"""Class-conditional DDPM with classifier-free guidance for 32x32 RGB cat/dog heads.

This file is self-contained (it imports only torch and the standard library) because the
export step copies it verbatim into ``submission/catdog_net.py``, and the graded
``generate.py`` may only import code that lives inside ``submission/``.
``tests/test_submission.py`` checks that the two copies are identical.

Notation (Ho et al. 2020, "Denoising Diffusion Probabilistic Models"; Nichol & Dhariwal
2021, "Improved DDPM"; Song et al. 2021, "DDIM"; Ho & Salimans 2022, "Classifier-Free
Diffusion Guidance"):

    x_0            a training image scaled to [-1, 1]
    abar_t         cumulative signal fraction, cosine schedule:
                   abar_t = f(t) / f(0),  f(t) = cos^2(((t / T) + s) / (1 + s) * pi / 2)
    forward        x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps,   eps ~ N(0, I)
    loss           || eps - eps_theta(x_t, t, c) ||^2,  c replaced by the null label with
                   probability p_uncond (so one network learns both eps(x,t,c), eps(x,t,0))
    guidance       eps_hat = eps_theta(x_t, t, null) + w * (eps_theta(x_t, t, c) - eps_theta(x_t, t, null))
                   (w = 1: plain conditional model; w > 1: stronger class signal)
    DDIM (eta=0)   x0_hat = (x_t - sqrt(1 - abar_t) eps_hat) / sqrt(abar_t), clipped to [-1, 1]
                   x_{t'} = sqrt(abar_t') x0_hat + sqrt(1 - abar_t') eps_hat

The network uses GroupNorm only (no BatchNorm) and no other cross-sample operation, so
every output image depends only on its own label and its own initial noise.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

NUM_CLASSES = 2          # 0 = cat, 1 = dog
NULL_LABEL = NUM_CLASSES  # extra embedding row used for the unconditional branch of CFG
_MASK63 = (1 << 63) - 1
_MASK64 = (1 << 64) - 1


# ---------------------------------------------------------------------------
# Noise schedule
# ---------------------------------------------------------------------------

def make_alphas_cumprod(timesteps: int = 1000, schedule: str = "cosine") -> torch.Tensor:
    """abar_t for t = 0..T-1 (float64)."""
    if schedule == "cosine":
        s = 0.008
        steps = torch.arange(timesteps + 1, dtype=torch.float64) / timesteps
        f = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
        betas = (1 - f[1:] / f[:-1]).clamp(max=0.999)
    elif schedule == "linear":
        betas = torch.linspace(1e-4, 0.02, timesteps, dtype=torch.float64)
    else:
        raise ValueError(f"unknown schedule {schedule!r}")
    return torch.cumprod(1 - betas, dim=0)


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------

def _groups(channels: int) -> int:
    return 32 if channels % 32 == 0 and channels >= 64 else 8


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of integer timesteps t [N] -> [N, dim]."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32)
                      / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=1)


class ResBlock(nn.Module):
    """GroupNorm-SiLU-Conv x2 with a scale-shift (FiLM) injection of time+class embedding."""

    def __init__(self, cin: int, cout: int, emb_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(cin), cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.emb = nn.Linear(emb_dim, 2 * cout)
        self.norm2 = nn.GroupNorm(_groups(cout), cout)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb(F.silu(emb))[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(self.dropout(F.silu(h)))
        return self.skip(x) + h


class AttnBlock(nn.Module):
    """Multi-head self-attention over the spatial positions of one image."""

    def __init__(self, channels: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(_groups(channels), channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(n, 3, self.heads, c // self.heads, h * w) \
            .permute(1, 0, 2, 4, 3).unbind(0)              # each [n, heads, hw, c/heads]
        out = F.scaled_dot_product_attention(q, k, v)      # attention within each image
        out = out.permute(0, 1, 3, 2).reshape(n, c, h, w)
        return x + self.proj(out)


class UNet(nn.Module):
    """DDPM-style U-Net: eps_theta(x_t, t, label). label == NULL_LABEL means unconditional."""

    def __init__(self, base_channels: int = 96, channel_mult=(1, 2, 2), num_res_blocks: int = 2,
                 attn_resolutions=(16, 8), dropout: float = 0.1, heads: int = 4,
                 image_size: int = 32):
        super().__init__()
        ch = base_channels
        emb_dim = 4 * ch
        self.base_channels = ch
        self.time_mlp = nn.Sequential(nn.Linear(ch, emb_dim), nn.SiLU(),
                                      nn.Linear(emb_dim, emb_dim))
        self.label_emb = nn.Embedding(NUM_CLASSES + 1, emb_dim)
        self.conv_in = nn.Conv2d(3, ch, 3, padding=1)

        self.down = nn.ModuleList()
        skips = [ch]
        cur, res = ch, image_size
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                block = nn.ModuleList([ResBlock(cur, ch * mult, emb_dim, dropout)])
                cur = ch * mult
                if res in attn_resolutions:
                    block.append(AttnBlock(cur, heads))
                self.down.append(block)
                skips.append(cur)
            if level != len(channel_mult) - 1:
                self.down.append(nn.ModuleList([nn.Conv2d(cur, cur, 3, stride=2, padding=1)]))
                skips.append(cur)
                res //= 2

        self.mid = nn.ModuleList([ResBlock(cur, cur, emb_dim, dropout), AttnBlock(cur, heads),
                                  ResBlock(cur, cur, emb_dim, dropout)])

        self.up = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mult))):
            for _ in range(num_res_blocks + 1):
                block = nn.ModuleList([ResBlock(cur + skips.pop(), ch * mult, emb_dim, dropout)])
                cur = ch * mult
                if res in attn_resolutions:
                    block.append(AttnBlock(cur, heads))
                self.up.append(block)
            if level != 0:
                self.up.append(nn.ModuleList([nn.Upsample(scale_factor=2, mode="nearest"),
                                              nn.Conv2d(cur, cur, 3, padding=1)]))
                res *= 2

        self.norm_out = nn.GroupNorm(_groups(cur), cur)
        self.conv_out = nn.Conv2d(cur, 3, 3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        emb = self.time_mlp(timestep_embedding(t, self.base_channels)) + self.label_emb(labels)
        h = self.conv_in(x)
        hs = [h]
        for block in self.down:
            if isinstance(block[0], ResBlock):
                h = block[0](h, emb)
                for extra in block[1:]:
                    h = extra(h)
            else:                                   # downsample
                h = block[0](h)
            hs.append(h)
        h = self.mid[0](h, emb)
        h = self.mid[1](h)
        h = self.mid[2](h, emb)
        for block in self.up:
            if isinstance(block[0], ResBlock):
                h = block[0](torch.cat([h, hs.pop()], dim=1), emb)
                for extra in block[1:]:
                    h = extra(h)
            else:                                   # upsample + conv
                h = block[1](block[0](h))
        return self.conv_out(F.silu(self.norm_out(h)))


# ---------------------------------------------------------------------------
# Diffusion wrapper: training loss and DDIM + CFG sampling
# ---------------------------------------------------------------------------

class CatDogDiffusion(nn.Module):
    """Holds the U-Net (all trainable parameters) and the fixed noise schedule."""

    def __init__(self, model_config: dict | None = None, timesteps: int = 1000,
                 schedule: str = "cosine", sample_steps: int = 50, guidance: float = 2.0):
        super().__init__()
        self.model_config = dict(model_config or {})
        self.unet = UNet(**self.model_config)
        self.timesteps = timesteps
        self.schedule = schedule
        self.sample_steps = sample_steps
        self.guidance = guidance
        # Not saved in the state dict: rebuilt from (timesteps, schedule).
        self.register_buffer("alphas_cumprod",
                             make_alphas_cumprod(timesteps, schedule).float(), persistent=False)

    # -- training ---------------------------------------------------------
    def loss(self, x0: torch.Tensor, labels: torch.Tensor, p_uncond: float = 0.1,
             generator: torch.Generator | None = None, t: torch.Tensor | None = None,
             noise: torch.Tensor | None = None) -> torch.Tensor:
        """Mean eps-prediction MSE. x0 in [-1, 1]; labels 0/1. t and noise may be fixed
        (used by the single-batch overfitting test)."""
        n = x0.shape[0]
        dev = x0.device
        if t is None:
            t = torch.randint(0, self.timesteps, (n,), device=dev, generator=generator)
        if noise is None:
            noise = torch.randn(x0.shape, device=dev, generator=generator)
        if p_uncond > 0:
            drop = torch.rand(n, device=dev, generator=generator) < p_uncond
            labels = torch.where(drop, torch.full_like(labels, NULL_LABEL), labels)
        abar = self.alphas_cumprod[t].view(n, 1, 1, 1)
        x_t = abar.sqrt() * x0 + (1 - abar).sqrt() * noise
        pred = self.unet(x_t, t, labels)
        return F.mse_loss(pred.float(), noise)

    # -- sampling ---------------------------------------------------------
    def ddim_timesteps(self, steps: int) -> list[int]:
        """Evenly spaced timesteps from T-1 down to 0."""
        ts = torch.linspace(self.timesteps - 1, 0, steps).round().long().tolist()
        return sorted(set(ts), reverse=True)

    @torch.no_grad()
    def sample(self, noise: torch.Tensor, labels: torch.Tensor, steps: int | None = None,
               guidance: float | None = None) -> torch.Tensor:
        """Deterministic DDIM (eta = 0) with classifier-free guidance.

        noise [N,3,32,32] is x_T; returns images in [0, 1]. Each output row depends only on
        its own noise row and label (the network has no cross-sample operations)."""
        steps = steps or self.sample_steps
        w = self.guidance if guidance is None else guidance
        x = noise
        n = x.shape[0]
        ab = self.alphas_cumprod
        ts = self.ddim_timesteps(steps)
        null = torch.full_like(labels, NULL_LABEL)
        for i, t in enumerate(ts):
            t_batch = torch.full((n,), t, device=x.device, dtype=torch.long)
            if w == 1.0:
                eps = self.unet(x, t_batch, labels)
            else:
                both = self.unet(torch.cat([x, x]), torch.cat([t_batch, t_batch]),
                                 torch.cat([labels, null]))
                eps_c, eps_u = both.chunk(2)
                eps = eps_u + w * (eps_c - eps_u)
            a_t = ab[t]
            x0 = ((x - (1 - a_t).sqrt() * eps) / a_t.sqrt()).clamp(-1, 1)
            eps = (x - a_t.sqrt() * x0) / (1 - a_t).sqrt()      # consistent with clipped x0
            a_prev = ab[ts[i + 1]] if i + 1 < len(ts) else torch.ones((), device=x.device)
            x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
        return ((x + 1) / 2).clamp(0, 1)


# ---------------------------------------------------------------------------
# Per-sample seeding
# ---------------------------------------------------------------------------

def _splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    return x ^ (x >> 31)


def sample_seed(seed: int, index: int) -> int:
    """A well-mixed 63-bit seed for sample `index` of a call with `seed`."""
    return _splitmix64(_splitmix64(seed & _MASK64) ^ index) & _MASK63


def initial_noise(n: int, seed: int, shape=(3, 32, 32)) -> torch.Tensor:
    """x_T for n samples on the CPU; row i depends only on (seed, i), not on n or labels."""
    rows = [torch.randn(shape, generator=torch.Generator().manual_seed(sample_seed(seed, i)))
            for i in range(n)]
    return torch.stack(rows) if rows else torch.zeros(0, *shape)


def build_from_export(blob: dict) -> CatDogDiffusion:
    """Rebuild the model from an exported weights dict (see catdog_gen/export.py)."""
    d = blob["diffusion"]
    s = blob["sampling"]
    model = CatDogDiffusion(blob["model_config"], timesteps=d["timesteps"],
                            schedule=d["schedule"], sample_steps=s["steps"],
                            guidance=s["guidance"])
    model.load_state_dict(blob["state_dict"])
    return model
