# Pet Head Synthesis

An image curation and conditional generation project for 32×32 RGB cat and dog portraits. The starting collection contained 76,250 unlabelled images. The central challenge was deciding whether a proposed crop showed a single, well-framed head in a real photograph. A detector can locate the right animal while still including most of its body or cutting off its ears.

## Building the training set

CLIP ranked whole images against descriptions of real animals, illustrations, toys and unrelated objects. Grounding DINO then localized potential heads in 23,009 shortlisted images. Independent cat and dog prompts often returned almost identical boxes around one face. Grouping these boxes by location made it possible to check for a second head without counting the same detection twice.

A quality scorer combined crop-level visual features with framing measurements. It learned from 398 explicit human judgments. After ranking candidates separately by species and removing near duplicates, the training set contained 3,988 cat crops and 5,242 dog crops from distinct source images.

In a separate 200-crop audit, a human reviewer accepted 196 crops. Estimated precision was 98.0% (95% Wilson interval, 95.0–99.2%). All four observed failures concerned framing: two heads were too large or too small, one was off-centre or tilted, and one had clipped ears or head. The audit measured quality among selected crops; estimating how many good heads were missed would require sampling the excluded images.

## Generating portraits

A class-conditional U-Net diffusion model was trained from scratch on the selected crops. The chosen checkpoint used exponential moving average weights from training step 80,000, 50 DDIM sampling steps and classifier-free guidance of 2. The class label controls the requested species, and a seed controls the initial noise.

| Local development evaluation | Result |
|---|---:|
| Class accuracy | 95.1% |
| FID | 44.31 |
| KID | 0.0040 |
| Distribution precision / recall | 0.736 / 0.554 |
| Estimated memorization rate | 0.25% |

The evaluation used 1,000 generated images per class. Increasing guidance from 2 to 3 at the same checkpoint raised class accuracy to 97.2%, while distribution recall fell to 0.508. Guidance 2 retained more visual variety at comparable FID. The 13.56-million-parameter model generated 1,000 images in about 117 seconds on an RTX 4060 Laptop GPU.
