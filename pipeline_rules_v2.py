"""Pure geometry, routing and decisions for the v2 pilot."""
from dataclasses import dataclass
from itertools import combinations
import hashlib
import json
import math

VERSION = "v2.0"
REAL = {"Real Cat", "Real Dog"}
LABELS = {"cat": 0, "dog": 1}
CLASSES = ("Real Cat", "Real Dog", "Cartoon Cat", "Cartoon Dog", "Toy / Print", "Other")
SCORE_KEYS = ("real_cat", "real_dog", "cartoon_cat", "cartoon_dog", "toy_print", "other")


def stable_id(*parts):
    return hashlib.sha256(json.dumps(parts, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class Detection:
    prompt: str
    query_id: int
    score: float
    box: tuple
    flags: tuple = ()

    @property
    def key(self):
        return self.prompt, self.query_id


@dataclass(frozen=True)
class Square:
    x0: int
    y0: int
    size: int
    angle: int = 0

    @property
    def key(self):
        return self.x0, self.y0, self.size, self.angle


def describe_legacy_scores(scores, score_floor=0.20, margin_floor=0.010):
    if len(scores) != len(CLASSES) or not all(map(math.isfinite, scores)):
        raise ValueError("Expected six finite CLIP scores")
    ranked = sorted(zip(CLASSES, scores, strict=True), key=lambda pair: -pair[1])
    (first, top), (second, runner) = ranked[:2]
    margin = top - runner
    reasons = []
    if top < score_floor:
        reasons.append("top_cosine_below_floor")
    if margin < margin_floor:
        reasons.append("top_two_too_close")
    return dict(scores=dict(zip(CLASSES, scores, strict=True)), top_class=first,
                top_score=top, second_class=second, second_score=runner,
                margin=margin, prediction="Uncertain" if reasons else first, reasons=reasons)


def verify_square(square, image_size):
    """Validate official geometry independently of the detector."""
    if square is None:
        return ("NO_USABLE_SQUARE",)
    x, y, side, angle = square.key
    w, h = image_size
    if any(type(value) is not int for value in (x, y, side)) or side < 40:
        return ("INVALID_SQUARE_SIZE_OR_COORDINATES",)
    if not isinstance(angle, (int, float)) or not math.isfinite(angle) or abs(angle) > 45:
        return ("INVALID_SQUARE_ANGLE",)
    center_x, center_y = x + side / 2, y + side / 2
    theta = math.radians(angle)
    half_extent = side / 2 * (abs(math.cos(theta)) + abs(math.sin(theta)))
    if (center_x - half_extent < -1e-9 or center_y - half_extent < -1e-9
            or center_x + half_extent > w + 1e-9 or center_y + half_extent > h + 1e-9):
        return ("INVALID_SQUARE_BOUNDS",)
    return ()


def valid_shape(box):
    return len(box) == 4 and all(map(math.isfinite, box)) and box[0] < box[2] and box[1] < box[3]


def extent(box):
    return max(box[2] - box[0], box[3] - box[1])


def area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def intersection(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def iou(a, b):
    overlap = intersection(a, b)
    return overlap / (area(a) + area(b) - overlap)


def nearest_integer(value, lower, upper):
    return max(lower, min(upper, math.ceil(value - 0.5)))


def plan_square(box, image_size, cfg):
    """Return a square that contains the box, a failure reason, and local warnings."""
    w, h = image_size
    if not valid_shape(box):
        return None, "BOX_DEGENERATE", ()
    left, top, right, bottom = box
    if left < 0 or top < 0 or right > w or bottom > h:
        return None, "BOX_OUTSIDE_SOURCE", ()
    if right - left < 32:
        return None, "BOX_WIDTH_PROXY_TOO_SMALL", ()
    long_side = extent(box)
    lo = max(40, math.ceil(long_side / cfg["max_head_fraction"]))
    hi = min(w, h, math.floor(long_side / cfg["min_head_fraction"]))
    if lo > hi:
        return None, "BOX_RATIO_PROXY_FAIL", ()
    cx, cy = (left + right) / 2, (top + bottom) / 2
    target = long_side / cfg["target_head_fraction"]
    for side in sorted(range(lo, hi + 1), key=lambda s: (abs(s - target), s)):
        x_min, x_max = max(0, math.ceil(right - side)), min(w - side, math.floor(left))
        y_min, y_max = max(0, math.ceil(bottom - side)), min(h - side, math.floor(top))
        if x_min > x_max or y_min > y_max:
            continue
        x0 = nearest_integer(cx - side / 2, x_min, x_max)
        y0 = nearest_integer(cy - side / 2, y_min, y_max)
        warnings = []
        if max(abs(cx - x0 - side / 2), abs(cy - y0 - side / 2)) > 0.10 * side:
            warnings.append("OFF_CENTER_AFTER_SHIFT")
        if min(left, top, w - right, h - bottom) < 0.05 * long_side:
            warnings.append("BOX_NEAR_SOURCE_EDGE")
        return Square(x0, y0, side), "", tuple(warnings)
    return None, "NO_INTEGER_SQUARE", ()


def distinct_regions(detections, image_size, cfg):
    """Select the best matching existing anchor; preserve ambiguous links."""
    keys = [d.key for d in detections]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate detection key")
    if not all(math.isfinite(d.score) for d in detections):
        raise ValueError("Non-finite detection score")
    plans = {d.key: plan_square(d.box, image_size, cfg) for d in detections}
    ordered = sorted(detections, key=lambda d: (
        plans[d.key][0] is None, len(plans[d.key][2]),
        area(d.box) if valid_shape(d.box) else math.inf, d.key))
    groups = []
    cutoff = cfg["duplicate_iou"]
    for d in ordered:
        matches = []
        if valid_shape(d.box):
            for group in groups:
                anchor = group["anchor"]
                if valid_shape(anchor.box):
                    overlap = iou(anchor.box, d.box)
                    if overlap >= cutoff:
                        matches.append((-overlap, anchor.key, group))
        matches.sort(key=lambda m: m[:2])
        if not matches:
            groups.append(dict(anchor=d, members=[d], links=[]))
            continue
        chosen = matches[0][2]
        chosen["members"].append(d)
        for neg_iou, anchor_key, other in matches[1:]:
            link = (d.key, chosen["anchor"].key, anchor_key, -neg_iou)
            chosen["links"].append(link)
            other["links"].append(link)
    return groups, plans


def compare(detections, image_size, cfg):
    """Common Directed and Crop grouping, with spatial and prompt evidence."""
    groups, plans = distinct_regions(detections, image_size, cfg)
    cross = []
    nested = []
    shaped = sorted((d for d in detections if valid_shape(d.box)), key=lambda d: d.key)
    cutoff = cfg["duplicate_iou"]
    for a, b in combinations(shaped, 2):
        overlap = iou(a.box, b.box)
        if a.prompt != b.prompt and overlap >= cutoff:
            cross.append((a.key, b.key, overlap))
    usable_anchors = [g["anchor"] for g in groups if valid_shape(g["anchor"].box)]
    for a, b in combinations(usable_anchors, 2):
        covered = intersection(a.box, b.box) / min(area(a.box), area(b.box))
        if covered >= cfg["nested_containment"] and iou(a.box, b.box) < cutoff:
            nested.append((a.key, b.key, covered))
    return dict(groups=groups, plans=plans, cross=cross, nested=nested)


def crop_id(pool_id, source_fingerprint, square):
    return stable_id("official-crop-v1-angle0", pool_id, source_fingerprint, square.key)


def generate_candidates(pool_id, fingerprint, image_size, analysis, dino_run_id):
    """Construct variants and preserve warning scope; clip joins happen separately."""
    plans = analysis["plans"]
    emitted = {}
    regions = []
    for group in analysis["groups"]:
        anchor, members = group["anchor"], group["members"]
        region_id = stable_id(VERSION, dino_run_id, pool_id, sorted(d.key for d in members))
        regions.append(region_id)
        source_meta = dict(pool_id=pool_id, source_fingerprint=fingerprint, region_id=region_id,
                           representative_key=anchor.key, representative_box=anchor.box,
                           representative_model_flags=list(anchor.flags),
                           representative_planning_flags=list(plans[anchor.key][2]),
                           representative_plan_error=plans[anchor.key][1],
                           other_member_flags=[dict(key=d.key, model_flags=d.flags,
                                                    planning_flags=plans[d.key][2],
                                                    plan_error=plans[d.key][1])
                                               for d in members if d.key != anchor.key],
                           members=[dict(key=d.key, score=d.score, box=d.box) for d in members],
                           image_cross_prompt=bool(analysis["cross"]))
        alternatives = sorted((d for d in members if d != anchor and plans[d.key][0]),
                              key=lambda d: (area(d.box), d.key))
        primary_square = plans[anchor.key][0]
        variants = [(anchor, "primary", primary_square)]
        seen_squares = {primary_square}
        for ordered, name in ((alternatives, "tight"), (reversed(alternatives), "wide")):
            for detection in ordered:
                square = plans[detection.key][0]
                if square not in seen_squares:
                    variants.append((detection, name, square))
                    seen_squares.add(square)
                    break
        for detection, variant, square in variants:
            local = list(plans[detection.key][2])
            member_keys = {d.key for d in members}
            if len({d.prompt for d in members}) > 1:
                local.append("DINO_REGION_CROSS_PROMPT")
            elif any(a in member_keys or b in member_keys for a, b, _ in analysis["cross"]):
                local.append("DINO_CROSS_REGION_CONFLICT")
            if group["links"]:
                local.append("AMBIGUOUS_REGION_MATCH")
            if any(anchor.key in (a, b) for a, b, _ in analysis["nested"]):
                local.append("NESTED_BOX_REVIEW")
            if square:
                x, y, s, _ = square.key
                bounds = (x, y, x + s, y + s)
                if any(g is not group and valid_shape(g["anchor"].box)
                       and intersection(bounds, g["anchor"].box) > 0 for g in analysis["groups"]):
                    local.append("POSSIBLE_SECOND_HEAD")
            cid = crop_id(pool_id, fingerprint, square) if square else ""
            key = cid or region_id
            if key in emitted:
                existing = emitted[key]
                existing["region_ids"].append(region_id)
                existing["region_evidence"].append(source_meta)
                existing["local_flags"] = sorted(set(existing["local_flags"] + local + ["MULTI_REGION_SAME_CROP"]))
                continue
            emitted[key] = dict(**source_meta, region_ids=[region_id], region_evidence=[source_meta],
                                crop_id=cid, square=square, variant=variant,
                                box=detection.box, detection_key=detection.key, dino_score=detection.score,
                                detection_flags=list(detection.flags), local_flags=sorted(set(local)),
                                geometry_errors=verify_square(square, image_size),
                                planning_error=plans[detection.key][1],
                                estimated_width=detection.box[2] - detection.box[0],
                                box_fraction=extent(detection.box) / square.size if square else None)
    return list(emitted.values()), regions


def assign_review_priority(row):
    if row.get("read_error") or not row.get("square"):
        return "GEOMETRY_REVIEW"
    if row["original_class"] in REAL:
        if row["original_class"] == row.get("crop_class") and not row["local_flags"]:
            return "PRIMARY_AGREE"
        return "PRIMARY_CONFLICT"
    if row.get("crop_class") in REAL:
        return "RECOVERY"
    return "NEGATIVE_CONTROL" if row.get("control_selected") else "PENDING"


def decide_candidate(row, review=None):
    """REVIEW keeps a candidate eligible. ACCEPT requires scoped, explicit visual evidence."""
    priority = assign_review_priority(row)
    reasons = list(row["local_flags"])
    original, crop = row["original_class"], row.get("crop_class")
    proposed = crop if crop in REAL else original if original in REAL else ""
    if original in REAL and crop in REAL and original != crop:
        reasons.append("SPECIES_DISAGREE")
        proposed = ""
    if crop and ((original in REAL and crop not in REAL) or
                 (original not in REAL and original != "Uncertain" and crop in REAL)):
        reasons.append("REALISM_DISAGREE")
    if priority == "RECOVERY":
        reasons.append("CLIP_RECOVERY_EVIDENCE")
    if not crop:
        reasons.append("CROP_CLIP_MISSING")

    def result(state, extra, species=""):
        return dict(decision=state, review_priority=priority, proposed_species=proposed,
                    verified_species=species, label=LABELS.get(species, ""),
                    reason_codes=list(dict.fromkeys(reasons + list(extra))),
                    blocking_checks=list(extra) if state == "REVIEW" else [])

    if row.get("input_mismatch") or row.get("read_error"):
        return result("REVIEW", ["INPUT_MISMATCH" if row.get("input_mismatch") else "READ_ERROR"])
    if row.get("square") and row["geometry_errors"]:
        return result("REJECT", row["geometry_errors"])
    if not row.get("square"):
        return result("REVIEW", ["NO_USABLE_SQUARE", row.get("planning_error") or "NO_DETECTION"])
    if not row.get("preview_verified"):
        return result("REVIEW", ["PREVIEW_NOT_VERIFIED"])
    if not review:
        return result("REVIEW", ["HUMAN_CHECK_REQUIRED"])
    if review.get("crop_id") != row["crop_id"] or review.get("preview_hash") != row.get("preview_hash"):
        return result("REVIEW", ["HUMAN_REVIEW_STALE"])
    if review.get("conflicting"):
        return result("REVIEW", ["CONFLICTING_HUMAN_REVIEW"])
    hard_reasons = []
    if review.get("real_photo") == "no":
        hard_reasons.append("HUMAN_NONPHOTO")
    if review.get("human_species") == "other":
        hard_reasons.append("HUMAN_NON_TARGET")
    if review.get("exactly_one_head") == "no":
        hard_reasons.append("HUMAN_HEAD_COUNT")
    if review.get("native_width_verified") == "no":
        hard_reasons.append("HUMAN_HEAD_UNDER_32PX")
    if any(review.get(k) == "bad" for k in ("head_quality", "crop_quality")) or any(
        review.get(k) == "no" for k in ("coverage_includes_ears", "centered", "upright", "visual_scale")
    ):
        hard_reasons.append("HUMAN_HEAD_OR_CROP_BAD")
    if hard_reasons:
        if review.get("final_decision") == "accept":
            return result("REVIEW", ["CONFLICTING_HUMAN_REVIEW"])
        return result("REJECT", hard_reasons)
    required = ("real_photo", "exactly_one_head", "coverage_includes_ears",
                "centered", "upright", "visual_scale", "native_width_verified")
    species = review.get("human_species")
    missing = [k.upper() + "_UNVERIFIED" for k in required if review.get(k) != "yes"]
    if species not in LABELS:
        missing.append("SPECIES_UNVERIFIED")
    if review.get("head_quality") != "good" or review.get("crop_quality") != "good":
        missing.append("HUMAN_QUALITY_UNVERIFIED")
    if review.get("final_decision") != "accept" or not review.get("reviewer") or not review.get("reviewed_at"):
        missing.append("HUMAN_SIGNOFF_REQUIRED")
    if review.get("final_decision") == "reject":
        missing.append("HUMAN_REJECT_REASON_MISSING")
    conflicts = set(row["local_flags"]) | ({"SPECIES_DISAGREE", "REALISM_DISAGREE"} & set(reasons))
    if conflicts and not str(review.get("conflict_resolution") or "").strip():
        missing.append("CONFLICT_ADJUDICATION_REQUIRED")
    return result("REVIEW", missing) if missing else result("ACCEPT", ["VERIFIED_BY_REVIEW"], species)

