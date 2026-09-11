#!/usr/bin/env python3
"""Build a complete S2V multi-view jobs JSONL directly from the corpus index.

The builder first selects one balanced, high-quality anchor per video, then reads each
anchor record and selects complementary frames containing the same entity. A video is
emitted atomically: exactly ``--views-per-video`` rows or no rows at all. An existing
anchor JSONL can still be supplied with ``--anchors`` for recovery or reuse.
"""

import argparse
import itertools
import json
import math
import os
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

import s2v_anchor_selection as anchor_selection


ROOT = Path(__file__).resolve().parent
DEFAULT_INDEX = ROOT / "dataset" / "s2v_easy_data" / "output" / "index.jsonl"
DEFAULT_OUTPUT = ROOT / "dataset" / "s2v_keyframe_recontext" / "jobs.multiview.jsonl"


def parse_args():
    parser = argparse.ArgumentParser("Build complete S2V multi-view jobs from index.jsonl")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--index", default=str(DEFAULT_INDEX),
                        help="Corpus index used to select anchors (default source).")
    source.add_argument("--anchors", help="Reuse an existing one-row-per-video anchor JSONL.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--views-per-video", type=int, default=4,
                        help="Total rows per accepted video, including the anchor.")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--limit", type=int, default=0,
                        help="Process at most N selected anchors (0 = all).")

    anchor = parser.add_argument_group("anchor selection from --index")
    anchor.add_argument("--target", type=int, default=0,
                        help="Maximum anchors after balancing (0 = all).")
    anchor.add_argument("--seed", type=int, default=0)
    anchor.add_argument("--variants", default=anchor_selection.DEFAULT_VARIANTS)
    anchor.add_argument("--scan-limit", type=int, default=0)
    anchor.add_argument("--anchor-min-area-frac", type=float, default=0.08)
    anchor.add_argument("--anchor-max-area-frac", type=float, default=0.75)
    anchor.add_argument("--anchor-min-sharpness", type=float, default=120.0)
    anchor.add_argument("--anchor-min-confidence", type=float, default=0.9)
    anchor.add_argument("--anchor-max-objects-in-frame", type=int, default=2)
    anchor.add_argument("--per-category", type=int, default=0)
    anchor.add_argument("--allow-anchor-not-in-hand", action="store_true")
    anchor.add_argument("--allow-anchor-clipped", action="store_true")

    extra = parser.add_argument_group("additional view selection")
    extra.add_argument("--min-area-frac", type=float, default=0.035)
    extra.add_argument("--max-area-frac", type=float, default=0.80)
    extra.add_argument("--min-confidence", type=float, default=0.75)
    extra.add_argument("--min-sharpness", type=float, default=40.0)
    extra.add_argument("--min-short-side", type=int, default=96)
    extra.add_argument("--edge-margin", type=float, default=0.005)
    extra.add_argument("--max-objects-in-frame", type=int, default=2)
    extra.add_argument("--min-diversity", type=float, default=0.05)
    extra.add_argument("--allow-not-in-hand", action="store_true")
    extra.add_argument("--allow-motion-blur", action="store_true")
    extra.add_argument("--strict-quality", action="store_true",
                       help="Reject candidates failing quality thresholds.")
    extra.add_argument("--allow-estimated-bbox", action="store_true", default=True)
    extra.add_argument("--no-estimated-bbox", dest="allow_estimated_bbox",
                       action="store_false")
    return parser.parse_args()


def safe(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-") or "x"


def valid_bbox(value):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    bbox = [float(item) for item in value]
    if not all(math.isfinite(item) for item in bbox):
        return None
    x0, x1 = sorted((bbox[0], bbox[2]))
    y0, y1 = sorted((bbox[1], bbox[3]))
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def crop_for_bbox(image, bbox, padding=0.06):
    height, width = image.shape[:2]
    x0, y0, x1, y1 = bbox
    pad_x, pad_y = (x1 - x0) * padding, (y1 - y0) * padding
    left = max(0, min(width - 1, int((x0 - pad_x) * width)))
    top = max(0, min(height - 1, int((y0 - pad_y) * height)))
    right = max(left + 1, min(width, math.ceil((x1 + pad_x) * width)))
    bottom = max(top + 1, min(height, math.ceil((y1 + pad_y) * height)))
    return image[top:bottom, left:right]


def descriptor(crop):
    resized = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    hog = cv2.HOGDescriptor((128, 128), (32, 32), (16, 16), (16, 16), 9)
    edges = hog.compute(gray).reshape(-1).astype(np.float32)
    structure = cv2.dct(gray.astype(np.float32) / 255.0)[:16, :16].reshape(-1)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    colour = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    colour = cv2.normalize(colour, None).reshape(-1).astype(np.float32)
    return edges, structure, colour


def cosine_distance(left, right):
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return 0.0
    return float(np.clip(1.0 - np.dot(left, right) / denominator, 0.0, 1.0))


def view_distance(left, right):
    return float(np.clip(
        0.65 * cosine_distance(left[0], right[0])
        + 0.25 * cosine_distance(left[1], right[1])
        + 0.10 * cv2.compareHist(left[2], right[2], cv2.HISTCMP_BHATTACHARYYA),
        0.0, 1.0,
    ))


def matching_entity(entities, entity_id):
    return next((item for item in entities if item.get("entity_id") == entity_id), None)


def matching_reference(record, entity_id):
    return next((item for item in record.get("references") or []
                 if item.get("entity_id") == entity_id), None)


def upstream_bbox_map(reference, frames):
    dimensions = {frame.get("frame_index"): (frame.get("width"), frame.get("height"))
                  for frame in frames}
    result = {}
    for view in (reference or {}).get("views") or []:
        frame_index = view.get("source_frame_index")
        bbox = valid_bbox(view.get("raw_bbox") or view.get("bbox"))
        if frame_index is None or bbox is None:
            continue
        if max(bbox) > 1.5:
            width, height = dimensions.get(frame_index, (None, None))
            if not width or not height:
                continue
            bbox = valid_bbox([bbox[0] / width, bbox[1] / height,
                               bbox[2] / width, bbox[3] / height])
        if bbox:
            result[frame_index] = bbox
    return result


def estimated_bbox(frame_index, known):
    ordered = sorted((index, np.asarray(bbox)) for index, bbox in known.items() if bbox)
    if not ordered:
        return None, "missing"
    lower = [(index, bbox) for index, bbox in ordered if index < frame_index]
    upper = [(index, bbox) for index, bbox in ordered if index > frame_index]
    if lower and upper:
        left_index, left = lower[-1]
        right_index, right = upper[0]
        weight = (frame_index - left_index) / (right_index - left_index)
        bbox, source = left * (1 - weight) + right * weight, "interpolated"
    else:
        _, bbox = min(ordered, key=lambda item: abs(item[0] - frame_index))
        source = "nearest_propagated"
    return valid_bbox(np.clip(bbox, 0.0, 1.0).tolist()), source


def quality_reason(entity, entities, bbox, image, sharpness, config):
    height, width = image.shape[:2]
    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    short_side = min((bbox[2] - bbox[0]) * width, (bbox[3] - bbox[1]) * height)
    margin = config["edge_margin"]
    if float(entity.get("confidence") or 0) < config["min_confidence"]:
        return "confidence", area, short_side
    if not config["allow_not_in_hand"] and entity.get("in_hand") is not True:
        return "not_in_hand", area, short_side
    if not config["allow_motion_blur"] and entity.get("motion_blur"):
        return "motion_blur", area, short_side
    if len(entities) > config["max_objects_in_frame"]:
        return "too_many_objects", area, short_side
    if not config["min_area_frac"] <= area <= config["max_area_frac"]:
        return "area", area, short_side
    if short_side < config["min_short_side"]:
        return "short_side", area, short_side
    if sharpness < config["min_sharpness"]:
        return "sharpness", area, short_side
    if bbox[0] <= margin or bbox[1] <= margin or bbox[2] >= 1 - margin or bbox[3] >= 1 - margin:
        return "clipped", area, short_side
    return None, area, short_side


def normalized_quality(area, confidence, sharpness, short_side):
    area_score = np.clip((area - 0.035) / 0.145, 0.0, 1.0)
    sharpness_score = np.clip((math.log1p(sharpness) - math.log1p(40)) /
                              (math.log1p(400) - math.log1p(40)), 0.0, 1.0)
    size_score = np.clip((short_side - 96) / 224, 0.0, 1.0)
    return float(0.30 * area_score + 0.25 * sharpness_score
                 + 0.25 * confidence + 0.20 * size_score)


def select_views(anchor_descriptor, candidates, count, min_diversity):
    best = None
    for combination in itertools.combinations(candidates, count):
        descriptions = [anchor_descriptor] + [item["descriptor"] for item in combination]
        distances = [view_distance(descriptions[i], descriptions[j])
                     for i in range(len(descriptions)) for j in range(i + 1, len(descriptions))]
        minimum, mean = min(distances), sum(distances) / len(distances)
        quality = sum(item["quality"] for item in combination) / count
        key = (0.60 * minimum + 0.25 * mean + 0.15 * quality, minimum, quality)
        if minimum >= min_diversity and (best is None or key > best[0]):
            best = key, combination, minimum, mean
    return None if best is None else best[1:]


def process_anchor(payload):
    anchor, config = payload
    rejects = Counter()
    try:
        record = json.loads(Path(anchor["record_json"]).read_text(encoding="utf-8"))
        anchor_image = cv2.imread(str(anchor["frame_path"]), cv2.IMREAD_COLOR)
        anchor_bbox = valid_bbox(anchor.get("bbox_norm"))
        if anchor_image is None or anchor_bbox is None:
            raise ValueError("missing anchor image or bbox")
        anchor_descriptor = descriptor(crop_for_bbox(anchor_image, anchor_bbox))
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as error:
        return "anchor_error", [], {type(error).__name__: 1}

    total_views = config["views_per_video"]
    parent_name = anchor["job_name"]
    base_row = dict(anchor)
    base_row.update({"view_rank": 0, "extra_view_rank": 0, "is_anchor": True,
                     "parent_job_name": parent_name,
                     "anchor_frame_index": anchor.get("frame_index")})
    if total_views == 1:
        return "accepted", [base_row], {}

    frames = record.get("candidate_frames") or []
    detections = record.get("frame_detections") or []
    entity_id = anchor.get("entity_id")
    reference = matching_reference(record, entity_id)
    score_map = {view.get("source_frame_index"): view.get("select_score") or {}
                 for view in (reference or {}).get("views") or []}
    upstream = upstream_bbox_map(reference, frames)
    known = {anchor.get("frame_index"): anchor_bbox, **upstream}
    for slot, frame in enumerate(frames):
        entities = detections[slot].get("entities") or [] if slot < len(detections) else []
        entity = matching_entity(entities, entity_id)
        bbox = valid_bbox((entity or {}).get("bbox"))
        if bbox:
            known[frame.get("frame_index")] = bbox

    candidates = []
    for slot, frame in enumerate(frames):
        frame_index = frame.get("frame_index")
        if frame_index == anchor.get("frame_index"):
            continue
        entities = detections[slot].get("entities") or [] if slot < len(detections) else []
        entity = matching_entity(entities, entity_id)
        bbox = valid_bbox((entity or {}).get("bbox"))
        bbox_source = "frame_detection"
        if entity is None or bbox is None:
            if not config["allow_estimated_bbox"]:
                rejects["missing_detection"] += 1
                continue
            bbox = upstream.get(frame_index)
            bbox_source = "upstream_view"
            if bbox is None:
                bbox, bbox_source = estimated_bbox(frame_index, known)
            entity = {"entity_id": entity_id, "bbox": bbox, "confidence": 0.0,
                      "in_hand": anchor.get("in_hand"), "motion_blur": False}
        image = cv2.imread(str(frame.get("path")), cv2.IMREAD_COLOR)
        if image is None or bbox is None:
            rejects["missing_image_or_bbox"] += 1
            continue
        crop = crop_for_bbox(image, bbox)
        if not crop.size:
            rejects["empty_crop"] += 1
            continue
        computed_sharpness = float(cv2.Laplacian(
            cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
        sharpness = float(score_map.get(frame_index, {}).get("sharpness", computed_sharpness))
        reason, area, short_side = quality_reason(
            entity, entities, bbox, image, sharpness, config)
        if reason:
            rejects["quality_" + reason] += 1
            if config["strict_quality"]:
                continue
        confidence = float(entity.get("confidence") or 0)
        quality = normalized_quality(area, confidence, sharpness, short_side)
        if reason:
            quality *= 0.75
        candidates.append({
            "frame_path": str(frame.get("path")), "frame_slot": slot,
            "frame_index": frame_index, "timestamp": frame.get("timestamp"),
            "bbox_norm": [round(value, 5) for value in bbox],
            "confidence": confidence, "in_hand": bool(entity.get("in_hand")),
            "other_objects": [item.get("name") for item in entities
                              if item.get("entity_id") != entity_id and item.get("name")],
            "bbox_source": bbox_source, "quality_warning": reason, "descriptor": descriptor(crop),
            "quality": quality,
            "view_score": {"area_frac": round(area, 6), "sharpness": round(sharpness, 3),
                           "computed_sharpness": round(computed_sharpness, 3),
                           "short_side_px": round(short_side, 2), "confidence": confidence},
        })

    count = total_views - 1
    if len(candidates) < count:
        rejects["fewer_than_required"] += 1
        return "insufficient_views", [], dict(rejects)
    selected = select_views(anchor_descriptor, candidates, count, config["min_diversity"])
    if selected is None:
        rejects["diversity"] += 1
        return "insufficient_diversity", [], dict(rejects)
    combination, minimum, mean = selected
    rows = [base_row]
    ordered = sorted(combination, key=lambda item: item["frame_index"])
    for rank, candidate in enumerate(ordered, 1):
        row = dict(anchor)
        row.update({key: value for key, value in candidate.items()
                    if key not in {"descriptor", "quality"}})
        row.update({"view_rank": rank, "extra_view_rank": rank, "is_anchor": False,
                    "parent_job_name": parent_name,
                    "anchor_frame_index": anchor.get("frame_index")})
        row["view_score"].update({"quality": round(candidate["quality"], 6),
                                  "set_min_distance": round(minimum, 6),
                                  "set_mean_distance": round(mean, 6)})
        row["job_name"] = "__".join([
            anchor["sample_id"], safe(anchor.get("original_id")),
            f"f{int(candidate['frame_index']):06d}", anchor["entity_id"], anchor["variant"],
        ])
        rows.append(row)
    return "accepted", rows, dict(rejects)


def select_anchors_from_index(index_path, args):
    selection_args = argparse.Namespace(
        target=args.target, seed=args.seed, scan_limit=args.scan_limit,
        min_area_frac=args.anchor_min_area_frac,
        max_area_frac=args.anchor_max_area_frac,
        min_sharpness=args.anchor_min_sharpness,
        min_confidence=args.anchor_min_confidence,
        require_in_hand=not args.allow_anchor_not_in_hand,
        require_not_clipped=not args.allow_anchor_clipped,
        max_objects_in_frame=args.anchor_max_objects_in_frame,
        per_category=args.per_category, per_sample=1,
    )
    print(f"Selecting anchors from {index_path} ...", flush=True)
    candidates, rejects, names, scanned, accepted = anchor_selection.collect_candidates(
        index_path, selection_args
    )
    if not candidates:
        raise ValueError("No anchor candidate passed the configured quality filters")
    anchors = anchor_selection.balanced_sample(candidates, names, selection_args)
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    if not variants:
        raise ValueError("--variants must contain at least one value")
    for index, row in enumerate(anchors):
        row["variant"] = variants[index % len(variants)]
        row["job_name"] = "__".join([
            row["sample_id"], anchor_selection.safe(row.get("original_id")),
            f"f{int(row['frame_index']):06d}", row["entity_id"], row["variant"],
        ])
    print(f"Index lines={scanned}, accepted samples={accepted}, "
          f"anchor candidates={len(candidates)}, selected anchors={len(anchors)}")
    print(f"Anchor rejections: {dict(rejects.most_common())}")
    return anchors


def load_anchors(path, limit):
    anchors, seen = [], set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = row.get("sample_id")
            if not sample_id or sample_id in seen:
                raise ValueError(f"{path}:{line_number}: duplicate or missing sample_id {sample_id!r}")
            seen.add(sample_id)
            anchors.append(row)
            if limit and len(anchors) >= limit:
                break
    return anchors


def main():
    args = parse_args()
    if args.views_per_video < 1:
        raise ValueError("--views-per-video must be at least 1")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    output_path = Path(args.output).expanduser().resolve()
    if args.anchors:
        anchors_path = Path(args.anchors).expanduser().resolve()
        if not anchors_path.is_file():
            raise FileNotFoundError(f"Anchor jobs not found: {anchors_path}")
        anchors = load_anchors(anchors_path, args.limit)
        source_description = str(anchors_path)
    else:
        index_path = Path(args.index).expanduser().resolve()
        if not index_path.is_file():
            raise FileNotFoundError(f"Corpus index not found: {index_path}")
        anchors = select_anchors_from_index(index_path, args)
        if args.limit:
            anchors = anchors[:args.limit]
        source_description = str(index_path)
    if not anchors:
        raise ValueError(f"No anchors selected from {source_description}")
    config = vars(args).copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    statuses, rejects = Counter(), Counter()
    written = 0
    with temporary.open("w", encoding="utf-8") as output:
        payloads = ((anchor, config) for anchor in anchors)
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for index, (status, rows, row_rejects) in enumerate(
                    executor.map(process_anchor, payloads, chunksize=16), 1):
                statuses[status] += 1
                rejects.update(row_rejects)
                for row in rows:
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written += 1
                if index % 1000 == 0 or index == len(anchors):
                    print(f"processed={index}/{len(anchors)} accepted={statuses['accepted']} "
                          f"rows={written}", flush=True)
    temporary.replace(output_path)
    print(f"Wrote {written} rows ({args.views_per_video} per accepted video) -> {output_path}")
    print(f"Video statuses: {dict(statuses)}")
    print(f"Candidate rejections: {dict(rejects.most_common())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
