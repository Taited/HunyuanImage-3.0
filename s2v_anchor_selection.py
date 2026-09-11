"""Internal anchor-selection utilities for the S2V multi-view jobs builder.

Use ``build_s2v_multiview_jobs.py`` as the public jobs-generation entrypoint.
"""
import json
import random
import re
from collections import Counter, defaultdict


DEFAULT_VARIANTS = "object_only"


def safe(value):
    """Filename-safe, and stable: the upstream ids contain '=' and other awkward characters."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-") or "x"


def canon_category(name, known):
    """Group 'book'/'books' together so the per-category cap is not trivially doubled."""
    key = " ".join(name.lower().split())
    if key.endswith("s") and key[:-1] in known:
        return key[:-1]
    return key


def frame_for_reference(record, reference):
    """The candidate frame a reference was cropped from, plus that frame's detections.

    ``candidate_frames`` and ``frame_detections`` are index-aligned; the reference stores
    the *video* frame index, so match on ``frame_index`` rather than on list position.
    """
    target = reference.get("source_frame_index")
    if target is None:
        return None
    frames = record.get("candidate_frames") or []
    detections = record.get("frame_detections") or []
    for i, frame in enumerate(frames):
        if frame.get("frame_index") != target:
            continue
        entities = detections[i].get("entities") if i < len(detections) else []
        return i, frame, (entities or [])
    return None


def passes(score, args):
    if not score:
        return "no_select_score"
    if not (args.min_area_frac <= score.get("area_frac", 0.0) <= args.max_area_frac):
        return "area"
    if score.get("sharpness", 0.0) < args.min_sharpness:
        return "sharpness"
    if score.get("confidence", 0.0) < args.min_confidence:
        return "confidence"
    if args.require_in_hand and score.get("in_hand", 0.0) != 1.0:
        return "not_in_hand"
    if args.require_not_clipped and score.get("not_clipped", 0.0) != 1.0:
        return "clipped"
    return None


def collect_candidates(index_path, args):
    candidates, rejects, names = [], Counter(), Counter()
    scanned = accepted = 0
    with index_path.open(encoding="utf-8") as handle:
        for line in handle:
            if args.scan_limit and scanned >= args.scan_limit:
                break
            scanned += 1
            record = json.loads(line)
            if record.get("state") != "accepted":
                rejects["state_rejected"] += 1
                continue
            accepted += 1
            for reference in record.get("references") or []:
                score = reference.get("select_score") or {}
                reason = passes(score, args)
                if reason:
                    rejects[reason] += 1
                    continue
                located = frame_for_reference(record, reference)
                if located is None:
                    rejects["frame_not_found"] += 1
                    continue
                slot, frame, entities = located
                if len(entities) > args.max_objects_in_frame:
                    rejects["too_many_objects"] += 1
                    continue
                entity = next((e for e in entities if e.get("entity_id") == reference.get("entity_id")), None)
                if entity is None or not entity.get("bbox"):
                    rejects["bbox_not_found"] += 1
                    continue
                if entity.get("motion_blur"):
                    rejects["motion_blur"] += 1
                    continue

                name = reference.get("name") or entity.get("name") or "object"
                names[" ".join(name.lower().split())] += 1
                others = [e.get("name") for e in entities
                          if e.get("entity_id") != entity.get("entity_id") and e.get("name")]
                candidates.append({
                    "sample_id": record["sample_id"],
                    "original_id": record.get("original_id"),
                    "dataset": record.get("dataset"),
                    "video": record.get("video"),
                    "record_json": record.get("record_json"),
                    "artifact_dir": record.get("artifact_dir"),
                    "frame_path": frame.get("path"),
                    "frame_slot": slot,
                    "frame_index": frame.get("frame_index"),
                    "timestamp": frame.get("timestamp"),
                    "entity_id": entity.get("entity_id"),
                    "object_name": name,
                    "other_objects": others,
                    "bbox_norm": [round(float(v), 5) for v in entity["bbox"]],
                    "confidence": entity.get("confidence"),
                    "in_hand": entity.get("in_hand"),
                    "reference_image": reference.get("selected_cutout"),
                    "select_score": score,
                })
    return candidates, rejects, names, scanned, accepted


def balanced_sample(candidates, names, args):
    """Round-robin over shuffled categories: maximal spread, random within each category."""
    rng = random.Random(args.seed)
    known = set(names)
    buckets = defaultdict(list)
    for candidate in candidates:
        candidate["category"] = canon_category(candidate["object_name"], known)
        buckets[candidate["category"]].append(candidate)

    order = sorted(buckets)
    rng.shuffle(order)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    picked, per_sample = [], Counter()
    # A zero cap means exhaust every category. Round-robin still keeps the beginning of a
    # target-limited output balanced, while an unlimited run eventually visits everything.
    rounds = args.per_category or max(map(len, buckets.values()))
    for round_index in range(rounds):
        for category in order:
            if args.target and len(picked) >= args.target:
                return picked
            bucket = buckets[category]
            taken = 0
            while bucket and taken == 0:
                candidate = bucket.pop()
                if per_sample[candidate["sample_id"]] >= args.per_sample:
                    continue
                per_sample[candidate["sample_id"]] += 1
                picked.append(candidate)
                taken = 1
        if all(not bucket for bucket in buckets.values()):
            break
    return picked
