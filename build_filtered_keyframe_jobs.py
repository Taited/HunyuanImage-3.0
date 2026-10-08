#!/usr/bin/env python3
"""Join MASt3R kept frame IDs with Gemma boxes for Hunyuan multi-view jobs.

One keyframe directory is one object group, matching MASt3R's primary-object
assumption. This does not perform instance tracking when a video has multiple
objects. The largest-area detection is selected exactly as in MASt3R.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

from PIL import Image

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset"


def box_area(detection):
    box = detection.get("bbox_2d")
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError("bbox_2d must contain four pixel coordinates")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for v in box):
        raise ValueError("bbox_2d coordinates must be finite numbers")
    x0, y0, x1, y1 = box
    return max(0, x1 - x0) * max(0, y1 - y0)


def build_jobs(filtered_paths, bbox_path, keyframes_dir):
    """Validate all kept frames before returning a deterministic list of jobs."""
    sessions = {}
    for session in json.loads(Path(bbox_path).read_text())["results"].values():
        name = session.get("keyframe_dir")
        if not name:
            continue
        if name in sessions:
            raise ValueError(f"Duplicate bbox keyframe_dir: {name}")
        sessions[name] = session

    directories = {}
    for path in filtered_paths:
        for item in json.loads(Path(path).read_text())["dirs"]:
            name = item["keyframe_dir"]
            if name in directories:
                raise ValueError(f"Duplicate filtered directory: {name}; do not mix merged and shard files")
            if not name or Path(name).name != name or name in (".", ".."):
                raise ValueError(f"Invalid keyframe_dir: {name!r}")
            directories[name] = item

    jobs = []
    for name, item in sorted(directories.items()):
        ids = item["kept_frame_ids"]
        if any(type(fid) is not int or fid < 0 for fid in ids) or len(ids) != len(set(ids)):
            raise ValueError(f"{name}: kept_frame_ids must be unique nonnegative integers")
        if item.get("n_kept", len(ids)) != len(ids):
            raise ValueError(f"{name}: n_kept disagrees with kept_frame_ids")
        if not ids:
            continue
        if name not in sessions:
            raise ValueError(f"{name}: missing Gemma bbox session")
        session = sessions[name]
        # Hash avoids collisions after filename sanitization, including punctuation.
        slug = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")[:96] or "object"
        sample_id = slug + "-" + hashlib.sha256(name.encode()).hexdigest()[:16]
        rows = []
        for fid in sorted(ids):
            filename = f"frame_{fid:06d}.jpg"
            frame = session["frames"].get(filename)
            if frame is None or not frame.get("detections"):
                raise ValueError(f"{name}/{filename}: missing frame metadata or detections")
            det = max(frame["detections"], key=box_area)
            if box_area(det) <= 0 or det.get("motion_blur") is True:
                raise ValueError(f"{name}/{filename}: invalid or blurred primary detection")
            image_path = (Path(keyframes_dir) / name / filename).resolve()
            with Image.open(image_path) as image:
                width, height = image.size
            if (frame.get("width"), frame.get("height")) != (width, height):
                raise ValueError(f"{name}/{filename}: image dimensions disagree with bbox metadata")
            x0, y0, x1, y1 = det["bbox_2d"]
            x0, x1 = max(0, x0), min(width, x1)
            y0, y1 = max(0, y0), min(height, y1)
            if x1 - x0 < 2 or y1 - y0 < 2:
                raise ValueError(f"{name}/{filename}: bbox outside image or smaller than two pixels")
            label = det.get("label") or next(iter(session.get("objects") or []), "")
            if not isinstance(label, str) or not label.strip():
                raise ValueError(f"{name}/{filename}: missing object name")
            rows.append({
                "job_name": f"{sample_id}__frame_{fid:06d}",
                "sample_id": sample_id,
                "entity_id": "primary_object",
                "object_name": label.strip(),
                "frame_path": str(image_path),
                "frame_index": fid,
                "bbox_norm": [x0 / width, y0 / height, x1 / width, y1 / height],
                "keyframe_dir": name,
                "video": session.get("gemma_key"),
                "source": "mast3r_filtered_keyframes",
                "_area": box_area(det),
            })
        anchor = min(rows, key=lambda row: (-row["_area"], row["frame_index"]))
        ordered = [anchor] + [row for row in rows if row is not anchor]
        for rank, row in enumerate(ordered):
            row.pop("_area")
            row.update(view_rank=rank, extra_view_rank=rank, is_anchor=rank == 0,
                       parent_job_name=anchor["job_name"], anchor_frame_index=anchor["frame_index"])
            jobs.append(row)
    if not jobs:
        raise ValueError("No kept frames to export")
    return jobs


def write_jobs(path, rows):
    """Replace the output only after every input has passed validation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", delete=False) as handle:
            temporary = Path(handle.name)
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filtered-json", type=Path, nargs="+",
                        default=[DATASET / "gemma_keyframe_filtered/filtered.json"],
                        help="One merged report OR all filtered.part*of*.json shards")
    parser.add_argument("--bbox-json", type=Path, default=DATASET / "gemma_keyframe_bbox.json")
    parser.add_argument("--keyframes-dir", type=Path,
                        default=DATASET / "datatang_session2_videos_fps24_012_keyframes",
                        help="Original keyframes, or the MASt3R --copy-frames output directory")
    parser.add_argument("--output", type=Path,
                        default=DATASET / "s2v_keyframe_recontext/jobs.filtered.jsonl")
    args = parser.parse_args()
    rows = build_jobs(args.filtered_json, args.bbox_json, args.keyframes_dir)
    write_jobs(args.output, rows)
    print(f"jobs={len(rows)} groups={len({r['sample_id'] for r in rows})} output={args.output}")


if __name__ == "__main__":
    main()
