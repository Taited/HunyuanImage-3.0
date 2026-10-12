#!/usr/bin/env python3
"""Build side-by-side input/edit pairs from Gemma consistency reviews.

Selection is done per consistency group (sample_id, entity_id).  Groups with
at least ``--primary-min-frames`` passing frames are preferred.  If those do
not fill ``--target-videos``, groups with at least ``--fallback-min-frames``
are used as a fallback.  A fixed seed makes both tiers deterministic.

The left panel is the exact source/reference used by the consistency model
(``model_input_path``, falling back to ``input_path``); the right panel is
``edited_path``.  Images are aspect-ratio preserving thumbnails centered on
equal-sized panels.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps


DEFAULT_SOURCE = Path("dataset/gemma_edited_frame_consistency.jsonl")
DEFAULT_OUTPUT = Path("dataset/annotation_10k")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-videos", type=int, default=10_000)
    parser.add_argument("--primary-min-frames", type=int, default=3)
    parser.add_argument("--fallback-min-frames", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--panel-size", type=int, default=900)
    parser.add_argument("--jpeg-quality", type=int, default=94)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned selection without creating any files.",
    )
    return parser.parse_args()


def load_passing_groups(source: Path) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    with source.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {source}:{line_number}") from exc
            if row.get("status") != "ok" or row.get("verdict", {}).get("pass") is not True:
                continue
            group_id = row.get("group_id")
            if not group_id or not row.get("sample_id"):
                raise ValueError(f"Missing group_id/sample_id at {source}:{line_number}")
            groups[group_id].append(row)

    for rows in groups.values():
        rows.sort(key=lambda row: (row.get("frame_index") is None,
                                   row.get("frame_index", -1),
                                   row.get("review_id", "")))
    return dict(groups)


def choose_one_group_per_video(groups: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Use the strongest entity group if a video unexpectedly has several."""
    selected: dict[str, tuple[str, list[dict]]] = {}
    for group_id, rows in groups.items():
        sample_id = rows[0]["sample_id"]
        current = selected.get(sample_id)
        if current is None or (len(rows), group_id) > (len(current[1]), current[0]):
            selected[sample_id] = (group_id, rows)
    return {sample_id: item[1] for sample_id, item in selected.items()}


def source_path(row: dict) -> Path:
    raw = row.get("model_input_path") or row.get("input_path")
    if not raw:
        raise ValueError(f"No input path for review_id={row.get('review_id')}")
    return Path(raw)


def group_has_all_images(rows: list[dict]) -> tuple[bool, str | None]:
    for row in rows:
        for role, path in (("input", source_path(row)),
                           ("edited", Path(row.get("edited_path") or ""))):
            if not path.is_file():
                return False, f"missing {role}: {path}"
    return True, None


def select_groups(
    videos: dict[str, list[dict]],
    target: int,
    primary_min: int,
    fallback_min: int,
    seed: int,
) -> tuple[list[tuple[str, list[dict], str]], Counter, list[tuple[str, str]]]:
    if target < 1:
        raise ValueError("--target-videos must be positive")
    if primary_min < fallback_min or fallback_min < 1:
        raise ValueError("Require primary-min-frames >= fallback-min-frames >= 1")

    rng = random.Random(seed)
    primary = [(sample_id, rows) for sample_id, rows in videos.items()
               if len(rows) >= primary_min]
    fallback = [(sample_id, rows) for sample_id, rows in videos.items()
                if fallback_min <= len(rows) < primary_min]
    primary.sort(key=lambda item: item[0])
    fallback.sort(key=lambda item: item[0])
    rng.shuffle(primary)
    rng.shuffle(fallback)

    selected: list[tuple[str, list[dict], str]] = []
    missing: list[tuple[str, str]] = []
    available = Counter()
    for tier, candidates in (("primary", primary), ("fallback", fallback)):
        available[tier] = len(candidates)
        for sample_id, rows in candidates:
            if len(selected) >= target:
                break
            valid, reason = group_has_all_images(rows)
            if not valid:
                missing.append((sample_id, reason or "missing image"))
                continue
            selected.append((sample_id, rows, tier))

    if len(selected) < target:
        raise RuntimeError(
            f"Only {len(selected)} image-complete videos satisfy the thresholds; "
            f"requested {target}. Missing-image groups skipped: {len(missing)}"
        )
    return selected, available, missing


def fit_panel(image: Image.Image, size: int) -> Image.Image:
    contained = ImageOps.contain(image.convert("RGB"), (size, size),
                                 Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (size, size), "#161616")
    panel.paste(contained, ((size - contained.width) // 2,
                            (size - contained.height) // 2))
    return panel


def make_pair(row: dict, panel_size: int) -> Image.Image:
    with Image.open(source_path(row)) as input_image, Image.open(row["edited_path"]) as edited:
        left = fit_panel(input_image, panel_size)
        right = fit_panel(edited, panel_size)
    canvas = Image.new("RGB", (panel_size * 2, panel_size), "#161616")
    canvas.paste(left, (0, 0))
    canvas.paste(right, (panel_size, 0))
    return canvas


def frame_filename(row: dict) -> str:
    index = row.get("frame_index")
    if not isinstance(index, int) or index < 0:
        raise ValueError(f"Invalid frame_index for review_id={row.get('review_id')}: {index}")
    return f"frame-{index:06d}.jpg"


def write_outputs(
    output: Path,
    selected: list[tuple[str, list[dict], str]],
    args: argparse.Namespace,
    available: Counter,
    missing: list[tuple[str, str]],
) -> None:
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {output}. Choose a new --output path."
        )
    output.mkdir(parents=True)
    manifest_path = output / "manifest.jsonl"
    tier_counts = Counter()
    frame_counts = Counter()

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for ordinal, (sample_id, rows, tier) in enumerate(selected, 1):
            video_dir = output / sample_id
            video_dir.mkdir()
            tier_counts[tier] += 1
            frame_counts[len(rows)] += 1
            for row in rows:
                filename = frame_filename(row)
                destination = video_dir / filename
                make_pair(row, args.panel_size).save(
                    destination,
                    format="JPEG",
                    quality=args.jpeg_quality,
                    subsampling=0,
                )
                record = {
                    "selection_index": ordinal,
                    "selection_tier": tier,
                    "sample_id": sample_id,
                    "group_id": row["group_id"],
                    "entity_id": row.get("entity_id"),
                    "object_name": row.get("object_name"),
                    "frame_index": row["frame_index"],
                    "pair_path": str(destination),
                    "left_input_path": str(source_path(row)),
                    "right_edited_path": row["edited_path"],
                    "review_id": row.get("review_id"),
                    "verdict": row.get("verdict"),
                }
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "source": str(args.source),
        "output": str(output),
        "seed": args.seed,
        "target_videos": args.target_videos,
        "primary_min_frames": args.primary_min_frames,
        "fallback_min_frames": args.fallback_min_frames,
        "panel_size": args.panel_size,
        "layout": "left=model_input_path (fallback input_path), right=edited_path",
        "selected_videos": len(selected),
        "selected_frames": sum(len(rows) for _, rows, _ in selected),
        "selected_by_tier": dict(sorted(tier_counts.items())),
        "selected_by_passing_frame_count": {
            str(key): value for key, value in sorted(frame_counts.items())
        },
        "available_candidates": dict(sorted(available.items())),
        "missing_image_groups_skipped": len(missing),
        "missing_image_examples": missing[:20],
    }
    with (output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> int:
    args = parse_args()
    groups = load_passing_groups(args.source)
    videos = choose_one_group_per_video(groups)
    selected, available, missing = select_groups(
        videos,
        args.target_videos,
        args.primary_min_frames,
        args.fallback_min_frames,
        args.seed,
    )
    planned = Counter(tier for _, _, tier in selected)
    print(
        f"Passing groups={len(groups)}, videos={len(videos)}, "
        f"selected={len(selected)} ({dict(planned)}), "
        f"frames={sum(len(rows) for _, rows, _ in selected)}, "
        f"missing-image groups skipped={len(missing)}",
        file=sys.stderr,
    )
    if not args.dry_run:
        write_outputs(args.output, selected, args, available, missing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
