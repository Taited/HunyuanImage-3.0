#!/usr/bin/env python3
"""Export complete four-keyframe S2V examples for weekly reporting."""

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset" / "s2v_object_only-hunyuan-distil"


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def frame_index(entry):
    if entry.get("frame_index") is not None:
        return int(entry["frame_index"])
    match = re.search(r"__f(\d{6})__", entry["job_name"])
    return int(match.group(1)) if match else -1


def existing_output(entry, base, shard=None):
    original = Path(entry["output_path"])
    if original.is_file():
        return original
    if shard is None:
        candidate = base / "videos" / entry["sample_id"] / "edited" / original.name
    else:
        candidate = base / shard / "edited" / original.name
    return candidate if candidate.is_file() else None


def fit_image(path, size):
    with Image.open(path) as source:
        image = source.convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=5, help="Number of four-frame videos")
    parser.add_argument(
        "--output", default=str(ROOT / "weekly_report_s2v_four_keyframes")
    )
    args = parser.parse_args()

    main_root = DATASET / "full_bbox_direct_v2"
    extra_root = DATASET / "extra_views_latest_all"
    output = Path(args.output).resolve()
    individual = output / "individual"
    strips = output / "strips"
    individual.mkdir(parents=True, exist_ok=True)
    strips.mkdir(parents=True, exist_ok=True)

    anchors = {}
    for manifest in sorted(main_root.glob("shard_*/manifest.jsonl")):
        shard = manifest.parent.name
        for entry in read_jsonl(manifest):
            if entry.get("quality_status") != "pass_background_check":
                continue
            path = existing_output(entry, main_root, shard)
            if path:
                anchors[entry["source_job_name"]] = (entry, path)

    extras = defaultdict(list)
    for manifest in sorted(extra_root.glob("manifest.shard_*.jsonl")):
        for entry in read_jsonl(manifest):
            if entry.get("quality_status") != "pass_background_check":
                continue
            path = existing_output(entry, extra_root)
            if path:
                extras[entry.get("parent_job_name")].append((entry, path))

    selected = []
    seen_objects = set()
    for parent, anchor in anchors.items():
        views = extras.get(parent, [])
        unique = {frame_index(entry): (entry, path) for entry, path in views}
        if len(unique) != 3:
            continue
        rows = [anchor, *unique.values()]
        rows.sort(key=lambda item: frame_index(item[0]))
        object_name = anchor[0].get("object_name", "object")
        normalized = object_name.lower().strip()
        if normalized in seen_objects:
            continue
        selected.append((object_name, rows))
        seen_objects.add(normalized)
        if len(selected) == args.count:
            break

    if len(selected) < args.count:
        raise RuntimeError(f"Only found {len(selected)} complete, passing four-frame examples")

    tile_size = (360, 640)
    label_height = 54
    overview_rows = []
    export_manifest = []
    font = ImageFont.load_default()

    for video_number, (object_name, rows) in enumerate(selected, 1):
        panels = []
        sample_id = rows[0][0]["sample_id"]
        for view_number, (entry, source_path) in enumerate(rows, 1):
            frame = frame_index(entry)
            filename = f"video_{video_number:02d}_frame_{view_number:02d}_f{frame:06d}.jpg"
            destination = individual / filename
            shutil.copy2(source_path, destination)
            panel = Image.new("RGB", (tile_size[0], tile_size[1] + label_height), "white")
            panel.paste(fit_image(source_path, tile_size), (0, 0))
            draw = ImageDraw.Draw(panel)
            draw.rectangle((0, tile_size[1], tile_size[0], tile_size[1] + label_height), fill="#20242b")
            draw.text((12, tile_size[1] + 8), f"Keyframe {view_number} | frame {frame}", fill="white", font=font)
            draw.text((12, tile_size[1] + 28), object_name[:48], fill="#cbd5e1", font=font)
            panels.append(panel)
            export_manifest.append(
                {
                    "video_number": video_number,
                    "sample_id": sample_id,
                    "object_name": object_name,
                    "keyframe_number": view_number,
                    "frame_index": frame,
                    "source_path": str(source_path),
                    "export_path": str(destination),
                }
            )

        strip = Image.new("RGB", (tile_size[0] * 4, tile_size[1] + label_height), "white")
        for index, panel in enumerate(panels):
            strip.paste(panel, (index * tile_size[0], 0))
        strip.save(strips / f"video_{video_number:02d}_{sample_id}.jpg", quality=94)
        overview_rows.append(strip)

    gap = 12
    overview = Image.new(
        "RGB",
        (tile_size[0] * 4, len(overview_rows) * (tile_size[1] + label_height) + gap * (len(overview_rows) - 1)),
        "#d8dee8",
    )
    y = 0
    for row in overview_rows:
        overview.paste(row, (0, y))
        y += row.height + gap
    overview.save(output / "overview_5videos_20frames.jpg", quality=94)

    with (output / "selected_examples.jsonl").open("w", encoding="utf-8") as handle:
        for entry in export_manifest:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Exported {len(export_manifest)} images from {len(selected)} videos to {output}")


if __name__ == "__main__":
    main()
