"""Flag canonical renders whose background is not clean white.

The failure mode that matters here is a render that kept the original scene or
the person instead of isolating the object.  Those always show up as a border
that is not white, so scoring the frame edge separates them from the good
renders cheaply and without a model.

Writes ``audit.jsonl`` plus contact sheets of the worst offenders for review.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = ROOT / "dataset" / "s2v_keyframe_recontext-hunyuan-canonical"


def border_whiteness(path, band=0.06, threshold=235):
    """Fraction of pixels in the frame's outer band that are near-white."""
    image = Image.open(path).convert("L")
    pixels = np.asarray(image, dtype=np.uint8)
    height, width = pixels.shape
    bh = max(1, int(height * band))
    bw = max(1, int(width * band))
    border = np.concatenate([
        pixels[:bh, :].ravel(), pixels[-bh:, :].ravel(),
        pixels[:, :bw].ravel(), pixels[:, -bw:].ravel(),
    ])
    return float((border >= threshold).mean())


def contact_sheet(paths, cell=380):
    images = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((cell, cell), Image.LANCZOS)
        images.append(image)
    if not images:
        return None
    columns = min(len(images), 4)
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell, rows * cell), (238, 238, 238))
    for index, image in enumerate(images):
        x = (index % columns) * cell + (cell - image.width) // 2
        y = (index // columns) * cell + (cell - image.height) // 2
        sheet.paste(image, (x, y))
    return sheet


def main():
    parser = argparse.ArgumentParser("Audit canonical product renders")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--min-white", type=float, default=0.90,
                        help="Border-whiteness below this is flagged for review.")
    parser.add_argument("--sheets", type=int, default=16, help="Worst-N to tile into review sheets.")
    args = parser.parse_args()

    output_root = Path(args.output_root).expanduser().resolve()
    entries = [json.loads(line) for line in (output_root / "manifest.jsonl").open(encoding="utf-8")
               if line.strip()]

    scored = []
    for entry in entries:
        path = Path(entry["output_path"])
        if not path.is_file():
            continue
        scored.append({"job_name": entry["job_name"], "object_name": entry.get("object_name"),
                       "white": round(border_whiteness(path), 4),
                       "output_path": str(path), "input_path": entry["input_path"]})
    scored.sort(key=lambda item: item["white"])

    audit_path = output_root / "audit.jsonl"
    with audit_path.open("w", encoding="utf-8") as handle:
        for item in scored:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    flagged = [item for item in scored if item["white"] < args.min_white]
    values = np.array([item["white"] for item in scored])
    print(f"Audited {len(scored)} renders")
    print(f"Border whiteness: mean {values.mean():.3f}, median {np.median(values):.3f}")
    for cut in (0.99, 0.95, 0.90, 0.75):
        print(f"  >= {cut:.2f}: {(values >= cut).sum():4d}  ({(values >= cut).mean() * 100:.1f}%)")
    print(f"Flagged below {args.min_white}: {len(flagged)}")
    for item in flagged[:20]:
        print(f"  {item['white']:.3f}  {item['object_name']}")

    worst = flagged[: args.sheets]
    if worst:
        sheet = contact_sheet([item["output_path"] for item in worst])
        sheet.save(output_root / "audit_worst_output.jpg", quality=90)
        sheet = contact_sheet([item["input_path"] for item in worst])
        sheet.save(output_root / "audit_worst_input.jpg", quality=90)
        print(f"Review sheets: {output_root}/audit_worst_output.jpg (+ _input)")
    print(f"Audit: {audit_path}")


if __name__ == "__main__":
    main()
