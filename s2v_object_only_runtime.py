"""Shared model, prompt, crop, and image-quality utilities for S2V inference.

This is an internal module. Use ``run_s2v_multiview_hunyuan_distil.py`` as the
inference entrypoint.
"""

import hashlib
import json
import sys
from pathlib import Path

from PIL import Image, ImageStat


ROOT = Path(__file__).resolve().parent
WENTAI_ROOT = ROOT.parent
DEFAULT_MODEL = WENTAI_ROOT / "weights" / "HunyuanImage-3.0-Instruct-Distil"
DISTIL_STEPS = 8


STRICT_SYSTEM_PROMPT = """You are a precise product-image editing engine.
Use the supplied photograph only as visual evidence for the identity and appearance of the named
target object. Execute the requested removal and reconstruction; do not preserve unrelated source
pixels or the source composition. The returned image contains one complete target object and no
human, hand, body, clothing, surrounding scene, prop, border, caption, poster layout, watermark, or
invented writing. Reconstruct occluded object regions from the visible geometry and material. Return
exactly one edited image and no explanatory text."""


PROMPTS = {
    "strict_v1": """IMAGE EDITING TASK

Target object: {object_name}

Extract the exact target object from the supplied photograph. Completely erase the person and every
human pixel, including the face, hair, body, clothing, arms, hands, fingers, skin and human shadows.
Erase the entire original background and every other object.

Where the target object is hidden by a hand, finger, body part or another object, reconstruct the
missing region so it continues the visible shape, geometry, colour, material, texture and genuine
surface markings of this same physical item. Keep its identity; do not redesign, duplicate or replace
it.

The finished image contains exactly one complete {object_name}, fully visible and centred at a useful
product-listing scale. The canvas is a uniform pure white RGB(255,255,255) seamless background, with
only a small soft contact shadow directly under the object. It is a clean, sharp, photorealistic
e-commerce product listing image. All empty space is plain white.""",
    "strict_v2": """Edit the reference photo into a plain product cutout of the {object_name}.

Keep only the same physical {object_name}. Remove the whole person, all hands and fingers, all body
parts and clothing, every other item, and the complete source scene. Rebuild portions of the
{object_name} covered in the reference by continuing its visible structure, proportions, material,
colour, texture and authentic markings. The reconstructed item is complete and physically coherent.

Place one complete {object_name} alone in the centre of an otherwise empty, uniformly pure-white
RGB(255,255,255) image. Show the full item with comfortable white margin and a subtle contact shadow.
The result is a straightforward photorealistic online-store product photo: one item on white, with no
graphic layout or added lettering.""",
    "strict_v3": """DIRECT IMAGE EDIT. The literal target is: {object_name}.

The supplied bbox crop is centred on the target. Use it only to identify the target's exact
appearance and boundaries. Keep only the same visible physical target named above. Remove all people, hands, fingers, bodies,
clothing, background and every non-target item. Target selection is literal: if the target is a tool,
remove the food, material, animal part, container or surface it touches; if the target is a component,
remove the larger parent product attached to it. An item is retained only when it is semantically part
of the exact target phrase "{object_name}".

Reconstruct target regions hidden by fingers or other removed content by continuing the target's
visible geometry, proportions, colour, material, texture and authentic markings. Produce one complete
and coherent target, not a duplicate and not a related product.

The output contains only one complete {object_name}, centred and fully visible on a uniform pure-white
RGB(255,255,255) background with comfortable margin and a subtle natural contact shadow. Plain,
sharp, photorealistic online-store product photography, with no layout or invented writing.""",
}


def output_job_name(source_name):
    for suffix in ("__object_only", "__person_swap", "__bg_swap", "__canonical_product"):
        if source_name.endswith(suffix):
            return source_name[:-len(suffix)] + "__object_only_product"
    return source_name + "__object_only_product"


def stable_seed(name, base_seed):
    digest = hashlib.sha256(f"{name}|{base_seed}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % (2**31 - 1)


def object_description(record):
    # object_description is supported for future enriched JSONL files; the current corpus uses
    # object_name. Never infer a description from job_name, entity_id or an image filename.
    value = record.get("object_description") or record.get("object_name")
    value = " ".join(str(value or "").split())
    if not value:
        raise ValueError(f"Job {record.get('job_name')} has no object_description/object_name")
    return value


def validate_distilled_checkpoint(model_id):
    required = ["config.json", "generation_config.json", "model.safetensors.index.json", "tokenizer.json"]
    missing = [name for name in required if not (model_id / name).is_file()]
    shards = sorted(model_id.glob("model-*-of-0032.safetensors"))
    if missing or len(shards) != 32:
        raise FileNotFoundError(
            f"Incomplete checkpoint at {model_id}: missing metadata={missing}, shards={len(shards)}/32"
        )
    config = json.loads((model_id / "config.json").read_text(encoding="utf-8"))
    if not (config.get("cfg_distilled") and config.get("use_meanflow")):
        raise ValueError(f"This runner requires the distilled MeanFlow checkpoint: {model_id}")


def build_max_memory(reserve_gib):
    import torch
    result = {}
    for index in range(torch.cuda.device_count()):
        free_bytes, _ = torch.cuda.mem_get_info(index)
        free_gib = free_bytes / 1024**3
        budget = free_gib - reserve_gib
        if budget <= 0:
            raise RuntimeError(f"GPU {index}: {free_gib:.1f} GiB free, below reserve {reserve_gib:.1f} GiB")
        result[index] = f"{budget:.0f}GiB"
        print(f"GPU {index}: {free_gib:.1f} GiB free, weight budget {budget:.0f} GiB", flush=True)
    return result


def load_model(model_id, args):
    sys.path.insert(0, str(ROOT))
    from hunyuan_image_3 import HunyuanImage3ForCausalMM
    extra = {}
    if args.gpu_reserve_gib > 0:
        extra["max_memory"] = build_max_memory(args.gpu_reserve_gib)
    model = HunyuanImage3ForCausalMM.from_pretrained(
        str(model_id), attn_implementation=args.attn_impl, dtype="auto", device_map="auto",
        moe_impl=args.moe_impl, moe_drop_tokens=True, local_files_only=True, **extra,
    )
    model.load_tokenizer(str(model_id))
    model.generation_config.diff_guidance_scale = args.guidance_scale
    return model


def read_manifest(path):
    entries = {}
    if not path.is_file():
        return entries
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("job_name"):
                entries[row["job_name"]] = row
    return entries


def border_whiteness(image, band_fraction=0.06, threshold=235):
    gray = image.convert("L")
    width, height = gray.size
    bw, bh = max(1, round(width * band_fraction)), max(1, round(height * band_fraction))
    bands = [gray.crop((0, 0, width, bh)), gray.crop((0, height - bh, width, height)),
             gray.crop((0, 0, bw, height)), gray.crop((width - bw, 0, width, height))]
    white = pixels = 0
    for band in bands:
        histogram = band.histogram()
        pixels += sum(histogram)
        white += sum(histogram[threshold:])
    return round(white / max(1, pixels), 4)


def bbox_crop(image, bbox_norm, padding):
    """Crop tightly around bbox with proportional padding; never change the crop's aspect ratio."""
    if not bbox_norm or len(bbox_norm) != 4:
        raise ValueError("bbox input mode requires a four-value bbox_norm")
    x0, y0, x1, y1 = [float(value) for value in bbox_norm]
    x0, x1 = sorted((x0 * image.width, x1 * image.width))
    y0, y1 = sorted((y0 * image.height, y1 * image.height))
    box_w, box_h = x1 - x0, y1 - y0
    if box_w < 2 or box_h < 2:
        raise ValueError(f"Invalid bbox after scaling: {bbox_norm}")
    pad_x, pad_y = box_w * padding, box_h * padding
    crop_box = (
        max(0, round(x0 - pad_x)), max(0, round(y0 - pad_y)),
        min(image.width, round(x1 + pad_x)), min(image.height, round(y1 + pad_y)),
    )
    return image.crop(crop_box), crop_box


def place_on_source_sized_canvas(image, source_size, occupancy=0.92):
    """Fit without distortion on a white canvas whose pixels exactly match the source frame."""
    max_size = (max(1, round(source_size[0] * occupancy)),
                max(1, round(source_size[1] * occupancy)))
    fitted = image.copy()
    fitted.thumbnail(max_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", source_size, "white")
    canvas.paste(fitted, ((source_size[0] - fitted.width) // 2,
                          (source_size[1] - fitted.height) // 2))
    return canvas


def compare_image(source, result):
    # result is guaranteed to have source.size, so comparison does not hide geometry changes.
    width, height = source.size
    sheet = Image.new("RGB", (width * 2 + 16, height), "white")
    sheet.paste(source, (0, 0))
    sheet.paste(result, (width + 16, 0))
    return sheet
