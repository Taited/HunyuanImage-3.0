"""Minimal multi-keyframe -> white-background three-view test for HunyuanImage 3 Distil."""

import argparse
from pathlib import Path

from PIL import Image

from hunyuan_image_3 import HunyuanImage3ForCausalMM


SYSTEM_PROMPT = """You are a precise multi-reference product rendering engine. All reference
images show the same physical object at different moments or angles. Combine their visual evidence
to preserve that object's identity, shape, proportions, materials, colours, texture and real
markings. Return exactly one image and no explanatory text."""

PROMPT = """Use all supplied keyframes as references for the same {object_name}. Remove every
person, hand, finger, body part, prop and original scene. Reconstruct only genuinely occluded parts
from the consistent evidence across the references.

Create one clean orthographic three-view product sheet containing exactly three depictions of this
same {object_name}, arranged left-to-right as: front view, right-side view, back view. Keep identical
scale, upright orientation, baseline, lighting, appearance and physical identity in all three views.
Leave generous even spacing; do not overlap or crop any view. The entire canvas and all space between
the views must be uniform pure white RGB(255,255,255), with only very subtle contact shadows. No
people, hands, scene, border, divider, caption, label, arrows, dimensions, logo added by the renderer,
or other text. Photorealistic e-commerce product photography, not a drawing or diagram."""


def parse_args():
    parser = argparse.ArgumentParser("Multi-keyframe three-view smoke test")
    parser.add_argument("--images", nargs="+", required=True, help="2+ keyframes of one object")
    parser.add_argument("--object-name", required=True)
    parser.add_argument("--model-id", default="../weights/HunyuanImage-3.0-Instruct-Distil")
    parser.add_argument("--output", default="outputs/multiframe_three_view.png")
    parser.add_argument("--image-size", default="768x1536", help="Landscape HxW target")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    return parser.parse_args()


def border_white_ratio(image, band=0.05, threshold=245):
    gray = image.convert("L")
    w, h = gray.size
    bw, bh = max(1, round(w * band)), max(1, round(h * band))
    regions = (gray.crop((0, 0, w, bh)), gray.crop((0, h - bh, w, h)),
               gray.crop((0, 0, bw, h)), gray.crop((w - bw, 0, w, h)))
    histograms = [region.histogram() for region in regions]
    white = sum(sum(hist[threshold:]) for hist in histograms)
    total = sum(sum(hist) for hist in histograms)
    return white / total


def main():
    args = parse_args()
    if len(args.images) < 2:
        raise ValueError("--images needs at least two keyframes")
    image_paths = [str(Path(path).expanduser().resolve()) for path in args.images]
    missing = [path for path in image_paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing input image: {missing[0]}")

    model = HunyuanImage3ForCausalMM.from_pretrained(
        args.model_id,
        attn_implementation="sdpa",
        dtype="auto",
        device_map="auto",
        moe_impl="eager",
        moe_drop_tokens=True,
        local_files_only=True,
    )
    model.load_tokenizer(args.model_id)
    model.generation_config.diff_guidance_scale = args.guidance_scale

    _, samples = model.generate_image(
        prompt=PROMPT.format(object_name=args.object_name),
        image=image_paths,
        seed=args.seed,
        image_size=args.image_size,
        use_system_prompt="custom",
        system_prompt=SYSTEM_PROMPT,
        bot_task="image",
        infer_align_image_size=False,
        diff_infer_steps=args.steps,
        max_new_tokens=2048,
        verbose=1,
    )
    result = samples[0].convert("RGB")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.save(output)
    print(f"saved={output.resolve()}")
    print(f"size={result.size}, border_white_ratio={border_white_ratio(result):.4f}")


if __name__ == "__main__":
    main()
