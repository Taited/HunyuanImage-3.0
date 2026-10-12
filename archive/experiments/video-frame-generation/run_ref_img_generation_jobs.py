import argparse
import json
import re
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from hunyuan_image_3 import HunyuanImage3ForCausalMM


ROOT = Path(__file__).resolve().parent
WEIGHTS_ROOT = Path('/mnt/data04/144632/qiwei.huang@videorebirth.com/wentai/weights')
DEFAULT_MODEL_PATH = WEIGHTS_ROOT / "HunyuanImage-3.0-Instruct"
DEFAULT_JOBS_PATH = ROOT / "dataset" / "prompt_hbn_v2_mini_enhanced_A_kinematic.json"
DEFAULT_IMAGE_ROOT = ROOT / "dataset" / "hbn_test_images"
DEFAULT_OUTPUT_ROOT = ROOT / "dataset" / "ref_img_generation"


# Template used when the video prompt contains parseable timed shot segments, so we
# can describe the action that has happened up to the mid-point of the video.
MID_FRAME_PROMPT_TEMPLATE = """You are editing the provided reference image, which is the FIRST FRAME of a short video.

Generate one realistic still image that represents a MIDDLE FRAME of that video, a natural in-progress moment about halfway through, after the opening motion has already begun but before the video ends.

Depict the subject frozen mid-action as a single clean still photograph. Do not depict the opening frame, the final frame, motion trails, storyboard panels, multiple frames, or any text explaining the video. Preserve the original subject identity, product appearance, clothing, materials, background style, lighting, camera perspective, and image quality from the reference image unless the action below explicitly changes them.

Opening state (reference image):
{first_frame}

Action that has taken place from the start up to the middle of the video:
{mid_action}

Mid-frame edit instruction:
Render the single frame at the mid-point of the described video, capturing the action partway through (roughly the moment the first half of the motion has just completed). The output must be a single natural photograph, not a video frame sequence."""


# Fallback template when no timed shot segments can be parsed from the prompt.
MID_FRAME_FALLBACK_TEMPLATE = """You are editing the provided reference image, which is the FIRST FRAME of a short video.

Generate one realistic still image that represents a MIDDLE FRAME of that video, a natural in-progress moment about halfway through the following action, after it has begun but before it finishes.

Depict the subject frozen mid-action as a single clean still photograph. Do not depict the opening frame, the final frame, motion trails, storyboard panels, multiple frames, or any text explaining the video. Preserve the original subject identity, product appearance, clothing, materials, background style, lighting, camera perspective, and image quality from the reference image unless the action below explicitly changes them.

Video description:
{video_prompt}

Mid-frame edit instruction:
Render the single frame at the mid-point of the described video, capturing the action partway through. The output must be a single natural photograph, not a video frame sequence."""


# Matches a timed shot segment: [Shot 1] [00:00-00:03] [Close-Up, ...] <description>
SHOT_RE = re.compile(
    r"\[Shot[^\]]*\]\s*\[(\d{2}):(\d{2})-(\d{2}):(\d{2})\]\s*(?:\[[^\]]*\])?\s*"
    r"(.*?)(?=\[Shot\b|\[Last frame\]|\Z)",
    flags=re.IGNORECASE | re.DOTALL,
)


def parse_args():
    parser = argparse.ArgumentParser("Batch first-frame image editing to video middle-frame reference generation")
    parser.add_argument("--jobs", default=str(DEFAULT_JOBS_PATH), help="Path to enhanced prompt JSON.")
    parser.add_argument("--image-root", default=str(DEFAULT_IMAGE_ROOT), help="Directory containing <job_key>.jpg images.")
    parser.add_argument("--model-id", default=str(DEFAULT_MODEL_PATH), help="Local model directory.")
    parser.add_argument("--model-name", default=None, help="Output subfolder name. Defaults to model directory name.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Root output directory.")
    parser.add_argument("--seeds", default="0", help="Comma-separated seed values.")
    parser.add_argument(
        "--image-size",
        default="match",
        help=(
            "Output size. 'match' (default): snap to the preset aspect ratio closest to each "
            "input image. 'auto': let the model choose the ratio. Or an explicit 'HxW' / 'W:H'."
        ),
    )
    parser.add_argument("--diff-infer-steps", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--attn-impl", default="sdpa", choices=["sdpa", "flash_attention_2"])
    parser.add_argument("--moe-impl", default="eager", choices=["eager", "flashinfer"])
    parser.add_argument("--use-system-prompt", default="en_unified")
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", help="Regenerate files that already exist.")
    parser.add_argument("--skip-missing", action="store_true", help="Skip jobs whose input image is missing.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned outputs without loading the model.")
    return parser.parse_args()


def resolve_path(path_value, base_dir=ROOT):
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def safe_name(value):
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return value.strip("._-") or "image"


def _clean(text):
    return " ".join(text.split()).strip()


def extract_first_frame_desc(video_prompt):
    match = re.search(
        r"\[First frame\]\s*:?\s*(.*?)(?=\[Shot\b|\[Last frame\]|\Z)",
        video_prompt,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return _clean(match.group(1)) if match else ""


def parse_shots(video_prompt):
    """Parse timed shot segments into a list of {start, end, desc} (seconds)."""
    shots = []
    for m in SHOT_RE.finditer(video_prompt):
        start = int(m.group(1)) * 60 + int(m.group(2))
        end = int(m.group(3)) * 60 + int(m.group(4))
        desc = _clean(m.group(5))
        if desc:
            shots.append({"start": start, "end": end, "desc": desc})
    return shots


def extract_mid_action(video_prompt):
    """Concatenate the shot descriptions covering the first half of the video.

    Returns the combined action text leading up to the mid-point, or None when no
    timed shot segments can be parsed (caller then uses the fallback template).
    """
    shots = parse_shots(video_prompt)
    if not shots:
        return None
    total = shots[-1]["end"]
    mid = total / 2.0
    # Keep every segment that begins before the mid-point (i.e. the first half of
    # the motion). For the common 2-segment [0-3]/[3-6] case this yields segment 1,
    # whose end-state is exactly the middle frame.
    covered = [sh["desc"] for sh in shots if sh["start"] < mid]
    if not covered:
        covered = [shots[0]["desc"]]
    return " ".join(covered)


def build_mid_frame_prompt(video_prompt):
    mid_action = extract_mid_action(video_prompt)
    if mid_action is None:
        return MID_FRAME_FALLBACK_TEMPLATE.format(video_prompt=_clean(video_prompt))
    first_frame = extract_first_frame_desc(video_prompt) or "(see the provided reference image)"
    return MID_FRAME_PROMPT_TEMPLATE.format(first_frame=first_frame, mid_action=mid_action)


def load_jobs(jobs_path, image_root, skip_missing=False):
    with jobs_path.open("r", encoding="utf-8") as f:
        jobs = json.load(f)

    if not isinstance(jobs, dict):
        raise ValueError("Prompt JSON must be an object keyed by image name.")

    normalized_jobs = []
    missing_inputs = []
    for idx, (image_key, job) in enumerate(jobs.items()):
        if not isinstance(job, dict):
            raise ValueError(f"Job {image_key!r} must be an object.")
        if not job.get("prompt"):
            raise ValueError(f"Job {image_key!r} is missing required field: prompt")

        image_path = image_root / f"{image_key}.jpg"
        if not image_path.exists():
            if skip_missing:
                missing_inputs.append(str(image_path))
                continue
            raise FileNotFoundError(f"Input image does not exist for job {image_key!r}: {image_path}")

        normalized_jobs.append(
            {
                "job_id": idx,
                "image_key": image_key,
                "image_name": safe_name(image_key),
                "input": str(image_path),
                "video_prompt": job["prompt"],
                "mid_action": extract_mid_action(job["prompt"]),
                "prompt": build_mid_frame_prompt(job["prompt"]),
            }
        )

    if missing_inputs:
        print(f"Skipped {len(missing_inputs)} jobs with missing input images.")

    return normalized_jobs


def parse_seeds(seeds_arg):
    seeds = [int(item.strip()) for item in seeds_arg.split(",") if item.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one seed value.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("--seeds contains duplicate seed values.")
    return seeds


def load_model(model_id, args):
    kwargs = dict(
        attn_implementation=args.attn_impl,
        dtype="auto",
        device_map="auto",
        moe_impl=args.moe_impl,
        moe_drop_tokens=True,
        local_files_only=True,
    )
    model = HunyuanImage3ForCausalMM.from_pretrained(str(model_id), **kwargs)
    model.load_tokenizer(str(model_id))
    return model


def resolve_image_size(image_size_arg, input_path):
    """Resolve the ``image_size`` passed to ``generate_image`` for one job.

    When ``--image-size match`` is used, return the input image's own "HxW" so the
    model snaps the output to the preset aspect ratio closest to the original.
    Otherwise pass the argument through ('auto' or an explicit 'HxW' / 'W:H').
    """
    if image_size_arg != "match":
        return image_size_arg
    with Image.open(input_path) as im:
        width, height = im.size
    return f"{height}x{width}"


def build_output_path(output_dir, job, seed_value, multi_seed=False):
    # Name the output after the input image. Only disambiguate by seed when
    # more than one seed is requested (otherwise files would overwrite).
    suffix = f"_seed_{seed_value}" if multi_seed else ""
    filename = f"{job['image_key']}{suffix}.png"
    return output_dir / filename


def main():
    args = parse_args()
    jobs_path = resolve_path(args.jobs)
    image_root = resolve_path(args.image_root)
    model_id = resolve_path(args.model_id)
    output_root = resolve_path(args.output_root)
    seeds = parse_seeds(args.seeds)

    if not jobs_path.exists():
        raise FileNotFoundError(f"Jobs JSON does not exist: {jobs_path}")
    if not image_root.exists():
        raise FileNotFoundError(f"Image root does not exist: {image_root}")
    if not model_id.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_id}")

    jobs = load_jobs(jobs_path, image_root, skip_missing=args.skip_missing)
    model_name = safe_name(args.model_name or model_id.name)
    output_dir = output_root / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(jobs)} jobs from: {jobs_path}")
    print(f"Using first-frame images from: {image_root}")
    print(f"Saving outputs to: {output_dir}")

    multi_seed = len(seeds) > 1
    if args.dry_run:
        for job in jobs:
            for seed_value in seeds:
                print(build_output_path(output_dir, job, seed_value, multi_seed))
        return

    model = load_model(model_id, args)
    print(f"Loaded local model from: {model_id}")

    total = len(jobs) * len(seeds)
    with tqdm(total=total, desc="Generating mid frames", unit="image") as progress:
        for job in jobs:
            for seed_value in seeds:
                save_path = build_output_path(output_dir, job, seed_value, multi_seed)
                progress.set_postfix(job=f"{job['job_id']:03d}", image=job["image_key"], seed=seed_value)

                if save_path.exists() and not args.overwrite:
                    progress.update(1)
                    continue

                _, samples = model.generate_image(
                    prompt=job["prompt"],
                    image=job["input"],
                    seed=seed_value,
                    image_size=resolve_image_size(args.image_size, job["input"]),
                    use_system_prompt=args.use_system_prompt,
                    bot_task="image",
                    infer_align_image_size=True,
                    diff_infer_steps=args.diff_infer_steps,
                    max_new_tokens=args.max_new_tokens,
                    verbose=args.verbose,
                )
                samples[0].save(save_path)
                progress.update(1)

    print("Done.")


if __name__ == "__main__":
    main()
