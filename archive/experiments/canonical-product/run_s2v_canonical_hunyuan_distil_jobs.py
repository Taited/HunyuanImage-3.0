"""Render S2V keyframe objects as canonical white-background product images.

This is the HunyuanImage-3.0-Instruct-Distil counterpart of DiffSynth-Studio's
``run_s2v_recontext_qwen_image_edit_2511_canonical_jobs.py``.  It consumes the
same ``jobs.jsonl`` and the *same already-cropped* input frames the Qwen run
used (``dataset/s2v_keyframe_recontext/input/<job_name>.jpg``), so the two
models can be compared on byte-identical inputs.

The task: keep only ``object_name`` from the input frame, remove the person and
the whole source scene, and re-render the object alone on a pure white seamless
studio background -- an e-commerce style canonical product shot.

The script is deliberately self-contained (no DiffSynth import) and shardable,
so several workers can each own a disjoint slice of the job list on their own
pair of GPUs.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent
WENTAI_ROOT = ROOT.parent
DIFFSYNTH_ROOT = WENTAI_ROOT / "DiffSynth-Studio"
DEFAULT_MODEL = WENTAI_ROOT / "weights" / "HunyuanImage-3.0-Instruct-Distil"
DEFAULT_JOBS = DIFFSYNTH_ROOT / "dataset" / "s2v_keyframe_recontext" / "jobs.jsonl"
DEFAULT_INPUT_DIR = DIFFSYNTH_ROOT / "dataset" / "s2v_keyframe_recontext" / "input"
DEFAULT_OUTPUT_ROOT = ROOT / "dataset" / "s2v_keyframe_recontext-hunyuan-canonical"

DISTIL_STEPS = 8
VARIANT = "canonical_product"

# ---------------------------------------------------------------------------
# Prompt profiles
# ---------------------------------------------------------------------------

# Profile "qwen_v1": the Qwen canonical prompt, verbatim, so the first Hunyuan
# run is a like-for-like control against the Qwen canonical renders.
CANONICAL_PRODUCT_PROMPT = (
    "Create a canonical e-commerce product photograph of only the {object_name} shown in the input. "
    "Remove the entire person, including all hands, fingers, arms and body parts. Remove every other "
    "object and remove the original background completely. Reconstruct only the parts of the "
    "{object_name} that are hidden by the person, keeping the reconstruction consistent with its "
    "visible structure.\n\n"
    "Keep the {object_name} recognizable as the exact same item: preserve its shape, proportions, "
    "colour, material, texture, finish and all clearly visible surface details. Preserve text, labels "
    "and markings that are genuinely visible on it, but invent no new text, logo, label, pattern, "
    "decoration, component or duplicate.\n\n"
    "Show exactly one {object_name}, whole and fully visible, upright in its natural resting "
    "orientation, centred in the frame, at a useful catalogue scale, viewed straight-on or from a "
    "slight three-quarter angle. Use a pure white, clean, seamless studio background (#FFFFFF) with "
    "soft even product lighting and only a subtle natural contact shadow directly beneath the item. "
    "No person, no body parts, no props, no supporting hand, no table, no floor line, no room, no "
    "scenery, no border, no caption and no watermark. Photorealistic, sharp, clean-edged, "
    "catalogue-grade product image."
)

# Profile "hunyuan_v2": purely positive description of the *target* image.
#
# The qwen_v1 wording ends in a long "no person, no props, ... no caption, no
# watermark" list.  Hunyuan's direct-image path has no negative-prompt channel,
# so that list reads as content to include: the first smoke run answered it by
# stamping an invented brand caption across a render.  Everything here is
# therefore phrased as what the finished photograph *contains*.
HUNYUAN_V2_PROMPT = (
    "A clean e-commerce catalogue photograph of a single {object_name}, isolated by itself on a "
    "pure white seamless studio background.\n\n"
    "It is the identical {object_name} from the reference image: the same shape, proportions, "
    "colour, material, texture, finish, and the same surface details, labels and printed text that "
    "are visible on it there. Whatever part of it the reference image hid is completed so the item "
    "reads as whole and undamaged.\n\n"
    "The {object_name} stands alone and complete at the centre of an empty white frame, upright in "
    "its natural resting orientation, filling most of the picture at catalogue scale, seen "
    "straight-on or from a slight three-quarter angle, under soft even studio lighting with a "
    "subtle contact shadow directly beneath it. The whole background is plain uniform white and "
    "empty. Sharp, clean-edged, photorealistic product photography."
)

# Profile "hunyuan_v3": one decisive removal instruction, then the same positive
# target description.  For frames where v2 leaves a hand or the original room in
# place, naming the removal once is what makes the edit happen; the negation is
# kept to a single short sentence so it cannot dominate the conditioning.
HUNYUAN_V3_PROMPT = (
    "Take only the {object_name} out of the reference photograph and re-shoot it by itself as a "
    "product image. The person holding it and the room around it are entirely absent from the "
    "result.\n\n"
    "A clean e-commerce catalogue photograph of a single {object_name} on a pure white seamless "
    "studio background. It is the identical {object_name} from the reference image: the same shape, "
    "proportions, colour, material, texture, finish, and the same surface details, labels and "
    "printed text that are visible on it there. Whatever part of it the reference image hid is "
    "completed so the item reads as whole and undamaged.\n\n"
    "It stands alone and complete at the centre of an empty white frame, upright in its natural "
    "resting orientation, filling most of the picture at catalogue scale, seen straight-on or from "
    "a slight three-quarter angle, under soft even studio lighting with a subtle contact shadow "
    "directly beneath it. The whole background is plain uniform white and empty. Sharp, "
    "clean-edged, photorealistic product photography."
)

# Profile "hunyuan_v4": qwen_v1's imperative edit framing, minus the two things
# that demonstrably hurt it on this checkpoint.
#
# Round-1 sweep findings this encodes:
#   * A prompt that only *describes* the wanted picture (v2/v3) makes the model
#     drop the reference and free-generate, so the removal stays imperative here.
#   * "e-commerce", "catalogue-grade" and "product image" pull the model towards
#     advertising layouts -- v2/v3 rendered poster headlines in fake type. The
#     wording is plain photographic instead.
#   * Ending on "no caption and no watermark" (qwen_v1) is answered by stamping
#     an invented caption on the render, so the empty background is stated as a
#     positive fact rather than as a list of banned elements.
HUNYUAN_V4_PROMPT = (
    "Edit the provided photograph. Remove the person completely, including every hand, finger, arm "
    "and body part, remove every other object, and remove the whole original background. Keep only "
    "the {object_name}.\n\n"
    "Rebuild whatever part of the {object_name} the person was covering, matching the structure, "
    "colour and material already visible on it. It must stay the exact same physical item: "
    "identical shape, proportions, colour, material, texture, finish and surface detail, and the "
    "same labels, printed text and markings it already carries. Do not restyle it, do not add "
    "lettering or decoration it does not have, and show only one of it.\n\n"
    "Place the {object_name} alone at the centre of a plain pure white studio background, whole and "
    "fully visible, upright in its natural resting orientation, large in the frame, under soft even "
    "lighting with a soft contact shadow beneath it. Every pixel around the {object_name} is flat "
    "white and empty. Photorealistic photograph, sharp and cleanly lit."
)

PROMPT_PROFILES = {
    "qwen_v1": CANONICAL_PRODUCT_PROMPT,
    "hunyuan_v2": HUNYUAN_V2_PROMPT,
    "hunyuan_v3": HUNYUAN_V3_PROMPT,
    "hunyuan_v4": HUNYUAN_V4_PROMPT,
}

# Only used with ``--use-system-prompt custom``.
EDIT_SYSTEM_PROMPT = """You are a precise image-editing model. Execute the user's requested edits decisively.
The supplied image is a reference to edit, not a finished image to copy. When the user asks for the
subject to be isolated on a new background, visibly and completely remove everything else; do not
merely retouch the image, enhance its quality, or change only lighting and colour. Preserve only the
elements the user explicitly asks to keep, and preserve their identity exactly. Return exactly one
edited image and no text."""


def build_prompt(profile, object_name):
    object_name = " ".join(str(object_name or "object").split())
    return PROMPT_PROFILES[profile].format(object_name=object_name)


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

def canonical_job_name(job_name):
    """Rewrite a job name's variant suffix to ``canonical_product``.

    Input crops are named after the *original* variant (``__object_only`` /
    ``__person_swap``), but every job here becomes a canonical render, so
    outputs carry the canonical suffix and cannot collide with earlier runs.
    """
    for suffix in ("__object_only", "__person_swap", "__bg_swap"):
        if job_name.endswith(suffix):
            return job_name[: -len(suffix)] + f"__{VARIANT}"
    return f"{job_name}__{VARIANT}"


def load_jobs(jobs_path, input_dir, args):
    jobs = []
    missing = []
    with jobs_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            source_name = record["job_name"]
            # Full-frame mode feeds the original selected keyframe to Hunyuan verbatim.
            # Cropped mode is retained only for reproducing the earlier Qwen-matched run.
            input_path = (
                Path(record["frame_path"])
                if args.input_mode == "full-frame"
                else input_dir / f"{source_name}.jpg"
            )
            if not input_path.is_file():
                missing.append(str(input_path))
                continue
            jobs.append(
                {
                    "job_name": canonical_job_name(source_name),
                    "source_job_name": source_name,
                    "input_path": input_path,
                    "record": record,
                }
            )
    if missing:
        print(f"WARNING: {len(missing)} jobs have no cropped input image, e.g. {missing[0]}")

    if args.job_names:
        wanted = {name.strip() for name in args.job_names.split(",") if name.strip()}
        jobs = [
            job for job in jobs
            if job["job_name"] in wanted or job["source_job_name"] in wanted
        ]
    if args.only_source_variant:
        jobs = [job for job in jobs if job["record"].get("variant") == args.only_source_variant]
    if args.num_shards > 1:
        jobs = [job for idx, job in enumerate(jobs) if idx % args.num_shards == args.shard_index]
    if args.limit:
        jobs = jobs[: args.limit]
    return jobs


def derive_seed(job_name, base_seed):
    """Stable per-job seed: a job renders identically however the list is sharded."""
    digest = hashlib.sha256(f"{job_name}|{base_seed}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % (2 ** 31 - 1)


def resolve_image_size(image_size, input_path):
    """Resolve the ``image_size`` string passed to ``generate_image`` for one job.

    ``match`` mirrors the input frame's own "HxW" so the render keeps the source
    aspect ratio -- the checkpoint ships 720x1280 and 1280x720 as resolutions in
    its group, so a portrait keyframe maps exactly rather than being squared off.
    Anything else ('auto', an explicit 'HxW' / 'W:H') is passed through.
    """
    if image_size != "match":
        return image_size
    with Image.open(input_path) as image:
        width, height = image.size
    return f"{height}x{width}"


def read_manifest(path):
    """Return {job_name: entry} for an existing manifest, newest entry winning."""
    entries = {}
    if not path.is_file():
        return entries
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("job_name"):
                entries[entry["job_name"]] = entry
    return entries


def side_by_side(left, right, gap=16):
    height = max(left.height, right.height)

    def fit(image):
        if image.height == height:
            return image
        width = max(1, round(image.width * height / image.height))
        return image.resize((width, height), Image.LANCZOS)

    left, right = fit(left), fit(right)
    canvas = Image.new("RGB", (left.width + gap + right.width, height), (255, 255, 255))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + gap, 0))
    return canvas


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def validate_checkpoint(model_id):
    required = ["config.json", "generation_config.json", "model.safetensors.index.json", "tokenizer.json"]
    missing = [name for name in required if not (model_id / name).is_file()]
    shards = sorted(model_id.glob("model-*-of-0032.safetensors"))
    if missing or len(shards) != 32:
        raise FileNotFoundError(
            f"Incomplete Hunyuan checkpoint at {model_id}: missing metadata={missing}, "
            f"model shards={len(shards)}/32"
        )
    with (model_id / "config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    return bool(config.get("cfg_distilled") and config.get("use_meanflow"))


def build_max_memory(reserve_gib):
    """Cap each visible GPU at (currently free - reserve) GiB.

    ``device_map="auto"`` fills a card up to its free memory at load time, which
    leaves nothing for activations when another process shares the card -- the
    weights load fine and the first denoising step then OOMs.  Reserving real
    headroom per card makes a run on a shared GPU survivable.
    """
    import torch

    max_memory = {}
    for index in range(torch.cuda.device_count()):
        free_bytes, _ = torch.cuda.mem_get_info(index)
        free_gib = free_bytes / (1024 ** 3)
        budget = free_gib - reserve_gib
        if budget <= 0:
            raise RuntimeError(
                f"GPU {index} has only {free_gib:.1f} GiB free, less than the "
                f"{reserve_gib:.1f} GiB activation reserve. Free the card or lower --gpu-reserve-gib."
            )
        max_memory[index] = f"{budget:.0f}GiB"
        print(f"GPU {index}: {free_gib:.1f} GiB free -> weight budget {budget:.0f} GiB", flush=True)
    return max_memory


def load_model(model_id, args):
    sys.path.insert(0, str(ROOT))
    from hunyuan_image_3 import HunyuanImage3ForCausalMM

    kwargs = {}
    if args.gpu_reserve_gib > 0:
        kwargs["max_memory"] = build_max_memory(args.gpu_reserve_gib)
    model = HunyuanImage3ForCausalMM.from_pretrained(
        str(model_id),
        attn_implementation=args.attn_impl,
        dtype="auto",
        device_map="auto",
        moe_impl=args.moe_impl,
        moe_drop_tokens=True,
        local_files_only=True,
        **kwargs,
    )
    model.load_tokenizer(str(model_id))
    model.generation_config.diff_guidance_scale = args.guidance_scale
    return model


# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        "Canonical white-background product renders with HunyuanImage-3.0-Instruct-Distil"
    )
    parser.add_argument("--jobs", default=str(DEFAULT_JOBS))
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR),
                        help="Directory of pre-cropped <job_name>.jpg frames (shared with the Qwen run).")
    parser.add_argument(
        "--input-mode",
        choices=["full-frame", "cropped"],
        default="cropped",
        help="Reference image source. 'cropped' (default) uses the bbox-recomposed DiffSynth input; "
             "'full-frame' uses jobs[].frame_path verbatim.",
    )
    parser.add_argument("--model-id", default=str(DEFAULT_MODEL))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--tag", default=None,
                        help="Optional run subdirectory under --output-root (e.g. a smoke-test name).")
    parser.add_argument("--limit", type=int, default=0, help="Render at most N jobs (0 = all).")
    parser.add_argument("--job-names", default=None,
                        help="Comma-separated job names to render (source or canonical form).")
    parser.add_argument("--only-source-variant", default=None,
                        choices=["object_only", "person_swap", "bg_swap"],
                        help="Restrict to jobs whose original variant is this.")
    parser.add_argument("--num-shards", type=int, default=1, help="Split jobs across N worker processes.")
    parser.add_argument("--shard-index", type=int, default=0, help="This worker's shard (0-based).")
    parser.add_argument("--prompt-profile", choices=sorted(PROMPT_PROFILES), default="qwen_v1")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--image-size", default="match",
                        help="'match' (default): render at each input frame's own aspect ratio. "
                             "'square': use --height x --width. Or 'auto' / an explicit 'HxW' / 'W:H'.")
    parser.add_argument("--diff-infer-steps", type=int, default=DISTIL_STEPS,
                        help="Diffusion steps (Distil=8, full Instruct=50).")
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=0, help="Base seed; each job derives its own.")
    parser.add_argument("--gpu-reserve-gib", type=float, default=14.0,
                        help="Per-GPU activation headroom held back from the weight budget. "
                             "Essential when a card is shared with another process; 0 disables the cap.")
    parser.add_argument("--attn-impl", default="sdpa", choices=["sdpa", "flash_attention_2"])
    parser.add_argument("--moe-impl", default="eager", choices=["eager", "flashinfer"])
    parser.add_argument("--use-system-prompt", default="en_unified")
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--bot-task", choices=["image", "recaption", "think_recaption"], default="image")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--no-compare", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Write prompts/manifest without loading the model or touching a GPU.")
    return parser.parse_args()


def main():
    args = parse_args()
    jobs_path = Path(args.jobs).expanduser().resolve()
    input_dir = Path(args.input_dir).expanduser().resolve()
    model_id = Path(args.model_id).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if args.tag:
        output_root = output_root / args.tag

    if not jobs_path.is_file():
        raise FileNotFoundError(f"Jobs file not found: {jobs_path}")
    if args.input_mode == "cropped" and not input_dir.is_dir():
        raise FileNotFoundError(f"Input crop directory not found: {input_dir}")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require 0 <= --shard-index < --num-shards.")

    is_distilled = validate_checkpoint(model_id)
    if is_distilled and args.diff_infer_steps != DISTIL_STEPS:
        raise ValueError("The distilled MeanFlow checkpoint must use --diff-infer-steps 8.")
    if not is_distilled and args.diff_infer_steps == DISTIL_STEPS:
        # Keep the Distil-friendly CLI default usable with only --model-id swapped.
        args.diff_infer_steps = 50
    model_label = "HunyuanImage-3.0-Instruct-Distil" if is_distilled else "HunyuanImage-3.0-Instruct"

    jobs = load_jobs(jobs_path, input_dir, args)
    edited_dir = output_root / "edited"
    compare_dir = output_root / "compare"
    for directory in (edited_dir, compare_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # Give every concurrently-running worker its own manifest file. Sharing one
    # would let the tidy-up rewrite at the end of a run truncate another worker's
    # entries; per-worker manifests are merged by merge_canonical_manifests.py.
    suffix = "" if args.num_shards == 1 else f".shard{args.shard_index}"
    if args.only_source_variant:
        suffix += f".{args.only_source_variant}"
    manifest_path = output_root / (f"manifest.dryrun{suffix}.jsonl" if args.dry_run
                                   else f"manifest{suffix}.jsonl")
    done = set() if args.overwrite else set(read_manifest(manifest_path))
    image_size = f"{args.height}x{args.width}" if args.image_size == "square" else args.image_size

    print(f"Model: {model_label} ({model_id})")
    print(f"Jobs: {len(jobs)} (shard {args.shard_index}/{args.num_shards}), profile={args.prompt_profile}, "
          f"size={image_size}, steps={args.diff_infer_steps}, guidance={args.guidance_scale}, "
          f"bot_task={args.bot_task}")
    print(f"Input mode: {args.input_mode}")
    if args.input_mode == "cropped":
        print(f"Inputs: {input_dir}")
    print(f"Output: {output_root}; already done: {len(done)}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}", flush=True)

    model = None
    generated = skipped = failed = 0
    manifest = manifest_path.open("a", encoding="utf-8")
    try:
        with tqdm(total=len(jobs), unit="render") as progress:
            for job in jobs:
                name = job["job_name"]
                record = job["record"]
                out_path = edited_dir / f"{name}.jpg"
                progress.set_postfix(obj=str(record.get("object_name"))[:20])
                if name in done or (out_path.exists() and not args.overwrite):
                    skipped += 1
                    progress.write(f"PROGRESS completed={progress.n + 1}/{len(jobs)} generated={generated} "
                                   f"skipped={skipped} failed={failed} job={name}")
                    progress.update(1)
                    continue

                prompt = build_prompt(args.prompt_profile, record.get("object_name"))
                job_seed = derive_seed(name, args.seed)
                job_image_size = resolve_image_size(image_size, job["input_path"])

                entry = {
                    "job_name": name,
                    "source_job_name": job["source_job_name"],
                    "sample_id": record.get("sample_id"),
                    "original_id": record.get("original_id"),
                    "video": record.get("video"),
                    "record_json": record.get("record_json"),
                    "frame_path": record.get("frame_path"),
                    "frame_index": record.get("frame_index"),
                    "timestamp": record.get("timestamp"),
                    "entity_id": record.get("entity_id"),
                    "object_name": record.get("object_name"),
                    "other_objects": record.get("other_objects"),
                    "bbox_norm": record.get("bbox_norm"),
                    "source_variant": record.get("variant"),
                    "variant": VARIANT,
                    "model": model_label,
                    "model_path": str(model_id),
                    "bot_task": args.bot_task,
                    "prompt_profile": args.prompt_profile,
                    "seed": job_seed,
                    "steps": args.diff_infer_steps,
                    "guidance_scale": args.guidance_scale,
                    "image_size_mode": image_size,
                    "requested_size": job_image_size,
                    "scene": "pure white seamless studio background",
                    "lighting": "soft even product lighting",
                    "input_path": str(job["input_path"]),
                    "output_path": str(out_path),
                    "prompt": prompt,
                }

                if args.dry_run:
                    manifest.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    manifest.flush()
                    progress.update(1)
                    continue

                if model is None:
                    model = load_model(model_id, args)

                try:
                    text_generation_kwargs = {"use_cache": True} if args.bot_task != "image" else {}
                    _, samples = model.generate_image(
                        prompt=prompt,
                        image=str(job["input_path"]),
                        seed=job_seed,
                        image_size=job_image_size,
                        use_system_prompt=args.use_system_prompt,
                        system_prompt=(
                            args.system_prompt or EDIT_SYSTEM_PROMPT
                            if args.use_system_prompt == "custom"
                            else args.system_prompt
                        ),
                        bot_task=args.bot_task,
                        infer_align_image_size=True,
                        diff_infer_steps=args.diff_infer_steps,
                        max_new_tokens=args.max_new_tokens,
                        verbose=args.verbose,
                        **text_generation_kwargs,
                    )
                except Exception as error:  # keep a long batch alive past one bad job
                    failed += 1
                    progress.write(f"FAILED job={name}: {type(error).__name__}: {error}")
                    progress.write(f"PROGRESS completed={progress.n + 1}/{len(jobs)} generated={generated} "
                                   f"skipped={skipped} failed={failed} job={name}")
                    progress.update(1)
                    continue

                result = samples[0].convert("RGB")
                entry["actual_size"] = list(result.size)
                result.save(out_path, quality=95)
                if not args.no_compare:
                    source = Image.open(job["input_path"]).convert("RGB")
                    side_by_side(source, result).save(compare_dir / f"{name}.jpg", quality=95)
                manifest.write(json.dumps(entry, ensure_ascii=False) + "\n")
                manifest.flush()
                generated += 1
                progress.write(f"PROGRESS completed={progress.n + 1}/{len(jobs)} generated={generated} "
                               f"skipped={skipped} failed={failed} job={name}")
                progress.update(1)
    finally:
        manifest.close()

    entries = read_manifest(manifest_path)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for entry in entries.values():
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Generated {generated}, skipped {skipped}, failed {failed}.")
    print(f"Edited outputs: {edited_dir}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
