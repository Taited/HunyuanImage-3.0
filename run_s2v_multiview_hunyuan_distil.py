#!/usr/bin/env python3
"""Run HunyuanImage object-only inference for complete multi-view S2V jobs."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from PIL import Image
from tqdm import tqdm

import s2v_object_only_runtime as base


ROOT = Path(__file__).resolve().parent
DEFAULT_JOBS = ROOT / "dataset" / "s2v_keyframe_recontext" / "jobs.multiview.jsonl"
DEFAULT_OUTPUT_ROOT = (
    ROOT / "dataset" / "s2v_object_only-hunyuan-distil" / "multiview"
)


def parse_args():
    parser = argparse.ArgumentParser(
        "HunyuanImage 3 Distil grouped inference for complete S2V multi-view jobs"
    )
    parser.add_argument("--jobs", default=str(DEFAULT_JOBS))
    parser.add_argument("--model-id", default=str(base.DEFAULT_MODEL))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--limit", type=int, default=0, help="Jobs per shard; 0 means all.")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--prompt-profile", choices=sorted(base.PROMPTS), default="strict_v2")
    parser.add_argument("--use-system-prompt", choices=["custom", "en_unified"], default="custom")
    parser.add_argument("--input-mode", choices=["bbox", "full-frame"], default="bbox")
    parser.add_argument("--bbox-padding", type=float, default=0.30)
    parser.add_argument("--min-native-white", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--diff-infer-steps", type=int, default=base.DISTIL_STEPS)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--attn-impl", choices=["sdpa", "flash_attention_2"], default="sdpa")
    parser.add_argument("--moe-impl", choices=["eager", "flashinfer"], default="eager")
    parser.add_argument("--gpu-reserve-gib", type=float, default=14.0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--save-native", action="store_true")
    parser.add_argument("--no-compare", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def safe_component(value):
    value = str(value)
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)
    cleaned = cleaned.strip("._")
    return cleaned or f"sample_{base.hashlib.sha256(value.encode()).hexdigest()[:16]}"


def sample_key(record):
    return str(record.get("sample_id") or record.get("video") or record["job_name"])


def artifact_stem(record):
    frame_index = int(record["frame_index"])
    rank = int(record.get("view_rank", record.get("extra_view_rank")) or 0)
    entity = safe_component(record.get("entity_id") or "entity")
    return f"frame-{frame_index:06d}__view-{rank:02d}__{entity}__object_only_product"


def load_and_assign_jobs(path, num_shards, shard_index):
    all_jobs = []
    group_shards = {}
    missing = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            key = sample_key(record)
            if key not in group_shards:
                group_shards[key] = len(group_shards) % num_shards
            if group_shards[key] != shard_index:
                continue
            frame_path = Path(record.get("frame_path", "")).expanduser()
            if not frame_path.is_file():
                print(f"WARNING line {line_number}: missing frame {frame_path}", flush=True)
                missing += 1
                continue
            all_jobs.append(
                {
                    "job_name": base.output_job_name(record["job_name"]),
                    "source_job_name": record["job_name"],
                    "input_path": frame_path.resolve(),
                    "object_name": base.object_description(record),
                    "record": record,
                    "sample_key": key,
                    "artifact_stem": artifact_stem(record),
                }
            )
    return all_jobs, len(group_shards), missing


def main():
    args = parse_args()
    jobs_path = Path(args.jobs).expanduser().resolve()
    model_id = Path(args.model_id).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    if not jobs_path.is_file():
        raise FileNotFoundError(f"Jobs JSONL not found: {jobs_path}")
    if args.diff_infer_steps != base.DISTIL_STEPS:
        raise ValueError(
            f"The distilled MeanFlow checkpoint must use exactly {base.DISTIL_STEPS} steps"
        )
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require num_shards >= 1 and 0 <= shard_index < num_shards")
    base.validate_distilled_checkpoint(model_id)

    jobs, total_samples, missing = load_and_assign_jobs(
        jobs_path, args.num_shards, args.shard_index
    )
    if args.limit:
        jobs = jobs[: args.limit]
    if not jobs:
        raise ValueError("No runnable jobs selected for this shard")

    shard_tag = f"shard_{args.shard_index:02d}"
    manifest_name = (
        f"manifest.dryrun.{shard_tag}.jsonl"
        if args.dry_run
        else f"manifest.{shard_tag}.jsonl"
    )
    manifest_path = output_root / manifest_name
    output_root.mkdir(parents=True, exist_ok=True)
    done = set() if args.overwrite else set(base.read_manifest(manifest_path))
    invalid_path = output_root / f"invalid_input.{shard_tag}.jsonl"
    invalid_done = set() if args.overwrite else set(base.read_manifest(invalid_path))

    assigned_samples = len({job["sample_key"] for job in jobs})
    print(f"Checkpoint: HunyuanImage-3.0-Instruct-Distil ({model_id})")
    print(
        f"Jobs: {len(jobs)}, videos: {assigned_samples}/{total_samples}, "
        f"shard={args.shard_index}/{args.num_shards}, missing={missing}"
    )
    print(
        f"Profile={args.prompt_profile}, input={args.input_mode}, "
        f"bbox_padding={args.bbox_padding:.2f}, steps={args.diff_infer_steps}"
    )
    print(f"Output: {output_root}; completed in manifest: {len(done)}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}", flush=True)

    model = None
    generated = skipped = failed = 0
    with manifest_path.open("a", encoding="utf-8") as manifest, \
            invalid_path.open("a", encoding="utf-8") as invalid_manifest:
        with tqdm(total=len(jobs), unit="edit") as progress:
            for job in jobs:
                name = job["job_name"]
                record = job["record"]
                sample_dir = output_root / "videos" / safe_component(job["sample_key"])
                edited_dir = sample_dir / "edited"
                compare_dir = sample_dir / "compare"
                native_dir = sample_dir / "native"
                reference_dir = sample_dir / "reference_inputs"
                stem = job["artifact_stem"]
                out_path = edited_dir / f"{stem}.jpg"
                model_input_path = reference_dir / f"{stem}.jpg"

                progress.set_postfix(
                    video=safe_component(job["sample_key"])[:16],
                    frame=record.get("frame_index"),
                )
                if name in done or name in invalid_done or (out_path.is_file() and not args.overwrite):
                    skipped += 1
                    progress.update(1)
                    continue

                try:
                    source = Image.open(job["input_path"]).convert("RGB")
                    input_size = source.size
                    if args.input_mode == "bbox":
                        model_input, crop_box = base.bbox_crop(
                            source, record.get("bbox_norm"), args.bbox_padding
                        )
                    else:
                        model_input, crop_box = source, (0, 0, source.width, source.height)
                except Exception as error:
                    failed += 1
                    invalid_entry = {
                        "job_name": name,
                        "source_job_name": job["source_job_name"],
                        "sample_id": record.get("sample_id"),
                        "video": record.get("video"),
                        "frame_index": record.get("frame_index"),
                        "input_path": str(job["input_path"]),
                        "bbox_norm": record.get("bbox_norm"),
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                    invalid_manifest.write(json.dumps(invalid_entry, ensure_ascii=False) + "\n")
                    invalid_manifest.flush()
                    invalid_done.add(name)
                    progress.write(
                        f"INVALID {name}: {type(error).__name__}: {error}"
                    )
                    progress.update(1)
                    continue

                prompt = base.PROMPTS[args.prompt_profile].format(
                    object_name=job["object_name"]
                )
                seed = base.stable_seed(name, args.seed)
                requested_size = f"{model_input.height}x{model_input.width}"
                entry = {
                    "job_name": name,
                    "source_job_name": job["source_job_name"],
                    "sample_id": record.get("sample_id"),
                    "video": record.get("video"),
                    "frame_index": record.get("frame_index"),
                    "frame_slot": record.get("frame_slot"),
                    "view_rank": record.get("view_rank", record.get("extra_view_rank")),
                    "is_anchor": record.get("is_anchor"),
                    "parent_job_name": record.get("parent_job_name"),
                    "anchor_frame_index": record.get("anchor_frame_index"),
                    "entity_id": record.get("entity_id"),
                    "object_name": job["object_name"],
                    "input_path": str(job["input_path"]),
                    "model_input_path": str(model_input_path),
                    "output_path": str(out_path),
                    "input_mode": args.input_mode,
                    "bbox_norm": record.get("bbox_norm"),
                    "bbox_padding": args.bbox_padding,
                    "crop_box": list(crop_box),
                    "model_input_size": list(model_input.size),
                    "input_size": list(input_size),
                    "requested_model_size": requested_size,
                    "model": "HunyuanImage-3.0-Instruct-Distil",
                    "model_path": str(model_id),
                    "prompt_profile": args.prompt_profile,
                    "system_prompt_profile": args.use_system_prompt,
                    "bot_task": "image",
                    "steps": args.diff_infer_steps,
                    "guidance_scale": args.guidance_scale,
                    "seed": seed,
                    "prompt": prompt,
                }
                if args.dry_run:
                    manifest.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    manifest.flush()
                    progress.update(1)
                    continue

                edited_dir.mkdir(parents=True, exist_ok=True)
                reference_dir.mkdir(parents=True, exist_ok=True)
                if not args.no_compare:
                    compare_dir.mkdir(parents=True, exist_ok=True)
                if args.save_native:
                    native_dir.mkdir(parents=True, exist_ok=True)
                model_input.save(model_input_path, quality=95)

                if model is None:
                    model = base.load_model(model_id, args)
                started = time.time()
                try:
                    _, samples = model.generate_image(
                        prompt=prompt,
                        image=str(model_input_path),
                        seed=seed,
                        image_size=requested_size,
                        use_system_prompt=args.use_system_prompt,
                        system_prompt=(
                            base.STRICT_SYSTEM_PROMPT
                            if args.use_system_prompt == "custom"
                            else None
                        ),
                        bot_task="image",
                        infer_align_image_size=True,
                        diff_infer_steps=args.diff_infer_steps,
                        max_new_tokens=args.max_new_tokens,
                        verbose=args.verbose,
                    )
                    native = samples[0].convert("RGB")
                    if args.save_native:
                        native.save(native_dir / f"{stem}.jpg", quality=95)
                    result = base.place_on_source_sized_canvas(native, input_size)
                    result.save(out_path, quality=95)
                    if not args.no_compare:
                        base.compare_image(source, result).save(
                            compare_dir / f"{stem}.jpg", quality=92
                        )
                    native_white = base.border_whiteness(native)
                    entry.update(
                        {
                            "native_output_size": list(native.size),
                            "final_output_size": list(result.size),
                            "fitted_to_source_canvas": True,
                            "native_border_whiteness": native_white,
                            "border_whiteness": base.border_whiteness(result),
                            "quality_status": (
                                "pass_background_check"
                                if native_white >= args.min_native_white
                                else "review_background"
                            ),
                            "seconds": round(time.time() - started, 2),
                        }
                    )
                    manifest.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    manifest.flush()
                    generated += 1
                    progress.write(
                        f"PROGRESS {progress.n + 1}/{len(jobs)} "
                        f"video={record.get('sample_id')} frame={record.get('frame_index')} "
                        f"native_white={native_white:.3f}"
                    )
                except Exception as error:
                    failed += 1
                    progress.write(
                        f"FAILED {name}: {type(error).__name__}: {error}"
                    )
                progress.update(1)

    entries = base.read_manifest(manifest_path)
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for entry in entries.values():
            manifest.write(json.dumps(entry, ensure_ascii=False) + "\n")

    if not args.dry_run:
        review_path = output_root / f"review_required.{shard_tag}.jsonl"
        with review_path.open("w", encoding="utf-8") as review:
            for entry in entries.values():
                if entry.get("quality_status") == "review_background":
                    review.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Generated {generated}, skipped {skipped}, failed {failed}")
    print(f"Manifest: {manifest_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
