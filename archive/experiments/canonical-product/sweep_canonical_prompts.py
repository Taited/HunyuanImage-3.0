"""Sweep prompt profiles / bot tasks over a fixed set of jobs on one model load.

Loading the 158 GiB checkpoint costs several minutes, which dwarfs an 8-step
render.  This runs the whole (profile x bot_task x system_prompt) matrix inside
a single load so a tuning round takes minutes rather than an hour.

Outputs land in ``<output-root>/<run>/<config>/`` alongside a contact sheet per
config, so configs can be compared image-by-image on identical inputs.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from run_s2v_canonical_hunyuan_distil_jobs import (
    DEFAULT_INPUT_DIR,
    DEFAULT_JOBS,
    DEFAULT_MODEL,
    EDIT_SYSTEM_PROMPT,
    PROMPT_PROFILES,
    build_prompt,
    canonical_job_name,
    derive_seed,
    load_model,
    validate_checkpoint,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_SWEEP_ROOT = ROOT / "dataset" / "s2v_keyframe_recontext-hunyuan-canonical" / "_sweeps"

# Six frames spanning both source variants and a range of difficulty: a mostly
# submerged object, small objects held up to camera, and a clean easy case.
DEFAULT_TEST_JOBS = [
    "s2v_90231a3859c34a5277c35e025660093c__31_-Scene-005-1082-1381_frames_145-297_sim_0__f000000__entity-000__object_only",
    "s2v_96d3b541bfe7f327f804cdae805e2a28__663567595910520_frames_20-196_sim_0__f000000__entity-000__object_only",
    "s2v_d54ed4db3fc91d750faf71bac3bf8240__248064759870306_frames_69-253_sim_0__f000061__entity-000__object_only",
    "s2v_163d4b27aac46b88770656b7f1f274d4__58325910983_frames_0-160_sim_0__f000000__entity-001__person_swap",
    "s2v_62e89f9103f5548bf74e8213d548b5c8__128447284510180_frames_24-208_sim_0__f000061__entity-000__person_swap",
    "s2v_b0f7679a222b684fca65f59aeced0858__80_-Scene-001-6-9121_frames_815-999_sim_0__f000000__entity-000__person_swap",
]


def parse_config(spec):
    """Parse ``profile[:bot_task[:system]]`` into a config dict.

    ``system`` is ``en_unified`` (the checkpoint's own edit system prompt) or
    ``custom`` (the decisive edit prompt from the batch runner).
    """
    parts = spec.split(":")
    profile = parts[0]
    bot_task = parts[1] if len(parts) > 1 and parts[1] else "image"
    system = parts[2] if len(parts) > 2 and parts[2] else "en_unified"
    if profile not in PROMPT_PROFILES:
        raise ValueError(f"Unknown profile {profile!r}; choose from {sorted(PROMPT_PROFILES)}")
    if bot_task not in {"image", "recaption", "think_recaption"}:
        raise ValueError(f"Unknown bot_task {bot_task!r}")
    if system not in {"en_unified", "custom"}:
        raise ValueError(f"Unknown system prompt {system!r}")
    return {"spec": spec, "name": f"{profile}__{bot_task}__{system}",
            "profile": profile, "bot_task": bot_task, "system": system}


def contact_sheet(paths, cell=420):
    """Tile renders into one reviewable sheet (inputs on the top row if given)."""
    images = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((cell, cell), Image.LANCZOS)
        images.append(image)
    if not images:
        return None
    columns = min(len(images), 3)
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell, rows * cell), (238, 238, 238))
    for index, image in enumerate(images):
        x = (index % columns) * cell + (cell - image.width) // 2
        y = (index // columns) * cell + (cell - image.height) // 2
        sheet.paste(image, (x, y))
    return sheet


def parse_args():
    parser = argparse.ArgumentParser("Prompt/bot-task sweep for canonical product renders")
    parser.add_argument("--configs", nargs="+", required=True,
                        help="Configs as profile[:bot_task[:system]], e.g. hunyuan_v2:image "
                             "hunyuan_v3:think_recaption:custom")
    parser.add_argument("--run-name", default="sweep", help="Subdirectory under --sweep-root.")
    parser.add_argument("--sweep-root", default=str(DEFAULT_SWEEP_ROOT))
    parser.add_argument("--jobs", default=str(DEFAULT_JOBS))
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--model-id", default=str(DEFAULT_MODEL))
    parser.add_argument("--job-names", nargs="*", default=None,
                        help="Source job names to render (default: the built-in six-frame set).")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--diff-infer-steps", type=int, default=8)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-reserve-gib", type=float, default=14.0)
    parser.add_argument("--attn-impl", default="sdpa", choices=["sdpa", "flash_attention_2"])
    parser.add_argument("--moe-impl", default="eager", choices=["eager", "flashinfer"])
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    configs = [parse_config(spec) for spec in args.configs]
    input_dir = Path(args.input_dir).expanduser().resolve()
    model_id = Path(args.model_id).expanduser().resolve()
    sweep_dir = Path(args.sweep_root).expanduser().resolve() / args.run_name
    sweep_dir.mkdir(parents=True, exist_ok=True)

    validate_checkpoint(model_id)

    records = {}
    with Path(args.jobs).expanduser().resolve().open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                record = json.loads(line)
                records[record["job_name"]] = record

    wanted = args.job_names if args.job_names else DEFAULT_TEST_JOBS
    jobs = []
    for source_name in wanted:
        if source_name not in records:
            raise KeyError(f"Job not found in {args.jobs}: {source_name}")
        input_path = input_dir / f"{source_name}.jpg"
        if not input_path.is_file():
            raise FileNotFoundError(f"No cropped input for {source_name}: {input_path}")
        jobs.append({"source_job_name": source_name,
                     "job_name": canonical_job_name(source_name),
                     "input_path": input_path,
                     "record": records[source_name]})

    image_size = f"{args.height}x{args.width}"
    print(f"Sweep: {len(configs)} configs x {len(jobs)} jobs -> {sweep_dir}")
    for config in configs:
        print(f"  config {config['name']}")
    print(f"Objects: {[job['record'].get('object_name') for job in jobs]}", flush=True)

    # Contact sheet of the inputs, so a config's sheet can be read against it.
    sheet = contact_sheet([job["input_path"] for job in jobs])
    if sheet is not None:
        sheet.save(sweep_dir / "_inputs.jpg", quality=92)

    model = load_model(model_id, args)
    print("Model loaded.", flush=True)

    log_path = sweep_dir / "sweep.jsonl"
    log = log_path.open("a", encoding="utf-8")
    try:
        total = len(configs) * len(jobs)
        with tqdm(total=total, unit="render") as progress:
            for config in configs:
                config_dir = sweep_dir / config["name"]
                config_dir.mkdir(parents=True, exist_ok=True)
                rendered = []
                for job in jobs:
                    record = job["record"]
                    object_name = record.get("object_name")
                    out_path = config_dir / f"{job['job_name']}.jpg"
                    progress.set_postfix(cfg=config["name"][:24], obj=str(object_name)[:16])
                    if out_path.exists() and not args.overwrite:
                        rendered.append(out_path)
                        progress.update(1)
                        continue

                    prompt = build_prompt(config["profile"], object_name)
                    job_seed = derive_seed(job["job_name"], args.seed)
                    started = time.time()
                    try:
                        text_kwargs = {"use_cache": True} if config["bot_task"] != "image" else {}
                        _, samples = model.generate_image(
                            prompt=prompt,
                            image=str(job["input_path"]),
                            seed=job_seed,
                            image_size=image_size,
                            use_system_prompt=config["system"],
                            system_prompt=EDIT_SYSTEM_PROMPT if config["system"] == "custom" else None,
                            bot_task=config["bot_task"],
                            infer_align_image_size=True,
                            diff_infer_steps=args.diff_infer_steps,
                            max_new_tokens=args.max_new_tokens,
                            verbose=args.verbose,
                            **text_kwargs,
                        )
                    except Exception as error:
                        progress.write(f"FAILED cfg={config['name']} job={object_name}: "
                                       f"{type(error).__name__}: {error}")
                        progress.update(1)
                        continue

                    samples[0].convert("RGB").save(out_path, quality=95)
                    rendered.append(out_path)
                    log.write(json.dumps({
                        "config": config["name"], "spec": config["spec"],
                        "job_name": job["job_name"], "object_name": object_name,
                        "seed": job_seed, "seconds": round(time.time() - started, 1),
                        "output_path": str(out_path), "prompt": prompt,
                    }, ensure_ascii=False) + "\n")
                    log.flush()
                    progress.write(f"PROGRESS cfg={config['name']} obj={object_name} "
                                   f"{time.time() - started:.0f}s")
                    progress.update(1)

                sheet = contact_sheet(rendered)
                if sheet is not None:
                    sheet.save(sweep_dir / f"{config['name']}.jpg", quality=92)
                    progress.write(f"SHEET {sweep_dir / (config['name'] + '.jpg')}")
    finally:
        log.close()

    print(f"Sweep done: {sweep_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
