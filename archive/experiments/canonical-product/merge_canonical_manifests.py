"""Merge the per-worker manifests of a sharded canonical render into manifest.jsonl.

Each worker writes ``manifest.shard<N>[.<variant>].jsonl`` so concurrent runs never
truncate one another.  This collapses them into a single ``manifest.jsonl`` and
reports which jobs are still missing an output image.
"""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = ROOT / "dataset" / "s2v_keyframe_recontext-hunyuan-canonical"


def main():
    parser = argparse.ArgumentParser("Merge sharded canonical manifests")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--jobs", default=None,
                        help="Optional jobs.jsonl; reports jobs with no rendered output.")
    args = parser.parse_args()

    output_root = Path(args.output_root).expanduser().resolve()
    edited_dir = output_root / "edited"
    shards = sorted(output_root.glob("manifest.shard*.jsonl"))
    if not shards:
        raise SystemExit(f"No shard manifests found under {output_root}")

    entries = {}
    for shard in shards:
        count = 0
        with shard.open(encoding="utf-8") as handle:
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
                    count += 1
        print(f"{shard.name}: {count} entries")

    # Only keep entries whose image actually landed on disk, so a partially
    # written run cannot be mistaken for a complete one downstream.
    complete = {name: entry for name, entry in entries.items()
                if (edited_dir / f"{name}.jpg").is_file()}
    dropped = len(entries) - len(complete)

    merged_path = output_root / "manifest.jsonl"
    with merged_path.open("w", encoding="utf-8") as handle:
        for name in sorted(complete):
            handle.write(json.dumps(complete[name], ensure_ascii=False) + "\n")

    print(f"\nMerged {len(complete)} entries -> {merged_path}"
          + (f" ({dropped} manifest rows had no image and were dropped)" if dropped else ""))
    print(f"Images on disk: {len(list(edited_dir.glob('*.jpg')))}")

    if args.jobs:
        from run_s2v_canonical_hunyuan_distil_jobs import canonical_job_name

        wanted = []
        with Path(args.jobs).expanduser().resolve().open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    wanted.append(canonical_job_name(json.loads(line)["job_name"]))
        missing = [name for name in wanted if name not in complete]
        print(f"Jobs total {len(wanted)}, missing {len(missing)}")
        for name in missing[:20]:
            print(f"  MISSING {name}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")


if __name__ == "__main__":
    main()
