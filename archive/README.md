# Historical experiment archive

Archived on 2026-10-12. These scripts are historical snapshots, not current pipeline entry points.
No datasets, images, checkpoints, credentials, virtual environments or logs are included.

## Restore before running

See manifest.json for every original workspace-relative path and SHA-256 checksum.
Back up any existing destination, then restore each file from its archived path to its original
path under the workspace root. Do not run scripts directly here: their relative paths and imports
expect the original layout. Restoring a group does not guarantee compatibility with current dependencies.

## Groups

- experiments/canonical-product: old canonical-image generation, launcher, prompt sweep and manifest merge; restore together.
- experiments/video-frame-generation: first-frame to middle/end-frame experiments.
- experiments/legacy-runs: old pending-queue launcher and standalone three-view smoke test.
- experiments/reporting: export examples from the old output layout.

Current multi-view pipeline entry points remain at repository root.
audit_canonical_outputs.py and tools/build_annotation_10k.py remain as auxiliary tools.
The controlled 2x2 experiment, including this project's runner, is archived in the sibling
multi-reference-data-preparation repository at archive/experiments/three-view-2x2/workspace/.
