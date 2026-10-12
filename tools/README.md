# Auxiliary data tools

- build_annotation_10k.py: reproducibly select passing consistency-review groups and create
  side-by-side annotation pairs. Defaults: dataset/gemma_edited_frame_consistency.jsonl -> dataset/annotation_10k.
  Run from the HunyuanImage-3.0 root with its Python environment; use --dry-run to preview selection.
- ../audit_canonical_outputs.py: score frame-edge whiteness from a manifest and export low-score
  contact sheets. This is a background heuristic, not a full identity or geometry quality assessment.

Historical reporting and generation experiments are under ../archive/.
