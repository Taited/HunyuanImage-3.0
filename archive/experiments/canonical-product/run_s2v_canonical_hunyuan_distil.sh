#!/bin/bash
# Launch sharded canonical-product renders on the last four GPUs.
#
# The 80B MoE checkpoint is ~158 GiB in bf16, so it needs two H200s per copy.
# Four cards therefore host two independent workers, each owning half the jobs.
#
# Each worker sizes its weight budget from the memory actually free on its cards
# (--gpu-reserve-gib), so a card shared with someone else's job still works as
# long as the shared card keeps ~85 GiB free.
#
#   ./run_s2v_canonical_hunyuan_distil.sh                     # all 300 jobs
#   PROFILE=hunyuan_v2 TAG=v2 ./run_s2v_canonical_hunyuan_distil.sh
#   EXTRA="--limit 4" TAG=smoke ./run_s2v_canonical_hunyuan_distil.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
PROFILE="${PROFILE:-qwen_v1}"
TAG="${TAG:-}"
GPU_PAIRS="${GPU_PAIRS:-4,5 6,7}"
RESERVE_GIB="${RESERVE_GIB:-14}"
EXTRA="${EXTRA:-}"

read -r -a PAIRS <<< "$GPU_PAIRS"
NUM_SHARDS=${#PAIRS[@]}

LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="canonical_${PROFILE}${TAG:+_$TAG}_$STAMP"

TAG_ARGS=()
if [[ -n "$TAG" ]]; then TAG_ARGS=(--tag "$TAG"); fi

echo "Run: $RUN_NAME"
echo "Shards: $NUM_SHARDS on GPU pairs: ${PAIRS[*]}"
for i in "${!PAIRS[@]}"; do
    LOG="$LOG_DIR/${RUN_NAME}_shard${i}.log"
    CUDA_VISIBLE_DEVICES="${PAIRS[$i]}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    nohup "$PYTHON" "$ROOT/run_s2v_canonical_hunyuan_distil_jobs.py" \
        --prompt-profile "$PROFILE" \
        --num-shards "$NUM_SHARDS" \
        --shard-index "$i" \
        --gpu-reserve-gib "$RESERVE_GIB" \
        "${TAG_ARGS[@]}" \
        $EXTRA \
        > "$LOG" 2>&1 &
    echo "  shard $i -> GPU ${PAIRS[$i]} pid $! log $LOG"
    # Stagger the launches: two workers reading 158 GiB of shards at once
    # saturates the filesystem and slows both loads down.
    sleep 20
done
wait
echo "All shards finished."
