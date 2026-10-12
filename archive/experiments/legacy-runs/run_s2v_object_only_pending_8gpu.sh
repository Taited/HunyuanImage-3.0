#!/bin/bash
# Run the deduplicated full_bbox_direct_v2 queue on four independent 2-GPU workers.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
JOBS="${JOBS:-$ROOT/dataset/s2v_keyframe_recontext/jobs.pending_full_bbox_direct_v2.jsonl}"
OUTPUT_BASE="${OUTPUT_BASE:-$ROOT/dataset/s2v_object_only-hunyuan-distil/full_bbox_direct_v2}"
GPU_PAIRS="${GPU_PAIRS:-0,1 2,3 4,5 6,7}"
SHARD_OFFSET="${SHARD_OFFSET:-3}"
RESERVE_GIB="${RESERVE_GIB:-14}"
STAGGER_SECONDS="${STAGGER_SECONDS:-20}"
EXTRA="${EXTRA:-}"
read -r -a PAIRS <<< "$GPU_PAIRS"
NUM_SHARDS=${#PAIRS[@]}
[[ "$NUM_SHARDS" -eq 4 ]] || { echo "Expected four GPU pairs, got $NUM_SHARDS" >&2; exit 2; }
[[ -f "$JOBS" ]] || { echo "Jobs file not found: $JOBS" >&2; exit 2; }
LOG_DIR="$OUTPUT_BASE/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
PID_FILE="$LOG_DIR/pending_8gpu_${STAMP}.pids"
echo "Jobs: $JOBS ($(wc -l < "$JOBS") rows)"
echo "Workers: $NUM_SHARDS on GPU pairs: ${PAIRS[*]}"
echo "PID file: $PID_FILE"
for i in "${!PAIRS[@]}"; do
    SHARD_NUMBER="$((SHARD_OFFSET + i))"
    OUTPUT_ROOT="$OUTPUT_BASE/shard_$(printf '%02d' "$SHARD_NUMBER")"
    LOG="$LOG_DIR/shard_$(printf '%02d' "$SHARD_NUMBER")_${STAMP}.log"
    mkdir -p "$OUTPUT_ROOT"
    CUDA_VISIBLE_DEVICES="${PAIRS[$i]}" PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    nohup "$PYTHON" "$ROOT/run_s2v_object_only_hunyuan_distil.py" \
        --jobs "$JOBS" --output-root "$OUTPUT_ROOT" \
        --num-shards "$NUM_SHARDS" --shard-index "$i" \
        --prompt-profile strict_v2 --input-mode bbox \
        --gpu-reserve-gib "$RESERVE_GIB" $EXTRA \
        > "$LOG" 2>&1 < /dev/null &
    PID=$!
    printf '%s\t%s\t%s\t%s\n' "$PID" "$i" "${PAIRS[$i]}" "$LOG" >> "$PID_FILE"
    echo "shard $i -> output shard_$(printf '%02d' "$SHARD_NUMBER"), GPU ${PAIRS[$i]}, PID $PID"
    echo "  log: $LOG"
    [[ "$i" -ge "$((NUM_SHARDS - 1))" ]] || sleep "$STAGGER_SECONDS"
done
echo "All workers are running in the background."
echo "Progress: tail -f $LOG_DIR/shard_0[3-6]_${STAMP}.log"
echo "Processes: ps -fp $(cut -f1 "$PID_FILE" | paste -sd, -)"
