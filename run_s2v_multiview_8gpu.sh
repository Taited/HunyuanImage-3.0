#!/bin/bash
# Run multi-view jobs on independently configured GPU workers.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
MODEL_ID="${MODEL_ID:-$ROOT/../weights/HunyuanImage-3.0-Instruct-Distil}"
JOBS="${JOBS:-$ROOT/dataset/s2v_keyframe_recontext/jobs.multiview.jsonl}"
OUTPUT_BASE="${OUTPUT_BASE:-$ROOT/dataset/s2v_object_only-hunyuan-distil/multiview}"
GPU_PAIRS="${GPU_PAIRS:-0,1 2,3 4,5 6,7}"
RESERVE_GIB="${RESERVE_GIB:-14}"
STAGGER_SECONDS="${STAGGER_SECONDS:-20}"
EXTRA="${EXTRA:-}"
WAIT_FOR_WORKERS="${WAIT_FOR_WORKERS:-0}"

read -r -a PAIRS <<< "$GPU_PAIRS"
read -r -a EXTRA_ARGS <<< "$EXTRA"
NUM_SHARDS=${#PAIRS[@]}
PIDS=()
[[ "$NUM_SHARDS" -gt 0 ]] || { echo "GPU_PAIRS is empty" >&2; exit 2; }
[[ -x "$PYTHON" ]] || {
    echo "Project Python not executable: $PYTHON" >&2
    exit 2
}
[[ -f "$JOBS" ]] || {
    echo "Jobs file not found: $JOBS" >&2
    exit 2
}

LOG_DIR="$OUTPUT_BASE/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
PID_FILE="$LOG_DIR/multiview_8gpu_${STAMP}.pids"

echo "Python: $PYTHON"
echo "Jobs: $JOBS ($(wc -l < "$JOBS") rows)"
echo "Output: $OUTPUT_BASE"
echo "Workers: $NUM_SHARDS on GPU pairs: ${PAIRS[*]}"
echo "PID file: $PID_FILE"

for i in "${!PAIRS[@]}"; do
    SHARD="$(printf '%02d' "$i")"
    LOG="$LOG_DIR/shard_${SHARD}_${STAMP}.log"
    CUDA_VISIBLE_DEVICES="${PAIRS[$i]}" \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    nohup "$PYTHON" "$ROOT/run_s2v_multiview_hunyuan_distil.py" \
        --jobs "$JOBS" \
        --model-id "$MODEL_ID" \
        --output-root "$OUTPUT_BASE" \
        --num-shards "$NUM_SHARDS" \
        --shard-index "$i" \
        --prompt-profile strict_v2 \
        --input-mode bbox \
        --gpu-reserve-gib "$RESERVE_GIB" \
        "${EXTRA_ARGS[@]}" \
        > "$LOG" 2>&1 < /dev/null &
    PID=$!
    PIDS+=("$PID")
    printf '%s\t%s\t%s\t%s\n' "$PID" "$i" "${PAIRS[$i]}" "$LOG" >> "$PID_FILE"
    echo "shard $i -> GPU ${PAIRS[$i]}, PID $PID"
    echo "  log: $LOG"
    [[ "$i" -eq "$((NUM_SHARDS - 1))" ]] || sleep "$STAGGER_SECONDS"
done

echo "All workers are running in the background."
echo "Progress: tail -f $LOG_DIR/shard_*_${STAMP}.log"
echo "Processes: ps -fp $(cut -f1 "$PID_FILE" | paste -sd, -)"
if [[ "$WAIT_FOR_WORKERS" == "1" ]]; then
    echo "Supervisor mode enabled; waiting for all workers."
    wait "${PIDS[@]}"
fi
