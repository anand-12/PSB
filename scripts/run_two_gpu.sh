#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
DATA="${DATA:-data}"
OUTPUT="${OUTPUT:-runs}"
SEED0="${SEED0:-0}"
SEED1="${SEED1:-1}"
POSTERIOR_LR="${POSTERIOR_LR:-0.03}"
POSTERIOR_TASK_DECAY="${POSTERIOR_TASK_DECAY:-1.0}"
POSTERIOR_MIN_LR="${POSTERIOR_MIN_LR:-0.0125}"

mkdir -p "$OUTPUT/logs"

# Download once before launching concurrent readers of the MNIST files.
"$PYTHON" -c "from gwd_cl.data import PermutedMNIST; PermutedMNIST('$DATA', tasks=10, seed=0)"

COMMON=(
  --tasks 10
  --particles 8
  --width 100
  --depth 2
  --task1-epochs 15
  --particle-refine-epochs 3
  --batch-size 512
  --eval-batch-size 2048
  --workers 0
  --learning-rate 1e-3
  --particle-jitter 0.03
  --score learned
  --score-steps 1500
  --score-batch-size 128
  --score-learning-rate 2e-3
  --sigma-max 0.50
  --sigma-min 0.005
  --solution-kernel-std 0.05
  --reverse-steps 1200
  --guidance-mode proximal
  --posterior-learning-rate "$POSTERIOR_LR"
  --posterior-task-decay "$POSTERIOR_TASK_DECAY"
  --posterior-min-learning-rate "$POSTERIOR_MIN_LR"
  --posterior-max-step 0.05
  --curvature-strength 1.0
  --fisher-batches 64
  --fisher-batch-size 64
  --guidance-ramp-power 1
  --permutation-seed 0
  --data "$DATA"
  --output "$OUTPUT"
  --tag paper
)

CUDA_VISIBLE_DEVICES="$GPU0" "$PYTHON" -u -m gwd_cl.train "${COMMON[@]}" --seed "$SEED0" \
  >"$OUTPUT/logs/paper_seed${SEED0}.log" 2>&1 &
PID0=$!
CUDA_VISIBLE_DEVICES="$GPU1" "$PYTHON" -u -m gwd_cl.train "${COMMON[@]}" --seed "$SEED1" \
  >"$OUTPUT/logs/paper_seed${SEED1}.log" 2>&1 &
PID1=$!

trap 'kill "$PID0" "$PID1" 2>/dev/null || true' INT TERM
wait "$PID0"
STATUS0=$?
wait "$PID1"
STATUS1=$?

if [[ "$STATUS0" -ne 0 || "$STATUS1" -ne 0 ]]; then
  echo "A run failed (seed $SEED0: $STATUS0, seed $SEED1: $STATUS1)."
  echo "Inspect $OUTPUT/logs/paper_seed${SEED0}.log and $OUTPUT/logs/paper_seed${SEED1}.log"
  exit 1
fi

"$PYTHON" -m gwd_cl.summarize \
  "$OUTPUT/paper_seed${SEED0}/summary.json" \
  "$OUTPUT/paper_seed${SEED1}/summary.json"
