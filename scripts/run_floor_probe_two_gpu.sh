#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
DATA="${DATA:-data}"
OUTPUT="${OUTPUT:-runs_floor_probe}"
mkdir -p "$OUTPUT/logs"
"$PYTHON" -c "from gwd_cl.data import PermutedMNIST; PermutedMNIST('$DATA', tasks=5, seed=0)"

COMMON=(
  --tasks 5 --particles 8 --width 100 --depth 2
  --task1-epochs 15 --particle-refine-epochs 3
  --batch-size 512 --eval-batch-size 2048 --workers 0
  --learning-rate 1e-3 --particle-jitter 0.03
  --score learned --score-steps 1500 --score-batch-size 128
  --score-learning-rate 2e-3
  --sigma-max 0.50 --sigma-min 0.005 --solution-kernel-std 0.05
  --reverse-steps 1200 --guidance-mode proximal
  --posterior-learning-rate 0.03 --posterior-task-decay 1.0
  --posterior-max-step 0.05 --guidance-ramp-power 1
  --curvature-strength 1.0 --curvature-damping 0.1
  --mobility-minimum 0.05 --mobility-maximum 3
  --fisher-batches 64 --fisher-batch-size 64
  --permutation-seed 0 --data "$DATA" --output "$OUTPUT" --seed 0
)

CUDA_VISIBLE_DEVICES="$GPU0" "$PYTHON" -u -m gwd_cl.train "${COMMON[@]}" \
  --posterior-min-learning-rate 0.010 --tag floor0010 \
  >"$OUTPUT/logs/floor0010.log" 2>&1 &
PID0=$!
CUDA_VISIBLE_DEVICES="$GPU1" "$PYTHON" -u -m gwd_cl.train "${COMMON[@]}" \
  --posterior-min-learning-rate 0.0125 --tag floor00125 \
  >"$OUTPUT/logs/floor00125.log" 2>&1 &
PID1=$!

trap 'kill "$PID0" "$PID1" 2>/dev/null || true' INT TERM
wait "$PID0"; STATUS0=$?
wait "$PID1"; STATUS1=$?
if [[ "$STATUS0" -ne 0 || "$STATUS1" -ne 0 ]]; then
  echo "A learning-rate-floor probe failed. Inspect $OUTPUT/logs/floor*.log"
  exit 1
fi

echo "posterior learning-rate floor 0.010:"
tail -n 2 "$OUTPUT/logs/floor0010.log" | head -n 1
echo "posterior learning-rate floor 0.0125:"
tail -n 2 "$OUTPUT/logs/floor00125.log" | head -n 1

