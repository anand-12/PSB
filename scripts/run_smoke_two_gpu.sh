#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
DATA="${DATA:-data}"
OUTPUT="${OUTPUT:-runs}"
mkdir -p "$OUTPUT/logs"
"$PYTHON" -c "from gwd_cl.data import PermutedMNIST; PermutedMNIST('$DATA', tasks=5, seed=0)"

COMMON=(
  --tasks 5 --particles 4 --width 100 --depth 2
  --task1-epochs 2 --particle-refine-epochs 1
  --batch-size 512 --eval-batch-size 2048 --workers 0
  --score-steps 100 --reverse-steps 100
  --sigma-max 0.50 --sigma-min 0.005 --solution-kernel-std 0.15
  --guidance-mode proximal --posterior-learning-rate 0.05 --posterior-max-step 0.05
  --permutation-seed 0 --data "$DATA" --output "$OUTPUT" --tag smoke
)

CUDA_VISIBLE_DEVICES="$GPU0" "$PYTHON" -u -m gwd_cl.train "${COMMON[@]}" --seed 0 \
  >"$OUTPUT/logs/smoke_seed0.log" 2>&1 &
PID0=$!
CUDA_VISIBLE_DEVICES="$GPU1" "$PYTHON" -u -m gwd_cl.train "${COMMON[@]}" --seed 1 \
  >"$OUTPUT/logs/smoke_seed1.log" 2>&1 &
PID1=$!
trap 'kill "$PID0" "$PID1" 2>/dev/null || true' INT TERM
wait "$PID0"; STATUS0=$?
wait "$PID1"; STATUS1=$?
if [[ "$STATUS0" -ne 0 || "$STATUS1" -ne 0 ]]; then
  echo "Smoke test failed. Inspect $OUTPUT/logs/smoke_seed*.log"
  exit 1
fi
"$PYTHON" -m gwd_cl.summarize "$OUTPUT/smoke_seed0" "$OUTPUT/smoke_seed1"
