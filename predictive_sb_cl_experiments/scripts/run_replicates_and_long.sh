#!/bin/sh
set -eu

PYTHON=${PYTHON:-/export/home/anandr/miniforge3/envs/test2/bin/python}
GPU0=${GPU0:-0}
GPU1=${GPU1:-1}

common="--particles 8 --first-steps 1200 --stages 10 --steps-per-stage 60 --memory-size 256 --current-probes 64 --target-steps 250 --data ../data"
best_sb="--endpoint-mix 0.5 --schedule power --schedule-power 0.25 --sinkhorn-epsilon-ratio 0.005"

for seed in 1 2; do
  CUDA_VISIBLE_DEVICES="$GPU0" "$PYTHON" -u -m ipsb.train \
    --method direct --tasks 5 $common --seed "$seed" \
    --output runs_replicates --tag direct > "direct_seed${seed}.log" 2>&1 &
  pid0=$!
  CUDA_VISIBLE_DEVICES="$GPU1" "$PYTHON" -u -m ipsb.train \
    --method sb --tasks 5 $common $best_sb --seed "$seed" \
    --output runs_replicates --tag sb_best > "sb_best_seed${seed}.log" 2>&1 &
  pid1=$!
  wait "$pid0"
  wait "$pid1"
done

CUDA_VISIBLE_DEVICES="$GPU0" "$PYTHON" -u -m ipsb.train \
  --method direct --tasks 10 $common --seed 0 \
  --output runs_long --tag direct > direct_long.log 2>&1 &
pid0=$!
CUDA_VISIBLE_DEVICES="$GPU1" "$PYTHON" -u -m ipsb.train \
  --method sb --tasks 10 $common $best_sb --seed 0 \
  --output runs_long --tag sb_best > sb_best_long.log 2>&1 &
pid1=$!
wait "$pid0"
wait "$pid1"
