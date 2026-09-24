#!/bin/sh
set -eu

PYTHON=${PYTHON:-/export/home/anandr/miniforge3/envs/test2/bin/python}
GPU0=${GPU0:-0}
GPU1=${GPU1:-1}

base="--tasks 10 --particles 8 --first-steps 1200 --stages 10 --steps-per-stage 60 --target-steps 250 --data ../data --seed 0"
best_sb="--endpoint-mix 0.5 --schedule power --schedule-power 0.25 --sinkhorn-epsilon-ratio 0.005"

CUDA_VISIBLE_DEVICES="$GPU0" "$PYTHON" -u -m ipsb.train \
  --method direct $base --memory-size 256 --current-probes 64 \
  --target-temperature 0 --output runs_endpoint --tag direct_map > direct_map.log 2>&1 &
pid0=$!
CUDA_VISIBLE_DEVICES="$GPU1" "$PYTHON" -u -m ipsb.train \
  --method sb $base $best_sb --memory-size 256 --current-probes 64 \
  --target-temperature 0 --output runs_endpoint --tag sb_map > sb_map.log 2>&1 &
pid1=$!
wait "$pid0"
wait "$pid1"

CUDA_VISIBLE_DEVICES="$GPU0" "$PYTHON" -u -m ipsb.train \
  --method direct $base --memory-size 1024 --current-probes 128 \
  --target-temperature 0 --output runs_endpoint --tag direct_map_m1k > direct_map_m1k.log 2>&1 &
pid0=$!
CUDA_VISIBLE_DEVICES="$GPU1" "$PYTHON" -u -m ipsb.train \
  --method sb $base $best_sb --memory-size 1024 --current-probes 128 \
  --target-temperature 0 --output runs_endpoint --tag sb_map_m1k > sb_map_m1k.log 2>&1 &
pid1=$!
wait "$pid0"
wait "$pid1"
