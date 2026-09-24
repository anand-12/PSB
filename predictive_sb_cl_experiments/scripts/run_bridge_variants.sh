#!/bin/sh
set -eu

PYTHON=${PYTHON:-/export/home/anandr/miniforge3/envs/test2/bin/python}
GPU0=${GPU0:-0}
GPU1=${GPU1:-1}

run() {
  gpu=$1
  method=$2
  tag=$3
  shift 3
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u -m ipsb.train \
    --method "$method" --tasks 5 --particles 8 --first-steps 1200 \
    --stages 10 --steps-per-stage 60 --memory-size 256 --current-probes 64 \
    --target-steps 250 --data ../data --output runs_variants --tag "$tag" "$@"
}

run "$GPU0" finetune finetune > finetune.log 2>&1 &
pid0=$!
run "$GPU1" sb sb_deterministic --no-bridge-noise > sb_deterministic.log 2>&1 &
pid1=$!
wait "$pid0"
wait "$pid1"

run "$GPU0" sb sb_low_noise --sinkhorn-epsilon-ratio 0.005 > sb_low_noise.log 2>&1 &
pid0=$!
run "$GPU1" sb sb_cosine --schedule cosine > sb_cosine.log 2>&1 &
pid1=$!
wait "$pid0"
wait "$pid1"

run "$GPU0" sb sb_high_entropy --sinkhorn-epsilon-ratio 0.5 > sb_high_entropy.log 2>&1

