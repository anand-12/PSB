#!/bin/sh
set -eu

PYTHON=${PYTHON:-/export/home/anandr/miniforge3/envs/test2/bin/python}
GPU0=${GPU0:-0}
GPU1=${GPU1:-1}

run() {
  gpu=$1
  tag=$2
  shift 2
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u -m ipsb.train \
    --method sb --tasks 5 --particles 8 --first-steps 1200 \
    --stages 10 --steps-per-stage 60 --memory-size 256 --current-probes 64 \
    --target-steps 250 --data ../data --output runs_lifting --tag "$tag" "$@"
}

run "$GPU0" endpoint25 --endpoint-mix 0.25 > endpoint25.log 2>&1 &
pid0=$!
run "$GPU1" endpoint50 --endpoint-mix 0.5 > endpoint50.log 2>&1 &
pid1=$!
wait "$pid0"
wait "$pid1"

run "$GPU0" endpoint75 --endpoint-mix 0.75 > endpoint75.log 2>&1 &
pid0=$!
run "$GPU1" power_fast \
  --endpoint-mix 0.5 --schedule power --schedule-power 0.25 > power_fast.log 2>&1 &
pid1=$!
wait "$pid0"
wait "$pid1"

run "$GPU0" power_low_noise \
  --endpoint-mix 0.5 --schedule power --schedule-power 0.25 \
  --sinkhorn-epsilon-ratio 0.005 > power_low_noise.log 2>&1
