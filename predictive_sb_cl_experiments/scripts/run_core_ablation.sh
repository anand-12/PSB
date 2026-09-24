#!/bin/sh
set -eu

PYTHON=${PYTHON:-/export/home/anandr/miniforge3/envs/test2/bin/python}
GPU0=${GPU0:-0}
GPU1=${GPU1:-1}

run() {
  gpu=$1
  method=$2
  shift 2
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u -m ipsb.train \
    --method "$method" --tasks 5 --particles 8 --first-steps 1200 \
    --stages 10 --steps-per-stage 60 --memory-size 256 --current-probes 64 \
    --target-steps 250 --data ../data --output runs_core --tag core "$@"
}

run "$GPU0" direct > direct.log 2>&1 &
pid0=$!
run "$GPU1" sb > sb.log 2>&1 &
pid1=$!
wait "$pid0"
wait "$pid1"

run "$GPU0" ot_path > ot_path.log 2>&1 &
pid0=$!
run "$GPU1" random_path > random_path.log 2>&1 &
pid1=$!
wait "$pid0"
wait "$pid1"

