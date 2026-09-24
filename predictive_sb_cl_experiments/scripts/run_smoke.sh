#!/bin/sh
set -eu

PYTHON=${PYTHON:-/export/home/anandr/miniforge3/envs/test2/bin/python}
GPU=${GPU:-0}

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -m ipsb.train \
  --method sb --tasks 2 --particles 4 --first-steps 20 \
  --stages 2 --steps-per-stage 5 --target-steps 5 \
  --memory-size 32 --current-probes 16 --eval-batch-size 5000 \
  --data ../data --output runs_smoke --tag smoke

