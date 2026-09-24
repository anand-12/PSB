#!/usr/bin/env bash
# Download results (without mid-run checkpoints) to ./results_tpu.
set -euo pipefail
source "$(dirname "$0")/config.env"
G=(--project "$PROJECT" --zone "$ZONE")
gcloud compute ssh "${G[@]}" "$VM" --command \
    "cd /mnt/data && tar czf /tmp/psb_results.tgz --exclude='_state' psb_results"
gcloud compute scp "${G[@]}" "$VM:/tmp/psb_results.tgz" ./psb_results.tgz
mkdir -p results_tpu && tar xzf psb_results.tgz -C results_tpu && rm psb_results.tgz
echo "Results in ./results_tpu/psb_results (see */summary.md)"
