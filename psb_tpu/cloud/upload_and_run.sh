#!/usr/bin/env bash
# Copy the zip to the VM, set up the environment, start the sweep in tmux.
#   bash cloud/upload_and_run.sh path/to/psb_tpu.zip
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/config.env"
ZIP="${1:?usage: upload_and_run.sh path/to/psb_tpu.zip}"
G=(--project "$PROJECT" --zone "$ZONE")
gcloud compute scp "${G[@]}" "$ZIP" "$VM:/tmp/psb_tpu.zip"
gcloud compute ssh "${G[@]}" "$VM" --command "
set -e
mountpoint -q /mnt/data || echo 'warning: /mnt/data not mounted (startup script still running?)'
cd /mnt/data && python3 -m zipfile -e /tmp/psb_tpu.zip . && cd psb_tpu   # the image has no unzip
bash setup.sh
tmux kill-session -t psb 2>/dev/null || true
tmux new -d -s psb 'GCS_BUCKET=$GCS_BUCKET COST_CAP_USD=$COST_CAP_USD SELF_DELETE=$SELF_DELETE bash /mnt/data/psb_tpu/run_all.sh 2>&1 | tee -a /mnt/data/run_all.log'
echo 'Sweep started in tmux session psb.'
"
