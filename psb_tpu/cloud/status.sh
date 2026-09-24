#!/usr/bin/env bash
# Show the latest progress lines from the VM.
set -euo pipefail
source "$(dirname "$0")/config.env"
gcloud compute ssh --project "$PROJECT" --zone "$ZONE" "$VM" \
    --command "tail -n ${1:-25} /mnt/data/run_all.log; echo; ls /mnt/data/psb_results 2>/dev/null"
