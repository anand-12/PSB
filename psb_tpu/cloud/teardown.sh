#!/usr/bin/env bash
# Delete the VM (billing stops). The data disk and its results are kept.
set -euo pipefail
source "$(dirname "$0")/config.env"
gcloud compute instances delete "$VM" --project "$PROJECT" --zone "$ZONE" --quiet
echo "VM deleted. Disk $DISK kept; delete it with:"
echo "  gcloud compute disks delete $DISK --zone $ZONE --project $PROJECT"
