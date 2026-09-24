#!/usr/bin/env bash
# Once per project: the persistent disk that outlives every VM.
set -euo pipefail
source "$(dirname "$0")/config.env"
gcloud compute disks create "$DISK" --project "$PROJECT" --zone "$ZONE" \
    --size "$DISK_SIZE" --type hyperdisk-balanced
