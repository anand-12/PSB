#!/usr/bin/env bash
# Keep trying TPU shapes until one actually boots.
#
# Only fail-fast provisioning is used (SPOT, then on-demand), so nothing ever
# sits in a Flex-start queue. Cheapest and most schedulable shapes come first.
# The winner is written to cloud/.winner and the script exits.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PROJECT="${PROJECT:-sodium-hue-408214}"
VM="${VM:-psb-tpu}"
MAX_RUN="${MAX_RUN:-8h}"
WINNER="$HERE/.winner"
G=(gcloud --project="$PROJECT")
rm -f "$WINNER"

log() { echo "$(date +%H:%M:%S) $*"; }

try_gce() {   # zone machine provisioning-model
    local zone=$1 machine=$2 model=$3 out status
    "${G[@]}" compute disks describe psb-data --zone "$zone" >/dev/null 2>&1 || \
        "${G[@]}" compute disks create psb-data --zone "$zone" --size 100GB \
            --type hyperdisk-balanced >/dev/null 2>&1
    out=$(timeout 600 "${G[@]}" compute instances create "$VM" --zone "$zone" \
        --machine-type "$machine" --provisioning-model "$model" \
        --instance-termination-action DELETE --max-run-duration "$MAX_RUN" \
        --maintenance-policy TERMINATE \
        --image-project ubuntu-os-accelerator-images \
        --image-family ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e \
        --boot-disk-size 100GB --boot-disk-type hyperdisk-balanced \
        --disk "name=psb-data,device-name=data-disk,mode=rw,boot=no" \
        --scopes cloud-platform \
        --metadata-from-file "startup-script=$HERE/startup.sh" 2>&1)
    status=$("${G[@]}" compute instances describe "$VM" --zone "$zone" \
        --format='value(status)' 2>/dev/null)
    if [ "$status" = RUNNING ]; then
        log "UP: $machine $model in $zone"
        printf 'kind=gce\nzone=%s\nmachine=%s\nmodel=%s\nvm=%s\n' "$zone" "$machine" "$model" "$VM" > "$WINNER"
        return 0
    fi
    [ -n "$status" ] && "${G[@]}" compute instances delete "$VM" --zone "$zone" --quiet >/dev/null 2>&1
    log "no: $machine $model $zone -- $(echo "$out" | grep -oiE 'stockout|does not have enough resources|quota[^.]*|not supported[^.]*' | head -1)"
    return 1
}

try_v5e() {   # zone accelerator-type [--spot]
    local zone=$1 type=$2 spot=${3:-} out status node="${VM}-v5e"
    out=$(timeout 600 "${G[@]}" compute tpus tpu-vm create "$node" --zone "$zone" \
        --accelerator-type "$type" --version v2-alpha-tpuv5-lite $spot 2>&1)
    status=$("${G[@]}" compute tpus tpu-vm describe "$node" --zone "$zone" \
        --format='value(state)' 2>/dev/null)
    if [ "$status" = READY ]; then
        log "UP: v5e $type ${spot:-on-demand} in $zone"
        printf 'kind=v5e\nzone=%s\nmachine=%s\nmodel=%s\nvm=%s\n' "$zone" "$type" "${spot:-ondemand}" "$node" > "$WINNER"
        return 0
    fi
    [ -n "$status" ] && "${G[@]}" compute tpus tpu-vm delete "$node" --zone "$zone" --quiet >/dev/null 2>&1
    log "no: v5e $type ${spot:-on-demand} $zone -- $(echo "$out" | grep -oiE 'stockout|unavailable|quota[^.]*|permission[^.]*|error[^.]*' | head -1)"
    return 1
}

round=0
while :; do
    round=$((round + 1))
    log "--- round $round ---"
    # v6e: CT6E family quota exists only in europe-west4 (48 chips)
    try_gce europe-west4-a ct6e-standard-1t SPOT     && break
    try_gce europe-west4-a ct6e-standard-4t SPOT     && break
    try_gce europe-west4-b ct6e-standard-1t SPOT     && break
    try_gce europe-west4-a ct6e-standard-1t STANDARD && break
    try_gce europe-west4-a ct6e-standard-4t STANDARD && break
    # v5e: legacy API, quota confirmed in the diagnostic report
    try_v5e europe-west4-b v5litepod-4 --spot        && break
    try_v5e us-west4-a     v5litepod-4 --spot        && break
    try_v5e europe-west4-b v5litepod-4               && break
    try_v5e us-west4-a     v5litepod-4               && break
    log "all shapes unavailable; retrying in 3 min"
    sleep 180
done
log "done -- see $WINNER"
cat "$WINNER"
