#!/usr/bin/env bash
# Keep a provisioned TPU VM alive while nobody is watching.
#
# Hunts for capacity (on-demand first, because Spot gets preempted), installs
# the package and the Python environment, then supervises: if the VM is
# preempted or hits its run-duration limit, it is recreated and reprovisioned.
# The sweep itself is NOT started -- that is left to you.
#
#   bash cloud/keep_tpu_up.sh                  # supervise for SUPERVISE_HOURS
#   touch cloud/.stop_supervisor               # stop after the current check
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PROJECT="${PROJECT:-sodium-hue-408214}"
VM="${VM:-psb-tpu}"
ZONE="${ZONE:-europe-west4-a}"
MAX_RUN="${MAX_RUN:-8h}"
SUPERVISE_HOURS="${SUPERVISE_HOURS:-12}"
ZIP="${ZIP:-$(dirname "$(dirname "$HERE")")/dist/psb_tpu.zip}"
G=(gcloud --project="$PROJECT")
STARTED=$(date +%s)
STATUS="$HERE/.tpu_status"

log() { echo "$(date +%H:%M:%S) $*"; }

# Shapes in order: stable first, then cheap, then v5e. Only fail-fast models.
SHAPES=(
    "$ZONE ct6e-standard-1t STANDARD"
    "$ZONE ct6e-standard-4t STANDARD"
    "$ZONE ct6e-standard-1t SPOT"
    "$ZONE ct6e-standard-4t SPOT"
    "europe-west4-b ct6e-standard-1t STANDARD"
)

create_one() {   # zone machine model
    local zone=$1 machine=$2 model=$3 status
    "${G[@]}" compute disks describe psb-data --zone "$zone" >/dev/null 2>&1 || \
        "${G[@]}" compute disks create psb-data --zone "$zone" --size 100GB \
            --type hyperdisk-balanced >/dev/null 2>&1
    timeout 600 "${G[@]}" compute instances create "$VM" --zone "$zone" \
        --machine-type "$machine" --provisioning-model "$model" \
        --instance-termination-action DELETE --max-run-duration "$MAX_RUN" \
        --maintenance-policy TERMINATE \
        --image-project ubuntu-os-accelerator-images \
        --image-family ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e \
        --boot-disk-size 100GB --boot-disk-type hyperdisk-balanced \
        --disk "name=psb-data,device-name=data-disk,mode=rw,boot=no" \
        --scopes cloud-platform \
        --metadata-from-file "startup-script=$HERE/startup.sh" >/dev/null 2>&1
    status=$("${G[@]}" compute instances describe "$VM" --zone "$zone" --format='value(status)' 2>/dev/null)
    if [ "$status" = RUNNING ]; then
        ZONE="$zone"
        log "UP: $machine $model in $zone"
        return 0
    fi
    [ -n "$status" ] && "${G[@]}" compute instances delete "$VM" --zone "$zone" --quiet >/dev/null 2>&1
    log "no capacity: $machine $model $zone"
    return 1
}

hunt() {
    local shape
    for shape in "${SHAPES[@]}"; do
        # shellcheck disable=SC2086
        create_one $shape && return 0
    done
    return 1
}

provision() {
    local i
    for i in $(seq 1 30); do
        timeout 90 "${G[@]}" compute ssh "$VM" --zone "$ZONE" --quiet --command "true" >/dev/null 2>&1 && break
        sleep 15
    done
    timeout 300 "${G[@]}" compute scp "$ZIP" "$VM:/tmp/psb_tpu.zip" --zone "$ZONE" --quiet >/dev/null 2>&1 || return 1
    # the TPU image has no unzip, so use python's zipfile
    timeout 1800 "${G[@]}" compute ssh "$VM" --zone "$ZONE" --quiet --command \
        "set -e; cd /mnt/data && python3 -m zipfile -e /tmp/psb_tpu.zip . && cd psb_tpu && bash setup.sh" 2>&1 | tail -5
    return "${PIPESTATUS[0]}"
}

while :; do
    [ -f "$HERE/.stop_supervisor" ] && { log "stop file present; exiting"; exit 0; }
    if [ $(( ($(date +%s) - STARTED) / 3600 )) -ge "$SUPERVISE_HOURS" ]; then
        log "supervised for $SUPERVISE_HOURS h; exiting (VM left as is)"
        exit 0
    fi
    state=$("${G[@]}" compute instances describe "$VM" --zone "$ZONE" --format='value(status)' 2>/dev/null)
    if [ "$state" = RUNNING ]; then
        sleep 60
        continue
    fi
    log "no running VM (state='${state:-none}'); hunting"
    if hunt; then
        if provision; then
            printf 'zone=%s\nvm=%s\nprovisioned=%s\n' "$ZONE" "$VM" "$(date -Is)" > "$STATUS"
            log "provisioned and ready in $ZONE"
            if [ "${RESUME_SWEEP:-0}" = 1 ]; then
                "${G[@]}" compute ssh "$VM" --zone "$ZONE" --quiet --command \
                    "tmux new -d -s psb 'bash /mnt/data/psb_tpu/run_all.sh'" >/dev/null 2>&1 \
                    && log "sweep resumed in tmux"
            fi
        else
            # never leave an unprovisioned VM running (and billing) idle
            log "provisioning failed; deleting the VM and retrying"
            "${G[@]}" compute instances delete "$VM" --zone "$ZONE" --quiet >/dev/null 2>&1
            sleep 60
        fi
    else
        log "no capacity anywhere; retrying in 3 min"
        sleep 180
    fi
done
