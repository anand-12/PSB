#!/usr/bin/env bash
# Create the TPU VM with the data disk attached and guardrails on.
#   bash cloud/create_vm.sh            # create
#   bash cloud/create_vm.sh --dry-run  # print the command only
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/config.env"
args=(compute instances create "$VM"
      --project "$PROJECT" --zone "$ZONE" --machine-type "$MACHINE"
      --provisioning-model "$PROVISIONING"
      --max-run-duration "$MAX_RUN" --instance-termination-action DELETE
      --image-project ubuntu-os-accelerator-images
      --image-family ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e
      --maintenance-policy TERMINATE --boot-disk-size 100GB
      --disk "name=$DISK,device-name=data-disk,mode=rw,boot=no"
      --metadata-from-file "startup-script=$HERE/startup.sh"
      --scopes cloud-platform)
if [ "$PROVISIONING" = FLEX_START ]; then
    args+=(--request-valid-for-duration 2h)   # give up if no capacity within 2h
fi
if [ "${1:-}" = --dry-run ]; then
    printf 'gcloud'; printf ' %q' "${args[@]}"; echo
    exit 0
fi
gcloud "${args[@]}"
echo "Requested. Flex-start waits for capacity; check with:"
echo "  gcloud compute instances describe $VM --zone $ZONE --project $PROJECT --format='value(status)'"
