#!/bin/sh
# Provision a v6e flex-start TPU VM for the ICLR sprint.
#
# Order of operations, first time only:
#   1. Create a NET-NEW GCP project (reusing an old TRC project causes the
#      "code 10, Failed to perform tenant project creation" error).
#   2. gcloud auth login && gcloud config set project "$PROJECT"
#   3. gcloud services enable tpu.googleapis.com
#   4. gcloud components install alpha --quiet
#   5. Run the program's pre-flight diagnostic:
#      curl -sSO https://gist.githubusercontent.com/RobMulla/ee1a530f9ff0bdb9aa5b493c7faf9aa2/raw/tpu_diagnostic.py && python3 tpu_diagnostic.py
#      If GPUS_ALL_REGIONS is 0, stop and email tpu-builders-support@google.com.
#   6. Create the data disk (once):
#      gcloud compute disks create psb-data --size=100GB --zone="$ZONE" --type=hyperdisk-balanced
set -eu

PROJECT=${PROJECT:?set PROJECT}
ZONE=${ZONE:-us-east5-a}          # v6e zones: us-east5-a/b, us-central1-a, europe-west4-a, southamerica-west1-a
MACHINE=${MACHINE:-ct6e-standard-4t}   # 4 chips = 4 JAX devices; ct6e-standard-1t for cheap dev
VM=${VM:-psb-v6e}
DISK=${DISK:-psb-data}
# Flex-start guarantees uninterrupted uptime once provisioned, up to 7 days.
# It CANNOT be stopped, only deleted, so idle time burns credits: keep this
# matched to the work actually queued.
RUN_FOR=${RUN_FOR:-8h}

gcloud compute instances create "$VM" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --machine-type="$MACHINE" \
  --provisioning-model=FLEX_START \
  --request-valid-for-duration=2h \
  --max-run-duration="$RUN_FOR" \
  --instance-termination-action=DELETE \
  --image-project=ubuntu-os-accelerator-images \
  --image-family=ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e \
  --maintenance-policy=TERMINATE \
  --boot-disk-size=100GB \
  --disk=name="$DISK",device-name=data-disk,mode=rw,boot=no \
  --metadata-from-file=startup-script="$(dirname "$0")/startup.sh"

echo "Status:"
echo "  gcloud compute instances describe $VM --zone=$ZONE"
echo "Connect:"
echo "  gcloud compute ssh $VM --zone=$ZONE"
echo "Delete when done (flex-start cannot be paused):"
echo "  gcloud compute instances delete $VM --zone=$ZONE --quiet"
