#!/bin/bash
# Startup script for the flex-start TPU VM. Mounts the persistent data disk and
# installs JAX. Flex-start VMs cannot be stopped and restarted, only deleted,
# so everything that must survive a VM lives on /mnt/data.

set -x

# Format the data disk only if it has no filesystem yet.
if [ -z "$(sudo blkid /dev/disk/by-id/google-data-disk)" ]; then
  sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard \
    /dev/disk/by-id/google-data-disk
fi

sudo mkdir -p /mnt/data
sudo mount -o discard,defaults /dev/disk/by-id/google-data-disk /mnt/data
sudo chmod a+w /mnt/data

# Python environment on the data disk so it survives VM deletion.
sudo apt-get update && sudo apt-get install -y python3-pip python3-venv git

if [ ! -d /mnt/data/tpu-env ]; then
  python3 -m venv /mnt/data/tpu-env
  /mnt/data/tpu-env/bin/pip install -U pip
  /mnt/data/tpu-env/bin/pip install "jax[tpu]" \
    -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
  /mnt/data/tpu-env/bin/pip install flax optax tensorflow-datasets numpy
fi

# Sanity check; shows up in the serial console log.
/mnt/data/tpu-env/bin/python -c "import jax; print('TPU devices:', jax.device_count())"
