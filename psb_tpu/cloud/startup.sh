#!/bin/bash
# VM startup script (passed by create_vm.sh): format once, then mount the data disk.
DEV=/dev/disk/by-id/google-data-disk
if [ -z "$(sudo blkid $DEV)" ]; then
    sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard $DEV
fi
sudo mkdir -p /mnt/data
sudo mount -o discard,defaults $DEV /mnt/data
sudo chmod a+w /mnt/data
