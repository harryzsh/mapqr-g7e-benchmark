#!/bin/bash
# download_nuscenes.sh - Download nuScenes dataset from public S3
set -e
DEST=/opt/dlami/nvme/MapQR/data/nuscenes
mkdir -p $DEST && cd $DEST

echo "[$(date)] Downloading core files..."
aws s3 cp s3://motional-nuscenes/public/v1.0/v1.0-mini.tgz . --no-sign-request &
aws s3 cp s3://motional-nuscenes/public/v1.0/v1.0-trainval_meta.tgz . --no-sign-request &
aws s3 cp s3://motional-nuscenes/public/v1.0/nuScenes-map-expansion-v1.3.zip . --no-sign-request &
aws s3 cp s3://motional-nuscenes/public/v1.0/can_bus.zip . --no-sign-request &
wait

echo "[$(date)] Downloading trainval blobs (batch 1: 01-04)..."
for i in 01 02 03 04; do
  aws s3 cp s3://motional-nuscenes/public/v1.0/v1.0-trainval${i}_blobs.tgz . --no-sign-request &
done
wait

echo "[$(date)] Downloading trainval blobs (batch 2: 05-08)..."
for i in 05 06 07 08; do
  aws s3 cp s3://motional-nuscenes/public/v1.0/v1.0-trainval${i}_blobs.tgz . --no-sign-request &
done
wait

echo "[$(date)] Downloading trainval blobs (batch 3: 09-10)..."
aws s3 cp s3://motional-nuscenes/public/v1.0/v1.0-trainval09_blobs.tgz . --no-sign-request &
aws s3 cp s3://motional-nuscenes/public/v1.0/v1.0-trainval10_blobs.tgz . --no-sign-request &
wait

echo "[$(date)] Extracting..."
tar -xzf v1.0-mini.tgz &
tar -xzf v1.0-trainval_meta.tgz &
unzip -qo nuScenes-map-expansion-v1.3.zip -d maps &
unzip -qo can_bus.zip -d /opt/dlami/nvme/MapQR/data/ &
wait
for f in v1.0-trainval*_blobs.tgz; do
  [ -f "$f" ] && tar -xzf "$f" &
done
wait
echo "[$(date)] Done. $(du -sh $DEST)"
