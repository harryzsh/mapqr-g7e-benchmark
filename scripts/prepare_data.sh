#!/bin/bash
# prepare_data.sh - Generate nuScenes pkl files for MapQR training
set -e

docker run --rm --gpus all \
  -v /opt/dlami/nvme/MapQR:/mapqr \
  -e PYTHONPATH=/mapqr:/mapqr/mmdetection3d \
  -e LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib \
  -w /mapqr \
  mapqr-g7e:latest bash -c "
echo '[$(date)] Generating trainval pkl...'
python tools/maptrv2/custom_nusc_map_converter.py \
  --root-path ./data/nuscenes --out-dir ./data/nuscenes \
  --extra-tag nuscenes --version v1.0 --canbus ./data

echo '[$(date)] PKL files:'
ls -lh ./data/nuscenes/*.pkl
"
