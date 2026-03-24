#!/bin/bash
# run_benchmark.sh - Run MapQR 8-GPU benchmark on g7e.48xlarge
# Requires: Docker image built, patches applied, GKT compiled, data prepared
set -e

MAPQR_DIR=/opt/dlami/nvme/MapQR
WORK_DIR=$MAPQR_DIR/work_dirs
NCCL_LIB=/opt/pytorch/lib/python3.13/site-packages/nvidia/nccl/lib/libnccl.so.2

# Start GPU monitor
echo "[$(date)] Starting GPU monitor..."
bash $(dirname "$0")/gpu_monitor.sh &
GPU_MON_PID=$!

run_training() {
  local NAME=$1
  local BATCH=$2
  local EXTRA_OPTS=$3
  
  echo ""
  echo "============================================"
  echo "[$(date)] Running: $NAME (batch=$BATCH/GPU)"
  echo "============================================"
  
  docker run --gpus all --shm-size=64g --ipc=host \
    -v $MAPQR_DIR:/mapqr \
    -v $NCCL_LIB:/opt/nccl/libnccl.so.2:ro \
    -e PYTHONPATH=/mapqr:/mapqr/mmdetection3d \
    -e LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib \
    -e LD_PRELOAD=/opt/nccl/libnccl.so.2 \
    -w /mapqr \
    mapqr-g7e:latest bash -c "
cd /mapqr && torchrun --nproc_per_node=8 --master_port=29500 \
  tools/train.py projects/configs/mapqr/mapqr_nusc_r50_24ep.py \
  --launcher pytorch \
  --work-dir /mapqr/work_dirs/$NAME \
  --deterministic \
  --cfg-options evaluation.interval=999 data.samples_per_gpu=$BATCH data.workers_per_gpu=8 log_config.interval=10 $EXTRA_OPTS
" 2>&1 | tee $WORK_DIR/${NAME}.log
  
  echo "[$(date)] $NAME complete"
}

# Baseline: batch=4/GPU (default)
run_training "baseline_8gpu_bs4" 4

# Optimized: batch=8/GPU
run_training "optimized_8gpu_bs8" 8

# Stop GPU monitor
kill $GPU_MON_PID 2>/dev/null

echo ""
echo "============================================"
echo "[$(date)] All benchmarks complete!"
echo "============================================"
echo "Results in: $WORK_DIR/"
echo "GPU monitor: gpu_monitor.csv"
