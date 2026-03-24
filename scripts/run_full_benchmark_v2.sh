#!/bin/bash
# run_full_benchmark_v2.sh — Complete benchmark suite for g7e.48xlarge
# Runs: synthetic GPU tests, full dataset training, DDP scaling, pin_memory test
set -e

MAPQR=/opt/dlami/nvme/MapQR
NCCL_LIB=/opt/pytorch/lib/python3.13/site-packages/nvidia/nccl/lib/libnccl.so.2
RESULTS=/opt/dlami/nvme/benchmark_results
mkdir -p $RESULTS

DOCKER_CMD="docker run --rm --gpus all --shm-size=64g --ipc=host \
  -v $MAPQR:/mapqr \
  -v $NCCL_LIB:/opt/nccl/libnccl.so.2:ro \
  -e PYTHONPATH=/mapqr:/mapqr/mmdetection3d \
  -e LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib \
  -e LD_PRELOAD=/opt/nccl/libnccl.so.2 \
  -w /mapqr \
  mapqr-g7e:ready"

echo "============================================"
echo "[$(date)] MapQR Benchmark Suite v2"
echo "============================================"

# Verify dataset
echo ""
echo "=== Dataset Check ==="
$DOCKER_CMD python3 -c "
import pickle
d=pickle.load(open('/mapqr/data/nuscenes/nuscenes_map_infos_temporal_train.pkl','rb'))
print(f'Train: {len(d[\"infos\"])} scenes')
"

# ==========================================
# Phase 1: Synthetic GPU Benchmark
# ==========================================
echo ""
echo "============================================"
echo "Phase 1: Synthetic GPU Benchmark"
echo "============================================"

# A1: Single GPU, bs=4
echo "[$(date)] A1: 1-GPU FP32 bs=4"
$DOCKER_CMD python /mapqr/scripts/benchmark_synthetic.py --batch-size 4 2>&1 | tee $RESULTS/A1_1gpu_fp32_bs4.txt

# A2: Single GPU, bs=8
echo "[$(date)] A2: 1-GPU FP32 bs=8"
$DOCKER_CMD python /mapqr/scripts/benchmark_synthetic.py --batch-size 8 2>&1 | tee $RESULTS/A2_1gpu_fp32_bs8.txt

# A3: 8-GPU, bs=4
echo "[$(date)] A3: 8-GPU FP32 bs=4"
$DOCKER_CMD torchrun --nproc_per_node=8 /mapqr/scripts/benchmark_synthetic.py --batch-size 4 2>&1 | tee $RESULTS/A3_8gpu_fp32_bs4.txt

# A4: 8-GPU, bs=8
echo "[$(date)] A4: 8-GPU FP32 bs=8"
$DOCKER_CMD torchrun --nproc_per_node=8 /mapqr/scripts/benchmark_synthetic.py --batch-size 8 2>&1 | tee $RESULTS/A4_8gpu_fp32_bs8.txt

# A5: 8-GPU, bs=8, BF16
echo "[$(date)] A5: 8-GPU BF16 bs=8"
$DOCKER_CMD torchrun --nproc_per_node=8 /mapqr/scripts/benchmark_synthetic.py --batch-size 8 --amp 2>&1 | tee $RESULTS/A5_8gpu_bf16_bs8.txt

# DDP scaling: 2-GPU and 4-GPU
echo "[$(date)] A6: 2-GPU FP32 bs=4"
docker run --rm --gpus '"device=0,1"' --shm-size=64g --ipc=host \
  -v $MAPQR:/mapqr -v $NCCL_LIB:/opt/nccl/libnccl.so.2:ro \
  -e PYTHONPATH=/mapqr:/mapqr/mmdetection3d \
  -e LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib \
  -e LD_PRELOAD=/opt/nccl/libnccl.so.2 -w /mapqr \
  mapqr-g7e:ready torchrun --nproc_per_node=2 /mapqr/scripts/benchmark_synthetic.py --batch-size 4 2>&1 | tee $RESULTS/A6_2gpu_fp32_bs4.txt

echo "[$(date)] A7: 4-GPU FP32 bs=4"
docker run --rm --gpus '"device=0,1,2,3"' --shm-size=64g --ipc=host \
  -v $MAPQR:/mapqr -v $NCCL_LIB:/opt/nccl/libnccl.so.2:ro \
  -e PYTHONPATH=/mapqr:/mapqr/mmdetection3d \
  -e LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib \
  -e LD_PRELOAD=/opt/nccl/libnccl.so.2 -w /mapqr \
  mapqr-g7e:ready torchrun --nproc_per_node=4 /mapqr/scripts/benchmark_synthetic.py --batch-size 4 2>&1 | tee $RESULTS/A7_4gpu_fp32_bs4.txt

# ==========================================
# Phase 2: Full Dataset Training (3 epochs)
# ==========================================
echo ""
echo "============================================"
echo "Phase 2: Full Dataset Training (3 epochs)"
echo "============================================"

run_train() {
  local NAME=$1
  local CONFIG=$2
  echo "[$(date)] $NAME"
  $DOCKER_CMD bash -c "cd /mapqr && torchrun --nproc_per_node=8 --master_port=29500 \
    tools/train.py $CONFIG --launcher pytorch \
    --work-dir /mapqr/work_dirs/$NAME --deterministic \
    --cfg-options total_epochs=3 runner.max_epochs=3" 2>&1 | tee $RESULTS/${NAME}.txt
}

# B1: Default config
run_train "B1_default" "projects/configs/mapqr/mapqr_nusc_r50_24ep.py --cfg-options evaluation.interval=999"

# B2: Batch scaling
run_train "B2_bs8" "projects/configs/mapqr/mapqr_nusc_r50_24ep.py --cfg-options evaluation.interval=999 data.samples_per_gpu=8"

# B3: All optimizations FP32
run_train "B3_optimized_fp32" "projects/configs/mapqr/mapqr_nusc_r50_24ep_optimized.py --cfg-options total_epochs=3 runner.max_epochs=3"

# B4: All optimizations BF16
run_train "B4_optimized_bf16" "projects/configs/mapqr/mapqr_nusc_r50_24ep_bf16_optimized.py --cfg-options total_epochs=3 runner.max_epochs=3"

echo ""
echo "============================================"
echo "[$(date)] ALL BENCHMARKS COMPLETE"
echo "Results in: $RESULTS/"
echo "============================================"
ls -la $RESULTS/
