#!/bin/bash
# run_full_benchmark_v2.sh — 8-GPU benchmark suite for g7e.48xlarge
set -e

MAPQR=/opt/dlami/nvme/MapQR
NCCL_LIB=/opt/pytorch/lib/python3.13/site-packages/nvidia/nccl/lib/libnccl.so.2
RESULTS=/opt/dlami/nvme/benchmark_results
mkdir -p $RESULTS

RUN="docker run --rm --gpus all --shm-size=64g --ipc=host \
  -v $MAPQR:/mapqr \
  -v $NCCL_LIB:/opt/nccl/libnccl.so.2:ro \
  -e PYTHONPATH=/mapqr:/mapqr/mmdetection3d \
  -e LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib \
  -e LD_PRELOAD=/opt/nccl/libnccl.so.2 \
  -w /mapqr mapqr-g7e:ready"

echo "============================================"
echo "[$(date)] MapQR 8-GPU Benchmark Suite"
echo "============================================"

# Dataset check
$RUN python3 -c "
import pickle
d=pickle.load(open('/mapqr/data/nuscenes/nuscenes_map_infos_temporal_train.pkl','rb'))
print(f'Train: {len(d[\"infos\"])} scenes')
"

# ==========================================
# Phase 1: Synthetic GPU (8-GPU, no I/O)
# ==========================================
echo ""
echo "=== Phase 1: Synthetic 8-GPU Benchmark ==="

echo "[$(date)] A1: 8-GPU FP32 bs=4"
$RUN torchrun --nproc_per_node=8 /mapqr/scripts/benchmark_synthetic.py --batch-size 4 2>&1 | tee $RESULTS/A1_8gpu_fp32_bs4.txt

echo "[$(date)] A2: 8-GPU FP32 bs=8"
$RUN torchrun --nproc_per_node=8 /mapqr/scripts/benchmark_synthetic.py --batch-size 8 2>&1 | tee $RESULTS/A2_8gpu_fp32_bs8.txt

echo "[$(date)] A3: 8-GPU BF16 bs=4"
$RUN torchrun --nproc_per_node=8 /mapqr/scripts/benchmark_synthetic.py --batch-size 4 --amp 2>&1 | tee $RESULTS/A3_8gpu_bf16_bs4.txt

echo "[$(date)] A4: 8-GPU BF16 bs=8"
$RUN torchrun --nproc_per_node=8 /mapqr/scripts/benchmark_synthetic.py --batch-size 8 --amp 2>&1 | tee $RESULTS/A4_8gpu_bf16_bs8.txt

# ==========================================
# Phase 2: Full Dataset 8-GPU (3 epochs)
# ==========================================
echo ""
echo "=== Phase 2: Full Dataset 8-GPU Training ==="

run_train() {
  local NAME=$1
  local CONFIG=$2
  echo "[$(date)] $NAME"
  $RUN bash -c "cd /mapqr && torchrun --nproc_per_node=8 --master_port=29500 \
    tools/train.py $CONFIG --launcher pytorch \
    --work-dir /mapqr/work_dirs/$NAME --deterministic \
    --cfg-options total_epochs=3 runner.max_epochs=3" 2>&1 | tee $RESULTS/${NAME}.txt
}

# B1: Default (bs=4, 4 workers)
run_train "B1_default_fp32" "projects/configs/mapqr/mapqr_nusc_r50_24ep.py --cfg-options evaluation.interval=999"

# B2: Optimized FP32 (bs=8, 24 workers, pin_memory, persistent_workers, prefetch)
run_train "B2_optimized_fp32" "projects/configs/mapqr/mapqr_nusc_r50_24ep_optimized.py"

# B3: Optimized BF16
run_train "B3_optimized_bf16" "projects/configs/mapqr/mapqr_nusc_r50_24ep_bf16_optimized.py"

echo ""
echo "============================================"
echo "[$(date)] ALL BENCHMARKS COMPLETE"
echo "============================================"
ls -la $RESULTS/
