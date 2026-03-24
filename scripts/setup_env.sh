#!/bin/bash
# setup_env.sh - Full environment setup for MapQR benchmark on g7e.48xlarge
# Run on a fresh g7e instance with DLAMI
set -e

echo "============================================"
echo "MapQR Benchmark Setup for g7e.48xlarge"
echo "============================================"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCHMARK_DIR="$(dirname "$SCRIPT_DIR")"
NVME=/opt/dlami/nvme

# Step 1: Verify GPUs
echo ""
echo "[Step 1] Verifying GPUs..."
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Found $GPU_COUNT GPUs"

# Step 2: Clone repos
echo ""
echo "[Step 2] Cloning repos..."
cd $NVME
[ -d MapQR ] || git clone https://github.com/HXMap/MapQR.git
mkdir -p MapQR/ckpts
[ -f MapQR/ckpts/resnet50-19c8e357.pth ] || \
  wget -q https://download.pytorch.org/models/resnet50-19c8e357.pth -O MapQR/ckpts/resnet50-19c8e357.pth

# Step 3: Build Docker image
echo ""
echo "[Step 3] Building Docker image (~20 min for mmcv compilation)..."
docker build -f $BENCHMARK_DIR/Dockerfile -t mapqr-g7e:latest $NVME/

# Step 4: Apply Docker-internal patches
echo ""
echo "[Step 4] Applying Docker patches..."
docker run --gpus all --name mapqr-patch mapqr-g7e:latest bash -c "$(cat $SCRIPT_DIR/apply_docker_patches.sh)"
docker commit mapqr-patch mapqr-g7e:latest
docker rm mapqr-patch

# Step 5: Apply source code patches
echo ""
echo "[Step 5] Applying Blackwell compatibility patches..."
cd $NVME/MapQR
bash $SCRIPT_DIR/apply_patches.sh .

# Step 6: Build GKT CUDA op
echo ""
echo "[Step 6] Building GKT CUDA op..."
docker run --gpus all --name gkt-build \
  -v $NVME/MapQR:/mapqr \
  -e TORCH_CUDA_ARCH_LIST="12.0" -e FORCE_CUDA=1 \
  -e LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib \
  mapqr-g7e:latest bash -c "
cd /mapqr/projects/mmdet3d_plugin/maptr/modules/ops/geometric_kernel_attn
python setup.py build install
python -c 'import GeometricKernelAttention; print(\"GKT: OK\")'
"
docker commit gkt-build mapqr-g7e:latest
docker rm gkt-build

# Step 7: Download nuScenes
echo ""
echo "[Step 7] Downloading nuScenes dataset (~300GB, ~15 min)..."
bash $SCRIPT_DIR/download_nuscenes.sh

# Step 8: Generate PKLs
echo ""
echo "[Step 8] Generating data PKLs..."
bash $SCRIPT_DIR/prepare_data.sh

echo ""
echo "============================================"
echo "Setup complete! Run benchmark with:"
echo "  bash $SCRIPT_DIR/run_benchmark.sh"
echo "============================================"
