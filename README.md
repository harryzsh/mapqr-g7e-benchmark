# MapQR Benchmark on AWS g7e.48xlarge (NVIDIA RTX PRO 6000 Blackwell)

## Overview

GPU benchmark of [MapQR](https://github.com/HXMap/MapQR) (vectorized HD map construction for autonomous driving) on AWS g7e.48xlarge with 8x NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs.

## Hardware

| Item | Details |
|------|---------|
| Instance | AWS g7e.48xlarge |
| Region | us-west-2c |
| GPU | 8x NVIDIA RTX PRO 6000 Blackwell Server Edition |
| GPU Memory | 96 GB GDDR7 per GPU (768 GB total) |
| GPU Memory Bandwidth | 1,597 GB/s per GPU |
| CUDA Cores | 24,064 per GPU |
| Tensor Cores | 752 5th-gen per GPU |
| TDP | 600W per GPU |
| vCPUs | 192 |
| System Memory | 2,048 GiB |
| Local NVMe SSD | 15.2 TB (4x 3.8 TB) |
| Network | 1,600 Gbps |
| Driver | 580.126.16 |
| CUDA | 13.0 (host) / 12.8 (Docker) |
| On-Demand Price | $33.14/hr |

## Software Stack

| Component | Version |
|-----------|---------|
| PyTorch | 2.7.0+cu128 (Docker) |
| NCCL | 2.28.9 (host, mounted via LD_PRELOAD) |
| mmcv-full | 1.7.2 (compiled from source for Blackwell arch 12.0) |
| mmdet | 2.28.2 |
| mmdet3d | 0.17.2 |
| Python | 3.10 |
| OS | Ubuntu 22.04 (Docker) / Ubuntu 24.04 (host) |

## Dataset

Full nuScenes v1.0-trainval dataset:
- ~28,000 training scenes, 6 camera views each
- 696 GB on NVMe SSD
- 4 cities: Boston, Pittsburgh, Las Vegas, Singapore

## Benchmark Results

### 8-GPU Training (Baseline: batch=4/GPU)

| Metric | Value |
|--------|-------|
| Total batch size | 32 (4 × 8 GPUs) |
| Epochs | 24 |
| Total training time | ~8 min |
| Time per epoch | ~20 sec |
| GPU memory | 17.5 GB / 96 GB (18%) |
| GPU utilization | 90-98% |
| Temperature | 35-40°C |
| Power draw | ~200W / 600W (33%) |

### 8-GPU Training (Optimized: batch=8/GPU)

| Metric | Value |
|--------|-------|
| Total batch size | 64 (8 × 8 GPUs) |
| Epochs | 24 |
| Total training time | ~9 min |
| Time per epoch | ~23 sec |
| GPU memory | 31.7 GB / 96 GB (33%) |
| GPU utilization | 97-99% |
| Temperature | 35-39°C |
| Power draw | 150-185W / 600W (25-31%) |

### Single GPU Training (batch=4)

| Epoch | Loss | Time/iter |
|-------|------|-----------|
| 1 | 392.10 | 1.135s |
| 2 | 209.45 | 1.089s |
| 3 | 181.43 | 1.106s |
| 4 | 158.73 | 1.103s |

### Key Findings

1. **GPU underutilization**: MapQR (ResNet50, dim=128) is too small for RTX PRO 6000's 96GB. Only 18-33% memory used.
2. **Power efficiency**: GPUs running at 25-33% TDP. The model doesn't saturate Tensor Cores.
3. **DDP efficiency**: 8-GPU gives ~4.5x speedup over 1-GPU (vs theoretical 8x). Similar to H200 benchmark findings.
4. **Batch scaling limited**: Deformable attention CUDA op constrains batch sizes (`batch % im2col_step == 0`). Max practical batch=8/GPU.
5. **Blackwell compatibility**: Required 13 patches to port MapQR from CUDA 11.x/PyTorch 1.9 to CUDA 12.8/PyTorch 2.7.

## Setup Guide

### Step 1: Launch EC2 Instance

```bash
# g7e.48xlarge in us-west-2 (or any region with availability)
# AMI: Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.10 (Ubuntu 24.04)
# Root EBS: 200GB gp3
# NVMe: auto-mounted at /opt/dlami/nvme (14TB)
```

### Step 2: Clone Repos

```bash
cd /opt/dlami/nvme
git clone https://github.com/HXMap/MapQR.git
git clone https://github.com/harryzsh/mapqr-g7e-benchmark.git
wget -q https://download.pytorch.org/models/resnet50-19c8e357.pth -O MapQR/ckpts/resnet50-19c8e357.pth
```

### Step 3: Build Docker Image

```bash
cd /opt/dlami/nvme
docker build -f mapqr-g7e-benchmark/Dockerfile -t mapqr-g7e:latest .
```

This builds mmcv-full from source with `TORCH_CUDA_ARCH_LIST="12.0"` for Blackwell. Takes ~20 min.

### Step 4: Apply Blackwell Compatibility Patches

```bash
cd /opt/dlami/nvme/MapQR
bash /opt/dlami/nvme/mapqr-g7e-benchmark/scripts/apply_patches.sh
```

See [PATCHES.md](docs/PATCHES.md) for details on all 13 patches.

### Step 5: Build GKT CUDA Op

Must be done inside Docker with GPU access:

```bash
docker run --gpus all --name gkt-build \
  -v /opt/dlami/nvme/MapQR:/mapqr \
  -e TORCH_CUDA_ARCH_LIST="12.0" -e FORCE_CUDA=1 \
  -e LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib \
  mapqr-g7e:latest bash -c "
cd /mapqr/projects/mmdet3d_plugin/maptr/modules/ops/geometric_kernel_attn
python setup.py build install
"
docker commit gkt-build mapqr-g7e:latest
docker rm gkt-build
```

### Step 6: Download nuScenes Dataset

```bash
bash /opt/dlami/nvme/mapqr-g7e-benchmark/scripts/download_nuscenes.sh
```

Downloads ~300GB from `s3://motional-nuscenes/public/v1.0/` (no auth needed). Takes ~15 min on g7e.

### Step 7: Generate Data PKLs

```bash
bash /opt/dlami/nvme/mapqr-g7e-benchmark/scripts/prepare_data.sh
```

### Step 8: Run Benchmark

```bash
bash /opt/dlami/nvme/mapqr-g7e-benchmark/scripts/run_benchmark.sh
```

## Blackwell Compatibility Issues & Fixes

Porting MapQR from CUDA 11.x / PyTorch 1.9 to Blackwell (CUDA 12.8+ / PyTorch 2.7) required 13 patches:

| # | Issue | Fix |
|---|-------|-----|
| 1 | mmdet3d CUDA ops won't compile for arch 12.0 | try/except fallbacks (MapQR doesn't use 3D ops) |
| 2 | GKT `AT_DISPATCH_FLOATING_TYPES` uses deprecated `value.type()` | Changed to `value.scalar_type()` |
| 3 | GKT `AT_ASSERTM` deprecated | Changed to `TORCH_CHECK` |
| 4 | GKT `.type().is_cuda()` deprecated | Changed to `.is_cuda()` |
| 5 | `numba.errors` moved | Changed to `numba.core.errors` |
| 6 | matplotlib `seaborn-whitegrid` renamed | Pin `matplotlib==3.5.3` (keeps old name) |
| 7 | `upath` package changed | Install `universal_pathlib` |
| 8 | timm model registry conflicts | Pin `timm==0.6.13` |
| 9 | mmcv registry duplicate registration error | Set `force=True` default in `_register_module` |
| 10 | `from tkinter.messagebox import NO` | Replace with `NO = "no"` |
| 11 | `torch.load` defaults to `weights_only=True` in PyTorch 2.6+ | Patch mmcv checkpoint.py to pass `weights_only=False` |
| 12 | NCCL 2.26 hangs on Blackwell multi-GPU (known bug #1637) | Mount host NCCL 2.28.9 via `LD_PRELOAD` |
| 13 | `dict_keys` not picklable in Python 3.10 (DDP spawn) | Force `multiprocessing.set_start_method("fork")` |

## TODOs

- [ ] Fix validation (`numpy.int64` has no attribute `intersects` — shapely/numpy compat)
- [ ] BF16 mixed precision benchmark
- [ ] PyTorch Profiler kernel-level analysis
- [ ] Cross-GPU comparison report (g7e vs g5 A10G vs p5en H200)

## References

- [MapQR](https://github.com/HXMap/MapQR) — ECCV 2024
- [NCCL Blackwell P2P fix](https://github.com/NVIDIA/nccl/issues/1637)
- [NVIDIA Blackwell PyTorch Migration Guide](https://forums.developer.nvidia.com/t/software-migration-guide-for-nvidia-blackwell-rtx-gpus/321330)
- [H200 Benchmark Report](https://github.com/yunfeilu92/h200-benchmark)
