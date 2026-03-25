# MapQR Benchmark on AWS g7e.48xlarge (NVIDIA RTX PRO 6000 Blackwell)

## Overview

Comprehensive GPU benchmark of [MapQR](https://github.com/HXMap/MapQR) (vectorized HD map construction for autonomous driving) on AWS g7e.48xlarge with 8x NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs. Includes full 24-epoch training, optimization sweep, and profiling analysis.

## Hardware

| Item | Details |
|------|---------|
| Instance | AWS g7e.48xlarge |
| Region | us-west-2c |
| GPU | 8x NVIDIA RTX PRO 6000 Blackwell Server Edition (sm_120) |
| GPU Memory | 96 GB GDDR7 per GPU (768 GB total) |
| GPU Memory Bandwidth | 1,597 GB/s per GPU |
| CUDA Cores | 24,064 per GPU |
| Tensor Cores | 752 5th-gen per GPU |
| TDP | 250W per GPU |
| vCPUs | 192 |
| System Memory | 768 GiB |
| Local NVMe SSD | 14 TB |
| Driver | 580.126.16 |
| CUDA | 13.0 (host) / 12.8 (Docker) |
| On-Demand Price | $33.14/hr |

## Software Stack

| Component | Version |
|-----------|---------|
| PyTorch | 2.7.0+cu128 |
| NCCL | 2.28.9 (host, mounted via LD_PRELOAD) |
| mmcv-full | 1.7.2 (compiled from source, arch 12.0) |
| mmdet | 2.28.2 |
| mmdet3d | 0.17.2 |
| Python | 3.10 (Docker) |
| OS | Ubuntu 22.04 (Docker) / Ubuntu 24.04 (host) |

## Dataset

Full nuScenes v1.0-trainval:
- 28,130 train keyframe samples (from 700 driving scenes)
- 6,019 val keyframe samples
- 6 camera views per sample (168,780 train images)
- 696 GB on NVMe SSD

## Full 24-Epoch Training Results (Best Config)

**Config:** FP32, batch=4/GPU (32 global), 24 workers, persistent_workers, prefetch=4, pin_memory

| Metric | Value |
|--------|-------|
| Total batch size | 32 (4 x 8 GPUs) |
| Iterations per epoch | 880 |
| Time per iteration | 1.20s |
| Time per epoch | 18.7 min (avg) |
| Total training time (24 epochs) | 7.5 hours |
| Throughput | **26.7 samples/sec (160 images/sec)** |
| Per-GPU throughput | 3.3 samples/sec (20 images/sec) |
| GPU memory | 17.5 GB / 96 GB (18%) |
| GPU utilization | 30-80% (avg ~60%) |
| Temperature | 34-42 C |
| Power draw | ~200W / 250W TDP |
| Final loss (epoch 24) | 13.86 |

### Loss Convergence

| Epoch | Loss | Checkpoint Time |
|-------|------|-----------------|
| 1 | ~97 | 16:24 |
| 2 | ~74 | 16:42 |
| 3 | ~65 | 17:00 |
| 4 | ~53 | 17:19 |
| 8 | ~33 | 20:00 |
| 12 | — | 22:46 |
| 16 | 19.2 | 00:00 |
| 20 | 15.2 | 01:16 |
| 24 | **13.86** | 02:32 |

### Epoch Timing (Epochs 9-24, single continuous run)

| Epoch | Duration |
|-------|----------|
| 10 | 19:00 |
| 11 | 18:24 |
| 12 | 19:06 |
| 13 | 17:48 |
| 14 | 18:16 |
| 15 | 19:37 |
| 16 | 18:34 |
| 17 | 18:38 |
| 18 | 19:14 |
| 19 | 19:04 |
| 20 | 18:27 |
| 21 | 19:19 |
| 22 | 19:20 |
| 23 | 19:01 |
| 24 | 18:40 |
| **Average** | **18 min 42 sec** |

## Optimization Sweep

Tested 8 configurations. All use 8 GPUs, full 28K dataset.

| # | Config | Time/iter | Samples/sec | vs Baseline | Verdict |
|---|--------|-----------|-------------|-------------|---------|
| 1 | **FP32 bs=4 24w optimized** | **1.20s** | **26.7** | **baseline** | **Best** |
| 2 | FP16 mixed (deform forced FP32) | 1.28s | 25.0 | -6% | Slower |
| 3 | FP16 mixed (deform FP16 enabled) | 1.32s | 24.2 | -9% | Slower |
| 4 | FP32 bs=4 workers=16 prefetch=8 | 1.20s | 26.7 | 0% | No change |
| 5 | FP32 bs=4 cudnn.benchmark=True | 2.16s* | 14.8* | -44% | Much slower |
| 6 | FP32 bs=4 TF32 Tensor Cores | 1.22s | 26.2 | -2% | No change |
| 7 | FP32 bs=8 optimized | 2.58s | 24.8 | -7% | Slower |
| 8 | FP32 bs=8 TF32 | 2.55s | 25.1 | -6% | Slower |
| 9 | FP16 bs=8 LR=1.2e-3 (H200 config) | 2.61s | 24.5 | -8% | Slower |
| 10 | NCCL tuned (Ring/LL128/NVL) | 1.29s | 24.8 | -7% | Slower |

*cudnn.benchmark tested on single GPU

### Why Optimizations Failed

- **FP16/BF16**: Deformable attention CUDA op is hardcoded to FP32. FP16 casting overhead negates any compute savings.
- **Batch=8**: Deformable attention doesn't scale linearly with batch size. 2x batch = 2.15x compute time.
- **cudnn.benchmark**: MapQR has variable-size inputs (image padding), causing constant re-benchmarking.
- **TF32**: mmcv's pre-compiled CUDA ops bypass PyTorch's matmul precision settings.
- **NCCL tuning**: Default auto-selection was already optimal for this GPU topology.

## Profiling Analysis

PyTorch Profiler, single GPU, 10 iterations, batch=4.

### Top CUDA Kernels by Time (1.908s total)

| Kernel | CUDA Time | % | Category |
|--------|-----------|---|----------|
| aten::addmm (linear layers) | 537ms | 28.2% | FP32 GEMM, no Tensor Cores |
| sgemm_largek_lds64 | 460ms | 24.1% | Plain FP32 GEMM |
| aten::copy_ (memory copies) | 247ms | 13.0% | Memory |
| cudnn_convolution | 228ms | 12.0% | Backbone conv2d |
| Memcpy HtoD (CPU to GPU) | 213ms | 11.2% | Data transfer |
| cdist_forward (loss) | 178ms | 9.3% | Distance computation |
| cudnn_batch_norm | 104ms | 5.5% | BatchNorm |
| bmm (attention) | 98ms | 5.2% | Matrix multiply |
| clamp_min_ (ReLU) | 87ms | 4.5% | Activation |
| add_ (residual) | 86ms | 4.5% | Skip connections |
| cutlass_80_tensorop | 44ms | 2.3% | Only Tensor Core usage |
| MultiScaleDeformableAttn | 44ms | 2.3% | Custom CUDA op |

### Key Profiling Findings

1. **24% of CUDA time is plain FP32 SGEMM** — not using Tensor Cores at all
2. **Only 2.3% uses Tensor Cores** (Ampere sm_80 via JIT, not native Blackwell sm_120)
3. **11% is CPU-to-GPU data transfer** (Memcpy HtoD) — data loading bottleneck
4. **Deformable attention is only 2.3%** — not the compute bottleneck, but blocks FP16

### JPEG Decode Benchmark

| Method | Per image | Per sample (6 cams) | Speedup |
|--------|-----------|---------------------|---------|
| mmcv.imread (CPU) | 6.6ms | 39ms | baseline |
| cv2.imread (CPU) | 6.5ms | 39ms | 1x |
| torchvision GPU decode (nvJPEG) | 0.7ms | 4ms | **10x** |

GPU decode can't be used in forked dataloader workers (CUDA fork limitation).

## Bottleneck Analysis

```
Iteration breakdown (1.20s total):
  Compute:     ~1.00s (83%) — FP32 CUDA ops compiled for sm_80
  Data load:   ~0.09s  (7%) — CPU JPEG decode, well parallelized
  Data transfer: ~0.11s (9%) — CPU to GPU memcpy (pageable)
  NCCL sync:   ~0.01s  (1%) — gradient allreduce
```

### Why Blackwell GPUs Are Underutilized

1. **mmcv CUDA ops compiled for sm_80 (Ampere)** — JIT to Blackwell sm_120, missing native kernel optimizations
2. **FP32-locked deformable attention** — blocks mixed precision for the entire model
3. **Legacy data pipeline** — CPU JPEG decode + pageable memory copies
4. **Only 18% GPU memory used** — model too small for 96GB GPUs
5. **GPU utilization 30-80%** — frequent idle periods between iterations (data starvation + NCCL sync)

## Recommended Improvements

### Achievable with code changes

| Improvement | Expected Gain | Effort |
|-------------|--------------|--------|
| Pure PyTorch rewrite (remove mmcv) | +100-130% throughput | High |
| CUDA deformable attn backward fix | +29% + 35% less memory | High |
| Rebatch vectorization (HeightKernelAttn) | +74% backward time | Medium |
| FFCV/DALI data pipeline | Eliminate 11% data transfer | Medium |
| GPU JPEG decode (nvJPEG) | 10x decode speedup | Medium |
| Recompile CUDA ops for sm_120 | 10-20% compute | Medium |

### Reference: H200 Pure PyTorch Results (colleague's Phase 2)

| Config | Throughput | vs mmcv baseline |
|--------|-----------|-----------------|
| mmcv bs4 FP16 (H200) | 22.9 s/s | baseline |
| mmcv bs8 FP16 optimized | 28.7 s/s | +25% |
| Pure PyTorch bs32 | 52.5 s/s | **+129%** |

The pure PyTorch rewrite eliminates mmcv decorator overhead, fixes CUDA backward, and vectorizes rebatch loops.

## Blackwell Compatibility Patches (15 total)

| # | Issue | Fix |
|---|-------|-----|
| 1 | mmdet3d version assertion | Removed |
| 2 | mmdet3d CUDA ops won't compile for arch 12.0 | try/except fallbacks |
| 3 | GKT `value.type()` deprecated | `value.scalar_type()` |
| 4 | GKT `AT_ASSERTM` deprecated | `TORCH_CHECK` |
| 5 | GKT `.type().is_cuda()` deprecated | `.is_cuda()` |
| 6 | `numba.errors` moved | `numba.core.errors` |
| 7 | matplotlib style renamed | Pin `matplotlib==3.5.3` |
| 8 | `upath` package changed | `universal_pathlib` |
| 9 | timm registry conflicts | Pin `timm==0.6.13` |
| 10 | mmcv duplicate registration | `force=True` default |
| 11 | tkinter.messagebox.NO removed | `NO = "no"` |
| 12 | torch.load weights_only default | `weights_only=False` |
| 13 | DDP pickle error (fork vs spawn) | `set_start_method("fork")` |
| 14 | NCCL 2.26 hangs on Blackwell | Mount host NCCL 2.28.9 via LD_PRELOAD |
| 15 | mmcv DDP `_use_replicated_tensor_module` | Use `self.module` directly |

## Cross-GPU Comparison

| GPU | Instance | Throughput (mmcv) | Config |
|-----|----------|-------------------|--------|
| 8x A100 40GB | p4d.24xlarge | 14.5 s/s | FP16 bs=4 |
| 8x H200 141GB | p5en.48xlarge | 28.7 s/s | FP16 bs=8 optimized |
| **8x RTX PRO 6000 96GB** | **g7e.48xlarge** | **26.7 s/s** | **FP32 bs=4 optimized** |

Note: Blackwell runs FP32 (not FP16) because mmcv CUDA ops are compiled for sm_80 and FP16 adds casting overhead. With native sm_120 compilation and pure PyTorch rewrite, Blackwell throughput is estimated at 60-70 s/s.

## Conclusion

1. **Best config for MapQR on g7e.48xlarge**: FP32, batch=4/GPU, 24 workers, persistent_workers, prefetch=4, pin_memory — **26.7 samples/sec, 160 images/sec**
2. **Full 24-epoch training**: 7.5 hours, loss 97 to 13.86, ~18.7 min/epoch
3. **Parameter tuning ceiling reached**: 10 optimization attempts, none faster than baseline
4. **Root cause**: mmcv 1.x legacy CUDA ops compiled for Ampere sm_80, FP32-locked deformable attention, CPU-bound data pipeline
5. **Path to 2x+ throughput**: Pure PyTorch rewrite (proven on H200 at +129%)

## References

- [MapQR](https://github.com/HXMap/MapQR) — ECCV 2024
- [NCCL Blackwell bug](https://github.com/NVIDIA/nccl/issues/1637)
- [NVIDIA Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/)
- [FFCV](https://ffcv.io/) — Fast data loading
