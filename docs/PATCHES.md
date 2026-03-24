# Blackwell Compatibility Patches — Detailed Documentation

## Why Patches Are Needed

MapQR was built for Python 3.8, PyTorch 1.9, CUDA 11.1, mmcv-full 1.4.
Blackwell GPUs (compute capability 12.0) require CUDA 12.8+, PyTorch 2.7+, NCCL 2.28+.

This ~4 year version gap causes 13 breaking changes.

## Patch Details

### Patch 1: mmdet3d Version Assertions

**File**: `mmdetection3d/mmdet3d/__init__.py`
**Problem**: Hard-coded version checks reject newer mmcv/mmdet/mmseg versions.
**Fix**: Remove all `assert` blocks that check version compatibility.

### Patch 2: mmdet3d CUDA Ops Fallbacks

**File**: `mmdetection3d/mmdet3d/ops/__init__.py` + 30 other files
**Problem**: mmdet3d's 3D CUDA ops (ball_query, iou3d, roiaware_pool3d, etc.) won't compile for Blackwell arch 12.0.
**Fix**: Replace `ops/__init__.py` with try/except wrapper. MapQR only uses BEV operations, not 3D point cloud ops, so these are safely skipped.

### Patch 3: GKT AT_DISPATCH_FLOATING_TYPES

**File**: `projects/.../geometric_kernel_attn_cuda.cu`
**Problem**: `AT_DISPATCH_FLOATING_TYPES(value.type(), ...)` — `value.type()` returns `DeprecatedTypeProperties` in PyTorch 2.x.
**Fix**: Change to `value.scalar_type()`.

### Patch 4: GKT AT_ASSERTM

**File**: Same CUDA file
**Problem**: `AT_ASSERTM` macro removed in PyTorch 2.x.
**Fix**: Replace with `TORCH_CHECK`.

### Patch 5: GKT .type().is_cuda()

**File**: Same CUDA file
**Problem**: `.type().is_cuda()` deprecated.
**Fix**: Change to `.is_cuda()`.

### Patch 6: numba Import Path

**File**: `mmdetection3d/mmdet3d/datasets/pipelines/data_augment_utils.py`
**Problem**: `from numba.errors import` — module moved in numba 0.56+.
**Fix**: Change to `from numba.core.errors import`.

### Patch 7: Matplotlib Seaborn Style

**Problem**: `seaborn-whitegrid` renamed to `seaborn-v0_8-whitegrid` in matplotlib 3.6+.
**Fix**: Pin `matplotlib==3.5.3` which keeps the original name. No code change needed.

### Patch 8: upath Package

**Problem**: `from upath import UPath` — the `upath` pip package changed, now need `universal_pathlib`.
**Fix**: Install `universal_pathlib` in Dockerfile.

### Patch 9: timm Registry Conflict

**Problem**: MapQR registers `EfficientNet` in mmcv's model registry, but newer timm also registers it.
**Fix**: Pin `timm==0.6.13` AND set `force=True` default in mmcv's `_register_module()` and `register_module()`.

### Patch 10: tkinter Import

**File**: `projects/.../bevformer/detectors/bevformer.py` (and `bevformer_fp16.py`)
**Problem**: `from tkinter.messagebox import NO` — tkinter not installed in Docker.
**Fix**: Replace with `NO = "no"` (it's just a string constant).

### Patch 11: torch.load weights_only

**Problem**: PyTorch 2.6+ defaults `weights_only=True` in `torch.load()`, which rejects legacy .pth files.
**Fix**: Patch mmcv's `checkpoint.py` to pass `weights_only=False`.

### Patch 12: NCCL Blackwell Multi-GPU Hang

**Problem**: NCCL ≤2.26 has a known bug ([#1637](https://github.com/NVIDIA/nccl/issues/1637)) where multi-GPU communication hangs on Blackwell GPUs. P2P is disabled for RTX-class GPUs, and the fallback path was broken.
**Fix**: Mount host's NCCL 2.28.9 library via `LD_PRELOAD` at container runtime. The host DLAMI (PyTorch 2.10+cu130) ships with the fixed version.

### Patch 13: dict_keys Pickle Error (DDP)

**File**: `tools/train.py`
**Problem**: Python 3.10 can't pickle `dict_keys` objects. PyTorch DDP with `spawn` start method tries to pickle the config, which contains `dict_keys`.
**Fix**: Add `multiprocessing.set_start_method("fork", force=True)` at the top of `train.py`. Fork doesn't need to pickle.
