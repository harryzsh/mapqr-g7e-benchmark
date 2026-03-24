# Lessons Learned

## 1. NCCL version matters for Blackwell multi-GPU
NCCL ≤2.26 hangs on Blackwell GPUs. Need NCCL 2.28+. The fix: mount host's newer NCCL via `LD_PRELOAD`.

## 2. Don't use `torchrun` spawn with mmcv 1.x on Python 3.10+
`dict_keys` objects aren't picklable in Python 3.10. Use `multiprocessing.set_start_method("fork")`.

## 3. Pin numpy LAST after all pip installs
Every package pulls in numpy 2.x which breaks mmcv/mmdet compiled for numpy 1.x. Always: `pip install "numpy==1.26.4" --force-reinstall --no-deps` as the final step.

## 4. mmcv-full 1.x compiles fine for Blackwell with TORCH_CUDA_ARCH_LIST="12.0"
Despite being old, mmcv-full 1.7.2 compiles from source on CUDA 12.8 with Blackwell arch. Takes ~15 min.

## 5. GKT CUDA op needs 3 API fixes for PyTorch 2.x
- `value.type()` → `value.scalar_type()`
- `AT_ASSERTM` → `TORCH_CHECK`
- `.type().is_cuda()` → `.is_cuda()`

## 6. mmdet3d 3D CUDA ops can be safely skipped for MapQR
MapQR only uses BEV operations. The 3D point cloud ops (ball_query, iou3d, etc.) that fail to compile are never called.

## 7. Deformable attention limits batch size
`batch % im2col_step == 0` constraint in mmcv's deformable attention CUDA kernel. Max practical batch=8/GPU for MapQR.

## 8. MapQR is too small for RTX PRO 6000
ResNet50 + dim=128 only uses 18-33% of 96GB GPU memory and 25-33% of 600W TDP. These GPUs are designed for much larger models.

## 9. Docker image layer export is slow for large images
33GB images take 3-5 min just to export layers. Plan for this.

## 10. g7e.48xlarge capacity is limited
Had to try us-east-1 (no capacity in both AZs) before finding availability in us-west-2c.
