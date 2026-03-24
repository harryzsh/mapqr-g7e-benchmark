#!/bin/bash
# apply_patches.sh - Apply all Blackwell/PyTorch 2.7 compatibility patches to MapQR
# Run from MapQR root directory
set -e

MAPQR_DIR="${1:-.}"
cd "$MAPQR_DIR"

echo "[1/13] Patching mmdet3d version assertions..."
python3 -c "
import re
with open('mmdetection3d/mmdet3d/__init__.py') as f:
    content = f.read()
content = re.sub(r'assert \([\s\S]*?\n\n', '\n', content)
open('mmdetection3d/mmdet3d/__init__.py','w').write(content)
"

echo "[2/13] Patching mmdet3d ops __init__.py (skip uncompilable CUDA ops)..."
cat > mmdetection3d/mmdet3d/ops/__init__.py << 'OPSEOF'
import warnings
_failed = []

def _try_import(name, module_path):
    try:
        import importlib
        mod = importlib.import_module(module_path, package='mmdet3d.ops')
        globals().update({k: getattr(mod, k) for k in dir(mod) if not k.startswith('_')})
    except Exception:
        _failed.append(name)

_try_import('ball_query', '.ball_query')
_try_import('furthest_point_sample', '.furthest_point_sample')
_try_import('gather_points', '.gather_points')
_try_import('group_points', '.group_points')
_try_import('interpolate', '.interpolate')
_try_import('iou3d', '.iou3d')
_try_import('norm', '.norm')
_try_import('paconv', '.paconv')
_try_import('pointnet_modules', '.pointnet_modules')
_try_import('roiaware_pool3d', '.roiaware_pool3d')
_try_import('spconv', '.spconv')
_try_import('voxel', '.voxel')

if _failed:
    warnings.warn(f'mmdet3d ops not loaded (Blackwell compat): {_failed}')
OPSEOF

echo "[3/13] Patching mmdet3d.ops imports in all files (try/except fallbacks)..."
python3 << 'PYEOF'
import os, re
for root_dir in ['mmdetection3d/mmdet3d', 'projects']:
    for dirpath, dirs, files in os.walk(root_dir):
        if '__pycache__' in dirpath: continue
        for fn in files:
            if not fn.endswith('.py'): continue
            fp = os.path.join(dirpath, fn)
            with open(fp) as f: content = f.read()
            if 'from mmdet3d.ops' not in content: continue
            new = re.sub(
                r'^(from mmdet3d\.ops(?:\.\w+)* import (.+))$',
                lambda m: 'try:\n    ' + m.group(1) + '\nexcept Exception:\n    ' +
                    '; '.join(n.strip() + ' = None' for n in m.group(2).split(',')),
                content, flags=re.MULTILINE)
            if new != content:
                open(fp, 'w').write(new)
PYEOF

echo "[4/13] Patching GKT CUDA source for PyTorch 2.x API..."
cd projects/mmdet3d_plugin/maptr/modules/ops/geometric_kernel_attn/src
sed -i 's/AT_DISPATCH_FLOATING_TYPES(value\.type(), /AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), /g' geometric_kernel_attn_cuda.cu
sed -i 's/\.type()\.is_cuda()/.is_cuda()/g' geometric_kernel_attn_cuda.cu
sed -i 's/AT_ASSERTM/TORCH_CHECK/g' geometric_kernel_attn_cuda.cu
cd "$MAPQR_DIR"

echo "[5/13] Patching numba import path..."
sed -i 's/from numba.errors import/from numba.core.errors import/' \
  mmdetection3d/mmdet3d/datasets/pipelines/data_augment_utils.py

echo "[6/13] Patching tkinter imports..."
sed -i 's|from tkinter.messagebox import NO|NO = "no"|' \
  projects/mmdet3d_plugin/bevformer/detectors/bevformer.py
sed -i 's|from tkinter.messagebox import NO|NO = "no"|' \
  projects/mmdet3d_plugin/bevformer/detectors/bevformer_fp16.py 2>/dev/null || true

echo "[7/13] Adding multiprocessing fork for DDP pickle compat..."
sed -i '/^from __future__/a import multiprocessing; multiprocessing.set_start_method("fork", force=True)' tools/train.py

echo "[8-13] Remaining patches applied inside Docker (see Dockerfile and run_benchmark.sh):"
echo "  [8] mmcv registry force=True (in Docker)"
echo "  [9] mmcv checkpoint weights_only=False (in Docker)"
echo "  [10] seaborn style (matplotlib 3.5.3 keeps original name)"
echo "  [11] NCCL 2.28.9 via LD_PRELOAD (at runtime)"
echo "  [12] timm==0.6.13 (in Dockerfile)"
echo "  [13] universal_pathlib for upath (in Dockerfile)"

echo ""
echo "=== All patches applied ==="
