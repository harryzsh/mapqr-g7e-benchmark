#!/bin/bash
# apply_docker_patches.sh - Patches applied inside Docker container after build
# Run inside Docker: docker run --gpus all -v /path/to/MapQR:/mapqr mapqr-g7e:latest bash apply_docker_patches.sh
set -e

echo "[1] Patching mmcv registry (allow duplicate model registration)..."
python3 -c "
f='/usr/local/lib/python3.10/dist-packages/mmcv/utils/registry.py'
c=open(f).read()
c=c.replace('def _register_module(self, module, module_name=None, force=False):',
            'def _register_module(self, module, module_name=None, force=True):')
c=c.replace('def register_module(self, name=None, force=False, module=None):',
            'def register_module(self, name=None, force=True, module=None):')
open(f,'w').write(c)
print('  registry patched')
"

echo "[2] Patching mmcv checkpoint (weights_only=False for legacy .pth files)..."
python3 -c "
f='/usr/local/lib/python3.10/dist-packages/mmcv/runner/checkpoint.py'
c=open(f).read()
c=c.replace('checkpoint = torch.load(filename, map_location=map_location)',
            'checkpoint = torch.load(filename, map_location=map_location, weights_only=False)')
open(f,'w').write(c)
print('  checkpoint patched')
"

echo "=== Docker patches applied ==="
