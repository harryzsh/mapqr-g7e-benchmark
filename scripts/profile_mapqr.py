from __future__ import division
import torch
import torch.profiler
import sys, importlib

sys.path.insert(0, '/mapqr')
sys.path.insert(0, '/mapqr/mmdetection3d')
importlib.import_module('projects.mmdet3d_plugin')

from mmcv import Config
from mmdet3d.models import build_model

cfg = Config.fromfile('/mapqr/projects/configs/mapqr/mapqr_nusc_r50_24ep.py')
cfg.model.pretrained = dict(img='/mapqr/ckpts/resnet50-19c8e357.pth')
model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
model.cuda().train()

bs = 4
dummy_img = torch.randn(bs * 6, 3, 480, 800).cuda()

print("Warming up backbone...")
for _ in range(5):
    with torch.no_grad():
        feats = model.img_backbone(dummy_img)
        feats = model.img_neck(feats)
torch.cuda.synchronize()

print("Profiling backbone+neck (20 iters)...")
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
) as prof:
    for _ in range(20):
        with torch.no_grad():
            feats = model.img_backbone(dummy_img)
            feats = model.img_neck(feats)
        torch.cuda.synchronize()

print("\n=== Top 20 CUDA Kernels by Self GPU Time ===")
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))

import time
torch.cuda.synchronize()
start = time.time()
for _ in range(50):
    with torch.no_grad():
        feats = model.img_backbone(dummy_img)
        feats = model.img_neck(feats)
    torch.cuda.synchronize()
elapsed = time.time() - start
print(f"\nBackbone+Neck: {elapsed/50*1000:.1f} ms/iter, {bs*6*50/elapsed:.0f} images/sec")

prof.export_chrome_trace("/mapqr/work_dirs/profile_trace.json")
print("Trace saved to /mapqr/work_dirs/profile_trace.json")
