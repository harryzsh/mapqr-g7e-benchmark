"""
MapQR Synthetic Data Benchmark
Measures pure GPU throughput without data loading overhead.
Inspired by github.com/qingzwang/pluto_efficiency

Usage:
  # Single GPU
  python benchmark_synthetic.py --batch-size 4

  # 8-GPU DDP
  torchrun --nproc_per_node=8 benchmark_synthetic.py --batch-size 4

  # BF16
  torchrun --nproc_per_node=8 benchmark_synthetic.py --batch-size 8 --amp
"""
from __future__ import division
import multiprocessing; multiprocessing.set_start_method("fork", force=True)

import argparse
import time
import torch
import torch.distributed as dist
import sys
import os
import importlib

sys.path.insert(0, '/mapqr')
sys.path.insert(0, '/mapqr/mmdetection3d')
importlib.import_module('projects.mmdet3d_plugin')

from mmcv import Config
from mmdet3d.models import build_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=4, help='Per-GPU batch size')
    parser.add_argument('--num-steps', type=int, default=50, help='Benchmark steps')
    parser.add_argument('--warmup-steps', type=int, default=5, help='Warmup steps')
    parser.add_argument('--amp', action='store_true', help='Enable BF16 mixed precision')
    parser.add_argument('--config', default='projects/configs/mapqr/mapqr_nusc_r50_24ep.py')
    # DDP args (set by torchrun)
    parser.add_argument('--local_rank', type=int, default=0)
    return parser.parse_args()


def create_dummy_data(batch_size, device):
    """Create dummy input matching MapQR's expected format."""
    # MapQR processes 6 camera views per sample
    # ResNet50 backbone expects (B*6, 3, H, W)
    img = torch.randn(batch_size, 6, 3, 480, 800, device=device)
    
    # Minimal img_metas
    img_metas = [dict(
        can_bus=torch.zeros(18).numpy(),
        lidar2img=[torch.eye(4).numpy() for _ in range(6)],
        img_shape=[(480, 800, 3)] * 6,
        scene_token=f'dummy_{i}',
        sample_idx=i,
    ) for i in range(batch_size)]
    
    return img, img_metas


def benchmark(args):
    rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    is_ddp = world_size > 1
    
    if is_ddp:
        dist.init_process_group('nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
    
    device = torch.device(f'cuda:{rank}')
    
    # Build model
    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = dict(img='/mapqr/ckpts/resnet50-19c8e357.pth')
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.to(device).train()
    
    if is_ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda') if args.amp else None
    
    # Get the raw model for backbone access
    raw_model = model.module if is_ddp else model
    
    # Timing storage
    forward_times = []
    backward_times = []
    optim_times = []
    total_times = []
    
    total_steps = args.warmup_steps + args.num_steps
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"MapQR Synthetic Benchmark")
        print(f"  GPUs: {world_size}, Batch/GPU: {args.batch_size}, Global: {args.batch_size * world_size}")
        print(f"  AMP: {args.amp}, Steps: {args.num_steps} (+{args.warmup_steps} warmup)")
        print(f"{'='*60}\n")
    
    for step in range(total_steps):
        # Generate data on GPU (no I/O)
        img = torch.randn(args.batch_size * 6, 3, 480, 800, device=device)
        
        # === Forward ===
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=args.amp):
            feats = raw_model.img_backbone(img)
            feats = raw_model.img_neck(feats)
            # Simulate loss (sum of features)
            loss = sum(f.mean() for f in feats)
        
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        
        # === Backward ===
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        
        # === Optimizer ===
        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
        
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        
        if step >= args.warmup_steps:
            forward_times.append((t1 - t0) * 1000)
            backward_times.append((t2 - t1) * 1000)
            optim_times.append((t3 - t2) * 1000)
            total_times.append((t3 - t0) * 1000)
    
    # Results
    if rank == 0:
        import statistics
        
        def stats(times):
            return statistics.mean(times), statistics.median(times), min(times), max(times)
        
        fwd_mean, fwd_med, fwd_min, fwd_max = stats(forward_times)
        bwd_mean, bwd_med, bwd_min, bwd_max = stats(backward_times)
        opt_mean, opt_med, opt_min, opt_max = stats(optim_times)
        tot_mean, tot_med, tot_min, tot_max = stats(total_times)
        
        global_batch = args.batch_size * world_size
        throughput = global_batch / (tot_mean / 1000)
        
        mem = torch.cuda.max_memory_allocated(device) / 1024**2
        
        print(f"{'Phase':<12} {'Mean':>8} {'Median':>8} {'Min':>8} {'Max':>8}")
        print(f"{'-'*48}")
        print(f"{'Forward':<12} {fwd_mean:>7.1f}ms {fwd_med:>7.1f}ms {fwd_min:>7.1f}ms {fwd_max:>7.1f}ms")
        print(f"{'Backward':<12} {bwd_mean:>7.1f}ms {bwd_med:>7.1f}ms {bwd_min:>7.1f}ms {bwd_max:>7.1f}ms")
        print(f"{'Optimizer':<12} {opt_mean:>7.1f}ms {opt_med:>7.1f}ms {opt_min:>7.1f}ms {opt_max:>7.1f}ms")
        print(f"{'Total':<12} {tot_mean:>7.1f}ms {tot_med:>7.1f}ms {tot_min:>7.1f}ms {tot_max:>7.1f}ms")
        print(f"\nThroughput: {throughput:.1f} samples/sec")
        print(f"GPU Memory: {mem:.0f} MB")
        print(f"Backward/Forward ratio: {bwd_mean/fwd_mean:.2f}x")
    
    if is_ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    args = parse_args()
    benchmark(args)
