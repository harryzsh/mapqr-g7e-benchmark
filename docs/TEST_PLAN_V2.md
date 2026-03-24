# Full Benchmark Test Plan (v2 — Full Dataset)

## Previous Results (INVALID — used mini dataset 323 scenes, not full 34K)

All timing numbers from earlier runs are invalid for epoch-level comparisons.
GPU metrics (utilization, memory, power, temperature, profiling) remain valid.

## Test Plan

### Phase 1: Synthetic Data Benchmark (Pure GPU Throughput)

No data loading — measures raw model compute speed with phased timing.

**Script:** `scripts/benchmark_synthetic.py`

**Tests:**
| Test | Batch/GPU | GPUs | Precision | What it measures |
|------|-----------|------|-----------|-----------------|
| A1 | 4 | 1 | FP32 | Single GPU baseline |
| A2 | 8 | 1 | FP32 | Batch scaling on 1 GPU |
| A3 | 4 | 8 | FP32 | DDP overhead |
| A4 | 8 | 8 | FP32 | DDP + larger batch |
| A5 | 8 | 8 | BF16 | Mixed precision impact |
| A6 | 4 | 2 | FP32 | DDP scaling (2 GPU) |
| A7 | 4 | 4 | FP32 | DDP scaling (4 GPU) |

**Metrics per test (50 steps, 5 warmup):**
- Forward time (ms)
- Backward time (ms)
- Optimizer step time (ms)
- Total step time (ms)
- Throughput (samples/sec)
- GPU memory (MB)

### Phase 2: Full Dataset Training (3 epochs each)

Real data loading from full 34K trainval dataset.

**Tests:**
| Test | Batch/GPU | Workers | pin_memory | persistent_workers | prefetch | Precision |
|------|-----------|---------|------------|-------------------|----------|-----------|
| B1 | 4 | 4 | False | False | 2 | FP32 | (default config)
| B2 | 8 | 8 | False | False | 2 | FP32 | (batch scaling)
| B3 | 8 | 24 | True | True | 4 | FP32 | (all optimizations)
| B4 | 8 | 24 | True | True | 4 | BF16 | (all optimizations + BF16)

**Metrics per test:**
- Epoch time (sec)
- Per-iteration time (ms)
- Loss at end of each epoch
- GPU utilization, memory, temperature, power (from gpu_monitor)

### Phase 3: DataLoader Isolation

Measure data loading overhead separately.

**Tests:**
| Test | Workers/GPU | What it measures |
|------|-------------|-----------------|
| C1 | 0 | Main process loading (worst case) |
| C2 | 4 | Default workers |
| C3 | 8 | More workers |
| C4 | 24 | Max workers (192 vCPUs / 8 GPUs) |

With and without pin_memory for each.

### Phase 4: DDP Scaling Curve

Fixed batch=4/GPU, vary GPU count.

| GPUs | Global Batch | Expected |
|------|-------------|----------|
| 1 | 4 | Baseline |
| 2 | 8 | ~1.9x |
| 4 | 16 | ~3.5x |
| 8 | 32 | ~6-7x |

### Phase 5: pin_memory Impact

Same config, toggle pin_memory only.

| pin_memory | Expected impact |
|------------|----------------|
| False | Baseline |
| True | ~5-10% faster (based on Pluto benchmark) |

## Performance Settings Applied

All optimized runs use:
```python
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=24,
    persistent_workers=True,
    prefetch_factor=4,
    pin_memory=True,
)
```

## Expected Outcomes

Based on Pluto benchmark patterns and our profiling:

1. **Synthetic vs real data gap**: ~10-15% overhead from data loading
2. **pin_memory**: ~5-10% improvement
3. **persistent_workers**: ~5% improvement (avoids worker restart per epoch)
4. **BF16**: Higher power draw, similar or slightly faster speed (model too small)
5. **DDP scaling**: ~85-90% efficiency at 8 GPUs (based on Pluto's 88.8%)
6. **Batch scaling**: Diminishing returns past bs=8 due to deformable attention constraint
