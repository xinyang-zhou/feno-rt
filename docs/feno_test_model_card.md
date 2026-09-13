# feno_test model card

## Purpose

`feno_test.pth` is a benchmark-only FENO checkpoint distributed separately
from the Git repository. It preserves the full 95,659,456-parameter model
topology so that FENO-RT latency, throughput, GPU memory, caching, scheduling,
and serving paths can be reproduced.

This is not an accuracy release. Training stopped after epoch 0, and no
scientific prediction-quality claim should be derived from its output.

## Files

| Asset | Size | SHA-256 |
|---|---:|---|
| `feno_test.pth` | 684,759,337 bytes | `4807ff56b2cb6968547083df8c715a0ad868ae8f4b8de1c51576374e57f94cf2` |
| `norm_params_freq.npz` | 1,038 bytes | `b9f754ea9ff8e39b779c64740c71da1581025e0f86e1e950b986848775a3cc4a` |

The checkpoint is a PyTorch state dictionary containing 395 tensors and no
optimizer, executable model object, or non-tensor metadata. Its tensor dtypes
are FP32 and complex64. FENO-RT loads it with `weights_only=True`.

## Compatibility

- FENO-RT 0.8.0;
- `feno_rt.config.DEFAULT_INFERENCE_CONFIG`;
- PyTorch 2.8.0 and CUDA 12.8 for the recorded GPU benchmark;
- strict state-dict loading with no missing or unexpected keys.

Run the verifier before benchmarking:

```bash
python scripts/verify_checkpoint.py \
  --manifest release/feno_test.manifest.json \
  --checkpoint /path/to/feno_test.pth \
  --normalization /path/to/norm_params_freq.npz \
  --device cuda:0
```

## Benchmark scope

The public benchmark uses one explicit GPU and deterministic synthetic input.
It reports synchronized latency distributions, throughput, peak allocated GPU
memory, checkpoint loading, cold/hot medium preparation, and cached/uncached
FENO execution. Deepwave is measured with the same velocity grid, receiver
count, output length, batch size, and GPU.

The comparison deliberately excludes numerical accuracy. Deepwave is a
finite-difference propagator while FENO is a learned surrogate, so these
performance numbers must not be described as proof of physical equivalence.

## Recorded single-GPU result

Environment: one NVIDIA GeForce RTX 5090, compute capability 12.0, PyTorch
2.8.0+cu128, CUDA 12.8, Deepwave 0.0.20, FP32, no
`CUDA_VISIBLE_DEVICES` remapping. Each row uses two warmup iterations and ten
measured iterations.

| Batch | FENO all-cache median | FENO uncached median | Deepwave median | FENO all-cache throughput | FENO hot speedup over Deepwave |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.219 ms | 42.158 ms | 9.248 ms | 820.1 req/s | 7.58x |
| 4 | 1.381 ms | 45.294 ms | 19.327 ms | 2,897.3 req/s | 14.00x |
| 8 | 1.583 ms | 51.229 ms | 33.175 ms | 5,055.1 req/s | 20.96x |

Cold medium preparation had a 35.512 ms median; hot medium lookup had a
1.872 ms median. The four-level cache reduced FENO latency relative to its
uncached path by 34.58x, 32.81x, and 32.37x for batches 1, 4, and 8.

The uncached FENO path was slower than Deepwave for every measured batch. The
reported advantage therefore applies to repeated queries that reuse a
materialized medium and request contexts. Full measurements, memory fields,
hashes, workload parameters, and timing distributions are in
`artifacts/benchmarks/feno_test_single_gpu.json`.

## Single-GPU profile

A separate Nsight Systems 2024.6.2 trace was collected on physical `cuda:0`
only. It covers checkpoint loading plus cold, hot-cache, and uncached paths, so
its percentages describe the complete profiling run rather than isolated
steady-state latency. The largest GPU-kernel groups were implicit FP32
convolution (14.9%), elementwise addition (12.7%), FFT kernels (14.9%
combined), and FP32 memory-efficient attention kernels (11.3% combined).

The full `.nsys-rep`, SQLite export, and CUDA API/kernel CSV summaries are
prepared as optional release assets alongside the checkpoint. They are useful
for bottleneck inspection, but the synchronized benchmark JSON above remains
the source of record for latency comparisons.

## Intended and excluded use

Intended use:

- FENO-RT systems and cache benchmarks;
- single-GPU profiling and regression testing;
- serving-path demonstrations.

Excluded use:

- scientific interpretation;
- accuracy, generalization, or production-readiness claims;
- safety-critical or operational seismic decisions;
- additional training presented as continuation of a validated model.
