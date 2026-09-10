# FENO-RT

Cache-aware inference runtime for a non-autoregressive scientific neural operator.

[![tests](https://github.com/xinyang-zhou/feno-rt/actions/workflows/ci.yml/badge.svg)](https://github.com/xinyang-zhou/feno-rt/actions/workflows/ci.yml)

This repository is an **inference-only source release**. It contains
runtime scheduling, tensor caches, CUDA Graph support, multi-GPU serving,
observability, trace replay, and a deterministic synthetic smoke workload.

It intentionally contains no training loop, loss implementation, optimizer,
training/evaluation dataset, research notebook, normalization artifact,
pretrained checkpoint, or benchmark produced from a private checkpoint.

## What is included

- byte-bounded tensor LRU caches with reference-counted leases;
- medium, decoder-static, geometry-prefix, and frequency/wavelet caches;
- cache-aware dynamic batching with FCFS, deadlines, starvation protection,
  cancellation, timeout, and recursive error isolation;
- FP32/TF32/BF16/FP16 inference policies, SDPA selection, `torch.compile`, and
  CUDA Graph buckets with persistent buffers;
- per-GPU replicas, cache-aware routing, stable affinity, hot-context
  replication, and one spawned process per GPU for CPU-result serving;
- worker-local GPU/pinned-CPU medium tiers with tensor-free handles,
  execution-window leases, promotion/demotion, and cost-aware eviction;
- aiohttp JSON/NumPy serving, Prometheus exposition, bounded JSONL traces,
  offline replay, controlled profiling, and cache flush controls;
- CPU and GPU tests that construct randomly initialized small models;
- a deterministic, redistribution-safe synthetic input and replay trace.

The forward model definitions under `feno_rt/models/` are included only as the
inference adapter required by the runtime. There is no code that trains those
modules and no trained parameter file in this repository.

## Architecture

```text
HTTP / replay client
        |
        v
request validation + context registry + trace/profiler
        |
        v
cache-aware replica router
        |
        v
one worker process per GPU
        |
        +-- dynamic batch scheduler and admission budgets
        +-- GPU <-> pinned-CPU medium tier
        +-- four-level reusable context cache
        +-- precision / compile / CUDA Graph execution
        +-- asynchronous D2H into leased shared output slots
```

See [docs/architecture.md](docs/architecture.md) for the request lifecycle and
the ownership rules that make eviction safe.

## CPU-only smoke test

Python 3.9+ is required. Install a PyTorch build suitable for your platform,
then install the project:

```bash
python -m pip install -r requirements/runtime.lock
python -m pip install -e '.[http]' --no-deps
python scripts/data/generate_demo.py --force
python benchmarks/benchmark_stage6_replay.py
python -m unittest discover -s tests -v
```

The smoke benchmark creates a small randomly initialized model and uses the
synthetic 16 x 16 medium in `data/demo/`. It verifies cache/replay/metrics and
profiling integration. Its timing is not a production performance claim.

## Serving a compatible checkpoint

No checkpoint or normalization statistics are distributed here. If you own a
compatible inference checkpoint and its normalization file, pass both paths
explicitly:

```bash
python scripts/serve_http.py \
  --checkpoint /path/to/compatible-inference-state-dict.pth \
  --normalization /path/to/normalization-stats.npz \
  --devices cuda:0
```

The loader uses `torch.load(..., weights_only=True)`. The server binds to
`127.0.0.1` by default and does not provide TLS or authentication; do not expose
it directly to an untrusted network.

The checked-in launch policy never starts four GPUs. Two-GPU commands require
the unremapped physical pair `cuda:1 cuda:2`; single-GPU use may select an
explicit device such as `cuda:0`.

## Repository scope

The allowlist and release audit are documented in
[docs/public_release_scope.md](docs/public_release_scope.md). In particular:

- `data/demo/` is synthetic and covered by its own CC0-1.0 dedication;
- `artifacts/benchmarks/stage6_synthetic_replay.json` is produced by a random
  small CPU model and is retained only as a functional example;
- all real data, derived arrays, model parameters, normalization statistics,
  private evaluation artifacts, and training/research material are excluded.

## Tests

CPU tests run in GitHub Actions. GPU-specific tests self-skip when their exact
device requirements are unavailable. Tests never download a model or dataset.

```bash
python -m unittest discover -s tests -v
```

## License

The inference-runtime source is released under the [MIT License](LICENSE).
The generated synthetic files in `data/demo/` are covered separately by the
CC0-1.0 dedication in `data/demo/LICENSE.md`.
