# FENO-RT

Cache-aware inference runtime for a non-autoregressive scientific neural operator.

[![tests](https://github.com/xinyang-zhou/feno-rt/actions/workflows/ci.yml/badge.svg)](https://github.com/xinyang-zhou/feno-rt/actions/workflows/ci.yml)

This repository is an **inference-only source release**. It contains
runtime scheduling, tensor caches, CUDA Graph support, multi-GPU serving,
observability, trace replay, and a deterministic synthetic smoke workload.

It intentionally contains no training loop, loss implementation, optimizer,
training/evaluation dataset, or research notebook. Large model assets are never
committed to Git. A separately distributed `feno_test.pth` checkpoint is
available only for reproducible runtime-efficiency testing and carries no
scientific-accuracy claim.

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
- a versioned release manifest, checkpoint verifier, and optional single-GPU
  FENO/Deepwave performance benchmark.

The forward model definitions under `feno_rt/models/` are included only as the
inference adapter required by the runtime. There is no code that trains those
modules and no trained parameter file in the Git repository.

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

## Benchmark-only checkpoint

The optional GitHub Release assets are named `feno_test.pth` and
`norm_params_freq.npz`. The checkpoint has the production model topology but
training stopped after epoch 0. It exists to reproduce latency, throughput,
memory, cache, and scheduling measurements; do not use it to assess prediction
quality. See [the model card](docs/feno_test_model_card.md) for the measured
single-GPU results and limitations.

After downloading both assets from the matching release, verify their hashes,
state-dict structure, strict model compatibility, and a single-GPU smoke
forward:

    python scripts/verify_checkpoint.py \
      --manifest release/feno_test.manifest.json \
      --checkpoint /path/to/feno_test.pth \
      --normalization /path/to/norm_params_freq.npz \
      --device cuda:0

The same files can be served explicitly:

    python scripts/serve_http.py \
      --checkpoint /path/to/feno_test.pth \
      --normalization /path/to/norm_params_freq.npz \
      --devices cuda:0

The loader uses `torch.load(..., weights_only=True)`. The server binds to
`127.0.0.1` by default and does not provide TLS or authentication; do not expose
it directly to an untrusted network.

The checked-in launch policy never starts four GPUs. Two-GPU commands require
the unremapped physical pair `cuda:1 cuda:2`; single-GPU use may select an
explicit device such as `cuda:0`.

## Single-GPU efficiency comparison

Install the optional Deepwave baseline after installing the runtime:

    python -m pip install -r requirements/benchmark.lock

Run the performance-only comparison on exactly one explicit GPU:

    python benchmarks/benchmark_feno_deepwave.py \
      --checkpoint /path/to/feno_test.pth \
      --normalization /path/to/norm_params_freq.npz \
      --device cuda:0

The benchmark uses a deterministic synthetic 700 x 700 velocity field, 1024
output samples, 700 receivers, synchronized wall-clock timing, warmup, repeated
measurements, and peak allocated GPU memory. It reports FENO uncached,
medium-only, and fully cached paths separately. The Deepwave comparison is an
efficiency baseline only; it is not a numerical-equivalence or accuracy claim.

## Repository scope

The allowlist and release audit are documented in
[docs/public_release_scope.md](docs/public_release_scope.md). In particular:

- `data/demo/` is synthetic and covered by its own CC0-1.0 dedication;
- `artifacts/benchmarks/stage6_synthetic_replay.json` is produced by a random
  small CPU model and is retained only as a functional example;
- all real data, derived arrays, private evaluation artifacts, and
  training/research material are excluded from Git;
- `feno_test.pth` and its normalization file are separately distributed
  release assets described by `release/feno_test.manifest.json`.

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
