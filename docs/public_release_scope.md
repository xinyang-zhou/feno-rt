# Public release scope

This repository is built from an explicit inference-only allowlist.

## Included

- `feno_rt/models/`: forward definitions required by the inference adapter;
- `feno_rt/runtime/`: runner, caches, scheduler, engine, CUDA Graph, routing,
  process workers, tier management, trace replay, and synthetic workloads;
- `feno_rt/observability/` and `feno_rt/serving/`;
- `scripts/serve_http.py`, `scripts/replay_trace.py`, and the deterministic
  synthetic-demo generator;
- tests that instantiate random small models and do not download assets;
- one CPU synthetic replay benchmark and its functional example artifact;
- the `feno_test` release manifest, verifier, model card, and single-GPU
  performance-only benchmark;
- dependency locks and CPU CI.

## Explicitly excluded

- training loops, losses, optimizers, schedulers, and distributed training
  launchers;
- raw or processed scientific data and train/test split indices;
- model checkpoints and normalization statistics inside the Git tree;
- research/evaluation notebooks and plots;
- private evaluation results and benchmark artifacts produced from trained
  checkpoints;
- dataset-generation and evaluation scripts;
- absolute paths, credentials, shell history, caches, and editor state.

## Separately distributed release assets

`feno_test.pth` and `norm_params_freq.npz` are GitHub Release assets rather
than tracked repository files. Their names, sizes, SHA-256 digests, compatibility
contract, and limitations are recorded in
`release/feno_test.manifest.json`. The checkpoint is for systems-performance
reproduction only and is not an accuracy release.

## Why forward definitions remain

The runtime adapter decomposes the forward graph into reusable medium,
decoder-static, geometry, and frequency stages. Those forward-only definitions
are therefore part of the inference implementation. They expose no parameter
values and the public tests use deterministic random initialization only.

## Pre-publish checks

Run the following before every public push:

```bash
python scripts/audit_public_release.py
python -m unittest discover -s tests -v
python benchmarks/benchmark_stage6_replay.py --output /tmp/feno-rt-smoke.json
```

Also inspect `git status --short` and verify that the repository-wide MIT
License remains present. Before uploading the separate assets, run
`scripts/verify_checkpoint.py` against the staged filenames and manifest.
