# Public benchmarks

The deterministic CPU smoke benchmark requires no checkpoint:

```bash
python scripts/data/generate_demo.py --force
python benchmarks/benchmark_stage6_replay.py
```

It constructs a small randomly initialized model, replays 24 synthetic
requests twice, checks deterministic output summaries, flushes all caches,
renders Prometheus metrics, and exports a temporary CPU profiler trace.

No private checkpoint, normalization statistic, dataset, or trained-model
performance result is required or included.

The optional single-GPU benchmark uses the separately distributed
`feno_test.pth` asset:

```bash
python benchmarks/benchmark_feno_deepwave.py \
  --checkpoint /path/to/feno_test.pth \
  --normalization /path/to/norm_params_freq.npz \
  --device cuda:0
```

It measures FENO checkpoint loading, cold and hot medium preparation,
medium-only decoding, the full four-level cache path, uncached execution, and
an optional Deepwave 0.0.20 forward baseline. Inputs are deterministic and
synthetic. Results are performance-only and make no accuracy claim.

For a quick smoke run, use `--batch-sizes 1 --warmup 1 --repeats 3`. The
canonical artifact uses the exact command and environment recorded in its JSON.
