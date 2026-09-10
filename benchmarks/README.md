# Public benchmark

The repository retains one deterministic CPU smoke benchmark:

```bash
python scripts/data/generate_demo.py --force
python benchmarks/benchmark_stage6_replay.py
```

It constructs a small randomly initialized model, replays 24 synthetic
requests twice, checks deterministic output summaries, flushes all caches,
renders Prometheus metrics, and exports a temporary CPU profiler trace.

No private checkpoint, normalization statistic, dataset, or trained-model
performance result is required or included.
