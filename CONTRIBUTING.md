# Contributing

Contributions must remain inference-only. Do not add training code, loss
functions, datasets, checkpoints, normalization artifacts, private benchmark
results, or research notebooks.

Before submitting a change, run:

```bash
python scripts/audit_public_release.py
python -m unittest discover -s tests -v
python benchmarks/benchmark_stage6_replay.py --output /tmp/feno-rt-smoke.json
```

Performance changes should include a reproducible workload, correctness check,
latency and throughput, memory usage where relevant, and a machine-readable
artifact. Negative results should be retained when they explain why an option
is disabled.
