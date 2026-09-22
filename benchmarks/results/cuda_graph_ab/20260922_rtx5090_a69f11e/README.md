# CUDA Graph A/B: reviewed publication

Measured source: `a69f11ecc1d656f02cac09cff6fc1dddb4892bb3`. RTX 5090,
steady all-hit cache, Graph off/on, batch sizes 1/2/4/8, three independent
processes per configuration, 1,000 requests per run. All 24 runs passed.

See [summary.md](summary.md) for latency, throughput, capture cost and memory,
and [summary.json](summary.json) for run-level variation and paired comparisons.
Latency samples remain in each `graph_*_run*.json` file.

These public files are sanitized copies of the collected results. Interpreter
paths are replaced with `python`, source paths are repository-relative, output
paths use `/tmp/feno-graph-ab/session`, and the physical GPU UUID is replaced
consistently by `GPU-ANON-0`. The same replacement applies to the recorded device
selection environment variable. This alias is not an actual device UUID.

Measurement samples, timing/metric values, execution parameters, correctness,
model/config/workload SHA256 and source commit are unchanged. Summaries and their
input-file hashes were regenerated after redaction. [publication.json](publication.json)
records the transformations and original/published result hashes.

To rebuild the summary from public files:

```bash
python benchmarks/summarize_graph_ab.py \
  --input-dir benchmarks/results/cuda_graph_ab/20260922_rtx5090_a69f11e
```

Run commands describe the original invocation with machine-specific values
replaced. To collect a new session, use real model paths and select an actual GPU
as described in [the benchmark instructions](../../../README.md).
