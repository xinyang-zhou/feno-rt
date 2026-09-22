# Contributing to FENO-RT

FENO-RT is an MIT-licensed, single-device inference runtime. Contributions should
fit the [current scope](docs/RELEASE_HISTORY.md): cache correctness and lifecycle,
budgeted scheduling, asynchronous execution, CUDA Graphs and reproducible analysis.

## Development

Use Python 3.9+ and an isolated environment. Install a CUDA-enabled PyTorch build
when testing GPU behavior; the CPU build is sufficient for most unit tests.

```bash
python -m pip install -r requirements/runtime.lock -r requirements/test.lock
python -m pip install -e . --no-deps
python -m unittest discover -s tests -v
python benchmarks/benchmark_core.py --device cpu --repeats 3
```

CPU runs skip CUDA integration tests. Changes to capture/replay, stream ownership
or GPU output lifetime should also run those tests on CUDA. Report which checks
actually ran and which were skipped. Formal benchmark dependencies are pinned;
the installed PyTorch CUDA build suffix and driver/runtime versions are recorded
separately in result metadata.

## Changes and pull requests

Describe the problem, resulting behavior and relevant validation. Add regression
coverage when changing correctness, cancellation/deadlines, cache ownership or
resource budgets. Keep runtime dependencies separate from optional analysis tools;
plotting currently uses `matplotlib==3.10.8`.

State performance claims with workload, cache state, device, measurement boundary,
repetitions and correctness evidence. Follow [the benchmark protocol](docs/PERFORMANCE.md)
and [reproduction instructions](docs/BENCHMARK_REPRODUCTION.md). Preserve negative
results and costs alongside speedups. Profiler observations are diagnostic evidence.

## Public files and local data

Keep working notes and raw captures in an ignored `.local/` directory or outside
the checkout. Formal runners require an output directory outside their source
checkout. Do not stage model files, environment files, binary profiler reports or
unreviewed collection archives.

Before submitting benchmark evidence, prepare a reviewed public copy: anonymize
machine paths and device/process/thread identifiers, preserve measurement samples
and model/config/input hashes, document transformations, and regenerate summary
file hashes. The [result index](benchmarks/results/README.md) lists examples.

After staging the intended files, run:

```bash
python scripts/check_publication.py
git diff --cached --check
git diff --cached --stat
```

The publication check reads the index, so it checks exactly the staged content.
It recognizes selected paths, credential forms, device identifiers and file types;
it does not replace reviewing the actual diff. CI also checks the committed tree.

Changes to previously published history require separate coordination: deleting
metadata in a new commit does not remove it from older commits or external clones.
