# Public synthetic demo

This directory contains only deterministic synthetic inputs. It does **not**
contain field observations, any private dataset, trained weights, or
model predictions.

Regenerate the binary velocity array and replay trace from the repository root:

```bash
python scripts/data/generate_demo.py --force
```

`velocity.npy` is a 16 x 16 float32 synthetic velocity field. `requests.jsonl`
contains 24 model-independent requests in the Stage 6 trace schema. Its results
are marked `unverified`; the benchmark establishes repeatability using the
locally installed model and records its own checksums.

The generated files are covered by [LICENSE.md](LICENSE.md). The rest of the
project and any separately distributed data or weights are not covered by that
dedication.
