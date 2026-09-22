# CUDA Graph A/B formal summary

Status: `passed`

- Commit: `a69f11ecc1d656f02cac09cff6fc1dddb4892bb3`
- GPU: `NVIDIA GeForce RTX 5090` (`GPU-f989bf75-9cc6-41aa-3cdf-754be94c4c1b`)
- PyTorch/CUDA: `2.8.0+cu128` / `12.8`
- Runs: 24 passed / 24 expected

Percentiles below are medians of independent run-level metrics; raw samples are not pooled.

## Run-level metrics

| Batch | Graph | Runs | Mean ms | p50 ms | p95 ms | p99 ms | req/s | Peak allocated MiB |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | off | 3 | 1.152 | 1.143 | 1.186 | 1.308 | 862.66 | 703.77 |
| 1 | on | 3 | 0.682 | 0.679 | 0.697 | 0.714 | 1448.64 | 714.79 |
| 2 | off | 3 | 1.204 | 1.187 | 1.256 | 1.393 | 1651.39 | 713.98 |
| 2 | on | 3 | 0.721 | 0.718 | 0.742 | 0.763 | 2740.49 | 728.01 |
| 4 | off | 3 | 1.293 | 1.286 | 1.327 | 1.407 | 3075.13 | 736.73 |
| 4 | on | 3 | 0.909 | 0.904 | 0.922 | 0.991 | 4357.28 | 757.38 |
| 8 | off | 3 | 1.486 | 1.474 | 1.519 | 1.619 | 5354.82 | 772.17 |
| 8 | on | 3 | 1.200 | 1.191 | 1.224 | 1.494 | 6617.15 | 805.34 |

## Paired Graph on/off comparison

| Batch | Pairs | Mean latency reduction | p99 reduction | Throughput speedup | Peak allocated delta MiB |
|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 41.11% | 45.89% | 1.688x | 11.02 |
| 2 | 3 | 40.07% | 43.61% | 1.660x | 14.03 |
| 4 | 3 | 29.97% | 29.58% | 1.423x | 20.65 |
| 8 | 3 | 19.48% | 16.85% | 1.239x | 33.17 |

## CUDA Graph setup and steady-state execution

| Batch | Capture ms | Capture allocated MiB | Static buffers MiB | Measured replay rate | Fallbacks | Padded requests |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 51.069 | 15.44 | 4.58 | 100.00% | 0 | 0 |
| 2 | 56.551 | 22.75 | 9.15 | 100.00% | 0 | 0 |
| 4 | 52.475 | 37.37 | 18.30 | 100.00% | 0 | 0 |
| 8 | 55.802 | 67.38 | 36.61 | 100.00% | 0 | 0 |

## Notes

- All percentiles are summarized from run-level metrics; raw samples are not pooled.
- Latency reduction is (Graph off - Graph on) / Graph off * 100%.
- Throughput speedup is Graph on / Graph off.
- Formal runs do not collect kernel-launch counts; version-matched profiler diagnostics must report those separately.
