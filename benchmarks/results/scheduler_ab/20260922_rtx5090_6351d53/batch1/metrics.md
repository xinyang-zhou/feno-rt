# Complete run-level metrics

Each cell is the median of three independent run-level values. Latency triplets are p50 / p95 / p99 in milliseconds. Request samples are not pooled.

| Scenario | Policy | req/s | Queue ms | Execution ms | E2E ms |
|:---|:---|---:|:---|:---|:---|
| no_reuse | fcfs | 106.87 | 5051.34 / 8588.96 / 8872.71 | 8.94 / 10.52 / 10.64 | 5060.67 / 8596.66 / 8880.26 |
| no_reuse | cache_aware | 68.76 | 9295.59 / 13824.82 / 14132.29 | 13.70 / 21.31 / 21.65 | 9309.78 / 13833.27 / 14140.06 |
| uniform_reuse | fcfs | 131.39 | 4643.76 / 7102.33 / 7242.07 | 6.64 / 10.90 / 11.71 | 4650.27 / 7106.08 / 7245.77 |
| uniform_reuse | cache_aware | 81.43 | 7827.78 / 11745.00 / 11908.89 | 10.94 / 19.45 / 21.15 | 7838.00 / 11749.75 / 11913.17 |
| long_tail_reuse | fcfs | 136.58 | 4416.05 / 6820.59 / 6957.22 | 6.74 / 12.42 / 13.04 | 4422.98 / 6824.43 / 6961.00 |
| long_tail_reuse | cache_aware | 90.87 | 6669.58 / 10467.39 / 10637.08 | 10.37 / 18.13 / 18.95 | 6680.20 / 10472.35 / 10641.32 |
| hotspot_reuse | fcfs | 139.10 | 4300.13 / 6688.89 / 6822.08 | 6.67 / 9.27 / 13.73 | 4307.23 / 6692.93 / 6825.88 |
| hotspot_reuse | cache_aware | 95.89 | 6250.81 / 9888.63 / 10058.71 | 9.93 / 14.11 / 18.46 | 6261.13 / 9893.94 / 10063.19 |

| Scenario | Policy | Batch | Geometry hit | Wavelet hit | Geometry shared | Wavelet shared | Scheduler us/req | Guard dispatches | Max dispatch age ms | Peak allocated MiB | Peak reserved MiB |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no_reuse | fcfs | 1.00 | 0.00% | 0.00% | 0.00% | 0.00% | 186.60 | 1000 | 8935.86 | 1677.96 | 3670.00 |
| no_reuse | cache_aware | 1.00 | 0.00% | 0.00% | 0.00% | 0.00% | 2578.56 | 1000 | 14199.16 | 1677.96 | 3672.00 |
| uniform_reuse | fcfs | 1.00 | 56.10% | 93.60% | 0.00% | 0.00% | 191.46 | 1000 | 7276.58 | 1070.71 | 3010.00 |
| uniform_reuse | cache_aware | 1.00 | 56.10% | 93.60% | 0.00% | 0.00% | 2551.17 | 1000 | 11944.46 | 1070.71 | 3010.00 |
| long_tail_reuse | fcfs | 1.00 | 81.80% | 96.80% | 0.00% | 0.00% | 184.01 | 1000 | 6986.25 | 934.87 | 2858.00 |
| long_tail_reuse | cache_aware | 1.00 | 81.80% | 96.80% | 0.00% | 0.00% | 2531.53 | 1000 | 10669.57 | 934.87 | 2858.00 |
| hotspot_reuse | fcfs | 1.00 | 93.90% | 99.20% | 0.00% | 0.00% | 182.70 | 1000 | 6853.85 | 857.52 | 2772.00 |
| hotspot_reuse | cache_aware | 1.00 | 93.90% | 99.20% | 0.00% | 0.00% | 2570.34 | 1000 | 10093.11 | 857.52 | 2772.00 |

Cache hit rates count cache lookups after within-batch deduplication; shared-request ratios use a different denominator. Guard dispatches do not imply a 50 ms queue bound.
