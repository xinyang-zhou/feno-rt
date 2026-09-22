# Complete run-level metrics

Each cell is the median of three independent run-level values. Latency triplets are p50 / p95 / p99 in milliseconds. Request samples are not pooled.

| Scenario | Policy | req/s | Queue ms | Execution ms | E2E ms |
|:---|:---|---:|:---|:---|:---|
| no_reuse | fcfs | 114.32 | 4977.30 / 8183.05 / 8368.38 | 8.67 / 10.48 / 10.62 | 4986.41 / 8188.30 / 8373.51 |
| no_reuse | cache_aware | 419.10 | 1306.01 / 1942.00 / 1990.32 | 15.45 / 20.33 / 21.13 | 1322.95 / 1954.06 / 2002.27 |
| uniform_reuse | fcfs | 130.68 | 4686.03 / 7135.28 / 7270.73 | 6.64 / 10.96 / 11.93 | 4692.64 / 7139.02 / 7274.37 |
| uniform_reuse | cache_aware | 490.79 | 1161.12 / 1664.64 / 1691.61 | 13.67 / 18.15 / 19.65 | 1177.62 / 1671.32 / 1699.80 |
| long_tail_reuse | fcfs | 134.92 | 4493.17 / 6886.84 / 7027.20 | 6.80 / 12.36 / 13.32 | 4500.11 / 6891.18 / 7031.01 |
| long_tail_reuse | cache_aware | 521.38 | 1074.47 / 1550.75 / 1572.76 | 11.99 / 18.02 / 19.39 | 1090.30 / 1556.72 / 1578.43 |
| hotspot_reuse | fcfs | 136.86 | 4402.90 / 6808.93 / 6939.60 | 6.92 / 10.04 / 13.96 | 4410.37 / 6813.23 / 6943.18 |
| hotspot_reuse | cache_aware | 578.60 | 918.36 / 1351.73 / 1377.34 | 10.29 / 17.59 / 18.54 | 929.92 / 1358.95 / 1384.99 |

| Scenario | Policy | Batch | Geometry hit | Wavelet hit | Geometry shared | Wavelet shared | Scheduler us/req | Guard dispatches | Max dispatch age ms | Peak allocated MiB | Peak reserved MiB |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no_reuse | fcfs | 1.00 | 0.00% | 0.00% | 0.00% | 0.00% | 182.41 | 1000 | 8409.70 | 1677.96 | 3670.00 |
| no_reuse | cache_aware | 7.81 | 0.00% | 0.00% | 0.00% | 0.00% | 292.21 | 1000 | 2013.02 | 1870.22 | 3670.00 |
| uniform_reuse | fcfs | 1.00 | 56.10% | 93.60% | 0.00% | 0.00% | 198.34 | 1000 | 7303.94 | 1070.71 | 3010.00 |
| uniform_reuse | cache_aware | 7.81 | 54.79% | 93.34% | 5.60% | 7.70% | 284.38 | 1000 | 1704.28 | 1266.49 | 3014.00 |
| long_tail_reuse | fcfs | 1.00 | 81.80% | 96.80% | 0.00% | 0.00% | 183.08 | 1000 | 7056.82 | 934.87 | 2858.00 |
| long_tail_reuse | cache_aware | 7.81 | 74.07% | 95.23% | 48.30% | 53.70% | 271.34 | 1000 | 1581.98 | 1130.09 | 2864.00 |
| hotspot_reuse | fcfs | 1.00 | 93.90% | 99.20% | 0.00% | 0.00% | 195.82 | 1000 | 6969.53 | 857.52 | 2772.00 |
| hotspot_reuse | cache_aware | 7.81 | 80.32% | 97.51% | 82.50% | 82.30% | 255.73 | 1000 | 1392.89 | 1052.82 | 2778.00 |

Cache hit rates count cache lookups after within-batch deduplication; shared-request ratios use a different denominator. Guard dispatches do not imply a 50 ms queue bound.
