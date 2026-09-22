# Async Engine diagnostic report

Diagnostic only: instrumentation changes host timing. No formal speedup claim.

- Source: `{'git_commit': '6351d53dec23e72c27c88b4c6650f7c4a9a572f3', 'git_dirty': False}`
- Environment: `{'python': '3.9.25', 'pytorch': '2.8.0+cu128', 'cuda': '12.8', 'device': 'cuda:0', 'gpu': 'NVIDIA GeForce RTX 5090'}`
- Workload: `{'scenario': 'hotspot_reuse', 'request_count': 64, 'arrival_interval_us': 0, 'medium_cache': 'warm', 'geometry_wavelet_cache': 'warm', 'requests_sha256': 'ca895b5b600c2802858363ad89abacd19925d506f238942800bb02d62d3c93e3'}`
- Execution: `{'policy': 'cache_aware', 'double_buffer': False, 'cuda_graphs': False, 'nvtx': True}`
- Correctness: `{'passed': True, 'relative_l2_error': 1.9839174569824536e-07, 'max_absolute_error': 6.556510925292969e-07, 'rtol': 1e-05, 'atol': 1e-05, 'probe_requests': 8, 'completed_probe_requests': 8, 'missing_probe_indices': [], 'nonfinite_probe_indices': [], 'mismatched_probe_indices': [], 'output_shape': [700, 1024], 'output_dtype': 'torch.float32', 'all_finite': True}`
- Trace health: `{'result_class': 'diagnostic', 'clock': 'perf_counter_ns', 'nvtx_enabled': True, 'dropped_events': 0, 'active_requests': 0}`

## Host phase durations

| Phase | Count | Sum ms | Mean ms | Max ms |
|:---|---:|---:|---:|---:|
| feno.batch_build | 8 | 1.459 | 0.182 | 0.199 |
| feno.cache_identity | 64 | 33.494 | 0.523 | 0.983 |
| feno.cache_lookup | 57 | 0.640 | 0.011 | 0.033 |
| feno.completion | 8 | 1.148 | 0.143 | 0.173 |
| feno.decoder_tail | 8 | 16.226 | 2.028 | 3.101 |
| feno.device_synchronize | 8 | 0.408 | 0.051 | 0.067 |
| feno.engine_window | 1 | 196.634 | 196.634 | 196.634 |
| feno.geometry_resolve | 8 | 5.544 | 0.693 | 0.814 |
| feno.input_prepare | 8 | 4.646 | 0.581 | 1.349 |
| feno.output | 8 | 0.026 | 0.003 | 0.003 |
| feno.output_ownership | 8 | 0.960 | 0.120 | 0.133 |
| feno.queue_wait | 64 | 5539.160 | 86.549 | 162.407 |
| feno.request_e2e | 64 | 6012.805 | 93.950 | 167.635 |
| feno.runner_execute | 8 | 34.276 | 4.284 | 6.378 |
| feno.scheduler_select | 8 | 3.079 | 0.385 | 0.626 |
| feno.wavelet_resolve | 8 | 4.079 | 0.510 | 0.547 |

Phases overlap across threads and requests and include nested scopes. Do not sum this table into an end-to-end total or GPU execution time.

## Request and scheduling metrics

```json
{
  "requests": {
    "submitted": 64,
    "succeeded": 64,
    "failed": 0,
    "cancelled": 0,
    "timed_out": 0,
    "pending": 0
  },
  "batches": 8,
  "isolated_retries": 0,
  "mean_batch_size": 8.0,
  "max_observed_batch_size": 8,
  "effective_batch_fill_ratio": 1.0,
  "throughput_requests_per_second": 330.2131407669261,
  "queue_latency": {
    "mean_ms": 86.54937715625,
    "p50_ms": 56.1804405,
    "p95_ms": 156.69782225,
    "p99_ms": 160.72486513,
    "max_ms": 162.407374
  },
  "execution_latency": {
    "mean_ms": 5.163156624999995,
    "p50_ms": 4.5548614999999995,
    "p95_ms": 9.252506,
    "p99_ms": 9.252506,
    "max_ms": 9.252506
  },
  "end_to_end_latency": {
    "mean_ms": 93.95007296874999,
    "p50_ms": 62.337063,
    "p95_ms": 161.86777819999998,
    "p99_ms": 165.92082813000002,
    "max_ms": 167.635404
  },
  "policy": "cache_aware",
  "limits": {
    "max_wait_us": 2000,
    "max_batch_size": 8,
    "max_query_tokens": 5600,
    "max_activation_bytes": 2147483648,
    "max_output_bytes": 536870912
  },
  "pipeline": {
    "double_buffered": false,
    "prepared_batches": 8,
    "mean_preparation_ms": 0.170603625
  },
  "cache": {
    "caches": {
      "medium": {
        "name": "medium",
        "capacity_bytes": 2415919104,
        "resident_bytes": 130457600,
        "resident_entries": 4,
        "pinned_entries": 0,
        "hits": 0,
        "misses": 0,
        "hit_rate": 0.0,
        "insertions": 0,
        "evictions": 0,
        "invalidations": 0,
        "rejections": 0
      },
      "geometry_prefix": {
        "name": "geometry_prefix",
        "capacity_bytes": 536870912,
        "resident_bytes": 5376000,
        "resident_entries": 15,
        "pinned_entries": 0,
        "hits": 20,
        "misses": 0,
        "hit_rate": 1.0,
        "insertions": 0,
        "evictions": 0,
        "invalidations": 0,
        "rejections": 0
      },
      "wavelet": {
        "name": "wavelet",
        "capacity_bytes": 536870912,
        "resident_bytes": 12582912,
        "resident_entries": 8,
        "pinned_entries": 0,
        "hits": 21,
        "misses": 0,
        "hit_rate": 1.0,
        "insertions": 0,
        "evictions": 0,
        "invalidations": 0,
        "rejections": 0
      }
    },
    "runtime": {
      "geometry_requests": 64,
      "geometry_unique": 20,
      "geometry_in_batch_dedup_hits": 44,
      "wavelet_requests": 64,
      "wavelet_unique": 21,
      "wavelet_in_batch_dedup_hits": 43,
      "geometry_dedup_rate": 0.6875,
      "wavelet_dedup_rate": 0.671875
    }
  },
  "scheduler": {
    "selection_calls": 8,
    "selection_total_ms": 3.22425,
    "selection_mean_us": 403.03125,
    "selection_max_us": 644.812,
    "selection_us_per_dispatched_request": 50.37890625,
    "dispatches": 8,
    "dispatched_requests": 64,
    "singleton_batch_ratio": 0.0,
    "geometry_shared_request_ratio": 0.828125,
    "geometry_in_batch_dedup_rate": 0.6875,
    "wavelet_shared_request_ratio": 0.8125,
    "wavelet_in_batch_dedup_rate": 0.671875,
    "deadline_guard_requests": 0,
    "starvation_guard_requests": 39,
    "max_dispatch_age_ms": 161.654222
  }
}
```

## Interpretation

- queue_wait ends at execution activation, including prefetched waiting time.
- runner_execute includes host dispatch, output ownership and the configured CUDA synchronization.
- output_ownership isolates the engine's output clone/split from the runner's own output stage.
- completion measures result delivery on the event loop; request_e2e ends when the future is resolved.
- GPU kernel activity, API launch gaps and CPU scheduling stalls require the matching Nsight report.
- Compare the same trace with single/double buffering and Graph off/on before attributing a bottleneck.
- Truncated traces or active requests invalidate phase-count analysis.
