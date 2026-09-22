# Async Engine diagnostic report

Diagnostic only: instrumentation changes host timing. No formal speedup claim.

- Source: `{'git_commit': '6351d53dec23e72c27c88b4c6650f7c4a9a572f3', 'git_dirty': False}`
- Environment: `{'python': '3.9.25', 'pytorch': '2.8.0+cu128', 'cuda': '12.8', 'device': 'cuda:0', 'gpu': 'NVIDIA GeForce RTX 5090'}`
- Workload: `{'scenario': 'hotspot_reuse', 'request_count': 64, 'arrival_interval_us': 0, 'medium_cache': 'warm', 'geometry_wavelet_cache': 'warm', 'requests_sha256': 'ca895b5b600c2802858363ad89abacd19925d506f238942800bb02d62d3c93e3'}`
- Execution: `{'policy': 'cache_aware', 'double_buffer': True, 'cuda_graphs': True, 'nvtx': True}`
- Correctness: `{'passed': True, 'relative_l2_error': 1.9839174569824536e-07, 'max_absolute_error': 6.556510925292969e-07, 'rtol': 1e-05, 'atol': 1e-05, 'probe_requests': 8, 'completed_probe_requests': 8, 'missing_probe_indices': [], 'nonfinite_probe_indices': [], 'mismatched_probe_indices': [], 'output_shape': [700, 1024], 'output_dtype': 'torch.float32', 'all_finite': True}`
- Trace health: `{'result_class': 'diagnostic', 'clock': 'perf_counter_ns', 'nvtx_enabled': True, 'dropped_events': 0, 'active_requests': 0}`

## Host phase durations

| Phase | Count | Sum ms | Mean ms | Max ms |
|:---|---:|---:|---:|---:|
| feno.batch_build | 8 | 1.928 | 0.241 | 0.290 |
| feno.cache_identity | 64 | 19.389 | 0.303 | 0.393 |
| feno.cache_lookup | 487 | 2.560 | 0.005 | 0.063 |
| feno.completion | 8 | 0.565 | 0.071 | 0.075 |
| feno.decoder_tail | 8 | 2.208 | 0.276 | 0.419 |
| feno.device_synchronize | 8 | 0.632 | 0.079 | 0.210 |
| feno.engine_window | 1 | 58.486 | 58.486 | 58.486 |
| feno.geometry_resolve | 8 | 4.827 | 0.603 | 0.630 |
| feno.input_prepare | 8 | 5.415 | 0.677 | 2.409 |
| feno.output | 8 | 0.019 | 0.002 | 0.003 |
| feno.output_ownership | 8 | 1.565 | 0.196 | 0.285 |
| feno.queue_wait | 64 | 1896.198 | 29.628 | 53.254 |
| feno.request_e2e | 64 | 2155.162 | 33.674 | 56.065 |
| feno.runner_execute | 8 | 20.081 | 2.510 | 4.367 |
| feno.scheduler_select | 8 | 8.003 | 1.000 | 1.633 |
| feno.wavelet_resolve | 8 | 3.673 | 0.459 | 0.479 |

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
  "throughput_requests_per_second": 1109.8879028825886,
  "queue_latency": {
    "mean_ms": 29.628097812500005,
    "p50_ms": 29.525853499999997,
    "p95_ms": 46.81405155,
    "p99_ms": 52.26577238,
    "max_ms": 53.254415
  },
  "execution_latency": {
    "mean_ms": 3.635918874999999,
    "p50_ms": 3.4676955,
    "p95_ms": 6.544601,
    "p99_ms": 6.544601,
    "max_ms": 6.544601
  },
  "end_to_end_latency": {
    "mean_ms": 33.67441123437501,
    "p50_ms": 33.071892500000004,
    "p95_ms": 49.940773750000005,
    "p99_ms": 55.04982604999999,
    "max_ms": 56.064671
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
    "double_buffered": true,
    "prepared_batches": 8,
    "mean_preparation_ms": 0.23416575
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
        "hits": 19,
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
        "hits": 19,
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
      "geometry_unique": 19,
      "geometry_in_batch_dedup_hits": 45,
      "wavelet_requests": 64,
      "wavelet_unique": 19,
      "wavelet_in_batch_dedup_hits": 45,
      "geometry_dedup_rate": 0.703125,
      "wavelet_dedup_rate": 0.703125
    }
  },
  "scheduler": {
    "selection_calls": 8,
    "selection_total_ms": 8.104363,
    "selection_mean_us": 1013.045375,
    "selection_max_us": 1645.509,
    "selection_us_per_dispatched_request": 126.630671875,
    "dispatches": 8,
    "dispatched_requests": 64,
    "singleton_batch_ratio": 0.0,
    "geometry_shared_request_ratio": 0.859375,
    "geometry_in_batch_dedup_rate": 0.703125,
    "wavelet_shared_request_ratio": 0.84375,
    "wavelet_in_batch_dedup_rate": 0.703125,
    "deadline_guard_requests": 0,
    "starvation_guard_requests": 1,
    "max_dispatch_age_ms": 50.540065000000006
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
