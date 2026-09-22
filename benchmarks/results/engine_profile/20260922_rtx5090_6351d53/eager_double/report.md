# Async Engine diagnostic report

Diagnostic only: instrumentation changes host timing. No formal speedup claim.

- Source: `{'git_commit': '6351d53dec23e72c27c88b4c6650f7c4a9a572f3', 'git_dirty': False}`
- Environment: `{'python': '3.9.25', 'pytorch': '2.8.0+cu128', 'cuda': '12.8', 'device': 'cuda:0', 'gpu': 'NVIDIA GeForce RTX 5090'}`
- Workload: `{'scenario': 'hotspot_reuse', 'request_count': 64, 'arrival_interval_us': 0, 'medium_cache': 'warm', 'geometry_wavelet_cache': 'warm', 'requests_sha256': 'ca895b5b600c2802858363ad89abacd19925d506f238942800bb02d62d3c93e3'}`
- Execution: `{'policy': 'cache_aware', 'double_buffer': True, 'cuda_graphs': False, 'nvtx': True}`
- Correctness: `{'passed': True, 'relative_l2_error': 1.9839174569824536e-07, 'max_absolute_error': 6.556510925292969e-07, 'rtol': 1e-05, 'atol': 1e-05, 'probe_requests': 8, 'completed_probe_requests': 8, 'missing_probe_indices': [], 'nonfinite_probe_indices': [], 'mismatched_probe_indices': [], 'output_shape': [700, 1024], 'output_dtype': 'torch.float32', 'all_finite': True}`
- Trace health: `{'result_class': 'diagnostic', 'clock': 'perf_counter_ns', 'nvtx_enabled': True, 'dropped_events': 0, 'active_requests': 0}`

## Host phase durations

| Phase | Count | Sum ms | Mean ms | Max ms |
|:---|---:|---:|---:|---:|
| feno.batch_build | 8 | 2.437 | 0.305 | 0.478 |
| feno.cache_identity | 64 | 19.993 | 0.312 | 0.584 |
| feno.cache_lookup | 198 | 1.164 | 0.006 | 0.024 |
| feno.completion | 8 | 0.981 | 0.123 | 0.280 |
| feno.decoder_tail | 8 | 13.574 | 1.697 | 2.505 |
| feno.device_synchronize | 8 | 0.272 | 0.034 | 0.043 |
| feno.engine_window | 1 | 113.752 | 113.752 | 113.752 |
| feno.geometry_resolve | 8 | 4.930 | 0.616 | 0.692 |
| feno.input_prepare | 8 | 4.726 | 0.591 | 1.303 |
| feno.output | 8 | 0.023 | 0.003 | 0.003 |
| feno.output_ownership | 8 | 0.809 | 0.101 | 0.111 |
| feno.queue_wait | 64 | 3219.002 | 50.297 | 84.951 |
| feno.request_e2e | 64 | 3603.280 | 56.301 | 90.648 |
| feno.runner_execute | 8 | 29.809 | 3.726 | 5.460 |
| feno.scheduler_select | 8 | 4.000 | 0.500 | 1.070 |
| feno.wavelet_resolve | 8 | 3.599 | 0.450 | 0.485 |

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
  "throughput_requests_per_second": 570.1469828231462,
  "queue_latency": {
    "mean_ms": 50.29691360937501,
    "p50_ms": 42.101679000000004,
    "p95_ms": 80.9786304,
    "p99_ms": 83.97146857,
    "max_ms": 84.951094
  },
  "execution_latency": {
    "mean_ms": 4.873711875000001,
    "p50_ms": 4.6069315,
    "p95_ms": 7.788336,
    "p99_ms": 7.788336,
    "max_ms": 7.788336
  },
  "end_to_end_latency": {
    "mean_ms": 56.301242390624985,
    "p50_ms": 47.5664675,
    "p95_ms": 85.94027080000001,
    "p99_ms": 89.66932309,
    "max_ms": 90.648127
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
    "mean_preparation_ms": 0.293287875
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
    "selection_total_ms": 4.102127,
    "selection_mean_us": 512.765875,
    "selection_max_us": 1081.105,
    "selection_us_per_dispatched_request": 64.095734375,
    "dispatches": 8,
    "dispatched_requests": 64,
    "singleton_batch_ratio": 0.0,
    "geometry_shared_request_ratio": 0.828125,
    "geometry_in_batch_dedup_rate": 0.6875,
    "wavelet_shared_request_ratio": 0.8125,
    "wavelet_in_batch_dedup_rate": 0.671875,
    "deadline_guard_requests": 0,
    "starvation_guard_requests": 26,
    "max_dispatch_age_ms": 80.812152
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
