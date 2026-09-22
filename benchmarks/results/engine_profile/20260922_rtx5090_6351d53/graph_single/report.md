# Async Engine diagnostic report

Diagnostic only: instrumentation changes host timing. No formal speedup claim.

- Source: `{'git_commit': '6351d53dec23e72c27c88b4c6650f7c4a9a572f3', 'git_dirty': False}`
- Environment: `{'python': '3.9.25', 'pytorch': '2.8.0+cu128', 'cuda': '12.8', 'device': 'cuda:0', 'gpu': 'NVIDIA GeForce RTX 5090'}`
- Workload: `{'scenario': 'hotspot_reuse', 'request_count': 64, 'arrival_interval_us': 0, 'medium_cache': 'warm', 'geometry_wavelet_cache': 'warm', 'requests_sha256': 'ca895b5b600c2802858363ad89abacd19925d506f238942800bb02d62d3c93e3'}`
- Execution: `{'policy': 'cache_aware', 'double_buffer': False, 'cuda_graphs': True, 'nvtx': True}`
- Correctness: `{'passed': True, 'relative_l2_error': 1.9839174569824536e-07, 'max_absolute_error': 6.556510925292969e-07, 'rtol': 1e-05, 'atol': 1e-05, 'probe_requests': 8, 'completed_probe_requests': 8, 'missing_probe_indices': [], 'nonfinite_probe_indices': [], 'mismatched_probe_indices': [], 'output_shape': [700, 1024], 'output_dtype': 'torch.float32', 'all_finite': True}`
- Trace health: `{'result_class': 'diagnostic', 'clock': 'perf_counter_ns', 'nvtx_enabled': True, 'dropped_events': 0, 'active_requests': 0}`

## Host phase durations

| Phase | Count | Sum ms | Mean ms | Max ms |
|:---|---:|---:|---:|---:|
| feno.batch_build | 8 | 0.746 | 0.093 | 0.099 |
| feno.cache_identity | 64 | 19.623 | 0.307 | 0.392 |
| feno.cache_lookup | 462 | 2.525 | 0.005 | 0.068 |
| feno.completion | 8 | 0.597 | 0.075 | 0.087 |
| feno.decoder_tail | 8 | 2.264 | 0.283 | 0.436 |
| feno.device_synchronize | 8 | 0.627 | 0.078 | 0.206 |
| feno.engine_window | 1 | 58.519 | 58.519 | 58.519 |
| feno.geometry_resolve | 8 | 4.912 | 0.614 | 0.645 |
| feno.input_prepare | 8 | 3.424 | 0.428 | 0.749 |
| feno.output | 8 | 0.019 | 0.002 | 0.003 |
| feno.output_ownership | 8 | 1.535 | 0.192 | 0.270 |
| feno.queue_wait | 64 | 1891.632 | 29.557 | 51.423 |
| feno.request_e2e | 64 | 2090.829 | 32.669 | 54.197 |
| feno.runner_execute | 8 | 18.169 | 2.271 | 2.640 |
| feno.scheduler_select | 8 | 7.832 | 0.979 | 1.634 |
| feno.wavelet_resolve | 8 | 3.647 | 0.456 | 0.474 |

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
  "throughput_requests_per_second": 1107.9693831357954,
  "queue_latency": {
    "mean_ms": 29.556756171875012,
    "p50_ms": 29.035927,
    "p95_ms": 48.6504811,
    "p99_ms": 50.877303049999995,
    "max_ms": 51.423239
  },
  "execution_latency": {
    "mean_ms": 2.706030999999997,
    "p50_ms": 2.5181875,
    "p95_ms": 4.125224,
    "p99_ms": 4.125224,
    "max_ms": 4.125224
  },
  "end_to_end_latency": {
    "mean_ms": 32.6692013125,
    "p50_ms": 32.0857545,
    "p95_ms": 51.43992085000001,
    "p99_ms": 53.66058665,
    "max_ms": 54.196871
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
    "mean_preparation_ms": 0.086291
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
    "selection_total_ms": 7.908381,
    "selection_mean_us": 988.547625,
    "selection_max_us": 1645.744,
    "selection_us_per_dispatched_request": 123.568453125,
    "dispatches": 8,
    "dispatched_requests": 64,
    "singleton_batch_ratio": 0.0,
    "geometry_shared_request_ratio": 0.859375,
    "geometry_in_batch_dedup_rate": 0.703125,
    "wavelet_shared_request_ratio": 0.84375,
    "wavelet_in_batch_dedup_rate": 0.703125,
    "deadline_guard_requests": 0,
    "starvation_guard_requests": 2,
    "max_dispatch_age_ms": 51.117095
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
