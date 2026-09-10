"""Dependency-free Prometheus exposition for FENO runtime statistics."""

from __future__ import annotations

from collections import Counter
from threading import RLock
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


LabelTuple = Tuple[Tuple[str, str], ...]


def _labels(**values: Any) -> LabelTuple:
    return tuple(sorted((str(key), str(value)) for key, value in values.items()))


class ServiceMetrics:
    """Thread-safe HTTP and administrative-operation counters."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._http_inflight = 0
        self._http_requests: Counter[LabelTuple] = Counter()
        self._http_duration_seconds: Counter[LabelTuple] = Counter()
        self._operations: Counter[LabelTuple] = Counter()

    def begin_http(self) -> None:
        with self._lock:
            self._http_inflight += 1

    def finish_http(
        self,
        *,
        method: str,
        route: str,
        status: int,
        duration_seconds: float,
    ) -> None:
        labels = _labels(method=method, route=route, status=int(status))
        with self._lock:
            self._http_inflight = max(0, self._http_inflight - 1)
            self._http_requests[labels] += 1
            self._http_duration_seconds[labels] += max(0.0, duration_seconds)

    def observe_operation(self, operation: str, status: str) -> None:
        with self._lock:
            self._operations[_labels(operation=operation, status=status)] += 1

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "http_inflight": self._http_inflight,
                "http_requests": dict(self._http_requests),
                "http_duration_seconds": dict(self._http_duration_seconds),
                "operations": dict(self._operations),
            }


def _escape_label(value: Any) -> str:
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace('"', '\\"')
    )


def _sample(name: str, value: Any, labels: Optional[Mapping[str, Any]] = None) -> str:
    suffix = ""
    if labels:
        rendered = ",".join(
            f'{key}="{_escape_label(label)}"' for key, label in sorted(labels.items())
        )
        suffix = "{" + rendered + "}"
    numeric = float(value)
    if numeric.is_integer():
        rendered_value = str(int(numeric))
    else:
        rendered_value = repr(numeric)
    return f"{name}{suffix} {rendered_value}"


def _family(
    lines: list,
    name: str,
    metric_type: str,
    help_text: str,
    samples: Iterable[str],
) -> None:
    rendered = list(samples)
    if not rendered:
        return
    lines.extend((f"# HELP {name} {help_text}", f"# TYPE {name} {metric_type}"))
    lines.extend(rendered)


def _tuple_labels(labels: LabelTuple) -> Dict[str, str]:
    return dict(labels)


def render_prometheus(
    runtime: Mapping[str, Any],
    service: Mapping[str, Any],
    *,
    registered_mediums: int = 0,
) -> str:
    """Render runtime and HTTP snapshots in Prometheus text format 0.0.4."""

    lines = []
    _family(
        lines,
        "feno_http_requests_total",
        "counter",
        "HTTP responses grouped by method, canonical route, and status.",
        (
            _sample("feno_http_requests_total", count, _tuple_labels(labels))
            for labels, count in service.get("http_requests", {}).items()
        ),
    )
    _family(
        lines,
        "feno_http_request_duration_seconds",
        "counter",
        "Cumulative HTTP request wall time.",
        (
            _sample(
                "feno_http_request_duration_seconds",
                duration,
                _tuple_labels(labels),
            )
            for labels, duration in service.get("http_duration_seconds", {}).items()
        ),
    )
    _family(
        lines,
        "feno_http_requests_in_flight",
        "gauge",
        "HTTP requests currently being served.",
        [_sample("feno_http_requests_in_flight", service.get("http_inflight", 0))],
    )
    _family(
        lines,
        "feno_registered_mediums",
        "gauge",
        "Medium contexts registered by the HTTP service.",
        [_sample("feno_registered_mediums", registered_mediums)],
    )
    _family(
        lines,
        "feno_admin_operations_total",
        "counter",
        "Administrative operations grouped by operation and status.",
        (
            _sample("feno_admin_operations_total", count, _tuple_labels(labels))
            for labels, count in service.get("operations", {}).items()
        ),
    )

    request_samples = []
    for state in ("submitted", "completed", "failed", "cancelled"):
        if state in runtime:
            request_samples.append(
                _sample("feno_runtime_requests_total", runtime[state], {"state": state})
            )
    _family(
        lines,
        "feno_runtime_requests_total",
        "counter",
        "Top-level routed request counters.",
        request_samples,
    )
    _family(
        lines,
        "feno_router_selections_total",
        "counter",
        "Replica-router selections by worker.",
        (
            _sample("feno_router_selections_total", count, {"worker_id": worker_id})
            for worker_id, count in runtime.get("selection_counts", {}).items()
        ),
    )
    router_gauges = {
        "feno_router_mean_seconds": float(runtime.get("routing_mean_us", 0.0)) / 1e6,
        "feno_completion_mean_seconds": float(
            runtime.get("mean_completion_latency_ms", 0.0)
        )
        / 1e3,
        "feno_cache_locality_selection_ratio": runtime.get(
            "cache_locality_selection_rate", 0.0
        ),
        "feno_selected_cache_hit_ratio": runtime.get(
            "selected_cache_entry_hit_rate", 0.0
        ),
    }
    for name, value in router_gauges.items():
        _family(lines, name, "gauge", name.replace("_", " ") + ".", [_sample(name, value)])

    worker_queue = []
    worker_memory = []
    worker_requests = []
    worker_requests_in_flight = []
    worker_batches = []
    worker_batch_size = []
    worker_d2h_copies = []
    worker_d2h_bytes = []
    worker_d2h_seconds = []
    cache_entries = []
    cache_bytes = []
    cache_capacity = []
    cache_events = []
    tier_transfer_events = []
    tier_transfer_bytes = []
    tier_source_rebuilds = []
    latency_quantiles = []
    runtime_workers = runtime.get("workers", ())
    if not isinstance(runtime_workers, (tuple, list)):
        runtime_workers = ()
    for worker in runtime_workers:
        worker_id = str(worker.get("worker_id", "unknown"))
        device = str(worker.get("device", "unknown"))
        base_labels = {"worker_id": worker_id, "device": device}
        worker_queue.append(
            _sample("feno_worker_queue_depth", worker.get("queue_depth", 0), base_labels)
        )
        for kind, key in (("free", "free_memory_bytes"), ("total", "total_memory_bytes")):
            worker_memory.append(
                _sample(
                    "feno_worker_memory_bytes",
                    worker.get(key, 0),
                    {**base_labels, "kind": kind},
                )
            )
        d2h = worker.get("d2h", {})
        worker_d2h_copies.append(
            _sample("feno_worker_d2h_copies_total", d2h.get("copies", 0), base_labels)
        )
        worker_d2h_bytes.append(
            _sample("feno_worker_d2h_bytes_total", d2h.get("bytes", 0), base_labels)
        )
        worker_d2h_seconds.append(
            _sample(
                "feno_worker_d2h_seconds_total",
                float(d2h.get("time_ms", 0.0)) / 1e3,
                base_labels,
            )
        )
        engine = worker.get("engine", {})
        for state, value in engine.get("requests", {}).items():
            if state == "pending":
                worker_requests_in_flight.append(
                    _sample("feno_worker_requests_in_flight", value, base_labels)
                )
                continue
            worker_requests.append(
                _sample(
                    "feno_worker_requests_total",
                    value,
                    {**base_labels, "state": state},
                )
            )
        worker_batches.append(
            _sample("feno_worker_batches_total", engine.get("batches", 0), base_labels)
        )
        worker_batch_size.append(
            _sample(
                "feno_worker_batch_size",
                engine.get("mean_batch_size", 0.0),
                {**base_labels, "stat": "mean"},
            )
        )
        worker_batch_size.append(
            _sample(
                "feno_worker_batch_size",
                engine.get("max_observed_batch_size", 0),
                {**base_labels, "stat": "max"},
            )
        )
        for latency_name in ("queue_latency", "execution_latency", "end_to_end_latency"):
            summary = engine.get(latency_name, {})
            for quantile, key in (("0.5", "p50_ms"), ("0.95", "p95_ms"), ("0.99", "p99_ms")):
                latency_quantiles.append(
                    _sample(
                        "feno_worker_latency_seconds",
                        float(summary.get(key, 0.0)) / 1e3,
                        {
                            **base_labels,
                            "phase": latency_name.removesuffix("_latency"),
                            "quantile": quantile,
                        },
                    )
                )
        for cache_name, cache in engine.get("cache", {}).get("caches", {}).items():
            labels = {**base_labels, "cache": cache_name}
            cache_entries.append(
                _sample(
                    "feno_cache_resident_entries",
                    cache.get("resident_entries", 0),
                    labels,
                )
            )
            cache_bytes.append(
                _sample(
                    "feno_cache_resident_bytes", cache.get("resident_bytes", 0), labels
                )
            )
            for event in (
                "hits",
                "misses",
                "insertions",
                "evictions",
                "invalidations",
                "rejections",
            ):
                cache_events.append(
                    _sample(
                        "feno_cache_events_total",
                        cache.get(event, 0),
                        {**labels, "event": event},
                    )
                )
        medium_tiers = worker.get("medium_tiers", {})
        for tier_name in ("gpu", "pinned_cpu"):
            tier = medium_tiers.get(tier_name, {})
            if not tier:
                continue
            labels = {**base_labels, "cache": f"medium_{tier_name}"}
            cache_entries.append(
                _sample(
                    "feno_cache_resident_entries",
                    tier.get("resident_entries", 0),
                    labels,
                )
            )
            cache_bytes.append(
                _sample(
                    "feno_cache_resident_bytes",
                    tier.get("resident_bytes", 0),
                    labels,
                )
            )
            cache_capacity.append(
                _sample(
                    "feno_cache_capacity_bytes",
                    tier.get("capacity_bytes", 0),
                    labels,
                )
            )
            for event in ("hits", "evictions", "capacity_overflows"):
                if event in tier:
                    cache_events.append(
                        _sample(
                            "feno_cache_events_total",
                            tier[event],
                            {**labels, "event": event},
                        )
                    )
        transfers = medium_tiers.get("transfers", {})
        for direction, event in (("h2d", "promotions"), ("d2h", "demotions")):
            tier_transfer_events.append(
                _sample(
                    "feno_medium_tier_transfers_total",
                    transfers.get(event, 0),
                    {**base_labels, "direction": direction},
                )
            )
            tier_transfer_bytes.append(
                _sample(
                    "feno_medium_tier_transfer_bytes_total",
                    transfers.get(f"{direction}_bytes", 0),
                    {**base_labels, "direction": direction},
                )
            )
        tier_source_rebuilds.append(
            _sample(
                "feno_medium_source_rebuilds_total",
                medium_tiers.get("source", {}).get("rebuilds", 0),
                base_labels,
            )
        )

    families = (
        ("feno_worker_queue_depth", "gauge", "Outstanding work by worker.", worker_queue),
        ("feno_worker_memory_bytes", "gauge", "CUDA memory information by worker.", worker_memory),
        (
            "feno_worker_requests_total",
            "counter",
            "Worker request counters by terminal state.",
            worker_requests,
        ),
        (
            "feno_worker_requests_in_flight",
            "gauge",
            "Worker requests not yet terminal.",
            worker_requests_in_flight,
        ),
        ("feno_worker_batches_total", "counter", "Executed batches by worker.", worker_batches),
        ("feno_worker_batch_size", "gauge", "Observed batch-size statistics.", worker_batch_size),
        (
            "feno_worker_d2h_copies_total",
            "counter",
            "Completed device-to-host copies.",
            worker_d2h_copies,
        ),
        (
            "feno_worker_d2h_bytes_total",
            "counter",
            "Device-to-host bytes copied.",
            worker_d2h_bytes,
        ),
        (
            "feno_worker_d2h_seconds_total",
            "counter",
            "Cumulative device-to-host copy time.",
            worker_d2h_seconds,
        ),
        ("feno_worker_latency_seconds", "gauge", "Worker latency quantiles.", latency_quantiles),
        ("feno_cache_resident_entries", "gauge", "Resident cache entry count.", cache_entries),
        ("feno_cache_resident_bytes", "gauge", "Resident cache tensor bytes.", cache_bytes),
        ("feno_cache_capacity_bytes", "gauge", "Configured cache capacity.", cache_capacity),
        ("feno_cache_events_total", "counter", "Cache lifecycle events.", cache_events),
        (
            "feno_medium_tier_transfers_total",
            "counter",
            "Medium tier promotion and demotion count.",
            tier_transfer_events,
        ),
        (
            "feno_medium_tier_transfer_bytes_total",
            "counter",
            "Medium tier transfer bytes.",
            tier_transfer_bytes,
        ),
        (
            "feno_medium_source_rebuilds_total",
            "counter",
            "Medium representations rebuilt from registered source.",
            tier_source_rebuilds,
        ),
    )
    for name, metric_type, help_text, samples in families:
        _family(lines, name, metric_type, help_text, samples)
    return "\n".join(lines) + "\n"
