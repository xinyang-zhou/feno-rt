"""Request lifecycle primitives for asynchronous FENO inference."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from time import perf_counter_ns
from typing import Any, Callable, Dict, Hashable, Mapping, Optional, Tuple


class RequestState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    FAILED = "failed"


class RequestTimeoutError(TimeoutError):
    """Raised when a request misses its absolute deadline."""


class EngineClosedError(RuntimeError):
    """Raised when work is submitted to an engine that is closing or closed."""


@dataclass(frozen=True)
class RequestCost:
    """Worst-case online memory/token cost used for batch admission."""

    query_tokens: int
    activation_bytes: int
    output_bytes: int

    def __post_init__(self) -> None:
        if self.query_tokens <= 0:
            raise ValueError("query_tokens must be positive")
        if self.activation_bytes <= 0:
            raise ValueError("activation_bytes must be positive")
        if self.output_bytes <= 0:
            raise ValueError("output_bytes must be positive")


@dataclass
class InferenceRequest:
    """One single-shot inference request stored in the dynamic queue."""

    request_id: str
    sequence_id: int
    context: Any
    source_position: Any
    frequency: Any
    receiver_positions: Optional[Any]
    positions_are_normalized: bool
    denormalize: bool
    cache_level: str
    priority: int
    cost: RequestCost
    batch_key: Hashable
    geometry_key: Hashable
    frequency_key: Hashable
    cache_keys: Mapping[str, Hashable]
    future: "asyncio.Future[Any]" = field(repr=False, compare=False)
    submitted_ns: int = field(default_factory=perf_counter_ns)
    enqueued_ns: int = field(default_factory=perf_counter_ns)
    deadline_ns: Optional[int] = None
    state: RequestState = RequestState.QUEUED
    started_ns: Optional[int] = None
    completed_ns: Optional[int] = None
    error: Optional[BaseException] = field(default=None, repr=False, compare=False)

    def age_us(self, now_ns: Optional[int] = None) -> float:
        current = perf_counter_ns() if now_ns is None else now_ns
        return max(0.0, (current - self.enqueued_ns) / 1_000.0)

    def deadline_slack_us(self, now_ns: Optional[int] = None) -> float:
        if self.deadline_ns is None:
            return float("inf")
        current = perf_counter_ns() if now_ns is None else now_ns
        return (self.deadline_ns - current) / 1_000.0

    def is_expired(self, now_ns: Optional[int] = None) -> bool:
        return self.deadline_slack_us(now_ns) <= 0.0

    @property
    def terminal(self) -> bool:
        """Return whether the request has produced a result, error, or cancellation."""
        return self.future.done()


class RequestHandle:
    """Awaitable result handle with best-effort queued/running cancellation."""

    def __init__(
        self,
        request: InferenceRequest,
        cancel_callback: Callable[[InferenceRequest], bool],
    ) -> None:
        self._request = request
        self._cancel_callback = cancel_callback

    @property
    def request_id(self) -> str:
        return self._request.request_id

    @property
    def done(self) -> bool:
        return self._request.future.done()

    def cancel(self) -> bool:
        return self._cancel_callback(self._request)

    async def result(self) -> Any:
        return await self._request.future

    def __await__(self):
        return self.result().__await__()


@dataclass
class RequestMetrics:
    """Mutable counters and latency samples owned by one engine."""

    submitted: int = 0
    succeeded: int = 0
    failed: int = 0
    cancelled: int = 0
    timed_out: int = 0
    batches: int = 0
    isolated_retries: int = 0
    batch_sizes: list = field(default_factory=list)
    queue_ms: list = field(default_factory=list)
    execution_ms: list = field(default_factory=list)
    end_to_end_ms: list = field(default_factory=list)
    started_ns: int = field(default_factory=perf_counter_ns)

    @staticmethod
    def _percentile(values: list, percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(float(value) for value in values)
        position = (len(ordered) - 1) * percentile / 100.0
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = position - lower
        return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction

    @classmethod
    def _latency_summary(cls, values: list) -> Dict[str, float]:
        return {
            "mean_ms": sum(values) / len(values) if values else 0.0,
            "p50_ms": cls._percentile(values, 50.0),
            "p95_ms": cls._percentile(values, 95.0),
            "p99_ms": cls._percentile(values, 99.0),
            "max_ms": max(values) if values else 0.0,
        }

    def snapshot(
        self,
        *,
        max_batch_size: int,
        include_raw_samples: bool = False,
    ) -> Dict[str, Any]:
        elapsed_s = max((perf_counter_ns() - self.started_ns) / 1e9, 1e-12)
        mean_batch = sum(self.batch_sizes) / len(self.batch_sizes) if self.batch_sizes else 0.0
        terminal = self.succeeded + self.failed + self.cancelled + self.timed_out
        snapshot = {
            "requests": {
                "submitted": self.submitted,
                "succeeded": self.succeeded,
                "failed": self.failed,
                "cancelled": self.cancelled,
                "timed_out": self.timed_out,
                "pending": max(0, self.submitted - terminal),
            },
            "batches": self.batches,
            "isolated_retries": self.isolated_retries,
            "mean_batch_size": mean_batch,
            "max_observed_batch_size": max(self.batch_sizes) if self.batch_sizes else 0,
            "effective_batch_fill_ratio": mean_batch / max_batch_size,
            "throughput_requests_per_second": self.succeeded / elapsed_s,
            "queue_latency": self._latency_summary(self.queue_ms),
            "execution_latency": self._latency_summary(self.execution_ms),
            "end_to_end_latency": self._latency_summary(self.end_to_end_ms),
        }
        if include_raw_samples:
            snapshot["raw_samples"] = {
                "batch_sizes": list(self.batch_sizes),
                "queue_latency_ms": list(self.queue_ms),
                "execution_latency_ms": list(self.execution_ms),
                "end_to_end_latency_ms": list(self.end_to_end_ms),
            }
        return snapshot
