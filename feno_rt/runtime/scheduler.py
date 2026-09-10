"""FCFS and cache-aware admission policies for Stage-3 dynamic batching."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
from time import perf_counter_ns
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from .request import InferenceRequest, RequestCost


class SchedulingPolicy(str, Enum):
    FCFS = "fcfs"
    CACHE_AWARE = "cache_aware"


@dataclass(frozen=True)
class DynamicBatchConfig:
    """Queue timing, admission budgets, and failure-handling controls."""

    policy: SchedulingPolicy = SchedulingPolicy.CACHE_AWARE
    max_wait_us: int = 2_000
    max_batch_size: int = 64
    max_query_tokens: int = 64 * 700
    max_activation_bytes: int = 2 * 1024 * 1024 * 1024
    max_output_bytes: int = 512 * 1024 * 1024
    deadline_guard_us: int = 500
    starvation_timeout_us: int = 50_000
    default_timeout_s: Optional[float] = None
    error_isolation: bool = True
    synchronize_device: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.policy, str):
            object.__setattr__(self, "policy", SchedulingPolicy(self.policy))
        if self.max_wait_us < 0:
            raise ValueError("max_wait_us must be non-negative")
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        for name in ("max_query_tokens", "max_activation_bytes", "max_output_bytes"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.deadline_guard_us < 0:
            raise ValueError("deadline_guard_us must be non-negative")
        if self.starvation_timeout_us <= 0:
            raise ValueError("starvation_timeout_us must be positive")
        if self.default_timeout_s is not None and self.default_timeout_s <= 0:
            raise ValueError("default_timeout_s must be positive")


@dataclass
class _BatchCost:
    requests: int = 0
    query_tokens: int = 0
    activation_bytes: int = 0
    output_bytes: int = 0

    def add(self, cost: RequestCost) -> None:
        self.requests += 1
        self.query_tokens += cost.query_tokens
        self.activation_bytes += cost.activation_bytes
        self.output_bytes += cost.output_bytes


CacheProbe = Callable[[InferenceRequest], Mapping[str, bool]]


class BaseBatchScheduler:
    def __init__(
        self,
        config: DynamicBatchConfig,
        *,
        cache_probe: Optional[CacheProbe] = None,
    ) -> None:
        self.config = config
        self._cache_probe = cache_probe or (lambda _request: {})

    def request_fits_empty_batch(self, request: InferenceRequest) -> bool:
        return self._fits(_BatchCost(), request.cost)

    def _fits(self, current: _BatchCost, incoming: RequestCost) -> bool:
        return (
            current.requests + 1 <= self.config.max_batch_size
            and current.query_tokens + incoming.query_tokens <= self.config.max_query_tokens
            and current.activation_bytes + incoming.activation_bytes
            <= self.config.max_activation_bytes
            and current.output_bytes + incoming.output_bytes <= self.config.max_output_bytes
        )

    def _admit_ordered(self, requests: Iterable[InferenceRequest]) -> List[InferenceRequest]:
        selected: List[InferenceRequest] = []
        cost = _BatchCost()
        for request in requests:
            if not self._fits(cost, request.cost):
                continue
            selected.append(request)
            cost.add(request.cost)
            if len(selected) >= self.config.max_batch_size:
                break
        return selected

    def is_saturated(
        self,
        batch: Sequence[InferenceRequest],
        pending: Sequence[InferenceRequest],
    ) -> bool:
        if len(batch) >= self.config.max_batch_size:
            return True
        if not batch:
            return False
        selected_ids = {request.request_id for request in batch}
        cost = _BatchCost()
        for request in batch:
            cost.add(request.cost)
        compatible = [
            request
            for request in pending
            if request.request_id not in selected_ids
            and request.batch_key == batch[0].batch_key
        ]
        return bool(compatible) and not any(
            self._fits(cost, request.cost) for request in compatible
        )

    def select_batch(
        self,
        pending: Sequence[InferenceRequest],
        *,
        now_ns: Optional[int] = None,
    ) -> List[InferenceRequest]:
        raise NotImplementedError


class FCFSScheduler(BaseBatchScheduler):
    """Strict arrival-order baseline; never skips an incompatible head item."""

    def select_batch(
        self,
        pending: Sequence[InferenceRequest],
        *,
        now_ns: Optional[int] = None,
    ) -> List[InferenceRequest]:
        del now_ns
        if not pending:
            return []
        ordered = sorted(pending, key=lambda request: request.sequence_id)
        batch_key = ordered[0].batch_key
        contiguous: List[InferenceRequest] = []
        for request in ordered:
            if request.batch_key != batch_key:
                break
            contiguous.append(request)
        return self._admit_ordered(contiguous)


class CacheAwareScheduler(BaseBatchScheduler):
    """Group cache-compatible work while honoring urgent and starved requests."""

    _RESIDENCY_WEIGHTS = {
        "medium": 2.0,
        "decoder": 8.0,
        "geometry": 20.0,
        "wavelet": 4.0,
    }

    def _forced_prefix(
        self,
        pending: Sequence[InferenceRequest],
        now_ns: int,
    ) -> List[InferenceRequest]:
        urgent = [
            request
            for request in pending
            if request.deadline_slack_us(now_ns) <= self.config.deadline_guard_us
        ]
        if urgent:
            return sorted(
                urgent,
                key=lambda request: (
                    request.deadline_ns if request.deadline_ns is not None else 2**63,
                    request.sequence_id,
                ),
            )
        starved = [
            request
            for request in pending
            if request.age_us(now_ns) >= self.config.starvation_timeout_us
        ]
        if starved:
            return sorted(starved, key=lambda request: request.sequence_id)
        return []

    def _rank_group(
        self,
        group: Sequence[InferenceRequest],
        now_ns: int,
        *,
        anchor: Optional[InferenceRequest] = None,
    ) -> List[InferenceRequest]:
        geometry_counts = Counter(request.geometry_key for request in group)
        frequency_counts = Counter(request.frequency_key for request in group)

        def rank(request: InferenceRequest):
            residency = self._cache_probe(request)
            resident_score = sum(
                weight for name, weight in self._RESIDENCY_WEIGHTS.items() if residency.get(name)
            )
            locality_score = (
                6.0 * (geometry_counts[request.geometry_key] - 1)
                + 1.5 * (frequency_counts[request.frequency_key] - 1)
            )
            age_score = min(request.age_us(now_ns) / self.config.starvation_timeout_us, 1.0)
            return (
                -(resident_score + locality_score + request.priority * 2.0 + age_score),
                request.sequence_id,
            )

        ranked = sorted(group, key=rank)
        if anchor is not None:
            ranked = [anchor] + [request for request in ranked if request is not anchor]
        return ranked

    def _group_score(
        self,
        selected: Sequence[InferenceRequest],
        now_ns: int,
    ) -> tuple:
        geometry_dedup = len(selected) - len({request.geometry_key for request in selected})
        frequency_dedup = len(selected) - len({request.frequency_key for request in selected})
        resident_score = 0.0
        for request in selected:
            residency = self._cache_probe(request)
            resident_score += sum(
                weight for name, weight in self._RESIDENCY_WEIGHTS.items() if residency.get(name)
            )
        oldest_age = max((request.age_us(now_ns) for request in selected), default=0.0)
        highest_priority = max((request.priority for request in selected), default=0)
        return (
            len(selected),
            geometry_dedup,
            frequency_dedup,
            resident_score,
            highest_priority,
            oldest_age,
            -min(request.sequence_id for request in selected),
        )

    def select_batch(
        self,
        pending: Sequence[InferenceRequest],
        *,
        now_ns: Optional[int] = None,
    ) -> List[InferenceRequest]:
        if not pending:
            return []
        current_ns = perf_counter_ns() if now_ns is None else now_ns
        forced = self._forced_prefix(pending, current_ns)
        if forced:
            anchor = forced[0]
            compatible = [request for request in pending if request.batch_key == anchor.batch_key]
            forced_ids = {request.request_id for request in forced}
            prefix = [request for request in forced if request.batch_key == anchor.batch_key]
            remainder = [request for request in compatible if request.request_id not in forced_ids]
            return self._admit_ordered(prefix + self._rank_group(remainder, current_ns))

        groups: Dict[object, List[InferenceRequest]] = defaultdict(list)
        for request in pending:
            groups[request.batch_key].append(request)
        candidates = []
        for group in groups.values():
            ranked = self._rank_group(group, current_ns)
            selected = self._admit_ordered(ranked)
            if selected:
                candidates.append(selected)
        return max(candidates, key=lambda batch: self._group_score(batch, current_ns))


def make_scheduler(
    config: DynamicBatchConfig,
    *,
    cache_probe: Optional[CacheProbe] = None,
) -> BaseBatchScheduler:
    if config.policy is SchedulingPolicy.FCFS:
        return FCFSScheduler(config, cache_probe=cache_probe)
    if config.policy is SchedulingPolicy.CACHE_AWARE:
        return CacheAwareScheduler(config, cache_probe=cache_probe)
    raise ValueError(f"unsupported scheduling policy: {config.policy}")
