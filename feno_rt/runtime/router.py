"""Replica routing policies for the Stage 5 multi-GPU runtime."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import Any, Dict, Mapping, Sequence


class ReplicaRoutingPolicy(str, Enum):
    ROUND_ROBIN = "round_robin"
    CACHE_AWARE = "cache_aware"


@dataclass(frozen=True)
class ReplicaRouterConfig:
    """Weights used to balance locality, queue pressure, and memory headroom."""

    medium_replica_weight: float = 24.0
    medium_cache_weight: float = 8.0
    decoder_cache_weight: float = 20.0
    geometry_cache_weight: float = 64.0
    wavelet_cache_weight: float = 8.0
    affinity_weight: float = 256.0
    queue_penalty: float = 6.0
    batch_boundary_penalty: float = 128.0
    memory_headroom_weight: float = 8.0
    missing_replica_penalty: float = 96.0
    minimum_free_memory_ratio: float = 0.03

    def __post_init__(self) -> None:
        non_negative = (
            "medium_replica_weight",
            "medium_cache_weight",
            "decoder_cache_weight",
            "geometry_cache_weight",
            "wavelet_cache_weight",
            "affinity_weight",
            "queue_penalty",
            "batch_boundary_penalty",
            "memory_headroom_weight",
            "missing_replica_penalty",
        )
        for name in non_negative:
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if not 0.0 <= self.minimum_free_memory_ratio < 1.0:
            raise ValueError("minimum_free_memory_ratio must be in [0, 1)")


@dataclass(frozen=True)
class WorkerRouteState:
    worker_id: str
    device: str
    healthy: bool
    queue_depth: int
    free_memory_bytes: int
    total_memory_bytes: int
    has_medium_replica: bool
    cache_residency: Mapping[str, bool]
    batch_capacity: int = 0
    affinity_match: bool = False

    @property
    def free_memory_ratio(self) -> float:
        if self.total_memory_bytes <= 0:
            return 0.0
        return max(
            0.0,
            min(1.0, self.free_memory_bytes / self.total_memory_bytes),
        )


@dataclass(frozen=True)
class RouteDecision:
    worker_id: str
    policy: ReplicaRoutingPolicy
    score: float
    locality_score: float
    queue_penalty: float
    memory_score: float
    missing_replica_penalty: float
    cache_residency: Mapping[str, bool]

    @property
    def cache_hits(self) -> int:
        return sum(bool(value) for value in self.cache_residency.values())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "policy": self.policy.value,
            "score": self.score,
            "locality_score": self.locality_score,
            "queue_penalty": self.queue_penalty,
            "memory_score": self.memory_score,
            "missing_replica_penalty": self.missing_replica_penalty,
            "cache_residency": dict(self.cache_residency),
        }


class BaseReplicaRouter:
    policy: ReplicaRoutingPolicy

    def select(self, states: Sequence[WorkerRouteState]) -> RouteDecision:
        raise NotImplementedError

    @staticmethod
    def _eligible(states: Sequence[WorkerRouteState]) -> Sequence[WorkerRouteState]:
        healthy = [state for state in states if state.healthy]
        if not healthy:
            raise RuntimeError("no healthy GPU worker is available")
        return healthy


class RoundRobinRouter(BaseReplicaRouter):
    """Deterministic baseline that intentionally ignores cache locality."""

    policy = ReplicaRoutingPolicy.ROUND_ROBIN

    def __init__(self) -> None:
        self._cursor = 0
        self._lock = Lock()

    def select(self, states: Sequence[WorkerRouteState]) -> RouteDecision:
        eligible = sorted(self._eligible(states), key=lambda state: state.worker_id)
        with self._lock:
            state = eligible[self._cursor % len(eligible)]
            self._cursor += 1
        return RouteDecision(
            worker_id=state.worker_id,
            policy=self.policy,
            score=0.0,
            locality_score=0.0,
            queue_penalty=0.0,
            memory_score=0.0,
            missing_replica_penalty=0.0,
            cache_residency=dict(state.cache_residency),
        )


class CacheAwareReplicaRouter(BaseReplicaRouter):
    """Jointly score cache locality, outstanding work, and free GPU memory."""

    policy = ReplicaRoutingPolicy.CACHE_AWARE

    def __init__(self, config: ReplicaRouterConfig = ReplicaRouterConfig()) -> None:
        self.config = config

    def _score(self, state: WorkerRouteState) -> RouteDecision:
        residency = state.cache_residency
        locality = (
            (self.config.medium_replica_weight if state.has_medium_replica else 0.0)
            + (self.config.medium_cache_weight if residency.get("medium") else 0.0)
            + (self.config.decoder_cache_weight if residency.get("decoder") else 0.0)
            + (self.config.geometry_cache_weight if residency.get("geometry") else 0.0)
            + (self.config.wavelet_cache_weight if residency.get("wavelet") else 0.0)
            + (self.config.affinity_weight if state.affinity_match else 0.0)
        )
        queue_depth = max(0, state.queue_depth)
        crossed_boundaries = (
            queue_depth // state.batch_capacity if state.batch_capacity > 0 else 0
        )
        queue_cost = (
            self.config.queue_penalty * queue_depth
            + self.config.batch_boundary_penalty * crossed_boundaries
        )
        memory_score = self.config.memory_headroom_weight * state.free_memory_ratio
        missing_cost = (
            0.0 if state.has_medium_replica else self.config.missing_replica_penalty
        )
        return RouteDecision(
            worker_id=state.worker_id,
            policy=self.policy,
            score=locality + memory_score - queue_cost - missing_cost,
            locality_score=locality,
            queue_penalty=queue_cost,
            memory_score=memory_score,
            missing_replica_penalty=missing_cost,
            cache_residency=dict(residency),
        )

    def select(self, states: Sequence[WorkerRouteState]) -> RouteDecision:
        healthy = self._eligible(states)
        with_headroom = [
            state
            for state in healthy
            if state.free_memory_ratio >= self.config.minimum_free_memory_ratio
        ]
        eligible = with_headroom or list(healthy)
        decisions = [(self._score(state), state) for state in eligible]
        decision, _ = max(
            decisions,
            key=lambda item: (
                item[0].score,
                -item[1].queue_depth,
                item[1].free_memory_ratio,
                item[1].worker_id,
            ),
        )
        return decision


def make_replica_router(
    policy: ReplicaRoutingPolicy,
    *,
    config: ReplicaRouterConfig = ReplicaRouterConfig(),
) -> BaseReplicaRouter:
    if isinstance(policy, str):
        policy = ReplicaRoutingPolicy(policy)
    if policy is ReplicaRoutingPolicy.ROUND_ROBIN:
        return RoundRobinRouter()
    if policy is ReplicaRoutingPolicy.CACHE_AWARE:
        return CacheAwareReplicaRouter(config)
    raise ValueError(f"unsupported replica routing policy: {policy}")
