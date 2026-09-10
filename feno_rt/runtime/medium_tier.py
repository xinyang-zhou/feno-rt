"""Lease-safe GPU and pinned-CPU tiers for materialized medium latents.

The public handle in this module never owns a tensor.  A worker-local manager
owns every materialized representation and may therefore demote or discard it
without leaving a hidden CUDA reference in the HTTP registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log2
from threading import RLock
from time import perf_counter_ns
from typing import Any, Dict, List, Optional

import torch

from .context_cache import MediumCacheKey, value_nbytes
from .model_runner import FENOModelRunner, MediumContext


_MIB = 1024 * 1024


class MediumTierError(RuntimeError):
    """Base error for worker-local medium tier operations."""


class UnknownMediumHandleError(MediumTierError):
    """Raised when a handle is used after it has been released."""


@dataclass(frozen=True)
class MediumTierConfig:
    """Capacity and cost policy for worker-local materialized medium state."""

    gpu_capacity_bytes: int = 64 * _MIB
    pinned_cpu_capacity_bytes: int = 128 * _MIB
    pin_cpu_memory: bool = True
    rebuild_ms_per_mib: float = 15.16
    promote_ms_per_mib: float = 0.25
    recency_half_life_accesses: float = 64.0

    def __post_init__(self) -> None:
        if self.gpu_capacity_bytes <= 0:
            raise ValueError("gpu_capacity_bytes must be positive")
        if self.pinned_cpu_capacity_bytes < 0:
            raise ValueError("pinned_cpu_capacity_bytes must be non-negative")
        if self.rebuild_ms_per_mib <= 0 or self.promote_ms_per_mib <= 0:
            raise ValueError("tier cost estimates must be positive")
        if self.recency_half_life_accesses <= 0:
            raise ValueError("recency_half_life_accesses must be positive")


@dataclass(frozen=True)
class MediumTierHandle:
    """Stable tensor-free token exported by a worker."""

    medium_id: str
    cache_key: MediumCacheKey


@dataclass
class _TierEntry:
    handle: MediumTierHandle
    velocity_cpu: torch.Tensor
    already_normalized: bool
    gpu_context: Optional[MediumContext] = None
    cpu_latent: Optional[torch.Tensor] = None
    gpu_bytes: int = 0
    cpu_bytes: int = 0
    leases: int = 0
    accesses: int = 0
    last_access: int = 0
    rebuild_cost_ms: float = 0.0
    promote_cost_ms: float = 0.0


class MediumTierLease:
    """Pins one GPU representation until the queued inference is resolved."""

    def __init__(
        self,
        manager: "MediumTierManager",
        key: MediumCacheKey,
        context: MediumContext,
    ) -> None:
        self._manager = manager
        self.key = key
        self.value = context
        self._released = False

    @property
    def released(self) -> bool:
        return self._released

    def release(self) -> None:
        if self._released:
            return
        self._manager.release(self.key)
        self._released = True

    def __enter__(self) -> MediumContext:
        return self.value

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.release()


class MediumTierManager:
    """Own exclusive GPU/CPU representations behind stable medium handles.

    Capacity is enforced after every operation.  If all possible GPU victims
    are leased, a new entry may temporarily exceed capacity; it is trimmed as
    soon as a lease is released.  A leased entry is never demoted, including
    during a forced cache flush.
    """

    def __init__(
        self,
        runner: FENOModelRunner,
        config: Optional[MediumTierConfig] = None,
    ) -> None:
        self.runner = runner
        self.config = config or MediumTierConfig()
        self._entries: Dict[MediumCacheKey, _TierEntry] = {}
        self._clock = 0
        self._gpu_bytes = 0
        self._cpu_bytes = 0
        self._lock = RLock()
        self._gpu_hits = 0
        self._pinned_hits = 0
        self._source_rebuilds = 0
        self._promotions = 0
        self._demotions = 0
        self._pinned_evictions = 0
        self._capacity_overflows = 0
        self._invalidations = 0
        self._h2d_bytes = 0
        self._d2h_bytes = 0
        self._h2d_time_ns = 0
        self._d2h_time_ns = 0

    def register(
        self,
        velocity: Any,
        *,
        medium_id: str,
        already_normalized: bool = False,
    ) -> MediumTierHandle:
        velocity_cpu = (
            torch.as_tensor(velocity).detach().to(device="cpu").contiguous().clone()
        )
        key = self.runner.medium_cache_key(
            velocity_cpu, already_normalized=already_normalized
        )
        handle = MediumTierHandle(str(medium_id), key)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._entries[key] = _TierEntry(
                    handle=handle,
                    velocity_cpu=velocity_cpu,
                    already_normalized=bool(already_normalized),
                )
            else:
                entry.handle = handle
        return handle

    def prepare(
        self,
        velocity: Any,
        *,
        medium_id: str,
        already_normalized: bool = False,
    ) -> MediumTierHandle:
        handle = self.register(
            velocity,
            medium_id=medium_id,
            already_normalized=already_normalized,
        )
        lease = self.acquire(handle)
        lease.release()
        return handle

    def _entry(self, key: MediumCacheKey) -> _TierEntry:
        entry = self._entries.get(key)
        if entry is None:
            raise UnknownMediumHandleError(
                "medium handle is not registered on this worker"
            )
        return entry

    def _touch(self, entry: _TierEntry) -> None:
        self._clock += 1
        entry.last_access = self._clock
        entry.accesses += 1

    def _retention_score(self, entry: _TierEntry, *, cpu: bool = False) -> float:
        age = max(0, self._clock - entry.last_access)
        recency = 0.5 ** (age / self.config.recency_half_life_accesses)
        size_bytes = entry.cpu_bytes if cpu else entry.gpu_bytes
        size_mib = max(size_bytes / _MIB, 1.0 / _MIB)
        rebuild = max(
            entry.rebuild_cost_ms,
            size_mib * self.config.rebuild_ms_per_mib,
        )
        promote = max(
            entry.promote_cost_ms,
            size_mib * self.config.promote_ms_per_mib,
        )
        recovery_cost = rebuild if cpu else max(rebuild - promote, promote)
        frequency = 1.0 + log2(1.0 + entry.accesses)
        return recovery_cost * frequency * recency / size_mib

    def _victim(
        self,
        *,
        cpu: bool,
        exclude: Optional[MediumCacheKey] = None,
    ) -> Optional[_TierEntry]:
        candidates = [
            entry
            for key, entry in self._entries.items()
            if key != exclude
            and entry.leases == 0
            and (entry.cpu_latent is not None if cpu else entry.gpu_context is not None)
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda entry: (
                self._retention_score(entry, cpu=cpu),
                entry.last_access,
                entry.handle.cache_key.velocity_digest,
            ),
        )

    def _drop_cpu(self, entry: _TierEntry) -> None:
        if entry.cpu_latent is None:
            return
        self._cpu_bytes -= entry.cpu_bytes
        entry.cpu_latent = None
        entry.cpu_bytes = 0
        self._pinned_evictions += 1

    def _make_cpu_copy(self, entry: _TierEntry) -> None:
        if entry.cpu_latent is not None or entry.gpu_context is None:
            return
        latent = entry.gpu_context.latent.detach()
        size_bytes = value_nbytes(latent)
        if size_bytes > self.config.pinned_cpu_capacity_bytes:
            return
        while self._cpu_bytes + size_bytes > self.config.pinned_cpu_capacity_bytes:
            victim = self._victim(cpu=True, exclude=entry.handle.cache_key)
            if victim is None:
                return
            self._drop_cpu(victim)
        started_ns = perf_counter_ns()
        pin_memory = (
            self.config.pin_cpu_memory
            and latent.device.type == "cuda"
            and torch.cuda.is_available()
        )
        cpu_latent = torch.empty(
            tuple(latent.shape),
            dtype=latent.dtype,
            device="cpu",
            pin_memory=pin_memory,
        )
        cpu_latent.copy_(latent, non_blocking=pin_memory)
        entry.cpu_latent = cpu_latent
        entry.cpu_bytes = size_bytes
        self._cpu_bytes += size_bytes
        self._d2h_bytes += size_bytes
        self._d2h_time_ns += perf_counter_ns() - started_ns

    def _demote(self, entry: _TierEntry) -> bool:
        if entry.gpu_context is None or entry.leases:
            return False
        self._make_cpu_copy(entry)
        self.runner.caches.invalidate_medium(entry.handle.cache_key, force=False)
        self._gpu_bytes -= entry.gpu_bytes
        entry.gpu_context = None
        entry.gpu_bytes = 0
        self._demotions += 1
        return True

    def _ensure_gpu_room(
        self,
        incoming_bytes: int,
        *,
        exclude: MediumCacheKey,
    ) -> bool:
        while self._gpu_bytes + incoming_bytes > self.config.gpu_capacity_bytes:
            victim = self._victim(cpu=False, exclude=exclude)
            if victim is None:
                return False
            self._demote(victim)
        return True

    def _trim_gpu(self) -> None:
        while self._gpu_bytes > self.config.gpu_capacity_bytes:
            victim = self._victim(cpu=False)
            if victim is None:
                return
            self._demote(victim)

    def acquire(self, handle: MediumTierHandle) -> MediumTierLease:
        with self._lock:
            entry = self._entry(handle.cache_key)
            self._touch(entry)
            if entry.gpu_context is not None:
                self._gpu_hits += 1
                context = entry.gpu_context
            elif entry.cpu_latent is not None:
                self._pinned_hits += 1
                cpu_latent = entry.cpu_latent
                size_bytes = entry.cpu_bytes
                entry.cpu_latent = None
                entry.cpu_bytes = 0
                self._cpu_bytes -= size_bytes
                self._ensure_gpu_room(size_bytes, exclude=handle.cache_key)
                started_ns = perf_counter_ns()
                latent = cpu_latent.to(
                    device=self.runner.device,
                    non_blocking=bool(cpu_latent.is_pinned()),
                )
                entry.promote_cost_ms = (
                    0.8 * entry.promote_cost_ms
                    + 0.2 * ((perf_counter_ns() - started_ns) / 1.0e6)
                    if entry.promote_cost_ms
                    else (perf_counter_ns() - started_ns) / 1.0e6
                )
                self._h2d_time_ns += perf_counter_ns() - started_ns
                self._h2d_bytes += size_bytes
                context = MediumContext(
                    medium_id=handle.medium_id,
                    latent=latent,
                    cache_key=handle.cache_key,
                )
                entry.gpu_context = context
                entry.gpu_bytes = size_bytes
                self._gpu_bytes += size_bytes
                self._promotions += 1
            else:
                self._source_rebuilds += 1
                started_ns = perf_counter_ns()
                context = self.runner.prepare_medium(
                    entry.velocity_cpu,
                    medium_id=handle.medium_id,
                    already_normalized=entry.already_normalized,
                    use_cache=False,
                )
                elapsed_ms = (perf_counter_ns() - started_ns) / 1.0e6
                entry.rebuild_cost_ms = (
                    0.8 * entry.rebuild_cost_ms + 0.2 * elapsed_ms
                    if entry.rebuild_cost_ms
                    else elapsed_ms
                )
                size_bytes = value_nbytes(context)
                if size_bytes > self.config.gpu_capacity_bytes:
                    raise MediumTierError(
                        f"medium latent requires {size_bytes} bytes, exceeding "
                        f"the GPU tier capacity {self.config.gpu_capacity_bytes}"
                    )
                self._ensure_gpu_room(size_bytes, exclude=handle.cache_key)
                entry.gpu_context = context
                entry.gpu_bytes = size_bytes
                self._gpu_bytes += size_bytes

            if self._gpu_bytes > self.config.gpu_capacity_bytes:
                self._capacity_overflows += 1
            entry.leases += 1
            if context.medium_id != handle.medium_id:
                context = MediumContext(
                    medium_id=handle.medium_id,
                    latent=context.latent,
                    cache_key=context.cache_key,
                )
            return MediumTierLease(self, handle.cache_key, context)

    def release(self, key: MediumCacheKey) -> None:
        with self._lock:
            entry = self._entry(key)
            if entry.leases <= 0:
                raise RuntimeError("medium tier entry is not leased")
            entry.leases -= 1
            self._trim_gpu()

    def residency(self, handle: MediumTierHandle) -> Dict[str, bool]:
        with self._lock:
            entry = self._entries.get(handle.cache_key)
            return {
                "registered": entry is not None,
                "gpu": entry is not None and entry.gpu_context is not None,
                "pinned_cpu": entry is not None and entry.cpu_latent is not None,
                "leased": entry is not None and entry.leases > 0,
            }

    def invalidate(
        self,
        handle: MediumTierHandle,
        *,
        force: bool = False,
    ) -> Dict[str, int]:
        del force
        with self._lock:
            entry = self._entries.get(handle.cache_key)
            if entry is None or entry.leases:
                return {
                    "gpu_medium": 0,
                    "pinned_medium": 0,
                    "source_registration": 0,
                }
            gpu_count = int(entry.gpu_context is not None)
            cpu_count = int(entry.cpu_latent is not None)
            dependent = self.runner.caches.invalidate_medium(
                handle.cache_key, force=False
            )
            if entry.gpu_context is not None:
                self._gpu_bytes -= entry.gpu_bytes
            if entry.cpu_latent is not None:
                self._cpu_bytes -= entry.cpu_bytes
            del self._entries[handle.cache_key]
            self._invalidations += 1
            return {
                **dependent,
                "gpu_medium": gpu_count,
                "pinned_medium": cpu_count,
                "source_registration": 1,
            }

    def evict(
        self,
        handle: MediumTierHandle,
        *,
        force: bool = False,
    ) -> Dict[str, int]:
        """Drop materialized state while keeping the source registration."""
        del force
        with self._lock:
            entry = self._entries.get(handle.cache_key)
            if entry is None or entry.leases:
                return {
                    "gpu_medium": 0,
                    "pinned_medium": 0,
                    "source_registration": 0,
                }
            gpu_count = int(entry.gpu_context is not None)
            cpu_count = int(entry.cpu_latent is not None)
            dependent = self.runner.caches.invalidate_medium(
                handle.cache_key, force=False
            )
            if entry.gpu_context is not None:
                self._gpu_bytes -= entry.gpu_bytes
                entry.gpu_context = None
                entry.gpu_bytes = 0
            if entry.cpu_latent is not None:
                self._cpu_bytes -= entry.cpu_bytes
                entry.cpu_latent = None
                entry.cpu_bytes = 0
            return {
                **dependent,
                "gpu_medium": gpu_count,
                "pinned_medium": cpu_count,
                "source_registration": 0,
            }

    def clear_materialized(self, *, force: bool = False) -> Dict[str, int]:
        del force
        with self._lock:
            gpu_count = 0
            cpu_count = 0
            for entry in self._entries.values():
                if entry.leases:
                    continue
                if entry.gpu_context is not None:
                    self.runner.caches.invalidate_medium(
                        entry.handle.cache_key, force=False
                    )
                    self._gpu_bytes -= entry.gpu_bytes
                    entry.gpu_context = None
                    entry.gpu_bytes = 0
                    gpu_count += 1
                if entry.cpu_latent is not None:
                    self._cpu_bytes -= entry.cpu_bytes
                    entry.cpu_latent = None
                    entry.cpu_bytes = 0
                    cpu_count += 1
            return {
                "gpu_medium": gpu_count,
                "pinned_medium": cpu_count,
                "source_registration": 0,
            }

    def gpu_medium_ids(self) -> List[str]:
        with self._lock:
            return sorted(
                entry.handle.medium_id
                for entry in self._entries.values()
                if entry.gpu_context is not None
            )

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            entries = [
                {
                    "medium_id": entry.handle.medium_id,
                    "velocity_digest": entry.handle.cache_key.velocity_digest,
                    "gpu": entry.gpu_context is not None,
                    "pinned_cpu": entry.cpu_latent is not None,
                    "leases": entry.leases,
                    "accesses": entry.accesses,
                    "retention_score": self._retention_score(
                        entry, cpu=entry.gpu_context is None
                    ),
                }
                for entry in self._entries.values()
            ]
            return {
                "policy": "cost_frequency_recency_per_byte",
                "registered_entries": len(self._entries),
                "gpu": {
                    "capacity_bytes": self.config.gpu_capacity_bytes,
                    "resident_bytes": self._gpu_bytes,
                    "resident_entries": sum(item["gpu"] for item in entries),
                    "leased_entries": sum(
                        item["gpu"] and item["leases"] > 0 for item in entries
                    ),
                    "hits": self._gpu_hits,
                    "capacity_overflows": self._capacity_overflows,
                },
                "pinned_cpu": {
                    "capacity_bytes": self.config.pinned_cpu_capacity_bytes,
                    "resident_bytes": self._cpu_bytes,
                    "resident_entries": sum(item["pinned_cpu"] for item in entries),
                    "actual_pinned_entries": sum(
                        entry.cpu_latent is not None
                        and entry.cpu_latent.is_pinned()
                        for entry in self._entries.values()
                    ),
                    "hits": self._pinned_hits,
                    "evictions": self._pinned_evictions,
                    "pin_memory": self.config.pin_cpu_memory,
                },
                "source": {
                    "registered_entries": len(self._entries),
                    "rebuilds": self._source_rebuilds,
                },
                "transfers": {
                    "promotions": self._promotions,
                    "demotions": self._demotions,
                    "h2d_bytes": self._h2d_bytes,
                    "d2h_bytes": self._d2h_bytes,
                    "h2d_time_ms": self._h2d_time_ns / 1.0e6,
                    "d2h_time_ms": self._d2h_time_ns / 1.0e6,
                },
                "invalidations": self._invalidations,
                "entries": entries,
            }
