"""Multi-GPU worker and routing orchestration for FENO inference."""

from __future__ import annotations

import asyncio
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from functools import partial
from hashlib import sha256
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union
from uuid import uuid4

import torch

from feno_rt.config import DEFAULT_INFERENCE_CONFIG, FENOModelConfig
from feno_rt.preprocessing import NormalizationStats

from .context_cache import FENOCacheConfig
from .engine import AsyncFENOEngine, DynamicBatchConfig
from .medium_tier import (
    MediumTierConfig,
    MediumTierHandle,
    MediumTierLease,
    MediumTierManager,
)
from .model_runner import FENOModelRunner, MediumContext
from .precision import PrecisionMode
from .request import RequestHandle, RequestState
from .router import (
    ReplicaRoutingPolicy,
    ReplicaRouterConfig,
    RouteDecision,
    WorkerRouteState,
    make_replica_router,
)


def _now_s() -> float:
    return perf_counter_ns() / 1.0e9


def _parse_cpu_list(value: str) -> Set[int]:
    cpus: Set[int] = set()
    for part in value.strip().split(","):
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            cpus.update(range(int(start), int(end) + 1))
        else:
            cpus.add(int(part))
    return cpus


def _cuda_local_cpus(device: torch.device) -> Tuple[int, ...]:
    index = device.index if device.index is not None else torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    pci_path = Path(
        "/sys/bus/pci/devices/"
        f"{properties.pci_domain_id:04x}:{properties.pci_bus_id:02x}:"
        f"{properties.pci_device_id:02x}.0/local_cpulist"
    )
    try:
        local = _parse_cpu_list(pci_path.read_text(encoding="utf-8"))
        allowed = set(os.sched_getaffinity(0))
    except (AttributeError, OSError, ValueError):
        return ()
    return tuple(sorted(local.intersection(allowed)))


def _pin_current_thread(cpus: Tuple[int, ...]) -> None:
    if not cpus:
        return
    try:
        os.sched_setaffinity(0, set(cpus))
    except (AttributeError, OSError):
        pass


@dataclass
class MultiGPUMediumContext:
    """Device-tensor-free public handle plus worker-local replica tokens.

    Device tensors are owned exclusively by each worker's MediumTierManager.
    """

    medium_id: str
    velocity_cpu: torch.Tensor
    already_normalized: bool = False
    replicas: Dict[str, Any] = field(default_factory=dict)
    replica_locks: Dict[str, asyncio.Lock] = field(default_factory=dict, repr=False)
    route_key_cache: Dict[str, Dict[Tuple[Any, ...], Dict[str, Any]]] = field(
        default_factory=dict, repr=False
    )
    route_affinity: Dict[Tuple[Any, ...], str] = field(default_factory=dict, repr=False)
    request_count: int = 0
    hot_replication_scheduled: bool = False

    @property
    def worker_ids(self) -> Tuple[str, ...]:
        return tuple(sorted(self.replicas))


class RoutedRequestHandle:
    """Request handle returned by :class:`MultiGPUFENOEngine`."""

    def __init__(
        self,
        base: RequestHandle,
        result_task: "asyncio.Task[torch.Tensor]",
        worker_id: str,
        decision: RouteDecision,
    ) -> None:
        self._base = base
        self._result_task = result_task
        self.worker_id = worker_id
        self.route_decision = decision

    @property
    def request_id(self) -> str:
        return self._base.request_id

    @property
    def state(self) -> RequestState:
        return self._base.state

    @property
    def done(self) -> bool:
        return self._result_task.done()

    def cancel(self) -> bool:
        return self._base.cancel()

    async def result(self) -> torch.Tensor:
        return await self._result_task

    def __await__(self):
        return self.result().__await__()


class GPUWorker:
    """One independent runner, cache, batcher, and stream pair on one GPU."""

    execution_mode = "thread"

    def __init__(
        self,
        worker_id: str,
        runner: FENOModelRunner,
        batch_config: Optional[DynamicBatchConfig] = None,
        async_d2h: bool = True,
        double_buffer: bool = False,
        memory_sample_interval_ms: float = 50.0,
        numa_affinity: bool = True,
        medium_tier_config: Optional[MediumTierConfig] = None,
    ) -> None:
        if runner.device.type != "cuda":
            raise ValueError("GPUWorker requires a CUDA FENOModelRunner")
        self.worker_id = str(worker_id)
        self.runner = runner
        self.device = torch.device(runner.device)
        self.medium_tiers = MediumTierManager(runner, medium_tier_config)
        self.async_d2h = bool(async_d2h)
        if memory_sample_interval_ms < 0:
            raise ValueError("memory_sample_interval_ms must be non-negative")
        self._memory_sample_interval_ns = int(memory_sample_interval_ms * 1.0e6)
        self._memory_sampled_ns = 0
        self._memory_sample = (0, 0)
        self.local_cpu_affinity = (
            _cuda_local_cpus(self.device) if numa_affinity else ()
        )
        executor_initializer = partial(
            _pin_current_thread, self.local_cpu_affinity
        )
        with torch.cuda.device(self.device):
            self.compute_stream = torch.cuda.Stream(device=self.device)
            self.d2h_stream = torch.cuda.Stream(device=self.device)

        config = batch_config or DynamicBatchConfig()
        if self.async_d2h and config.synchronize_device:
            config = replace(config, synchronize_device=False)
        self.engine = AsyncFENOEngine(
            runner,
            config=config,
            double_buffer=double_buffer,
            execution_stream=self.compute_stream,
            executor_initializer=executor_initializer,
            context_acquirer=self.medium_tiers.acquire,
        )
        self._d2h_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"feno-d2h-{self.worker_id}",
            initializer=executor_initializer,
        )
        self._outstanding = 0
        self._control_depth = 0
        self._closed = False
        self._d2h_copies = 0
        self._d2h_bytes = 0
        self._d2h_time_ns = 0

    async def start(self) -> None:
        await self.engine.start()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self.engine.close()
        self._d2h_executor.shutdown(wait=True)

    def reserve(self) -> None:
        self._outstanding += 1

    def release(self) -> None:
        self._outstanding = max(0, self._outstanding - 1)

    @property
    def queue_depth(self) -> int:
        return self._outstanding + self._control_depth

    def _memory_info(self, *, fresh: bool = False) -> Tuple[int, int]:
        now_ns = perf_counter_ns()
        expired = (
            not self._memory_sampled_ns
            or now_ns - self._memory_sampled_ns >= self._memory_sample_interval_ns
        )
        if fresh or expired:
            with torch.cuda.device(self.device):
                free_memory, total_memory = torch.cuda.mem_get_info(self.device)
            self._memory_sample = (int(free_memory), int(total_memory))
            self._memory_sampled_ns = now_ns
        return self._memory_sample

    async def prepare_medium(
        self,
        velocity_cpu: torch.Tensor,
        medium_id: str,
        already_normalized: bool = False,
    ) -> MediumTierHandle:
        self._control_depth += 1
        try:
            return await self.engine.run_serialized(
                self.medium_tiers.prepare,
                velocity_cpu,
                medium_id=medium_id,
                already_normalized=already_normalized,
            )
        finally:
            self._control_depth = max(0, self._control_depth - 1)

    async def invalidate_medium(
        self, context: MediumTierHandle, *, force: bool = False
    ) -> Dict[str, int]:
        return await self.engine.run_serialized(
            self.medium_tiers.invalidate,
            context,
            force=force,
        )

    async def evict_medium(
        self, context: MediumTierHandle, *, force: bool = False
    ) -> Dict[str, int]:
        return await self.engine.run_serialized(
            self.medium_tiers.evict,
            context,
            force=force,
        )

    async def clear_caches(self, *, force: bool = False) -> Dict[str, int]:
        def clear() -> Dict[str, int]:
            tier_counts = self.medium_tiers.clear_materialized(force=force)
            cache_counts = self.runner.clear_caches(force=force)
            return {**cache_counts, **tier_counts}

        return await self.engine.run_serialized(
            clear,
        )

    async def acquire_medium(
        self, handle: MediumTierHandle
    ) -> Tuple[MediumContext, MediumTierLease]:
        lease = await self.engine.run_serialized(
            self.medium_tiers.acquire,
            handle,
        )
        return lease.value, lease

    async def release_medium_lease(self, lease: Optional[MediumTierLease]) -> None:
        if lease is not None and not lease.released:
            await self.engine.run_serialized(lease.release)

    def request_cache_keys(
        self,
        handle: MediumTierHandle,
        source_position: Any,
        frequency: Any,
        *,
        receiver_positions: Optional[Any] = None,
        positions_are_normalized: bool = False,
    ) -> Dict[str, Any]:
        return self.runner.request_cache_keys_for_medium(
            handle.cache_key,
            source_position,
            frequency,
            receiver_positions=receiver_positions,
            positions_are_normalized=positions_are_normalized,
        )

    def route_state(
        self,
        medium_context: Optional[MediumTierHandle],
        source_position: Any,
        frequency: Any,
        *,
        receiver_positions: Optional[Any] = None,
        positions_are_normalized: bool = False,
        cache_keys: Optional[Mapping[str, Any]] = None,
        probe_cache: bool = True,
        probe_memory: bool = True,
        affinity_match: bool = False,
    ) -> WorkerRouteState:
        residency: Mapping[str, bool] = {}
        tier_residency = (
            self.medium_tiers.residency(medium_context)
            if medium_context is not None
            else {
                "registered": False,
                "gpu": False,
                "pinned_cpu": False,
                "leased": False,
            }
        )
        has_replica = bool(tier_residency["registered"])
        if medium_context is not None and probe_cache:
            keys = (
                dict(cache_keys)
                if cache_keys is not None
                else self.request_cache_keys(
                    medium_context,
                    source_position,
                    frequency,
                    receiver_positions=receiver_positions,
                    positions_are_normalized=positions_are_normalized,
                )
            )
            residency = self.runner.cache_residency(keys)
            residency["medium"] = bool(tier_residency["gpu"])
        if probe_memory:
            free_memory, total_memory = self._memory_info()
        else:
            free_memory, total_memory = self._memory_sample or (1, 1)
            if total_memory <= 0:
                free_memory, total_memory = (1, 1)
        return WorkerRouteState(
            worker_id=self.worker_id,
            device=str(self.device),
            healthy=not self._closed,
            queue_depth=self.queue_depth,
            free_memory_bytes=int(free_memory),
            total_memory_bytes=int(total_memory),
            has_medium_replica=has_replica,
            cache_residency=residency,
            batch_capacity=self.engine.config.max_batch_size,
            affinity_match=affinity_match,
        )

    async def submit(self, *args: Any, **kwargs: Any) -> RequestHandle:
        return await self.engine.submit(*args, **kwargs)

    async def resolve(
        self, handle: RequestHandle, return_cpu: bool = True
    ) -> torch.Tensor:
        output = await handle
        if not return_cpu or output.device.type != "cuda":
            return output
        if not self.async_d2h:
            return output.detach().to(device="cpu")

        started_ns = perf_counter_ns()
        cpu_output = torch.empty_like(
            output, device="cpu", pin_memory=True, memory_format=torch.preserve_format
        )
        with torch.cuda.device(self.device), torch.cuda.stream(self.d2h_stream):
            self.d2h_stream.wait_stream(self.compute_stream)
            cpu_output.copy_(output.detach(), non_blocking=True)
            event = torch.cuda.Event()
            event.record(self.d2h_stream)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._d2h_executor, event.synchronize)
        self._d2h_copies += 1
        self._d2h_bytes += cpu_output.numel() * cpu_output.element_size()
        self._d2h_time_ns += perf_counter_ns() - started_ns
        return cpu_output

    def stats(self) -> Dict[str, Any]:
        free_memory = total_memory = 0
        if not self._closed:
            free_memory, total_memory = self._memory_info(fresh=True)
        return {
            "worker_id": self.worker_id,
            "device": str(self.device),
            "execution_mode": self.execution_mode,
            "healthy": not self._closed,
            "queue_depth": self.queue_depth,
            "local_cpu_affinity": list(self.local_cpu_affinity),
            "free_memory_bytes": int(free_memory),
            "total_memory_bytes": int(total_memory),
            "peak_memory": {
                "allocated_bytes": int(torch.cuda.max_memory_allocated(self.device)),
                "reserved_bytes": int(torch.cuda.max_memory_reserved(self.device)),
            },
            "d2h": {
                "async": self.async_d2h,
                "copies": self._d2h_copies,
                "bytes": self._d2h_bytes,
                "time_ms": self._d2h_time_ns / 1.0e6,
            },
            "engine": self.engine.stats(),
            "medium_tiers": self.medium_tiers.snapshot(),
        }


class MultiGPUFENOEngine:
    """Route requests across independent per-GPU FENO workers."""

    def __init__(
        self,
        workers: Sequence[GPUWorker],
        routing_policy: Union[str, ReplicaRoutingPolicy] = ReplicaRoutingPolicy.CACHE_AWARE,
        router_config: Optional[ReplicaRouterConfig] = None,
        hot_medium_threshold: int = 8,
        return_cpu: bool = True,
    ) -> None:
        if not workers:
            raise ValueError("at least one GPU worker is required")
        worker_ids = [worker.worker_id for worker in workers]
        devices = [worker.device for worker in workers]
        if len(set(worker_ids)) != len(worker_ids):
            raise ValueError("worker IDs must be unique")
        if len(set(devices)) != len(devices):
            raise ValueError("worker devices must be unique")
        if hot_medium_threshold < 1:
            raise ValueError("hot_medium_threshold must be at least one")

        self.workers = list(workers)
        self._workers = {worker.worker_id: worker for worker in workers}
        self.router = make_replica_router(
            routing_policy, config=router_config or ReplicaRouterConfig()
        )
        self.routing_policy = ReplicaRoutingPolicy(routing_policy)
        self.hot_medium_threshold = int(hot_medium_threshold)
        self.return_cpu = bool(return_cpu)
        self._route_lock = asyncio.Lock()
        self._replication_tasks: Dict[int, Set["asyncio.Task[Any]"]] = {}
        self._started = False
        self._closed = False
        self._selection_counts: Counter[str] = Counter()
        self._request_ids: Set[str] = set()
        self._submitted = 0
        self._completed = 0
        self._failed = 0
        self._cancelled = 0
        self._locality_selections = 0
        self._selected_cache_hits = 0
        self._selected_cache_probes = 0
        self._route_key_cache_hits = 0
        self._route_key_cache_misses = 0
        self._route_affinity_hits = 0
        self._route_affinity_misses = 0
        self._route_affinity_migrations = 0
        self._route_affinity_fast_hits = 0
        self._routing_time_ns = 0
        self._replications = 0
        self._cache_flushes = 0
        self._latency_ns = 0

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        devices: Sequence[Union[str, torch.device]],
        config: FENOModelConfig = DEFAULT_INFERENCE_CONFIG,
        normalization: Optional[NormalizationStats] = None,
        batch_config: Optional[DynamicBatchConfig] = None,
        precision: Union[str, PrecisionMode] = PrecisionMode.FP32,
        sdpa_backend: str = "auto",
        cache_config: Optional[FENOCacheConfig] = None,
        medium_tier_config: Optional[MediumTierConfig] = None,
        routing_policy: Union[str, ReplicaRoutingPolicy] = ReplicaRoutingPolicy.CACHE_AWARE,
        router_config: Optional[ReplicaRouterConfig] = None,
        hot_medium_threshold: int = 8,
        async_d2h: bool = True,
        double_buffer: bool = False,
        enable_cuda_graphs: bool = False,
        return_cpu: bool = True,
        process_isolation: Optional[bool] = None,
    ) -> "MultiGPUFENOEngine":
        if not devices:
            raise ValueError("devices must not be empty")
        use_processes = (
            len(devices) > 1 and return_cpu
            if process_isolation is None
            else bool(process_isolation)
        )
        if use_processes and not return_cpu:
            raise ValueError(
                "process-isolated workers require CPU result delivery"
            )
        workers: List[Any] = []
        if use_processes:
            from .process_worker import ProcessGPUWorker

            try:
                for index, device_value in enumerate(devices):
                    workers.append(
                        ProcessGPUWorker(
                            worker_id=f"gpu-{index}",
                            checkpoint_path=checkpoint_path,
                            device=device_value,
                            config=config,
                            normalization=normalization,
                            batch_config=batch_config,
                            precision=precision,
                            sdpa_backend=sdpa_backend,
                            cache_config=cache_config,
                            medium_tier_config=medium_tier_config,
                            double_buffer=double_buffer,
                            enable_cuda_graphs=enable_cuda_graphs,
                        )
                    )
            except BaseException:
                for worker in workers:
                    worker.shutdown_now()
                raise
        else:
            for index, device_value in enumerate(devices):
                device = torch.device(device_value)
                runner = FENOModelRunner.from_checkpoint(
                    checkpoint_path,
                    config=config,
                    normalization=normalization,
                    device=device,
                    precision=precision,
                    sdpa_backend=sdpa_backend,
                    cache_config=cache_config,
                )
                if enable_cuda_graphs:
                    runner.enable_cuda_graphs()
                workers.append(
                    GPUWorker(
                        worker_id=f"gpu-{index}",
                        runner=runner,
                        batch_config=batch_config,
                        async_d2h=async_d2h,
                        double_buffer=double_buffer,
                        medium_tier_config=medium_tier_config,
                    )
                )
        return cls(
            workers,
            routing_policy=routing_policy,
            router_config=router_config,
            hot_medium_threshold=hot_medium_threshold,
            return_cpu=return_cpu,
        )

    async def __aenter__(self) -> "MultiGPUFENOEngine":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.close()

    async def start(self) -> None:
        if self._started:
            return
        if self._closed:
            raise RuntimeError("engine is closed")
        await asyncio.gather(*(worker.start() for worker in self.workers))
        self._started = True

    async def close(self) -> None:
        if self._closed:
            return
        tasks = [task for group in self._replication_tasks.values() for task in group]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.gather(*(worker.close() for worker in self.workers))
        self._closed = True

    async def prepare_medium(
        self,
        velocity: torch.Tensor,
        medium_id: Optional[str] = None,
        already_normalized: bool = False,
        replicas: int = 1,
    ) -> MultiGPUMediumContext:
        if replicas < 1 or replicas > len(self.workers):
            raise ValueError("replicas must be between one and the worker count")
        velocity_cpu = torch.as_tensor(velocity).detach().to(device="cpu").contiguous().clone()
        if medium_id is None:
            digest = sha256(velocity_cpu.view(torch.uint8).numpy().tobytes()).hexdigest()
            medium_id = f"medium-{digest[:16]}"
        context = MultiGPUMediumContext(
            medium_id=str(medium_id),
            velocity_cpu=velocity_cpu,
            already_normalized=bool(already_normalized),
            replica_locks={worker.worker_id: asyncio.Lock() for worker in self.workers},
        )
        await asyncio.gather(
            *(
                self._ensure_replica(context, worker.worker_id)
                for worker in self.workers[:replicas]
            )
        )
        return context

    async def _ensure_replica(
        self, context: MultiGPUMediumContext, worker_id: str
    ) -> Any:
        existing = context.replicas.get(worker_id)
        if existing is not None:
            return existing
        lock = context.replica_locks[worker_id]
        async with lock:
            existing = context.replicas.get(worker_id)
            if existing is not None:
                return existing
            worker = self._workers[worker_id]
            replica = await worker.prepare_medium(
                context.velocity_cpu,
                medium_id=context.medium_id,
                already_normalized=context.already_normalized,
            )
            context.replicas[worker_id] = replica
            self._replications += 1
            return replica

    async def replicate_medium(
        self,
        context: MultiGPUMediumContext,
        worker_ids: Optional[Iterable[str]] = None,
    ) -> None:
        target_ids = list(worker_ids) if worker_ids is not None else list(self._workers)
        unknown = set(target_ids).difference(self._workers)
        if unknown:
            raise KeyError(f"unknown worker IDs: {sorted(unknown)}")
        await asyncio.gather(
            *(self._ensure_replica(context, worker_id) for worker_id in target_ids)
        )

    def _schedule_hot_replication(self, context: MultiGPUMediumContext) -> None:
        if context.hot_replication_scheduled or len(context.replicas) == len(self.workers):
            return
        if context.request_count < self.hot_medium_threshold:
            return
        context.hot_replication_scheduled = True
        group = self._replication_tasks.setdefault(id(context), set())
        for worker in self.workers:
            if worker.worker_id not in context.replicas:
                group.add(asyncio.create_task(self._ensure_replica(context, worker.worker_id)))

    async def wait_for_replication(self, context: MultiGPUMediumContext) -> None:
        tasks = list(self._replication_tasks.pop(id(context), set()))
        if tasks:
            await asyncio.gather(*tasks)

    async def release_medium(
        self, context: MultiGPUMediumContext, *, force: bool = False
    ) -> Dict[str, Dict[str, int]]:
        """Remove one medium and all of its dependent device-local cache state."""
        await self.wait_for_replication(context)
        invalidated: Dict[str, Dict[str, int]] = {}
        for worker_id, replica in list(context.replicas.items()):
            worker = self._workers[worker_id]
            invalidated[worker_id] = await worker.invalidate_medium(
                replica,
                force=force,
            )
        context.replicas.clear()
        context.route_key_cache.clear()
        context.route_affinity.clear()
        return invalidated

    async def flush_caches(
        self,
        context: Optional[MultiGPUMediumContext] = None,
        *,
        worker_ids: Optional[Iterable[str]] = None,
        force: bool = False,
    ) -> Dict[str, Dict[str, int]]:
        """Flush all caches or one medium on selected workers.

        Calls are serialized with each worker's inference stream. Callers must
        still prevent new requests for the affected contexts while flushing.
        """
        target_ids = list(worker_ids) if worker_ids is not None else list(self._workers)
        unknown = set(target_ids).difference(self._workers)
        if unknown:
            raise KeyError(f"unknown worker IDs: {sorted(unknown)}")
        if len(set(target_ids)) != len(target_ids):
            raise ValueError("worker_ids must not contain duplicates")

        async def flush_one(worker_id: str) -> Tuple[str, Dict[str, int]]:
            worker = self._workers[worker_id]
            if context is None:
                result = await worker.clear_caches(
                    force=force,
                )
            else:
                replica = context.replicas.get(worker_id)
                if replica is None:
                    result = {}
                else:
                    result = await worker.evict_medium(
                        replica,
                        force=force,
                    )
            return worker_id, result

        pairs = await asyncio.gather(*(flush_one(worker_id) for worker_id in target_ids))
        if context is not None:
            for worker_id in target_ids:
                context.route_key_cache.pop(worker_id, None)
            context.route_affinity = {
                signature: worker_id
                for signature, worker_id in context.route_affinity.items()
                if worker_id not in target_ids
            }
        self._cache_flushes += 1
        return dict(pairs)

    def _route_states(
        self,
        context: MultiGPUMediumContext,
        source_position: Any,
        frequency: Any,
        *,
        receiver_positions: Optional[Any] = None,
        positions_are_normalized: bool = False,
        signature: Optional[Tuple[Any, ...]] = None,
    ) -> List[WorkerRouteState]:
        probe_locality = self.routing_policy is ReplicaRoutingPolicy.CACHE_AWARE
        if probe_locality and signature is None:
            signature = self._request_signature(
                source_position,
                frequency,
                receiver_positions,
                positions_are_normalized,
            )
        states: List[WorkerRouteState] = []
        affinity_worker_id = context.route_affinity.get(signature)
        honor_affinity = True
        if affinity_worker_id is not None:
            owner = self._workers[affinity_worker_id]
            capacity = max(1, owner.engine.config.max_batch_size)
            owner_boundary = owner.queue_depth // capacity
            honor_affinity = not any(
                worker.worker_id in context.replicas
                and worker.queue_depth // max(1, worker.engine.config.max_batch_size)
                < owner_boundary
                for worker in self.workers
                if worker.worker_id != affinity_worker_id
            )
        for worker in self.workers:
            local_context = context.replicas.get(worker.worker_id)
            cache_keys = None
            if probe_locality and local_context is not None:
                worker_cache = context.route_key_cache.setdefault(worker.worker_id, {})
                cache_keys = worker_cache.get(signature)
                if cache_keys is None:
                    cache_keys = worker.request_cache_keys(
                        local_context,
                        source_position,
                        frequency,
                        receiver_positions=receiver_positions,
                        positions_are_normalized=positions_are_normalized,
                    )
                    if len(worker_cache) >= 4_096:
                        worker_cache.pop(next(iter(worker_cache)))
                    worker_cache[signature] = cache_keys
                    self._route_key_cache_misses += 1
                else:
                    self._route_key_cache_hits += 1
            states.append(
                worker.route_state(
                    local_context,
                    source_position,
                    frequency,
                    receiver_positions=receiver_positions,
                    positions_are_normalized=positions_are_normalized,
                    cache_keys=cache_keys,
                    probe_cache=probe_locality,
                    probe_memory=probe_locality,
                    affinity_match=(
                        probe_locality
                        and honor_affinity
                        and affinity_worker_id == worker.worker_id
                    ),
                )
            )
        return states

    @staticmethod
    def _tensor_signature(value: Any) -> str:
        tensor = torch.as_tensor(value).detach().to(device="cpu").contiguous()
        digest = sha256()
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    @classmethod
    def _request_signature(
        cls,
        source_position: Any,
        frequency: Any,
        receiver_positions: Optional[Any],
        positions_are_normalized: bool,
    ) -> Tuple[Any, ...]:
        return (
            cls._tensor_signature(source_position),
            cls._tensor_signature(frequency),
            None
            if receiver_positions is None
            else cls._tensor_signature(receiver_positions),
            bool(positions_are_normalized),
        )

    def _affinity_fast_decision(
        self,
        context: MultiGPUMediumContext,
        signature: Optional[Tuple[Any, ...]],
        source_position: Any,
        frequency: Any,
        receiver_positions: Optional[Any],
        positions_are_normalized: bool,
    ) -> Optional[RouteDecision]:
        if signature is None:
            return None
        worker_id = context.route_affinity.get(signature)
        if worker_id is None or worker_id not in context.replicas:
            return None
        owner = self._workers[worker_id]
        if owner._closed:
            return None
        capacity = max(1, owner.engine.config.max_batch_size)
        owner_boundary = owner.queue_depth // capacity
        other_boundaries = [
            worker.queue_depth // max(1, worker.engine.config.max_batch_size)
            for worker in self.workers
            if worker.worker_id != worker_id
            and worker.worker_id in context.replicas
            and not worker._closed
        ]
        if other_boundaries and owner_boundary > min(other_boundaries):
            return None
        state = owner.route_state(
            context.replicas[worker_id],
            source_position,
            frequency,
            receiver_positions=receiver_positions,
            positions_are_normalized=positions_are_normalized,
            probe_cache=False,
            probe_memory=True,
            affinity_match=True,
        )
        minimum_ratio = getattr(
            getattr(self.router, "config", None), "minimum_free_memory_ratio", 0.0
        )
        if state.free_memory_ratio < minimum_ratio:
            return None
        self._route_affinity_fast_hits += 1
        return self.router.select([state])

    async def submit(
        self,
        medium_context: MultiGPUMediumContext,
        source_position: Any,
        frequency: Any,
        *,
        receiver_positions: Optional[Any] = None,
        positions_are_normalized: bool = False,
        denormalize: bool = False,
        cache_level: str = "all",
        request_id: Optional[str] = None,
        timeout_s: Optional[float] = None,
        deadline_s: Optional[float] = None,
        priority: int = 0,
    ) -> RoutedRequestHandle:
        if self._closed:
            raise RuntimeError("engine is closed")
        if not self._started:
            await self.start()
        if timeout_s is not None and deadline_s is not None:
            raise ValueError("specify timeout_s or deadline_s, not both")
        if timeout_s is not None and timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if priority < 0:
            raise ValueError("priority must be non-negative")
        absolute_deadline = deadline_s
        if timeout_s is not None:
            absolute_deadline = _now_s() + timeout_s
        resolved_request_id = request_id or uuid4().hex

        async with self._route_lock:
            if resolved_request_id in self._request_ids:
                raise ValueError(f"duplicate request_id: {resolved_request_id}")
            routing_started_ns = perf_counter_ns()
            try:
                route_signature = (
                    self._request_signature(
                        source_position,
                        frequency,
                        receiver_positions,
                        positions_are_normalized,
                    )
                    if self.routing_policy is ReplicaRoutingPolicy.CACHE_AWARE
                    else None
                )
                decision = self._affinity_fast_decision(
                    medium_context,
                    route_signature,
                    source_position,
                    frequency,
                    receiver_positions,
                    positions_are_normalized,
                )
                if decision is None:
                    states = self._route_states(
                        medium_context,
                        source_position,
                        frequency,
                        receiver_positions=receiver_positions,
                        positions_are_normalized=positions_are_normalized,
                        signature=route_signature,
                    )
                    decision = self.router.select(states)
                if route_signature is not None:
                    if route_signature in medium_context.route_affinity:
                        self._route_affinity_hits += 1
                        if (
                            medium_context.route_affinity[route_signature]
                            != decision.worker_id
                        ):
                            medium_context.route_affinity[route_signature] = decision.worker_id
                            self._route_affinity_migrations += 1
                    else:
                        medium_context.route_affinity[route_signature] = decision.worker_id
                        self._route_affinity_misses += 1
            finally:
                self._routing_time_ns += perf_counter_ns() - routing_started_ns
            worker = self._workers[decision.worker_id]
            worker.reserve()
            self._request_ids.add(resolved_request_id)
            self._selection_counts[worker.worker_id] += 1
            self._submitted += 1
            non_medium_hits = sum(
                bool(value)
                for name, value in decision.cache_residency.items()
                if name != "medium"
            )
            if non_medium_hits > 0:
                self._locality_selections += 1
            self._selected_cache_hits += decision.cache_hits
            self._selected_cache_probes += len(decision.cache_residency)

        try:
            local_handle = await self._ensure_replica(
                medium_context, worker.worker_id
            )
            base_handle = await worker.submit(
                local_handle,
                source_position,
                frequency,
                receiver_positions=receiver_positions,
                positions_are_normalized=positions_are_normalized,
                denormalize=denormalize,
                cache_level=cache_level,
                request_id=resolved_request_id,
                timeout_s=None,
                deadline_s=absolute_deadline,
                priority=priority,
            )
        except BaseException:
            worker.release()
            self._request_ids.discard(resolved_request_id)
            self._failed += 1
            raise

        medium_context.request_count += 1
        self._schedule_hot_replication(medium_context)
        task = asyncio.create_task(self._resolve(worker, base_handle))
        return RoutedRequestHandle(base_handle, task, worker.worker_id, decision)

    async def _resolve(
        self,
        worker: GPUWorker,
        handle: RequestHandle,
    ) -> torch.Tensor:
        started_ns = perf_counter_ns()
        try:
            output = await worker.resolve(handle, return_cpu=self.return_cpu)
            self._completed += 1
            return output
        except asyncio.CancelledError:
            self._cancelled += 1
            raise
        except BaseException:
            self._failed += 1
            raise
        finally:
            worker.release()
            self._latency_ns += perf_counter_ns() - started_ns

    async def infer(
        self,
        medium_context: MultiGPUMediumContext,
        source_position: Any,
        frequency: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        return await (
            await self.submit(
                medium_context, source_position, frequency, **kwargs
            )
        )

    def stats(self) -> Dict[str, Any]:
        return {
            "routing_policy": self.routing_policy.value,
            "submitted": self._submitted,
            "completed": self._completed,
            "failed": self._failed,
            "cancelled": self._cancelled,
            "mean_completion_latency_ms": (
                self._latency_ns / max(1, self._completed + self._failed + self._cancelled) / 1.0e6
            ),
            "selection_counts": dict(self._selection_counts),
            "cache_locality_selection_rate": self._locality_selections / max(1, self._submitted),
            "selected_cache_entry_hit_rate": (
                self._selected_cache_hits / max(1, self._selected_cache_probes)
            ),
            "routing_mean_us": self._routing_time_ns / max(1, self._submitted) / 1.0e3,
            "route_key_cache": {
                "hits": self._route_key_cache_hits,
                "misses": self._route_key_cache_misses,
                "hit_rate": self._route_key_cache_hits
                / max(1, self._route_key_cache_hits + self._route_key_cache_misses),
            },
            "route_affinity": {
                "hits": self._route_affinity_hits,
                "misses": self._route_affinity_misses,
                "hit_rate": self._route_affinity_hits
                / max(1, self._route_affinity_hits + self._route_affinity_misses),
                "migrations": self._route_affinity_migrations,
                "fast_path_hits": self._route_affinity_fast_hits,
            },
            "medium_replications": self._replications,
            "cache_flushes": self._cache_flushes,
            "hot_medium_threshold": self.hot_medium_threshold,
            "worker_execution_modes": {
                worker.worker_id: getattr(worker, "execution_mode", "thread")
                for worker in self.workers
            },
            "workers": [worker.stats() for worker in self.workers],
        }
