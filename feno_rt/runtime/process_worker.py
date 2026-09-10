"""Spawn-isolated GPU worker used to avoid cross-device CUDA launch contention."""

from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import dataclass, replace
from functools import partial
from hashlib import sha256
from pathlib import Path
from threading import Condition, Lock, RLock
from time import perf_counter_ns
import traceback
from typing import Any, Dict, Mapping, Optional, Set, Tuple, Union

import torch

from feno_rt.config import FENOModelConfig
from feno_rt.preprocessing import (
    NormalizationStats,
    prepare_frequencies,
    prepare_source_receiver_batch,
)

from .context_cache import FENOCacheConfig
from .engine import AsyncFENOEngine
from .medium_tier import (
    MediumTierConfig,
    MediumTierHandle,
    MediumTierManager,
)
from .model_runner import FENOModelRunner
from .request import RequestHandle
from .router import WorkerRouteState
from .scheduler import DynamicBatchConfig


class ProcessWorkerError(RuntimeError):
    """Raised when an isolated GPU worker rejects an RPC or exits."""


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


def _memory_snapshot(device: torch.device) -> Dict[str, int]:
    free_memory, total_memory = torch.cuda.mem_get_info(device)
    return {
        "free_memory_bytes": int(free_memory),
        "total_memory_bytes": int(total_memory),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _worker_process_main(
    connection: Any,
    shared_outputs: torch.Tensor,
    checkpoint_path: str,
    device_value: str,
    config: FENOModelConfig,
    normalization: Optional[NormalizationStats],
    precision: str,
    sdpa_backend: str,
    cache_config: Optional[FENOCacheConfig],
    medium_tier_config: Optional[MediumTierConfig],
    enable_cuda_graphs: bool,
) -> None:
    """Own one CUDA context and execute a serial RPC stream."""

    device = torch.device(device_value)
    contexts: Dict[str, MediumTierHandle] = {}
    d2h_copies = 0
    d2h_bytes = 0
    d2h_time_ns = 0
    overflow_transfers = 0
    shared_output_registered = False
    try:
        torch.set_num_threads(1)
        with torch.cuda.device(device):
            local_cpu_affinity = _cuda_local_cpus(device)
            _pin_current_thread(local_cpu_affinity)
            runner = FENOModelRunner.from_checkpoint(
                checkpoint_path,
                config=config,
                normalization=normalization,
                device=device,
                precision=precision,
                sdpa_backend=sdpa_backend,
                cache_config=cache_config,
            )
            medium_tiers = MediumTierManager(runner, medium_tier_config)
            if enable_cuda_graphs:
                runner.enable_cuda_graphs()
            compute_stream = torch.cuda.Stream(device=device)
            d2h_stream = torch.cuda.Stream(device=device)
            register_status = int(
                torch.cuda.cudart().cudaHostRegister(
                    shared_outputs.data_ptr(),
                    shared_outputs.numel() * shared_outputs.element_size(),
                    1,
                )
            )
            shared_output_registered = register_status == 0
            torch.cuda.reset_peak_memory_stats(device)
        connection.send(
            {
                "ok": True,
                "result": {
                    "gpu_name": torch.cuda.get_device_name(device),
                    "local_cpu_affinity": list(local_cpu_affinity),
                    "shared_output_registered": shared_output_registered,
                    **_memory_snapshot(device),
                },
            }
        )
    except BaseException:
        connection.send({"ok": False, "error": traceback.format_exc()})
        connection.close()
        return

    while True:
        try:
            message = connection.recv()
        except EOFError:
            break
        operation = message.get("op")
        try:
            if operation == "close":
                connection.send({"ok": True, "result": None})
                break
            if operation == "prepare_medium":
                medium_id = str(message["medium_id"])
                with torch.cuda.device(device), torch.cuda.stream(compute_stream):
                    handle = medium_tiers.prepare(
                        message["velocity"],
                        medium_id=medium_id,
                        already_normalized=bool(message["already_normalized"]),
                    )
                    lease = medium_tiers.acquire(handle)
                    context = lease.value
                    runner.prepare_decoder_context(context)
                    # A newly replicated process has not executed the cached
                    # decoder path yet. Pay that one-time CUDA/library setup
                    # while replication is still off the request path, so its
                    # first routed batch is not an artificial cold outlier.
                    center = config.domain_extent / 2.0
                    runner.forward_batch(
                        context,
                        [[center, center]],
                        [10.0],
                        cache_level="all",
                    )
                compute_stream.synchronize()
                lease.release()
                contexts[medium_id] = handle
                connection.send(
                    {
                        "ok": True,
                        "result": {
                            "medium_id": medium_id,
                            "gpu_medium_ids": medium_tiers.gpu_medium_ids(),
                            **_memory_snapshot(device),
                        },
                    }
                )
                continue
            if operation == "forward":
                medium_id = str(message["medium_id"])
                handle = contexts[medium_id]
                with torch.cuda.device(device), torch.cuda.stream(compute_stream):
                    lease = medium_tiers.acquire(handle)
                    context = lease.value
                    output = runner.forward_batch(
                        context,
                        message["sources"],
                        message["frequencies"],
                        receiver_positions=message.get("receivers"),
                        positions_are_normalized=bool(
                            message["positions_are_normalized"]
                        ),
                        denormalize=bool(message["denormalize"]),
                        cache_level=str(message["cache_level"]),
                    )
                started_ns = perf_counter_ns()
                output_numel = output.numel()
                if output_numel > int(shared_outputs.shape[1]):
                    raise RuntimeError(
                        f"predicted output has {output_numel} elements, "
                        f"shared slot has {shared_outputs.shape[1]}"
                    )
                slot_value = message.get("output_slot")
                if slot_value is None:
                    cpu_output = torch.empty_like(
                        output,
                        device="cpu",
                        pin_memory=True,
                        memory_format=torch.preserve_format,
                    )
                    overflow_transfers += 1
                else:
                    slot_index = int(slot_value)
                    cpu_output = shared_outputs[
                        slot_index,
                        :output_numel,
                    ].view(output.shape)
                with torch.cuda.device(device), torch.cuda.stream(d2h_stream):
                    d2h_stream.wait_stream(compute_stream)
                    cpu_output.copy_(output.detach(), non_blocking=True)
                    event = torch.cuda.Event()
                    event.record(d2h_stream)
                event.synchronize()
                lease.release()
                d2h_copies += int(output.shape[0])
                d2h_bytes += output_numel * torch.float32.itemsize
                d2h_time_ns += perf_counter_ns() - started_ns
                connection.send(
                    {
                        "ok": True,
                        "result": {
                            "shape": tuple(output.shape),
                            "numel": output_numel,
                            "output": cpu_output if slot_value is None else None,
                            "gpu_medium_ids": medium_tiers.gpu_medium_ids(),
                        },
                    }
                )
                continue
            if operation == "reset_cache_stats":
                runner.reset_cache_stats()
                connection.send({"ok": True, "result": None})
                continue
            if operation == "reset_peak_memory_stats":
                torch.cuda.reset_peak_memory_stats(device)
                connection.send({"ok": True, "result": None})
                continue
            if operation == "invalidate_medium":
                medium_id = str(message["medium_id"])
                handle = contexts.pop(medium_id, None)
                result = (
                    {}
                    if handle is None
                    else medium_tiers.invalidate(
                        handle,
                        force=bool(message["force"]),
                    )
                )
                connection.send({"ok": True, "result": result})
                continue
            if operation == "evict_medium":
                medium_id = str(message["medium_id"])
                handle = contexts.get(medium_id)
                result = (
                    {}
                    if handle is None
                    else medium_tiers.evict(
                        handle,
                        force=bool(message["force"]),
                    )
                )
                connection.send(
                    {
                        "ok": True,
                        "result": {
                            "counts": result,
                            "gpu_medium_ids": medium_tiers.gpu_medium_ids(),
                        },
                    }
                )
                continue
            if operation == "clear_caches":
                tier_counts = medium_tiers.clear_materialized(
                    force=bool(message["force"])
                )
                result = {
                    **runner.clear_caches(force=bool(message["force"])),
                    **tier_counts,
                }
                connection.send({"ok": True, "result": result})
                continue
            if operation == "stats":
                connection.send(
                    {
                        "ok": True,
                        "result": {
                            **_memory_snapshot(device),
                            "cache": runner.cache_metrics(),
                            "medium_tiers": medium_tiers.snapshot(),
                            "execution": runner.execution_config(),
                            "shared_output_registered": shared_output_registered,
                            "d2h": {
                                "async": True,
                                "copies": d2h_copies,
                                "bytes": d2h_bytes,
                                "time_ms": d2h_time_ns / 1.0e6,
                                "overflow_transfers": overflow_transfers,
                            },
                        },
                    }
                )
                continue
            raise ValueError(f"unknown process-worker operation: {operation!r}")
        except BaseException:
            candidate = locals().get("lease")
            if candidate is not None and not candidate.released:
                try:
                    candidate.release()
                except BaseException:
                    pass
            connection.send({"ok": False, "error": traceback.format_exc()})

    if shared_output_registered:
        torch.cuda.cudart().cudaHostUnregister(shared_outputs.data_ptr())
    connection.close()


@dataclass(frozen=True)
class ProcessMediumContext:
    """Parent-process token referring to a device-local medium context."""

    medium_id: str
    cache_key: Tuple[str, str, str]
    latent: str = "remote"


class _OutputSlotLease:
    def __init__(self, runner: "_ProcessRunnerProxy", slot_index: int) -> None:
        self._runner = runner
        self._slot_index = slot_index
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._runner._release_output_slot(self._slot_index)

    def __del__(self) -> None:
        self.release()


class _ProcessRunnerProxy:
    """Small CPU-side runner facade consumed by :class:`AsyncFENOEngine`."""

    CACHE_LEVELS = FENOModelRunner.CACHE_LEVELS
    forward_output_owned = True

    def __init__(
        self,
        connection: Any,
        process: Any,
        shared_outputs: torch.Tensor,
        worker_id: str,
        config: FENOModelConfig,
    ) -> None:
        self._connection = connection
        self._process = process
        self._shared_outputs = shared_outputs
        self._worker_id = worker_id
        self._rpc_lock = Lock()
        self._cache_lock = RLock()
        self._slot_condition = Condition()
        self._available_slots = list(range(int(shared_outputs.shape[0])))
        self._resident: Dict[str, Set[Any]] = {
            "medium": set(),
            "decoder": set(),
            "geometry": set(),
            "wavelet": set(),
        }
        self._medium_keys: Dict[str, Any] = {}
        self._cached_metrics: Dict[str, Any] = {}
        self.config = config
        self.dtype = torch.float32
        # The parent executor performs blocking IPC and receives an owned CPU
        # tensor. CUDA synchronization is entirely child-process local.
        self.device = torch.device("cpu")

    def _acquire_output_slot(self) -> Optional[int]:
        with self._slot_condition:
            return self._available_slots.pop() if self._available_slots else None

    def _release_output_slot(self, slot_index: int) -> None:
        with self._slot_condition:
            self._available_slots.append(slot_index)
            self._slot_condition.notify()

    def _rpc(self, operation: str, **payload: Any) -> Any:
        with self._rpc_lock:
            if not self._process.is_alive():
                raise ProcessWorkerError(
                    f"{self._worker_id} process exited with {self._process.exitcode}"
                )
            try:
                self._connection.send({"op": operation, **payload})
                response = self._connection.recv()
            except (BrokenPipeError, EOFError, OSError) as error:
                raise ProcessWorkerError(
                    f"{self._worker_id} process RPC failed"
                ) from error
        if not response.get("ok"):
            raise ProcessWorkerError(response.get("error", "remote worker failed"))
        return response.get("result")

    @staticmethod
    def _tensor_digest(*values: torch.Tensor) -> str:
        digest = sha256()
        for value in values:
            tensor = value.detach().to(device="cpu").contiguous()
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def request_cache_keys(
        self,
        context: ProcessMediumContext,
        source_position: Any,
        frequency: Any,
        *,
        receiver_positions: Optional[Any] = None,
        positions_are_normalized: bool = False,
    ) -> Dict[str, Any]:
        sources, receivers = prepare_source_receiver_batch(
            source_position,
            self.config,
            receiver_positions=receiver_positions,
            positions_are_normalized=positions_are_normalized,
            device=torch.device("cpu"),
            dtype=self.dtype,
        )
        frequencies = prepare_frequencies(
            frequency,
            sources.shape[0],
            device=torch.device("cpu"),
            dtype=self.dtype,
        )
        medium_key = context.cache_key
        decoder_key = ("decoder", medium_key)
        geometry_key = (
            "geometry",
            decoder_key,
            self._tensor_digest(sources[0], receivers[0]),
        )
        effective = (
            frequencies[0]
            .detach()
            .to(device="cpu", dtype=torch.float32)
            .reshape(1)
            .contiguous()
        )
        wavelet_key = ("wavelet", effective.view(torch.uint8).numpy().tobytes().hex())
        return {
            "medium": medium_key,
            "decoder": decoder_key,
            "geometry": geometry_key,
            "wavelet": wavelet_key,
        }

    def cache_residency(self, keys: Mapping[str, Any]) -> Dict[str, bool]:
        with self._cache_lock:
            return {
                name: key in self._resident[name]
                for name, key in keys.items()
                if name in self._resident
            }

    def _sync_gpu_mediums(self, medium_ids: Any) -> None:
        resident_ids = {str(value) for value in medium_ids}
        with self._cache_lock:
            self._resident["medium"] = {
                key
                for medium_id, key in self._medium_keys.items()
                if medium_id in resident_ids
            }
            self._resident["decoder"] = {
                ("decoder", key) for key in self._resident["medium"]
            }

    def prepare_medium(
        self,
        velocity: torch.Tensor,
        *,
        medium_id: str,
        already_normalized: bool = False,
    ) -> ProcessMediumContext:
        result = self._rpc(
            "prepare_medium",
            velocity=torch.as_tensor(velocity).detach().cpu().numpy(),
            medium_id=medium_id,
            already_normalized=already_normalized,
        )
        context = ProcessMediumContext(
            medium_id=result["medium_id"],
            cache_key=("process-medium", self._worker_id, result["medium_id"]),
        )
        self._medium_keys[context.medium_id] = context.cache_key
        self._sync_gpu_mediums(result.get("gpu_medium_ids", (context.medium_id,)))
        with self._cache_lock:
            if context.medium_id in result.get("gpu_medium_ids", ()):
                self._resident["medium"].add(context.cache_key)
                self._resident["decoder"].add(("decoder", context.cache_key))
        return context

    def forward_batch(
        self,
        context: ProcessMediumContext,
        sources: torch.Tensor,
        frequencies: torch.Tensor,
        *,
        receiver_positions: Optional[torch.Tensor] = None,
        positions_are_normalized: bool = False,
        denormalize: bool = False,
        cache_level: str = "all",
    ) -> torch.Tensor:
        output_slot = self._acquire_output_slot()
        try:
            result = self._rpc(
                "forward",
                medium_id=context.medium_id,
                sources=sources.detach().cpu().numpy(),
                frequencies=frequencies.detach().cpu().numpy(),
                receivers=(
                    None
                    if receiver_positions is None
                    else receiver_positions.detach().cpu().numpy()
                ),
                positions_are_normalized=positions_are_normalized,
                denormalize=denormalize,
                cache_level=cache_level,
                output_slot=output_slot,
            )
        except BaseException:
            if output_slot is not None:
                self._release_output_slot(output_slot)
            raise
        self._sync_gpu_mediums(result.get("gpu_medium_ids", (context.medium_id,)))
        shape = tuple(int(value) for value in result["shape"])
        output_numel = int(result["numel"])
        if output_slot is None:
            return result["output"].view(shape)
        if output_numel > int(self._shared_outputs.shape[1]):
            self._release_output_slot(output_slot)
            raise ProcessWorkerError("remote output shape exceeds shared allocation")
        owned_output = self._shared_outputs[
            output_slot,
            :output_numel,
        ].view(shape)
        owned_output._feno_output_lease = _OutputSlotLease(self, output_slot)
        return owned_output

    def reset_cache_stats(self) -> None:
        self._rpc("reset_cache_stats")

    def cache_metrics(self) -> Dict[str, Any]:
        return dict(self._cached_metrics)

    def runtime_stats(self) -> Dict[str, Any]:
        result = self._rpc("stats")
        self._cached_metrics = dict(result.get("cache", {}))
        return result

    def reset_peak_memory_stats(self) -> None:
        self._rpc("reset_peak_memory_stats")

    def invalidate_medium(
        self, context: ProcessMediumContext, *, force: bool = False
    ) -> Dict[str, int]:
        result = self._rpc(
            "invalidate_medium",
            medium_id=context.medium_id,
            force=force,
        )
        with self._cache_lock:
            for resident in self._resident.values():
                resident.clear()
            self._medium_keys.pop(context.medium_id, None)
        return result

    def evict_medium(
        self, context: ProcessMediumContext, *, force: bool = False
    ) -> Dict[str, int]:
        result = self._rpc(
            "evict_medium",
            medium_id=context.medium_id,
            force=force,
        )
        self._sync_gpu_mediums(result.get("gpu_medium_ids", ()))
        return dict(result.get("counts", {}))

    def clear_caches(self, *, force: bool = False) -> Dict[str, int]:
        result = self._rpc("clear_caches", force=force)
        with self._cache_lock:
            for resident in self._resident.values():
                resident.clear()
        return result

    def close(self) -> None:
        try:
            self._rpc("close")
        finally:
            self._connection.close()


class ProcessGPUWorker:
    """GPU worker whose model and CUDA streams live in a spawned process."""

    execution_mode = "process"

    def __init__(
        self,
        worker_id: str,
        checkpoint_path: Union[str, Path],
        device: Union[str, torch.device],
        config: FENOModelConfig,
        normalization: Optional[NormalizationStats],
        batch_config: Optional[DynamicBatchConfig] = None,
        *,
        precision: str = "fp32",
        sdpa_backend: str = "auto",
        cache_config: Optional[FENOCacheConfig] = None,
        medium_tier_config: Optional[MediumTierConfig] = None,
        double_buffer: bool = False,
        enable_cuda_graphs: bool = False,
        startup_timeout_s: float = 120.0,
    ) -> None:
        self.worker_id = str(worker_id)
        self.device = torch.device(device)
        if self.device.type != "cuda" or self.device.index is None:
            raise ValueError("ProcessGPUWorker requires an explicitly indexed CUDA device")
        self.async_d2h = True
        self._outstanding = 0
        self._control_depth = 0
        self._closed = False
        resolved_config = batch_config or DynamicBatchConfig()
        output_elements = (
            resolved_config.max_batch_size
            * config.num_receivers
            * config.output_steps
        )
        self._shared_outputs = torch.empty(
            (2, output_elements),
            dtype=torch.float32,
        ).share_memory_()
        context = mp.get_context("spawn")
        parent_connection, child_connection = context.Pipe(duplex=True)
        self._process = context.Process(
            target=_worker_process_main,
            name=f"feno-{self.worker_id}",
            args=(
                child_connection,
                self._shared_outputs,
                str(checkpoint_path),
                str(self.device),
                config,
                normalization,
                getattr(precision, "value", str(precision)),
                sdpa_backend,
                cache_config,
                medium_tier_config,
                enable_cuda_graphs,
            ),
        )
        self._process.start()
        child_connection.close()
        self._connection = parent_connection
        if not parent_connection.poll(startup_timeout_s):
            self._process.terminate()
            self._process.join(timeout=5)
            raise TimeoutError(f"{self.worker_id} did not start within {startup_timeout_s}s")
        ready = parent_connection.recv()
        if not ready.get("ok"):
            self._process.join(timeout=5)
            raise ProcessWorkerError(ready.get("error", "worker startup failed"))
        metadata = ready["result"]
        self.gpu_name = str(metadata["gpu_name"])
        self.local_cpu_affinity = tuple(metadata["local_cpu_affinity"])
        self._memory_sample = (
            int(metadata["free_memory_bytes"]),
            int(metadata["total_memory_bytes"]),
        )
        self.runner = _ProcessRunnerProxy(
            parent_connection,
            self._process,
            self._shared_outputs,
            self.worker_id,
            config,
        )
        if resolved_config.synchronize_device:
            resolved_config = replace(resolved_config, synchronize_device=False)
        self.engine = AsyncFENOEngine(
            self.runner,
            config=resolved_config,
            double_buffer=double_buffer,
            executor_initializer=partial(
                _pin_current_thread,
                self.local_cpu_affinity,
            ),
        )
        self._last_runtime_stats: Dict[str, Any] = metadata

    @property
    def queue_depth(self) -> int:
        return self._outstanding + self._control_depth

    def reserve(self) -> None:
        self._outstanding += 1

    def release(self) -> None:
        self._outstanding = max(0, self._outstanding - 1)

    async def start(self) -> None:
        await self.engine.start()

    async def close(self) -> None:
        if self._closed:
            return
        await self.engine.close()
        self.shutdown_now()

    def shutdown_now(self) -> None:
        """Close or terminate this worker during synchronous startup cleanup."""

        if self._closed and not self._process.is_alive():
            return
        try:
            self.runner.close()
        except BaseException:
            pass
        self._process.join(timeout=10)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)
        self._closed = True

    async def prepare_medium(
        self,
        velocity_cpu: torch.Tensor,
        medium_id: str,
        already_normalized: bool = False,
    ) -> ProcessMediumContext:
        self._control_depth += 1
        try:
            return await self.engine.run_serialized(
                self.runner.prepare_medium,
                velocity_cpu,
                medium_id=medium_id,
                already_normalized=already_normalized,
            )
        finally:
            self._control_depth = max(0, self._control_depth - 1)

    async def acquire_medium(
        self, context: ProcessMediumContext
    ) -> Tuple[ProcessMediumContext, None]:
        return context, None

    async def release_medium_lease(self, lease: None) -> None:
        del lease

    def request_cache_keys(
        self,
        context: ProcessMediumContext,
        source_position: Any,
        frequency: Any,
        *,
        receiver_positions: Optional[Any] = None,
        positions_are_normalized: bool = False,
    ) -> Dict[str, Any]:
        return self.runner.request_cache_keys(
            context,
            source_position,
            frequency,
            receiver_positions=receiver_positions,
            positions_are_normalized=positions_are_normalized,
        )

    async def invalidate_medium(
        self, context: ProcessMediumContext, *, force: bool = False
    ) -> Dict[str, int]:
        return await self.engine.run_serialized(
            self.runner.invalidate_medium,
            context,
            force=force,
        )

    async def evict_medium(
        self, context: ProcessMediumContext, *, force: bool = False
    ) -> Dict[str, int]:
        return await self.engine.run_serialized(
            self.runner.evict_medium,
            context,
            force=force,
        )

    async def clear_caches(self, *, force: bool = False) -> Dict[str, int]:
        return await self.engine.run_serialized(
            self.runner.clear_caches,
            force=force,
        )

    def route_state(
        self,
        medium_context: Optional[ProcessMediumContext],
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
        del source_position, frequency, receiver_positions, positions_are_normalized
        residency: Mapping[str, bool] = {}
        if medium_context is not None and probe_cache:
            residency = self.runner.cache_residency(dict(cache_keys or {}))
        free_memory, total_memory = self._memory_sample
        return WorkerRouteState(
            worker_id=self.worker_id,
            device=str(self.device),
            healthy=not self._closed and self._process.is_alive(),
            queue_depth=self.queue_depth,
            free_memory_bytes=free_memory,
            total_memory_bytes=total_memory,
            has_medium_replica=medium_context is not None,
            cache_residency=residency,
            batch_capacity=self.engine.config.max_batch_size,
            affinity_match=affinity_match,
        )

    async def submit(self, *args: Any, **kwargs: Any) -> RequestHandle:
        return await self.engine.submit(*args, **kwargs)

    async def resolve(
        self, handle: RequestHandle, return_cpu: bool = True
    ) -> torch.Tensor:
        del return_cpu
        return await handle

    def reset_peak_memory_stats(self) -> None:
        self.runner.reset_peak_memory_stats()

    def stats(self) -> Dict[str, Any]:
        if not self._closed:
            self._last_runtime_stats = self.runner.runtime_stats()
            self._memory_sample = (
                int(self._last_runtime_stats["free_memory_bytes"]),
                int(self._last_runtime_stats["total_memory_bytes"]),
            )
        runtime = self._last_runtime_stats
        return {
            "worker_id": self.worker_id,
            "device": str(self.device),
            "execution_mode": self.execution_mode,
            "pid": self._process.pid,
            "healthy": not self._closed and self._process.is_alive(),
            "queue_depth": self.queue_depth,
            "local_cpu_affinity": list(self.local_cpu_affinity),
            "free_memory_bytes": int(runtime.get("free_memory_bytes", 0)),
            "total_memory_bytes": int(runtime.get("total_memory_bytes", 0)),
            "peak_memory": {
                "allocated_bytes": int(runtime.get("peak_allocated_bytes", 0)),
                "reserved_bytes": int(runtime.get("peak_reserved_bytes", 0)),
            },
            "shared_output": {
                "slots": int(self._shared_outputs.shape[0]),
                "bytes": self._shared_outputs.numel()
                * self._shared_outputs.element_size(),
                "cuda_host_registered": bool(
                    runtime.get("shared_output_registered", False)
                ),
                "available_slots": len(self.runner._available_slots),
                "overflow_transfers": int(
                    runtime.get("d2h", {}).get("overflow_transfers", 0)
                ),
            },
            "d2h": dict(runtime.get("d2h", {"async": True})),
            "engine": self.engine.stats(),
            "execution": dict(runtime.get("execution", {})),
            "medium_tiers": dict(runtime.get("medium_tiers", {})),
        }
