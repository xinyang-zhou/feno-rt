"""Asynchronous cache-aware dynamic batching engine for FENO inference."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from hashlib import sha256
from time import perf_counter_ns
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from uuid import uuid4

import torch

from feno_rt.preprocessing import prepare_frequencies, prepare_source_receiver_batch

from .request import (
    EngineClosedError,
    InferenceRequest,
    RequestCost,
    RequestHandle,
    RequestMetrics,
    RequestState,
    RequestTimeoutError,
)
from .scheduler import BaseBatchScheduler, DynamicBatchConfig, make_scheduler


@dataclass(frozen=True)
class _PreparedBatch:
    requests: Tuple[InferenceRequest, ...]
    sources: torch.Tensor
    frequencies: torch.Tensor
    receivers: Optional[torch.Tensor]
    preparation_ms: float


class AsyncRequestQueue:
    """Condition-backed async queue that waits for a useful micro-batch."""

    def __init__(
        self,
        config: DynamicBatchConfig,
        terminal_callback: Callable[[InferenceRequest], None],
    ) -> None:
        self._config = config
        self._terminal_callback = terminal_callback
        self._pending: List[InferenceRequest] = []
        self._condition = asyncio.Condition()
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    async def put(self, request: InferenceRequest) -> None:
        async with self._condition:
            if self._closed:
                raise EngineClosedError("the request queue is closed")
            self._pending.append(request)
            self._condition.notify()

    async def wake(self) -> None:
        async with self._condition:
            self._condition.notify_all()

    def _remove_terminal_and_expired(self, now_ns: int) -> None:
        retained: List[InferenceRequest] = []
        for request in self._pending:
            if request.terminal or request.future.cancelled():
                continue
            if request.is_expired(now_ns):
                request.state = RequestState.FAILED
                request.completed_ns = now_ns
                error = RequestTimeoutError(
                    f"request {request.request_id} expired while waiting in the queue"
                )
                request.error = error
                if not request.future.done():
                    request.future.set_exception(error)
                self._terminal_callback(request)
                continue
            retained.append(request)
        self._pending = retained

    def _dispatch_ready(
        self,
        batch: Sequence[InferenceRequest],
        scheduler: BaseBatchScheduler,
        now_ns: int,
    ) -> bool:
        if self._closed:
            return True
        if len(self._pending) >= self._config.max_batch_size:
            return True
        if scheduler.is_saturated(batch, self._pending):
            return True
        oldest_age = max((request.age_us(now_ns) for request in self._pending), default=0.0)
        if oldest_age >= self._config.max_wait_us:
            return True
        earliest_slack = min(
            (request.deadline_slack_us(now_ns) for request in self._pending),
            default=float("inf"),
        )
        return earliest_slack <= self._config.deadline_guard_us

    def _wait_seconds(self, now_ns: int) -> Optional[float]:
        if not self._pending:
            return None
        oldest_age = max(request.age_us(now_ns) for request in self._pending)
        wait_us = max(0.0, self._config.max_wait_us - oldest_age)
        deadline_us = min(
            (request.deadline_slack_us(now_ns) for request in self._pending),
            default=float("inf"),
        )
        wake_us = min(wait_us, max(0.0, deadline_us - self._config.deadline_guard_us))
        return max(wake_us / 1e6, 1e-6)

    async def next_batch(
        self,
        scheduler: BaseBatchScheduler,
    ) -> Optional[List[InferenceRequest]]:
        async with self._condition:
            while True:
                now_ns = perf_counter_ns()
                self._remove_terminal_and_expired(now_ns)
                if self._closed and not self._pending:
                    return None
                if self._pending:
                    batch = scheduler.select_batch(self._pending, now_ns=now_ns)
                    if not batch:
                        raise RuntimeError("scheduler could not admit the head request")
                    if self._dispatch_ready(batch, scheduler, now_ns):
                        selected_ids = {request.request_id for request in batch}
                        self._pending = [
                            request
                            for request in self._pending
                            if request.request_id not in selected_ids
                        ]
                        return list(batch)
                    timeout = self._wait_seconds(now_ns)
                else:
                    timeout = None

                if timeout is None:
                    await self._condition.wait()
                else:
                    try:
                        await asyncio.wait_for(self._condition.wait(), timeout)
                    except asyncio.TimeoutError:
                        pass

    async def close(self, *, cancel_pending: bool = False) -> None:
        async with self._condition:
            self._closed = True
            if cancel_pending:
                now_ns = perf_counter_ns()
                for request in self._pending:
                    if request.terminal:
                        continue
                    request.state = RequestState.FAILED
                    request.completed_ns = now_ns
                    request.future.cancel()
                    self._terminal_callback(request)
                self._pending.clear()
            self._condition.notify_all()


class AsyncFENOEngine:
    """Single-worker async inference engine with dynamic micro-batching.

    One background coroutine owns scheduling, while one executor thread owns
    model execution.  Batch failures are recursively bisected so a malformed
    request does not fail unrelated requests.
    """

    def __init__(
        self,
        runner: Any,
        config: Optional[DynamicBatchConfig] = None,
        *,
        double_buffer: bool = False,
        execution_stream: Optional[torch.cuda.Stream] = None,
        executor_initializer: Optional[Callable[[], None]] = None,
        context_acquirer: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        self.runner = runner
        self.config = config or DynamicBatchConfig()
        self.double_buffer = bool(double_buffer)
        self.execution_stream = execution_stream
        self.executor_initializer = executor_initializer
        self.context_acquirer = context_acquirer
        if execution_stream is not None and torch.device(runner.device).type != "cuda":
            raise ValueError("execution_stream requires a CUDA model runner")
        self._metrics = RequestMetrics()
        self._terminal_ids = set()
        self._request_ids = set()
        self._sequence = 0
        self._preparation_ms: List[float] = []
        self._prefetched_batches = 0
        self._worker_task: Optional["asyncio.Task[None]"] = None
        self._executor: Optional[ThreadPoolExecutor] = None
        self._queue: Optional[AsyncRequestQueue] = None
        self._closed = False
        self._start_lock = asyncio.Lock()

    async def __aenter__(self) -> "AsyncFENOEngine":
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        await self.close(graceful=exc is None)

    async def start(self) -> None:
        async with self._start_lock:
            if self._closed:
                raise EngineClosedError("the engine is closed")
            if self._worker_task is not None:
                return
            self._executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="feno-worker",
                initializer=self.executor_initializer,
            )
            self._queue = AsyncRequestQueue(self.config, self._record_terminal)
            self._worker_task = asyncio.create_task(self._worker_loop(), name="feno-batcher")

    def _cache_probe(self, request: InferenceRequest) -> Mapping[str, bool]:
        probe = getattr(self.runner, "cache_residency", None)
        if probe is None:
            return {}
        return probe(dict(request.cache_keys))

    @staticmethod
    def _digest_tensor(value: torch.Tensor) -> str:
        tensor = value.detach().to(device="cpu").clone().contiguous()
        digest = sha256()
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def _prepare_identity(
        self,
        context: Any,
        source_position: Any,
        frequency: Any,
        receiver_positions: Optional[Any],
        positions_are_normalized: bool,
        denormalize: bool,
        cache_level: str,
    ) -> Tuple[Tuple[Any, ...], Any, Any, Dict[str, Any]]:
        sources, receivers = prepare_source_receiver_batch(
            source_position,
            self.runner.config,
            receiver_positions=receiver_positions,
            positions_are_normalized=positions_are_normalized,
            device=torch.device("cpu"),
            dtype=self.runner.dtype,
        )
        frequencies = prepare_frequencies(
            frequency,
            sources.shape[0],
            device=torch.device("cpu"),
            dtype=self.runner.dtype,
        )
        if sources.shape[0] != 1:
            raise ValueError("submit accepts exactly one source position per request")

        key_builder = getattr(self.runner, "request_cache_keys", None)
        cache_keys: Dict[str, Any] = {}
        if key_builder is not None:
            cache_keys = key_builder(
                context,
                source_position,
                frequency,
                receiver_positions=receiver_positions,
                positions_are_normalized=positions_are_normalized,
            )
        medium_key = cache_keys.get(
            "medium",
            getattr(context, "cache_key", None)
            or (getattr(context, "medium_id", None), id(getattr(context, "latent", context))),
        )
        geometry_key = cache_keys.get(
            "geometry",
            (medium_key, self._digest_tensor(sources[0]), self._digest_tensor(receivers[0])),
        )
        frequency_key = (
            cache_keys["wavelet"]
            if "wavelet" in cache_keys
            else self._digest_tensor(frequencies[0:1])
        )
        batch_key = (
            medium_key,
            bool(positions_are_normalized),
            bool(denormalize),
            cache_level,
            int(receivers.shape[1]),
        )
        return batch_key, geometry_key, frequency_key, cache_keys

    def estimate_request_cost(self) -> RequestCost:
        """Estimate worst-case decoder buffers for one request.

        Output bytes are exact. Activation bytes cover geometry queries at each
        decoder layer plus wavelet token/K/V buffers and the final output. It is
        intentionally conservative and is an admission budget, not a profiler.
        """
        config = self.runner.config
        dtype_bytes = torch.empty((), dtype=self.runner.dtype).element_size()
        query_tokens = int(config.num_receivers)
        query_state_bytes = query_tokens * int(config.decoder_dim) * dtype_bytes
        wavelet_bytes = int(config.output_steps) * int(config.decoder_dim) * dtype_bytes
        output_bytes = query_tokens * int(config.output_steps) * dtype_bytes
        activation_bytes = (
            query_state_bytes * (2 * int(config.decoder_depth) + 4)
            + wavelet_bytes * 3
            + output_bytes
        )
        return RequestCost(
            query_tokens=query_tokens,
            activation_bytes=activation_bytes,
            output_bytes=output_bytes,
        )

    async def submit(
        self,
        context: Any,
        source_position: Any,
        frequency: Any,
        *,
        receiver_positions: Optional[Any] = None,
        positions_are_normalized: bool = False,
        denormalize: bool = False,
        cache_level: str = "all",
        timeout_s: Optional[float] = None,
        deadline_s: Optional[float] = None,
        priority: int = 0,
        request_id: Optional[str] = None,
    ) -> RequestHandle:
        """Validate and enqueue one request, returning an awaitable handle.

        ``deadline_s`` is an absolute ``time.monotonic()``-style timestamp.
        ``timeout_s`` is relative to submission; at most one may be supplied.
        """
        submitted_ns = perf_counter_ns()
        if timeout_s is not None and deadline_s is not None:
            raise ValueError("pass timeout_s or deadline_s, not both")
        if timeout_s is None and deadline_s is None:
            timeout_s = self.config.default_timeout_s
        if timeout_s is not None and timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if priority < 0:
            raise ValueError("priority must be non-negative")
        if self._closed:
            raise EngineClosedError("the engine is closed")
        await self.start()
        cache_levels = getattr(self.runner, "CACHE_LEVELS", None)
        if cache_levels is not None and cache_level not in cache_levels:
            raise ValueError(f"cache_level must be one of {sorted(cache_levels)}")
        source_snapshot = (
            torch.as_tensor(source_position, dtype=self.runner.dtype).detach().cpu().clone()
        )
        frequency_snapshot = (
            torch.as_tensor(frequency, dtype=self.runner.dtype).detach().cpu().clone()
        )
        receiver_snapshot = (
            None
            if receiver_positions is None
            else torch.as_tensor(receiver_positions, dtype=self.runner.dtype)
            .detach()
            .cpu()
            .clone()
        )
        batch_key, geometry_key, frequency_key, cache_keys = self._prepare_identity(
            context,
            source_snapshot,
            frequency_snapshot,
            receiver_snapshot,
            positions_are_normalized,
            denormalize,
            cache_level,
        )
        cost = self.estimate_request_cost()
        enqueued_ns = perf_counter_ns()
        if deadline_s is not None:
            deadline_ns = int(deadline_s * 1e9)
        elif timeout_s is not None:
            deadline_ns = submitted_ns + int(timeout_s * 1e9)
        else:
            deadline_ns = None
        loop = asyncio.get_running_loop()
        resolved_request_id = request_id or uuid4().hex
        if resolved_request_id in self._request_ids:
            raise ValueError(f"duplicate request_id: {resolved_request_id}")
        request = InferenceRequest(
            request_id=resolved_request_id,
            sequence_id=self._sequence,
            context=context,
            source_position=source_snapshot,
            frequency=frequency_snapshot,
            receiver_positions=receiver_snapshot,
            positions_are_normalized=positions_are_normalized,
            denormalize=denormalize,
            cache_level=cache_level,
            priority=priority,
            cost=cost,
            batch_key=batch_key,
            geometry_key=geometry_key,
            frequency_key=frequency_key,
            cache_keys=cache_keys,
            future=loop.create_future(),
            submitted_ns=submitted_ns,
            enqueued_ns=enqueued_ns,
            deadline_ns=deadline_ns,
        )
        scheduler = make_scheduler(self.config, cache_probe=self._cache_probe)
        if not scheduler.request_fits_empty_batch(request):
            raise ValueError(
                "one request exceeds the configured query-token, activation, or output budget"
            )
        self._request_ids.add(resolved_request_id)
        self._sequence += 1
        self._metrics.submitted += 1
        assert self._queue is not None
        try:
            await self._queue.put(request)
        except BaseException:
            self._metrics.submitted -= 1
            self._request_ids.remove(resolved_request_id)
            raise
        try:
            # An unbounded Condition.put() may complete without suspending.
            # Yield once so the batcher is not starved by a burst of submitters.
            await asyncio.sleep(0)
        except asyncio.CancelledError:
            self._cancel_request(request)
            raise
        return RequestHandle(request, self._cancel_request)

    async def infer(self, *args: Any, **kwargs: Any) -> Any:
        handle = await self.submit(*args, **kwargs)
        return await handle

    def _run_on_execution_stream(
        self,
        function: Callable[[], Any],
        *,
        synchronize: bool,
    ) -> Any:
        device = torch.device(self.runner.device)
        if self.execution_stream is None:
            result = function()
            if synchronize and device.type == "cuda":
                torch.cuda.synchronize(device)
            return result

        with torch.cuda.device(device):
            self.execution_stream.wait_stream(torch.cuda.default_stream(device))
            with torch.cuda.stream(self.execution_stream):
                result = function()
            if synchronize:
                self.execution_stream.synchronize()
        return result

    def _run_serialized_sync(self, function: Callable[[], Any]) -> Any:
        return self._run_on_execution_stream(function, synchronize=True)

    async def run_serialized(
        self,
        function: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Run control work on the same executor and CUDA stream as inference."""
        if self._closed:
            raise EngineClosedError("the engine is closed")
        await self.start()
        assert self._executor is not None
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self._run_serialized_sync,
            partial(function, *args, **kwargs),
        )

    def _cancel_request(self, request: InferenceRequest) -> bool:
        if request.terminal or request.future.done():
            return False
        request.state = RequestState.FAILED
        request.completed_ns = perf_counter_ns()
        cancelled = request.future.cancel()
        if cancelled:
            self._record_terminal(request)
            if self._queue is not None:
                asyncio.create_task(self._queue.wake())
        return cancelled

    def _record_terminal(self, request: InferenceRequest) -> None:
        if request.request_id in self._terminal_ids:
            return
        self._terminal_ids.add(request.request_id)
        if request.future.cancelled():
            self._metrics.cancelled += 1
        elif isinstance(request.error, RequestTimeoutError):
            self._metrics.timed_out += 1
        elif request.error is not None:
            self._metrics.failed += 1
        else:
            self._metrics.succeeded += 1
        if request.started_ns is not None:
            self._metrics.queue_ms.append((request.started_ns - request.enqueued_ns) / 1e6)
        if request.completed_ns is not None:
            self._metrics.end_to_end_ms.append((request.completed_ns - request.submitted_ns) / 1e6)

    def _stack_receivers(self, requests: Sequence[InferenceRequest]) -> Optional[torch.Tensor]:
        if all(request.receiver_positions is None for request in requests):
            return None
        config = self.runner.config
        default_receivers = torch.zeros(config.num_receivers, 2, dtype=self.runner.dtype)
        default_receivers[:, 0] = float(config.receiver_depth)
        default_receivers[:, 1] = torch.arange(config.num_receivers, dtype=self.runner.dtype)
        values = [
            default_receivers
            if request.receiver_positions is None
            else torch.as_tensor(request.receiver_positions, dtype=self.runner.dtype)
            for request in requests
        ]
        return torch.stack(values)

    def _prepare_batch(self, requests: Sequence[InferenceRequest]) -> _PreparedBatch:
        started_ns = perf_counter_ns()
        sources = torch.stack(
            [torch.as_tensor(request.source_position, dtype=self.runner.dtype).reshape(2) for request in requests]
        )
        frequencies = torch.stack(
            [torch.as_tensor(request.frequency, dtype=self.runner.dtype).reshape(()) for request in requests]
        )
        receivers = self._stack_receivers(requests)
        preparation_ms = (perf_counter_ns() - started_ns) / 1e6
        return _PreparedBatch(
            requests=tuple(requests),
            sources=sources,
            frequencies=frequencies,
            receivers=receivers,
            preparation_ms=preparation_ms,
        )

    def _run_batch_sync(self, prepared: _PreparedBatch) -> List[torch.Tensor]:
        def run() -> List[torch.Tensor]:
            requests = prepared.requests
            first = requests[0]
            context = first.context
            context_lease = (
                self.context_acquirer(context)
                if self.context_acquirer is not None
                else None
            )
            if context_lease is not None:
                context = context_lease.value
            try:
                output = self.runner.forward_batch(
                    context,
                    prepared.sources,
                    prepared.frequencies,
                    receiver_positions=prepared.receivers,
                    positions_are_normalized=first.positions_are_normalized,
                    denormalize=first.denormalize,
                    cache_level=first.cache_level,
                )
            finally:
                if context_lease is not None:
                    context_lease.release()
            if output.shape[0] != len(requests):
                raise RuntimeError(
                    f"runner returned batch {output.shape[0]} for {len(requests)} requests"
                )
            owned_output = (
                output
                if getattr(self.runner, "forward_output_owned", False)
                else output.clone()
            )
            output_lease = getattr(owned_output, "_feno_output_lease", None)
            results = [owned_output[index] for index in range(len(requests))]
            if output_lease is not None:
                for result in results:
                    result._feno_output_lease = output_lease
            return results

        return self._run_on_execution_stream(
            run,
            synchronize=self.config.synchronize_device,
        )

    async def _execute_isolated(
        self,
        prepared: _PreparedBatch,
    ) -> List[Tuple[InferenceRequest, Optional[torch.Tensor], Optional[BaseException]]]:
        requests = prepared.requests
        if not requests:
            return []
        assert self._executor is not None
        loop = asyncio.get_running_loop()
        try:
            outputs = await loop.run_in_executor(
                self._executor,
                self._run_batch_sync,
                prepared,
            )
            return [
                (request, output, None)
                for request, output in zip(requests, outputs)
            ]
        except Exception as error:
            if not self.config.error_isolation or len(requests) == 1:
                return [(request, None, error) for request in requests]
            self._metrics.isolated_retries += 1
            midpoint = len(requests) // 2
            left = await self._execute_isolated(self._prepare_batch(requests[:midpoint]))
            right = await self._execute_isolated(self._prepare_batch(requests[midpoint:]))
            return left + right

    def _deliver(
        self,
        results: Iterable[Tuple[InferenceRequest, Optional[torch.Tensor], Optional[BaseException]]],
        execution_ms: float,
    ) -> None:
        completed_ns = perf_counter_ns()
        for request, output, error in results:
            if request.terminal or request.future.cancelled():
                continue
            request.completed_ns = completed_ns
            self._metrics.execution_ms.append(execution_ms)
            if request.is_expired(completed_ns):
                request.state = RequestState.FAILED
                timeout = RequestTimeoutError(
                    f"request {request.request_id} missed its deadline during execution"
                )
                request.error = timeout
                request.future.set_exception(timeout)
            elif error is not None:
                request.state = RequestState.FAILED
                request.error = error
                request.future.set_exception(error)
            else:
                request.future.set_result(output)
            self._record_terminal(request)

    async def _next_prepared_batch(
        self,
        scheduler: BaseBatchScheduler,
    ) -> Optional[_PreparedBatch]:
        assert self._queue is not None
        requests = await self._queue.next_batch(scheduler)
        if requests is None:
            return None
        prepared = self._prepare_batch(requests)
        self._preparation_ms.append(prepared.preparation_ms)
        self._prefetched_batches += 1
        return prepared

    def _activate_prepared(
        self,
        prepared: _PreparedBatch,
    ) -> Optional[_PreparedBatch]:
        now_ns = perf_counter_ns()
        active: List[InferenceRequest] = []
        for request in prepared.requests:
            if request.terminal or request.future.cancelled():
                continue
            if request.is_expired(now_ns):
                request.state = RequestState.FAILED
                request.completed_ns = now_ns
                error = RequestTimeoutError(
                    f"request {request.request_id} expired after batch prefetch"
                )
                request.error = error
                if not request.future.done():
                    request.future.set_exception(error)
                self._record_terminal(request)
                continue
            active.append(request)
        if not active:
            return None
        if len(active) != len(prepared.requests):
            prepared = self._prepare_batch(active)
            self._preparation_ms.append(prepared.preparation_ms)
        started_ns = perf_counter_ns()
        for request in prepared.requests:
            request.state = RequestState.RUNNING
            request.started_ns = started_ns
        return prepared

    async def _single_buffer_loop(self, scheduler: BaseBatchScheduler) -> None:
        while True:
            prepared = await self._next_prepared_batch(scheduler)
            if prepared is None:
                return
            active = self._activate_prepared(prepared)
            if active is None:
                continue
            self._metrics.batches += 1
            self._metrics.batch_sizes.append(len(active.requests))
            started_ns = perf_counter_ns()
            results = await self._execute_isolated(active)
            execution_ms = (perf_counter_ns() - started_ns) / 1e6
            self._deliver(results, execution_ms)

    async def _worker_loop(self) -> None:
        scheduler = make_scheduler(self.config, cache_probe=self._cache_probe)
        assert self._queue is not None
        if not self.double_buffer:
            await self._single_buffer_loop(scheduler)
            return
        prepared = await self._next_prepared_batch(scheduler)
        while prepared is not None:
            active = self._activate_prepared(prepared)
            if active is None:
                prepared = await self._next_prepared_batch(scheduler)
                continue
            self._metrics.batches += 1
            self._metrics.batch_sizes.append(len(active.requests))
            next_batch_task = asyncio.create_task(
                self._next_prepared_batch(scheduler)
            )
            started_ns = perf_counter_ns()
            results = await self._execute_isolated(active)
            execution_ms = (perf_counter_ns() - started_ns) / 1e6
            self._deliver(results, execution_ms)
            prepared = await next_batch_task

    def stats(self) -> Dict[str, Any]:
        metrics = self._metrics.snapshot(max_batch_size=self.config.max_batch_size)
        metrics["policy"] = self.config.policy.value
        metrics["limits"] = {
            "max_wait_us": self.config.max_wait_us,
            "max_batch_size": self.config.max_batch_size,
            "max_query_tokens": self.config.max_query_tokens,
            "max_activation_bytes": self.config.max_activation_bytes,
            "max_output_bytes": self.config.max_output_bytes,
        }
        metrics["pipeline"] = {
            "double_buffered": self.double_buffer,
            "prepared_batches": self._prefetched_batches,
            "mean_preparation_ms": (
                sum(self._preparation_ms) / len(self._preparation_ms)
                if self._preparation_ms else 0.0
            ),
        }
        cache_metrics = getattr(self.runner, "cache_metrics", None)
        metrics["cache"] = cache_metrics() if cache_metrics is not None else {}
        return metrics

    async def close(self, *, graceful: bool = True) -> None:
        if self._closed:
            return
        self._closed = True
        if self._queue is not None:
            await self._queue.close(cancel_pending=not graceful)
        if self._worker_task is not None:
            await self._worker_task
        if self._executor is not None:
            self._executor.shutdown(wait=True)
        self._worker_task = None
        self._executor = None
