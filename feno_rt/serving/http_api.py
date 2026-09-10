"""Optional aiohttp JSON/NumPy API for :class:`MultiGPUFENOEngine`."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from io import BytesIO
import json
from time import perf_counter
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch

try:
    from aiohttp import web
except ImportError as error:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "FENO HTTP serving requires the optional 'aiohttp' package"
    ) from error

from feno_rt.runtime.request import (
    EngineClosedError,
    RequestTimeoutError,
)
from feno_rt.observability import ServiceMetrics, render_prometheus
from feno_rt.runtime.replay import output_summary


NPY_CONTENT_TYPE = "application/x-npy"


@dataclass
class _MediumRecord:
    context: Any
    inflight: int = 0
    deleting: bool = False
    flushing: bool = False


@dataclass
class HTTPServerHandle:
    """Handle for a programmatically started aiohttp server."""

    runner: Any
    host: str
    port: int

    async def close(self) -> None:
        await self.runner.cleanup()


class FENOHTTPService:
    """Expose medium preparation, inference, stats, and release over HTTP."""

    def __init__(
        self,
        engine: Any,
        *,
        close_engine: bool = True,
        max_request_bytes: int = 64 * 1024 * 1024,
        max_json_output_elements: int = 2_000_000,
        metrics: Optional[ServiceMetrics] = None,
        trace_recorder: Optional[Any] = None,
        profiler: Optional[Any] = None,
    ) -> None:
        if max_request_bytes <= 0:
            raise ValueError("max_request_bytes must be positive")
        if max_json_output_elements <= 0:
            raise ValueError("max_json_output_elements must be positive")
        self.engine = engine
        self.close_engine = bool(close_engine)
        self.max_request_bytes = int(max_request_bytes)
        self.max_json_output_elements = int(max_json_output_elements)
        self.service_metrics = metrics or ServiceMetrics()
        self.trace_recorder = trace_recorder
        self.profiler = profiler
        self._mediums: Dict[str, _MediumRecord] = {}
        self._lock = asyncio.Lock()
        self._global_flush = False
        self._preparing_mediums = 0
        self._preparing_ids = set()
        self._started = False
        self._closed = False

    def create_app(self) -> "web.Application":
        app = web.Application(
            client_max_size=self.max_request_bytes,
            middlewares=[self._error_middleware],
        )
        app.router.add_get("/health", self.health)
        app.router.add_get("/metrics", self.prometheus_metrics)
        app.router.add_get("/v1/stats", self.stats)
        app.router.add_post("/v1/mediums", self.prepare_medium)
        app.router.add_delete("/v1/mediums/{context_id}", self.delete_medium)
        app.router.add_post("/v1/infer", self.infer)
        app.router.add_post("/v1/cache/flush", self.flush_caches)
        app.router.add_get("/v1/traces/status", self.trace_status)
        app.router.add_post("/v1/traces/start", self.start_trace)
        app.router.add_post("/v1/traces/stop", self.stop_trace)
        app.router.add_get("/v1/profiler/status", self.profiler_status)
        app.router.add_post("/v1/profiler/start", self.start_profiler)
        app.router.add_post("/v1/profiler/stop", self.stop_profiler)
        app.on_startup.append(self._on_startup)
        app.on_cleanup.append(self._on_cleanup)
        return app

    @web.middleware
    async def _error_middleware(self, request: "web.Request", handler: Any):
        started = perf_counter()
        status = 500
        self.service_metrics.begin_http()
        try:
            try:
                response = await handler(request)
            except web.HTTPException as error:
                status = error.status
                raise
            except KeyError as error:
                response = self._error_response(
                    404, "not_found", str(error).strip("'")
                )
            except RequestTimeoutError as error:
                response = self._error_response(504, "request_timeout", str(error))
            except EngineClosedError as error:
                response = self._error_response(503, "engine_closed", str(error))
            except (TypeError, ValueError) as error:
                response = self._error_response(400, "invalid_request", str(error))
            except asyncio.CancelledError:
                status = 499
                raise
            except Exception:
                response = self._error_response(
                    500, "internal_error", "inference failed"
                )
            status = response.status
            return response
        finally:
            self.service_metrics.finish_http(
                method=request.method,
                route=self._canonical_route(request),
                status=status,
                duration_seconds=perf_counter() - started,
            )

    @staticmethod
    def _canonical_route(request: "web.Request") -> str:
        route = getattr(getattr(request, "match_info", None), "route", None)
        resource = getattr(route, "resource", None)
        canonical = getattr(resource, "canonical", None)
        if canonical:
            return str(canonical)
        if request.path.startswith("/v1/mediums/"):
            return "/v1/mediums/{context_id}"
        return request.path

    @staticmethod
    def _error_response(status: int, code: str, message: str) -> "web.Response":
        return web.json_response(
            {"error": {"code": code, "message": message}}, status=status
        )

    async def _on_startup(self, _app: "web.Application") -> None:
        if self._closed:
            raise RuntimeError("HTTP service is closed")
        if not self._started:
            await self.engine.start()
            self._started = True

    async def _on_cleanup(self, _app: "web.Application") -> None:
        if self._closed:
            return
        if self.trace_recorder is not None and self.trace_recorder.status()["active"]:
            try:
                self.trace_recorder.stop()
            except Exception:
                self.service_metrics.observe_operation("trace_stop", "failed")
        if self.profiler is not None and self.profiler.status()["active"]:
            try:
                self.profiler.stop()
            except Exception:
                self.service_metrics.observe_operation("profiler_stop", "failed")
        if self.close_engine:
            await self.engine.close()
        self._closed = True

    async def _json_body(self, request: "web.Request") -> Mapping[str, Any]:
        try:
            body = json.loads(await request.text())
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError("request body is not valid JSON") from error
        if not isinstance(body, Mapping):
            raise ValueError("JSON body must be an object")
        return body

    @staticmethod
    def _query_bool(value: Optional[str], default: bool = False) -> bool:
        if value is None:
            return default
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes"}:
            return True
        if normalized in {"0", "false", "no"}:
            return False
        raise ValueError(f"invalid boolean value: {value}")

    @staticmethod
    def _json_bool(value: Any, name: str, default: bool = False) -> bool:
        if value is None:
            return default
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be a JSON boolean")
        return value

    @staticmethod
    def _validate_context_id(value: Any) -> str:
        context_id = str(value).strip()
        if not context_id:
            raise ValueError("context_id must not be empty")
        if len(context_id) > 128:
            raise ValueError("context_id must contain at most 128 characters")
        return context_id

    async def health(self, _request: "web.Request") -> "web.Response":
        return web.json_response(
            {
                "status": "ok" if self._started and not self._closed else "starting",
                "mediums": len(self._mediums),
            }
        )

    async def prometheus_metrics(self, _request: "web.Request") -> "web.Response":
        payload = render_prometheus(
            self.engine.stats(),
            self.service_metrics.snapshot(),
            registered_mediums=len(self._mediums),
        )
        return web.Response(
            text=payload,
            headers={"Content-Type": "text/plain; version=0.0.4; charset=utf-8"},
        )

    async def stats(self, _request: "web.Request") -> "web.Response":
        async with self._lock:
            registry = {
                context_id: {
                    "inflight": record.inflight,
                    "replicas": list(getattr(record.context, "worker_ids", ())),
                }
                for context_id, record in self._mediums.items()
            }
        operations = {
            "trace": None
            if self.trace_recorder is None
            else self.trace_recorder.status(),
            "profiler": None if self.profiler is None else self.profiler.status(),
            "global_cache_flush_active": self._global_flush,
            "preparing_mediums": self._preparing_mediums,
        }
        return web.json_response(
            {"registry": registry, "runtime": self.engine.stats(), "operations": operations}
        )

    @staticmethod
    def _worker_ids(value: Any) -> Optional[Tuple[str, ...]]:
        if value is None:
            return None
        if not isinstance(value, list) or not value:
            raise ValueError("worker_ids must be a non-empty JSON array")
        worker_ids = tuple(str(item).strip() for item in value)
        if any(not item for item in worker_ids):
            raise ValueError("worker_ids must not contain empty values")
        if len(set(worker_ids)) != len(worker_ids):
            raise ValueError("worker_ids must not contain duplicates")
        return worker_ids

    @staticmethod
    def _clear_context_routes(context: Any, worker_ids: Tuple[str, ...]) -> None:
        route_keys = getattr(context, "route_key_cache", None)
        if route_keys is not None:
            for worker_id in worker_ids:
                route_keys.pop(worker_id, None)
        affinity = getattr(context, "route_affinity", None)
        if affinity is not None:
            for signature, worker_id in list(affinity.items()):
                if worker_id in worker_ids:
                    del affinity[signature]

    async def flush_caches(self, request: "web.Request") -> "web.Response":
        body = await self._json_body(request)
        scope = str(body.get("scope", "all")).lower()
        if scope not in {"all", "medium"}:
            raise ValueError("cache flush scope must be 'all' or 'medium'")
        worker_ids = self._worker_ids(body.get("worker_ids"))
        force = self._json_bool(body.get("force"), "force")
        context_id: Optional[str] = None
        records = []
        context = None

        async with self._lock:
            if self._global_flush:
                return self._error_response(
                    409, "maintenance_busy", "a global cache flush is active"
                )
            if scope == "medium":
                if body.get("context_id") is None:
                    raise ValueError("medium cache flush requires context_id")
                context_id = self._validate_context_id(body["context_id"])
                record = self._mediums.get(context_id)
                if record is None:
                    raise KeyError(f"unknown context_id: {context_id}")
                records = [record]
                context = record.context
            else:
                records = list(self._mediums.values())
                if self._preparing_mediums:
                    return self._error_response(
                        409,
                        "cache_busy",
                        "global cache flush blocked by medium preparation",
                    )
            busy = sum(record.inflight for record in records)
            if busy:
                return self._error_response(
                    409,
                    "cache_busy",
                    f"cache flush blocked by {busy} in-flight request(s)",
                )
            if any(record.deleting or record.flushing for record in records):
                return self._error_response(
                    409, "maintenance_busy", "another maintenance operation is active"
                )
            for record in records:
                record.flushing = True
            if scope == "all":
                self._global_flush = True

        try:
            result = await self.engine.flush_caches(
                context,
                worker_ids=worker_ids,
                force=force,
            )
            affected_workers = tuple(result)
            if scope == "all":
                for record in records:
                    self._clear_context_routes(record.context, affected_workers)
            self.service_metrics.observe_operation(f"cache_flush_{scope}", "succeeded")
        except BaseException:
            self.service_metrics.observe_operation(f"cache_flush_{scope}", "failed")
            raise
        finally:
            async with self._lock:
                for record in records:
                    record.flushing = False
                if scope == "all":
                    self._global_flush = False
        return web.json_response(
            {
                "scope": scope,
                "context_id": context_id,
                "workers": result,
                "force": force,
            }
        )

    def _trace_unavailable(self) -> "web.Response":
        return self._error_response(
            503, "trace_unavailable", "request recording is not configured"
        )

    async def trace_status(self, _request: "web.Request") -> "web.Response":
        if self.trace_recorder is None:
            return self._trace_unavailable()
        return web.json_response(self.trace_recorder.status())

    async def start_trace(self, request: "web.Request") -> "web.Response":
        if self.trace_recorder is None:
            return self._trace_unavailable()
        body = await self._json_body(request)
        metadata = body.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("trace metadata must be a JSON object")
        try:
            status = self.trace_recorder.start(
                name=body.get("name"),
                metadata={
                    **dict(metadata),
                    "runtime_routing_policy": self.engine.stats().get(
                        "routing_policy", "unknown"
                    ),
                },
            )
        except (RuntimeError, FileExistsError) as error:
            self.service_metrics.observe_operation("trace_start", "rejected")
            return self._error_response(409, "trace_conflict", str(error))
        except BaseException:
            self.service_metrics.observe_operation("trace_start", "failed")
            raise
        self.service_metrics.observe_operation("trace_start", "succeeded")
        return web.json_response(status, status=201)

    async def stop_trace(self, _request: "web.Request") -> "web.Response":
        if self.trace_recorder is None:
            return self._trace_unavailable()
        try:
            status = self.trace_recorder.stop()
        except RuntimeError as error:
            self.service_metrics.observe_operation("trace_stop", "rejected")
            return self._error_response(409, "trace_conflict", str(error))
        except BaseException:
            self.service_metrics.observe_operation("trace_stop", "failed")
            raise
        self.service_metrics.observe_operation("trace_stop", "succeeded")
        return web.json_response(status)

    def _profiler_unavailable(self) -> "web.Response":
        return self._error_response(
            503, "profiler_unavailable", "runtime profiling is not configured"
        )

    async def profiler_status(self, _request: "web.Request") -> "web.Response":
        if self.profiler is None:
            return self._profiler_unavailable()
        return web.json_response(self.profiler.status())

    async def start_profiler(self, request: "web.Request") -> "web.Response":
        if self.profiler is None:
            return self._profiler_unavailable()
        body = await self._json_body(request)
        try:
            status = self.profiler.start(name=body.get("name"))
        except (RuntimeError, FileExistsError) as error:
            self.service_metrics.observe_operation("profiler_start", "rejected")
            return self._error_response(409, "profiler_conflict", str(error))
        except BaseException:
            self.service_metrics.observe_operation("profiler_start", "failed")
            raise
        self.service_metrics.observe_operation("profiler_start", "succeeded")
        return web.json_response(status, status=201)

    async def stop_profiler(self, _request: "web.Request") -> "web.Response":
        if self.profiler is None:
            return self._profiler_unavailable()
        try:
            status = self.profiler.stop()
        except RuntimeError as error:
            self.service_metrics.observe_operation("profiler_stop", "rejected")
            return self._error_response(409, "profiler_conflict", str(error))
        except BaseException:
            self.service_metrics.observe_operation("profiler_stop", "failed")
            raise
        self.service_metrics.observe_operation("profiler_stop", "succeeded")
        return web.json_response(status)

    async def _decode_medium(
        self, request: "web.Request"
    ) -> Tuple[Any, Optional[str], bool, int]:
        if request.content_type in {NPY_CONTENT_TYPE, "application/octet-stream"}:
            payload = await request.read()
            try:
                velocity = np.load(BytesIO(payload), allow_pickle=False)
            except (ValueError, EOFError, OSError) as error:
                raise ValueError("request body is not a valid .npy array") from error
            if not isinstance(velocity, np.ndarray):
                raise ValueError("only a single .npy array is accepted")
            medium_id = request.query.get("medium_id")
            already_normalized = self._query_bool(
                request.query.get("already_normalized"), False
            )
            replicas = int(request.query.get("replicas", "1"))
        else:
            body = await self._json_body(request)
            if "velocity" not in body:
                raise ValueError("JSON medium request requires 'velocity'")
            velocity = np.asarray(body["velocity"], dtype=np.float32)
            medium_id = body.get("medium_id")
            already_normalized = self._json_bool(
                body.get("already_normalized"), "already_normalized"
            )
            replicas = int(body.get("replicas", 1))
        if medium_id is not None:
            medium_id = self._validate_context_id(medium_id)
            async with self._lock:
                if medium_id in self._mediums:
                    raise web.HTTPConflict(
                        text=json.dumps(
                            {
                                "error": {
                                    "code": "context_exists",
                                    "message": f"context already exists: {medium_id}",
                                }
                            }
                        ),
                        content_type="application/json",
                    )
        return velocity, medium_id, already_normalized, replicas

    async def prepare_medium(self, request: "web.Request") -> "web.Response":
        velocity, medium_id, already_normalized, replicas = await self._decode_medium(
            request
        )
        async with self._lock:
            if self._global_flush:
                return self._error_response(
                    409, "maintenance_busy", "a global cache flush is active"
                )
            if medium_id is not None and (
                medium_id in self._mediums or medium_id in self._preparing_ids
            ):
                return self._error_response(
                    409, "context_exists", f"context already exists: {medium_id}"
                )
            self._preparing_mediums += 1
            if medium_id is not None:
                self._preparing_ids.add(medium_id)
        try:
            context = await self.engine.prepare_medium(
                velocity,
                medium_id=medium_id,
                already_normalized=already_normalized,
                replicas=replicas,
            )
        except BaseException:
            async with self._lock:
                self._preparing_mediums -= 1
                if medium_id is not None:
                    self._preparing_ids.discard(medium_id)
            raise
        context_id = self._validate_context_id(context.medium_id)
        async with self._lock:
            self._preparing_mediums -= 1
            if medium_id is not None:
                self._preparing_ids.discard(medium_id)
            if context_id in self._mediums:
                raise web.HTTPConflict(
                    text=json.dumps(
                        {
                            "error": {
                                "code": "context_exists",
                                "message": f"context already exists: {context_id}",
                            }
                        }
                    ),
                    content_type="application/json",
                )
            self._mediums[context_id] = _MediumRecord(context=context)
        return web.json_response(
            {
                "context_id": context_id,
                "replicas": list(getattr(context, "worker_ids", ())),
                "velocity_shape": list(np.asarray(velocity).shape),
            },
            status=201,
        )

    async def _acquire_context(self, context_id: str) -> _MediumRecord:
        async with self._lock:
            record = self._mediums.get(context_id)
            if record is None:
                raise KeyError(f"unknown context_id: {context_id}")
            if record.deleting or record.flushing:
                operation = "deleted" if record.deleting else "flushed"
                raise web.HTTPConflict(
                    text=json.dumps(
                        {
                            "error": {
                                "code": "context_maintenance",
                                "message": (
                                    f"context is being {operation}: {context_id}"
                                ),
                            }
                        }
                    ),
                    content_type="application/json",
                )
            record.inflight += 1
            return record

    async def _release_context(self, context_id: str) -> None:
        async with self._lock:
            record = self._mediums.get(context_id)
            if record is not None:
                record.inflight = max(0, record.inflight - 1)

    async def delete_medium(self, request: "web.Request") -> "web.Response":
        context_id = self._validate_context_id(request.match_info["context_id"])
        async with self._lock:
            record = self._mediums.get(context_id)
            if record is None:
                raise KeyError(f"unknown context_id: {context_id}")
            if record.inflight:
                return self._error_response(
                    409,
                    "context_busy",
                    f"context has {record.inflight} in-flight request(s)",
                )
            if record.flushing:
                return self._error_response(
                    409, "maintenance_busy", "the context is being flushed"
                )
            record.deleting = True
        release = getattr(self.engine, "release_medium", None)
        try:
            invalidated = await release(record.context) if release is not None else {}
        except BaseException:
            async with self._lock:
                current = self._mediums.get(context_id)
                if current is record:
                    current.deleting = False
            raise
        async with self._lock:
            if self._mediums.get(context_id) is record:
                del self._mediums[context_id]
        return web.json_response(
            {"context_id": context_id, "released": True, "invalidated": invalidated}
        )

    def _record_trace(
        self,
        request_payload: Mapping[str, Any],
        result_payload: Mapping[str, Any],
    ) -> None:
        if self.trace_recorder is None:
            return
        try:
            recorded = self.trace_recorder.record(request_payload, result_payload)
        except Exception:
            self.service_metrics.observe_operation("trace_record", "failed")
            return
        if recorded:
            self.service_metrics.observe_operation("trace_record", "succeeded")

    async def infer(self, request: "web.Request") -> "web.Response":
        body = await self._json_body(request)
        if "context_id" not in body:
            raise ValueError("inference request requires 'context_id'")
        if "source_position" not in body or "frequency" not in body:
            raise ValueError("inference request requires source_position and frequency")
        response_format = str(
            body.get(
                "response_format",
                "npy" if NPY_CONTENT_TYPE in request.headers.get("Accept", "") else "json",
            )
        ).lower()
        if response_format not in {"json", "npy"}:
            raise ValueError("response_format must be 'json' or 'npy'")
        context_id = self._validate_context_id(body["context_id"])
        positions_are_normalized = self._json_bool(
            body.get("positions_are_normalized"), "positions_are_normalized"
        )
        denormalize = self._json_bool(body.get("denormalize"), "denormalize")
        cache_level = str(body.get("cache_level", "all"))
        timeout_s = None if body.get("timeout_s") is None else float(body["timeout_s"])
        priority = int(body.get("priority", 0))
        trace_request = {
            "context_id": context_id,
            "source_position": body["source_position"],
            "frequency": body["frequency"],
            "receiver_positions": body.get("receiver_positions"),
            "positions_are_normalized": positions_are_normalized,
            "denormalize": denormalize,
            "cache_level": cache_level,
            "timeout_s": timeout_s,
            "priority": priority,
            "original_request_id": body.get("request_id"),
        }
        record = await self._acquire_context(context_id)
        try:
            try:
                handle = await self.engine.submit(
                    record.context,
                    body["source_position"],
                    body["frequency"],
                    receiver_positions=body.get("receiver_positions"),
                    positions_are_normalized=positions_are_normalized,
                    denormalize=denormalize,
                    cache_level=cache_level,
                    timeout_s=timeout_s,
                    priority=priority,
                    request_id=body.get("request_id"),
                )
                output = await handle
            except BaseException as error:
                self._record_trace(
                    trace_request,
                    {
                        "status": (
                            "cancelled"
                            if isinstance(error, asyncio.CancelledError)
                            else "failed"
                        ),
                        "error_type": type(error).__name__,
                    },
                )
                raise
        finally:
            await self._release_context(context_id)

        output_cpu = torch.as_tensor(output).detach().to(device="cpu").contiguous()
        summary = output_summary(output_cpu)
        self._record_trace(
            trace_request,
            {
                "status": "succeeded",
                "request_id": str(handle.request_id),
                "worker_id": str(getattr(handle, "worker_id", "unknown")),
                **summary,
            },
        )
        headers = {
            "X-FENO-Request-ID": str(handle.request_id),
            "X-FENO-Worker-ID": str(getattr(handle, "worker_id", "unknown")),
            "X-FENO-Shape": ",".join(str(value) for value in output_cpu.shape),
            "X-FENO-Dtype": str(output_cpu.dtype).removeprefix("torch."),
        }
        if response_format == "npy":
            buffer = BytesIO()
            np.save(buffer, output_cpu.numpy(), allow_pickle=False)
            return web.Response(
                body=buffer.getvalue(), content_type=NPY_CONTENT_TYPE, headers=headers
            )
        if output_cpu.numel() > self.max_json_output_elements:
            raise ValueError(
                "output is too large for JSON; request response_format='npy'"
            )
        return web.json_response(
            {
                "request_id": handle.request_id,
                "worker_id": getattr(handle, "worker_id", None),
                "shape": list(output_cpu.shape),
                "dtype": str(output_cpu.dtype).removeprefix("torch."),
                "output": output_cpu.tolist(),
            },
            headers=headers,
        )


async def start_http_server(
    service: FENOHTTPService,
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
) -> HTTPServerHandle:
    """Start the HTTP service without taking ownership of the event loop."""
    runner = web.AppRunner(service.create_app())
    await runner.setup()
    site = web.TCPSite(runner, host=host, port=port)
    await site.start()
    sockets = getattr(site._server, "sockets", ())
    bound_port = int(sockets[0].getsockname()[1]) if sockets else int(port)
    return HTTPServerHandle(runner=runner, host=host, port=bound_port)
