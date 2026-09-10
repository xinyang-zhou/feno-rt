"""CPU-only Stage 6 metrics, profiling, trace, and HTTP control tests."""

import asyncio
from dataclasses import dataclass, field
import json
from pathlib import Path
import tempfile
import unittest

from aiohttp.test_utils import TestClient, TestServer
import torch

from feno_rt.observability import RuntimeProfiler, ServiceMetrics, render_prometheus
from feno_rt.runtime import RequestTraceRecorder, load_trace, replay_trace
from feno_rt.serving import FENOHTTPService


class FakeProfile:
    def __init__(self):
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def export_chrome_trace(self, path):
        Path(path).write_text('{"traceEvents": []}\n', encoding="utf-8")


@dataclass
class FakeContext:
    medium_id: str
    worker_ids: tuple = ("gpu-0", "gpu-1")
    route_key_cache: dict = field(
        default_factory=lambda: {"gpu-0": {"key": "value"}}
    )
    route_affinity: dict = field(default_factory=lambda: {("key",): "gpu-0"})


class FakeHandle:
    def __init__(self, output, request_id, worker_id="gpu-0", gate=None):
        self.output = output
        self.request_id = request_id
        self.worker_id = worker_id
        self.gate = gate

    async def result(self):
        if self.gate is not None:
            await self.gate.wait()
        return self.output

    def __await__(self):
        return self.result().__await__()


class FakeStage6Engine:
    def __init__(self):
        self.started = 0
        self.closed = 0
        self.submitted = 0
        self.completed = 0
        self.flushes = []
        self.flush_started = asyncio.Event()
        self.flush_gate = None
        self.submit_started = asyncio.Event()
        self.result_gate = None

    async def start(self):
        self.started += 1

    async def close(self):
        self.closed += 1

    async def prepare_medium(
        self, velocity, *, medium_id=None, already_normalized=False, replicas=1
    ):
        return FakeContext(medium_id or "derived", tuple(f"gpu-{i}" for i in range(replicas)))

    async def release_medium(self, context):
        return {worker_id: {"medium": 1} for worker_id in context.worker_ids}

    async def flush_caches(self, context=None, *, worker_ids=None, force=False):
        self.flush_started.set()
        if self.flush_gate is not None:
            await self.flush_gate.wait()
        selected = tuple(worker_ids or ("gpu-0", "gpu-1"))
        self.flushes.append(
            {
                "context": None if context is None else context.medium_id,
                "worker_ids": selected,
                "force": force,
            }
        )
        return {worker_id: {"medium": 1} for worker_id in selected}

    async def submit(self, context, source_position, frequency, **kwargs):
        self.submitted += 1
        self.completed += 1
        self.submit_started.set()
        output = torch.tensor(source_position, dtype=torch.float32) + float(frequency)
        return FakeHandle(
            output,
            kwargs.get("request_id") or f"request-{self.submitted}",
            worker_id=f"gpu-{(self.submitted - 1) % 2}",
            gate=self.result_gate,
        )

    def stats(self):
        return {
            "routing_policy": "cache_aware",
            "submitted": self.submitted,
            "completed": self.completed,
            "failed": 0,
            "cancelled": 0,
            "selection_counts": {"gpu-0": self.submitted},
            "routing_mean_us": 12.0,
            "workers": [],
        }


class Stage6PrimitiveTest(unittest.IsolatedAsyncioTestCase):
    async def test_trace_record_load_and_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = RequestTraceRecorder(directory)
            recorder.start(name="unit-trace", metadata={"purpose": "test"})
            recorder.record(
                {
                    "context_id": "medium",
                    "source_position": [2.0, 3.0],
                    "frequency": 5.0,
                    "positions_are_normalized": False,
                    "denormalize": False,
                    "cache_level": "all",
                    "timeout_s": None,
                    "priority": 0,
                },
                {
                    "status": "succeeded",
                    "checksum": 15.0,
                    "shape": [2],
                    "dtype": "float32",
                    "finite": True,
                },
            )
            stopped = recorder.stop()
            records = load_trace(Path(directory) / stopped["file"])
            self.assertEqual(len(records), 2)
            engine = FakeStage6Engine()
            replay = await replay_trace(
                engine,
                {"medium": FakeContext("medium")},
                records,
                wave_size=1,
                request_prefix="unit-replay",
            )
            self.assertEqual(replay["status_counts"], {"succeeded": 1})
            self.assertEqual(replay["verified_checksums"], 1)
            self.assertEqual(replay["max_relative_checksum_error"], 0.0)

    async def test_profiler_path_and_lifecycle_are_bounded(self):
        profiles = []

        def factory(_activities):
            profile = FakeProfile()
            profiles.append(profile)
            return profile

        with tempfile.TemporaryDirectory() as directory:
            profiler = RuntimeProfiler(
                Path(directory), include_cuda=False, profiler_factory=factory
            )
            status = profiler.start(name="unit-profile")
            self.assertTrue(status["active"])
            with self.assertRaises(RuntimeError):
                profiler.start(name="second")
            stopped = profiler.stop()
            self.assertTrue((Path(directory) / stopped["file"]).is_file())
            self.assertTrue(profiles[0].started)
            self.assertTrue(profiles[0].stopped)
            with self.assertRaises(ValueError):
                profiler.start(name="../escape")

    async def test_prometheus_renderer_has_stable_labels(self):
        metrics = ServiceMetrics()
        metrics.begin_http()
        metrics.finish_http(
            method="GET",
            route="/health",
            status=200,
            duration_seconds=0.01,
        )
        runtime = FakeStage6Engine().stats()
        runtime["workers"] = [
            {
                "worker_id": "gpu-1",
                "device": "cuda:1",
                "d2h": {"copies": 2, "bytes": 1024, "time_ms": 0.5},
                "engine": {
                    "requests": {"submitted": 2, "succeeded": 1, "pending": 1}
                },
                "medium_tiers": {
                    "gpu": {
                        "resident_entries": 2,
                        "resident_bytes": 4096,
                        "capacity_bytes": 8192,
                        "hits": 3,
                        "capacity_overflows": 0,
                    },
                    "pinned_cpu": {
                        "resident_entries": 1,
                        "resident_bytes": 2048,
                        "capacity_bytes": 4096,
                        "hits": 1,
                        "evictions": 0,
                    },
                    "transfers": {
                        "promotions": 1,
                        "demotions": 2,
                        "h2d_bytes": 2048,
                        "d2h_bytes": 4096,
                    },
                    "source": {"rebuilds": 2},
                },
            }
        ]
        text = render_prometheus(runtime, metrics.snapshot(), registered_mediums=2)
        self.assertIn("# TYPE feno_http_requests_total counter", text)
        self.assertIn('method="GET",route="/health",status="200"', text)
        self.assertIn("feno_registered_mediums 2", text)
        self.assertIn("# TYPE feno_worker_requests_total counter", text)
        self.assertIn("# TYPE feno_worker_requests_in_flight gauge", text)
        self.assertIn("feno_worker_d2h_bytes_total", text)
        self.assertIn('cache="medium_gpu"', text)
        self.assertIn("feno_cache_capacity_bytes", text)
        self.assertIn("feno_medium_tier_transfers_total", text)
        self.assertIn("feno_medium_source_rebuilds_total", text)

    async def test_trace_record_count_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = RequestTraceRecorder(directory, max_records=1)
            recorder.start(name="bounded")
            self.assertTrue(recorder.record({"value": 1}, {"status": "ok"}))
            with self.assertRaisesRegex(RuntimeError, "record limit"):
                recorder.record({"value": 2}, {"status": "ok"})
            self.assertEqual(recorder.stop()["records"], 1)


class Stage6HTTPControlTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.engine = FakeStage6Engine()
        self.recorder = RequestTraceRecorder(root / "traces")
        self.profiler = RuntimeProfiler(
            root / "profiles",
            include_cuda=False,
            profiler_factory=lambda _activities: FakeProfile(),
        )
        self.service = FENOHTTPService(
            self.engine,
            trace_recorder=self.recorder,
            profiler=self.profiler,
        )
        self.client = TestClient(TestServer(self.service.create_app()))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()
        self.temporary.cleanup()

    async def _prepare(self):
        response = await self.client.post(
            "/v1/mediums",
            json={
                "medium_id": "stage6-medium",
                "velocity": [[1.0, 2.0], [3.0, 4.0]],
                "replicas": 2,
            },
        )
        self.assertEqual(response.status, 201)

    async def test_metrics_trace_profiler_and_cache_flush_endpoints(self):
        await self._prepare()
        trace_start = await self.client.post(
            "/v1/traces/start",
            json={"name": "http-trace", "metadata": {"test": True}},
        )
        self.assertEqual(trace_start.status, 201, await trace_start.text())
        profile_start = await self.client.post(
            "/v1/profiler/start", json={"name": "http-profile"}
        )
        self.assertEqual(profile_start.status, 201, await profile_start.text())

        inference = await self.client.post(
            "/v1/infer",
            json={
                "context_id": "stage6-medium",
                "source_position": [2.0, 3.0],
                "frequency": 5.0,
            },
        )
        self.assertEqual(inference.status, 200)
        self.assertEqual((await inference.json())["output"], [7.0, 8.0])

        profile_stop = await self.client.post("/v1/profiler/stop")
        self.assertEqual(profile_stop.status, 200)
        trace_stop = await self.client.post("/v1/traces/stop")
        self.assertEqual(trace_stop.status, 200)
        trace_body = await trace_stop.json()
        self.assertEqual(trace_body["records"], 1)
        trace = load_trace(Path(self.temporary.name) / "traces" / trace_body["file"])
        self.assertEqual(trace[-1]["result"]["checksum"], 15.0)

        flush = await self.client.post(
            "/v1/cache/flush",
            json={
                "scope": "medium",
                "context_id": "stage6-medium",
                "worker_ids": ["gpu-0"],
            },
        )
        self.assertEqual(flush.status, 200, await flush.text())
        self.assertEqual(self.engine.flushes[-1]["worker_ids"], ("gpu-0",))

        metrics = await self.client.get("/metrics")
        self.assertEqual(metrics.status, 200)
        text = await metrics.text()
        self.assertIn("feno_http_requests_total", text)
        self.assertIn("feno_admin_operations_total", text)
        self.assertIn("feno_registered_mediums 1", text)

    async def test_busy_cache_flush_is_rejected(self):
        await self._prepare()
        self.engine.result_gate = asyncio.Event()
        inference = asyncio.create_task(
            self.client.post(
                "/v1/infer",
                json={
                    "context_id": "stage6-medium",
                    "source_position": [1.0, 2.0],
                    "frequency": 3.0,
                },
            )
        )
        await self.engine.submit_started.wait()
        flush = await self.client.post(
            "/v1/cache/flush",
            json={"scope": "medium", "context_id": "stage6-medium"},
        )
        self.assertEqual(flush.status, 409)
        self.assertEqual((await flush.json())["error"]["code"], "cache_busy")
        self.engine.result_gate.set()
        self.assertEqual((await inference).status, 200)

    async def test_global_flush_blocks_new_medium_preparation(self):
        await self._prepare()
        self.engine.flush_gate = asyncio.Event()
        flushing = asyncio.create_task(
            self.client.post("/v1/cache/flush", json={"scope": "all"})
        )
        await self.engine.flush_started.wait()
        preparing = await self.client.post(
            "/v1/mediums",
            json={
                "medium_id": "blocked-medium",
                "velocity": [[1.0, 2.0], [3.0, 4.0]],
            },
        )
        self.assertEqual(preparing.status, 409)
        self.assertEqual((await preparing.json())["error"]["code"], "maintenance_busy")
        self.engine.flush_gate.set()
        self.assertEqual((await flushing).status, 200)


if __name__ == "__main__":
    unittest.main()
