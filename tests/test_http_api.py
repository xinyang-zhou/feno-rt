"""HTTP JSON/NumPy contract and lifecycle tests without a GPU dependency."""

import asyncio
from dataclasses import dataclass
from io import BytesIO
import unittest

import numpy as np
import torch
from aiohttp.test_utils import TestClient, TestServer

from feno_rt.serving import FENOHTTPService


@dataclass
class FakeContext:
    medium_id: str
    worker_ids: tuple = ("gpu-1", "gpu-2")


class FakeHandle:
    def __init__(self, output, request_id="request-1", gate=None):
        self.output = output
        self.request_id = request_id
        self.worker_id = "gpu-1"
        self.gate = gate

    async def result(self):
        if self.gate is not None:
            await self.gate.wait()
        return self.output

    def __await__(self):
        return self.result().__await__()


class FakeEngine:
    def __init__(self):
        self.started = 0
        self.closed = 0
        self.released = []
        self.submissions = []
        self.submit_started = asyncio.Event()
        self.result_gate = None

    async def start(self):
        self.started += 1

    async def close(self):
        self.closed += 1

    async def prepare_medium(
        self, velocity, *, medium_id=None, already_normalized=False, replicas=1
    ):
        array = np.asarray(velocity)
        if array.ndim != 2:
            raise ValueError("velocity must be rank two")
        return FakeContext(medium_id or "derived-medium", ("gpu-1",) * replicas)

    async def release_medium(self, context):
        self.released.append(context.medium_id)
        return {"gpu-1": {"medium": 1}}

    async def submit(self, context, source_position, frequency, **kwargs):
        self.submissions.append((context, source_position, frequency, kwargs))
        self.submit_started.set()
        output = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        return FakeHandle(
            output,
            request_id=kwargs.get("request_id") or "request-1",
            gate=self.result_gate,
        )

    def stats(self):
        return {"submitted": len(self.submissions), "workers": 2}


class HTTPAPITest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = FakeEngine()
        self.service = FENOHTTPService(
            self.engine, max_json_output_elements=100
        )
        self.client = TestClient(TestServer(self.service.create_app()))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()

    async def _prepare_json(self, medium_id="medium-json"):
        return await self.client.post(
            "/v1/mediums",
            json={
                "medium_id": medium_id,
                "velocity": [[1.0, 2.0], [3.0, 4.0]],
                "already_normalized": True,
                "replicas": 2,
            },
        )

    async def test_json_prepare_infer_stats_and_delete(self):
        health = await self.client.get("/health")
        self.assertEqual(health.status, 200)
        self.assertEqual((await health.json())["status"], "ok")

        prepared = await self._prepare_json()
        self.assertEqual(prepared.status, 201)
        self.assertEqual((await prepared.json())["context_id"], "medium-json")

        duplicate = await self._prepare_json()
        self.assertEqual(duplicate.status, 409)

        response = await self.client.post(
            "/v1/infer",
            json={
                "context_id": "medium-json",
                "source_position": [2.0, 3.0],
                "frequency": 10.0,
                "request_id": "http-request",
                "response_format": "json",
            },
        )
        self.assertEqual(response.status, 200)
        payload = await response.json()
        self.assertEqual(payload["shape"], [2, 3])
        self.assertEqual(payload["output"], [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
        self.assertEqual(response.headers["X-FENO-Worker-ID"], "gpu-1")

        stats = await self.client.get("/v1/stats")
        self.assertEqual((await stats.json())["registry"]["medium-json"]["inflight"], 0)
        deleted = await self.client.delete("/v1/mediums/medium-json")
        self.assertEqual(deleted.status, 200)
        self.assertEqual(self.engine.released, ["medium-json"])

    async def test_npy_medium_and_npy_result(self):
        buffer = BytesIO()
        np.save(buffer, np.ones((2, 2), dtype=np.float32), allow_pickle=False)
        prepared = await self.client.post(
            "/v1/mediums?medium_id=medium-npy&already_normalized=true&replicas=1",
            data=buffer.getvalue(),
            headers={"Content-Type": "application/x-npy"},
        )
        self.assertEqual(prepared.status, 201)

        response = await self.client.post(
            "/v1/infer",
            json={
                "context_id": "medium-npy",
                "source_position": [2.0, 3.0],
                "frequency": 25.0,
                "response_format": "npy",
            },
        )
        self.assertEqual(response.status, 200)
        self.assertEqual(response.content_type, "application/x-npy")
        output = np.load(BytesIO(await response.read()), allow_pickle=False)
        np.testing.assert_array_equal(output, np.arange(6, dtype=np.float32).reshape(2, 3))

    async def test_unknown_context_and_invalid_body_are_structured_errors(self):
        missing = await self.client.post(
            "/v1/infer",
            json={
                "context_id": "missing",
                "source_position": [1.0, 2.0],
                "frequency": 10.0,
            },
        )
        self.assertEqual(missing.status, 404)
        self.assertEqual((await missing.json())["error"]["code"], "not_found")
        invalid = await self.client.post("/v1/mediums", json={"medium_id": "bad"})
        self.assertEqual(invalid.status, 400)
        self.assertEqual((await invalid.json())["error"]["code"], "invalid_request")

        await self._prepare_json("validation")
        submitted_before = len(self.engine.submissions)
        invalid_format = await self.client.post(
            "/v1/infer",
            json={
                "context_id": "validation",
                "source_position": [1.0, 2.0],
                "frequency": 10.0,
                "response_format": "csv",
            },
        )
        self.assertEqual(invalid_format.status, 400)
        invalid_boolean = await self.client.post(
            "/v1/infer",
            json={
                "context_id": "validation",
                "source_position": [1.0, 2.0],
                "frequency": 10.0,
                "denormalize": "false",
            },
        )
        self.assertEqual(invalid_boolean.status, 400)
        self.assertEqual(len(self.engine.submissions), submitted_before)

    async def test_busy_context_cannot_be_deleted(self):
        await self._prepare_json("busy")
        self.engine.result_gate = asyncio.Event()
        inference = asyncio.create_task(
            self.client.post(
                "/v1/infer",
                json={
                    "context_id": "busy",
                    "source_position": [1.0, 2.0],
                    "frequency": 10.0,
                },
            )
        )
        await self.engine.submit_started.wait()
        deletion = await self.client.delete("/v1/mediums/busy")
        self.assertEqual(deletion.status, 409)
        self.assertEqual((await deletion.json())["error"]["code"], "context_busy")
        self.engine.result_gate.set()
        self.assertEqual((await inference).status, 200)

    async def test_cleanup_closes_owned_engine(self):
        self.assertEqual(self.engine.started, 1)
        await self.client.close()
        self.assertEqual(self.engine.closed, 1)


if __name__ == "__main__":
    unittest.main()
