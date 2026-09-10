"""HTTP integration test pinned to physical GPUs 1 and 2 only."""

import asyncio
from io import BytesIO
import os
import unittest

from aiohttp import ClientSession
import numpy as np
import torch

from feno_rt.config import FENOModelConfig
from feno_rt.models.feno_freq import FENOFreq
from feno_rt.runtime import (
    DynamicBatchConfig,
    FENOModelRunner,
    GPUWorker,
    MultiGPUFENOEngine,
    ReplicaRouterConfig,
)
from feno_rt.serving import FENOHTTPService, start_http_server


def small_config() -> FENOModelConfig:
    return FENOModelConfig(
        original_size=16,
        velocity_height=16,
        velocity_width=16,
        latent_height=4,
        latent_width=4,
        output_steps=16,
        receiver_depth=1,
        num_receivers=8,
        encoder_dim=16,
        encoder_depth=1,
        encoder_heads=4,
        decoder_dim=16,
        decoder_depth=2,
        decoder_heads=4,
        fno_modes1=4,
        fno_modes2=4,
        fno_width=16,
        patch_size=2,
        position_embedding_dim=8,
        frequency_condition_dim=8,
        dropout_rate=0.0,
    )


def has_physical_test_pair() -> bool:
    return (
        not os.environ.get("CUDA_VISIBLE_DEVICES")
        and torch.cuda.is_available()
        and torch.cuda.device_count() >= 3
    )


@unittest.skipUnless(
    has_physical_test_pair(),
    "requires direct physical cuda:1/cuda:2 access; four-GPU tests are never used",
)
class Stage5HTTPGPUIntegrationTest(unittest.IsolatedAsyncioTestCase):
    async def test_npy_api_routes_across_exact_two_gpu_pair(self):
        config = small_config()
        torch.manual_seed(52)
        template = FENOFreq(config.encoder_config(), config.decoder_config())
        state_dict = {
            name: value.detach().cpu() for name, value in template.state_dict().items()
        }
        batch_config = DynamicBatchConfig(
            max_wait_us=5_000,
            max_batch_size=4,
            max_query_tokens=32,
            max_activation_bytes=10_000_000,
            max_output_bytes=10_000_000,
        )
        workers = []
        for worker_id, device in (("gpu-1", "cuda:1"), ("gpu-2", "cuda:2")):
            model = FENOFreq(config.encoder_config(), config.decoder_config())
            model.load_state_dict(state_dict)
            runner = FENOModelRunner(
                model,
                config=config,
                device=device,
                model_version="stage5-http-two-gpu-test",
            )
            workers.append(
                GPUWorker(
                    worker_id,
                    runner,
                    batch_config,
                    async_d2h=True,
                )
            )
        engine = MultiGPUFENOEngine(
            workers,
            routing_policy="cache_aware",
            router_config=ReplicaRouterConfig(
                decoder_cache_weight=0.0,
                geometry_cache_weight=0.0,
                wavelet_cache_weight=0.0,
                affinity_weight=0.0,
                queue_penalty=1_000.0,
            ),
            hot_medium_threshold=2,
            return_cpu=True,
        )
        service = FENOHTTPService(engine, close_engine=True)
        server = await start_http_server(service, host="127.0.0.1", port=0)
        base_url = f"http://{server.host}:{server.port}"
        velocity_buffer = BytesIO()
        np.save(
            velocity_buffer,
            np.linspace(-1.0, 1.0, 16 * 16, dtype=np.float32).reshape(16, 16),
            allow_pickle=False,
        )
        sources = [
            [2.0, 3.0],
            [8.0, 10.0],
            [4.0, 5.0],
            [11.0, 12.0],
            [2.0, 12.0],
            [12.0, 2.0],
            [6.0, 9.0],
            [9.0, 6.0],
        ]
        frequencies = [10.0, 25.0, 15.0, 20.0, 5.0, 30.0, 35.0, 40.0]

        try:
            async with ClientSession() as client:
                prepared = await client.post(
                    f"{base_url}/v1/mediums",
                    params={
                        "medium_id": "http-two-gpu-medium",
                        "already_normalized": "true",
                        "replicas": "2",
                    },
                    data=velocity_buffer.getvalue(),
                    headers={"Content-Type": "application/x-npy"},
                )
                self.assertEqual(prepared.status, 201, await prepared.text())
                prepared_body = await prepared.json()
                self.assertEqual(prepared_body["replicas"], ["gpu-1", "gpu-2"])

                async def infer(source, frequency):
                    response = await client.post(
                        f"{base_url}/v1/infer",
                        json={
                            "context_id": "http-two-gpu-medium",
                            "source_position": source,
                            "frequency": frequency,
                            "response_format": "npy",
                        },
                    )
                    if response.status != 200:
                        self.fail(
                            f"HTTP inference failed with {response.status}: "
                            f"{await response.text()}"
                        )
                    worker_id = response.headers["X-FENO-Worker-ID"]
                    output = np.load(
                        BytesIO(await response.read()), allow_pickle=False
                    )
                    return worker_id, output

                first = await asyncio.gather(
                    *(infer(source, frequency) for source, frequency in zip(sources, frequencies))
                )
                second = await asyncio.gather(
                    *(infer(source, frequency) for source, frequency in zip(sources, frequencies))
                )
                for (_, first_output), (_, second_output) in zip(first, second):
                    self.assertEqual(first_output.shape, (8, 16))
                    self.assertTrue(np.isfinite(first_output).all())
                    np.testing.assert_allclose(
                        first_output, second_output, rtol=1e-5, atol=1e-6
                    )

                stats_response = await client.get(f"{base_url}/v1/stats")
                self.assertEqual(stats_response.status, 200)
                stats = await stats_response.json()
                selections = stats["runtime"]["selection_counts"]
                self.assertGreater(selections.get("gpu-1", 0), 0)
                self.assertGreater(selections.get("gpu-2", 0), 0)

                metrics_response = await client.get(f"{base_url}/metrics")
                self.assertEqual(metrics_response.status, 200)
                metrics = await metrics_response.text()
                self.assertIn("feno_http_requests_total", metrics)
                self.assertIn('device="cuda:1"', metrics)
                self.assertIn('device="cuda:2"', metrics)

                flushed = await client.post(
                    f"{base_url}/v1/cache/flush",
                    json={
                        "scope": "medium",
                        "context_id": "http-two-gpu-medium",
                    },
                )
                self.assertEqual(flushed.status, 200, await flushed.text())
                flush_body = await flushed.json()
                self.assertEqual(
                    set(flush_body["workers"]), {"gpu-1", "gpu-2"}
                )
                after_flush_worker, after_flush_output = await infer(
                    sources[0], frequencies[0]
                )
                self.assertIn(after_flush_worker, {"gpu-1", "gpu-2"})
                np.testing.assert_allclose(
                    after_flush_output, first[0][1], rtol=1e-5, atol=1e-6
                )

                deleted = await client.delete(
                    f"{base_url}/v1/mediums/http-two-gpu-medium"
                )
                self.assertEqual(deleted.status, 200, await deleted.text())
                invalidated = (await deleted.json())["invalidated"]
                self.assertEqual(set(invalidated), {"gpu-1", "gpu-2"})

                missing = await client.post(
                    f"{base_url}/v1/infer",
                    json={
                        "context_id": "http-two-gpu-medium",
                        "source_position": sources[0],
                        "frequency": frequencies[0],
                    },
                )
                self.assertEqual(missing.status, 404)
        finally:
            await server.close()


if __name__ == "__main__":
    unittest.main()
