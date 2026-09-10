"""Two-GPU Stage 5 integration test pinned to physical GPUs 1 and 2."""

import asyncio
import os
from pathlib import Path
import tempfile
import unittest

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


def small_config():
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


def has_physical_test_pair():
    return (
        not os.environ.get("CUDA_VISIBLE_DEVICES")
        and torch.cuda.is_available()
        and torch.cuda.device_count() >= 3
    )


@unittest.skipUnless(
    has_physical_test_pair(),
    "requires direct access to physical cuda:1 and cuda:2; four-GPU tests are not used",
)
class Stage5TwoGPUIntegrationTest(unittest.IsolatedAsyncioTestCase):
    async def test_checkpoint_workers_use_processes_and_overflow_safely(self):
        config = small_config()
        torch.manual_seed(53)
        model = FENOFreq(config.encoder_config(), config.decoder_config())
        velocity = torch.linspace(-1.0, 1.0, 16 * 16).reshape(16, 16)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "small-feno.pth"
            torch.save(model.state_dict(), checkpoint)
            engine = MultiGPUFENOEngine.from_checkpoint(
                checkpoint,
                devices=("cuda:1", "cuda:2"),
                config=config,
                batch_config=DynamicBatchConfig(
                    max_wait_us=0,
                    max_batch_size=1,
                    max_query_tokens=8,
                    max_activation_bytes=10_000_000,
                    max_output_bytes=10_000_000,
                ),
                routing_policy="round_robin",
                return_cpu=True,
            )
            outputs = []
            first_snapshot = None
            try:
                context = await engine.prepare_medium(
                    velocity,
                    medium_id="process-overflow",
                    already_normalized=True,
                    replicas=2,
                )
                for index in range(5):
                    output = await asyncio.wait_for(
                        engine.infer(
                            context,
                            [float(2 + index), float(3 + index)],
                            float(10 + index),
                        ),
                        timeout=30.0,
                    )
                    outputs.append(output)
                    if first_snapshot is None:
                        first_snapshot = output.clone()
                stats = engine.stats()

                self.assertTrue(all(torch.isfinite(output).all() for output in outputs))
                self.assertIsNotNone(first_snapshot)
                torch.testing.assert_close(outputs[0], first_snapshot)
                self.assertEqual(
                    stats["worker_execution_modes"],
                    {"gpu-0": "process", "gpu-1": "process"},
                )
                self.assertTrue(
                    all(
                        worker["shared_output"]["cuda_host_registered"]
                        for worker in stats["workers"]
                    )
                )
                self.assertGreaterEqual(
                    sum(
                        worker["shared_output"]["overflow_transfers"]
                        for worker in stats["workers"]
                    ),
                    1,
                )
            finally:
                outputs.clear()
                await engine.close()

    async def test_replica_workers_hot_copy_and_async_delivery(self):
        config = small_config()
        torch.manual_seed(51)
        template = FENOFreq(config.encoder_config(), config.decoder_config())
        state_dict = {name: value.detach().cpu() for name, value in template.state_dict().items()}
        workers = []
        for worker_id, device in (("gpu-1", "cuda:1"), ("gpu-2", "cuda:2")):
            model = FENOFreq(config.encoder_config(), config.decoder_config())
            model.load_state_dict(state_dict)
            runner = FENOModelRunner(
                model,
                config=config,
                device=device,
                model_version="stage5-two-gpu-test",
            )
            workers.append(
                GPUWorker(
                    worker_id,
                    runner,
                    DynamicBatchConfig(
                        max_wait_us=5_000,
                        max_batch_size=4,
                        max_query_tokens=32,
                        max_activation_bytes=10_000_000,
                        max_output_bytes=10_000_000,
                    ),
                    async_d2h=True,
                )
            )

        router_config = ReplicaRouterConfig(
            decoder_cache_weight=0.0,
            geometry_cache_weight=0.0,
            wavelet_cache_weight=0.0,
            queue_penalty=100.0,
        )
        engine = MultiGPUFENOEngine(
            workers,
            routing_policy="cache_aware",
            router_config=router_config,
            hot_medium_threshold=2,
            return_cpu=True,
        )
        velocity = torch.linspace(-1.0, 1.0, 16 * 16).reshape(16, 16)
        sources = [[2.0, 3.0], [8.0, 10.0], [4.0, 5.0], [11.0, 12.0]]
        frequencies = [10.0, 25.0, 15.0, 20.0]

        try:
            context = await engine.prepare_medium(
                velocity,
                medium_id="stage5-medium",
                already_normalized=True,
                replicas=1,
            )
            self.assertEqual(context.worker_ids, ("gpu-1",))

            warm_handles = await asyncio.gather(
                engine.submit(context, sources[0], frequencies[0]),
                engine.submit(context, sources[1], frequencies[1]),
            )
            warm_outputs = await asyncio.gather(*warm_handles)
            await engine.wait_for_replication(context)
            self.assertEqual(context.worker_ids, ("gpu-1", "gpu-2"))

            handles = await asyncio.gather(
                *[
                    engine.submit(context, source, frequency)
                    for source, frequency in zip(sources[2:], frequencies[2:])
                ]
            )
            outputs = warm_outputs + list(await asyncio.gather(*handles))
            resident_context, tier_lease = await workers[0].acquire_medium(
                context.replicas["gpu-1"]
            )
            try:
                expected = workers[0].runner.forward_batch(
                    resident_context, sources, frequencies
                ).detach().cpu()
            finally:
                await workers[0].release_medium_lease(tier_lease)
            torch.testing.assert_close(
                torch.stack(outputs), expected, rtol=1e-5, atol=1e-6
            )

            stats = engine.stats()
            self.assertEqual(stats["completed"], 4)
            self.assertEqual(stats["failed"], 0)
            self.assertEqual(stats["medium_replications"], 2)
            self.assertGreater(stats["selection_counts"].get("gpu-1", 0), 0)
            self.assertGreater(stats["selection_counts"].get("gpu-2", 0), 0)
            self.assertTrue(all(output.device.type == "cpu" for output in outputs))
            self.assertTrue(all(output.is_pinned() for output in outputs))
            self.assertEqual(sum(item["d2h"]["copies"] for item in stats["workers"]), 4)
        finally:
            await engine.close()


if __name__ == "__main__":
    unittest.main()
