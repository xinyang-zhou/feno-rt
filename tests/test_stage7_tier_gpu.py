"""Single-GPU Stage 7.1 integration test pinned to physical cuda:0."""

from dataclasses import fields, is_dataclass
import gc
import os
import unittest

import torch

from feno_rt.config import FENOModelConfig
from feno_rt.models.feno_freq import FENOFreq
from feno_rt.runtime import (
    DynamicBatchConfig,
    FENOModelRunner,
    GPUWorker,
    MediumTierConfig,
    MultiGPUFENOEngine,
    value_nbytes,
)


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


def contains_cuda_tensor(value, seen=None) -> bool:
    if seen is None:
        seen = set()
    if id(value) in seen:
        return False
    seen.add(id(value))
    if torch.is_tensor(value):
        return value.device.type == "cuda"
    if is_dataclass(value) and not isinstance(value, type):
        return any(
            contains_cuda_tensor(getattr(value, item.name), seen)
            for item in fields(value)
        )
    if isinstance(value, dict):
        return any(contains_cuda_tensor(item, seen) for item in value.values())
    if isinstance(value, (tuple, list, set)):
        return any(contains_cuda_tensor(item, seen) for item in value)
    return False


@unittest.skipUnless(
    not os.environ.get("CUDA_VISIBLE_DEVICES")
    and torch.cuda.is_available()
    and torch.cuda.device_count() >= 1,
    "requires direct access to physical cuda:0",
)
class Stage7TierGPUIntegrationTest(unittest.IsolatedAsyncioTestCase):
    async def test_tensor_free_handle_tiers_and_flush_rebuild(self) -> None:
        config = small_config()
        torch.manual_seed(73)
        runner = FENOModelRunner(
            FENOFreq(config.encoder_config(), config.decoder_config()),
            config=config,
            device="cuda:0",
            model_version="stage7.1-gpu-test",
        )
        velocity = torch.linspace(-1.0, 1.0, 16 * 16).reshape(16, 16)
        sample = runner.prepare_medium(
            velocity, already_normalized=True, use_cache=False
        )
        entry_bytes = value_nbytes(sample)
        del sample
        gc.collect()
        torch.cuda.empty_cache()

        worker = GPUWorker(
            "gpu-0",
            runner,
            DynamicBatchConfig(
                max_wait_us=0,
                max_batch_size=2,
                max_query_tokens=16,
                max_activation_bytes=10_000_000,
                max_output_bytes=10_000_000,
            ),
            medium_tier_config=MediumTierConfig(
                gpu_capacity_bytes=entry_bytes,
                pinned_cpu_capacity_bytes=2 * entry_bytes,
            ),
        )
        engine = MultiGPUFENOEngine(
            [worker],
            routing_policy="cache_aware",
            return_cpu=True,
        )
        contexts = []
        try:
            for index in range(3):
                contexts.append(
                    await engine.prepare_medium(
                        velocity + index * 0.01,
                        medium_id=f"tier-{index}",
                        already_normalized=True,
                    )
                )

            self.assertTrue(all(not contains_cuda_tensor(item) for item in contexts))
            initial = worker.medium_tiers.snapshot()
            self.assertEqual(initial["registered_entries"], 3)
            self.assertEqual(initial["gpu"]["resident_entries"], 1)
            self.assertEqual(initial["pinned_cpu"]["resident_entries"], 2)
            self.assertEqual(
                initial["pinned_cpu"]["actual_pinned_entries"],
                initial["pinned_cpu"]["resident_entries"],
            )

            first = await engine.infer(contexts[0], [4.0, 5.0], 10.0)
            self.assertTrue(torch.isfinite(first).all())
            promoted = worker.medium_tiers.snapshot()
            self.assertGreaterEqual(promoted["transfers"]["promotions"], 1)
            self.assertLessEqual(
                promoted["gpu"]["resident_bytes"],
                promoted["gpu"]["capacity_bytes"],
            )

            flushed = await engine.flush_caches(contexts[0])
            self.assertEqual(
                flushed["gpu-0"]["source_registration"], 0
            )
            handle = contexts[0].replicas["gpu-0"]
            self.assertTrue(worker.medium_tiers.residency(handle)["registered"])
            self.assertFalse(worker.medium_tiers.residency(handle)["gpu"])

            rebuilt = await engine.infer(contexts[0], [4.0, 5.0], 10.0)
            torch.testing.assert_close(first, rebuilt)
            self.assertGreater(
                worker.medium_tiers.snapshot()["source"]["rebuilds"],
                initial["source"]["rebuilds"],
            )

            released = await engine.release_medium(contexts[0])
            self.assertEqual(
                released["gpu-0"]["source_registration"], 1
            )
        finally:
            await engine.close()


if __name__ == "__main__":
    unittest.main()
