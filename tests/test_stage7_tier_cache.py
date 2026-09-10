"""CPU-only policy and lifecycle tests for Stage 7.1 medium tiers."""

from dataclasses import fields
import unittest

import torch

from feno_rt.config import FENOModelConfig
from feno_rt.models.feno_freq import FENOFreq
from feno_rt.runtime import (
    FENOModelRunner,
    MediumTierConfig,
    MediumTierHandle,
    MediumTierManager,
    UnknownMediumHandleError,
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
        decoder_depth=1,
        decoder_heads=4,
        fno_modes1=4,
        fno_modes2=4,
        fno_width=16,
        patch_size=2,
        position_embedding_dim=8,
        frequency_condition_dim=8,
        dropout_rate=0.0,
    )


class MediumTierManagerTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(71)
        config = small_config()
        self.runner = FENOModelRunner(
            FENOFreq(config.encoder_config(), config.decoder_config()),
            config=config,
            device="cpu",
            model_version="stage7.1-test",
        )
        self.velocities = [
            torch.full((16, 16), float(index), dtype=torch.float32)
            for index in range(1, 5)
        ]
        sample = self.runner.prepare_medium(
            self.velocities[0],
            already_normalized=True,
            use_cache=False,
        )
        self.entry_bytes = value_nbytes(sample)

    def manager(self, gpu_entries: int, cpu_entries: int) -> MediumTierManager:
        return MediumTierManager(
            self.runner,
            MediumTierConfig(
                gpu_capacity_bytes=gpu_entries * self.entry_bytes,
                pinned_cpu_capacity_bytes=cpu_entries * self.entry_bytes,
                pin_cpu_memory=False,
                recency_half_life_accesses=8.0,
            ),
        )

    def prepare(
        self, manager: MediumTierManager, index: int
    ) -> MediumTierHandle:
        return manager.prepare(
            self.velocities[index],
            medium_id=f"medium-{index}",
            already_normalized=True,
        )

    def test_handle_is_tensor_free_and_stable(self) -> None:
        manager = self.manager(1, 1)
        handle = self.prepare(manager, 0)

        self.assertTrue(all(not torch.is_tensor(getattr(handle, item.name)) for item in fields(handle)))
        self.assertEqual(handle.medium_id, "medium-0")
        self.assertTrue(manager.residency(handle)["gpu"])

    def test_gpu_eviction_demotes_and_pinned_hit_promotes(self) -> None:
        manager = self.manager(1, 2)
        first = self.prepare(manager, 0)
        second = self.prepare(manager, 1)

        self.assertTrue(manager.residency(first)["pinned_cpu"])
        self.assertTrue(manager.residency(second)["gpu"])
        lease = manager.acquire(first)
        self.assertEqual(lease.value.medium_id, "medium-0")
        self.assertTrue(manager.residency(first)["gpu"])
        lease.release()

        snapshot = manager.snapshot()
        self.assertEqual(snapshot["gpu"]["resident_entries"], 1)
        self.assertLessEqual(
            snapshot["gpu"]["resident_bytes"], snapshot["gpu"]["capacity_bytes"]
        )
        self.assertEqual(snapshot["pinned_cpu"]["hits"], 1)
        self.assertGreaterEqual(snapshot["transfers"]["demotions"], 2)
        self.assertEqual(snapshot["transfers"]["promotions"], 1)

    def test_lease_blocks_inflight_demotion_and_trims_overflow(self) -> None:
        manager = self.manager(1, 2)
        first = self.prepare(manager, 0)
        lease = manager.acquire(first)

        second = self.prepare(manager, 1)

        self.assertTrue(manager.residency(first)["gpu"])
        self.assertFalse(manager.residency(second)["gpu"])
        self.assertGreaterEqual(manager.snapshot()["gpu"]["capacity_overflows"], 1)
        lease.release()
        self.assertLessEqual(
            manager.snapshot()["gpu"]["resident_bytes"], self.entry_bytes
        )

    def test_cost_frequency_recency_policy_keeps_hot_entry(self) -> None:
        manager = self.manager(2, 2)
        hot = self.prepare(manager, 0)
        cold = self.prepare(manager, 1)
        for _ in range(5):
            lease = manager.acquire(hot)
            lease.release()

        newest = self.prepare(manager, 2)

        self.assertTrue(manager.residency(hot)["gpu"])
        self.assertTrue(manager.residency(newest)["gpu"])
        self.assertFalse(manager.residency(cold)["gpu"])
        self.assertEqual(
            manager.snapshot()["policy"], "cost_frequency_recency_per_byte"
        )

    def test_flush_retains_source_registration_for_rebuild(self) -> None:
        manager = self.manager(1, 1)
        handle = self.prepare(manager, 0)
        before = manager.snapshot()["source"]["rebuilds"]

        counts = manager.clear_materialized()

        self.assertEqual(counts["gpu_medium"], 1)
        self.assertTrue(manager.residency(handle)["registered"])
        self.assertFalse(manager.residency(handle)["gpu"])
        lease = manager.acquire(handle)
        lease.release()
        self.assertEqual(manager.snapshot()["source"]["rebuilds"], before + 1)

    def test_invalidation_never_removes_a_leased_entry(self) -> None:
        manager = self.manager(1, 1)
        handle = self.prepare(manager, 0)
        lease = manager.acquire(handle)
        self.assertEqual(manager.invalidate(handle, force=True)["source_registration"], 0)
        lease.release()
        self.assertEqual(manager.invalidate(handle)["source_registration"], 1)
        with self.assertRaises(UnknownMediumHandleError):
            manager.acquire(handle)

    def test_config_rejects_unusable_capacities(self) -> None:
        with self.assertRaises(ValueError):
            MediumTierConfig(gpu_capacity_bytes=0)
        with self.assertRaises(ValueError):
            MediumTierConfig(pinned_cpu_capacity_bytes=-1)


if __name__ == "__main__":
    unittest.main()
