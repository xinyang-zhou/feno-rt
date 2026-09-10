"""End-to-end correctness tests for all Stage-2 cache levels."""

import unittest

import torch

from feno_rt.config import FENOModelConfig
from feno_rt.models.feno_freq import FENOFreq
from feno_rt.preprocessing import NormalizationStats
from feno_rt.runtime import FENOModelRunner


class Stage2CacheTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = FENOModelConfig(
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
        cls.normalization = NormalizationStats(
            v_mean=2000.0,
            v_std=500.0,
            seis_mean=0.25,
            seis_std=2.0,
        )
        torch.manual_seed(7)
        model = FENOFreq(cls.config.encoder_config(), cls.config.decoder_config())
        cls.runner = FENOModelRunner(
            model,
            config=cls.config,
            normalization=cls.normalization,
            device="cpu",
            model_version="stage2-test-model",
        )
        cls.velocity = torch.linspace(1500.0, 2500.0, 16 * 16).reshape(16, 16)

    def setUp(self) -> None:
        self.runner.clear_caches(force=True)
        self.runner.reset_cache_stats()

    def test_all_cache_levels_match_original_decoder(self) -> None:
        sources = torch.tensor([[2.0, 3.0], [8.0, 10.0], [2.0, 3.0]])
        frequencies = torch.tensor([10.0, 25.0, 10.0])
        context = self.runner.prepare_medium(self.velocity)
        expected = self.runner.forward_batch(
            context, sources, frequencies, cache_level="medium"
        )

        for cache_level in ("decoder", "geometry", "all"):
            with self.subTest(cache_level=cache_level):
                actual = self.runner.forward_batch(
                    context, sources, frequencies, cache_level=cache_level
                )
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_medium_decoder_geometry_and_wavelet_hits_are_recorded(self) -> None:
        first_context = self.runner.prepare_medium(self.velocity)
        second_context = self.runner.prepare_medium(self.velocity)
        self.assertIs(first_context.latent, second_context.latent)

        sources = torch.tensor([[2.0, 3.0], [2.0, 3.0], [9.0, 7.0]])
        frequencies = torch.tensor([10.0, 10.0, 25.0])
        self.runner.forward_batch(first_context, sources, frequencies, cache_level="all")
        self.runner.forward_batch(first_context, sources, frequencies, cache_level="all")
        metrics = self.runner.cache_metrics()

        self.assertEqual(metrics["caches"]["medium"]["hits"], 1)
        self.assertGreaterEqual(metrics["caches"]["decoder_context"]["hits"], 1)
        self.assertEqual(metrics["caches"]["geometry_prefix"]["resident_entries"], 2)
        self.assertEqual(metrics["caches"]["wavelet"]["resident_entries"], 2)
        self.assertGreaterEqual(metrics["caches"]["geometry_prefix"]["hits"], 2)
        self.assertGreaterEqual(metrics["caches"]["wavelet"]["hits"], 2)
        self.assertEqual(metrics["runtime"]["geometry_in_batch_dedup_hits"], 2)
        self.assertEqual(metrics["runtime"]["wavelet_in_batch_dedup_hits"], 2)

    def test_exact_float_frequency_is_part_of_wavelet_key(self) -> None:
        context = self.runner.prepare_medium(self.velocity)
        source = torch.tensor([[2.0, 3.0], [2.0, 3.0]])
        frequency = torch.tensor(10.0, dtype=torch.float32)
        adjacent = torch.nextafter(frequency, torch.tensor(float("inf")))

        self.runner.forward_batch(
            context,
            source,
            torch.stack((frequency, adjacent)),
            cache_level="all",
        )

        self.assertEqual(
            self.runner.cache_metrics()["caches"]["wavelet"]["resident_entries"],
            2,
        )

    def test_cascade_invalidation_removes_medium_dependent_state(self) -> None:
        context = self.runner.prepare_medium(self.velocity)
        self.runner.forward_batch(
            context,
            [[2.0, 3.0]],
            [10.0],
            cache_level="all",
        )
        invalidated = self.runner.caches.invalidate_medium(context.cache_key)

        self.assertEqual(invalidated["medium"], 1)
        self.assertEqual(invalidated["decoder_context"], 1)
        self.assertEqual(invalidated["geometry_prefix"], 1)
        self.assertEqual(
            self.runner.cache_metrics()["caches"]["wavelet"]["resident_entries"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
