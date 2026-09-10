"""Correctness tests for the Stage-1 cached model runner."""

import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from feno_rt.config import FENOModelConfig
from feno_rt.models.feno_freq import FENOFreq
from feno_rt.preprocessing import (
    NormalizationStats,
    build_velocity_input,
    prepare_frequencies,
    prepare_source_receiver_batch,
)
from feno_rt.runtime import FENOModelRunner


class ModelRunnerTest(unittest.TestCase):
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
        torch.manual_seed(0)
        model = FENOFreq(
            cls.config.encoder_config(),
            cls.config.decoder_config(),
        ).eval()
        cls.runner = FENOModelRunner(
            model,
            config=cls.config,
            normalization=cls.normalization,
            device="cpu",
        )
        cls.velocity = torch.linspace(1500.0, 2500.0, 16 * 16).reshape(16, 16)
        cls.sources = torch.tensor([[2.0, 3.0], [8.0, 10.0]])
        cls.frequencies = torch.tensor([10.0, 25.0])

    def test_cached_runner_matches_legacy_execution(self) -> None:
        context = self.runner.prepare_medium(self.velocity)
        actual = self.runner.forward_batch(
            context,
            self.sources,
            self.frequencies,
            cache_level="medium",
        )

        model_input = build_velocity_input(
            self.velocity,
            self.config,
            self.normalization,
            device=torch.device("cpu"),
        )
        source_batch, receiver_batch = prepare_source_receiver_batch(
            self.sources,
            self.config,
            device=torch.device("cpu"),
        )
        frequency_batch = prepare_frequencies(
            self.frequencies,
            batch_size=len(self.sources),
            device=torch.device("cpu"),
        )
        with torch.inference_mode():
            latent = self.runner.model.encoder(model_input)
            expected = self.runner.model.decoder(
                latent.expand(len(self.sources), -1, -1),
                source_batch,
                receiver_batch,
                frequency_batch,
            )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_scalar_frequency_is_broadcast(self) -> None:
        context = self.runner.prepare_medium(self.velocity)
        output = self.runner.forward_batch(context, self.sources, 20.0)
        self.assertEqual(
            tuple(output.shape),
            (2, self.config.num_receivers, self.config.output_steps),
        )

    def test_denormalization(self) -> None:
        context = self.runner.prepare_medium(self.velocity)
        normalized = self.runner.forward_batch(
            context,
            self.sources[:1],
            self.frequencies[:1],
        )
        physical = self.runner.forward_batch(
            context,
            self.sources[:1],
            self.frequencies[:1],
            denormalize=True,
        )
        expected = normalized * self.normalization.seis_std + self.normalization.seis_mean
        torch.testing.assert_close(physical, expected, rtol=0, atol=0)

    def test_rejects_frequency_count_mismatch(self) -> None:
        context = self.runner.prepare_medium(self.velocity)
        with self.assertRaisesRegex(ValueError, "Expected 2 frequencies"):
            self.runner.forward_batch(context, self.sources, [10.0, 20.0, 30.0])


if __name__ == "__main__":
    unittest.main()
