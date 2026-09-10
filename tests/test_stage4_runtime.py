"""Unit tests for Stage 4 precision, SDPA, and compile controls."""

import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from feno_rt.config import FENOModelConfig
from feno_rt.models.feno_freq import FENOFreq
from feno_rt.runtime import FENOModelRunner, PrecisionMode, PrecisionPolicy


def _config() -> FENOModelConfig:
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


class Stage4RuntimeTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(4)
        self.config = _config()
        self.runner = FENOModelRunner(
            FENOFreq(
                self.config.encoder_config(),
                self.config.decoder_config(),
            ),
            config=self.config,
            device="cpu",
        )
        self.velocity = torch.linspace(1500.0, 2500.0, 16 * 16).reshape(16, 16)
        self.sources = torch.tensor([[2.0, 3.0], [8.0, 10.0]])
        self.frequencies = torch.tensor([10.0, 25.0])

    def test_precision_policy_and_cache_identity(self) -> None:
        self.assertEqual(PrecisionMode.parse("BF16"), PrecisionMode.BF16)
        self.assertEqual(self.runner.precision_signature, "fp32")
        self.assertEqual(
            self.runner.execution_config()["decoder_static_precision"], "fp32"
        )
        context = self.runner.prepare_medium(self.velocity, already_normalized=True)
        keys = self.runner.request_cache_keys(
            context,
            self.sources[:1],
            self.frequencies[:1],
        )
        self.assertEqual(keys["medium"].dtype, "fp32")
        self.assertEqual(keys["wavelet"].dtype, "fp32")
        with self.assertRaisesRegex(ValueError, "requires a CUDA device"):
            PrecisionPolicy.create("tf32", "cpu")

    def test_sdpa_math_matches_auto_on_cached_path(self) -> None:
        context = self.runner.prepare_medium(self.velocity, already_normalized=True)
        expected = self.runner.forward_batch(
            context, self.sources, self.frequencies, cache_level="all"
        )
        self.runner.set_sdpa_backend("math")
        actual = self.runner.forward_batch(
            context, self.sources, self.frequencies, cache_level="all"
        )
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        self.assertEqual(self.runner.execution_config()["sdpa_backend"], "math")
        with self.assertRaisesRegex(ValueError, "sdpa_backend"):
            self.runner.set_sdpa_backend("unknown")

    def test_torch_compile_eager_backend_preserves_output(self) -> None:
        context = self.runner.prepare_medium(self.velocity, already_normalized=True)
        expected = self.runner.forward_batch(
            context, self.sources, self.frequencies, cache_level="all"
        )
        config = self.runner.enable_torch_compile(
            backend="eager", mode=None, dynamic=False
        )
        actual = self.runner.forward_batch(
            context, self.sources, self.frequencies, cache_level="all"
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue(config["enabled"])
        self.runner.disable_torch_compile()
        self.assertFalse(self.runner.execution_config()["torch_compile"]["enabled"])


if __name__ == "__main__":
    unittest.main()
