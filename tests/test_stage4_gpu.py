"""Explicit GPU integration tests for Stage 4 runtime features."""

import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from feno_rt.config import FENOModelConfig
from feno_rt.models.feno_freq import FENOFreq
from feno_rt.runtime import FENOModelRunner


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


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class Stage4GPUIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(8)
        self.config = _config()
        self.velocity = torch.linspace(-1.0, 1.0, 16 * 16).reshape(16, 16)
        self.sources = torch.tensor([[2.0, 3.0], [8.0, 10.0], [4.0, 5.0]])
        self.frequencies = torch.tensor([10.0, 25.0, 15.0])

    def _runner(self, precision: str = "fp32") -> FENOModelRunner:
        return FENOModelRunner(
            FENOFreq(self.config.encoder_config(), self.config.decoder_config()),
            config=self.config,
            device="cuda:0",
            precision=precision,
        )

    def test_low_precision_outputs_are_finite(self) -> None:
        for precision in ("tf32", "bf16", "fp16"):
            runner = self._runner(precision)
            context = runner.prepare_medium(self.velocity, already_normalized=True)
            static_context = runner.prepare_decoder_context(context)
            expected_static_dtype = {
                "tf32": torch.float32,
                "bf16": torch.bfloat16,
                "fp16": torch.float32,
            }[precision]
            self.assertEqual(static_context.layers[-1].tokens.dtype, expected_static_dtype)
            output = runner.forward_batch(
                context, self.sources, self.frequencies, cache_level="all"
            )
            self.assertTrue(torch.isfinite(output).all().item(), precision)

    def test_flash_sdpa_and_cuda_graph_replay(self) -> None:
        runner = self._runner("bf16")
        context = runner.prepare_medium(self.velocity, already_normalized=True)
        runner.set_sdpa_backend("flash")
        expected = runner.forward_batch(
            context, self.sources, self.frequencies, cache_level="all"
        )
        runner.enable_cuda_graphs(buckets=(1, 4), fallback_on_error=False)
        first = runner.forward_batch(
            context, self.sources, self.frequencies, cache_level="all"
        )
        second = runner.forward_batch(
            context, self.sources, self.frequencies, cache_level="all"
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(first, expected, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(second, expected, rtol=1e-5, atol=1e-5)
        metrics = runner.execution_config()["cuda_graph"]
        self.assertEqual(metrics["captures"], 1)
        self.assertEqual(metrics["replays"], 2)
        self.assertEqual(metrics["padded_slots"], 2)
        self.assertGreater(metrics["static_buffer_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
