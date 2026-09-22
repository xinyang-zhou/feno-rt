"""GPU integration test for CUDA Graph capture and replay."""

import sys
import unittest
import asyncio
import gc
from unittest.mock import patch
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from feno_rt.config import FENOModelConfig
from feno_rt.models.feno_freq import FENOFreq
from feno_rt.runtime import FENOModelRunner, AsyncFENOEngine, DynamicBatchConfig


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
class CUDAGraphGPUIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(8)
        self.config = _config()
        self.velocity = torch.linspace(-1.0, 1.0, 16 * 16).reshape(16, 16)
        self.sources = torch.tensor([[2.0, 3.0], [8.0, 10.0], [4.0, 5.0]])
        self.frequencies = torch.tensor([10.0, 25.0, 15.0])

    def _runner(self) -> FENOModelRunner:
        return FENOModelRunner(
            FENOFreq(self.config.encoder_config(), self.config.decoder_config()),
            config=self.config,
            device="cuda:0",
        )

    def test_capture_and_replay_match_eager_output(self) -> None:
        runner = self._runner()
        context = runner.prepare_medium(self.velocity, already_normalized=True)
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

    def test_replay_preserves_owned_outputs_and_allocations_stabilize(self):
        runner = self._runner()
        context = runner.prepare_medium(self.velocity, already_normalized=True)
        runner.enable_cuda_graphs(buckets=(4,), fallback_on_error=False)
        first = runner.forward_batch(context, self.sources, self.frequencies)
        saved = first.clone()
        changed = self.frequencies + 1
        # Populate all cache signatures and allocator paths before the check.
        result = runner.forward_batch(context, self.sources, changed)
        del result
        gc.collect()
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        for _ in range(30):
            result = runner.forward_batch(context, self.sources, changed)
            del result
        gc.collect()
        torch.cuda.synchronize()
        self.assertLessEqual(torch.cuda.memory_allocated(), before + 4096)
        torch.testing.assert_close(first, saved, rtol=0, atol=0)
        self.assertEqual(runner.execution_config()["cuda_graph"]["resident_graphs"], 1)

    def test_capture_failure_falls_back_and_is_not_retried(self):
        runner = self._runner()
        context = runner.prepare_medium(self.velocity, already_normalized=True)
        expected = runner.forward_batch(context, self.sources, self.frequencies)
        runner.enable_cuda_graphs(buckets=(4,))
        with patch.object(runner._cuda_graph_runner, "_capture", side_effect=RuntimeError("injected")) as capture:
            for _ in range(2):
                actual = runner.forward_batch(context, self.sources, self.frequencies)
                torch.testing.assert_close(actual, expected)
        self.assertEqual(capture.call_count, 1)
        self.assertEqual(runner.execution_config()["cuda_graph"]["fallbacks"], 2)

    def test_async_engine_can_replay_graph_on_its_execution_thread(self):
        runner = self._runner()
        context = runner.prepare_medium(self.velocity, already_normalized=True)
        expected = runner.forward_batch(context, self.sources, self.frequencies)
        runner.enable_cuda_graphs(buckets=(1, 4), fallback_on_error=False)

        async def execute():
            async with AsyncFENOEngine(runner, DynamicBatchConfig(max_batch_size=3),
                                       double_buffer=True) as engine:
                handles = await asyncio.gather(*[
                    engine.submit(context, source, frequency)
                    for source, frequency in zip(self.sources, self.frequencies)
                ])
                return torch.stack(await asyncio.gather(*handles))

        actual = asyncio.run(execute())
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
