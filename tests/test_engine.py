"""Async execution, cancellation, timeout, and isolation tests."""

import asyncio
import unittest

import torch

from feno_rt.config import FENOModelConfig
from feno_rt.models.feno_freq import FENOFreq
from feno_rt.preprocessing import NormalizationStats
from feno_rt.runtime import (
    AsyncFENOEngine,
    DynamicBatchConfig,
    FENOModelRunner,
    RequestState,
    RequestTimeoutError,
    SchedulingPolicy,
)


def make_runner():
    config = FENOModelConfig(
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
    normalization = NormalizationStats(
        v_mean=2000.0,
        v_std=500.0,
        seis_mean=0.25,
        seis_std=2.0,
    )
    torch.manual_seed(11)
    model = FENOFreq(config.encoder_config(), config.decoder_config())
    runner = FENOModelRunner(
        model,
        config=config,
        normalization=normalization,
        device="cpu",
    )
    velocity = torch.linspace(1500.0, 2500.0, 16 * 16).reshape(16, 16)
    return runner, runner.prepare_medium(velocity)


class FaultInjectingRunner:
    def __init__(self, runner):
        self._runner = runner

    def __getattr__(self, name):
        return getattr(self._runner, name)

    def forward_batch(self, context, source_positions, frequencies, **kwargs):
        sources = torch.as_tensor(source_positions)
        if torch.any(sources[:, 0] == 13.0):
            raise ValueError("injected bad request")
        return self._runner.forward_batch(
            context,
            source_positions,
            frequencies,
            **kwargs,
        )


class AsyncFENOEngineTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.runner, self.context = make_runner()

    async def test_dynamic_batch_matches_direct_runner(self):
        config = DynamicBatchConfig(
            policy=SchedulingPolicy.CACHE_AWARE,
            max_wait_us=20_000,
            max_batch_size=3,
            max_query_tokens=24,
            max_activation_bytes=10_000_000,
            max_output_bytes=10_000_000,
        )
        sources = [[2.0, 3.0], [8.0, 10.0], [2.0, 3.0]]
        frequencies = [10.0, 25.0, 10.0]
        async with AsyncFENOEngine(
            self.runner, config, double_buffer=True
        ) as engine:
            handles = await asyncio.gather(
                *[
                    engine.submit(self.context, source, frequency)
                    for source, frequency in zip(sources, frequencies)
                ]
            )
            actual = await asyncio.gather(*handles)
            stats = engine.stats(include_raw_samples=True)

        expected = self.runner.forward_batch(self.context, sources, frequencies)
        torch.testing.assert_close(torch.stack(actual), expected, rtol=1e-5, atol=1e-6)
        self.assertEqual(stats["batches"], 1)
        self.assertEqual(stats["max_observed_batch_size"], 3)
        self.assertEqual(stats["requests"]["succeeded"], 3)
        self.assertTrue(stats["pipeline"]["double_buffered"])
        self.assertEqual(stats["pipeline"]["prepared_batches"], 1)
        self.assertEqual(stats["scheduler"]["dispatches"], 1)
        self.assertEqual(stats["scheduler"]["dispatched_requests"], 3)
        self.assertAlmostEqual(
            stats["scheduler"]["geometry_shared_request_ratio"], 2.0 / 3.0
        )
        self.assertAlmostEqual(
            stats["scheduler"]["wavelet_shared_request_ratio"], 2.0 / 3.0
        )
        self.assertEqual(stats["raw_samples"]["batch_sizes"], [3])
        self.assertEqual(len(stats["raw_samples"]["queue_latency_ms"]), 3)
        self.assertEqual(len(stats["raw_samples"]["execution_latency_ms"]), 3)
        self.assertEqual(len(stats["raw_samples"]["end_to_end_latency_ms"]), 3)

    async def test_queued_request_can_be_cancelled(self):
        config = DynamicBatchConfig(
            max_wait_us=50_000,
            max_batch_size=4,
            max_query_tokens=32,
        )
        async with AsyncFENOEngine(self.runner, config) as engine:
            handle = await engine.submit(self.context, [2.0, 3.0], 10.0)
            self.assertTrue(handle.cancel())
            with self.assertRaises(asyncio.CancelledError):
                await handle
            await asyncio.sleep(0)
            stats = engine.stats()

        self.assertTrue(handle.done)
        self.assertEqual(stats["requests"]["cancelled"], 1)

    async def test_queue_deadline_expires_before_dispatch(self):
        config = DynamicBatchConfig(
            max_wait_us=20_000,
            max_batch_size=4,
            max_query_tokens=32,
            deadline_guard_us=0,
        )
        async with AsyncFENOEngine(self.runner, config) as engine:
            handle = await engine.submit(
                self.context,
                [2.0, 3.0],
                10.0,
                timeout_s=0.001,
            )
            with self.assertRaises(RequestTimeoutError):
                await handle
            stats = engine.stats()

        self.assertTrue(handle.done)
        self.assertEqual(stats["requests"]["timed_out"], 1)
        self.assertEqual(stats["batches"], 0)

    def test_request_state_only_tracks_active_work_and_failure(self):
        self.assertEqual(
            set(RequestState),
            {RequestState.QUEUED, RequestState.RUNNING, RequestState.FAILED},
        )

    async def test_submit_snapshots_mutable_inputs(self):
        config = DynamicBatchConfig(
            policy=SchedulingPolicy.CACHE_AWARE,
            max_wait_us=20_000,
            max_batch_size=4,
            max_query_tokens=32,
        )
        source = torch.tensor([2.0, 3.0])
        original = source.clone()
        expected = self.runner.forward_batch(
            self.context,
            [original.tolist()],
            [10.0],
        )[0]
        async with AsyncFENOEngine(self.runner, config) as engine:
            handle = await engine.submit(self.context, source, 10.0)
            source.fill_(14.0)
            actual = await handle

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    async def test_single_request_over_budget_is_rejected(self):
        config = DynamicBatchConfig(max_query_tokens=1)
        async with AsyncFENOEngine(self.runner, config) as engine:
            with self.assertRaisesRegex(ValueError, "exceeds"):
                await engine.submit(self.context, [2.0, 3.0], 10.0)

    async def test_batch_failure_isolates_bad_request(self):
        config = DynamicBatchConfig(
            max_wait_us=20_000,
            max_batch_size=3,
            max_query_tokens=24,
            error_isolation=True,
        )
        runner = FaultInjectingRunner(self.runner)
        async with AsyncFENOEngine(runner, config) as engine:
            handles = await asyncio.gather(
                engine.submit(self.context, [2.0, 3.0], 10.0),
                engine.submit(self.context, [13.0, 3.0], 10.0),
                engine.submit(self.context, [8.0, 10.0], 25.0),
            )
            results = await asyncio.gather(*handles, return_exceptions=True)
            stats = engine.stats()

        self.assertIsInstance(results[1], ValueError)
        self.assertIsInstance(results[0], torch.Tensor)
        self.assertIsInstance(results[2], torch.Tensor)
        self.assertEqual(stats["requests"]["succeeded"], 2)
        self.assertEqual(stats["requests"]["failed"], 1)
        self.assertGreaterEqual(stats["isolated_retries"], 1)


if __name__ == "__main__":
    unittest.main()
