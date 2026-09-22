"""Lifecycle traces remain balanced across completion, timeout and cancellation."""

import asyncio
import unittest

from feno_rt.runtime import AsyncFENOEngine, DynamicBatchConfig
from feno_rt.runtime.diagnostics import EngineTrace
from tests.test_engine import make_runner


class TraceTest(unittest.TestCase):
    def test_bounded_trace_and_exception_scope(self):
        trace = EngineTrace(max_events=1)
        with self.assertRaisesRegex(ValueError, "test"):
            with trace.span("fails"):
                raise ValueError("test")
        with trace.span("dropped"):
            pass
        snapshot = trace.snapshot()
        self.assertEqual(len(snapshot["traceEvents"]), 1)
        self.assertEqual(snapshot["metadata"]["dropped_events"], 1)
        self.assertGreaterEqual(snapshot["traceEvents"][0]["dur"], 0)


class EngineTraceTest(unittest.IsolatedAsyncioTestCase):
    async def test_success_trace_covers_queue_worker_and_completion(self):
        runner, context = make_runner()
        trace = EngineTrace()
        runner.trace = trace
        config = DynamicBatchConfig(max_batch_size=2, max_wait_us=0)
        async with AsyncFENOEngine(runner, config, trace=trace, double_buffer=True) as engine:
            handles = await asyncio.gather(*[
                engine.submit(context, [2, 3], 10) for _ in range(2)
            ])
            await asyncio.gather(*handles)
        snapshot = trace.snapshot()
        self.assertEqual(snapshot["metadata"]["active_requests"], 0)
        events = snapshot["traceEvents"]
        names = {e["name"] for e in events}
        self.assertTrue({"feno.request_e2e", "feno.queue_wait", "feno.scheduler_select",
                         "feno.batch_build", "feno.runner_execute", "feno.completion",
                         "feno.decoder_tail"}.issubset(names))
        self.assertEqual(sum(e["name"] == "feno.request_e2e" for e in events), 2)
        self.assertTrue(all(e["dur"] >= 0 for e in events))

    async def test_cancel_and_timeout_close_request_ranges(self):
        runner, context = make_runner()
        trace = EngineTrace()
        config = DynamicBatchConfig(max_wait_us=200_000, max_batch_size=8, deadline_guard_us=0)
        async with AsyncFENOEngine(runner, config, trace=trace) as engine:
            cancelled = await engine.submit(context, [2, 3], 10)
            cancelled.cancel()
            expired = await engine.submit(context, [2, 3], 10, timeout_s=0.001)
            await asyncio.gather(cancelled, expired, return_exceptions=True)
        snapshot = trace.snapshot()
        self.assertEqual(snapshot["metadata"]["active_requests"], 0)
        statuses = {e["args"].get("status") for e in snapshot["traceEvents"]
                    if e["name"] == "feno.request_e2e"}
        self.assertEqual(statuses, {"cancelled", "timed_out"})
