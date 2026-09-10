"""Deterministic policy tests for Stage-3 batch scheduling."""

import unittest
from time import perf_counter_ns

from feno_rt.runtime.request import InferenceRequest, RequestCost
from feno_rt.runtime.scheduler import (
    CacheAwareScheduler,
    DynamicBatchConfig,
    FCFSScheduler,
    SchedulingPolicy,
)


def make_request(
    sequence_id,
    batch_key,
    geometry_key,
    frequency_key,
    *,
    enqueued_ns=None,
    deadline_ns=None,
    query_tokens=1,
):
    return InferenceRequest(
        request_id=f"request-{sequence_id}",
        sequence_id=sequence_id,
        context=object(),
        source_position=(0.0, 0.0),
        frequency=10.0,
        receiver_positions=None,
        positions_are_normalized=False,
        denormalize=False,
        cache_level="all",
        priority=0,
        cost=RequestCost(
            query_tokens=query_tokens,
            activation_bytes=16,
            output_bytes=16,
        ),
        batch_key=batch_key,
        geometry_key=geometry_key,
        frequency_key=frequency_key,
        cache_keys={},
        future=None,
        enqueued_ns=perf_counter_ns() if enqueued_ns is None else enqueued_ns,
        deadline_ns=deadline_ns,
    )


class SchedulerPolicyTest(unittest.TestCase):
    def test_fcfs_does_not_skip_incompatible_head_items(self):
        config = DynamicBatchConfig(policy=SchedulingPolicy.FCFS, max_batch_size=4)
        scheduler = FCFSScheduler(config)
        pending = [
            make_request(0, "medium-a", "g1", "f1"),
            make_request(1, "medium-b", "g2", "f1"),
            make_request(2, "medium-a", "g1", "f2"),
        ]

        selected = scheduler.select_batch(pending)

        self.assertEqual([request.sequence_id for request in selected], [0])

    def test_cache_aware_groups_compatible_requests_across_queue(self):
        config = DynamicBatchConfig(policy=SchedulingPolicy.CACHE_AWARE, max_batch_size=4)
        scheduler = CacheAwareScheduler(config)
        pending = [
            make_request(0, "medium-a", "g1", "f1"),
            make_request(1, "medium-b", "g2", "f1"),
            make_request(2, "medium-a", "g1", "f2"),
        ]

        selected = scheduler.select_batch(pending)

        self.assertEqual({request.sequence_id for request in selected}, {0, 2})

    def test_all_three_admission_budgets_are_enforced(self):
        config = DynamicBatchConfig(
            max_batch_size=8,
            max_query_tokens=2,
            max_activation_bytes=32,
            max_output_bytes=32,
        )
        scheduler = CacheAwareScheduler(config)
        pending = [make_request(index, "a", "g", "f") for index in range(4)]

        selected = scheduler.select_batch(pending)

        self.assertEqual(len(selected), 2)
        self.assertEqual(sum(request.cost.query_tokens for request in selected), 2)
        self.assertEqual(sum(request.cost.activation_bytes for request in selected), 32)
        self.assertEqual(sum(request.cost.output_bytes for request in selected), 32)

    def test_near_deadline_request_preempts_locality(self):
        now_ns = perf_counter_ns()
        config = DynamicBatchConfig(
            max_batch_size=2,
            deadline_guard_us=1_000,
            starvation_timeout_us=100_000,
        )
        scheduler = CacheAwareScheduler(config)
        pending = [
            make_request(0, "hot", "g1", "f1", enqueued_ns=now_ns),
            make_request(1, "hot", "g1", "f1", enqueued_ns=now_ns),
            make_request(
                2,
                "urgent",
                "g2",
                "f2",
                enqueued_ns=now_ns,
                deadline_ns=now_ns + 500_000,
            ),
        ]

        selected = scheduler.select_batch(pending, now_ns=now_ns)

        self.assertEqual(selected[0].sequence_id, 2)

    def test_starved_request_preempts_locality(self):
        now_ns = perf_counter_ns()
        config = DynamicBatchConfig(
            max_batch_size=2,
            deadline_guard_us=0,
            starvation_timeout_us=1_000,
        )
        scheduler = CacheAwareScheduler(config)
        pending = [
            make_request(
                0,
                "starved",
                "g0",
                "f0",
                enqueued_ns=now_ns - 2_000_000,
            ),
            make_request(1, "hot", "g1", "f1", enqueued_ns=now_ns),
            make_request(2, "hot", "g1", "f1", enqueued_ns=now_ns),
        ]

        selected = scheduler.select_batch(pending, now_ns=now_ns)

        self.assertEqual(selected[0].sequence_id, 0)


    def test_all_starved_compatible_requests_keep_fcfs_order(self):
        now_ns = perf_counter_ns()
        config = DynamicBatchConfig(
            max_batch_size=2,
            deadline_guard_us=0,
            starvation_timeout_us=1_000,
        )
        scheduler = CacheAwareScheduler(config)
        pending = [
            make_request(0, "old", "g0", "f0", enqueued_ns=now_ns - 3_000_000),
            make_request(1, "old", "g1", "f1", enqueued_ns=now_ns - 2_000_000),
            make_request(2, "old", "hot", "f2", enqueued_ns=now_ns),
            make_request(3, "old", "hot", "f2", enqueued_ns=now_ns),
        ]

        selected = scheduler.select_batch(pending, now_ns=now_ns)

        self.assertEqual([request.sequence_id for request in selected], [0, 1])

if __name__ == "__main__":
    unittest.main()
