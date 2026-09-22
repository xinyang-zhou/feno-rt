"""Correctness diagnosis must distinguish unavailable output from numeric errors."""

import unittest
import torch

from benchmarks.benchmark_scheduler_ab import compare_correctness_probes


class ReferenceRunner:
    def forward_uncached(self, *args, **kwargs):
        return torch.ones(1, 2, 3)


class SchedulerCorrectnessTest(unittest.TestCase):
    def compare(self, outputs):
        requests = [dict(medium_id="m", source_position=[0, 0], frequency_hz=10)] * 2
        return compare_correctness_probes(
            torch, ReferenceRunner(), {"m": None}, requests, outputs, [0, 1],
            rtol=1e-5, atol=1e-5,
        )

    def test_missing_probe_is_not_reported_as_nan(self):
        result = self.compare({0: torch.ones(2, 3)})
        self.assertFalse(result["passed"])
        self.assertTrue(result["all_finite"])
        self.assertEqual(result["missing_probe_indices"], [1])
        self.assertEqual(result["completed_probe_requests"], 1)
        self.assertEqual(result["nonfinite_probe_indices"], [])

    def test_nonfinite_and_mismatched_outputs_are_distinct(self):
        result = self.compare({0: torch.full((2, 3), float("nan")), 1: torch.zeros(2, 3)})
        self.assertFalse(result["passed"])
        self.assertFalse(result["all_finite"])
        self.assertEqual(result["nonfinite_probe_indices"], [0])
        self.assertEqual(result["mismatched_probe_indices"], [1])
        self.assertEqual(result["missing_probe_indices"], [])

    def test_broadcastable_shape_mismatch_is_rejected_without_subtraction(self):
        result = self.compare({0: torch.ones(2, 1), 1: torch.ones(2, 3)})
        self.assertFalse(result["passed"])
        self.assertEqual(result["mismatched_probe_indices"], [0])
