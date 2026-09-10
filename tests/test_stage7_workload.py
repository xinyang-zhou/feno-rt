"""CPU-only tests for Stage 7.0 workload and capacity planning."""

import unittest

import numpy as np

from feno_rt.runtime import (
    MultiMediumWorkloadConfig,
    array_digest,
    build_multi_medium_trace,
    build_velocity_variants,
    lru_hit_curve,
    plan_tier_capacities,
    summarize_multi_medium_trace,
    trace_digest,
)


class Stage7WorkloadTest(unittest.TestCase):
    def test_velocity_variants_are_deterministic_distinct_and_bounded(self):
        base = np.linspace(1500.0, 4500.0, 16 * 16, dtype=np.float32).reshape(
            16, 16
        )
        config = MultiMediumWorkloadConfig(
            medium_count=4,
            request_count=16,
            geometry_cardinality=4,
            perturbation_fraction=0.01,
            seed=17,
        )
        left = build_velocity_variants(base, config)
        right = build_velocity_variants(base, config)
        self.assertEqual(tuple(left), tuple(right))
        self.assertTrue(np.array_equal(left["medium-000"], base))
        self.assertFalse(np.shares_memory(left["medium-000"], base))
        self.assertEqual(
            [array_digest(value) for value in left.values()],
            [array_digest(value) for value in right.values()],
        )
        self.assertEqual(len({array_digest(value) for value in left.values()}), 4)
        for value in left.values():
            self.assertTrue(np.isfinite(value).all())
            self.assertGreater(float(value.min()), 0.0)

    def test_trace_is_reproducible_and_covers_working_set(self):
        config = MultiMediumWorkloadConfig(
            medium_count=4,
            request_count=64,
            geometry_cardinality=8,
            frequencies=(5.0, 10.0, 20.0),
            seed=23,
        )
        left = build_multi_medium_trace(config, domain_extent=15.0)
        right = build_multi_medium_trace(config, domain_extent=15.0)
        self.assertEqual(trace_digest(left), trace_digest(right))
        self.assertEqual(len(left), 64)
        self.assertEqual(
            {item["context_id"] for item in left},
            {f"medium-{index:03d}" for index in range(4)},
        )
        self.assertEqual({item["frequency"] for item in left}, {5.0, 10.0, 20.0})
        self.assertTrue(
            all(
                0.0 < coordinate < 15.0
                for item in left
                for coordinate in item["source_position"]
            )
        )

    def test_lru_hit_curve_is_exact_and_monotonic(self):
        curve = lru_hit_curve(["a", "b", "a", "c", "a", "b"])
        self.assertEqual([row["hits"] for row in curve], [0, 2, 3])
        self.assertEqual([row["misses"] for row in curve], [6, 4, 3])
        self.assertTrue(
            all(
                left["hit_rate"] <= right["hit_rate"]
                for left, right in zip(curve, curve[1:])
            )
        )

    def test_capacity_plan_covers_selected_hot_set(self):
        config = MultiMediumWorkloadConfig(
            medium_count=4,
            request_count=128,
            geometry_cardinality=8,
            frequencies=(5.0, 10.0),
            seed=29,
        )
        trace = build_multi_medium_trace(config, domain_extent=15.0)
        plan = plan_tier_capacities(
            trace,
            {
                "medium": 100,
                "decoder_context": 1_000,
                "geometry_prefix": 50,
                "wavelet": 25,
            },
            velocity_bytes_per_medium=1_024,
            target_medium_hit_rate=0.5,
            headroom_fraction=0.25,
            allocation_quantum_bytes=1_024,
        )
        self.assertGreaterEqual(plan["policy"]["predicted_medium_lru_hit_rate"], 0.5)
        self.assertEqual(plan["working_set"]["medium_count"], 4)
        self.assertEqual(plan["working_set"]["frequency_count"], 2)
        for name, raw in plan["gpu"]["raw_component_bytes"].items():
            self.assertGreaterEqual(
                plan["gpu"]["recommended_component_capacity_bytes"][name], raw
            )
        self.assertEqual(plan["disk"]["minimum_source_velocity_bytes"], 4_096)

    def test_summary_and_invalid_configuration_are_explicit(self):
        config = MultiMediumWorkloadConfig(
            medium_count=4,
            request_count=16,
            geometry_cardinality=4,
            seed=31,
        )
        trace = build_multi_medium_trace(config, domain_extent=15.0)
        summary = summarize_multi_medium_trace(trace)
        self.assertEqual(summary["requests"], 16)
        self.assertEqual(summary["medium_count"], 4)
        with self.assertRaises(ValueError):
            MultiMediumWorkloadConfig(medium_count=1)
        with self.assertRaises(ValueError):
            plan_tier_capacities(
                trace,
                {"medium": 1},
                velocity_bytes_per_medium=1,
            )


if __name__ == "__main__":
    unittest.main()
