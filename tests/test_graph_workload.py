"""Tests for deterministic CUDA Graph A/B workload manifests."""

import unittest

from benchmarks.graph_workload import (
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT_DIR,
    build_manifest,
    check_manifests,
    load_config,
)


class GraphWorkloadTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(DEFAULT_CONFIG)

    def test_checked_in_manifests_match_configuration(self):
        self.assertTrue(
            check_manifests(
                self.config,
                DEFAULT_OUTPUT_DIR,
                self.config["workload"]["batch_sizes"],
            )
        )

    def test_batch_templates_are_nested_and_have_exact_request_counts(self):
        manifests = {
            batch_size: build_manifest(self.config, batch_size)
            for batch_size in self.config["workload"]["batch_sizes"]
        }
        largest = manifests[max(manifests)]["batch_template"]

        for batch_size, manifest in manifests.items():
            self.assertEqual(manifest["request_count"], 1000)
            self.assertEqual(
                manifest["invocation_count"] * batch_size,
                manifest["request_count"],
            )
            self.assertEqual(manifest["batch_template"], largest[:batch_size])
            self.assertEqual(manifest["request_sequence"], "repeat_fixed_batch")
            self.assertEqual(
                manifest["cache_state"],
                {"medium": "warm", "geometry": "warm", "wavelet": "warm"},
            )

    def test_manifest_does_not_depend_on_graph_mode(self):
        manifest = build_manifest(self.config, 8)

        self.assertNotIn("graph_enabled", manifest)
        self.assertNotIn("mode", manifest)

    def test_unknown_batch_size_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not configured"):
            build_manifest(self.config, 3)


if __name__ == "__main__":
    unittest.main()
