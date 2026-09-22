"""Tests for deterministic scheduler A/B request traces."""

import unittest

from benchmarks.scheduler_workload import (
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT_DIR,
    build_manifest,
    check_manifests,
    load_config,
    scenario_names,
)


class SchedulerWorkloadTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(DEFAULT_CONFIG)

    def test_checked_in_manifests_match_configuration(self):
        self.assertTrue(check_manifests(self.config, DEFAULT_OUTPUT_DIR))

    def test_all_scenarios_have_the_same_request_identity_contract(self):
        for name in scenario_names(self.config):
            with self.subTest(name=name):
                manifest = build_manifest(self.config, name)
                requests = manifest["requests"]
                self.assertEqual(manifest["request_count"], 1000)
                self.assertEqual(len(requests), 1000)
                self.assertEqual(
                    [item["sequence_index"] for item in requests], list(range(1000))
                )
                self.assertEqual(len({item["request_id"] for item in requests}), 1000)
                self.assertEqual(
                    [item["medium_id"] for item in requests[:4]],
                    [
                        f"medium-{(index * 3 + list(scenario_names(self.config)).index(name)) % 4}"
                        for index in range(4)
                    ],
                )

    def test_reuse_scenarios_cover_expected_range(self):
        manifests = {
            name: build_manifest(self.config, name)
            for name in scenario_names(self.config)
        }
        self.assertEqual(manifests["no_reuse"]["reuse"]["geometry_reuse_ratio"], 0.0)
        self.assertEqual(manifests["no_reuse"]["reuse"]["wavelet_reuse_ratio"], 0.0)
        self.assertGreater(
            manifests["hotspot_reuse"]["reuse"]["geometry_reuse_ratio"], 0.9
        )
        self.assertGreater(
            manifests["hotspot_reuse"]["reuse"]["wavelet_reuse_ratio"], 0.9
        )

    def test_policy_is_not_encoded_in_trace(self):
        for name in scenario_names(self.config):
            manifest = build_manifest(self.config, name)
            self.assertNotIn("policy", manifest)
            self.assertNotIn("scheduler", manifest)


if __name__ == "__main__":
    unittest.main()
