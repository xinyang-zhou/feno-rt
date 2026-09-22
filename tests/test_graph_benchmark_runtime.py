"""CPU-only structural tests for the formal Graph A/B runner."""

import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from benchmarks.benchmark_graph_ab import (
    ensure_external_new_output,
    latency_summary,
    locked_dependency_versions,
    load_checked_manifest,
    normalized_graph_metrics,
    query_nvidia_gpus,
    resolve_physical_gpu,
    zero_graph_metrics,
)
from benchmarks.graph_workload import DEFAULT_CONFIG, load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class GraphBenchmarkRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(DEFAULT_CONFIG)

    def test_latency_summary_uses_linear_interpolation(self):
        summary = latency_summary([1.0, 2.0, 3.0, 4.0])

        self.assertEqual(summary["mean_ms"], 2.5)
        self.assertEqual(summary["p50_ms"], 2.5)
        self.assertAlmostEqual(summary["p95_ms"], 3.85)
        self.assertAlmostEqual(summary["p99_ms"], 3.97)
        self.assertEqual(summary["max_ms"], 4.0)

    def test_checked_manifest_is_loaded(self):
        path, manifest = load_checked_manifest(self.config, 4)

        self.assertTrue(path.is_file())
        self.assertEqual(manifest["batch_size"], 4)
        self.assertEqual(manifest["invocation_count"], 250)

    def test_formal_output_must_be_outside_repository_and_new(self):
        with self.assertRaisesRegex(ValueError, "outside the repository"):
            ensure_external_new_output(PROJECT_ROOT / "result.json")

        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "result.json"
            ensure_external_new_output(candidate)
            candidate.touch()
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                ensure_external_new_output(candidate)

    def test_disabled_graph_metrics_are_explicit_zeros(self):
        metrics = zero_graph_metrics()

        self.assertTrue(metrics)
        self.assertEqual(set(metrics.values()), {0})

    def test_graph_metrics_separate_setup_from_measured_replays(self):
        before = {
            "requests": 21,
            "captures": 1,
            "replays": 21,
            "capture_failures": 0,
            "fallbacks": 0,
            "padded_requests": 0,
            "padded_slots": 0,
            "resident_graphs": 1,
            "static_buffer_bytes": 4096,
        }
        after = dict(before, requests=1021, replays=1021)

        metrics = normalized_graph_metrics(after, before)

        self.assertEqual(metrics["measured_requests"], 1000)
        self.assertEqual(metrics["measured_captures"], 0)
        self.assertEqual(metrics["measured_replays"], 1000)
        self.assertEqual(metrics["measured_fallbacks"], 0)
        self.assertEqual(metrics["measured_replay_rate"], 1.0)

    def test_runtime_lock_uses_exact_versions(self):
        self.assertEqual(
            locked_dependency_versions(),
            {
                "kappamodules": "0.1.112",
                "numpy": "2.0.2",
                "torch": "2.8.0",
            },
        )

    @patch("benchmarks.benchmark_graph_ab.run_command")
    def test_nvidia_smi_gpu_metadata_is_parsed(self, run_command):
        run_command.return_value = (
            "0, GPU-example, Example GPU, 81920, Enabled, 700.0, "
            "2100, 1593, 570.00"
        )

        rows = query_nvidia_gpus()

        self.assertEqual(
            rows,
            [
                {
                    "index": 0,
                    "uuid": "GPU-example",
                    "name": "Example GPU",
                    "memory_total_mib": 81920,
                    "persistence_mode": "Enabled",
                    "power_limit_watts": 700.0,
                    "application_clocks": "graphics=2100 MHz,memory=1593 MHz",
                    "driver": "570.00",
                }
            ],
        )

    def test_gpu_mapping_prefers_runtime_uuid(self):
        uuid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        rows = [
            {"index": 0, "uuid": "GPU-zero"},
            {"index": 1, "uuid": f"GPU-{uuid}"},
        ]
        properties = SimpleNamespace(uuid=uuid)

        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0"}, clear=False):
            resolved = resolve_physical_gpu(0, properties, rows)

        self.assertEqual(resolved["uuid"], f"GPU-{uuid}")

    def test_ambiguous_numeric_gpu_mapping_is_rejected(self):
        rows = [
            {"index": 0, "uuid": "GPU-zero"},
            {"index": 1, "uuid": "GPU-one"},
        ]
        properties = SimpleNamespace()

        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0"}, clear=False):
            with self.assertRaisesRegex(RuntimeError, "select the GPU by UUID"):
                resolve_physical_gpu(0, properties, rows)

    def test_gpu_uuid_environment_mapping_accepts_nvidia_prefix(self):
        uuid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        rows = [
            {"index": 0, "uuid": "GPU-zero"},
            {"index": 1, "uuid": f"GPU-{uuid}"},
        ]

        with patch.dict(
            "os.environ", {"CUDA_VISIBLE_DEVICES": f"GPU-{uuid}"}, clear=False
        ):
            resolved = resolve_physical_gpu(0, SimpleNamespace(), rows)

        self.assertEqual(resolved["index"], 1)


if __name__ == "__main__":
    unittest.main()
