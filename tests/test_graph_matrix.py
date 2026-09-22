"""CPU-only tests for Graph A/B matrix planning and run-level summaries."""

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from benchmarks.graph_workload import DEFAULT_CONFIG, load_config
from benchmarks.run_graph_ab_matrix import (
    attempt_log_file,
    build_plan,
    load_resumable_session,
    new_session,
)
from benchmarks.summarize_graph_ab import (
    build_summary,
    file_sha256,
    load_results,
    render_markdown,
    summarize_directory,
    write_json_atomic,
)
from tests.test_benchmark_validation import make_valid_result


class GraphMatrixTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(DEFAULT_CONFIG)

    def test_plan_contains_counterbalanced_fresh_process_matrix(self):
        plan = build_plan(self.config)

        self.assertEqual(len(plan), 24)
        self.assertEqual(
            len(
                {
                    (item["mode"], item["batch_size"], item["repeat_index"])
                    for item in plan
                }
            ),
            24,
        )
        self.assertEqual(
            [(item["mode"], item["batch_size"]) for item in plan[:4]],
            [("off", 1), ("on", 1), ("off", 2), ("on", 2)],
        )
        repeat_two = [item for item in plan if item["repeat_index"] == 2]
        self.assertEqual(
            [(item["mode"], item["batch_size"]) for item in repeat_two[:4]],
            [("on", 8), ("off", 8), ("on", 4), ("off", 4)],
        )
        self.assertEqual(attempt_log_file(plan[0], 1), plan[0]["log_file"])
        self.assertTrue(attempt_log_file(plan[0], 2).endswith(".attempt2.log"))

    def _result(self, mode, batch_size, repeat_index, config_sha):
        result = deepcopy(make_valid_result())
        graph_enabled = mode == "on"
        latency_ms = float(batch_size) * (0.75 if graph_enabled else 1.0)
        invocation_count = 1000 // batch_size
        measured_seconds = invocation_count * latency_ms / 1000.0
        result.update(
            run_id=f"graph_{mode}_batch{batch_size}_run{repeat_index}",
            repeat_index=repeat_index,
        )
        result["source"]["config_sha256"] = config_sha
        result["workload"].update(
            sha256=f"{batch_size:064x}",
            batch_size=batch_size,
            invocation_count=invocation_count,
            source_shape=[batch_size, 2],
            receiver_shape=[batch_size, 700, 2],
            frequency_shape=[batch_size],
        )
        result["execution"].update(
            graph_enabled=graph_enabled,
            selected_graph_bucket=batch_size if graph_enabled else None,
            deterministic=False,
        )
        result["timing"].update(
            measured_wall_time_seconds=measured_seconds,
            batch_latency_ms=[latency_ms] * invocation_count,
        )
        result["metrics"]["latency"] = {
            "mean_ms": latency_ms,
            "p50_ms": latency_ms,
            "p95_ms": latency_ms,
            "p99_ms": latency_ms,
            "max_ms": latency_ms,
        }
        result["metrics"]["throughput_requests_per_second"] = (
            1000 / measured_seconds
        )
        if graph_enabled:
            result["metrics"]["memory"].update(
                capture_wall_time_ms=5.0,
                capture_allocated_delta_bytes=1024,
                capture_reserved_delta_bytes=2048,
            )
            result["metrics"]["graph"].update(
                requests=invocation_count + 21,
                captures=1,
                replays=invocation_count + 21,
                resident_graphs=1,
                static_buffer_bytes=4096,
                measured_requests=invocation_count,
                measured_replays=invocation_count,
                measured_replay_rate=1.0,
            )
        return result

    def test_resume_requires_the_same_commit_config_and_model_artifacts(self):
        artifacts = {
            "checkpoint": {"identifier": "model.pth", "sha256": "a" * 64},
            "normalization": {"identifier": "norm.npz", "sha256": "b" * 64},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session_path = root / "session.json"
            session = new_session(
                self.config,
                DEFAULT_CONFIG,
                root,
                "c" * 40,
                artifacts,
            )
            write_json_atomic(session_path, session)

            resumed = load_resumable_session(
                session_path,
                self.config,
                DEFAULT_CONFIG,
                "c" * 40,
                artifacts,
            )
            self.assertEqual(resumed["status"], "running")
            self.assertEqual(len(resumed["source"]["resume_commands"]), 1)

            changed = deepcopy(artifacts)
            changed["checkpoint"]["sha256"] = "d" * 64
            with self.assertRaisesRegex(ValueError, "different model artifacts"):
                load_resumable_session(
                    session_path,
                    self.config,
                    DEFAULT_CONFIG,
                    "c" * 40,
                    changed,
                )

    def test_summary_uses_run_level_metrics_and_paired_comparisons(self):
        config_sha = file_sha256(DEFAULT_CONFIG)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for item in build_plan(self.config):
                result = self._result(
                    item["mode"],
                    item["batch_size"],
                    item["repeat_index"],
                    config_sha,
                )
                path = root / item["result_file"]
                path.write_text(json.dumps(result), encoding="utf-8")
                paths.append(path)

            summary = build_summary(
                self.config,
                DEFAULT_CONFIG,
                load_results(paths),
            )
            regenerated = summarize_directory(
                DEFAULT_CONFIG,
                root,
                root / "summary.json",
                root / "summary.md",
            )
            self.assertTrue((root / "summary.json").is_file())
            self.assertTrue((root / "summary.md").is_file())

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(regenerated["status"], "passed")
        self.assertEqual(summary["protocol"]["expected_run_count"], 24)
        self.assertEqual(summary["protocol"]["passed_run_count"], 24)
        self.assertFalse(summary["protocol"]["raw_samples_pooled"])
        self.assertEqual(len(summary["groups"]), 8)
        self.assertEqual(len(summary["comparisons"]), 4)
        for comparison in summary["comparisons"]:
            self.assertEqual(comparison["paired_run_count"], 3)
            self.assertAlmostEqual(
                comparison["metrics"]["mean_latency_reduction_percent"][
                    "median"
                ],
                25.0,
            )
            self.assertAlmostEqual(
                comparison["metrics"]["throughput_speedup"]["median"],
                4.0 / 3.0,
            )
        markdown = render_markdown(summary)
        self.assertIn("raw samples are not pooled", markdown)
        self.assertIn("| 8 | on | 3 |", markdown)
        self.assertIn("CUDA Graph setup and steady-state execution", markdown)
        self.assertIn("100.00%", markdown)


if __name__ == "__main__":
    unittest.main()
