"""CPU-only tests for scheduler matrix planning and run-level summaries."""

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from benchmarks.run_scheduler_ab_matrix import (
    attempt_log_file,
    build_plan,
    load_resumable_session,
    new_session,
)
from benchmarks.scheduler_workload import DEFAULT_CONFIG, load_config
from benchmarks.summarize_scheduler_ab import (
    build_summary,
    file_sha256,
    load_results,
    render_markdown,
    render_reuse_throughput_svg,
    summarize_directory,
    write_json_atomic,
)
from tests.test_benchmark_validation import make_valid_scheduler_result


class SchedulerMatrixTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(DEFAULT_CONFIG)

    def test_plan_contains_counterbalanced_fresh_process_matrix(self):
        plan = build_plan(self.config)

        self.assertEqual(len(plan), 24)
        self.assertEqual(
            len(
                {
                    (item["policy"], item["scenario"], item["repeat_index"])
                    for item in plan
                }
            ),
            24,
        )
        self.assertEqual(
            [(item["policy"], item["scenario"]) for item in plan[:4]],
            [
                ("fcfs", "no_reuse"),
                ("cache_aware", "no_reuse"),
                ("fcfs", "uniform_reuse"),
                ("cache_aware", "uniform_reuse"),
            ],
        )
        repeat_two = [item for item in plan if item["repeat_index"] == 2]
        self.assertEqual(
            [(item["policy"], item["scenario"]) for item in repeat_two[:4]],
            [
                ("cache_aware", "hotspot_reuse"),
                ("fcfs", "hotspot_reuse"),
                ("cache_aware", "long_tail_reuse"),
                ("fcfs", "long_tail_reuse"),
            ],
        )
        self.assertEqual(attempt_log_file(plan[0], 1), plan[0]["log_file"])
        self.assertTrue(attempt_log_file(plan[0], 2).endswith(".attempt2.log"))

    def _result(self, policy, scenario, repeat_index, config_sha):
        result = deepcopy(make_valid_scheduler_result())
        speedups = {
            "no_reuse": 1.05,
            "uniform_reuse": 1.20,
            "long_tail_reuse": 1.40,
            "hotspot_reuse": 1.60,
        }
        speedup = speedups[scenario] if policy == "cache_aware" else 1.0
        throughput = 1000.0 * speedup
        latency = 4.0 / speedup
        queue = 3.0 / speedup
        batch_size = 8 if policy == "cache_aware" else 1
        batch_sizes = [batch_size] * (1000 // batch_size)
        result.update(
            run_id=f"scheduler_{policy}_{scenario}_run{repeat_index}",
            repeat_index=repeat_index,
        )
        result["source"]["config_sha256"] = config_sha
        scenario_hashes = {
            "no_reuse": "1",
            "uniform_reuse": "2",
            "long_tail_reuse": "3",
            "hotspot_reuse": "4",
        }
        result["workload"].update(
            name=scenario,
            sha256=scenario_hashes[scenario] * 64,
        )
        result["execution"]["policy"] = policy
        result["timing"].update(
            measured_wall_time_seconds=1000.0 / throughput,
            raw_samples={
                "batch_sizes": batch_sizes,
                "queue_latency_ms": [queue] * 1000,
                "execution_latency_ms": [1.0] * 1000,
                "end_to_end_latency_ms": [latency] * 1000,
            },
        )
        result["metrics"].update(
            throughput_requests_per_second=throughput,
            batches=len(batch_sizes),
            mean_batch_size=float(batch_size),
            max_observed_batch_size=batch_size,
            effective_batch_fill_ratio=batch_size / 8.0,
            queue_latency={
                "mean_ms": queue,
                "p50_ms": queue,
                "p95_ms": queue,
                "p99_ms": queue,
                "max_ms": queue,
            },
            execution_latency={
                "mean_ms": 1.0,
                "p50_ms": 1.0,
                "p95_ms": 1.0,
                "p99_ms": 1.0,
                "max_ms": 1.0,
            },
            end_to_end_latency={
                "mean_ms": latency,
                "p50_ms": latency,
                "p95_ms": latency,
                "p99_ms": latency,
                "max_ms": latency,
            },
        )
        result["metrics"]["scheduler"].update(
            selection_calls=len(batch_sizes),
            dispatches=len(batch_sizes),
            singleton_batch_ratio=1.0 if batch_size == 1 else 0.0,
            geometry_shared_request_ratio=0.75 if policy == "cache_aware" else 0.0,
            geometry_in_batch_dedup_rate=0.5 if policy == "cache_aware" else 0.0,
            wavelet_shared_request_ratio=0.75 if policy == "cache_aware" else 0.0,
            wavelet_in_batch_dedup_rate=0.5 if policy == "cache_aware" else 0.0,
        )
        result["metrics"]["cache"]["caches"]["geometry_prefix"]["hit_rate"] = (
            0.8 if policy == "cache_aware" else 0.5
        )
        result["metrics"]["cache"]["caches"]["wavelet"]["hit_rate"] = (
            0.9 if policy == "cache_aware" else 0.75
        )
        return result

    def test_resume_requires_same_commit_config_and_model_artifacts(self):
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

    def test_summary_uses_paired_runs_and_generates_reuse_plot(self):
        config_sha = file_sha256(DEFAULT_CONFIG)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for item in build_plan(self.config):
                result = self._result(
                    item["policy"],
                    item["scenario"],
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
                root / "reuse_throughput.svg",
            )
            self.assertTrue((root / "summary.json").is_file())
            self.assertTrue((root / "summary.md").is_file())
            self.assertTrue((root / "reuse_throughput.svg").is_file())

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(regenerated["status"], "passed")
        self.assertEqual(summary["protocol"]["expected_run_count"], 24)
        self.assertEqual(summary["protocol"]["passed_run_count"], 24)
        self.assertFalse(summary["protocol"]["raw_samples_pooled"])
        self.assertEqual(len(summary["groups"]), 8)
        self.assertEqual(len(summary["comparisons"]), 4)
        hotspot = next(
            item for item in summary["comparisons"] if item["scenario"] == "hotspot_reuse"
        )
        self.assertAlmostEqual(
            hotspot["metrics"]["throughput_speedup"]["median"], 1.6
        )
        markdown = render_markdown(summary)
        self.assertIn("raw samples are not pooled", markdown)
        self.assertIn("reuse_throughput.svg", markdown)
        self.assertIn("Paired Cache-Aware vs FCFS", markdown)
        svg = render_reuse_throughput_svg(summary)
        self.assertIn("<svg", svg)
        self.assertIn("cache_aware", svg)


if __name__ == "__main__":
    unittest.main()
