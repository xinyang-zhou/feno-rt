"""Standard-library tests for benchmark artifact validation."""

import json
import unittest
from copy import deepcopy
from pathlib import Path

from benchmarks.validate_result import validate_document


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_valid_result():
    samples = [1.0] * 1000
    return {
        "schema_version": "1.0.0",
        "result_class": "formal",
        "status": "passed",
        "benchmark": "cuda_graph_ab",
        "run_id": "graph_off_batch1_run1",
        "repeat_index": 1,
        "created_at_utc": "2026-01-01T00:00:00Z",
        "source": {
            "git_commit": "a" * 40,
            "git_dirty": False,
            "command": ["python", "benchmarks/benchmark_graph_ab.py"],
            "config_path": "benchmarks/configs/graph_ab.json",
            "config_sha256": "b" * 64,
        },
        "environment": {
            "hardware": {
                "gpus": [
                    {
                        "logical_index": 0,
                        "physical_index": 1,
                        "name": "Example GPU",
                        "uuid": "GPU-example",
                        "total_memory_bytes": 1024,
                        "persistence_mode": "Enabled",
                        "power_limit_watts": None,
                        "application_clocks": None,
                    }
                ],
                "system_gpu_count": 2,
                "visible_cuda_device_count": 1,
                "cpu_model": "Example CPU",
                "logical_cpu_count": 1,
                "cpu_affinity": [0],
                "system_memory_bytes": 1024,
                "other_gpu_processes": False,
            },
            "software": {
                "python": "3.9.0",
                "pytorch": "2.8.0",
                "cuda_runtime": "12.8",
                "cudnn": None,
                "driver": "example",
                "operating_system": "Linux",
                "kernel": "example",
                "python_executable": "/usr/bin/python",
                "torch_num_threads": 1,
                "torch_num_interop_threads": 1,
                "dependency_lock": {
                    "identifier": "requirements/runtime.lock",
                    "sha256": "1" * 64,
                },
                "direct_dependencies": {
                    "kappamodules": "0.1.112",
                    "numpy": "2.0.2",
                    "torch": "2.8.0",
                },
            },
            "environment_variables": {"CUDA_VISIBLE_DEVICES": "1"},
        },
        "model": {
            "random_initialized": False,
            "seed": 7,
            "checkpoint": {"identifier": "checkpoint", "sha256": "c" * 64},
            "normalization": {"identifier": "normalization", "sha256": "d" * 64},
            "config": {"decoder_dim": 128},
            "parameter_count": 1,
            "parameter_dtype": "torch.float32",
        },
        "workload": {
            "name": "steady_all_hit",
            "schema_version": "1.0.0",
            "sha256": "e" * 64,
            "velocity_sha256": "f" * 64,
            "seed": 7,
            "request_count": 1000,
            "batch_size": 1,
            "invocation_count": 1000,
            "arrival_pattern": "closed_loop",
            "cache_state": {
                "medium": "warm",
                "geometry": "warm",
                "wavelet": "warm",
            },
            "source_shape": [1, 2],
            "receiver_shape": [1, 700, 2],
            "frequency_shape": [1],
        },
        "execution": {
            "device": "cuda:0",
            "graph_enabled": False,
            "graph_buckets": [1, 2, 4, 8],
            "selected_graph_bucket": None,
            "profiler_enabled": False,
            "warmup_invocations": 20,
            "dtype": "torch.float32",
            "tf32_enabled": False,
            "cudnn_benchmark": False,
            "deterministic": True,
        },
        "timing": {
            "clock": "perf_counter_ns",
            "synchronization": "device_before_after",
            "latency_unit": "ms",
            "setup_included": False,
            "measured_wall_time_seconds": 1.0,
            "batch_latency_ms": samples,
        },
        "metrics": {
            "latency": {
                "mean_ms": 1.0,
                "p50_ms": 1.0,
                "p95_ms": 1.0,
                "p99_ms": 1.0,
                "max_ms": 1.0,
            },
            "throughput_requests_per_second": 1000.0,
            "memory": {
                "baseline_allocated_bytes": 100,
                "peak_allocated_bytes": 120,
                "peak_reserved_bytes": 128,
                "capture_wall_time_ms": None,
                "capture_allocated_delta_bytes": None,
                "capture_reserved_delta_bytes": None,
            },
            "cache": {},
            "graph": {
                "requests": 0,
                "captures": 0,
                "replays": 0,
                "capture_failures": 0,
                "fallbacks": 0,
                "padded_requests": 0,
                "padded_slots": 0,
                "resident_graphs": 0,
                "static_buffer_bytes": 0,
                "measured_requests": 0,
                "measured_captures": 0,
                "measured_replays": 0,
                "measured_capture_failures": 0,
                "measured_fallbacks": 0,
                "measured_padded_requests": 0,
                "measured_padded_slots": 0,
                "measured_replay_rate": 0.0,
            },
        },
        "correctness": {
            "passed": True,
            "relative_l2_error": 0.0,
            "max_absolute_error": 0.0,
            "rtol": 1e-5,
            "atol": 1e-5,
            "output_shape": [1, 700, 1024],
            "output_dtype": "torch.float32",
            "all_finite": True,
        },
        "artifacts": {
            "result_path": "benchmarks/results/example.json",
            "workload_path": "benchmarks/results/workload.json",
        },
        "notes": [],
    }


class BenchmarkValidationTest(unittest.TestCase):
    def test_checked_in_graph_configuration_is_valid(self):
        with (PROJECT_ROOT / "benchmarks/configs/graph_ab.json").open(
            "r", encoding="utf-8"
        ) as handle:
            config = json.load(handle)

        errors, kind, _status = validate_document(config)

        self.assertEqual(kind, "config")
        self.assertEqual(errors, [])

    def test_graph_configuration_rejects_unsupported_model_modes(self):
        with (PROJECT_ROOT / "benchmarks/configs/graph_ab.json").open(
            "r", encoding="utf-8"
        ) as handle:
            config = json.load(handle)
        config["model"].update(
            config_source="custom",
            model_source="random_initialized",
            dtype="float16",
        )

        errors, _kind, _status = validate_document(config, kind="config")

        self.assertTrue(any("config_source" in error for error in errors))
        self.assertTrue(any("model_source" in error for error in errors))
        self.assertTrue(any("dtype" in error for error in errors))

    def test_graph_configuration_rejects_unimplemented_workload_modes(self):
        with (PROJECT_ROOT / "benchmarks/configs/graph_ab.json").open(
            "r", encoding="utf-8"
        ) as handle:
            config = json.load(handle)
        config["workload"]["name"] = "cold"
        config["workload"]["arrival_pattern"] = "trace"
        config["workload"]["cache_state"]["geometry"] = "cold"

        errors, _kind, _status = validate_document(config, kind="config")

        self.assertTrue(any("workload.name" in error for error in errors))
        self.assertTrue(any("arrival_pattern" in error for error in errors))
        self.assertTrue(any("cache_state.geometry" in error for error in errors))

    def test_valid_formal_result_is_accepted(self):
        errors, kind, _status = validate_document(make_valid_result())

        self.assertEqual(kind, "result")
        self.assertEqual(errors, [])

    def test_dirty_source_is_rejected(self):
        result = make_valid_result()
        result["source"]["git_dirty"] = True

        errors, _kind, _status = validate_document(result)

        self.assertTrue(any("git_dirty" in error for error in errors))

    def test_summary_must_match_raw_samples(self):
        result = deepcopy(make_valid_result())
        result["metrics"]["latency"]["p99_ms"] = 0.5

        errors, _kind, _status = validate_document(result)

        self.assertTrue(any("p99_ms" in error for error in errors))

    def test_environment_requires_dependency_identity(self):
        result = deepcopy(make_valid_result())
        result["environment"]["software"]["dependency_lock"]["sha256"] = "unknown"

        errors, _kind, _status = validate_document(result)

        self.assertTrue(any("dependency_lock.sha256" in error for error in errors))

    def test_passed_graph_result_rejects_fallback(self):
        result = deepcopy(make_valid_result())
        result["execution"]["graph_enabled"] = True
        result["execution"]["selected_graph_bucket"] = 1
        result["metrics"]["graph"].update(
            requests=1021,
            captures=1,
            replays=1000,
            fallbacks=1,
            resident_graphs=1,
            static_buffer_bytes=1024,
            measured_requests=1000,
            measured_replays=1000,
            measured_fallbacks=1,
            measured_replay_rate=1.0,
        )

        errors, _kind, _status = validate_document(result)

        self.assertTrue(any("fallback" in error for error in errors))

    def test_measured_graph_rate_must_match_counters(self):
        result = deepcopy(make_valid_result())
        result["metrics"]["graph"]["measured_replay_rate"] = 0.5

        errors, _kind, _status = validate_document(result)

        self.assertTrue(any("measured_replay_rate" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
