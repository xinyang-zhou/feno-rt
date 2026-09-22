"""Validate FENO-RT benchmark configurations and formal result JSON files.

The core validator uses only the Python standard library.  When jsonschema
with Draft 2020-12 support is installed, the document is also checked against
the complete schema in ``benchmarks/schema``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Mapping, Sequence, Tuple


SCHEMA_DIR = Path(__file__).resolve().parent / "schema"
SCHEMA_VERSION = "1.0.0"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite_number(value: Any) -> bool:
    return _is_number(value) and math.isfinite(float(value))


def _mapping(value: Any, path: str, errors: List[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(f"{path} must be an object")
        return {}
    return value


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _close(actual: Any, expected: float) -> bool:
    return _finite_number(actual) and math.isclose(
        float(actual), expected, rel_tol=1e-6, abs_tol=1e-9
    )


def _validate_graph_config(document: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    required = {
        "schema_version",
        "benchmark",
        "device",
        "seed",
        "model",
        "workload",
        "measurement",
        "execution",
        "cuda_graph",
        "correctness",
    }
    for name in sorted(required - document.keys()):
        errors.append(f"missing top-level field: {name}")
    if document.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if document.get("benchmark") != "cuda_graph_ab":
        errors.append("benchmark must be cuda_graph_ab")
    if not re.fullmatch(r"cuda:[0-9]+", str(document.get("device", ""))):
        errors.append("device must have the form cuda:<logical-index>")
    seed = document.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        errors.append("seed must be a non-negative integer")

    model = _mapping(document.get("model"), "model", errors)
    if model.get("config_source") != "default_inference":
        errors.append("model.config_source must be default_inference")
    if model.get("model_source") != "checkpoint":
        errors.append("model.model_source must be checkpoint")
    if not model.get("checkpoint_env"):
        errors.append("checkpoint model requires model.checkpoint_env")
    if not model.get("normalization_env"):
        errors.append("checkpoint model requires model.normalization_env")
    if model.get("dtype") != "float32":
        errors.append("formal Graph A/B model.dtype must be float32")

    workload = _mapping(document.get("workload"), "workload", errors)
    if workload.get("name") != "steady_all_hit":
        errors.append("workload.name must be steady_all_hit")
    if workload.get("arrival_pattern") != "closed_loop":
        errors.append("workload.arrival_pattern must be closed_loop")
    request_count = workload.get("request_count")
    if not _positive_int(request_count) or request_count < 1000:
        errors.append("workload.request_count must be at least 1000")
    batch_sizes = workload.get("batch_sizes")
    if (
        not isinstance(batch_sizes, list)
        or not batch_sizes
        or any(not _positive_int(value) for value in batch_sizes)
        or len(set(batch_sizes)) != len(batch_sizes)
    ):
        errors.append("workload.batch_sizes must contain unique positive integers")
        batch_sizes = []
    if _positive_int(request_count):
        for batch_size in batch_sizes:
            if request_count % batch_size:
                errors.append(
                    f"workload.request_count must be divisible by batch size {batch_size}"
                )
    frequencies = workload.get("frequency_hz")
    if (
        not isinstance(frequencies, list)
        or not frequencies
        or any(not _finite_number(value) or value <= 0 for value in frequencies)
        or len(set(frequencies)) != len(frequencies)
    ):
        errors.append(
            "workload.frequency_hz must contain unique positive finite numbers"
        )
    if workload.get("request_sequence") != "repeat_fixed_batch":
        errors.append("workload.request_sequence must be repeat_fixed_batch")
    velocity = _mapping(workload.get("velocity"), "workload.velocity", errors)
    if velocity.get("generator") != "physical_sine_cosine_v1":
        errors.append("workload.velocity.generator must be physical_sine_cosine_v1")
    for name in ("z_sine_amplitude", "x_cosine_amplitude"):
        if not _finite_number(velocity.get(name)):
            errors.append(f"workload.velocity.{name} must be finite")
    for name in ("z_cycles", "x_cycles"):
        value = velocity.get(name)
        if not _finite_number(value) or value <= 0:
            errors.append(f"workload.velocity.{name} must be positive and finite")
    source_positions = workload.get("source_positions")
    if (
        not isinstance(source_positions, list)
        or not source_positions
        or any(
            not isinstance(position, list)
            or len(position) != 2
            or any(not _finite_number(value) or value < 0 for value in position)
            for position in source_positions
        )
    ):
        errors.append("workload.source_positions must contain non-negative coordinate pairs")
        source_positions = []
    elif len({tuple(position) for position in source_positions}) != len(source_positions):
        errors.append("workload.source_positions must be unique")
    if batch_sizes and len(source_positions) < max(batch_sizes):
        errors.append("workload.source_positions must cover the largest batch size")
    if workload.get("positions_are_normalized") is not False:
        errors.append("Graph A/B source positions must use physical grid coordinates")
    if workload.get("receiver_positions") != "model_default":
        errors.append("Graph A/B requires model_default receiver positions")
    cache_state = _mapping(workload.get("cache_state"), "workload.cache_state", errors)
    for name in ("medium", "geometry", "wavelet"):
        if cache_state.get(name) != "warm":
            errors.append(f"steady_all_hit requires cache_state.{name}=warm")

    measurement = _mapping(document.get("measurement"), "measurement", errors)
    if not _positive_int(measurement.get("independent_runs")) or measurement.get(
        "independent_runs", 0
    ) < 3:
        errors.append("measurement.independent_runs must be at least 3")
    if not _positive_int(measurement.get("warmup_invocations")):
        errors.append("measurement.warmup_invocations must be positive")
    if measurement.get("clock") != "perf_counter_ns":
        errors.append("measurement.clock must be perf_counter_ns")
    if measurement.get("synchronization") != "device_before_after":
        errors.append("measurement.synchronization must be device_before_after")
    if measurement.get("profiler_enabled") is not False:
        errors.append("Formal configuration requires profiler_enabled=false")
    if measurement.get("save_raw_batch_latency_samples") is not True:
        errors.append("Formal configuration must save raw batch latency samples")
    if measurement.get("record_peak_memory") is not True:
        errors.append("Formal configuration must record peak memory")

    execution = _mapping(document.get("execution"), "execution", errors)
    for name in ("tf32_enabled", "cudnn_benchmark", "deterministic_algorithms"):
        if not isinstance(execution.get(name), bool):
            errors.append(f"execution.{name} must be boolean")

    graph = _mapping(document.get("cuda_graph"), "cuda_graph", errors)
    modes = graph.get("modes")
    if not isinstance(modes, list) or any(
        not isinstance(value, str) for value in modes
    ) or set(modes) != {"off", "on"}:
        errors.append("cuda_graph.modes must contain exactly off and on")
    buckets = graph.get("buckets")
    if (
        not isinstance(buckets, list)
        or not buckets
        or any(not _positive_int(value) for value in buckets)
        or len(set(buckets)) != len(buckets)
    ):
        errors.append("cuda_graph.buckets must contain unique positive integers")
        buckets = []
    for batch_size in batch_sizes:
        if batch_size not in buckets:
            errors.append(f"batch size {batch_size} needs a matching CUDA Graph bucket")
    if graph.get("fallback_on_error") is not False:
        errors.append("Formal Graph A/B requires fallback_on_error=false")
    if graph.get("capture_outside_steady_state") is not True:
        errors.append("Graph capture must be outside steady-state timing")
    if not _positive_int(graph.get("warmup_iterations")):
        errors.append("cuda_graph.warmup_iterations must be positive")

    correctness = _mapping(document.get("correctness"), "correctness", errors)
    if correctness.get("reference") != "eager_same_commit":
        errors.append("correctness.reference must be eager_same_commit")
    for name in ("rtol", "atol"):
        value = correctness.get(name)
        if not _finite_number(value) or value < 0:
            errors.append(f"correctness.{name} must be a non-negative finite number")
    if correctness.get("require_finite") is not True:
        errors.append("correctness.require_finite must be true")
    return errors


def _validate_formal_result(document: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    required = {
        "schema_version",
        "result_class",
        "status",
        "benchmark",
        "run_id",
        "repeat_index",
        "created_at_utc",
        "source",
        "environment",
        "model",
        "workload",
        "execution",
        "timing",
        "metrics",
        "correctness",
        "artifacts",
        "notes",
    }
    for name in sorted(required - document.keys()):
        errors.append(f"missing top-level field: {name}")
    if document.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if document.get("result_class") != "formal":
        errors.append("result_class must be formal")
    if document.get("status") not in {"passed", "failed"}:
        errors.append("status must be passed or failed")
    if not _positive_int(document.get("repeat_index")):
        errors.append("repeat_index must be positive")
    created_at = document.get("created_at_utc")
    if not isinstance(created_at, str):
        errors.append("created_at_utc must be an ISO 8601 UTC string")
    else:
        try:
            parsed_created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        except ValueError:
            errors.append("created_at_utc must be an ISO 8601 UTC string")
        else:
            if parsed_created_at.utcoffset() != timezone.utc.utcoffset(parsed_created_at):
                errors.append("created_at_utc must use UTC")

    source = _mapping(document.get("source"), "source", errors)
    if not COMMIT_PATTERN.fullmatch(str(source.get("git_commit", ""))):
        errors.append("source.git_commit must be a full lowercase 40-character SHA")
    if source.get("git_dirty") is not False:
        errors.append("Formal result requires source.git_dirty=false")
    if not SHA256_PATTERN.fullmatch(str(source.get("config_sha256", ""))):
        errors.append("source.config_sha256 must be a lowercase SHA256")

    environment = _mapping(document.get("environment"), "environment", errors)
    hardware = _mapping(environment.get("hardware"), "environment.hardware", errors)
    for name in ("system_gpu_count", "visible_cuda_device_count"):
        if not _positive_int(hardware.get(name)):
            errors.append(f"environment.hardware.{name} must be positive")
    for name in ("logical_cpu_count", "system_memory_bytes"):
        if not _positive_int(hardware.get(name)):
            errors.append(f"environment.hardware.{name} must be positive")
    affinity = hardware.get("cpu_affinity")
    if (
        not isinstance(affinity, list)
        or not affinity
        or any(not _non_negative_int(value) for value in affinity)
        or len(set(affinity)) != len(affinity)
    ):
        errors.append(
            "environment.hardware.cpu_affinity must contain unique non-negative integers"
        )
    gpus = hardware.get("gpus")
    if not isinstance(gpus, list) or not gpus:
        errors.append("environment.hardware.gpus must contain the measured GPU")
    software = _mapping(environment.get("software"), "environment.software", errors)
    dependency_lock = _mapping(
        software.get("dependency_lock"), "environment.software.dependency_lock", errors
    )
    if not SHA256_PATTERN.fullmatch(str(dependency_lock.get("sha256", ""))):
        errors.append("environment.software.dependency_lock.sha256 must be a lowercase SHA256")
    direct_dependencies = software.get("direct_dependencies")
    if not isinstance(direct_dependencies, Mapping) or not direct_dependencies:
        errors.append("environment.software.direct_dependencies must not be empty")
    for name in ("torch_num_threads", "torch_num_interop_threads"):
        if not _positive_int(software.get(name)):
            errors.append(f"environment.software.{name} must be positive")

    execution = _mapping(document.get("execution"), "execution", errors)
    if execution.get("profiler_enabled") is not False:
        errors.append("Formal result requires execution.profiler_enabled=false")
    for name in ("tf32_enabled", "cudnn_benchmark", "deterministic"):
        if not isinstance(execution.get(name), bool):
            errors.append(f"execution.{name} must be boolean")

    workload = _mapping(document.get("workload"), "workload", errors)
    if not SHA256_PATTERN.fullmatch(str(workload.get("sha256", ""))):
        errors.append("workload.sha256 must be a lowercase SHA256")
    if not SHA256_PATTERN.fullmatch(str(workload.get("velocity_sha256", ""))):
        errors.append("workload.velocity_sha256 must be a lowercase SHA256")
    request_count = workload.get("request_count")
    batch_size = workload.get("batch_size")
    invocation_count = workload.get("invocation_count")
    if not _positive_int(request_count) or request_count < 1000:
        errors.append("workload.request_count must be at least 1000")
    if not _positive_int(batch_size):
        errors.append("workload.batch_size must be positive")
    if not _positive_int(invocation_count):
        errors.append("workload.invocation_count must be positive")
    if all(_positive_int(value) for value in (request_count, batch_size, invocation_count)):
        if request_count != batch_size * invocation_count:
            errors.append(
                "workload.request_count must equal batch_size * invocation_count"
            )

    timing = _mapping(document.get("timing"), "timing", errors)
    if timing.get("setup_included") is not False:
        errors.append("timing.setup_included must be false")
    raw_samples = timing.get("batch_latency_ms")
    if (
        not isinstance(raw_samples, list)
        or not raw_samples
        or any(not _finite_number(value) or value < 0 for value in raw_samples)
    ):
        errors.append("timing.batch_latency_ms must contain finite non-negative samples")
        raw_samples = []
    if _positive_int(invocation_count) and len(raw_samples) != invocation_count:
        errors.append(
            "raw batch latency sample count must equal workload.invocation_count"
        )
    measured_seconds = timing.get("measured_wall_time_seconds")
    if not _finite_number(measured_seconds) or measured_seconds <= 0:
        errors.append("timing.measured_wall_time_seconds must be positive and finite")

    metrics = _mapping(document.get("metrics"), "metrics", errors)
    latency = _mapping(metrics.get("latency"), "metrics.latency", errors)
    if raw_samples:
        expected_latency = {
            "mean_ms": sum(raw_samples) / len(raw_samples),
            "p50_ms": _percentile(raw_samples, 50.0),
            "p95_ms": _percentile(raw_samples, 95.0),
            "p99_ms": _percentile(raw_samples, 99.0),
            "max_ms": max(raw_samples),
        }
        for name, expected in expected_latency.items():
            if not _close(latency.get(name), expected):
                errors.append(
                    f"metrics.latency.{name} does not match raw samples: expected {expected}"
                )
    if _positive_int(request_count) and _finite_number(measured_seconds) and measured_seconds > 0:
        expected_throughput = request_count / measured_seconds
        if not _close(metrics.get("throughput_requests_per_second"), expected_throughput):
            errors.append(
                "metrics.throughput_requests_per_second does not match request_count / "
                "measured_wall_time_seconds"
            )

    memory = _mapping(metrics.get("memory"), "metrics.memory", errors)
    baseline = memory.get("baseline_allocated_bytes")
    peak_allocated = memory.get("peak_allocated_bytes")
    peak_reserved = memory.get("peak_reserved_bytes")
    for name, value in (
        ("baseline_allocated_bytes", baseline),
        ("peak_allocated_bytes", peak_allocated),
        ("peak_reserved_bytes", peak_reserved),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            errors.append(f"metrics.memory.{name} must be a non-negative integer")
    if isinstance(baseline, int) and isinstance(peak_allocated, int) and peak_allocated < baseline:
        errors.append("peak_allocated_bytes cannot be lower than baseline_allocated_bytes")
    if (
        isinstance(peak_allocated, int)
        and isinstance(peak_reserved, int)
        and peak_reserved < peak_allocated
    ):
        errors.append("peak_reserved_bytes cannot be lower than peak_allocated_bytes")

    graph_metrics = _mapping(metrics.get("graph"), "metrics.graph", errors)
    graph_metric_names = (
        "requests",
        "captures",
        "replays",
        "capture_failures",
        "fallbacks",
        "padded_requests",
        "padded_slots",
        "resident_graphs",
        "static_buffer_bytes",
        "measured_requests",
        "measured_captures",
        "measured_replays",
        "measured_capture_failures",
        "measured_fallbacks",
        "measured_padded_requests",
        "measured_padded_slots",
    )
    for name in graph_metric_names:
        if not _non_negative_int(graph_metrics.get(name)):
            errors.append(f"metrics.graph.{name} must be a non-negative integer")
    measured_replay_rate = graph_metrics.get("measured_replay_rate")
    if (
        not _finite_number(measured_replay_rate)
        or measured_replay_rate < 0
        or measured_replay_rate > 1
    ):
        errors.append("metrics.graph.measured_replay_rate must be between zero and one")
    measured_requests = graph_metrics.get("measured_requests")
    measured_replays = graph_metrics.get("measured_replays")
    measured_fallbacks = graph_metrics.get("measured_fallbacks")
    if all(
        _non_negative_int(value)
        for value in (measured_requests, measured_replays, measured_fallbacks)
    ):
        if measured_requests != measured_replays + measured_fallbacks:
            errors.append(
                "metrics.graph.measured_requests must equal measured_replays + "
                "measured_fallbacks"
            )
        expected_replay_rate = (
            measured_replays / measured_requests if measured_requests else 0.0
        )
        if not _close(measured_replay_rate, expected_replay_rate):
            errors.append(
                "metrics.graph.measured_replay_rate does not match measured counters"
            )
    graph_enabled = execution.get("graph_enabled")
    selected_bucket = execution.get("selected_graph_bucket")
    graph_buckets = execution.get("graph_buckets", [])
    if (
        not isinstance(graph_buckets, list)
        or any(not _positive_int(value) for value in graph_buckets)
        or len(set(graph_buckets)) != len(graph_buckets)
    ):
        errors.append("execution.graph_buckets must contain unique positive integers")
        graph_buckets = []
    capture_wall_time = memory.get("capture_wall_time_ms")
    if capture_wall_time is not None and (
        not _finite_number(capture_wall_time) or capture_wall_time < 0
    ):
        errors.append("metrics.memory.capture_wall_time_ms must be non-negative or null")
    for name in ("capture_allocated_delta_bytes", "capture_reserved_delta_bytes"):
        value = memory.get(name)
        if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
            errors.append(f"metrics.memory.{name} must be an integer or null")
    if graph_enabled is True:
        if not _positive_int(selected_bucket):
            errors.append("Graph-enabled result requires selected_graph_bucket")
        elif selected_bucket not in graph_buckets or (
            _positive_int(batch_size) and selected_bucket < batch_size
        ):
            errors.append("selected_graph_bucket must cover batch_size and exist in graph_buckets")
        if document.get("status") == "passed":
            if graph_metrics.get("capture_failures") != 0 or graph_metrics.get("fallbacks") != 0:
                errors.append(
                    "passed Graph A/B result cannot contain capture failures or fallbacks"
                )
            if _non_negative_int(graph_metrics.get("captures")) and graph_metrics.get(
                "captures", 0
            ) < 1:
                errors.append("passed Graph-enabled result must contain a captured graph")
            if (
                _positive_int(invocation_count)
                and _non_negative_int(graph_metrics.get("replays"))
                and graph_metrics.get("replays", 0) < invocation_count
            ):
                errors.append(
                    "passed Graph-enabled result needs at least one replay per measured invocation"
                )
            if _positive_int(invocation_count) and measured_requests != invocation_count:
                errors.append(
                    "passed Graph-enabled result requires one measured Graph request per "
                    "invocation"
                )
            if _positive_int(invocation_count) and measured_replays != invocation_count:
                errors.append(
                    "passed Graph-enabled result requires one measured replay per invocation"
                )
            for name in (
                "measured_captures",
                "measured_capture_failures",
                "measured_fallbacks",
                "measured_padded_requests",
                "measured_padded_slots",
            ):
                if graph_metrics.get(name) != 0:
                    errors.append(
                        f"passed exact-bucket steady state requires metrics.graph.{name}=0"
                    )
            if measured_replay_rate != 1.0:
                errors.append(
                    "passed Graph-enabled result requires measured_replay_rate=1"
                )
            if selected_bucket == batch_size and (
                graph_metrics.get("padded_requests") != 0
                or graph_metrics.get("padded_slots") != 0
            ):
                errors.append("matching Graph bucket cannot report padded requests or slots")
            for name in (
                "capture_wall_time_ms",
                "capture_allocated_delta_bytes",
                "capture_reserved_delta_bytes",
            ):
                if memory.get(name) is None:
                    errors.append(f"passed Graph-enabled result requires metrics.memory.{name}")
    elif graph_enabled is False:
        if selected_bucket is not None:
            errors.append("Graph-disabled result requires selected_graph_bucket=null")
        for name in (
            "requests",
            "captures",
            "replays",
            "capture_failures",
            "fallbacks",
            "padded_requests",
            "padded_slots",
            "resident_graphs",
            "static_buffer_bytes",
            "measured_requests",
            "measured_captures",
            "measured_replays",
            "measured_capture_failures",
            "measured_fallbacks",
            "measured_padded_requests",
            "measured_padded_slots",
            "measured_replay_rate",
        ):
            if graph_metrics.get(name) != 0:
                errors.append(f"Graph-disabled result requires metrics.graph.{name}=0")
        for name in (
            "capture_wall_time_ms",
            "capture_allocated_delta_bytes",
            "capture_reserved_delta_bytes",
        ):
            if memory.get(name) is not None:
                errors.append(f"Graph-disabled result requires metrics.memory.{name}=null")
    else:
        errors.append("execution.graph_enabled must be boolean")

    model = _mapping(document.get("model"), "model", errors)
    if model.get("random_initialized") is False and model.get("checkpoint") is None:
        errors.append("non-random model requires checkpoint identity")

    correctness = _mapping(document.get("correctness"), "correctness", errors)
    if document.get("status") == "passed":
        if correctness.get("passed") is not True:
            errors.append("status=passed requires correctness.passed=true")
        if correctness.get("all_finite") is not True:
            errors.append("status=passed requires correctness.all_finite=true")
    return errors


def _full_schema_errors(document: Mapping[str, Any], kind: str) -> Tuple[List[str], str]:
    try:
        from jsonschema import Draft202012Validator, FormatChecker
    except ImportError:
        return [], "full schema validation skipped (install jsonschema>=4.18)"

    schema_name = (
        "graph_ab_config.schema.json" if kind == "config" else "formal_result.schema.json"
    )
    with (SCHEMA_DIR / schema_name).open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = [
        f"schema: {error.json_path}: {error.message}"
        for error in sorted(validator.iter_errors(document), key=lambda item: list(item.path))
    ]
    return errors, f"validated against {schema_name}"


def validate_document(document: Any, *, kind: str = "auto") -> Tuple[List[str], str, str]:
    if not isinstance(document, Mapping):
        return ["top-level JSON value must be an object"], "unknown", "schema not run"
    resolved_kind = kind
    if resolved_kind == "auto":
        resolved_kind = "result" if "result_class" in document else "config"
    if resolved_kind == "config":
        errors = _validate_graph_config(document)
    elif resolved_kind == "result":
        errors = _validate_formal_result(document)
    else:
        return [f"unsupported document kind: {resolved_kind}"], resolved_kind, "schema not run"
    schema_errors, schema_status = _full_schema_errors(document, resolved_kind)
    return errors + schema_errors, resolved_kind, schema_status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="JSON configuration or formal result")
    parser.add_argument("--kind", choices=("auto", "config", "result"), default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        with args.path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        print(f"error: cannot read {args.path}: {error}", file=sys.stderr)
        return 2

    errors, kind, schema_status = validate_document(document, kind=args.kind)
    if errors:
        print(f"invalid {kind} document: {args.path}", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(f"valid {kind} document: {args.path}")
    print(schema_status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
