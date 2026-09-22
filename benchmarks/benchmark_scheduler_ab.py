"""Run one formal FCFS/cache-aware scheduler comparison in a fresh process."""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.benchmark_graph_ab import (  # noqa: E402
    array_sha256,
    collect_environment,
    display_path,
    ensure_external_new_output,
    file_sha256,
    git_identity,
    resolve_model_paths,
    verified_dependency_versions,
    write_json_atomic,
)
from benchmarks.scheduler_workload import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT_DIR,
    build_manifest,
    load_config,
    manifest_path,
    manifest_text,
    scenario_by_name,
)
from benchmarks.validate_result import validate_document  # noqa: E402


def load_runtime_dependencies():
    try:
        import numpy as np
        import torch
    except ImportError as error:
        raise RuntimeError(
            "benchmark_scheduler_ab.py requires the locked runtime dependencies"
        ) from error

    from feno_rt.config import FENOModelConfig
    from feno_rt.preprocessing import NormalizationStats
    from feno_rt.runtime import (
        AsyncFENOEngine,
        DynamicBatchConfig,
        FENOModelRunner,
        SchedulingPolicy,
    )

    return (
        np,
        torch,
        FENOModelConfig,
        NormalizationStats,
        AsyncFENOEngine,
        DynamicBatchConfig,
        FENOModelRunner,
        SchedulingPolicy,
    )


def load_checked_manifest(
    config: Mapping[str, Any], scenario_name: str
) -> Tuple[Path, Dict[str, Any]]:
    path = manifest_path(DEFAULT_OUTPUT_DIR, scenario_name)
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    expected = build_manifest(config, scenario_name)
    if manifest_text(manifest) != manifest_text(expected):
        raise ValueError(f"tracked scheduler workload manifest is stale: {path}")
    return path, manifest


def build_velocity_family(
    np: Any,
    model_config: Any,
    normalization: Any,
    definition: Mapping[str, Any],
    family_index: int,
):
    if definition["generator"] != "physical_sine_cosine_family_v1":
        raise ValueError(f"unsupported velocity generator: {definition['generator']}")
    z = np.linspace(0.0, 1.0, model_config.velocity_height, dtype=np.float32)
    x = np.linspace(0.0, 1.0, model_config.velocity_width, dtype=np.float32)
    z_grid, x_grid = np.meshgrid(z, x, indexing="ij")
    phase = np.float32(float(definition["phase_step"]) * family_index)
    z_scale = np.float32(2.0 * np.pi * float(definition["z_cycles"]))
    x_scale = np.float32(2.0 * np.pi * float(definition["x_cycles"]))
    normalized = (
        np.float32(definition["z_sine_amplitude"])
        * np.sin(z_scale * z_grid + phase)
        + np.float32(definition["x_cosine_amplitude"])
        * np.cos(x_scale * x_grid - phase)
    )
    return (
        np.float32(normalization.v_mean)
        + np.float32(normalization.v_std) * normalized
    ).astype(np.float32)


def _group_requests_by_medium(
    requests: Sequence[Mapping[str, Any]],
) -> Dict[str, List[Mapping[str, Any]]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for request in requests:
        grouped.setdefault(str(request["medium_id"]), []).append(request)
    return grouped


def warmup_runner(
    runner: Any,
    contexts: Mapping[str, Any],
    requests: Sequence[Mapping[str, Any]],
    *,
    max_batch_size: int,
) -> None:
    grouped = _group_requests_by_medium(requests)
    for medium_id, items in grouped.items():
        for start in range(0, len(items), max_batch_size):
            batch = items[start : start + max_batch_size]
            runner.forward_batch(
                contexts[medium_id],
                [item["source_position"] for item in batch],
                [item["frequency_hz"] for item in batch],
                positions_are_normalized=False,
                cache_level="all",
            )


async def execute_trace(
    *,
    engine_class: Any,
    runner: Any,
    engine_config: Any,
    contexts: Mapping[str, Any],
    requests: Sequence[Mapping[str, Any]],
    double_buffer: bool,
    probe_indices: Sequence[int],
) -> Tuple[Dict[str, Any], Dict[int, Any], List[str], int, int]:
    retained_outputs: Dict[int, Any] = {}
    errors: List[str] = []
    probe_set = set(int(index) for index in probe_indices)

    async with engine_class(
        runner,
        engine_config,
        double_buffer=double_buffer,
    ) as engine:
        started_ns = perf_counter_ns()

        async def run_request(index: int, request: Mapping[str, Any]) -> None:
            target_ns = started_ns + int(request["arrival_offset_us"]) * 1_000
            delay_s = (target_ns - perf_counter_ns()) / 1e9
            if delay_s > 0:
                await asyncio.sleep(delay_s)
            try:
                handle = await engine.submit(
                    contexts[str(request["medium_id"])],
                    request["source_position"],
                    request["frequency_hz"],
                    positions_are_normalized=False,
                    cache_level="all",
                    timeout_s=float(request["timeout_us"]) / 1e6,
                    priority=int(request["priority"]),
                    request_id=str(request["request_id"]),
                )
                output = await handle
            except BaseException as error:
                errors.append(
                    f"{request['request_id']}: {type(error).__name__}: {error}"
                )
                return
            if index in probe_set:
                retained_outputs[index] = output

        tasks = [
            asyncio.create_task(run_request(index, request))
            for index, request in enumerate(requests)
        ]
        await asyncio.gather(*tasks)
        measured_completed_ns = perf_counter_ns()
        stats = engine.stats(include_raw_samples=True)
    return stats, retained_outputs, errors, started_ns, measured_completed_ns


def compare_correctness_probes(
    torch: Any,
    runner: Any,
    velocities: Mapping[str, Any],
    requests: Sequence[Mapping[str, Any]],
    retained_outputs: Mapping[int, Any],
    probe_indices: Sequence[int],
    *,
    rtol: float,
    atol: float,
) -> Dict[str, Any]:
    relative_errors: List[float] = []
    maximum_errors: List[float] = []
    output_shape: Optional[List[int]] = None
    output_dtype: Optional[str] = None
    all_finite = True
    passed = True
    missing_indices: List[int] = []
    nonfinite_indices: List[int] = []
    mismatch_indices: List[int] = []
    for index in probe_indices:
        request = requests[index]
        candidate = retained_outputs.get(index)
        if candidate is None:
            passed = False
            missing_indices.append(index)
            continue
        reference = runner.forward_uncached(
            velocities[str(request["medium_id"])],
            [request["source_position"]],
            [request["frequency_hz"]],
            positions_are_normalized=False,
        )[0]
        output_shape = list(candidate.shape)
        output_dtype = str(candidate.dtype)
        finite = bool(torch.isfinite(candidate).all().item()) and bool(
            torch.isfinite(reference).all().item()
        )
        all_finite = all_finite and finite
        if not finite:
            passed = False
            nonfinite_indices.append(index)
            continue
        if candidate.shape != reference.shape or candidate.dtype != reference.dtype:
            passed = False
            mismatch_indices.append(index)
            continue
        difference = (candidate - reference).float()
        relative = torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(
            reference.float()
        ).clamp_min(1e-12)
        relative_errors.append(float(relative.item()))
        maximum_errors.append(float(difference.abs().max().item()))
        try:
            torch.testing.assert_close(candidate, reference, rtol=rtol, atol=atol)
        except AssertionError:
            passed = False
            mismatch_indices.append(index)
    return {
        "passed": passed and all_finite and len(retained_outputs) == len(probe_indices),
        "relative_l2_error": max(relative_errors) if relative_errors else None,
        "max_absolute_error": max(maximum_errors) if maximum_errors else None,
        "rtol": rtol,
        "atol": atol,
        "probe_requests": len(probe_indices),
        "completed_probe_requests": len(probe_indices) - len(missing_indices),
        "missing_probe_indices": missing_indices,
        "nonfinite_probe_indices": nonfinite_indices,
        "mismatched_probe_indices": mismatch_indices,
        "output_shape": output_shape or [],
        "output_dtype": output_dtype or "unknown",
        "all_finite": all_finite,
    }


def evenly_spaced_probe_indices(request_count: int, probe_count: int) -> List[int]:
    if probe_count < 1 or probe_count > request_count:
        raise ValueError("correctness probe count must be within the request trace")
    if probe_count == 1:
        return [0]
    return sorted(
        {
            round(index * (request_count - 1) / (probe_count - 1))
            for index in range(probe_count)
        }
    )


def validate_measured_stats(
    stats: Mapping[str, Any], request_count: int
) -> List[str]:
    notes: List[str] = []
    requests = stats["requests"]
    if int(requests["submitted"]) != request_count:
        notes.append("submitted request count does not match the trace")
    if int(requests["succeeded"]) != request_count:
        notes.append("not every measured request succeeded")
    for name in ("failed", "cancelled", "timed_out", "pending"):
        if int(requests[name]) != 0:
            notes.append(f"measured requests.{name} is not zero")
    raw = stats["raw_samples"]
    for name in (
        "queue_latency_ms",
        "execution_latency_ms",
        "end_to_end_latency_ms",
    ):
        if len(raw[name]) != request_count:
            notes.append(f"raw {name} count does not match the request trace")
    if sum(int(value) for value in raw["batch_sizes"]) != request_count:
        notes.append("raw batch sizes do not sum to the request count")
    scheduler = stats["scheduler"]
    if int(scheduler["dispatched_requests"]) != request_count:
        notes.append("scheduler dispatched request count does not match the trace")
    return notes


def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    ensure_external_new_output(args.output)
    config = load_config(args.config)
    config_errors, config_kind, _schema_status = validate_document(config, kind="config")
    if config_errors:
        raise ValueError(f"invalid {config_kind} document: {'; '.join(config_errors)}")
    scenario_by_name(config, args.scenario)
    if args.policy not in config["scheduler"]["policies"]:
        raise ValueError(f"policy {args.policy} is not configured")
    if not 1 <= args.repeat_index <= config["measurement"]["independent_runs"]:
        raise ValueError("repeat-index is outside configured independent runs")

    git_commit, git_dirty = git_identity()
    if git_dirty:
        raise RuntimeError("formal benchmark requires a clean Git working tree")
    manifest_file, manifest = load_checked_manifest(config, args.scenario)
    checkpoint_path, normalization_path = resolve_model_paths(config)

    (
        np,
        torch,
        FENOModelConfig,
        NormalizationStats,
        AsyncFENOEngine,
        DynamicBatchConfig,
        FENOModelRunner,
        SchedulingPolicy,
    ) = load_runtime_dependencies()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the formal scheduler A/B benchmark")
    if config["model"]["dtype"] != "float32":
        raise ValueError("formal scheduler A/B currently supports only float32")

    device = torch.device(config["device"])
    torch.cuda.set_device(device)
    execution = config["execution"]
    if execution["cuda_graph_enabled"]:
        raise ValueError("scheduler A/B fixes CUDA Graph off to isolate scheduling")
    torch.backends.cuda.matmul.allow_tf32 = execution["tf32_enabled"]
    torch.backends.cudnn.allow_tf32 = execution["tf32_enabled"]
    torch.backends.cudnn.benchmark = execution["cudnn_benchmark"]
    torch.use_deterministic_algorithms(execution["deterministic_algorithms"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    dependency_versions = verified_dependency_versions()
    environment = collect_environment(torch, device, dependency_versions)

    model_config = FENOModelConfig()
    normalization = NormalizationStats.load(normalization_path)
    runner = FENOModelRunner.from_checkpoint(
        checkpoint_path,
        config=model_config,
        normalization=normalization,
        device=device,
    )
    if runner.dtype != torch.float32:
        raise ValueError(f"expected float32 checkpoint, found {runner.dtype}")

    contexts: Dict[str, Any] = {}
    velocities: Dict[str, Any] = {}
    velocity_hashes: Dict[str, str] = {}
    for medium in manifest["mediums"]:
        medium_id = str(medium["medium_id"])
        velocity = build_velocity_family(
            np,
            model_config,
            normalization,
            manifest["velocity"],
            int(medium["family_index"]),
        )
        velocities[medium_id] = velocity
        velocity_hashes[medium_id] = array_sha256(velocity)
        contexts[medium_id] = runner.prepare_medium(velocity, medium_id=medium_id)

    warmup_count = int(config["measurement"]["warmup_requests"])
    warmup_runner(
        runner,
        contexts,
        manifest["requests"][:warmup_count],
        max_batch_size=int(config["scheduler"]["max_batch_size"]),
    )
    torch.cuda.synchronize(device)

    # The formal trace starts with warm medium contexts and cold dependent caches.
    runner.caches.geometry.clear(force=True)
    runner.caches.wavelet.clear(force=True)
    runner.reset_cache_stats()
    gc.collect()
    torch.cuda.synchronize(device)
    baseline_allocated = int(torch.cuda.memory_allocated(device))
    torch.cuda.reset_peak_memory_stats(device)

    scheduler_config = config["scheduler"]
    engine_config = DynamicBatchConfig(
        policy=SchedulingPolicy(args.policy),
        max_wait_us=int(scheduler_config["max_wait_us"]),
        max_batch_size=int(scheduler_config["max_batch_size"]),
        max_query_tokens=int(scheduler_config["max_query_tokens"]),
        max_activation_bytes=int(scheduler_config["max_activation_bytes"]),
        max_output_bytes=int(scheduler_config["max_output_bytes"]),
        deadline_guard_us=int(scheduler_config["deadline_guard_us"]),
        starvation_timeout_us=int(scheduler_config["starvation_timeout_us"]),
        error_isolation=bool(scheduler_config["error_isolation"]),
        synchronize_device=bool(scheduler_config["synchronize_device"]),
    )
    probe_indices = evenly_spaced_probe_indices(
        len(manifest["requests"]),
        int(config["measurement"]["correctness_probe_requests"]),
    )
    stats, retained_outputs, request_errors, started_ns, completed_ns = asyncio.run(
        execute_trace(
            engine_class=AsyncFENOEngine,
            runner=runner,
            engine_config=engine_config,
            contexts=contexts,
            requests=manifest["requests"],
            double_buffer=bool(scheduler_config["double_buffer"]),
            probe_indices=probe_indices,
        )
    )
    torch.cuda.synchronize(device)
    measured_wall_time_seconds = (completed_ns - started_ns) / 1e9
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    measured_notes = validate_measured_stats(stats, int(manifest["request_count"]))
    measured_notes.extend(request_errors)

    correctness_config = config["correctness"]
    correctness = compare_correctness_probes(
        torch,
        runner,
        velocities,
        manifest["requests"],
        retained_outputs,
        probe_indices,
        rtol=float(correctness_config["rtol"]),
        atol=float(correctness_config["atol"]),
    )

    raw_samples = stats.pop("raw_samples")
    status = "passed" if not measured_notes and correctness["passed"] else "failed"
    result = {
        "schema_version": "1.0.0",
        "result_class": "formal",
        "status": status,
        "benchmark": config["benchmark"],
        "run_id": f"scheduler_{args.policy}_{args.scenario}_run{args.repeat_index}",
        "repeat_index": int(args.repeat_index),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "command": [sys.executable, *sys.argv],
            "config_path": display_path(args.config),
            "config_sha256": file_sha256(args.config),
        },
        "environment": environment,
        "model": {
            "random_initialized": False,
            "seed": int(config["seed"]),
            "checkpoint": {
                "identifier": checkpoint_path.name,
                "sha256": file_sha256(checkpoint_path),
            },
            "normalization": {
                "identifier": normalization_path.name,
                "sha256": file_sha256(normalization_path),
            },
            "config": model_config.to_dict(),
            "parameter_count": sum(parameter.numel() for parameter in runner.model.parameters()),
            "parameter_dtype": str(runner.dtype),
        },
        "workload": {
            "name": manifest["name"],
            "schema_version": manifest["schema_version"],
            "sha256": file_sha256(manifest_file),
            "seed": manifest["seed"],
            "request_count": manifest["request_count"],
            "arrival_pattern": manifest["arrival_pattern"],
            "arrival_interval_us": manifest["requests"][0]["arrival_offset_us"]
            if len(manifest["requests"]) == 1
            else manifest["requests"][1]["arrival_offset_us"]
            - manifest["requests"][0]["arrival_offset_us"],
            "slo_timeout_us": manifest["requests"][0]["timeout_us"],
            "medium_count": len(manifest["mediums"]),
            "velocity_sha256": velocity_hashes,
            "positions_are_normalized": manifest["positions_are_normalized"],
            "receiver_positions": manifest["receiver_positions"],
            "cache_state": manifest["cache_state"],
            "reuse": manifest["reuse"],
        },
        "execution": {
            "device": str(device),
            "policy": args.policy,
            "profiler_enabled": False,
            "warmup_requests": warmup_count,
            "dtype": str(runner.dtype),
            "graph_enabled": False,
            "tf32_enabled": bool(execution["tf32_enabled"]),
            "cudnn_benchmark": bool(execution["cudnn_benchmark"]),
            "deterministic": bool(execution["deterministic_algorithms"]),
            "double_buffer": bool(scheduler_config["double_buffer"]),
            "scheduler": {
                name: scheduler_config[name]
                for name in (
                    "max_wait_us",
                    "max_batch_size",
                    "max_query_tokens",
                    "max_activation_bytes",
                    "max_output_bytes",
                    "deadline_guard_us",
                    "starvation_timeout_us",
                    "error_isolation",
                    "synchronize_device",
                )
            },
        },
        "timing": {
            "clock": "perf_counter_ns",
            "setup_included": False,
            "measured_wall_time_seconds": measured_wall_time_seconds,
            "raw_samples": raw_samples,
        },
        "metrics": {
            "throughput_requests_per_second": (
                stats["requests"]["succeeded"] / measured_wall_time_seconds
            ),
            "requests": stats["requests"],
            "batches": stats["batches"],
            "mean_batch_size": stats["mean_batch_size"],
            "max_observed_batch_size": stats["max_observed_batch_size"],
            "effective_batch_fill_ratio": stats["effective_batch_fill_ratio"],
            "queue_latency": stats["queue_latency"],
            "execution_latency": stats["execution_latency"],
            "end_to_end_latency": stats["end_to_end_latency"],
            "scheduler": stats["scheduler"],
            "cache": stats["cache"],
            "pipeline": stats["pipeline"],
            "memory": {
                "baseline_allocated_bytes": baseline_allocated,
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
            },
        },
        "correctness": correctness,
        "artifacts": {
            "result_path": str(args.output.resolve()),
            "workload_path": display_path(manifest_file),
        },
        "notes": measured_notes,
    }
    write_json_atomic(args.output, result)
    result_errors, result_kind, schema_status = validate_document(result, kind="result")
    if result_errors:
        raise ValueError(f"invalid {result_kind} document: {'; '.join(result_errors)}")
    print(f"saved: {args.output.resolve()}")
    print(schema_status)
    print(f"status: {status}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--policy", choices=("fcfs", "cache_aware"), required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--repeat-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_benchmark(args)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
