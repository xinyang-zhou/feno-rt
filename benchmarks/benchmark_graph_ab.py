"""Run one formal CUDA Graph A/B configuration in one fresh process.

One invocation measures exactly one Graph mode, batch size, and repeat index.
An external orchestrator is responsible for launching the complete matrix in
independent processes.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_LOCK = PROJECT_ROOT / "requirements/runtime.lock"
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.graph_workload import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT_DIR,
    build_manifest,
    load_config,
    manifest_path,
    manifest_text,
)
from benchmarks.validate_result import validate_document  # noqa: E402


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: Any) -> str:
    array = value.copy(order="C")
    digest = hashlib.sha256()
    digest.update(str(tuple(array.shape)).encode("utf-8"))
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def run_command(arguments: Sequence[str]) -> str:
    completed = subprocess.run(
        list(arguments),
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def git_identity() -> Tuple[str, bool]:
    commit = run_command(("git", "rev-parse", "HEAD"))
    status = run_command(("git", "status", "--porcelain", "--untracked-files=all"))
    return commit, bool(status)


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def ensure_external_new_output(path: Path) -> None:
    resolved = path.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("formal run output must initially be written outside the repository")
    if resolved.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {resolved}")


def load_checked_manifest(
    config: Mapping[str, Any], batch_size: int
) -> Tuple[Path, Dict[str, Any]]:
    path = manifest_path(DEFAULT_OUTPUT_DIR, batch_size)
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    expected = build_manifest(config, batch_size)
    if manifest_text(manifest) != manifest_text(expected):
        raise ValueError(f"tracked workload manifest is stale: {path}")
    return path, manifest


def load_runtime_dependencies():
    try:
        import numpy as np
        import torch
    except ImportError as error:
        raise RuntimeError(
            "benchmark_graph_ab.py requires the locked runtime dependencies"
        ) from error

    from feno_rt.config import FENOModelConfig
    from feno_rt.preprocessing import NormalizationStats
    from feno_rt.runtime import FENOModelRunner

    return np, torch, FENOModelConfig, NormalizationStats, FENOModelRunner


def locked_dependency_versions(path: Path = RUNTIME_LOCK) -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line:
            raise ValueError(f"runtime lock entry is not exact: {line}")
        name, version = (part.strip() for part in line.split("==", 1))
        if not name or not version:
            raise ValueError(f"invalid runtime lock entry: {line}")
        versions[name] = version
    if not versions:
        raise ValueError(f"runtime lock contains no dependencies: {path}")
    return versions


def verified_dependency_versions() -> Dict[str, str]:
    resolved: Dict[str, str] = {}
    for name, expected in locked_dependency_versions().items():
        installed = importlib.metadata.version(name)
        if installed != expected and not installed.startswith(f"{expected}+"):
            raise RuntimeError(
                f"installed {name}=={installed} does not match runtime lock {expected}"
            )
        resolved[name] = installed
    return resolved


def parse_optional_float(value: str) -> Optional[float]:
    stripped = value.strip()
    if not stripped or stripped.upper() in {"N/A", "[N/A]", "NOT SUPPORTED"}:
        return None
    return float(stripped)


def query_nvidia_gpus() -> List[Dict[str, Any]]:
    fields = (
        "index",
        "uuid",
        "name",
        "memory.total",
        "persistence_mode",
        "power.limit",
        "clocks.applications.graphics",
        "clocks.applications.memory",
        "driver_version",
    )
    output = run_command(
        (
            "nvidia-smi",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        )
    )
    rows = []
    for values in csv.reader(io.StringIO(output)):
        if not values:
            continue
        if len(values) != len(fields):
            raise RuntimeError(f"unexpected nvidia-smi GPU row: {values}")
        values = [value.strip() for value in values]
        persistence_mode = values[4]
        graphics_clock = values[6]
        memory_clock = values[7]
        clocks = None
        if graphics_clock.upper() not in {"N/A", "[N/A]"} or memory_clock.upper() not in {
            "N/A",
            "[N/A]",
        }:
            clocks = f"graphics={graphics_clock} MHz,memory={memory_clock} MHz"
        rows.append(
            {
                "index": int(values[0]),
                "uuid": values[1],
                "name": values[2],
                "memory_total_mib": int(values[3]),
                "persistence_mode": (
                    None
                    if persistence_mode.upper() in {"N/A", "[N/A]", "NOT SUPPORTED"}
                    else persistence_mode
                ),
                "power_limit_watts": parse_optional_float(values[5]),
                "application_clocks": clocks,
                "driver": values[8],
            }
        )
    if not rows:
        raise RuntimeError("nvidia-smi returned no GPUs")
    return rows


def canonical_gpu_uuid(value: Any) -> str:
    resolved = str(value).strip().lower()
    if resolved.startswith("gpu-"):
        return resolved[4:]
    return resolved


def resolve_physical_gpu(
    logical_index: int, properties: Any, rows: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any]:
    property_uuid = getattr(properties, "uuid", None)
    if property_uuid is not None:
        property_uuid = canonical_gpu_uuid(property_uuid)
        for row in rows:
            if canonical_gpu_uuid(row["uuid"]) == property_uuid:
                return row

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        tokens = [token.strip() for token in visible.split(",") if token.strip()]
        if logical_index >= len(tokens):
            raise RuntimeError("logical device is outside CUDA_VISIBLE_DEVICES")
        token = tokens[logical_index]
        if token.isdigit():
            if len(rows) == 1:
                return rows[0]
            raise RuntimeError(
                "cannot safely map a numeric CUDA_VISIBLE_DEVICES ordinal on a "
                "multi-GPU host because CUDA and nvidia-smi ordering may differ; "
                "select the GPU by UUID"
            )
        else:
            canonical_token = canonical_gpu_uuid(token)
            for row in rows:
                canonical_row = canonical_gpu_uuid(row["uuid"])
                if canonical_row == canonical_token or canonical_row.startswith(
                    canonical_token
                ):
                    return row
    else:
        if len(rows) == 1 and logical_index == 0:
            return rows[0]
    raise RuntimeError(
        "cannot resolve logical CUDA device to a physical GPU; set "
        "CUDA_VISIBLE_DEVICES to a GPU UUID"
    )


def has_other_compute_processes(gpu_uuid: str) -> bool:
    output = run_command(
        (
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        )
    )
    current_pid = os.getpid()
    for values in csv.reader(io.StringIO(output)):
        if len(values) < 2:
            continue
        uuid = values[0].strip()
        try:
            pid = int(values[1].strip())
        except ValueError:
            continue
        if uuid == gpu_uuid and pid != current_pid:
            return True
    return False


def cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def system_memory_bytes() -> int:
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    page_size = os.sysconf("SC_PAGE_SIZE")
    page_count = os.sysconf("SC_PHYS_PAGES")
    return int(page_size * page_count)


def cpu_affinity() -> List[int]:
    if hasattr(os, "sched_getaffinity"):
        return sorted(int(value) for value in os.sched_getaffinity(0))
    return list(range(int(os.cpu_count() or 1)))


def collect_environment(
    torch: Any, device: Any, dependency_versions: Mapping[str, str]
) -> Dict[str, Any]:
    logical_index = device.index
    if logical_index is None:
        logical_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    rows = query_nvidia_gpus()
    physical = resolve_physical_gpu(logical_index, properties, rows)
    cudnn_version = torch.backends.cudnn.version()
    environment_names = (
        "CUDA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER",
        "CUBLAS_WORKSPACE_CONFIG",
        "PYTORCH_CUDA_ALLOC_CONF",
        "CUDA_LAUNCH_BLOCKING",
        "CUDA_MODULE_LOADING",
        "NVIDIA_TF32_OVERRIDE",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    )
    return {
        "hardware": {
            "gpus": [
                {
                    "logical_index": logical_index,
                    "physical_index": int(physical["index"]),
                    "name": str(properties.name),
                    "uuid": str(physical["uuid"]),
                    "total_memory_bytes": int(properties.total_memory),
                    "persistence_mode": physical["persistence_mode"],
                    "power_limit_watts": physical["power_limit_watts"],
                    "application_clocks": physical["application_clocks"],
                }
            ],
            "system_gpu_count": len(rows),
            "visible_cuda_device_count": int(torch.cuda.device_count()),
            "cpu_model": cpu_model(),
            "logical_cpu_count": int(os.cpu_count() or 1),
            "cpu_affinity": cpu_affinity(),
            "system_memory_bytes": system_memory_bytes(),
            "other_gpu_processes": has_other_compute_processes(str(physical["uuid"])),
        },
        "software": {
            "python": platform.python_version(),
            "pytorch": str(torch.__version__),
            "cuda_runtime": str(torch.version.cuda),
            "cudnn": None if cudnn_version is None else str(cudnn_version),
            "driver": str(physical["driver"]),
            "operating_system": platform.platform(),
            "kernel": platform.release(),
            "python_executable": sys.executable,
            "torch_num_threads": int(torch.get_num_threads()),
            "torch_num_interop_threads": int(torch.get_num_interop_threads()),
            "dependency_lock": {
                "identifier": display_path(RUNTIME_LOCK),
                "sha256": file_sha256(RUNTIME_LOCK),
            },
            "direct_dependencies": dict(dependency_versions),
        },
        "environment_variables": {
            name: os.environ.get(name) for name in environment_names
        },
    }


def build_velocity(np: Any, config: Any, normalization: Any, definition: Mapping[str, Any]):
    if definition["generator"] != "physical_sine_cosine_v1":
        raise ValueError(f"unsupported velocity generator: {definition['generator']}")
    z = np.linspace(0.0, 1.0, config.velocity_height, dtype=np.float32)
    x = np.linspace(0.0, 1.0, config.velocity_width, dtype=np.float32)
    z_grid, x_grid = np.meshgrid(z, x, indexing="ij")
    z_scale = np.float32(2.0 * np.pi * float(definition["z_cycles"]))
    x_scale = np.float32(2.0 * np.pi * float(definition["x_cycles"]))
    normalized = (
        np.float32(definition["z_sine_amplitude"]) * np.sin(z_scale * z_grid)
        + np.float32(definition["x_cosine_amplitude"]) * np.cos(x_scale * x_grid)
    )
    return (
        np.float32(normalization.v_mean)
        + np.float32(normalization.v_std) * normalized
    ).astype(np.float32)


def percentile(values: Sequence[float], level: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * level / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def latency_summary(samples: Sequence[float]) -> Dict[str, float]:
    return {
        "mean_ms": sum(samples) / len(samples),
        "p50_ms": percentile(samples, 50.0),
        "p95_ms": percentile(samples, 95.0),
        "p99_ms": percentile(samples, 99.0),
        "max_ms": max(samples),
    }


def compare_outputs(
    torch: Any,
    candidates: Iterable[Any],
    reference: Any,
    rtol: float,
    atol: float,
):
    relative_errors: List[float] = []
    maximum_errors: List[float] = []
    reference_finite = bool(torch.isfinite(reference).all().item())
    all_finite = reference_finite
    passed = reference_finite
    for candidate in candidates:
        finite = bool(torch.isfinite(candidate).all().item())
        all_finite = all_finite and finite
        if not reference_finite or not finite:
            passed = False
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
    return {
        "passed": passed and all_finite,
        "relative_l2_error": max(relative_errors) if relative_errors else None,
        "max_absolute_error": max(maximum_errors) if maximum_errors else None,
        "rtol": rtol,
        "atol": atol,
        "output_shape": list(reference.shape),
        "output_dtype": str(reference.dtype),
        "all_finite": all_finite,
    }


def zero_graph_metrics() -> Dict[str, Any]:
    return {
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
    }


def normalized_graph_metrics(
    raw: Mapping[str, Any], before_measurement: Mapping[str, Any]
) -> Dict[str, Any]:
    total_names = (
        "requests",
        "captures",
        "replays",
        "capture_failures",
        "fallbacks",
        "padded_requests",
        "padded_slots",
        "resident_graphs",
        "static_buffer_bytes",
    )
    metrics = {name: int(raw[name]) for name in total_names}
    delta_names = (
        "requests",
        "captures",
        "replays",
        "capture_failures",
        "fallbacks",
        "padded_requests",
        "padded_slots",
    )
    for name in delta_names:
        metrics[f"measured_{name}"] = int(raw[name]) - int(
            before_measurement[name]
        )
    measured_requests = metrics["measured_requests"]
    metrics["measured_replay_rate"] = (
        metrics["measured_replays"] / measured_requests
        if measured_requests
        else 0.0
    )
    return metrics


def resolve_model_paths(config: Mapping[str, Any]) -> Tuple[Path, Path]:
    model = config["model"]
    if model["model_source"] != "checkpoint":
        raise ValueError("formal Graph A/B currently requires a checkpoint model")
    checkpoint_value = os.environ.get(model["checkpoint_env"])
    normalization_value = os.environ.get(model["normalization_env"])
    if not checkpoint_value:
        raise ValueError(f"missing environment variable: {model['checkpoint_env']}")
    if not normalization_value:
        raise ValueError(f"missing environment variable: {model['normalization_env']}")
    checkpoint = Path(checkpoint_value).expanduser().resolve()
    normalization = Path(normalization_value).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if not normalization.is_file():
        raise FileNotFoundError(f"normalization not found: {normalization}")
    return checkpoint, normalization


def assert_all_hit(cache_metrics: Mapping[str, Any]) -> List[str]:
    notes: List[str] = []
    caches = cache_metrics.get("caches", {})
    for name in ("medium", "geometry_prefix", "wavelet"):
        misses = int(caches.get(name, {}).get("misses", 0))
        if misses:
            notes.append(f"{name} cache recorded {misses} measured misses")
    return notes


def write_json_atomic(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    ensure_external_new_output(args.output)
    config = load_config(args.config)
    config_errors, config_kind, _schema_status = validate_document(config, kind="config")
    if config_errors:
        raise ValueError(f"invalid {config_kind} document: {'; '.join(config_errors)}")
    if args.mode not in config["cuda_graph"]["modes"]:
        raise ValueError(f"mode {args.mode} is not configured")
    if args.batch_size not in config["workload"]["batch_sizes"]:
        raise ValueError(f"batch size {args.batch_size} is not configured")
    if not 1 <= args.repeat_index <= config["measurement"]["independent_runs"]:
        raise ValueError("repeat-index is outside configured independent runs")

    git_commit, git_dirty = git_identity()
    if git_dirty:
        raise RuntimeError("formal benchmark requires a clean Git working tree")
    manifest_file, manifest = load_checked_manifest(config, args.batch_size)
    checkpoint_path, normalization_path = resolve_model_paths(config)

    np, torch, FENOModelConfig, NormalizationStats, FENOModelRunner = (
        load_runtime_dependencies()
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the formal Graph A/B benchmark")
    if config["model"]["dtype"] != "float32":
        raise ValueError("formal Graph A/B currently supports only float32")

    device = torch.device(config["device"])
    torch.cuda.set_device(device)
    execution_config = config["execution"]
    torch.backends.cuda.matmul.allow_tf32 = execution_config["tf32_enabled"]
    torch.backends.cudnn.allow_tf32 = execution_config["tf32_enabled"]
    torch.backends.cudnn.benchmark = execution_config["cudnn_benchmark"]
    torch.use_deterministic_algorithms(execution_config["deterministic_algorithms"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    dependency_versions = verified_dependency_versions()
    environment = collect_environment(torch, device, dependency_versions)

    model_config = FENOModelConfig()
    normalization = NormalizationStats.load(normalization_path)
    velocity = build_velocity(np, model_config, normalization, manifest["velocity"])
    velocity_digest = array_sha256(velocity)
    sources = [item["source_position"] for item in manifest["batch_template"]]
    frequencies = [item["frequency_hz"] for item in manifest["batch_template"]]
    for source in sources:
        if any(value < 0 or value > model_config.domain_extent for value in source):
            raise ValueError(f"source position outside model domain: {source}")

    runner = FENOModelRunner.from_checkpoint(
        checkpoint_path,
        config=model_config,
        normalization=normalization,
        device=device,
    )
    if runner.dtype != torch.float32:
        raise ValueError(f"expected float32 checkpoint, found {runner.dtype}")
    context = runner.prepare_medium(velocity)

    # Populate geometry and wavelet caches, then take an all-hit eager reference.
    prefill_output = runner.forward_batch(context, sources, frequencies, cache_level="all")
    reference_output = runner.forward_batch(context, sources, frequencies, cache_level="all")
    torch.cuda.synchronize(device)
    reference_cpu = reference_output.detach().cpu()

    graph_config = config["cuda_graph"]
    capture_wall_time_ms: Optional[float] = None
    capture_allocated_delta: Optional[int] = None
    capture_reserved_delta: Optional[int] = None
    setup_candidates = []
    if args.mode == "on":
        runner.enable_cuda_graphs(
            buckets=tuple(graph_config["buckets"]),
            warmup_iterations=graph_config["warmup_iterations"],
            fallback_on_error=graph_config["fallback_on_error"],
        )
        torch.cuda.synchronize(device)
        allocated_before = torch.cuda.memory_allocated(device)
        reserved_before = torch.cuda.memory_reserved(device)
        capture_started = perf_counter_ns()
        captured_output = runner.forward_batch(
            context, sources, frequencies, cache_level="all"
        )
        torch.cuda.synchronize(device)
        capture_wall_time_ms = (perf_counter_ns() - capture_started) / 1e6
        capture_allocated_delta = int(
            torch.cuda.memory_allocated(device) - allocated_before
        )
        capture_reserved_delta = int(
            torch.cuda.memory_reserved(device) - reserved_before
        )
        setup_candidates.append(captured_output.detach().cpu())

    warmup_output = None
    for _ in range(config["measurement"]["warmup_invocations"]):
        warmup_output = runner.forward_batch(
            context, sources, frequencies, cache_level="all"
        )
    torch.cuda.synchronize(device)
    if warmup_output is not None:
        setup_candidates.append(warmup_output.detach().cpu())

    graph_metrics_before_measurement = (
        runner.execution_config()["cuda_graph"]
        if args.mode == "on"
        else zero_graph_metrics()
    )

    del prefill_output, reference_output, warmup_output
    if args.mode == "on":
        del captured_output
    gc.collect()
    torch.cuda.synchronize(device)

    runner.reset_cache_stats()
    baseline_allocated = int(torch.cuda.memory_allocated(device))
    torch.cuda.reset_peak_memory_stats(device)
    samples: List[float] = []
    measured_output = None
    torch.cuda.synchronize(device)
    measurement_started = perf_counter_ns()
    for _ in range(manifest["invocation_count"]):
        torch.cuda.synchronize(device)
        started = perf_counter_ns()
        measured_output = runner.forward_batch(
            context, sources, frequencies, cache_level="all"
        )
        torch.cuda.synchronize(device)
        samples.append((perf_counter_ns() - started) / 1e6)
    measured_seconds = (perf_counter_ns() - measurement_started) / 1e9

    if measured_output is None:
        raise RuntimeError("measurement produced no output")
    measured_output_cpu = measured_output.detach().cpu()
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    cache_metrics = runner.cache_metrics()
    cache_notes = assert_all_hit(cache_metrics)

    correctness_config = config["correctness"]
    correctness = compare_outputs(
        torch,
        [*setup_candidates, measured_output_cpu],
        reference_cpu,
        rtol=float(correctness_config["rtol"]),
        atol=float(correctness_config["atol"]),
    )
    if args.mode == "on":
        raw_graph_metrics = runner.execution_config()["cuda_graph"]
        graph_metrics = normalized_graph_metrics(
            raw_graph_metrics, graph_metrics_before_measurement
        )
        selected_bucket = args.batch_size
    else:
        graph_metrics = zero_graph_metrics()
        selected_bucket = None

    graph_notes = []
    if graph_metrics["capture_failures"] or graph_metrics["fallbacks"]:
        graph_notes.append("CUDA Graph capture failure or fallback was recorded")
    passed = correctness["passed"] and not cache_notes and not graph_notes
    measured_gpu_uuid = environment["hardware"]["gpus"][0]["uuid"]
    environment["hardware"]["other_gpu_processes"] = bool(
        environment["hardware"]["other_gpu_processes"]
        or has_other_compute_processes(measured_gpu_uuid)
    )
    parameter_count = sum(parameter.numel() for parameter in runner.model.parameters())

    result = {
        "schema_version": "1.0.0",
        "result_class": "formal",
        "status": "passed" if passed else "failed",
        "benchmark": config["benchmark"],
        "run_id": f"graph_{args.mode}_batch{args.batch_size}_run{args.repeat_index}",
        "repeat_index": int(args.repeat_index),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "git_commit": git_commit,
            "git_dirty": False,
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
            "parameter_count": int(parameter_count),
            "parameter_dtype": str(runner.dtype),
        },
        "workload": {
            "name": manifest["name"],
            "schema_version": manifest["schema_version"],
            "sha256": file_sha256(manifest_file),
            "velocity_sha256": velocity_digest,
            "seed": int(manifest["seed"]),
            "request_count": int(manifest["request_count"]),
            "batch_size": int(manifest["batch_size"]),
            "invocation_count": int(manifest["invocation_count"]),
            "arrival_pattern": manifest["arrival_pattern"],
            "cache_state": dict(manifest["cache_state"]),
            "source_shape": [args.batch_size, 2],
            "receiver_shape": [
                args.batch_size,
                model_config.num_receivers,
                2,
            ],
            "frequency_shape": [args.batch_size],
        },
        "execution": {
            "device": str(device),
            "graph_enabled": args.mode == "on",
            "graph_buckets": list(graph_config["buckets"]),
            "selected_graph_bucket": selected_bucket,
            "profiler_enabled": False,
            "warmup_invocations": int(config["measurement"]["warmup_invocations"]),
            "dtype": str(runner.dtype),
            "tf32_enabled": bool(execution_config["tf32_enabled"]),
            "cudnn_benchmark": bool(execution_config["cudnn_benchmark"]),
            "deterministic": bool(execution_config["deterministic_algorithms"]),
        },
        "timing": {
            "clock": config["measurement"]["clock"],
            "synchronization": config["measurement"]["synchronization"],
            "latency_unit": "ms",
            "setup_included": False,
            "measured_wall_time_seconds": measured_seconds,
            "batch_latency_ms": samples,
        },
        "metrics": {
            "latency": latency_summary(samples),
            "throughput_requests_per_second": manifest["request_count"]
            / measured_seconds,
            "memory": {
                "baseline_allocated_bytes": baseline_allocated,
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
                "capture_wall_time_ms": capture_wall_time_ms,
                "capture_allocated_delta_bytes": capture_allocated_delta,
                "capture_reserved_delta_bytes": capture_reserved_delta,
            },
            "cache": cache_metrics,
            "graph": graph_metrics,
        },
        "correctness": correctness,
        "artifacts": {
            "result_path": str(args.output.resolve()),
            "workload_path": display_path(manifest_file),
        },
        "notes": [
            "Latency samples are synchronized batch-invocation wall-clock durations.",
            *cache_notes,
            *graph_notes,
        ],
    }
    errors, _kind, schema_status = validate_document(result, kind="result")
    if errors:
        raise RuntimeError("generated invalid formal result: " + "; ".join(errors))
    write_json_atomic(args.output.resolve(), result)
    print(f"saved: {args.output.resolve()}")
    print(schema_status)
    print(
        json.dumps(
            {
                "run_id": result["run_id"],
                "status": result["status"],
                "mean_ms": result["metrics"]["latency"]["mean_ms"],
                "p99_ms": result["metrics"]["latency"]["p99_ms"],
                "throughput_requests_per_second": result["metrics"][
                    "throughput_requests_per_second"
                ],
            },
            indent=2,
        )
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("off", "on"), required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--repeat-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_benchmark(args)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
