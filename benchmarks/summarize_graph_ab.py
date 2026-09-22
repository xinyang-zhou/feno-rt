"""Summarize independent formal CUDA Graph A/B runs without pooling samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.graph_workload import DEFAULT_CONFIG, load_config  # noqa: E402
from benchmarks.validate_result import validate_document  # noqa: E402


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def write_json_atomic(path: Path, document: Mapping[str, Any]) -> None:
    write_text_atomic(
        path,
        json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
    )


def describe(values: Sequence[float]) -> Dict[str, Any]:
    resolved = [float(value) for value in values]
    if not resolved:
        raise ValueError("cannot summarize an empty metric")
    mean = statistics.fmean(resolved)
    cv_percent = None
    if mean != 0.0:
        cv_percent = statistics.pstdev(resolved) / abs(mean) * 100.0
    return {
        "count": len(resolved),
        "median": statistics.median(resolved),
        "min": min(resolved),
        "max": max(resolved),
        "cv_percent": cv_percent,
    }


def expected_keys(config: Mapping[str, Any]) -> List[Tuple[str, int, int]]:
    modes = config["cuda_graph"]["modes"]
    batches = config["workload"]["batch_sizes"]
    repeats = int(config["measurement"]["independent_runs"])
    return [
        (str(mode), int(batch), repeat)
        for repeat in range(1, repeats + 1)
        for batch in batches
        for mode in modes
    ]


def result_key(result: Mapping[str, Any]) -> Tuple[str, int, int]:
    mode = "on" if result["execution"]["graph_enabled"] else "off"
    return (
        mode,
        int(result["workload"]["batch_size"]),
        int(result["repeat_index"]),
    )


def load_results(paths: Iterable[Path]) -> List[Tuple[Path, Dict[str, Any]]]:
    loaded: List[Tuple[Path, Dict[str, Any]]] = []
    for path in sorted((item.resolve() for item in paths), key=str):
        with path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        errors, kind, _schema_status = validate_document(result, kind="result")
        if errors:
            raise ValueError(f"invalid {kind} {path}: {'; '.join(errors)}")
        loaded.append((path, result))
    return loaded


def _identity(result: Mapping[str, Any]) -> Dict[str, Any]:
    environment = result["environment"]
    hardware = environment["hardware"]
    gpu = hardware["gpus"][0]
    software = environment["software"]
    model_config_sha = hashlib.sha256(
        _canonical_json(result["model"]["config"]).encode("utf-8")
    ).hexdigest()
    return {
        "git_commit": result["source"]["git_commit"],
        "config_sha256": result["source"]["config_sha256"],
        "checkpoint_sha256": result["model"]["checkpoint"]["sha256"],
        "normalization_sha256": result["model"]["normalization"]["sha256"],
        "velocity_sha256": result["workload"]["velocity_sha256"],
        "gpu_uuid": gpu["uuid"],
        "gpu_name": gpu["name"],
        "gpu_logical_index": gpu["logical_index"],
        "gpu_physical_index": gpu["physical_index"],
        "gpu_total_memory_bytes": gpu["total_memory_bytes"],
        "gpu_persistence_mode": gpu["persistence_mode"],
        "gpu_power_limit_watts": gpu["power_limit_watts"],
        "gpu_application_clocks": gpu["application_clocks"],
        "system_gpu_count": hardware["system_gpu_count"],
        "visible_cuda_device_count": hardware["visible_cuda_device_count"],
        "cpu_model": hardware["cpu_model"],
        "logical_cpu_count": hardware["logical_cpu_count"],
        "cpu_affinity": hardware["cpu_affinity"],
        "system_memory_bytes": hardware["system_memory_bytes"],
        "python": software["python"],
        "pytorch": software["pytorch"],
        "cuda_runtime": software["cuda_runtime"],
        "cudnn": software["cudnn"],
        "driver": software["driver"],
        "operating_system": software["operating_system"],
        "kernel": software["kernel"],
        "torch_num_threads": software["torch_num_threads"],
        "torch_num_interop_threads": software["torch_num_interop_threads"],
        "dependency_lock_sha256": software["dependency_lock"]["sha256"],
        "environment_variables": environment["environment_variables"],
        "model_config_sha256": model_config_sha,
        "parameter_count": result["model"]["parameter_count"],
        "parameter_dtype": result["model"]["parameter_dtype"],
    }


def _group_metrics(results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    selectors = {
        "mean_latency_ms": lambda item: item["metrics"]["latency"]["mean_ms"],
        "p50_latency_ms": lambda item: item["metrics"]["latency"]["p50_ms"],
        "p95_latency_ms": lambda item: item["metrics"]["latency"]["p95_ms"],
        "p99_latency_ms": lambda item: item["metrics"]["latency"]["p99_ms"],
        "throughput_requests_per_second": lambda item: item["metrics"][
            "throughput_requests_per_second"
        ],
        "baseline_allocated_bytes": lambda item: item["metrics"]["memory"][
            "baseline_allocated_bytes"
        ],
        "peak_allocated_bytes": lambda item: item["metrics"]["memory"][
            "peak_allocated_bytes"
        ],
        "peak_reserved_bytes": lambda item: item["metrics"]["memory"][
            "peak_reserved_bytes"
        ],
    }
    return {
        name: describe([selector(result) for result in results])
        for name, selector in selectors.items()
    }


def _reduction_percent(off_value: float, on_value: float) -> float:
    if off_value <= 0:
        raise ValueError("Graph-off latency must be positive")
    return (off_value - on_value) / off_value * 100.0


def _paired_comparison(
    batch_size: int,
    pairs: Sequence[Tuple[int, Mapping[str, Any], Mapping[str, Any]]],
) -> Dict[str, Any]:
    raw_pairs = []
    for repeat_index, off, on in pairs:
        off_latency = off["metrics"]["latency"]
        on_latency = on["metrics"]["latency"]
        off_throughput = off["metrics"]["throughput_requests_per_second"]
        on_throughput = on["metrics"]["throughput_requests_per_second"]
        raw_pairs.append(
            {
                "repeat_index": repeat_index,
                "graph_off_run_id": off["run_id"],
                "graph_on_run_id": on["run_id"],
                "mean_latency_reduction_percent": _reduction_percent(
                    off_latency["mean_ms"], on_latency["mean_ms"]
                ),
                "p50_latency_reduction_percent": _reduction_percent(
                    off_latency["p50_ms"], on_latency["p50_ms"]
                ),
                "p95_latency_reduction_percent": _reduction_percent(
                    off_latency["p95_ms"], on_latency["p95_ms"]
                ),
                "p99_latency_reduction_percent": _reduction_percent(
                    off_latency["p99_ms"], on_latency["p99_ms"]
                ),
                "throughput_speedup": on_throughput / off_throughput,
                "peak_allocated_delta_bytes": on["metrics"]["memory"][
                    "peak_allocated_bytes"
                ]
                - off["metrics"]["memory"]["peak_allocated_bytes"],
                "peak_reserved_delta_bytes": on["metrics"]["memory"][
                    "peak_reserved_bytes"
                ]
                - off["metrics"]["memory"]["peak_reserved_bytes"],
            }
        )
    metric_names = (
        "mean_latency_reduction_percent",
        "p50_latency_reduction_percent",
        "p95_latency_reduction_percent",
        "p99_latency_reduction_percent",
        "throughput_speedup",
        "peak_allocated_delta_bytes",
        "peak_reserved_delta_bytes",
    )
    return {
        "batch_size": batch_size,
        "paired_run_count": len(raw_pairs),
        "metrics": {
            name: describe([pair[name] for pair in raw_pairs])
            for name in metric_names
        },
        "pairs": raw_pairs,
    }


def build_summary(
    config: Mapping[str, Any],
    config_path: Path,
    loaded: Sequence[Tuple[Path, Mapping[str, Any]]],
) -> Dict[str, Any]:
    if not loaded:
        raise ValueError("no formal result files were found")
    expected = set(expected_keys(config))
    by_key: Dict[Tuple[str, int, int], Mapping[str, Any]] = {}
    input_records = []
    for path, result in loaded:
        key = result_key(result)
        if key in by_key:
            raise ValueError(f"duplicate result for mode={key[0]}, batch={key[1]}, run={key[2]}")
        by_key[key] = result
        expected_run_id = f"graph_{key[0]}_batch{key[1]}_run{key[2]}"
        if result["run_id"] != expected_run_id:
            raise ValueError(
                f"run_id {result['run_id']} does not match matrix key {expected_run_id}"
            )
        input_records.append(
            {
                "run_id": result["run_id"],
                "path": path.name,
                "sha256": file_sha256(path),
                "status": result["status"],
                "mode": key[0],
                "batch_size": key[1],
                "repeat_index": key[2],
            }
        )

    unexpected = sorted(set(by_key) - expected)
    if unexpected:
        raise ValueError(f"unexpected matrix entries: {unexpected}")
    identities = {_canonical_json(_identity(result)) for result in by_key.values()}
    if len(identities) != 1:
        raise ValueError("formal runs do not share one source/model/GPU/software identity")
    identity = _identity(next(iter(by_key.values())))
    actual_config_sha = file_sha256(config_path.resolve())
    if identity["config_sha256"] != actual_config_sha:
        raise ValueError("formal run config SHA does not match the summary configuration")

    workload_hashes: Dict[int, set] = {}
    for (mode, batch, _repeat), result in by_key.items():
        if result["benchmark"] != config["benchmark"]:
            raise ValueError(f"unexpected benchmark in {result['run_id']}")
        if result["workload"]["request_count"] != config["workload"]["request_count"]:
            raise ValueError(f"request count mismatch in {result['run_id']}")
        expected_invocations = config["workload"]["request_count"] // batch
        if result["workload"]["invocation_count"] != expected_invocations:
            raise ValueError(f"invocation count mismatch in {result['run_id']}")
        if result["execution"]["graph_enabled"] != (mode == "on"):
            raise ValueError(f"Graph mode mismatch in {result['run_id']}")
        expected_bucket = batch if mode == "on" else None
        if result["execution"]["selected_graph_bucket"] != expected_bucket:
            raise ValueError(f"selected Graph bucket mismatch in {result['run_id']}")
        if result["execution"]["graph_buckets"] != config["cuda_graph"]["buckets"]:
            raise ValueError(f"Graph bucket configuration mismatch in {result['run_id']}")
        if result["execution"]["warmup_invocations"] != config["measurement"][
            "warmup_invocations"
        ]:
            raise ValueError(f"warmup count mismatch in {result['run_id']}")
        expected_execution = config["execution"]
        execution_pairs = (
            ("tf32_enabled", "tf32_enabled"),
            ("cudnn_benchmark", "cudnn_benchmark"),
            ("deterministic", "deterministic_algorithms"),
        )
        for result_name, config_name in execution_pairs:
            if result["execution"][result_name] != expected_execution[config_name]:
                raise ValueError(
                    f"execution setting {result_name} mismatch in {result['run_id']}"
                )
        workload_hashes.setdefault(batch, set()).add(result["workload"]["sha256"])
    for batch, hashes in workload_hashes.items():
        if len(hashes) != 1:
            raise ValueError(f"Graph on/off did not share one workload for batch {batch}")

    passed = {key: value for key, value in by_key.items() if value["status"] == "passed"}
    missing = sorted(expected - set(by_key))
    failed_run_ids = sorted(
        result["run_id"] for result in by_key.values() if result["status"] != "passed"
    )
    groups = []
    for batch in config["workload"]["batch_sizes"]:
        for mode in config["cuda_graph"]["modes"]:
            group_runs = [
                passed[key]
                for key in sorted(passed, key=lambda item: item[2])
                if key[0] == mode and key[1] == batch
            ]
            if not group_runs:
                continue
            group: Dict[str, Any] = {
                "mode": mode,
                "batch_size": batch,
                "run_count": len(group_runs),
                "metrics": _group_metrics(group_runs),
            }
            if mode == "on":
                capture_times = [
                    result["metrics"]["memory"]["capture_wall_time_ms"]
                    for result in group_runs
                ]
                group["graph_setup"] = {
                    "capture_wall_time_ms": describe(capture_times),
                    "capture_allocated_delta_bytes": describe(
                        [
                            result["metrics"]["memory"][
                                "capture_allocated_delta_bytes"
                            ]
                            for result in group_runs
                        ]
                    ),
                    "capture_reserved_delta_bytes": describe(
                        [
                            result["metrics"]["memory"][
                                "capture_reserved_delta_bytes"
                            ]
                            for result in group_runs
                        ]
                    ),
                    "static_buffer_bytes": describe(
                        [
                            result["metrics"]["graph"]["static_buffer_bytes"]
                            for result in group_runs
                        ]
                    ),
                    "resident_graphs": describe(
                        [
                            result["metrics"]["graph"]["resident_graphs"]
                            for result in group_runs
                        ]
                    ),
                }
                group["graph_execution"] = {
                    "selected_bucket": batch,
                    "measured_replay_rate": describe(
                        [
                            result["metrics"]["graph"]["measured_replay_rate"]
                            for result in group_runs
                        ]
                    ),
                    "measured_fallbacks": describe(
                        [
                            result["metrics"]["graph"]["measured_fallbacks"]
                            for result in group_runs
                        ]
                    ),
                    "measured_padded_requests": describe(
                        [
                            result["metrics"]["graph"][
                                "measured_padded_requests"
                            ]
                            for result in group_runs
                        ]
                    ),
                }
            groups.append(group)

    comparisons = []
    for batch in config["workload"]["batch_sizes"]:
        pairs = []
        for repeat in range(1, config["measurement"]["independent_runs"] + 1):
            off = passed.get(("off", batch, repeat))
            on = passed.get(("on", batch, repeat))
            if off is not None and on is not None:
                pairs.append((repeat, off, on))
        if pairs:
            comparisons.append(_paired_comparison(batch, pairs))

    if missing:
        status = "incomplete"
    elif failed_run_ids:
        status = "failed"
    else:
        status = "passed"
    shared_gpu = any(
        result["environment"]["hardware"]["other_gpu_processes"]
        for result in by_key.values()
    )
    notes = [
        "All percentiles are summarized from run-level metrics; raw samples are not pooled.",
        "Latency reduction is (Graph off - Graph on) / Graph off * 100%.",
        "Throughput speedup is Graph on / Graph off.",
        (
            "Formal runs do not collect kernel-launch counts; version-matched profiler "
            "diagnostics must report those separately."
        ),
    ]
    if shared_gpu:
        notes.append("At least one run observed another compute process on the measured GPU.")
    return {
        "schema_version": "1.0.0",
        "summary_class": "formal",
        "status": status,
        "benchmark": config["benchmark"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "identity": identity,
        "protocol": {
            "expected_run_count": len(expected),
            "observed_run_count": len(by_key),
            "passed_run_count": len(passed),
            "independent_runs_per_configuration": config["measurement"][
                "independent_runs"
            ],
            "request_count_per_run": config["workload"]["request_count"],
            "raw_samples_pooled": False,
            "profiler_enabled": False,
            "shared_gpu_observed": shared_gpu,
        },
        "missing_matrix_entries": [
            {"mode": mode, "batch_size": batch, "repeat_index": repeat}
            for mode, batch, repeat in missing
        ],
        "failed_run_ids": failed_run_ids,
        "inputs": sorted(
            input_records,
            key=lambda item: (
                item["batch_size"],
                item["repeat_index"],
                item["mode"],
            ),
        ),
        "groups": groups,
        "comparisons": comparisons,
        "notes": notes,
    }


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _format_number(value: float, digits: int = 3) -> str:
    if not math.isfinite(float(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def render_markdown(summary: Mapping[str, Any]) -> str:
    identity = summary["identity"]
    lines = [
        "# CUDA Graph A/B formal summary",
        "",
        f"Status: `{summary['status']}`",
        "",
        f"- Commit: `{identity['git_commit']}`",
        f"- GPU: `{identity['gpu_name']}` (`{identity['gpu_uuid']}`)",
        f"- PyTorch/CUDA: `{identity['pytorch']}` / `{identity['cuda_runtime']}`",
        f"- Runs: {summary['protocol']['passed_run_count']} passed / "
        f"{summary['protocol']['expected_run_count']} expected",
        "",
        (
            "Percentiles below are medians of independent run-level metrics; "
            "raw samples are not pooled."
        ),
        "",
        "## Run-level metrics",
        "",
        (
            "| Batch | Graph | Runs | Mean ms | p50 ms | p95 ms | p99 ms | "
            "req/s | Peak allocated MiB |"
        ),
        "|---:|:---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in summary["groups"]:
        metrics = group["metrics"]
        lines.append(
            "| {batch} | {mode} | {runs} | {mean} | {p50} | {p95} | {p99} | "
            "{throughput} | {memory} |".format(
                batch=group["batch_size"],
                mode=group["mode"],
                runs=group["run_count"],
                mean=_format_number(metrics["mean_latency_ms"]["median"]),
                p50=_format_number(metrics["p50_latency_ms"]["median"]),
                p95=_format_number(metrics["p95_latency_ms"]["median"]),
                p99=_format_number(metrics["p99_latency_ms"]["median"]),
                throughput=_format_number(
                    metrics["throughput_requests_per_second"]["median"], 2
                ),
                memory=_format_number(
                    metrics["peak_allocated_bytes"]["median"] / (1024 * 1024), 2
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Paired Graph on/off comparison",
            "",
            (
                "| Batch | Pairs | Mean latency reduction | p99 reduction | "
                "Throughput speedup | Peak allocated delta MiB |"
            ),
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for comparison in summary["comparisons"]:
        metrics = comparison["metrics"]
        lines.append(
            "| {batch} | {pairs} | {mean}% | {p99}% | {speedup}x | {memory} |".format(
                batch=comparison["batch_size"],
                pairs=comparison["paired_run_count"],
                mean=_format_number(
                    metrics["mean_latency_reduction_percent"]["median"], 2
                ),
                p99=_format_number(
                    metrics["p99_latency_reduction_percent"]["median"], 2
                ),
                speedup=_format_number(metrics["throughput_speedup"]["median"], 3),
                memory=_format_number(
                    metrics["peak_allocated_delta_bytes"]["median"] / (1024 * 1024),
                    2,
                ),
            )
        )
    lines.extend(
        [
            "",
            "## CUDA Graph setup and steady-state execution",
            "",
            (
                "| Batch | Capture ms | Capture allocated MiB | Static buffers MiB | "
                "Measured replay rate | Fallbacks | Padded requests |"
            ),
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for group in summary["groups"]:
        if group["mode"] != "on":
            continue
        setup = group["graph_setup"]
        execution = group["graph_execution"]
        lines.append(
            "| {batch} | {capture} | {capture_memory} | {static_memory} | "
            "{replay_rate}% | {fallbacks} | {padding} |".format(
                batch=group["batch_size"],
                capture=_format_number(
                    setup["capture_wall_time_ms"]["median"], 3
                ),
                capture_memory=_format_number(
                    setup["capture_allocated_delta_bytes"]["median"]
                    / (1024 * 1024),
                    2,
                ),
                static_memory=_format_number(
                    setup["static_buffer_bytes"]["median"] / (1024 * 1024), 2
                ),
                replay_rate=_format_number(
                    execution["measured_replay_rate"]["median"] * 100.0, 2
                ),
                fallbacks=_format_number(
                    execution["measured_fallbacks"]["median"], 0
                ),
                padding=_format_number(
                    execution["measured_padded_requests"]["median"], 0
                ),
            )
        )
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {note}" for note in summary["notes"])
    if summary["missing_matrix_entries"]:
        lines.append(f"- Missing entries: `{summary['missing_matrix_entries']}`")
    if summary["failed_run_ids"]:
        lines.append(f"- Failed runs: `{summary['failed_run_ids']}`")
    return "\n".join(lines) + "\n"


def summarize_directory(
    config_path: Path, input_dir: Path, output_json: Path, output_markdown: Path
) -> Dict[str, Any]:
    config = load_config(config_path)
    config_errors, kind, _schema_status = validate_document(config, kind="config")
    if config_errors:
        raise ValueError(f"invalid {kind}: {'; '.join(config_errors)}")
    paths = input_dir.glob("graph_*_batch*_run*.json")
    summary = build_summary(config, config_path, load_results(paths))
    write_json_atomic(output_json, summary)
    write_text_atomic(output_markdown, render_markdown(summary))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_json = args.output_json or args.input_dir / "summary.json"
    output_markdown = args.output_markdown or args.input_dir / "summary.md"
    summary = summarize_directory(
        args.config.resolve(),
        args.input_dir.resolve(),
        output_json.resolve(),
        output_markdown.resolve(),
    )
    print(f"saved: {output_json.resolve()}")
    print(f"saved: {output_markdown.resolve()}")
    print(f"status: {summary['status']}")
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
