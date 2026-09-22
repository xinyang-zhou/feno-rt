"""Summarize formal FCFS/cache-aware runs without pooling request samples."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.scheduler_workload import DEFAULT_CONFIG, load_config  # noqa: E402
from benchmarks.summarize_graph_ab import (  # noqa: E402
    describe,
    file_sha256,
    write_json_atomic,
    write_text_atomic,
)
from benchmarks.validate_result import validate_document  # noqa: E402


def expected_keys(config: Mapping[str, Any]) -> List[Tuple[str, str, int]]:
    policies = config["scheduler"]["policies"]
    scenarios = [item["name"] for item in config["workloads"]["scenarios"]]
    repeats = int(config["measurement"]["independent_runs"])
    return [
        (str(policy), str(scenario), repeat)
        for repeat in range(1, repeats + 1)
        for scenario in scenarios
        for policy in policies
    ]


def result_key(result: Mapping[str, Any]) -> Tuple[str, str, int]:
    return (
        str(result["execution"]["policy"]),
        str(result["workload"]["name"]),
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
        if result.get("benchmark") != "scheduler_ab":
            raise ValueError(f"unexpected benchmark in {path}")
        loaded.append((path, result))
    return loaded


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


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


def _cache_hit_rate(result: Mapping[str, Any], name: str) -> float:
    return float(result["metrics"]["cache"]["caches"][name]["hit_rate"])


def _group_metrics(results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    selectors = {
        "throughput_requests_per_second": lambda item: item["metrics"][
            "throughput_requests_per_second"
        ],
        "queue_p50_ms": lambda item: item["metrics"]["queue_latency"]["p50_ms"],
        "queue_p95_ms": lambda item: item["metrics"]["queue_latency"]["p95_ms"],
        "queue_p99_ms": lambda item: item["metrics"]["queue_latency"]["p99_ms"],
        "end_to_end_p50_ms": lambda item: item["metrics"]["end_to_end_latency"][
            "p50_ms"
        ],
        "end_to_end_p95_ms": lambda item: item["metrics"]["end_to_end_latency"][
            "p95_ms"
        ],
        "end_to_end_p99_ms": lambda item: item["metrics"]["end_to_end_latency"][
            "p99_ms"
        ],
        "mean_batch_size": lambda item: item["metrics"]["mean_batch_size"],
        "effective_batch_fill_ratio": lambda item: item["metrics"][
            "effective_batch_fill_ratio"
        ],
        "geometry_shared_request_ratio": lambda item: item["metrics"]["scheduler"][
            "geometry_shared_request_ratio"
        ],
        "wavelet_shared_request_ratio": lambda item: item["metrics"]["scheduler"][
            "wavelet_shared_request_ratio"
        ],
        "geometry_cache_hit_rate": lambda item: _cache_hit_rate(item, "geometry_prefix"),
        "wavelet_cache_hit_rate": lambda item: _cache_hit_rate(item, "wavelet"),
        "scheduler_us_per_request": lambda item: item["metrics"]["scheduler"][
            "selection_us_per_dispatched_request"
        ],
        "starvation_guard_requests": lambda item: item["metrics"]["scheduler"][
            "starvation_guard_requests"
        ],
        "deadline_violations": lambda item: item["metrics"]["requests"]["timed_out"],
        "peak_allocated_bytes": lambda item: item["metrics"]["memory"][
            "peak_allocated_bytes"
        ],
    }
    return {
        name: describe([selector(result) for result in results])
        for name, selector in selectors.items()
    }


def _reduction_percent(baseline: float, candidate: float) -> float:
    if baseline <= 0:
        raise ValueError("paired baseline must be positive")
    return (baseline - candidate) / baseline * 100.0


def _paired_comparison(
    scenario: str,
    pairs: Sequence[Tuple[int, Mapping[str, Any], Mapping[str, Any]]],
) -> Dict[str, Any]:
    raw_pairs = []
    for repeat_index, fcfs, cache_aware in pairs:
        fcfs_metrics = fcfs["metrics"]
        aware_metrics = cache_aware["metrics"]
        raw_pairs.append(
            {
                "repeat_index": repeat_index,
                "fcfs_run_id": fcfs["run_id"],
                "cache_aware_run_id": cache_aware["run_id"],
                "throughput_speedup": aware_metrics["throughput_requests_per_second"]
                / fcfs_metrics["throughput_requests_per_second"],
                "end_to_end_p99_reduction_percent": _reduction_percent(
                    fcfs_metrics["end_to_end_latency"]["p99_ms"],
                    aware_metrics["end_to_end_latency"]["p99_ms"],
                ),
                "queue_p99_reduction_percent": _reduction_percent(
                    fcfs_metrics["queue_latency"]["p99_ms"],
                    aware_metrics["queue_latency"]["p99_ms"],
                ),
                "batch_fill_ratio_delta": aware_metrics["effective_batch_fill_ratio"]
                - fcfs_metrics["effective_batch_fill_ratio"],
                "geometry_shared_request_ratio_delta": aware_metrics["scheduler"][
                    "geometry_shared_request_ratio"
                ]
                - fcfs_metrics["scheduler"]["geometry_shared_request_ratio"],
                "wavelet_shared_request_ratio_delta": aware_metrics["scheduler"][
                    "wavelet_shared_request_ratio"
                ]
                - fcfs_metrics["scheduler"]["wavelet_shared_request_ratio"],
                "geometry_cache_hit_rate_delta": _cache_hit_rate(
                    cache_aware, "geometry_prefix"
                )
                - _cache_hit_rate(fcfs, "geometry_prefix"),
                "wavelet_cache_hit_rate_delta": _cache_hit_rate(cache_aware, "wavelet")
                - _cache_hit_rate(fcfs, "wavelet"),
                "scheduler_us_per_request_delta": aware_metrics["scheduler"][
                    "selection_us_per_dispatched_request"
                ]
                - fcfs_metrics["scheduler"]["selection_us_per_dispatched_request"],
                "peak_allocated_delta_bytes": aware_metrics["memory"][
                    "peak_allocated_bytes"
                ]
                - fcfs_metrics["memory"]["peak_allocated_bytes"],
            }
        )
    metric_names = tuple(
        name
        for name in raw_pairs[0]
        if name not in {"repeat_index", "fcfs_run_id", "cache_aware_run_id"}
    )
    return {
        "scenario": scenario,
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
        raise ValueError("no formal scheduler result files were found")
    expected = set(expected_keys(config))
    by_key: Dict[Tuple[str, str, int], Mapping[str, Any]] = {}
    inputs = []
    for path, result in loaded:
        key = result_key(result)
        if key in by_key:
            raise ValueError(
                f"duplicate result for policy={key[0]}, scenario={key[1]}, run={key[2]}"
            )
        by_key[key] = result
        expected_run_id = f"scheduler_{key[0]}_{key[1]}_run{key[2]}"
        if result["run_id"] != expected_run_id:
            raise ValueError(
                f"run_id {result['run_id']} does not match matrix key {expected_run_id}"
            )
        inputs.append(
            {
                "run_id": result["run_id"],
                "path": path.name,
                "sha256": file_sha256(path),
                "status": result["status"],
                "policy": key[0],
                "scenario": key[1],
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
    if identity["config_sha256"] != file_sha256(config_path.resolve()):
        raise ValueError("formal run config SHA does not match the summary configuration")

    workload_hashes: Dict[str, set] = {}
    request_count = int(config["workloads"]["request_count"])
    for (policy, scenario, _repeat), result in by_key.items():
        if result["workload"]["request_count"] != request_count:
            raise ValueError(f"request count mismatch in {result['run_id']}")
        if result["execution"]["policy"] != policy:
            raise ValueError(f"scheduler policy mismatch in {result['run_id']}")
        if result["execution"]["graph_enabled"]:
            raise ValueError(f"CUDA Graph must remain disabled in {result['run_id']}")
        if result["execution"]["warmup_requests"] != config["measurement"][
            "warmup_requests"
        ]:
            raise ValueError(f"warmup request count mismatch in {result['run_id']}")
        workload_hashes.setdefault(scenario, set()).add(result["workload"]["sha256"])
    for scenario, hashes in workload_hashes.items():
        if len(hashes) != 1:
            raise ValueError(
                f"FCFS/cache-aware did not share one trace for scenario {scenario}"
            )

    passed = {key: value for key, value in by_key.items() if value["status"] == "passed"}
    missing = sorted(expected - set(by_key))
    failed_run_ids = sorted(
        result["run_id"] for result in by_key.values() if result["status"] != "passed"
    )
    scenario_order = [item["name"] for item in config["workloads"]["scenarios"]]
    groups = []
    for scenario in scenario_order:
        for policy in config["scheduler"]["policies"]:
            group_runs = [
                passed[key]
                for key in sorted(passed, key=lambda item: item[2])
                if key[0] == policy and key[1] == scenario
            ]
            if not group_runs:
                continue
            groups.append(
                {
                    "scenario": scenario,
                    "policy": policy,
                    "run_count": len(group_runs),
                    "reuse": group_runs[0]["workload"]["reuse"],
                    "metrics": _group_metrics(group_runs),
                }
            )

    comparisons = []
    repeats = int(config["measurement"]["independent_runs"])
    for scenario in scenario_order:
        pairs = []
        for repeat in range(1, repeats + 1):
            fcfs = passed.get(("fcfs", scenario, repeat))
            cache_aware = passed.get(("cache_aware", scenario, repeat))
            if fcfs is not None and cache_aware is not None:
                pairs.append((repeat, fcfs, cache_aware))
        if pairs:
            comparisons.append(_paired_comparison(scenario, pairs))

    status = "incomplete" if missing else "failed" if failed_run_ids else "passed"
    shared_gpu = any(
        result["environment"]["hardware"]["other_gpu_processes"]
        for result in by_key.values()
    )
    notes = [
        "All percentiles are summarized from run-level metrics; raw samples are not pooled.",
        "Each FCFS/cache-aware pair reads the same checked-in request trace.",
        "Throughput speedup is cache-aware / FCFS.",
        "Latency reduction is (FCFS - cache-aware) / FCFS * 100%.",
        "CUDA Graph is disabled for every run so the comparison isolates scheduling.",
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
            "independent_runs_per_configuration": repeats,
            "request_count_per_run": request_count,
            "raw_samples_pooled": False,
            "profiler_enabled": False,
            "cuda_graph_enabled": False,
            "shared_gpu_observed": shared_gpu,
        },
        "missing_matrix_entries": [
            {"policy": policy, "scenario": scenario, "repeat_index": repeat}
            for policy, scenario, repeat in missing
        ],
        "failed_run_ids": failed_run_ids,
        "inputs": sorted(
            inputs,
            key=lambda item: (
                scenario_order.index(item["scenario"]),
                item["repeat_index"],
                item["policy"],
            ),
        ),
        "groups": groups,
        "comparisons": comparisons,
        "notes": notes,
    }


def _format_number(value: float, digits: int = 3) -> str:
    if not math.isfinite(float(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def render_markdown(summary: Mapping[str, Any]) -> str:
    identity = summary["identity"]
    lines = [
        "# FCFS vs Cache-Aware scheduler formal summary",
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
        "![Reuse degree versus throughput](reuse_throughput.svg)",
        "",
        "## Run-level metrics",
        "",
        (
            "| Scenario | Policy | Runs | req/s | Queue p99 ms | E2E p99 ms | "
            "Mean batch | Fill | Geometry shared | Wavelet shared | Scheduler us/req |"
        ),
        "|:---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in summary["groups"]:
        metrics = group["metrics"]
        lines.append(
            "| {scenario} | {policy} | {runs} | {throughput} | {queue} | {e2e} | "
            "{batch} | {fill}% | {geometry}% | {wavelet}% | {overhead} |".format(
                scenario=group["scenario"],
                policy=group["policy"],
                runs=group["run_count"],
                throughput=_format_number(
                    metrics["throughput_requests_per_second"]["median"], 2
                ),
                queue=_format_number(metrics["queue_p99_ms"]["median"]),
                e2e=_format_number(metrics["end_to_end_p99_ms"]["median"]),
                batch=_format_number(metrics["mean_batch_size"]["median"], 2),
                fill=_format_number(
                    metrics["effective_batch_fill_ratio"]["median"] * 100.0, 1
                ),
                geometry=_format_number(
                    metrics["geometry_shared_request_ratio"]["median"] * 100.0, 1
                ),
                wavelet=_format_number(
                    metrics["wavelet_shared_request_ratio"]["median"] * 100.0, 1
                ),
                overhead=_format_number(metrics["scheduler_us_per_request"]["median"], 2),
            )
        )
    lines.extend(
        [
            "",
            "## Paired Cache-Aware vs FCFS comparison",
            "",
            (
                "| Scenario | Pairs | Throughput speedup | E2E p99 reduction | "
                "Queue p99 reduction | Fill delta | Geometry hit delta | Peak delta MiB |"
            ),
            "|:---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for comparison in summary["comparisons"]:
        metrics = comparison["metrics"]
        lines.append(
            "| {scenario} | {pairs} | {speedup}x | {e2e}% | {queue}% | "
            "{fill} | {hit} | {memory} |".format(
                scenario=comparison["scenario"],
                pairs=comparison["paired_run_count"],
                speedup=_format_number(metrics["throughput_speedup"]["median"], 3),
                e2e=_format_number(
                    metrics["end_to_end_p99_reduction_percent"]["median"], 2
                ),
                queue=_format_number(
                    metrics["queue_p99_reduction_percent"]["median"], 2
                ),
                fill=_format_number(metrics["batch_fill_ratio_delta"]["median"], 3),
                hit=_format_number(
                    metrics["geometry_cache_hit_rate_delta"]["median"], 3
                ),
                memory=_format_number(
                    metrics["peak_allocated_delta_bytes"]["median"] / (1024 * 1024),
                    2,
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


def render_reuse_throughput_svg(summary: Mapping[str, Any]) -> str:
    width, height = 760, 440
    left, right, top, bottom = 80, 30, 40, 95
    chart_width = width - left - right
    chart_height = height - top - bottom
    policies = ("fcfs", "cache_aware")
    colors = {"fcfs": "#6b7280", "cache_aware": "#2563eb"}
    group_map = {
        (group["scenario"], group["policy"]): group for group in summary["groups"]
    }
    scenarios = []
    for comparison in summary["comparisons"]:
        scenario = comparison["scenario"]
        group = group_map.get((scenario, "fcfs")) or group_map.get(
            (scenario, "cache_aware")
        )
        if group is not None:
            scenarios.append(
                (scenario, float(group["reuse"]["geometry_reuse_ratio"]))
            )
    scenarios.sort(key=lambda item: item[1])
    values = [
        float(group_map[(scenario, policy)]["metrics"]["throughput_requests_per_second"]["median"])
        for scenario, _reuse in scenarios
        for policy in policies
        if (scenario, policy) in group_map
    ]
    maximum = max(values, default=1.0) * 1.1

    def x_position(index: int) -> float:
        if len(scenarios) <= 1:
            return left + chart_width / 2
        return left + chart_width * index / (len(scenarios) - 1)

    def y_position(value: float) -> float:
        return top + chart_height * (1.0 - value / maximum)

    lines = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="white"/>',
        (
            '<text x="380" y="24" text-anchor="middle" font-family="sans-serif" '
            'font-size="17">Reuse degree vs throughput</text>'
        ),
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" stroke="#111827"/>',
        (
            f'<line x1="{left}" y1="{top + chart_height}" '
            f'x2="{left + chart_width}" y2="{top + chart_height}" '
            'stroke="#111827"/>'
        ),
    ]
    for tick in range(5):
        value = maximum * tick / 4
        y = y_position(value)
        lines.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_width}" '
            f'y2="{y:.1f}" stroke="#e5e7eb"/>'
        )
        lines.append(
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-family="sans-serif" font-size="11">{value:.0f}</text>'
        )
    for index, (scenario, reuse) in enumerate(scenarios):
        x = x_position(index)
        lines.append(
            f'<text x="{x:.1f}" y="{top + chart_height + 22}" '
            f'text-anchor="middle" font-family="sans-serif" font-size="11">'
            f"{html.escape(scenario)}</text>"
        )
        lines.append(
            f'<text x="{x:.1f}" y="{top + chart_height + 38}" '
            f'text-anchor="middle" font-family="sans-serif" font-size="10">'
            f"geometry reuse {reuse * 100:.1f}%</text>"
        )
    for policy in policies:
        points = []
        for index, (scenario, _reuse) in enumerate(scenarios):
            group = group_map.get((scenario, policy))
            if group is None:
                continue
            value = float(
                group["metrics"]["throughput_requests_per_second"]["median"]
            )
            points.append((x_position(index), y_position(value), value))
        if points:
            encoded = " ".join(f"{x:.1f},{y:.1f}" for x, y, _value in points)
            lines.append(
                f'<polyline points="{encoded}" fill="none" '
                f'stroke="{colors[policy]}" stroke-width="2.5"/>'
            )
            for x, y, value in points:
                lines.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{colors[policy]}"/>'
                )
                lines.append(
                    f'<text x="{x:.1f}" y="{y - 8:.1f}" text-anchor="middle" '
                    f'font-family="sans-serif" font-size="10" '
                    f'fill="{colors[policy]}">{value:.0f}</text>'
                )
    legend_x = left + 10
    for index, policy in enumerate(policies):
        x = legend_x + index * 150
        lines.append(
            f'<line x1="{x}" y1="{height - 20}" x2="{x + 24}" '
            f'y2="{height - 20}" stroke="{colors[policy]}" stroke-width="3"/>'
        )
        lines.append(
            f'<text x="{x + 30}" y="{height - 16}" font-family="sans-serif" '
            f'font-size="12">{html.escape(policy)}</text>'
        )
    lines.append(
        f'<text x="18" y="{top + chart_height / 2:.1f}" '
        f'transform="rotate(-90 18 {top + chart_height / 2:.1f})" '
        'text-anchor="middle" font-family="sans-serif" font-size="12">'
        "requests/s</text>"
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def summarize_directory(
    config_path: Path,
    input_dir: Path,
    output_json: Path,
    output_markdown: Path,
    output_svg: Path,
) -> Dict[str, Any]:
    config = load_config(config_path)
    errors, kind, _schema_status = validate_document(config, kind="config")
    if errors:
        raise ValueError(f"invalid {kind}: {'; '.join(errors)}")
    paths = input_dir.glob("scheduler_*_run*.json")
    summary = build_summary(config, config_path, load_results(paths))
    write_json_atomic(output_json, summary)
    write_text_atomic(output_markdown, render_markdown(summary))
    write_text_atomic(output_svg, render_reuse_throughput_svg(summary))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--output-svg", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_json = args.output_json or args.input_dir / "summary.json"
    output_markdown = args.output_markdown or args.input_dir / "summary.md"
    output_svg = args.output_svg or args.input_dir / "reuse_throughput.svg"
    summary = summarize_directory(
        args.config.resolve(),
        args.input_dir.resolve(),
        output_json.resolve(),
        output_markdown.resolve(),
        output_svg.resolve(),
    )
    print(f"saved: {output_json.resolve()}")
    print(f"saved: {output_markdown.resolve()}")
    print(f"saved: {output_svg.resolve()}")
    print(f"status: {summary['status']}")
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
