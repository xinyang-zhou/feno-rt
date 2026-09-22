"""Build deterministic request traces for the scheduler A/B benchmark."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "benchmarks/configs/scheduler_ab.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "benchmarks/workloads"


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise ValueError("scheduler A/B configuration must be a JSON object")
    return document


def scenario_by_name(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    matches = [
        scenario
        for scenario in config["workloads"]["scenarios"]
        if scenario["name"] == name
    ]
    if len(matches) != 1:
        raise ValueError(f"unknown or duplicate scheduler scenario: {name}")
    return matches[0]


def _weighted_index(rng: random.Random, size: int, exponent: float) -> int:
    weights = [1.0 / math.pow(index + 1, exponent) for index in range(size)]
    threshold = rng.random() * sum(weights)
    cumulative = 0.0
    for index, weight in enumerate(weights):
        cumulative += weight
        if threshold <= cumulative:
            return index
    return size - 1


def _pool_index(
    rng: random.Random,
    *,
    request_index: int,
    size: int,
    scenario: Mapping[str, Any],
) -> int:
    pattern = scenario["pattern"]
    if pattern == "unique":
        return request_index
    if pattern == "uniform":
        return rng.randrange(size)
    if pattern == "zipf":
        return _weighted_index(rng, size, float(scenario["zipf_exponent"]))
    if pattern == "hotspot":
        if rng.random() < float(scenario["hotspot_probability"]):
            return 0
        return 1 + rng.randrange(size - 1) if size > 1 else 0
    raise ValueError(f"unsupported scheduler workload pattern: {pattern}")


def _source_position(index: int) -> Sequence[float]:
    row, column = divmod(index, 32)
    return [float(23 + row * 10), float(10 + column * 20)]


def _frequency_hz(index: int, *, unique: bool) -> float:
    return float(8.0 + index * (0.01 if unique else 2.0))


def build_manifest(config: Mapping[str, Any], scenario_name: str) -> Dict[str, Any]:
    workloads = config["workloads"]
    scenario = scenario_by_name(config, scenario_name)
    request_count = int(workloads["request_count"])
    medium_count = int(workloads["medium_count"])
    geometry_pool_size = int(scenario["geometry_pool_size"])
    frequency_pool_size = int(scenario["frequency_pool_size"])
    if request_count < 1 or medium_count < 1:
        raise ValueError("request_count and medium_count must be positive")
    if geometry_pool_size < 1 or frequency_pool_size < 1:
        raise ValueError("scheduler workload pools must be positive")
    if geometry_pool_size > 1024:
        raise ValueError("geometry_pool_size exceeds the deterministic source grid")

    scenario_index = [
        item["name"] for item in workloads["scenarios"]
    ].index(scenario_name)
    rng = random.Random(int(config["seed"]) + 10_007 * scenario_index)
    requests = []
    for request_index in range(request_count):
        medium_index = (request_index * 3 + scenario_index) % medium_count
        geometry_index = _pool_index(
            rng,
            request_index=request_index,
            size=geometry_pool_size,
            scenario=scenario,
        )
        frequency_index = _pool_index(
            rng,
            request_index=request_index,
            size=frequency_pool_size,
            scenario=scenario,
        )
        requests.append(
            {
                "request_id": f"{scenario_name}-{request_index:04d}",
                "sequence_index": request_index,
                "arrival_offset_us": int(workloads["arrival_interval_us"])
                * request_index,
                "timeout_us": int(workloads["slo_timeout_us"]),
                "medium_id": f"medium-{medium_index}",
                "geometry_id": f"geometry-{geometry_index}",
                "wavelet_id": f"frequency-{frequency_index}",
                "source_position": list(_source_position(geometry_index)),
                "frequency_hz": _frequency_hz(
                    frequency_index,
                    unique=scenario["pattern"] == "unique",
                ),
                "priority": 0,
            }
        )

    unique_mediums = len({item["medium_id"] for item in requests})
    unique_geometries = len(
        {(item["medium_id"], item["geometry_id"]) for item in requests}
    )
    unique_wavelets = len({item["wavelet_id"] for item in requests})
    reuse = {
        "unique_mediums": unique_mediums,
        "unique_geometries": unique_geometries,
        "unique_wavelets": unique_wavelets,
        "medium_reuse_ratio": 1.0 - unique_mediums / request_count,
        "geometry_reuse_ratio": 1.0 - unique_geometries / request_count,
        "wavelet_reuse_ratio": 1.0 - unique_wavelets / request_count,
    }
    mediums = [
        {
            "medium_id": f"medium-{index}",
            "family_index": index,
        }
        for index in range(medium_count)
    ]
    return {
        "schema_version": workloads["schema_version"],
        "benchmark": config["benchmark"],
        "name": scenario_name,
        "seed": int(config["seed"]) + 10_007 * scenario_index,
        "request_count": request_count,
        "arrival_pattern": workloads["arrival_pattern"],
        "positions_are_normalized": workloads["positions_are_normalized"],
        "receiver_positions": workloads["receiver_positions"],
        "cache_state": dict(workloads["cache_state"]),
        "velocity": dict(workloads["velocity"]),
        "mediums": mediums,
        "reuse": reuse,
        "requests": requests,
    }


def manifest_text(manifest: Mapping[str, Any]) -> str:
    return json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def manifest_path(output_dir: Path, scenario_name: str) -> Path:
    return output_dir / f"scheduler_{scenario_name}.json"


def scenario_names(config: Mapping[str, Any]) -> Iterable[str]:
    return [item["name"] for item in config["workloads"]["scenarios"]]


def write_manifests(config: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in scenario_names(config):
        path = manifest_path(output_dir, name)
        path.write_text(manifest_text(build_manifest(config, name)), encoding="utf-8")
        print(f"wrote {path.relative_to(PROJECT_ROOT)}")


def check_manifests(config: Mapping[str, Any], output_dir: Path) -> bool:
    valid = True
    for name in scenario_names(config):
        path = manifest_path(output_dir, name)
        expected = manifest_text(build_manifest(config, name))
        if not path.exists():
            print(f"missing {path.relative_to(PROJECT_ROOT)}", file=sys.stderr)
            valid = False
        elif path.read_text(encoding="utf-8") != expected:
            print(f"stale {path.relative_to(PROJECT_ROOT)}", file=sys.stderr)
            valid = False
        else:
            print(f"valid {path.relative_to(PROJECT_ROOT)}")
    return valid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.write:
        write_manifests(config, args.output_dir)
        return 0
    return 0 if check_manifests(config, args.output_dir) else 1


if __name__ == "__main__":
    raise SystemExit(main())
