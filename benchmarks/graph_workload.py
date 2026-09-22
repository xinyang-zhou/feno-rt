"""Build deterministic workload manifests for the CUDA Graph A/B benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "benchmarks/configs/graph_ab.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "benchmarks/workloads"


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise ValueError("Graph A/B configuration must be a JSON object")
    return document


def build_manifest(config: Mapping[str, Any], batch_size: int) -> Dict[str, Any]:
    workload = config["workload"]
    configured_batches = workload["batch_sizes"]
    if batch_size not in configured_batches:
        raise ValueError(f"batch size {batch_size} is not configured")
    request_count = int(workload["request_count"])
    if request_count % batch_size:
        raise ValueError("request_count must be divisible by batch_size")

    positions = workload["source_positions"][:batch_size]
    frequencies = workload["frequency_hz"]
    batch_template = [
        {
            "slot": slot,
            "source_position": [float(value) for value in position],
            "frequency_hz": float(frequencies[slot % len(frequencies)]),
        }
        for slot, position in enumerate(positions)
    ]
    if len(batch_template) != batch_size:
        raise ValueError("source_positions does not cover batch_size")

    return {
        "schema_version": workload["schema_version"],
        "benchmark": config["benchmark"],
        "name": workload["name"],
        "seed": int(config["seed"]),
        "request_count": request_count,
        "batch_size": batch_size,
        "invocation_count": request_count // batch_size,
        "arrival_pattern": workload["arrival_pattern"],
        "request_sequence": workload["request_sequence"],
        "positions_are_normalized": workload["positions_are_normalized"],
        "receiver_positions": workload["receiver_positions"],
        "velocity": dict(workload["velocity"]),
        "cache_state": dict(workload["cache_state"]),
        "batch_template": batch_template,
    }


def manifest_text(manifest: Mapping[str, Any]) -> str:
    return json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def manifest_path(output_dir: Path, batch_size: int) -> Path:
    return output_dir / f"graph_ab_batch{batch_size}.json"


def write_manifests(
    config: Mapping[str, Any], output_dir: Path, batch_sizes: Iterable[int]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for batch_size in batch_sizes:
        path = manifest_path(output_dir, batch_size)
        path.write_text(
            manifest_text(build_manifest(config, batch_size)), encoding="utf-8"
        )
        print(f"wrote {path.relative_to(PROJECT_ROOT)}")


def check_manifests(
    config: Mapping[str, Any], output_dir: Path, batch_sizes: Iterable[int]
) -> bool:
    valid = True
    for batch_size in batch_sizes:
        path = manifest_path(output_dir, batch_size)
        expected = manifest_text(build_manifest(config, batch_size))
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
    batch_sizes = config["workload"]["batch_sizes"]
    if args.write:
        write_manifests(config, args.output_dir, batch_sizes)
        return 0
    return 0 if check_manifests(config, args.output_dir, batch_sizes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
