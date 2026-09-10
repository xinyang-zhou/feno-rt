#!/usr/bin/env python3
"""Generate the deterministic, redistribution-safe Stage 6 demo inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "data" / "demo"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, velocity_name: str) -> None:
    metadata: Dict[str, Any] = {
        "schema_version": 1,
        "kind": "metadata",
        "created_at": "2026-01-01T00:00:00+00:00",
        "metadata": {
            "description": "Deterministic synthetic Stage 6 replay trace",
            "context_id": "demo",
            "velocity_file": velocity_name,
            "synthetic": True,
        },
    }
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, sort_keys=True) + "\n")
        for sequence in range(24):
            row = sequence % 6
            column = (sequence * 5) % 8
            request = {
                "context_id": "demo",
                "source_position": [2.0 + row * 1.8, 1.0 + column * 1.7],
                "frequency": 3.0 + float(sequence % 5) * 0.75,
                "receiver_positions": None,
                "positions_are_normalized": False,
                "denormalize": False,
                "cache_level": "all",
                "timeout_s": 30.0,
                "priority": sequence % 3,
            }
            record = {
                "schema_version": 1,
                "kind": "request",
                "sequence": sequence,
                "recorded_at": "2026-01-01T00:00:00+00:00",
                "request": request,
                # The public trace deliberately contains no model output. The
                # benchmark validates determinism by comparing two local runs.
                "result": {"status": "unverified"},
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def generate(output_dir: Path, *, force: bool = False) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    velocity_path = output_dir / "velocity.npy"
    trace_path = output_dir / "requests.jsonl"
    manifest_path = output_dir / "manifest.json"
    targets = (velocity_path, trace_path, manifest_path)
    existing = [path for path in targets if path.exists()]
    if existing and not force:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"refusing to overwrite existing demo files: {names}")
    if force:
        for path in existing:
            path.unlink()

    axis = np.linspace(-1.0, 1.0, 16, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(axis, axis, indexing="ij")
    anomaly = np.exp(-10.0 * ((grid_x - 0.25) ** 2 + (grid_y + 0.2) ** 2))
    velocity = (
        2_000.0
        + 350.0 * (grid_x + 1.0) / 2.0
        + 240.0 * anomaly.astype(np.float32)
    ).astype(np.float32)
    np.save(velocity_path, velocity, allow_pickle=False)
    _write_jsonl(trace_path, velocity_path.name)

    manifest = {
        "schema_version": 1,
        "generator": "scripts/data/generate_demo.py",
        "deterministic": True,
        "license": "CC0-1.0",
        "files": {
            velocity_path.name: {
                "sha256": _sha256(velocity_path),
                "shape": list(velocity.shape),
                "dtype": str(velocity.dtype),
            },
            trace_path.name: {
                "sha256": _sha256(trace_path),
                "requests": 24,
            },
        },
    }
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace only the three known generated files in the output directory",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(json.dumps(generate(arguments.output_dir, force=arguments.force), indent=2))
