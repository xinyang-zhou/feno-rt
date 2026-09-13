#!/usr/bin/env python3
"""Verify separately distributed FENO-RT checkpoint release assets."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any, Dict, Mapping

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from feno_rt.config import DEFAULT_INFERENCE_CONFIG
from feno_rt.models.feno_freq import FENOFreq
from feno_rt.preprocessing import NormalizationStats
from feno_rt.runtime import FENOModelRunner


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported release manifest schema")
    if payload.get("purpose") != "performance-only":
        raise ValueError("feno_test must remain a performance-only asset")
    return payload


def verify_file(path: Path, specification: Mapping[str, Any], label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if path.name != specification["filename"]:
        raise ValueError(
            f"{label} filename mismatch: expected {specification['filename']}, "
            f"found {path.name}"
        )
    actual_size = path.stat().st_size
    if actual_size != specification["size_bytes"]:
        raise ValueError(
            f"{label} size mismatch: expected {specification['size_bytes']}, "
            f"found {actual_size}"
        )
    actual_digest = file_sha256(path)
    if actual_digest != specification["sha256"]:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {specification['sha256']}, "
            f"found {actual_digest}"
        )
    return actual_digest


def synthetic_velocity(normalization: NormalizationStats) -> np.ndarray:
    config = DEFAULT_INFERENCE_CONFIG
    z = np.linspace(0.0, 1.0, config.velocity_height, dtype=np.float32)
    x = np.linspace(0.0, 1.0, config.velocity_width, dtype=np.float32)
    z_grid, x_grid = np.meshgrid(z, x, indexing="ij")
    normalized = (
        0.45 * np.sin(2.0 * np.pi * z_grid)
        + 0.20 * np.cos(4.0 * np.pi * x_grid)
    )
    return (
        normalization.v_mean + normalization.v_std * normalized
    ).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument(
        "--device",
        default=None,
        help="optional single device smoke test, for example cuda:0",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    checkpoint_spec = manifest["checkpoint"]
    normalization_spec = manifest["normalization"]

    checkpoint_digest = verify_file(
        args.checkpoint, checkpoint_spec, "checkpoint"
    )
    normalization_digest = verify_file(
        args.normalization, normalization_spec, "normalization"
    )

    with np.load(args.normalization) as values:
        actual_keys = sorted(values.files)
    if actual_keys != sorted(normalization_spec["keys"]):
        raise ValueError(
            f"normalization keys mismatch: expected {normalization_spec['keys']}, "
            f"found {actual_keys}"
        )
    normalization = NormalizationStats.load(args.normalization)
    if normalization.v_std <= 0 or normalization.seis_std <= 0:
        raise ValueError("normalization standard deviations must be positive")

    state_dict = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True
    )
    if not isinstance(state_dict, Mapping):
        raise TypeError("checkpoint must contain a state-dict mapping")
    if not all(isinstance(key, str) for key in state_dict):
        raise TypeError("all state-dict keys must be strings")
    tensor_values = [value for value in state_dict.values() if torch.is_tensor(value)]
    tensor_count = len(tensor_values)
    non_tensor_entries = len(state_dict) - len(tensor_values)
    dtypes = sorted({str(value.dtype) for value in tensor_values})
    if tensor_count != checkpoint_spec["tensor_count"]:
        raise ValueError("checkpoint tensor count does not match the manifest")
    if non_tensor_entries != checkpoint_spec["non_tensor_entries"]:
        raise ValueError("checkpoint contains unexpected non-tensor entries")
    if dtypes != sorted(checkpoint_spec["dtypes"]):
        raise ValueError(f"checkpoint dtype mismatch: found {dtypes}")
    for key, value in state_dict.items():
        if not torch.isfinite(value).all():
            raise ValueError(f"checkpoint contains a non-finite tensor: {key}")

    config = DEFAULT_INFERENCE_CONFIG
    model = FENOFreq(config.encoder_config(), config.decoder_config()).eval()
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(f"strict state-dict load failed: {incompatible}")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != checkpoint_spec["parameter_count"]:
        raise ValueError(
            f"parameter count mismatch: expected {checkpoint_spec['parameter_count']}, "
            f"found {parameter_count}"
        )
    del tensor_values
    del state_dict

    report: Dict[str, Any] = {
        "schema_version": 1,
        "status": "passed",
        "purpose": manifest["purpose"],
        "package_version": manifest["package_version"],
        "checkpoint": {
            "filename": args.checkpoint.name,
            "size_bytes": args.checkpoint.stat().st_size,
            "sha256": checkpoint_digest,
            "tensor_count": tensor_count,
            "parameter_count": parameter_count,
            "dtypes": dtypes,
            "strict_load": True,
            "all_tensors_finite": True,
        },
        "normalization": {
            "filename": args.normalization.name,
            "size_bytes": args.normalization.stat().st_size,
            "sha256": normalization_digest,
            "keys": actual_keys,
            "positive_standard_deviations": True,
        },
    }

    if args.device is not None:
        device = torch.device(args.device)
        if device.type != "cuda" or device.index is None:
            raise ValueError("--device must be one explicit CUDA device")
        if not torch.cuda.is_available() or device.index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device is not available: {device}")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        runner = FENOModelRunner(
            model,
            config=config,
            normalization=normalization,
            device=device,
            model_version=f"release:{checkpoint_digest}",
        )
        velocity = synthetic_velocity(normalization)
        source = np.asarray(
            [[config.velocity_height // 2, config.velocity_width // 2]],
            dtype=np.float32,
        )
        context = runner.prepare_medium(velocity, use_cache=False)
        reference = runner.forward_batch(
            context, source, np.asarray([15.0], dtype=np.float32), cache_level="medium"
        )
        cached = runner.forward_batch(
            context, source, np.asarray([15.0], dtype=np.float32), cache_level="all"
        )
        torch.cuda.synchronize(device)
        if tuple(cached.shape) != (
            1,
            config.num_receivers,
            config.output_steps,
        ):
            raise ValueError(f"unexpected smoke output shape: {tuple(cached.shape)}")
        if not torch.isfinite(reference).all() or not torch.isfinite(cached).all():
            raise ValueError("single-GPU smoke output contains non-finite values")
        difference = torch.linalg.vector_norm(cached - reference)
        denominator = torch.linalg.vector_norm(reference).clamp_min(1e-12)
        report["single_gpu_smoke"] = {
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "output_shape": list(cached.shape),
            "outputs_finite": True,
            "cached_vs_reference_relative_l2": float(
                (difference / denominator).item()
            ),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        }

    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
