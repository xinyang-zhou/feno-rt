#!/usr/bin/env python3
"""Single-GPU, performance-only FENO-RT and Deepwave comparison."""

from __future__ import annotations

import argparse
import gc
from hashlib import sha256
import importlib.metadata
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from feno_rt.config import DEFAULT_INFERENCE_CONFIG, FENOModelConfig
from feno_rt.preprocessing import NormalizationStats
from feno_rt.runtime import FENOModelRunner


def synchronize(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def synthetic_velocity(
    config: FENOModelConfig, normalization: NormalizationStats
) -> np.ndarray:
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


def workload(
    config: FENOModelConfig,
    batch_size: int,
    frequency_hz: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + batch_size)
    margin = 40
    sources = rng.integers(
        low=margin,
        high=min(config.velocity_height, config.velocity_width) - margin,
        size=(batch_size, 2),
        dtype=np.int64,
    ).astype(np.float32)
    frequencies = np.full(batch_size, frequency_hz, dtype=np.float32)
    return sources, frequencies


def measure(
    function: Callable[[], Any],
    *,
    device: torch.device,
    batch_size: int,
    warmup: int,
    repeats: int,
) -> Dict[str, Any]:
    for _ in range(warmup):
        output = function()
        del output
    synchronize(device)

    baseline_allocated = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    samples: List[float] = []
    output_shape = None
    for _ in range(repeats):
        start = time.perf_counter()
        output = function()
        synchronize(device)
        samples.append((time.perf_counter() - start) * 1000.0)
        if torch.is_tensor(output):
            output_shape = list(output.shape)
        del output

    values = np.asarray(samples, dtype=np.float64)
    median_ms = float(np.median(values))
    return {
        "mean_ms": float(values.mean()),
        "median_ms": median_ms,
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
        "stdev_ms": float(statistics.pstdev(samples)),
        "requests_per_second": batch_size / (median_ms / 1000.0),
        "incremental_peak_allocated_bytes": max(
            0,
            torch.cuda.max_memory_allocated(device) - baseline_allocated,
        ),
        "output_shape": output_shape,
    }


def run_feno(
    args: argparse.Namespace,
    device: torch.device,
    config: FENOModelConfig,
    normalization: NormalizationStats,
    velocity: np.ndarray,
) -> Dict[str, Any]:
    synchronize(device)
    load_start = time.perf_counter()
    runner = FENOModelRunner.from_checkpoint(
        args.checkpoint,
        config=config,
        normalization=normalization,
        device=device,
        precision=args.precision,
        sdpa_backend=args.sdpa_backend,
    )
    synchronize(device)
    checkpoint_load_ms = (time.perf_counter() - load_start) * 1000.0
    model_resident_bytes = torch.cuda.memory_allocated(device)

    cold_prepare = measure(
        lambda: runner.prepare_medium(velocity, use_cache=False),
        device=device,
        batch_size=1,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    context = runner.prepare_medium(velocity, use_cache=True)
    synchronize(device)
    context_resident_bytes = (
        torch.cuda.memory_allocated(device) - model_resident_bytes
    )
    hot_prepare = measure(
        lambda: runner.prepare_medium(velocity, use_cache=True),
        device=device,
        batch_size=1,
        warmup=args.warmup,
        repeats=args.repeats,
    )

    results = []
    for batch_size in args.batch_sizes:
        sources, frequencies = workload(
            config, batch_size, args.frequency_hz, args.seed
        )
        reference = runner.forward_batch(
            context, sources, frequencies, cache_level="medium"
        )
        cached = runner.forward_batch(
            context, sources, frequencies, cache_level="all"
        )
        synchronize(device)
        if not torch.isfinite(reference).all() or not torch.isfinite(cached).all():
            raise ValueError("FENO produced non-finite output")
        relative_l2 = (
            torch.linalg.vector_norm(cached - reference)
            / torch.linalg.vector_norm(reference).clamp_min(1e-12)
        ).item()
        output_shape = list(cached.shape)
        del reference
        del cached

        medium_only = measure(
            lambda: runner.forward_batch(
                context, sources, frequencies, cache_level="medium"
            ),
            device=device,
            batch_size=batch_size,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        all_cached = measure(
            lambda: runner.forward_batch(
                context, sources, frequencies, cache_level="all"
            ),
            device=device,
            batch_size=batch_size,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        uncached = measure(
            lambda: runner.forward_uncached(velocity, sources, frequencies),
            device=device,
            batch_size=batch_size,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        results.append(
            {
                "batch_size": batch_size,
                "output_shape": output_shape,
                "cache_correctness_relative_l2": relative_l2,
                "medium_only": medium_only,
                "all_cached": all_cached,
                "uncached": uncached,
                "all_cache_speedup_vs_medium_only": (
                    medium_only["median_ms"] / all_cached["median_ms"]
                ),
                "all_cache_speedup_vs_uncached": (
                    uncached["median_ms"] / all_cached["median_ms"]
                ),
            }
        )
        print(
            f"FENO batch={batch_size}: hot={all_cached['median_ms']:.3f} ms "
            f"uncached={uncached['median_ms']:.3f} ms"
        )

    payload = {
        "checkpoint_load_ms": checkpoint_load_ms,
        "model_resident_bytes": model_resident_bytes,
        "medium_context_resident_bytes": context_resident_bytes,
        "cold_prepare_medium": cold_prepare,
        "hot_prepare_medium": hot_prepare,
        "execution_config": runner.execution_config(),
        "results": results,
    }
    del context
    del runner
    gc.collect()
    torch.cuda.empty_cache()
    synchronize(device)
    return payload


def ricker_batch(
    frequencies: np.ndarray,
    nt: int,
    dt: float,
    device: torch.device,
) -> torch.Tensor:
    frequency = torch.as_tensor(
        frequencies, device=device, dtype=torch.float32
    ).reshape(-1, 1)
    time_axis = torch.arange(nt, device=device, dtype=torch.float32).reshape(1, -1)
    time_axis = time_axis * dt
    peak_time = 1.5 / frequency
    argument = torch.pi * frequency * (time_axis - peak_time)
    return ((1.0 - 2.0 * argument.square()) * torch.exp(-argument.square())).unsqueeze(1)


def run_deepwave(
    args: argparse.Namespace,
    device: torch.device,
    config: FENOModelConfig,
    velocity: np.ndarray,
) -> Dict[str, Any]:
    import deepwave

    velocity_device = torch.as_tensor(
        velocity, device=device, dtype=torch.float32
    )
    receiver = torch.zeros(
        config.num_receivers, 2, device=device, dtype=torch.long
    )
    receiver[:, 0] = config.receiver_depth
    receiver[:, 1] = torch.arange(
        config.num_receivers, device=device, dtype=torch.long
    )
    dt = args.duration_seconds / (config.output_steps - 1)
    max_velocity = float(velocity.max())
    results = []
    for batch_size in args.batch_sizes:
        sources, frequencies = workload(
            config, batch_size, args.frequency_hz, args.seed
        )
        source_locations = torch.as_tensor(
            sources, device=device, dtype=torch.long
        ).unsqueeze(1)
        receiver_locations = receiver.unsqueeze(0).expand(
            batch_size, -1, -1
        ).contiguous()
        source_amplitudes = ricker_batch(
            frequencies,
            config.output_steps,
            dt,
            device,
        )

        def forward() -> torch.Tensor:
            with torch.inference_mode():
                return deepwave.scalar(
                    velocity_device,
                    args.grid_spacing_m,
                    dt,
                    source_amplitudes=source_amplitudes,
                    source_locations=source_locations,
                    receiver_locations=receiver_locations,
                    accuracy=args.deepwave_accuracy,
                    pml_width=args.pml_width,
                    pml_freq=args.frequency_hz,
                    max_vel=max_velocity,
                )[-1]

        timing = measure(
            forward,
            device=device,
            batch_size=batch_size,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        expected_shape = [
            batch_size,
            config.num_receivers,
            config.output_steps,
        ]
        if timing["output_shape"] != expected_shape:
            raise ValueError(
                f"Deepwave output shape mismatch: expected {expected_shape}, "
                f"found {timing['output_shape']}"
            )
        results.append({"batch_size": batch_size, "forward": timing})
        print(
            f"Deepwave batch={batch_size}: "
            f"forward={timing['median_ms']:.3f} ms"
        )

    return {
        "version": importlib.metadata.version("deepwave"),
        "grid_spacing_m": args.grid_spacing_m,
        "dt_seconds": dt,
        "duration_seconds": args.duration_seconds,
        "accuracy": args.deepwave_accuracy,
        "pml_width": args.pml_width,
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--frequency-hz", type=float, default=15.0)
    parser.add_argument("--duration-seconds", type=float, default=0.5)
    parser.add_argument("--grid-spacing-m", type=float, default=10.0)
    parser.add_argument("--deepwave-accuracy", type=int, choices=(2, 4, 6, 8), default=4)
    parser.add_argument("--pml-width", type=int, default=20)
    parser.add_argument(
        "--precision",
        choices=("fp32", "tf32", "bf16", "fp16"),
        default="fp32",
    )
    parser.add_argument("--sdpa-backend", default="auto")
    parser.add_argument("--skip-deepwave", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "benchmarks"
        / "feno_test_single_gpu.json",
    )
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats <= 0:
        parser.error("warmup must be non-negative and repeats must be positive")
    if not args.batch_sizes or any(value <= 0 for value in args.batch_sizes):
        parser.error("batch sizes must be positive")
    if args.frequency_hz <= 0 or args.duration_seconds <= 0:
        parser.error("frequency and duration must be positive")
    if args.grid_spacing_m <= 0 or args.pml_width < 0:
        parser.error("grid spacing must be positive and PML width non-negative")
    return args


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or device.index is None:
        raise ValueError("benchmark requires one explicit CUDA device")
    if not torch.cuda.is_available() or device.index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device is not available: {device}")
    torch.cuda.set_device(device)

    config = DEFAULT_INFERENCE_CONFIG
    normalization = NormalizationStats.load(args.normalization)
    velocity = synthetic_velocity(config, normalization)
    feno = run_feno(args, device, config, normalization, velocity)
    deepwave = None
    if not args.skip_deepwave:
        deepwave = run_deepwave(args, device, config, velocity)

    comparisons = []
    if deepwave is not None:
        for feno_result, deepwave_result in zip(
            feno["results"], deepwave["results"]
        ):
            deepwave_ms = deepwave_result["forward"]["median_ms"]
            comparisons.append(
                {
                    "batch_size": feno_result["batch_size"],
                    "feno_hot_speedup_over_deepwave": (
                        deepwave_ms / feno_result["all_cached"]["median_ms"]
                    ),
                    "feno_uncached_speedup_over_deepwave": (
                        deepwave_ms / feno_result["uncached"]["median_ms"]
                    ),
                }
            )

    properties = torch.cuda.get_device_properties(device)
    payload = {
        "schema_version": 1,
        "scope": "single_gpu_performance_only",
        "accuracy_claim": False,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {
            "filename": args.checkpoint.name,
            "size_bytes": args.checkpoint.stat().st_size,
            "sha256": file_sha256(args.checkpoint),
        },
        "normalization": {
            "filename": args.normalization.name,
            "sha256": file_sha256(args.normalization),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "compute_capability": [
                properties.major,
                properties.minor,
            ],
            "total_device_memory_bytes": properties.total_memory,
        },
        "workload": {
            "synthetic": True,
            "velocity_shape": [
                config.velocity_height,
                config.velocity_width,
            ],
            "output_steps": config.output_steps,
            "receivers": config.num_receivers,
            "batch_sizes": args.batch_sizes,
            "frequency_hz": args.frequency_hz,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "seed": args.seed,
        },
        "feno": feno,
        "deepwave": deepwave,
        "comparisons": comparisons,
        "limitations": [
            "This benchmark compares runtime efficiency, not numerical accuracy.",
            "The checkpoint stopped after epoch 0 and is not an accuracy release.",
            "Deepwave grid spacing and time sampling define a synthetic comparison workload.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
