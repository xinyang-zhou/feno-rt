"""Replay one recorded FENO trace on at most two explicitly selected GPUs."""

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from feno_rt.config import DEFAULT_INFERENCE_CONFIG
from feno_rt.paths import ARTIFACTS_DIR
from feno_rt.preprocessing import NormalizationStats
from feno_rt.runtime import (
    DynamicBatchConfig,
    MediumTierConfig,
    MultiGPUFENOEngine,
    load_trace,
    replay_trace,
)


PHYSICAL_TWO_GPU_PAIR = ("cuda:1", "cuda:2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--velocity", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="path to a compatible inference-only state dict (not distributed here)",
    )
    parser.add_argument(
        "--normalization",
        type=Path,
        required=True,
        help="path to matching inference normalization statistics",
    )
    parser.add_argument("--devices", nargs="+", default=list(PHYSICAL_TWO_GPU_PAIR))
    parser.add_argument("--wave-size", type=int, default=32)
    parser.add_argument("--already-normalized", action="store_true")
    parser.add_argument("--gpu-medium-cache-mib", type=int, default=64)
    parser.add_argument("--pinned-medium-cache-mib", type=int, default=128)
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACTS_DIR / "replays" / "latest.json",
    )
    args = parser.parse_args()
    if not 1 <= len(args.devices) <= 2:
        parser.error("trace replay supports one or two GPU workers, never four")
    if len(args.devices) == 2:
        if tuple(args.devices) != PHYSICAL_TWO_GPU_PAIR:
            parser.error("two-GPU replay is restricted to physical cuda:1 and cuda:2")
        if os.environ.get("CUDA_VISIBLE_DEVICES"):
            parser.error(
                "unset CUDA_VISIBLE_DEVICES so cuda:1/cuda:2 remain physical GPU 1/2"
            )
    if args.wave_size < 1:
        parser.error("wave-size must be positive")
    if args.gpu_medium_cache_mib < 1 or args.pinned_medium_cache_mib < 0:
        parser.error("invalid medium tier capacities")
    return args


async def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for production trace replay")
    for value in args.devices:
        device = torch.device(value)
        if device.type != "cuda" or device.index is None:
            raise ValueError(f"an explicit CUDA index is required: {value}")
        if device.index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device is not available: {value}")
    records = load_trace(args.trace)
    context_ids = sorted(
        {
            str(record["request"]["context_id"])
            for record in records
            if record.get("kind") == "request"
        }
    )
    if len(context_ids) != 1:
        raise ValueError(
            "the CLI accepts exactly one recorded context; replay multi-medium traces "
            "through the Python API with an explicit context mapping"
        )
    velocity = np.load(args.velocity, allow_pickle=False)
    if not isinstance(velocity, np.ndarray) or velocity.ndim != 2:
        raise ValueError("velocity must be one rank-two .npy array")
    normalization = NormalizationStats.load(args.normalization)
    batch_config = DynamicBatchConfig(
        max_wait_us=2_000,
        max_batch_size=32,
        max_query_tokens=32 * DEFAULT_INFERENCE_CONFIG.num_receivers,
        max_activation_bytes=2 * 1024 * 1024 * 1024,
        max_output_bytes=512 * 1024 * 1024,
    )
    engine = MultiGPUFENOEngine.from_checkpoint(
        args.checkpoint,
        devices=args.devices,
        config=DEFAULT_INFERENCE_CONFIG,
        normalization=normalization,
        batch_config=batch_config,
        medium_tier_config=MediumTierConfig(
            gpu_capacity_bytes=args.gpu_medium_cache_mib * 1024 * 1024,
            pinned_cpu_capacity_bytes=args.pinned_medium_cache_mib * 1024 * 1024,
        ),
        routing_policy="cache_aware",
        hot_medium_threshold=8,
        async_d2h=True,
        return_cpu=True,
    )
    try:
        context = await engine.prepare_medium(
            velocity,
            medium_id=context_ids[0],
            already_normalized=args.already_normalized,
            replicas=len(args.devices),
        )
        result = await replay_trace(
            engine,
            {context_ids[0]: context},
            records,
            wave_size=args.wave_size,
        )
        runtime = engine.stats()
    finally:
        await engine.close()
    return {
        "schema_version": 1,
        "stage": 6,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "trace": str(args.trace),
        "devices": list(args.devices),
        "four_gpu_tests": "skipped by design",
        "result": result,
        "runtime": runtime,
        "acceptance": {
            "all_requests_succeeded": result["status_counts"].get("succeeded", 0)
            == result["trace_requests"],
            "checksums_match": result["max_relative_checksum_error"] <= 1e-6,
        },
    }


def main() -> None:
    args = parse_args()
    artifact = asyncio.run(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(artifact["result"], indent=2))
    print(json.dumps(artifact["acceptance"], indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
