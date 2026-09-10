"""Run the optional FENO-RT HTTP API without elevated privileges."""

import argparse
import os
import sys
from pathlib import Path

import torch
from aiohttp import web

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from feno_rt.config import DEFAULT_INFERENCE_CONFIG
from feno_rt.observability import RuntimeProfiler
from feno_rt.paths import ARTIFACTS_DIR
from feno_rt.preprocessing import NormalizationStats
from feno_rt.runtime import (
    DynamicBatchConfig,
    MediumTierConfig,
    MultiGPUFENOEngine,
    RequestTraceRecorder,
)
from feno_rt.serving import FENOHTTPService


PHYSICAL_TWO_GPU_PAIR = ("cuda:1", "cuda:2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--precision",
        choices=("fp32", "tf32", "bf16", "fp16"),
        default="fp32",
    )
    parser.add_argument("--sdpa-backend", default="auto")
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--max-wait-us", type=int, default=2_000)
    parser.add_argument("--hot-medium-threshold", type=int, default=8)
    parser.add_argument("--gpu-medium-cache-mib", type=int, default=64)
    parser.add_argument("--pinned-medium-cache-mib", type=int, default=128)
    parser.add_argument("--max-request-mib", type=int, default=64)
    parser.add_argument(
        "--trace-dir", type=Path, default=ARTIFACTS_DIR / "traces"
    )
    parser.add_argument("--trace-max-mib", type=int, default=128)
    parser.add_argument("--trace-max-records", type=int, default=100_000)
    parser.add_argument(
        "--profile-dir", type=Path, default=ARTIFACTS_DIR / "profiles"
    )
    parser.add_argument("--cuda-graphs", action="store_true")
    args = parser.parse_args()
    if len(args.devices) > 2:
        parser.error("this Stage 5 launcher never starts more than two GPU workers")
    if len(args.devices) == 2:
        if tuple(args.devices) != PHYSICAL_TWO_GPU_PAIR:
            parser.error("two-GPU serving is restricted to physical cuda:1 and cuda:2")
        if os.environ.get("CUDA_VISIBLE_DEVICES"):
            parser.error(
                "unset CUDA_VISIBLE_DEVICES so cuda:1/cuda:2 remain physical GPU 1/2"
            )
    if not args.devices:
        parser.error("at least one device is required")
    if not 1 <= args.port <= 65_535:
        parser.error("port must be between 1 and 65535")
    if args.max_batch_size < 1 or args.max_wait_us < 0:
        parser.error("invalid dynamic batching limits")
    if args.max_request_mib < 1:
        parser.error("max-request-mib must be positive")
    if args.gpu_medium_cache_mib < 1 or args.pinned_medium_cache_mib < 0:
        parser.error("invalid medium tier capacities")
    if args.trace_max_mib < 1 or args.trace_max_records < 1:
        parser.error("trace capacity limits must be positive")
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GPU serving")
    for value in args.devices:
        device = torch.device(value)
        if device.type != "cuda" or device.index is None:
            raise ValueError(f"an explicit CUDA index is required: {value}")
        if device.index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device is not available: {value}")

    normalization = NormalizationStats.load(args.normalization)
    batch_config = DynamicBatchConfig(
        max_wait_us=args.max_wait_us,
        max_batch_size=args.max_batch_size,
        max_query_tokens=args.max_batch_size * DEFAULT_INFERENCE_CONFIG.num_receivers,
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
        precision=args.precision,
        sdpa_backend=args.sdpa_backend,
        routing_policy="cache_aware",
        hot_medium_threshold=args.hot_medium_threshold,
        async_d2h=True,
        enable_cuda_graphs=args.cuda_graphs,
        return_cpu=True,
    )
    service = FENOHTTPService(
        engine,
        close_engine=True,
        max_request_bytes=args.max_request_mib * 1024 * 1024,
        trace_recorder=RequestTraceRecorder(
            args.trace_dir,
            max_bytes=args.trace_max_mib * 1024 * 1024,
            max_records=args.trace_max_records,
        ),
        profiler=RuntimeProfiler(args.profile_dir, include_cuda=True),
    )
    web.run_app(
        service.create_app(),
        host=args.host,
        port=args.port,
        print=lambda message: print(message, flush=True),
    )


if __name__ == "__main__":
    main()
