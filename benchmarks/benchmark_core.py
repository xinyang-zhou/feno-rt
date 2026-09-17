"""Small reproducible benchmark for the focused runtime features."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Callable, Dict, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from feno_rt.config import FENOModelConfig
from feno_rt.models import FENOFreq
from feno_rt.runtime import AsyncFENOEngine, DynamicBatchConfig, FENOModelRunner


def make_config() -> FENOModelConfig:
    return FENOModelConfig(
        original_size=16,
        velocity_height=16,
        velocity_width=16,
        latent_height=4,
        latent_width=4,
        output_steps=32,
        receiver_depth=1,
        num_receivers=16,
        encoder_dim=32,
        encoder_depth=2,
        encoder_heads=4,
        decoder_dim=32,
        decoder_depth=2,
        decoder_heads=4,
        fno_modes1=4,
        fno_modes2=4,
        fno_width=32,
        patch_size=2,
        position_embedding_dim=8,
        frequency_condition_dim=16,
        dropout_rate=0.0,
    )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(
    function: Callable[[], torch.Tensor],
    *,
    device: torch.device,
    repeats: int,
) -> Tuple[Dict[str, float], torch.Tensor]:
    samples = []
    output = function()
    synchronize(device)
    for _ in range(repeats):
        started = perf_counter()
        output = function()
        synchronize(device)
        samples.append((perf_counter() - started) * 1_000.0)
    return {
        "mean_ms": sum(samples) / len(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }, output


async def benchmark_batching(
    runner: FENOModelRunner,
    context: Any,
    *,
    requests: int,
) -> Dict[str, Any]:
    config = DynamicBatchConfig(
        max_wait_us=2_000,
        max_batch_size=8,
        max_query_tokens=8 * runner.config.num_receivers,
        max_activation_bytes=256 * 1024 * 1024,
        max_output_bytes=64 * 1024 * 1024,
    )
    started = perf_counter()
    async with AsyncFENOEngine(runner, config, double_buffer=True) as engine:
        handles = await asyncio.gather(
            *(
                engine.submit(
                    context,
                    [float(index % 12 + 1), float((index * 3) % 12 + 1)],
                    float((10, 15, 20, 25)[index % 4]),
                )
                for index in range(requests)
            )
        )
        outputs = await asyncio.gather(*handles)
        stats = engine.stats()
    synchronize(runner.device)
    elapsed = perf_counter() - started
    return {
        "requests": requests,
        "elapsed_ms": elapsed * 1_000.0,
        "throughput_requests_per_second": requests / elapsed,
        "output_checksum": sum(float(item.double().sum()) for item in outputs),
        "batches": stats["batches"],
        "max_observed_batch_size": stats["max_observed_batch_size"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--cuda-graphs", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1 or args.requests < 1:
        parser.error("repeats and requests must be positive")
    if args.cuda_graphs and torch.device(args.device).type != "cuda":
        parser.error("--cuda-graphs requires a CUDA device")
    return args


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    torch.manual_seed(7)
    config = make_config()
    runner = FENOModelRunner(
        FENOFreq(config.encoder_config(), config.decoder_config()),
        config=config,
        device=device,
    )
    velocity = torch.linspace(-1.0, 1.0, 16 * 16).reshape(16, 16)
    sources = torch.tensor([[2.0, 3.0], [8.0, 10.0], [2.0, 3.0], [6.0, 4.0]])
    frequencies = torch.tensor([10.0, 25.0, 10.0, 15.0])
    context = runner.prepare_medium(velocity, already_normalized=True)

    uncached, reference = measure(
        lambda: runner.forward_uncached(
            velocity,
            sources,
            frequencies,
            already_normalized=True,
        ),
        device=device,
        repeats=args.repeats,
    )
    runner.forward_batch(context, sources, frequencies, cache_level="all")
    cached, cached_output = measure(
        lambda: runner.forward_batch(context, sources, frequencies, cache_level="all"),
        device=device,
        repeats=args.repeats,
    )
    relative_error = float(
        torch.linalg.vector_norm((cached_output - reference).float())
        / torch.linalg.vector_norm(reference.float()).clamp_min(1e-12)
    )

    result: Dict[str, Any] = {
        "device": str(device),
        "uncached": uncached,
        "three_level_cache": cached,
        "cache_speedup": uncached["mean_ms"] / cached["mean_ms"],
        "cache_relative_l2_error": relative_error,
        "cache_metrics": runner.cache_metrics(),
        "dynamic_batching": asyncio.run(
            benchmark_batching(runner, context, requests=args.requests)
        ),
    }

    if args.cuda_graphs:
        runner.enable_cuda_graphs(buckets=(1, 2, 4, 8), fallback_on_error=False)
        graph_timing, graph_output = measure(
            lambda: runner.forward_batch(context, sources, frequencies, cache_level="all"),
            device=device,
            repeats=args.repeats,
        )
        result["cuda_graph"] = {
            **graph_timing,
            "relative_l2_error": float(
                torch.linalg.vector_norm((graph_output - cached_output).float())
                / torch.linalg.vector_norm(cached_output.float()).clamp_min(1e-12)
            ),
            "metrics": runner.execution_config()["cuda_graph"],
        }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
