#!/usr/bin/env python3
"""Run the reproducible CPU-only Stage 6 synthetic trace benchmark."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import tempfile
from time import perf_counter
from typing import Any, Dict

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from feno_rt.config import FENOModelConfig
from feno_rt.models.feno_freq import FENOFreq
from feno_rt.observability import RuntimeProfiler, ServiceMetrics, render_prometheus
from feno_rt.runtime import (
    AsyncFENOEngine,
    DynamicBatchConfig,
    FENOModelRunner,
    load_trace,
    replay_trace,
)


DEMO_DIR = ROOT / "data" / "demo"
DEFAULT_OUTPUT = ROOT / "artifacts" / "benchmarks" / "stage6_synthetic_replay.json"


def make_config() -> FENOModelConfig:
    return FENOModelConfig(
        original_size=16,
        velocity_height=16,
        velocity_width=16,
        latent_height=4,
        latent_width=4,
        output_steps=16,
        receiver_depth=1,
        num_receivers=8,
        encoder_dim=16,
        encoder_depth=1,
        encoder_heads=4,
        decoder_dim=16,
        decoder_depth=2,
        decoder_heads=4,
        fno_modes1=4,
        fno_modes2=4,
        fno_width=16,
        patch_size=2,
        position_embedding_dim=8,
        frequency_condition_dim=8,
        dropout_rate=0.0,
    )


def _profile_smoke_test() -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="feno-stage6-profile-") as directory:
        profiler = RuntimeProfiler(Path(directory), include_cuda=False)
        profiler.start(name="cpu-smoke")
        left = torch.arange(64, dtype=torch.float32).reshape(8, 8)
        torch.mm(left, left)
        result = profiler.stop()
        path = Path(directory) / result["file"]
        return {
            "exported": path.is_file() and path.stat().st_size > 0,
            "bytes": path.stat().st_size if path.is_file() else 0,
            "filename": result["file"],
        }


async def run_benchmark() -> Dict[str, Any]:
    torch.manual_seed(20260910)
    torch.set_num_threads(1)
    config = make_config()
    model = FENOFreq(config.encoder_config(), config.decoder_config()).eval()
    runner = FENOModelRunner(
        model,
        config=config,
        normalization=None,
        device="cpu",
        model_version="stage6-public-synthetic-v1",
    )
    velocity = np.load(DEMO_DIR / "velocity.npy", allow_pickle=False)
    trace_path = DEMO_DIR / "requests.jsonl"
    trace = load_trace(trace_path)
    context = runner.prepare_medium(velocity, medium_id="demo", already_normalized=True)
    engine = AsyncFENOEngine(
        runner,
        DynamicBatchConfig(max_wait_us=2_000, max_batch_size=8),
        double_buffer=True,
    )
    try:
        cold = await replay_trace(
            engine,
            {"demo": context},
            trace,
            wave_size=8,
            verify_checksums=False,
            request_prefix="stage6-cold",
        )
        hot = await replay_trace(
            engine,
            {"demo": context},
            trace,
            wave_size=8,
            verify_checksums=False,
            request_prefix="stage6-hot",
        )
        runtime = engine.stats()
    finally:
        await engine.close()

    checksum_scale = max(abs(float(cold["aggregate_checksum"])), 1.0)
    checksum_relative_error = abs(
        float(cold["aggregate_checksum"]) - float(hot["aggregate_checksum"])
    ) / checksum_scale
    cache_before_flush = runner.cache_metrics()["caches"]
    cleared = runner.clear_caches()
    cache_after_flush = runner.cache_metrics()["caches"]
    cache_empty = all(
        snapshot["resident_entries"] == 0
        for snapshot in cache_after_flush.values()
    )
    prometheus = render_prometheus(
        runtime,
        ServiceMetrics().snapshot(),
        registered_mediums=1,
    )
    profile = _profile_smoke_test()
    accepted = {
        "all_requests_succeeded": cold["status_counts"] == {"succeeded": 24}
        and hot["status_counts"] == {"succeeded": 24},
        "repeat_checksum_exact": checksum_relative_error == 0.0,
        "cache_flush_empty": cache_empty,
        "prometheus_rendered": "feno_http_requests_in_flight" in prometheus,
        "cpu_profile_exported": profile["exported"],
    }
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "stage6-public-synthetic-replay",
        "device": "cpu",
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "seed": 20260910,
        "trace": {
            "path": str(trace_path.relative_to(ROOT)),
            "records": len(trace),
            "requests": 24,
            "wave_size": 8,
        },
        "cold": cold,
        "hot": hot,
        "hot_speedup": cold["elapsed_s"] / max(hot["elapsed_s"], 1e-12),
        "repeat_checksum_relative_error": checksum_relative_error,
        "cache": {
            "before_flush": cache_before_flush,
            "cleared_entries": cleared,
            "after_flush": cache_after_flush,
        },
        "profiler": profile,
        "prometheus_bytes": len(prometheus.encode("utf-8")),
        "acceptance": accepted,
        "all_acceptance_passed": all(accepted.values()),
        "notes": [
            "CPU-only by design; this public smoke benchmark consumes no GPU.",
            "Timing is descriptive, not a production GPU performance claim.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    started = perf_counter()
    report = asyncio.run(run_benchmark())
    report["wall_time_s"] = perf_counter() - started
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["all_acceptance_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
