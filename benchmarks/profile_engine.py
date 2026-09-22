"""Small diagnostic trace of the complete engine; never a formal benchmark."""

import argparse
import asyncio
from dataclasses import asdict
from functools import partial
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.benchmark_graph_ab import file_sha256, git_identity, write_json_atomic
from benchmarks.benchmark_scheduler_ab import (
    build_velocity_family, compare_correctness_probes, evenly_spaced_probe_indices,
    execute_trace, warmup_runner,
)
from benchmarks.scheduler_workload import DEFAULT_CONFIG, build_manifest, load_config


def small_config():
    from feno_rt.config import FENOModelConfig
    return FENOModelConfig(
        original_size=16, velocity_height=16, velocity_width=16,
        latent_height=4, latent_width=4, output_steps=16, receiver_depth=1,
        num_receivers=8, encoder_dim=16, encoder_depth=1, encoder_heads=4,
        decoder_dim=16, decoder_depth=2, decoder_heads=4,
        fno_modes1=4, fno_modes2=4, fno_width=16, patch_size=2,
        position_embedding_dim=8, frequency_condition_dim=8, dropout_rate=0.0,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--toy", action="store_true", help="small random model for functional checks")
    parser.add_argument("--policy", choices=["fcfs", "cache_aware"], default="cache_aware")
    parser.add_argument("--scenario", default="hotspot_reuse")
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--arrival-interval-us", type=int, default=0)
    parser.add_argument("--single-buffer", action="store_true")
    parser.add_argument("--cuda-graphs", action="store_true")
    parser.add_argument("--all-hit", action="store_true")
    parser.add_argument("--nvtx", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    import numpy as np
    import torch
    from feno_rt.config import FENOModelConfig
    from feno_rt.models import FENOFreq
    from feno_rt.preprocessing import NormalizationStats
    from feno_rt.runtime import AsyncFENOEngine, DynamicBatchConfig, FENOModelRunner
    from feno_rt.runtime.diagnostics import EngineTrace

    config = load_config(args.config)
    if not 8 <= args.requests <= config["workloads"]["request_count"]:
        raise ValueError("requests must be between 8 and the configured trace length")
    if args.arrival_interval_us < 0:
        raise ValueError("arrival interval must be non-negative")
    if args.output_dir.exists():
        raise ValueError("use a new diagnostic output directory")
    device = torch.device(args.device)
    if (args.cuda_graphs or args.nvtx) and device.type != "cuda":
        raise ValueError("CUDA Graph/NVTX require a CUDA device")
    torch.set_num_threads(1)
    torch.manual_seed(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    commit, dirty = git_identity()
    source = dict(git_commit=commit, git_dirty=dirty)
    manifest = build_manifest(config, args.scenario)
    manifest["requests"] = manifest["requests"][:args.requests]
    for index, request in enumerate(manifest["requests"]):
        request["arrival_offset_us"] = index * args.arrival_interval_us
        if args.toy:
            request["source_position"] = [float(index % 12), float((index * 3) % 12)]
    checkpoint = None
    normalization_identity = None
    if args.toy:
        model_config = small_config()
        normalization = NormalizationStats(v_mean=2000, v_std=500, seis_mean=0, seis_std=1)
        runner = FENOModelRunner(
            FENOFreq(model_config.encoder_config(), model_config.decoder_config()),
            config=model_config, normalization=normalization, device=device,
        )
    else:
        checkpoint_path = Path(os.environ[config["model"]["checkpoint_env"]])
        normalization_path = Path(os.environ[config["model"]["normalization_env"]])
        checkpoint = dict(identifier=checkpoint_path.name, sha256=file_sha256(checkpoint_path))
        normalization_identity = dict(identifier=normalization_path.name, sha256=file_sha256(normalization_path))
        model_config = FENOModelConfig()
        normalization = NormalizationStats.load(normalization_path)
        runner = FENOModelRunner.from_checkpoint(
            checkpoint_path, config=model_config, normalization=normalization, device=device,
        )
    contexts, velocities = {}, {}
    for medium in manifest["mediums"]:
        key = medium["medium_id"]
        velocity = build_velocity_family(np, model_config, normalization,
                                         manifest["velocity"], medium["family_index"])
        velocities[key] = velocity
        contexts[key] = runner.prepare_medium(velocity, medium_id=key)
    scheduler = {k: v for k, v in config["scheduler"].items()
                 if k not in ("policies", "double_buffer")}
    if args.cuda_graphs:
        runner.enable_cuda_graphs(buckets=(1, 2, 4, 8), fallback_on_error=False)
        first = manifest["requests"][0]
        for bucket in (1, 2, 4, 8):
            runner.forward_batch(contexts[first["medium_id"]],
                                 [first["source_position"]] * bucket,
                                 [first["frequency_hz"]] * bucket)
    warmup_runner(runner, contexts, manifest["requests"],
                  max_batch_size=scheduler["max_batch_size"])
    if not args.all_hit:
        runner.caches.geometry.clear()
        runner.caches.wavelet.clear()
    runner.reset_cache_stats()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    trace = EngineTrace(nvtx=args.nvtx)
    runner.trace = trace
    probes = evenly_spaced_probe_indices(args.requests, 8)
    with trace.span("feno.engine_window"):
        stats, outputs, errors, start, end = asyncio.run(execute_trace(
            engine_class=partial(AsyncFENOEngine, trace=trace), runner=runner,
            engine_config=DynamicBatchConfig(policy=args.policy, **scheduler),
            contexts=contexts, requests=manifest["requests"],
            double_buffer=not args.single_buffer, probe_indices=probes,
        ))
    runner.trace = None
    graph_metrics = runner.execution_config()
    correctness = compare_correctness_probes(
        torch, runner, velocities, manifest["requests"], outputs, probes,
        rtol=config["correctness"]["rtol"], atol=config["correctness"]["atol"],
    )
    args.output_dir.mkdir(parents=True)
    trace.write(args.output_dir / "trace.json")
    write_json_atomic(args.output_dir / "requests.json", manifest["requests"])
    result = dict(
        result_class="diagnostic", benchmark="engine_profile", source=source,
        command=[sys.executable, *sys.argv], config_sha256=file_sha256(args.config),
        model=dict(random_initialized=args.toy, seed=config["seed"], config=asdict(model_config),
                   checkpoint=checkpoint, normalization=normalization_identity),
        environment=dict(python=platform.python_version(), pytorch=torch.__version__,
                         cuda=torch.version.cuda, device=str(device),
                         gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None),
        workload=dict(scenario=args.scenario, request_count=args.requests,
                      arrival_interval_us=args.arrival_interval_us,
                      medium_cache="warm", geometry_wavelet_cache="warm" if args.all_hit else "cold",
                      requests_sha256=file_sha256(args.output_dir / "requests.json")),
        execution=dict(policy=args.policy, double_buffer=not args.single_buffer,
                       cuda_graphs=args.cuda_graphs, nvtx=args.nvtx),
        measured_window_ms=(end-start)/1e6, stats=stats, graphs=graph_metrics,
        correctness=correctness, errors=errors, trace_metadata=trace.snapshot()["metadata"],
    )
    write_json_atomic(args.output_dir / "diagnostic.json", result)
    print(f"saved diagnostic artifacts to {args.output_dir}")
    health = result["trace_metadata"]
    return 0 if (not errors and correctness["passed"] and
                 not health["dropped_events"] and not health["active_requests"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
