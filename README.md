# FENO-RT

一个面向非自回归科学神经算子的轻量推理运行时。项目只聚焦四项能力：

- 三级计算缓存：medium-static、geometry-prefix、wavelet；
- medium 失效时对依赖缓存执行级联失效；
- 支持容量预算、deadline 和饥饿保护的动态 batching；
- 按 batch bucket 捕获和复用在线尾部的 CUDA Graph。

仓库不包含训练代码、多 GPU 调度、HTTP 服务、可观测性平台或模型权重。

## Project scope and versions

当前 `main`（开发版本 `0.9.0.dev0`）是聚焦单设备执行链路的运行时核心。
它保留缓存、调度、异步组批和 CUDA Graph 等关键机制，外围服务能力不属于
当前源码范围。

历史 [`v0.8.0`](https://github.com/xinyang-zhou/feno-rt/tree/v0.8.0)
是完整 serving 版本，包含多 GPU worker、HTTP、Prometheus、trace replay 和
GPU↔pinned CPU medium tier。两个版本的能力边界与代码演进见
[docs/RELEASE_HISTORY.md](docs/RELEASE_HISTORY.md)。

## Architecture

```text
velocity
  -> medium-static cache (encoder latent + decoder K/V)
     -> geometry prefix cache
        -> frequency/wavelet cache
           -> eager or CUDA Graph online tail

concurrent requests
  -> AsyncRequestQueue
     -> FCFS / cache-aware scheduler
        -> dynamic batch
           -> FENOModelRunner
```

缓存由单个 model runner 私有持有。medium key 只包含规范化速度模型摘要，
geometry key 组合 medium 与震源/接收器几何，wavelet key 使用精确频率 bit。
删除一个 medium 时会同时失效对应的 geometry-prefix；wavelet 与 medium
无关，因此会保留。

详细设计见 [docs/architecture.md](docs/architecture.md)。

## Install

Python 3.9+：

```bash
python -m pip install -r requirements/runtime.lock
python -m pip install -e . --no-deps
```

CUDA Graph 需要 CUDA 设备，其余功能可以在 CPU 上运行。

## Minimal usage

```python
import asyncio
import torch

from feno_rt.config import FENOModelConfig
from feno_rt.models import FENOFreq
from feno_rt.runtime import AsyncFENOEngine, DynamicBatchConfig, FENOModelRunner

config = FENOModelConfig(
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
model = FENOFreq(config.encoder_config(), config.decoder_config())
runner = FENOModelRunner(model, config=config, device="cpu")
velocity = torch.linspace(-1.0, 1.0, 16 * 16).reshape(16, 16)
medium = runner.prepare_medium(velocity, already_normalized=True)


async def main():
    batch = DynamicBatchConfig(max_batch_size=8, max_query_tokens=64)
    async with AsyncFENOEngine(runner, batch) as engine:
        handles = await asyncio.gather(
            engine.submit(medium, [2.0, 3.0], 10.0),
            engine.submit(medium, [8.0, 10.0], 25.0),
        )
        outputs = await asyncio.gather(*handles)
        print(outputs[0].shape, engine.stats())


asyncio.run(main())
```

在 CUDA 上启用图捕获：

```python
runner = FENOModelRunner(model, config=config, device="cuda:0")
runner.enable_cuda_graphs(buckets=(1, 2, 4, 8))
```

## Tests and benchmark

```bash
python -m unittest discover -s tests -v
python benchmarks/benchmark_core.py --device cpu
python benchmarks/benchmark_core.py --device cuda:0 --cuda-graphs
```

CPU 环境会自动跳过 CUDA Graph 集成测试。

`benchmark_core.py` 用于功能和数值正确性 smoke test，不构成正式性能结论。
正式实验的环境记录、计时边界、重复方式和结果准入规则见
[docs/PERFORMANCE.md](docs/PERFORMANCE.md)。
CUDA Graph A/B 的单进程正式运行入口和服务器命令见
[benchmarks/README.md](benchmarks/README.md)。

## CUDA Graph performance

在 NVIDIA GeForce RTX 5090 上，使用同一模型、同一 workload 和 all-hit cache，
Graph off/on 各运行 3 个独立进程，每个配置测量 1000 个请求。表中 latency 是一次
batch invocation 的同步 wall-clock 时间，数值为 3 次 run-level 指标的中位数。

| Batch | Graph off mean | Graph on mean | Mean latency reduction | Throughput speedup | Peak allocated delta |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.152 ms | 0.682 ms | 41.11% | 1.688x | +11.02 MiB |
| 2 | 1.204 ms | 0.721 ms | 40.07% | 1.660x | +14.03 MiB |
| 4 | 1.293 ms | 0.909 ms | 29.97% | 1.423x | +20.65 MiB |
| 8 | 1.486 ms | 1.200 ms | 19.48% | 1.239x | +33.17 MiB |

CUDA Graph 对小 batch 收益最大，因为固定的 host/framework dispatch 与 kernel launch
开销在小 batch 总耗时中占比更高。该结论不表示单个 GPU kernel 的计算变快。

完整的 p50/p95/p99、capture 成本、正确性、环境信息、24 个原始 run 和日志见
[正式结果](benchmarks/results/cuda_graph_ab/20260922_rtx5090_a69f11e/summary.md)。

## Repository layout

```text
feno_rt/models/             forward-only model definition
feno_rt/runtime/
  context_cache.py          three byte-bounded LRU caches and cascade invalidation
  model_runner.py           staged cached execution
  request.py                request lifecycle and cost model
  scheduler.py              FCFS and cache-aware batching
  engine.py                 asynchronous batching engine
  cuda_graph.py             bucketed CUDA Graph capture/replay
benchmarks/
  benchmark_core.py          functional smoke benchmark
  benchmark_graph_ab.py      one-process formal CUDA Graph A/B runner
  graph_workload.py          deterministic workload manifest generator
  run_graph_ab_matrix.py     isolated-process formal matrix orchestrator
  summarize_graph_ab.py      run-level A/B summary generator
  configs/                   versioned experiment configurations
  schema/                    machine-readable result contracts
  workloads/                 deterministic workload definitions
  results/                   raw formal results and summaries
tests/
docs/PERFORMANCE.md
docs/RELEASE_HISTORY.md
```

## License

MIT
