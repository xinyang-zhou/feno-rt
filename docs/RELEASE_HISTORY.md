# Release history and repository scope

FENO-RT 的版本演进分为完整 serving 系统和聚焦运行时核心两条清晰边界。
历史版本继续由 Git tag 保留；当前 `main` 不重复携带已经移出的外围服务代码。

## Current main

当前 `main` 使用开发版本 `0.9.0.dev0`，聚焦单设备推理的核心执行链路：

- runner 私有的 medium-static、geometry-prefix 和 wavelet 三级 Tensor Cache；
- 按 tensor payload bytes 管理容量的 LRU 和 lease 生命周期保护；
- medium 到 geometry-prefix 的依赖感知级联失效；
- 带多重资源预算、deadline 和 starvation 保护的动态 batching；
- FCFS 与 cache-aware 调度策略；
- 单执行线程的异步 engine、取消、超时和请求错误隔离；
- 按 batch bucket 捕获并复用 decoder tail 的 CUDA Graph；
- CPU 单元测试和 CUDA Graph GPU 集成测试。

当前源码不提供多 GPU 编排、HTTP API、Prometheus、trace replay、进程 worker、
GPU↔pinned CPU medium tier、模型权重或训练代码。

## v0.8.0

[`v0.8.0`](https://github.com/xinyang-zhou/feno-rt/tree/v0.8.0) 是历史完整
serving release。除推理核心外，该 tag 还包含：

- per-GPU replica、cache-aware router 和独立 worker process；
- GPU↔pinned CPU medium tier、promotion/demotion 和 execution-window lease；
- HTTP serving、Prometheus metrics、bounded trace 和 offline replay；
- precision policy、`torch.compile` 和服务侧 profiling 控制；
- synthetic workload、release manifest 和单 GPU benchmark 工具。

这些能力仍可通过固定 tag 查看和复现，但不应被理解为当前 `main` 的默认能力。

## Cache terminology change

`v0.8.0` 将 medium、decoder-static、geometry-prefix 和 frequency/wavelet 描述为
四级可复用上下文。当前 `main` 把 encoder latent 与 decoder static projected K/V
统一封装进 `MediumContext`，由一个 medium-static cache 管理，因此当前术语是三级
cache：

```text
medium-static
  -> geometry-prefix
wavelet (frequency-dependent and medium-independent)
```

medium 是 geometry-prefix 的依赖根；wavelet 不依赖 medium，因此 medium 失效时
不会清除 wavelet entry。

## Version identification

性能结果、问题报告和复现实验应记录完整 Git commit，而不只记录包版本。开发版本
`0.9.0.dev0` 表示当前源码位于 `v0.8.0` 之后且尚未对应新的稳定 release tag。
