# Workloads

Formal benchmark 的 workload 必须能够由固定 trace 或确定性生成配置重建。

每个 workload 需要记录：

- workload 名称和 schema 版本；
- 随机种子；
- 请求数量和到达模式；
- source、receiver 和 frequency 的生成方式；
- medium、geometry 和 frequency 的复用分布；
- 每层 cache 在计时开始前的状态；
- trace 文件或生成配置的 SHA256。

生成 workload 后不得在 Graph on/off 或不同调度策略之间修改请求内容和顺序。若更新
生成算法或字段含义，必须提升 workload schema 版本，不能覆盖旧结果使用的定义。

当前 Graph A/B 使用 `configs/graph_ab.json` 中的 `steady_all_hit` 确定性定义。配置
显式保存 velocity 生成公式、八个 source position、frequency 序列和默认 receiver
规则。每个 batch 使用 source 列表的前 N 项，并重复同一个固定 batch；因此 on/off
模式的请求内容、顺序和复用关系完全一致。

生成并检查 tracked workload manifests：

```bash
python benchmarks/graph_workload.py --write
python benchmarks/graph_workload.py --check
```

Scheduler A/B 使用 4 个确定性请求 trace：`no_reuse`、`uniform_reuse`、
`long_tail_reuse` 和 `hotspot_reuse`。FCFS 与 Cache-Aware 必须读取同一个对应文件，
策略本身不编码在 trace 中：

```bash
python benchmarks/scheduler_workload.py --write
python benchmarks/scheduler_workload.py --check
```

这些 trace 固定请求顺序、到达时间、timeout、medium、source、frequency 和 priority。
配置或生成算法变化后，必须重新生成 manifest 并使检查通过。
