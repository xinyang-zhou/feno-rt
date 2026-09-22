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

当前 Graph A/B 使用 `configs/graph_ab.json` 中的 `steady_all_hit` 确定性定义。请求
trace 的落盘格式将在正式 runner 写入时与结果一起保存并计算 SHA256。
