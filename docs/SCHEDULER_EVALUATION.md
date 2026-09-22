# FCFS vs Cache-Aware scheduler evaluation

本文定义 FENO-RT 调度实验的问题、控制变量和结果解释边界。正式性能数字只有在服务器
矩阵全部运行、正确性与 schema 校验通过并提交原始结果后才会写入本文和主 README。

`6351d53` 的两套 RTX 5090 矩阵已通过完整性和正确性检查，公开脱敏结果见
[调度实验](../benchmarks/results/scheduler_ab/20260922_rtx5090_6351d53/README.md)。主矩阵
吞吐比为 3.481–4.228x；batch=1 对照为 0.616–0.687x，必须同时报告适用范围和退化。

## Question

实验回答：在相同模型、请求 trace、资源预算和 SLO 下，Cache-Aware 调度能否通过改善
batch compatibility、批内 geometry/wavelet 共享或 cache locality，提高吞吐且不以不可
接受的 queue p99、deadline violation 或 starvation 为代价。

CUDA Graph 在全部 run 中关闭，避免把 Graph replay 收益混入调度收益。FCFS 是严格到达
顺序基线，遇到不兼容的队首请求时不会跨过它组批；Cache-Aware 可以在 pending queue
中选择兼容请求，同时保留 deadline guard 和 starvation guard。

## Controlled workload

[`scheduler_ab.json`](../benchmarks/configs/scheduler_ab.json) 固定 1000 个 burst 请求、
4 个 medium、batch 上限 8、同一资源预算和 30 秒请求 timeout。medium context 在计时前
预热，geometry-prefix 与 wavelet cache 在计时前清空。四个确定性 trace 只改变复用分布：

| Scenario | Purpose |
|---|---|
| `no_reuse` | geometry 和 wavelet 几乎不复用，用于观察 Cache-Aware 的退化边界 |
| `uniform_reuse` | 在有限池中均匀抽样 |
| `long_tail_reuse` | Zipf 长尾访问 |
| `hotspot_reuse` | 80% 请求集中到一个热点 |

每个策略对读取同一份 checked-in JSON trace。策略名称不写入 trace；trace 的 SHA256 会
写入每个结果。每种策略与场景运行 3 个全新进程，执行顺序按 repeat 反转，共 24 个 run。

## Metrics and interpretation

`14678d6` 使用 5 秒 timeout，原配置保存在
[`scheduler_ab_5s_legacy.json`](../benchmarks/configs/scheduler_ab_5s_legacy.json)。
当前协议采用 30 秒完整完成窗口，适用于两种策略和所有场景。30 秒结果不能用来证明
原 5 秒 SLO；复现旧协议需使用旧 commit 的 trace，不能混用当前 30 秒 trace。
新增 `scheduler_batch1_ab.json` 固定两策略 batch 上限为 1，帮助分析组批收益与调度开销。
主矩阵和该对照分别运行、分别汇总；不合并样本。任一矩阵不完整或有失败时不生成性能对比。

trace 的相邻请求使用不同 medium，严格 FCFS 因兼容性限制只能形成 batch=1；Cache-Aware
可以跨过不兼容项组批。因此主矩阵的全部收益不能归因于缓存命中。`no_reuse` 消除的是
geometry/wavelet 复用，medium 仍有复用。batch=1 对照消除 batch size 差异，但仍受
50 ms starvation guard 影响，不能视为纯缓存收益的精确分解。

`starvation_guard_requests` 统计触发等待阈值的 dispatch 请求数；50 ms 不是最大 queue
latency 保证。正确性字段分别报告探针缺失、非有限值和数值不匹配；缺失探针会使运行
失败，但不表示已返回的输出含有 NaN/Inf。

每个 run 保存原始 batch size 以及逐请求 queue、execution 和 end-to-end latency。
正式汇总报告：

- throughput 和 end-to-end p50/p95/p99；
- queue p50/p95/p99，用于识别为等待 locality 支付的代价；
- mean batch size、fill ratio、批内 geometry/wavelet 共享比例；
- medium、geometry-prefix、wavelet 的 hit/miss/eviction 与运行时去重指标；
- deadline violation、starvation guard 次数和最长 dispatch wait；
- scheduler selection 的总时间、均值、最大值和每请求 CPU 开销；
- baseline/peak allocated、peak reserved memory；
- 对 8 个均匀抽取请求执行 uncached reference 正确性检查。

最终结论必须区分两个来源：更大的有效 batch，以及同一 batch 内或跨 batch 的缓存复用。
如果吞吐提高但 queue/end-to-end p99 或 SLO 明显恶化，不能只报告吞吐。`no_reuse` 是必须
公开的反例；若 Cache-Aware 在该场景有额外调度开销或没有收益，也应保留。

## Reproduction

先安装 lock 中的依赖，并确认工作树干净、目标 GPU 独占。模型文件通过配置指定的环境
变量解析，但路径本身不提交到仓库：

```bash
export FENO_CHECKPOINT=/path/to/feno_test.pth
export FENO_NORMALIZATION=/path/to/norm_params_freq.npz
export CUDA_VISIBLE_DEVICES=GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
python benchmarks/run_scheduler_ab_matrix.py \
  --output-dir /tmp/feno-scheduler-ab/<session-name>
```

该目录必须在仓库外。矩阵完成后先检查 `summary.md`、`session.json` 和 24 个 run 的状态，
保留全部成功与失败记录。原始结果可能包含机器路径和设备标识；发布前按
[复现步骤](BENCHMARK_REPRODUCTION.md) 准备公开副本并检查元数据。

## Scope

主矩阵刻意使用固定 request shape、单一 burst 到达模式和统一 timeout，以隔离复用程度与
调度策略。它不代表所有负载强度或 SLO。deadline、starvation 和三类 admission budget
已有行为测试；不同到达率、混合 request shape 和多档 SLO 属于后续压力实验，不能从本
矩阵外推结论。
