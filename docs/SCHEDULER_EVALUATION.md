# FCFS vs Cache-Aware scheduler evaluation

本文定义 FENO-RT 调度实验的问题、控制变量和结果解释边界。正式性能数字只有在服务器
矩阵全部运行、正确性与 schema 校验通过并提交原始结果后才会写入本文和主 README。

## Question

实验回答：在相同模型、请求 trace、资源预算和 SLO 下，Cache-Aware 调度能否通过改善
batch compatibility、批内 geometry/wavelet 共享或 cache locality，提高吞吐且不以不可
接受的 queue p99、deadline violation 或 starvation 为代价。

CUDA Graph 在全部 run 中关闭，避免把 Graph replay 收益混入调度收益。FCFS 是严格到达
顺序基线，遇到不兼容的队首请求时不会跨过它组批；Cache-Aware 可以在 pending queue
中选择兼容请求，同时保留 deadline guard 和 starvation guard。

## Controlled workload

[`scheduler_ab.json`](../benchmarks/configs/scheduler_ab.json) 固定 1000 个 burst 请求、
4 个 medium、batch 上限 8、同一资源预算和 5 秒请求 timeout。medium context 在计时前
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
  --output-dir /home/xinyang/feno-rt-results/scheduler_<session-name>
```

该目录必须在仓库外。矩阵完成后先检查 `summary.md`、`session.json` 和 24 个 run 的状态，
再将整个结果目录复制到 `benchmarks/results/scheduler_ab/` 并提交。

## Scope

主矩阵刻意使用固定 request shape、单一 burst 到达模式和统一 timeout，以隔离复用程度与
调度策略。它不代表所有负载强度或 SLO。deadline、starvation 和三类 admission budget
已有行为测试；不同到达率、混合 request shape 和多档 SLO 属于后续压力实验，不能从本
矩阵外推结论。
