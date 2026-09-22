# Published benchmark evidence

本目录保存经过审阅与脱敏的性能证据。Formal 结果保留全部原始测量样本，汇总遵循
`docs/PERFORMANCE.md`。Diagnostic 时间线放在独立目录中，不混入 Formal 性能汇总。

## Published results

- [CUDA Graph A/B：RTX 5090，commit `a69f11e`](cuda_graph_ab/20260922_rtx5090_a69f11e/README.md)
  - 2026-09-22 UTC；
  - batch 1/2/4/8，Graph off/on，每组 3 个独立进程；
  - 每个配置 1000 个请求，steady all-hit cache；
  - 24/24 runs 通过正确性与结果校验。
- [调度主矩阵与 batch=1 对照：RTX 5090，commit `6351d53`](scheduler_ab/20260922_rtx5090_6351d53/README.md)
  - 两套矩阵各 24 个 run，共 48/48 通过；
  - 每 run 1000 请求、30 秒 timeout、Graph/profiler 关闭；
  - 同时报告组批收益、singleton 退化、尾延迟及显存代价。
- [完整 Engine 诊断：RTX 5090，commit `6351d53`](engine_profile/20260922_rtx5090_6351d53/README.md)
  - 真实模型、64 请求、eager/Graph × single/double buffer；
  - 四组匿名 host 时间线和 GPU 区间证据；
  - 单次诊断采集，不作为正式加速数字。

建议目录格式：

```text
results/
└── <benchmark>/
    └── <YYYYMMDD>_<gpu-slug>_<commit12>/
        ├── graph_<mode>_batch<batch>_run<repeat>.json
        ├── logs/
        ├── session.json
        ├── summary.json
        └── summary.md
```

Scheduler 结果使用 `scheduler_ab/<YYYYMMDD>_<gpu-slug>_<commit12>/`，原始 run 命名为
`scheduler_<policy>_<scenario>_run<repeat>.json`，并额外保存自动生成的
`reuse_throughput.svg`。使用主矩阵和独立对照时分别放在 `main/`、`batch1/` 中。

每个 run 文件必须自包含关键环境、模型、workload、计时、显存和正确性信息。机器绝对
路径可以用于当次运行日志，但可提交结果应优先保存仓库相对路径、非敏感标识和 SHA256。
正式运行先写入仓库外的目录；完成校验后建立公开副本，去除个人路径、真实设备 UUID
和 OS 进程/线程标识，记录字段变更，再重建引用文件 SHA 的汇总。源码 `git_dirty`
状态应在运行时采集，脱敏后的副本通过 `publication.json` 说明与原始文件的区别。

提交前验证每个 run：

```bash
python benchmarks/validate_result.py benchmarks/results/.../run_....json
```

结果规则：

- 原始 latency samples 不得从 run 文件删除；
- failed correctness run 保留为诊断证据；完整矩阵未通过时不发布性能对比；
- 不手工修改生成的 percentile、throughput 或 speedup；
- summary 必须列出输入 run 路径和 SHA256；
- summary 的 percentile 必须来自独立 run-level 指标，不能合并三轮 raw samples；
- `.nsys-rep`、`.sqlite`、checkpoint 和模型权重不放入本目录。

每个已发布数据集的 `publication.json` 记录处理方式与来源。二进制原始捕获、
环境转储、收集压缩包以及个人记录只保留本地，不作为公开数据集的一部分。
