# Formal benchmark results

本目录只保存符合 `docs/PERFORMANCE.md` 的原始 Formal JSON 和由这些 JSON 生成的汇总。
Diagnostic profiler 产物和 Historical 结果不能混入当前版本的 Formal 汇总。

## Published results

- [CUDA Graph A/B：RTX 5090，commit `a69f11e`](cuda_graph_ab/20260922_rtx5090_a69f11e/summary.md)
  - 2026-09-22 UTC；
  - batch 1/2/4/8，Graph off/on，每组 3 个独立进程；
  - 每个配置 1000 个请求，steady all-hit cache；
  - 24/24 runs 通过正确性与结果校验。

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
`reuse_throughput.svg`。在正式服务器矩阵完成前，不建立占位结果目录或发布性能数字。

每个 run 文件必须自包含关键环境、模型、workload、计时、显存和正确性信息。机器绝对
路径可以用于当次运行日志，但可提交结果应优先保存仓库相对路径、非敏感标识和 SHA256。
建议正式运行先写入仓库外的临时目录，全部验证通过后再整体复制到本目录；源码
`git_dirty` 状态应在创建本次输出前采集。

提交前验证每个 run：

```bash
python benchmarks/validate_result.py benchmarks/results/.../run_....json
```

结果规则：

- 原始 latency samples 不得从 run 文件删除；
- failed correctness run 保留，但不进入 summary；
- 不手工修改生成的 percentile、throughput 或 speedup；
- summary 必须列出输入 run 路径和 SHA256；
- summary 的 percentile 必须来自独立 run-level 指标，不能合并三轮 raw samples；
- `.nsys-rep`、`.sqlite`、checkpoint 和模型权重不放入本目录。
