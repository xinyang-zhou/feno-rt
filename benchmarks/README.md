# Benchmarks

本目录将功能 smoke benchmark 与正式性能实验分开。

调度实验使用统一 30 秒 deadline，协议与解释边界见
[调度实验设计](../docs/SCHEDULER_EVALUATION.md)。测试、两套调度矩阵和 Engine 采集命令见
[实验复现步骤](../docs/BENCHMARK_REPRODUCTION.md)。

`profile_engine.py` 与 `summarize_engine_profile.py` 只生成 diagnostic 材料，不进入正式
性能汇总。`scheduler_batch1_ab.json` 是固定 batch=1 的独立对照配置。

RTX 5090 的[调度结果及退化对照](results/scheduler_ab/20260922_rtx5090_6351d53/README.md)
和[完整 Engine 诊断](results/engine_profile/20260922_rtx5090_6351d53/README.md)包含公开脱敏样本。
`analyze_engine_sqlite.py` 提取测量窗口内的匿名 GPU 区间；`plot_performance_results.py`
从公开 JSON 重建图表，额外依赖 `matplotlib==3.10.8`，不属于推理运行时依赖。

- `benchmark_core.py`：使用小型随机模型检查主要运行路径和数值等价性；
- `configs/`：版本化实验配置；
- `schema/`：配置和结果的机器可读契约；
- `workloads/`：确定性 workload 定义或请求 trace；
- `results/`：可追溯的原始 Formal 结果和汇总。

正式性能实验必须遵守 [`docs/PERFORMANCE.md`](../docs/PERFORMANCE.md)。配置文件只保存
可移植参数；checkpoint、normalization 和设备映射等机器相关信息在运行时解析，并将
解析后的标识、SHA256 和环境写入每个结果文件。

## Graph A/B baseline configuration

[`configs/graph_ab.json`](configs/graph_ab.json) 固定以下主要变量：

- batch size 为 1、2、4、8；
- 每个配置 1000 个请求；
- 3 个独立进程；
- 20 次 warmup invocation；
- medium、geometry 和 wavelet 均为 all-hit；
- Graph off/on；
- profiler 关闭；
- 保存全部原始 batch latency samples。

当前参考 runner 只实现 `steady_all_hit`。cold 或 partial-hit 需要单独定义 cache clear、
预填充和复用边界后另建配置，不能仅修改标签复用本 runner。

[`graph_workload.py`](graph_workload.py) 从配置生成 batch 1/2/4/8 的 tracked workload
manifest。manifest 明确固定 velocity 公式、source、frequency、receiver 规则和请求顺序，
Graph on/off 必须读取同一个对应文件。

配置中的 `checkpoint_env` 和 `normalization_env` 是环境变量名称，不是文件路径。
正式结果必须记录解析后文件的 SHA256，不能只记录环境变量或本机绝对路径。
多 GPU 机器建议先运行 `nvidia-smi --query-gpu=index,uuid,name --format=csv,noheader`
取得 UUID，并用 UUID 设置 `CUDA_VISIBLE_DEVICES`。正式 runner 会优先按运行时 UUID
解析 logical/physical GPU 映射；无法消除枚举顺序歧义时会拒绝生成结果。

## Validation

先验证 checked-in 配置：

```bash
python benchmarks/validate_result.py benchmarks/configs/graph_ab.json --kind config
```

正式结果写入后运行：

```bash
python benchmarks/validate_result.py path/to/run.json
```

校验器始终执行协议中的关键语义检查。如果环境安装了 `jsonschema`，还会使用
Draft 2020-12 schema 执行完整结构校验。校验通过只表示结果格式和内部关系有效，
不代表性能结论已经经过人工审阅。

开发环境可安装固定的测试依赖以启用完整 schema 校验；CI 会强制安装该文件：

```bash
python -m pip install -r requirements/test.lock
```

## One formal run

一次命令只运行一个 Graph mode、一个 batch size 和一个 repeat，并将结果先写到仓库
外部。checkpoint 与 normalization 通过配置指定的环境变量解析：

```bash
export FENO_CHECKPOINT=/path/to/feno_test.pth
export FENO_NORMALIZATION=/path/to/norm_params_freq.npz
export CUDA_VISIBLE_DEVICES=GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
python benchmarks/benchmark_graph_ab.py \
  --mode off \
  --batch-size 1 \
  --repeat-index 1 \
  --output /tmp/feno-graph-ab/graph_off_batch1_run1.json
```

正式 runner 会拒绝 dirty Git worktree、仓库内部输出路径、已存在的输出文件、不匹配的
workload manifest、缺失的模型文件、与 `requirements/runtime.lock` 不一致的直接依赖，
以及不在配置矩阵内的参数。CUDA Graph capture 在正式 steady-state 循环之前完成并
单独记录。

`batch_latency_ms` 保存每次同步 batch invocation 的原始 service time；throughput 使用
包含 invocation 间 Python 开销的完整 measurement window 计算，而不是简单用原始
latency samples 求和。

## Complete formal matrix

确认服务器使用干净的 commit、独占目标 GPU，并设置模型文件后，一条命令运行完整矩阵：

```bash
export FENO_CHECKPOINT=/path/to/feno_test.pth
export FENO_NORMALIZATION=/path/to/norm_params_freq.npz
export CUDA_VISIBLE_DEVICES=GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
python benchmarks/run_graph_ab_matrix.py \
  --output-dir /tmp/feno-graph-ab/<session-name>
```

执行前可以只查看 24 个子进程命令，不创建结果目录：

```bash
python benchmarks/run_graph_ab_matrix.py \
  --output-dir /tmp/feno-graph-ab/dry-run \
  --dry-run
```

矩阵包含 2 个 Graph mode × 4 个 batch size × 3 次重复。每个配置由新的 Python
进程运行，所有进程串行使用同一张 GPU；相邻的 on/off 构成一对，第二轮反转 mode 和
batch 顺序以减小固定执行顺序造成的漂移。`session.json` 在每个子进程后原子更新，日志
写入 `logs/`。若基础设施错误或人工中断，在相同 commit 和配置下恢复：

```bash
python benchmarks/run_graph_ab_matrix.py \
  --output-dir /tmp/feno-graph-ab/<session-name> \
  --resume
```

全部运行后自动生成：

- `summary.json`：run-level median/min/max/CV、逐 repeat 配对的 reduction/speedup 和
  输入文件 SHA256；
- `summary.md`：用于人工复核的表格；
- 24 个原始 Formal JSON：保留全部 batch latency samples；
- 24 个独立进程日志和一个记录真实执行顺序的 `session.json`。

汇总器不会把三轮 raw samples 合并后重新计算 percentile。也可以对已有目录单独重建
汇总：

```bash
python benchmarks/summarize_graph_ab.py \
  --input-dir /tmp/feno-graph-ab/<session-name>
```

## Scheduler A/B baseline configuration

[`configs/scheduler_ab.json`](configs/scheduler_ab.json) 比较严格 FCFS 和 Cache-Aware
调度。两种策略逐对读取完全相同的 checked-in trace；CUDA Graph 固定关闭，batch limit、
资源预算、SLO、模型和 cache 初始状态保持一致。

主矩阵包含 4 种复用分布：几乎无 geometry/wavelet 复用、均匀复用、长尾复用和高热点
复用。每个 trace 有 1000 个请求和 4 个已预热 medium，geometry/wavelet 在计时前清空。
2 种策略 × 4 个 trace × 3 次独立进程，共 24 个 run。

先检查配置和 trace：

```bash
python benchmarks/validate_result.py benchmarks/configs/scheduler_ab.json --kind config
python benchmarks/scheduler_workload.py --check
```

一次单独运行：

```bash
python benchmarks/benchmark_scheduler_ab.py \
  --policy fcfs \
  --scenario uniform_reuse \
  --repeat-index 1 \
  --output /tmp/feno-scheduler-ab/scheduler_fcfs_uniform_reuse_run1.json
```

完整矩阵：

```bash
export FENO_CHECKPOINT=/path/to/feno_test.pth
export FENO_NORMALIZATION=/path/to/norm_params_freq.npz
export CUDA_VISIBLE_DEVICES=GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
python benchmarks/run_scheduler_ab_matrix.py \
  --output-dir /tmp/feno-scheduler-ab/<session-name>
```

结果目录必须位于 Git 仓库外，且首次运行时不存在。runner 要求干净工作树，并验证模型
文件、依赖 lock、GPU 映射和 trace。执行中断后可在同一 commit、配置和模型文件上加
`--resume`。完整输出包含 24 个原始 JSON、日志、`session.json`、`summary.json`、
`summary.md` 和 `reuse_throughput.svg`。

结果同时记录吞吐、queue/execution/end-to-end p50/p95/p99、batch fill、各层 cache
指标、批内共享比例、deadline/starvation、调度器 CPU 开销、峰值显存和正确性。汇总只对
run-level 指标取统计量，不合并三轮原始请求 samples。实验设计与结论准入见
[`docs/SCHEDULER_EVALUATION.md`](../docs/SCHEDULER_EVALUATION.md)。
