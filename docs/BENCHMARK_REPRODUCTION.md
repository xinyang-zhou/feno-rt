# GPU benchmark reproduction

在仓库根目录执行以下命令。正式矩阵要求干净的 Git checkout，并将完整源码 commit、
配置、模型和输入 trace 的 SHA256 记录在结果中。输出目录放在仓库外，每次实验使用新目录。

## Environment

```bash
git rev-parse HEAD
git status --short
python -m pip install -r requirements/runtime.lock -r requirements/test.lock
python -m pip install -e . --no-deps
export FENO_CHECKPOINT=/path/to/feno_test.pth
export FENO_NORMALIZATION=/path/to/norm_params_freq.npz
export CUDA_VISIBLE_DEVICES=GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

将占位路径和 GPU UUID 替换为实际值。配对实验使用相同 GPU、checkpoint、normalization
和软件环境，尽量独占设备。模型文件不提交到仓库。若当前 checkout 有修改，使用独立的
干净 checkout 运行正式实验，保留原工作文件。

## Scheduler comparison and batch-size control

```bash
bash scripts/validate_server.sh /tmp/feno-validation/session-01
```

脚本先执行全部测试，再运行两套独立矩阵，每套包含 2 个策略 × 4 个场景 × 3 次独立进程：

| Directory | Configuration | Purpose |
|---|---|---|
| `scheduler_main` | `scheduler_ab.json`，batch 上限 8 | FCFS 与 Cache-Aware 完整调度对比 |
| `scheduler_batch1` | `scheduler_batch1_ab.json`，batch 上限 1 | 消除 batch size 差异的对照 |

每个 run 测量 1000 个请求，统一 30 秒 timeout，关闭 profiler 和 CUDA Graph。两套矩阵
读取相同请求 trace，分别汇总。计时边界、缓存状态和归因限制见
[SCHEDULER_EVALUATION.md](SCHEDULER_EVALUATION.md)。30 秒结果不能证明旧 5 秒 SLO。

保留 `commit.txt`、`tests.log`，以及两套矩阵各自的 `session.json`、`summary.json`、
`summary.md`、`reuse_throughput.svg`、24 个原始 JSON 和 `logs/`。脚本在测试或矩阵失败时
停止，已经生成的结果会保留；不要删除失败记录或只挑选通过的运行。

中断后，仅在 commit、配置和模型均未改变时，使用矩阵工具的 `--resume`：

```bash
python benchmarks/run_scheduler_ab_matrix.py \
  --output-dir /tmp/feno-validation/session-01/scheduler_main --resume
python benchmarks/run_scheduler_ab_matrix.py \
  --config benchmarks/configs/scheduler_batch1_ab.json \
  --output-dir /tmp/feno-validation/session-01/scheduler_batch1 --resume
```

只对已经创建的 session 使用 `--resume`；它不会覆盖已记录的失败 run。改变协议、源码
或模型后使用新目录，不混合不同实验的结果。也可按 [benchmarks/README.md](../benchmarks/README.md)
直接运行单套矩阵。

## Paired Engine diagnostics

`nsys` 必须支持所用 GPU、驱动以及 `--cuda-graph-trace=graph`。可用 `FENO_NSYS_BIN`
指定 Nsight Systems CLI，用 `FENO_PYTHON` 指定 Python。以下四组统一真实模型、
64 个 burst 请求、all-hit cache 和 Cache-Aware 策略，仅改变 Graph 与双缓冲开关：

```bash
bash scripts/profile_engine_nsys.sh /tmp/feno-profile/eager-single \
  --requests 64 --all-hit --single-buffer
bash scripts/profile_engine_nsys.sh /tmp/feno-profile/eager-double \
  --requests 64 --all-hit
bash scripts/profile_engine_nsys.sh /tmp/feno-profile/graph-single \
  --requests 64 --all-hit --single-buffer --cuda-graphs
bash scripts/profile_engine_nsys.sh /tmp/feno-profile/graph-double \
  --requests 64 --all-hit --cuda-graphs
```

每组保留 `engine.nsys-rep`、`engine.sqlite`，以及 `diagnostic/` 中的 `diagnostic.json`、
`trace.json`、`requests.json` 和 `report.md`。比较前核对模型、输入 SHA、请求数量和到达
间隔；正确性必须通过，`dropped_events`、`active_requests` 和请求错误数必须为零。

分析只选 `feno.engine_window`，预热和 capture 位于窗口外。队列等待、host service
和 GPU execution 的解释见 [ENGINE_PROFILING.md](ENGINE_PROFILING.md)。这些带 profiler
的耗时用于瓶颈诊断，不用于正式性能表。`--toy` 只验证采集路径，不能替代真实模型。

## Review and publication

验收正式结果时检查完整矩阵、每次运行的请求完成数、正确性、依赖、模型和 trace 身份。
报告 run-level 统计和配对结果，保留无收益场景及尾延迟、显存、调度开销等代价。

原始 JSON、日志和 Nsight 报告可能包含个人目录、命令行及机器标识。原始文件在本地保留，
先准备并检查公开副本，再提交图表与可复现结果。移除个人路径；需要设备配对身份时使用
一致的匿名标识。不要直接提交整份未经审阅的运行目录或二进制报告。

公开副本若经过脱敏，注明处理字段，重新生成其文件校验值和汇总引用；保留源码、配置、
模型、输入 SHA 与性能样本，不修改测量值或运行状态。公开材料不包含个人进度、讲稿或模型文件。
