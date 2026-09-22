# Performance evaluation protocol

本文规定 FENO-RT 性能实验的公共协议。除非结果满足本文的正式实验要求，否则不得
作为项目的正式性能结论。

当前仓库已发布一组按本协议采集的 CUDA Graph A/B 正式 GPU 结果：
[RTX 5090 / commit `a69f11e`](../benchmarks/results/cuda_graph_ab/20260922_rtx5090_a69f11e/summary.md)。
该结果包含 24 个独立进程 run 的原始 samples、环境与模型身份、正确性、显存、capture
成本和自动生成的汇总。`benchmark_core.py` 仍只是小模型 smoke benchmark，只用于检查
运行路径、输出正确性和基本指标是否可用。

## Result classes

性能材料分为三类，三者不能混用：

| Class | Purpose | Requirements | May support a headline claim |
|---|---|---|---|
| Formal | 比较端到端延迟、吞吐和显存 | profiler 关闭，大样本，独立重复，保存原始结果 | 是 |
| Diagnostic | 使用 NVTX、Nsight Systems 或其他 profiler 解释原因 | 可使用少量稳定请求，必须记录 profiler 配置 | 否 |
| Historical | 保存旧 tag 或旧 commit 的结果 | 必须标注版本、环境、命令和原始产物 | 仅限对应版本 |

Diagnostic 结果可以解释 kernel launch、同步、memcpy 和时间线空白，但不能替代
unprofiled Formal 结果。Historical 结果不能描述成当前 `main` 的性能。

## Required experiment identity

每个 Formal run 必须保存足够信息，使另一台同配置机器能够重现实验。

### Source

- 完整 Git commit SHA；
- 工作树是否干净；
- benchmark 入口和完整命令行；
- 配置文件的内容或 SHA256；
- 结果 schema 版本。

存在未提交代码时，结果可以用于开发调试，但不能进入正式汇总。
`git_dirty` 在实验会话启动、创建本次输出文件之前采集。建议先写入仓库外的临时结果
目录，整组实验结束并验证后再复制到 `benchmarks/results/`，避免前一轮生成的结果影响
后续独立 run 的源码状态判断。

### Hardware

- GPU 名称、总显存和 device UUID；
- physical GPU 与进程内 logical device 的映射；
- GPU 数量；
- driver 版本；
- persistence、power limit、application clocks 或其他可见的频率设置；
- CPU 型号、逻辑核数和系统内存；
- 操作系统与 kernel 版本；
- 实验期间是否存在共享 GPU 的其他进程。

Formal run 应尽量独占目标 GPU。无法独占时，必须记录共享状态，且不能隐藏由资源
竞争造成的异常波动。

### Software

- Python、PyTorch、CUDA runtime 和 cuDNN 版本；
- 安装方式和依赖 lock file；
- dtype、TF32 和 deterministic 设置；
- 影响执行的环境变量，例如 `CUDA_VISIBLE_DEVICES`；
- CUDA Graph、compile 或其他执行开关。

### Model

- 完整 `FENOModelConfig`；
- checkpoint 路径的非敏感标识与 SHA256；
- normalization 文件 SHA256；
- 参数数量和参数 dtype。

如果实验使用确定性随机初始化模型，必须记录随机种子并明确标记
`random_initialized=true`。这类实验可以比较相同 shape 下的运行时性能和数值等价性，
但不能支持模型精度结论。

### Workload

- workload 名称、schema 版本和随机种子；
- 输入 trace 或生成配置的 SHA256；
- 请求总数、batch size 和调用次数；
- source、receiver、frequency 的 shape、dtype 和取值生成方式；
- 请求到达模式；
- medium、geometry 和 frequency 的复用分布；
- cache 初始状态；
- CUDA Graph bucket、padding 和 fallback 条件。

“cold”或“partial hit”不能作为未定义的简称。结果必须分别说明 medium、geometry 和
wavelet cache 在计时开始前的状态，以及在哪个边界执行 clear 或预填充。

## Formal run rules

### Warmup and setup

模型加载、medium preparation、cache 预填充、CUDA context 初始化和 allocator 初始化
属于 setup，不进入 steady-state latency。warmup 次数必须固定并记录。

CUDA Graph 第一次 capture 的时间和显存单独报告。比较 steady-state replay 时，capture
必须在计时前完成；不得把 eager 的 steady-state 与 Graph 的首次 capture 调用直接比较。

### Timing boundary

直接测量 GPU runner 时使用单调高精度时钟，并在每个测量区间前后同步目标 CUDA
device 或 execution stream：

```text
synchronize
start timestamp
measured operation
synchronize
end timestamp
```

计时操作必须包含正常返回给调用方所需的 GPU 输出所有权处理，但不包含仅为统计而做
的 CPU checksum、JSON 序列化或额外 D2H copy。若使用 CUDA Events，结果中必须注明，
且不能与 wall-clock latency 混用同一个字段名。

异步 engine 需要分别记录：

- queue latency：enqueue 到 batch execution 开始；
- execution latency：batch execution 开始到结果可交付；
- end-to-end latency：submit 到对应 future 完成。

### Sample count and repetitions

- 每个配置至少测量 1000 个请求；
- 每个配置至少运行 3 个独立进程；
- 每次独立运行都重新创建 model、runner 和 CUDA context；
- warmup 不计入 1000 个测量请求；
- profiler、NVTX capture 和 GPU metrics sampling 在 Formal run 中关闭；
- 每次运行保存全部原始 latency samples，不只保存摘要。

这里的 request count 是实际请求数，不是 batch invocation 数。例如 batch size 8 下，
125 次完整 batch invocation 对应 1000 个请求。

### Batch latency semantics

对于直接 `forward_batch` 测量：

- `batch_latency_ms` 是一次 batch invocation 的同步 wall-clock 时间；
- batch 中每个请求都经历该次 batch service 时间；
- `batch_latency_ms / batch_size` 只能标为 `amortized_ms_per_request`；
- 不得把 amortized 值描述为单请求端到端 latency。

吞吐按测量窗口内完成的请求数计算：

```text
throughput_requests_per_second = completed_requests / measured_wall_time_seconds
```

## Required metrics

每个独立 run 至少保存：

| Metric | Definition |
|---|---|
| latency mean/p50/p95/p99/max | 从该 run 的原始 wall-clock samples 计算 |
| throughput | 测量窗口内完成请求数除以 wall-clock 时间 |
| baseline allocated bytes | steady-state 测量前已分配的 CUDA memory |
| peak allocated bytes | steady-state 测量期间 `max_memory_allocated` |
| peak reserved bytes | steady-state 测量期间 `max_memory_reserved` |
| correctness relative L2 | 相对 reference 输出的 L2 误差 |
| correctness max abs | 相对 reference 输出的最大绝对误差 |
| cache counters | 每层 hit、miss、entry、byte、eviction 和 rejection |
| Graph counters | capture、replay、fallback、padding 和 static buffer bytes |

Graph setup 还需单独保存 capture wall time、capture 前后 allocated/reserved memory 和
resident graph 数量。显存字段统一使用 bytes 保存，展示时再转换为 MiB 或 GiB。
累计 Graph counter 之外还要保存正式测量窗口的 counter 差值；exact-bucket 主实验要求
每个 measured invocation 对应一次已有 Graph replay，且 measured capture、fallback 和
padding 均为零。

steady-state 显存测量应在 setup、cache 预填充和 Graph capture 完成后同步设备，先记录
baseline，再调用 `reset_peak_memory_stats`，最后进入正式测量。这样 steady-state peak
不会与模型加载或 Graph capture 的临时峰值混在一起。

## Correctness gate

性能结果只有通过正确性检查后才有效。每个优化配置必须与同一 commit、同一模型和
同一输入上的 reference path 比较，并保存：

- relative L2 error；
- max absolute error；
- 使用的 `rtol` 和 `atol`；
- 输出 shape、dtype 和有限值检查；
- correctness pass/fail。

容差必须在运行前由配置确定，不能看到结果后放宽。发生 NaN、Inf、shape 不一致或
超出容差时，该 run 必须保留失败记录，但不得进入性能汇总。

## Repetition and summary rules

每个独立 run 单独计算 mean、p50、p95、p99、throughput 和 memory。最终 summary：

- 列出所有独立 run 的指标；
- 对 run-level 指标报告 median；
- 同时报告 min/max 或变异系数，用于展示运行间波动；
- 保留全部原始 samples；
- 不能只把三轮 samples 合并后报告一个 percentile；
- 不能静默删除最慢 run 或离群 sample。

若需要排除系统异常，排除条件必须在实验前定义，并在结果中保留被排除 run 及原因。

## CUDA Graph A/B contract

CUDA Graph 的 Formal A/B 除 Graph 开关外必须保持以下条件一致：

- Git commit、模型、checkpoint 和 dtype；
- GPU、软件环境和进程亲和设置；
- workload、请求顺序和随机种子；
- batch size、cache 初始状态和 warmup；
- 同步方式、计时边界和输出所有权；
- 正确性容差和显存统计边界。

每个 batch size 使用匹配的 Graph bucket 进行主要比较；padding 场景另行报告。Graph
capture 在 steady-state 测量前完成并单独计时。每轮 Graph on/off 使用成对 workload，
记录执行顺序，并在独立进程中运行。

必须同时报告：

- Graph on/off 的 run-level latency、throughput 和显存；
- latency reduction 与 throughput speedup 的计算公式；
- capture/setup 成本；
- replay、fallback、padding 和 resident graph 数；
- eager 与 Graph 输出的正确性；
- 不适用或回退的输入 signature。

Graph 收益应描述为减少 host/framework dispatch 和 CUDA launch 开销。除非另有独立的
GPU kernel 证据，不得声称 Graph 加快了单个 kernel 的计算。

## Claim admission policy

正式写入 README、release note 或性能摘要的数字必须同时满足：

1. 来自 profiler 关闭的 Formal run；
2. 满足最小请求数和独立重复次数；
3. correctness gate 通过；
4. 能定位到原始结果、配置、workload 和 Git commit；
5. 明确硬件、batch、cache 状态和统计口径；
6. 同时披露 latency、throughput、显存和适用范围；
7. 与 Diagnostic 和 Historical 结果明确分开。

缺少原始数据、复现命令或版本身份的数字不得作为当前项目的性能结论。

## Formal result checklist

发布结果前逐项确认：

- [ ] 工作树干净并记录完整 commit；
- [ ] 环境、模型和 workload identity 完整；
- [ ] cache 状态和 Graph bucket 明确；
- [ ] warmup、setup 和 steady-state 边界分离；
- [ ] 每配置至少 1000 请求和 3 个独立进程；
- [ ] profiler 关闭；
- [ ] 保存原始 latency samples；
- [ ] p50、p95、p99、throughput 和显存齐全；
- [ ] correctness gate 通过；
- [ ] Graph capture 成本和 fallback 状态齐全；
- [ ] summary 能追溯到原始结果和复现命令。

## Reference Graph A/B command

仓库中的参考编排器用 24 个独立、串行的 Python 进程完成 Graph on/off、batch
1/2/4/8、每组 3 次的正式矩阵，并自动保存 session、原始结果、日志和 run-level 汇总：

```bash
CUDA_VISIBLE_DEVICES=GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx \
  python benchmarks/run_graph_ab_matrix.py \
  --output-dir /tmp/feno-graph-ab/<session-name>
```

运行前还需要设置 `FENO_CHECKPOINT` 和 `FENO_NORMALIZATION`。完整命令、dry-run 和
中断恢复方法见 [`benchmarks/README.md`](../benchmarks/README.md)。
