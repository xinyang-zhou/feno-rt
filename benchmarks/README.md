# Benchmarks

本目录将功能 smoke benchmark 与正式性能实验分开。

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

配置中的 `checkpoint_env` 和 `normalization_env` 是环境变量名称，不是文件路径。
正式结果必须记录解析后文件的 SHA256，不能只记录环境变量或本机绝对路径。

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
