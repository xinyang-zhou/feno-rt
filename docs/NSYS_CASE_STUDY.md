# CUDA Graph：执行机制与正式 A/B

## 问题与假设

缓存命中后，在线尾部仍需提交多个 GPU 操作。对于小 batch，host/framework dispatch
与 kernel launch 的固定开销可能占据较大比例。CUDA Graph 将静态尾部预先捕获，使用
一次 replay 提交，目标是减少这部分开销。

验证该机制需要把 CUDA API、GPU queue 与 kernel 时间线对应起来。GPU Active/请求耗时
不是 SM 利用率；GPU Projection 是关联 GPU 操作的首尾跨度，包含空隙。仅凭 wall-clock
加速比不能证明单个 kernel 的计算变快。

## 实现与证据

`CUDAGraphTailRunner` 对 batch bucket 和输入 signature 建图，复制输入到固定地址，
replay 后 clone 输出保证调用方所有权。未填满 bucket 时 padding，返回前裁剪；capture
失败可回退 eager。Graph 默认关闭，使用 `enable_cuda_graphs` 显式启用。

静态输入复制与输出 clone 仍有成本，必须计入端到端耗时。采用 graph 级追踪时，内部
kernel 被折叠显示，不能把更少的独立 launch 记录解释成相同数量的 kernel 消失。

## 正式验证

[RTX 5090 正式结果](../benchmarks/results/cuda_graph_ab/20260922_rtx5090_a69f11e/summary.md)
对应 `a69f11ecc1d656f02cac09cff6fc1dddb4892bb3`，Graph off/on、batch 1/2/4/8、每配置
1000 请求、3 个独立进程，共 24/24 通过。全部关闭 profiler，并把 capture 放在计时窗口外。

batch=1 的 run-level mean 中位数为 1.152→0.682 ms，成对延迟降幅中位数 41.11%，
吞吐比中位数 1.688x，峰值 allocated 增加 11.02 MiB。batch=8 的对应延迟降幅 19.48%。
不同统计量分别取中位数，因此不要求“两个中位数的比”等于“成对比值的中位数”。

适用范围是相同模型、all-hit、固定 bucket 的同步 runner invocation，不包含 Engine
排队。capture 约 51–57 ms，需要单独考虑摊销；收益随 batch 增大而缩小，符合固定提交
开销占比下降的假设。正式结果不采集 kernel launch 次数，不能仅凭这些结果量化
host dispatch、kernel 计算和内存复制分别贡献了多少收益。

## 剩余问题

geometry/wavelet 命中后仍有 `cat/index_select`，Graph replay 前后还有复制与 clone。
完整 Engine 的配对诊断应检查瓶颈是否转移到排队、调度、组批或输出所有权处理，
再判断是否值得优化这些路径。

新的采集入口、时间线语义及服务器命令见 [ENGINE_PROFILING.md](ENGINE_PROFILING.md)。
真实模型的[完整 Engine 案例](../benchmarks/results/engine_profile/20260922_rtx5090_6351d53/README.md)
进一步显示 admission、身份构造和调度的 host 开销；Graph 下仍有明显 GPU 提交空隙，
不能仅因 single/double buffer 总耗时接近就判断设备计算成为瓶颈。
