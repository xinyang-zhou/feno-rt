# 从 FENO-RT 迁移到 LLM Serving

核对日期：2026-09-22。下列是设计概念对照，不表示 FENO-RT 已实现 LLM 的生成循环或分页 KV。
外部实现会演进，具体开关以目标项目所用 commit 的文档为准。

## 缓存、调度和执行

| FENO-RT | LLM Serving 对应概念 | 可迁移原则 | 关键差异 |
|---|---|---|---|
| medium/geometry/wavelet 中间 Tensor Cache | prefix/KV cache | 按真实输入依赖复用计算；key 必须覆盖语义 | FENO 缓存固定阶段产物，LLM KV 随序列增长 |
| runner 私有 key 作用域 | model/adapter/cache salt 身份 | 避免跨模型、精度或租户错误命中 | FENO 当前没有共享跨租户 cache |
| byte LRU 与 lease | KV block 引用计数与回收 | 使用中的资源不能被淘汰 | lease 本身不提供分页、block table 或 PagedAttention |
| medium→geometry 显式失效 | prefix 依赖与 block 身份 | 上游变化必须处理下游有效性 | FENO 的 wavelet 与 medium 独立；不是一条 token 前缀树 |
| cache-aware batching | locality/prefix-aware scheduling | 在局部性、batch fill、公平性间权衡 | FENO 请求一次前向结束，LLM 请求经历多轮 decode |
| micro-batch queue | continuous batching | 在调度点选择当前可执行工作 | continuous batching 可在生成迭代边界加入/移除序列 |
| query/activation/output 预算 | token budget 与 KV capacity | admission 同时受算力和内存约束 | FENO 估计成本不等于 LLM 实际 KV block 分配量 |
| bucketed tail Graph | full/piecewise CUDA Graph | 静态地址、shape、capture 与 replay 分离 | LLM 还需处理 prefill/decode 混合与 attention backend 兼容性 |

vLLM 的 prefix caching 以 block hash 识别可复用内容，block 有请求引用计数；它包含分配、
追加、释放和淘汰流程。FENO 的 lease 对应其中“使用中保护”这一部分，不能据此声称实现了
完整 KV 管理。[vLLM prefix caching](https://docs.vllm.ai/en/latest/design/prefix_caching/)

vLLM 的 CUDA Graph 设计包含 full、piecewise 及运行时选择。FENO 仅捕获缓存解析后的尾部，
需要比较的是静态 buffer、bucket、fallback 和显存成本这些约束。
[vLLM CUDA Graphs](https://docs.vllm.ai/en/latest/design/cuda_graphs/)

## Prefill、decode 与指标

Prefill 处理输入 token，decode 逐步产生后续 token。较大 prefill 通常更容易形成大矩阵
计算，低 batch decode 常受权重/KV 访存与提交开销影响；模型、并发和长度变化后需重新测量。
Chunked prefill 将长输入分块，在 token 预算内与 decode 调度；其目标涉及吞吐和 ITL/TTFT
权衡。FENO 没有 token 生成循环，因此不能直接把 batch service time 改名为 TPOT。
[vLLM chunked prefill](https://docs.vllm.ai/en/latest/configuration/optimization/#chunked-prefill)

| 指标 | 定义 | 与当前 FENO 的关系 |
|---|---|---|
| TTFT | 请求到首个输出 token 的时间 | FENO 当前一次性交付整个输出，没有首 token 阶段 |
| ITL | 相邻输出 token 的时间间隔 | FENO 不适用 |
| TPOT | 通常为首 token 之后生成耗时除以后续 token 数；必须注明工具口径 | FENO 不适用 |
| queue latency | 入队到执行开始 | 两者都适用，需说明 prefetch 是否算等待 |
| E2E | 请求提交到最终结果可交付 | 两者都适用；需说明包含哪些客户端/服务边界 |
| throughput/goodput | 总处理率/满足给定 SLO 的处理率 | FENO 调度实验保存超时数，不能用成功请求数隐藏被丢弃请求 |

## 并行与服务拆分

Tensor parallel 把单层张量计算分给多设备，并引入集合通信；expert parallel 将 MoE expert
分配给设备，并引入 token 路由和负载不均问题。它们与简单的多副本请求分发不同。FENO
`v0.8.0` 的 per-GPU replica 不能描述成 tensor/expert parallel。

Prefill/decode 分离让两阶段在不同资源池执行，需要 KV 传输、路由、排队以及失败恢复。
SGLang 的部署文档给出了分离的 prefill/decode worker 和传输后端。设计推论是：隔离阶段
干扰必须与额外传输、协调成本一起测量，不能假设分离必然降低延迟。
[SGLang PD disaggregation](https://docs.sglang.io/docs/advanced_features/pd_disaggregation)
