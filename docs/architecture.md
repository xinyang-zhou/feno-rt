# Core architecture

FENO-RT 将一次完整前向拆成三个具有不同输入依赖的阶段，而不是套用自回归
模型的 token KV cache。

## Three cache levels

| Cache | Key dependency | Cached value |
|---|---|---|
| medium-static | canonical velocity digest | encoder latent, decoder tokens and projected K/V |
| geometry-prefix | medium key, source and receiver geometry | frequency-independent query |
| wavelet | exact frequency bits | wavelet tokens and projected K/V |

每个缓存都是按 tensor payload 字节限制容量的线程安全 LRU。缓存 entry 可以
通过 lease 暂时固定，被固定的 entry 不参与淘汰。

三个缓存都由单个 `FENOModelRunner` 私有持有，因此模型、归一化、dtype 和
设备由 runner 作用域保证，不重复写入 cache key。medium 是 geometry-prefix
的依赖根。调用 `FENOCacheBundle.invalidate_medium` 时，运行时先删除
geometry-prefix，再删除 medium。wavelet 与 medium 无关，不会被误删。

速度模型在 CPU 上转换为规范化的 float32 表示后计算 digest，因此物理速度
输入和等价的 `already_normalized=True` 输入会命中同一个 medium entry。
`MediumContext` 带有 runner 所有权标记，不能跨 runner 复用。

## Dynamic batching

`AsyncFENOEngine` 接收单请求并在后台形成 micro-batch。每个 batch 同时受
以下预算约束：

- request count；
- query token count；
- estimated activation bytes；
- output bytes。

FCFS 是严格到达顺序基线。cache-aware scheduler 会按 batch compatibility、
缓存驻留、重复 geometry/frequency、priority 和等待时间排序。接近 deadline
或等待超过 starvation threshold 的请求优先。

模型执行由单独的 executor thread 串行拥有。一个 batch 失败时，引擎可以递归
二分重试，从而只让非法请求失败。

## CUDA Graph

CUDA Graph 只覆盖已经完成三级缓存查找后的在线尾部。运行时按 batch size
选择 bucket，为 prefix、frequency、wavelet K/V 和输出建立持久 buffer。
不足 bucket 的 batch 使用最后一个样本填充，返回前再裁剪。

图缓存 key 包含 bucket、输入尾部形状和 dtype。第一次请求捕获，后续相同
signature 的请求更新静态 buffer 并 replay。返回值会 clone，避免下一次
replay 修改调用方仍在使用的结果。
