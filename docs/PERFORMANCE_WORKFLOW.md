# 一页性能分析流程

```mermaid
flowchart TD
    A[固定 commit / GPU / 模型 / workload / SLO] --> B[定义 cache 与计时边界]
    B --> C[预热后建立无 profiler 基线]
    C --> D{正确性与样本完整?}
    D -->|否| E[保存失败原始证据 / 诊断 / 新 commit 重跑]
    E --> A
    D -->|是| F[小样本 NVTX / Nsight 分解请求链路]
    F --> G{主导耗时在哪里?}
    G -->|排队或组批| H[检查 workload / SLO / batch fill / 调度 CPU]
    G -->|host 提交空隙| I[检查 Graph / 输入准备 / 微小操作 / 同步]
    G -->|GPU 重 kernel| J[用 Nsight Compute 分析计算与访存]
    H --> K[提出可验证假设 / 估计 Amdahl 上限]
    I --> K
    J --> K
    K --> L[最小改动 / 正确性与生命周期测试]
    L --> M[无 profiler 独立 A/B / p50 p95 p99 吞吐显存]
    M --> N{结果稳定且可追溯?}
    N -->|否| E
    N -->|是| O[报告适用边界 / 剩余瓶颈 / 决定继续或停止]
```

三个不能互换的时间：queue 是等待；service 包含 host 提交与同步；GPU Active 是实际
GPU 操作活跃区间。Projection 是首尾跨度。分位数不相加，重叠 phase 的累计时间不相加。

正式结果必须保存原始 samples、配置/trace/model SHA、硬件软件身份、每个独立 run、
正确性和错误记录。失败运行不能静默删除，诊断耗时不能作为正式速度提升。
