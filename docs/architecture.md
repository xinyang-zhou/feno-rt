# Runtime architecture

FENO-RT serves a single-shot, non-autoregressive scientific operator. It reuses
intermediate tensors according to their real input dependencies instead of
applying an autoregressive token-cache design.

## Reusable computation

```text
medium input
  -> encoded medium latent
     -> decoder-static context and projected K/V
        -> geometry prefix(medium, source, receivers)
           -> output(..., frequency)

frequency
  -> wavelet/frequency context and projected K/V
     -> joins the geometry prefix in the online tail
```

Structured cache keys include model version, input digest, normalization,
precision, device, and downstream inputs as appropriate. Invalidating a medium
cascades to dependent decoder and geometry entries. Frequency state is
independent and is not cleared unnecessarily.

## Request lifecycle

1. A CPU medium source is registered with one or more workers.
2. The external context stores stable worker tokens, not CUDA tensors.
3. The router combines replica affinity, cache residency, queue pressure, and
   memory headroom.
4. The selected dynamic batcher groups compatible requests while enforcing
   query-token, activation-byte, and output-byte budgets.
5. Deadlines and starvation protection override locality ranking.
6. A medium lease is acquired only when a batch starts actual execution.
7. A pinned entry is promoted H2D; an absent materialized entry is rebuilt from
   its registered CPU source.
8. Duplicate geometry and frequency work is computed once per batch.
9. The selected eager/compiled/CUDA-Graph path produces the output.
10. Multi-GPU CPU-result serving copies D2H into a leased host-registered shared
    slot, falling back to an owned pinned result if all slots are held.
11. Execution leases are released and any temporary capacity overflow is
    trimmed immediately.

## Medium ownership

`MediumTierHandle` is a tensor-free identity. `MediumTierManager` is the sole
worker-local owner of materialized GPU and pinned-CPU latents. A medium has at
most one materialized representation in these two tiers.

GPU eviction demotes to page-locked CPU when capacity permits and invalidates
dependent GPU decoder/geometry entries. A pinned hit promotes back to GPU.
Leased entries are never victims. If every candidate is leased, execution may
temporarily exceed the configured byte capacity and converges after release.

The eviction score combines estimated rebuild/transfer cost, access frequency,
recency, and bytes. This avoids treating a cheap cold tensor and an expensive
hot tensor as equivalent simply because of LRU order.

## Process isolation

For multi-GPU CPU-result serving, each GPU is owned by a spawned process. The
child owns its CUDA context, model, streams, cache, and medium tier. The parent
owns routing and communicates through a narrow RPC interface. This design also
keeps large outputs out of Python pipes by using shared host memory.

Single-GPU serving stays in-process. GPU-tensor-returning compute-only paths are
also in-process because device tensors are deliberately not transported across
process boundaries.

## Operational controls

- Prometheus text includes HTTP, request, routing, worker, cache, tier, and D2H
  counters/gauges.
- Bounded JSONL traces store replay inputs and output summaries, not full output
  arrays.
- Profiler sessions write only inside a configured directory and accept a
  restricted filename alphabet.
- A materialized flush preserves source registration and handles; context
  release removes source registration as well.
