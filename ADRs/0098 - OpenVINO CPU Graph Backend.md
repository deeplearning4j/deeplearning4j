# ADR 0098 - OpenVINO CPU Graph Backend

## Status
Implemented

Proposed by: Adam Gibson (April 2026)

## Context

ADR 0067 covers fused SDPA via OneDNN's Graph API for ~40 ops. However, OneDNN's op coverage is limited — a typical transformer model uses 200+ distinct ops, leaving most segments unfusible. OpenVINO's inference engine supports ~200 ops via its opset13, offering much broader fusion coverage on CPU.

The `GraphBackend` interface (used by CUDA Graphs, OneDNN, ACL, MLX, NNAPI, MLIR) provides the integration point: `canFuseSegment`, `compileSegment`, `executeSegment`.

## Decision

Implement `OpenVinoGraphBackend` as a `GraphBackend` in the CPU backend chain, placed before OneDNN due to its broader op coverage.

### Backend Chain Ordering

`NativeDynamicShapePlan_segments.cpp` builds the runtime chain via `getCpuGraphBackendChain()`:

```
MLX → OpenVINO → OneDNN → ACL → NNAPI → ArmHybrid → MLIR
```

OpenVINO before OneDNN because it covers ~200 ops vs OneDNN's ~40. When mode is `SLOT_BY_SLOT` the chain is empty. GPU-only modes (`TRITON`, `CUDA_GRAPHS`) bypass the chain on CUDA builds but fall through to it on CPU-only builds.

### Mixed-Segment (Island) Execution

Ops that cannot be mapped to OpenVINO opset13 (e.g., SSM recurrence) split a segment into contiguous "OV islands" of mappable ops separated by "NativeRange" blocks. Execution interleaves compiled `ov::InferRequest` calls with `NativeSlotExecutor` callbacks for unmappable ops.

### Runtime Configuration

The constructor configures `ov::Core` for single-request autoregressive decode:

| Setting | Value | Rationale |
|---|---|---|
| Performance mode | LATENCY | Single stream, all threads intra-op (THROUGHPUT rejected — replicates model buffers, causes OOM) |
| Hyper-threading | Disabled | Reduces contention on shared resources |
| CPU pinning | Enabled | Prevents thread migration overhead |
| Core selection | P-cores only | On hybrid CPUs (Intel 12th gen+) |
| Disk cache | `~/.nd4j/openvino_cache` | Avoids recompilation across runs |

### FP16 Handling

At startup, queries OneDNN (`dnnl_get_effective_cpu_isa`) to detect AVX512-FP16 / AMX-FP16. If absent (e.g., AMD Ryzen), all FP16 parameters are promoted to FP32 before inference. Promoted tensors are cached per island to avoid per-token allocation overhead.

### Compilation Caching

Two-level cache:
1. **Segment-level LRU** (`cacheLru_` / `cache_`): unlimited entries (a prior 772-entry limit caused eviction thrashing on a 1913-slot Qwen model)
2. **Topology-level** (`modelCache_`): shares `ov::CompiledModel` across transformer layers with identical op structure — each segment holds only a lightweight `ov::InferRequest`

### Location

- `libnd4j/include/graph/cpu/OpenVinoGraphBackend.h`
- `libnd4j/include/graph/cpu/OpenVinoGraphBackend.cpp`

## Consequences

- Broader CPU fusion coverage (~200 ops vs OneDNN's ~40)
- Island execution handles mixed segments without falling back to fully unfused execution
- LATENCY mode avoids model buffer replication that caused OOM with THROUGHPUT mode
- Topology sharing reduces compiled model memory across identical transformer layers

## Related ADRs

- [0067](0067%20-%20Scaled%20Dot-Product%20Attention%20Optimization.md) — OneDNN SDPA fusion (narrower scope, same CPU backend)
- [0091](0091%20-%20LlamaCpp%20OneDNN%20cuDNN%20Backend%20Classifiers.md) — backend classifier Maven profiles
- [0061](0061%20-%20DynamicShapePlan%20Execution.md) — DSP execution engine that drives segment scheduling
