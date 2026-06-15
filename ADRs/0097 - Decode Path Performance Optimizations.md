# ADR 0097 - Decode Path Performance Optimizations

## Status
Implemented

Proposed by: Adam Gibson (April–June 2026)

## Context

Transformer LLM decode is bottlenecked by kernel launch overhead and global memory bandwidth. A 28-layer model (e.g., Qwen-0.8B) executes 280+ individual CUDA kernels per decode step. Profiling showed that kernel launch latency and memory bandwidth — not compute — dominated decode time.

ADR 0067 covers the fused SDPA/Flash Attention kernels (multi-backend). ADR 0089 covers CUDA graph capture/replay orchestration. This ADR covers the additional fused ops, batched GEMM optimizations, and composite replay infrastructure that together moved VLM decode from ~12 tok/s to ~70 tok/s.

## Decision

### Fused Transformer Ops

Three new CUDA kernels that combine multiple transformer operations into single launches.

**`skip_rms_norm`** — Residual add + RMS normalization in one kernel:
```
hidden = input + skip [+ bias]
output = hidden * rsqrt(mean(hidden²) + eps) * gamma
```
Eliminates a separate add kernel and one global memory round-trip per layer.

**`rms_norm_linear`** — RMS normalization + linear projection:
```
output = matmul(rms_norm(x, gamma, eps), W)
```
The intermediate normalized tensor never materializes in global memory. Backward pass (`rms_norm_linear_bp`) recomputes the norm. Both declared in `libnd4j/include/ops/declarable/headers/llm.h`, implemented in `helpers/cuda/rms_norm.cu`.

**Fused warp-shuffle attention softmax** — Single kernel for decode-path `Q@K^T + softmax + attn@V`:
- `__shfl_down_sync` warp-shuffle reductions for max and sum (no atomics)
- GQA support via `kvHead = qHead / headsPerKvHead`
- Eliminates 6 graph nodes per layer (two cuBLAS calls, permute copies, standalone softmax)

Implemented in `libnd4j/include/helpers/cuda/FlashAttentionHelper.cu`.

### Batched GEMM Mixed-Type Cast

DSP detects mixed-type GEMM groups (FLOAT32 activations × HALF weights) at plan compile time and pre-allocates persistent cast scratch buffers:

- `castScratch` via `cudaMalloc` (not pool) per group
- `batchedGemmCastFloat2Half` kernel casts activations in-place before cuBLAS dispatch
- Eliminates ~240 CUDA memory ops/step (was `cudaMalloc×2 + cudaFreeAsync×2` per group × 60 groups)

Impact: 41→49.76 tok/s (+21%).

Implemented in `libnd4j/include/graph/impl/NativeDynamicShapePlan_batchgemm.cu`.

### Composite Replay

`compositeReplay()` interleaves Triton CUDA graph islands with native gap-op segments into a unified replay schedule:

- `REPLAY_UNIT_TRITON_ISLAND`: captured CUDA graph for a fusible segment
- `REPLAY_UNIT_GAP`: native ops that cannot be captured (cuBLAS, dynamic shapes)
- Pre-allocated `compositeReplayHandles` per unit

The **frozen fast path** (`platformTryFrozenFastPath`) handles external input H2D sync, cross-stream ordering, and arg table refresh in a single tight codepath — bypassing the heavier `phaseReplay` segment iteration for the common decode steady state.

Implemented in `NativeDynamicShapePlan_gpubackend.cu` and `NativeDynamicShapePlan.cpp`.

### Additional Decode Hot-Path Optimizations

- Skip frozen constant slots during execution (`allFrozenConstants` detection post-auto-seal)
- Skip helper dispatch + timing memset in frozen DSP steady state
- Skip redundant error clearing and O(3) variable-only `syncExternalInputs`
- Gate `REQUIRE_TRUE` validation to first 3 decode steps
- `argTableStable` tracking for internal-only pointer changes (skip refresh + ext input sync)
- Active gap slot cache (skip 97% of slot iterations in compositeReplay)

## Consequences

- Decode throughput improved from ~12 to ~70 tok/s on RTX 4090 (SmolDocling VLM, Qwen-0.8B decoder)
- Kernel launch count reduced by ~84 per step from fused ops alone
- Batched GEMM cast eliminated a per-step allocation storm
- Composite replay unified Triton and native execution into a single schedulable pipeline

## Related ADRs

- [0067](0067%20-%20Scaled%20Dot-Product%20Attention%20Optimization.md) — fused SDPA / Flash Attention
- [0089](0089%20-%20CUDA%20Graph%20Capture%20and%20Replay.md) — CUDA graph capture/replay lifecycle
- [0071](0071%20-%20Triton%20Graph%20Backend.md) — Triton kernel fusion backend
- [0094](0094%20-%20DSP%20Buffer%20Coloring%20Pooling%20and%20Passivation.md) — GPU memory reduction for DSP plans
