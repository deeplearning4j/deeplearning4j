# PR08: C++ Platform Backends — Sub-Split Plan

**Total files:** 427 (382 platform + 45 graph backends)
**Merge layer:** 3
**Recommendation:** Split into 4 sub-PRs by platform family

Execution infrastructure (40 files) stays in the main PR08 since it is
shared across all backends.

## Sub-PR Summary

| Sub-PR | Name | Files | Description |
|---|---|---:|---|
| PR08a | ARM + Apple Backends | ~173 | armcompute/, accelerate/, mps/ |
| PR08b | Intel/x86 Backends | ~72 | mkldnn/oneDNN |
| PR08c | CUDA Ecosystem Backends | ~96 | llamacpp/, cudnn/, miopen/ |
| PR08d | Experimental Backends + Graph Backends | ~86 | mlir/, pjrt/, vlm/, graph backends, execution infra |

**Total: 427 files** (382 platform + 45 graph backends)

Note: 40 execution infrastructure files (`libnd4j/include/execution/`) are
included in PR08d since they are closely tied to multi-backend dispatch.

---

## PR08a: ARM + Apple Backends (~173 files)

All mobile/edge and Apple silicon backends.

### Files

| Path | Files | Contents |
|---|---:|---|
| `platform/armcompute/` | 125 | ARM Compute Library ops: conv2d, pool, gemm, batchnorm, softmax, etc. |
| `platform/accelerate/` | 28 | Apple Accelerate: vDSP/BLAS for avgpool, conv2d, gemm, softmax |
| `platform/mps/` | 20 | Metal Performance Shaders: GPU ops for Apple silicon |

### Graph backends included
| Path | Files | Contents |
|---|---:|---|
| `graph/cpu/AclGraphBackend.*` | 2 | ARM Compute Library graph backend |
| `graph/cpu/ArmHybridGraphBackend.*` | 2 | ARM hybrid (CPU+NPU) graph backend |
| `graph/cpu/MlxGraphBackend.*` | 2 | Apple MLX framework graph backend |
| `graph/metal/MetalReplayHandle.*` | 2 | Metal command buffer replay |

### Merge order
Independent from PR08b-d (no cross-platform references).

### Review focus
- ARM NEON intrinsics correctness
- MPS shader compilation
- Accelerate vDSP precision

### ADRs
- `ADRs/0073 - Hexagon MLIR Backend.md` (duplicate 0073, needs renumbering; shared mobile focus)

---

## PR08b: Intel/x86 Backends (~72 files)

oneDNN/MKL-DNN backend for x86 CPU acceleration.

### Files

| Path | Files | Contents |
|---|---:|---|
| `platform/mkldnn/` | 72 | oneDNN ops: batchnorm, conv2d/3d, deconv, gru, lstm, matmul, pool, softmax, tanh, etc. |

### Graph backends included
| Path | Files | Contents |
|---|---:|---|
| `graph/cpu/OneDnnGraphBackend.*` | 2 | oneDNN graph API backend |
| `graph/cpu/OpenVinoGraphBackend.*` | 2 | OpenVINO inference backend |

### Merge order
Independent from PR08a/c/d.

### Review focus
- oneDNN primitive creation and caching
- Memory format conversions (nchw ↔ blocked)
- Thread pool integration

### ADRs
- `ADRs/0058 - Multi-Backend Kernel Selection and Management.md`
- `ADRs/0059 - Multi-Backend Op Execution System.md`

---

## PR08c: CUDA Ecosystem Backends (~96 files)

NVIDIA CUDA backends: LLaMA.cpp CUDA kernels, cuDNN, and AMD MIOpen (ROCm).

### Files

| Path | Files | Contents |
|---|---:|---|
| `platform/llamacpp/` | 59 | LLaMA.cpp ops: attention, gated_delta_rule, matmul, rmsnorm, rope — includes `cuda/` subdirectory |
| `platform/cudnn/` | 30 | cuDNN ops: batchnorm, conv2d/3d, ctcloss, dropout, pool2d/3d |
| `platform/miopen/` | 5 | MIOpen/ROCm: AMD GPU equivalents of cuDNN ops |

### Graph backends included
| Path | Files | Contents |
|---|---:|---|
| `graph/hip/HipGraphReplayHandle.*` | 2 | HIP (AMD ROCm) graph replay |

### Merge order
Independent from PR08a/b/d, but llamacpp depends on PR07d (op helpers).

### Review focus
- LLaMA.cpp CUDA kernel FP16 handling (known bug history: fused_rope FP16 NaN)
- cuDNN descriptor lifecycle
- MIOpen API compatibility

### ADRs
- `ADR-LlamaCppBackend.md` (root-level, to be moved to ADRs/)

---

## PR08d: Experimental Backends + Graph Backends + Execution Infrastructure (~86 files)

Emerging/experimental backends, all remaining graph backends, and the
shared execution infrastructure (LaunchContext, AffinityManager, etc.).

### Files — Platform ops

| Path | Files | Contents |
|---|---:|---|
| `platform/mlir/` | 19 | MLIR JIT: op lowering to MLIR dialects |
| `platform/vlm/` | 14 | VLM-specific ops: vision encoder, embedding merger |
| `platform/pjrt/` | 10 | TPU via PJRT: SameDiff → XLA HLO IR |

### Files — Graph backends

| Path | Files | Contents |
|---|---:|---|
| `graph/cpu/` (remaining) | 13 | NnapiGraphBackend, MlirCpuGraphBackend, and other CPU graph backends |
| `graph/hexagon/` | 8 | Qualcomm Hexagon NPU graph backend |
| `graph/tpu/` | 8 | TPU graph backend |
| `graph/levelzero/` | 2 | Intel Level Zero (oneAPI) replay handle |
| `graph/vulkan/` | 2 | Vulkan compute replay handle |

### Files — Execution infrastructure

| Path | Files | Contents |
|---|---:|---|
| `execution/` | 40 | AffinityManager, LaunchContext, DataTransferManager, CudaGraphScheduler, MegaKernel*, ThreadPool, StreamManager |

### Merge order
Should merge AFTER PR08a-c (experimental backends may reference
patterns established in mature backends).

### Review focus
- MLIR dialect definitions — verify op coverage
- PJRT buffer management — XLA owns buffers, must coordinate with ND4J
- VLM ops — vision encoder correctness
- Execution infrastructure — thread safety, stream management

### ADRs
- `ADRs/0072 - TPU Backend.md`
- `ADRs/0057 - MLIR JIT Compilation Backend.md` (duplicate 0057, needs renumbering)
- `ADRs/0057 - ZLUDA Transpiler Support.md` (duplicate 0057, needs renumbering)
- `ADRs/0073 - Hexagon MLIR Backend.md` (duplicate 0073, needs renumbering)

---

## Recommended Merge Sequence

```
PR08a (ARM/Apple) ──┐
PR08b (Intel/x86) ──┼──→  PR08d (experimental + infra)
PR08c (CUDA/ROCm) ──┘
```

PR08a/b/c are independent and can merge in any order or in parallel.
PR08d (experimental + execution infra) should merge last.
