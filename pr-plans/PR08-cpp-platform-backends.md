# PR08: C++ Platform Backends

**Estimated files:** ~467 (382 platform + 45 graph backends + 40 execution infra)
**Merge layer:** 3
**Complexity:** Medium (volume; each backend is relatively independent)
**Reviewers:** Platform-specific reviewers

## Description

Platform-specific op implementations: ARM Compute Library, oneDNN/MKL,
LLaMA.cpp, cuDNN, Apple Accelerate, MLIR, Metal (MPS), VLM ops,
PJRT (TPU), MIOpen (ROCm), plus graph backend implementations for
each platform.

## Sub-Split Plan → [`PR08-sub-split.md`](PR08-sub-split.md)

Recommended 4-way split (verified against actual diff):

| Sub-PR | Name | Files |
|---|---|---:|
| PR08a | ARM + Apple Backends | ~173 |
| PR08b | Intel/x86 Backends | ~72 |
| PR08c | CUDA Ecosystem Backends | ~96 |
| PR08d | Experimental Backends + Graph Backends | ~86 |

Merge order: PR08a/b/c (parallel) → PR08d

## File Categories

### ARM Compute Library (~125 files)
- `libnd4j/include/ops/declarable/platform/armcompute/*.cpp`
- `libnd4j/include/ops/declarable/platform/armcompute/*.h`
- Ops: avgpool, batchnorm, col2im, conv2d, deconv2d, depthwiseconv2d, gemm, im2col, lrn, matmul, maxpool, etc.

### oneDNN/MKL (~72 files)
- `libnd4j/include/ops/declarable/platform/mkldnn/*.cpp`
- `libnd4j/include/ops/declarable/platform/mkldnn/*.h`
- Ops: batchnorm, conv2d, conv3d, deconv, gru, lstm, matmul, pool2d, pool3d, softmax, tanh, etc.

### LLaMA.cpp (~59 files)
- `libnd4j/include/ops/declarable/platform/llamacpp/*.cpp`
- `libnd4j/include/ops/declarable/platform/llamacpp/*.h`
- `libnd4j/include/ops/declarable/platform/llamacpp/cuda/*.cu`
- Ops: attention, gated_delta_rule, matmul, rmsnorm, rope, etc.

### cuDNN (~30 files)
- `libnd4j/include/ops/declarable/platform/cudnn/*.cu`
- `libnd4j/include/ops/declarable/platform/cudnn/*.h`
- Ops: batchnorm, conv2d, conv3d, ctcloss, dropout, pool2d, pool3d, etc.

### Apple Accelerate (~28 files)
- `libnd4j/include/ops/declarable/platform/accelerate/*.cpp`
- `libnd4j/include/ops/declarable/platform/accelerate/*.h`
- Ops: avgpool, conv2d, gemm, maxpool, softmax, etc.

### Metal/MPS (~20 files)
- `libnd4j/include/ops/declarable/platform/mps/*.mm`
- `libnd4j/include/ops/declarable/platform/mps/*.h`

### MLIR (~18 files)
- `libnd4j/include/ops/declarable/platform/mlir/*.cpp`
- `libnd4j/include/ops/declarable/platform/mlir/*.h`
- `libnd4j/include/mlir/` — MLIR dialect files

### VLM ops (~12 files)
- `libnd4j/include/ops/declarable/platform/vlm/*.cpp`
- `libnd4j/include/ops/declarable/platform/vlm/cuda/*.cu`

### PJRT/TPU (~10 files)
- `libnd4j/include/ops/declarable/platform/pjrt/*.cpp`
- `libnd4j/include/ops/declarable/platform/pjrt/*.h`

### MIOpen/ROCm (~5 files)
- `libnd4j/include/ops/declarable/platform/miopen/*.cpp`

### Graph backends (~45 files)
- `libnd4j/include/graph/cpu/` (21 files) — AclGraphBackend, ArmHybridGraphBackend,
  CpuIRBuilder, FunctionalReplayHandle, MlirCpuGraphBackend, MlxGraphBackend,
  MlxIRBuilder, NnapiGraphBackend, OneDnnGraphBackend, OpenVinoGraphBackend,
  SymbolicShapeRanges
- `libnd4j/include/graph/hexagon/` (8 files) — HexagonGraphBackend, HexagonIRBuilder,
  HexagonReplayHandle, HexagonRuntimeManager
- `libnd4j/include/graph/tpu/` (8 files) — HloIRBuilder, PjrtClientManager,
  TpuGraphBackend, TpuReplayHandle
- `libnd4j/include/graph/cuda/` (2 files) — CudaGraphReplayHandle.cu/.h
- `libnd4j/include/graph/hip/` (2 files) — HipGraphReplayHandle
- `libnd4j/include/graph/levelzero/` (2 files) — LevelZeroReplayHandle
- `libnd4j/include/graph/metal/` (2 files) — MetalReplayHandle
- `libnd4j/include/graph/vulkan/` (2 files) — VulkanReplayHandle

### Execution infrastructure (~40 files)
- `libnd4j/include/execution/` — AffinityManager, LaunchContext, DataTransferManager,
  CudaGraphScheduler, StreamManager, ThreadPool, ModelParallel, ZludaRuntime,
  CallableInterface, PipelineScheduler, TensorPartition
- `libnd4j/include/execution/megakernel/` — BarrierManager, MegaKernelInterpreter,
  MegaKernelScheduler, MegaKernelTask, MegaKernelTaskBuilder

### ADRs (7 — only those actually changed in the diff; note: duplicate numbers not yet renumbered)
- `ADRs/0058 - Multi-Backend Kernel Selection and Management.md` — Multi-level kernel selection with runtime benchmarking
- `ADRs/0059 - Multi-Backend Op Execution System.md` — Runtime multi-backend loading with automatic op routing by device
- `ADRs/0072 - TPU Backend.md` — TPU backend via PJRT API mapping SameDiff to XLA HLO IR
- `ADRs/0057 - MLIR JIT Compilation Backend.md` — MLIR JIT for graph-level fusion and cross-platform codegen (duplicate 0057, needs renumbering)
- `ADRs/0057 - ZLUDA Transpiler Support.md` — ZLUDA runtime transpiler to run CUDA on AMD/Intel GPUs (duplicate 0057, needs renumbering)
- `ADRs/0073 - Hexagon MLIR Backend.md` — Qualcomm Hexagon NPU backend for INT8/INT16 mobile inference (duplicate 0073, needs renumbering)
- `ADR-LlamaCppBackend.md` (root-level, to be moved to ADRs/) — Backend classifier profiles for LLaMA.cpp, oneDNN, cuDNN
