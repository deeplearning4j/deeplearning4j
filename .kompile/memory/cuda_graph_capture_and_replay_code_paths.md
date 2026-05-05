---
name: CUDA graph capture and replay code paths
description: Key file locations for monolithic/composite CUDA graph capture, replay, and optimization
type: reference
---

## CUDA Graph Capture/Replay Code Paths

### Monolithic Capture (NATIVE_ONLY_CAPTURE)
- **Entry**: `libnd4j/include/graph/impl/NativeDynamicShapePlan_gpubackend.cu:3048` — beginCapture
- **Slot loop**: lines 3112-3121 — iterates ALL slots via executeSlot()
- **Node count check**: line 3134 — cudaGraphGetNodes after capture
- **End capture**: line 3189 — handle->endCapture()

### Monolithic Replay (frozen fast path)
- **Entry**: `libnd4j/include/graph/impl/NativeDynamicShapePlan_cuda.cu:155` — platformTryFrozenFastPath()
- **Decision**: line 202-205 — fastPathApplicable checks replayHandle ready
- **Replay**: line 376 — seg.exec.replayHandle->replay(stream)
- **Composite**: line 438 — compositeReplay() when monolithic not available

### Composite Replay
- **Main function**: `libnd4j/include/graph/impl/NativeDynamicShapePlan_gpubackend.cu` — compositeReplay()
- **Gap prezero**: lines 1034-1065 — zeroes gap slot outputs before replay
- **Arg table refresh**: lines 996-1021 — TritonGraphBackend::refreshArgTablesForReplay

### executeSlot (per-op execution during capture)
- **Function**: `libnd4j/include/graph/impl/NativeDynamicShapePlan_slotexec.cpp:1516`
- **Frozen constant skip**: line 2119 — skips when shapesFrozen_ && executeCount_ >= 2
- **View handling**: line 1662-1735 — tryCreateViewForSlot for view-capable ops
- **Frozen fast path**: line 1631-1640 — gate conditions

### Key Op Implementations
- **reshape_no_copy**: `libnd4j/include/ops/declarable/generic/shape/reshape_no_copy.cpp`
  - Line 43: calls output->assign(input) when buffers differ — THIS PRODUCES GRAPH NODES
  - Line 27: identity reshape returns OK (no node)
  
- **Op traits**: `libnd4j/include/ops/impl/OpTraitTable.cpp:286`
  - reshape_no_copy: VIEW | DATADEP (prevents freezing)

### Autoregressive Decode Loop
- **File**: `libnd4j/include/ops/declarable/helpers/cuda/autoregressive_decode.cu`
- **Graph replay**: line 574 — plan->executeSteadyState()
- **KV scatter**: lines 817-867 — POST-graph, copies present KV to static buffers
- **Stream sync**: line 787 — cudaStreamSynchronize (primary serialization point)
- **planOwnsKvScatter gate**: line 817 — skips manual scatter when plan handles it

### GenerationPipeline
- **cachePositionExtIdx**: `nd4j/samediff-llm/.../GenerationPipeline.java:1954` — set to -1 for VLM/ONNX (no in-place KV write)
