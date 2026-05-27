# PR09: DSP/Graph Execution (C++)

**Estimated files:** ~125 (graph/ minus generated/, gpu/, and platform graph backends)
**Merge layer:** 3
**Complexity:** High — core execution engine
**Reviewers:** Core C++ team, DSP specialists

## Description

The C++ DynamicShapePlan execution engine: NativeDynamicShapePlan (plan compilation,
execution, CUDA graph capture/replay), graph execution context (Variable, VariableSpace,
Context), DSP diagnostics/lifecycle, segment management, frozen plan execution,
slot execution, and control flow logic. This is the core of the DSP runtime.

## File Categories

### NativeDynamicShapePlan (~13 impl files)
- `libnd4j/include/graph/impl/NativeDynamicShapePlan.cpp` — core plan logic
- `libnd4j/include/graph/impl/NativeDynamicShapePlan_batchgemm.cu` — batched GEMM dispatch
- `libnd4j/include/graph/impl/NativeDynamicShapePlan_batchzero.cu` — batch zero optimization
- `libnd4j/include/graph/impl/NativeDynamicShapePlan_cublas.cu` — cuBLAS gap op execution
- `libnd4j/include/graph/impl/NativeDynamicShapePlan_cuda.cu` — CUDA device management
- `libnd4j/include/graph/impl/NativeDynamicShapePlan_cudagraph.cu` — CUDA graph capture/replay state machine (~1700 lines)
- `libnd4j/include/graph/impl/NativeDynamicShapePlan_cuda_stubs.cpp` — CPU-only stubs for CUDA functions
- `libnd4j/include/graph/impl/NativeDynamicShapePlan_gpubackend.cpp` — GPU backend dispatch (CPU side)
- `libnd4j/include/graph/impl/NativeDynamicShapePlan_gpubackend.cu` — GPU backend dispatch (CUDA side)
- `libnd4j/include/graph/impl/NativeDynamicShapePlan_prereplay.cu` — pre-replay argument refresh
- `libnd4j/include/graph/impl/NativeDynamicShapePlan_segments.cpp` — segment lifecycle management
- `libnd4j/include/graph/impl/NativeDynamicShapePlan_slotexec.cpp` — slot execution (CPU)
- `libnd4j/include/graph/impl/NativeDynamicShapePlan_slotexec_cuda.cu` — slot execution (CUDA)

### Graph execution infrastructure (~18 impl files)
- `libnd4j/include/graph/impl/Context.cpp`
- `libnd4j/include/graph/impl/ContextPrototype.cpp`
- `libnd4j/include/graph/impl/DeviceExecutionContext.cpp`
- `libnd4j/include/graph/impl/DeviceExecutionContext_cuda.cu`
- `libnd4j/include/graph/impl/ExecutionState.cpp`
- `libnd4j/include/graph/impl/ExecutionState_cuda.cu`
- `libnd4j/include/graph/impl/FlatUtils.cpp`
- `libnd4j/include/graph/impl/FlowPath.cpp`
- `libnd4j/include/graph/impl/FrozenPlan.cpp`
- `libnd4j/include/graph/impl/FusionPass.cpp`
- `libnd4j/include/graph/impl/Graph.cpp`
- `libnd4j/include/graph/impl/GraphAnalysisUtils.cpp`
- `libnd4j/include/graph/impl/GraphExecutioner.cpp`
- `libnd4j/include/graph/impl/GraphHolder.cpp`
- `libnd4j/include/graph/impl/GraphReplayFactory.cpp`
- `libnd4j/include/graph/impl/NativePlanCache.cpp`
- `libnd4j/include/graph/impl/NativePlanCompiler.cpp`
- `libnd4j/include/graph/impl/Node.cpp`

### Plan/segment/slot impl files (~9)
- `libnd4j/include/graph/impl/PlanDefinition.cpp`
- `libnd4j/include/graph/impl/PlanTopology.cpp`
- `libnd4j/include/graph/impl/ReplayCacheManager.cpp`
- `libnd4j/include/graph/impl/ResourceBinder.cpp`
- `libnd4j/include/graph/impl/SegmentExecutor.cpp`
- `libnd4j/include/graph/impl/SlotArray.cpp`
- `libnd4j/include/graph/impl/SlotBufferOwnership.cpp`
- `libnd4j/include/graph/impl/Stash.cpp`
- `libnd4j/include/graph/impl/VariableProxy.cpp`

### Graph readers (~4 impl files)
- `libnd4j/include/graph/impl/SdnbReader.cpp`
- `libnd4j/include/graph/impl/SdzReader.cpp`
- `libnd4j/include/graph/impl/Variable.cpp`
- `libnd4j/include/graph/impl/VariableSpace.cpp`

### DSP diagnostics (~2 impl files)
- `libnd4j/include/graph/impl/DspDiagnostics.cpp`

### Subgraph analysis (~1)
- `libnd4j/include/graph/analysis/ConvexSubgraphAnalyzer.cpp`
- `libnd4j/include/graph/analysis/ConvexSubgraphAnalyzer.h`

### Control flow logic (~22 files: 11 headers + 11 impls)
- `libnd4j/include/graph/execution/LogicConditional.h` + `impl/LogicConditional.cpp`
- `libnd4j/include/graph/execution/LogicEnter.h` + `impl/LogicEnter.cpp`
- `libnd4j/include/graph/execution/LogicExecutor.h` + `impl/LogicExecutor.cpp`
- `libnd4j/include/graph/execution/LogicExit.h`
- `libnd4j/include/graph/execution/LogicExpose.h` + `impl/LogicExpose.cpp`
- `libnd4j/include/graph/execution/LogicLoopCond.h` + `impl/LogicLoopCond.cpp`
- `libnd4j/include/graph/execution/LogicMerge.h` + `impl/LogicMerge.cpp`
- `libnd4j/include/graph/execution/LogicNextIteration.h` + `impl/LogicNextIteration.cpp`
- `libnd4j/include/graph/execution/LogicReturn.h` + `impl/LogicReturn.cpp`
- `libnd4j/include/graph/execution/LogicScope.h` + `impl/LogicScope.cpp`
- `libnd4j/include/graph/execution/LogicSwitch.h` + `impl/LogicSwitch.cpp`
- `libnd4j/include/graph/execution/LogicWhile.h` + `impl/LogicWhile.cpp`

### Exception handling (~2)
- `libnd4j/include/graph/exceptions/impl/unresolved_input_exception.cpp`
- `libnd4j/include/graph/exceptions/impl/unresolved_output_exception.cpp`

### Graph profiling (~4)
- `libnd4j/include/graph/profiling/OpTimingTracker.h`
- `libnd4j/include/graph/profiling/impl/GraphProfile.cpp`
- `libnd4j/include/graph/profiling/impl/GraphProfilingHelper.cpp`
- `libnd4j/include/graph/profiling/impl/OpTimingTracker.cpp`

### Headers (~35+)
- `libnd4j/include/graph/CaptureStateGuard.h`
- `libnd4j/include/graph/Context.h`
- `libnd4j/include/graph/ContextPrototype.h`
- `libnd4j/include/graph/DeviceExecutionContext.h`
- `libnd4j/include/graph/DspAnalysisUtils.h`
- `libnd4j/include/graph/DspConstants.h`
- `libnd4j/include/graph/DspDiagnostics.h`
- `libnd4j/include/graph/DspExecutionTrace.h`
- `libnd4j/include/graph/DspGraphTypes.h`
- `libnd4j/include/graph/DspHashUtils.h`
- `libnd4j/include/graph/DspLifecycleContext.h`
- `libnd4j/include/graph/DspPhaseUtils.h`
- `libnd4j/include/graph/DspSegmentLifecycle.h`
- `libnd4j/include/graph/DspStreamGuard.h`
- `libnd4j/include/graph/DspThreadState.h`
- `libnd4j/include/graph/DspVerifyUtils.h`
- `libnd4j/include/graph/ExecutionState.h`
- `libnd4j/include/graph/FlowPath.h`
- `libnd4j/include/graph/FrozenPlan.h`
- `libnd4j/include/graph/FusionPass.h`
- `libnd4j/include/graph/GraphAnalysisUtils.h`
- `libnd4j/include/graph/GraphBackendCommon.h`
- `libnd4j/include/graph/GraphBackend.h`
- `libnd4j/include/graph/GraphExecutioner.h`
- `libnd4j/include/graph/GraphHolder.h`
- `libnd4j/include/graph/GraphReplayHandle.h`
- `libnd4j/include/graph/LegacyOpTypeCodes.h`
- `libnd4j/include/graph/ModeContract.h`
- `libnd4j/include/graph/NativeDynamicShapePlan.h`
- `libnd4j/include/graph/NativePlanCache.h`
- `libnd4j/include/graph/NativePlanCompiler.h`
- `libnd4j/include/graph/OpContextLifecycleTracker.h`
- `libnd4j/include/graph/OpDetection.h`
- `libnd4j/include/graph/PlanDefinition.h`
- `libnd4j/include/graph/PlanExecutionContext.h`
- `libnd4j/include/graph/PlanTopology.h`
- `libnd4j/include/graph/ReplayCacheManager.h`
- `libnd4j/include/graph/ResourceBinder.h`
- `libnd4j/include/graph/ResultWrapper.h`
- `libnd4j/include/graph/SdnbReader.h`
- `libnd4j/include/graph/SdzReader.h`
- `libnd4j/include/graph/SegmentAnalysisTypes.h`
- `libnd4j/include/graph/SegmentExecutor.h`
- `libnd4j/include/graph/SlotArray.h`
- `libnd4j/include/graph/SlotBufferOwnership.h`
- `libnd4j/include/graph/Variable.h`
- `libnd4j/include/graph/VariableProxy.h`
- `libnd4j/include/graph/VariableSpace.h`

### ADRs (8 — only those actually changed in the diff)
- `ADRs/0061 - DynamicShapePlan Execution.md` — Core DSP architecture: compile → warmup → freeze → replay lifecycle
- `ADRs/0078 - DSP Diagnostic Framework Extensions.md` — STREAM_SYNC, MULTI_DEVICE, GRAPH_REPLAY diagnostic categories
- `ADRs/0079 - NativeDynamicShapePlan Structural Refactoring.md` — Decomposed 18K-line impl into immutable definition vs. mutable state
- `ADRs/0080 - Triton Fusion Replay Correctness and Accuracy Validation.md` — Fixed stale pinned copies, op trait misclassification, GELU mismatch
- `ADRs/0081 - DSP View Shape Correctness and Execution Comparison Diagnostics.md` — Fixed view builder ignoring ONNX permutation inputs
- `ADRs/0082 - CUDA Graph Replay Pointer Stability and Frozen Steady-State.md` — Fixed argTableStable, external input pollution, frozen-constant detection
- `ADRs/0083 - Thread-Local Cast Cache Leak Prevention.md` — Fixed ~250 MB/step GPU leak in cuBLAS cast cache
- `ADRs/0084 - DSP Execution State Simplification.md` — Removed redundant ExecutionPhase enum, pruned dead SlotState values

### Root-level ADRs (2 — to be moved into ADRs/)
- `ADR-CudaGraphReplay.md` — Full capture/replay lifecycle with Triton sub-kernels and gap ops
- `ADR-DeviceTransferManagement.md` — Five-priority multi-GPU memory management framework

## Review Focus

- NativeDynamicShapePlan_cudagraph.cu — CUDA graph capture/replay state machine, verify pointer stability
- NativeDynamicShapePlan_segments.cpp — segment lifecycle management
- NativeDynamicShapePlan_slotexec*.cpp — slot execution on CPU and CUDA
- ModeContract.h — execution mode dispatch logic (struct-based SSOT)
- CaptureStateGuard.h — RAII guard for capture state transitions
- SegmentExecutor — verify no stale data paths between segments
