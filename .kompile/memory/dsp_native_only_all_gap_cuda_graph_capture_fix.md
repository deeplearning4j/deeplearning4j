---
name: DSP native-only all-gap CUDA graph capture fix
description: All-gap DSP segments in AUTO/TRITON must be captured as native-only monolithic CUDA graphs, not zero-node/slot-by-slot or composite gap replay.
type: project
---

2026-05-27: Fixed native-only all-gap DSP capture path. In NativeDynamicShapePlan_gpubackend.cu, all-gap AUTO/TRITON segments now set nativeOnlyGraphCapture, skip empty Triton capture, execute native slots on the CUDA capture stream, tag compiledByBackend as CUDA, and avoid building a composite replay schedule. In NativeDynamicShapePlan_cuda.cu, frozen fast path now treats CUDA-tagged monolithic captures with gap slots as valid monolithic replay handles instead of routing/demoting to composite gap execution. Target test passed: DspConcurrentPlanSharingTest#testPoolThreadGpuHeavyGraph on nd4j-cuda-12.9. Diagnostics: ALL_GAPS_NATIVE_CAPTURE, NATIVE_ONLY_CAPTURE captured 5 graph nodes, GRAPH CAPTURE COMPLETE, READY (monolithic handle), repeated REPLAY_SUCCESS, graphReplays advanced; no ZERO_KERNEL_SBS/captureProducedNoKernels/composite gap replay for the captured native-only path.
