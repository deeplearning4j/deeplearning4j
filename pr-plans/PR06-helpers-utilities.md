# PR06: Helpers & Utilities (C++)

**Estimated files:** ~87
**Merge layer:** 2
**Complexity:** Medium
**Reviewers:** Core C++ team

## Description

Shared C++ helper classes used across ops: MmulHelper (matrix multiply),
ShapeUtils, FlashAttention, CUTLASS GEMM, kernel auto-tuner, constant helpers,
BLAS helpers, debugging utilities, and linear algebra helpers (SVD, Hessenberg, etc.).

## Files (87)

### Core computational helpers
- `libnd4j/include/helpers/MmulHelper.h`
- `libnd4j/include/helpers/cpu/MmulHelper.cpp`
- `libnd4j/include/helpers/cuda/MmulHelper.cu`
- `libnd4j/include/helpers/impl/MmulHelper.cpp`
- `libnd4j/include/helpers/FlashAttentionHelper.h`
- `libnd4j/include/helpers/cuda/FlashAttentionHelper.cu`
- `libnd4j/include/helpers/impl/FlashAttentionHelper.cpp`
- `libnd4j/include/helpers/CutlassGemmHelper.h`
- `libnd4j/include/helpers/cuda/CutlassGemmHelper.cu`
- `libnd4j/include/helpers/impl/CutlassGemmHelper.cpp`
- `libnd4j/include/helpers/CutlassHelper.h`
- `libnd4j/include/helpers/cuda/CutlassHelper.cu`
- `libnd4j/include/helpers/impl/CutlassHelper.cpp`
- `libnd4j/include/helpers/AttentionWorkspace.h`
- `libnd4j/include/helpers/impl/AttentionHelper.cpp`
- `libnd4j/include/helpers/impl/AttentionWorkspace.cpp`

### Shape utilities
- `libnd4j/include/helpers/ShapeUtils.h`
- `libnd4j/include/helpers/impl/ShapeUtils.cpp`
- `libnd4j/include/helpers/ShapeBuilders.h`
- `libnd4j/include/helpers/impl/ShapeBuilders.cpp`
- `libnd4j/include/helpers/shape.h`
- `libnd4j/include/helpers/reshapeNoCopy.h`
- `libnd4j/include/helpers/DirectShapeTrie.h`
- `libnd4j/include/helpers/impl/DirectShapeTrie.cpp`
- `libnd4j/include/helpers/DirectTadTrie.h`
- `libnd4j/include/helpers/impl/DirectTadTrie.cpp`

### BLAS/cuBLAS helpers
- `libnd4j/include/helpers/BlasHelper.h`
- `libnd4j/include/helpers/impl/BlasHelper.cpp`
- `libnd4j/include/helpers/cublasHelper.h`
- `libnd4j/include/helpers/cpu/cublasHelper.cpp`
- `libnd4j/include/helpers/cuda/cublasHelper.cu`
- `libnd4j/include/helpers/MklBlasHelper.h`
- `libnd4j/include/helpers/MklVmlHelper.h`

### Kernel selection & auto-tuning
- `libnd4j/include/helpers/KernelAutoTuner.h`
- `libnd4j/include/helpers/impl/KernelAutoTuner.cpp`
- `libnd4j/include/helpers/KernelPerformanceRegistry.h`
- `libnd4j/include/helpers/impl/KernelPerformanceRegistry.cpp`
- `libnd4j/include/helpers/KernelSelectionEnvironment.h`
- `libnd4j/include/helpers/impl/KernelSelectionEnvironment.cpp`
- `libnd4j/include/helpers/KernelPluginTemplate.h`
- `libnd4j/include/helpers/DynamicKernelLoader.h`
- `libnd4j/include/helpers/impl/DynamicKernelLoader.cpp`
- `libnd4j/include/helpers/HelperVersionRegistry.h`
- `libnd4j/include/helpers/impl/HelperVersionRegistry.cpp`

### Constant helpers
- `libnd4j/include/helpers/ConstantHelper.h`
- `libnd4j/include/helpers/cpu/ConstantHelper.cpp`
- `libnd4j/include/helpers/cuda/ConstantHelper.cu`
- `libnd4j/include/helpers/ConstantShapeHelper.h`
- `libnd4j/include/helpers/impl/ConstantShapeHelper.cpp`
- `libnd4j/include/helpers/impl/ConstantTadHelper.cpp`

### Shape buffer creators
- `libnd4j/include/helpers/cpu/CpuShapeBufferCreator.cpp`
- `libnd4j/include/helpers/cuda/CudaShapeBufferCreator.cu`

### Debug/logging/transfer
- `libnd4j/include/helpers/DebugHelper.h`
- `libnd4j/include/helpers/DebugInfo.h`
- `libnd4j/include/helpers/logger.h`
- `libnd4j/include/helpers/impl/logger.cpp`
- `libnd4j/include/helpers/TransferMetrics.h`
- `libnd4j/include/helpers/impl/TransferMetrics.cpp`
- `libnd4j/include/helpers/PointersManager.h`
- `libnd4j/include/helpers/cuda/PointersManager.cu`

### CUDA utilities
- `libnd4j/include/helpers/impl/CudaLaunchHelper.cpp`
- `libnd4j/include/helpers/cuda/OpTimingTracker_cuda.cu`

### TAD calculators
- `libnd4j/include/helpers/cpu/TadCalculator.cpp`
- `libnd4j/include/helpers/cuda/TadCalculator.cu`

### Reduction loop helpers
- `libnd4j/include/helpers/cpu/loops/IndexReductionLoops.hpp`
- `libnd4j/include/helpers/cpu/loops/Reduction3Loops.hpp`
- `libnd4j/include/helpers/cpu/loops/ReductionLoops.hpp`
- `libnd4j/include/helpers/cpu/loops/ReductionLoops_bool.cpp`
- `libnd4j/include/helpers/cpu/loops/ReductionLoops_float.hpp`
- `libnd4j/include/helpers/cpu/loops/ReductionLoops_long.cpp`
- `libnd4j/include/helpers/cpu/loops/ReductionLoops_same.cpp`

### Linear algebra helpers
- `libnd4j/include/helpers/HessenbergAndSchur.h`
- `libnd4j/include/helpers/impl/HessenbergAndSchur.cpp`
- `libnd4j/include/helpers/hhColPivQR.h`
- `libnd4j/include/helpers/impl/hhColPivQR.cpp`
- `libnd4j/include/helpers/hhSequence.h`
- `libnd4j/include/helpers/impl/hhSequence.cpp`
- `libnd4j/include/helpers/impl/EigenValsAndVecs.cpp`
- `libnd4j/include/helpers/impl/FullPivLU.cpp`
- `libnd4j/include/helpers/impl/jacobiSVD.cpp`
- `libnd4j/include/helpers/impl/Sqrtm.cpp`
- `libnd4j/include/helpers/cpu/svd.cpp`

### Miscellaneous
- `libnd4j/include/helpers/BitwiseUtils.h`
- `libnd4j/include/helpers/LoopKind.h`
- `libnd4j/include/helpers/Loops.h`
- `libnd4j/include/helpers/mman.h`
- `libnd4j/include/helpers/impl/OmpLaunchHelper.cpp`

### ADRs (2 — only those actually changed in the diff)
- `ADRs/0055-Kernel_Selection_And_Dynamic_Loading.md` — Runtime auto-tuning, persistent performance caching, dynamic shared-library plugin loading
- `ADR-OpTimingTracker.md` (root-level, to be moved to ADRs/) — Lock-free ring-buffer op timing with phase-level granularity, Chrome Trace/CSV export
