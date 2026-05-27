# PR10: Triton/NVRTC/PTX Backend (C++)

**Estimated files:** ~47
**Merge layer:** 3 (depends on PR09 DSP graph)
**Complexity:** High
**Reviewers:** CUDA/Triton specialists

## Description

Triton graph backend (JIT kernel compilation, IR builder, section fusion, cache),
NVRTC/PTX graph backends, GPU kernel launcher, capture buffer registry,
fusion scoring, and op category tables. This is the Triton compilation and
dispatch pipeline that sits on top of the DSP execution engine.

## Files (47)

### Triton graph backend (~22)
- `libnd4j/include/graph/gpu/TritonGraphBackend.cpp`
- `libnd4j/include/graph/gpu/TritonGraphBackend.h`
- `libnd4j/include/graph/gpu/TritonGraphBackend_binary.cpp`
- `libnd4j/include/graph/gpu/TritonGraphBackend_cache.cpp`
- `libnd4j/include/graph/gpu/TritonGraphBackend_compile.cu` (CUDA, not .cpp)
- `libnd4j/include/graph/gpu/TritonGraphBackend_execute.cu` (CUDA, not .cpp)
- `libnd4j/include/graph/gpu/TritonGraphBackend_internal.h` (header, not .cpp)
- `libnd4j/include/graph/gpu/TritonGraphBackend_kernel.cu` (CUDA, not .cpp)
- `libnd4j/include/graph/gpu/TritonGraphBackend_lru.cpp`
- `libnd4j/include/graph/gpu/TritonGraphBackend_preload.cpp`
- `libnd4j/include/graph/gpu/TritonCacheBundle.cpp`

### Triton IR builder (~15)
- `libnd4j/include/graph/gpu/TritonIRBuilder.cpp`
- `libnd4j/include/graph/gpu/TritonIRBuilder.h`
- `libnd4j/include/graph/gpu/TritonIRBuilder_analysis.cpp`
- `libnd4j/include/graph/gpu/TritonIRBuilder_cuda.cu` (CUDA, not .cpp)
- `libnd4j/include/graph/gpu/TritonIRBuilder_emitters.cpp`
- `libnd4j/include/graph/gpu/TritonIRBuilder_internal.h` (header, not .cpp)
- `libnd4j/include/graph/gpu/TritonIRBuilder_kernels.cpp`
- `libnd4j/include/graph/gpu/TritonIRBuilder_module.cpp`
- `libnd4j/include/graph/gpu/TritonIRBuilder_sections.cpp`
- `libnd4j/include/graph/gpu/TritonIRBuilder_types.h` (header, not .cpp)
- `libnd4j/include/graph/gpu/TritonIRBuilder_types.cpp`

### Triton dispatch
- `libnd4j/include/graph/gpu/TritonTargetDispatch.cpp`
- `libnd4j/include/graph/gpu/TritonTargetDispatch.h`

### NVRTC backend
- `libnd4j/include/graph/gpu/NvrtcGraphBackend.cu`
- `libnd4j/include/graph/gpu/NvrtcGraphBackend.h`
- `libnd4j/include/graph/gpu/NvrtcKernelBuilder.cu`
- `libnd4j/include/graph/gpu/NvrtcKernelBuilder.h`
- `libnd4j/include/graph/gpu/NvrtcKernelCache.cu`
- `libnd4j/include/graph/gpu/NvrtcKernelCache.h`

### PTX backend
- `libnd4j/include/graph/gpu/PtxGraphBackend.cu`
- `libnd4j/include/graph/gpu/PtxGraphBackend.h`

### GPU kernel launcher
- `libnd4j/include/graph/gpu/GpuKernelLauncher.cu`
- `libnd4j/include/graph/gpu/GpuKernelLauncher.h`

### Fusion infrastructure
- `libnd4j/include/graph/FusionPass.h`
- `libnd4j/include/graph/impl/FusionPass.cpp`
- `libnd4j/include/graph/gpu/FusionScoring.cpp`
- `libnd4j/include/graph/gpu/FusionScoring.h`
- `libnd4j/include/graph/gpu/CaptureBufferRegistry.cu`
- `libnd4j/include/graph/gpu/CaptureBufferRegistry.h`
- `libnd4j/include/graph/gpu/JitGraphBackendCommon.cu`
- `libnd4j/include/graph/gpu/JitGraphBackendCommon.h`

### Op category/config tables
- `libnd4j/include/graph/gpu/OpCategoryTable.h`
- `libnd4j/include/graph/gpu/SectionTypeConfig.h`
- `libnd4j/include/graph/gpu/ViewRecipe.h`
- `libnd4j/include/graph/gpu/SymbolicShapeRanges.h`
- `libnd4j/include/graph/cpu/SymbolicShapeRanges.cpp`

### ADR (1)
- `ADRs/0071 - Triton Graph Backend.md` — OpenAI Triton as kernel fusion backend for DSP, OpTraitTable.cpp as SSOT for mappability

## Review Focus

- TritonIRBuilder — kernel code generation correctness
- TritonGraphBackend_cache/lru — cache eviction and stale key prevention
- FusionPass — section fusion logic must not break accuracy
- OpCategoryTable — op classification affects dispatch decisions
