# ADR: Triton Graph Backend

## Status

Implemented

Proposed by: Adam Gibson (January 2025)

Discussed with: Development Team

## Context

NativeDynamicShapePlan executes SameDiff graphs as sequences of individual op dispatches — one kernel launch per op. For a vision encoder with 1962 ops, this means 1962 separate kernel launches, each with ~5μs overhead plus intermediate global memory traffic between ops. While CUDA Graphs can capture and replay these launch sequences (eliminating launch overhead), they still execute each kernel separately — intermediate results are written to global memory and read back for the next kernel.

True kernel fusion — combining multiple ops into a single kernel where intermediate values stay in registers or shared memory — can deliver 2-5x speedups over CUDA Graphs for fusible segments. This is particularly impactful for the element-wise chains that dominate attention and normalization layers:

```
Unfused (3 kernels, 2 intermediate buffers):
  add(x, y) → global_mem → relu(result) → global_mem → mul(result, z)

Fused (1 kernel, 0 intermediate buffers):
  add_relu_mul(x, y, z) — intermediates stay in registers
```

OpenAI's Triton compiler provides an MLIR-based infrastructure for generating fused GPU kernels from high-level IR. Triton supports multiple GPU targets (NVIDIA via PTX, AMD via AMDGCN, Intel via SPIR-V) through a single compilation pipeline, making it a natural choice for LibND4J's multi-backend architecture.

### Why Not Just CUDA Graphs?

CUDA Graphs capture a recording of kernel launches and replay them. This eliminates launch overhead but not memory traffic:

| Metric | Slot-by-Slot | CUDA Graphs | Triton Fusion |
|--------|-------------|-------------|---------------|
| Kernel launches | N | 1 (replay) | 1 (fused) |
| Intermediate memory stores | N-1 | N-1 | 0 |
| Memory bandwidth | O(N) | O(N) | O(1) per segment |
| GPU occupancy | Low (launch gaps) | High | Highest |

For element-wise chains (add→relu→mul→...), Triton fusion eliminates all intermediate global memory traffic, achieving near-peak memory bandwidth utilization.

## Decision

We implement a Triton-based graph backend that compiles fusible segments of NativeDynamicShapePlan into optimized GPU kernels, with multi-target support for NVIDIA, AMD, and Intel GPUs.

### Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                    NativeDynamicShapePlan                            │
│                                                                     │
│  Segment detection: buildSegments() identifies fusible op ranges    │
│                                                                     │
│  Execution priority:                                                │
│    1. TritonGraphBackend  → fused kernel (fusible segments)         │
│    2. CudaGraphBackend    → captured replay (non-fusible segments)  │
│    3. Slot-by-slot        → individual ops (always-works fallback)  │
└──────────────────────────────┬─────────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────────┐
│                    TritonGraphBackend                                │
│                    (implements GraphBackend)                         │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Compilation Cache                                             │  │
│  │  Key: {startSlot, endSlot, shapeKey}                          │  │
│  │  Value: CompiledKernel (GPU module, function, launch config)  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  canFuseSegment() → checks ≥2 mappable ops, ≥50% fusion fraction  │
│  compileSegment() → IR build + compile + load + cache              │
│  executeSegment() → launch cached kernel                           │
└──────────────────────────────┬─────────────────────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                 ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────────┐
│ TritonIRBuilder  │ │ TritonTarget     │ │ GPU Driver Module    │
│                  │ │ Dispatch         │ │                      │
│ NativeSlots →    │ │ MLIR → Binary:   │ │ Binary → Module:     │
│ Triton MLIR IR   │ │  NVIDIA: PTX     │ │  cuModuleLoadData    │
│                  │ │  AMD: AMDGCN     │ │  hipModuleLoadData   │
│ SSA value sharing│ │  Intel: SPIR-V   │ │  zeModuleCreate      │
│ No intermediate  │ │                  │ │                      │
│ global stores    │ │ Binary → Launch: │ │ Kernel → Launch:     │
│                  │ │  cuLaunchKernel  │ │  grid, block, args   │
│                  │ │  hipModuleLaunch │ │                      │
│                  │ │  zeCommandList   │ │                      │
└──────────────────┘ └──────────────────┘ └──────────────────────┘
```

### GraphBackend Interface

All graph backends (Triton, CUDA Graphs, oneDNN Graph, ACL Dynamic Fusion) implement the same abstract interface:

```cpp
class GraphBackend {
public:
    virtual bool isAvailable() const = 0;
    virtual bool canFuseSegment(NativeSlot* slots, int start, int end) = 0;
    virtual bool compileSegment(GraphSegment& seg, NativeSlot* slots,
                                NDArray** externalInputs, int numExternalInputs,
                                NDArray** outputSlots, int totalOutputSlots,
                                LongType shapeKey) = 0;
    virtual Status executeSegment(GraphSegment& seg, NativeSlot* slots,
                                  NDArray** externalInputs, int numExternalInputs,
                                  NDArray** outputSlots, int totalOutputSlots,
                                  void* stream) = 0;
    virtual void invalidateCache() = 0;
    virtual const char* name() const = 0;
    virtual std::vector<CompilationAuditEntry> getLastCompilationAudit() const = 0;
};
```

`CompilationAuditEntry` tracks per-op compilation status, identifying which ops were successfully compiled vs. skipped (skipped ops produce stale outputs on graph replay).

### Triton IR Builder

`TritonIRBuilder` constructs Triton MLIR IR (TTIR) from sequences of NativeSlots. The key advantage over CUDA Graphs is SSA value sharing — fused ops share intermediate values in the IR without global memory stores.

**Supported Op Categories (~40 ops)**:

| Category | Ops | Triton IR |
|----------|-----|-----------|
| Binary Elementwise | add, subtract, multiply, divide, minimum, maximum | arith.addf, arith.subf, arith.mulf, arith.divf, arith.minimumf, arith.maximumf |
| Unary Elementwise | relu, sigmoid, tanh, gelu, exp, log, abs, sqrt, square, pow, clamp | arith.maximumf(x,0), math.exp + arith patterns, math.tanh, math.exp, math.log, math.absf, math.sqrt |
| MatMul | matmul, mmul, batch_matmul | tt.dot |
| Reduction | reduce_sum, reduce_max, reduce_min, reduce_mean, reduce_prod | tt.reduce |
| Normalization | softmax, log_softmax, layer_norm | Compound multi-op patterns |
| Cast | type cast | arith.extf, arith.truncf, arith.sitofp, etc. |

**IR Construction Pipeline**:

```
1. Register Triton dialects (triton, arith, math)
2. Identify unique buffer references crossing segment boundary
3. Build function signature: tt.ptr<dtype> for each buffer + n_elements I32
4. Create entry block:
   a. tt.get_program_id(0) → thread indexing
   b. tt.load for each input
   c. Map each NativeSlot to Triton IR ops (SSA value sharing)
   d. tt.store for outputs
5. Return TritonIRModule with MLIR handle + kernel metadata
```

**Tile Configuration Selection**:

| Segment Type | BLOCK_SIZE | numWarps | numStages | Grid |
|-------------|-----------|----------|-----------|------|
| Element-wise only | 1024 | 4 | 3 | 1D: ceil(N/1024) |
| MatMul-dominant | 128 (M,N,K=32) | 8 | 3 | 2D: ceil(M/128) × ceil(N/128) |
| Reduction-dominant | 1024 | 4 | 2 | 1D: ceil(N/1024) |

### Multi-Target Dispatch

`TritonTargetDispatch` handles GPU detection, compilation, and kernel launching for three GPU vendors:

**Target Detection Priority**:
1. **HIP** (AMD): preferred because `hipDeviceProp_t.gcnArchName` gives exact architecture (e.g., "gfx1100")
2. **Level Zero** (Intel): preferred for Intel GPUs (Ponte Vecchio "pvc", Alchemist "xehpg")
3. **CUDA** (NVIDIA): for NVIDIA GPUs (e.g., "sm_89" from compute capability)

**Compilation Pipeline**:

```
TTIR (Triton IR)
  → TTGIR (Triton GPU IR) — tile mapping, shared memory planning
    → LLVM IR — standard LLVM optimizations
      → Target ISA:
         NVIDIA: PTX assembly (loaded via cuModuleLoadDataEx)
         AMD: AMDGCN ELF binary (loaded via hipModuleLoadData)
         Intel: SPIR-V binary (loaded via zeModuleCreate)
```

**ZLUDA Compatibility**: Under ZLUDA (which intercepts CUDA API calls to target AMD/Intel hardware), Triton bypasses ZLUDA's interception entirely. It detects the actual GPU hardware via HIP or Level Zero and produces native binaries (AMDGCN or SPIR-V) instead of PTX. ZLUDA's `cuModuleLoadDataEx` expects PTX, so Triton loads its binaries through the native driver API directly.

### Segment Selection Criteria

`canFuseSegment()` applies conservative criteria:
- Minimum 1 Triton-mappable op in the segment (`MIN_MAPPABLE_OPS = 1`)
- At least 50% of segment ops are Triton-mappable
- All ops in a fusion group must be mappable (no fallback within fused kernel)
- Non-mappable ops break the segment, creating separate fusion groups

### Shape-Aware Caching

Compiled kernels are cached by `{startSlot, endSlot, shapeKey}`:

```cpp
struct CacheKey {
    int startSlot;
    int endSlot;
    LongType shapeKey;  // Hash of input/output shapes
};
std::unordered_map<CacheKey, CompiledKernel> compiledKernels_;
```

When shapes change (e.g., growing KV cache in autoregressive generation), the shape key changes, triggering recompilation. Previous compiled kernels are retained in cache for the case where shapes cycle (e.g., batch dimension alternating between prefill and decode).

### Build Configuration

Triton support is optional and controlled by CMake flags:

```cmake
# Enable Triton backend
cmake -DHELPERS_triton=ON -Dlibnd4j.chip=cuda ..

# FindTriton.cmake searches:
#   TRITON_ROOT environment variable
#   ${CMAKE_BINARY_DIR}/triton_install (ExternalProject)
#   /usr/local, /usr

# Backend-specific libraries detected automatically:
#   TritonNVIDIAGPU (NVIDIA)
#   TritonAMDGPU (AMD)
#   TritonIntelGPU (Intel)
```

All Triton code is guarded by `#if HAVE_TRITON ... #endif`. When Triton is not available, NativeDynamicShapePlan falls back to CUDA Graphs or slot-by-slot execution transparently.

## Consequences

### Advantages

**Kernel Fusion**: Eliminates intermediate global memory stores between fused ops. For element-wise chains (add→relu→mul), this achieves near-peak memory bandwidth vs. 3x bandwidth waste with separate kernels.

**Multi-GPU Vendor Support**: Single codebase compiles to NVIDIA PTX, AMD AMDGCN, and Intel SPIR-V. This is the only fusion path that works on all three GPU vendors without per-vendor kernel implementations.

**Compilation Caching**: Shape-keyed caching ensures kernels are compiled once and reused across decode steps. In autoregressive generation with stable shapes, compilation overhead is amortized over hundreds of steps.

**Persistent Cache Reuse**: Compiled Triton PTX for sub-segments is also cached on disk, reducing repeated startup compile cost across process restarts and iterative benchmarking sessions.

**Graceful Fallback**: Non-fusible segments fall back to CUDA Graphs or slot-by-slot execution. The system never crashes due to unsupported ops — it just runs them without fusion.

**Compilation Audit**: Per-op audit trail makes it easy to diagnose why specific ops weren't fused, enabling targeted improvements to op coverage.

**Coverage and Correctness Improvements**: Triton IR handling has been expanded for shape/index-heavy and layout-sensitive ops used in VLM decode paths (including gather, range, shape-of, set-scalar, permute/tile/strided-slice, concat/split axis handling, and conv2d sectioning).

**Runtime Robustness**: Section-boundary handling and kernel synchronization paths were hardened to avoid stale outputs and intermittent CUDA runtime failures.

### Disadvantages

**Triton Library Dependency**: Requires Triton 3.2.0+ installation. Not all deployment environments have Triton available, especially embedded or edge devices.

**Compilation Latency**: First compilation of a segment takes 10-100ms depending on segment size and GPU target. This is amortized but impacts first-token latency.

**Limited Op Coverage**: ~40 ops are mappable to Triton IR. Complex ops (custom CUDA kernels, multi-output ops) break fusion boundaries. Coverage will expand over time but will never reach 100%.

**Shape Change Recompilation**: Each unique shape combination requires a new compilation. For autoregressive generation with growing KV cache, this means periodic recompilations as the sequence length grows.

**Memory Overhead**: Compiled kernel binaries and GPU modules consume host and device memory. For large graphs with many segments and shape variations, cache memory can be significant.

## References

- libnd4j/include/graph/gpu/TritonGraphBackend.h, TritonGraphBackend.cpp
- libnd4j/include/graph/gpu/TritonIRBuilder.h, TritonIRBuilder.cpp
- libnd4j/include/graph/gpu/TritonTargetDispatch.h, TritonTargetDispatch.cpp
- libnd4j/include/graph/GraphBackend.h (abstract interface)
- libnd4j/cmake/FindTriton.cmake
- platform-tests/src/test/java/org/eclipse/deeplearning4j/nd4j/autodiff/samediff/TritonGraphBackendTest.java
- OpenAI Triton: https://github.com/openai/triton
- ADR 0061 - DynamicShapePlan Execution (NativeDynamicShapePlan integration)
- ADR 0058 - Multi-Backend Kernel Selection and Management
