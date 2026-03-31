# ADR: Triton Graph Backend

## Status

Implemented

Proposed by: Adam Gibson (January 2025)

Updated by: Runtime maintainers (March 31, 2026)

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
│  GraphExecutionMode (non-cascading — each mode is a complete path): │
│    GEM_TRITON       → Triton fused kernels + CUDA graph replay      │
│    GEM_CUDA_GRAPHS  → CUDA graph capture/replay only                │
│    GEM_SLOT_BY_SLOT → individual op dispatch (no fusion/graphs)     │
│    GEM_AUTO         → selects best available backend per segment    │
│                                                                     │
│  Per-segment ExecutionPhase tracking:                               │
│    WARMUP → COMPILING → COMPILED → REPLAYING (capturable)          │
│    SLOT_BY_SLOT (non-capturable segments, always)                   │
│                                                                     │
│  Failure = hard error. No cascading fallback between modes.         │
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

### GraphSegment Structure

`GraphSegment` separates immutable definition from mutable execution state:

```cpp
struct GraphSegment {
  // ── Immutable definition (set at buildSegments, never changes) ─────
  int startSlot;
  int endSlot;
  bool isCapturable;
  LongType shapeKey;

  // ── Mutable execution state (changes per-execution) ────────────────
  struct ExecState {
    int executionCount = 0;
    bool captureFailed = false;
    std::unique_ptr<GraphReplayHandle> replayHandle;
    LongType cachedShapeKey = 0;
    std::string compiledByBackend;   // "Triton", "CUDA", "slot-by-slot", etc.
    bool argTableStable = false;     // Fast-replay: skip refresh when stable
    ExecutionPhase currentPhase = ExecutionPhase::WARMUP;
    // ... OOM retry, JIT kernel, symbolic shape, batch-zero entries
    void reset();
  };

  ExecState exec;
};
```

**Key design points:**
- **Immutable definition** (`startSlot`, `endSlot`, `isCapturable`, `shapeKey`) is set once at `buildSegments()` time and never changes.
- **Mutable `ExecState`** tracks everything that changes during execution: counters, replay handles, compilation status, and the `ExecutionPhase`.
- **`ExecutionPhase`** is the ACTUAL runtime mode of a segment (not the user's preference). It progresses: `WARMUP` -> `COMPILING` -> `COMPILED` -> `REPLAYING` for capturable segments, or stays at `SLOT_BY_SLOT` for non-capturable segments.
- **No `pendingClose` or `deferredClose`**: The memory model uses one persistent array per slot. Arrays are reused across executions without close/reopen cycles.

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
| Fused Attention (decode, seqQ=1) | blockM=1, blockN=auto | 4 | 1 | Attention grid |
| Fused Attention (prefill) | blockM=seqQ, blockN=auto | 4 | 1 | Attention grid |

The attention tile config is selected by `chooseFusedAttentionTileConfig()` which sets `blockM = seqQ` for single-token decode. This eliminates 98% wasted compute — previous blockM=64 wasted 63/64 threads on masked positions.

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

### Shape-Aware Caching and Recompilation

Compiled kernels are cached by `{startSlot, endSlot, shapeKey}`:

```cpp
struct CacheKey {
    int startSlot;
    int endSlot;
    LongType shapeKey;  // Hash of input/output shapes
};
std::unordered_map<CacheKey, CompiledKernel> compiledKernels_;
```

When shapes change (e.g., growing KV cache in autoregressive generation), the shape key changes, triggering recompilation via the shape key cache. Previous compiled kernels are retained in cache for the case where shapes cycle (e.g., batch dimension alternating between prefill and decode).

**No adaptive splitting**: Segments are NOT binary-split on shape instability. Instead, the segment definition remains fixed (immutable `startSlot`/`endSlot`), and shape changes invalidate the segment's `ExecState` (resetting `cachedShapeKey` and `replayHandle`), causing recompilation with the new shapes. The shape key cache retains previously compiled kernels, so returning to a previously-seen shape is a cache hit with no recompilation.

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

All Triton code is guarded by `#if HAVE_TRITON ... #endif`. When Triton is not available, the user selects a different `GraphExecutionMode` (e.g., `GEM_CUDA_GRAPHS` or `GEM_SLOT_BY_SLOT`). Each mode is a complete, non-cascading execution path.

## Recent Optimizations (March 2026)

### Section Fusion Scoring

A heuristic-based cost model (`FusionScoring.cpp`) evaluates whether merging adjacent Triton sections into mega-kernels is beneficial. The scoring considers:

- **Grid compatibility**: Sections with incompatible grid dimensions (1D vs 2D) cannot merge
- **Shared memory estimation**: Combined shared memory must fit GPU limits
- **Memory traffic savings**: Bytes of intermediate global memory eliminated by fusion
- **Kernel launch overhead savings**: ~15μs per eliminated kernel launch
- **Register pressure penalties**: Excessive register usage reduces GPU occupancy
- **Attention neighborhood bonuses**: +50.0 score for sections adjacent to attention ops

Sections are merged when the net fusion score exceeds `TRITON_FUSION_MIN_SCORE` (default: 5.0).

**Configuration**:
- `ND4J_TRITON_SECTION_FUSION` (default: true) — enable section fusion
- `ND4J_TRITON_FUSION_SCORING` (default: true) — enable cost-model scoring
- `ND4J_TRITON_FUSION_MIN_SCORE` (default: 5.0) — minimum score threshold

### Attention Neighborhood Fusion

Sections adjacent to `FUSED_ATTENTION` sections (e.g., `GATHER`, `CONCAT`, `STACK` for KV cache operations) can now be fused with surrounding element-wise ops. This reduces section fragmentation in attention patterns common in transformer decoders.

The fusion scoring gives a significant bonus (+50.0) to attention-adjacent sections, allowing `GATHER`/`CONCAT`/`STACK` sections to merge with nearby element-wise ops even when the raw memory traffic savings alone would not justify fusion.

**Configuration**: `ND4J_TRITON_FUSE_ATTENTION_NEIGHBORHOODS` (default: true)

**Performance impact** (SmolDocling, RTX 4090): 83.98 → 86.37 tok/s steady-state decode

### Attention Kernel Decode Optimization

The fused attention kernel's tile configuration was previously using a fixed `blockM=64` for all cases. For single-token decode (seqQ=1), this wasted 63/64 of compute on masked positions.

**Solution**: `chooseFusedAttentionTileConfig()` now sets `blockM = seqQ`:
- Single-token decode (seqQ=1): blockM=1 — each thread block processes exactly one query token
- Multi-token prefill: blockM=seqQ — adapts to actual sequence length

Additionally, **K buffer validation** was added for static KV cache mode. The validator checks shape capacity (not content length) when using dual-buffer KV, enabling Triton compilation during decode where it previously fell back to native.

**Warps/stages tuning results** (SmolDocling, RTX 4090):

| Config | Steady-state tok/s |
|--------|--------------------|
| warps=2, stages=1 (baseline) | 69.56 |
| warps=4, stages=1 (optimal) | 86.91 (+24.9%) |
| warps=4, stages=2 | 82.74 |
| warps=2, stages=2 | 71.23 |

**Configuration**:
- `ND4J_TRITON_NUM_WARPS` (default: auto — selected by tile config)
- `ND4J_TRITON_NUM_STAGES` (default: auto)
- `Environment.tritonAttentionBlockN()` (0 = auto)

### Triton Fusion Optimization Flags

Fine-grained control over which fusion passes are applied during IR construction:

| Flag | Default | Purpose |
|------|---------|---------|
| `ND4J_TRITON_FUSE_IDENTITY_SHAPES` | true | Fuse identity reshape/expand_dims/squeeze into adjacent ops |
| `ND4J_TRITON_FUSE_CAST_CHAINS` | true | Merge consecutive cast operations |
| `ND4J_TRITON_SPECIALIZE_PERMUTE_SEQ1` | true | Optimize permutes for seq=1 (decode) |
| `ND4J_TRITON_FUSE_ATTENTION_NEIGHBORHOODS` | true | Prefer larger compile ranges around attention-adjacent data movement |
| `ND4J_TRITON_FUSED_MATMUL` | false | Fuse matmul→bias→activation (HIGH RISK — cuBLAS faster for M=1) |

### Verification & Debugging Infrastructure

Triton now includes a comprehensive set of diagnostic and verification flags:

**Verification** (correctness validation):
- `ND4J_TRITON_VERIFY_KERNELS` — run both Triton and native path, compare outputs element-wise
- `ND4J_TRITON_VERIFY_KEEP_NATIVE` — keep native outputs during verify (detect error accumulation)
- `ND4J_TRITON_VERIFY_FULL_SNAPSHOT` — save/restore ALL outputSlots during verify (corruption detection)

**Debugging** (execution control):
- `ND4J_TRITON_SKIP_KERNELS` — skip Triton, run native fallback (isolate Triton issues)
- `ND4J_TRITON_MAX_SUBKERNEL_INDEX` — limit sub-kernel execution (-1 = unlimited)
- `ND4J_TRITON_FORCE_RECAPTURE` — force CUDA graph re-capture every step
- `ND4J_TRITON_CAPTURE_MIN_EXEC` (default: 2) — execution count before graph capture

**Dump** (IR and compilation inspection):
- `ND4J_TRITON_KERNEL_DUMP` — save generated Triton MLIR and compiled PTX to disk
- `ND4J_TRITON_DUMP_SECTIONS` — output section breakdown to stderr
- `ND4J_TRITON_DUMP_ARGS` — output argument mapping to stderr
- `ND4J_TRITON_LOG_ALL_PATTERNS` — log pattern matching details during IR construction

### Compilation Type Control

Fine-grained control over which section types are compiled vs. falling back to native:

- `ND4J_TRITON_COMPILE_ALL` (default: false) — compile all section types
- `ND4J_TRITON_INCLUDE_TYPES` (comma-separated whitelist, e.g., `REDUCTION,NORMALIZATION,GATHER`)
- `ND4J_TRITON_EXCLUDE_OPS` (comma-separated op blacklist, e.g., `matmul,softmax`)

This enables incremental enablement of Triton compilation for new section types while keeping production defaults conservative.

## Consequences

### Advantages

**Kernel Fusion**: Eliminates intermediate global memory stores between fused ops. For element-wise chains (add→relu→mul), this achieves near-peak memory bandwidth vs. 3x bandwidth waste with separate kernels.

**Multi-GPU Vendor Support**: Single codebase compiles to NVIDIA PTX, AMD AMDGCN, and Intel SPIR-V. This is the only fusion path that works on all three GPU vendors without per-vendor kernel implementations.

**Compilation Caching**: Shape-keyed caching ensures kernels are compiled once and reused across decode steps. In autoregressive generation with stable shapes, compilation overhead is amortized over hundreds of steps.

**Persistent Cache Reuse**: Compiled Triton PTX for sub-segments is also cached on disk, reducing repeated startup compile cost across process restarts and iterative benchmarking sessions.

**Non-Cascading Execution Modes**: Each `GraphExecutionMode` is a complete, self-contained path. Failure within a mode is a hard error — there is no cascading fallback between modes (e.g., Triton failure does NOT cascade to CUDA Graphs). Non-fusible segments within a Triton-mode plan use CUDA graph capture for the non-fusible ranges ("gap ops"), but this is part of the Triton execution path, not a fallback to a different mode.

**Compilation Audit**: Per-op audit trail makes it easy to diagnose why specific ops weren't fused, enabling targeted improvements to op coverage.

**Coverage and Correctness Improvements**: Triton IR handling has been expanded for shape/index-heavy and layout-sensitive ops used in VLM decode paths (including gather, range, shape-of, set-scalar, permute/tile/strided-slice, concat/split axis handling, and conv2d sectioning).

**Runtime Robustness**: Section-boundary handling and kernel synchronization paths were hardened to avoid stale outputs and intermittent CUDA runtime failures.

**Cost-Model Fusion**: FusionScoring provides data-driven section merge decisions, replacing heuristic-only approaches. Attention neighborhood bonuses specifically optimize the fragmented section patterns common in transformer decoder attention.

**Decode-Optimized Attention**: Dynamic tile configuration (`blockM=seqQ`) eliminates 98% wasted compute for single-token decode, achieving +24.9% throughput improvement with optimal warp/stage tuning.

**Comprehensive Verification**: Built-in golden-comparison mode (`VERIFY_KERNELS`) enables continuous correctness validation without separate test infrastructure, including full-snapshot mode for detecting subtle corruption.

### Disadvantages

**Triton Library Dependency**: Requires Triton 3.2.0+ installation. Not all deployment environments have Triton available, especially embedded or edge devices.

**Compilation Latency**: First compilation of a segment takes 10-100ms depending on segment size and GPU target. This is amortized but impacts first-token latency.

**Limited Op Coverage**: ~40 ops are mappable to Triton IR. Complex ops (custom CUDA kernels, multi-output ops) break fusion boundaries. Coverage will expand over time but will never reach 100%. Section types with `fusionVerified=false` (e.g., SPLIT, CONCAT, CONST_GEN) are excluded from Triton compilation to prevent SIGABRT crashes — they must be individually verified before enabling.

**Shape Change Recompilation**: Each unique shape combination requires a new compilation. For autoregressive generation with growing KV cache, this means periodic recompilations as the sequence length grows. *Mitigated by*: shape key caching retains all previously compiled kernels (returning to a seen shape is a cache hit), symbolic shape ranges (see ADR 0061) reduce recompilation frequency, and disk cache persistence (see ADR 0061) eliminates cross-process recompilation. Segments are NOT adaptively split on shape instability — the segment definition stays fixed, and only the `ExecState` is reset for recompilation.

**Memory Overhead**: Compiled kernel binaries and GPU modules consume host and device memory. For large graphs with many segments and shape variations, cache memory can be significant.

## References

### Core Triton Backend

- libnd4j/include/graph/gpu/TritonGraphBackend.h, TritonGraphBackend.cpp
- libnd4j/include/graph/gpu/TritonGraphBackend_internal.h
- libnd4j/include/graph/gpu/TritonGraphBackend_binary.cpp
- libnd4j/include/graph/gpu/TritonGraphBackend_cache.cpp
- libnd4j/include/graph/gpu/TritonGraphBackend_compile.cu
- libnd4j/include/graph/gpu/TritonGraphBackend_execute.cu
- libnd4j/include/graph/gpu/TritonGraphBackend_kernel.cu

### IR Builder

- libnd4j/include/graph/gpu/TritonIRBuilder.h, TritonIRBuilder.cpp
- libnd4j/include/graph/gpu/TritonIRBuilder_internal.h
- libnd4j/include/graph/gpu/TritonIRBuilder_analysis.cpp
- libnd4j/include/graph/gpu/TritonIRBuilder_types.cpp
- libnd4j/include/graph/gpu/TritonIRBuilder_emitters.cpp
- libnd4j/include/graph/gpu/TritonIRBuilder_kernels.cpp
- libnd4j/include/graph/gpu/TritonIRBuilder_module.cpp
- libnd4j/include/graph/gpu/TritonIRBuilder_sections.cpp
- libnd4j/include/graph/gpu/TritonIRBuilder_cuda.cu

### Fusion & Configuration

- libnd4j/include/graph/gpu/FusionScoring.cpp
- libnd4j/include/graph/gpu/SectionTypeConfig.h
- libnd4j/include/graph/GraphBackendCommon.h

### Target Dispatch

- libnd4j/include/graph/gpu/TritonTargetDispatch.h, TritonTargetDispatch.cpp

### cuBLAS Lt Integration

- libnd4j/include/helpers/cuda/MmulHelper.cu (tryLtMatmul)
- libnd4j/include/helpers/cuda/cublasHelper.cu (Lt handle management)
- libnd4j/include/helpers/cublasHelper.h

### Build & Configuration

- libnd4j/cmake/FindTriton.cmake
- libnd4j/include/system/Environment.h (Triton configuration flags)
- nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/factory/Environment.java

### Tests & Benchmarks

- platform-tests/src/test/java/org/eclipse/deeplearning4j/nd4j/autodiff/samediff/TritonGraphBackendTest.java
- platform-tests/src/test/java/org/eclipse/deeplearning4j/vlm/TestSmolDoclingOptimizedPipeline.java
- nd4j/samediff-llm/src/main/java/org/eclipse/deeplearning4j/model/benchmark/BenchmarkConfig.java
- nd4j/samediff-llm/src/main/java/org/eclipse/deeplearning4j/model/benchmark/BenchmarkConfigApplier.java

### Related

- OpenAI Triton: https://github.com/openai/triton
- ADR 0061 - DynamicShapePlan Execution (NativeDynamicShapePlan integration)
- ADR 0058 - Multi-Backend Kernel Selection and Management
- ADR 0067 - Scaled Dot-Product Attention Optimization
