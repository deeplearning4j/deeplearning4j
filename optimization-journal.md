# CUDA Graph Optimization Journal

## What We Achieved
- **Frozen constant detection**: 250/3220 slots skip execution during graph capture/replay
  - Graph nodes: 8522 → 8082 (-5.2%)
  - Value-independent ops (shape_of, zeros_like, ones_like, create, size_at, rank) break external dependency chains
- **Performance**: P50=23ms (43.5 tok/s), stable across all changes
- **Accuracy**: 38/100 unique tokens baseline maintained

---

## What NOT To Do

### 1. Do NOT gate output nullify (memset) for CUDA graph replay

**What we tried**: Skip `cached->nullify()` for "fully-writing" ops (add, matmul, cast, etc.) to reduce graph memset nodes from 3238 to ~940.

**What happened**: Output degraded to 3/100 unique tokens (degenerate).

**Why it fails**: CUDA kernels that fully overwrite their output during normal (slot-by-slot) execution do NOT fully overwrite during graph replay. We confirmed this by testing:
- Slot-by-slot with gating: **41/100 unique** (better than baseline — works perfectly)
- CUDA graph replay with gating: **3/100 unique** (catastrophically broken)

**Root cause**: Unknown. Possibly CUDA graph runtime optimizations, kernel launch parameter caching, or memory visibility differences between graph replay and normal execution. The graph records the kernel but replays it in a context where output bytes aren't all overwritten.

**Rule**: Graph memsets MUST stay unconditional. Never gate `cached->nullify()` on `needsZeroedOutput` when the result will be captured into a CUDA graph.

### 2. Do NOT change the C++ `isFullyWritingOp()` expecting it to affect the decode plan

**What we tried**: Modified `NativePlanCompiler::isFullyWritingOp()` in C++ to expand/shrink the fully-writing ops set.

**What happened**: Zero effect on the decode plan. The graph had identical memset counts regardless of C++ changes.

**Why it fails**: The `needsZeroedOutput` flag is set by the **Java-side** `DynamicShapePlanCompiler.java` (line 97-145) during plan compilation, serialized into the plan bytes, and read as-is by the C++ side. The C++ `NativePlanCompiler::isFullyWritingOp()` is only used when compiling from FlatGraph directly (a different code path). The Java-side `FULLY_WRITING_OPS` set is much larger and already marks ALL 3220 decode ops as not needing zeroed output.

**Rule**: To change `needsZeroedOutput` for the decode plan, edit `DynamicShapePlanCompiler.java`, not `NativePlanCompiler.cpp`.

### 3. Do NOT assume the NDArray constructor leaves buffers uninitialized

**What we tried**: Gate `out->nullify()` for new allocations in Step 3.

**What happened**: No effect — the 940 memsets in the graph remained unchanged.

**Why it fails**: `new NDArray(order, shape, dt)` constructor calls `_buffer->setToZeroBuffers()` (NDArray.hXX line 413). Every newly allocated NDArray is already zeroed. Our external `nullify()` call was redundant for new allocations. The 940 internal memsets come from:
- NDArray constructor zeroing
- Ops calling `nullify()` internally (e.g., `strided_slice` line 765, `create` line 34)
- CUBLAS workspace zeroing

**Rule**: Don't try to optimize away memsets for new NDArray allocations — the constructor already does it.

### 4. Do NOT put frozenConstantSlot check inside frozenContextReady block

**What we tried (earlier session)**: `frozenConstantSlot` check was nested inside `if (slot.frozenContextReady)`.

**What happened**: During graph capture, `frozenContextReady` is disabled (our capture fix), so frozen constant slots went through the full execution path — their kernels and memsets got recorded into the graph.

**Why it fails**: Graph capture disables `frozenContextReady` to ensure all contexts are freshly configured with capture buffers. But frozen constant slots don't NEED fresh configuration — their output never changes.

**Fix**: Move `frozenConstantSlot` check BEFORE `frozenContextReady` so it fires during capture too. This eliminated 440 graph nodes.

### 5. Do NOT refresh ALL inputs during capture in the frozen fast path

**What we tried (earlier session)**: When `tl_graphExecutionActive`, refresh all inputs from `outputSlots_` (not just external/view-producer).

**What happened**: SIGSEGV in `NDArray::applyTrueBroadcast` during capture.

**Why it fails**: Refreshing inputs while keeping frozen-context outputs creates mismatches. The output arrays were configured for the original inputs, not capture-time inputs. The context becomes internally inconsistent.

**Fix**: Disable `frozenContextReady` entirely during capture (save/restore). One-time cost for full context reconfiguration.

### 6. Do NOT assume op name consistency between Java and C++

**What we tried**: Added `zeros_like` to C++ FULLY_WRITING_OPS.

**What happened**: No effect because the Java side uses `zeroslike`, `zeros_as`, `zeros_like` as separate entries, and the actual op name from ONNX import was `zeroslike`.

**Why it matters**: Op names have synonyms (DECLARE_SYN in C++). The canonical name (`zeros_as`), the common name (`zeros_like`), and the ONNX-imported name (`zeroslike`) are all different strings. Both Java and C++ FULLY_WRITING_OPS sets must include ALL synonyms.

**Rule**: Always include all synonyms when adding ops to any set. Check DECLARE_SYN in the op declaration files.

---

## What DOES Work

### Frozen constant detection with value-independent ops
- Ops like `shape_of`, `zeros_like`, `ones_like`, `create` produce identical output every step when shapes are frozen
- These break the external dependency chain — downstream ops that only consume frozen outputs also become frozen
- 250/3220 slots detected (188 from value-independent ops, 62 from pure constant propagation)
- Cascading: if shape_of → strided_slice → create, ALL become frozen

### PointersManager H2D content dedup during capture
- Dimension/axis arrays (e.g., [0,1] for reduce, [2] for softmax axis) uploaded identically by many ops
- FNV-1a hash of content + size as cache key, skip alloc+memcpy on hit
- Saved 596 memcpy + 596 pinned host allocs per capture
- Zero risk — only active during graph capture, cleared on capture start/end

### Capture audit with auto top-10
- Per-op node counting during capture (getNumNodesDuringCapture before/after each slot)
- Auto-prints top-10 ops by node count after capture when timing is enabled
- Identified `onnx_multi_head_attention` (24 nodes each) as optimization target

### Unlimited element-wise fusion chain length
- `MAX_CHAIN_LENGTH` set to `INT_MAX` (was 8, then 16)
- "Fusion" is buffer reuse (in-place chaining), NOT kernel merge
- Each op still launches its own kernel — no register pressure concern
- Longest chain in SmolDocling: 4 ops (practical limit is model structure)

### Unconditional nullify during graph capture
- The memset gets recorded in the graph and replays every step
- Ensures no stale data leaks through during graph replay
- Cost: ~3238 memset nodes in graph, adds ~0ms to graph replay (GPU memset is fast)

### View-producer skip during nullify
- Ops that return views (permute, transpose) share their input's DataBuffer
- Nullifying the view zeros the shared buffer → corrupts upstream data
- `slotIsViewProducer_[si]` flag prevents this

---

## Graph Node Breakdown (SmolDocling decode, 3220 slots)

| Component | Before | After memcpy dedup | Notes |
|-----------|--------|-------------------|-------|
| Kernels | 2956 | 2956 | Op computations (unchanged) |
| Memcpys | 1828 | **1232** | **-596 (33% reduction)** via PointersManager cache |
| Memsets | 3238 | 3238 | Output buffer zeroing (CANNOT reduce for graph replay) |
| MemAllocs | 30 | 30 | Capture workspace allocations |
| MemFrees | 30 | 30 | Capture workspace frees |
| **Total** | **8082** | **7486** | **-596 nodes (7.4% reduction)** |

### Capture audit findings
- **Top node contributor**: `onnx_multi_head_attention` at 24 nodes each (12 instances)
- **982 host-only ops** (30%) contribute 0 graph nodes (shape_of, frozen constants, etc.)
- **2238 ops** actually generate graph work
- **Capture workspace utilization**: 124MB / 512MB (23.7%)

### Frozen constant savings
- 250 slots skipped → -63 kernels, -126 memcpys, -251 memsets = -440 nodes

### PointersManager memcpy dedup savings
- Content-based FNV hash cache for H2D copies during capture
- Identical dimension/axis arrays (e.g., [0,1] for reduce) shared across ops
- -596 memcpy nodes, -596 pinned host allocs, reduced capture workspace usage

### Remaining 1232 memcpy nodes — what they are
- Each `onnx_multi_head_attention` contributes ~8 memcpys (QKV shapes, dim arrays)
- These are unique per-attention-head dimension arrays (non-dedupable)
- Other sources: gather index arrays, reshape targets, broadcast dims
- **Cannot be further reduced** without caching at the op level (each op creates
  unique dimension arrays with different values)

### Theoretical further reduction
- If graph memsets could be gated: -2298 memsets → 5188 nodes (31% reduction)
- Blocked by CUDA graph replay correctness issue (unknown root cause)

---

## Potential Optimizations to Explore

### Priority 1: Reduce GPU execution time (~22ms/step)

#### A. Op fusion — increase chain length ✅ DONE
- **Changed**: `MAX_CHAIN_LENGTH` from 8 → `INT_MAX` (no limit)
- **Result**: No kernel count change (longest chain in model is 4 ops)
- Fusion is buffer reuse only — no register pressure or kernel merge

#### B. Eliminate redundant memcpys ✅ PARTIALLY DONE (1828→1232)
- **Done**: PointersManager content-based dedup cache during capture (-596 nodes)
- **Remaining 1232**: unique per-op dimension arrays (non-dedupable content)
  - Each attention op has unique QKV dimension arrays
  - Gather/scatter ops have unique index arrays
- **Not practical to reduce further** without op-level caching or GPU-side dim computation
- **Capture audit** now identifies top ops by node count for targeting

#### C. Investigate CUDA graph replay memset elimination
- 3238 memsets are 40% of graph nodes. Removing them would be the single biggest optimization.
- Slot-by-slot execution with gated memsets produces 41/100 unique (correct output).
- Graph replay with gated memsets produces 3/100 (broken).
- **Next steps**:
  1. Use `cuda-memcheck` or `compute-sanitizer --tool racecheck` on the captured graph
  2. Binary search: gate memsets for ONLY one op type (e.g., only `cast`) during capture
  3. Check if CUDA graph execution reorders independent nodes → uninitialized read race
  4. Check if cublas GEMM kernel reads from output (beta parameter handling)
  5. Test with `cudaGraphInstantiateWithFlags(cudaGraphInstantiateFlagAutoFreeOnLaunch)`
- **Risk**: High — root cause unknown. May be unfixable due to CUDA runtime behavior.

#### D. Remove 30 MemAlloc/MemFree nodes from graph
- These come from `cudaMallocAsync` calls during capture (op temporary workspace).
- The 512MB capture workspace (`tl_captureWorkspace`) exists to prevent this, but 30 ops
  still allocate dynamically.
- **Test plan**: Add logging in `CudaMemoryPool::allocate` during capture to identify which
  ops trigger dynamic allocation. Redirect to capture workspace stack allocator.
- **Risk**: Low — capture workspace already handles most allocations.

### Priority 2: Reduce per-step overhead (~1ms outside GPU)

#### E. Eliminate 3 per-step capture buffer copies
- Currently 3 buffers copied per step: attention_mask, position_ids, and one other.
- These are `SOURCE_PLACEHOLDER` buffers with `neverSkipCopy = true`.
- **Optimization**: For static KV cache, attention_mask is incrementally modified (append 1 at end).
  Could use a CUDA kernel to update in-place instead of full D2D copy.
- **Test plan**: Profile copy sizes. If <1KB, overhead is negligible (~200us total).
- **Risk**: Low — copies are already fast.

#### F. Reduce `cudaStreamSynchronize` calls
- Line 744: sync after every graph replay to let Java read outputs.
- **Optimization**: Use CUDA events instead of full stream sync. Record an event after graph
  launch, poll event in Java before reading output. Allows CPU work during GPU execution.
- **Test plan**: Replace sync with `cudaEventRecord` + `cudaEventSynchronize`. Measure if
  CPU-side overhead (Java output gathering) overlaps with GPU execution.
- **Risk**: Medium — requires Java-side changes to event-based polling.

### Priority 3: Memory optimizations

#### G. Configurable capture workspace size ✅ DONE
- **Env var**: `ND4J_DSP_CAPTURE_WORKSPACE_MB` (default 512MB, range 1-4096)
- **Measured utilization**: 124MB / 512MB (23.7%) for SmolDocling
- Logs utilization % after capture for per-model tuning
- Default kept at 512MB to avoid overfitting to one model

#### H. NDArray constructor zeroing elimination
- `new NDArray()` always calls `setToZeroBuffers()` (NDArray.hXX:413).
- For temporary arrays that will be fully written by the next op, this memset is wasted.
- **Optimization**: Add a `DataBuffer(size, dtype, workspace, /* zero= */ false)` constructor
  variant. Use it in Step 3 when `needsZeroedOutput = false`.
- **Test plan**: Only affects first allocation (not cached reuse). Measure warmup time reduction.
- **Risk**: Medium — must ensure NO code path reads uninitialized buffers.

### Priority 4: Accuracy improvements

#### I. Investigate 38% vs 41% token diversity gap (graph vs no-graph)
- Without CUDA graphs: 41/100 unique tokens
- With CUDA graphs: 38/100 unique tokens
- This 3-token gap suggests CUDA graph replay introduces subtle numerical differences.
- **Test plan**: Run both modes with `dumpOutputs=true`, diff the output arrays at each step.
  Find which step first diverges and which op produces different values.
- **Risk**: May be inherent floating-point ordering difference (acceptable).

#### J. Test with longer sequences (1000+ tokens)
- Current test only runs 100 tokens. Long-running inference may expose:
  - Memory leaks (growth per step)
  - Numerical drift (accumulated floating-point error)
  - KV cache overflow
  - GC pressure
- **Test plan**: `-Dvlm.test.maxTokens=1000`, monitor poolUsed growth, unique token ratio.
- **Risk**: Low — read-only test.

#### K. Test with different models
- Current optimizations are validated only on SmolDocling.
- Other VLM models may have different op distributions, larger KV caches, or different
  attention patterns that exercise different code paths.
- **Test plan**: Run on a second model (e.g., a text-only decoder) to verify generalization.

### Priority 5: Architecture changes — toward minimal overhead

**Ideal**: 1 kernel launch, 1 data upload, 0 allocations per step.
**Current**: 2956 kernels, 1232 memcpys, 3238 memsets, 30 allocs per step (in graph).

#### What's achievable vs what's not

| Target | Current | Achievable? | How |
|--------|---------|-------------|-----|
| 0 allocations | 30 | **Yes** | Identify 30 ops that bypass capture workspace, redirect them |
| 0 memcpys | 1232 | **Partially** | ~596 already eliminated. Remaining are unique per-op data. Could pre-compute all dim arrays on GPU at plan compile time |
| 0 memsets | 3238 | **No** | Required for CUDA graph replay correctness (see "What NOT To Do #1") |
| 1 kernel | 2956 | **No (graph)** | CUDA graphs replay individual kernels; can't merge. Would need a code-generation approach (Triton, custom CUDA) |
| Fewer kernels | 2956 | **Yes** | True kernel fusion via Triton backend or custom fused kernels for common patterns (attention, layer_norm+gelu+mul) |

#### L. Pre-compute dimension arrays at plan compile time
- Currently: each op creates dimension arrays on host, uploads H2D during capture
- Better: at compile time, allocate one GPU buffer with ALL dimension arrays packed contiguously
- Each op gets a pre-computed device pointer offset — zero H2D copies during capture
- **Eliminates all remaining 1232 memcpy nodes**
- **Risk**: Medium — requires plan compiler to know all ops' dimension needs upfront

#### M. True kernel fusion via Triton/custom CUDA
- Current "fusion" is buffer reuse, not kernel merge
- True fusion: generate one kernel for `layer_norm → mul → add → gelu` chain
- Triton backend already exists in codebase (`TritonGraphBackend.cpp`)
- Each fused kernel eliminates N-1 launches + intermediate buffers
- 12 attention layers × 6 fusible chains = ~72 fused ops → ~200 fewer kernel launches
- **Risk**: High — requires Triton runtime, op-specific code generation

#### N. Multi-stream graph execution
- Current: single mega-graph on one stream
- Independent subgraphs (e.g., QKV projections) could run in parallel
- **Risk**: High — complex sync, may not help on single-GPU

#### O. Persistent kernel approach
- One mega-kernel that processes entire decode step
- **Risk**: Very high — complete rewrite of execution model

---

## Time Budget Breakdown (per decode step)

| Phase | Time | % | Where |
|-------|------|---|-------|
| Capture buffer copy | ~0.3ms | 1.3% | 3 D2D cudaMemcpyAsync |
| Graph launch + GPU exec | ~22ms | 95.7% | cudaGraphLaunch → kernel execution |
| KV scatter | ~0.4ms | 1.7% | Direct CUDA kernel for KV cache update |
| Stream sync | ~0ms | 0% | GPU finishes before sync returns |
| Java overhead | ~0.6ms | 2.6% | Input prep, output read, token sampling |
| **Total** | **~23ms** | **100%** | **43.5 tok/s** |

**Bottleneck**: 95.7% of time is GPU kernel execution inside the graph. The remaining
optimizations that matter are:
1. **True kernel fusion** (reduce 2956 kernels) — requires Triton or custom CUDA
2. **Pre-computed dimension arrays** (eliminate 1232 memcpys) — plan compiler change
3. **Attention kernel optimization** (each of 12 attn ops = 24 graph nodes = ~288 total)

Everything else (workspace, capture overhead, Java) is in the noise.

---

## Things to Test Before Any Optimization

1. **Baseline stability**: Run the 100-token test 5 times, verify P50 is consistently 23ms
   and unique tokens is consistently 38/100. If results vary, investigate noise sources.

2. **Memory stability over 1000 tokens**: Verify no memory leak per step. Expected: ~1MB/step
   growth from KV cache, not 40MB/step (the old leak).

3. **Graph re-capture trigger**: Intentionally change a shape mid-run (if possible) to verify
   the graph invalidation + re-capture path works correctly.

4. **Multi-model sequential**: Load SmolDocling, run inference, unload, load another model,
   run inference. Verify no memory corruption from cached graph state.

5. **Stress test frozen constants**: Verify that frozen constant slots produce identical output
   at step 1 vs step 100 by adding a checksum comparison.
