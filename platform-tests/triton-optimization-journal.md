# Triton Optimization Journal — SmolDocling on RTX 4090

## Test Configuration
- Model: SmolDocling 360M (vision encoder + decoder + embed_tokens)
- GPU: RTX 4090 24GB
- Test: `TestSmolDoclingOptimizedPipeline#testOptimizedDoclingPipeline`
- PDF: pathfinder-mythic.pdf page 10
- Tokens: 20 (quick) / 100 (full)

## Baseline (REDUCE_OVERHEAD, no Triton)
- **Token diversity: 60% (12/20 unique)**
- **Decode throughput: 7.74 tok/s** (129ms/step avg)
- CUDA graph capture NOT active (single mega-segment, capture fails on gather/shape ops)
- Triton kernel launches: 0
- This is the correctness reference

## Test 1: MAX_AUTOTUNE (Triton, pre-fix)
- **Token diversity: 10% (2/20 unique)** — all `<doctag>` from step 2+
- **Decode throughput: 0.04 tok/s** (avg 26594ms due to Triton compilation at step 2)
- Triton kernel launches: 4122 (206/step)
- Step 2 took 502s (Triton JIT compilation)
- After compilation: ~109ms/step (9 tok/s) — fast but WRONG

### Root Cause 1: Triton rms_norm missing weight parameter (FIXED in session 1)
- `identifySections()` merged NORMALIZATION ops (rms_norm, softmax) into ELEMENTWISE sections
  via `canMergeWithElementwise()` returning true for NORMALIZATION
- The merged ELEMENTWISE section passes `isFallbackSection()` and gets compiled by Triton
- `emitNormalizationOp("rms_norm")` computes `x * rsqrt(mean(x^2) + eps)` but IGNORES
  the second input (weight/scale parameter)
- **Fix**: Changed `canMergeWithElementwise()` to return false for REDUCTION and NORMALIZATION

### Root Cause 2: Binary pow op treated as unary (FOUND in session 2)
- `pow` mapped as `UNARY_ELEMENTWISE` in op table — IR builder uses only first input
- Exponent read from `slot.tArgs[0]`, defaulting to 2.0f if not set
- ONNX `Pow` is binary: `pow(base, exponent_tensor)` where both are tensors
- For RoPE: `pow(10000, freq_indices)` → computed as `pow(10000, 2.0)` = 100,000,000 constant
- **All positional information destroyed** — model can't distinguish token positions
- This explains why step 1 logits are identical (no position-dependent computation yet)
  but step 2+ diverges completely (position encoding needed for attention)
- **Fix**: Changed `pow`/`Pow` to `BINARY_ELEMENTWISE`, added `custom.pow` handler in
  `emitBinaryElementwise()`: `exp(exponent * log(base))`. Added unary fallback for
  single-input pow ops (scalar exponent in tArgs).
- **Files**: `TritonIRBuilder.cpp` lines 249-250, 1876-1882, 3745-3768, 5893-5908

### Root Cause 3: Stale native error propagation (FIXED in session 1)
- DSP graph capture failures left non-zero errorReference
- token_sample failed with "slot 0 (gather) failed with status 50" (stale error)
- Fix: `setError(0, "")` in NativeOps_dsp.cu + clearLastError in Java paths

## Session 2 Changes

### Phase 2: Allow Triton Fallback During CUDA Graph Capture
- **Environment.h**: Added `_tritonAllowFallbackCapture{true}` flag
- **TritonGraphBackend.cpp**: Guards at capture rejection points check flag
- When enabled, cuBLAS/native ops execute during capture and get recorded into CUDA graph

### Phase 3: Cast Elimination Pass
- **FusionPass.cpp**: Detects consecutive cast pairs (A→B followed by B→A) and marks as identity
- **Default**: `false` — needs validation before enabling

### Phase 4: Matmul Segmentation
- **NativeDynamicShapePlan.cpp**: Breaks segments at matmul/attention boundaries
- **Default**: `false` (opt-in)

### Phase 5: FP16 Compute for Matmuls
- **MmulHelper.cu**: Auto-cast FP32 inputs to HALF when `dspFp16Compute=true`
- **Default**: `false` (opt-in)

### Triton Verify/Skip Mode (debugging)
- `ND4J_TRITON_SKIP_KERNELS=1`: Replace all Triton sub-kernels with native slot-by-slot
- `ND4J_TRITON_VERIFY_KERNELS=1`: Run both Triton and native, compare outputs per sub-kernel
- **File**: `TritonGraphBackend.cpp` executeSegment

## Architecture Notes

### Segment Compilation Flow
1. `identifySections()` classifies ops into section types (ELEMENTWISE, MATMUL, FUSED_ATTENTION, etc.)
2. `canMergeWithElementwise()` allows SHAPE_MANIPULATION and IDENTITY to merge into ELEMENTWISE
3. `isFallbackSection()` returns true for anything NOT ELEMENTWISE or IDENTITY
4. Only ELEMENTWISE sections get Triton sub-kernels; everything else runs via `fallbackRangeExecutor_`
5. SmolDocling decoder: 3840 slots → 229 Triton sub-kernels + native fallback for gaps

### Gap Filling During Execution
- `executeSegment()` tracks `nextSlotToRun` and calls `fallbackRangeExecutor_` for gaps
- During CUDA graph capture, fallback allocation fails → capture fails → falls back to direct dispatch
- Direct dispatch succeeds because stream is not in capture mode
- The `fallbackRanges` vector in CompiledSegment is declared but NEVER populated (not a bug — gaps are detected via slot tracking)

## Test 2: MAX_AUTOTUNE (post-pow-fix + fallback-fix) — PASSED
- **Token diversity: 60% (12/20 unique)** — matches REDUCE_OVERHEAD baseline exactly
- **Decode throughput: 2.71 tok/s** overall (10.5 tok/s steady-state after JIT)
- Triton sub-kernels: 417 per step
- Step 2: 4538ms (Triton JIT compilation), steps 3+: ~95ms (10.5 tok/s)
- CUDA graph capture NOT active (`hasGraph=0` in logs)
- Direct Triton dispatch works correctly

### Root Cause 3 (updated): tl_graphExecutionActive during fallback (FIXED session 2)
- Triton path set `tl_graphExecutionActive=true` for the entire segment execution
- Fallback executor (matmul, attention, concat) ran with this flag true
- Flag causes ops to use capture workspace for allocation (too small) → status 50 failures
- Sticky CUDA error cascades → all subsequent ops fail (illegal memory access)
- **Fix**: Save/restore `tl_graphExecutionActive`, set to `false` during fallback execution

## Session 3: Performance — CUDA Graph Capture with Triton

### Current State
- Triton accuracy: CORRECT (60% diversity matching baseline)
- Triton steady-state: ~95ms/step (10.5 tok/s) — direct dispatch, no CUDA graphs
- Baseline CUDA graphs (no Triton): ~20ms/step (50 tok/s)
- Target: 2.5ms/step (400 tok/s)

### Root Cause 4: SIGSEGV during cudaGraphLaunch (capture recorded invalid MemFree nodes)

**Problem**: CUDA graph capture with Triton succeeded (graph created and instantiated), but
`cudaGraphLaunch` crashed with SIGSEGV in libcuda.so.

**Analysis**: During graph capture, `executeSegmentSlotBySlot` (called by Triton's fallback
executor for matmul/attention gaps) performed these capture-breaking operations:

1. **`flushPendingClose` (line 1587)**: Called `delete arr` which triggered
   `DataBuffer::deleteSpecial()` → `cudaFreeAsync(ptr, captureStream)` for memory allocated
   OUTSIDE the capture. CUDA forbids freeing graph-external memory within a captured stream.
   The resulting MemFree graph nodes contain stale addresses on launch → SIGSEGV.

2. **OOM retry path**: `cudaStreamSynchronize` + `flushPendingClose` + `cudaMemPoolTrimTo`
   all break capture semantics.

3. **CudaMemoryPool::free** for non-workspace memory: Same MemFree graph node issue.

**Fix** (5 changes):
- `executeSegmentSlotBySlot`: Detect capture mode via `cudaStreamIsCapturing`, skip
  `flushPendingClose` and OOM retry during capture
- `DataBuffer::deleteSpecial`: Return early during capture for non-workspace owned memory
  (don't reset `_specialBuffer`, don't update MemoryCounter — memory stays allocated)
- `CudaMemoryPool::free`: Skip ALL frees during capture as defense-in-depth
- Post-capture cleanup: `flushPendingClose` after capture attempt to free accumulated arrays
- **Files**: `NativeDynamicShapePlan.cpp`, `DataBuffer.cu`, `CudaMemoryPool.cu`

### Root Cause 5: executionCount warmup path (pre-existing, fixed earlier)
- `executeSegmentWithGpuGraph` returned early at executionCount==0 without incrementing
- Fixed: increment before return

### Root Cause 6: No capture workspace for Triton path (pre-existing, fixed earlier)
- Triton capture block didn't set up `tl_captureWorkspace` infrastructure
- Fixed: Added 512MB workspace allocation + thread-local setup matching non-Triton path

### Root Cause 7: External input address instability (pre-existing, fixed earlier)
- `inputs_embeds` and `input_ids` created new arrays each step → address key mismatch
- Fixed: `reusableEmbeddings`/`reusableInputIds` in StaticKvCacheDecodeLoop.java

### Capture Safety Audit
- Triton sub-kernels: `cuLaunchKernel` ✓ (capture-safe)
- cuBLAS matmul: ✓ (capture-safe, uses LaunchContext stream = capture stream)
- PointersManager: ✓ (`synchronize()` skips when `tl_graphExecutionActive`, `allocateDevMem` uses pool → workspace, destructor frees via pool → no-op during capture)
- DataBuffer operations: ✓ (`syncToPrimary` returns early, `syncToSpecial` uses capture stream, `nullify` uses `captureSafeStreamOrDefault()`, `deleteSpecial` skips during capture)

### Test 3: Pending (after build)
