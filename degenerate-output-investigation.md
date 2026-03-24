# Degenerate Output Investigation

## Current State
- **Throughput**: FIXED (91 tok/s) — device context restore in FrozenDecodeStep.allocateBuffers()
- **Accuracy**: BROKEN — 13/50 unique tokens, repeating `<picture><loc_*>` pattern
- Working commit: `52ad5cf5d5` | Broken commit: `a7d964de31` (single commit gap)

## Fixes Applied So Far
1. `FrozenDecodeStep.java` — device context save/restore in allocateBuffers() (throughput, CONFIRMED)
2. `DecoderUtils.java` — removed `!nativeDecodeInputs` guard on bias putScalar
3. `MmulHelper.cu` — reject mixed FP16×FP32 in tryLtMatmul (`aType != bType → return false`)

## All Changes in Broken Commit a7d964de31

### MmulHelper.cu (354 lines added) — 4 distinct changes:

**S1. cuBLAS Lt fast path** (lines 143-310)
- New `tryLtMatmul()` for M=1, N≥16384 vocab projections
- ALREADY FIXED: reject mixed input types
- Status: Fixed, not the primary cause

**S2. Capture-aware cast reuse** (lines ~750-790)
- `tl_lastCaptureCastSource{A,B}` + `tl_lastCaptureCastArray{A,B}` track the last cast
- If same source pointer `A` seen again during graph execution, reuses the cached cast
  buffer WITHOUT re-executing the assign (cast kernel)
- Theory: During CUDA graph REPLAY, source pointer identity doesn't mean same data.
  The activation tensor address is fixed (capture buffer), but contents change each step.
  Pointer-based reuse skips the cast → stale FP16 values → all matmuls see same activations
  → identical logits every step → degenerate loop.
- **File**: `libnd4j/include/helpers/cuda/MmulHelper.cu`, search for `sameLogicalA`
- **Change**: Remove the `sameLogicalA`/`sameLogicalB` shortcut paths. Keep the
  normal cache path that calls `cached->assign(effA)` which re-executes the cast.
- **Rebuild**: .cu file, ~4-5 min

**S3. Row-vector fast path** (lines ~902-940)
- New `rowVectorFastPath` for M=1 that swaps operand order to cuBLAS col-major
- Condition: `M == 1 && !transA && transB && pA->ews() == 1 && pB->strideAt(1) == 1 && pC->ews() == 1`
- Uses `N,1,K` dimensions with `CUBLAS_OP_N, CUBLAS_OP_N` instead of original trans flags
- Theory: The leading dimensions or operand swap could be wrong for certain stride patterns.
  `ldaFast = pB->strideAt(0)` might not be correct if B was transposed via view (stride trick).
- **File**: `libnd4j/include/helpers/cuda/MmulHelper.cu`, search for `rowVectorFastPath`
- **Change**: Make `rowVectorFastPath = false` (always use original cuBLAS path)
- **Rebuild**: .cu file, ~4-5 min

**S4. mmulNxN capture cast reuse** (lines ~1248-1280)
- Same pointer-based cast reuse but in `mmulNxN` path using `tl_captureCastReuse{A,B}` maps
- Theory: Same stale-cast issue as S2 but for batched/higher-dimensional matmuls
- **File**: same, search for `tl_captureCastReuseA.find`
- **Change**: Disable the reuse map lookups in mmulNxN
- **Rebuild**: .cu file, ~4-5 min

### NDArray.hXX (39 lines) — 1 change:

**S5. nullptr host buffers in copyDataForAssign/asT**
- On CUDA, passes `nullptr` instead of `buffer()` when both arrays have device buffers
- Affects: `copyDataForAssign` (2 code paths) and `asT()`
- Theory: If any kernel's CPU fallback path or host-side logic dereferences the host pointer,
  it would crash or read zeros. But more subtly, calling `buffer()` may have been triggering
  a necessary D→H sync as a side effect. Skipping it could leave host data stale for
  later reads that don't go through `prepareSpecialUse`.
- **File**: `libnd4j/include/array/NDArray.hXX`
- **Change**: Revert the 3 `#if defined(SD_CUDA)` blocks
- **Rebuild**: HEADER = 30-45 min full rebuild. Try LAST.

### NativeOpsHelpers_Context.cpp (51 lines) — 1 change:

**S6. Shape sync optimization**
- Previously: ALL inputs `forceSyncToHost()` before shape calculation
- Now: only inputs ≤4096 elements, plus "where"/"unique" ops
- Theory: A shape function in the VLM decode graph reads values from a large input and
  gets stale host data → wrong output shape → cascading wrong results
- **File**: `libnd4j/include/legacy/impl/NativeOpsHelpers_Context.cpp`
- **Change**: `shouldSyncInputForShape` returns true always
- **Rebuild**: .cpp file, ~4 min

### FusionScoring.cpp + TritonIRBuilder_sections.cpp — 1 change:

**S7. Attention neighborhood fusion**
- `tritonFuseAttentionNeighborhoods` flag (default: true) adds +50 score bonus to merge
  GATHER/CONCAT/STACK sections adjacent to attention ops
- TritonIRBuilder allows these types to merge with elementwise sections
- Theory: Incorrectly fused sections produce wrong Triton kernel → wrong attention output
- **File**: `libnd4j/include/system/Environment.h` has `_tritonFuseAttentionNeighborhoods`
- **Test**: Set `tritonFuseAttentionNeighborhoods(false)` via Environment before decode.
  Could potentially be done from Java side without C++ rebuild.
- **Rebuild**: None if toggled from Java. Header change if modifying default.

## Recommended Investigation Order

1. **S2 + S4: Capture cast reuse** — PRIME SUSPECT. Fast .cu rebuild. The pointer-identity
   assumption is fundamentally wrong for CUDA graph replay where buffer addresses are fixed
   but contents change. This directly explains the "stuck in a loop" symptom.

2. **S3: Row-vector fast path** — Easy to disable. Same .cu file, can combine with S2/S4 fix.

3. **S7: Attention neighborhood fusion** — Can test from Java without rebuild by calling
   `Nd4j.getEnvironment().setTritonFuseAttentionNeighborhoods(false)` (if wired through).
   Check if the Java Environment interface has this method.

4. **S6: Shape sync** — .cpp file, fast rebuild. Low probability but easy to rule out.

5. **S5: NDArray.hXX nullptr** — LAST. Header = slow rebuild. Lower probability since
   CUDA kernels don't use host buffers.

## Tests Written
- `MmulMixedPrecisionRegressionTest.java` — 6/6 passing (SameDiff FP32×FP16 matmul)
- `StaticKvDecodeRegressionTest#testAttnBiasUpdatesWithNativeDecodeInputs` — passing
