---
name: dsp-accuracy-regression-shared
description: Shared CPU+CUDA DSP accuracy regression issues identified May 2 2026
type: project
---

## Shared CPU+CUDA DSP Accuracy Regression Issues (May 2 2026)

These issues affect BOTH backends. Investigation of commits April 29 - May 2 (since last-good `9bb2680e2b`).

### TIER 1 CRITICAL

**1. [DONE] prezeroSegmentOutputs skip — UNCOMMITTED REGRESSION**
- File: `NativeDynamicShapePlan_segments.cpp:928-933`
- **FIXED (earlier in session)**: Unconditional `prezeroSegmentOutputs(seg, stream)` restored at line 933. The guard `if (!(shapesFrozen_ && executeCount_ >= 2))` has been removed.
- Ops with DATADEP trait (gather, concat, argmax, reshape) have `needsZeroedOutput=true`
- When prezero is skipped, these ops read stale data from previous step

**2. [DONE] BFS kMaxBfs=256 truncation — COMMITTED BUG, UNCOMMITTED FIX**
- File: `NativeDynamicShapePlan_slotexec.cpp:233`
- **FIXED (earlier in session)**: kMaxBfs bumped to 4096 at line 233
- VLM models with 400+ slots: BFS silently truncates, returns false when true
- Slots incorrectly classified as non-dynamic → incorrectly frozen → stale outputs

**3. [DONE] rms_norm_linear reshape fix — UNCOMMITTED**
- File: `llm_ops.cpp` (rms_norm_linear op)
- **FIXED (earlier in session)**: `reshape(order, shape, false)` (zero-copy view) + directWrite guard + assign-back
- Without fix: rank>2 rms_norm_linear silently drops results

**4. [DONE] backfillCachedOutputShapes early-return guard (COMMITTED)**
- File: `NativeDynamicShapePlan_slotexec.cpp`
- **FIXED this session**: Removed `if (slot.state_ >= NativeSlot::SlotState::SHAPE_CACHED) return;` guard
- phaseShapeInferenceOnly pre-pass sets SHAPE_CACHED with prefill shapes; the guard blocked shape correction when actual execution produced different shapes (decode vs prefill)
- Pre-existing guard `if (!slot.shapeCache.cachedOutputShapes.empty()) return;` remains and is sufficient

**5. [DONE] SameDiff.dup() DSP flag propagation — COMMITTED BUG, UNCOMMITTED FIX**
- File: `SameDiff.java`
- **FIXED (earlier in session)**: dup() now propagates all DSP flags: graphExecutionMode, dspAutoCompileEnabled, dspNativeAutoCompileEnabled, dspFallbackToAutoIfTritonUnavailable, placementStrategy, customDevicePlacement

### TIER 2 HIGH

**6. [DONE] silu/swish_mul in-place aliasing — FIXED in HEAD (commit 529e26f702)**
- File: `llm_ops.cpp:312` (silu), `llm_ops.cpp:791,798,807` (swish_mul)
- Bug: when output aliases input, computed sigmoid(x)^2 instead of x*sigmoid(x)
- Fix: alias guard `if (output->buffer() == input->buffer())` before choosing safe vs fast path
- Minor gap: uses ->buffer() not ->dataBuffer(), could miss GPU-only aliasing

**7. shapeFunctionOverride validation skip at executeCount_ >= 3 (COMMITTED)**
- File: `NativeDynamicShapePlan_slotexec.cpp` (3 sites: lines 1752, 2670, 4016)
- File: `DeclarableOp.cpp:993`
- Skips validateNonEmptyInput, validateArguments, validateDataTypes, prepareOutputs
- Generally safe for frozen decode (shapes don't change), but masks bugs during development
- **STATUS**: Still open — acceptable for frozen decode, but tracked for awareness

**8. [FIXED] GraphOptimizer DCE pass**
- File: `GraphOptimizer.java`
- New DCE (dead code elimination) pass was seeding BFS only from the graph's direct output variables
- KV cache update ops are not in the graph output list — they're side-effecting ops that update in-place
- DCE incorrectly pruned KV cache scatter/update ops, causing model to lose context between tokens
- **FIXED this session**: BFS seed now includes `sd.outputs()` (all declared outputs) AND follows `varControlDeps` edges — KV cache update ops are now reachable and preserved

**9. [FIXED] GenerationPipeline logits name change**
- File: `GenerationPipeline.java`
- Default logits name had changed "logits" → "lm_logits"
- If model doesn't have "lm_logits" output, logits lookup failed silently
- **FIXED this session**: Added runtime auto-discovery fallback — scans all graph outputs for a variable with "logit" in the name when the configured name is not found
- **Note**: This was NOT the root cause of Qwen3.5 CPU garbage; Qwen3.5 uses "lm_logits" which already matched. Fix is defensive for other models.

### TIER 3 MEDIUM

**10. gather/concat DATADEP trait (COMMITTED)**
- File: `OpTraitTable.cpp`
- gather and concat gained DATADEP trait — suppresses isFullyWriting, forces needsZeroedOutput=true
- Correct behavior, but combined with prezero skip (#1), these ops get no prezero
- Fix #1 resolves this interaction

**11. rms_norm_linear NORM→MATMUL trait change (COMMITTED)**
- File: `OpTraitTable.cpp`
- Changed from NORM to MATMUL category — both include FULLY_WRITING so no functional change
- But MATMUL may have different segment compilation behavior

**12. [DONE] NormalizationFusionOptimizations stripTrivialOps (FIXED in HEAD)**
- File: `NormalizationFusionOptimizations.java`
- Was stripping through reshape ops — caused wrong-shaped fusion inputs
- Fixed: restricted to cast/identity only

---

**Why:** These issues affect both CPU and CUDA because they're in shared DSP infrastructure (slot execution, segment management, shape caching) or shared op implementations.

**Status (May 2 2026 session end):**
- DONE: Items 1, 2, 3, 4, 5, 6, 8, 9, 12
- OPEN (acceptable): Item 7 (shapeFunctionOverride validation skip — safe for frozen decode)
- RESOLVED INTERACTION: Item 10 (resolved by Fix 1)

**IMPORTANT — May 1 baseline was NEVER correct.** The "last-good" commit `9bb2680e2b` also produced garbage output when properly tested — it only passed due to lenient thresholds in the test harness. The regression hunt assumed a correct baseline that did not exist.

**CPU rebuild required.** Most C++ fixes in this session have not yet been compiled into the CPU binary. A full CPU rebuild is needed before final accuracy verification.
