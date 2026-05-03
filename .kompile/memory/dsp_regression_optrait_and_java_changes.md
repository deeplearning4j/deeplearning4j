---
name: dsp-regression-optrait-and-java-changes
description: OpTraitTable, DeclarableOp, SameDiff, and other Java-side changes since 9bb2680e2b
type: project
---

## OpTraitTable & Java-Side Changes Since 9bb2680e2b (May 2 2026)

### OpTraitTable.cpp — BOTH

**rms_norm_linear NORM→MATMUL (COMMITTED):**
- Both categories include FULLY_WRITING — no functional change for prezero
- MATMUL may have different segment compilation behavior (e.g., cuBLAS dispatch)

**gather/concat gained DATADEP trait (COMMITTED):**
- Suppresses isFullyWriting → forces needsZeroedOutput=true
- Correct behavior: these ops have data-dependent output sizes
- Combined with prezero skip regression → these ops get NO prezero = stale data

**argmax/argmin gained DATADEP (COMMITTED):**
- Same effect as gather/concat — forces needsZeroedOutput=true

**reshape/reshape_no_copy gained DATADEP (COMMITTED):**
- Questionable: reshape always fully writes its output buffer
- DATADEP forces unnecessary prezero for reshape ops

**xw_plus_b added as MATMUL (COMMITTED):**
- Correct categorization

**Bulk additions for non-LLM ops (COMMITTED):**
- Many ops added to trait table — broadens coverage

### DeclarableOp.cpp — BOTH

**shapeFunctionOverride gate at line 993 (COMMITTED):**
- Skips: validateNonEmptyInput, validateArguments, validateDataTypes, prepareOutputs
- Helper dispatch NOT bypassed (guard was reverted — positive)
- timingRecord storage: zero-init → alignas char array (perf, no functional change)

### SameDiff.java — BOTH

**dup() DSP flag propagation (COMMITTED BUG, UNCOMMITTED FIX):**
- Bug: dup() via SDNB doesn't propagate DSP config flags
- Fix propagates: graphExecutionMode, dspAutoCompileEnabled, dspNativeAutoCompileEnabled, dspFallbackToAutoIfTritonUnavailable, placementStrategy, customDevicePlacement
- Without fix: dup()'d SameDiff may have DSP disabled or in wrong mode

### DynamicShapePlanExecutor.java — BOTH

**KV max-allocation now ACTIVE (COMMITTED):**
- Was commented out — now active in normal execution path
- setShapeOnlyMode and configureMaxAllocationForKvCache overloads added
- If max-allocation miscalculates, KV cache may be undersized

### GraphExecutionMode.java — BOTH

**SHAPE_INFERENCE_ONLY(18) added (COMMITTED):**
- New execution mode for shape-only pre-pass
- Used by phaseShapeInferenceOnly in C++

### GGMLModelImport.java — BOTH

**forInference() defaults (UNCOMMITTED):**
- importModel defaults changed to ConversionOptions.forInference()
- CPU weights converted to FP32
- Correct for inference — prevents FP16 overflow on CPUs without AMX

### NDArray changes — BOTH

**validateIntegrity() removed from hot paths (COMMITTED):**
- Reduces corruption detection in exchange for performance
- If corruption occurs, it's detected later (or not at all)

**NDArray constructor canary gated to debug mode (COMMITTED):**
- Canary values only written in debug builds
- Release builds don't detect use-after-free or buffer overruns via canary

**ConstantShapeBuffer alignment check (COMMITTED — POSITIVE):**
- Detects shape info buffer corruption early
- Positive defensive check

**Why:** These changes control op behavior (traits), validation (DeclarableOp), and Java-side configuration (SameDiff). They set the stage for how the native plan executes.
**How to apply:** The DATADEP trait additions are correct but depend on prezero being unconditional. The SameDiff.dup() fix must be kept.
