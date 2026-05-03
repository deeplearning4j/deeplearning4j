---
name: dsp-accuracy-regression-cpu
description: CPU-specific DSP accuracy regression issues identified May 2 2026
type: project
---

## CPU-Specific DSP Accuracy Regression Issues (May 2 2026)

Investigation of commits April 29 - May 2 (since last-good commit `9bb2680e2b`).

### TIER 1 CRITICAL

**1. [FIXED] MKL SDPA prefill drops attention bias — causal masking absent**
- File: `libnd4j/include/ops/declarable/platform/mkldnn/sdpa.cpp`
- Decode path (seqQ=1, lines 696-708): `biasPtr` applied via `cblas_saxpy` BEFORE softmax ✓
- Prefill path (seqQ>1, lines 737-776): `biasPtr` extracted but NEVER applied between Q@K^T and softmax ✗
- `biasF32` was deleted at line 728 before the prefill loop began
- Result: CPU prefill attention scores were completely unmasked — model attended to future tokens
- **FIXED this session**: `cblas_saxpy` bias application added between Q@K^T (sgemm) and softmax in prefill loop

**2. [NOT A REGRESSION] OpenVINO moved BEFORE OneDNN in CPU backend chain**
- File: `NativeDynamicShapePlan_segments.cpp`
- **CONFIRMED NOT A REGRESSION**: Ordering was ALREADY the same in last-good commit `9bb2680e2b`
- OpenVINO-before-OneDNN was the pre-existing ordering — not introduced between 9bb2680e2b and HEAD
- Do NOT chase this as a regression cause

**3. [FIXED] Persistent nativeRangeSegments_ for CPU graph replay (stale replay)**
- File: `NativeDynamicShapePlan_segments.cpp`
- CPU graph replay uses `FunctionalReplayHandle` with lambda captures
- Persistent `nativeRangeSegments_` map caused replayed lambdas to reference stale slot arrays after invalidation
- **FIXED this session**: `clearNativeRangeSegmentsForSlotRange()` added to `invalidateForRebuild()` — clears stale entries on every invalidation

### TIER 2 HIGH

**4. [FIXED] fusedRoPECached cos/sin stride assumption**
- File: `libnd4j/include/helpers/cpu/fused_llm_ops.cpp`
- Rewritten fusedRoPECached had hardcoded stride offsets instead of using `strideAt()`
- Non-contiguous cos/sin tensors (e.g., from slicing a larger cache) produced wrong rotary embeddings
- **FIXED this session**: Replaced hardcoded stride offsets with `strideAt(dim)` calls throughout fusedRoPECached
- **Note**: Qwen3.5 uses non-cached FusedRoPE (not fusedRoPECached), so this fix does NOT affect Qwen3.5 accuracy

**5. [FIXED] invFreq stack buffer overflow risk**
- File: `libnd4j/include/helpers/cpu/fused_llm_ops.cpp`
- `float invFreq[512]` on stack — headDim > 1024 would overflow (headDim/2 > 512)
- Current models use headDim=128, but future models may exceed this
- **FIXED this session**: Changed to heap allocation `new float[halfRotate]` with proper delete[]

**6. [POSITIVE FIX — DO NOT REVERT] rmsNormLinear_ float32 GEMM on CPUs without AMX-FP16**
- File: `libnd4j/include/ops/declarable/helpers/cpu/rms_norm.cpp`
- Uses float32 accumulator for GEMM — prevents zeros on CPUs without AMX-FP16 instructions
- This is a POSITIVE fix that resolved FP16 accuracy issues on CPU

### TIER 3 MEDIUM

**7. [NOT A REGRESSION] Cascade failure silent fallback**
- File: `NativeDynamicShapePlan_segments.cpp`
- **CONFIRMED NOT A REGRESSION**: Code already throws AND sets `compilationFailed = true` — both behaviors were present in last-good commit `9bb2680e2b`
- Silent fallback path already existed; the throw path was never removed

**Why:** These are CPU-specific because they involve MKL/OneDNN/OpenVINO backends, CPU graph replay (FunctionalReplayHandle), and CPU-specific helper implementations.

### FIXES FROM CPU TEST ROUND 2 (May 2 2026 continued)

**8. [FIXED] MKL SDPA prefill bias buffer overread**
- File: `libnd4j/include/ops/declarable/platform/mkldnn/sdpa.cpp:756-763`
- `cblas_saxpy(seqQ * seqKV, ...)` assumed bias was `[1,1,seqQ,seqKV]` but actual shape is `[1,1,1,seqKV]`
- Read `(seqQ-1) * seqKV` floats of uninitialized heap memory → non-deterministic prefill logits across runs
- **FIXED this session**: Row-by-row application with `biasF32->sizeAt(-2)` broadcast detection

**9. [FIXED] EOS token mis-resolution causing premature decode termination**
- File: `HuggingFaceTokenizer.java:522` — `resolveSpecialToken("<|im_end|>")` with `addSpecialTokens=false`
- BPE-fragmented the special token string to a low common token ID (e.g., 11 = comma)
- GGUF metadata has correct `eos_id=248046`, but tokenizer resolved wrong ID
- SLOT_BY_SLOT: only 3 tokens generated because token 11 matched mis-resolved EOS → loop exited
- **FIXED this session**: Added `encode(tokenStr, true)` fallback + `eosTokenId < 100` guard in `buildStopTokenIds()`

**CPU BUILD STATUS (May 2 2026):** CPU rebuild in progress with SDPA fix #8. Java fix #9 already installed via `mvn install -DskipTests -pl nd4j/samediff-llm`.
