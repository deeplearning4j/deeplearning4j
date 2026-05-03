---
name: dsp-accuracy-fix-plan
description: Complete ordered fix plan for DSP accuracy regression — 18 fixes applied May 2 2026 session
type: project
---

## DSP Accuracy Fix Plan (May 2 2026) — Complete Session Summary

All fixes applied this session. Last-good commit reference: `9bb2680e2b`.

**IMPORTANT BASELINE NOTE:** The May 1 baseline (`9bb2680e2b`) was NEVER correct — it also produced garbage output when tested properly. It only appeared to pass due to lenient thresholds in the test harness. The regression hunt was chasing a non-existent correct baseline.

---

### SHARED (CPU + CUDA) FIXES

**Fix 1: [DONE] REVERT prezero skip**
- File: `libnd4j/include/graph/impl/NativeDynamicShapePlan_segments.cpp:928-933`
- Restored unconditional `prezeroSegmentOutputs(seg, stream)` at line 933
- Removed guard `if (!(shapesFrozen_ && executeCount_ >= 2))`
- Ops with DATADEP trait (gather, concat, argmax) need prezero on every step

**Fix 2: [DONE] BFS kMaxBfs bump to 4096**
- File: `libnd4j/include/graph/impl/NativeDynamicShapePlan_slotexec.cpp:233`
- Changed kMaxBfs from 256 → 4096
- VLM models with 400+ slots: BFS was silently truncating, classifying live slots as non-dynamic → incorrectly frozen

**Fix 3: [DONE] rms_norm_linear reshape fix**
- File: `libnd4j/include/ops/declarable/generic/nn/llm_ops.cpp`
- `reshape(order, shape, false)` zero-copy view + directWrite guard + assign-back
- Without fix: rank>2 rms_norm_linear silently dropped results

**Fix 4: [DONE] SameDiff.dup() DSP flag propagation**
- File: `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/autodiff/samediff/SameDiff.java`
- dup() now propagates: graphExecutionMode, dspAutoCompileEnabled, dspNativeAutoCompileEnabled, dspFallbackToAutoIfTritonUnavailable, placementStrategy, customDevicePlacement

**Fix 5: [DONE] backfillCachedOutputShapes early-return guard**
- File: `libnd4j/include/graph/impl/NativeDynamicShapePlan_slotexec.cpp`
- Removed `if (slot.state_ >= NativeSlot::SlotState::SHAPE_CACHED) return;` guard
- phaseShapeInferenceOnly pre-pass was setting SHAPE_CACHED with prefill shapes; guard blocked shape correction on decode step
- Pre-existing guard `if (!slot.shapeCache.cachedOutputShapes.empty()) return;` remains and is sufficient

**Fix 6: [DONE] GraphOptimizer DCE BFS seed fix**
- File: `nd4j/.../GraphOptimizer.java`
- BFS seed now includes `sd.outputs()` (all declared outputs) AND follows `varControlDeps` edges
- Previous BFS only seeded from direct graph outputs — missed KV cache update ops (side-effecting, not in output list)
- DCE was incorrectly pruning KV cache scatter/update ops → model lost context between decode tokens

**Fix 7: [DONE] GenerationPipeline logits auto-discovery fallback**
- File: `nd4j/samediff-llm/src/main/java/org/eclipse/deeplearning4j/llm/generation/GenerationPipeline.java`
- Added runtime auto-discovery: scans all graph outputs for a variable with "logit" in the name when configured name not found
- Default changed "logits" → "lm_logits" — this fallback handles models that use either name
- Not the Qwen3.5 root cause (Qwen3.5 uses "lm_logits" correctly), but defensive fix for other models

---

### CPU-SPECIFIC FIXES

**Fix 8: [DONE] MKL SDPA prefill bias application**
- File: `libnd4j/include/ops/declarable/platform/mkldnn/sdpa.cpp:756-763`
- Applied `cblas_saxpy` bias (causal mask) between Q@K^T sgemm and softmax in prefill path (seqQ>1)
- Decode path (seqQ=1) already had bias correctly applied
- Without fix: CPU prefill attention was completely unmasked — model attended to future tokens

**Fix 9: [DONE] fusedRoPECached cos/sin stride fix**
- File: `libnd4j/include/helpers/cpu/fused_llm_ops.cpp`
- Replaced hardcoded stride offsets with `strideAt(dim)` calls throughout fusedRoPECached
- Non-contiguous cos/sin tensors (from slicing larger caches) were producing wrong rotary embeddings
- Note: Qwen3.5 uses non-cached FusedRoPE — this fix does not affect Qwen3.5 directly

**Fix 10: [DONE] invFreq stack → heap allocation**
- File: `libnd4j/include/helpers/cpu/fused_llm_ops.cpp`
- Changed `float invFreq[512]` stack buffer to `new float[halfRotate]` heap allocation
- Stack overflow risk when headDim > 1024 (headDim/2 > 512)

**Fix 11: [DONE] nativeRangeSegments_ stale replay cleanup**
- File: `libnd4j/include/graph/impl/NativeDynamicShapePlan_segments.cpp`
- Added `clearNativeRangeSegmentsForSlotRange()` call inside `invalidateForRebuild()`
- Stale lambda captures in persistent nativeRangeSegments_ map referenced freed slot arrays after invalidation

---

### CUDA-SPECIFIC FIXES

**Fix 12: [DONE] MmulHelper.cu dims.x/dims.y swap**
- File: `libnd4j/include/helpers/cuda/MmulHelper.cu:977`
- `getMMulDims()` returns `dim3(blocksPerGrid, threadsPerBlock, sharedMem)` — x=blocks, y=threads
- Call had `(dims.y, dims.x, dims.z, ...)` — args transposed
- On large matrices where blocksPerGrid > 1024, swapped value exceeded thread-per-block limit → CUDA error 9
- Became fatal with FP16 `forInference()` weights (triggers more large-matrix code paths)
- Fixed to `(dims.x, dims.y, dims.z, stream, ...)`

**Fix 13: [DONE] markExternalInputVariable + gpubackend markWarmupDone**
- Files: `NativeDynamicShapePlan.h`, `NativeDynamicShapePlan.cpp`, `NativeDynamicShapePlan_gpubackend.cpp`, `autoregressive_decode.cu`
- `markExternalInputVariable()` was calling `invalidateForRebuild()` which destroyed CUDA graphs on first decode call
- `gpubackend.cpp SegmentLifecycle::markWarmupDone(seg.exec)` fix prevents premature re-entry into warmup

**Fix 14: [DONE] static_cast → dynamic_cast for FunctionalReplayHandle**
- File: `libnd4j/include/graph/impl/NativeDynamicShapePlan_segments.cpp` (lines 966 and 1501)
- CPU graph replay uses FunctionalReplayHandle; CUDA uses CudaGraphReplayHandle
- Wrong static_cast on CUDA handle → undefined behavior (null dereference or silent corruption)
- Restored dynamic_cast at both sites

**Fix 15: [DONE] kvInPlaceWrite dtype REQUIRE_TRUE assertion**
- File: `onnx_multi_head_attention.cpp` + CUDA helper
- Mixed-type auto-cast at entry casts key/value to query dtype
- Added REQUIRE_TRUE assertion in BOTH CUDA and CPU implementations to catch dtype mismatches at runtime

**Fix 16: [DONE] AttentionFusion permute branch removal**
- File: `AttentionFusionOptimizations.java`
- Removed Permute branch from `extractQKFromMatmul()` at 4 locations
- Permute absorption was changing K layout assumptions — only true Transpose ops should be absorbed

**Fix 17: [DONE] autoregressive_decode.cu debug printf gating**
- File: `libnd4j/include/ops/declarable/helpers/cuda/autoregressive_decode.cu`
- Gated debug printfs behind `env_isVerbose()` check
- Unconditional printfs were writing to stdout on every token in production

---

### NOT-A-BUG CONFIRMATIONS

**N1: onnx_mha syncToDevice removal — NOT A BUG**
- File: `onnx_multi_head_attention.cpp`
- Investigated this session: removed syncToDevice calls were genuine no-ops (arrays already device-current)
- Do NOT revert

**N2: OpenVINO before OneDNN ordering — NOT A REGRESSION**
- File: `NativeDynamicShapePlan_segments.cpp`
- Confirmed: ordering was identical in last-good commit `9bb2680e2b`
- Was pre-existing; did not change between baseline and HEAD

---

### POSITIVE FIXES (DO NOT REVERT)

**P1: rmsNormLinear_ float32 GEMM accumulator**
- File: `libnd4j/include/ops/declarable/helpers/cpu/rms_norm.cpp`
- Float32 accumulator prevents zeros on CPUs without AMX-FP16
- This is a POSITIVE fix — resolved FP16 accuracy issues on CPU

**P2: GGMLModelImport forInference() weights**
- File: `nd4j/nd4j-ggml/src/main/java/org/nd4j/ggml/GGMLModelImport.java`
- Kept uncommitted change — marks GGML weights as inference-mode for proper FP16 handling

---

### FIXES FROM CPU TEST ROUND 2 (May 2 2026 continued)

**Fix 18: [DONE] MKL SDPA prefill bias buffer overread**
- File: `libnd4j/include/ops/declarable/platform/mkldnn/sdpa.cpp:756-763`
- `cblas_saxpy(seqQ * seqKV, ...)` read past end of `[1,1,1,seqKV]` bias buffer
- Caused non-deterministic prefill logits (heap junk added to attention scores)
- Fixed: row-by-row application with `biasF32->sizeAt(-2)` broadcast detection

**Fix 19: [DONE] EOS token mis-resolution (premature decode termination)**
- File: `nd4j/.../HuggingFaceTokenizer.java:522` — added `encode(tokenStr, true)` fallback
- File: `nd4j/.../GenerationPipeline.java` — added `buildStopTokenIds()` with `eosTokenId < 100` guard
- `resolveSpecialToken("<|im_end|>")` with `addSpecialTokens=false` BPE-fragmented to wrong ID
- GGUF metadata has correct `eos_id=248046`, but tokenizer resolved a different (low) ID
- SLOT_BY_SLOT generated only 3 tokens because token 11 (comma) matched mis-resolved EOS

---

## Fix Count Summary

| Category | Count |
|---|---|
| Shared CPU+CUDA fixes | 7 |
| CPU-specific fixes | 6 |
| CUDA-specific fixes | 6 |
| Not-a-bug confirmations | 2 |
| Positive fixes (keep) | 2 |
| **Total changes addressed** | **20 + 2 confirmations** |

## Next Steps

1. **CPU rebuild in progress** — SDPA bias fix (#18) needs native recompile. Java fix (#19) already installed.
2. **CUDA build needed** — first attempt OOM'd. Will restart after CPU build completes.
3. **shapeFunctionOverride validation gating** — still open, acceptable for frozen decode. Not a blocking regression.
