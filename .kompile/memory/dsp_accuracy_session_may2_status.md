---
name: dsp-accuracy-session-may2-status
description: May 2 session status — TWO ROOT CAUSES FOUND (CPU causal mask + CUDA staging sync). Fixes applied, builds in progress.
type: project
---

## DSP Accuracy Session Status (May 2 2026) — ROOT CAUSES FOUND

### End Goals (UNCHANGED)
- **CPU**: TestQwen35Pipeline must output text containing "France" (Qwen3.5-0.8B)
- **CUDA**: run-benchmark.sh --tokens 250 must output text about "mythic heroes" (SmolDocling VLM)

### Current State: BOTH ROOT CAUSES IDENTIFIED, FIXES APPLIED, BUILDS IN PROGRESS

#### CPU Root Cause: MKL SDPA never reads causal mask at input[8]
- **File:** sdpa.cpp PLATFORM_IMPL(dot_product_attention_v2, ENGINE_CPU)
- **Bug:** Bias detection only checks input[5] as bias when input[6] is empty. When KV cache is active (input[5]=keyCache, input[6]=valueCache), input[6] is never empty, so the causal mask at input[8] is never read. The generic op reads input[8] correctly but the MKL platform override runs instead.
- **Impact:** All 6 attention layers run non-causal during prefill. Confirmed: even with DSP+optimizer disabled, first token = 314 (' of').
- **Fix:** Three-way bias detection for KV-cache-active, prefill-empty-cache, and legacy paths.
- **Status:** Code edited, CPU build in progress.

#### CUDA Root Cause: Frozen fast-path skips ensureAndSyncStagingBuffers()
- **File:** NativeDynamicShapePlan_cuda.cu, platformTryFrozenFastPath
- **Bug:** Does H2D syncToSpecial() but never D2D-copies into staging buffers. CUDA graphs read from staging buffer addresses. Composite path calls ensureAndSyncStagingBuffers() correctly.
- **Impact:** All native-loop tokens are 0 because graph replay reads stale capture-time data.
- **Fix:** Added ensureAndSyncStagingBuffers() call matching composite path pattern.
- **Status:** Code edited, CUDA build waiting on CPU build to finish (OOM risk).

### Investigation Methods That Led to Root Causes
1. **CPU isolation test**: Ran with `-Dqwen.graph.optimizer=false -Dqwen.dsp=false -Dqwen.config=SLOT_BY_SLOT` — still token 314. Proved op-level bug, not DSP/optimizer.
2. **Subagent investigation of MKL SDPA**: Traced input layout from DotProductAttentionV2.java through generic op (correct) to MKL platform impl (broken). Found input[8] never read.
3. **Subagent investigation of frozen fast-path**: Compared platformTryFrozenFastPath vs compositeReplay — found missing ensureAndSyncStagingBuffers() call.
4. **DSP validation preflight**: `ref=12015 test=0` at step 2 confirmed SLOT_BY_SLOT produces correct tokens but optimized path produces zeros.

### Ruled Out This Session
- GDN gate sign convention: correct (Java negative → C++ exp → 0-1 decay)
- GDN beta projection: correct per reference
- Fused-chain intermediate outputs: isOnlyConsumedOnce guard is correct
- ssm_a sign convention: fix applied (aLog.neg()), changed DSP prefill tokens but not SLOT_BY_SLOT — deeper root cause found

### Total Fixes in Working Tree (25+)
See individual CPU/CUDA status files for complete lists. All prior fixes are correct and preserved.
