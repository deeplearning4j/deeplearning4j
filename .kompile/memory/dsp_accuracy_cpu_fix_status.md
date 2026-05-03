---
name: dsp-accuracy-cpu-fix-status
description: CPU Qwen3.5 accuracy fix status — softplus alpha=0 root cause found, build in progress May 2 2026
type: project
---

## CPU Qwen3.5 Accuracy Fix Status (May 2 2026 — Updated)

### END GOAL: TestQwen35Pipeline with nd4j-native must output text containing "France"
- Prompt: "What is the capital of France?"
- Model: Qwen3.5-0.8B GGUF
- Test: `cd platform-tests && mvn test -Dtest=TestQwen35Pipeline#testQwen35Pipeline -Dbackend.artifactId=nd4j-native -Dqwen.config=SLOT_BY_SLOT`

### ROOT CAUSE FOUND: OneDNN softplus alpha=0 → ALL inf output

**File:** libnd4j/include/ops/declarable/platform/mkldnn/softplus.cpp:87-88

OneDNN `eltwise_soft_relu` formula: `log(1 + exp(alpha * x)) / alpha`
- With `alpha=0`: numerator=log(2), denominator=0 → ALL outputs = +inf
- This kills ALL 18 GDN layers: softplus(inf) → gate_decay=-inf → exp(-inf)=0 → no state memory

**Evidence from debug+verbose trace (line 73808 of qwen-debug-verbose.log):**
- softplus input: `[2.92, -3.70, -2.35, ..., 6.40, -6.45, -0.17]` (normal values)
- softplus output: `[inf, inf, inf, ..., inf, inf, inf]` (ALL infinity)
- GDN gate decay input[4]: `[-inf, -inf, -inf, ..., -inf, -inf, -inf]` (ALL negative infinity)
- exp(-inf) = 0, so GDN state update = `0 * S + beta * k * delta` → no recurrent memory

**Fix:** Changed `alpha=0.f` to `alpha=1.f` in forward (line 88) and backward (lines 171, 176)

### VERIFIED FIXES IN CODE
1. MKL SDPA causal mask — reads input[8] correctly (verified: first token changed from 314 to 303)
2. OneDNN softplus alpha=1.0 (FIX APPLIED, BUILD IN PROGRESS)
3. L2-norm eps=1e-6 (was 1e-12) in LLaMAArchitecture.java:744,748
4. Prezero skip guard removed — unconditional prezeroSegmentOutputs
5. BFS kMaxBfs raised to 4096
6. causal_conv1d kernel flip — kk instead of K-1-kk
7. fusedRoPECached stride fix
8. invFreq heap allocation
9. nativeRangeSegments_ stale replay cleanup
10. rmsNormLinear FP32 accumulator
11. EOS token resolution
12. SameDiff.dup() DSP flag propagation
13. GGMLModelImport forInference()

### CURRENT STATUS: CPU build in progress with softplus fix
- Build: `/tmp/cpu-build-softplus-fix.log`
- After build: run `mvn test -Dtest=TestQwen35Pipeline#testQwen35Pipeline -Dbackend.artifactId=nd4j-native -Dqwen.config=SLOT_BY_SLOT`
- Expected: GDN layers now produce meaningful output → model generates "France"

### NEXT IF STILL FAILING
- Check if logsigmoid.cpp has same alpha=0 bug (also uses eltwise_soft_relu)
- Check GDN state write-back transpose logic
- Check if causal_conv1d output is correct now that softplus works
