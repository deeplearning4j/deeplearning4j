---
name: rms-norm-linear-fusion-multi-consumer-bug
description: FuseRMSNormLinear removes rms_norm var when multiple matmuls consume it (Qwen Q/K/V shared norm)
type: project
---

## FuseRMSNormLinear Multi-Consumer Bug

**Root cause:** `FuseRMSNormLinearPattern` in `NormalizationFusionOptimizations.java` removes `rms_norm_N` output variable even when multiple matmuls consume it (Q/K/V projections sharing a normalization in Qwen3.5).

**Why:** The `hasOnlyConsumer` check used `helper.getVariable()` which returns cached data from the start of the optimization pass. Prior optimizations may have changed consumer lists, making the cache stale. The live graph (`sd.getVariables()`) must be used instead.

**Error:** `Op 'matmul_186' input[0] references missing variable 'rms_norm_78'`

**Fix (applied):**
1. Pre-check: Use `sd.getVariables().get(rmsNormOutVar)` (live graph, not helper cache) to verify exactly 1 consumer before fusing
2. Safety net: Before `removeVariable(rmsNormOutVar)`, re-check `remainingUsers` from the live graph

**File:** `nd4j/.../optimize/optimizations/NormalizationFusionOptimizations.java` lines 675-693 and 732-744

**How to apply:** When debugging optimizer validation errors post-fusion, always check if `hasOnlyConsumer` uses stale helper cache vs live graph. Multi-consumer shared normalization is common in GQA models.
