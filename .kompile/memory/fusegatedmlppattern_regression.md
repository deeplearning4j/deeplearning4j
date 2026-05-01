---
name: FuseGatedMLPPattern regression
description: FuseGatedMLPPattern disabled — absorbing matmuls into fused_gemm_swiglu regressed throughput from ~51 to 48.2 tok/s
type: project
---

## FuseGatedMLPPattern Disabled (2026-04-28)

The 3-stage activation fusion chain (sigmoid*x→swish, swish*y→swish_mul, swish_mul(matmul,matmul)→fused_gemm_swiglu) regressed throughput from ~51 to 48.2 tok/s when fully enabled.

**Root cause:** FuseGatedMLPPattern (stage 3) absorbs two separate matmul ops into a single fused_gemm_swiglu op. This caused three problems:
1. fused_gemm_swiglu is MATMUL category in OpTraitTable.cpp — breaks elementwise Triton island chains
2. DSP can no longer batch the two separate matmul ops together
3. The C++ generic impl allocates temp buffers + does 2 sequential MmulHelper::mmul calls

**Fix:** Removed FuseGatedMLPPattern from ActivationFusionOptimizations.java. The chain now stops at swish_mul (BINARY_ELEMENTWISE), which stays in Triton islands and lets DSP batch the matmuls.

**Also fixed:** OptimizationHelper stale cache bug — HashMap caches initialized once, never updated when optimizers create new ops/variables. Added liveSd fallback with backfill to getVariable()/getOp().

**Result:** 50.98 tok/s (back to baseline). Stages 1+2 still fire (30+30 applications for SmolDocling).

**Files modified:**
- ActivationFusionOptimizations.java — removed FuseGatedMLPPattern inner class
- OptimizationHelper.java — added liveSd fallback for cache misses
- TestFusedGemmSwigluCorrectness.java — updated testGraphOptimizerFusion to expect swish_mul
