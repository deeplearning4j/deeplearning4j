---
name: onednn-softplus-alpha-zero-bug
description: "CRITICAL ROOT CAUSE: OneDNN softplus alpha=0 causes division by zero → all inf output, kills ALL GDN gate decay"
type: project
---

## OneDNN Softplus alpha=0 Bug — ROOT CAUSE OF GDN FAILURE (May 2 2026)

**File:** libnd4j/include/ops/declarable/platform/mkldnn/softplus.cpp:87-88

**Bug:** OneDNN `eltwise_soft_relu` formula is `log(1 + exp(alpha * x)) / alpha`. Code passed `alpha=0.f`, causing:
- Numerator: `log(1 + exp(0 * x)) = log(2)` (constant for all inputs)
- Denominator: `0`
- Result: `log(2) / 0 = +inf` for ALL inputs

**Evidence from debug trace:** softplus input was `[2.92, -3.70, -2.35, ..., 6.40, -6.45, -0.17]` (perfectly normal). Output was `[inf, inf, inf, ..., inf, inf, inf]`.

**Cascade:**
1. softplus outputs ALL `inf`
2. gate decay = `-exp(A_log) * inf = -inf`
3. exp(gate_decay) = exp(-inf) = 0
4. GDN state update: `S = 0 * S + beta * k * delta` — no memory, state overwritten every step
5. All 18 GDN layers produce near-zero output
6. Model degenerates to attention-only (6 layers), outputs prompt echo

**Fix:** Changed `alpha=0.f` to `alpha=1.f` on lines 87-88 (forward) and lines 171-176 (backward).

**Why:** The OneDNN API for soft_relu requires alpha=1 for standard softplus. alpha=0 is a degenerate case that produces infinity. The generic op (transform::SoftPlus) computes correctly but is overridden by the OneDNN platform helper.

**How to apply:** Any OneDNN eltwise_soft_relu usage MUST use alpha=1.0, never alpha=0.0.

**Status:** Fix applied, CPU build in progress
