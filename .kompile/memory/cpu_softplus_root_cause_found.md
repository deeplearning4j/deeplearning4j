---
name: cpu-softplus-root-cause-found
description: "CPU ROOT CAUSE: OneDNN softplus alpha=0 → all inf, kills GDN gate decay. Fix: alpha=1.0. Build in progress."
type: project
---

## CPU Root Cause: OneDNN softplus alpha=0 (May 2 2026)

**File:** libnd4j/include/ops/declarable/platform/mkldnn/softplus.cpp:87-88
**Bug:** `eltwise_soft_relu` with alpha=0.f → log(1+exp(0*x))/0 = log(2)/0 = +inf for ALL inputs
**Evidence:** debug+verbose trace showed softplus input normal [-11..+8], output ALL inf
**Cascade:** softplus=inf → gate=-inf → exp(-inf)=0 → GDN state zeroed every step → 18 GDN layers dead → attention-only echo
**Fix:** alpha=0.f → alpha=1.f (forward line 88, backward lines 171,176)
**Status:** CPU build in progress with fix

Also fixed: L2-norm eps 1e-12→1e-6 in LLaMAArchitecture.java:744,748

**CUDA causal mask off-by-one:** CONFIRMED present and correct (kvJustWritten = currentPosition-1). Not the CUDA issue.
