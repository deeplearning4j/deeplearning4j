---
name: cpu-sdpa-causal-mask-verified-correct
description: CPU SDPA causal mask flow verified correct — NOT the cause of 'ofof.' output
type: project
---

## CPU SDPA Causal Mask — Verified Correct (May 2 2026)

Investigation confirmed all aspects of the causal mask flow are working:
1. Mask built correctly as [1,1,prefillLen,maxKvLen] with -1e9 at future positions
2. Correctly passed as input[8] to dot_product_attention_v2
3. Correctly detected in platform impl (!hasKvCache && width>8)
4. Correctly sliced with .dup() to [1,1,prefillLen,prefillLen]
5. Correctly applied per-row in MKL prefill loop (cblas_saxpy)
6. Decode path bias also correctly applied

**Not the cause of 'ofof.' output.**

**Why:** Eliminated causal mask as potential root cause for CPU accuracy regression
**How to apply:** Focus CPU accuracy investigation on GDN layer computation, not attention masking
