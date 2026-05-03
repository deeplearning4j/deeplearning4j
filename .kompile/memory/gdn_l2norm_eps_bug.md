---
name: gdn-l2norm-eps-bug
description: "CRITICAL: GDN L2-norm eps=1e-12 vs reference eps=1e-6 — causes near-zero vector amplification corrupting GDN state"
type: project
---

## GDN L2-Normalization Epsilon Bug (May 2 2026)

**File:** nd4j/nd4j-ggml/src/main/java/org/nd4j/ggml/architecture/LLaMAArchitecture.java:744,748

**Bug:** L2 normalization used `eps=1e-12` instead of reference `eps=1e-6`
- Reference (HuggingFace modeling_qwen3_5.py): `l2norm(query, dim=-1, eps=1e-6)` using `rsqrt(sum(x*x) + 1e-6)`
- Ours: `sd.math.sqrt(qNormSq.add(1e-12))` — 6 orders of magnitude too small

**Impact:** Near-zero Q/K vectors get amplified ~1e6x instead of clamped at ~1e3 norm. This corrupts the GDN state matrix S on early tokens, causing all 18 GDN layers to produce near-zero output. The model degenerates to attention-only (6 layers), producing echo/repetition of prompt words instead of meaningful generation.

**Fix:** Changed both `1e-12` to `1e-6` at lines 744 and 748. nd4j-ggml jar reinstalled.

**Why:** The epsilon acts as a floor for the L2 norm denominator. With 1e-12, vectors with sum-of-squares ~1e-12 get divided by ~1e-6, amplifying them 1e6x. The reference 1e-6 limits amplification to ~1e3x.

**How to apply:** Any L2 normalization for GDN Q/K must use eps=1e-6 to match HuggingFace reference. This is NOT a general epsilon — it's specific to the GDN kernel's expectation.

**Status:** Fix applied, nd4j-ggml jar installed, CPU test running
