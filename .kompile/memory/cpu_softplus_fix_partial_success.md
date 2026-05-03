---
name: cpu-softplus-fix-partial-success
description: "CPU softplus alpha=1 fix: tokens now real words but wrong ('ofof.' not France) — partial success May 2 2026"
type: project
---

## CPU Qwen3.5 Softplus Fix — Partial Success (May 2 2026)

### Fix Applied
OneDNN softplus alpha changed from 0.f to 1.f in:
- Forward: softplus.cpp:91 (alpha=1.f, beta=0.f)
- Backward: softplus.cpp:176,180 (alpha=1.f, beta=0.f)

### Result
- **Before fix**: All garbage tokens, GDN layers completely dead (inf from softplus → exp(-inf)=0 → no state memory)
- **After fix**: Real tokens generated: [314, 1020, 13] → text=' ofof.'
- nativeCount=1 — only 1 native decode token generated (13), suggests early stop or low token count
- Test PASSED quality validator but output does NOT contain "France"
- 0.17 tok/s on CPU (expected slow, CPU SLOT_BY_SLOT mode)

### Analysis
- Softplus fix is CORRECT and necessary — GDN layers are now alive
- BUT output 'ofof.' is wrong — model not producing coherent France response
- Possible remaining issues:
  1. Only 3 tokens generated total (prefill+warmup+1 native). `-Dqwen.tokens=20` should give 18 native tokens
  2. Token 13 may be hitting EOS check incorrectly
  3. MKL SDPA prefill bias (causal mask) still not applied in prefill path — plan item B1
  4. L2 norm eps 1e-6 fix needs verification
  5. Need more tokens to assess actual quality

### Next Steps
- Run with more tokens (50+) and check if early stop is the issue
- Verify MKL SDPA prefill causal mask application (sdpa.cpp:724-760)
- If still wrong after more tokens, investigate MKL SDPA bias path

**Why:** Softplus was the single biggest CPU bug (killed 18/24 layers), but may not be the only one
**How to apply:** Continue CPU investigation with SDPA prefill bias as next suspect
