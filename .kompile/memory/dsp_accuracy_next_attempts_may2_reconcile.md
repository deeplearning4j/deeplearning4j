---
name: dsp-accuracy-next-attempts-may2-reconcile
description: May 2 reconciliation — updated with root causes found. Items 1,3,4 resolved. Remaining items for if fixes don't fully solve it.
type: project
---

# DSP Accuracy Reconciliation — UPDATED (May 2 2026 late session)

## STATUS: Two root causes found and fixed, awaiting rebuild verification

### RESOLVED ITEMS from original reconciliation

**Item 1 (CPU prefill isolation) — RESOLVED: ROOT CAUSE FOUND**
Ran the isolation test. With optimizer+DSP disabled, prefill still produced token 314. This proved the bug is at the op level. Subagent traced it to MKL SDPA never reading the causal mask at input[8]. Fix applied to sdpa.cpp.

**Item 2 (Fused-chain intermediate outputs) — RESOLVED: NO BUG**
Subagent investigated NativeDynamicShapePlan_slotexec.cpp fused chain handling. The `isOnlyConsumedOnce` guard at fusion-candidate time ensures no non-chain slot reads intermediate outputs. Zero-filled stub arrays at intermediate slots are never consumed by live computation. Correct by construction.

**Item 3 (CUDA frozen fast-path staging) — RESOLVED: ROOT CAUSE FOUND**
Subagent compared platformTryFrozenFastPath vs compositeReplay. Confirmed the frozen fast-path does H2D sync but never D2D-copies into staging buffers. Fix applied to NativeDynamicShapePlan_cuda.cu.

**Item 4 (markExternalInputVariable invalidation depth) — PARTIALLY RESOLVED**
The invalidateSegmentCaptures method was already added in prior work. The staging sync fix (item 3) is the missing piece — once staging buffers are refreshed, the captured graph reads correct data. The compile/capture invalidation was already deep enough.

**Item 5 (CUDA GDN/conv feedback) — NOT APPLICABLE TO CURRENT TARGET**
SmolDocling VLM is a standard transformer, does NOT use GDN/conv recurrent states. This fix is needed for CUDA Qwen3.5 but is a separate task, not blocking either current end goal.

### REMAINING ITEMS (only relevant if the two root cause fixes don't fully solve it)

**Item 6 (Dynamic-shape classification for variable externals)**
If CUDA VLM still produces wrong output after staging fix, check whether value-changing external inputs are treated as static by the frozen fast-path. External non-placeholder inputs are intentionally not classified as dynamic, which could leave slots reusing frozen context when values changed.

**Item 7 (CUDA silu/swish_mul alias detection)**
Low priority. The `output->buffer() == input->buffer()` check on CUDA might miss special-buffer aliasing. Not believed to be the current root cause but worth hardening.

**Item 8 (rms_norm_linear rank/view coverage)**
The rank>2 reshape fix is directionally correct. Verify with non-contiguous rank-3 inputs if other tests fail.

**Item 9 (Test harness reference-prompt assertion)**
The quality validator can pass obviously wrong text. Needs explicit France-prompt assertion. Not blocking — we verify manually.

**Item 10 (CUDA VLM diagnostics)**
After fix verification, if partially fixed, use DSP diagnostics (EXECUTE/MEMORY/VERIFY) to trace where tokens become wrong.

## PRACTICAL PRIORITY NOW
1. Wait for CPU build → test CPU Qwen → verify causal mask fix
2. Start CUDA build → test CUDA VLM → verify staging fix
3. If either still fails, revisit items 6-10 above
