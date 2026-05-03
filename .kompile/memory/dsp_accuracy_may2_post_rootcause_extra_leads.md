---
name: dsp-accuracy-may2-post-rootcause-extra-leads
description: "May 2 post-root-cause memory search addendum: what is obsolete, what remains worth trying if CPU/CUDA fixes fail verification"
type: project
---

# DSP Accuracy May 2 Post-Root-Cause Extra Leads (2026-05-02)

## Current memory status

The active May 2 memory now records two root causes as found and fixed, awaiting rebuild verification:

- CPU Qwen3.5: MKL SDPA platform impl never read the causal mask at input[8], so prefill attention was non-causal even with optimizer and DSP disabled. Fix is in `sdpa.cpp`, needs CPU rebuild/test.
- CUDA VLM SmolDocling: CUDA frozen fast-path skipped `ensureAndSyncStagingBuffers()`, so graph replay read stale capture-time staging data and produced zeros after the first two Java tokens. Fix is in `NativeDynamicShapePlan_cuda.cu`, needs CUDA rebuild/test.

No May 2 memory found a verified post-rebuild pass yet. The last recorded results are still pre-verification failures: CPU first token `314` and CUDA native-loop zeros.

## Do not keep chasing these unless new evidence appears

- The prezero outer guard is gone in current code. `NativeDynamicShapePlan_segments.cpp` now calls `prezeroSegmentOutputs(seg, stream)` unconditionally, and the platform function performs internal filtering. Old task-result warnings about the outer guard are historical.
- The `backfillCachedOutputShapes` state-based early return is gone in current `NativeDynamicShapePlan_slotexec.cpp`. Current code only returns when `cachedOutputShapes` is already nonempty, then backfills from live output arrays.
- Fused-chain intermediate slots were reinvestigated after the earlier warning. Current memory says `isOnlyConsumedOnce` prevents live downstream consumers of those intermediate stubs. Treat this as ruled out unless a new trace shows an intermediate fused-chain output slot is actually consumed outside the fused chain.
- CUDA GDN/conv recurrent-state feedback remains a real CUDA Qwen parity task, but the current CUDA target is SmolDocling VLM, which memory says is a standard transformer without GDN/conv state.

## Additional fallback leads if the two root-cause fixes fail verification

1. `markExternalInputVariable` still does not reset plan-level `compilationDone_`.

   Current code clears segment captures via `SegmentLifecycle::invalidateSegmentCaptures`, clears `compiledByBackend`, resets `argTableStable`, resets captured address keys, and sets `pointersStable_ = false`, but it does not set `compilationDone_ = false`. `phaseCompile()` is gated by `if (!compilationDone_)`. If CUDA still fails after the staging-sync fix, verify with DSP diagnostics that invalidated segments actually rebuild/recompile/capture after external inputs become variable. If not, try setting `compilationDone_ = false` in the `needsFullInvalidation` path or add a diagnostic assertion that no invalidated segment with empty `compiledByBackend` is skipped as if the plan were fully compiled.

2. CUDA `silu` and `swish_mul` alias detection still uses host `buffer()` equality.

   Current `llm_ops.cpp` still checks `output->buffer() == input->buffer()` / `output->buffer() == x->buffer()` / `output->buffer() == y->buffer()`. Raw May 2 task output ranked this high because CUDA DSP arrays can be device-authoritative while host buffers are stale or not representative of device aliasing. Active memory now downranks it because staging explains the all-zero token symptom. Keep it as the next CUDA correctness hardening item if staging fix removes zeros but output is still semantically wrong or logits look crushed. Prefer DataBuffer/specialBuffer-aware alias detection or object identity checks, plus CPU/CUDA in-place tests for `silu` and `swish_mul`.

3. If CPU still samples token `314` after the SDPA fix, stay on SDPA/bias application before jumping back to DSP.

   The fix added a three-way bias selection: KV-cache active reads input[8], empty-cache prefill with width > 8 reads input[8], legacy no-cache reads input[5]. If verification still fails, add reusable diagnostics around the MKL rank-4 path to confirm `attentionBias != nullptr`, branch taken (`useInPlaceKv` vs prefill), `biasLastDim`, `kSeqDim`, `biasSeqQ`, and that the prefill loop actually applies the causal rows before softmax. This is the shortest route to validating the root cause fix.

4. Confirm the patched CUDA fast path actually executes during the benchmark.

   The current fast path computes input address stability before calling `ensureAndSyncStagingBuffers()`. This is probably okay because `computeSegmentInputAddrKey()` intentionally skips variable external input addresses, so staged-vs-raw variable pointers do not affect the key. Still, if CUDA verification fails, check DSP diagnostics for `FROZEN_FAST_PATH: staging buffers synced` and the validation pair (`ref=... test=...`). If that message does not appear, the benchmark may be using composite replay or another path and the root-cause fix is not being exercised.

5. Preserve explicit semantic verification.

   Existing memories repeatedly note the quality validator allowed wrong text to pass. The verification target is not just Maven success: CPU must answer the France prompt with `France`, and CUDA must produce text about `mythic heroes`. Record the tee logs and the first few sampled token IDs after rebuild.
