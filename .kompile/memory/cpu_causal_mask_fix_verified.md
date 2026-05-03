---
name: cpu-causal-mask-fix-verified
description: CPU causal mask fix CONFIRMED working — output changed from garbage to coherent echo. GDN layers not contributing, model echoes prompt words.
type: project
---

## CPU Causal Mask Fix — VERIFIED WORKING (May 2 2026)

### Fix Confirmed
MKL SDPA causal mask fix (input[8] reading) definitively changed output:
- Before: token 314 (` of`) — random/meaningless
- After: model echoes last words of prompt — coherent but semantically wrong

### Evidence: Model Is Echoing Prompt
- "capital of France?" → `' of of,'` (still 314 on fresh import — ssm_a fix not applied?)
- "poem about the sea" → `' and the and the the...'`
- "photosynthesis in one sentence" → `' in one in one in one...'`

Pattern: model repeats the last 1-2 words of the prompt. This is classic attention-only behavior where GDN (recurrent) layers are NOT contributing meaningful hidden state. The 6 attention layers can see and copy prompt tokens, but the 18 GDN layers are producing near-zero or identity outputs.

### Main Config Test vs Reference Test Discrepancy
- Main test (testQwen35Pipeline): PASSED SLOT_BY_SLOT, output `' in one...'` — uses cached SDZ model
- Reference test (testQwen35ReferencePrompts): FAILED, output `' of of,'` — uses fresh GGMLModelImport.importModel()
- The fresh import gets token 314 (old broken), the cached model gets 303 (new) — the ssm_a sign fix in LLaMAArchitecture.java may not be compiled into the nd4j-ggml jar used by the fresh import

### Next Investigation: GDN Layer Output
The GDN layers use gated_delta_rule op. If their state update is broken (e.g., state always zero, or gate always 1 causing no decay), they produce identity/near-zero outputs and the model degenerates to attention-only behavior.

Candidates:
1. GDN state initialization — is stateIn always zero for prefill? Should it be?
2. Gate decay values — are they reasonable (0.01-0.5) or degenerate (0 or 1)?
3. Beta values — is sigmoid(beta_proj) producing 0 (no update) or 1 (full update)?
4. Q/K/V L2 normalization — is this squashing values too much?
5. Q scaling by 1/sqrt(headDim) — is this double-scaling (GDN doesn't use attention)?

**Why:** Causal mask was the attention-layer root cause. Now GDN layers need separate investigation.
**How to apply:** Add diagnostic logging to GDN op to dump gate/beta/state values during prefill.


## 2026-05-02 11:10


## Additional Insights - May 2 follow-up

### Recheck the “cached SDZ vs fresh import” assumption

Current source for `TestQwen35Pipeline.loadModel()` imports GGUF fresh via `GGMLModelImport.importModel(...)` for both `testQwen35Pipeline` and `testQwen35ReferencePrompts`. The memory note that the main test uses a cached SDZ while the reference test uses fresh import may be stale or from a different loader path.

If the main and reference tests still diverge, first suspect one of these before assuming model-cache behavior:
- stale installed Maven artifact or test classpath still using an older `nd4j-ggml` jar,
- different `-Dqwen.*` properties between runs,
- shared JVM/static state across test methods,
- a different cached-model path not visible in the current `loadModel()` implementation.

Concrete verification to add: print or inspect the runtime class location for `org.nd4j.ggml.architecture.LLaMAArchitecture` during the failing test, then confirm the installed jar contains the expected `ssm_a` sign and SDPA causal-mask code. If source was changed but token 314 persists, reinstall the Java module that owns `LLaMAArchitecture` before retesting.

### Highest-priority GDN suspect: Q scaling after L2 normalization

`LLaMAArchitecture` currently L2-normalizes Q and K for GDN, then scales Q by `1 / sqrt(headDimKV)`. That scale is standard for softmax attention logits, but Gated Delta Rule uses the state readout `S_t^T q_t`; it is not a softmax dot-product. After Q has already been normalized, scaling it again by about `1/sqrt(128) ~= 0.088` can make the GDN residual contribution much too small.

This matches the observed post-SDPA failure mode: generation is no longer random token 314, but behaves like attention-only prompt copying or short echo loops. Treat this as a high-priority candidate to verify against the reference Qwen/GDN implementation. The test is not to bypass GDN, but to confirm whether the imported GDN equations should omit this attention-style Q scaling.

### Existing GDN tests are insufficient for this failure

`TestGatedDeltaRule` mostly checks shape, finite outputs, state changes, asymmetric dimensions, and dtype handling. It does not compare against a scalar/reference implementation or model-scale expected statistics. Passing it does not rule out sign, scale, recurrence-order, beta/gate, or transpose mistakes.

Add a focused parity test with small fixed tensors comparing the Java/native CPU op against a simple scalar implementation of:
- `prediction = state * k`,
- `delta = v - exp(gate) * prediction`,
- `state = exp(gate) * state + beta * outer(k, delta)`,
- `output = state^T * q`.

Also add a Qwen-scale diagnostic case using L2-normalized Q/K and realistic beta/gate ranges, because the current random tests will not catch a residual that is consistently attenuated in real model conditions.

### Use existing layer diagnostics before adding new logging

`TestQwenLayerDiagnostics#testLayerDiagnostics` is already the right entry point. Run it from `platform-tests` with tee and inspect the checkpoint stats around the first GDN layer:

```bash
cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && \
  mvn test -Dtest=TestQwenLayerDiagnostics#testLayerDiagnostics \
  2>&1 | tee /tmp/qwen-layer-diag.log
```

The important comparison is `embedded` vs `post_attn_0` vs `layer_out_0`. If `post_attn_0` is almost unchanged from `embedded`, or the GDN output norm is tiny, the bug is before the later full-attention layers and likely in GDN import/op scaling rather than SDPA.

Useful additions to the diagnostic checkpoints:
- `gdn_q_norm_0` before and after the extra `1/sqrt(headDimKV)` scale,
- `gdn_k_norm_0`,
- `gdn_beta_0`,
- `gdn_gate_decay_0` and `exp(gate_decay)`,
- `gdn_out_0`,
- `gdn_state_out_0` norm after prefill.

### Zero initial GDN state is expected

Starting prefill with zero `gdn_state_in` is not inherently wrong. The recurrent state should be built across the prompt tokens inside the GDN op. The key checks are whether `stateOut` becomes meaningfully non-zero after prefill and whether decode steps feed the previous layer state back in correctly.

So do not chase “state starts at zero” by itself. Chase “state stays near zero”, “state is not consumed on decode”, or “GDN output is tiny despite non-zero state”.

### Token 314 after a source fix points to classpath or split failure

If a run still emits token 314 (` of`) after the CPU SDPA causal-mask fix, separate two cases:

1. The SDPA fix is not actually on the runtime classpath. Confirm by checking the installed/test jar for the changed implementation and reinstalling the owning module.
2. SDPA is fixed, but fresh import is still missing another fix such as the `ssm_a` sign handling. Source currently uses `aLog.neg()` before `exp`, so a mismatch between source and runtime artifact is plausible.

The follow-up should make the runtime artifact visible rather than infer from output alone.

### Revalidate the QKV/GDN input path, not just the native op

The causal conv1d flip fix may be correct, but GDN depends on the full imported path: qkv projection, conv + SiLU, q/k/v split, reshape, Q/K norm, beta projection, and gate decay. Layer diagnostics should include q/k/v and beta/gate statistics before entering the native `gated_delta_rule` op. If those stats are already collapsed or repeated, the native op may be doing exactly what it was given.
