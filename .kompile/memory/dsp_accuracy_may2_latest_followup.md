---
name: dsp-accuracy-may2-latest-followup
description: "May 2 latest follow-up: CPU iArgs fixed early stop but quality still wrong; causal_conv1d flip likely backwards; CUDA still zeros after replay fixes"
type: project
---

## DSP Accuracy May 2 Latest Follow-up

### Latest CPU result after iArgs fix

`/tmp/cpu-qwen-test.log` shows the iArgs stop-token parsing fix did resolve the one-token native decode problem:
- `nativeCount=48` for 50-token runs, not `nativeCount=1`.
- The output is still wrong.
- SLOT_BY_SLOT generated repetitive `of` / `.` tokens: `[314, 1020, 13, 314, 13, ...]`.
- Later configs generated high-diversity multilingual garbage and failed coherence.
- The full matrix eventually failed from physical memory pressure: `physicalBytes (58279M) > maxPhysicalBytes (57344M)`.

Interpretation: early stop was a real bug and is now past the first gate, but CPU semantic correctness is still broken in the prefill/GDN path. For verification, run one config at a time instead of the full matrix until correctness is stable.

### New high-priority CPU lead: causal_conv1d weight indexing likely repaired in the wrong direction

The current memory claimed PyTorch cross-correlation means `weight[0]` multiplies the current timestep. That is not true for Qwen3.5's exact reference expression:

`F.conv1d(mixed_qkv, weight.unsqueeze(1), padding=K-1)[:, :, :seq_len]`

With left padding and taking the first `seq_len` outputs, the current timestep aligns with `weight[K-1]`; older timesteps align with lower kernel indices. Existing `TestCausalConv1d` also expects `weight[K-1]` to multiply the current timestep.

Current CPU/CUDA helpers use `srcT = t - kk` with `weight[kk]`, so `weight[0]` multiplies current input. That likely corrupts Q/K/V in all 18 GDN layers and fits the post-softplus/post-iArgs failure mode.

Next concrete attempt:
1. Change CPU and CUDA causal_conv1d lag loop to use `weight[K - 1 - kk]` when `srcT = t - kk`.
2. Run `cd platform-tests && mvn test -Dtest=TestCausalConv1d -Dbackend.artifactId=nd4j-native 2>&1 | tee /tmp/causal-conv1d-regression-check.log`.
3. Reinstall native/backend artifacts as required, then run a single Qwen config only: `TestQwen35Pipeline#testQwen35Pipeline -Dbackend.artifactId=nd4j-native -Dqwen.config=SLOT_BY_SLOT -Dqwen.max.tokens=50` with tee.

### Q scaling suspicion demoted

Local HuggingFace Qwen3.5 reference explicitly does:
- optional Q/K L2 norm with eps `1e-6`, then
- `query = query * (1 / sqrt(query.shape[-1]))`.

So Q scaling after L2 norm should not be the first thing to remove. Keep it unless direct parity against the reference proves otherwise.

### GDN parity test still needed

Even after causal_conv1d is corrected, add a scalar/reference parity test for `gated_delta_rule`. The native formula looks algebraically equivalent to the HuggingFace recurrent fallback, but the test suite still does not prove reference parity for signs, transposes, dtype casts, or realistic gate/beta ranges.

### CUDA latest status

`/tmp/bench-cuda-debug.log` after the recent CUDA build still fails semantically:
- DSP validation: token match rate `2/10`, first divergent token at step 2: ref `12015`, test `0`.
- Benchmark: `[216, 49229, 30341, 0, 0, ...]` in debug run; latest non-debug was `[216, 49229, 0, 0, ...]`.
- Force-recapture matched OPTIMAL, so the remaining zero-token failure is probably not only stale replay cache. The diagnostic `NO-CAPTURE` path crashed, which blocks the cleanest isolation.

Next CUDA attempts:
1. Fix the `NO-CAPTURE` diagnostic crash so there is a non-captured baseline.
2. Compare logits/token IDs at step 2 between OPTIMAL, FORCE_RECAPTURE, and NO_CAPTURE.
3. Keep the causal mask off-by-one and composite replay gap fixes verified in the binary, but do not assume they are sufficient.
4. If causal_conv1d is fixed for CPU/CUDA, rebuild CUDA too because the same helper indexing exists in `causal_conv1d.cu`.

### Harness note

The full Qwen matrix is too memory-heavy while broken. Use single-config runs for root-cause verification, then restore matrix runs after SLOT_BY_SLOT is semantically correct.
