# HANDOFF — Fixed-Buffer LLM Decode: reuse forward-fix + prefill accuracy bug

**Date:** 2026-07-07 · **Branch:** `ag_new_release_updates_2` · **Model under test:** Qwen3.5-0.8B (Q4_K_M), GDN-hybrid (Gated-DeltaNet + short-conv recurrent layers interleaved with attention layers).

Persistent memory note with the full trail: `~/.claude/.../memory/dsp-fixed-buffer-decode-demotion.md`.

---

## TL;DR — where this stands

There are **THREE independent things** on the fixed-buffer decode path. Do not conflate them.

1. **DONE + COMMITTED** — `d54e4fbe1d` (the "unseal", completes #52). Fixed the DSP demotion **crash**. Not in scope anymore.

2. **DONE (uncommitted) but NOT bit-exact — the buffer-reuse forward-fix ("Design A").** Implemented in `GenerationPipeline.java` + `InGraphKvState.java`. It **works structurally**: reuses the frozen decode plan across `generate()` calls, eliminates the ~130 s/gen re-warm (measured ~15 s for 2 generates). BUT `reuse != fresh` token-for-token, because on reuse the warmup decode step runs as a **frozen replay** while on fresh it runs **slot-by-slot** → slightly different GDN state → tips a sensitive argmax. User **deferred** bit-exactness of this ("do A, then root-cause fixed-vs-variable first"). Candidate fix if/when you return to it: make the fixed-buffer native decode start from the **frozen-computed** recurrent state on BOTH fresh and reuse (on fresh, re-run the warmup step frozen after freeze) so both use the same state. **NEVER force slot-by-slot — user was explicit.**

3. **THE ACTIVE BUG (what the user wants fixed) — the fixed-buffer path diverges from the variable-buffer path, and the divergence ORIGINATES IN THE PREFILL.** This is a **pre-existing model/native bug, independent of #2 and the reuse work.** Details below. **This is where a fresh agent should start.**

---

## The active bug (#3): fixed-buffer prefill diverges from variable-buffer

### What is measured (all reproducible — see "How to reproduce")

- Greedy, same 25-token prompt. Fixed-buffer (`maxPrefillLength=64, maxKvCacheLength=128`) vs variable-buffer output **agree on tokens 0–3, then diverge at token index 4** and greedy-wander apart.
- **It is STRUCTURAL, not numerical.** Padding the prompt to 64 (KV 128) vs 96 (KV 192) — genuinely different plan shapes + kernels — gives **bit-identical** output (`diagPaddingSensitivity`). So the fixed path is shape-robust; the fixed-vs-variable difference is a real code-path/computation difference, not a fragile near-tie.
- **The divergence begins in the PREFILL, at the last REAL token.** The pipeline already logs `[GGUF-KV] Prefill last-pos logits: min/max/mean` (GenerationPipeline ~line 942). At `samplePos=24` (`actualPrefillLen=25`):
  - `pad64` vs `pad96` prefill logits: identical to **~1e-6** (`max 29.069515` vs `29.069520`) — pure fp32 kernel noise, argmax-robust.
  - `fixed` vs `variable` prefill logits: differ by **~1.4** (`max 29.07` vs `27.69`; `min -8.79` vs `-9.02`) — **six orders of magnitude** larger than kernel noise.
- Same argmax (both first token = `248068`), so token 0 matches, but the corrupted KV/GDN state propagates and tips the decode at token 4.

### The core question

The model is **causal**. Padding sits at positions 25+ (future relative to position 24). Future/padding **must be causally invisible** to position 24. But it isn't (1.4 logit shift). **Something mixes a future/padding position into the real-token computation at position 24.**

### Ruled OUT (with evidence — don't re-derive these)

- **Numerical / shape-kernel noise** — refuted by `pad64==pad96` to 1e-6.
- **The recurrent (GDN) INITIAL state fed to decode** — refuted by token 1 (warmup) matching.
- **Prefill real-token KV / attention masking of padding** — token 0 (prefill argmax) matches; and `buildPaddedPrefillCausalMask` (GenerationPipeline:2174) real-token rows are **identical** to `buildInGraphCausalMask` (DecoderInputBuilder:570) — only diff is fully-masked padding rows. Masked entries use `-1e9`/`-65504`, which underflow `exp` to exactly 0 in fp32 → masked positions contribute nothing.
- **`CausalConv1d`** (`libnd4j/include/ops/declarable/helpers/cuda/causal_conv1d.cu`) — **VERIFIED strictly causal.** Kernel line ~59: `srcT = t - kk`, so `out[t] = Σ_kk w[K-1-kk]·x[t-kk]`, reading only `x[t] … x[t-K+1]` (K=4, state width 3). No future read. **NOT the leak.** (I originally suspected this and was wrong — check the code, don't assume from the name.)

### Prime remaining suspects (UNVERIFIED — a fresh agent should trace, not guess)

The two cross-position ops in a GDN layer are the conv (ruled out) and the **GatedDeltaRule** scan. The GDN layer is built in `nd4j/nd4j-ggml/src/main/java/org/nd4j/ggml/architecture/LLaMAArchitecture.java` (method around line 673–860; conv at 748, `GatedDeltaRule` at 839). Note `GatedDeltaRule(sd, q, k, v, beta, gateDecay, gdnStateIn)` receives **NO attention/padding mask** — if its CUDA impl is **chunk-wise** and does an intra-chunk matmul, verify that the intra-chunk causal masking is correct AND that it doesn't treat padding positions as valid. CUDA impl: `libnd4j/include/ops/declarable/helpers/cuda/gated_delta_net_block.cu` (also `.../generic/nn/gated_delta_net_block.cpp`, and a llamacpp platform variant `.../platform/llamacpp/cuda/gated_delta_ops.cu`). Also worth a second look: the attention layers' actual application of the mask (the mask *builder* is correct; confirm the attention op consumes it correctly for a padded query block).

Caveat: naive reasoning says a causal scan at position 24 can only see 0..24, so it "shouldn't" leak — yet the measurement says it does. **Trust the measurement. Trace it.**

### THE NEXT STEP (do this first)

Value-trace the **prefill** and find the **first layer/op whose position-24 output diverges** between fixed and variable. Two ways:

1. **Targeted (preferred):** run `sd.output(...)` (or `decoder.output`) requesting the per-layer GDN/attention intermediates as extra outputs (`gdn_out_N`, `gdn_conv_N`, attention layer outputs) for BOTH a padded [1,64] prefill and an unpadded [1,25] prefill, slice position 24, and diff. First layer that differs (beyond ~1e-6) is the culprit; whether it's a GDN layer or an attention layer tells you which op to fix.
2. **Coarse:** `Nd4j.getEnvironment().setDebug(true); Nd4j.getEnvironment().setVerbose(true);` at the top of the test, run a single prefill each way, diff the per-op min/max/mean in the logs to bracket the first diverging op. (Huge logs — capture with `tee`, grep.)

Then fix that op to be strictly causal / padding-aware at the boundary, or (if it's a genuinely non-causal-by-design op like a chunk GDN) make the fixed-buffer path not feed trailing padding into it. **Left-padding is NOT a general fix for a recurrent model — it just moves the leak to position 0.**

---

## How to reproduce (all from `platform-tests/`, CUDA backend)

Test file (NEW this session): `platform-tests/src/test/java/org/eclipse/deeplearning4j/llm/generation/TestFixedBufferDecodeReuse.java`. Key methods:

- `diagIsolatedFixedVsVariable` — fixed vs variable on **two separate model imports** (no shared-executor contamination; frees the first model before importing the second or 3 resident 0.8B models blow the 24 GB physical cap). Logs `[ISO]` arrays + `fixed==variable?`.
- `diagPaddingSensitivity` — pad64 vs pad96 (the structural-vs-numerical discriminator). Logs `[PAD]`.
- `reuseAcrossGeneratesIsConsistentAndCoherent` / `reusePreservesCorrectnessAcrossPromptChange` / `reuseSurvivesConfigSwap` — the reuse (#2) regression gate. Currently `reuseAcross...` FAILS (reuse != fresh, per #2 above).

```bash
cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && \
/home/agibsonccc/dev-apps/mvn/bin/mvn test -Dtest=TestFixedBufferDecodeReuse#diagIsolatedFixedVsVariable \
  -Dbackend.artifactId=nd4j-cuda-12.9 2>&1 | tee /tmp/iso.log
# read the [ISO]/[PAD] lines and the "Prefill last-pos logits" lines from the tee log
```

The prefill-logit evidence is already captured in `/tmp/iso-fixed-vs-var.log` and `/tmp/pad-sensitivity.log` from this session (grep `"Prefill last-pos logits"`).

---

## Files changed this session (uncommitted unless noted)

- **COMMITTED** `d54e4fbe1d` — `libnd4j/include/graph/impl/NativeDynamicShapePlan.cpp` (the unseal; #1).
- `nd4j/samediff-llm/.../generation/GenerationPipeline.java` — Design A buffer-reuse (#2). ~16 edits: new field `cachedFixedBufferState`; `prefillWarmupAndFreeze(..., InGraphKvState reuseState)` reuse branches (skip `clearNodeOutputsOnly`, in-place overwrite of prefill+decode ext inputs, skip re-freeze, write back into the same state object); caller `generateSimpleWithInGraphKvCache` caches/reuses; `startSession` invalidates; `close()` frees. NOTE this file also had pre-existing (not-mine) ADR-0105 continuation edits at session start.
- `nd4j/samediff-llm/.../generation/InGraphKvState.java` — added `prefillInputMap` field + free it in `close()`.
- `platform-tests/.../llm/generation/TestFixedBufferDecodeReuse.java` — NEW (tests + diagnostics above).
- Memory note `dsp-fixed-buffer-decode-demotion.md` — updated with the full root-cause trail.

Build of `samediff-llm` (Java-only) is green: `/home/agibsonccc/dev-apps/mvn/bin/mvn install -DskipTests -pl :samediff-llm`. A fix to the native GDN op will need the CUDA build (see below).

---

## Ground rules (from AGENTS.md — the ones that bite)

- **NEVER** `ccache -C`/`--clear`, `git checkout <file>`/`stash`/`reset --hard`/`clean`, `make` directly, `mvn test` from repo root, `export VAR=` before `mvn test`, or pipe build/test output through `tail`. Always `tee` and read the log.
- **NEVER force slot-by-slot / hardcode GraphExecutionMode** as a workaround. Fix root causes.
- CUDA build (needed for a native GDN fix; ~10 min incremental, keep `-Dlibnd4j.chip=cuda`):
  ```bash
  /home/agibsonccc/dev-apps/mvn/bin/mvn -Pcuda -Dlibnd4j.triton=ON -Dlibnd4j.chip=cuda \
    -Dlibnd4j.buildthreads=12 -Dlibnd4j.log=libnd4j-build.log \
    -pl libnd4j,:nd4j-cuda-12.9 clean install -DskipTests 2>&1 | tee cuda-build-output.log
  ```
- All tests from `platform-tests/` only, `-Dbackend.artifactId=nd4j-cuda-12.9`, piped through `tee`.
- **Measure, don't reason** about DSP/model runtime behavior — this whole investigation flipped twice on measurement (numerical→structural; conv-suspect→conv-cleared). Verify with a test, not an argument.
