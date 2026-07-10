# Decoder Dev Journal

Working journal for the consolidated decode-strategy work in `GenerationPipeline`
(ADR 0106 — masked multi-position decode substrate). Newest entries at the bottom.

## Goal

Finish **every** decode strategy, all in `GenerationPipeline` (not the other decoder classes),
each tested individually (ops) then end-to-end (full decode), across **every** config:

| Strategy | B×W | Status |
|---|---|---|
| greedy | 1×1 | works (native `autoregressive_decode`) |
| sampling (temp/top-k/top-p/rep-pen) | 1×1 | CPU ok; **CUDA top-k/top-p ignored → Piece 3** |
| speculative | 1×W chain/tree | prototype exists, unwired → revive in GenerationPipeline |
| contrastive | 1×(k+1) siblings | not built; needs last-hidden output |
| beam | B×1 | not built; divergent per-beam KV |

## Guardrails (from the AFK reminder, 2026-07-06)

- **Max-length allocation + masking** is the pattern (already how prefill/decode + KV work). Freeze the
  plan once at the max window; mask inactive slots. Do **not** reshape per step.
- **Do NOT modify DSP.** It's well tested. If a DSP issue appears → **stop immediately and
  `CronDelete cfcb517f`**; the user will look in the morning.
- Everything in `GenerationPipeline`; do not route through `StaticKvCacheDecodeLoop` /
  `SpeculativeDecodeLoop` / `TextGenerator`.
- Test individual ops first, then full decoders. Test every config and their differences.
- A few key DSP regression tests only — not the full gate. List any others wanted in the commit message.
- New ops → samediff op-codegen, follow conventions.
- CUDA kernels: framework conventions — no `#ifdef SD_CUDA` in `.cpp`; `SD_KERNEL`/`SD_HOST` macros,
  `BUILD_SINGLE_SELECTOR`, `getLaunchDims`, `DebugHelper::checkGlobalErrorCode`, in `.cu` files only.
- CPU code: multi-threaded (`PRAGMA_OMP_*`), framework-standard optimized loops.
- Be mindful of resources; wait for other tests/builds to finish before starting mine.

## Pieces (Phase 1 of ADR 0106)

1. **DecodeStrategy config surface + honest guard** — DONE. `SamplingConfig.DecodeStrategy`
   {AUTO,CONTRASTIVE,BEAM} + `numBeams`/`lengthPenalty`/`penaltyAlpha`/`contrastiveTopK` + presets
   `beam()`/`contrastive()` + `isBeam()`/`isContrastive()`. `GenerationPipeline.guardDecodeStrategy`
   throws (not silently greedy) until the substrate lands. Compiles.
2. **Fixed-width grid mask builder** — DONE. `DecoderInputBuilder.buildInGraphWindowMask` /
   `buildInGraphWindowPositionIds` + `chainParents`/`siblingParents`/`chain|siblingPositions`.
   Additive bias `[1,1,W,maxKv]`, W=1 identical to `buildInGraphDecodeMask`. Test
   `MaskedDecodeWindowTest` 5/5 green (CPU).
3. **CUDA `token_sample` top-k/top-p parity** — DONE ✅. Single `tempTopKTopPSampleKernel`
   (weight-space, block-cooperative binary-search thresholds). CUDA build SUCCESS (token_sample.cu
   genuinely recompiled). `TokenSampleParityTest` 6/6 on **CUDA** and 6/6 on **CPU**. Sampling decoder
   config (temp/top-k/top-p) now correct on both backends.
4. **Native masked multi-position substrate (B=1×W_max)** — pending.
5. **Phase 1 DSP regression gate (a few key tests)** — pending.

## Log

### 2026-07-06
- Pieces 1–2 landed and green (Java, CPU). Config surface + mask builder consolidated onto
  `SamplingConfig` / `DecoderInputBuilder` (no new classes) per the consolidation preference.
- Confirmed native CUDA decode samples via `tokenSample`/`tokenSampleWithPenalties`
  (cuda/autoregressive_decode.cu:748-762); CUDA `tokenSample` ignores top-k/top-p
  (cuda/token_sample.cu:193) → Piece 3 is the correct, bounded fix.
- Box idle (GPU 0/1 ~0-10%, no builds running); scheduled 30-min reminder cron `cfcb517f`.
- Next: implement Piece 3, build CUDA, test the sampling op individually, then wire/verify the
  sampling decode config end-to-end.
- Piece 3 CODE DONE: replaced CUDA `tempSoftmaxSampleKernel` with `tempTopKTopPSampleKernel`
  (cuda/token_sample.cu) — temp→top-k→softmax→top-p→sample in softmax-weight space, block-cooperative
  binary-search thresholds, no scratch buffer, mirrors CPU order exactly. `.cu`-only, no `#ifdef
  SD_CUDA`, CPU path untouched (already OMP). Individual-op test `TokenSampleParityTest` written
  (backend-agnostic correctness: topK=1→argmax, tiny topP→argmax, topK→∈topK-set, batch). CUDA build
  running (bg bks1dqn2r); will install nd4j-cuda + run the op test on CUDA (+ CPU sanity) when it lands.
- Piece 3 VERIFIED end-to-end at op level: `TokenSampleParityTest` 6/6 on CUDA and CPU.
- Full-decode test v1 (`TestSamplingDecodeConfigs`, 9 generates, N=24, variable buffers): ran ~35 min
  and was EXTERNALLY KILLED near the end (8/9 generates done) — NOT a DSP issue. Root cause: the
  variable-buffer path recompiles the whole DSP plan every generate (`PLAN_DESTRUCTION reason=
  PLAN_CHANGED`), so 9 generates × recompile ≈ 35 min → hit a background time cap. NO crash dump, NO
  err700/900, NO OOM; DSP succeeded throughout and 3 sampling configs produced valid coherent output
  (`<think></think> It sounds like a classic...`). `hasNaN=false`.
- Fix (matches the "max-length allocation + masking" mandate): set `maxPrefillLength=64` +
  `maxKvCacheLength=128` so the plan **freezes once and replays** across generates → fast; N=16.
  Re-running (bg bjlh1iqfo). LESSON for full-decode tests: always use fixed buffers or they recompile
  per call and time out.
- Contrastive feasibility: GGUF decoders name the final pre-lm_head norm `output_norm`
  (e.g. GemmaArchitecture.java:223 `buildRMSNorm(...,"output_norm",...)`), so `h_t` is likely a
  requestable decode output → contrastive can be pure-Java in GenerationPipeline (top-k via SamplerUtils
  → W=k sibling masked forward requesting logits+`output_norm` → degeneration score → winner + per-layer
  KV slot copy). Verify the exact output var name against a loaded model before implementing.

### 2026-07-06 — STOPPED on a DSP issue (cron cfcb517f cancelled, per the AFK reminder)

**What happened.** The full-decode test v2, configured with the *fixed-buffer fast path*
(`maxPrefillLength=64` + `maxKvCacheLength=128`) so the DSP plan freezes once and replays, FAILED with a
hard DSP error on all 3 methods (Qwen 0.8B GDN-hybrid, CUDA):

```
DSP execution failed. No fallback to standard path. Fix the DSP executor.
Native plan execution failed with status -1: [PHASE_TRANSITION] plan REPLAYING -> SHAPES_FROZEN
reason=segment no longer satisfies replay steady state: seg[0-1340] backend=2 execPhase=BUILDING:WARMUP
segExecCount=0 handleReady=0 compositeReady=0 argStable=0 execCount=15 frozenExec=4 kind=demotion
```
It also produced degenerate output (`<think></think> （诡诡）))))`) before failing — suggests the
padded-prefill causal mask for this recurrent (GDN/conv-state) model interacts badly with frozen replay.

**Not caused by this branch's decoder work.** Evidence:
- `TokenSampleParityTest` passes 6/6 on **CUDA** and 6/6 on **CPU** (the CUDA sampling fix is correct).
- Full-decode v1 (SAME test, **variable buffers**, no `maxPrefillLength`) ran fine for ~35 min with
  coherent output across 3 sampling configs, `hasNaN=false`, no DSP error (it was only externally killed
  for running too long — 9 generates × per-call DSP recompile).
- `token_sample.cu` is post-logits sampling; it is not part of DSP plan freeze/replay.

So the demotion is a **pre-existing DSP frozen-replay behavior** on the fixed-buffer path, exposed by
choosing that config for test speed — the exact "max-length allocation + masking" fast path the reminder
points at, so it blocks the intended fast decode path and needs the user's eyes.

**Reproducer:** `TestSamplingDecodeConfigs` with `.maxPrefillLength(64).maxKvCacheLength(128)` on the shared
pipeline across ~9 generates (fails by ~execCount=15). Test reverted to variable buffers (works, slower).

**Action taken:** stopped all DSP-touching work, cancelled cron `cfcb517f`, reverted the test config,
did NOT modify DSP. Solid tested work (Pieces 1–3) is intact and green. Strategy decoders
(contrastive/beam/speculative — Piece 4+) are BLOCKED on this fixed-buffer replay-demotion, since they
rely on the frozen-plan + masking fast path. Awaiting user review in the morning.

### 2026-07-07 — Strategy surface consolidated; native B/W substrate still gates execution

User direction: DSP work is still in flight; continue feature consolidation and save testing for last.

What landed in this pass:
- `SamplingConfig.DecodeStrategy` now models the full ADR 0106 strategy set: `AUTO`, `GREEDY`, `SAMPLE`,
  `SPECULATIVE`, `CONTRASTIVE`, `BEAM`.
- Added explicit presets/helpers: `sample(...)`, `speculative()`, `isSampling()`, `isSpeculative()`;
  existing AUTO behavior stays backward-compatible.
- `GenerationPipeline` now resolves every requested strategy into one ADR 0106 policy shape:
  - greedy/sample → `B=1,W=1`
  - speculative → `B=1,W=maxSpeculativeTokens`
  - contrastive → `B=1,W=contrastiveTopK`, requiring hidden-state output
  - beam → `B=numBeams,W=1`
- All GenerationPipeline decode entry points now use the same resolver and reject multi-hypothesis
  strategies at the native boundary instead of silently running scalar greedy/sample or routing through
  `SpeculativeDecodeLoop`, `StaticKvCacheDecodeLoop`, or `TextGenerator`.

Important boundary found while inspecting the native op: current `autoregressive_decode` still hard-codes
scalar substrate assumptions (`inputIds [1, seqLen]`, attention mask `[1,1,seqLen,maxKvLen]`, one logits
stream, one token stream). True speculative/contrastive/beam execution therefore still requires extending
that native op/helper interface to fixed `B_max x W_max`; the Java policy surface is ready for it, but the
old scalar op cannot honestly execute these strategies yet. No DSP changes made.

### 2026-07-07 — `token_sample` is now the native policy boundary

User direction: integrate the ADR 0106 strategy plan with current `token_sample` so all sampling lives in
one place with different configurations.

What landed:
- Added `TokenSampleStrategy`, `TokenSampleConfig`, `TokenSampleResult`, and `tokenSamplePolicy(...)` to
  `token_sample.h`.
- Implemented CPU and CUDA `tokenSamplePolicy(...)` for scalar `GREEDY` / `SAMPLE`, delegating to the
  existing `tokenSample` / `tokenSampleWithPenalties` helpers. Non-scalar policies now fail fast inside
  the selector instead of accidentally executing as greedy.
- `AutoregressiveDecodeConfig` now carries a `TokenSampleConfig`.
- `autoregressive_decode` parses the policy envelope from `tArgs[4..13]` while preserving the existing
  iArg layout for plan handles, KV indices, recurrent-state indices, and stop IDs.
- CPU and CUDA `autoregressive_decode` helpers now call `tokenSamplePolicy(...)` instead of owning their
  own greedy-vs-sampling branch.
- Java `AutoregressiveDecode` now emits the policy envelope in tArgs and exposes `withDecodePolicy(...)`
  for future B/W substrate call sites. This keeps policy metadata out of iArgs so current variable KV/stop
  packing stays stable.

Verification:
- Java compile: `/home/agibsonccc/dev-apps/mvn/bin/mvn install -DskipTests -pl :nd4j-api,:samediff-llm`
  passed.
- CUDA build: `/home/agibsonccc/dev-apps/mvn/bin/mvn -Pcuda -Dlibnd4j.triton=ON -Dlibnd4j.chip=cuda
  -Dlibnd4j.buildthreads=16 -Dlibnd4j.log=libnd4j-build.log -pl libnd4j,:nd4j-cuda-12.9 clean install
  -DskipTests` passed (`libnd4j` 05:17, `nd4j-cuda-12.9` 01:30).
- Focused test: `TestAutoregressiveDecodeIArgs` passed 4/4 from `platform-tests`; milestone `dee1dfed`.

Still not done: the actual masked B/W forward is not implemented yet. This pass only makes the selector
contract and native op metadata ready, and centralizes scalar token selection through `token_sample`.

### 2026-07-07 — Sampling-policy parity gaps closed before B/W substrate

Follow-up from the sampling-gap audit:
- Extended `SamplingConfig` with `minP`, frequency penalty, presence penalty, `typicalP`, beam groups,
  diversity penalty, and return-sequence count. Scalar native execution consumes `minP` + frequency/presence;
  `typicalP` now fails fast instead of silently no-oping until a native typical-p filter exists.
- Extended the `AutoregressiveDecode` tArg policy envelope to carry `minP`, frequency/presence penalties,
  `minNewTokens`, generated-token offset, and a full-width seed split into two 32-bit exact lanes.
- `GenerationPipeline` now applies the resolved ADR 0106 policy envelope to every native
  `autoregressive_decode` handoff instead of only passing legacy temp/top-k/top-p/repetition args.
- Fixed the Java warmup tokens on the static/native path: they now use the same `sampleToken` policy helper
  instead of hard-coded argmax, and both GGUF + static warmups honor stop-token suppression under
  `minNewTokens`.
- `tokenSamplePolicy` now owns stop-token floor masking for CPU and CUDA before greedy/sample selection;
  native decode derives per-step seeds from the configured base seed (`seed + step`) and passes stop IDs / 
  generated offsets into the policy.
- `GenerationConfig` now parses `min_p`, `frequency_penalty`, and `presence_penalty` alongside existing
  HF fields. Beam-only extras remain represented but execution is still blocked by the B/W substrate gate.

No DSP changes, no mode forcing, no cache clearing, and no alternate decode loops.
