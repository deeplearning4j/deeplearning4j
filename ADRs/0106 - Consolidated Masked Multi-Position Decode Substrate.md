# ADR 0106: Consolidated Masked Multi-Position Decode Substrate (Beam, Contrastive, Speculative)

## Status

Proposed

Proposed by: Adam Gibson (6 Jul 2026)

## Context

`GenerationPipeline` (module `nd4j/samediff-llm`, package `org.eclipse.deeplearning4j.llm.generation`)
today offers exactly two token-selection strategies — greedy and stochastic sampling
(temperature / top-k / top-p / repetition penalty) — via `SamplingConfig`. The only samplers are
`GreedySampler` and `CompositeSampler`. There is **no beam search and no contrastive search**. Adding
them naively means two more decode loops, but the codebase already carries **three disjoint decode
implementations**, and adding more would compound the divergence:

1. **Native single-sequence decode (wired, fast, frozen).** `autoregressive_decode`
   (`libnd4j/include/ops/declarable/generic/nn/autoregressive_decode.cpp:85`; helper
   `.../helpers/cpu/autoregressive_decode.cpp:116` and `.../cuda/autoregressive_decode.cu`) runs the
   *entire* decode loop in C++ against a frozen DSP plan: `plan->execute()` → `token_sample` →
   in-place KV write → embed lookup → repeat, with **no per-step Java round-trip**. It is strictly
   `batch=1`, `window=1`, and its only model output consumed is logits (`logitsOutputIdx`;
   `AutoregressiveDecodeConfig` has no hidden-state field — `.../helpers/autoregressive_decode.h:58`).
2. **Java speculative / tree (unwired, dynamic-shape, prototype).** `TreeAttentionVerifier`,
   `SpeculativeDecodeLoop`, `DraftModelSpeculator`, `NgramSpeculator`, `SpeculativeKVCacheManager`.
   `TreeAttentionVerifier.buildTreeAttentionMask` (`TreeAttentionVerifier.java:177`) already expresses
   *many hypotheses in one masked forward* — it packs candidates along the **sequence** dimension and
   builds a `[1,1,W,past+W]` mask where each node attends only to its ancestors + past KV, reading all
   `W` position-logits from one pass (`verifyTree`, `:266`). But it runs the model via
   `decoder.output(...)` (Java per-step; `DraftModelSpeculator.java:421`), its masks are **dynamically
   sized** (grow per step → no DSP freeze / no CUDA-graph replay), it selects with **argmax only**, and
   it is **not wired**: every `new SpeculativeDecodeLoop(...)` is in a test, `buildDraftSpeculator()`
   (`GenerationPipeline.java:3530`) has no live caller, and `create()` logs a "NOT yet implemented in
   decode loops" warning (`GenerationPipeline.java:448`).
3. **`token_sample` selection primitive (native, both platforms).** `tokenSample` /
   `tokenSampleWithPenalties` (`.../helpers/token_sample.h:31,52`) implement temperature / top-k /
   top-p / penalties / min-p on CPU (`.../cpu/token_sample.cpp`); the CUDA kernel
   (`.../cuda/token_sample.cu:193`) implements only greedy-argmax and temperature-softmax sampling —
   **top-k / top-p are silently ignored on CUDA** (a pre-existing correctness gap).

### Key enabling observation

Beam, contrastive, and speculative decoding are the **same shape**: a *masked forward over a fixed
grid of positions/hypotheses*, followed by a *per-step policy* over the resulting position-logits.
Today's single-sequence decode is the `[B=1 × W=1]` special case; speculative is `[1 × W]`;
contrastive is `[1 × (k+1)]`; beam is `[B × 1]`. The DSP idiom already used for the *sequence*
dimension — fix a **maximum** (`maxPrefillLength` / `maxKvCacheLength`), pad, and **mask**, so the
plan freezes and CUDA-graph-replays once — generalizes directly to the *candidate / hypothesis*
dimension. `token_sample` already provides the per-position selection primitive to build on.

## Decision

Build **one** substrate — a frozen, masked, multi-position decode step — and express greedy, sampling,
speculative, contrastive, and beam decoding as thin **policies** over it. Do **not** add parallel
beam/contrastive loops.

### The substrate

- **Frozen `[B_max × W_max]` masked multi-position forward.** Extend `autoregressive_decode` so the
  per-step forward runs a fixed grid of `B_max` hypotheses (batch dim) × `W_max` candidate positions
  (window dim). Shape is fixed at the max and driven by masking (a generalized, **fixed-width**
  `buildTreeAttentionMask`: `[B,1,W,past+W]`, position grid `[B,W]`), so the DSP plan freezes and
  captures **once**. `W=1, B=1` is the current path and MUST remain behaviorally identical
  (regression-safe special case). Implemented on **both** CPU and CUDA.
- **Optional last-hidden-state output.** Add a hidden-state plan output + ext index to
  `AutoregressiveDecodeConfig` (contrastive needs `h_v` per candidate). Off by default; only wired when
  the policy requires it and the decoder graph exposes its final pre-logits hidden state (add the
  output to the graph where a model does not already expose it).
- **KV layout.** *Window* dim = one shared past + masked sibling candidates (contrastive, speculative).
  *Batch* dim = `B` **divergent** per-hypothesis KV histories + a **reorder** on prune (beam). Reuse
  `SpeculativeKVCacheManager` zero-copy checkpoint/rollback and add a batch-dim reorder/gather
  primitive.
- **One selection primitive.** `token_sample` is the single per-position selector for every policy;
  close its CUDA top-k / top-p gap as part of this work so all backends agree.

### Policies (thin consumers of the substrate)

| Strategy | B | W | Adds over substrate |
|---|---|---|---|
| greedy / sample (today) | 1 | 1 | nothing — `token_sample` |
| speculative (revive) | 1 | K (chain/tree) | tree mask + accept/rollback (prototype logic) |
| contrastive | 1 | k+1 siblings | **last-hidden output** + degeneration score `(1−α)·p − α·max cos-sim` |
| beam | B | 1 | per-beam divergent KV + top-B-over-B×vocab + reorder + length penalty |

- **Policy execution.** Greedy/sample stays **fully native in-loop** (unchanged — no round-trip). For
  the multi-hypothesis policies the substrate forward is native and frozen; the policy bookkeeping runs
  **once per window/step**, not per token, so any Java round-trip is amortized over the window (this is
  how speculative verification is meant to work). Phasing permits a **Java policy over native windowed
  forwards** first (correctness), moving hot policy logic into the native loop later if the benchmark
  demands it. Either way the *forward* is the single consolidated substrate.

### Consolidation (retire, don't add)

- `TreeAttentionVerifier.buildTreeAttentionMask` → the substrate's fixed-width mask builder.
- `SpeculativeDecodeLoop` (Java, dynamic, `decoder.output()` per-step) → a speculative **policy** on the
  native frozen substrate; the disjoint prototype path is removed, not left alongside.
- The banned `StaticKvCacheDecodeLoop` stays retired (it is not on this path).
- **Java surface.** A single `DecodeStrategy` enum on `SamplingConfig` (`GREEDY`, `SAMPLE`,
  `CONTRASTIVE`, `BEAM`, `SPECULATIVE`) plus params (`numBeams`, `lengthPenalty`, `penaltyAlpha`,
  `contrastiveTopK`, speculation width); **one** decode entry in `GenerationPipeline` selects the
  policy. Composes with the runtime-mutable active `SamplingConfig` (`setSamplingConfig`, this branch),
  so one 5 GB model load can be driven across strategies without reload.

### Phasing (each phase independently valuable and DSP-gated)

1. **Substrate, `B=1 × W_max`, masked, frozen** — generalize the native step; prove `W=1` parity → no
   behavior change. Both platforms. + CUDA `token_sample` top-k/top-p fix.
2. **Revive speculative on the substrate** — fold the prototype in; retire the Java loop; run the
   speculative + lossless-equivalence tests.
3. **Contrastive** — add last-hidden-state output + degeneration-penalty policy.
4. **Beam** — extend to `B_max` batch dim + KV reorder. **Highest risk** (CUDA-graph capture at `B>1`
   is the known freeze/capture landmine zone; see the DSP-frozen-phase and capture-arena history), so
   it is last and gated behind a capture-feasibility spike; we can stop before it.

### Invariants

- `W=1, B=1` is bit-identical to the pre-ADR native decode (greedy determinism preserved).
- The plan freezes/captures **once** at `[B_max × W_max]`; no per-step reshape; the frozen-plan
  ext-input pointer-stability contract (ADR 0105) is preserved — hypotheses are activated by mask, not
  by reallocation.
- `token_sample` is the only place a token is chosen, on every backend.

## Consequences

- One decode engine instead of three; the unwired Java speculative prototype is **revived and
  absorbed** rather than left as parallel dead code; the CUDA sampling correctness gap is closed.
- New additive public API: `DecodeStrategy` + strategy params on `SamplingConfig`; beam / contrastive /
  speculative selectable via config and the runtime `setSamplingConfig` path.
- Native op interface grows (grid dims `B/W`, policy selector, strategy params, optional hidden-state
  output, `[B,1,W,past+W]` grid mask). New frozen shapes `[B×W]`; **CUDA-graph capture at `B>1` is the
  principal risk** and is quarantined to Phase 4.
- Performance: the `[1×1]` path must be regression-neutral (the mandatory DSP gate + `lateSteady tok/s`
  benchmark verify this each phase). Beam and contrastive are quality modes and are expected to be
  slower per token; that cost is inherent to multi-hypothesis search, not a regression.
- Memory: beam holds `B` KV histories; contrastive/speculative hold one past + a `W`-wide window.
  Bounded by the configured `B_max` / `W_max`.

## Alternatives considered

- **Separate beam and contrastive decode loops** — rejected: proliferation, the exact anti-pattern the
  three existing disjoint paths already demonstrate.
- **Pure-Java per-step loops on `decoder.output()`** (what the current speculative prototype does) —
  rejected as the target: bypasses the native frozen path, incurs a Java round-trip per token, and is
  what left the speculative code dead and divergent. Retained only as a transitional policy-execution
  option (per *window*, not per token) behind the native substrate.
- **Windowed / bounded beam on the shared-past window** — rejected as the primary beam design because
  it is not textbook-exact (beams share one past, so long-range divergence is lost); kept as a cheaper
  fallback if `B>1` CUDA-graph capture proves infeasible in the Phase 4 spike.
- **Re-freeze / re-capture the plan at `batch=N` per call** — rejected: per-call recapture cost and DSP
  lifecycle churn. The fixed-max + mask idiom freezes once and masks inactive rows.
- **Leave speculative decoding out of scope** — rejected under the consolidation constraint: the
  substrate *is* the speculative mechanism, so building it without wiring speculative would preserve
  the dead parallel prototype.

## Validation

- **Per-phase DSP regression gate** (the mandatory batch in `AGENTS.md`) — beam/contrastive/speculative
  all run through DSP; the gate catches cross-test contamination and frozen-plan / capture regressions.
- **Phase 1 parity** — `W=1, B=1` token-for-token identical to the current decode
  (`TestQwen35Pipeline`, `TestSmolDoclingOptimizedPipeline`), `lateSteady tok/s` neutral vs. the
  committed baseline.
- **Phase 2 speculative** — `SpeculativeDecodeLoopTest` + the lossless-equivalence oracle
  (`DraftModelSpeculativeDecodingExample`, SmolLM2 135M → 1.7B): output identical to greedy, acceptance
  rate > 0, both backends.
- **Phase 3 contrastive** — degeneration / repetition reduction vs. greedy on a repetition-prone prompt;
  numeric parity of the degeneration score CPU vs. CUDA.
- **Phase 4 beam** — beam output matches a reference beam search (log-prob ordering, length penalty) on
  a small model; both backends.
- New test classes under `platform-tests/.../llm/generation/`; all runs `tee`-logged.

## Related

- ADR 0096 (LLM Generation Pipeline), ADR 0097 (Decode Path Performance Optimizations),
  ADR 0105 (Generation Session Continuation — the frozen-plan pointer-stability contract this builds
  on). Depends on the runtime-mutable `SamplingConfig` change on this branch
  (`GenerationPipeline.setSamplingConfig`).

## Open questions / follow-ups

- `B_max` / `W_max` defaults and configuration surface (proposed defaults: `W_max=8`, `B_max=4`).
- Per-model availability of a final hidden-state output for contrastive — audit which GGUF/ONNX
  decoders expose it vs. need the output added to the graph.
- Phase 4 CUDA-graph capture feasibility at `B>1` — a spike gates the batch-dim beam commit; the
  windowed-beam fallback is the contingency.
- Whether the speculative policy's bookkeeping (tree verify) and beam's top-B selection eventually move
  into the native loop for throughput, or stay as per-window Java policies.
