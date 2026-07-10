# ADR 0105: Generation Session Continuation (Resume / Incremental Decode)

## Status

Proposed

Proposed by: Adam Gibson (6 Jul 2026)

## Context

`GenerationPipeline` (module `nd4j/samediff-llm`, package
`org.eclipse.deeplearning4j.llm.generation`) generates text with a single
`generate(prompt, maxNewTokens)` call. For single-model GGUF decoders this runs the in-graph-KV
path (`generateSimpleWithInGraphKvCache`): prefill → warmup decode → freeze DSP plan → native
`autoregressive_decode` C++ loop, then it frees the static KV / recurrent buffers.

Downstream (kompile) runs **small-context** local models (e.g. lfm2.5-1.2b, 512-token context) for
structured extraction. When a call hits the token budget it returns
`GenerationResult{finishReason = MAX_TOKENS}` with a truncated body (e.g. half-written JSON). The
only prior recourse was to re-call `generate(prompt + textSoFar)`, which **re-tokenizes and
re-prefills the entire prior context** — O(n²) across turns — and is indistinguishable from a fresh
request.

We want a first-class way to **continue decoding from where the prior call stopped, reusing the
already-populated in-graph KV cache** (no session reset, no re-prefill), bounded by the model's
context window.

### Key enabling fact

The native `autoregressive_decode` op is **already resumable**: its C++ loop
(`libnd4j/include/ops/declarable/helpers/cpu/autoregressive_decode.cpp:159`) starts at
`currentPosition = prefillSeqLen` (an iArg), writes the *fed* token's K/V at that position, samples
the next token, and advances — honoring an arbitrary start position and reading pre-populated KV
buffers. **No C++ / nd4j-api change is required.** The invariant that makes resume exact: the last
sampled token's K/V is not written until it is fed, so continuation feeds the last generated token
at absolute cache position `P + G - 1` (`P` = prompt length, `G` = tokens generated so far) — exactly
mirroring the existing prefill→warmup→native handoff.

## Decision

Add a **`GenerationSession`** (task Option 1) as the continuation primitive, layering `continueFrom`
(result-threaded) and `append` (token injection) on top. All changes are Java-only in `samediff-llm`.

- **Refactor, don't fork.** `generateSimpleWithInGraphKvCache` is split into
  `prefillWarmupAndFreeze(...) → InGraphKvState` (prefill + warmup + freeze, retaining all buffers /
  handles / ext-indices) and `runInGraphNativeDecode(state, n, isContinuation)` (the native loop,
  which no longer frees the retained buffers). The one-shot method is now
  `build → run(false) → close` and is **behaviorally identical** to before.
- **`InGraphKvState`** holds the retained static-KV / recurrent buffers, decode-step tensors, frozen
  plan handles, resolved ext indices, and the running decode state (`cachePosition`,
  `lastGeneratedToken`, `generatedSoFar`, RNG). The **same INDArray objects** are reused across
  calls, so the frozen-plan ext-input pointers and any captured CUDA-graph device pointers stay
  valid — the pointer-stability contract that `fixedBuffers` mode already relies on. `close()` frees
  them exactly once.
- **Capacity is a configurable mode; default is static, sized to the context ceiling**
  (`KvContinuationMode.STATIC_CONTEXT_CEILING`). A session pre-sizes its STATIC KV buffer once to
  `prefillLen + capacity` (capacity resolved from an explicit arg, else `config.sessionCapacity`,
  else `config.maxKvCacheLength − prefillLen`, else `config.maxNewTokens`). When the buffer fills,
  continuation returns `MAX_TOKENS` ("context full"). `GROWABLE` is reserved and throws
  `UnsupportedOperationException` (this ADR's follow-up).
- **Numerical-identity contract.** For greedy decoding with the default repetition penalty, one
  logical generation over K calls is token-for-token identical to a single `generate()` of the summed
  budget — validated by `TestGenerationSessionContinuation` (generate 20 + continue 20 == generate
  40). With sampling or a repetition penalty `≠ 1.0` the continuation is valid but not bit-identical
  (the Java/C++ RNG and the per-invocation penalty history do not carry across the seam).
- **Concurrency is thread-confined + lock-free** (deadlock-free by construction). One session per
  pipeline (`AtomicReference` CAS guard); sessions are bound to their creating thread (the decoder's
  `InferenceSession` and frozen plan are thread-affine) and enforce it fail-fast; no monitor is held
  across a native decode. `pipeline.close()` frees an open session's buffers before the decoder.
- **Degenerate-loop ("thinking trap") safety.** Every path is hard-bounded by KV capacity — no
  unbounded loop. The `continueToCompletion` convenience additionally runs a configurable
  `RepetitionGuard` (default on) that stops early on a periodic-tail repetition and reports
  `FinishReason.REPETITION`. Detection lives only in the loop convenience, never in the pure
  primitives, so it cannot perturb the identity contract.

## Consequences

- Truncated structured output can be self-healed by continuing into the unused context budget instead
  of blindly bumping `maxNewTokens` and re-prefilling. The downstream loop is
  `while (result.isTruncated() && session.getRemainingCapacity() > 0) result = session.continueGeneration(n)`.
- New public API on `GenerationPipeline`: `startSession(String[, int])`, `continueFrom`, and the
  nested `GenerationSession` (`generate`, `continueGeneration`, `continueToCompletion`, `append`,
  `cancel`, `getFullText`, `getRemainingCapacity`, `close`). New `GenerationResult.sessionId` and
  `FinishReason.REPETITION` (both additive). New config: `sessionCapacity`, `kvContinuationMode`,
  `repetitionGuard`.
- Continuation beyond the pre-sized static buffer is **not** supported in this release; extending past
  the original budget requires the `GROWABLE` KV cache (reallocate + re-establish frozen-plan pointer
  stability + re-capture CUDA graphs) — deferred to a follow-up ADR.

## Alternatives considered

- **Result-threaded `continueFrom` as the primitive (Option 2)** — reduces to Option 1 under the hood
  (state must live somewhere keyed to the last result), so it is provided as a convenience over the
  session, not as the core.
- **Growable / dynamic KV now** — larger blast radius on the frozen-plan / CUDA-graph pointer
  contract and the DSP regression gate; deferred. The static-to-context-ceiling default covers the
  motivating small-context case.

## Validation

`platform-tests/.../llm/generation/TestGenerationSessionContinuation` (CUDA, Qwen3.5-0.8B GGUF, a
GDN-hybrid model that also exercises recurrent-state retention): identity (continue == single-shot),
`continueToCompletion` termination (observed `REPETITION` and capacity-full), lifecycle/close/reopen,
and `append`. The one-shot path is unchanged and remains covered by `TestQwen35Pipeline`.
