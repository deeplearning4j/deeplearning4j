# ADR 0111 — Constrained Decoding (JSON / Tool-Call) v1

**Status:** Accepted  
**Date:** 2026-07-12  
**Author:** kompile-agent (handoff R2)  
**Relates to:** SDX_MOBILE_LLM_C_API_HANDOFF.md §R2, DECODER_DEV_JOURNAL.md pieces 4-5

---

## Context

The samediff-llm `GenerationPipeline` produces free-form text. When an LLM is wired into a
tool-dispatch or structured-data extraction loop the caller needs the output to be
syntactically valid in a known grammar — typically a JSON object, or a canonical
`{"tool": "<name>", "args": {...}}` envelope for agent tool-call routing.

Without constrained decoding the pipeline can produce truncated or syntactically invalid JSON,
forcing callers to implement fragile retry/repair loops.

The KGR / graph-reasoning MCP layer (kompile) expects every model reply to be parseable as a
tool-call envelope with `tool` ∈ `{ask_graph_verify, graph_reasoning_query, ask_graph_query}`
before dispatching. A hard token-level constraint is the correct place to enforce this.

### Constraints on scope

* Pieces 1–3 of `DECODER_DEV_JOURNAL.md` (DecodeStrategy, Java mask builder, CUDA
  `token_sample` parity) are done and stable.
* Pieces 4–5 (masked multi-position substrate) are in-flight by other contributors.
* This ADR covers **only additive changes at the Java sampling layer** — no touch of the
  native `AutoregressiveDecode` op, no DSP-shape changes, no new ops.

---

## Decision

### Package: `org.eclipse.deeplearning4j.llm.generation.constraint`

| Class | Role |
|---|---|
| `TextConstraint` | Interface: `canExtend(currentText, piece)`, `isAccepting(currentText)`, `reset()`, `type()` |
| `JsonObjectConstraint` | Accepts any syntactically valid single JSON object (brace/bracket/string state machine) |
| `ToolCallConstraint(enumNames)` | Accepts `{"tool": "<enum>", "args": <free-form JSON>}` — 5-phase automaton |
| `ConstraintConfig` | @Data @Builder: `type`, `toolNames`, `evalTopK=256`; factory `jsonObject()` / `toolCall(String... names)` |
| `ConstraintVocabCache` | LRU-style cache (cap 512): amortises full-vocab sweep across steps with the same emitted prefix |
| `ConstraintMasker` | Stateful per-generation wrapper: `maskLogits(float[], eosTokenId, idToPiece)` + `tokenEmitted(int, idToPiece)` |

### Wire-in point

`GenerationPipeline.sampleToken` is overloaded with a 7-argument variant that:
1. Evaluates `ConstraintMasker.maskLogits` when a constraint is active.
2. Applies standard temperature / top-k / top-p / min-p sampling on the masked logits.
3. Calls `masker.tokenEmitted` after selection so the automaton advances.

The original 5-argument `sampleToken` delegates to the 7-argument variant with
`masker=null`, preserving zero behavior change when no constraint is configured.

### Bypassing the native `AutoregressiveDecode` op

The native op runs the full decode loop in C++ with no Java callbacks. When a constraint
is active a **Java step-by-step decode loop** replaces the native op for the constrained
run. The Java loop calls `decoder.output()` once per token, picking up the masker on every
step. The DSP plan is untouched; the native path is used unmodified for unconstrained runs.

### Masking strategy

The masker evaluates the top-`evalTopK` (default 256) tokens first (cheap partial sort).

* **Strategy A** (some top-K tokens are allowed): keep only those; zero the rest.
* **Strategy B** (no top-K tokens allowed): widen to full vocab and zero only disallowed tokens.

EOS is allowed only when `isAccepting(emittedText)` is true, preventing premature generation end.

### Configuration surface

Added to `SamplingConfig`:

```java
private ConstraintConfig constraintConfig;
public boolean hasConstraint() { return constraintConfig != null; }
```

Caller API:

```java
SamplingConfig sampling = SamplingConfig.builder()
    .doSample(true)
    .temperature(0.7f)
    .constraintConfig(ConstraintConfig.toolCall("ask_graph_verify", "graph_reasoning_query", "ask_graph_query"))
    .build();
pipeline.generate(prompt, 128, sampling);
```

### `options_json` contract (SDX C ABI layer)

```json
{
  "constraint": {
    "type": "json_object",
    "tools": []
  }
}
```

```json
{
  "constraint": {
    "type": "tool_call",
    "tools": ["ask_graph_verify", "graph_reasoning_query", "ask_graph_query"]
  }
}
```

`type` is one of `"json_object"` | `"tool_call"`.  
`tools` is required (may be empty) for `tool_call`, ignored for `json_object`.  
Unknown `type` values throw `IllegalArgumentException` at config-build time.

---

## Alternatives considered

### Grammar-based masking (GBNF / Lark)
Full grammar → token-trie masking is more general but substantially heavier
(grammar compilation, trie build per vocab). V2 item — see below.

### Post-hoc repair
Retry until valid JSON, or strip/repair after generation. Adds latency proportional to
failure rate, which is high on small (0.5B) models without guidance.

### Native-side masking (C++ callback into Java)
Would require DSP-shape changes and a new native op. Blocked on pieces 4-5. This v1
design is deliberately designed to coexist with pieces 4-5 (pure Java-path, additive).

---

## Consequences

### Positive
* Zero behavior change when `constraintConfig` is null — native decode path unchanged.
* Orthogonal to DSP / pieces 4-5; merge conflict surface is one field in `SamplingConfig`.
* `ConstraintVocabCache` amortises the full-vocab sweep: at evalTopK=256 most steps are
  O(n) once then O(k) on repeated prefixes.
* `ToolCallConstraint` is stateless (phase detection from accumulated text alone) —
  no mutable state to reset across retries.

### Negative / risks
* Java decode loop overhead vs native loop (extra JNI round-trips per step).
  Perf target: constrained overhead < 50% on CPU (validated by `ConstrainedDecodingIntegrationTest`).
* ToolCallConstraint requires the model to have seen the exact prefix
  `{"tool": "` during training; weaker models may drift before the constraint takes hold.
  `evalTopK` widening mitigates this in most cases.

---

## V2 items (out of scope here)

| Item | Rationale |
|---|---|
| Full JSON-Schema constraint (object shape, field types, required/optional) | Needs grammar compiler, high complexity |
| GBNF/BNF grammar support | General, but trie build cost is non-trivial at 32K+ vocab |
| Native-path masking hook (pieces 4-5) | Removes Java-loop overhead; blocked until substrate lands |
| Constraint serialisation in `.kgraph` model bundles | For reproducible constrained fine-tune replay |

---

## Files added/changed

**New (main):**
- `nd4j/samediff-llm/src/main/java/org/eclipse/deeplearning4j/llm/generation/constraint/TextConstraint.java`
- `nd4j/samediff-llm/src/main/java/org/eclipse/deeplearning4j/llm/generation/constraint/JsonObjectConstraint.java`
- `nd4j/samediff-llm/src/main/java/org/eclipse/deeplearning4j/llm/generation/constraint/ToolCallConstraint.java`
- `nd4j/samediff-llm/src/main/java/org/eclipse/deeplearning4j/llm/generation/constraint/ConstraintConfig.java`
- `nd4j/samediff-llm/src/main/java/org/eclipse/deeplearning4j/llm/generation/constraint/ConstraintVocabCache.java`
- `nd4j/samediff-llm/src/main/java/org/eclipse/deeplearning4j/llm/generation/constraint/ConstraintMasker.java`

**Modified:**
- `nd4j/samediff-llm/src/main/java/org/eclipse/deeplearning4j/llm/generation/sampling/SamplingConfig.java` — added `constraintConfig` field + `hasConstraint()`
- `nd4j/samediff-llm/src/main/java/org/eclipse/deeplearning4j/llm/generation/GenerationPipeline.java` — 7-arg `sampleToken`, `constraintMasker` creation, constrained Java decode loop

**New (test):**
- `nd4j/samediff-llm/src/test/java/org/eclipse/deeplearning4j/llm/generation/constraint/ConstraintAutomatonTest.java` — 27 unit tests (automaton logic, masking, config factories)
- `nd4j/samediff-llm/src/test/java/org/eclipse/deeplearning4j/llm/generation/constraint/ConstrainedDecodingIntegrationTest.java` — live model tests (skipped if assets absent)

**New (doc):**
- `ADRs/0111 - Constrained Decoding (JSON - Tool-Call) v1.md` (this file)
