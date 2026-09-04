# ADR 0119: Unified KvCacheManager Facade over InGraphKvState (Hybrid-Recurrent Android Decode)

## Status

Proposed

Proposed by: Tensor G3 qualification session, 2026-09 (Android one-off collapse workstream, parked pending frozen-plan lease fix)

## Context

### Two KV systems, one used

The repo carries two separate KV-cache stacks:

1. **`KvCacheManager` family** (`samediff-llm/.../generation/kvcache/`, ~8,850 LOC across 28 files):
   `KvCacheManager` interface with `KvCacheStrategy` = `STATIC / PAGED / QUANTIZED / TURBOQUANT`,
   implemented by `UnifiedKvCacheManager` (1,387 LOC), plus `TieredKVCacheManager`,
   `BeamKVCacheManager`, `SpeculativeKVCacheManager`, `KVCacheDiskOffloader`, `KVCacheHostOffloader`,
   `PagedKVCache`, `PerLayerPagedKVCache`, `QuantizedPagedKVCache`, `EvictablePagedKVCache`,
   eviction policies (`H2O`, `StreamingLLM`, sink-aware), `KVCachePrefixTree`/`RadixPrefixCache`
   prefix sharing, and checkpointing.

2. **`InGraphKvState`** — the state object actually executed by the production decode path
   (`GenerationPipeline` in-graph fixed-buffer path). It owns the static KV buffers, the 24+12
   hybrid recurrent state pairs (`past_gdn_state.*` gated-delta-rule, `past_conv_state.*`
   causal-conv) for Qwen3.5-style models, the quantized KV (V1 archive / V2 live layout),
   the decode-step scalars at stable addresses, the frozen-plan handles, and the pooled
   per-sample scratch (`reusedScratch`).

**Production construction sites for the manager family today: zero reachable from Android.**
`UnifiedKvCacheManager` is constructed only inside `StaticKvCacheDecodeLoop` (banned dead code per
AGENTS.md and `GenerationPipeline:1295`); `SpeculativeKVCacheManager` only inside
`DraftModelSpeculator` (not in the chat path); `TieredKVCacheManager`, `BeamKVCacheManager`,
`KVCacheDiskOffloader`, `KVCacheHostOffloader` have **no callers at all**.

### The Android one-off problem

The Android app independently maintains a **preparation-profile vocabulary** with no runtime
counterpart:

- `ModelPreparationOptions.kt` defines `KvCacheOptimization` (`OFF/INT8/FP8_E4M3/FP8_E5M2`,
  wire 0–3) — a *dtype* picker only. It maps to `kvQuantFormat` in the canonical SDZ profile
  and selects quantized storage **inside `InGraphKvState`** (V2 live INT8). It does **not**
  select a manager.
- `ChatViewModel`, `AppPreferences`, `OptimizedModelCacheRepository`, and
  `ModelOptimizationOptionsPane` each hand-roll their own serialization/projection of the same
  field list (`weightOptimization`, `kvCacheOptimization`, `tensorBatchSize`,
  `useMemoryMapping`, `diagnosticMode`) — five near-identical copies.
- The runtime (`SdxGgufModelPreparer` calibration; the packaged decode session) reads
  `kvQuantFormat` from the profile and applies it; everything else in the Android KV picker is
  cosmetic.

Result: the Android UI promises "KV cache optimization" but can only toggle dtype; the
manager family that could page, evict, tier, offload, or share prefixes is unreachable from
the product; and the same five-field vocabulary is maintained in five places that can drift.

### The impedance mismatch

`KvCacheManager` is specified over classic attention: `initializeFromPrefill(...,
ModelIOConfig.KVCacheNames kvNames, ...)`, `scatterNewEntries(outputs, kvNames)` — K/V
`present_key_values.{L}` pairs. Qwen3.5 hybrid decode on device has **36 state pairs**: 12
`past_gdn_state` (gated delta rule) and 24 `past_conv_state` (causal conv), addressed through
`InGraphKvState.gdnStateExtIndices/convStateExtIndices` in the frozen plan. A naive adapter
over `kvNames` alone covers ~1/3 of the state surface and silently drops the recurrent pairs
that dominate memory on this architecture.

Additionally, the frozen-DSP-plan contract requires **pointer-stable** buffers across decode
steps; any manager behavior that reallocates or relocates live storage (disk offload, host
swap, compaction) must be quarantined behind plan-safe boundaries (session start/end,
prefill/freeze boundary), not per-token.

## Decision

Introduce a **`HybridRecurrentKvManagerAdapter`**: the single `KvCacheManager` implementation
that fronts `InGraphKvState`'s full state surface, and make it the **only** path by which
Android selects KV behavior. Collapse the Android side to one vocabulary owned by one file.

### 1. Adapter (producer side, `samediff-llm`)

```java
final class HybridRecurrentKvManagerAdapter implements KvCacheManager {
    // Wraps InGraphKvState; exposes:
    //  - attention KV pairs        -> kvNames key/value names        (existing contract)
    //  - gdnStateExtIndices pairs  -> recurrent state names          (new contract below)
    //  - convStateExtIndices pairs -> recurrent state names          (new contract below)
    // Strategy semantics map to InGraphKvState fields, NOT to new storage:
    //  STATIC     -> staticKvBuffers/recurrentStateBuffers as today
    //  QUANTIZED  -> quantizedKvBuffers/kvScaleBuffers (V2 live INT8/FP8), kvQuantFormat passthrough
    //  PAGED      -> REJECTED at construction under frozen plans (pointer instability), see below
    //  TURBOQUANT -> passthrough to existing TURBOQUANT fields on GenerationPipelineConfig
}
```

Extend `KvCacheManager` with **one** new capability method rather than new interfaces:

```java
/** Names of all cache-state buffers this manager owns, including non-attention
 *  recurrent state (gated delta rule, conv) when the model has them. */
default List<StateSegment> stateSegments() { return List.of(); }
```

`StateSegment(name, kind ∈ {ATTENTION_K, ATTENTION_V, GDN_STATE, CONV_STATE}, INDArray buffer,
long capacityTokens)` — managers that only understand attention K/V may ignore the other kinds;
the adapter always reports all of them so tiering/eviction policies can be made state-aware
later without another interface break.

### 2. Plan-safety boundary (hard rule)

Under a frozen DSP plan (shapes frozen / CUDA graph captured), the adapter **fails closed** on
any strategy whose semantics require relocation of live buffers:

- `PAGED` block eviction, disk offload, host swap: throw `IllegalStateException` at
  `scatterNewEntries` time if `InGraphKvState.rotatingSlotMap == null` and the plan is frozen.
- The only relocation-safe point is the session/prefill boundary (`prefillWarmupAndFreeze`
  before freeze, or `close()`); the adapter exposes `boolean canRelocateNow()` from the state's
  lifecycle phase so managers consult rather than assume.

This preserves the pointer-stability contract that ADR 0118 documents as the reason
`GROWABLE` is unimplemented. Nothing in this ADR changes frozen-plan behavior.

### 3. Android collapse (consumer side)

`ModelPreparationOptions.kt` becomes the **single source of truth** for the preparation
vocabulary. Add one field:

```kotlin
enum class KvCacheStrategyOption(val label: String, val wire: String) {
    STATIC_IN_GRAPH("In-graph static (default)", "static_in_graph"),  // today's behavior
    QUANTIZED("Quantized (dtype by KvCacheOptimization)", "quantized"),
    ;
}
data class ModelPreparationOptions(
    ...,
    val kvCacheStrategy: KvCacheStrategyOption = KvCacheStrategyOption.QUANTIZED, // current default == INT8 path
)
```

Collapse points (each deletes a hand-rolled copy):
- `ChatViewModel.preparationFields` — replaced by
  `ModelPreparationOptions.toDiagnostics(): Map<String,String>`.
- `AppPreferences` getter/setter — replaced by `ModelPreparationOptions.toWire(): Bundle` /
  `fromWire(bundle)` (the existing `fromWire` moves onto the data class).
- `OptimizedModelCacheRepository.profileLabel` — replaced by
  `ModelPreparationOptions.profileLabel`.
- `SdxModelPreparationProcess` extras keys — moved to the data class as constants with the
  Bundle methods.
- Wire JSON (`optionsJson()`) gains `"kvCacheStrategy"`; the canonical profile/
  `GenerationPipelineConfig.builder()` path (`SdxGgufModelPreparer:317-321`) already switches
  on `kvQuantFormat > 0` and needs only the strategy field threaded through.

### 4. What managers become reachable, and when

| Strategy | Reachable after this ADR | Notes |
|---|---|---|
| `STATIC` (in-graph) | ✅ default | identical behavior, now selected through the vocabulary |
| `QUANTIZED` | ✅ | dtype via existing `KvCacheOptimization` (V2 live storage) |
| `TURBOQUANT` | ✅ passthrough | config fields already exist; Android toggle added |
| `PAGED`/tiering/offload | ❌ rejected under frozen plans | requires ADR 0118 VMM work first; adapter surfaces `canRelocateNow()` so the UI can gray them out instead of failing at runtime |

The UI gains one dropdown ("Cache strategy") with unsafe entries disabled + tooltip, replacing
the implicit "everything is STATIC/QUANTIZED" assumption. No new strategy claims runtime
semantics it does not have.

### 5. Migration

One release: adapter + vocabulary land together. `kvQuantFormat` wire values 0–3 are
unchanged (bit-compatible with existing SDZ caches); `"kvCacheStrategy": "quantized"` is the
new default spelling of today's behavior, so no cache invalidation occurs. The banned
`StaticKvCacheDecodeLoop` is **not** revived; the adapter fronts `InGraphKvState` directly.

## Consequences

**Positive**
- Five Android field-list copies collapse to one data class; vocabulary drift becomes
  impossible at compile time.
- The manager family gains a single, state-complete entry point (`stateSegments()`) that
  covers hybrid-recurrent models, instead of three orphaned entry points that assume classic
  attention.
- `PAGED`/offload become honestly represented in the product (grayed out with the ADR 0118
  dependency noted) instead of absent-but-implied.

**Negative / risks**
- `KvCacheManager` interface grows one default method (source-compatible; implementors in-tree
  updated in the same commit).
- The adapter must track `InGraphKvState` lifecycle phases; a miss would let a relocation
  strategy run mid-decode — mitigated by the fail-closed rule and by `canRelocateNow()`
  being derived from the state's own flags, not caller claims.
- `UnifiedKvCacheManager` remains gated behind `StaticKvCacheDecodeLoop` for classic models;
  it does **not** silently become the production path.

**Neutral**
- `TieredKVCacheManager`, `BeamKVCacheManager`, `KVCacheDiskOffloader`, `KVCacheHostOffloader`
  stay orphaned until ADR 0118 (VMM) or speculative/beam features call for them; they are not
  deleted (they are referenced by tests/policies) and not wired preemptively.

## Related
- ADR 0118: Virtual-Memory-Reserved Growable KV Cache (CUDA VMM) — the prerequisite for
  relocation-capable strategies under frozen plans.
- `GenerationPipeline:1295` route contract — unchanged; the adapter fronts `InGraphKvState`,
  never `StaticKvCacheDecodeLoop`.
- Frozen multi-plan lease retention investigation (2026-09) — `retainExternalInputsForPlan`
  per-plan references interact with pooled scratch; any manager work must respect the
  single-owner pooled-buffer rule until that fix lands.
