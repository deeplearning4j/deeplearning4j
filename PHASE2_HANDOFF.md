# Phase 2 Handoff: Replay-Unit Consolidation

## Mission

Consolidate Phase 1's correct ordered replay schedule into **fewer, larger, still-phase-closed** replay units.  
**The job is not to "make replay work"** — that's done. The job is to reduce replay-unit count and kernel launches without reopening lifecycle variance.

## Entry Criteria (ALL MUST BE TRUE)

- [x] Internal-gap mixed segments no longer replay out of order
- [x] View/layout prep handled as phase recipes, not fallback execution
- [x] Shape-only prep folded or explicitly classified
- [x] Materializing prep runs as ordered standalone replay units
- [x] Standalone graph tests pass (15/15, 0 failures)
- [x] Decode validation passes
- [x] Build clean: libnd4j + nd4j-cuda-12.9 with `-Dlibnd4j.triton=ON`

## Hard Invariants (DO NOT VIOLATE)

1. **No fallback transitions.** Never add a "merge if it works, else fall back" path.
2. **No post-replay internal-gap fixups.** No running extra ops after replay to "clean up."
3. **No silent downgrade from replay to slot-by-slot.** If consolidation fails, fail the consolidation — don't silently skip it.
4. **No monolithic graph spanning unsupported internal gaps.** Phase 1 already fixed this; don't reintroduce it.
5. **Every consolidation must preserve a phase-closed replay schedule.** Merged units must not cross phase boundaries.
6. **If a prep op is still materialized, document exactly why.** Name the consumer kernel and why it needs dense data.

## Current Evidence Files

| File | Purpose |
|------|---------|
| `/tmp/dsp_invalid_segment_buckets.txt` | Invalid segment bucket distribution |
| `/tmp/dsp_invalid_segment_composition.txt` | Full segment composition |
| `/tmp/dsp_decode_phase_contract.log` | Op histogram and view-capable counts |

## Key Source Files

| File | Role in Phase 2 |
|------|----------------|
| `libnd4j/include/graph/impl/NativeDynamicShapePlan_gpubackend.cpp` | Replay schedule signature, consolidation planner, diagnostics, validation |
| `libnd4j/include/graph/gpu/TritonIRBuilder_sections.cpp` | Section re-coalescing rules after Phase 1 cleanup |
| `libnd4j/include/graph/gpu/SectionTypeConfig.h` | Only if legality/profitability decisions need config exposure |
| `libnd4j/include/graph/gpu/OpCategoryTable.h` | Semantic op classification (already used by Phase 1) |
| `libnd4j/include/graph/gpu/ViewRecipe.h` | Phase 1 view recipe types (reference only) |

**Avoid touching** `libnd4j/include/graph/NativeDynamicShapePlan.h` unless there is absolutely no other option. Phase 1 already modified it; further changes risk cache invalidation.

## Priority Buckets

| Bucket | Count | Priority | Target Outcome |
|--------|------:|----------|----------------|
| `concat_ladder+gather_ladder` | 37 | **P0** | Shape/view churn disappears; adjacent payload units merge |
| `attention_tail+concat_ladder+gather_ladder` | 21 | **P0** | Attention absorbs prep; ≤1 materializing unit before attention |
| `attention_tail` | 8 | **P1** | Attention consumes logical layout directly |
| `attention_tail+stack_chain+concat_ladder+gather_ladder` | 1 | **P1** | Stress test; if this works, the rest follow |
| `gather_ladder [200-399]` | 1 (large) | **P1** | Simplest large-scale consolidation case |
| `simple_const_gather` | 22 | **P2** | Shape-only pieces disappear |
| `kv_mask_or_slice+concat_ladder+gather_ladder` | 1 | Defer | Only if it falls out naturally |
| `normalization_tail` | 1 | Defer | Only if it falls out naturally |

## Op Types To Target

| Category | Ops | Phase 2 Treatment |
|----------|-----|-------------------|
| DATA_MOVEMENT | gather, concat, stack, tile | Merge adjacent materializing units when phase-closed |
| SHAPE_MANIPULATION | broadcast_to, reshape_no_copy, permute, expand_dims, squeeze, strided_slice | Already recipes after Phase 1; absorb into consumers |
| CONSTANT_GENERATION | shape_of, create, range, ones_as | Already folded after Phase 1; verify no regression |
| FUSED_ATTENTION | onnx_multi_head_attention | Consume more logical layouts directly; eliminate prep kernels |

### Interpretation Rules

- `reshape`, `reshape_no_copy`, `permute`, `expand_dims`, `squeeze` → **recipe territory** after Phase 1. Do NOT re-execute as materializing.
- `broadcast_to` and `strided_slice` → **classify semantically**. If they allocate or repeat data, they're materializing. If they're zero-copy/view-like, keep as recipes. Use `OpCategoryTable` traits, not name matching.
- `gather` → **shape-only ONLY if it reads shape/meta tensors**. Any payload gather stays materializing.

## Concrete Deliverables

### 1. Replay Schedule Signature

Add a stable "replay schedule signature" for each segment in `NativeDynamicShapePlan_gpubackend.cpp`:

```cpp
struct ReplayScheduleSignature {
  enum UnitKind { TRITON_ISLAND, VIEW_RECIPE, SHAPE_RECIPE, MATERIALIZED_PREP };
  struct Unit { UnitKind kind; int startSlot; int endSlot; TritonOpCategory primaryOp; };
  std::vector<Unit> units;
  uint64_t hash;  // Stable hash for cross-step comparison
};
```

The signature encodes: ordered units, unit kind, participating slots, op types. This becomes the basis for deciding when two decode steps can share the same replay structure.

### 2. Consolidation Pass

Add a consolidation pass that merges adjacent units when:
- The merged unit remains phase-closed (no cross-phase dependencies)
- Buffer pointers are shape-stable across the merge boundary
- The profitability gate passes (see §5)

**Do NOT** use generic "merge nearby sections" heuristics. Use **explicit lowering rules**:

```
RULE: If unit[i] is MATERIALIZED_PREP(gather) AND unit[i+1] is MATERIALIZED_PREP(concat)
      AND concat consumes gather's output as its sole non-constant input
      → Merge into single GATHER_CONCAT unit

RULE: If unit[i] is VIEW_RECIPE(reshape/permute) AND unit[i+1] is TRITON_ISLAND
      → Absorb view recipe into Triton island's argument pre-processing

RULE: If unit[i] is SHAPE_RECIPE AND unit[i+1] is TRITON_ISLAND
      → Fold shape result into Triton island's shape inference; skip standalone unit
```

### 3. Explicit Fusion Lowerings

Build at least these dedicated fusion paths:

| Pattern | Lowering | Expected Savings |
|---------|----------|-----------------|
| gather + concat | Single fused kernel with indexed read + concatenating write | 2 kernels → 1 |
| gather + reshape/view recipe | Gather writes directly into reshaped layout | 2 units → 1 |
| stack + concat | Fused stack-and-concat (both are axis manipulations) | 2 kernels → 1 |
| broadcast_to + reshape_no_copy | Broadcast writes directly into target layout | 2 kernels → 1 |

### 4. Attention ABI Reduction

Teach attention lowering to consume more logical layouts directly:

- `reshape_no_copy` → attention reads with adjusted indexing
- `permute` → attention reads with adjusted dimension order
- `expand_dims` → attention treats 3D as 4D with implied dim=1
- `broadcast_to` (GQA/MQA) → attention handles key/value expansion internally
- Small `concat` (2-3 tensors, axis 0 or 1) → attention reads multiple inputs
- Small `stack` → same as concat

**Goal:** Remove prep kernels before `onnx_multi_head_attention`, not just move them around.

### 5. Profitability Gate

Every consolidation decision must pass:

```
if (kernelsRemoved >= 1) AND
   (bytesMaterialized < threshold OR materialization is fundamental) AND
   (unit is reusable across decode steps OR launch count decreases)
   → APPROVE consolidation
else
   → REJECT; log reason
```

This prevents over-fusing tiny prep that adds complexity without reducing replay cost.

## What This Phase Should NOT Do

- ❌ No fallback transitions
- ❌ No post-replay "fixup" execution
- ❌ No silent downgrade from replay to slot-by-slot
- ❌ No merging of units that cross phase boundaries just to recover a monolith
- ❌ No reintroducing capture buffers
- ❌ No benchmark-only fusions that weaken correctness rules
- ❌ No one-off workaround flags or caller-side guards

## Test Plan

### New Test File: `TritonReplayConsolidationTest.java`

Required test cases:

1. `gather -> concat -> reshape/view recipe` — verify fused lowering
2. `shape_of -> gather -> concat -> reshape` — verify shape-only folding
3. `stack -> concat -> attention` — verify attention-tail absorption
4. `broadcast_to -> reshape_no_copy -> attention` — verify view absorption
5. Minimal reproduction of `[200-399]` gather ladder — verify large-scale consolidation
6. Minimal reproduction of `[2676-2730]` attention tail — verify stress test
7. Replay invariance across multiple decode steps with same frozen shapes

### Extend Existing Tests

- `TritonDataMovementTest.java` — consolidation tests for gather/concat/stack ladders
- `TritonFusedAttentionTest.java` — attention-tail absorption tests
- `TestDspValidation.java` — full decode validation after consolidation

### Assertions

- Same outputs as Phase 1 baseline
- Same ordered replay signature across decode steps
- **Fewer replay units** than Phase 1 baseline
- **Fewer kernels per token** than Phase 1 baseline
- No new TRITON_REPLAY_PHASE_VIOLATION
- No address-drift invalidations caused by consolidation

## Performance Validation

Measure before/after on targeted buckets:

| Metric | How to Measure |
|--------|---------------|
| Replay units per decode step | DSP diagnostics: count units in signature |
| Kernels per decode step | TritonGraphBackend counters |
| Standalone prep kernels eliminated | Diff of replay schedules |
| Tokens/sec | `platform-tests/run-benchmark.sh` |

## Build Commands

```bash
# Native build (always use this):
/home/agibsonccc/dev-apps/mvn/bin/mvn -Pcuda -Dlibnd4j.chip=cuda \
  -Dlibnd4j.buildthreads=12 -Dlibnd4j.triton=ON \
  -pl libnd4j,:nd4j-cuda-12.9 clean install -DskipTests

# Focused test:
cd platform-tests && /home/agibsonccc/dev-apps/mvn/bin/mvn test \
  -Dtest=TritonReplayConsolidationTest

# Full validation:
cd platform-tests && ./run-benchmark.sh
```

## Definition of Done

This phase is complete when:

1. ✅ Replay remains strictly phase-closed on all targeted buckets
2. ✅ No new TRITON_REPLAY_PHASE_VIOLATION
3. ✅ No new address-drift invalidations caused by consolidation
4. ✅ Replay-unit count **decreases** on targeted buckets relative to Phase 1 baseline
5. ✅ Kernel count per decode step **decreases** on targeted buckets
6. ✅ Throughput from `run-benchmark.sh` improves or at minimum does not regress
7. ✅ Any remaining materialized prep step is **documented** with the consuming kernel reason
8. ✅ All new tests pass
9. ✅ Full decode validation passes

## What Good Looks Like

- Phase 1 produced a **correct** ordered replay schedule.
- Phase 2 turns that schedule into **fewer reusable** replay units.
- Fusion improves because prep is either **absorbed** or **proven necessary**, not because lifecycle rules got softer.
- The final logs explain **exactly which prep remains and why**.
- Performance improves because of **fewer replay units**, not because of hidden lifecycle workarounds.
