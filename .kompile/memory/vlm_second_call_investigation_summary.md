---
name: vlm-second-call-investigation-summary
description: "Master investigation summary: 15+ agents, all ruled-out causes, remaining suspects for second-call zero logits"
type: project
---

## VLM Second-Call Zero Logits — Investigation Summary (2026-05-02)

### Problem
Second `generateNative()` call produces all-zero logits (`[216, 49229, 0, 0, 0, ...]`) instead of "mythic heroes" text. First call works perfectly.

### Confirmed Fix (necessary but insufficient)
- **prezeroSegmentOutputs guard** (NativeDynamicShapePlan_segments.cpp:~930): `if (!(shapesFrozen_ && executeCount_ >= 2)) { prezeroSegmentOutputs(seg, stream); }` — prevents zeroing frozen output buffers every step.

### RULED OUT (with evidence from 15+ parallel agents)

| Suspect | Why Ruled Out |
|---------|---------------|
| segDispatchWarmup demotion | Safe — phaseWarmup handles initial warmup, bypasses segDispatchWarmup |
| Weight DataBuffer corruption via destroySession | Ordering correct, all 3 protection passes work |
| DataBuffer.cu changes | NO changes between good/bad commits |
| SameDiff.java changes | NO changes between good/bad commits |
| InferenceSession.java changes | NO changes between good/bad commits |
| KV max-allocation double-configure | No double-configure, proper reset on new plan |
| BenchmarkConfigApplier.resetModelState race | Cleanup is correct — resetModelState fully cleans before generateNative |
| generateNative lifecycle guard bypass | Bypassed correctly — resetModelState already did cleanup |
| invalidateForRebuild on fresh plan | No-op — needsFullInvalidation=false on fresh plan |
| markExternalInputVariable on fresh plan | Only sets variable flag, no invalidation |
| executeSteadyState warm-up skipping | 4-way gate forces 3 full execute() calls before steady state |
| External input H2D sync in steady state | Same platformTryFrozenFastPath for both plans |
| fused chain aliasing | Investigated, not root cause |
| backfillCachedOutputShapes | Investigated, not root cause |
| phaseShapeInferenceOnly disable | Pre-pass disabled by default (shapePrePassDone_=true in constructor) |

### REMAINING SUSPECTS (ranked by likelihood)

1. **slotexec.cpp 1021-line diff** — Massive change with many interacting modifications. Key changes not yet fully traced for second-plan behavior:
   - Warmup window `executeCount_ == 0` → `executeCount_ <= 1`
   - `isDynamicShape` BFS-based check (`hasTransitiveDynamicUpstream`)
   - `discardCachedSlotArray` replacing raw delete
   - `shapeOnlyMode_` guards throughout
   - Fused chain intermediate separate allocation rework
   - Max-size allocation rework with `db->expand()`
   - Frozen fast-path gate changes at lines 1564-1570

2. **NativeDynamicShapePlan.cpp 564-line diff** — Large change, key areas:
   - `shapePrePassDone_` initialized to `true` (pre-pass disabled)
   - `setOutputSlotMaxSizes` eager expand
   - `writeOutputSlot` phase-violation check
   - `phaseWarmup` clearing cachedOutputShapes

3. **Something specific to the second plan's output slot allocation** — The first plan works, the second doesn't. The C++ plan cache is properly cleared. A fresh plan is compiled. But something in how the second plan allocates/populates output slots during its warmup phase produces zeros instead of real values.

### Key Architecture Facts Learned
- `setShapesFrozen(false)` is architecturally BANNED (throws exception)
- `markExternalInputVariable` on fresh plan is a benign no-op
- `executeSteadyState` requires `shapesFrozen && executeCount>=3 && planPhase>=REPLAYING && !verifyKernels`
- `resetModelState` order: resetSession → clearPlaceholderOverrides → clearPlaceholders → clearOpInputs → clearDynamicShapePlanCache → setGraphExecutionMode(AUTO)
- `configureMaxAllocationForKvCache` fires exactly once per plan, before freeze

**Why:** Tracking the full investigation to avoid repeating ruled-out paths.
**How to apply:** Focus investigation on the 1021-line slotexec.cpp diff and the 564-line NativeDynamicShapePlan.cpp diff. The bug is in how the second plan's output slots are populated during warmup/replay phases — something in the large diff changed this behavior.
