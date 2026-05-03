---
name: vlm-decode-zeros-regression-full-diff-analysis
description: "Complete diff analysis 9bb2680e2b→a26ab20ba9: ranked suspects for second-plan all-zero logits"
type: project
---

# VLM Decode Zeros Regression — Full Diff Analysis (2026-05-02)

## Regression Range
- **Last good:** `9bb2680e2b` (Apr 30) — MKL SDPA prefill heap overrun fix
- **First broken:** `a26ab20ba9` (May 1) — "BROKEN: restored working tree snapshot"
- 157 files changed, ~77k insertions / ~67k deletions (monolithic snapshot commit)

## Symptom
- First `generateNative()` call: correct tokens (mythic heroes text)
- Second `generateNative()` call: all-zero logits → token 0 on every step
- Affects ALL configs: SLOT_BY_SLOT, TRITON, OPTIMAL
- prezero fix was necessary (first plan zeros) but NOT sufficient (second plan still zeros)

## Confirmed Fix Applied
- `prezeroSegmentOutputs` guard restored in NativeDynamicShapePlan_segments.cpp ~line 930
- `if (!(shapesFrozen_ && executeCount_ >= 2)) { prezeroSegmentOutputs(...); }`

## Ranked Suspects (from 8 parallel subagent investigations)

### TIER 1 — Highest Priority

**#1: KV cache max-allocation ENABLED (was explicitly disabled)**
- File: DynamicShapePlanExecutor.java, executeNativePlan ~line 3193
- Old code had comment: "Max-allocation DISABLED because giving ops wrong-shaped pre-allocated buffer causes KERNEL_FAILURE (50)"
- New code: REMOVED warning, ENABLED `configureMaxAllocationForKvCache` after first step
- Also called explicitly in GenerationPipeline.java BEFORE setShapesFrozen(true) at lines 1608 (ONNX) and 912 (GGUF)
- Double-call issue: auto-configure at line 3193 AND explicit at line 1608
- C++ side: `setOutputSlotMaxSizes` calls `db->expand(maxBytes)` eagerly on existing DataBuffers
- Risk: expand() could alias/corrupt buffers, or oversized buffers cause shape/buffer mismatch

**#2: segDispatchWarmup() INVERTED — promote→demote**
- File: NativeDynamicShapePlan_gpubackend.cpp, lines 337-350
- Old: promoted view-capable slots to FROZEN before warmup
- New: demotes ALL slots above WARMUP back to WARMUP
- CRITICAL BUG FOUND: demotion clears `state_` but NOT `cachedOutputShapes`
- phaseWarmup() clears BOTH state_ AND cachedOutputShapes — asymmetry
- Stale dtype in shape cache survives demotion → corrupts array creation

### TIER 2 — Likely Contributing

**#3: Warmup window extended + isDynamicShape gating**
- File: NativeDynamicShapePlan_slotexec.cpp, lines 2170 and 3796
- `executeCount_ == 0` → `executeCount_ <= 1` (extends warmup through first decode)
- `isDynamicShape` only set when BFS confirms dynamic upstream (was unconditional)
- Placeholder-driven shape changes (prefill→decode) don't set isDynamicShape
- Could allow premature frozen fast-path with stale context on executeCount_>=2

**#4: Weight DataBuffer corruption via destroySession()**
- File: SameDiff.java, destroySession() lines 4738-4880
- If associateArrayWithVariable replaced a CONSTANT's Java wrapper during Plan 1
- Original DataBuffer becomes unreachable via sd.variables()
- protectedAddresses (native ptr check) catches SOME but not all cases
- executor.getProtectedConstantBuffers() is still available (executor not closed yet when destroySession reads it)

### TIER 3 — Possible

**#5: setOutputSlotMaxSizes eager db->expand()**
- File: NativeDynamicShapePlan.cpp, setOutputSlotMaxSizes function
- Companion to #1 — the C++ side
- If expand() is called on a DataBuffer aliasing a weight → corruption

**#6: shapeOnlyMode_ left enabled**
- New JNI bridge: setPlanShapeOnlyMode in NativeOps_dsp.cu
- If called with true and never reset → all ops skip execution
- No evidence it's called during benchmark flow

**#7: nativeRangeSegments_ persistent map**
- File: NativeDynamicShapePlan_segments.cpp, nativeSlotCallback lambda
- Persistent GraphSegment objects stored by range key
- New plan starts with empty map — should be fine
- But old plan destructor interaction unclear

## Key Architecture Facts
- Second generateNative creates EVERYTHING fresh: new InferenceSession, new DynamicShapePlanExecutor, new Java DynamicShapePlan, new C++ NativeDynamicShapePlan
- Model weight arrays survive (protected by destroySession)
- nativePlanCache C++ object survives but is emptied
- segDispatchWarmup fires during phaseReplay (not phaseWarmup)
- It fires on executeCount_=1 (second execute call) when lifecycleState==NEEDS_WARMUP

## What's Been Ruled Out
- DataBuffer.cu syncToSpecial capture guard — no effect
- DspSegmentLifecycle.h resetExecuteCount removal — doesn't fire during benchmark
- Fused chain aliasing revert — no effect
- backfillCachedOutputShapes SHAPE_CACHED fast-exit — no effect
- phaseShapeInferenceOnly auto pre-pass disable — no effect (re-enabled)
- SameDiff.java — NO changes between good/bad commits
- InferenceSession.java — NO changes between good/bad commits
- DataBuffer.cu — NO changes between good/bad commits
- NativeDynamicShapePlan_cuda.cu — NO changes between good/bad commits
