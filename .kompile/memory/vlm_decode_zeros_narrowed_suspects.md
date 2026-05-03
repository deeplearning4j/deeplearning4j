---
name: vlm-decode-zeros-narrowed-suspects
description: "Narrowed suspects after 12+ agents: segDispatchWarmup ruled out, weight protection safe, focus on autoregressive_decode context setup"
type: project
---

# VLM Decode Zeros — Narrowed Investigation (2026-05-02, round 3)

## RULED OUT (with evidence)

### segDispatchWarmup demotion — SAFE
- phaseWarmup (initial path) clears BOTH state_ AND cachedOutputShapes
- segDispatchWarmup only fires during mid-replay invalidation, not initial warmup
- State demotion makes shapeCacheValid() return false — stale shapes inaccessible
- Frozen fast-path can't fire during segDispatchWarmup (all slots demoted to WARMUP)

### Weight DataBuffer corruption via destroySession — SAFE
- Ordering correct: destroySession reads protectedConstantBuffers BEFORE executor.close() nulls it
- Pass 1 (variables scan) + Pass 2 (executor protection) + Pass 3 (native address) all work
- trimMemoryPool only frees pool-managed free blocks, not live allocations
- associateArrayWithVariable is NOT called during DSP execution
- embeddingTable is never refreshed but points to a live CONSTANT array — safe

### DataBuffer::expand() aliasing — LOW RISK
- expand() operates on plan-owned output arrays (KV cache slots), NOT weight arrays
- It reallocates both host and device buffers, copies data, frees old
- Has throwIfFrozen() guard — throws if frozen plan references it
- New pointers are given to the plan's output slots — the plan knows about them

## STILL SUSPECT

### #1: autoregressive_decode's OpaqueContext setup (HIGHEST)
- C++ op zeros ALL outputs unconditionally at entry (lines 271-276)
- If decode loop produces zero logits OR exits early → output stays all-zeros
- OpaqueContext carries ALL external inputs including weights
- For second call: new executor creates new OpaqueContext
- QUESTION: Is the new OpaqueContext properly populated with weight arrays?
- QUESTION: Does getCachedOpContext() create a fresh context with all external inputs?

### #2: markExternalInputVariable → invalidateForRebuild interaction
- autoregressive_decode calls markExternalInputVariable for all mutable inputs
- This triggers invalidateForRebuild → resetExecuteCount → executeCount_=0
- Same for both first and second calls — shouldn't be differential
- BUT: invalidateForRebuild now also calls resetFrozenConstantDetection (new in broken commit)
- Could resetFrozenConstantDetection corrupt frozen constant state on a fresh plan?

### #3: KV max-allocation timing
- configureMaxAllocationForKvCache called BEFORE setShapesFrozen(true) — correct
- setOutputSlotMaxSizes calls db->expand() on output slot DataBuffers
- Plan is NOT frozen at this point — throwIfFrozen won't fire
- Auto-configure at Java line 3193 has maxKvCacheLength=0 at first decode — doesn't fire
- Only the explicit configure fires — one call, correct

## KEY ARCHITECTURAL FACT
- Second plan goes through IDENTICAL lifecycle as first plan
- Both have brand new C++ plans, fresh executors, fresh sessions
- The ONLY difference is that model weights went through a destroySession cycle
- If weights are intact (verified safe), the second plan should work identically

## NEXT INVESTIGATION NEEDED
1. How OpaqueContext is created and populated for the autoregressive_decode op
2. Whether getCachedOpContext() returns a context with ALL weight arrays populated
3. What happens if any external input in the OpaqueContext is null/freed
4. The exact Java code path from executor.getCachedOpContext() to the C++ op invocation
