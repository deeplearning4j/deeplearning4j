---
name: vlm-decode-zeros-deep-investigation
description: "Deep investigation findings from 9 parallel agents: expand(), frozen path, autoregressive_decode, weight protection"
type: project
---

# VLM Decode Zeros — Deep Investigation (2026-05-02, round 2)

## Critical Finding 1: expand() Changes GPU Pointers
- File: libnd4j/include/array/cuda/DataBuffer.cu lines 151-244
- expand() allocates a NEW _specialBuffer via CudaMemoryPool, copies old data via cudaMemcpy D2D, frees old buffer
- BOTH _primaryBuffer and _specialBuffer CHANGE — new pointers
- Has throwIfFrozen() guard — throws if frozen plan references this buffer
- setOutputSlotMaxSizes (NativeDynamicShapePlan.cpp:4369-4413) calls expand() on outputSlots_[slotIdx]->dataBuffer()
- These are plan-owned output arrays, NOT weight arrays — expand only touches KV cache output slots

## Critical Finding 2: autoregressive_decode Zeros Outputs Unconditionally
- File: helpers/cuda/autoregressive_decode.cu lines 271-276
- generatedTokenIds->assign(zero) runs at ENTRY — before any decode
- If the C++ function returns early for ANY reason (null plan, failed status, bad args), output stays all-zeros
- Plan handle reconstructed from two int32 iArgs (lines 183-187) — if stale pointer, UB/crash

## Critical Finding 3: markExternalInputVariable Triggers Full Plan Rebuild
- autoregressive_decode calls markExternalInputVariable() for all mutable inputs (lines 443-450)
- markExternalInputVariable → invalidateForRebuild → resetExecuteCount() → executeCount_=0
- This means EVERY autoregressive_decode call resets the plan — first step goes through phaseWarmup
- This is the SAME for both first and second generateNative calls

## Critical Finding 4: segDispatchWarmup Demotion Asymmetry
- segDispatchWarmup (gpubackend.cpp:337-350) demotes slots to WARMUP but does NOT clear cachedOutputShapes
- phaseWarmup (NativeDynamicShapePlan.cpp:2982-2987) clears BOTH state_ AND cachedOutputShapes
- This asymmetry means stale dtype in shape cache survives demotion
- segDispatchWarmup fires during phaseReplay (executeCount_=1, second execute call)
- Called from executeSegmentWithGpuGraph when lifecycleState==NEEDS_WARMUP

## Critical Finding 5: Frozen Fast-Path Gate
- Gate at slotexec.cpp:1564-1570: requires shapesFrozen_ && executeCount_ >= 2 && contextPool_ != null && frozenContextReady() && !isDynamicShape
- frozenContextReady() = state_ >= FROZEN (NativeDynamicShapePlan.h:441)
- contextPool_ entries are ALWAYS non-null (allocated at plan construction)
- Fast-path does NOT call calculateOutputShape — reuses frozen context, only refreshes input pointers
- If output buffers contain zeros from initialization and warmup didn't populate them correctly → zeros propagate

## Critical Finding 6: Second Call Creates Everything Fresh
- Brand new InferenceSession, DynamicShapePlanExecutor, Java DynamicShapePlan, C++ NativeDynamicShapePlan
- Model weight arrays survive (protected by destroySession)
- nativePlanCache C++ container survives but emptied
- The second plan goes through exact same lifecycle as the first
- SO: the bug must be in surviving state (weights) or in the cleanup between calls

## Key Question Still Open
- If everything is truly fresh for the second call, WHY does it produce zeros?
- Possible: weight DataBuffers were closed during destroySession despite protection
- Possible: CUDA memory pool recycling causes pointer aliasing between old freed buffers and new allocations
- Possible: expand() on first plan's KV slots changed a buffer that was later freed, and pool recycled the address
- Need to verify: are weight arrays actually intact after destroySession?
