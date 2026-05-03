---
name: vlm-second-call-investigation-kv-maxalloc
description: KV max-allocation flow RULED OUT as root cause of second-call zero logits
type: project
---

## KV Max-Allocation Flow — RULED OUT (2026-05-02)

Investigation of `configureMaxAllocationForKvCache` across 4 overloads in DynamicShapePlanExecutor.java.

### Key Findings

1. **No double-configure**: `maxAllocationConfigured` flag + `maxKvCacheLength > 0` guard ensure it fires exactly once per plan. Auto-configure at line 3193 never fires because `maxKvCacheLength=0` during warmup decode. Only the explicit call at GenerationPipeline.java:1608 fires.

2. **Proper reset on second plan**: `clearDynamicShapePlanCache()` → `compileNativePlan()` → line 1180 resets `maxAllocationConfigured=false`. New executor starts fresh with `maxKvCacheLength=0`.

3. **`setOutputSlotMaxSizes` (C++ NativeDynamicShapePlan.cpp:4369-4413)**: Eagerly calls `db->expand(maxBytes)` on existing DataBuffers. `expand()` changes device pointers but is called BEFORE `setShapesFrozen(true)` (GenerationPipeline.java:1580-1620). So expanded buffers get captured into CUDA graphs with their final pointers.

4. **`maxAllocatedSlots_` set**: Marks slots as fixed-address, preventing reallocation on subsequent steps.

5. **The "KERNEL_FAILURE (50)" comment was removed**: KV max-allocation was re-enabled. But the flow is architecturally sound — it runs before freeze, before capture.

**Why:** Investigating whether pointer changes from `db->expand()` could cause stale device pointers in CUDA graphs on the second call.
**How to apply:** KV max-allocation is not the root cause. The flow is correct: expand → freeze → capture. No pointers change after capture.
