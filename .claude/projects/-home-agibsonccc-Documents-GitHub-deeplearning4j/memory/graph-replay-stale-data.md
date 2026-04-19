---
name: CUDA Graph Replay Stale Data Root Cause
description: Graph capture bakes in H2D/D2D nodes that overwrite fresh data on replay — root cause of OPTIMAL degeneration
type: project
---

## CUDA Graph Replay Stale Data Bug (Apr 13, 2026)

**Status**: PARTIALLY FIXED — SLOT_BY_SLOT correct, OPTIMAL still degenerates

**Root cause**: During CUDA graph capture, `syncToSpecial()` in `DataBuffer.cu` records H2D memcpy nodes from pinned host workspace to device buffers. On replay, these baked-in nodes copy STALE capture-time data over fresh data computed by preceding segments.

**Evidence**:
- SLOT_BY_SLOT: CORRECT (mythic heroes text)
- DIAG_TRITON_noGC (Triton without graph capture): CORRECT
- DIAG_TRITON_noGC_VERIFY: every individual Triton kernel matches slot-by-slot
- OPTIMAL (Triton + graph capture): DEGENERATE (repeating tokens)
- Reusable embedding buffers make it worse (stable address + baked H2D = stale overwrites)
- 305 VIEW_RECIPE_FAIL per step in composite segments

**Fix applied (partial)**:
- `writeSpecial()` on external input DataBuffers before `tl_graphExecutionActive=true` (gpubackend.cpp:4213-4235)
- This prevents H2D recording for external inputs but doesn't fix internal intermediate buffer H2D nodes

**Why**: `DataBuffer.cu:840-846` creates pinned workspace copies during capture for ALL buffers with `isSpecialActual()=false`. These persist for graph lifetime. On replay, the graph's H2D nodes copy from this STALE pinned workspace.

**How to apply**: The fix needs to either (1) update pinned workspace with current data before each replay, (2) use `cudaGraphExecKernelNodeSetParams` to update H2D node params, or (3) eliminate H2D recording for internal buffers during capture by marking them device-actual.

**Fresh buffers workaround**: Using fresh buffer allocations each step (`.dup()`) forces address drift, which invalidates the captured graph and forces re-capture. This gives correct output at ~4.8 tok/s but prevents pointer stability needed for fast replay (~90+ tok/s).

**Key files**:
- `DataBuffer.cu:818-862` — syncToSpecial during capture (H2D recording)
- `NativeDynamicShapePlan_gpubackend.cpp:4206-4280` — capture setup + writeSpecial fix
- `NativeDynamicShapePlan_gpubackend.cpp:3248-3285` — composite segment slot-by-slot fallback
- `TritonGraphBackend_kernel.cu:1080-1200` — refreshArgTablesForReplay
