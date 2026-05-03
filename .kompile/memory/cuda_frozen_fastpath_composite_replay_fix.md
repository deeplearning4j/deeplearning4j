---
name: cuda-frozen-fastpath-composite-replay-fix
description: "CUDA VLM zeros fix: frozen fast-path in executeSlot skips re-execution during composite replay gap ops"
type: project
---

## CUDA Frozen Fast-Path Composite Replay Fix (2026-05-02)

**Root cause:** SmolDocling VLM produces `[216, 49229, 30341, 0, 0, 0, ...]` — first 3 tokens real, rest zero. At `executeCount_ >= 4`, `executeSteadyState` activates the fast path which uses composite replay. Gap ops in composite replay call `executeSlot`, which hits the frozen fast-path gate and reuses CACHED outputs from the previous execution instead of re-executing with fresh inputs (embeddings, masks, positions that change every decode step).

**Fix:** Added `!tl_dspReplayActive` to the frozen fast-path gate in `NativeDynamicShapePlan_slotexec.cpp:1567`, guarded by `#ifdef SD_CUDA` (since `tl_dspReplayActive` is only defined in `DataBuffer.cu`).

```cpp
if (!(shapesFrozen_ && executeCount_ >= 4 &&
#ifdef SD_CUDA
    !tl_dspReplayActive &&
#endif
    contextPool_[stepIdx] != nullptr && slot.frozenContextReady() &&
    ...)) break;
```

**Why:** `tl_dspReplayActive` is already set to `true` in `compositeReplay()` (NativeDynamicShapePlan_gpubackend.cu:1207-1370) before calling `executeSlot` for gap ops. But executeSlot never checked it, allowing the frozen fast-path to activate during replay gaps and return stale cached outputs.

**How to apply:** Any future frozen fast-path optimization must respect `tl_dspReplayActive` — gap ops during composite replay always need fresh execution since their inputs (ext inputs like embeddings, positions, masks) change every decode step.
