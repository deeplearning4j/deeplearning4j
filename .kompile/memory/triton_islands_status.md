---
name: triton-islands-status
description: "Triton island/gap status: skip_rms_norm emitter fully implemented, all gaps captured into merged graph, no fragmentation issue"
type: project
---

## Triton Island/Gap Status (2026-04-28)

### skip_rms_norm Triton Emitter — FULLY IMPLEMENTED
- TritonIRBuilder.cpp: registered as NORMALIZATION with `isTritonMappable=true`
- TritonIRBuilder_emitters.cpp: shares `rmsnorm` branch (residual add handled by module builder)
- TritonIRBuilder_module.cpp: both 1D and sectioned paths handle skip input + hidden output

### Compilation Config (OPTIMAL)
- `compileAll=true`, `tritonIncludeTypes` includes NORMALIZATION
- Only ELEMENTWISE sections compiled as Triton kernels (101 sections)
- 12 NORMALIZATION sections route to native but are capture-safe → folded into merged graph

### Composite Replay Structure
- 508 Triton islands + gaps → merged into 1 CUDA graph (`mergedGroups=1`)
- `gapExec=0us` — all gaps captured through stream capture
- `mergedLaunch=~2100us` — single `cudaGraphLaunch` per step

### Diagnostic Fix (2026-04-28)
Fixed 5 functions in both CPU (`NativeOps_dsp.cpp`) and CUDA (`NativeOps_dsp.cu`) to be composite-aware. Also added `hasCompositeHandles()` awareness to CPU stubs (`_cuda_stubs.cpp`): cleanup, free, and count functions now handle composite handles. `hasCompositeHandles()` is defined in `_gpubackend.cpp` which compiles on ALL platforms (not GPU-only).

**Why:** Previously reported "CUDA/EMPTY 0 replays" because diagnostics only checked the monolithic replayHandle (an EMPTY sentinel for composite captures).

**How to apply:** The island/gap structure is working correctly. Don't try to reduce gap count or improve island coverage — the merged capture already handles everything. Focus on reducing total op count inside the graph.
