---
name: CUDA Qwen stream ordering fix May 4
description: "[project] GDN state feedback uses wrong stream — assign() memcpy on LC default stream while plan reads on DSP stream. Fixed with explicit cudaMemcpyAsync on decode loop stream."
type: project
---

## Stream Ordering Bug in GDN State Feedback (May 4 2026)

**Status**: Fix applied, build in progress

### Problem

CUDA Qwen SLOT_BY_SLOT: first decode step correct (token 271), second step produces token 58 (wrong, should be 248068). The GDN state output from step 0 is not properly visible to step 1.

### Root Cause

In `autoregressive_decode.cu`, after `plan->execute()`, the GDN state feedback used `dst->assign(src)`. The `assign()` function uses `thisArray->getContext()->getCudaStream()` for its D2D memcpy (NDArray.hXX:3378-3385). Since plan output arrays have `LaunchContext::defaultContext()` as their context, the memcpy runs on the LC DEFAULT stream.

But `plan->execute()` uses a DSP execution stream (`execCtx->dspStream`). When `executeSteadyState()` is used, the cross-stream event ordering exists between the DSP stream and LC default stream. But there are edge cases (null lcDefaultStream, same stream) where the event is skipped, leaving a race condition.

### Fix

Replace `dst->assign(src)` with explicit `cudaMemcpyAsync` on `*stream` (the decode loop's stream, which IS the DSP stream passed to plan->execute()):

```cpp
size_t bytes = src->lengthOf() * src->sizeOfT();
cudaMemcpyAsync(dst->specialBuffer(), src->specialBuffer(),
                bytes, cudaMemcpyDeviceToDevice, *stream);
dst->tickWriteDevice();
```

This guarantees FIFO ordering: plan kernels → memcpy → next plan kernels, all on the same CUDA stream.

### File
`libnd4j/include/ops/declarable/helpers/cuda/autoregressive_decode.cu` lines 645-674

### Next Steps
- Verify Qwen outputs correct tokens with stream fix
- If correct: revert `execute()` → `executeSteadyState()` for performance
- Then benchmark VLM for performance regression check

**Why:** Stream ordering race causes stale GDN state on second decode step.
**How to apply:** Always use the decode loop's stream for state feedback, never rely on array context streams.
