---
name: dsp-composite-replay-async-nosync-clean
description: DspCompositeReplayTest passes cleanly with async-only DSP composite replay diagnostics
type: project
---

**Date:** 2026-05-27 Asia/Tokyo
**Test:** `DspCompositeReplayTest` with full DSP diagnostics passed 181/181. Log: `/tmp/dsp-composite-replay-nosync.log`; report: `/tmp/dsp-composite-replay-nosync.json`.

**Diagnostic counts:** rawFallback=0, lifecycle=0, missingOrdered=0, capturePreallocWaitFail=0, ZERO_KERNEL_SBS=0, captureProducedNoKernels=0, COMPOSITE_CAPTURE_FAIL=0, REPLAY_SUCCESS=1798, COMPOSITE_REPLAY_EXIT=1336. No `cudaDeviceSynchronize`, `cudaStreamSynchronize`, or `cudaEventSynchronize` in the tee log.

**Coverage value:** This exercises mixed native/Triton composite replay, gap/island sequencing, merged island replay, shape transitions, and late-step correctness after the async prealloc event fix.

**Milestone:** 0ea5ddcf.
