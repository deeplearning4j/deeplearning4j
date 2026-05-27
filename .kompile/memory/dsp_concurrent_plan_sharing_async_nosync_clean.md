---
name: dsp-concurrent-plan-sharing-async-nosync-clean
description: Full DspConcurrentPlanSharingTest passes with clean async DSP diagnostics after prealloc event capture fix
type: project
---

**Date:** 2026-05-27 Asia/Tokyo
**Build:** `/home/agibsonccc/dev-apps/mvn/bin/mvn -Pcuda -Dlibnd4j.triton=ON -Dlibnd4j.chip=cuda -Dlibnd4j.buildthreads=12 -Dlibnd4j.log=libnd4j-build.log -pl libnd4j,:nd4j-cuda-12.9 clean install -DskipTests 2>&1 | tee cuda-build-output.log` succeeded at 08:41:01 JST.
**Test:** `DspConcurrentPlanSharingTest` with full DSP diagnostics passed 51/51 at 08:44:49 JST. Log: `/tmp/dsp-concurrent-plan-sharing-nosync-v2.log`; report: `/tmp/dsp-concurrent-plan-sharing-nosync-v2.json`.

**Key diagnostic counts:** rawFallback=0, lifecycle=0, missingOrdered=0, capturePreallocWaitFail=0, ZERO_KERNEL_SBS=0, captureProducedNoKernels=0, nativeOnly=384, GRAPH CAPTURE COMPLETE=112, READY monolithic=2604, REPLAY_SUCCESS=2604. Log contained no `cudaDeviceSynchronize`, `cudaStreamSynchronize`, or `cudaEventSynchronize`.

**Root cause/fix:** Triton preallocation readiness event was being waited inside CUDA graph capture, creating an illegal dependency on uncaptured work from another stream. Pre-capture warmup already orders the preallocation event before capture, so capture-time execution skips that wait and logs that the dependency was already ordered. This removed `COMPOSITE_CAPTURE_FAIL`, `markOomDeferred` lifecycle violations, and missing ordered range executor fallback in the concurrent plan sharing suite.

**Milestone:** 97230f27.
