---
name: dsp-perf-recovery-commits
description: "Git commit reference for the DSP perf recovery chain: 10→50 tok/s after Triton refactor regression, organized by phase (infrastructure, correctness, performance)"
type: reference
---

# DSP Performance Recovery Commits (post-Triton refactor)

Reference for the commit chain that recovered from 10 tok/s back to ~50 tok/s after the Triton architecture change caused a regression from 86.7 tok/s. All on branch `ag_new_release_updates_2`.

## Pre-Triton Peak
- `52ad5cf5d5` (03-19) — 86.7 tok/s, monolithic CUDA graph (cuBLAS+cuDNN), single cudaGraphLaunch

## Infrastructure Commits (04-18 to 04-19)
- `0629d8e120` — DSP graph/Triton infrastructure: diagnostics, slot execution, OpCategory, IR builder
- `65aedd0233` — Plan-owned staging buffers for stable arg table pointers
- `943783d473` — Update capturedInputAddrKey after arg table refresh with staging addresses

## Correctness Fixes (04-20 to 04-27)
- `38081a955a` — use-after-free in writeOutputSlot (slot 51 shapeInfo nullptr crash)
- `cc7267db18` — DSP plan cache lookup + placeholder buffer protection for KV cache lifecycle
- `3370b75e83` — Unfreeze frozen constants feeding value-dependent shape ops
- `72da39b45a` — includeTypesHash in Triton SegmentCacheKey (stale cache hits)
- `533d3fa478` — Populate externalInputRanks_ to unblock FusionPass matmul+bias detection

## Performance Recovery (04-20 to 04-27, chronological)
- `3075ec44ac` — Plan cache + batchedGemm + staging re-enable (10→17.6)
- `3c6ed79824` — Frozen fast path + composite replay (6.8→23.8)
- `bc8695b7cc` — Skip frozen constants/identity ops in gap loop
- `cee34171d6` — Eliminate redundant cudaStreamSynchronize in decode token read
- `637b26cc38` — DspStreamGuard ordering + variable ext input indices
- `6f88736fcd` — Remove redundant blocking sync + debug memory queries
- `70d0b04d08` — scatter_nd_update FULLY_WRITING (eliminate prezero)
- `158f30a383` — Active gap slot cache (skip 97% of slot iterations)
- `eac54da587` — Triton island compilation + CUDA graph replay (→50.6)
- `056cb35b34` — Reclassify gap isCaptureSafe at capture time
- `020d93aa26` — Fused warp-shuffle softmax in attention decode path
- `432b78bd33` — KV scatter pre-alloc + fast-path staging buffer sync
- `78cd9d9316` — Bypass launchAsync overhead in composite replay fast path
- `b4cede988f` — Deduplicate cross-stream sync between executeSteadyState and platformTryFrozenFastPath
- `d4f8175283` — Skip error message heap alloc/free when no error is set

## Failed/Reverted (for future reference — don't re-attempt)
- `21dc91b1d6` — Mega-graph (monolithic CUDA graph with gaps) → 49.6% accuracy, BANNED
- mergeViewGaps (04-26) — -5.4% perf, view/identity gaps in CUDA graph adds overhead, REVERTED
- Dirty-generation counter (04-27) — slight regression, REVERTED
- generatedTokenIds direct write (04-27) — correctness failure (device-authoritative buffer), REVERTED
