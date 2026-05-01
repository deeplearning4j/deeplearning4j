---
name: vlm-decode-loop-optimization-todo
description: Prioritized TODO list for VLM decode perf — Java-side opts proved ineffective, bottleneck is C++ native loop
type: project
---

# VLM Decode Loop Optimization TODO

**Goal**: 50 tok/s → 100+ tok/s
**Model**: SmolDocling-256M, RTX 4090, batch=1 seq=1 decode

## CRITICAL DISCOVERY (2026-04-29)

The VLM benchmark ALREADY uses the native `autoregressive_decode` C++ op via
`GenerationPipeline.generateNative()`. The bottleneck is NOT the Java decode loop.

## Attempted & Failed (all Java-side)

| # | Task | Result | Why |
|---|---|---|---|
| 1 | Replace causal mask putScalar with GPU kernel | **REGRESSED** 50→48.4 tok/s | Buffer is only ~5KB; ScalarSet op dispatch slower than putScalar+H2D |
| 2 | GPU-side stop-token check | **N/A** | Already done in native C++ op |
| 3 | GPU-side embedding lookup | **N/A** | Already done via embedLookupKernel in native op |
| 4 | GPU-side position/mask/inputIds updates | **SKIPPED** | Same approach as #1, proven ineffective |
| 5 | Non-zero stream for D2H readback | **SKIPPED** | Sync waits for compute that must finish anyway |

## Key Lesson
The 50 tok/s bottleneck is inside the C++ native decode loop (`executeSteadyState()`),
not in Java-side per-step overhead. All the Java-side optimizations targeted the wrong layer.

## Remaining Viable Approaches (all C++ side)

1. **Reduce 22 composite CUDA graph islands** — split ops force island boundaries
2. **Profile executeSteadyState() C++ overhead** — nsys the native loop, not the Java loop
3. **Reduce per-step cudaStreamSynchronize cost** — token readback drain
4. **Reduce KV scatter overhead** — batch scatter dispatch
5. **Reduce plan executor per-step overhead** — ext input sync, dirty tracking