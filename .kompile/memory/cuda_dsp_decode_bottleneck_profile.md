---
name: cuda-dsp-decode-bottleneck-profile
description: "Per-step decode profiling: 14ms GPU execute (74%), 4ms CPU launch (22%), Tier 1a/1b/1c all implemented, GQA decode kernel built"
type: project
---

# CUDA DSP Decode Bottleneck Profile (updated 2026-04-28)

**Model**: SmolDocling-256M, RTX 4090, batch=1 seq=1 decode
**Current**: ~53 tok/s (~19ms/step)
**Target**: 100+ tok/s (<10ms/step)

## Per-Step Timing Breakdown

```
DECODE_STEP_TIMING[50]:  total=19191us wire=0us plan=5124us  samp+sync=13739us (syncOnly=13700us) postSync=326us
DECODE_STEP_TIMING[100]: total=19252us wire=0us plan=3665us  samp+sync=15290us (syncOnly=15271us) postSync=295us
DECODE_STEP_TIMING[150]: total=18941us wire=1us plan=4742us  samp+sync=13976us (syncOnly=13954us) postSync=221us
DECODE_STEP_TIMING[200]: total=20503us wire=0us plan=4325us  samp+sync=15831us (syncOnly=15792us) postSync=347us
```

## Current Decode Loop Structure (autoregressive_decode.cu)

Per-step pipeline (after all Tier 1 optimizations):
1. **Wire ext inputs** — pointer assignments (~0us)
2. **executeSteadyState** — compositeReplay CPU launch (~4ms)
3. **D2D token copy** (Tier 1a) — `cudaMemcpyAsync(D2D)` from sampledToken to generatedTokenIds
4. **Mask/position updates** (Tier 1b) — launched BEFORE sync to overlap with GPU
5. **D2H readback** (Tier 1c) — `cudaMemcpyAsync(D2H)` to pinned memory
6. **cudaStreamSynchronize** — blocks until ALL queued GPU work completes (~14ms)
7. **Stop check** — compare nextTokenId to stop tokens
8. **KV scatter** — batched D2D copy of present KV into static buffers
9. **Embed lookup + inputIds update** — uses nextTokenId from D2H

## Critical Finding

The 14ms sync IS the GPU graph execution time, not wasted wait. Single-stream FIFO means D2H and compute cannot overlap. Reducing this requires either:
- Fewer/faster GPU kernels (fusion, kernel optimization)
- Second stream + CUDA events for CPU/GPU pipeline overlap

## nsys GPU Kernel Breakdown (50 tokens, pre-skip_rms_norm)

| # | Kernel | %GPU | Total ms | Calls | Avg us |
|---|--------|------|----------|-------|--------|
| 1 | fusedAttention3DKernel | 24.9% | 103 | 30 | 3,439 |
| 2 | simpleReduce(Mean) | 16.4% | 68 | 150 | 454 |
| 3 | simpleScalar(Sum long) | 13.5% | 56 | 12 | 4,660 |
| 4 | transformAny(Assign) | 7.5% | 31 | 2322 | 13 |
| 5 | scalarMultiply | 7.1% | 29 | 173 | 170 |
| 6 | CUTLASS GEMM | 3.7% | 16 | 160 | 97 |
| 7 | splitCuda | 2.2% | 9 | 368 | 25 |
| 8 | softMaxCuda | 2.0% | 9 | 48 | 177 |
| 9 | tileKernelDouble | 2.0% | 9 | 256 | 33 |
| 10 | Assign(float→fp16) | 1.7% | 7 | 786 | 9 |

NOTE: This nsys data is from BEFORE skip_rms_norm (commit `31e5078e02`) and BEFORE fusedGQADecodeKernel. Items 2+5 should be reduced by skip_rms_norm. Items 4+9 should be eliminated by fusedGQADecodeKernel. Need fresh nsys.
