---
name: vlm-decode-perf-state-apr28
description: VLM decode 52.00 tok/s after SkipSimplifiedLayerNormalization fusion + DSP infra fixes. GPU compute (14.9ms→~13ms) is bottleneck.
type: project
---

## VLM Decode Performance State — 2026-04-28

### Current Result
- **52.00 tok/s** (250 tokens, SmolDocling-256M, RTX 4090, batch=1 seq=1)
- Target: 100+ tok/s
- Correctness: PASS
- Commit: dc97577cee

### Optimization History
| Change | tok/s | Delta | Commit |
|---|---|---|---|
| Baseline | 48.34 | — | prior |
| GATHER_DIAG gate + redispatch skip + reusable array | 48.67 | +0.7% | dc97577cee |
| **SkipSimplifiedLayerNormalization registration** | **52.00** | **+7.6%** | dc97577cee |

### Per-Step Timing (measured pre-SkipSimplified fix, needs re-measurement)
- plan (executeSteadyState + compositeReplay): 2.7ms (async queue)
- cudaStreamSynchronize: 14.9ms → est ~13ms now (GPU compute bottleneck)
- postSync: ~50us
- total: ~17.7ms → est ~19.2ms now (52 tok/s)

### Decode Plan Structure
- 1 monolithic merged CUDA graph, 0 gaps
- GPU compute is the bottleneck, not DSP infrastructure
- cudaStreamSynchronize accounts for ~85% of per-step time

### Files Changed (committed)
1. `DynamicShapePlanExecutor.java` — GATHER_DIAG gate, redispatch skip, reusable array
2. `MicrosoftOnnxExtensions.kt` — SkipSimplifiedLayerNormalization registration

### Bottleneck Analysis
At 52 tok/s = 19.2ms/step. Need 10ms/step for 100 tok/s.
- GPU kernel execution inside monolithic CUDA graph: ~13ms (est)
- Per-step C++ overhead (wire, plan call, post-sync): ~6ms
- Main GPU consumers: matmul (35%), attention (25%), gather (11%), sigmoid (9%)
- Matmuls at batch=1 seq=1 are tiny GEMVs — cuBLAS launch overhead dominates

### Next Optimization Targets (one at a time)
1. **--dsp-timing run** to get updated per-step breakdown with new model
2. **TF32 matmul mode** — faster FP32 compute for matmul-heavy workload
3. **--op-timing run** to verify kernel count reduction from fused model
4. **Further ONNX op fusions** — check for more decomposed ops
5. **Batch matmul fusion** — combine multiple small GEMVs into batched GEMM
