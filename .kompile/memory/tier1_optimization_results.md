---
name: tier1-optimization-results
description: "Tier 1 results: 44→52 tok/s (+17.5%), plan overhead halved, postSync 300→35us, correctness PASS"
type: project
---

# Tier 1 Optimization Results (2026-04-27)

**Before**: 44.01 tok/s (19.2ms/step)
**After**: 51.72 tok/s (17.1ms/step)
**Delta**: +7.7 tok/s (+17.5%)
**Correctness**: PASS (identical token sequence)

## Per-Change Impact

| Change | Metric | Before | After | Savings |
|---|---|---|---|---|
| 1a: D2D token store (eliminate p() H2D) | postSync | 300us | 35us | -265us |
| 1b: Mask/pos updates before sync | plan | 4200us | 2280us | -1920us |
| 1c: Pinned memory for D2H | syncOnly | ~14500us | ~14800us | marginal |

## Key Insight

Moving mask/position update kernels BEFORE `cudaStreamSynchronize` was the biggest win. These 5 kernels (attn mask, causal mask, attn_mask_reformat, position_ids, D2D token copy) now launch on the stream and overlap with the CUDA graph replay execution. The CPU enqueues them in ~20us, and they execute on GPU while the graph is still running.

## Remaining Bottleneck

`cudaStreamSynchronize` still blocks for ~14.8ms (GPU graph execution time). The `plan` CPU overhead dropped to 2.3ms. Total step = 2.3ms (CPU) + 14.8ms (GPU sync) + 35us (post) ≈ 17.1ms.

To reach 100 tok/s (10ms/step), need Tier 2: eliminate the sync entirely via device-side token pipeline.
