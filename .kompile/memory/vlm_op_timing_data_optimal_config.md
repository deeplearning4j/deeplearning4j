---
name: VLM op timing data OPTIMAL config
description: Per-op timing from OPTIMAL.csv showing 250-token decode costs for performance analysis
type: project
---

## VLM Op Timing Data (OPTIMAL config, 250 tokens)

Source: `platform-tests/op-timing/OPTIMAL.csv`

### Dominant Ops (by total time)
| Op | Calls | TotalMs | AvgUs | Notes |
|---|---|---|---|---|
| autoregressive_decode | 1 | 4119.4 | 4119351 | Outer loop (entire decode) |
| onnx_multi_head_attention | 90 | 168.9 | 1876 | 90 attention layers, P50=768us |
| reshape_no_copy | 604 | 138.1 | 228.7 | Many do actual memcpy (non-contiguous) |
| matmul | 604 | 22.5 | 37.2 | cuBLAS GEMM, P50=24us |
| gather | 989 | 21.2 | 21.5 | Embedding/indexing |
| concat | 552 | 15.4 | 27.9 | KV/QKV concatenation |
| broadcast_to | 248 | 13.3 | 53.7 | Mask broadcasting |
| equals | 248 | 8.5 | 34.2 | Comparison ops |
| reshape | 1725 | 8.1 | 4.7 | Mostly zero-copy views |

### Zero-Cost Ops (pure metadata, no graph nodes)
| Op | Calls | AvgUs | Notes |
|---|---|---|---|
| expand_dims | 569 | 0.39 | View only |
| permute | 360 | 0.26 | View only |

### Moderate Ops
| Op | Calls | TotalMs | AvgUs |
|---|---|---|---|
| stack | 180 | 4.1 | 22.8 |
| sigmoid | 101 | 3.1 | 30.4 |
| Where | 254 | 3.1 | 12.1 |
| cast | 526 | 2.7 | 5.2 |
| add | 191 | 2.3 | 12.2 |
| shape_of | 371 | 1.6 | 4.4 |
| multiply | 191 | 1.6 | 8.1 |
| fused_rope | 180 | 1.2 | 6.6 |
| skip_rms_norm | 180 | 0.8 | 4.5 |

### Key Insight
Total op execution time is ~415ms for 250 tokens = 1.66ms/token of GPU compute.
At 60 tok/s (16.6ms/token), only ~10% is actual compute. The rest is overhead:
- CUDA graph scheduling: ~5.5-11ms
- Stream sync: ~1-3ms
- KV scatter: ~1ms
- Token copy + embedding: ~0.5ms

**Why:** These numbers prove that graph node scheduling dominates. Reducing nodes from 2742 to ~500 would cut scheduling from ~8ms to ~1.5ms, achieving 100+ tok/s.
**How to apply:** Focus optimization on reducing graph nodes, not on making individual ops faster.
