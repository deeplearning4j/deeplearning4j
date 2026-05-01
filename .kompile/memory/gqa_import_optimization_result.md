---
name: gqa-import-optimization-result
description: "GroupQueryAttention→OnnxMultiHeadAttention import change: 0% perf impact, reverted"
type: project
---

## GQA Import Optimization Attempt (2026-04-29)

**Change:** Modified GroupQueryAttention.kt to emit single `onnx_multi_head_attention` op instead of manual Q@K^T→softmax→@V + repeat_kv subgraph.

**Result:** 0% impact (50.23 vs 51 tok/s baseline). Reverted.

**Why:** The `onnx_multi_head_attention` op internally does the same GPU compute (reshape, FlashAttention forward4D). Routing to `fusedGQADecodeCuda` instead of `forward4DDecode` made no difference for SmolDocling's small config (9Q/3KV heads, headDim=64). The prior `GQA forward4DDecode` attempt also failed (-31%).

**Key finding:** The attention kernel dispatch path (fusedGQA vs cuBLAS GEMV) is NOT the bottleneck. Both paths produce ~same GPU compute time. The 14ms syncOnly is dominated by the sheer number of ops (509 islands) and memory bandwidth, not kernel choice.

**Decoder op histogram (with change applied):**
onnx_multi_head_attention=30, matmul=211, reshape=674, gather=365, expand_dims=190, concat=184, cast=133, reshape_no_copy=181, shape_of=124, permute=120, fused_rope=60, skip_rms_norm=60, stack=60, Where=64, add=63, broadcast_to=62, equals=62, multiply=60, sigmoid=30, tile=1
