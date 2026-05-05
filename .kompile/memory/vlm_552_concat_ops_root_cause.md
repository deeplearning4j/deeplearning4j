---
name: VLM 552 concat ops root cause
description: 552 concats are shape-assembly + KV cache from ONNX Attention import, ~18 per layer × 28 layers
type: project
---

## 552 Concat Ops Root Cause (May 4 2026)

### Source
The 552 concat ops come from ONNX import in `Attention.kt` handleMicrosoftAttention() path:
- **~16 shape-assembly concats per layer**: Build reshape target shapes dynamically (e.g., [batch, seq, heads, headDim]) by concatenating scalar dimensions
- **2 KV cache sequence concats per layer**: `_kToUse = concat(past_K, current_K)` along seq dim
- For 28-layer SmolDocling: 28 × ~20 = ~560 nodes (matches 552)

### Key Files
- `nd4j/samediff-import/samediff-import-onnx/src/main/kotlin/org/nd4j/samediff/frameworkimport/onnx/definitions/implementations/Attention.kt` — lines 165-351, handleMicrosoftAttention()
- `nd4j/samediff-import/samediff-import-onnx/src/main/kotlin/org/nd4j/samediff/frameworkimport/onnx/definitions/implementations/MultiHeadAttention.kt` — delegates to native OnnxMultiHeadAttention
- `libnd4j/include/ops/declarable/generic/nn/llm_ops.cpp` — onnx_multi_head_attention implementation, already supports in-place KV write (kvInPlaceWrite, lines 195-238)

### Fix Options
1. **Enable planOwnsKvScatter for VLM**: Inject `cache_position` variable during ONNX import → cachePositionExtIdx becomes valid → eliminates 2 KV concats/layer
2. **Route through OnnxMultiHeadAttention**: Replace handleMicrosoftAttention() fallback with native op delegation → eliminates all 16 shape-assembly concats/layer
3. **Better: Ensure shape-assembly concats are frozen constants**: During decode, shapes are FIXED. These concat ops that build [batch,1,heads,dim] shapes should be FROZEN_CONSTANT and skipped. If freeze detection is working, they should already be eliminated.

### Critical Question
The model ALREADY uses onnx_multi_head_attention (90 calls in timing). So SmolDocling IS going through the native attention op. But 552 concat nodes still exist in the graph. Are they being properly detected as FROZEN_CONSTANT? If shapes are fixed during decode, shape-assembly concats produce the same output every step → should freeze. If they DON'T freeze, that's the bug.

**Why:** Eliminating 552 graph nodes saves ~1.5ms/step (from 16.6ms → 15ms = 67 tok/s). Combined with other node reductions, can reach 100+.
**How to apply:** First verify freeze detection works for shape concats. If it does, the issue is elsewhere. If it doesn't, fix freeze detection first (cheapest fix).
