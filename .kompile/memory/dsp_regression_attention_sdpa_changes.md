---
name: dsp-regression-attention-sdpa-changes
description: Attention and SDPA changes since 9bb2680e2b — MKL prefill bias, onnx_mha, FlashAttention, dot_product_attention
type: project
---

## Attention & SDPA Changes Since 9bb2680e2b (May 2 2026)

### MKL SDPA (sdpa.cpp) — CPU

**Heap overrun fix (9bb2680e2b — LAST GOOD COMMIT):**
- Scratch buffer sized for seqQ*seqKV instead of just seqKV
- Also fixed invalid cblas_sgemm_batch_strided strides
- This was the LAST GOOD commit

**Prefill bias bug (LATENT, UNFIXED):**
- File: `libnd4j/include/ops/declarable/platform/mkldnn/sdpa.cpp`
- Decode path (seqQ=1, lines 696-708): biasPtr applied via cblas_saxpy BEFORE softmax ✓
- Prefill path (seqQ>1, lines 728-776): biasF32 DELETED at line 728 before prefill loop ✗
- Causal masking completely absent during CPU prefill
- Model attends to future tokens → wrong attention weights → cascading error
- Fix: move biasF32 deletion AFTER prefill loop, add bias to scores between Q@K^T and softmax
- For prefill: bias shape [1,1,seqQ,seqKV] → apply row-by-row to each head's [seqQ,seqKV] score matrix

**executeSDPA4D_WithBias (COMMITTED):**
- Dead code — never called from any path
- Can be removed in cleanup

### onnx_multi_head_attention.cpp — BOTH

**AttentionWorkspace removed (COMMITTED):**
- Output reshaped directly instead of through workspace
- No nullify before FlashAttention — output buffer may contain stale data
- On CUDA: if previous step's data is in device buffer, no memset to clear it

**In-place KV write (COMMITTED):**
- New `useInPlaceKv` path using `kvInPlaceWrite` helper
- Writes directly to KV cache buffer instead of concat + copy
- Risk: dtype mismatch if mixed-type auto-cast at entry casts key/value but kvInPlaceWrite uses original buffer

**syncToDevice removed (COMMITTED):**
- Removed from KV concat path
- On CUDA: host buffer may be authoritative after concat → GPU reads stale device data
- On CPU: no effect (host IS device)

**Mixed-type auto-cast at entry (COMMITTED):**
- Casts key/value/pastKey/pastValue to query dtype
- If downstream code accesses original uncasted buffers, dtype mismatch

### dot_product_attention_v2.cpp — BOTH

**V/K rank auto-promotion 3D→4D (COMMITTED):**
- Automatically promotes rank-3 V/K to rank-4
- If promotion assumptions are wrong, attention layout is wrong

**rankOf() >= 2 guard for KV cache (COMMITTED):**
- Prevents crash on scalar/1D inputs
- Positive defensive check

**Prefill bias fallback at input[8] (COMMITTED):**
- Looks for bias at input index 8 as fallback
- If wrong input is at index 8, wrong bias applied

**DECLARE_SHAPE_FN refactored (COMMITTED):**
- Uses raw inputShape pointers instead of wrapped access
- If pointer arithmetic is wrong, output shape is wrong

### FlashAttentionHelper.cpp — BOTH

**FP16/BF16 softmax overflow prevention on CPU (COMMITTED — POSITIVE):**
- Promotes to FP32 for softmax computation
- Prevents NaN/Inf in attention weights

**GQA decode fast path on CUDA (COMMITTED):**
- Separate dispatch for grouped-query attention decode
- If head grouping logic is wrong, attention pattern is wrong

**Why:** Attention is the core of every transformer model. Any bug in Q@K^T, softmax, or score@V produces wrong output for every token.
**How to apply:** The MKL SDPA prefill bias is the top CPU-specific fix. The onnx_mha syncToDevice removal is the top CUDA-specific attention risk.
