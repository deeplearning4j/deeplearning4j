---
name: dsp-regression-helper-impl-changes
description: CPU and CUDA helper implementation changes since 9bb2680e2b — rmsNorm, fusedRoPE, FlashAttention
type: project
---

## Helper Implementation Changes Since 9bb2680e2b (May 2 2026)

### CPU helpers/cpu/rms_norm.cpp — CPU

**rmsNorm_ float32 accumulator (COMMITTED — POSITIVE):**
- Accumulates in float32 before converting back to input dtype
- Prevents FP16 overflow during norm computation
- Without this: FP16 norm computation produces NaN/Inf → zeros after softmax

**rmsNormLinear_ float32 GEMM (COMMITTED — POSITIVE):**
- Uses float32 GEMM strategy for FP16 inputs
- Prevents zeros on CPUs without AMX-FP16 instructions
- AMX-FP16 is only on Sapphire Rapids+ — most CPUs need this fallback

**Mixed-type gamma support (COMMITTED — POSITIVE):**
- Casts gamma to input dtype before applying
- Prevents silent type mismatch in norm * gamma multiplication

### CUDA helpers/cuda/rms_norm.cu — CUDA

**Dual-type gamma template <T, G> (COMMITTED — POSITIVE):**
- Eliminates gamma cast allocation on CUDA
- Template handles gamma type independently from input type
- More efficient than CPU approach (no runtime cast)

**M/K shape fix for batched rank-3+ inputs (COMMITTED — POSITIVE):**
- Correct M (rows) and K (columns) extraction for rank-3 inputs
- Without this: GEMM dimensions wrong → buffer overrun or wrong results

### CPU helpers/cpu/fused_llm_ops.cpp (~500 lines changed) — CPU

**fusedRoPE full rewrite (COMMITTED):**
- Typed dispatch via BUILD_SINGLE_SELECTOR
- Rank-3 support (was rank-4 only)
- Precomputed invFreq table for performance
- RISK: `float invFreq[512]` stack buffer — headDim > 1024 (headDim/2 > 512) overflows
- Current models use headDim=128, safe for now

**fusedRoPECached rewrite (COMMITTED — MEDIUM RISK):**
- Assumes cos/sin tensors have specific contiguous stride layout
- Non-contiguous cos/sin (from slicing larger cache) produce wrong rotary embeddings
- Wrong RoPE → wrong attention patterns → wrong outputs for every token
- Affects every layer that uses RoPE (i.e., every attention layer)

### FlashAttentionHelper.cpp — BOTH

**FP16/BF16 softmax overflow prevention on CPU (COMMITTED — POSITIVE):**
- Promotes to FP32 for softmax computation
- Prevents NaN/Inf in half-precision attention

**GQA decode fast path on CUDA (COMMITTED):**
- Separate dispatch for grouped-query attention in decode mode
- If head grouping calculation is wrong, GQA attention pattern is wrong
- Most modern LLMs use GQA (Llama, Qwen, Mistral)

**Why:** Helpers contain the actual math. A bug in rmsNorm, RoPE, or attention affects every layer of every model.
**How to apply:** The positive fixes (float32 accumulators, mixed-type gamma, M/K shape) must be preserved. The fusedRoPECached stride assumption is a latent risk for non-contiguous inputs.
