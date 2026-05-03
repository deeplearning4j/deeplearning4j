---
name: cuda-softmax-inplace-corruption
description: "Root cause of CUDA VLM all-zero tokens: fusedCausalMaskSoftmaxKernel in-place corruption"
type: project
---

## fusedCausalMaskSoftmaxKernel in-place write corruption

**File:** `libnd4j/include/helpers/cuda/FlashAttentionHelper.cu` lines 113-228

**Bug:** The kernel's Pass 2 writes `exp(val - max)` to `output[j]` while reading from `input[j]` in the same loop. ALL three call sites pass `input == output` (in-place). When `logitsOut == nullptr` (2 of 3 call sites), Pass 2 clobbers original logits as it writes exp values, causing later loop iterations to read already-exponentiated values instead of raw logits. Result: corrupted softmax probabilities → wrong attention output → wrong tokens.

**Call sites (all in FlashAttentionHelper.cpp):**
- Line 222: noGQA fused path (logitsBuffer may be non-null — only safe when it IS)
- Line 524: GQA fallback path (logitsBuffer usually null — BROKEN)
- Line 721: forward4DDecode (always nullptr — BROKEN)

**Fix (2026-05-02):** Restructured to 3-pass softmax:
- Pass 1: find max (unchanged)
- Pass 2: compute sum of exp(x-max) WITHOUT writing to output
- Pass 3: read input, compute exp(x-max)/sum, write to output

Safe because: each thread handles non-overlapping j values (stride=blockDim.x), so reads input[j] before writing output[j] at same index within a single iteration.

**Why:** Commit 020d93aa26 introduced the fusedCausalMaskSoftmaxCuda call but the kernel was never designed for in-place operation. The old code used `ops::helpers::softmax()` which handles in-place correctly.

**How to apply:** Any future kernel that accepts separate input/output pointers must handle the in-place case (input==output). Test with both separate and aliased buffers.
