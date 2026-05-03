---
name: cpu-qwen-france-success
description: CPU Qwen3.5 outputs 'The capital of France is Paris' — causal_conv1d weight flip fix confirmed
type: project
---

**Date:** 2026-05-02

**Fix:** causal_conv1d weight indexing changed from `kk` to `K-1-kk` in both CPU and CUDA kernels.

**Files changed:**
- `libnd4j/include/ops/declarable/helpers/cpu/causal_conv1d.cpp:83`
- `libnd4j/include/ops/declarable/helpers/cuda/causal_conv1d.cu:68`

**Root cause:** PyTorch `F.conv1d` with left-padding (`padding=K-1`) and truncation `[:,:,:L]` means `weight[K-1]` multiplies the current timestep, not `weight[0]`. The kernel was applying weights in reversed order — `weight[0]` hit current input instead of `weight[K-1]`.

**Result:** CPU Qwen3.5-0.8B GGUF now produces:
```
The capital of France is **Paris**.
Paris has been the capital of France since 1792...
```
nativeCount=48 tokens, all coherent, correct factual content.

**Why:** This affects all 18 causal_conv1d layers in Qwen's GDN (Gated Delta Network) blocks. Wrong weight ordering caused garbled hidden states that cascaded through all layers.

**How to apply:** This fix is critical for any model using causal_conv1d (Mamba, GDN, etc). The weight convention must match PyTorch's F.conv1d left-padding semantics.
