---
name: causal-conv1d-weight-flip-fix-may2
description: "Fixed causal_conv1d weight indexing: weight[K-1-kk] for PyTorch left-padded conv semantics, affects all 18 GDN layers"
type: project
---

## causal_conv1d Weight Index Fix (2026-05-02)

**Root cause:** Both CPU and CUDA causal_conv1d used `weight[kk]` with `srcT = t - kk`, making `weight[0]` multiply the current input. But PyTorch's `F.conv1d(x, w.unsqueeze(1), padding=K-1)[:, :, :seq_len]` with left-padding means `weight[K-1]` should multiply the current timestep.

**Fix:** Changed weight index from `kk` to `K-1-kk` in both:
- `libnd4j/include/ops/declarable/helpers/cpu/causal_conv1d.cpp:80`
- `libnd4j/include/ops/declarable/helpers/cuda/causal_conv1d.cu:67`

**Impact:** All 18 GDN layers in Qwen3.5 use causal_conv1d for Q/K/V projections. Wrong weight indexing corrupts every layer's attention computation. This is likely the primary remaining CPU accuracy bug after the iArgs fix.

**Verification needed:** Run `TestCausalConv1d` first (unit test), then `TestQwen35Pipeline` with single config.

**Why:** The earlier memory entry `causal_conv1d_kernel_flip_bug.md` had a correction noting the original "fix" was backwards. The tests (`TestCausalConv1d#testSingleTimestep`, `testMultiTimestepFullWindow`) confirm `weight[K-1]` should multiply current input.

**How to apply:** Both CPU and CUDA need rebuild. This fix also affects CUDA SmolDocling if it uses GDN layers (it doesn't — SmolDocling is a VLM, not a GDN model).
