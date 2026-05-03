---
name: causal-conv1d-kernel-flip-bug
description: "CRITICAL: causal_conv1d weight indexing flipped vs PyTorch cross-correlation convention — corrupts all 18 GDN layers"
type: project
---

## causal_conv1d Kernel Flip Bug (May 2 2026)

**Files:** `libnd4j/include/ops/declarable/helpers/cpu/causal_conv1d.cpp:80` AND `libnd4j/include/ops/declarable/helpers/cuda/causal_conv1d.cu:67`

**Bug:** Weight index used `K-1-kk` (convolution with flip) instead of `kk` (cross-correlation, matching PyTorch F.conv1d).

- `kk=0` is current timestep, `kk=K-1` is oldest
- Old code: `weight[d, K-1-kk]` → weight[K-1] multiplied current input (wrong)
- Fixed: `weight[d, kk]` → weight[0] multiplies current input (correct PyTorch convention)

**Impact:** ALL 18 GDN layers in Qwen3.5 (0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22) received corrupted Q/K/V projections from flipped causal convolution. This is the root cause of garbage output from the model.

**Why:** PyTorch F.conv1d performs cross-correlation (no kernel flip). GGUF weights are stored in this convention. Our code was performing true convolution (with flip), reversing the causal filter entirely.

**How to apply:** Both CPU and CUDA implementations fixed. Any future causal_conv1d code must use cross-correlation convention (no flip) to match PyTorch/GGUF weights.


## 2026-05-02 13:10


## Correction - May 2 latest review

The earlier interpretation above is likely backwards.

Local reference checked:
- `/home/agibsonccc/miniconda3/envs/qwen310/lib/python3.10/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py`
- Qwen3.5 fallback path uses `F.conv1d(mixed_qkv, weight.unsqueeze(1), padding=K-1)[:, :, :seq_len]`.
- PyTorch `conv1d` is cross-correlation, but because the input is left-padded and the first `seq_len` outputs are taken, the current timestep is multiplied by `weight[K-1]`, not `weight[0]`.

Existing project tests agree with that:
- `TestCausalConv1d#testSingleTimestep` says for `L=1`, only `weight[:, K-1]` should contribute.
- `TestCausalConv1d#testMultiTimestepFullWindow` expects `weight[1] * x[t] + weight[0] * x[t-1]` for `K=2`.

Current helper code does the opposite:
- `causal_conv1d.cpp`: `srcT = t - kk`, `weight[d, kk]`, so `weight[0]` multiplies current input.
- `causal_conv1d.cu` has the same current-weight mapping.

Actionable next step: repair CPU and CUDA causal_conv1d indexing to match the reference/test convention: when iterating lag `kk` with `srcT = t - kk`, use kernel index `K - 1 - kk`. Then run `TestCausalConv1d` from `platform-tests` with tee before retesting Qwen. This is not a workaround; it restores the PyTorch padded-conv semantics used by Qwen3.5 GDN.

This also demotes the earlier “Q scaling after L2 norm” suspicion: the local HuggingFace reference explicitly scales query by `1 / sqrt(query.shape[-1])` after optional L2 normalization in both chunk and recurrent gated-delta-rule paths. Do not remove Q scaling unless a direct parity test proves otherwise.
