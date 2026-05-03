---
name: cpu-sdpa-causal-mask-root-cause
description: "ROOT CAUSE: MKL SDPA platform impl never reads causal mask at input[8] — all prefill attention is non-causal"
type: project
---

## CPU Qwen3.5 Root Cause: MKL SDPA Causal Mask Not Read (May 2 2026)

### The Bug
`sdpa.cpp` PLATFORM_IMPL(dot_product_attention_v2, ENGINE_CPU) has broken bias detection at lines 1111-1129.

When KV cache is active (inputs 5=keyCache, 6=valueCache), the code checks:
```cpp
if (extraInput != nullptr && !extraInput->isEmpty() &&
    (extraInput2 == nullptr || extraInput2->isEmpty()))
```
This is ALWAYS FALSE because input[6] (valueCache) is not empty. So attentionBias stays nullptr.

The causal mask is at input[8] but the MKL impl never reads it. The generic op (dot_product_attention_v2.cpp) correctly reads input[8] at lines 240-281, but PLATFORM_CHECK returns true for FP32 rank-4, so the MKL impl runs instead.

### Impact
ALL prefill attention is non-causal — every token attends to every other token including future tokens. This corrupts all hidden states from layer 1 onward. Output: token 314 (' of') for "capital of France?" even with DSP and optimizer disabled.

### Fix Applied
Added three-way bias detection:
1. useInPlaceKv → read input[8] (KV cache active, bias at input[8])
2. !hasKvCache && width > 8 → read input[8] (prefill with empty cache placeholders)
3. Legacy path → read input[5] as bias (no KV cache at all)
Also added slicedBiasOwner for bias wider than K's seq dim, with cleanup at all return points.

**Why:** The MKL platform override was added for performance but didn't replicate the generic op's input[8] bias handling.
**How to apply:** File: sdpa.cpp, around line 1111. Status: APPLIED, needs CPU rebuild to verify.
