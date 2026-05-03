---
name: cuda-qwen-gdn-state-fix-may3
description: "CUDA Qwen root cause: missing GDN/conv state feedback in autoregressive_decode.cu — fix applied, build in progress"
type: project
---

## CUDA Qwen GDN State Feedback Fix (May 3 2026, 09:15 JST)

### ROOT CAUSE
The CUDA `autoregressive_decode.cu` was completely missing the "Step 2b: GDN/conv recurrent state feedback" loop that the CPU implementation has at lines 320-420.

Without this, on CUDA:
- GDN state ext inputs (`past_gdn_state.{L}`, `past_conv_state.{L}`) stay frozen at warmup values
- Every subsequent decode step sees stale recurrent state
- GDN layers produce wrong hidden states → wrong logits → garbage tokens (204082, 76238, etc.)

### FIX APPLIED
1. Added GDN/conv state D2D copy loop after plan execution, before token sampling (using `cudaMemcpyAsync` D2D)
2. Added `markExternalInputVariable()` for all GDN/conv state ext indices (so plan's staging buffers track them correctly)

### FILES MODIFIED
- `libnd4j/include/ops/declarable/helpers/cuda/autoregressive_decode.cu`

### STATUS
- Build in progress at /tmp/cuda-gdn-fix-build.log
- Expected result: CUDA Qwen should now output "France" like CPU does

### WHY CPU WORKED
CPU `autoregressive_decode.cpp` at lines 320-420 has `dst->assign(src)` for both GDN and conv state pairs every step. CUDA was missing this entirely.
