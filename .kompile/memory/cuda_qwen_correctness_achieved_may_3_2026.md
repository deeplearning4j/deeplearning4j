---
name: CUDA Qwen correctness achieved May 3 2026
description: CUDA Qwen outputs France (token 271) — CUTLASS stride fix + FP16 autocast removal confirmed working
type: project
---

## CUDA Qwen Correctness — ACHIEVED (May 3 2026)

TestQwenLayerDiagnostics on CUDA produces token 271 ("Paris") for "What is the capital of France?"

### Root Causes Fixed
1. **CUTLASS stride mismatch** (PRIMARY): `CutlassGemmHelper::gemm()` dispatched RowMajor kernel to column-major permuted weight views. Fix: stride contiguity check `strideAt(1) != 1` → fall through to cuBLAS.
2. **FP16 autocast** (SECONDARY): `MmulHelper::mmulMxM` auto-cast FP32×FP32 to HALF. Removed.
3. **fusedGQADecodeKernel thread oversubscription**: LaunchDims.cu capped at 256 threads (was escalating to 1024 causing error 701).
4. **KV cache ext inputs unmarked variable**: autoregressive_decode.cu now marks all KV ext inputs so frozen fast-path refreshes them.

### Files Modified
- `libnd4j/include/helpers/cuda/CutlassGemmHelper.cu` — stride check
- `libnd4j/include/helpers/cuda/MmulHelper.cu` — FP16 autocast removal
- `libnd4j/include/execution/cuda/LaunchDims.cu` — GQA decode thread cap
- `libnd4j/include/ops/declarable/helpers/cuda/autoregressive_decode.cu` — KV marking
- `libnd4j/include/ops/declarable/helpers/cpu/autoregressive_decode.cpp` — KV marking (CPU)

### Remaining
- VLM benchmark not yet producing "mythic heroes" text (separate issue)
- VLM benchmark at 55 tok/s, target 100+
