---
name: "VLM EOS root cause: mmulMxV mixed-type GEMV bug"
description: mmulMxV has no mixed-type cast — HALF weight × FLOAT32 activation dispatches to usualGemv which interprets HALF as FLOAT32 → garbage → EOS
type: project
---

## VLM EOS-on-Step-2 TRUE Root Cause: mmulMxV Mixed-Type Bug (May 3 2026)

### Problem
GraphOptimizer pre-casts large weight matrices (rank≥2, ≥1024 elements) to HALF. During native decode (M=1), matmul routes through `mmulMxV`. The GEMV path had its FP16 autocast removed but had NO mixed-type handling. When weight=HALF and activation=FLOAT32:
- `AX = false` (types differ)
- `typeHalfFloat = false` (requires AX=true)
- Falls into `usualGemv` dispatched on `xType=FLOAT32`
- `usualGemv` reads A's HALF buffer interpreting bytes as FLOAT32 → garbage logits
- argmax on garbage → position 0 → token 0 (EOS for SmolDocling)

### Why GEMM Path Was Fine
The `mmulMxM` path (used for M>1 prefill) has explicit mixed-type casting at lines 879-922 with a cast cache for graph capture. It correctly casts FLOAT32→HALF before cuBLAS.

### Fix Applied
Added mixed-type casting to mmulMxV (MmulHelper.cu lines 1156-1181):
- A=HALF, X=FLOAT32 → cast X to HALF → typeHalfFloat → cublasGemmEx HALF×HALF→F32
- A=FLOAT32, X=HALF → cast X to FLOAT32 → typeFloat → cublasSgemv (pure FP32)
- Also handles BF16×FLOAT32 cases

Only casts X (the small activation vector), never the large weight matrix A.

### Status
Fix in MmulHelper.cu, CUDA build in progress. Not yet tested.
