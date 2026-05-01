---
name: reshape-view-bypass-regression
description: "REVERTED: reshape_no_copy ARRAY_COPY_OFFSET_INPUT_0 view bypass caused -29% regression (50→35 tok/s) — non-contiguous views kill cuBLAS GEMV perf"
type: project
---

## reshape_no_copy View Bypass via ARRAY_COPY_OFFSET_INPUT_0 — REVERTED (04-27)

**Trial**: Skip tryCreateViewForSlot C-contiguity gate when output shape has ARRAY_COPY_OFFSET_INPUT_0 (reshapeNoAlloc already verified valid view strides).

**Result**: 35.39 tok/s (was ~50) — **-29% regression**. Correctness PASS.

**Op timing delta**:
- reshape_no_copy: 103ms → 92ms (-11ms, P99: 3072→1536µs) ✓ improved
- matmul: 193ms → 258ms (+65ms) ✗ massive regression
- Net: -54ms worse

**Root cause**: The assign() copy in reshape_no_copy's slow path is NOT waste — it produces C-contiguous dense outputs. Downstream cuBLAS GEMV (M=1 decode) is highly sensitive to input memory layout. Non-contiguous views from permute chains cause cuBLAS to use strided access patterns, ~34% slower per call.

**Key insight**: reshape_no_copy's "expensive" copies are load-bearing for cuBLAS perf. The 103ms spent copying is amortized by faster 193ms matmul. Without copies, matmul costs 258ms → net worse.

**Why:** The contiguity gate in tryCreateViewForSlot exists for a reason — downstream ops (especially cuBLAS) need dense contiguous inputs.
**How to apply:** NEVER bypass reshape_no_copy's copy path to create non-contiguous views. If optimizing reshape, ensure outputs remain C-contiguous.
