---
name: VLM decode regression root cause May 3
description: onnx_multi_head_attention.cpp workspace buffer removal + syncToDevice removal caused VLM EOS on step 1
type: project
---

# VLM Decode Regression Root Cause (May 3 2026)

## Root Cause: onnx_multi_head_attention.cpp changes between 11005b4ae6 and HEAD

Three changes introduced between working commit (11005b4ae6) and HEAD broke SmolDocling VLM decode:

### 1. Workspace buffer removal → stale data leakage (PRIMARY)
- Working: `AttentionWorkspace::getInstance()->getBuffer()` + `nullify()` → FlashAttention writes into zeroed buffer → `output->assign()`
- Broken: `output->reshape('c', outShape4d, false)` direct write — no zeroing, stale data leaks through masked attention positions
- **Fix**: Restored workspace buffer + nullify pattern

### 2. syncToDevice() removal → host-actual KV data not flushed (SECONDARY)
- Working: `kPastSlice->syncToDevice()` etc after every assign to KV slices
- Broken: All syncToDevice() calls removed
- Investigation proved assign() and FlashAttention use SAME stream (no race), BUT syncToDevice() serves a different purpose: it flushes host-actual data from permuted ext input views
- **Fix**: Restored syncToDevice() calls after all slice assigns

### 3. Validation disabled after step 3 (MASKING)
- `if (step < 3)` guards silently disabled all REQUIRE_TRUE checks in autoregressive_decode.cu
- Bugs on step 4+ were invisible
- **Fix**: All validation runs every step (O(1) cost, negligible)

## What SmolDocling Uses
- Standard KV concat mode (NO cache_position, NO external causal mask)
- Internal attn_mask_reformat subgraph for attention bias
- padKvToStaticSize for static KV buffers from prefill

## Status: Fixes coded, build in progress, NOT yet tested
**Why:** Previous build (20:15 May 3) does NOT contain these fixes.

**How to apply:** Build and test with `run-benchmark.sh --tokens 250`. Success = text about "mythic heroes".
