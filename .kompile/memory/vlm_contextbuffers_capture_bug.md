---
type: project
title: VLM VISION_ENCODE crash — ContextBuffers lazy init during CUDA graph capture
created: 2026-05-05
status: fix-applied-building
---

# VLM ContextBuffers Capture Bug (May 5 2026)

## Root Cause

`ContextBuffers::initialize()` uses `cudaMallocAsync(&ptr, size, 0)` on stream 0 (legacy stream). When called during CUDA graph capture (Triton compilation phase), this creates an implicit dependency between the legacy stream and the capturing stream → error 906 (`cudaErrorStreamCaptureImplicit`) → capture invalidated → all subsequent ops fail with error 901 (`cudaErrorStreamCaptureInvalidated`).

**Symptom:** VLM benchmark crashes at slot 274 (zeroslike) during VISION_ENCODE with `cudaMemsetAsync failed; Error code: [901]`.

**Evidence:**
- `WARNING: ContextBuffers: _reductionPointer cudaMallocAsync failed on device 0 (error 906)`
- `WARNING: getCudaStream() returning null stream - context may not be initialized`
- Then: `slot 274 (zeroslike) frozen-ctx exec exception: cudaMemsetAsync failed; Error code: [901]`

## Fix Applied

In `ContextBuffers::initialize()`:
1. Check `tl_graphExecutionActive && tl_graphCaptureStream != nullptr`
2. If true, use capture stream for `cudaMallocAsync` instead of stream 0
3. Skip `cudaSetDevice`, `trimPool`, and `cudaStreamSynchronize` during capture
4. All error-path `cudaFreeAsync` calls also use `allocStream` instead of 0

**File:** `libnd4j/include/execution/cuda/ContextBuffers.cu`

## Why ContextBuffers::initialize() Is Called During Capture

ContextBuffers is `thread_local` per CUDA device. During warmup, the main thread initializes ContextBuffers for device 0. But after warmup completes and before Triton capture begins, `ContextBuffers::release()` may be called (sets `_initialized = false`). When capture starts and an op accesses `reductionBuffer()` or `execStream()`, the lazy initialization fires on stream 0 inside capture.

## Status

Build in progress. Will test VLM benchmark after build completes.
