---
name: VLM CUDA graph node count analysis
description: "Root cause of 60 tok/s: 2742 CUDA graph nodes cause ~5.5-11ms driver scheduling overhead per replay"
type: project
---

## VLM Performance Root Cause (May 4 2026)

**Current**: ~60 tok/s (16.6ms/token)
**Target**: 100+ tok/s (10ms/token)
**Previous with inaccurate results**: 100+ tok/s (user confirmed this was achievable)

### Root Cause: CUDA Graph Node Count Overhead

The monolithic CUDA graph has **2742 captured kernel nodes**. NVIDIA driver scheduling overhead at ~2-4µs/node = **5.5-11ms per cudaGraphLaunch()**, when actual GPU compute is <2ms.

### Node Breakdown (from OPTIMAL.csv op-timing)

**Real compute ops (~332 actual compute, but each may produce multiple graph nodes):**
- onnx_multi_head_attention: 90 calls (each is multi-kernel internally, ~5-10 nodes each = 450-900 graph nodes)
- matmul: 604 calls (each = 1 cuBLAS GEMM, 1-2 nodes = ~604-1208 nodes)
- concat: 552 calls (each launches a kernel = ~552 nodes)
- cast: 526 calls (type conversion kernel = ~526 nodes)

**Zero-cost view ops (DO NOT produce graph nodes):**
- expand_dims: 569 calls, 0.390µs avg — pure metadata, no kernel
- permute: 360 calls, 0.256µs avg — pure metadata, no kernel
- reshape: 1725 calls, 4.709µs avg — mostly zero-copy
- shape_of: 371 calls — metadata only

**Mixed ops (some produce nodes, some don't):**
- reshape_no_copy: 604 calls, 228µs avg — DOES launch memcpy when buffers differ (line 43 in reshape_no_copy.cpp calls output->assign(input))

### Key Code Paths

1. **Capture path**: `NativeDynamicShapePlan_gpubackend.cu:3112` — NATIVE_ONLY_CAPTURE loops through ALL slots via executeSlot()
2. **Replay path**: `NativeDynamicShapePlan_cuda.cu:376` — single cudaGraphLaunch via seg.exec.replayHandle->replay(stream)
3. **View handling**: `NativeDynamicShapePlan_slotexec.cpp:703` — tryCreateViewForSlot() creates NDArray view without kernels
4. **Frozen constant skip**: `NativeDynamicShapePlan_slotexec.cpp:2119` — skips when shapesFrozen_ && executeCount_ >= 2

### Optimization Paths (prioritized)

1. **Reduce concat ops** — 552 nodes from KV cache concatenation. If planOwnsKvScatter can be enabled for VLM/ONNX, these get eliminated. Currently disabled because `cachePositionExtIdx = -1` for VLM models (GenerationPipeline.java:1954).

2. **Reduce reshape_no_copy copies** — 604 calls where many trigger assign() due to non-contiguous inputs. Making upstream outputs contiguous would make these zero-copy views.

3. **Composite replay instead of monolithic** — split the graph into smaller islands. The infrastructure exists (compositeReplay in gpubackend.cu) but the monolithic path is what's being used in OPTIMAL config.

4. **Skip frozen constants during capture** — frozen constant slots ARE skipped after executeCount >= 2, but capture happens earlier. Verify timing.

### Architecture Notes

- Monolithic vs composite decision: `NativeDynamicShapePlan_cuda.cu:202-210`
- fastPathApplicable requires replayHandle != null && isReady()
- compositeFastPathApplicable uses hasCompositeHandles()
- KV scatter is POST-graph (autoregressive_decode.cu:817), not inside the graph
- The 552 concat ops are INSIDE the graph (likely QKV projection concatenation within attention)

**Why:** Understanding this is critical for the 60→100 tok/s optimization. The overhead is all in NVIDIA driver scheduling of graph nodes, not GPU compute.
**How to apply:** Focus on reducing graph node count (eliminate concats, make reshape_no_copy zero-copy) rather than optimizing individual kernel performance.
