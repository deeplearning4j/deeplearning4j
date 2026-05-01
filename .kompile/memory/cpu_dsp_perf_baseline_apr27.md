---
name: cpu-dsp-perf-baseline-apr27
description: "CPU performance baseline 3 tokens/sec, DSP decode benchmark, Triton ON enables oneDNN+OpenVINO multi-backend chain, build command, working tree modifications"
type: project
---

# CPU DSP Performance Baseline: 3 tok/s

**Recorded**: 2026-04-27
**Model**: SmolDocling-256M (30-layer decoder), CPU, batch=1 seq=1 decode
**Current best**: 3 tok/s
**Branch**: `ag_new_release_updates_2`

## Build Command (CPU + Triton)

```bash
/home/agibsonccc/dev-apps/mvn/bin/mvn -Pcpu -Dlibnd4j.triton=ON -Dlibnd4j.buildthreads=12 \
  -Dlibnd4j.log=libnd4j-build.log \
  -pl libnd4j,:nd4j-cpu-backend-common,:nd4j-native \
  clean install -DskipTests 2>&1 | tee cpu-build-output.log
```

- **ALWAYS** use `-Dlibnd4j.triton=ON` — even for CPU. Triton ON auto-enables the multi-backend chain (oneDNN + OpenVINO).
- Without Triton flag, only generic C++ execution — no accelerated backends.

## Benchmark Command

```bash
cd platform-tests && ./run-benchmark.sh --backend cpu --tokens N
```

- CPU backend uses `testDecodeStepValidation` for validation (NOT `testOutputAccuracy`, which is CUDA/Triton-specific)
- Default validation method is overridden in `run-benchmark.sh` when `--backend cpu`
- CPU adds `-Dnd4j.omp.numthreads=$(nproc)` for multi-threaded BLAS
- Skips CUDA-only flags: nsys, draft model

## Multi-Backend Chain Architecture

When Triton=ON for CPU, DSP segments go through a backend cascade:
1. **OpenVINO** — tries to fuse segment into an OpenVINO CompiledModel (Intel CPU graph optimization)
2. **oneDNN** — tries to fuse segment via oneDNN graph compiled_partition
3. If neither can fuse → slot-by-slot native execution (only for segments with no fusible ops like pure permute/reshape/identity)

Falling back to slot-by-slot when a backend SHOULD be able to compile is BANNED — fix the compilation.

## Active Working Tree Modifications (2026-04-27)

### OpenVINO Backend (`OpenVinoGraphBackend.cpp/.h`)
- **Topology-based CompiledModel cache**: new `computeIslandTopologyHash()` produces slot-index-agnostic hash from op sequence + input shapes. Segments from different transformer layers with identical topology share one CompiledModel (each gets its own InferRequest). Cache keyed by FNV-1a hash in `modelCache_` with `modelCacheMtx_`.
- **strided_slice iArgs fix**: corrected mask parsing order — nd4j iArgs are `[beginMask, ellipsisMask, endMask, ...]`, not `[beginMask, endMask, ellipsisMask, ...]`. Was previously swapping endMask and ellipsisMask.

### oneDNN batched_gemm (`mkldnn/batched_gemm.cpp`)
- **Re-enabled oneDNN batched_gemm**: was previously DISABLED ("significant primitive creation overhead"). Now hoists primitive descriptor, memory descriptors, and memory objects outside the batch loop. Reuses single `matmul_prim` for all batches (just `set_data_handle()` per batch). Only creates per-batch primitive when beta varies.
- **PLATFORM_CHECK re-enabled**: supports FLOAT32, BF16, FP16 for rank-2 inputs.

### oneDNN SDPA (`mkldnn/sdpa.cpp`)
- **GQA (Grouped Query Attention) support**: detects `numKVHeads != numHeads`, uses `cblas_sgemm_batch_strided` with `stride_b=0` to reuse same K/V head across Q head groups.
- **KV cache scatter in SDPA**: new code path handles `kvCacheK`/`kvCacheV`/`cachePosInput` directly inside the oneDNN SDPA op (writes current K/V into cache at position, then uses full cache as K/V).
- **GQA type check**: GQA only supported for FP32 via MKL batch GEMM path.

### DSP Segments (`NativeDynamicShapePlan_segments.cpp`)
- **Distinguish "no fusible ops" from "compilation failure"**: tracks `anyBackendAttemptedCompile`. If NO backend could fuse (all returned `canFuseSegment=false`), demotes to slot-by-slot native execution (legitimate — segment is pure reshape/permute). If backends TRIED and ALL failed, throws hard error.

### CUDA Stubs for CPU Build (`NativeDynamicShapePlan_cuda_stubs.cpp`)
- **NativeSlotExecutor install in fast path**: `platformTryFrozenFastPath` now installs `setNativeSlotExecutor` callback on OneDNN/OpenVINO backends before `executeSegment()`, allowing backends to call back into slot-by-slot for unmappable op ranges within a segment. Clears after execution.

### GPU Backend (`NativeDynamicShapePlan_gpubackend.cu`)
- **Merged capture: skip gap slots entirely** during CUDA graph capture instead of executing them on capture stream. Gap slots (view/identity/frozen-constant) are zero-work metadata ops — executing them during capture triggered ConstantHelper H2D copies / cudaMemcpyAsync that poisoned capture (error 901). During replay, compositeReplay() handles merged gaps natively.
- **TLS state save/restore**: saves `tl_graphCaptureStream`, `tl_captureHostWorkspace*` before merged capture.

### NativeOps DSP (both CPU + CUDA)
- **Output pointer initialization fix**: `outputPtrs` now initialized to `nullptr` instead of from `opContext->outputArray(i)`. Java side places dummy `Nd4j.empty(FLOAT)` arrays — pre-populating from those dummies caused plan outputs to be silently skipped, leaving dummy shape `[]` arrays that crashed downstream.

### Java Side
- **DynamicShapePlanExecutor**: resets `shapesFrozen=false` and `executionCount=0` on plan change. Previously, frozen state from old plan leaked into new plan.
- **GenerationPipeline**: freezes shapes after first decode step in `generateSimpleWithKvCache()` (matching `generateNative()` behavior).
- **BenchmarkConfigApplier**: resets `graphExecutionMode` to `AUTO` between benchmark configs. Previously, SLOT_BY_SLOT leaked between configs.

### Other
- **fused_rope**: now accepts `ALL_INTS` as input types (position indices are integer).
- **AttentionHelper**: removed debug `sd_printf` and unnecessary `syncToDevice()` calls.
- **Dependencies.cmake**: fixed OpenVINO `HAVE_OPENVINO` / `OPENVINO_LIB` scope propagation from recursive `setup_openvino()` call.
- **run-benchmark.sh**: adds `mergedCaptureThroughViews=true` to both benchmark and validation args. Bumped `forkedProcessTimeoutInSeconds` to 3600.
- **pom.xml**: added `ND4J_TRITON_MERGED_CAPTURE_THROUGH_VIEWS` env var passthrough.

## Op Timing Profile (CPU OPTIMAL.csv)

Top ops by total time:
- `autoregressive_decode`: 4921ms (the decode loop wrapper)
- `matmul`: 418ms / 678 calls (616us avg) — dominated by one 75ms outlier (prefill?)
- `onnx_multi_head_attention`: 332ms / 100 calls (3.3ms avg) — this is the attention bottleneck
- `reshape_no_copy`: 234ms / 546 calls (428us avg, bimodal: P50=3us vs P90=3072us)
- `gather`: 158ms / 1175 calls (134us avg)
- `sigmoid`: 72ms / 112 calls (646us avg, high variance)

## Key CPU Commits (chronological)

- `9c9b70a117` — CPU: fix specialBuffer() throw, complete VLM decoder execution on CPU
- `ac8b205196` — Fix DSP zero-fallback on CPU: empty KV arrays, OpaqueNDArray ref safety
- `1ad8358f00` — Zero-fallback CPU VLM decode: dynamic shapes, empty array handling
- `6c15912069` — CPU benchmark: enable multi-threaded BLAS (OMP_NUM_THREADS=nproc)
- `6967c3a019` — Revert non-capturable→CPU_GRAPH routing (slower)
- `004f062dfc` — Remove isDataDependent capturability limitation
- `22d6d37a45` — Trait-based CPU segmentation: split at untraited op boundaries
- `d1f3ed1922` — Revert matmul segmentation change (no perf difference on CPU)
- `91c8d2dc44` — OpenVINO: multi-threaded THROUGHPUT mode, per-op exception logging
- `d03e9acb78` — CPU backend: full VLM pipeline, DSP infrastructure, OpenVINO/OneDNN optimization
- `1300fa0e5e` — Fix CPU link: guard MmulHelper epilogue with SD_CUDA
- `9d4f2f588c` — Fix OpenMP: add compile flags to OBJECT target for CPU build
