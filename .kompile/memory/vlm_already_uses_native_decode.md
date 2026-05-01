---
name: vlm-already-uses-native-decode
description: VLM benchmark already uses native autoregressive_decode C++ op, not Java decode loop — bottleneck is inside C++ native loop
type: project
---

# VLM Benchmark Already Uses Native Decode Path

**Discovered**: 2026-04-29

The SmolDocling VLM benchmark at 50 tok/s is ALREADY running through:
- `GenerationPipeline.generate(prefillEmbeddings, promptTokenIds, maxTokens)` (line 541 of TestSmolDoclingOptimizedPipeline.java)
- → `generateNative(...)` (line 1272 of GenerationPipeline.java)
- → C++ `autoregressive_decode` op with GPU-side mask/position/embed update kernels

**Why:** The earlier analysis assumed the Java `StaticKvCacheDecodeLoop` was the bottleneck (per-step putScalar + H2D sync). That analysis was wrong — the benchmark doesn't use `StaticKvCacheDecodeLoop`. It uses the native C++ loop which already has GPU kernels for all updates and only one `cudaStreamSynchronize` per step for the token D2H readback.

**Implication:** The 50→100 tok/s gap is inside the C++ native decode loop, likely:
1. The `executeSteadyState()` call — composite CUDA graph replay with 22 islands
2. The single per-step `cudaStreamSynchronize` for token readback
3. KV scatter batch kernel dispatch overhead
4. Any remaining sync/host-side overhead in the C++ plan executor

**What does NOT help:**
- Replacing Java putScalar with GPU ops (already native)
- GPU-side stop-token check (already done in C++)
- GPU-side embedding lookup (already done via embedLookupKernel in C++)
- Migrating to native op (already migrated)

**Where to look next:** Profile `executeSteadyState()` + the per-step C++ overhead in `autoregressiveDecodeCuda()` in `libnd4j/include/ops/declarable/helpers/cuda/autoregressive_decode.cu`.
