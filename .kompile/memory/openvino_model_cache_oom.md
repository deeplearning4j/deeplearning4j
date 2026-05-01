---
name: openvino-model-cache-oom
description: OpenVINO modelCache_ had no eviction — CompiledModels (~50-200MB each) accumulated across plan redispatches causing CPU OOM
type: project
---

# OpenVINO Model Cache OOM Bug (fixed 2026-04-27)

**Symptom**: `physicalBytes (36556M) > maxPhysicalBytes (34816M)` during CPU LLM benchmark with AUTO mode.

**Root cause**: `OpenVinoGraphBackend::modelCache_` (topology-based CompiledModel cache) had NO eviction. Each `shared_ptr<ov::CompiledModel>` holds ~50-200MB of internal OpenVINO working buffers. During LLM generation, plan redispatch (prefill→decode shape change) creates new CompiledModels, but old ones were never freed because `modelCache_` kept a reference alive even after segment cache entries were evicted.

**Fix**:
1. `evictCacheIfOverLimitLocked()` now sweeps `modelCache_` after evicting segment entries — removes CompiledModels where `use_count()==1` (only held by topology cache, no active segment references).
2. Lowered `kMaxCacheEntries` from 128 to 32 — each InferRequest holds significant working buffers.

**Files**: `OpenVinoGraphBackend.cpp` (eviction), `OpenVinoGraphBackend.h` (kMaxCacheEntries)

**Why:** OpenVINO CompiledModel objects are heavyweight — they contain compiled IR, thread pools, and pre-allocated tensor buffers. The topology cache was designed to share compiled models across transformer layers (good), but lacked lifecycle management for cross-plan-shape accumulation.

**How to apply:** Any new caching of heavyweight native objects (CompiledModel, compiled_partition, etc.) MUST have eviction tied to the plan cache lifecycle. Check `use_count()` after segment eviction.
