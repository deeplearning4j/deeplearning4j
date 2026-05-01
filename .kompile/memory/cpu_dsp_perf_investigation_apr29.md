---
name: cpu-dsp-perf-investigation-apr29
description: "CPU DSP deep investigation: OpenVINO IS working, segment merge bug found (BenchmarkConfig default), compute-bound at 0.12 tok/s on Ryzen 5950X AVX2"
type: project
---

# CPU DSP Performance Investigation (2026-04-29)

**Model**: Qwen3.5-0.8B (Q4_K_M GGUF, 24-layer Mamba2 hybrid), FP32
**CPU**: Ryzen 5950X (Zen 3, AVX2 only — NO AVX512, NO FP16 hardware)
**Branch**: `ag_new_release_updates_2`
**Baseline**: 0.07-0.09 tok/s → improved to 0.12-0.15 tok/s

## Key Finding: OpenVINO IS Activating Successfully

DSP diagnostics (`-Dnd4j.dsp.diagnostics=ALL -Dnd4j.dsp.diagnostics.level=full`) confirmed:
- **Backend chain**: 2 backends available (OpenVINO priority 1, OneDNN priority 2)
- **OpenVINO compiles ALL segments**: `canFuseSegment: mappable=1761/1761 ovCompilable=1725 canFuse=true`
- **Validation passes**: `OpenVINO VALIDATION OK: all 1761 ops covered (compiled=1725 nativeHandled=36)`
- **36 native-deferred ops**: gated_delta_rule (18), fused_rope (12), dot_product_attention_v2 (6) — executed via NativeSlotExecutor callback
- **Merged segment**: After freeze, 232 segments → 1 segment [0-1760] with 33 OV islands + 32 native ranges
- **Frozen fast path active**: `CPU_FROZEN_FAST_PATH: segments=1 executeCount=2` on step 2+

## Bug Found: BenchmarkConfig.create("OPTIMAL") Disabled Segment Merging

### Root Cause
`TestLLMBenchmarkSuite.testOptimalBaseline()` used `BenchmarkConfig.create("OPTIMAL")` instead of `BenchmarkConfig.optimal()`. The `create()` factory returns a blank config with Java primitive defaults. The field `boolean dspFreezeMergeSegments` defaulted to `false` (Java boolean default). `BenchmarkConfigApplier` then explicitly called `env.setDspFreezeMergeSegments(false)`, overriding the C++ default of `true`.

**Effect**: `resegmentForFreeze()` returned immediately at line 4015 (`!dspFreezeMergeSegments → return`), leaving 232 pre-freeze segments intact instead of merging to 1.

### Fix Applied
1. `BenchmarkConfig.java`: Changed field default `boolean dspFreezeMergeSegments = true;`
2. `TestLLMBenchmarkSuite.java`: Changed 3 instances of `BenchmarkConfig.create("OPTIMAL")` → `BenchmarkConfig.optimal()`

### Verified
With fix: `RESEGMENT: 232 -> 1 segments (shapes frozen, merge enabled)` confirmed in diagnostics.

## Architecture: CPU DSP Execution Lifecycle

### Plan lifecycle (per decode step)
1. **Step 0 (unfrozen)**: Slot-by-slot warmup, AUTO_SEAL freezes shapes
2. **resegmentForFreeze()**: Rebuilds segments with frozen MAX_SEGMENT_SIZE=100,000 → 1 segment
3. **Step 1 (frozen)**: phaseReplay → `executeSegmentWithCpuGraph()`:
   - Warmup (executionCount==0): runs slot-by-slot to populate output shapes
   - Compile (executionCount==1): OpenVINO `canFuseSegment` → `compileSegment` → `executeSegment`
   - Sets `resolvedCpuBackend = OpenVINO`
4. **Step 2+ (frozen fast path)**: `platformTryFrozenFastPath` → `executeSegmentWithSpecificBackend` with cached backend

### Key code paths
- `getCpuGraphBackendChain()` at `NativeDynamicShapePlan_segments.cpp:281` — built once, cached
- `resolveBackendForSegment()` at `NativeDynamicShapePlan.cpp:3960` — GEM_AUTO → `platformResolveBackend(false)` → CPU_GRAPH
- `platformResolveBackend()` at `NativeDynamicShapePlan_cuda_stubs.cpp:463` — returns CPU_GRAPH when HAVE_ONEDNN or HAVE_OPENVINO
- `executeSegmentWithCpuGraph()` at `NativeDynamicShapePlan_segments.cpp:416` — warmup→compile→execute lifecycle
- `platformTryFrozenFastPath()` at `NativeDynamicShapePlan_cuda_stubs.cpp:68` — frozen fast path entry

### GEM_SLOT_BY_SLOT kill switch
`getCpuGraphBackendChain()` returns EMPTY chain immediately if `graphExecutionMode_ == GEM_SLOT_BY_SLOT` (line 292).
`tritonSkipKernels()` forces `GEM_SLOT_BY_SLOT` at line 1463 — but defaults to `false`, only triggered by `ND4J_TRITON_SKIP_KERNELS=1`.

## Compile-Time Guards

Both backends are compile-time gated:
- `HAVE_ONEDNN`: set by `-DHELPERS_onednn=ON` in CMake (default OFF, enabled by build config)
- `HAVE_OPENVINO`: set by `-DSD_TRITON=ON` (via `-Dlibnd4j.triton=ON`)
- CPU build config.h confirms: `HAVE_ONEDNN 1`, `HAVE_OPENVINO 1`
- **ALWAYS use `-Dlibnd4j.triton=ON` for CPU builds** — it enables OpenVINO, not just Triton

## Performance: Compute-Bound Bottleneck

The 0.12-0.15 tok/s IS the OpenVINO-accelerated speed. The bottleneck is raw FP32 compute on AVX2:

### Qwen 0.8B op histogram (1761 total ops)
- `matmul`: 187 ops — [1,1,1024]×[1024,1024] each = ~2M FLOPs → 374M FLOPs/token total
- `gated_delta_rule`: 18 ops — Mamba2 SSM (complex, native-deferred)
- `fused_rope`: 12 ops (native-deferred)
- `dot_product_attention_v2`: 6 ops (native-deferred)
- `cast`: 108, `add_scalar`/`divide`/`sqrt`: 115 each, `multiply`: 260, `add`: 66, `reduce_mean`: 79

### Why it's slow
- **AVX2 peak**: ~16 FP32 ops/cycle × 4.9GHz = ~78 GFLOPS (single core)
- **Actual**: ~0.033 GFLOPS (1000x below peak) — massive overhead from:
  - 36 native-deferred ops break OpenVINO graph into 33 islands + 32 native ranges
  - Each native range calls back through NativeSlotExecutor → slot-by-slot dispatch
  - Memory-bound ops (reshape, permute, gather) dominate over compute
  - OpenVINO CPU plugin overhead for small tensors

### What would help
- AVX512/AMX CPU for hardware FP16/BF16
- Reducing native-deferred ops (making gated_delta_rule OpenVINO-mappable)
- INT8 quantization via OpenVINO's native quantization pipeline
- Multi-threaded execution across cores (OMP)

## Uncommitted Changes (as of investigation)

### C++ files (from prior session, not yet committed)
- `NativeDynamicShapePlan_segments.cpp`: validateSlotRange gated on `executeCount_ < 4`
- `DeclarableOp.cpp`: OpTimingRecord deferred construction, helper dispatch skip when `shapeFunctionOverride()`, UB fix for `timingRecord.usedHelper`

### Java files (from this session)
- `BenchmarkConfig.java`: `dspFreezeMergeSegments` default → `true`
- `TestLLMBenchmarkSuite.java`: `BenchmarkConfig.create("OPTIMAL")` → `BenchmarkConfig.optimal()`

## Diagnostic Commands

```bash
# Run with full DSP diagnostics
cd platform-tests && /home/agibsonccc/dev-apps/mvn/bin/mvn test \
  -Dtest=TestLLMBenchmarkSuite#testOptimalBaseline \
  -Dbench.max.tokens=5 -Dbench.models=qwen -Dbackend.artifactId=nd4j-native \
  -Dnd4j.dsp.diagnostics=ALL -Dnd4j.dsp.diagnostics.level=full \
  2>&1 | tee /tmp/dsp-cpu-diag.log

# Key grep patterns for diagnostic output
grep 'chain built' log            # Backend chain construction
grep 'canFuseSegment' log         # Per-segment fusion decision
grep 'RESEGMENT' log              # Segment merging on freeze
grep 'CPU_FROZEN_FAST_PATH' log   # Frozen fast path activation
grep 'PRE-EXECUTE.*backend=' log  # Per-segment backend execution
grep 'VALIDATION' log             # Compile validation results
```
