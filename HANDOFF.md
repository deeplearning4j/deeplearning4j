# Handoff: VLM Accuracy Regression Fix

## Goal
Running `platform-tests/run-benchmark.sh --tokens 250` against pathfinder-mythic.pdf page 10 should produce text about "mythic heroes" / "CREATING A MYTHIC CHARACTER". Instead it produces `<doctag><picture><loc_1><loc_0><loc_500><loc_500><other></picture> </doctag><end_of_utterance>` (25 tokens, EOS immediate). Location tags are WRONG — they should be text content.

The regression was introduced in commit `38081a955a` (fix: use-after-free in writeOutputSlot causing slot 51 shapeInfo nullptr crash). The prior commit `3075ec44ac` produced correct output.

## What Was Done (This Session)

### ShapeKeyState Refactoring — COMPLETE, BUILDS, NOT THE BUG

Replaced raw `GraphSegmentDef::shapeKey` (LongType) with structured `ShapeKeyState` struct across **18 files**. This was done to make the shape key lifecycle explicit and traceable. The refactoring is clean and correct — **zero spurious recompiles** in benchmark output.

**Files modified (all in `libnd4j/include/graph/`):**
- `NativeDynamicShapePlan.h` — added `ShapeKeyState` struct, `SegmentDispatchEvent` enum, `DSP_SEG_EVENT` macro
- `impl/NativeDynamicShapePlan_gpubackend.cpp` — main dispatch logic migrated
- `impl/NativeDynamicShapePlan_segments.cpp` — CPU segment dispatch migrated  
- `impl/NativeDynamicShapePlan_cuda.cu` — CUDA teardown migrated
- `impl/NativeDynamicShapePlan_cuda_stubs.cpp` — CPU stubs migrated
- `gpu/TritonGraphBackend_kernel.cu` — cache key lookups migrated
- `gpu/TritonGraphBackend_execute.cu` — cache key lookups migrated
- `gpu/NvrtcGraphBackend.cu`, `gpu/PtxGraphBackend.cu` — JIT cache keys migrated
- `cpu/AclGraphBackend.cpp`, `cpu/ArmHybridGraphBackend.cpp`, `cpu/MlirCpuGraphBackend.cpp`, `cpu/MlxGraphBackend.cpp`, `cpu/NnapiGraphBackend.cpp`, `cpu/OneDnnGraphBackend.cpp`, `cpu/OpenVinoGraphBackend.cpp` — CPU backends migrated
- `legacy/cpu/NativeOps_dsp.cpp`, `legacy/cuda/NativeOps_dsp.cu` — JSON diagnostics updated

**Benchmark results with refactoring:**
```
[DSP_EVENT] seg[0-1937] WARMUP_START → COMPILE_START → COMPILE_DONE → SHAPE_KEY_STORED (STABLE, 8 execs)
[DSP_EVENT] seg[0-2742] WARMUP_START → COMPILE_START → COMPILE_DONE → SHAPE_KEY_STORED (STABLE, 20 execs)
```
No DRIFTED, no RECOMPILE_TRIGGERED, no INVALIDATE. Shape keys are working correctly.

### What DIDN'T Fix It

The shape key was **not** the root cause. The actual output is wrong in a different way than expected:
- Only 25 tokens generated (EOS hit immediately)
- Output is `<doctag><picture>...` — generic document structure, not page 10 content
- This suggests the **vision encoder output** is wrong or the **prompt construction** is broken
- The model immediately decides "this is a picture" and emits the structure, meaning it's not "seeing" the actual page content from the vision encoder

## Where to Look Next

The regression is in commit `38081a955a`. Key changes in that commit:

1. **`DynamicShapePlanExecutor.java`** — `cachedConstantValues` (HashMap of small constants ≤32 elements, dup'd) → `protectedConstantBuffers` (IdentityHashMap of ALL constant DataBuffers, strong refs)
   - Old code: restored dead constants from cached copies
   - New code: throws RuntimeException if any protected constant is stale
   - **Hypothesis**: The vision encoder SameDiff has its own plan. When the decoder plan's `protectedConstantBuffers` holds references to vision encoder constants, it may prevent proper cleanup/re-resolution between encoder and decoder execution.

2. **`closeSlotArrayCache`** — added `outputProtectedBuffers` set and a check `if (protectedConstantBuffers == null || !protectedConstantBuffers.containsKey(buf))` before un-poisoning buffers
   - **Hypothesis**: Real intermediates marked constant by `directExecHelper()` are now being treated as protected constants and never freed, causing stale data to persist.

3. **Vision encoder tiling**: `encodeImageTiled()` calls `visionEncoder.output()` multiple times on the same SameDiff. The first execution compiles and captures the plan. Subsequent tile executions should use the same plan. If `protectedConstantBuffers` prevents buffer turnover between tile executions, stale data from tile 1 may leak into tile 2's output.

4. **Output: the model is generating generic document structure** (doctag/picture/location), not text content. This means the vision embedding reaching the decoder is garbage or zeros — the model defaults to "there's a picture here" when it can't interpret the visual features.

## Key Diagnostic Already Captured

```
seg[0-1937] = vision encoder (8 tile executions, correct lifecycle)
seg[0-2742] = decoder (20 executions, correct lifecycle)
```

Both segments are STABLE with correct shape keys. The bug is NOT in DSP dispatch/shape handling. It's in **data flow** — the vision encoder's output isn't correctly reaching the decoder.

## Branch & Build State

- Branch: `ag_new_release_updates_2`
- Build: **CUDA build succeeds** with all changes (full rebuild after header modification)
- The benchmark runs but produces wrong output

## Files with Uncommitted Changes (DO NOT `git checkout` these)

All the shapeKeyState refactoring files listed above, plus prior uncommitted work:
- `libnd4j/include/helpers/MmulHelper.h`
- `libnd4j/include/helpers/cuda/MmulHelper.cu`
- `platform-tests/dsp-diagnostics.json`
- `platform-tests/src/test/java/.../TestDspCaptureConfigMatrix.java` (untracked)
- `platform-tests/src/test/java/.../TestDspMergedSegmentReplay.java` (untracked)

## Rules

- NEVER use `git checkout`, `git stash`, `git reset --hard`, or `git clean` — BANNED
- NEVER use workarounds — fix root causes directly
- Test output via tee: `cd platform-tests && bash run-benchmark.sh --tokens 250 2>&1 | tee /tmp/bench.log`
- Build: `/home/agibsonccc/dev-apps/mvn/bin/mvn -Pcuda -Dlibnd4j.triton=ON -Dlibnd4j.chip=cuda -Dlibnd4j.buildthreads=12 -Dlibnd4j.log=libnd4j-build.log -pl libnd4j,:nd4j-cuda-12.9 clean install -DskipTests 2>&1 | tee /tmp/build.log`
- Location tags in output are WRONG — expected output is text about mythic heroes
- The user handles builds themselves — write code and let them know when to build
