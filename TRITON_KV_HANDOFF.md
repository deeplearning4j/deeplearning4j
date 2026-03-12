# Triton Mega-Kernel KV Cache Bug — Handoff Prompt

## Problem
SmolDocling VLM decoder produces **2 unique tokens** (degenerate) when using the Triton mega-kernel backend. Without Triton (CUDA slot-by-slot), it produces ~67% unique tokens correctly.

## Root Cause
The Triton mega-kernel fuses all 3840 decoder ops into ONE kernel. The 30 `onnx_multi_head_attention` ops' **present_key/present_value outputs are internal intermediates** — the kernel never writes them to `outputSlots[]`. After the kernel, `scatterKvEntries()` reads from `slotArrayCache_[presentOutputSlotIdx]` which contains stale/zeroed data. The static KV buffer never updates → identical output every step.

## What Was Done
A `composePresentKv` fix was added in `TritonGraphBackend::executeSegment()` (lines ~1985-2100 of `TritonGraphBackend.cpp`). It:
1. Iterates segment slots looking for `"onnx_multi_head_attention"` ops
2. Reads K/V projection outputs from `outputSlots[inputSourceIndices[1/2]]` (these ARE forced external at lines 5308-5325 of `TritonIRBuilder.cpp`)
3. Scatters into `outputSlots[outputSlotIndices[1/2]]` (present_key/present_value) at `lastPos`
4. Then `kvScatter` reads `present[lastPos]` → `staticBuf[cachePos]`

File-based diagnostic logging was added (`/tmp/triton_compose_kv.log`) using `fprintf` since `sd_printf` goes to stderr which surefire doesn't capture.

## What Has NOT Been Verified
**Every test run so far had `HAVE_TRITON:BOOL=OFF`** due to missing `-Dlibnd4j.triton=ON` maven flag. The composePresentKv code was **never actually exercised**. We don't know:
1. Whether the attention ops match the `"onnx_multi_head_attention"` string check in the NativeSlot structs
2. Whether `outputSlots[inputSourceIndices[1]]` has valid K projection data after kernel execution
3. Whether the present output slot has correct shape `[1,3,780,64]`
4. Whether the scatter offsets are correct

## Current Build State
- `HAVE_TRITON:BOOL=ON` in CMakeCache.txt
- `make -j4 nd4jcuda` completed successfully (incremental, TritonGraphBackend.cpp recompiled)
- **BUT** the nd4j-cuda-12.9 jar was NOT reinstalled — need to run:
  ```bash
  cd /home/agibsonccc/Documents/GitHub/deeplearning4j && \
  /home/agibsonccc/dev-apps/mvn/bin/mvn -Pcuda -Dlibnd4j.chip=cuda \
    -pl :nd4j-cuda-12.9,:nd4j-cuda-12.9-preset \
    -Dlibnd4j.triton=ON install -DskipTests -Dlibnd4j.build=skip
  ```
  (The `-Dlibnd4j.build=skip` avoids re-running cmake/make since the .so is already built)

## Key Files
| File | What | Lines |
|------|------|-------|
| `libnd4j/include/graph/gpu/TritonGraphBackend.cpp` | composePresentKv scatter code + file-based diagnostics | ~1985-2100 |
| `libnd4j/include/graph/gpu/TritonIRBuilder.cpp` | Forces K/V proj as external outputs | 5308-5325 |
| `libnd4j/include/graph/gpu/TritonIRBuilder.cpp` | SSA pass-through: `present_key = K_proj` | ~4373 |
| `libnd4j/include/graph/impl/NativeDynamicShapePlan.cpp` | `scatterKvEntries()` reads slotArrayCache | ~3902 |
| `libnd4j/include/graph/NativeDynamicShapePlan.h` | NativeSlot struct, inputSourceIndices encoding | 96-113 |

## Build Command
```bash
/home/agibsonccc/dev-apps/mvn/bin/mvn -Pcuda -Dlibnd4j.chip=cuda -Dlibnd4j.buildthreads=4 -Dlibnd4j.triton=ON \
  -pl libnd4j,:nd4j-cuda-12.9,:nd4j-cuda-12.9-preset,:nd4j-api \
  -Dlibnd4j.log=libnd4j-build.log clean install -DskipTests
```
**CRITICAL**: Must use `-Dlibnd4j.triton=ON`. Verify `HAVE_TRITON:BOOL=ON` in `libnd4j/blasbuild/cuda/CMakeCache.txt`.

## Test Command
```bash
cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && \
  mvn test -Dtest=TestSmolDoclingOptimizedPipeline#testOptimizedDoclingPipeline
```
After test, check `/tmp/triton_compose_kv.log` for diagnostic output.

## Next Steps
1. Install the nd4j-cuda jar (see command above)
2. Run the test
3. Check `/tmp/triton_compose_kv.log`:
   - If file exists with attention ops found → composePresentKv runs, check shapes/offsets
   - If file exists with "NO attention ops found" → opName or numInputs/numOutputs filter is wrong, log shows first 20 slot names
   - If file doesn't exist → executeSegment is never called (Triton backend not active)
4. If composePresentKv runs but output is still degenerate → scatter offsets or K proj data may be wrong
5. If composePresentKv doesn't run → fix the matching criteria

## Performance Issue (Secondary)
Single mega-kernel serializes 3840 ops with 256 threads → ~233ms/step vs ~25ms for CUDA graph slot-by-slot. Fundamental throughput problem. Not blocking correctness fix.

## Data Flow Diagram
```
Triton mega-kernel:
  reads: externalInputs[] (static KV buffers, attention_mask, embeddings, weights)
  writes: outputSlots[] (only for "external" outputs — K/V proj ARE external, present_key/value are NOT)

composePresentKv (post-kernel):
  reads: outputSlots[K_proj_slot] (current token's K projection, shape ~[1,3,1,64])
  writes: outputSlots[present_key_slot] at lastPos (shape [1,3,780,64])

scatterKvEntries (post-segment):
  reads: slotArrayCache_[present_key_slot][lastPos] (same buffer as outputSlots[present_key_slot])
  writes: externalInputs[past_key_ext_idx][cachePos] (static KV buffer for next step)
```
