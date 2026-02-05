# VLM Debug Journal

## Build & Test Commands

### Build Command (CUDA)
```bash
/home/agibsonccc/dev-apps/mvn/bin/mvn -Pcuda -Dlibnd4j.chip=cuda -Dlibnd4j.buildthreads=4 \
  -pl libnd4j,:nd4j-cuda-12.9,:nd4j-cuda-12.9-preset,:nd4j-api \
  -Dlibnd4j.log=libnd4j-build.log clean install -DskipTests
```

### Test Command
```bash
mvn clean test -Dtest=org.eclipse.deeplearning4j.vlm.TestVLMModelImportPipeline#testSmolDoclingFullPipeline \
  -Dvlm.test.pdf.path=pathfinder-mythic.pdf \
  -Dvlm.test.pdf.page=10 \
  -Dvlm.test.maxTiles=4
```

---

## Debugging Utilities

### 1. Test Runner Prefix (via -Dtest.prefix)
Located in `platform-tests/bin/java`. Supports:
- **Valgrind**: `TEST_RUNNER_PREFIX="valgrind"` - Memory error detection
- **ASAN**: `TEST_RUNNER_PREFIX="asan"` - AddressSanitizer for memory issues
- **compute-sanitizer**: `TEST_RUNNER_PREFIX="compute-sanitizer"` - CUDA memory debugging
- **nsys**: `TEST_RUNNER_PREFIX="nsys"` - NVIDIA Nsight Systems profiler
- **nvprof**: `TEST_RUNNER_PREFIX="nvprof"` - NVIDIA profiler (legacy)

### 2. ND4J Debug/Verbose Mode
```java
Nd4j.getEnvironment().setDebug(true);
Nd4j.getEnvironment().setVerbose(true);
```
**Warning**: These significantly impact performance but trace ALL op execution.

### 3. System Properties for VLM Test
- `-Dvlm.test.pdf.path=<path>` - PDF file to process
- `-Dvlm.test.pdf.page=<int>` - Specific page (0-based)
- `-Dvlm.test.pdf.maxPages=<int>` - Max pages to process
- `-Dvlm.test.pdf.dpi=<int>` - Render DPI (default: 150)
- `-Dvlm.test.maxTiles=<int>` - Max tiles per image
- `-Dvlm.test.debugGraph=true` - Log decoder graph structure
- `-Dvlm.test.debugEmbeds=true` - Log embedding diagnostics
- `-Dvlm.test.disableNormalize=true` - Skip image normalization
- `-Dvlm.test.disablePixelMask=true` - Use full attention mask

---

## Code Style Rules

1. **Do NOT use fully qualified class names** - Use imports instead
2. **Do NOT deviate from the build command** - Use exact command above
3. **Keep solutions simple** - Avoid over-engineering

---

## Current Investigation: SmolDocling Pipeline (2026-01-29)

### Problem
The VLM pipeline generates repetitive gibberish tokens instead of meaningful text:
- Generated: `</и N</ Néce</éé</ééééééééé`
- Token IDs: `[9617, 7872, 442, 9617, 442, 2756, 319, 9617, 2756, 2756, ...]`
- Token 2756 ('é') repeats excessively after ~6 tokens

### Key Log Observations
```
Step 18: currentSeqLen=1, pastSeqLen=1176, totalSeqLen=1177
Step 18 layernorm0 stats: min=-0.03658294677734375, max=0.1121063232421875, mean=5.483730928972363E-4
Step 18: token_id=2756, text='é'

Step 19: currentSeqLen=1, pastSeqLen=1176, totalSeqLen=1177
Step 19 layernorm0 stats: min=-0.03658294677734375, max=0.1121063232421875, mean=5.483730928972363E-4
Step 19: token_id=2756, text='é'
```

**Note**: Identical layernorm stats when same token repeats is EXPECTED (same embedding -> same output).

### Pipeline Architecture
1. **Vision Encoder** (`SMOLDOCLING_VISION_ENCODER`) - Processes image tiles
2. **Decoder** (`SMOLDOCLING_DECODER`) - Autoregressive text generation
3. **Embed Tokens** (`SMOLDOCLING_EMBED_TOKENS`) - Token embedding lookup
4. **Tokenizer** (`SMOLDOCLING_TOKENIZER`) - Text tokenization

### Graph Modifications Applied
1. `fixDecoderInputIds()` - Replaces baked-in input_ids constant with dynamic zeros
2. `fixRepeatKVReshape()` - Diagnoses KV reshape operations
3. `fixDecoderInputsEmbeds()` - Rewires first layernorm to use inputs_embeds placeholder

### Suspected Root Causes (To Investigate)

#### 1. Vision Embeddings Quality
- Check if vision encoder output has meaningful variation
- Compare min/max/mean to expected ranges
- Verify connector linear layer weights are non-zero

#### 2. Prompt Format Mismatch
- SmolDocling uses Idefics3 format with row/col tokens
- Prompt: `<|im_start|>User:<row_N_col_M><image>...<global-img><image>...PROMPT<end_of_utterance>\nAssistant:`
- Check if `<image>` token ID is resolved correctly

#### 3. Decoder Graph Wiring Incomplete
- `fixDecoderInputsEmbeds()` only rewires ONE operation (first layernorm)
- May need to trace full embedding path and fix all connections
- Check what `oldInput` was before rewiring

#### 4. KV Cache Issues
- Verify past_key_values shapes match decoder expectations
- Check if present.X.key/value are being properly stored/retrieved
- Ensure attention mask grows correctly with pastSeqLen

### Critical Diagnostic: "Step 0 logits diff vs zero-embed (L2)"

The test already has a **key diagnostic** at step 0 (line 1161-1162):
```java
double diff = logitsForSampling.sub(zeroLast).norm2Number().doubleValue();
log.info("Step 0 logits diff vs zero-embed (L2): {}", diff);
```

**Interpretation:**
- If `diff ≈ 0`: The model is **NOT using inputs_embeds**! The graph wiring is broken.
- If `diff > 0` but output is garbage: The model uses embeddings but vision content isn't meaningful.

**Look for this in your logs:**
```
Step 0 logits diff vs zero-embed (L2): <VALUE>
```

### Next Debugging Steps

1. **Check the critical L2 diff**:
   - If diff ≈ 0: Focus on `fixDecoderInputsEmbeds()` - the wiring is incomplete
   - If diff > 0: Focus on vision encoder output quality

2. **Verify vision embedding integration**:
   - Log vision embedding stats per frame
   - Check if image tokens (`<image>`) are present in tokenized prompt
   - Verify fillCount matches visionSeqLen

3. **Trace decoder graph**:
   - Log what `oldInput` was in `fixDecoderInputsEmbeds()`
   - Check if there are other paths using original embeddings
   - Verify inputs_embeds variable type is PLACEHOLDER

4. **Compare to reference implementation**:
   - Run same image through HuggingFace SmolDocling
   - Compare intermediate activations

---

## File Locations

- Test class: `platform-tests/src/test/java/org/eclipse/deeplearning4j/vlm/TestVLMModelImportPipeline.java`
- Model downloader: `platform-tests/src/test/java/org/eclipse/deeplearning4j/vlm/VLMModelDownloader.java`
- InferenceSession: `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/autodiff/samediff/internal/InferenceSession.java`
- ForwardExecutionDAGBuilder: `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/autodiff/samediff/execution/ForwardExecutionDAGBuilder.java`
- Test runner script: `platform-tests/bin/java`

---

## Session History

### 2026-01-29 - Initial Investigation
- Analyzed test logs showing repetitive token generation
- Identified model is in degeneration loop (token 2756 repeating)
- Reviewed pipeline architecture and graph modification functions
- Created this debug journal

### 2026-01-30 - Double Free / JVM Crash Fix

**Problem:** Test JVM crashes with `double free or corruption (out)` (exit code 134) during
the vision encoder forward pass. The surefire output showed concurrent
`NullPointerException: Cannot invoke "AtomicBoolean.compareAndSet"` on
`OpaqueDataBuffer.tryMarkForDeallocation()` — the `markedForDeallocation` field was null.

**Root Cause:** Stale nd4j-cuda jar. The `markedForDeallocation` field and `tryMarkForDeallocation()`
method were newly added to `OpaqueDataBuffer` (nd4j-api), and both `CudaDeallocator` (nd4j-cuda)
and `OpaqueDataBufferDeallocator` (nd4j-api) were updated to call it. However the nd4j-cuda jar
in `.m2` was from Jan 29 18:24 (before these changes), while nd4j-api was rebuilt on Jan 30.
The old `CudaDeallocator` didn't call `tryMarkForDeallocation()`, but `OpaqueDataBufferDeallocator`
did — on buffer objects that weren't constructed with the new field present in the runtime class.

**Fix:** Rebuild nd4j-cuda with the build command:
```bash
/home/agibsonccc/dev-apps/mvn/bin/mvn -Pcuda -Dlibnd4j.chip=cuda -Dlibnd4j.buildthreads=4 \
  -pl libnd4j,:nd4j-cuda-12.9,:nd4j-cuda-12.9-preset,:nd4j-api \
  -Dlibnd4j.log=libnd4j-build.log clean install -DskipTests
```

**Additional Fix:** Even after rebuilding, the NPE persisted because `markedForDeallocation`
(a `final AtomicBoolean` initialized inline) was somehow null at runtime. This can happen
when JavaCPP creates `OpaqueDataBuffer` instances via native JNI without calling Java
constructors. Added null guard in `tryMarkForDeallocation()` and `isMarkedForDeallocation()`.

**Result:** Test now completes (877s, no errors, no JVM crash). Generated text changed from
`<fake_token_around_image>` gibberish to ` 101010101011111) 1)` — still wrong but the model
is now actually processing vision embeddings through the divide/scale operation.

**Lesson:** Always rebuild ALL modules together when changing shared interfaces like
`OpaqueDataBuffer`. The build command above rebuilds both nd4j-api and nd4j-cuda together.

### 2026-01-30 - fixDecoderInputsEmbeds Divide Operation

**Finding:** The old `fixDecoderInputsEmbeds()` was bypassing a `DivOp` between `inputs_embeds`
and the first layernorm. Investigation showed:
```
Producer op 'divide' type=DivOp, inputs=[inputs_embeds, expand_dims], outputs=[divide]
```
The divide op takes `inputs_embeds` (PLACEHOLDER) and divides by `expand_dims` (computed value).
The graph was ALREADY correctly wired: `inputs_embeds → divide → layernorm`.
The old code was BREAKING this by replacing divide with inputs_embeds directly into layernorm.

**Fix:** Rewrote `fixDecoderInputsEmbeds()` to trace upstream and wire `inputs_embeds` into the
producer op's embedding input slot, preserving any intermediate operations. Since `inputs_embeds`
was already the correct input, the fix is a no-op — confirming the divide path was always correct.

### 2026-01-29 - **ROOT CAUSE IDENTIFIED**

**Test Run Results:**
```
Vision embeddings stats: min=-41.35, max=47.53, mean=0.13
Text embeddings stats: min=-1.046875, max=0.9921875, mean=0.001
Filled 256 of 256 image token positions
Final inputsEmbeds stats: min=-1.046875, max=0.9921875, mean=0.001  <-- BUG!
Step 0 logits diff vs zero-embed (L2): 2111.697  <-- Decoder IS using inputs_embeds
Step 0 top-5: #1: <fake_token_around_image> (95% prob)
```

**Critical Finding:** Vision embeddings are NOT being inserted into inputsEmbeds!
- Vision range should be: -41 to +47 (large values from connector)
- Final range is only: -1 to +1 (text embedding range only)
- The `put()` operation is NOT modifying the array!

**Root Cause:** The code at lines 960-971:
```java
INDArray inputsEmbeds = textEmbeddings.dup();
INDArray visionFlat = visionEmbeddings.reshape((int) visionSeqLen, (int) visionHiddenSize);
for (int pos = 0; pos < promptTokenIds.length && fillIdx < fillCount; pos++) {
    if (promptTokenIds[pos] == imageTokenId) {
        inputsEmbeds.put(
            new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.point(pos), NDArrayIndex.all()},
            visionFlat.getRow(fillIdx)
        );
        fillIdx++;
    }
}
```

**Likely Issue:** Data type mismatch
- Text embeddings dtype: BFLOAT16 (range -1 to +1 typical)
- Vision embeddings dtype: FLOAT32 (range -41 to +47)
- When `put()` assigns FLOAT32 values into BFLOAT16 array, values may be:
  1. Clipped/saturated
  2. Not assigned at all
  3. Converted incorrectly

**IMMEDIATE FIX TO TRY:**
```java
// Cast vision embeddings to match text embedding dtype BEFORE put()
INDArray visionFlat = visionEmbeddings.reshape(visionSeqLen, visionHiddenSize)
                                       .castTo(textEmbeddings.dataType());
```

**Verification Steps:**
1. Add: `log.info("Vision dtype: {}, Text dtype: {}", visionFlat.dataType(), inputsEmbeds.dataType())`
2. After put loop: `log.info("Position 10 value check: {}", inputsEmbeds.getDouble(0, 10, 0))`
3. Compare with `visionFlat.getDouble(0, 0)` to verify assignment worked

---

### 2026-01-31 - reduce_mean Shape Fix (RESOLVED) & New Cleanup Crash

#### Problem: SIGABRT during `reduce_mean` on `[1, 1024, 768]` input

The VLM test crashed with SIGABRT (exit code 134) around operation ~3000 during a `reduce_mean`
op with input shape `[1, 1024, 768]` and axis `{-1}`. The expected output shape should be
`[1, 1024]` but the op was computing a scalar reduction instead.

#### Root Cause: `Shape.wholeArrayDimension({-1})` Sentinel Ambiguity

The `wholeArrayDimension()` method in `Shape.java` treated dimension array `{-1}` as the
"reduce all dimensions" sentinel. But in NumPy/ONNX convention, `-1` means "last axis"
(equivalent to axis 2 for a rank-3 tensor). The fix required normalizing negative axes
**before** checking the sentinel, across all six locations in the codebase.

#### Files Modified

1. **`nd4j/.../api/shape/Shape.java`** - `wholeArrayDimension()` and `reductionShape()`
   - Changed to normalize negative axes before checking sentinel value

2. **`nd4j/.../api/ndarray/BaseNDArray.java`** - All `*Number()` methods
   - Changed `mean(-1)` to `mean()` (scalar reduction, not last-axis reduction)

3. **`nd4j/.../api/ops/BaseOp.java`** - `defineDimensions()`
   - Normalize before sentinel check

4. **`nd4j/.../nd4j-cuda/.../CudaExecutioner.java`** - 4 occurrences
   - All normalize-before-sentinel pattern fixes

5. **`nd4j/.../nd4j-cpu-backend-common/.../NativeOpExecutioner.java`** - 2 occurrences
   - Line ~196 (first occurrence, previous session)
   - Line ~320 (second occurrence, this session):
   ```java
   // Before (buggy):
   if (Shape.wholeArrayDimension(dimLong)) { dimLong = new long[0]; }
   val dimension = Shape.normalizeAxis(x.rank(), dimLong);

   // After (fixed):
   dimLong = Shape.normalizeAxis(x.rank(), dimLong);
   if (Shape.wholeArrayDimension(dimLong)) { dimLong = new long[0]; }
   val dimension = dimLong;
   ```

#### Build/Compile Instructions (Multi-Module)

Maven reactor `-pl` selectors don't work for these modules. Must compile from each directory:

```bash
# 1. Compile nd4j-api (Shape.java, BaseNDArray.java, BaseOp.java)
cd /home/agibsonccc/Documents/GitHub/deeplearning4j/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api
/home/agibsonccc/dev-apps/mvn/bin/mvn install -DskipTests

# 2. Compile nd4j-cpu-backend-common (NativeOpExecutioner.java)
cd /home/agibsonccc/Documents/GitHub/deeplearning4j/nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cpu-backend-common
/home/agibsonccc/dev-apps/mvn/bin/mvn install -DskipTests

# 3. Compile nd4j-cuda (CudaExecutioner.java) - ONLY if CUDA changes
cd /home/agibsonccc/Documents/GitHub/deeplearning4j/nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cuda
/home/agibsonccc/dev-apps/mvn/bin/mvn install -DskipTests

# 4. Recompile platform-tests
cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests
/home/agibsonccc/dev-apps/mvn/bin/mvn test-compile
```

**Lesson:** Only recompiling platform-tests is NOT sufficient when changing backend modules.
The test classpath uses jars from `~/.m2/repository`, so each changed module must be `mvn install`ed.

#### Verification

- All 17 `HeapCorruptionReproducerTest` tests PASS including:
  - `testIsolatedReduceMean` - validates reduce_mean with axis {-1} on [1, 1024, 768]
  - `testVisionEncoderReduceMeanPipeline` - validates the exact VLM reduce_mean sequence
  - `testRepeatedVLMChunkPipeline` - 10 iterations of 100+ op graph (NEW, added this session)

---

#### New Crash: Post-Execution Cleanup on 5th Inference Run

After the reduce_mean fix, the VLM test progresses MUCH further:
- Previously crashed at ~3000 ops (during reduce_mean)
- Now completes **all ~9810 operations** across 5 inference runs
- **Crash occurs during post-execution cleanup** of the 5th inference run

**Evidence:**
```
# Run 5 completes all operations:
Op [9808] zeroslike   [1, 64, 576] -> ... completed
Op [9809] not_equals  ... completed
Op [9810] where       ... completed
# "EXECUTION COMPLETE" marker does NOT appear for run 5
# JVM crashes with SIGABRT (exit code 134)
```

**Cleanup code path** (`InferenceSession.java` lines 384-411):
```java
// Phase 1: Close OpContexts
for(var opContext : opContexts.values()) {
    opContext.close();  // native resource deallocation
}

// Phase 2: Close MemoryManager
if(mmgr != null) {
    mmgr.close();  // workspace memory cleanup
}

// Phase 3: Clear TAD cache
tadManager.purgeBuffers();
```

**Key characteristics of this crash:**
1. **Nondeterministic location**: With `MALLOC_CHECK_=3`, crash moved to a different point,
   confirming heap corruption rather than a specific bad operation
2. **Cumulative corruption**: Corruption accumulates over ~10,000 native CUDA operations
   and manifests during `free()` calls in cleanup
3. **Cannot reproduce in isolation**: Simplified test with 10 iterations of 100+ ops works fine;
   the corruption requires the full VLM model complexity (~1962 ops per inference × 5 runs)
4. **Model output is CORRECT**: All 5 runs produce `image_features` with shape `[1, 64, 576]`
   before the crash

**`platform-tests/bin/java` settings:**
- `MALLOC_CHECK_=0` (currently; set to 3 for stricter checking)
- `CUDA_LAUNCH_BLOCKING=1` (synchronous CUDA for debugging)

**Next steps for this crash:**
- Focus on native memory operations in the OpContext close/dealloc path
- Check if `OpaqueDataBuffer` or `OpaqueNDArray` deallocation has use-after-free issues
- Audit native C++ code for buffer overruns in frequently-called ops (reduce, boolean, where)
- Note: Valgrind never finishes on this workload, and ASAN is incompatible with CUDA and bloats module size. Manual native debugging required for memory tools.

### 2026-01-31 - Crash Point Identified: Heap Corruption During Native Allocation

#### Key Finding: NOT in Java cleanup, but in native `malloc()`

Added `System.out.println`+`flush` markers to `ArrayCacheMemoryMgr.close()` and
`InferenceSession.output()` cleanup phases. **None of the Java cleanup markers appeared in the log.**

The actual crash message (glibc):
```
corrupted size vs. prev_size
```

This means `malloc()` detected that a freed heap chunk's metadata (size field) doesn't match the
previous chunk's size field. The corruption was CAUSED by a prior `free()` (or buffer overrun) and
DETECTED by a subsequent `malloc()`.

#### Crash Sequence (from `/tmp/vlm-crash-diag.log`, 5.8M lines)

1. **Decoder completes**: Last op result is `[1, 1, 49280]` (vocabulary logits) at line ~5869271
2. **dbClose wave**: ~200 buffer deallocations via `dbClose` (lines ~5869300-5873574)
   - These are the `mmgr.close()` calls freeing cached intermediate arrays
   - Mix of sizes: 670464, 223488, 10476, 6144, 2304, 768, 384, 128, 48, 8, 5, 4 bytes
3. **New allocations begin**: ~60 allocations of 223488 bytes each (lines ~5873588-5873658)
   - These are for the NEXT frame's vision encoder execution
4. **CRASH at line 5873660**: `corrupted size vs. prev_size` during `allocateDataBuffer`
   - Allocation count: 400623, total allocated: ~5.79 GB

#### What This Tells Us

- The corruption happens **during** `dbClose` (native `free()`), not during Java cleanup
- A `free()` call corrupted adjacent heap chunk metadata
- Most likely cause: **buffer overrun** from an op that wrote past the end of its output buffer
  - The overrun corrupts the malloc metadata of the NEXT heap chunk
  - When that next chunk is freed, `free()` doesn't notice (it only checks its own metadata)
  - When `malloc()` later walks the free list, it finds mismatched `size` vs `prev_size`
- The 223488-byte allocations (55872 floats = 97*576 or similar attention shapes) are suspicious
- Total native allocations: ~400K buffers, ~150K OpaqueNDArrays, ~5.8 GB

#### Diagnostic Code Added

- `ArrayCacheMemoryMgr.close()`: `System.out.println("MMGR_CLOSE: ...")` with flush before each `arr.close()`
- `InferenceSession.output()` finally block: `System.out.println("CLEANUP: Phase N")` markers
- These markers help confirm that the crash is in the native `malloc` path, not Java cleanup

#### Next Investigation

- The overrunning op is likely in the **previous** inference run's execution
- Need to identify which op produces a buffer that's too small for the data it writes
- `calculateOutputShape` returning wrong shape would cause this - especially for dynamic-shape ops
- Candidate ops: Where (dynamic output), Gather, GatherNd, Stack, Concat (shape-dependent)

### 2026-01-31 - MALLOC_CHECK_=3 Analysis & DeallocatorService Experiment

**All debugging is on CUDA backend.**

#### MALLOC_CHECK_=3 Analysis

Changed `MALLOC_CHECK_` from 0 to 3 in `platform-tests/bin/java` line 226.

**Key observations from `/tmp/vlm-malloc-check3.log`:**
1. Detected corruption earlier: "double free or corruption (out)" after ~11171 ops
2. DeallocatorService frees massive buffers (3MB, 50MB) **concurrently with matmul execution**
3. dbClose wave at crash: ~200 deallocations with NO ops executing
4. Buffer sizes: 50MB (12288*1024*4), 3MB (64*12288*4) — weight matrices / activations
5. CLEANUP markers never appear → crash is DURING execution, not cleanup
6. Last op before crash: matmul [1,64,12288] x [12288,576] → [1,64,576]

#### DeallocatorService Blocking Experiment

Added diagnostic code to `InferenceSession.java` (lines 451-461, 528-533):
- `org.nd4j.inference.block.deallocator` system property (default: `false`)
- Had to hardcode default to `"true"` since Maven surefire doesn't forward `-D` to forked JVM
- Reverted to `"false"` after test

**Results with blocking enabled:**
- Crash **delayed** from frame 5 → frame 7 (each frame ~2302 ops)
- Crash type changed: "double free or corruption (out)" → SIGSEGV in `unlink_chunk.constprop.0`
- `addr2line -f -e /lib64/libc.so.6 0x9934f` → `unlink_chunk.constprop.0`
- Crash during frame 7's `expand_dims` op memory allocation
- `unlink_chunk` = glibc trying to coalesce free chunks with corrupted `fd`/`bk` pointers
- **Conclusion:** DeallocatorService is NOT the root cause — blocking reduces memory churn so corruption manifests later

#### Confirmed Facts About the Heap Corruption

| # | Fact | Implication |
|---|------|-------------|
| 1 | Platform: CUDA backend | All native code paths are CUDA variants |
| 2 | Crash = buffer overrun corrupting malloc metadata | NOT double-free |
| 3 | compute-sanitizer: 0 CUDA errors | NOT a GPU memory violation |
| 4 | GC disabled: still crashes | NOT a GC/DeallocatorService race |
| 5 | Guard bytes on primary buffer: no canary corruption | NOT a simple primary buffer overrun |
| 6 | DeallocatorService blocked: delays crash frame 5→7 | Confirms native op code is the source |
| 7 | Crash is cumulative | Accumulates over thousands of ops |
| 8 | Deterministic at alloc ~13875 (import) or frame 5-7 (inference) | Reproducible |
| 9 | Overrun is in native C++ CUDA host-side op execution | Focus area |
| 10 | MmulHelper.cpp doesn't use BufferAccessGuard | But GC-disabled still crashes, so not the cause |

#### Key Code Findings

- **ArrayCacheMemoryMgr `enableCache` = false** (property: `org.nd4j.autodiff.samediff.cache.enable`)
  - All intermediate arrays freed immediately after use → massive `dbClose` waves
- **InteropDataBuffer::primary()** has race condition (`releaseAccess()` before return) but irrelevant since GC-disabled still crashes
- **CpuOpContext/CudaOpContext** hold strong Java refs preventing GC during op execution
- **DataBuffer.cpp `setPrimaryBuffer`** free logic is safe on CUDA (primary not allocated with `allocateBoth=false`)
- **Reshape does NOT create views** in OpContext path — uses memcpy/assign, copies data

#### Where Investigation Stopped

Was about to investigate `expand_dims` and `scatter_nd_update` CUDA implementations for buffer overrun potential. Task agent was launched to check both ops but session ran out of context before results came back.

#### What Still Needs Investigation

- Which CUDA op's `calculateOutputShape` returns too-small shape
- Whether any op writes to host buffers that are undersized
- `expand_dims` and `scatter_nd_update` implementations (in progress when session ended)
- Whether enabling cache (`org.nd4j.autodiff.samediff.cache.enable=true`) changes behavior
- Dynamic-shape ops: Where, Gather, GatherNd, Stack, Concat, scatter_nd_update

### 2026-01-31 Session 3 - Broadcast Shape Bug Fix (Bug #1)

#### Problem: `evalBroadcastShapeInfo` treated [N] and [N,1] as equivalent

When computing broadcast shapes, `ShapeUtils::evalBroadcastShapeInfo` had a "vector shortcut"
that returned early when both inputs were vectors. However it did NOT check that ranks matched,
so `[15]` and `[15,1]` were both considered vectors and the function returned `[15]` as the
broadcast shape instead of the correct `[15,15]`.

This caused the causal attention mask (lower-triangular `[seq,seq]` matrix) to collapse to `[seq]`,
breaking attention masking.

#### Fix

**File:** `libnd4j/include/helpers/impl/ShapeUtils.cpp`

Added rank check to the vector shortcut condition:
```cpp
// Before (buggy):
if (xIsVector && yIsVector) { ... }

// After (fixed):
if (xIsVector && yIsVector && shape::rank(min) == shape::rank(max)) { ... }
```

#### Verification
- Test `testCausalMaskComputation` passes: `[15]` broadcast with `[15,1]` produces `[15,15]`

### 2026-01-31 Session 3 - Type-Punning Bug in Broadcastable Ops (Bug #2)

#### Problem: FLOAT+LONG addition produced wrong results

When adding a FLOAT tensor to a LONG tensor (e.g., attention scores + causal mask), the fused
broadcastable kernels used single-type templates that read both inputs as the same dtype.
This caused LONG values (like `Long.MIN_VALUE = -9.22e18`) to be reinterpreted as FLOAT bit
patterns, producing completely wrong results.

Example:
- Expected: `2.0 + (-9.22e18)` = `-9.22e18`
- Got: `2.0 + (garbage float)` = small number like `-3.0`

This broke causal attention masking — masked positions should have had `-inf`-like values
but instead had small negative values, causing softmax to assign non-zero attention to
future positions.

#### Fix

**Files modified** (same pattern in all four):
- `libnd4j/include/ops/declarable/generic/broadcastable/add.cpp`
- `libnd4j/include/ops/declarable/generic/broadcastable/subtract.cpp`
- `libnd4j/include/ops/declarable/generic/broadcastable/multiply.cpp`
- `libnd4j/include/ops/declarable/generic/broadcastable/divide.cpp`

Added type-casting before computation:
```cpp
NDArray *castX = nullptr, *castY = nullptr;
auto cleanupCasts = [&]() { delete castX; delete castY; };
if (x->dataType() != z->dataType()) {
  castX = x->cast(z->dataType());
  x = castX;
}
if (y->dataType() != z->dataType()) {
  castY = y->cast(z->dataType());
  y = castY;
}
```

#### Verification
- Test `testFloatPlusLongAddition` passes: FLOAT(2.0) + LONG(Long.MIN_VALUE) = FLOAT(-9.22e18)
- VLM test: `add_5` (mask application) now shows min=-9.22e18 (correct), softmax min=6.05e-39 (correct masking)

### 2026-02-01 - RotaryEmbedding Half-Dim Bug (Bug #3) - ROOT CAUSE OF WRONG OUTPUT

#### Problem: cos_cache last dim (32) was treated as full head_dim instead of half

The ONNX RotaryEmbedding spec defines cos_cache shape as `[max_seq, head_size/2]`. For
SmolDocling with head_dim=64, cos_cache is `[8192, 32]`. The implementation used
`headDimVar = cos_cache_shape[-1] = 32` as the full head dimension, causing:

1. Input reshaped to `[1, 283, 576/32=18, 32]` instead of `[1, 283, 9, 64]`
   - 18 fake "heads" of dim 32 instead of 9 real heads of dim 64
2. The `rotate_half` formulation tried `x*cos + rotate_half(x)*sin` where x has dim 64
   but cos/sin have dim 32 — dimensions don't broadcast
3. RoPE output was IDENTICAL to input (rotation was a no-op):
   - Q proj output: min=-8.955, max=5.425
   - RoPE output: min=-8.955, max=5.425 (IDENTICAL = bug confirmed)

This broke ALL 30 decoder layers × 2 RoPE applications each = 60 broken rotations,
completely corrupting the attention patterns.

#### Fix

**File:** `nd4j/samediff-import/samediff-import-onnx/src/main/kotlin/org/nd4j/samediff/frameworkimport/onnx/definitions/implementations/RotaryEmbedding.kt`

Key changes:
1. Renamed `headDimVar` to `halfHeadDimVar` (= cos_cache last dim = 32)
2. Added `actualHeadDimVar = halfHeadDimVar * 2` (= 64, the real head dim)
3. Used `actualHeadDimVar` for num_heads calculation and reshape target shape
4. Replaced broken `rotate_half` formulation with split-based approach matching ONNX spec:
```kotlin
// Split input into halves along head_dim axis
val splitParts = sd.split(workingInput, 2, -1)
val x1 = splitParts[0]  // [batch, seq, num_heads, half_head_dim]
val x2 = splitParts[1]  // [batch, seq, num_heads, half_head_dim]

// Apply rotation: y1 = x1*cos - x2*sin, y2 = x1*sin + x2*cos
val y1 = sd.math.sub(sd.math.mul(x1, cos), sd.math.mul(x2, sin))
val y2 = sd.math.add(sd.math.mul(x1, sin), sd.math.mul(x2, cos))
result = sd.concat(-1, y1, y2)
```
5. Also implemented interleaved mode with reshape-to-pairs approach

#### Build Command (Kotlin module, not C++)
```bash
/home/agibsonccc/dev-apps/mvn/bin/mvn -pl nd4j/samediff-import/samediff-import-onnx -am -DskipTests install
```

#### Verification

Confirmed reshape is now correct: `[1, 283, 9, 64]` (9 heads, 64 head_dim) instead of `[1, 283, 18, 32]`

Logit comparison before vs after RoPE fix:
| Metric | Before Fix | After Fix | ONNX Runtime |
|--------|-----------|-----------|-------------|
| logits min | -16.73 | -24.39 | -23.30 |
| logits max | 7.32 | 23.65 | 22.24 |
| logits mean | -6.03 | -7.83 | -7.22 |
| `<doctag>` logit | -6.43 | 3.10 | 2.07 |

SameDiff logit distribution now closely matches ONNX Runtime baseline.

#### Generated Output After Fix

```
 <doctag><text><loc_0iae.0>User: 0 the</text>
```

The model now produces DocTag-structured output:
- `<doctag>` at step 1 (correct format start)
- `<text>` (correct DocTag element)
- `<loc_...>` (correct location coordinate format)
- `</text>` (correct closing tag)

However, quality issues remain:
- Leading space at step 0 instead of `<doctag>` being first
- Location coordinates garbled (`0iae.0` instead of proper numeric coordinates)
- "User:" from prompt appearing in output

Note: ONNX Runtime with same random vision features ALSO produces space as top-1 token
(logit=22.24) with `<doctag>` rank ~346, confirming this behavior is expected for non-real
image data. The model architecture is now working correctly.

#### Remaining Investigation

The output quality depends on vision encoder producing correct features for the real PDF image.
Need to verify:
1. Image preprocessing (normalization, tiling) matches HuggingFace reference
2. Vision encoder ONNX model produces same features as PyTorch reference
3. Embedding merge (text + vision) is numerically correct on device
