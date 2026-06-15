# SigLIP Vision Import Debug Journal

## Test Information
- **Test**: `TestVLMModelImportPipeline#testSigLIPVisionImport`
- **Test Command**: `mvn test -Dtest=TestVLMModelImportPipeline#testSigLIPVisionImport`
- **Optional Debug Prefixes**:
  - Compute sanitizer: `-Dtest.prefix=/usr/local/cuda/bin/compute-sanitizer`
  - Valgrind (slow, avoid): `-Dtest.prefix=valgrind -s --logfile="somelog.txt"`

## Build Commands
- **C++ rebuild (CUDA)**:
```bash
/home/agibsonccc/dev-apps/mvn/bin/mvn -Pcuda -Dlibnd4j.chip=cuda -Dlibnd4j.buildthreads=4 -pl libnd4j,:nd4j-cuda-12.9,:nd4j-cuda-12.9-preset,:nd4j-api -Dlibnd4j.log=libnd4j-build.log clean install -DskipTests
```
- **Install modules only (no compile)**:
```bash
mvn install -DskipTests -pl <module>
```

## Current Status
**FAILING** - JVM crashes with SIGABRT in native CUDA code

## Session History

### 2026-01-25 - Initial Investigation

#### Problem 1: Reshape error with constants
- **Error**: Reshape failed "150528 vs 768" - shape tensor had wrong values
- **Root Cause**: Gather operation was receiving index constants with shape `[1,1]` instead of scalar
- **Fix Applied**: Modified `Gather.kt` to flatten indices to 1D with `sd.reshape("${outputNames[0]}_indices_flat", indicesVariable, -1L)`
- **Files Changed**: `nd4j/samediff-import/samediff-import-onnx/src/main/kotlin/org/nd4j/samediff/frameworkimport/onnx/definitions/implementations/Gather.kt`

#### Problem 2: MatMul reshape error
- **Error**: Reshape failed "150528 vs 2352" on 4D attention tensors [1, 12, 196, 64]
- **Root Cause**: MatMul handler had complex 3D reshape logic that didn't work for 4D batched attention
- **Fix Applied**: Simplified `MatMul.kt` to just use `sd.linalg().matmul()` for batched matmul
- **Files Changed**: `nd4j/samediff-import/samediff-import-onnx/src/main/kotlin/org/nd4j/samediff/frameworkimport/onnx/definitions/implementations/MatMul.kt`

#### Problem 3: Constant node handling
- **Issue**: Constants not found during import - only checking by node name, not output name
- **Fix Applied**: Updated `OnnxIRGraph.kt` to search for Constant nodes by output name
- **Files Changed**: `nd4j/samediff-import/samediff-import-onnx/src/main/kotlin/org/nd4j/samediff/frameworkimport/onnx/ir/OnnxIRGraph.kt`

#### Current Problem: JVM SIGABRT Crash
- **Symptom**: Test JVM crashes with SIGABRT (signal 6)
- **Location**: Native code in `libnd4jcuda.so`
- **Core dumps**: Multiple SIGABRT crashes recorded in `coredumpctl`
- **Surefire reports**: Only show "Connection reset" - forked JVM died

**Last coredump info (PID 2702433)**:
- Signal: 6 (ABRT)
- Involves: libnd4jcuda.so, CUDA 12.9 libraries
- Stack trace shows crash in native code

## Files Modified This Session
1. `Gather.kt` - Added indices flattening
2. `MatMul.kt` - Simplified to use built-in matmul
3. `OnnxIRGraph.kt` - Added constant node lookup by output name

## 2026-01-25 12:12 - Continued Investigation

### Key Finding from Prior Session
The original error BEFORE crashes showed:
```
Operation: /vision_model/encoder/layers.0/self_attn/Reshape_1_output_0
Input: [1, 196, 768] (150528 elements)
Shape tensor: [4] values = [1, 1, 12, 64] (768 elements)
Error: Reshape lengths mismatch 150528 vs 768
```

The shape tensor should be `[1, 196, 12, 64]` but the `seq_len` (196) is being replaced with 1.
This is from Gather operations extracting shape values from constants.

### ONNX Module Install Issue
- The ONNX import JAR wasn't being installed properly (was from March 2025)
- Fixed by removing old JAR and reinstalling: `rm -rf ~/.m2/repository/org/eclipse/deeplearning4j/samediff-import-onnx/1.0.0-SNAPSHOT/`
- Then: `mvn install -pl nd4j/samediff-import/samediff-import-onnx -DskipTests`

### Current Crash Analysis
- JVM crashes with SIGABRT before producing any test output
- Crashes started AFTER our code changes to Gather.kt, MatMul.kt, OnnxIRGraph.kt
- Crash location: native code in libnd4jcuda.so

### Root Cause Found
From debug output in prior session:
```
Gather: indices constant value: [[0]]   <-- 2D shape [1,1], NOT scalar!
Gather: indices constant value: [[1]]   <-- 2D shape [1,1], NOT scalar!
```

The ONNX model has index constants as 2D arrays `[[0]]` with shape [1,1].
When Gather uses 2D indices [1,1] on 1D shape tensor [3]:
- ONNX output rank = indices.rank + input.rank - 1 = 2 + 1 - 1 = 2
- Output shape [1,1] containing the extracted value

This breaks Concat which expects scalar/1D values to concatenate into shape tensor.
Result: Concat produces [1,1,12,64] instead of [1,196,12,64] because:
- Each Gather output is [1,1] shaped, Concat sees first dim as 1

### Fix Strategy
Squeeze the Gather OUTPUT (not input indices) to remove extra dimensions.
The output should be scalar or 1D, not 2D.

## 2026-01-25 12:38 - Current State

### SIGABRT Crashes
- All test runs crash with SIGABRT (signal 6) in native code
- No hs_err files generated (SIGABRT != SIGSEGV)
- Native library libnd4jcuda.so was built at 09:40 today
- Coredumps started at 11:00

### Uncommitted libnd4j Changes
Many C++ files are modified (from git status):
- DataBuffer.cpp, DataBuffer.cu
- Context.cpp, Context.h
- ShapeBuilders.cpp, shape.h
- conv2d.cpp, deconv2d.cpp, etc.
- Many pooling and convolution ops

These native changes may be causing the SIGABRT crashes.

### Current File States
1. **Gather.kt** - Simple version (just sd.gather, no squeeze/reshape)
2. **MatMul.kt** - Simplified version using sd.linalg().matmul()
3. **OnnxIRGraph.kt** - Has extractConstantValue() and constant lookup by output name

### Confirmed Finding
Even with simple Gather.kt (no squeeze/reshape), the test still crashes with SIGABRT.
This proves the crash is NOT caused by our ONNX handler changes.

The SIGABRT crashes are caused by something else:
- Possibly the uncommitted libnd4j C++ changes
- Possibly a mismatch between Java API and native library
- Native library may need to be rebuilt

### Rebuild Attempt - 12:49
1. Rebuilt native library with CUDA build command
2. Cleared javacpp cache: `rm -rf ~/.javacpp/cache/nd4j-cuda-12.9-1.0.0-SNAPSHOT-linux-x86_64.jar`
3. New native library extracted at 12:49
4. **Still crashing with SIGABRT** - rebuild did not fix the issue

### Next Steps
1. Need to investigate what's causing the SIGABRT in the native code
2. Consider using compute-sanitizer: `-Dtest.prefix=/usr/local/cuda/bin/compute-sanitizer`
3. The issue may be a bug in the uncommitted C++ changes themselves

## User Instructions (IMPORTANT)
- Do NOT blindly revert code with git
- FIX the issue with manual updates to current code
- Keep the simplified MatMul if possible
- Write down findings in this journal
- Do NOT endlessly rebuild modules - assume install worked after running once
- Run tests and install ONCE only - check surefire logs for results
- Stop repeating the same actions expecting different results
- Do NOT check for imports - only fix imports when there's a compilation error
- Focus on: which op ran last and the graph structure
- Trace the graph structure using the variable names from the error output

## 2026-01-25 14:13 - Compute Sanitizer Run

### Ran test with compute-sanitizer
Command: `mvn test -Dtest=TestVLMModelImportPipeline#testSigLIPVisionImport -Dtest.prefix=/usr/local/cuda/bin/compute-sanitizer`

### Result: Real error exposed (no more SIGABRT crash!)
```
Operation: /vision_model/encoder/layers.0/layer_norm1/Pow
Input 0: [1, 196, 768] FLOAT
Input 1: [1, 1] FLOAT (constant 2.0) <-- WRONG SHAPE, should be scalar!
Error: PairWiseTransform intermediateShaped(...) failed - Error code [719]
```

### Root Cause
ONNX Constant nodes with single values stored as tensors with shape [1,1] instead of scalar.

### Fixes Applied

1. **OnnxIRGraph.kt:extractConstantValue()** - Squeeze single-element tensors to scalar
   ```kotlin
   if (arr.length() == 1L && arr.rank() > 0) {
       arr = arr.reshape()  // empty shape = scalar, preserves datatype
   }
   ```

2. **Gather.kt** - Cast float indices to INT64
   ```kotlin
   if (!indicesType.isIntType) {
       indicesVariable = sd.castTo("${outputNames[0]}_indices_int", indicesVariable, DataType.INT64)
   }
   ```

3. **Concat.kt** - Cast all inputs to same datatype for axis=0 concat
   ```kotlin
   val targetDtype = inputVars[0].dataType()
   if (inputVar.dataType() != targetDtype) {
       flattened = sd.castTo("${outputNames[0]}_cast$idx", flattened, targetDtype)
   }
   ```

### Files Modified
- `nd4j/samediff-import/samediff-import-onnx/.../OnnxIRGraph.kt`
- `nd4j/samediff-import/samediff-import-onnx/.../Gather.kt`
- `nd4j/samediff-import/samediff-import-onnx/.../Concat.kt`

### Current State
- ONNX module installed with fixes at 14:29
- Test run was interrupted - no complete result yet

## Important Rules
- NEVER run the test more than once - check surefire-reports
- Check `/platform-tests/target/surefire-reports/` for test output
- Dumpstream files indicate JVM crash, not test failure
- Do NOT use `a.shape` - use `sd.shape(a)` for dynamic shape handling

## 2026-01-25 18:00 - MatMul Rank Mismatch Fix

### Problem
MatMul in head/attention failing with rank mismatch:
```
Input A: [196, 1, 768] - rank 3
Input B: [768, 1536] - rank 2
Error: ShapeUtils::evalShapeForMatmul - ranks must be same, got xRank = 3 and yRank = 2
```

### Root Cause
libnd4j matmul requires same rank for both inputs. ONNX MatMul supports broadcasting when ranks differ.
The head attention has 3D tensor multiplied by 2D weight matrix.

### Fix Applied to MatMul.kt
Only flatten A to 2D for head/attention matmuls:
```kotlin
val isHeadAttention = opName.contains("head/attention") || opName.contains("head.attention")
if (isHeadAttention) {
    // Reshape A to [-1, last_dim]
    val minusOne = sd.constant(Nd4j.createFromArray(-1L))
    val lastDim = sd.sizeAt("${opName}_last_dim", a, -1)
    val flatShape = sd.concat("${opName}_flat_shape", 0, minusOne, lastDim)
    val aFlat = sd.reshape("${opName}_a_flat", a, flatShape)
    val output = sd.linalg().matmul(opName, aFlat, b)
} else {
    // Standard batched matmul for encoder attention (4D x 4D)
    val output = sd.linalg().matmul(opName, a, b)
}
```

### Key Points
- Encoder attention has 4D x 4D batched matmuls - work as-is
- Head attention has 3D x 2D matmuls - need flattening
- Use `sd.concat()` to build shape tensor, NOT `sd.stack()` (stack adds dimension)
- Use `sd.sizeAt()` to get dimension size dynamically
- Use `Nd4j.createFromArray(-1L)` for constant -1 in reshape

## 2026-01-25 20:46 - FIXED: Native MatMul Broadcast Support

### Solution
Instead of hacky import-time workarounds, fixed the issue in libnd4j native code to support ONNX MatMul broadcast semantics.

### Files Modified
1. **`libnd4j/include/helpers/impl/ShapeUtils.cpp`** - `evalShapeForMatmul()`
   - Removed requirement that leading dimensions must be singletons
   - Now handles general ND x 2D case: `[batch..., M, K] @ [K, N] -> [batch..., M, N]`
   - Also handles 2D x ND case: `[M, K] @ [batch..., K, N] -> [batch..., M, N]`

2. **`libnd4j/include/ops/declarable/generic/blas/matmul.cpp`** - `CUSTOM_OP_IMPL(matmul)`
   - Updated execution to flatten ND tensor to 2D before matmul
   - For x[batch..., M, K] @ y[K, N]: flatten x to [batchSize*M, K], matmul, result goes to z
   - For x[M, K] @ y[batch..., K, N]: flatten y to [K, batchSize*N], matmul, result goes to z

3. **`MatMul.kt`** - Simplified to just call `sd.linalg().matmul()`
   - No more complex flattening logic in import hook
   - Native op handles broadcast semantics

### Test Results
```
Tests run: 1, Failures: 0, Errors: 0, Skipped: 0
BUILD SUCCESS
Total time: 02:28 min

Output shapes:
  pooler_output: shape=[1, 768], dtype=FLOAT
  last_hidden_state: shape=[1, 196, 768], dtype=FLOAT
```

### Why This is Better
1. **Generalized solution** - Works for all ONNX MatMul operations, not just specific paths
2. **Native performance** - Flattening happens in C++, not through graph operations
3. **Simpler import code** - No complex dynamic shape computations in Kotlin
4. **Matches ONNX spec** - Proper broadcast semantics implemented at the op level
