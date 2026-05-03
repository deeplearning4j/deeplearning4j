---
name: cuda-rms-norm-linear-gemm-dims-swap
description: CUDA rms_norm_linear usualGemm blocksPerGrid/threadsPerBlock swap bug — error code 9, invalid configuration argument
type: bug
---

# CUDA rms_norm_linear GEMM Failure — usualGemm dims.x/dims.y Swap Bug

Documented: 2026-05-02

## Symptom

ALL Qwen test configs on CUDA fail during prefill with:
```
MMUL cuda gemv case failed, error code 9 = invalid configuration argument
```
This error appears at MmulHelper.cu:550 (usualGemm path) and MmulHelper.cu:628 (usualGemv path).

It fires at slot 1881 — the first `rms_norm_linear` op during prefill — when model weights are FP16 (from `forInference()` / `DEQUANTIZE_TO_FLOAT16` on GPU).

## Root Cause: dims.x and dims.y SWAPPED in usualGemm call

### getMMulDims() return convention (LaunchDims.cu:512)

```cpp
dim3 getMMulDims(int length, int sizeofDataType) {
    int threadsPerBlock = SD_MAX_NUM_THREADS / 2;  // == 512
    int blocksPerGrid = (length + threadsPerBlock - 1) / threadsPerBlock;
    int sharedMem = ...;
    // ...
    return dim3(blocksPerGrid, threadsPerBlock, sharedMem);
    // x = blocksPerGrid, y = threadsPerBlock, z = sharedMem
}
```

### Actual call site (MmulHelper.cu:976-980)

```cpp
dim3 dims = getMMulDims(C->lengthOf(), DataTypeUtils::sizeOf(cType));
BUILD_SINGLE_SELECTOR_THRICE(aType, usualGemm,
    (dims.y, dims.x, dims.z, stream, ...),
    //  ^^^^^  ^^^^^  swapped!
    SD_NUMERIC_TYPES)
```

### usualGemm signature (MmulHelper.cu:543)

```cpp
SD_HOST static void usualGemm(
    const int blocksPerGrid,     // receives dims.y = 512 (threadsPerBlock)
    const int threadsPerBlock,   // receives dims.x = (length+511)/512 — can be >> 1024!
    const int sharedMem,         // receives dims.z (correct)
    ...)
```

### Why this causes error 9

CUDA enforces a maximum of 1024 threads per block. When a large matrix (e.g., hidden_size=2048) is the output:
- `length = M * N` for the matmul output (e.g., 2048 for a single-row matmul)
- `blocksPerGrid (correct) = ceil(2048/512) = 4`
- `threadsPerBlock (correct) = 512`

After the swap:
- `threadsPerBlock passed = dims.x = 4` → launches with only 4 threads per block (wrong but valid)
- `blocksPerGrid passed = dims.y = 512` → launches 512 blocks (harmless)

Wait — actually the dim3 indexing means:
- For a `dim3(blocksPerGrid=4, threadsPerBlock=512, sharedMem=...)` returned value
- `dims.x = 4` (blocksPerGrid)
- `dims.y = 512` (threadsPerBlock)

The call is `usualGemm(dims.y=512, dims.x=4, dims.z, ...)` so usualGemm receives:
- blocksPerGrid = 512 (was threadsPerBlock value)
- threadsPerBlock = 4 (was blocksPerGrid value) 
- This is valid for small matrices (threadsPerBlock=4 is legal)

For LARGE matrices where `blocksPerGrid > 1024`:
- `length = large_M * large_N` e.g., seqLen=512 * hidden=2048 = 1,048,576
- `blocksPerGrid (correct) = ceil(1,048,576/512) = 2048`
- The call passes `threadsPerBlock = dims.x = 2048` → ILLEGAL (>1024) → error 9

## Why forInference() FP16 Specifically Triggers This

### The cuBLAS type guard

```cpp
const bool typeHalf = ABC && effAType == HALF && major >= 6;
const bool typeHalfFloat = AB && effAType == HALF && cType == FLOAT32 && major >= 6;

if (!typeDouble && !typeFloat && !typeHalf && !typeIntFloat && !typeHalfFloat) {
    // usualGemm fallback — bug lives here
}
```

For standard FP16 (HALF×HALF→HALF or HALF×HALF→FLOAT32 on sm_6+), `typeHalf` or `typeHalfFloat` is true and cuBLAS handles it. The `usualGemm` fallback is NOT normally reached for FP16 on modern GPUs.

### Why it fails anyway

The `rmsNormLinearGeneralLauncher` (M>1 / prefill path) does:
```cpp
NDArray normalized(input->ordering(), shapeVec, input->dataType(), context);
rmsNorm(context, input, gamma, &normalized, epsilon);
MmulHelper::mmul(&normalized, weight, output, 1.0, 0.0);
```

`normalized` has `input->dataType()` = HALF. `weight->dataType()` = HALF. But `output->dataType()` depends on what DSP shape inference allocated. If during prefill the output array was allocated as a different dtype (e.g., FLOAT32 but with wrong type inference), or on a pre-Pascal GPU (major < 6), or if CUTLASS dispatch triggers and then falls back, the usualGemm path can be reached.

The observation of 668 cast ops (vs 36 in successful runs) confirms that forInference() adds extensive FP16 dequantize casts. This changes the dtype graph significantly, and shape inference for the rms_norm_linear output slot may produce a type that slips past all cuBLAS guards.

### Confirmed triggering condition

- Successful tests (May 1 morning): `cast=36 ops`, default FP32 options, NOT using forInference()
- Failing tests (May 1 afternoon): `cast=668 ops`, FP16 weights from `forInference()`, hits slot 1881

## The Actual Fix

**Fix the dims.x/dims.y swap in MmulHelper.cu:976-980.**

Change:
```cpp
BUILD_SINGLE_SELECTOR_THRICE(aType, usualGemm,
    (dims.y, dims.x, dims.z, stream, ...),
    SD_NUMERIC_TYPES)
```

To:
```cpp
BUILD_SINGLE_SELECTOR_THRICE(aType, usualGemm,
    (dims.x, dims.y, dims.z, stream, ...),
    SD_NUMERIC_TYPES)
```

This makes `blocksPerGrid = dims.x` (the computed grid size) and `threadsPerBlock = dims.y` (512), matching the getMMulDims() convention.

**File:** `libnd4j/include/helpers/cuda/MmulHelper.cu` line 977

## Note on usualGemv Path

The GemV path (mmulMxV, line 1222-1228) is CORRECT — it uses named variables:
```cpp
const int blocksPerGrid = dims.x;
const int threadsPerBlock = dims.y;
BUILD_SINGLE_SELECTOR_THRICE(xType, usualGemv, (blocksPerGrid, threadsPerBlock, stream, ...), ...)
```
Only the mmulMxM path at line 977 has the swap bug.

## Impact

- Blocks ALL Qwen CUDA prefill with FP16 weights (any rms_norm_linear in M>1 prefill)
- Only affects types that don't route to cuBLAS (type combos outside typeDouble/typeFloat/typeHalf/typeIntFloat/typeHalfFloat)
- Pre-Pascal GPUs (major < 6) are always affected since typeHalf=false there
- On Pascal+ with pure FP16 weights, the failure mode may require specific dtype mismatches in DSP shape allocation

## Relationship to DSP Accuracy Regression

This bug is separate from (but compounds) the accuracy regression issues in dsp_accuracy_regression_cuda.md. The dims swap is a pre-existing bug in the fallback kernel launch path that becomes visible when forInference() FP16 weights route GEMM calls through unusual type combos not covered by cuBLAS fast paths.

## Verification Test

Run CUDA Qwen test with forInference() after fixing the swap:
```bash
cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && \
  mvn test -Dtest=TestQwen35Pipeline -Dbackend.artifactId=nd4j-cuda-12.9 \
  2>&1 | tee /tmp/qwen-cuda-fp16-fix.log
```
Expected: no "error code 9 = invalid configuration argument" at slot 1881.
