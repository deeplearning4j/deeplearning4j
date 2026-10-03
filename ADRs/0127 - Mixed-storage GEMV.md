# ADR 0127: Mixed-storage GEMV

## Status

Accepted (2026-10-03). Builds on ADR 0100 (the graph optimizer stores weights
as FP16 by default).

## Context

The graph optimizer stores weights in low precision (`nd4j.optimizer.weightDtype`,
FP16 by default, ADR 0100) while activations stay FLOAT. A decode step therefore
multiplies a FLOAT activation row by a HALF (or BFLOAT16) weight matrix. No BLAS
reads such a pair: cuBLAS and cuBLASLt take A and B of one type (FP8 pairs
aside), and host BLAS takes one FLOAT32 or DOUBLE type throughout. MmulHelper
widened the weights instead:

- CUDA `mmulMxV` cast the whole weight to FLOAT32 on every call. SmolDocling
  decode made 31 such copies of ~885k elements per token, 2.9 ms per token.
- CUDA `mmulMxM` kept a FLOAT32 copy of each weight in the cast cache, refreshed
  on every call unless the weight's buffer was constant: twice the weight memory,
  and twice the bytes per product.
- Narrowing the activation to HALF instead loses activation precision across
  layers, which compounds over a model's depth.

On CPU, `mmulMxV` and `dot` threw on any mix of types, and `mmulMxM` cast mixed
operands to A's type, which narrowed a FLOAT32 operand to HALF when A was HALF.

The matmul op's shape function picked its output type by enum ordinal, which
ranks BFLOAT16 (17) and every integer type above FLOAT32 (5) and DOUBLE (6):
FLOAT x BFLOAT16 gave BFLOAT16. Java's `Mmul` returns the wider input type.

## Decision

### The mixed GEMV

A matrix-vector product over mixed float storage reads every operand in place:

    y[r] = alpha * sum_k W[r, k] * x[k] + beta * y[r]

W, x and y may each be any float type the selectors dispatch (`ALL_FLOATS`:
HALF, BFLOAT16, FLOAT32, DOUBLE) and are read through their own strides, so
views, transposes and strided operands need no copy. Products and sums are in
`ProductAccumulator<W, X, Y>`: FLOAT for HALF, BFLOAT16 and FLOAT operands,
DOUBLE when any operand is DOUBLE. `MixedGemvLayout` (`helpers/matmul.h`)
describes the product.

It takes `mmulMxV` and the `mmulMxM` products with one output row (M == 1; the
matrix is B) or one output column (N == 1; the matrix is A):

- On CUDA, when the matrix's and the vector's types differ
  (`mixedGemvApplies`) and no cuBLASLt epilogue is pending. A matrix and a
  vector of one type into an output of another stay on cuBLAS (HALF x HALF into
  FLOAT through `cublasGemmEx`) or on the compute-type path below.
- On CPU, for every float product whose types are not all one, since host BLAS
  takes none of them. `dot` follows the same rule.

CUDA has two kernels, chosen by the matrix's layout
(`MixedGemvLayout::depthMajor`):

- Depth-major W (a row's weights adjacent): one warp per row. The lanes stride
  over the row's depth and a warp shuffle adds their sums.
- Row-major W (neighbouring rows adjacent, as a 'c' [K, N] weight is for
  [1, K] x [K, N]): tiles of 32 rows, one row per lane, so a warp's loads are
  coalesced. The block's warps split the depth, and thread row 0 adds their
  partial sums in a fixed order.

Each thread loads eight products before it adds them (in its summation order),
so a decode-size product keeps enough loads in flight. The launches are named
`mixed_gemv_rows` (4096 blocks of 128 threads) and `mixed_gemv_columns` (1024
blocks of 512 threads) in `LaunchDims`, and can be overridden by environment
variable. Neither kernel allocates, synchronizes or reads launch-dependent
state, so both are capturable, and a given launch always sums in the same
order.

On CPU, a depth-major matrix gives one dot product per row; otherwise a thread
sweeps the depth across a block of 256 rows, reading W in storage order.

### Other mixed storage

Any other mix (integer operands, or one of the CPU paths above that the mixed
GEMV does not take) computes in `mixedGemmComputeType`: FLOAT32, DOUBLE when any
operand is DOUBLE, INT64 when all are integers. Both backends share it. No
operand is narrowed, and the result is assigned into the caller's output.

### Output type

`matmulOutputType` is the type of a matmul's output: the wider input type, the
first on a tie, a float input over an integer one. It is Java's
`Mmul.promoteMatmulOutputDataType`. The op's shape function and every output
MmulHelper allocates use it.

## Consequences

- A decode-step product reads the weights once in their storage type: no copy
  per call, and no FLOAT32 copies of HALF weights in the cast cache for these
  products. SmolDocling decode on CUDA (`run-benchmark.sh --ideal`, 250
  tokens): lateSteady 72.7 and 73.7 tok/s in two runs before, 76.0 tok/s after;
  decode time 12.3 s and 12.0 s before, 11.5 s after.
- Results differ in the last bits from the widened cuBLAS path: the products are
  the same (widening is exact) but the summation order is not.
- CPU products over mixed types work instead of throwing, and no CPU path
  narrows an operand.
- A natively inferred matmul output of FLOAT x BFLOAT16 is FLOAT, as Java's.

## References

- `libnd4j/include/ops/declarable/helpers/matmul.h` (`MixedGemvLayout`,
  `mixedGemvApplies`, `ProductAccumulator`, `mixedGemmComputeType`,
  `matmulOutputType`)
- `helpers/cuda/MmulHelper.cu` (`mixedGemvRowKernel`, `mixedGemvColumnKernel`),
  `helpers/cpu/MmulHelper.cpp` (`mixedGemv_`)
- `execution/cuda/LaunchDims` (`mixed_gemv_rows`, `mixed_gemv_columns`)
- `ops/declarable/generic/blas/matmul.cpp` (shape function)
- `platform-tests`: `MixedDtypeGemvTest`, `MmulMixedPrecisionRegressionTest`
