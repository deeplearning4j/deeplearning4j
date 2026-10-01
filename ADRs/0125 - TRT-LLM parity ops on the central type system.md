# ADR 0125: TRT-LLM parity ops on the central type system

## Status

Accepted (2026-10-01). Extends ADR 0122 (FP8 and NVFP4 storage types) and
ADR 0123 (the format policies of the weight-only GEMM) to the TRT-LLM parity
ops.

## Context

The Java API has seven ops that mirror TensorRT-LLM kernels: `fp8_matmul`,
`smooth_quant`, `awq_matmul`, `gpu_top_k_sample`, `gpu_top_p_sample`,
`decoder_masked_mha` and `fused_norm_quantize`. Their wrappers had drifted from
the native library:

- Six of the wrappers named native ops that did not exist. `smooth_quant`
  existed, but its contract differed from the wrapper's.
- `TRTLLMFeatureParityOpsTest` checked for CUDA by comparing a backend name
  that never matched, so every test was skipped. Once the check read
  `Nd4j.getExecutioner().type()`, all seven tests errored.
- The wrappers passed formats as integer codes: `fp8Format` 0/1 on
  `Fp8Matmul`, and `QUANT_INT8`/`QUANT_FP8` on `FusedNormQuantize`. The FP8
  test fed INT8 arrays in place of FP8 ones.
- The generated `SDNN`/`NDNN` `fusedNormQuantize` threw away the per-row
  scales, so its codes could not be dequantized.

## Decision

### The data type carries the format

No op takes a format code.

- An FP8 operand is a `FLOAT8` (E4M3) or `FLOAT8_E5M2` array, and the storage
  type selects the format.
- A quantized output gets its type from the graph. `fused_norm_quantize` takes
  it from its data type argument (INT8 by default). `smooth_quant` quantizes
  activations onto the grid of the weight codes' type.
- The grid bound is `DataTypeUtils::max(Q)`: 127 for INT8, 448 for E4M3,
  57344 for E5M2. No op hardcodes a bound.

Type admission uses the central lists. `DECLARE_TYPES` uses `ALL_FLOATS`,
`ALL_INTS`, or `DataType::ANY` backed by an `isR`/`isZ`/`isU` check in the op.
Compute dispatches over `SD_FLOAT_TYPES` and accumulates in
`simdOps::AggregateType<T>`. No op or helper keeps its own type list.

The selectors do not dispatch FP8. FP8 enters and leaves the compute type
through the central `NDArray::assign`: widening is exact, and narrowing from
float rounds to nearest even (a double narrows through float).

`assign` runs as `execTransformAny`, whose `SD_COMMON_TYPES` matrix leaves FP8
out. Conversions into and out of FP8 go through `TransformAnyFp8<F8>`
(`loops/transform_any_fp8.h`):

- The FP8 side is fixed per encoding. `TransformAny<T, F8>` converts into it,
  and `TransformAny<F8, T>` converts out of it.
- T, the only free parameter, is dispatched by the central single-type
  selectors. It ranges over `SD_COMMON_TYPES`; T is also an FP8 encoding when
  converting from one encoding to the other.
- The CUDA pairs are split along with the arithmetic matrix
  (`transform_any_{i}.cu`), so no single translation unit instantiates them
  all.

### Shared helpers, not per-op kernels

**`helpers::scaledGemm`** (`helpers/int8_gemm.h`, `helpers/impl/scaled_gemm.cpp`)
- Computes C = (op(A) @ op(B)) · scaleA · scaleB + bias.
- A and B keep their own storage types. Each converts into the aggregate type
  of C, with a copy only when the type or layout differs, and the backend GEMM
  multiplies them there.
- `fp8_matmul`, `smooth_quant`, `awq_matmul` and the projections of
  `decoder_masked_mha` all use it.

**`helpers::symmetricRowScale` and `helpers::symmetricQuantize`**
(`helpers/symmetric_quant.h`, with CPU and CUDA implementations)
- Quantize symmetrically onto the grid of any signed integer or floating type.
- The grids are policies:
  - `SymmetricIntegerGrid` rounds to nearest even and maps NaN to 0.
  - `SymmetricFloatingGrid` clamps and leaves the rounding to the storage
    conversion.
- The bound must be exact in the accumulator; otherwise the helper throws. A
  zero scale quantizes to zero.

**The samplers** draw through the existing `tokenSampleDraw` helper. The top-p
penalties apply `applyLogitPenalties` to a copy of the sampled position, so the
logits stay read-only.

### Op contracts

| Op | Inputs | Outputs | Arguments |
|---|---|---|---|
| `fp8_matmul` | A [M,K], B [K,N] (any floating storage, FP8 included), scaleA (1 or [M]), scaleB (1 or [N]), bias [N] optional | C [M,N], in the data type argument, else scaleA's type | iArgs transposeA, transposeB |
| `smooth_quant` | X [...,K], W codes [N,K] (signed integer or floating), s [K], actScale (1 or [K]), weightScale (1 or [N]), bias optional; or X and s only | Y [...,N] in X's type; X / s with two inputs | iArg transposeWeight |
| `awq_matmul` | X [...,K], packed codes [ceil(K·bits/8), N], scales and zeros [ceil(K/group), N] (zeros optional, default 2^(bits−1)), bias optional | Y [...,N] in X's type | iArgs groupSize, numBits (1, 2, 4, 8) |
| `fused_norm_quantize` | X [...,F], gamma [F], beta [F] optional | codes [...,F] in the quantized type, scales [...] in X's type | iArg normType (RMSNorm, LayerNorm), tArg epsilon, dArg quantized type |
| `gpu_top_k_sample` | logits [vocab], [B,vocab] or [B,S,vocab] (last position), uniforms [B] optional | INT64 tokens [B], probabilities [B] under the kept distribution | iArgs k, seed; tArg temperature |
| `gpu_top_p_sample` | logits as above, uniforms optional, token history optional | as above | iArg seed; tArgs p, temperature, repetition, frequency and presence penalties |
| `decoder_masked_mha` | hidden [B,S,H], fused QKV weight, output weight, past K/V [B,kvHeads,past,d]; or Q, K, V | output, present K and V in input 0's type | iArgs heads, kvHeads (0 = from the shapes), headDim, useRoPE or causal, ropeBase; tArg mask filter value |

Every op returns `EMPTY_EXECUTE`: it runs on empty inputs rather than being
skipped. Two rules follow:

- An empty optional input stands for an absent one, so the Java builders can
  keep input positions when an optional input sits between required ones.
- Empty extents produce empty or zero-filled outputs, never unwritten ones.

Traits are declared on each op with `addTraits()`:

- The GEMMs declare `MATMUL | EXTERNAL_WORKSPACE | FULLY_WRITING`.
- `fused_norm_quantize` declares `NORMALIZATION | EXTERNAL_WORKSPACE | FULLY_WRITING`.
- `decoder_masked_mha` declares `ATTENTION | EXTERNAL_WORKSPACE | FULLY_WRITING`.
- The samplers declare `FULLY_WRITING | STATEFUL`. Without a positive seed they
  draw fresh entropy.

### Java API

- `FusedNormQuantize` takes a `DataType` for the codes, and `QUANT_INT8` and
  `QUANT_FP8` are removed. The generated `fusedNormQuantize` returns both codes
  and scales (`SDVariable[]` / `INDArray[]`).
- `Fp8Matmul` no longer has format arguments. The output type is an optional
  data type argument.
- `causalConv1dWithPrefix` and `gatedDeltaRuleWithPrefix` now come from the
  op-codegen DSL (`NeuralNetwork.kt`) instead of hand edits to generated
  files. Their op classes reject a null `actualSequenceLength`.

## Consequences

- These ops are compositions: convert, backend GEMM, scale. They are not fused
  kernels.
  - `fp8_matmul` widens FP8 into the aggregate type instead of feeding FP8
    tensor cores. A cuBLASLt FP8 path, or FP8 in the weight-only GEMM
    (ADR 0123), is the route to FP8 throughput.
  - `scaledGemm` copies operands whose type or layout differs from the
    output's.
- `fp8_quantize` (`headers/nn.h`) is older than this ADR and still selects its
  format with an integer code. It should move to the data type the same way.
- Callers of the removed `FusedNormQuantize` and `Fp8Matmul` signatures must
  pass data types. The repository has no other callers.

## Validation

`TRTLLMFeatureParityOpsTest` (CUDA) checks each op against a closed form:

- `fp8_matmul` on real `FLOAT8` operands equals K · scaleA · scaleB everywhere.
- `smooth_quant` with unit scales equals the row sums of the ties-to-even
  rounded activations.
- `awq_matmul` on 0x01-packed codes sums the even input channels.
- The top-k draw ranks among the k most likely tokens. The top-p draw lies in
  the nucleus.
- `decoder_masked_mha` keeps the past cache as the prefix of the present cache.
- `fused_norm_quantize` codes reconstruct the RMS-normalized rows to within
  half a step, and each row's largest magnitude takes the code 127.

`DecoderMaskedMhaTest`, `TokenSampleParityTest`,
`OpTraitTableComprehensiveTest` and the MTP prefix-binding tests cover the
remaining contracts.
