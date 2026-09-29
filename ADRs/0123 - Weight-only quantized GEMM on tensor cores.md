# ADR 0123: Weight-only quantized GEMM on tensor cores

## Status

Accepted, step 1 in progress. ModelOpt NVFP4 with BF16/FP16 activations is
implemented (`helpers/cuda/WeightOnlyGemm.cu`); the FP32-activation (3xTF32)
policy and the GGML/AWQ/GPTQ formats are not yet implemented.

Measured on GB10, sm_121, 2026-09-27, kernel time under nsys:

- 17408x5120: 313 us (160 GB/s).
- 5120x17408: 344 us (146 GB/s).
- The previous scalar kernel reached 49-78 GB/s.
- The standalone probe of the same mainloop reaches 230 us (below target: >= 200 GB/s).
- Qwen3.6-27B-NVFP4, 250 tokens: greedy 1.31 -> 1.55 tok/s.
- MTP multi-row commit: 2.26 -> 2.52 tok/s.
- Default-commit parity: 0 emission deltas.

## Context

Several operators multiply activations by block-quantized weights, and every one
of them runs on CUDA cores or dequantizes to a dense copy first:

| Format | Operator / helper | Block along K | Current CUDA path |
|---|---|---|---|
| ModelOpt NVFP4 (E2M1 + E4M3 block scale + FP32 global) | `modelopt_nvfp4_linear` | 16 | scalar tiled kernel |
| GGML Q8_0 | `ggml_qmatmul` | 32 | scalar kernel |
| GGML Q4_K / Q6_K | `ggml_qmatmul` | 256 super-block, 32/16 sub-blocks | scalar kernels |
| AWQ / GPTQ INT4 (group scale + zero) | `weight_dequant` | group size (typically 128) | dequantize, then dense GEMM |

On Qwen3.6-27B-NVFP4 the NVFP4 MLP projections now bound decode throughput
(73,406 calls per 250-token run, 0.65-0.92 ms each on the scalar kernel,
49-78 GB/s, against ~250 GB/s that a bandwidth-bound GEMM reaches on GB10).
The GGUF models (for example Qwen3.5-0.8B Q4_K_M) run the scalar
`ggml_qmatmul` kernels.

Library coverage on sm_121 (checked against this build): cuBLASLt's block-scaled
FP4 GEMM is W4A4 only, which changes these models' arithmetic (ADR 0122 rules it
out for NVFP4, and GGML/AWQ are weight-only by definition). Bundled CUTLASS 3.7
has mixed-input GEMMs only for SM90 (TMA/wgmma), which sm_121 lacks. It does
provide the SM80 warp-level primitives that sm_121 executes:
`cutlass::arch::Mma<GemmShape<16,8,16>, ...>` (`mma.sync.m16n8k16`, BF16/FP16
inputs, FP32 accumulation), `cp_async` and `ldsm`.

All four formats share one shape: low-bit values packed along K in blocks whose
size is a multiple of 16 (the MMA's K), each block with its own scale and
optionally a min/zero point.

## Decision

Add one weight-only quantized GEMM primitive, `z[rows, N] = x[rows, K] .
dequant(W)[N, K]^T`, built from a format-independent tensor-core mainloop and two
policies. Operators keep their names, signatures and native fallbacks and call
the primitive through a helper API (alongside `MmulHelper::ltMatmulScaled`).

### Mainloop (format-independent)

- `mma.m16n8k16.row.col` through `cutlass::arch::Mma`: A = activation tile
  (16 rows x 16 K), B = dequantized weight tile (16 K x 8 output columns).
  Rows are padded to 16; decode (<= 16 rows) is one M tile, prefill iterates.
- Packed weights and block metadata stream from the operator's own storage
  layout through a multi-stage `cp_async` pipeline into shared memory; no
  repacked weight copy, no dense intermediate. Activations for a CTA's K range
  are staged once per CTA and shared by its warps (`ldsm`).
- FP32 accumulation. A CTA owns a fixed block of output columns and the whole
  K; its warps split K into fixed ranges and reduce partial tiles through shared
  memory in a fixed order (see below for the few-wave cross-CTA split). Tile shape, K
  ranges and reduction order are compile-time constants independent of the row
  count: results are bit-reproducible under capture/replay and a row's result
  never depends on how many rows the call carries.
- Few-wave shapes split K across blocks. When a shape's column groups fill
  fewer than 4 waves of resident blocks and K leaves every split warp at least
  two chunks, `kSplitBlocks = 4` blocks share a column group. Each block reduces
  its warps as above and stores an FP32 partial tile in persistent per-device
  scratch. The group's last block to arrive (an atomic ticket, reset by that
  block) adds the partials in ascending block order. There is no extra launch
  and no per-call allocation. The scratch grows only outside stream capture.
  - The split depends only on the shape and the device's SM count, so row
    invariance and replay determinism hold.
  - On GB10 this covers only the Qwen3.6-27B down projection (N=5120,
    K=17408): 640 column groups against 192 resident blocks, whose partial
    last wave cost about 19% of that kernel.
  - Measured back-to-back at 250 tokens, greedy went from 10.66 to 10.89 tok/s
    (+2.2%). The earlier separate-launch split-K with per-call scratch was ~2%
    slower. `SD_WEIGHT_ONLY_SPLIT_BLOCKS` (1, 2 or 4; platform-tests
    `-Dnd4j.weightOnly.splitBlocks`) overrides the split; 1 restores the
    unsplit order.

### Weight-format policy

A format supplies, as compile-time policy: the block length along K (a multiple
of 16), how to address a block's packed values and metadata for output column n
and block b, and `dequantize(...)`, which produces the MMA-dtype values of a B
fragment. Each format's dequantization reproduces its existing kernel's
arithmetic exactly, so dequantized weights are bit-identical to today's:

- NVFP4 (ADR 0122): FP32 block x global scale, FP32 E2M1 x scale, round to the
  activation dtype. One MMA K-step is exactly one scale block.
- GGML Q8_0 / Q4_K / Q6_K: the ggml block formulas as implemented by the current
  `ggml_qmatmul` kernels.
- AWQ / GPTQ INT4: (q - zero) * scale per group, as in `weight_dequant`.

### Activation-precision policy

- BF16 and FP16 activations enter the MMA unchanged; with weights rounded to the
  same dtype, every product is exact in FP32.
- FP32 activations use a split-precision MMA (3xTF32: hi*hi + hi*lo + lo*hi, the
  scheme CUTLASS calls fast-accurate FP32). Products are FP32-accurate rather
  than bit-exact FP32 (the lo*lo term is dropped); the extra MMAs are free in a
  bandwidth-bound GEMM. Operators whose contract states exact FP32 products are
  amended to "FP32-accurate products, FP32 accumulation" (ADR 0122).

### Admission

Per format and activation dtype: dense row-major operands (the stride proof
already used by the ModelOpt tiled path), K a whole number of blocks, 16-byte
aligned activation, weight and output bases. Admission depends only on shapes,
dtypes and layout, never on runtime state, so a given linear never alternates
between paths. Everything else keeps the existing native kernels. Launch
configuration is registered in `LaunchDims.h`/`.cu` and validated before launch.

## Rollout

1. Mainloop + activation policies + NVFP4 format; `modelopt_nvfp4_linear` uses it.
2. GGML Q8_0 / Q4_K / Q6_K in `ggml_qmatmul`.
3. AWQ / GPTQ INT4, replacing dequantize-then-dense.

Each step lands with its format's tests and measurements before the next.

## Alternatives rejected

- W4A4 through cuBLASLt/CUTLASS block-scaled FP4: changes model arithmetic.
- Dequantizing weights to BF16 at load plus cuBLAS: ~4x resident weight memory
  and ~4x bytes read per token for a bandwidth-bound workload.
- One hand-written kernel per format: duplicates the pipeline, determinism and
  row-invariance machinery that each format would need anyway.
- CUTLASS SM90 mixed-input collectives: not executable on sm_121. A CUTLASS
  upgrade for SM120 collectives adds W4A4, not weight-only GEMM.

## Validation

Per format, in `platform-tests`:

- Dequantization parity: dequantized weights bit-identical to the existing
  kernel's.
- GEMM against an exact double reference with an FP32 accumulation-error bound;
  BF16, FP16 and FP32 activations; rows 1/5/16/17/128; K spanning several CTA
  K-ranges.
- Row invariance at real model shapes (`testRowResultsIndependentOfRowCount`).
- Non-admitted layouts still produce the native kernels' results.
- Microbenchmark at real shapes: >= 200 GB/s for decode (NVFP4 17408x5120 and
  5120x17408, rows 1 and 5; scalar kernel today: 49-78 GB/s).

Model level: the 27B NVFP4 gate (default commit, 250 tokens, zero emission
deltas, throughput reported separately from load) for step 1; the GGUF MTP
tests (`TestQwen35MtpDecode`) for step 2; the required DSP regression gate and
a CPU build for every step.

## Consequences

Dequantized weights stay bit-identical in every format; outputs change only
through FP32 accumulation order (and, for FP32 activations, split-precision
products). The scalar kernels remain as the general view-safe paths.
