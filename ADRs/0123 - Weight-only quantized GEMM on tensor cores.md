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

2026-09-30, after the few-wave split and bulk-copy staging (sm_90+). Kernel
medians at 1 row come from a standalone probe that rotates 12 weight sets
through L2:
- 17408x5120: 240 us direct, 214 us staged (235 GB/s).
- 5120x17408: 241 us direct, 226 us staged (222 GB/s).
- Greedy, back-to-back at 250 tokens: 10.92 tok/s direct, 11.52 tok/s staged,
  with identical tokens (after the split combine moved to release/acquire).

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
- Packed weights and block metadata are read in the operator's own storage
  layout: no repacked weight copy, no dense intermediate. A and B fragments
  apply the same permutation of the MMA's K positions, which leaves every
  product unchanged and only fixes the accumulation order. With it, a lane
  reads its 32 packed weights of a 128-element chunk with one coalesced 16-byte
  load, and its activations with one 8-byte load per row and MMA step. The
  direct kernel loads weights from global memory into registers one chunk
  ahead of the MMAs. On sm_90+ the bulk kernel stages them in shared memory
  instead (see Bulk-copy staging). Each warp reads its activations from global
  memory.
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
  - The ticket orders the partials with release/acquire, as a grid barrier
    does. After the block's closing barrier, thread 0 takes the ticket with a
    release RMW (`atom.release.gpu`). Only the last arrival issues an acquire
    fence (`fence.acq_rel.gpu`), before its threads read the scratch through
    L2.
  - A `__threadfence()` in every thread instead compiles on sm_121 to a
    sequentially consistent fence plus an L1 invalidation per warp
    (`MEMBAR.SC.GPU` ... `CCTL.IVALL`). Four split units per column group
    kept invalidating the L1 that holds the resident blocks' activation
    rows.
  - The effect showed only in the model: the bulk down projection ran at
    226 µs alone but 238-242 µs between the layer's FP8 GEMMs. A standalone
    probe reproduced it with the FP8 GEMMs' reads filling L2 first: 249 µs
    with the per-thread fence, 230 µs with release/acquire, with
    bit-identical results. Without the preceding reads: 230 and 224 µs.
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

### Bulk-copy staging (sm_90+)

A second kernel runs the same work units, K ranges, reduction and split
combine, but stages each unit's weights in shared memory before its MMAs.

**Staging.**
- Thread 0 arms an mbarrier with the tile's byte count. It then issues the
  unit's `cp.async.bulk` copies from global to shared memory with an L2
  evict-first policy, since the weights are read once. The copies cover the
  unit's 8 packed rows over its K range and their block-scale rows.
- Every thread waits on the barrier's phase once per unit. Lanes then read the
  same 16-byte packed words and 2-byte scale pairs from the tile that the
  direct kernel reads from global memory.
- A block issues its first copies before preparing its constants.
- Each later unit is staged after the previous unit's last barrier. A
  `fence.proxy.async` orders the tile's generic-proxy reads before the
  async-proxy writes.

**Layout.**
- Packed rows 2p and 2p+1 lie 64 bytes apart modulo 128. The two 64-byte
  segments a quarter-warp reads therefore fall in distinct banks.
- A scale row (K/16 bytes) may start only 2-byte aligned, so it is copied as
  its 16-byte-aligned superset.
- Scale rows are an odd multiple of 16 bytes apart, so the 8 rows a warp reads
  at once fall in distinct banks.
- A K = 5120 tile is 23,232 bytes.

**Admission and equivalence.**
- The bulk kernel needs 16-byte-aligned block scales, because bulk copies need
  aligned sources. It also needs a kernel image built for PTX 9.0 or later.
  Otherwise the direct kernel runs.
- Both kernels accumulate in the same order, so their results are
  bit-identical. The choice may therefore depend on the row count without
  affecting row invariance or replay.

**Selection.** Measured on GB10 (48 SMs, 128 KB of L1 and shared memory per
SM). The bulk kernel runs only when all four conditions hold:

| Condition | Reason | Bulk against direct |
|---|---|---|
| A unit spans >= 4096 of K | A unit waits for its copy once; short ranges do not amortize the wait | N=17408, 1 row: K=2048 and 3072 1-2% slower, 4096 4% faster, 5120 11% faster |
| >= 4 waves of units (kSplitWaveLimit x resident blocks) | One block's copy overlaps other blocks' MMAs only while every SM keeps cycling units | K=5120, 1 row: 512 units 11% slower, 640 5% slower, 768 2% faster, 2176 11% faster |
| rows x the unit's K range x activation bytes <= 64 KB | The tile's shared memory is carved out of L1, which must still hold the activation slice | N=17408, K=5120: 6 rows (60 KB) 1% faster, 7 rows (70 KB) 13% slower, 16 rows 49% slower |
| Tile fits two resident blocks per SM | Staging pays only while a second block computes during the first one's copies | Unsplit K=12288 and 17408 (one block per SM) 5-7% slower; K=9216 (two per SM) 14% faster |

At 8 rows, the direct kernel alone, forced to the same carve-out, runs 33%
slower. That supports L1 as the cause of the third condition.

**Qwen3.6-27B-NVFP4.** At decode (1 and 5 rows), every NVFP4 projection
stages. Kernel medians for 1 row, direct against bulk:

| Projection | Shape | Direct | Bulk |
|---|---|---|---|
| gate, up | K=5120, 2176 units | 240 µs (209 GB/s) | 214 µs (235 GB/s) |
| down | K=17408 split across 4 blocks, 2560 units of 4352 | 241 µs (208 GB/s) | 226 µs (222 GB/s) |
| lm_head | 31,040 units | 3290 µs (217 GB/s) | 2944 µs (243 GB/s) |

Prefill takes the direct kernel.

Measured back-to-back at 250 tokens, alternating the two kernels twice (see
Override), greedy went from 10.87 and 10.97 to 11.53 and 11.51 tok/s. That is
about 4.8 ms per token (+5.7%), with the same tokens in all four runs. The
kernel medians above predict about 4.7 ms per token (64 layers plus the
lm_head). Before the split combine's ticket used release/acquire (see
Mainloop), the same A/B gave only +2.8% (11.02 and 11.06 against 11.34 and
11.35): in the model, the bulk down projection took 242 µs instead of 226 µs.
With release/acquire it takes 222 µs (nsys, 64 layers x 20 tokens), and the
weight-only kernels take 1.17 ms less per token. Gate, up and every other
kernel are unchanged.

**Override.** `SD_WEIGHT_ONLY_STAGING` (platform-tests
`-Dnd4j.weightOnly.staging`) replaces the selection rule:
- `direct` never stages.
- `bulk` stages every shape whose operands and tile the device admits.
- Empty keeps the rule.

It changes timing only. It serves to A/B the kernels in one thermal state and
to re-measure the rule on another device.

**Open.** At K = 32 (mod 64), every other packed row starts 16 bytes into a
32-byte sector, so a lane group's 64-byte load touches three sectors instead
of two. At K=5152 with only 512 units, the bulk kernel was 14% faster
(68 µs against 79 µs), probably for that reason; the cause is not isolated.
Aligning the direct kernel's loads is a possible follow-up.

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
- Bulk against direct kernel, bit for bit
  (`testNvfp4BulkStagingMatchesDirectKernel`). The test reads which kernel
  launched from the BACKEND diagnostics. Its shapes sit on both sides of each
  selection condition. They include several units per block, partial chunks,
  unaligned scale rows, uneven splits, and block scales that are only 2-byte
  aligned, which must take the direct kernel.
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
