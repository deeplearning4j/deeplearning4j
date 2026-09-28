# ADR 0122: ModelOpt Qwen packed inference

## Status

Implementation in progress; native, full-model and MTP validation pending.

## Context

NVIDIA Qwen3.6-27B-NVFP4 revision
`0893e1606ff3d5f97a441f405d5fc541a6bdf404` is a mixed-precision SafeTensors
checkpoint, not a GGUF file or uniformly W4A4 model. Its configuration declares
W4A16_NVFP4 MLP/output projections, statically scaled FP8 attention projections,
BF16 auxiliary weights and a dense MTP predictor. GB10 reports compute capability
12.1; an SM86/PTX build is not evidence of native Blackwell FP4 execution.

## Decision

- Keep packed U8 E2M1 weights, E4M3 block scales and FP32 global scales distinct.
  Decode even K elements from low nibbles and odd elements from high nibbles.
- `modelopt_nvfp4_linear(X,W,blockScale,globalScale)` implements W4A16, not W4A4.
  Form FP32 block/global scale products, multiply decoded E2M1 values, round
  weights to activation dtype, and accumulate products in FP32. No whole-weight
  dense intermediate or GGUF requantization.
- `modelopt_fp8_linear(X,W,weightScale,inputScale)` implements static saturated
  E4M3 activation quantization and scaled FP32 products. FP8 and NVFP4 policies
  must not be conflated. Both ops have one integer argument: output FLOAT when
  1, otherwise activation dtype when 0. Activations support FLOAT/HALF/BFLOAT16.
- Supply CPU reference and CUDA implementations, stride-aware non-aliasing
  outputs, op-local traits and ordinary DSP native execution/capture admission.
  Dense matmul fusion requires the dense operand ABI, not merely a MATMUL trait.
- Reuse the Qwen hybrid SameDiff graph through explicit HF configuration and
  tensor adapters. Preserve explicit head dimensions, grouped GDN heads,
  one-centered norms, HF RoPE layout, independent predictor caches and shared
  packed LM head. Import text and the bundled MTP predictor; vision is outside
  this API's scope.
- Read generation_config.json separately: primary EOS and all stop IDs are
  generation metadata, not interchangeable with text_config training EOS.
- Import large tensors with bounded raw-byte staging and long offsets;
  low-precision storage must not undergo accidental numerical casts.

## Dense projection arithmetic: SERIAL_FMA

The connected-prefix W=1/W=5 probe demonstrated different FLOAT GEMM results
for identical logical operands immediately around a BF16 midpoint. Deterministic
cuBLAS replay is not a shape-invariant dot-product contract. Correct rounding of
the real dot product is not required; an identical recurrence is.

`matmul` reserves optional `iArgs[3]` (after transX/transY/transZ) for arithmetic:
absent or 0 is the unchanged legacy implementation; 1 is `SERIAL_FMA`. It is op
semantics, not an environment flag, caller DOUBLE conversion or execution mode.
The initial native implementation accepts matching HALF/BFLOAT16/FLOAT/DOUBLE
input/output storage. Its accumulator is FLOAT except for DOUBLE storage. Each
output starts at positive zero and applies `fma(x[k], y[k], sum)` in ascending K
order, including k=0. Then alpha is converted to accumulator precision and its
product with sum rounded; if beta is nonzero, one `fma(beta, oldOutput, result)`
follows. Beta zero never reads old output. Storage conversion occurs once. CPU
requires round-to-nearest arithmetic; CUDA uses explicit round-to-nearest FMA
(the FLOAT intrinsic preserves subnormals even under FTZ compilation). NaN payload
identity across architectures is not promised. CPU thread FP environments must
retain IEEE gradual underflow; no host/global FP mode is changed by this op.

The helper shares coordinate/stride projection and scalar recurrence across CPU
and CUDA. Native shape inference remains `ShapeUtils::evalShapeForMatmul`, so its
existing rank/broadcast restrictions remain: vector/vector, vector/matrix,
matrix/vector, equal-rank batches, and ND/2D or 2D/ND. Matrix transpose and output
transpose use the normal op flags; strided views do not require flattening or
materialization. Output/input DataBuffer aliasing is rejected. Existing empty
shape semantics are retained. CUDA uses the named `matmul_serial_fma` launch,
context stream, primary/special-use lifecycle, no workspace and no host fallback.

Java `Mmul` exposes constructor-selected `Arithmetic` without a policy setter or
shadow state. The argument participates in Java equality/hash, FlatBuffers
`extraInteger` serialization/restoration, native slot argument copies and DSP
segment keys (which already hash the argument count and every value). The dense
HALF/BFLOAT16 inference path in `QuantizedLinear` requests the contract, retaining
its existing FLOAT operand conversions and output cast. Packed GGML and ModelOpt
operators are not changed. Differentiating the rounded recurrence via ordinary
`matmul_bp` is explicitly rejected by the Java wrapper.

Triton now has a source-level SERIAL_FMA recipe in its per-element matmul emitter,
used by standalone and sectioned module entry points instead of `tt.dot`. It
starts at +0, carries one accumulator through ascending K with target-neutral
`math.fma`, and expresses the separately rounded alpha product as
`math.fma(alpha, sum, -0)` before the optional beta FMA. No TF32 truncation or
associative reduction is used. Beta zero emits no old-output load. The final
storage conversion precedes any consumer; serial matmuls do not absorb epilogues.
Existing FusionPass guards remain in force. Ordinary matmul is unchanged.

Performance of the recipe keeps the recurrence fixed. Parallelism is the output
count (one output per lane), and each lane's K loop is latency-bound on its
loads. The emitter therefore:

- software-pipelines the K loop: it carries the next 64 steps' raw operands,
  loading chunk i + 1 while chunk i's ascending FMAs run. A deeper buffer spills
  registers.
- loads a K-contiguous operand with 8-byte rows as one 64-bit word per lane
  per 4 BF16/HALF (2 FLOAT) steps. Each step's element is extracted by
  shift/truncate/bitcast, i.e. exactly the stored value. The extraction is
  lane-local, so no layout conversion is needed. An earlier `[BLOCK, 8]` tile
  form round-tripped through shared memory and spilled.
- records that alignment on the kernel argument (`requiredAlignment`). Every
  later binding, direct or consolidated, fails closed if it is violated.
- for decode shapes, targets about four single-warp programs per SM, with at
  least 32 outputs per program so every lane of the warp is used.

Weight layout decides the achievable bandwidth. With lanes over N, a weight
stored [N, K] and read transposed (K-contiguous) makes every lane stream a
different row: about 110 GB/s cold on GB10. Stored [K, N] c-order, one K step
of a warp is one contiguous segment: about 220 GB/s. On CUDA,
`SerialMatmulLayoutOptimizations` stores a constant consumed only through
`permute(W, 1, 0)`/`transpose(W)` as B of untransposed SERIAL_FMA matmuls as
[K, N] c-order, replacing the original.
- Values and each output's K order are unchanged, so results are bit-identical.
- Memory does not grow.
- CPU SERIAL_FMA walks K per output and keeps [N, K].

On Qwen3.6-27B (GB10):
- The `in_proj_a`/`in_proj_b` pair went from 4.3 ms to 0.09 ms per layer.
  Ranges for independent sections are also concurrent (ADR 0061).
- One MTP draft pass of dense BF16 SERIAL matmuls went from about 7.3 ms to
  4.3 ms.
- At 250 tokens, greedy decode is 9.5 tok/s and multi-row MTP is 13.8 tok/s.

With a cold L2, the pair's 5120-step chain was bounded by DRAM latency: with one
register-resident chunk in flight per lane, every 64-step chunk waited a full
DRAM round trip (a hand-written CUDA kernel with that pipeline also took about
108 µs).

#### Pipelined K loop for narrow projections

That bound belonged to the pipeline, not the recurrence. A CUDA probe that
stages each lane's K-contiguous row through a multi-stage `cp.async` ring
(16-byte copies, 64-step stages, 4 stages) hides the DRAM latency entirely
(cold equals warm) and runs the 96-output pair in about 20 µs, bit-identical to
the serial reference.

The compiled recipe now expresses the same thing:
- SERIAL_FMA's chunked K loop issues its loads inside the loop body and carries
  `tt.num_stages = 4`. The TTGIR pipeline runs `AssignLatencies` (module default
  1 stage, so only loops that request stages are pipelined) and `ScheduleLoops`
  before `Pipeline`, which turns the per-lane wide loads (≥4 bytes) into
  asynchronous copies into a shared-memory ring.
- A 16-byte-aligned K-contiguous operand loads two words per lane as one
  `[block, 2]` tile split back into consecutive words; the alignment is declared
  to Triton as `tt.divisibility` on the splatted base pointer (indirect-table
  pointers are `int_to_ptr` results the axis analysis cannot see through) and is
  enforced on every binding through `requiredAlignment`. Without Triton's
  Coalesce pass (skipped for modules above 128 ops) the tile's default layout
  still splits each lane's pair across two threads, so the copies are 8 bytes.
- The accumulation order is untouched: bit-identity tests cover K spanning many
  chunks and a scalar tail, 1 and 5 rows, partly masked programs, and [N,K] view,
  [N,K] transpose-B and [K,N] storage.

The per-lane path needs K-contiguous storage, while the coalesced [K,N] form is
better once a matmul is bandwidth-bound. `QuantizedLinear` therefore builds a
dense SERIAL_FMA projection with at most 1024 outputs as `matmul(x, W,
transposeB)` over the original [N,K] weight (no permute for constant folding to
materialize as [K,N]); wider projections keep the permuted view and the [K,N]
storage above. Measured on Qwen3.6-27B (per call, 4/16-program kernels vs 80+):

| Kernel | [K,N] coalesced | [N,K] pipelined |
|---|---|---|
| GDN in_proj_a/b pair, W=1 | 98.7 µs | 39.5 µs |
| same pair, W=5 | 99.5 µs | 40.6 µs |
| draft MLP/attention matmuls (80-320 programs) | 0.54-1.97 ms | 0.92-3.35 ms |

Greedy width-1 decode went from 10.36 to 10.70 tok/s and MTP from about 23.3 to
23.6 tok/s at 250 tokens, with identical greedy tokens and 0/250 MTP deltas.

The initial compiled admission domain is NVIDIA, matching HALF/BFLOAT16/FLOAT/
DOUBLE storage, nonempty rank>=2 matrices, equal-rank batches and ND/2D or 2D/ND,
positive-stride inputs (including F-order/offset/stepped views), and dense logical
C-order output. All three transpose flags and alpha/beta are implemented. Vectors,
empty tensors, mixed storage, output aliasing, other output layouts, unavailable
live shape metadata, unequal ND batch ranks, mismatched batch dimensions, and
indices/spans above INT_MAX-16384 are explicitly rejected before cache lookup
and by every module entry point. Structural mappability alone is not concrete
admission. Explicit op exclusions reject admission; once admitted, serial matmul
is compiled even where legacy matmul would be a native ordered range. A failed
serial leaf compilation reports failure, never native substitution.

Source inspection of the pinned Triton lowering maps `math.fma` to LLVM FMA;
the NVPTX target uses RN. Serial-containing kernel entries explicitly carry
LLVM `denormal-fp-math` and `denormal-fp-math-f32` passthrough attributes set to
`ieee,ieee`; Triton's kernel FuncOp conversion retains these. The emitted
operations have no fastmath/reassociation flags.
The alpha product retains its own rounding point even when ordinary FP fusion
is enabled. Device tests must still inspect generated PTX for RN/non-FTZ FMA and
verify FLOAT subnormals/signed zero against native; this source inspection is not
a runtime parity claim. AMD/Intel are not admitted or claimed validated. DSP
shape keys already hash every iArg (including the fourth) and the bit patterns of
tArgs on normal and symbolic paths before Triton's in-memory lookup. Triton's
shared compile/execute lookup hash additionally includes serial arguments and
all three concrete dtype/shape/stride signatures, so symbolic ranges or reused
frozen keys cannot reuse a differently shaped serial loop. The disk cache hashes
emitted TTIR, so the serial recipe has separate cache identity.

The optional oneDNN/ACL/Accelerate/MLIR eager helpers and compiled
oneDNN/OpenVINO/ACL/NNAPI/MLX/MLIR/ARM-hybrid/Hexagon/HIP/StableHLO paths reject the
flag. Vulkan's exact argument gate rejects nonzero fourth arguments. NVRTC/PTX
already reject matmul by category. Native cuBLAS batching and Lt epilogue/cast-sink
fusion reject the explicit contract.

### Source-phase acceptance blocker

Java graph rewrite guards require scope expansion beyond `Mmul` and
`QuantizedLinear`: `HorizontalFusionOptimizations` replaces matched matmuls with
legacy `sd.mmul`; `LinearFusionOptimizations` emits `xw_plus_b`;
`NormalizationFusionOptimizations` emits `rms_norm_linear`;
`AttentionFusionOptimizations` extracts/replaces matmuls without checking the
fourth argument; `AlgebraicOptimizations.ScalarIntoWeightFolding` reassociates the
scale into the weight. These sites must reject SERIAL_FMA before mutation (or
implement an exactly equivalent replacement). No optimizer-disable workaround
was added. Constant-chain folding already rejects every nonzero iArg, and the
transpose absorber copies the entire iArg array. Quantization passes that rewrite
operands also require a deliberate arithmetic-policy audit. Until these guards
are consolidated, the implementation is NOT optimizer-proof or accepted.

This batch is source-only. No build, test, device parity, capture/replay, or
performance validation has run. The parent owns the consolidated SM121 build,
focused window-parity test and wider DSP regression gates.

## FP8 linears on tensor cores (cuBLASLt)

`modelopt_fp8_linear` runs, when its shape admits it, as one scaled cuBLASLt
FP8 GEMM: the activation is quantized once per call (the same saturated E4M3
quantizer, `modelOptFp8Quantize`) into a dense scratch operand, and
`MmulHelper::ltMatmulScaled` computes
`(inputScale * weightScale) * E4M3(X) . E4M3(W)^T` with FP32 accumulation. Every
product is the same exact FP32 product as before; the FP32 accumulation order is
the library's. Admission is shape-only (K and N * sizeof(output) multiples of 16
bytes, dense row-major W and Z, 16-byte aligned bases); anything else, and
FLOAT32 activations never, stay on the native kernels.

Algorithm selection is a pure function of the problem: no split-K reduction, no
workspace, and no dependence on capture mode or workspace availability, so
results are bit-reproducible under graph capture and replay. Calls of up to 16
rows share the algorithm selected for 16 rows, so a row's result does not depend
on the call's width (the W=1/W=5 concern above); `TestModelOptLinear#
testRowResultsIndependentOfRowCount` pins this at the 27B shapes. The native
tiled kernel's bit-exact lane contract stays pinned for the FP8 shapes that do
not meet the alignment rules.

Measured on GB10 (Qwen3.6-27B shapes, 5 rows): 3.2-5.4 ms per call on the
original native kernel, 0.18-0.28 ms on cuBLASLt. The 27B losslessness gate
(default single-row commit) passes with zero emission deltas; multi-row commit
now does too (next section). NVFP4 W4A16 on tensor cores is ADR 0123.

## Multi-row commit: window rows and lifecycle phases must agree

Multi-row speculative commit emits the verification window's rows directly, so
two properties must hold bit for bit:

1. **Row invariance.** Row r of a W-row target forward equals the r-th of W
   chained width-1 forwards from the same state.
2. **Lifecycle parity.** A plan's native slot-by-slot warmup (its first
   executions) and its compiled replay give the same bits. The two decode modes
   reach those phases at different steps, so any warmup/replay difference
   becomes a greedy-vs-MTP difference.

The earlier 123 emission deltas (first at token 126) were entirely lifecycle
parity. Row invariance already held in steady replay.

- **Attention.** Triton decoded with a tiled online softmax. Native's grouped-query
  kernel (`fusedGQAAttentionWithScores4DKernel`) uses ascending-d FMA logits, scale
  then bias, max, `__expf`, a 256-thread strided sum with shuffle-down trees,
  `rcp.rn` normalization and an ascending-key FMA P·V. For decode windows (≤ 8
  rows, grouped heads, FLOAT/HALF/BFLOAT16, current K/V in the query dtype),
  `emitGgufDecodeAttentionKernel` now emits that exact recipe per query row
  (`emitNativeOrderedDecodeAttention`); decode tiles use one query row per program.
  - Each pairwise step of the sum tree is a single two-element addition, so the
    bits don't depend on Triton's thread mapping.
  - Keys past the window are masked by the decode bias: they contribute an exact
    +0 and are skipped, so the kernel stays at the flash kernel's speed (57–75 µs
    per call vs 110–220 µs native).
  - Other attention contracts keep the flash kernel and are not yet exact.
- **RMSNorm.** The replicated reduction order was right, but the mean used
  `arith.divf`, which Triton lowers to approximate FP32 division. It now uses the
  precise division, like native `div.rn`.

Guards:
- `DspDecodeRowInvarianceTest`: every decode op, both properties, 27B shapes.
- `TestQwenNvfp4Import#windowRowsMatchChainedScalarTarget`: both properties over
  every layer of the real model.
- The `EMIT_LOGITS` VERIFY diagnostic fingerprints each emitted token's logits,
  so two runs can be compared for the first numeric (not just argmax) difference.

At 250 tokens on GB10, multi-row commit gives 0/250 deltas at 14.4 tok/s
(acceptance 0.56); greedy is 9.5 tok/s and single-row commit 4.0 tok/s. On
Qwen3.5-0.8B (`run-benchmark-mtp.sh`), multi-row is also token-exact: 22.3 tok/s
at acceptance 0.41, versus 9.3 tok/s single-row. Multi-row commit is therefore
the default. `SD_MTP_MULTI_ROW_COMMIT=0` (`-Dnd4j.mtp.multiRowCommit=0` in
platform-tests) selects single-row commit.

## Accepted-prefix commit by checkpoint selection

After a partial acceptance, the linear-attention layers have already advanced
their recurrent state through the rejected draft rows. Multi-row commit used to
repair this by re-running the target over the accepted prefix, a second full
forward (~103 ms of a ~228 ms step on the 27B). The GDN and conv ops instead
export per-row state checkpoints (`gated_delta_rule_with_prefix`,
`causal_conv1d_with_prefix`), and the commit copies checkpoint `consumed - 1`
into the live state: about 150 MB of device copies instead of a forward pass.

- **Default.** `nd4j.mtp.prefixSelect` is `auto` by default: selection is used
  whenever every recurrent layer exports a checkpoint, and is off for models
  without recurrent layers or graphs built without the outputs. `select` requires
  it (an incomplete binding is a preparation error), and `off` keeps the rerun.
  Graph builders (`ModelArchitecture`, `ModelOptQwenConfig`) export the
  checkpoints unless the mode is `off`.
- **Bounds.** The trailer holds up to 64 layers per kind (GDN and conv
  separately); the 27B has 48 of each.
- **No rerun snapshots.** When selection is admitted for the whole window before
  verification, the pre-verification state and KV-row snapshots (read only by
  the rerun) are skipped. A commit-time rejection after that fails closed rather
  than rerunning without a snapshot.
- **Admission cost.** The cross-layer overlap check is a sorted sweep over the
  checkpoint and state byte ranges (O(n log n)); the earlier pairwise loop cost
  18 ms per partial-accept step at 96 layers.

Together with removing the host-blocking stream synchronize after DSP input
staging (consumers are ordered by events), the 27B reaches 22.5 tok/s at 250
tokens (acceptance 0.56, 0/250 emission deltas vs greedy 9.6 tok/s).

## Optional NVFP4 MTP predictor

ModelOpt exports leave the MTP predictor dense (BF16). Its SERIAL_FMA projections
read about 850 MB per draft pass. `ModelOptQwenImporter.importTextOnly(...,
quantizeMtpNvfp4)` converts the predictor's projections at import with
`ModelOptNvfp4Quantizer`, a bit-exact port of ModelOpt's `nvfp4_tensor.py`
recipe:

- **Scales:** a global scale `amax / 2688`, and E4M3 block scales of
  `amax_block / (6 * global)` (1 for an all-zero block).
- **Elements:** E2M1, round to nearest with ties to even, packed with the even
  column in the low nibble.
- **Execution:** the projections run as `modelopt_nvfp4_linear` (W4A16 through
  WeightOnlyGemm).

Only draft proposals, and so acceptance, can change; verification and emitted
tokens are the target's. On the 250-token gate:

| Predictor | Per-step time | Acceptance | MTP throughput |
|---|---|---|---|
| Dense (default) | baseline | 0.56 | 14.47 tok/s |
| NVFP4 | ~6 ms faster | 0.53 | 14.52 tok/s |

The two cancel, so the option stays off by default.

Enabling it exposed a serializer defect that any small one-byte-float constant
would hit: SDNB `dup()` dropped inline FLOAT8/FLOAT8_E5M2 arrays and scalars, and
widened BFLOAT16/HALF scalars to FLOAT. Inline loading now restores those types
exactly (`LargeSameDiffSerializationTest#testInlineLowPrecisionConstantsSurviveDup`).

## Consequences and validation

The CUDA NVFP4 implementations are packed FMA kernels, NOT FP4 Tensor Core
kernels (see ADR 0123); FP8 runs on tensor cores through cuBLASLt as described
above. No complete model support is claimed by their existence. The reference checkpoint is W4A16; changing its
activation quantizer to obtain W4A4 acceleration is not a compatible optimization.
A dedicated optimized implementation must preserve the same numerical contract.

Isolated tests live in platform-tests/TestModelOptLinear and cover layouts,
raw encodings, output types, serialization, scale validation and FP8 subnormal
rounding. TestQwenNvfp4Import has separate opt-ins for storage import, generation
and native predictor MTP. MTP validation requires positive proposals/acceptance
and token equality to greedy on identical prompts, not n-gram substitution.
The DSP regression gate and model-level validation must pass before acceptance.

Sources: pinned NVIDIA config, hf_quant_config and generation_config; NVIDIA
Model-Optimizer 0.45.0 nvfp4_tensor.py (including block-scale range clamping and
its PR1397 reference); Transformers/vLLM Qwen3.5 architecture and MTP definitions.
Full FP8 KV-cache export semantics are not yet implemented by this importer;
current integration uses the existing floating cache ABI and must be reported
as such rather than presented as equivalent exported FP8-cache execution.

## Steady-state plan ABI: all-preset source coverage

`executeSteadyStatePlan(Pointer, OpaqueContext, Pointer)` is a native instance
method, not a default Java delegate to ordinary execution. Presets explicitly
objectify this method and emit `@Override`; `NativeOps` retains its throwing
unsupported default for genuinely absent implementations. The native wrapper
selects `NativeDynamicShapePlan::executeSteadyState` and shares the ordinary
entry's input order, requested-output mapping, borrowed-output publication,
stream conversion/completion, and error propagation. Lifecycle admission remains
inside the plan; an early call is not evidence that replay has been reached.

| Preset | Native owner / exposure | Coverage boundary |
|---|---|---|
| Nd4jCpuPresets | `legacy/cpu/NativeOps_dsp.cpp` | Native steady-state dispatch; CPU and CPU graph helpers |
| Nd4jCudaPresets | `legacy/cuda/NativeOps_dsp.cu` | Native steady-state dispatch; same CUDA stream-storage conversion and output completion as ordinary entry; ZLUDA uses this source, not NVIDIA validation |
| Nd4jMinimalPresets | Inherits CPU mapping via `super.map`; links `nd4jcpu` | Same native owner, not a duplicated implementation |
| Nd4jTpuPresets | `MainBuildFlow.cmake` CPU-derived legacy collection; `BuildTPU.cmake` configures `nd4jtpu` | Real shared plan entry; selected TPU graph runtime remains responsible for capability/execution |
| Nd4jVulkanPresets | `legacy/vulkan/NativeOps_dsp_plan.cpp` | Real steady-state selection under `VulkanExecutionStreamGuard`, same stream synchronization and non-owning output publication; prior skip removed |
| Nd4jMetalPresets | MPS/MLX CPU-derived profile, `nd4jcpu` | Corrected nonexistent `nd4j_metal` link name; helper is now a NativeOps base with a subclass-accessible constructor; full macOS binding generation/device validation still required |
| Nd4jHexagonPresets | CPU-derived legacy collection, `nd4jhexagon` | Shared native entry; supplied missing NativeOps helper and precise context/pointer mappings; full Hexagon binding generation/device validation still required |
| SdxRuntimePresets | `legacy/impl/DspRuntimeC.cpp`, `dsp/runtime/dsp_runtime_c.h` | Separate C ABI: additive `sdxRunSteadyState` and `sdxRunSteadyStateAllocating`, automatically parsed from its existing header; never expose NativeOps pointers through this transport |
| LiteRtLmPresets | LiteRT-LM `c/engine.h` | Not a NativeOps/NativeDynamicShapePlan runtime; no foreign ABI injected |
| TokenizersPresets | Rust wrapper `tokenizers_c.h` | Tokenization only; no plan/runtime to expose |

No new per-profile native copies or CMake admission were necessary for the
CPU-derived artifacts: the existing source collection already compiles the
same CPU native wrapper. This says nothing about support for a particular op,
dtype or accelerator graph: normal backend admission and errors are unchanged.
The preexisting Metal availability probe remains a stub; it is not runtime
support evidence.

SDX preserves its own context mutex, tensor validation, public-to-plan input
mapping, cached backend stream, strict-backend policy, execution report and
caller-buffer copy/borrowed-output lifetime contract. Existing `sdxRun` and
`sdxRunAllocating` remain ordinary execution; the new explicit entries select
steady-state execution through the same `runInternal` owner. No ABI-v1 struct
layout, existing behavior or generation call path is changed. Standalone SDX
reuses the central object library (`BuildSDX.cmake`); Linux `sdx*` and Apple
`_sdx*` exports include the additive entries automatically. Windows export-list
generation scans `SDX_API` declarations in the public header. Staged SDK headers
must be refreshed by the parent build, not edited in place.

`SteadyStatePlanApiCoverageTest` enumerates all seven NativeOps presets and their
native owners from checkout sources, including minimizer inheritance, plus SDX
and the two unrelated transports. It separately checks **every installed**
generated binding with `Class.forName(..., false, ...)` and `getDeclaredMethod`:
the exact method must be public, native, nonstatic and declared on the binding,
not inherited from the unsupported interface default. Missing artifacts are
not treated as runtime proof; their source rows are still mandatory. This test
does not initialize native libraries or require unavailable GPUs.

Validation for this batch: **no builds, regeneration, tests or hardware runs**,
as requested. Parent consolidation must regenerate available bindings and run
this API coverage class plus the targeted steady-state lifecycle tests with an
explicit `-Dbackend.artifactId`. All other backend regeneration, native linking,
execution parity and performance remain unvalidated; a pure-Java preset compile
alone does not establish those properties.
