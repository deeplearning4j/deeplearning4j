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

## Consequences and validation

Current CUDA implementations are packed FMA kernels, NOT FP4 Tensor Core
kernels. No Tensor Core acceleration, throughput or complete model support is
claimed by their existence. The reference checkpoint is W4A16; changing its
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
