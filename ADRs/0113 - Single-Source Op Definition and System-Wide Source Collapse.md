# ADR: Single-Source Op Definition and System-Wide Source Collapse

## Status
Proposed

## Date
2026-07-13

## Context

libnd4j's op/kernel source multiplies along a grid: **926 ops (214 legacy
functors + 712 declarable) × up to ~7 authored artifacts each × up to 15
execution engines**. The result is ~280k LOC of kernel-ish code with large,
measured, mechanical duplication (full per-backend and per-op census in the
**Measured coverage** section below):

- **~209 op helpers implemented twice** (`helpers/cpu/*.cpp` ~45k LOC +
  `helpers/cuda/*.cu` ~65k LOC; 8 of the 209 stem pairs are allocator/runtime
  infra, ~201 are op-level). ~25% of the paired content is byte-identical
  scalar math retyped on both sides; ~35% of the CUDA side is launch/stream
  scaffolding; ~20% is duplicated iteration boilerplate; only ~12% is a
  genuinely divergent GPU algorithm.
- **~85k LOC across 9 platform-override dirs** (mkldnn, cudnn, mps, accelerate,
  armcompute, mlir, pjrt, vlm, miopen — llamacpp was removed this session, see
  ADR-0112). These hold **952 `(op, engine)` wrapper pairs over 353 distinct
  op names**; the llamacpp audit's **80–97%-boilerplate** finding generalizes —
  the surviving dirs share the same 3-zone wrapper skeleton (extract → validate
  → one accelerated call) behind per-dir utils files, with **~25.5k LOC of
  wrapper boilerplate collapsible while every kernel stays**.
- **Op-body boilerplate**: 2,292 `REQUIRE_TRUE` validation walls; 560
  `DECLARE_SHAPE_FN` (154 trivial passthrough/scalar), 741 `DECLARE_TYPES`
  blocks; an identical ~30–60-line weights-broadcast + reduction `switch`
  copy-pasted verbatim across 10 loss ops (~900 LOC); ~25-line axis-parsing
  shape-fn preambles across ~18 reduce ops.
- **Loop engine / instantiation** duplication (`broadcasting_bool` ↔ `_int`,
  `reduce_bool` ↔ `_long` ↔ `_same` ↔ `_float`) and the `.cu.in` /
  `BUILD_*_SELECTOR` type-instantiation machinery.

### What already works and MUST be preserved

This ADR builds on existing strengths, not a rewrite:

1. **Backend selection is already `#ifdef`-free in op bodies.** The generic op
   is one backend-agnostic file that calls `helpers::opFn(...)`; the backend is
   chosen at **link time** by which of `helpers/cpu/*.cpp` (g++) or
   `helpers/cuda/*.cu` (nvcc) is compiled. No `#ifdef SD_CUDA` in op code.
2. **Multi-backend runtime dispatch already exists.**
   `DECLARE_PLATFORM`/`PLATFORM_IMPL`/`PLATFORM_CHECK` register a
   `(op, engine)` helper; `KernelDispatchHelper::dispatchWithAutoTune`
   (`DeclarableOp.cpp:1058`) collects all usable helpers across **all** engines
   and picks one, falling back to the generic implementation when none is
   usable. This is the multi-dispatch backbone.
3. **Single-source functors already work for ~214 legacy ops** (327 macro
   entries; names shared across SCALAR/PAIRWISE/BROADCAST). `ops.h` +
   `op_macros_*.h` define each elementwise/reduce op **once** as an
   `SD_HOST_DEVICE` functor with no per-op backend file and no `#ifdef`; the
   `loops/cpu` and `loops/cuda` engines consume every functor. This is the
   existing proof that "define once, generate the engines" works at scale.
4. **op-codegen already single-sources the Java SD/ND namespaces** from one
   authored op body.

## Measured coverage

The collapse scope was measured across both axes of the grid (all file:line
evidence in the campaign notes). Two views: **backends** (the 9 platform-
override dirs + the CPU/CUDA generic pairs) and **ops** (the 926-op catalog
binned by *shape of computation*).

### Backend coverage — 9 platform-override dirs

Every dir follows the same 3-zone wrapper skeleton (input/output extract →
`REQUIRE_TRUE` + `PLATFORM_CHECK` requirements → **one** accelerated call into
its per-dir utils/kernel file). The wrapper is collapsible; the kernel is not
touched.

| Dir | Files | LOC | Wrappers | Engine | Wrapper LOC collapsible | `PLATFORM_CHECK` table-driveable |
|---|---:|---:|---:|---|---:|---|
| mkldnn | 72 | 19.9k | 137 | ENGINE_CPU | ~4.1k | yes (dtype-set + rank≤6 + offset==0) |
| cudnn | 34 | 12.5k | 86 | ENGINE_CUDA | ~2.6k | yes (RAII descriptors already factored) |
| mps | 20 | 11.6k | 261 | ENGINE_CPU→MPS | ~5.2k | yes (single dtype + contiguous) |
| accelerate | 28 | 9.7k | 129 | ENGINE_CPU | ~5.2k | mostly (BNNS conv descriptors → utils) |
| armcompute | 129 | 15.0k | 127 | ENGINE_CPU | ~3.2k | yes (`ArmFunction<T>` already abstracts) |
| mlir | 19 | 6.5k | 141 | ENGINE_CPU-JIT | ~2.8k | **yes — ideal pilot** (IMPL already 3 lines) |
| pjrt | 10 | 5.2k | 41 | ENGINE_TPU | ~1.6k | partial (per-op HLO build stays) |
| vlm | 12 | 3.9k | 16 | CPU + CUDA | ~0.5k | yes |
| miopen | 5 | 1.5k | 14 | ENGINE_ZLUDA_AMD | ~0.3k | **yes — easiest** (identical dtype block ×14) |
| **Total** | **329** | **~85.8k** | **952** | 9 engines / 353 distinct ops | **~25.5k** | — |

**Cross-backend overlap (the highest-value `DEFINE_PLATFORM_OP` targets).** 15
ops are accelerated by **6+ backends** — each is the *same wrapper written 6–7
times*: `conv2d`, `softmax`, `batchnorm`, `relu`, `sigmoid`, `tanh` (7 each);
`maxpool2d`, `avgpool2d`, `layer_norm`, `reduce_sum`, `reduce_mean`,
`reduce_max`, `log_softmax`, `relu6`, `elu`, `conv2d_bp` (6 each). Collapsing
these first turns ~100 wrapper bodies into 15 op-specs × N one-line override
entries, sharing one `Requirements` table per op across all its backends —
kernels untouched.

**CPU/CUDA generic pairs (209 stems) binned by computation shape:**

| Family | Pairs | Collapse vehicle | Examples |
|---|---:|---|---|
| INFRA (not op-level) | 8 | — (allocator/runtime) | ConstantHelper, MmulHelper, cublasHelper |
| MAP (elementwise+params) | 26 | `DECLARE_HELPER_MAP` | activations, addBias, clip, dropout, 9 updaters |
| INDEX_REMAP (`z[c]=x[f(c)]`) | 20 | `INDEX_REMAP` engine | roll, reverse, tile, one_hot, diag, s_t_b, gather |
| TAD_REDUCE (per-TAD) | 13 | `TAD_REDUCE` on L0 block-reduce | segment, histogram, ismax, rms_norm, percentile |
| SCATTER_ACC (scatter-add) | 10 | `SCATTER_ACC` engine | scatter, confusion, dynamic, merge, prefix |
| COMPOSABLE (closed-form/compose) | 18 | demote to `impl/` | zeta, polyGamma, betaInc, range, cross, image_resize |
| STAYS HAND-WRITTEN | 114 | override only | 26 sparse, 45+ LLM/attn, 21 conv/pool, lup/svd, sorts |

The first six families (~87 pairs) collapse onto ~4 shared generic engines;
the 114 STAYS pairs keep their `.cu`/utils kernels and register as overrides.

### Op coverage — 926-op catalog

Of the 712 **declarable** ops (the 214 legacy functors are already collapsed),
binned by whether the op is single-source-collapsible or genuinely stays a
hand-written per-backend kernel:

| Bin | Ops | % | Collapsible? |
|---|---:|---:|---|
| A ALREADY-SINGLE-SOURCE (broadcastable/list/logic) | 56 | 7.9% | yes (done) |
| B LEGACY-FUNCTOR-ELIGIBLE (activations+bp, updaters) | 56 | 7.9% | yes (math already a functor) |
| C INDEX_REMAP (reshape/gather/pad/space-transforms) | 78 | 11.0% | yes (coord-map) |
| D TAD_REDUCE (segment/reduce+bp/arg/top_k) | 57 | 8.0% | yes (TAD pattern) |
| E SCATTER_ACC (dense scatter yes; sparse suite no) | 58 | 8.1% | partial |
| F NORM/ATTENTION/FUSED/LOSS | 116 | 16.3% | CPU side + shells share; GPU stays |
| G LINALG-EXTERNAL (lu/svd/qr/cholesky/eig) | 11 | 1.5% | no (LAPACK/cuSolver) |
| H CONV/POOL (+bp, im2col, upsample) | 43 | 6.0% | no (tiled kernels) |
| I RECURRENT/SCAN (lstm/gru/sru/ssm/rwkv/gla) | 31 | 4.4% | no (scan deps) |
| J SIGNAL/AUDIO/IMAGE | 40 | 5.6% | mixed (~60% remap) |
| F4 helper-complex residual (biasadd, matmul_bp, nlp…) | 177 | 24.9% | ~120 composable / 57 complex |
| **Total** | **712** | | **~51% collapsible / ~49% stays** |

Two sub-catalogs sharpen the target:

- **`_bp` gradients: 182 ops (26% of the catalog), 67% composable.** 122 are
  already written as NDArray reduce-sum/broadcast composition with no
  platform-specific code (`add_bp`, `reduce_mean_bp`, `layer_norm_bp`,
  `softmax_bp`, `matmul_bp`, `concat_bp`, …); 25 are scatter-add-shaped; only
  35 are genuinely custom (conv/pool/rnn/hard-attention BP). The 122 are a
  near-free Layer-1/2 collapse.
- **Validation boilerplate confirms Layer-1 scale.** 2,292 `REQUIRE_TRUE`
  (332 sparse / 327 nn / 289 recurrent / 210 transforms / 198 conv / 173
  loss); the loss family averages 9.6 checks/op and repeats one 30-line
  weights-broadcast + reduction block verbatim across 10 ops; 154 of 560
  shape-fns are trivial passthrough/scalar.

**Bottom line: ~362 of 712 declarable ops (≈51%) collapse to a single-source
spec; ~350 keep a hand-written kernel that becomes an *override*, not a
deletion.** The 214 legacy functors are already there. No backend and no
kernel is removed — the collapse is of the *authoring surface*, not the
capability set.

### Substrate, loop-engine, and within-backend coverage (Layers 0, 3-intra, 4)

The op/backend tables above cover the *op* and *wrapper* axes. Three more pools
complete the grid — the shared CUDA substrate under every kernel (Layer 0), the
setup scaffolding *inside* the workers (Layer 3-intra, complementing the
cross-backend overlap), and the loop engines (Layer 4).

**Layer 0 — CUDA substrate (no `.cuh` device header exists today; ~650
duplication sites).** Six pools, split additive vs consistency-refactor:

| Pool | Sites | Fix |
|---|---|---|
| Warp/block reduce | 5 named re-defs of `warpReduceSum`/`blockReduceSum` (layer_norm, fused_llm_ops, activations, segment_softmax, rms_norm) + 18 raw-`__shfl` inline files (24 files total) | **additive** `warp_reduce.cuh` |
| Host-wrapper dispatch | ~501 `prepareSpecialUse`→`BUILD_*_SELECTOR`→`registerSpecialUse`→`synchronize` sites / 188 files | **additive** `SD_CUDA_DISPATCH` |
| SMEM shape-cache preamble | ~84 macro-able single-tensor thread-0 cache sites (42 loops/cuda files) | **additive** `SD_SMEM_SHAPE_CACHE` |
| TAD pack idiom | 44 CUDA files (`tadForDimensions`→`numberOfTads`/`primaryOffsets`) | **additive** `makeTadPacks()` |
| Atomics bypass | 8 files call raw `atomicAdd`/`atomicCAS` (sparse_blas.cu:113 re-implements the double-CAS `sd_atomicCAS<double>` already provides — a latent bug) | **consistency** |
| Formula SSOT | sigmoid/SiLU retyped in 10 files; tanh-GELU constants re-declared in 11; erf-GELU `1/√2` in 2; RMS-norm `rsqrt(sumSq+eps)` in 9 | **consistency** (`math/gelu_constants.h`, `sd_silu`, `sd_invRms`) |

**Layer 3-intra — within-backend clone families (~11k LOC, complements the
cross-backend wrapper pool; kernels untouched).** Inside a single dir, sibling
ops share descriptor/setup scaffolding collapsible to a per-dir family worker:

| Cluster | Files | Collapsible LOC | Shared scaffold |
|---|---:|---:|---|
| mkldnn eltwise/activation (**biggest**) | 16+5 | ~2,400 | one `eltwise_forward` body ×16; only the `algorithm::eltwise_*` enum differs (`eltwise_math.cpp` already proves the worker) |
| armcompute stamps (activation/binary/compare/reduce) | 37 | ~2,945 | `ArmFunction<T>` + one layer-type enum per op |
| mkldnn conv | 8 | ~1,300 | oneDNN memory-desc + primitive-desc + reorder |
| cudnn conv | 5 | ~1,200 | TensorDesc/FilterDesc/ConvDesc + algo-search + workspace |
| updater family (cpu+cuda) | 18 | ~1,745 | identical shape-cache + threads loop; only the update formula differs → functor-stamp |
| cudnn RNN | 3 | ~700 | RNN/dropout descriptor + v6/v8 dispatch; `copyRNNWeights(numGates)` |
| mps activation / accelerate BNNS | ~11 | ~675 | MPSImage lifecycle / BNNS descriptor build |

~98 files, **~11k LOC** of intra-worker scaffolding on top of the ~25.5k wrapper
LOC — the two Layer-3 pools are complementary, not overlapping.

**Layer 4 — loop engines (highest blast radius, ABI-constrained, done LAST).**
The type-variant engines are near-identical: `broadcasting`/`scalar`/`pairwise`
`_bool` vs `_int` iteration skeletons are **95–98% byte-identical** (only the
`Z` type, an `extraParams` arg, and the `DISPATCH` arity suffix differ); the
four `reduce_{float,same,bool,long}` engines are ~70% shared. **But** each
engine is hardwired to a disjoint op-number list in `legacy_ops.h`, and those
opNums are **external ABI** (embedded in serialized SameDiff graphs / ndarray
files). A merge is safe *only* if it preserves every opNum and adds a type-tag
branch before the `switch` — renumbering is a silent data-corruption bug. The
instantiation machinery is two systems: `split_heavy_op` (13 ops × 13 type
slices = 169 generated TUs) and `comb_compilation_units` (25 `.cu.in`/`.cpp.in`
templates → 300–500 TUs, selective-rendering-filtered). `NativeOps_dsp.cpp`↔`.cu`
(~4.3k LOC) is the **JNI export facade**, excluded (structural, not a merge
target).

## Prior art — how other frameworks solve this

Eight frameworks were surveyed for the same three questions this ADR faces:
(1) where an op is authored once, (2) how the different op kinds are expressed,
(3) how a generic impl and a backend override coexist.

| Framework | Authoring surface | Generic substrate | Override escape | Precedence |
|---|---|---|---|---|
| PyTorch | `native_functions.yaml` → torchgen | ufunc functor + `TensorIterator` (template lib) | `dispatch:` backend rows | runtime backend key > `CompositeImplicitAutograd` alias |
| TensorFlow | `REGISTER_OP` + `REGISTER_KERNEL_BUILDER` | Eigen functor, `Device` template param (template lib) | per-device kernel | `.Priority()` + `DEVICE_DEFAULT` fallback |
| ONNX Runtime | ONNX opset + per-EP kernel registry | hand CPU-EP kernels | EP `Compile()` / fusion | EP priority order, first-come; CPU EP last = fallback |
| ggml | central `ggml_op` enum + inline shape | per-backend switch | backend switch case | `supports_op()`; CPU `default: true` |
| Kokkos | one struct, `KOKKOS_INLINE_FUNCTION` | `parallel_for/reduce`, `ExecSpace` (template lib) | `ExecSpace` specialization → cuBLAS | compile-time specialization |
| CUTLASS | 5-layer templates + Python gen | template metaprogramming (template lib) | dispatch-policy tag | base `static_assert(false)`; only specializations |
| MLIR/IREE | `linalg.generic` (maps + body) | progressive dialect lowering (compiler IR) | `library_call` / `custom_call` | library pass before codegen |
| XLA | HLO graph | HLO per-backend lowering (compiler IR) | `custom_call` + FFI | library passes (cuBLAS/cuDNN) before codegen |

### Two conclusions the survey settles

**1. The precedence "fork" is not a fork — the field is unanimous.** Every
framework encodes *generic-is-fallback, a backend override shadows it for just
that backend*: PyTorch's `CompositeImplicitAutograd` alias key (runtime backend
keys outrank it), TF's `.Priority()` + `DEVICE_DEFAULT`, ORT's priority-ordered
EP partition with the CPU EP registered last, ggml's `supports_op()` with CPU
`default: return true`, and the zero-cost compile-time specializations of
Kokkos/CUTLASS/Eigen/NumPy-`ReplaceLoopBySignature`. libnd4j's
`KernelDispatchHelper` + `PLATFORM_CHECK` already implements exactly this (the
DSP plan cache hoists the check out of the hot path). The model needs no change
— only formalization.

**2. The substrate should be template-library, not compiler-IR.** All four
production runtimes (PyTorch, TF, ORT, ggml) and all four single-source-C++
libraries (Kokkos, CUTLASS, Eigen, NumPy) build the generic from a
template/functor + a registry. The compiler-IR camp (MLIR/IREE, XLA) authors in
an IR — but every one still *falls back to a hand/library kernel for anything
hot* via `library_call` / `custom_call` / BYOC; the IR fills gaps, the hand
kernel wins. Decisively, Kokkos/Eigen/NumPy show **libnd4j already is a
hand-rolled version of this**: `SD_HOST_DEVICE` ≡ `KOKKOS_INLINE_FUNCTION` /
`EIGEN_DEVICE_FUNC` (identical `__host__ __device__` expansion),
`BUILD_SINGLE_SELECTOR` ≡ the NumPy ufunc `funcs[]`/`types[]` table, `.cu.in` ≡
CUTLASS's ETI generator, `PLATFORM_IMPL`/`PLATFORM_CHECK` ≡ NumPy
`ReplaceLoopBySignature` / Kokkos `ExecSpace` specialization. The ADR's job is to
*formalize the machinery libnd4j already has*, keeping the MLIR JIT + Triton as
high-value **override lanes** (peers of cuBLAS/oneDNN), not the mandatory generic
fallback — the more so since MLIR-CUDA is CPU-only today.

### Four elements prior art adds that this ADR lacked

1. **A capability query + compile-time backend-partition (ORT + ggml).** ORT's
   `IKernelLookup::LookUpKernel(node)` and ggml's `supports_op(backend, tensor)
   →bool` answer "can backend B run op O at these dtypes/shapes?" *without
   attempting execution*, run as a graph pass assigning each node to the
   highest-priority capable backend up front. libnd4j has no such queryable
   predicate. High-fit move: add `bool supportsOp(op, descriptor)` per backend
   and do the assignment inside `DynamicShapePlanCompiler::compile()` — which
   already does section-fusion and cuBLAS-island detection — so selection is
   compile-time, not per-call. The single most valuable *new* structural piece,
   and it is additive.
2. **A structured reduction driver (Kokkos/Eigen/NumPy).** All express reductions
   as a split-join functor `(init, accumulate, join)` behind a driver. libnd4j
   hand-writes warp-shuffle reductions in `loops/cuda/reduction/reduce_*.chpp` —
   exactly where the `sPartials` Z-vs-InterType bug (HALF reduces garbage; see
   the BGE-fp16-NaN root) originated. A `(init, accumulate, join)` driver on
   Layer-0's `block_reduce` kills that bug family structurally and is the home
   for the `TAD_REDUCE` family.
3. **A backend-obligation manifest (PyTorch `RegistrationDeclarations.h`).** A
   generated machine-readable list of which ops a backend MUST implement vs
   inherits from the generic. libnd4j's 15-engine matrix has none; generating it
   from the op registry gives every backend author their exact surface.
4. **Override-by-symbol (MLIR `library_call` / XLA `custom_call`).** The formal
   "authored generic, overridden by a named hand kernel" pattern — exactly what
   `DEFINE_PLATFORM_OP(engine, name, worker)` expresses. Confirms the Layer-3
   macro design.

### Open decision (for sign-off)

The survey framed this as "template-library vs compiler-IR," but grounding it in
the tree shows that is a false dichotomy: **libnd4j already runs both and they
compose.** The template-library layer (`SD_HOST_DEVICE` functors + `loops/`
engines + `PLATFORM_IMPL`) is the eager + fallback path; a mature **trait-driven
compiler-IR layer** already fuses ops at the DSP segment level via three
multi-target emitters — `TritonIRBuilder` (GPU → PTX/AMDGCN/SPIR-V, with
category emitters `emitUnaryElementwise`/`emitBinaryElementwise`/
`emitComparisonOp`/`emitReductionOp`/`emitNormalizationOp` + `emitNativeCuda*`
primitives), `CpuIRBuilder` (CPU → MLIR → LLVM), and `MlxIRBuilder` (Metal → MLX)
— gated by `isTritonMappable(opName)` over `OpCategoryTable` and joined by an
op declaring its `OP_TRAIT_<category>`. The GPU compiler path is Triton and it is
mature; the earlier "MLIR is CPU-only" caveat mis-scoped it (MLIR is the *CPU*
lowering; Triton is the *GPU* one).

So the real decision is the **authoring stance** for the ~87 collapsible pairs,
not "which engine." The three concrete forms — template-library baseline,
compiler-IR-primary, and unify-the-authoring hybrid — are outlined end-to-end in
**Appendix A** with worked examples and the exact existing/new infra for each.
Recommendation: **hybrid-unify** (Appendix A.3) — one op-spec that emits both the
functor (eager/fallback, every dtype, standalone calls) and the trait+opcode
registration (fused GPU/CPU/Metal via the existing emitters), with the ORT-style
capability query in DSP compile choosing between them per op. This matches what
libnd4j already is and banks the mature emitter stack as the fast lane rather
than rebuilding it.

## Decision

Adopt a **"define once, generate the cross-product"** model and drive source
collapse at every layer of the grid, reusing the four mechanisms above rather
than replacing them.

An op is **authored as a single spec**: its computation (an `SD_HOST_DEVICE`
functor, or a composition of existing ops, or an explicit "requires a custom
kernel" marker per engine) plus its metadata (signature, shape rule, allowed
types, traits, requirements). From that one spec, macros/codegen emit the
declaration, the generic body, the generic engine binding, the type
instantiations, and the registration points for per-engine overrides.

### Non-goals (explicit, load-bearing)

- **No backend is removed.** This is a source-collapse, not a capability cut.
  Every one of the 15 engines remains a first-class override target.
- **No hand-optimized kernel is deleted.** CUDA (and cuDNN/oneDNN/MPS/…)
  kernels that beat the generic stay — they move from *mandatory per-op files*
  to *optional overrides* registered through the platform layer. The generic
  functor/engine is the fallback everyone inherits, not a replacement for a
  faster kernel.
- **Runtime dispatch semantics are unchanged.** `KernelDispatchHelper` still
  selects; the collapse only changes how the candidates are *authored*.
- **The type system is not touched** (per prior directive): no changes to type
  lists or selector type axes.

## The model — five collapse layers

**Layer 0 — shared substrate (mostly additive, unlocks the rest; ~650
duplication sites across 6 pools — see coverage table).** A device-primitive
header (`warp_reduce.cuh` — `warpReduceSum`/`Max`, `blockReduceSum`; no `.cuh`
device header exists today, so 5 named re-defs + 18 raw-`__shfl` files converge),
CUDA wiring macros (`SD_CUDA_DISPATCH` for the ~501-site PointersManager
sequence, `SD_SMEM_SHAPE_CACHE` for ~84 thread-0 preambles, `makeTadPacks()` for
44 TAD-idiom files), and validation macros (`REQUIRE_RANK_*`, `REQUIRE_SAME_SHAPE`,
`REQUIRE_SAME_DTYPE`, `RETURN_OK_IF_EMPTY`). Consistency pass: route the 8 files
that bypass `math::atomics::sd_atomic*` back through it (fixes the sparse_blas
double-CAS latent bug), and pull the retyped sigmoid/GELU/RMS-norm math into
shared inlines (`math/gelu_constants.h`, `sd_silu`, `sd_invRms`).

**Layer 1 — op declaration / shape / types / requirements.** Collapse the
2,292-site `REQUIRE_TRUE` walls, the 10-op loss reduction blocks (→
`loss_utils.h::applyLossReduction` + `VALIDATE_LOSS_INPUTS` + `LOSS_OUTPUT_SHAPE`),
the 154 trivial shape-fns (→ `SAME_SHAPE_AS_INPUT`/`CONSTANT_SCALAR_SHAPE` tags),
and the reduce shape-fn preambles (→ `DECLARE_STANDARD_REDUCE_SHAPE_FN`). The
182-op `_bp` catalog is a near-free adjunct here: 122 are already NDArray
composition. Extend op-codegen's "define once" down into the C++ declaration
surface.

**Layer 2 — generic implementation (the ~209 pairs; ~87 collapsible onto ~4
engines, 114 stay as overrides).** Author the op math once as
an `SD_HOST_DEVICE` functor (or a composition for composable ops); shared
**generic engines** consume it exactly as `loops/` consumes the 324 legacy
functors. Families keyed by *shape of computation*, not by op:
`DECLARE_HELPER_MAP` (elementwise+params), `INDEX_REMAP` (`z[coords]=x[f(coords)]`
— `roll`/`reverse`/`tile`/`one_hot`/`diag`/`triu`/`trace`/`invertPermutation`/…),
`TAD_REDUCE` (on Layer 0's block-reduce — the 7 segment files converge here),
`SCATTER_ACC` (`scatter`/`confusion`/`embedding_lookup_bp`). The generic engine
is the **fallback for all 15 engines**, not "the CPU+CUDA impl."

**Layer 3 — platform overrides (952 wrappers / ~85.8k LOC across 9 dirs;
~25.5k LOC collapsible — the largest and lowest-risk pool).** A per-engine
`DEFINE_PLATFORM_OP(engine, name, worker)` macro generalizes the mkldnn/cudnn
utils-house pattern so each override is one line + a worker, collapsing the
80–97% wrapper boilerplate **while keeping every optimized kernel**.
Requirements become declarative (`supportedDtypes`, `rankRange`, `contiguous`)
via `Requirements` additions (`expectRankIn`, `expectContiguous`), retiring the
hand-written `PLATFORM_CHECK` bodies. Pilot on the **15 ops shared by 6+
backends** (`conv2d`/`softmax`/`batchnorm`/activations/pooling/`reduce_*`) —
each is one op-spec + N one-line entries replacing 6–7 hand-copied wrappers,
and `mlir`/`miopen` are the cleanest first dirs (their `PLATFORM_CHECK` bodies
are already uniform dtype-set tables). A complementary **intra-worker** pass
(~11k LOC across ~98 files) lifts the shared descriptor/setup scaffolding inside
each dir into a family worker (`DEFINE_MKLDNN_ELTWISE`/`DEFINE_ARMCOMPUTE_*`,
`cudnnConvForward`, `copyRNNWeights(numGates)`, an updater functor-stamp) —
`eltwise_math.cpp` already proves the worker pattern.

**Layer 4 — loop engines / type instantiation (highest blast radius, last).**
Dedup the 95–98%-identical `broadcasting`/`scalar`/`pairwise` `_bool`↔`_int`
variants and the ~70%-shared `reduce_{float,same,bool,long}` engines, and the
`.cu.in` expansion (2 systems: `split_heavy_op` 13×13=169 TUs +
`comb_compilation_units` → 300–500 TUs). **Hard constraint:** `legacy_ops.h`
opNums are external ABI (serialized into SameDiff graphs / ndarray files) — a
merge must preserve every opNum and add a type-tag branch before the `switch`;
renumbering is a silent data-corruption bug. Shared infrastructure, done only
after Layers 0–3 prove out; `NativeOps_dsp` (JNI facade) is excluded.

### "One op file" defined

The single authored file is the **op spec**: the computation + metadata. The
declaration, generic body, per-engine bindings, type instantiations, and
dispatch wiring are generated or macro-stamped. A backend that wants to beat the
generic adds one `DEFINE_PLATFORM_OP` entry; it does not fork the op. The sole
`#ifdef __CUDACC__` lives inside the shared engine/macro layer, never in an op.

## What stays a hand-written per-backend kernel

Explicitly excluded from generic-engine collapse (~114 of the 209 CPU/CUDA stem
pairs, ~350 of the 712 declarable ops — the macro would only hide complexity and
cost correctness/perf): warp-shuffle Welford norms; cuSolver / cuFFT paths
(`lup`, `svd`, the 11 LINALG-EXTERNAL ops); sort-based ops (`nth_element`,
`top_k`); shared-memory-tiled convolutions (the 43 CONV/POOL ops); the 26 sparse
SpGEMM/SpMM kernels; atomics-based segment/scatter reduce where it diverges; the
45+ LLM/attention kernels; and the scan/recurrence kernels (`rwkv_wkv6/7`,
`ssm_scan`, `gated_linear_attn`, the 31 RECURRENT/SCAN ops). These remain
explicit `.cu` (and cuDNN/oneDNN/MPS/…) implementations, registered as overrides
— their *wrapper* still collapses (Layer 3), only the *kernel* stays.

## Phasing and gates

Sequenced lowest-risk / highest-yield first; each phase gated by the DSP
regression suite + a benchmark sweep + per-backend op tests, with LOC /
compile-time / binary-size deltas recorded before expanding:

1. **Layer 0 substrate** (additive; no op behavior change) — `warp_reduce.cuh` +
   the atomics/formula consistency pass first, since Layer 2's `TAD_REDUCE`
   depends on the shared block-reduce.
2. **Layer 3 platform-wrapper collapse** (biggest LOC, lowest algorithmic risk —
   boilerplate only, kernels untouched). Pilot `DEFINE_PLATFORM_OP` on `mlir` or
   `miopen` (cleanest checks) + the 15 six-backend ops; then the intra-worker
   family workers (`DEFINE_MKLDNN_ELTWISE` is the biggest single win at ~2.4k LOC).
3. **Layer 2 generic families** — pilot `one_hot` (INDEX_REMAP), one segment op
   (TAD_REDUCE), one MAP op; measure; then roll out.
4. **Layer 1 op-body / loss / reduce macros** (+ the 122 composable `_bp` ops).
5. **Layer 4 loop-engine / instantiation dedup** (last) — gated additionally on a
   **serialized-model round-trip test** (opNum ABI), not just op correctness.

## Risks

- **Hot headers**: validation macros land in `op_boilerplate.h`; Layer-0 device
  headers touch many CUDA TUs. Both force broad recompiles — batch edits, expect
  full rebuilds, never on a shared/contended build.
- **nvcc compile cost** if any collapse routes more TUs through nvcc; ccache
  mitigates but measure it.
- **Dispatch precedence across 15 engines** must stay well-defined when the
  generic becomes a real fallback for ops that previously had none.
- **Per-backend correctness**: every collapsed op must pass on **each** engine
  it targets, not just CPU+CUDA — the test matrix widens.
- **Layer-4 opNum ABI (hard blocker)**: `legacy_ops.h` op numbers are serialized
  into SameDiff graphs and ndarray files. Any engine-variant merge must preserve
  every opNum and add a type-tag branch before the dispatch `switch` —
  renumbering is silent data corruption. This is why Layer 4 is last and gated on
  a serialized-model round-trip test, not just op correctness.

## Consequences

- Estimated collapse (coverage preserved), by layer: Layer 2 ~20–30k LOC (~87 of
  209 stem pairs onto ~4 engines); Layer 3 ~25.5k wrapper LOC + ~11k intra-worker
  scaffolding (952 wrappers / ~98 clone-family files, kernels untouched); Layer 1
  ~1.9k LOC op-body/loss/shape-fn macros; Layer 0 ~650 duplication sites (mostly
  additive headers); Layer 4 gated on the opNum-ABI constraint. **Order ~60k+ LOC
  of authoring surface** removed while every kernel, backend, and op-number is
  preserved. Reaches ~362 of 712 declarable ops (≈51%); the 214 legacy functors
  are already collapsed.
- A bug in shared math is fixed once and propagates to every backend, instead
  of drifting between hand-copied `.cpp`/`.cu` pairs.
- New ops and new backends both get cheaper: a new op is one spec; a new backend
  inherits every generic op for free and overrides only what it accelerates.
- Ties into the AGENTS.md compose-first gate (a future ADR): new helpers must
  first try composition / a generic family before a hand kernel.

## Appendix A — the three substrates, concretely

The Prior-art survey named a "template-library vs compiler-IR" fork; the tree
shows **libnd4j already runs both, composed**. Two layers exist today:

- **Template-library layer** (eager + fallback, per-op): `SD_HOST_DEVICE`
  functors (`ops.h` 214 legacy + helpers), the `loops/cpu` + `loops/cuda`
  engines, `BUILD_*_SELECTOR` type dispatch, `PLATFORM_IMPL` overrides, and
  `KernelDispatchHelper` selection. Always correct, every dtype, works for
  standalone calls outside any graph.
- **Compiler-IR layer** (fused, per-DSP-segment): three trait-driven,
  multi-target emitters — `TritonIRBuilder` (GPU → TTIR → PTX/AMDGCN/SPIR-V;
  category emitters `emitUnaryElementwise`/`emitBinaryElementwise`/
  `emitComparisonOp`/`emitTernaryOp`/`emitReductionOp`/`emitNormalizationOp` +
  `emitNativeCuda{Exp,Log,Pow,Cos,Sin,Div,…}`), `CpuIRBuilder` (CPU → MLIR →
  LLVM JIT, via `MlirCpuGraphBackend`), and `MlxIRBuilder` (Metal → MLX). An op
  joins by declaring `OP_TRAIT_<category>` in `DECLARE_TYPES` and having an
  `OpCategoryTable` entry (`isTritonMappable`); no hand kernel for mapped
  categories. There is also the separate per-op MLIR eager JIT (`MLIREngine`,
  141 `DECLARE_PLATFORM(mlir)` ops, CPU).

The three "options" are therefore three **authoring stances**, not three engines.

### A.1 — Template-library baseline

- **Authoring (one point):** one `SD_HOST_DEVICE` functor in a shared header,
  assigned to a family (`MAP` / `INDEX_REMAP` / `TAD_REDUCE` / `SCATTER_ACC`), or
  a composition body for composables.
- **CPU + GPU:** the same functor is consumed by a shared `loops/cpu` and
  `loops/cuda` family engine — exactly how the 214 legacy functors already ride
  both. One source, compiled twice; no `#ifdef` in the body (the Kokkos/Eigen
  model, which `SD_HOST_DEVICE` ≡ `KOKKOS_INLINE_FUNCTION` already gives).
- **Other backends / overrides:** a faster kernel shadows via
  `PLATFORM_IMPL(engine, …)`; the functor is the fallback; `KernelDispatchHelper`
  selects. Unchanged.
- **Generated:** decl / shape-fn / types (Layer-1 macros) + type instantiations
  (`BUILD_*_SELECTOR`).
- **New infra:** the ~4 family engines + the stamp that emits decl+functor.
  **Exists:** `SD_HOST_DEVICE`, the loops pattern, `BUILD_*_SELECTOR`,
  `PLATFORM_IMPL`, `KernelDispatchHelper`.
- **Example — `one_hot` (INDEX_REMAP):** `z[coords] = (coords[axis] ==
  x[coords\axis]) ? on : off` authored once; rides the shared `INDEX_REMAP`
  engine on CPU+CUDA.
- **Risk / maintenance:** lowest. But each family engine's parallel structure is
  hand-authored once (then every op in the family rides it), and this path only
  *fuses* on GPU if the op ALSO declares a trait for the compiler-IR lane — i.e.
  it composes with A.2, it does not replace it.

### A.2 — Compiler-IR primary

- **Authoring (one point):** declare `OP_TRAIT_<category>` in `DECLARE_TYPES` +
  add an `OpCategoryTable` entry mapping the op name to an emitter clause /
  arith opcode. For a mapped category, **no hand kernel at all.**
- **CPU + GPU + Metal:** `TritonIRBuilder` emits fused TTIR → PTX/AMDGCN/SPIR-V;
  `CpuIRBuilder` emits MLIR → LLVM; `MlxIRBuilder` emits MLX — all from the same
  trait, fused with neighboring ops at the DSP segment level. This is the "one
  definition → all targets" ceiling, and it is **real and mature here** (Triton
  is multi-target, with disk cache + autotune + register-split).
- **Overrides:** a hand kernel registers via `DEFINE_PLATFORM_OP` / an
  override-by-symbol; the emitter is skipped for that `(op, engine)`. Same
  precedence.
- **New infra:** extend the category emitters to families they don't yet cover
  (data-movement / gather / scatter — `INDEX_REMAP`/`SCATTER_ACC` aren't emitter-
  backed today) **and** a per-op eager fallback for when a mapped op is called
  *outside* a DSP segment (the emitters are segment-scoped). **Exists:** the whole
  `TritonIRBuilder`/`CpuIRBuilder`/`MlxIRBuilder` stack, `isTritonMappable`,
  `OpCategoryTable`, and the elementwise/comparison/reduction/normalization
  emitters.
- **Example — a new fused activation:** declare `OP_TRAIT_UNARY_ELEMENTWISE` +
  map name→emit clause → it fuses on GPU with zero `.cu`, today. Contrast
  `one_hot`, which needs a *new* data-movement emitter category first.
- **Risk / maintenance:** highest ceiling and most of it is already built, but
  the "no functor at all" property only holds for emitter-covered categories and
  *inside* DSP; standalone/eager and unmapped categories still need a functor.
  Emitters are `mlir::Value` C++ — higher skill to edit than a functor, but a fix
  propagates to GPU+CPU+Metal at once.

### A.3 — Hybrid-unify (recommended)

- **Reality:** an op today *already* declares traits (feeding the compiler-IR
  fused lane) AND carries a functor/helper (eager, fallback, standalone, unmapped
  categories, every dtype). Both hang off one op declaration.
- **Authoring (the ADR unification):** ONE op-spec emits **both** — the
  `SD_HOST_DEVICE` functor (baseline) and the `OP_TRAIT_<category>` + opcode
  registration (fused fast lane) — plus decl/shape/types. "Define once, generate
  the cross-product" = generate functor + trait-entry + declaration from one
  spec, because both layers already key off the same op identity.
- **Runtime precedence (per op, chosen at DSP compile time by the ORT-style
  capability query):** in a DSP segment and trait-mappable → the fused emitter
  (fastest); else a `PLATFORM_IMPL` override if present; else the template-library
  functor (always correct).
- **New infra:** the unified op-spec + generator (functor + trait-entry + decl)
  and the capability-query/partition in `DynamicShapePlanCompiler::compile()`.
  **Exists:** both substrates in full.
- **Example — `one_hot`:** the spec emits the `INDEX_REMAP` functor (eager +
  fallback now) and, once a data-movement emitter category is added, the trait
  entry (fused later) — with no change to the authored spec. Elementwise /
  reduction / normalization ops get both lanes today.
- **Risk / maintenance:** two lanes to keep correct + benchmarked — but that is
  already the status quo; the only new thing is the *authoring* unification + a
  parity test asserting functor ≡ emitter. This is why it is recommended: it
  formalizes what libnd4j is, banks the mature emitter stack as the fast lane,
  and keeps the functor as the always-correct floor.

### Recommendation

Adopt **A.3**: template-library is the correctness floor, the trait-driven
emitters are the fused fast lane, and a single op-spec authors both. A.1 is the
floor *inside* A.3; A.2 is the fast lane *inside* A.3. Sequence: formalize the
op-spec on elementwise/reduction/normalization families first (both lanes already
exist for them), then extend the emitter categories to data-movement while the
functor covers those in the interim.

The buildable contract for this generation — the op-spec schema, the
present-driven **resolver** (`resolve(op, engine, present) → provider`), the
build-time **AOT** pipeline, the coverage/gap manifest, and an **observe-only**
bootstrap that predicts today's selection before generating anything — is
specified as a parallel concept in **ADR-0114**. Committed direction there:
build-time generation + AOT (runtime JIT is the fast lane, constrained to agree
with the AOT resolution).
