# ADR: Kernel Expression DSL for the Emitter Lane

## Status

**Proposed — foundation landed, deliberately UNWIRED (2026-07-16).** The
`libnd4j/include/graph/kernelspec/` module exists and compiles in every chip
build, but no emitter, table, or execution path consults it yet. Existing
emitters are kept untouched; wiring is gated on the parity tests described in
§6.

## Date

2026-07-16

## Context

A Jul-16 audit of the Triton and Vulkan compilation/emitter infrastructure
found four independent emitter codebases that each re-derive per-op semantics,
plus five hand-synced per-op capability catalogs:

1. **Triton** (`graph/gpu/TritonIRBuilder*`): in-process MLIR C++ builders →
   TTIR → TTGIR → LLVM → PTX/AMDGCN. Per-op math lives in hardcoded if-else
   chains keyed on a `tritonIrOp` string inside the category emitters
   (`TritonIRBuilder_emitters.cpp`). Two tables must stay in sync by hand —
   `buildOpTable()` (emission) and `getOpCategoryTable()` (routing); an op in
   the second but not the first routes to Triton and dies with
   `KERNEL_FAILURE` (documented at `TritonIRBuilder.cpp:462-467`).
2. **Vulkan** (`graph/vulkan/VulkanSegmentRecorder` + `VulkanOpLowerings` +
   `VulkanKernelEmitterCatalog`): textual MLIR (linalg) per-op policies →
   4-stage pass pipeline → SPIR-V → `VkPipeline`. The catalog is data-driven
   (recipe, dtype mask, layout mask, rank bounds, traits, argument schema) but
   keyed by descriptor hash, disjoint from the name-keyed Triton tables.
   The Vulkan recorder rejects descriptor, dtype, layout, rank, or argument
   combinations that its catalog cannot implement. Catalog acceptance alone does
   not prove that lowering, SPIR-V creation, dispatch, and replay succeeded.
   tArgs are baked into the MLIR text (`pushConstantBytes == 0` everywhere).
3. **CPU MLIR** (`graph/cpu/CpuIRBuilder`): copy-paste-parallel `emit*` API
   (no shared base class with Triton) over arith/math/scf/memref, lowered via
   the shared `MLIREngine::buildCPUPipeline()`.
4. **MLX** (`graph/cpu/MlxIRBuilder`): same category decomposition, but emits
   MLX lazy arrays rather than MLIR.

The related metadata surfaces are the operations' own `addTraits`
implementations, `OpCategoryTable.h`, Triton `buildOpTable()`,
`VulkanKernelEmitterCatalog`, and the Java
`DynamicShapePlanCompiler` trait-mirror constants. Op-local `addTraits`
remains the framework trait authority; a generated emitter specification must
validate against or generate those op-local declarations, never replace them
with a centralized traits table. Adding one expression-shaped op end-to-end
still requires several emitter and routing surfaces to agree.

ADR-0113 (Appendix A) already frames the emitters as the "compiler-IR fast
lane" and recommends hybrid authoring; ADR-0114 specifies the full op-spec +
resolver + AOT model and is deferred as the north star. This ADR is the
tractable slice of that north star scoped **to the emitter lane only**: it
needs none of the resolver/AOT grid.

## Decision

Introduce a three-layer kernel-authoring DSL, additive at every step:

**Layer 1 — single-source `KernelSpec` registry.** One authored record per op
carrying name + aliases, category, `OP_TRAIT_*` mask, dtype capability mask,
scalar parameters (tArgs-backed, with defaults), and the op's math. The
registry can generate backend emitter catalogs and consistency checks. Framework
traits remain declared by each operation's `addTraits`; any future code
generation must update that op-local declaration rather than introduce a
central trait registry.

**Layer 2 — `KernelExpr`, a portable expression AST for op bodies.** ~30 node
kinds (arith, math, comparisons, select, logic, scalar params, constants) plus
composite helpers (`sigmoid`, `silu`, `mish`, `clamp`, ...). Reductions are
authored as an `(init, combine, finalize)` triple — the same decomposition
Vulkan's `reductionCallbacksFor` `{initValue, combineOp, finalizeOp}` and
Triton's `tt.reduce` combiner regions already use, so the triple maps 1:1 onto
every backend. One shared MLIR interpreter (`emitKernelExpr`) serves all
MLIR-value backends, with a small `MlirEmitPolicy` supplying the
precision-sensitive primitives per backend (Triton substitutes its
`emitNativeCuda*` libdevice emitters; CPU uses the stock math dialect; Vulkan
supplies linalg-body constants). MLX gets its own small interpreter later.

**Layer 3 — control surface.** The spec carries per-backend enablement,
hand-written-override pins (`handWritten(engine)` — standard
override-beats-emitter precedence, unchanged), and later per-op tile/fusion
hints unified with the existing `ND4J_TRITON_*` knobs and the ADR-0104 PGO
work.

**Deliberately out of scope:** structural kernels — matmul tiling, flash
attention, convolution, sorts/scans, scatter, normalization multi-pass
patterns. These stay hand-written recipes the spec references by name,
matching ADR-0113's "the IR fills gaps, the hand kernel wins" conclusion.

## What landed in this change (all unwired)

- `graph/kernelspec/KernelExpr.h` + `impl/KernelExpr.cpp` — dependency-free
  AST, authoring sugar, structural validation, printer.
- `graph/kernelspec/KernelSpec.h` + `impl/KernelSpec.cpp` — spec struct,
  validation (arity vs category, reduction-triple conventions, scalar-param
  bounds), thread-safe registry, fluent builder.
- `impl/KernelSpecPilots.cpp` — 8 pilot specs mirroring ops that already
  exist in the hand-written emitters (`swish_mul`, `elu`, `clipbyvalue`,
  `hardsigmoid`, `mish`, `reduce_sum`, `reduce_mean`, `reduce_max`), behind an
  explicit idempotent `registerPilotKernelSpecs()` — no static initializer.
- `KernelSpecConsistency.h` + `impl/KernelSpecConsistency.cpp` — drift check
  of registered specs against the shared `getOpCategoryTable()` (phase 0 of
  the consistency gate; the Triton `buildOpTable` and Vulkan-catalog legs land
  when those tables expose enumeration).
- `KernelExprMlirEmitter.h` + `impl/KernelExprMlirEmitterCheck.cpp` — the
  shared header-only MLIR interpreter + a `#if HAVE_MLIR` compile-verification
  anchor TU (same gate `CpuIRBuilder.cpp` uses; empty TU on non-MLIR builds).

Sources are picked up automatically by the `GRAPH_SOURCES` recursive glob
(`MainBuildFlow.cmake:188`); no build-system or existing-file edits were made.

## Wiring plan (future phases, each gated)

1. **P0 consistency gate**: run `crossCheckKernelSpecsWithOpCategoryTable()`
   in a platform-tests check; extend to `buildOpTable` (needs a small
   enumeration accessor on `TritonIRBuilder`) and the Vulkan catalog.
2. **P1 Triton**: one new generic case in
   `emitUnaryElementwise`/`emitBinaryElementwise` that resolves the slot's op
   in the `KernelSpecRegistry` and calls `emitKernelExpr` with a
   Triton-flavored `MlirEmitPolicy` (libdevice hooks). Parity test: DSL
   emission vs existing hand-written emission, token/numeric-exact, before any
   hand-written case is retired.
3. **P2 CPU + Vulkan**: same interpreter with the default policy inside
   `CpuIRBuilder`; Vulkan gains a generic elementwise policy whose linalg body
   is generated from the AST, plus catalog registration generated from the
   spec (including `opIsRecordable` guards derived from the spec's
   dtype/rank/layout declarations — required because Vulkan hard-fails on
   gaps).
4. **P3 authoring host**: extend `codegen/op-codegen`'s Kotlin `Op` schema
   (traits/category/kernel fields are absent today) so specs are authored
   next to the existing Java-generation source and the C++ registration is
   generated — per the standing codegen-first mandate and ADR-0114's
   build-time-generation decision.
5. **P4 extensions**: matmul epilogues (Triton's `emitMatmulKernel` already
   takes `epilogueOps`), Vulkan push constants for scalar params (removes the
   baked-tArgs pipeline explosion), MLX interpreter, integer/bool dtype paths.

## Constraints the design respects

- **`OP_TRAIT_*` 32-bit space is full** (bit 31 used): the spec reuses the
  existing mask verbatim and adds its own orthogonal category enum; it never
  needs new trait bits.
- **Traits are not purely static** (`NativePlanCompiler` clears
  `OP_TRAIT_DATA_DEPENDENT` for 3-input `where`): such ops stay outside the
  DSL or get an explicit escape hatch later.
- **Name keying (Triton) vs descriptor-hash keying (Vulkan)**: the spec is
  name+alias keyed; the Vulkan leg derives the descriptor hash from the op
  class at generation time.
- **Scalars are baked in v1** on both backends (matches current behavior);
  per-step-varying scalars stay unsupported until Vulkan push constants land.
- **Shapes stay baked / shape-keyed caches stay** — the DSL neither fixes nor
  worsens recompilation-per-shape.
- **No emitter or kernel is removed, ever** (ADR-0113 non-goals). Hand
  kernels shadow DSL emission through the existing precedence.

## Consequences

- Once wired, a new expression-shaped op is a 5-10 line spec that lights up on
  Triton, CPU-MLIR, and Vulkan simultaneously, fuses into segments
  automatically, and cannot desync across the catalogs it generates.
- Until wired, the module is inert: zero behavior change, zero risk, and the
  consistency checker already gives a machine-readable drift report between
  the DSL's view and `OpCategoryTable`.
- The MLIR interpreter is compile-verified only on `HAVE_MLIR` builds (mlir
  helper, vulkan chip) and by direct syntax checks; the default CPU/CUDA
  builds compile it out — acceptable while unwired, revisit at P1.

## Relationship to other ADRs

- **ADR-0113**: this is Appendix A.2/A.3's emitter lane given a single
  authoring surface; the template-library floor and `KernelDispatchHelper`
  precedence are untouched.
- **ADR-0114**: deferred north star; this ADR implements only its
  `TRAIT_EMITTER` lane's authoring, observe-only-first philosophy included
  (the consistency checker predicts the tables before anything generates
  them).
- **ADR-0104** (Triton tile PGO): Layer-3 per-op tile hints are the natural
  join point.
- **ADR-0115** (Vulkan disk caches): DSL-emitted Vulkan kernels flow through
  the same `mlirToSpirv` + tiered cache path unchanged.
