# ADR: Op Definition Resolver and AOT Generation — Specification

## Status

**Deferred — north-star concept, parked for later revisit (2026-07-13).** The
full realization of this spec is effectively a *custom op-authoring language with
auto-compilation to all backends*. That is the right long-run direction, but it
is a large build whose payoff is realized only once most of the machinery exists
— too big to justify starting now. This document is kept as the **buildable
blueprint to revisit when there is appetite for the language + auto-compile
step**. The committed generation direction, when it is picked up, is **build-time
generation + AOT materialization** (runtime JIT is the fast lane, constrained to
agree with the AOT resolution via Invariant I4), and the safe first move is the
**observe-only** bootstrap (§10).

## Conclusion (2026-07-13)

The exercise was worthwhile even though the resolver is being parked:

1. It **settled the precedence model** — unanimous across eight frameworks
   (ADR-0113 Prior art): generic-fallback + per-backend shadow override, which
   `KernelDispatchHelper` already implements. Nothing to build there.
2. It **corrected the substrate framing** — libnd4j already runs the hybrid
   (template-library floor + the mature trait-driven Triton/MLIR/MLX emitters);
   "compiler-IR vs template-library" was a false fork (ADR-0113 Appendix A).
3. It produced this **buildable contract** (the resolver, op-spec, AOT pipeline,
   invariants, gap manifest, observe-only bootstrap) so that revisiting is a
   pick-up, not a restart.

**What is actionable now — with none of this machinery — is the macro-based
common-pattern consolidation in ADR-0113 (Layers 0/1/3 + WS9).** That work
reduces real duplication (warp-reduce driver, `SD_CUDA_DISPATCH`, loss/validation
macros, `DEFINE_MKLDNN_ELTWISE` / `DEFINE_ARMCOMPUTE_*` / `cudnnConvForward`
family workers, updater functor-stamp) as ordinary macro/refactor edits — no
resolver, no codegen, no op-spec, no DSL. It stands on its own and is the correct
near-term path; the resolver/language is the deferred end-state that this
macro work incidentally moves toward (each family worker is a hand-rolled
instance of what the generator would emit).

## Date

2026-07-13

## Purpose

Define, precisely and language-neutrally, a **resolver** — a pure function that,
for each op × engine cell of the dispatch grid, selects which provider fills it
*from what is present in the build* — and the **AOT generator** that materializes
the resolver's output into registrations, type instantiations, and a coverage
manifest. The resolver is the single mechanism behind "define once, generate the
cross-product": one op-spec in, the whole present-dependent grid out.

Non-goals: no op is migrated; no existing file is changed; no kernel is written;
the runtime JIT path is not specified here (only constrained via I4). This is a
blueprint, validated observe-only against today's tree, so we can critique the
model before building it.

## 1. Vocabulary

- **Op** — a declarable operation, identified by a stable `name` and stable
  `opNum` (ABI; serialized into models; never generated or renumbered).
- **Engine** — a provider-granularity backend: `CPU`, `CUDA`, `ONEDNN`, `CUDNN`,
  `MPS`, `ACCELERATE`, `ARMCOMPUTE`, `MLIR`, `MLX`, `PJRT`, `MIOPEN`, `TRITON`,
  `VLM`. Finer than libnd4j's coarse `ENGINE_CPU`/`ENGINE_CUDA` macro argument
  (which several helpers share); the resolver operates at helper granularity and
  maps down to `ENGINE_*` + `HELPERS_*`/`HAVE_*` at emit time.
- **Cell** — one `(op, engine)` pair. The grid is ops × engines.
- **Provider** — what fills a cell. Exactly one of:
  `HAND_OVERRIDE(engine, symbol)` · `TRAIT_EMITTER(engine, category)` ·
  `GENERIC_ENGINE(family)` · `COMPOSITION` · `ABSENT(reason)`.
- **Lane** — a class of provider: the *override* lane (hand kernels / platform
  wrappers), the *emitter* lane (trait-driven IR: Triton/MLIR/MLX), the *generic*
  lane (template-library functor on the `loops/` engines), the *composition* lane.
- **Present-set** — what exists for a given build: enabled engines + discovered
  per-op artifacts (§3).
- **Floor engine(s)** — the engine(s) that must provide every op or the op is
  globally unavailable: `CPU` always, plus `CUDA` when the CUDA chip is built.
- **Resolution** — the resolver's output: for each op, a map `engine → Provider`.

## 2. The op-spec (single source of truth)

One authored record per op. All generation reads only this + the present-set.
Fields (schema, not syntax):

```
OpSpec {
  name        : string            // stable
  opNum       : int               // stable ABI key; authored, never generated
  signature   : { nIn, nOut, tArgs, iArgs, bArgs }
  shapeRule   : ShapeRef          // named shape-fn OR declarative descriptor
                                  //   (SAME_AS_INPUT(k) | SCALAR(dtype) | fn:<name>)
  typeRule    : { inTypes, outTypes, sameMode, promotion }
  traits      : set<OpTrait>      // OP_TRAIT_* categories (drives emitter mapping)
  compute     : Functor { family: MAP|INDEX_REMAP|TAD_REDUCE|SCATTER_ACC,
                          body:  SymbolRef }        // SD_HOST_DEVICE functor
              | Composition { body: SymbolRef }     // NDArray/SameDiff body
              | TraitOnly {}                         // emitter-only; MUST have a
                                                     //   mappable trait
  overrides   : map<engine, SymbolRef>?  // OPTIONAL pinned hand kernels; if
                                         //   omitted, discovery fills it (§3)
  requirements: Predicate         // declarative: dtypes, rankRange, contiguous →
                                  //   becomes supportsOp() (runtime, deferred)
}
```

`compute` is the "define once." `traits` is what enrolls the op in the emitter
lane. `overrides` is normally *discovered*, not written (§3). Everything else the
generator needs — declaration, shape-fn, types, registrations, instantiations —
is derived.

## 3. The present model — "what's present"

Three inputs; the first two feed the AOT resolver, the third is runtime (I4).

1. **EnabledEngines** — derived from build flags: `HAVE_TRITON`, `HAVE_CUDNN`,
   `HAVE_MKLDNN`(→ONEDNN), `HAVE_MLIR`, `HAVE_MLX`, `HELPERS_*`, chip = cpu|cuda.
   A CPU-only build yields `{CPU, MLIR?, ONEDNN?, ARMCOMPUTE?, ACCELERATE?}`.
2. **ArtifactIndex** — per op, which lanes exist, from **discovery reconciled
   against declaration**:
   - `overrideSymbols : map<engine, symbol>` — presence of
     `platform/<engine>/<op>.{cpp,cu,mm}` and `helpers/{cpu,cuda}/<op>.*`, OR the
     spec's pinned `overrides`.
   - `emitterEngines : set<engine>` — engines whose IR emitter covers the op's
     trait category: `TRITON`/`MLIR`/`MLX` when `isTritonMappable(op.name)` /
     the `OpCategoryTable` entry exists for a `traits` category the emitter
     implements (elementwise/comparison/logical/ternary/reduction/normalization
     today; data-movement not yet).
   - `genericPresent : bool` — a `Functor`/family or `Composition` body exists.
3. **CapabilityProbe (runtime, deferred)** — `supportsOp(op, engine, descriptor)
   → bool`: is the engine usable *now* (device up, lib loaded, dtype/rank ok).
   Not consumed by AOT; constrains it via I4.

**Discovery vs declaration.** Policy: *discovery proposes, the spec confirms.*
The generator scans presence and proposes providers; the op-spec may pin or
exclude; a build check **fails on drift** — a declared symbol that is absent, or
a present file that no spec claims. This avoids both magic (typo silently drops
an override) and verbosity (hand-listing every cell). PyTorch is fully
declarative (`dispatch:`); TF/ORT discover static registrars at link time; this
sits deliberately between.

## 4. The resolver

```
resolve(op, enabledEngines, artifactIndex) -> Resolution   // map engine -> Provider

for each engine e in enabledEngines:                        // fixed iteration order
  provider(op, e) =
     if e in artifactIndex.overrideSymbols   -> HAND_OVERRIDE(e, sym)          // 1
     elif e in artifactIndex.emitterEngines  -> TRAIT_EMITTER(e, op.category)  // 2
     elif genericHostable(e) and artifactIndex.genericPresent
                                             -> GENERIC_ENGINE(op.family)      // 3
     elif op.compute is Composition and genericHostable(e)
                                             -> COMPOSITION                     // 3'
     else                                    -> ABSENT(reason(op, e))          // 4
```

- **Precedence is a fixed total order:** `HAND_OVERRIDE > TRAIT_EMITTER >
  GENERIC_ENGINE / COMPOSITION > ABSENT`. An override always shadows; the emitter
  is the next fastest; the generic/composition is the floor.
- `genericHostable(e)` is true for `CPU` and `CUDA` only (the `loops/` engines
  host the template-library generic). Other engines are override- or
  emitter-only; a non-hosting engine with neither is legitimately `ABSENT` — its
  ops route to the floor at runtime.
- **Purity & totality:** no side effects; *every* cell resolves to exactly one
  Provider including `ABSENT`. `reason(op,e)` is machine-readable
  (`engine-not-enabled` | `no-override` | `category-not-emitter-mapped` |
  `not-generic-hostable` | `type-unsupported`).
- **Determinism:** identical `(op-spec, present-set)` ⇒ identical Resolution,
  byte-for-byte, with engines emitted in fixed order.

Coverage rule (checked after resolution): an op is **COVERED** iff every floor
engine has a non-`ABSENT` provider. Per-engine `ABSENT` is normal; a floor gap is
a build error unless the op is explicitly flagged `partial`.

## 5. AOT generation pipeline

Runs at build time (in `buildnativeoperations.sh`, reusing op-codegen
`generate.sh` + the `.cu.in` + `configure_file` machinery). Stages:

1. **Load** all `OpSpec`s.
2. **Index** — build `ArtifactIndex` per op via discovery ⋈ declaration (§3),
   failing on drift.
3. **Resolve** — `resolve()` each op over `EnabledEngines` → per-op `Resolution`.
4. **Check** — enforce Invariants (§7): totality, floor coverage, precedence,
   opNum stability, type completeness.
5. **Emit** — deterministically, sorted by `opNum` then fixed engine order:
   - op **declaration** + shape-fn + types (once per op),
   - per-cell **registration**: `HAND_OVERRIDE` → register the symbol;
     `TRAIT_EMITTER` → record trait/opcode mapping (kernel itself is runtime-JIT,
     nothing to AOT-emit beyond the mapping row); `GENERIC_ENGINE`/`COMPOSITION`
     → instantiate the generic engine binding for that engine **and** the full
     authored **type cross-product** (`BUILD_*_SELECTOR` / `.cu.in`),
   - the **gap manifest** (§6),
   - `include_ops.h` / registration aggregation.
6. **Materialize idempotently** — content-hash each generated file; write only on
   change. Same present-set ⇒ no rebuild (ccache-safe).

"Depending on what's present" is literal: a CPU-only build emits no CUDA/CUDNN
cells; `-Dlibnd4j.helper=onednn` emits the ONEDNN overrides; `HAVE_TRITON=OFF`
drops the emitter rows and the floor carries those ops.

## 6. Generated artifacts

1. **Registration TU(s)** — the resolved per-cell wiring.
2. **Type-instantiation TUs** — generic providers × authored type set (types are
   untouchable; the full cross-product is generated, minus slices an override
   claims).
3. **Resolution manifest** (`resolution-manifest.json`, machine-readable):
   ```
   { op: { opNum, coverage: COVERED|PARTIAL,
           cells: { <engine>: { provider: OVERRIDE|EMITTER|GENERIC|COMPOSITION|ABSENT,
                                symbol?, category?, reason? } } },
     summary: { ops, engines, cellsProvided, cellsAbsent,
                floorGaps: [...], emitterCoverageByCategory: {...} } }
   ```
4. **Obligation report** (PyTorch `RegistrationDeclarations.h` analogue): per
   engine, the ops it *must* implement (no floor / explicitly required) vs the
   ops it *inherits* from the generic. This is the machine-readable answer to
   "what must a new backend author fill?"

## 7. Invariants & guards

- **I1 Determinism.** Same `(specs, present-set)` → byte-identical output.
- **I2 Totality.** Every `(op, enabledEngine)` cell resolves to exactly one
  Provider (incl. `ABSENT`); no undefined cells.
- **I3 Floor coverage.** Every op is non-`ABSENT` on every floor engine, else the
  build fails (unless `partial`).
- **I4 Build ≡ Runtime.** The AOT Resolution equals what the runtime resolver
  would choose for the same present-set (asserted by test). Prevents "fuses in
  JIT, missing in AOT" drift.
- **I5 opNum stability.** `opNum` is read from the spec, never generated or
  renumbered (serialized-model ABI).
- **I6 Type completeness.** `GENERIC`/`COMPOSITION` providers instantiate the
  full authored type set; the resolver never narrows types.
- **I7 Precedence total order.** `OVERRIDE > EMITTER > GENERIC/COMPOSITION >
  ABSENT`, fixed and global.
- **I8 No silent holes.** Every `ABSENT` carries a machine-readable `reason`; the
  manifest is a required build artifact and diffing it gates CI (a cell moving to
  `ABSENT` unexpectedly fails the build).

## 8. Abstract interfaces (spec, not code)

```
OpSpec                                             // §2
ArtifactIndex   = { overrideSymbols, emitterEngines, genericPresent }
Provider        = HAND_OVERRIDE | TRAIT_EMITTER | GENERIC_ENGINE | COMPOSITION | ABSENT
Resolution      = map<engine, Provider>

discoverArtifacts(op, tree, enabledEngines) -> ArtifactIndex        // §3, pure over a snapshot
resolve(op, enabledEngines, artifactIndex)  -> Resolution           // §4, pure
checkInvariants(op, resolution)             -> [Violation]          // §7
emit(op, resolution)                        -> [GeneratedFile]      // §5, deterministic
manifest(allResolutions)                    -> ResolutionManifest   // §6
supportsOp(op, engine, descriptor)          -> bool                 // runtime, DEFERRED; must satisfy I4
```

The generator is `emit ∘ (checkInvariants; resolve) ∘ discoverArtifacts`, folded
over all specs, then `manifest`.

## 9. Worked examples

**`tanh`** — traits `{UNARY_ELEMENTWISE, ACTIVATION}`; functor present; ONEDNN +
CUDNN + MPS wrappers present.
- Build `{CPU, CUDA, TRITON, ONEDNN, CUDNN, MLIR}`:
  `CPU→GENERIC(MAP)` · `CUDA→GENERIC(MAP)` · `ONEDNN→OVERRIDE(tanhMKLDNN)` ·
  `CUDNN→OVERRIDE(tanhCUDNN)` · `TRITON→EMITTER(UNARY_ELEMENTWISE)` ·
  `MLIR→EMITTER(UNARY_ELEMENTWISE)`. COVERED (floor CPU+CUDA generic).
- Build `{CPU}` (no triton/onednn/cudnn): `CPU→GENERIC(MAP)`; all other engines
  omitted (`engine-not-enabled`, not a gap). COVERED.

**`one_hot`** — trait `INDEX_REMAP`-shaped (data-movement, **no emitter category
yet**); functor present; no overrides.
- Any build: `CPU→GENERIC(INDEX_REMAP)` · `CUDA→GENERIC(INDEX_REMAP)`;
  `ONEDNN/CUDNN/MPS/TRITON→ABSENT(category-not-emitter-mapped | no-override)`.
  COVERED via floor. Manifest flags "no accelerated lane; add a data-movement
  emitter category" as an improvement target — visible, not silent.

**`matmul`** — trait `MATMUL`; cuBLAS/cuBLASLt + oneDNN GEMM overrides present;
generic present.
- `CUDA→OVERRIDE(cublasLt)` (shadows emitter+generic) · `ONEDNN→OVERRIDE(gemm)` ·
  `TRITON→EMITTER(MATMUL)` · `CPU→GENERIC` (or ONEDNN override at CPU). Shows
  precedence: overrides win where present, emitter/generic fill the rest.

## 10. Observe-only bootstrap (the parallel concept)

The first realization changes nothing in the tree:

- **Phase 0 — report-only.** Implement `discoverArtifacts` + `resolve` + `manifest`
  and run them over the **current** tree and the **current** build flags. Emit
  **only** the resolution manifest + obligation report. No registrations, no
  `.cu`, no edits to any op.
- **Validation.** Assert the manifest reproduces today's observed selection:
  the `HAND_OVERRIDE` cells match the present `PLATFORM_IMPL` registrars, the
  `TRAIT_EMITTER` cells match `isTritonMappable`, and the `GENERIC` cells match
  the ops with a functor/loops binding and no override. Where the manifest and
  reality disagree, the *model* is wrong — fix the spec, not the tree.
- **Only after** the manifest matches reality for CPU and CUDA builds does
  `emit` get turned on for a single pilot family (elementwise), behind a flag,
  compared byte-for-byte against the hand-written registrations it replaces.

This keeps the resolver a *parallel concept* — an observer that earns trust by
predicting the existing system before it is allowed to generate it.

## 11. Open questions (deferred, not blocking the spec)

- **Discovery vs declaration** final policy per lane (recommended:
  discovery-proposes/spec-confirms with fail-on-drift; §3).
- **Engine enumeration** and which engines are `genericHostable` beyond CPU/CUDA
  (e.g. does a future portable-parallel layer host the generic on more engines?).
- **Composition lane** materialization: AOT-generated NDArray body vs a runtime
  SameDiff sub-graph.
- **opNum allocation** authority and the append-only registry that guards I5.
- **Runtime resolver / JIT boundary**: where `supportsOp` + per-op JIT live in
  `DynamicShapePlanCompiler::compile()`, and the test that enforces I4.
- **Variant / dynamic-shape ops** (`OP_TRAIT_VALUE_DEPENDENT_SHAPE`,
  multi-output, wide ops): how one spec expresses them.
- **Manifest-diff gate**: what CI treats as an allowed vs breaking coverage
  change (I8).

## Relationship to ADR-0113

This resolver is the generation mechanism for ADR-0113 Appendix A.3
(hybrid-unify): the op-spec (§2) is A.3's single authored spec; the emitter lane
is A.2; the generic lane is A.1; `resolve()` is the "generate the cross-product,"
and the runtime half of I4 is the ORT-style capability query ADR-0113 folds into
DSP compile. ADR-0113 remains the strategy; this is the buildable contract for
its Layer-2/3 generation, kept deliberately parallel until the observe-only
manifest proves it.
