# DSP CPU Codegen Tier — Triton-CPU Integration Design

Date: 2026-07-02 · Author: perf audit session (Claude) · Status: DESIGN — pre-implementation
Companion: `DSP_PERF_HANDOFF.md` (Part I gaps, Part II strategy; this doc expands WS-L/H4)
ADR note: intentionally NOT an ADR yet — distill into one after implementation results.

---

## 1. Summary

DSP has no compiled execution tier on CPU: steady-state CPU inference is a serial
slot-by-slot interpreter, optionally accelerated per-segment by oneDNN or OpenVINO.
Meanwhile the CPU build already downloads, compiles, and links the official
`triton-lang/triton-cpu` backend — toolchain, LLVM, headers, `HAVE_TRITON_CPU=1` —
but not one line of runtime code can reach it, because all Triton consumer sources
are excluded from non-CUDA compilation. This design closes that gap: compile the
existing (target-agnostic) Triton section analysis and kernel emitters on CPU, add a
CPU lowering target and ORC-JIT launcher behind `TritonTargetDispatch`, and register
a `TritonCpuGraphBackend` in the CPU backend chain, selected per replay-unit by
measurement (the WS-G auction), never by fiat. The result is the chip-agnostic
thesis made real on CPU: one section/fusion analysis, N lowerings, with oneDNN,
OpenVINO, microkernel libraries, and the interpreter as competing candidates.

## 2. Background and problem statement

### 2.1 DSP execution model (context)

DSP compiles a SameDiff graph into segments; Triton-mappable slot runs become
"islands" (captured/compiled units), the rest are live "gaps". On GPU, islands are
Triton-compiled and CUDA-graph-replayed. Per-chip replay is abstracted by
`GraphReplayHandle` (`graph/impl/GraphReplayFactory.cpp:19-55` routes CUDA /
HIP-via-ZLUDA / LevelZero-via-ZLUDA / Vulkan / Metal / TPU / Hexagon / CPU).

### 2.2 CPU today (all verified 2026-07-02)

- CPU "replay" is a no-op counter: `FunctionalReplayHandle::replay()`
  (`graph/impl/FunctionalReplayHandle.cpp:69-79`); every token re-runs the serial
  interpreter loop (`NativeDynamicShapePlan_segments.cpp:1006-1021`). No inter-op
  parallelism.
- Compiled options are backend-chain only: OpenVINO (all-or-nothing per segment) and
  oneDNN (island-style partial fusion), else `EMULATED_REPLAY`
  (`graph/cpu/NativeDynamicShapePlan_cuda_stubs.cpp:566-578`).
- The in-house MLIR JIT (`CpuIRBuilder`, ADR 0085) is excluded from the build:
  `HAVE_MLIR=0` in the current CPU `config.h`, and `MainBuildFlow.cmake:213-224`
  removes `CpuIRBuilder.cpp` / `MlirCpu*` / `ArmHybridGraphBackend.cpp` when MLIR is
  off.

### 2.3 The paradox this design resolves

`-Dlibnd4j.triton=ON` (mandatory for CPU builds per project docs) currently does the
following on CPU (verified against `blasbuild/cpu` cache and generated config):

| What happens | Evidence |
|---|---|
| Auto-enables oneDNN helper on x86 | `buildnativeoperations.sh:2688-2705` |
| Gates OpenVINO ("triggered by SD_TRITON=ON, not helpers") | `cmake/Options.cmake:87`, `Dependencies.cmake:2602-2605` |
| Downloads + builds `triton-lang/triton-cpu` @ pinned commit `c4ccb98…` | `Dependencies.cmake:1573, 1603-1604` |
| Builds LLVM from source @ pinned `20902f0…`, host targets only | `Dependencies.cmake:1650-1657, 1781-1787` |
| Builds with `TRITON_CODEGEN_BACKENDS "cpu"`; Python dependency patched out | `Dependencies.cmake:1616-1618, 2005-2008` |
| Installs to `triton_cpu_install` / `triton_cpu_llvm_install`; reuse-if-present | `Dependencies.cmake:1438-1443, 1446-1453` |
| Defines `HAVE_TRITON=1` AND `HAVE_TRITON_CPU=1` | `blasbuild/cpu/include/config.h:11-12`, CMakeCache |
| Links `libtriton.a` + MLIR/LLVM static libs (MLIR core whole-archive) into the object lib | `Dependencies.cmake:1464-1523`, `MainBuildFlow.cmake:871-872` |

And yet, nothing can use it:

| Why it is unreachable | Evidence |
|---|---|
| ALL `graph/gpu/*.cpp` excluded from non-CUDA builds (emitters, backend, dispatch, cache) | `MainBuildFlow.cmake:189-198` ("CUDA-only infrastructure") |
| `HAVE_TRITON_CPU` has zero references in any .cpp/.cu/.h | only `config.h.in:12` + cmake config generation |
| One generated, unused Java constant | `Nd4jCpu.java:239` |
| CPU backend chain has no Triton entry | `cuda_stubs.cpp:566-578` |

Net cost today: an hours-cold LLVM+triton-cpu toolchain build, gigabytes of disk,
whole-archive MLIR core in the shipped `.so` — for zero runtime benefit. The
consumer code exists (GPU emitters) and the toolchain exists (triton-cpu); they are
two halves of a bridge that were never joined.

### 2.4 Why a CPU codegen tier matters (perf motivation)

From the Part I audit: CPU decode/embedding loses to (a) the serial interpreter loop
with per-slot guard overhead, (b) BLAS-only compute with no cross-op fusion outside
oneDNN/OpenVINO coverage, (c) attention/elementwise chains executed op-by-op with
full materialization between ops. A section-fused JIT tier attacks (b) and (c)
directly, and pairs with the E1 micro-schedule for (a). Historical reference
workload: BGE embedding [32×512] at 68-82s/pass before the threading fixes.

## 3. Goals and non-goals

**Goals**
1. A reachable CPU codegen execution tier driven by the SAME section analysis and
   kernel emitters as the GPU (`TritonIRBuilder_sections/_module/_kernels/...`).
2. Selected empirically per unit (WS-G auction) against oneDNN, OpenVINO,
   microkernels, and the interpreter — never enabled by fiat.
3. No Python anywhere at runtime or build-time beyond what exists today (the build
   already patches Python out of triton-cpu).
4. Disk-cached JIT artifacts keyed by host CPU features (AVX2/AVX-512/AMX), reusing
   `TritonCacheBundle`.
5. Pluggable lowering (H4/I1): the CPU target sits behind the same seam that could
   host IREE or MLIR-direct later.

**Non-goals**
- Replacing oneDNN/OpenVINO (they remain chain members and auction candidates).
- Matmul-first coverage: initial scope is ELEMENTWISE/IDENTITY (+ REDUCTION/
  NORMALIZATION second); GEMM on CPU is better served by oneDNN/libxsmm candidates
  initially (see §8 microkernels).
- Windows (Triton already disabled there — `Dependencies.cmake:1421-1426`).
- NUMA/multi-socket placement; deferred.
- The CPU frozen micro-schedule itself (Part I E1) — separate workstream; this tier
  plugs into it.

## 4. Design overview

### 4.1 Components

```
            (existing, becomes CPU-compiled)          (new)
 Segment → TritonIRBuilder_sections → TTIR module → TritonTargetDispatch
                                                     ├─ CUDA target (existing: PTX → cuModule)
                                                     └─ CPU target  (NEW: triton-cpu passes → LLVM → ORC JIT)
                                                              │
                                                    CpuKernelLauncher (NEW)
                                                     grid → thread-pool loop → jitted fn ptrs
                                                              │
                                    TritonCpuGraphBackend (NEW, registered in getCpuGraphBackendChain)
                                                              │
                                    FunctionalReplayHandle (existing, becomes real:
                                    replay() executes the jitted-section schedule + native gap ops)
```

### 4.2 Execution flow (steady state)

1. Segment resolution: `TritonCpuGraphBackend::compileSegment` runs the existing
   section builder over the segment (unchanged code), emits one TTIR module per
   section group (unchanged emitters).
2. Lowering: the new CPU target in `TritonTargetDispatch` drives triton-cpu's pass
   pipeline (TritonToTritonCPU conversion → TritonCPU dialect → vector/scf → LLVM
   dialect), then hands the LLVM module to an ORC LLJIT instance. Exact pass list is
   lifted from the pinned commit's `third_party/cpu` backend driver during the P0
   spike (see §10 open questions) — do not guess it; extract it.
3. JIT + cache: object code cached via the existing `TritonCacheBundle` disk cache;
   cache key = existing TTIR/config hash + **host CPU feature set** (AVX2/AVX-512
   flags/AMX) in place of the GPU arch. Warm process start = load object, no MLIR.
4. Launch: `CpuKernelLauncher` executes the grid: for each program id (block), call
   the jitted kernel function with the triton-cpu ABI (args array + program ids).
   Outer grid dimension parallelized on the libnd4j thread pool; each program body
   is single-threaded by construction (triton-cpu model). No H2D, no arg-table
   device copy — the "arg table" degenerates to a host pointer array; the entire
   GPU arg-refresh machinery is unnecessary on this path.
5. Replay: `FunctionalReplayHandle` for the segment holds the ordered schedule of
   {jitted-section launches + native gap ops}; `replay()` executes it. This makes
   CPU "replay" mean what it means on every other chip (I2), and slots directly
   into the E1 micro-schedule when that lands.

### 4.3 Key design decisions

- **D1 — Reuse the Triton-dialect emitters, do not write a second linalg emitter.**
  Single source of kernel truth across chips is the entire point (H4). The linalg
  path (`CpuIRBuilder`) stays as an alternative lowering, not a parallel emitter to
  maintain (§8).
- **D2 — Lowering is a plug-in.** `TritonTargetDispatch` gains a target interface
  (compile(TTIR, config) → {callable, metadata}); CUDA/PTX is one implementation,
  triton-cpu is the second, IREE/MLIR-direct could be a third. Capability descriptor
  per target (canJIT, vectorISA, preferredBlockMultiple, launchOverheadHint) feeds
  the scheduler/auction (I1).
- **D3 — Grid semantics on CPU.** numWarps/numStages are GPU concepts: the CPU tile
  profile in `selectTileConfig` (`TritonIRBuilder_analysis.cpp:858-952` gains a CPU
  branch) sets blockSize = vector-width multiples (e.g. 8-64 elems × dtype), one
  stage, and grid = ceil(N/block). Parallelism comes from grid mapping onto the
  thread pool, NOT from OMP inside the kernel.
- **D4 — One LLVM in the process.** The ORC JIT uses the SAME
  `triton_cpu_llvm_install` LLVM that libtriton.a links (already in the binary).
  HARD RULE: never link a second LLVM (e.g. from a separately-pinned HAVE_MLIR
  build) into the same image — ODR/TypeID chaos. If HAVE_MLIR is enabled later, it
  must be rebased onto the triton-cpu LLVM pin first (§8, CpuIRBuilder).
- **D5 — ABI fidelity.** Kernel call ABI (argument packing, grid-coord passing,
  scratch/shared-memory emulation if any) is lifted verbatim from the pinned
  triton-cpu commit during P0 — an executable spike, not documentation reading, is
  the acceptance test.
- **D6 — Thread-budget coordination.** One owner for CPU parallelism per plan
  execution: when the launcher parallelizes a grid, intra-op OMP and BLAS threads
  are capped accordingly (coordinates with Part I E2/E3; `SD_BLAS_SERIALIZE`
  interplay: jitted sections never call BLAS, so no serialization concern inside
  this tier).
- **D7 — Verify-then-race, scoped fallback.** A jitted section becomes eligible only
  after its warmup output matches the interpreter reference within `run-validation`
  tolerances (G2). Failures deopt THAT SECTION to the interpreter, with cooldown +
  re-attempt (F1) — never plan-wide, never permanent, never silent (F2 WARN).
- **D8 — Initial section-type scope.** Phase 1 compiles ELEMENTWISE + IDENTITY
  sections only (the types already `compiledByDefault` on GPU —
  `SectionTypeConfig.h:64,82`), then REDUCTION/NORMALIZATION (tt.reduce paths).
  MATMUL/ATTENTION sections stay native on CPU initially: oneDNN/libxsmm are the
  right first candidates there, and the GPU-side emitters for those carry GPU
  assumptions (tensor-core dots, shared-mem tiling) that do not transfer.

## 5. Build system changes (maps to WS-L L1)

1. **File-granular inclusion instead of directory exclusion.** Replace the blanket
   `graph/gpu/*.cpp` removal for non-CUDA (`MainBuildFlow.cmake:189-198`) with two
   lists:
   - CPU-eligible (compile when `HAVE_TRITON`, any chip): `TritonIRBuilder_*.cpp`,
     `TritonGraphBackend_cache.cpp`, `_lru.cpp`, `_binary.cpp` (non-CUDA parts),
     `TritonTargetDispatch.cpp`, `TritonCacheBundle.cpp`.
   - CUDA-only (unchanged): all `*.cu` (`TritonGraphBackend_{compile,execute,kernel}.cu`,
     `TritonCudaDriverDispatch.cu`, `DspCudaDispatch.cu`, `ResourceBinder_cuda.cu`,
     `NativeDynamicShapePlan_*.cu`).
2. **Include hygiene pass.** The CPU-eligible .cpp files must compile without CUDA
   headers: wrap residual `SD_CUDA`-specific includes/blocks (e.g. PTX/NVRTC
   sections of `_binary.cpp`, driver bits of `TritonTargetDispatch.cpp`) in
   `#if defined(SD_CUDA)`. Budget explicit time for this; it is the main mechanical
   risk of L1.
3. **New sources**: `graph/cpu/TritonCpuGraphBackend.{h,cpp}`,
   `graph/cpu/CpuKernelLauncher.{h,cpp}`, CPU target implementation for
   `TritonTargetDispatch` (new .cpp under `graph/cpu/`, so it never lands in a CUDA
   source list).
4. **Linkage**: already done — `triton_interface` links on any build where the
   target exists (`MainBuildFlow.cmake:871-872`). Measure binary-size delta before/
   after (whole-archive MLIR core is already paid today).
5. **Config plumbing** (repo rule): `dspCpuTriton` (AUTO/ON/OFF) +
   `dspCpuTritonThreads` in `system/Environment.h/.cpp`, mirrored in Java
   `Environment.java` + `ND4JSystemProperties`, preset-exposed.

## 6. Runtime integration

1. **Backend chain**: `getCpuGraphBackendChain()` (`_segments.cpp:451-467` region)
   gains `TritonCpuGraphBackend` guarded by `HAVE_TRITON_CPU` + `dspCpuTriton`.
   Initial order: Triton-CPU → OpenVINO → oneDNN → interpreter, but ordering is a
   default only — the WS-G auction decides per unit once it lands; before then,
   AUTO enables Triton-CPU only for sections it verifies (D7).
2. **FunctionalReplayHandle** gains a schedule payload: ordered list of
   {jittedKernel(fn, gridDims, argSlots) | nativeSlotRange}. `replay()` executes it;
   `state_=READY` requires all member sections jitted + verified. Existing warmup/
   freeze lifecycle drives it — no new lifecycle states.
3. **Diagnostics**: reuse DSP_DIAG JIT/EXECUTE categories; per-section compile-time
   and per-launch timing feed the same ledger as GPU (G3) so CPU/GPU decisions are
   comparable in one place.
4. **Java**: no API change required for the tier itself. The `prewarm()` API (Part I
   C2) applies unchanged: prewarming a CPU plan runs JIT + verification off the
   request path.

## 7. Validation plan (per AGENTS.md; all from `platform-tests/`, all through `tee`)

1. New tests (CPU backend): `TritonCpuSectionExecutionTest` (per section type:
   jitted output == interpreter output across dtypes/shapes incl. odd lengths and
   broadcast cases), `TritonCpuBackendChainTest` (chain selection, deopt + re-promote
   path, cache round-trip across two JVM runs).
2. Existing gates: full CPU DSP regression batch; `run-validation.sh` accuracy on
   CPU; `run-llm-benchmarks.sh --backend cpu --test baseline --models qwen`.
3. Perf acceptance: BGE embedding pass time and elementwise-heavy segment
   microbenchmarks — the tier must beat the interpreter by a stated margin (target:
   ≥1.5× on fused elementwise chains) or the auction simply won't pick it (which is
   also an acceptable outcome — the ledger documents it).
4. Binary-size + build-time delta recorded in the PR description (the toolchain is
   already built today, so expected delta is source-compile only).

## 8. Alternatives considered (liveness checked 2026-07-02 via direct README fetches; web search unavailable)

**Same-slot alternatives (section-codegen JIT):**
- **triton-cpu (CHOSEN)** — official experimental CPU backend, live ("long-lived
  development branch", WIP, no tagged releases → pinned commit is correct). Only
  option consuming the Triton dialect the GPU emitters already produce. Risk:
  experimental status; mitigated by D2 plug-in seam + P0 spike + pinned commit.
- **In-house CpuIRBuilder + upstream MLIR (ADR 0085)** — exists, excluded
  (`HAVE_MLIR=0`). Linalg-based, second emitter to maintain, LLVM-pin conflict with
  the triton-cpu toolchain (D4). KEEP as fallback lowering; enable only after
  rebasing its LLVM onto the triton-cpu pin.
- **IREE (llvm-cpu)** — VERIFIED ALIVE: LF AI & Data project, stable+nightly
  releases, AMD MLPerf 2025 submission, Apache-2+LLVM-exception. Mature CPU codegen
  (data tiling, ukernels). Consumes linalg/StableHLO (pairs with CpuIRBuilder's
  emitter, not the Triton one); heavier, own runtime/artifact model. THE HEDGE if
  triton-cpu stalls — pluggable behind D2.
- **triton-shared (Microsoft)** — README opens "This repository is no longer
  maintained." DO NOT plan around it.

**Graph-backend-slot alternatives (peers of oneDNN/OpenVINO, coarser granularity):**
- **XLA:CPU via PJRT** — cheapest meaningful experiment in this whole space:
  `graph/tpu/PjrtClientManager` + `HloIRBuilder` already exist in-tree; the same
  PJRT client API loads the CPU plugin. Recommended as a P5 side experiment/data
  point, not the primary (whole-segment granularity, no per-section auction).
- **Apache TVM (Relax/MetaSchedule)** — alive; autotuning philosophy matches WS-G,
  but tuning workflow is offline/Python-centric — conflicts with the no-Python
  discipline. Not selected.
- ONNX Runtime — circular for this codebase. Skip.

**Microkernel-slot (COMPLEMENTARY, not alternatives — G4 auction hand-kernel candidates):**
- **libxsmm** — VERIFIED ALIVE (BSD-3, active CI): JIT'd small/batched GEMM
  microkernels, AVX-512/AMX, C API. Direct answer to CPU decode GEMV and batched-
  GEMM groups; needs no compiler integration. Register as auction candidate in
  parallel with this design.
- **oneDNN brgemm/ukernel API** — already linked; same role.
- **XNNPACK** (elementwise/conv microkernels), **KleidiAI** (ARM; pairs with the
  in-tree `ArmHybridGraphBackend.cpp`, currently excluded).

**Dead/avoid:** Glow (archived), PlaidML/TensorComprehensions (dead). BladeDISC:
README fetched but activity unverified — check commit history before any reliance.

## 9. Implementation phases

- **P0 — Spike (de-risk, ~days).** Standalone C++ test (not wired to DSP): build a
  small TTIR module (reuse an emitter or hand-write TTIR text), drive the pinned
  triton-cpu pass pipeline + ORC JIT, execute vs a reference loop. Exit: correct
  output, measured launch overhead, ABI + pass list documented from source.
  NO further investment until P0 passes.
- **P1 — Build restructure (L1, §5).** CPU-eligible Triton sources compile on CPU;
  include-hygiene pass; binary-size delta measured. Exit: CPU build green, no
  runtime behavior change.
- **P2 — CPU target + launcher + cache (§4).** `TritonTargetDispatch` CPU target,
  `CpuKernelLauncher`, feature-keyed disk cache. Exit: sections JIT + execute under
  the new unit tests; cache round-trip across process restart.
- **P3 — Backend-chain wiring (§6).** `TritonCpuGraphBackend`, FunctionalReplayHandle
  schedule, config flags, verify-then-race + scoped deopt. Exit: CPU DSP regression
  batch green with `dspCpuTriton=ON`; BGE + qwen CPU benchmarks recorded vs baseline.
- **P4 — Tuning + auction.** CPU tile profile in `selectTileConfig`; PGO race (H1)
  over CPU block sizes; register under the G1 auction with oneDNN/OpenVINO/
  interpreter/libxsmm candidates. Exit: ledger shows per-section winners; no
  section slower than interpreter survives.
- **P5 — Parallel experiments (optional).** XLA:CPU-via-PJRT data point; CpuIRBuilder
  LLVM-pin rebase study; ARM path (KleidiAI + ArmHybridGraphBackend revival).

## 10. Risks and open questions

| Risk | Mitigation |
|---|---|
| triton-cpu ABI/pass drift across pins (experimental upstream) | Pinned commit + P0 spike as executable spec + D2 plug-in seam (IREE hedge) |
| CUDA include leakage blocks P1 | Explicit include-hygiene pass budgeted; CUDA-only code stays in .cu |
| Two LLVMs in one image (if HAVE_MLIR later) | D4 hard rule: single LLVM pin; rebase CpuIRBuilder before enabling |
| Thread oversubscription (grid pool × OMP × BLAS) | D6 single-owner budget; jitted kernels never spawn threads |
| JIT latency on first request | Disk cache (P2) + `prewarm()` (Part I C2) |
| Binary size growth | Already paying whole-archive MLIR today; measure delta in P1; `SDX_INCLUDE_TRITON` packaging flag exists for SDX distribution decisions |
| Wrong perf bet (tier loses to oneDNN everywhere) | Acceptable by design: auction + ledger document it; scope was chosen (D8) where fusion wins are most likely |

Open questions (answer during P0/P1, do not block on them now):
1. Exact kernel ABI and pass pipeline at pin `c4ccb98` (extract from
   `third_party/cpu` in the vendored source).
2. Do any ELEMENTWISE-section emitters produce GPU-only ops (barriers, async copies)
   that need a CPU-legal variant, or is the elementwise subset already clean?
3. Cache-key granularity for CPU features (per-ISA-tier vs exact cpuid flags).
4. Interaction detail with E1: does the micro-schedule own gap ops between jitted
   sections, or does FunctionalReplayHandle? (Proposed: FunctionalReplayHandle owns
   the whole segment schedule; E1 uses it.)

## 11. References

- `DSP_PERF_HANDOFF.md` — Part I gaps (E1-E5), Part II strategy (G/H/I), WS-L audit.
- ADRs: 0055/0058 (kernel selection + autotune registry), 0061 (DSP execution),
  0085 (MLIR JIT backend), 0089 (CUDA graph capture/replay), 0093 (plan disk
  persistence), 0097 (decode path perf), 0098 (OpenVINO CPU backend).
- Evidence index (verified this session): `Dependencies.cmake:1389-1657, 2005-2008,
  2602-2605`; `MainBuildFlow.cmake:189-224, 871-872`; `buildnativeoperations.sh:
  2688-2705`; `blasbuild/cpu/include/config.h:11-14`; `cuda_stubs.cpp:566-578`;
  `FunctionalReplayHandle.cpp:69-79`; `SectionTypeConfig.h:62-83`;
  `TritonIRBuilder_analysis.cpp:858-952`.
- Upstream (fetched 2026-07-02): triton-lang/triton-cpu README (live, WIP);
  microsoft/triton-shared README (unmaintained); iree-org/iree README (LF AI&Data,
  releases, MLPerf); libxsmm README (active CI, BSD-3).
