# ADR 0117: Triton AMD Lowering for ZLUDA/AMD DSP Dispatch

## Status

Accepted (Jul 2026). Implemented; pending burn-in on AMD hardware.

## Context

DSP dispatches compiled segments to multiple JIT targets (Triton, NVRTC, PTX,
CPU MLIR). The Triton execution path is already target-abstracted: all module
load / kernel lookup / launch flows through `TritonTargetDispatch`
(`TritonGraphBackend_binary.cpp`, `_lru.cpp`, `_kernel.cu`), which detects
NVIDIA/AMD/Intel at runtime and uses the CUDA driver, HIP, or Level Zero
respectively. Build wiring (`Dependencies.cmake` `TRITON_GPU_TARGET`,
`TRITON_CODEGEN_BACKENDS`) already builds LLVM's AMDGPU target and Triton's
`TritonAMDGPUToLLVM` passes for `SD_HIP` or `SD_ZLUDA + ZLUDA_TARGET_BACKEND=AMD`.

Three gaps prevented end-to-end AMD execution:

1. **No HSACO link step.** Phase 6 emitted `CodeGenFileType::AssemblyFile` for
   all targets. PTX text is directly loadable by CUDA; AMDGCN assembly is not
   loadable by `hipModuleLoadData`, which requires a linked executable code
   object (HSACO).
2. **Hardcoded wavefront width.** TTIR→TTGIR set `threadsPerWarp = 32`
   unconditionally; AMD CDNA/GCN (gfx9xx) are 64-wide.
3. **CUDA-only kernel attribute call.** `configureCudaKernelSharedMemory`
   called `cuFuncSetAttribute` on the loaded kernel handle unconditionally —
   invalid on a `hipFunction_t`.

## Decision

Expose Triton lowering for AMD inside DSP by closing those gaps in the
dispatch layer — no new execution mode, no Java surface changes:

1. **AMD codegen emits a relocatable object and links it in-process via
   `amd_comgr`** (`AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE`, ISA name
   `amdgcn-amd-amdhsa--<full gcnArchName>`). comgr ships with every ROCm
   install, which both native HIP and ZLUDA+AMD builds require. Gated on
   `TRITON_HAS_HIP && __has_include(<amd_comgr/amd_comgr.h>)`; a clear
   diagnostic fires when absent. Feature suffixes (`gfx90a:xnack-`) are
   stripped for the LLVM TargetMachine CPU but preserved for the comgr ISA.
2. **Wave width derives from the arch**: 32 for gfx10/11/12 (RDNA) and
   NVIDIA, 64 otherwise (CDNA/GCN).
3. **The >48KB dynamic-shared opt-in is skipped on non-NVIDIA targets**
   (`TritonTargetDispatch::detectTarget()` guard); the shared-memory limit
   check itself still applies via the attributes HIP/ZLUDA report.

ZLUDA remains the intended vehicle: the nd4j CUDA backend runs on ZLUDA while
Triton kernels bypass ZLUDA's PTX-only module loader by compiling AMDGCN and
loading through HIP directly (existing design, `TritonTargetDispatch.cpp`
header comments).

## Addendum: HIP graph replay wired (same day)

The pre-existing `graph/hip/` scaffold (`HipGraphBackend` — per-island
`hipGraph_t`/`hipGraphExec_t` capture via `IslandCapturePolicy::forRocm()`,
dlopen-opaque `HipRuntimeManager`, `HipGraphReplayHandle`) is now wired in:

- Gates widened from `SD_HIP` to
  `SD_HIP || ZLUDA_TARGET_AMD || HAVE_MIOPEN` across all six files (the
  backend/manager are dlopen-opaque; the replay handle needs HIP headers,
  which ZLUDA+AMD builds have).
- `getGpuGraphBackend()` (`NativeDynamicShapePlan_gpubackend.cpp`) gained the
  `GEM_HIP_GRAPHS` branch mirroring TPU/Hexagon: explicit mode selects it,
  AUTO tries it after Triton/NVRTC/PTX/TPU/Hexagon when
  `HipGraphBackend::isAvailable()` (i.e. `libamdhip64.so` dlopens — always
  false on NVIDIA hosts, so AUTO there is unchanged).
- `ModeContract` case 9 now sets `requiresCompilation + needsJitBackend +
  forceSyncDuringCapture + skipFrozenConstsDuringCapture +
  requiresDeterministicCublas` — without `needsJitBackend` the dispatcher
  returned nullptr before reaching the branch (this was the actual unwiring).
- Enum plumbing already existed on both sides (`GEM_HIP_GRAPHS = 9`,
  Java `GraphExecutionMode.HIP_GRAPHS(9)`); no Java changes were needed.

Under ZLUDA+AMD, `hipStreamBeginCapture` on the plan stream records both
ZLUDA-translated launches and directly-launched HIP Triton kernels because
ZLUDA streams are `hipStream_t` underneath — the same identity the Triton
module-loading bypass relies on. This replaces the "no graph replay under
ZLUDA" limitation. First bring-up must validate the stream-identity
assumption and port the CUDA replay invariants review (broad pre-replay
sync consumption, post-replay fixup, external-input address rechecks) if
gaps surface.

## Consequences

- On ZLUDA+AMD builds, DSP's TRITON mode can compile and execute segments on
  AMD GPUs; unsupported sections keep falling back to native ordered
  execution as on NVIDIA. CUDA-graph modes remain unavailable under ZLUDA.
- NVIDIA builds are unaffected (all changes are target-gated; default builds
  don't include HIP/comgr headers).
- Cooperative-launch (grid-sync) sections still fall back to standard launch
  on AMD; multi-phase kernels needing grid sync stay native there.
- Untested on real AMD hardware — first bring-up should run the DSP matrix
  (`run-dsp-matrix.sh`) and `TritonOrderedReductionCorrectnessTest` on an
  RDNA card under ZLUDA.
