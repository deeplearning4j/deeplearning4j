# ADR 0102: Accelerator and CPU-Architecture CI Test Tiers

## Status

Proposed

Proposed by: Adam Gibson (2 Jul 2026)

## Context

`platform-tests` already has tag-based test tiering (`smoke`, `full-ci` via `-Psmoke`/`-Pfull-ci`)
and backend/helper profiles (`test-zluda`, `test-tpu`, `test-onednn`, `test-armcompute`, `test-mps`,
`multi-backend-*`). The GitHub Actions side has build-deploy workflows per platform classifier and
snapshot-consuming test workflows (`run-tests.yml`, `run-cpu-tests-sanity-checks.yml`, etc.).

An audit (see `PLATFORM_TESTS_CI_AUDIT.md`) found the two sides were not reconciled:

1. The `test-zluda` and `test-tpu` profiles selected JUnit tag groups (`zluda,rocm,amd-gpu` / `tpu`)
   that **no test carried** — running them executed zero tests.
2. The `test-zluda` profile never injected `ZLUDA_PATH` into the forked test JVM's
   `LD_LIBRARY_PATH` (the base surefire config hardcodes CUDA toolkit paths), so ZLUDA's
   `libcuda.so` could never intercept the driver.
3. Tags used by profiles/tests (`zluda`, `rocm`, `amd-gpu`, `tpu`, `multi-backend`, `multi-device`,
   `backend-discovery`) had no `TagNames` constants.
4. `linux-x86_64-avx512` artifacts are built in CI but never executed: GitHub-hosted x64 runners are
   a hardware lottery (AMD EPYC 7763 has no AVX-512).
5. Test workflows passed `-DexcludedGroups=...`, which is silently ignored because the pom binds
   surefire `excludedGroups` to `${excludedTests}`; likewise `export OMP_NUM_THREADS=1` before
   `mvn test` is overridden by the pom's `environmentVariables` (`${omp.num.threads}`, default 32).

## Decision

### Tag layer

- Add `TagNames` constants: `ZLUDA`, `ROCM`, `AMD_GPU`, `TPU`, `HEXAGON`, `MULTI_BACKEND`,
  `MULTI_DEVICE`, `BACKEND_DISCOVERY`. Tag semantics: *accelerator tags mark environment
  requirements*; the existing `smoke`/`full-ci` tags mark *cost tiers*. A test may carry both.
- Accelerator-tagged tests must self-gate with JUnit assumptions on their environment variables
  (`ZLUDA_PATH`, `PJRT_PATH`/`TPU_LIBRARY_PATH`) so unfiltered runs skip rather than fail.

### ZLUDA tier

- `ZludaSmokeTest` (tags `zluda,rocm,amd-gpu`) exercises what ZLUDA v6 supports: backend init,
  elementwise ops, cuBLAS GEMM vs a pure-Java reference, reductions/transforms. cuDNN and CUDA
  graph capture/replay are out of scope — ZLUDA does not reliably support stream capture, so DSP
  `CUDA_GRAPHS` mode must not be routed through ZLUDA.
- `test-zluda` profile now forwards `ZLUDA_PATH`/`ZLUDA_TARGET` and prepends
  `$ZLUDA_PATH[:$ZLUDA_PATH/lib]` to the forked JVM's `LD_LIBRARY_PATH`, and enables JavaCPP
  `pathsFirst` + a ZLUDA-first `java.library.path` so ZLUDA's cublas/nvrtc drop-ins can intercept
  the JavaCPP-extracted NVIDIA libraries. Groups are overridable via `-Dzluda.test.groups`.
- CI: `run-zluda-smoke-tests.yml` on a self-hosted RDNA GPU runner (labels
  `[self-hosted, linux, amd-gpu]`); downloads a pinned ZLUDA release and normalizes the layout to
  the `ZLUDA_PATH/lib/libcuda.so` convention `JZludaBackend` expects. There are no hosted AMD GPU
  runners, and AMD's CDNA cloud (MI300X) is not a ZLUDA target — consumer RDNA hardware is required.

### TPU tier

- `JTpuBackend.canRun()` is a stub (`false`) until PJRT bindings land, so the tier validates the
  layers that exist: classpath + SPI registration, in-process `dlopen` of a PJRT/libtpu library,
  and the `Nd4jTpuHelper` availability contract (`TpuBackendSmokeTest`, tag `tpu`).
- Zero-hardware smoke: `pip install libtpu` provides the same `libtpu.so` used on Cloud TPU VMs
  (TPU VMs are x86_64 hosts), which loads on any hosted runner. `run-tpu-smoke-tests.yml` runs this
  on `ubuntu-24.04` on every dispatch, plus an optional self-hosted Cloud TPU VM job (spot
  v5litepod-1 ≈ $0.5/hr; only one process may hold the TPU — surefire stays at `forkCount=1`).

### Hexagon tier

- Same treatment as TPU: `nd4j-hexagon` had the identical `org.nd4j` groupId bit-rot and
  old-interface scaffolding; repaired to current APIs (`HexagonExecutioner` skeleton throwing
  until hexagon-mlir bindings land, `HexagonEnvironment`/`HexagonOpContext` placeholders,
  `canRun()==false` locked by `HexagonBackendSmokeTest`). New `-Phexagon` reactor profile
  (nd4j-hexagon + nd4j-hexagon-preset), `-Ptest-hexagon` platform-tests profile (group `hexagon`,
  `HEXAGON_MLIR_PATH` wiring), published from the cross-platform linux leg. hexagon-mlir is BSD-3
  open source (Qualcomm, Dec 2025), so the native `libnd4jhexagon.so` build (committed
  `SD_HEXAGON` CMake routing + `graph/hexagon/*` sources) is a from-source-buildable follow-up.

### CPU-architecture tiers

- arm64: native, free GitHub-hosted `ubuntu-*-arm` runners (already used by existing workflows).
- AVX-512: `run-avx512-sde-tests.yml` runs the `smoke`/`full-ci` tiers against the
  `linux-x86_64-avx512` snapshot artifacts under Intel SDE (`sde64 -skx --`) when the host lacks
  AVX-512, injected via the existing `-Dtest.prefix`/`bin/java` wrapper mechanism (the wrapper
  passes unknown prefixes through verbatim). SDE is 5–20x slower: correctness only, never for
  performance claims.
- ppc64le/s390x (IBM hosted-runner OSS program) and riscv64 (RISE runners,
  `ubuntu-24.04-riscv`) are documented as follow-ups, not implemented.

### Native build integration (TPU, ZLUDA, Hexagon)

- **TPU native**: `pjrt_c_api.h` vendored in-tree (`include/external/pjrt/`, pinned openxla
  commit); `BuildTPU.cmake`/`TpuConfiguration.cmake` restructured to the MainBuildFlow
  `target_sources` pattern; `-Dlibnd4j.chip=tpu` → `libnd4jtpu.so` (~192MB), libtpu dlopen'd at
  runtime (no link-time proprietary dep, verified via `ldd`), no python. `build-deploy-linux-tpu.yml`.
- **ZLUDA native**: `--zluda <ON|AMD|INTEL|AUTO>` in `buildnativeoperations.sh` →
  `-DSD_ZLUDA=ON [+ -DSD_ZLUDA_TARGET]` → existing `ZludaConfiguration.cmake`; `-Dlibnd4j.zluda`
  pom passthrough. Rides the **cuda** chip build (binary drop-in, no separate lib). CI-only
  validation (`build-deploy-linux-zluda.yml`, `install` not `deploy`) — a local build is avoided
  because a global `SD_ZLUDA` define would invalidate the CUDA ccache; a published ZLUDA-flavored
  artifact would need its own classifier, not the stock CUDA coordinates.
- **Hexagon native**: `-Phexagon` reactor + `libnd4j.chip=hexagon` → `libnd4jhexagon.so`,
  `BuildHexagon.cmake` modernized like BuildTPU, hexagon-mlir runtime dlopen'd (BSD-3, no SDK at
  build time). `build-deploy-linux-hexagon.yml`.
- **AMD via ROCm PJRT (the recommended AMD path, not a CUDA-compat bridge)**: AMD is NOT a separate
  backend. The ROCm PJRT plugin `xla_rocm_plugin.so` (from the `jax-rocm7-pjrt` wheel) is loaded by
  the SAME native `PjrtClientManager` as TPU — the uniform `GetPjrtApi()` C ABI. `PjrtClientManager`
  was generalized (`.cpp`-only, TPU target) to resolve any plugin via `PJRT_PLUGIN_LIBRARY_PATH` /
  `ROCM_PJRT_PATH` (direct `.so` or a dir), falling back to the libtpu defaults. This sidesteps the
  CUDA-Graphs-on-AMD problem that sinks ZLUDA/HIP because XLA owns graph scheduling, and covers
  **both CDNA (MI300X) and RDNA**. Plugin obtained **python-free** via
  `libnd4j/scripts/fetch-pjrt-plugin.sh rocm` (curl+jq+unzip of the wheel). `RocmPjrtSmokeTest`
  (`-Ptest-rocm`, tags `rocm,amd-gpu`) validates existence + `GetPjrtApi` export (pure-Java ELF
  dynsym reader — no toolchain) anywhere; real in-process load needs a ROCm 7 host (deps:
  `libamdhip64.so.7`/`librocblas.so.5`/`libMIOpen.so.1`). `run-rocm-pjrt-smoke-tests.yml` is
  two-tier (hosted static-validate + optional self-hosted AMD). Limit: carries only XLA/HLO-lowerable
  ops (the SameDiff/decode subgraph), NOT the custom `.cu` kernel library — that gap needs SCALE or a
  HIP port (see the compile-time-bridge research).
- CPU/CUDA regression builds pass after the shared-tree-free native changes (only chip-specific
  sources + cmake touched).

### Mixed-runtime replay (islands + gaps) across backends

- `IslandCapturePolicy` (`libnd4j/include/graph/IslandCapturePolicy.{h,cpp}`) — backend-neutral
  partitioner that splits a slot range into capturable islands + eager gaps from op-traits + node
  counts, with per-backend profiles: `forCuda()` (permissive — today's behavior), `forRocm()`
  (≤128-node islands, attention/host-callback/dynamic-index → gaps, matching vLLM PIECEWISE +
  ROCm/hip #3887's stale-pointer threshold), `forMetal()` (≤64-node islands for the MTLIndirect­
  CommandBuffer command cap, but **attention stays captured** because MLX fuses it via
  `mx::fast::scaled_dot_product_attention` — the key Apple/ROCm divergence). Proven by a
  standalone gtest: 25/25 (no GPU needed). This generalizes the CUDA backend's implicit
  `perIslandSafetyBytes` bail-out into an explicit policy and is what lets hipGraph/Metal-ICB
  replay survive their platform bugs by sizing islands under the safe threshold.
- `HipGraphBackend`/`HipRuntimeManager` (`libnd4j/include/graph/hip/`, `#ifdef SD_HIP`) — the ROCm
  GraphBackend on the same seam, dlopen-opaque `libamdhip64` (no ROCm headers, compiles on Linux),
  per-island `hipGraphLaunch` replay + eager gaps, consuming `IslandCapturePolicy::forRocm()`.
  Syntax-verified here; live replay + the slot-executor capture-stream injection validate on AMD.
- Apple/Metal: `MlxGraphBackend` (island graph via MLX `mx::compile`) + `MetalReplayHandle`
  (`MTLIndirectCommandBuffer` replay) + MPS per-op helpers (gaps) already realize the same
  mixed runtime natively — no bit-rot (already on current `NativeSlot` sub-structs). Gap: the
  JavaCPP MLX preset must be generated on macOS. jax-metal PJRT is NOT viable (unmaintained,
  not dlopen-able) — Apple is native MLX, not PJRT-reuse. `nd4j-metal`/`-preset` Java stubs +
  `-Ptest-metal` + `MetalBackendSmokeTest` + `run-mlx-smoke-tests.yml` (macos-14) built here.

### Workflow fixes

- `-DexcludedGroups=...` → `-DexcludedTests=...` and `-Domp.num.threads=1` in all four test
  workflows; `run-cpu-tests-sanity-checks.yml` gains a `testTier` input (`classes|smoke|full-ci`)
  wiring the tag tiers into CI.

## Consequences

### Advantages

- `-Ptest-zluda` / `-Ptest-tpu` now select real tests with correct environment wiring; ZLUDA and
  TPU readiness can be exercised from GitHub Actions today (TPU without hardware).
- AVX-512 artifacts get deterministic execution coverage for the first time.
- Group exclusions in CI actually take effect (previously the full suite minus nothing was eligible).

### Disadvantages / risks

- ZLUDA interception of JavaCPP-extracted `libcublas`/`libnvrtc` (absolute-path `System.load`)
  is best-effort via `pathsFirst`; may require OS-level library replacement on the runner. The
  smoke test reports what was actually loaded to make iteration cheap.
- The TPU tier locks the current stub contract (`canRun()==false`); when PJRT bindings land the
  smoke test fails intentionally, forcing an upgrade to real device-enumeration checks.
- SDE emulation cannot validate performance, only correctness.
