# Platform-Tests + GitHub Actions Audit — Tags, Multi-CPU, ZLUDA, TPU

Date: 2026-07-02 · Branch: `ag_new_release_updates_2` · Companion ADR: `ADRs/0102 - Accelerator and CPU-Architecture CI Test Tiers.md`

This document is (1) an audit of the platform-tests tagging/profile architecture and the GitHub
Actions coverage, (2) research findings on what is available for ZLUDA / TPU / multi-CPU testing
from GitHub Actions in 2026, and (3) a record of the changes made to close the gaps.

---

## 1. platform-tests architecture (as audited)

### 1.1 Tag inventory

`TagNames` (`nd4j/nd4j-common-tests/src/main/java/org/nd4j/common/tests/tags/TagNames.java`) defines
31 constants (now 38). Three orthogonal families are in use:

| Family | Tags | Purpose |
|---|---|---|
| **Cost tier** | `smoke` (~20 classes, <30s, no GPU/downloads), `full-ci` (~27 classes, ~5 min), untagged = nightly/weekly | CI runtime budget selection via `-Psmoke` / `-Pfull-ci` (`surefire.groups`) |
| **Resource/exclusion** | `long-running-test` (~48 files), `large-resources` (~45), `downloads` (~14), `multi-threaded`, `manual`, `needs-verify` | Excluded on low-spec CI via `-DexcludedTests=...` |
| **Domain/environment** | `samediff`, `rng`, `java-only`, `file-io`, `spark`, `keras`, `onnx`, `tensorflow`, … plus string-literals `dsp`(15), `sparse`(6), `gnn`(6), `multi-backend`(5), `multi-gpu`(3), `vlm`, `cuda`, `cudnn`, `onednn`, `armcompute`, … | Domain slicing; environment requirements |

**Findings (tags):**

- **F1.** ~26 tags used as string literals had no `TagNames` constants (`dsp`, `sparse`, `gnn`,
  `multi-backend`, `multi-gpu`, `zluda`, `tpu`, …). → *Partially fixed:* added constants for the
  seven CI-relevant ones (`ZLUDA`, `ROCM`, `AMD_GPU`, `TPU`, `MULTI_BACKEND`, `MULTI_DEVICE`,
  `BACKEND_DISCOVERY`). `dsp`/`sparse`/`gnn` remain literal-only (domain tags; consolidation is a
  cheap follow-up).
- **F2. (critical)** The pom profiles `test-zluda` (groups `zluda,rocm,amd-gpu`) and `test-tpu`
  (group `tpu`) selected tags that **zero tests carried** — both profiles ran 0 tests. → *Fixed:*
  new `ZludaSmokeTest` and `TpuBackendSmokeTest` in
  `platform-tests/src/test/java/org/eclipse/deeplearning4j/nd4j/backends/`.

### 1.2 Profiles (platform-tests/pom.xml)

Tier: `smoke`, `full-ci` (set `surefire.groups`). Helper: `test-onednn`, `test-cudnn`,
`test-armcompute`, `test-mps`, `test-cpu-only`, `test-autotuning`, `test-helper-comparison`.
Backend/topology: `cpu-dep`/`cpu-profile` (`-Dbackend.artifactId=nd4j-native`), `multi-backend`,
`multi-backend-dual` (CPU+CUDA), `multi-backend-all` (CPU+CUDA+ZLUDA+TPU), `test-zluda`
(env-activated by `ZLUDA_PATH`), `test-tpu`, `test-device-routing`, `native-tests[-cuda]`,
`all-tests`/`skip-native-tests`, `jemalloc`.

Key plumbing facts a CI author must know:

- **Backend swap** is dependency-level: `-Dbackend.artifactId=nd4j-native|nd4j-cuda-12.9` +
  `-Dplatform.classifier=<javacpp platform[-extension]>`. Default is **nd4j-cuda-12.9** — CPU CI
  must always pass the override.
- **`excludedGroups` is bound to `${excludedTests}`** (pom line ~1606). `-DexcludedGroups=...` on
  the CLI is silently ignored. Use `-DexcludedTests=...`.
- **Forked-JVM env is pom-controlled** (surefire `<environmentVariables>`): `OMP_NUM_THREADS` etc.
  come from `${omp.num.threads}` (default 32) — shell `export` before `mvn test` does not reach the
  test JVM. Use `-Domp.num.threads=N`.
- **`LD_LIBRARY_PATH` was hardcoded** to CUDA toolkit paths for the forked JVM (line ~1617) — this
  is why ZLUDA interception could not work (see §3).
- **`bin/java` wrapper** (`<jvm>${project.basedir}/bin/java</jvm>`): special-cases
  valgrind/asan/compute-sanitizer/nsys/nvprof prefixes from `-Dtest.prefix`
  (`TEST_RUNNER_PREFIX`), and passes **any other prefix through verbatim** — which is what makes
  the Intel SDE AVX-512 leg possible with zero wrapper changes (§5.2).
- `forkCount=1`, `reuseForks=false`, heap 8g / off-heap 24g defaults (CI overrides to 4g/6g).

### 1.3 CPU architecture/extension selection

- `-Dlibnd4j.extension=avx2|avx512` → javacpp classifier suffix (`linux-x86_64-avx2`, …);
  avx512 maps to `-march=skylake-avx512`; avx2 auto-adds onednn to helpers.
- Helpers: `onednn, cudnn, armcompute, mlir, mps, accelerate, miopen, pjrt` (`-Dlibnd4j.helpers`).
- Platforms handled by `buildnativeoperations.sh`: linux-x86_64, windows-x86_64, linux-ppc64le,
  linux-armhf, linux-arm64, android-{arm64,x86,x86_64}, macosx-arm64.
- Backend parameterization in tests: `BaseNd4jTestWithBackends#configs()` →
  `ServiceLoader<Nd4jBackend>` filtered by `canRun()` and `-Dbackends=`. Note `JCublasBackend.canRun()`
  **throws** on zero devices; CPU-only runs must remove the CUDA jar from the classpath
  (`-Dbackend.artifactId=nd4j-native`), not rely on priorities.

---

## 2. GitHub Actions audit

18 workflows, **all `workflow_dispatch`-only** (no push/PR/schedule triggers anywhere).
Remote = `deeplearning4j/deeplearning4j` (canonical org repo).

### 2.1 Coverage matrix (before this change)

| Target | Built? | Tested? | Notes |
|---|---|---|---|
| linux-x86_64 (+onednn/avx2/avx512/compile, centos-compat) | ✅ stable | ⚠️ test workflows exist but **never ran** in the last 60 recorded runs | `build-deploy-linux-x86_64[-compat].yml` |
| linux-arm64 (+armcompute/onednn/compile) | ✅ | ⚠️ same | native `ubuntu-22.04-arm` runners |
| macosx-arm64 (+mps/compile) | ✅ | ⚠️ same | `macos-14` |
| windows-x86_64 (+onednn/avx2/avx512) | ✅ (4/5 recent) | ❌ | MSYS2/MinGW |
| android-arm64 (+armcompute/nnapi), android-x86_64 | ✅ | ❌ | NDK r27d |
| cuda 12.6 / 12.9 linux | ⚠️ unstable — mostly **cancelled** (1 success/11) | ❌ | OOM/timeout on hosted runners |
| cuda 12.6 / 12.9 windows | ❌ **0 successes in all recorded runs** | ❌ | consistently failing |
| macosx-x86_64 | ❌ deleted 2026-05 | ❌ | revivable on `macos-15-intel` (available until Aug 2027) |
| linux-armhf, android-arm32/x86, Jetson cuda-arm64 | ❌ deleted | ❌ | |
| linux-ppc64le, s390x, riscv64 | ❌ never | ❌ | free OSS runner programs exist (§5.3) |
| ROCm/AMD (ZLUDA), TPU | ❌ none | ❌ | → fixed by this change (dispatch workflows) |

Test workflows (`run-tests.yml` 16-suite dispatcher, `run-cpu-tests-sanity-checks.yml` 12-platform
matrix, `run-cpu-integration-tests.yml`, `run-gpu-tests-sanity-checks.yml` self-hosted, legacy
`cpu-sanity-check-tests.yaml`) consume **snapshot native jars** from
`central.sonatype.com/repository/maven-snapshots/` and build only Java modules
(`-pl :platform-tests --also-make -pl !:libnd4j`) — a sound architecture; they simply were not
being executed, and used `-Dtest=` class lists rather than the tag tiers.

### 2.2 Bugs found in the test workflows (now fixed)

- **F3.** `-DexcludedGroups="long-running-test,large-resources,downloads"` was a **no-op** in all
  four test workflows (pom binds that surefire parameter to `${excludedTests}`) — heavy tests were
  never actually excluded. → replaced with `-DexcludedTests=...`.
- **F4.** `export OMP_NUM_THREADS=1` before `mvn test` was **overridden** by the pom's surefire
  `<environmentVariables>` (default 32 threads — thread oversubscription on 3–4 core runners).
  → replaced with `-Domp.num.threads=1`.
- **F5.** No workflow used the `smoke`/`full-ci` tag tiers. → `run-cpu-tests-sanity-checks.yml`
  gained a `testTier` input (`classes|smoke|full-ci`).
- **F6.** Legacy `cpu-sanity-check-tests.yaml` (checkout@v2, openblas 0.3.19, full unfiltered
  `mvn clean test`) and `test_multiple_arch.yaml` (CUDA 11.x cache warmer) are vestigial —
  recommend deletion (left in place; user call).

---

## 3. ZLUDA — findings and what was implemented

### 3.1 State of ZLUDA (mid-2026)

- **v6** (≈June 2026), Apache-2.0/MIT, back to an unfunded weekend project after the AMD (2022–24)
  and anonymous-sponsor (→2025) phases. Blog: <https://vosen.github.io/ZLUDA/blog/zluda-update-q1q2-2026/>
- GPUs: **RDNA1–RDNA4 consumer only** (ROCm 6.3/6.4, ROCm 7 supported since Q4 2025).
  **CDNA (MI250/MI300X) is not a target** — AMD Developer Cloud / MI300X rentals are useless for ZLUDA.
- API coverage: driver API mostly complete; **NVRTC/PTX JIT is the most mature path** (llama.cpp
  CUDA backend fully works, ~native-ROCm perf); **cuBLAS partial** (~44 fns via rocBLAS, added Q3
  2025); **cuDNN incomplete** (MIOpen mapping, known failures); cuSPARSE crate exists, depth unknown;
  **CUDA Graphs: no reliable stream-capture support** (`cuGraphExecUpdate_v2` merged, other
  instantiation entry points open) — **DSP `CUDA_GRAPHS` mode must not be routed through ZLUDA**.
- PyTorch still not working under ZLUDA (their 2026 goal, vLLM-focused).

### 3.2 nd4j-specific integration facts

- The repo already had: `nd4j-zluda` module (SPI backend, priority 90, delegates to
  `JCublasNDArray`), reactor profiles `-Pzluda[-amd|-intel]`, `test-zluda` pom profile
  (env-activated by `ZLUDA_PATH`), a `ZLUDA_PATH` intercept in `JCublasBackend.canRun()`, and
  ADR 0087. What was missing: tagged tests (F2), and env wiring (below).
- **F7. (critical)** The forked test JVM's `LD_LIBRARY_PATH` was hardcoded to CUDA toolkit paths, so
  ZLUDA's `libcuda.so` could never be resolved. → *Fixed:* `test-zluda` now overrides
  `LD_LIBRARY_PATH` to `$ZLUDA_PATH:$ZLUDA_PATH/lib:<cuda paths>` and forwards
  `ZLUDA_PATH`/`ZLUDA_TARGET`.
- `libcuda.so.1` is loaded from the **system** (never bundled by JavaCPP) → `LD_LIBRARY_PATH`
  interception works for the driver. But `libcublas`/`libnvrtc`/`libcudart` are extracted from the
  preset jars and loaded by **absolute path** — ZLUDA's drop-in replacements for those need JavaCPP
  `pathsFirst` + a ZLUDA-first `java.library.path` (now set by the profile) and matching versioned
  sonames in the ZLUDA dir; worst case, replace the libs at the OS level on the runner. Expect one
  iteration on real hardware; `ZludaSmokeTest#zludaEnvironmentReport` prints everything needed.
- **F8.** `JZludaBackend.checkZludaAvailable()` only accepts `ZLUDA_PATH/lib/libcuda.so`; ZLUDA
  releases unpack flat. The workflow normalizes the layout (creates `lib/` symlink); the smoke test
  warns on mismatch.
- **F9.** ZLUDA consumes the **PTX** embedded in fatbins. Verify snapshot artifacts embed PTX
  (`cuobjdump --list-ptx libnd4jcuda.so`); if the CUDA build ever switches to `-real`-only
  architectures, ZLUDA (and forward-compat) breaks.

### 3.3 CI recipe (implemented: `.github/workflows/run-zluda-smoke-tests.yml`)

- **No hosted AMD GPU runners exist** (GitHub's GPU runners are NVIDIA T4, private-repo only).
  ZLUDA itself tests on self-hosted runners (`gpu_small`/`gpu_large`); PyTorch/TheRock use
  AMD-provided self-hosted runners. Cheapest practical: a one-time ~$400 RDNA3 box (RX 7800 XT)
  registered as `[self-hosted, linux, amd-gpu]` with ROCm 6.4.
- Workflow: downloads a pinned ZLUDA release (or `system`), normalizes layout, builds Java modules
  + `nd4j-zluda` from source with CUDA snapshot jars, runs `-Ptest-zluda` (default
  `-Dtest=ZludaSmokeTest`; broaden with `testGroups=smoke` → runs the generic smoke tier under the
  ZLUDA environment via the new `-Dzluda.test.groups` override).
- Local: `ZLUDA_PATH=/opt/zluda platform-tests/run-zluda-smoke-tests.sh`.

Alternatives if ZLUDA proves too immature: **SCALE** (Spectral Compute; recompiles CUDA for AMD,
v1.7 May 2026, commercial license for production) and **HIPIFY** (source port; large effort).
chipStar targets Intel/SPIR-V, not practical here.

---

## 4. TPU — findings and what was implemented

### 4.1 State of the nd4j TPU backend

`nd4j-tpu` + `nd4j-tpu-preset` exist (reactor `-Ptpu`), SPI-registered, but **stubs**:
`JTpuBackend.canRun()` hardcodes `false` ("until native bindings are ready");
`Nd4jTpuHelper.isTpuAvailable()` just checks `PJRT_PATH` non-empty; PJRT is listed as a libnd4j
helper (`pjrt`) but unbound. ADR 0072 covers the design.
**F10.** `JTpuBackend.getPriority()` = `BACKEND_PRIORITY_GPU + 10` (110) — *above CUDA*. Harmless
while `canRun()` is false, but the day bindings land, a TPU-present machine would silently outrank
CUDA. Revisit before enabling.

### 4.2 What's available for TPU CI (2026)

- **No hosted/third-party TPU runners exist.** Pattern used by JAX (self-hosted runners on
  Google-managed TPU VMs, labels like `linux-x86-ct5lp-224-8tpu`) and PyTorch/XLA.
- Cheapest real hardware: **v5litepod-1** (`ct5lp-hightpu-1t`) on-demand ≈$1.60/chip-h, spot
  ≈$0.48–0.64/h, per-second billing → a 10-min smoke ≈ $0.10–0.30. v6e has 0 default quota.
  **TPU Research Cloud** grants free quota (30-day renewable; publish-results terms).
- TPU VMs are **ordinary x86_64 Ubuntu hosts** — JDK + GitHub runner install normally. Gotchas:
  only **one process** may hold the TPU (`forkCount=1`), image sets `LD_PRELOAD=tcmalloc` (unset it),
  spot VMs can vanish mid-job.
- **Zero-hardware smoke:** `pip install libtpu` ships the real `libtpu.so` (x86_64) — it *loads*
  anywhere even without TPU hardware; the PJRT CPU plugin inside jaxlib wheels is the fallback.
  Non-Python precedents for dlopen'ing PJRT directly: gomlx/gopjrt (Go), Elixir EXLA, Reactant.jl.

### 4.3 Implemented

- `TpuBackendSmokeTest` (tag `tpu`): SPI/classpath registration check, in-process `System.load` of
  the PJRT/libtpu `.so` (from `-Dpjrt.path`/`-Dtpu.library.path` or `PJRT_PATH`/`TPU_LIBRARY_PATH`),
  and the `Nd4jTpuHelper` contract. Locks `canRun()==false` so the stub can't silently "activate" —
  when bindings land the test fails on purpose, demanding real device-enumeration checks.
- `test-tpu` profile now forwards `pjrt.path`/`tpu.library.path` ↔ `PJRT_PATH`/`TPU_LIBRARY_PATH`
  to the forked JVM; groups overridable via `-Dtpu.test.groups`.
- `.github/workflows/run-tpu-smoke-tests.yml`: hosted `pjrt-cpu-smoke` job (pip libtpu → dlopen
  smoke on `ubuntu-24.04`, no hardware) + optional `tpu-vm-smoke` job for a self-hosted TPU VM
  runner (input `tpuRunnerLabels`).
- Local: `platform-tests/run-tpu-smoke-tests.sh` (auto-discovers libtpu via python3).

Next real step for TPU (out of scope here): bind the PJRT C API (`xla/pjrt/c/pjrt_c_api.h`) via the
`nd4j-tpu-preset` JavaCPP preset and flip `canRun()` to enumerate devices — the smoke tier and CI
workflow are already in place to validate it.

---

## 5. Multi-CPU coverage — findings and recommendations

### 5.1 Hosted runner reality (2026)

- x64 `ubuntu-24.04`: 4 cores/16GB, **CPU lottery** (AMD EPYC 7763 = AVX2-only vs some Intel) —
  AVX-512 is non-deterministic (actions/runner#1069). AVX2 universal.
- arm64 `ubuntu-24.04-arm` / `ubuntu-22.04-arm`: free for public repos (GA Aug 2025), Azure Cobalt
  100 (Neoverse N2, SVE2). Already used by the arm64 workflows. `windows-11-arm` also free.
- macOS: `macos-14/15` arm64; **`macos-13` retired Dec 2025** → x64 macs live on `macos-15-intel`
  until Aug 2027 (option to revive the deleted mac-x86_64 build).
- GPU runners: NVIDIA T4, $0.052/min, private repos — a paid option to unblock the failing hosted
  CUDA test story.

### 5.2 Deterministic AVX-512 (implemented)

`run-avx512-sde-tests.yml`: builds Java modules against `linux-x86_64-avx512` snapshot jars and runs
the `smoke`/`full-ci` tier under **Intel SDE** (`sde64 -skx --`, or clx/icl/spr) when the host lacks
AVX-512 — injected through the existing `-Dtest.prefix` → `TEST_RUNNER_PREFIX` → `bin/java` pass-through.
5–20x slowdown; correctness only. Precedent: llama.cpp CI. This is the first time the avx512
artifacts get executed anywhere.

### 5.3 Other architectures (recommended, not implemented)

| Target | Path | Practicality |
|---|---|---|
| ppc64le | **IBM hosted-runner OSS program** (apply via IBM/actionspz "New Project" issue) — free native runners; JDK available | High — but needs onboarding + a `linux-ppc64le` build workflow (deleted years ago; `buildnativeoperations.sh` still supports it) |
| s390x | Same IBM program | Low priority |
| riscv64 | **RISE runners**: install `rise-risc-v-runners` GitHub App → `runs-on: ubuntu-24.04-riscv` (Scaleway EM-RV1, 4×1.85GHz) — free for OSS | Medium — cold libnd4j build may exceed limits on 4 slow cores; needs sccache warm job first; Temurin JDK 21 exists for riscv64 |
| QEMU emulation | `uraimo/run-on-arch-action` etc. | **Not viable for libnd4j builds** (5–10x; bytedeco abandoned it once native arm64 runners shipped). Only for running prebuilt smoke tests |

### 5.4 Caching

Native build workflows already use a patched **sccache** (GHA cache backend, retry layer; ADR 0095).
Note GitHub's cache is 10GB free (paid beyond, since Nov 2025) — keep per-(platform,extension) keys,
and prefer a scheduled warm job + `restore-only` PR reads if PR triggers are ever enabled.

---

## 6. Changes made (this audit)

**Tests/tags/pom**
- `nd4j/nd4j-common-tests/.../TagNames.java` — +7 constants (`ZLUDA`, `ROCM`, `AMD_GPU`, `TPU`,
  `MULTI_BACKEND`, `MULTI_DEVICE`, `BACKEND_DISCOVERY`).
- `platform-tests/src/test/java/org/eclipse/deeplearning4j/nd4j/backends/ZludaSmokeTest.java` — new.
- `platform-tests/src/test/java/org/eclipse/deeplearning4j/nd4j/backends/TpuBackendSmokeTest.java` — new.
- `platform-tests/pom.xml` — `test-zluda`: LD_LIBRARY_PATH/ZLUDA_PATH/ZLUDA_TARGET wiring, JavaCPP
  pathsFirst, `zluda.test.groups`/`zluda.target` properties; `test-tpu`: PJRT_PATH/TPU_LIBRARY_PATH
  wiring, `tpu.test.groups` property.
- `platform-tests/run-zluda-smoke-tests.sh`, `platform-tests/run-tpu-smoke-tests.sh` — new runner
  scripts (match `run-smoke-tests.sh` conventions).

**Workflows**
- `.github/workflows/run-zluda-smoke-tests.yml` — new (self-hosted AMD/Intel GPU).
- `.github/workflows/run-tpu-smoke-tests.yml` — new (hosted PJRT-CPU smoke + optional TPU VM job).
- `.github/workflows/run-avx512-sde-tests.yml` — new (deterministic AVX-512 via Intel SDE).
- `run-cpu-tests-sanity-checks.yml` — `testTier` input (smoke/full-ci), `-DexcludedTests` fix,
  `-Domp.num.threads=1` fix.
- `run-cpu-integration-tests.yml`, `run-gpu-tests-sanity-checks.yml`, `run-tests.yml` — same two fixes.

**Docs**
- `ADRs/0102 - Accelerator and CPU-Architecture CI Test Tiers.md` — new.
- This file.

**Backend module repair (nd4j-tpu was unbuildable)**
- `nd4j-tpu/pom.xml` — parent + dependency groupIds were the pre-Eclipse `org.nd4j` (resolved a stale
  parent from `.m2`; the module could not even be read by Maven since the namespace migration —
  profile-gating hid it). Fixed to `org.eclipse.deeplearning4j`; `junit:junit` → managed JUnit 5;
  added the `nd4j-tpu-preset` dependency (mirrors nd4j-cuda/-preset structure).
- `JTpuBackend` — implements current `allowsOrder()`; `getEnvironment()` throws until PJRT lands.
- `JTpuNDArray` (abstract scaffold), `TpuExecutioner` (current-API skeleton, all exec paths throw
  "PJRT bindings not yet implemented"), `TpuEnvironment`/`JTpuNDArrayFactory`/`TpuOpContext`
  (placeholders decoupled from fast-moving interfaces) — the module now compiles and tracks the
  real `OpExecutioner` contract. `Nd4jTpuPresets` — `OpExclusionUtils.processOps` rename;
  `nd4j-tpu-preset/pom.xml` — removed the self-referencing platform-classified dep (cannot exist
  until a native TPU lib is published). `Nd4jTpuHelper.isTpuAvailable()` hardened against leaked
  `${env.PJRT_PATH}` Maven placeholders.
- Verified: all modules build; TPU smoke tier green locally (2 pass, 1 skip without libtpu).

**Build-cycle + CI integration (subagent work)**
- `build-deploy-cross-platform.yml` — the linux-x86_64 leg now activates `-Pzluda,tpu,hexagon`, so
  `nd4j-zluda`, `nd4j-tpu(-preset)`, `nd4j-hexagon(-preset)` snapshots actually get published
  (previously the snapshot-consuming test workflows could never resolve them). Other OS legs
  unchanged.
- **Hexagon enabled with the same playbook as TPU**: `nd4j-hexagon/pom.xml` had the identical
  stale-`org.nd4j` dependency groupIds + unmanaged junit; `HexagonEnvironment`/`HexagonOpContext`
  → placeholders, `HexagonExecutioner` → current-API skeleton (throws until hexagon-mlir bindings
  land), `HexagonBackend` gets `allowsOrder()` + guarded `getEnvironment()`, preset
  `processOps` fix. New: `-Phexagon` reactor profile, `-Ptest-hexagon` platform-tests profile
  (`hexagon.test.groups`/`HEXAGON_MLIR_PATH`), `TagNames.HEXAGON`, `HexagonBackendSmokeTest`
  (SPI registration + `canRun()==false` lock + optional runtime dlopen + device-info contract),
  `run-hexagon-smoke-tests.sh`. hexagon-mlir is BSD-3 open source — better than TPU's binary
  libtpu; the native build is genuinely from-source-able.
- `build-zluda-validation.yml` — new hosted workflow, no python/GPU/CUDA-toolkit: builds the
  zluda+tpu modules from source, gates on **PTX evidence** in the published `libnd4jcuda.so`
  (readelf `.nv_fatbin` sections + `strings` PTX markers; slim `cuda-cuobjdump-12-9` apt package
  documented as the definitive alternative), statically verifies a pinned ZLUDA release exports
  `cuInit`/`cuDeviceGetCount`/`cuModuleLoadData`, and uploads a SUMMARY.md artifact.
- `run-zluda-smoke-tests.yml` — advisory steps added: NVRTC stub presence in the ZLUDA install and
  PTX presence in the nd4j-cuda jar (warnings only; the AMD runner's real execution is the gate).
- TPU **native** build — **WORKING**: `pjrt_c_api.h` vendored at `libnd4j/include/external/pjrt/`
  (pinned openxla commit, `libnd4j/scripts/vendor-pjrt-header.sh`), `BuildTPU.cmake`/
  `TpuConfiguration.cmake` restructured to build from the vendored header with libtpu dlopen'd at
  runtime (no python, no link-time proprietary deps — verified via `ldd`). Build command:

  ```bash
  /home/agibsonccc/dev-apps/mvn/bin/mvn -Ptpu -Dlibnd4j.buildthreads=12 \
    -Dlibnd4j.log=libnd4j-build.log \
    -pl libnd4j,:nd4j-tpu-preset,:nd4j-tpu install -DskipTests 2>&1 | tee tpu-build.log
  ```

  Produces `libnd4j/blasbuild/tpu/libnd4jtpu.so` (~192MB, exports the `HloIRBuilder`/
  `TpuGraphBackend`/`PjrtClientManager` layer) + installs both TPU jars. The `-Ptpu` profile in
  `libnd4j/pom.xml` sets `libnd4j.chip=tpu` → `javacpp-cppbuild-compile-tpu` → `-DSD_TPU=true`.
  C++ bit-rot fixed across all `graph/tpu/*` + `platform/pjrt/*` files (API drift: DataType enum
  names, `reduceAlongDimension` target-overloads, non-const-ref shape constructors, banned
  `make_unique` into raw pointers, `PjrtClient.get()` accessor, stray/duplicate deletes).
  Also: `build-deploy-linux-tpu.yml` (mirrors build-deploy-linux-x86_64 with sccache) and the
  python-free `run-tpu-smoke-tests.yml` (curl+jq wheel extraction).

**Native ZLUDA build wiring — DONE (build-cycle integration):**
- `buildnativeoperations.sh` gained `--zluda <ON|AMD|INTEL|AUTO>` → `-DSD_ZLUDA=ON`
  [`+ -DSD_ZLUDA_TARGET=<t>`], consumed by the existing `cmake/ZludaConfiguration.cmake`
  (`setup_zluda[_amd|_intel]()`, `setup_miopen()`; `MainBuildFlow.cmake` calls it when `SD_ZLUDA=ON`).
  `libnd4j/pom.xml` threads `-Dlibnd4j.zluda` (default OFF) into all three cppbuild executions.
- ZLUDA piggybacks the **cuda chip** build (it's a binary libcuda.so drop-in — no separate `.so`).
  **A local `SD_ZLUDA=ON` build was deliberately NOT run**: adding that global define to
  `blasbuild/cuda` would invalidate the entire CUDA ccache (multi-hour rebuild of the working tree).
  New `build-deploy-linux-zluda.yml` validates it in CI on a fresh runner — `install` only, never
  `deploy` (a ZLUDA-flavored `libnd4jcuda.so` must not overwrite the standard CUDA snapshot
  coordinates; publishing would need a distinct classifier), downloads a pinned ZLUDA release into
  `ZLUDA_PATH`, and gates on `SD_ZLUDA=ON` + "ZLUDA configuration complete" appearing in the log.
  A real ZLUDA build additionally needs a ROCm (AMD) or oneAPI (Intel) host; miopen flows via the
  generic `-Dlibnd4j.helpers=miopen` → `-DHELPERS_miopen=ON` path.

**Native Hexagon build wiring — DONE (build-cycle integration):**
- `-Phexagon` reactor profile (nd4j-hexagon + nd4j-hexagon-preset); `libnd4j/pom.xml` `hexagon`
  profile (`libnd4j.chip=hexagon` → `javacpp-cppbuild-compile-hexagon` execution +
  `assembly-hexagon.xml`); `buildnativeoperations.sh` chip case (`hexagon` → `-DSD_HEXAGON=true`,
  lib `nd4jhexagon`); `BuildHexagon.cmake` rewritten to the MainBuildFlow `target_sources` pattern
  (matches the modernized `BuildTPU.cmake`) — adds `graph/hexagon/*.cpp`, defines `SD_HEXAGON`/
  `HAVE_HEXAGON_MLIR`, links `-ldl` (libhexagon_mlir_runtime.so dlopen'd at runtime, opaque void*,
  no Qualcomm SDK at build time — hexagon-mlir is BSD-3 open source). `build-deploy-linux-hexagon.yml`
  (mirrors build-deploy-linux-tpu, sccache, verifies `libnd4jhexagon.so`). The native C++ in
  `graph/hexagon/*` has the same DSP-API bit-rot the TPU files had (NativeSlot→wiring sub-struct,
  DSP_DIAG category rename) — being fixed against the `TpuGraphBackend.cpp` reference (subagent).
- nd4j-hexagon module itself: same `org.nd4j`→`org.eclipse` pom rot + old-interface scaffolding as
  nd4j-tpu; repaired identically (Executioner skeleton, Env/OpContext placeholders,
  `canRun()==false` locked by `HexagonBackendSmokeTest`, preset cycle broken + `nd4j-presets-common`
  dep + `module-info.java` added). Builds green; smoke tier 3 tests pass (1 skip, no runtime).

## 7. Follow-up recommendations (still open)

1. **Enable a PR gate**: the cheapest meaningful gate is `run-cpu-tests-sanity-checks.yml` with
   `testTier=smoke` on `{linux-x86_64, linux-arm64}` only (snapshot jars, ~10 min) — currently
   *nothing* runs automatically on push/PR.
2. **Fix hosted CUDA builds** (linux: cancelled/OOM; windows: 0-for-17) or move them to larger/
   self-hosted runners; until then GPU test workflows have no fresh snapshots to consume. The
   PTX gate in `build-zluda-validation.yml` also depends on fresh CUDA snapshots, and PTX presence
   requires the CUDA build to keep emitting virtual (`compute_XX`) targets, not `-real`-only.
3. **ppc64le via IBM program + riscv64 via RISE** (§5.3); optionally revive macosx-x86_64 on
   `macos-15-intel` before Aug 2027.
4. **Retire** `cpu-sanity-check-tests.yaml` and `test_multiple_arch.yaml` (vestigial).
5. **TPU bindings**: with the native `libnd4jtpu.so` build in place, generate the JavaCPP bindings
   (`javacpp.generate` profile), flip `JTpuBackend.canRun()` to consult `PjrtClientManager`, and
   fix the priority footgun (F10: `BACKEND_PRIORITY_GPU + 10` outranks CUDA) BEFORE enabling.
   `TpuBackendSmokeTest` will fail intentionally when `canRun()` changes, forcing the upgrade.
6. **Tag hygiene**: add TagNames constants for `dsp`/`sparse`/`gnn`/`multi-gpu` literals; consider
   tagging a curated CUDA smoke set with `full-ci` so the GPU sanity workflow can use tiers too.
7. On first real ZLUDA hardware run: iterate on cublas/nvrtc interception using the
   `zludaEnvironmentReport` output (F7 note) — JavaCPP loads those by absolute path, so
   `pathsFirst` interception is best-effort until proven on hardware.
8. **Native ZLUDA build**: `--zluda`/`-Dlibnd4j.zluda` wiring lands with the TPU subagent; an
   actual `SD_ZLUDA=ON` compile needs a ROCm host (self-hosted AMD runner) and would light up the
   in-tree MIOpen platform ops.
9. **Hexagon native build**: the Java modules, reactor profile, smoke tier and publishing are done;
   remaining is the native `libnd4jhexagon.so` path — modernize `BuildHexagon.cmake` to the same
   MainBuildFlow pattern as the rewritten `BuildTPU.cmake`, add a `javacpp-cppbuild-compile-hexagon`
   execution to `libnd4j/pom.xml`, fix `graph/hexagon/*.cpp` bit-rot, and source the hexagon-mlir
   toolchain (BSD-3, buildable from source — no proprietary blob needed, unlike libtpu).

### Quick usage

```bash
# ZLUDA (AMD box with ROCm + ZLUDA installed)
ZLUDA_PATH=/opt/zluda platform-tests/run-zluda-smoke-tests.sh -Dzluda.target=AMD

# TPU wiring smoke — no TPU hardware needed (pip install libtpu)
platform-tests/run-tpu-smoke-tests.sh

# Tag tiers locally
platform-tests/run-smoke-tests.sh                      # ~30s
platform-tests/run-full-ci-tests.sh                    # ~5min

# One-time module installs the new profiles need (from repo root)
/home/agibsonccc/dev-apps/mvn/bin/mvn install -DskipTests -Pzluda -pl :nd4j-zluda
/home/agibsonccc/dev-apps/mvn/bin/mvn install -DskipTests -Ptpu -pl :nd4j-tpu,:nd4j-tpu-preset
```
