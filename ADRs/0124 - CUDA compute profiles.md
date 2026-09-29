# ADR 0124: CUDA compute profiles

## Status

Accepted (2026-09-30). Amends ADR 0041.

Resolver checked on 2026-09-30 with `libnd4j/cmake/CudaConfiguration.cmake` in
CMake script mode. It ran against the installed CUDA 13.1 `nvcc`, and against
stubs that print the `nvcc --list-gpu-code` output of CUDA 12.6, 12.8 and 12.9.
Those toolkits are not installed on the checking machine.

| Input | Resolved targets |
|---|---|
| CUDA 13.1 or 12.9, `release` | `sm_80 sm_86 sm_90 sm_100 sm_120`, PTX `compute_120` |
| CUDA 12.6, `release` | `sm_80 sm_86 sm_90`, PTX `compute_90` |
| CUDA 13.1, `dev`, GB10 (found through `nvidia-smi`) | `sm_121`, no PTX |
| CUDA 12.8, `dev`, GB10 | `sm_120` (runs on 12.1), no PTX |
| CUDA 13.1, `dev`, RTX 3090 + RTX 4090 | `sm_86 sm_89`, no PTX |
| `dev`, no GPU visible | `sm_86`, PTX `compute_86` |
| `libnd4j.compute=8.6 9.0` / `12.1` / `80,86` | exactly those SASS targets, no PTX |
| `libnd4j.compute=12.1` on CUDA 12.6 | configure error: the toolkit cannot generate `sm_121` |
| `libnd4j.compute=Maxwell`, profile `foo` | configure error |

## Context

The CUDA library is built for one CUDA configuration at a time. A checkout's
POMs describe a single configuration (CUDA 12.6, 12.9 or 13.1, and ZLUDA on
12.9), and `change-cuda-versions.sh <version>` switches between them. The
release plans (`release/{aws,azure,gcp}/release-plan.json`) and the CI
workflows list every configuration and run `change-cuda-versions.sh` in each
job before Maven.

Until this change the GPU targets did not depend on the configuration or on
who the build was for:

- Without `libnd4j.compute`, `buildnativeoperations.sh` set `COMPUTE=all`.
  CMake turned `all` and `auto` into `sm_86` SASS plus `compute_86` PTX. So
  every local build, including one on a DGX Spark (GB10, `sm_121`), produced
  that pair.
  - On any GPU newer than Ada, the driver JIT-compiles the PTX of the whole
    library when it loads. The library is 264 MB.
  - Kernels could not use instructions newer than compute capability 8.6, such
    as FP8 conversions (8.9) or TMA bulk copies (9.0). For example, the FP8
    case of `CutlassGemmHelper.cu` is disabled because no build targets
    `compute_89`.
  - A100 (8.0) cannot run `sm_86` SASS or `compute_86` PTX.
- The x86_64 and Windows release shards passed `-Dlibnd4j.compute=8.6 9.0`, and
  `build-scripts/release/native-platform.sh` used the same list by default.
  - An explicit list builds SASS only, and SASS does not run across major
    versions. So the published artifacts had no code for compute capability
    10.x or 12.x (B200, RTX 50, RTX PRO Blackwell), and none for A100.
  - ADR 0041 had also filed A100 under 8.6 and RTX 40, L4 and L40 under 9.0.
    Those GPUs are 8.0 and 8.9.
- `build-scripts/build-common.sh` hardcoded `8.6 9.0` for CUDA 12 and
  `8.6 9.0 10.0 12.0` for CUDA 13.
- The version updater behind `change-cuda-versions.sh` also rewrote YAML and
  JSON files. Each switch therefore collapsed the release plans and workflows
  onto the one configuration it selected.

Developers want quick builds for their own GPU. Releases have to cover the GPUs
users have with the toolkit each configuration ships. Explicit compute lists
must keep working.

## Decision

### Target selection

`libnd4j.compute` (`-cc` / `--compute`) and the new `libnd4j.compute.profile`
(`--compute-profile`, CMake `SD_COMPUTE_PROFILE`) pick the targets.
`resolve_cuda_architectures` in `CudaConfiguration.cmake` applies them in this
order:

1. **Explicit targets.** `libnd4j.compute` takes compute capabilities with or
   without a dot, separated by spaces or commas (`"8.6 9.0"`, `12.1`, `80,86`).
   - It builds SASS for exactly those targets and no PTX.
   - It overrides the profile.
   - Every target is checked against `nvcc --list-gpu-code`. A target the
     toolkit cannot generate, or a value that is not a compute capability,
     stops the configure.
   - Codenames such as `Maxwell` were already rejected before this change; only
     the README still listed them.
2. **`dev` profile.** This is the default when `libnd4j.compute` is empty.
   - It builds SASS for each distinct GPU architecture on the build machine and
     no PTX. CMake's native detection finds the GPUs, with `nvidia-smi` as the
     fallback when CMake is older than 3.24 or the driver is older than the
     toolkit.
   - For a GPU newer than the toolkit, it builds the closest older SASS of the
     same major version. For example, CUDA 12.8 on GB10 builds `sm_120`.
   - With no visible GPU, or when cross-compiling, it builds `sm_86` SASS plus
     `compute_86` PTX, the previous default.
3. **`release` profile.** It builds SASS for each `SD_CUDA_RELEASE_ARCHITECTURES`
   target (`80 86 90 100 120`) that the toolkit can generate, plus PTX for the
   newest of them:

   | CUDA configuration | SASS | PTX |
   |---|---|---|
   | 13.1, 12.9 | `sm_80 sm_86 sm_90 sm_100 sm_120` | `compute_120` |
   | 12.6 (predates Blackwell) | `sm_80 sm_86 sm_90` | `compute_90` |

   SASS for X.Y runs on X.Z when Z ≥ Y, so the 12.9 and 13.1 artifacts run
   natively on:

   - 8.0 (A100, A30)
   - 8.6 and 8.9 (RTX 30/40, A10, A40, L4, L40)
   - 9.0 (H100, H200)
   - 10.0 and 10.3 (B200, B300)
   - 12.0 and 12.1 (RTX 50, RTX PRO Blackwell, DGX Spark)

   GPUs newer than every SASS target JIT-compile the PTX.

The older keywords map to profiles: `auto` and `native` mean `dev`, and `all`
means `release`.

Two builds do not use the resolver:

- ZLUDA keeps `resolve_zluda_ptx_architectures`, which derives its virtual
  architecture from the ROCm SDK.
- `SD_GCC_FUNCTRACE` builds pin `sm_86`.

Maven sets `libnd4j.compute.profile` to `dev` in `libnd4j/pom.xml` and passes
both properties to `buildnativeoperations.sh` on every CUDA build. Empty
values are passed as empty arguments. The script checks the profile name. It
also sizes `nvcc --threads` and the per-job RAM budget (14 GB per nvcc thread)
from the resolved architecture count.

### Release builds

- **Architecture contract.** Every CUDA release shard declares exactly one of
  `-Dlibnd4j.compute=<targets>` or `-Dlibnd4j.compute.profile=release`.
  `cuda_architecture_contract` in `release/aws/build-platform.py` rejects any
  other combination and never accepts `dev`. ZLUDA shards declare neither.
- **Shard targets.**
  - The x86_64 and Windows shards for CUDA 12.6, 12.9 and 13.1 use
    `profile=release`.
  - The Linux arm64 CUDA 13.1 shard stays at `-Dlibnd4j.compute=12.1` for DGX
    Spark (ADR 0041). It is cross-compiled, so `dev` would fall back to
    `sm_86`.
- **Local release scripts.** `build-scripts/release/native-platform.sh` passes
  `DL4J_COMPUTE` and `DL4J_COMPUTE_PROFILE` (default `release`).
  `build-scripts/build-common.sh` passes `CUDA_COMPUTE` (default empty) and
  `CUDA_COMPUTE_PROFILE` (default `release`).

### CUDA configuration selection

`change-cuda-versions.sh` (`contrib/version-updater`, `CudaFileUpdater`) rewrites
POMs only. Release plans and workflows own their configuration lists and select
a configuration per job. `run-tests.yml`, `run-zluda-smoke-tests.yml` and
`build-zluda-validation.yml` run `change-cuda-versions.sh "${BACKEND#nd4j-cuda-}"`
before building.

## Consequences

**Builds and ccache**

- A `dev` build on a single-GPU machine compiles one architecture, so each
  kernel is built once and nothing is JIT-compiled at load. On GB10 the device
  code is `sm_121` SASS, which lets kernels use instructions up to compute
  capability 12.1.
- Kernels that use instructions newer than 8.0 still need a guard or a runtime
  check. The release targets start at `sm_80`, and a `dev` build targets
  whatever GPU it is built on.
- Switching a build tree between profiles or target lists changes every CUDA
  translation unit's flags, so ccache misses on all of them and the next
  build is a full rebuild.
  - This is also true the first time a tree that was built with the old
    default meets the new one, for example `sm_86` + `compute_86` → `sm_121`
    on GB10.
  - Builds that must share a ccache should pin `libnd4j.compute`.

**Release artifacts**

- A release build compiles every CUDA translation unit for five SASS targets
  and one PTX target, where the 13.1 shards previously built two SASS targets.
  CUDA 12.6 cannot generate the Blackwell targets, so its shards build three
  SASS targets and one PTX target.
  - Compile time and library size grow with the target count.
  - The measured single-target `libnd4j.so` is 264 MB. Release sizes have not
    been measured yet.
  - `buildnativeoperations.sh` runs up to four `nvcc` threads per translation
    unit and budgets RAM per thread, so the release shards trade some make
    parallelism for thread parallelism.
- Turing (7.5) and older remain outside the release targets, as in ADR 0041.
  Users can still build them with `-Dlibnd4j.compute=7.5` on toolkits that
  generate them.
- Jetson Thor (compute capability 11.0 under CUDA 13) is not covered by the
  release targets: `sm_100` SASS and `compute_120` PTX do not run on it. Adding
  it needs its own arm64 classifier.
- The release target list lives in one CMake variable,
  `SD_CUDA_RELEASE_ARCHITECTURES`. `buildnativeoperations.sh` repeats its size
  (5 targets, or 3 before CUDA 12.8) to size `nvcc --threads`. Adding a target
  means updating the variable, that count, the README table and this ADR.
