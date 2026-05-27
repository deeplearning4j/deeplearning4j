# PR01: Build System & CI

**Estimated files:** ~155
**Merge layer:** 0 (no dependencies)
**Complexity:** Low
**Reviewers:** Build/infra team

## Description

CI workflows (GitHub Actions), CMake build system, Maven POM version alignment,
ccache setup, Triton cmake wiring, build scripts, and packaging configuration.
This is pure infrastructure — no runtime code changes.

## File Categories

### GitHub Actions workflows (~17)
- `.github/workflows/build-deploy-android-arm64.yml`
- `.github/workflows/build-deploy-android-x86_64.yml`
- `.github/workflows/build-deploy-cross-platform.yml`
- `.github/workflows/build-deploy-linux-arm64.yml`
- `.github/workflows/build-deploy-linux-cuda-12.6.yml`
- `.github/workflows/build-deploy-linux-cuda-12.9.yml`
- `.github/workflows/build-deploy-linux-x86_64-compat.yml`
- `.github/workflows/build-deploy-linux-x86_64.yml`
- `.github/workflows/build-deploy-mac-arm64.yml`
- `.github/workflows/build-deploy-mac.yml`
- `.github/workflows/build-deploy-windows-cuda-12.6.yml`
- `.github/workflows/build-deploy-windows-cuda-12.9.yml`
- `.github/workflows/build-deploy-windows.yml`
- `.github/workflows/cpu-sanity-check-tests.yaml`
- `.github/workflows/java-hotfix-release.yml`
- `.github/workflows/publish-sdk-release.yml`
- `.github/workflows/run-cpu-integration-tests.yml`
- `.github/workflows/run-cpu-tests-sanity-checks.yml`
- `.github/workflows/run-gpu-tests-sanity-checks.yml`
- `.github/workflows/run-tests.yml`
- `.github/workflows/test_multiple_arch.yaml`

### GitHub Actions composite actions (~8)
- `.github/actions/build-centos/build.sh`
- `.github/actions/build-centos/Dockerfile`
- `.github/actions/install-cmake-linux/action.yml`
- `.github/actions/publish-sdk-jars/action.yml`
- `.github/actions/publish-sdx-runtime-sdk/action.yml`
- `.github/actions/setup-ccache-linux/action.yml`
- `.github/actions/setup-ccache-macos/action.yml`
- `.github/actions/setup-ccache-windows/action.yml`

### CMake build system (~48)
- `libnd4j/CMakeLists.txt`
- `libnd4j/CMakeLists.txt.onednn.in`
- `libnd4j/cmake/android-arm64.cmake`
- `libnd4j/cmake/android-arm.cmake`
- `libnd4j/cmake/android-x86.cmake`
- `libnd4j/cmake/BuildCPU.cmake`
- `libnd4j/cmake/BuildHexagon.cmake`
- `libnd4j/cmake/BuildSDX.cmake`
- `libnd4j/cmake/BuildTPU.cmake`
- `libnd4j/cmake/CompileMemoryProfiling.cmake`
- `libnd4j/cmake/CompilerFlags.cmake`
- `libnd4j/cmake/CompilerOptimizations.cmake`
- `libnd4j/cmake/CudaCleanup.cmake`
- `libnd4j/cmake/CudaConfiguration.cmake`
- `libnd4j/cmake/Dependencies.cmake`
- `libnd4j/cmake/DuplicateInstantiationDetection.cmake`
- `libnd4j/cmake/FindMLIR.cmake`
- `libnd4j/cmake/FindMLX.cmake`
- `libnd4j/cmake/FindNCCL.cmake`
- `libnd4j/cmake/FindOpenVINO.cmake`
- `libnd4j/cmake/FindTriton.cmake`
- `libnd4j/cmake/GenCompilation.cmake`
- `libnd4j/cmake/HelperConfiguration.cmake`
- `libnd4j/cmake/HexagonConfiguration.cmake`
- `libnd4j/cmake/install_openvino.cmake`
- `libnd4j/cmake/install_triton.cmake`
- `libnd4j/cmake/install_triton.sh`
- `libnd4j/cmake/JNIConfiguration.cmake`
- `libnd4j/cmake/MainBuildFlow.cmake`
- `libnd4j/cmake/nvcc_filter.py`
- `libnd4j/cmake/Options.cmake`
- `libnd4j/cmake/PartialLinking.cmake`
- `libnd4j/cmake/patch_openvino.cmake`
- `libnd4j/cmake/patch_triton.cmake`
- `libnd4j/cmake/patch_triton_cpu.cmake`
- `libnd4j/cmake/patch_triton_no_amd.sh`
- `libnd4j/cmake/Platform.cmake`
- `libnd4j/cmake/PlatformOptimizations.cmake`
- `libnd4j/cmake/PostBuild.cmake`
- `libnd4j/cmake/Ppstep.cmake`
- `libnd4j/cmake/sdx_exports.lds`
- `libnd4j/cmake/SdxRuntimePackage.cmake`
- `libnd4j/cmake/SelectiveRenderingCore.cmake`
- `libnd4j/cmake/SmartCcache.cmake`
- `libnd4j/cmake/smart_ccache.py`
- `libnd4j/cmake/TemplateCorrelation.cmake`
- `libnd4j/cmake/TemplateProcessing.cmake`
- `libnd4j/cmake/TpuConfiguration.cmake`
- `libnd4j/cmake/TypeMST.cmake`
- `libnd4j/cmake/TypeProfiles.cmake`
- `libnd4j/cmake/TypeRegistryGenerator.cmake`
- `libnd4j/cmake/ZludaConfiguration.cmake`

### Build scripts (~15)
- `libnd4j/buildnativeoperations.sh`
- `libnd4j/flatc-generate.sh`
- `libnd4j/assembly-tpu.xml`
- `libnd4j/tools/sdx-compile.sh`
- `libnd4j/tools/sdx-generate-bindings.sh`
- `build-cuda.sh`
- `change-cuda-versions.sh`
- `update-op-registry.bat`
- `update-op-registry.sh`
- `test-ant.xml`
- `build-scripts/build-cuda-backend*.sh` (11 files)

### Maven POMs (~40+)
- `pom.xml` (root)
- `libnd4j/pom.xml`
- `nd4j/pom.xml`
- `nd4j/nd4j-backends/nd4j-backend-impls/pom.xml`
- All module `pom.xml` files (nd4j-api, nd4j-cuda, nd4j-native, nd4j-ggml, samediff-llm, etc.)
- `codegen/blas-lapack-generator/pom.xml`
- `codegen/op-codegen/pom.xml`
- `contrib/benchmarking_nd4j/pom.xml`
- `contrib/blas-lapack-generator/pom.xml`
- `datavec/datavec-excel/pom.xml`
- `deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-ui-model/pom.xml`
- `omnihub/pom.xml`
- `platform-tests/pom.xml`
- `resources/pom.xml`

### Copilot/editor config
- `.github/copilot-instructions.md`

### ADRs (6 — only those actually changed in the diff)
- `ADRs/0030 - Type Promotion.md` — Smaller type-limited artifact to reduce binary size
- `ADRs/0039 - Selective rendering type system.md` — CMake-level semantic filtering to avoid template combinatorial explosion
- `ADRs/0041 - CUDA Architecture Reduction.md` — Drop pre-Ampere compute capabilities to cut build time 75%
- `ADRs/0042 - Android NDK Migration.md` — Upgrade from NDK r21d to r27d (LLVM 18)
- `ADRs/0045 - Android Cross-Compilation Modernization.md` — Modernized Android CMake toolchain files
- `ADRs/0047 - Comprehensive Template Instantiation Migration.md` — Platform-aware type equivalence classes for cross-platform linker correctness

### DSP runtime Java binding POM
- `libnd4j/include/dsp/runtime/bindings/java/pom.xml`
