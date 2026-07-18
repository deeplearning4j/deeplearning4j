# Vulkan Backend Build and Validation

## Architecture

Vulkan is a standalone device backend. Its native chip, JavaCPP preset, bindings,
and classified resources live under the Vulkan modules and build tree. CPU and
CUDA are separate backends; neither is an execution path or link dependency of
`libnd4jvulkan`.

Vulkan artifacts are classified by target OS/ABI. The platform Vulkan loader then
selects AMD, NVIDIA, Intel, or mobile hardware at runtime. A CUDA version is not a
Vulkan classifier dimension.

Triton remains enabled because it is the universal DSP/replay build gate. This
does not make Vulkan depend on the CUDA backend.

## Canonical build

Run the Maven reactor from the repository root. Do not invoke CMake or a native
build tool directly.

~~~bash
/home/agibsonccc/dev-apps/mvn/bin/mvn \
  --log-file maven-vulkan-build.log \
  -f pom.xml -Pvulkan -Dlibnd4j.vulkan \
  -Dlibnd4j.triton=ON -Dlibnd4j.buildthreads=12 \
  -Dlibnd4j.log=libnd4j-vulkan-build.log \
  -DskipTests \
  -pl libnd4j,:nd4j-vulkan-preset,:nd4j-vulkan install
~~~

The POM and JavaCPP preset own compiler selection, link paths, native-resource
inclusion, and classifier layout. Do not set `java.library.path`, `LD_PRELOAD`,
`CUDA_VISIBLE_DEVICES`, or Vulkan loader/device overrides to make a test pass.

## Strict kernel gate

All tests run from `platform-tests/` and retain a Maven log.

~~~bash
cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests
/home/agibsonccc/dev-apps/mvn/bin/mvn \
  --log-file vulkan-strict-kernel-gate.log \
  -Ptest-vulkan -Dlibnd4j.triton=ON -Dlibnd4j.buildthreads=12 \
  -Dnd4j.vulkan.test.requireTriton=true \
  -Dnd4j.vulkan.test.requireHardware=true \
  -Dtest=VulkanKernelEmitterStrictReplayTest test
~~~

A passing strict case proves one real Vulkan pipeline, changed-input replay, an
independent numerical oracle, and no slot-by-slot or fallback execution.

## Supporting gates

~~~bash
/home/agibsonccc/dev-apps/mvn/bin/mvn \
  --log-file vulkan-integration-gate.log \
  -Ptest-vulkan -Dlibnd4j.triton=ON -Dlibnd4j.buildthreads=12 \
  -Dnd4j.vulkan.test.requireTriton=true \
  -Dnd4j.vulkan.test.requireHardware=true \
  -Dtest=VulkanReplayEquivalenceTest,VulkanMultiDeviceTest test
~~~

`VulkanReplayEquivalenceTest` contains integration-level DSP and eager numerical
checks. Only methods that explicitly inspect the DSP plan and assert
`vulkan-native` execution are replay-coverage evidence. `VulkanMultiDeviceTest`
covers device ownership, caches, concurrent devices, and real device kernels.

## Diagnostics

Use `-Dorg.bytedeco.javacpp.logger.debug=true` when the preset/compiler/linker
selection needs inspection. Fix the POM, preset, CMake dependency, or downloaded
dependency patch at its source; do not compensate with machine-specific paths.

## Coverage reporting

Report the Vulkan implementation and driver, physical device, dtype/layout/rank
case, commit, exact Maven command, and retained log. Loader smoke, catalog
membership, compilation, dispatched correctness, and performance are separate
claims.
