# Vulkan Android Target

## Current status

The build system has Android Vulkan target wiring for `android-arm64`
(`arm64-v8a`) and `android-x86_64`, with API level 24 as the minimum configured
platform. Android uses the NDK/sysroot Vulkan device API and target `libvulkan`;
it must not resolve a host Vulkan loader.

This wiring is not yet proof of real-device correctness or performance. No
throughput number should be published until a physical device run retains the
device fingerprint, Vulkan driver/version, commit, exact command, raw result, and
complete log.

## Build ownership

The Maven profiles, JavaCPP platform properties, and libnd4j build script own the
cross compiler, sysroot, ABI, API level, resource classifier, and native link.
Do not configure CMake manually, hardcode NDK paths into project files, disable
other backends through environment variables, or set `java.library.path`.

An Android ARM64 reactor invocation has this shape:

~~~bash
/home/agibsonccc/dev-apps/mvn/bin/mvn \
  --log-file maven-vulkan-android-arm64.log \
  -f pom.xml -Pandroid-arm64 -Pvulkan \
  -Djavacpp.platform=android-arm64 -Dlibnd4j.vulkan \
  -Dlibnd4j.triton=ON -Dlibnd4j.buildthreads=12 \
  -Dlibnd4j.log=libnd4j-vulkan-android-arm64.log \
  -DskipTests \
  -pl libnd4j,:nd4j-vulkan-preset,:nd4j-vulkan install
~~~

`android-x86_64` uses the corresponding profile and JavaCPP platform value. The
resulting `nd4j-vulkan` classifier is selected by the target OS/ABI; it is not
split by GPU vendor. Vulkan device enumeration selects Adreno, Mali, or another
installed implementation at runtime.

## Runtime requirements

- Android API 24 or newer.
- A physical device exposing the Vulkan features required by the selected dtype
  and storage contract.
- The Android platform Vulkan loader; no desktop loader or ICD manifest.
- Application packaging that includes the tooling-produced Android classifier.

## Validation boundary

An emulator or software Vulkan implementation can validate packaging and
correctness plumbing only. It cannot validate Android GPU drivers, UMA behavior,
subgroup behavior, thermals, or throughput.

Cross-device peer migration currently requires external-memory/semaphore
capabilities that are not universal on Vulkan 1.0/Android. Single-device Android
execution is a separate capability and must be tested independently.

## Required real-device evidence

1. Backend and device enumeration through ordinary JavaCPP/POM loading.
2. The strict one-op pipeline gate for the supported dtype/layout/rank matrix.
3. DSP persistent replay with changed inputs and no fallback.
4. Memory, lifecycle, and repeated-create/destroy tests.
5. Only after correctness: retained model and performance benchmarks.

No synthetic CSV rows, hypothetical device numbers, or nonexistent harness
scripts are treated as validation.
