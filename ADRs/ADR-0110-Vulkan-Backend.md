# ADR-0110: Standalone Vulkan Backend

## Status

Accepted. The standalone backend and its device/replay foundation are implemented.
Kernel coverage and platform qualification remain active work.

## Context

Vulkan is a compute API available across multiple GPU vendors and operating
systems. ND4J needs a Vulkan backend that behaves like a normal device backend:
it owns its devices and memory, loads beside other backends in one process, and
executes only work for which it has a real Vulkan implementation.

Vulkan is not a CPU execution mode and is not an extension of the CUDA backend.
CPU, CUDA, and Vulkan may share chip-neutral utilities, but they must not select
or enter one another's backend paths.

## Decision

### Standalone chip

Vulkan is a first-class libnd4j chip:

- `--chip vulkan` selects `SD_VULKAN`.
- The native library is `blasbuild/vulkan/libnd4jvulkan.so`.
- Maven's `vulkan` profile owns native build, JavaCPP generation, packaging,
  and platform classifier selection.
- Vulkan code and dependencies do not get injected into CPU or CUDA presets.

The supported build entry point is Maven. Direct CMake invocations are
development diagnostics, not the documented installation procedure.

### Compiler and runtime stack

The Vulkan chip requires MLIR and Triton support. Triton is the project-wide DSP
compiler knob; it is not a CUDA-only option. Vulkan and CUDA use the managed
shared LLVM/MLIR runtime selected by the build, so both shared libraries can
coexist in one process without symbol hiding, static isolation, `LD_PRELOAD`,
or device masking.

The build must prefer its managed dependencies consistently from CMake through
JavaCPP compilation and linking. A host-specific LLVM, such as a Linuxbrew
installation, must not leak into one stage of the build.

### Device and execution model

`VulkanDeviceManager`, per-device contexts, memory pools, dual-buffer storage,
and `VulkanReplayHandle` provide the native device and DSP replay layers.
Vulkan records and submits real compute pipelines. An unsupported operation,
dtype, layout, rank, or argument combination is rejected; it is not executed by
CPU, CUDA, an emulated device, or a hidden eager path.

DSP capture/replay is available only when every recorded slot has a supported
Vulkan implementation. Normal Vulkan eager/integration tests are useful but do
not, by themselves, establish replay or emitted-kernel coverage.

### Packaging and hardware selection

The Vulkan artifact classifier describes the operating-system/ABI platform.
It is intentionally vendor-neutral: the Vulkan loader enumerates the installed
ICDs and physical devices at runtime. NVIDIA, AMD, Intel, Android vendors, and
software ICDs do not require separate Java artifacts merely because the vendor
differs.

JavaCPP presets and Maven dependency inclusion decide which native libraries are
packaged and loaded. Tests and applications must not manually set
`java.library.path`, `LD_PRELOAD`, `CUDA_VISIBLE_DEVICES`, or an ICD
override to manufacture a passing configuration.

### Android

The repository has Android ARM64 and x86_64 cross-build profiles with API 24 as
the minimum configured API. Android resolves Vulkan through the NDK/platform
device API. This wiring is not a claim of real-device qualification: Adreno,
Mali, and other Android driver/device matrices require retained hardware
evidence before support is declared.

## Verification contract

The authoritative emitted-kernel gate is
`VulkanKernelEmitterStrictReplayTest`. Each case must:

1. construct the intended op through SameDiff;
2. require the real Vulkan backend and Triton-enabled DSP;
3. reach native Vulkan replay with no slot fallback;
4. change inputs between executions; and
5. match an independent reference.

`VulkanReplayEquivalenceTest` and the device-tier suites provide additional
integration, memory, and numerical checks. They are not substitutes for the
strict emitted-kernel gate.

## Consequences

- CPU and CUDA behavior remain independent of Vulkan.
- Multiple device backends and their shared libraries are expected to coexist.
- Coverage claims follow retained strict-test evidence, not catalog entries,
  eager calculations, or source presence.
- Unsupported coverage remains visible work rather than being hidden by a
  fallback.
- Platform support is reported separately for build wiring and real-hardware
  qualification.

## References

- `libnd4j/VULKAN_SPIRV_COVERAGE.md`
- `VULKAN_BUILD_SETUP.md`
- `VULKAN_ANDROID_BUILD.md`
- ADR-0111: Vulkan device management
- ADR-0112: Vulkan Java layer, kernel registration, and tests
