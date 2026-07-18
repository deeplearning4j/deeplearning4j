# Vulkan Backend Status

**Updated:** 2026-07-17

**Status:** The standalone build, JavaCPP packaging, device runtime, memory model,
multi-device support, MLIR-to-SPIR-V compilation, and DSP capture/replay path are
implemented. This does not mean every ND4J operation is covered or that every
catalog entry has passed on every Vulkan implementation.

## What counts as support

Vulkan support has three distinct levels:

1. **Catalog eligibility:** `VulkanKernelEmitterCatalog` accepts a particular
   descriptor, dtype, layout, rank, and argument schema.
2. **Shader compilation:** the accepted recipe lowers through the managed shared
   MLIR stack and creates a Vulkan compute pipeline.
3. **Dispatched correctness:** `VulkanKernelEmitterStrictReplayTest` proves that
   the pipeline executed, replayed changed inputs, produced an independent
   reference result, and used neither slot-by-slot execution nor fallback.

Only level 3 is reported as confirmed kernel coverage. Catalog membership and
test-source presence are not hardware evidence.

Unsupported descriptor combinations are rejected during Vulkan recordability
validation. A strict Vulkan segment must never execute through CPU, CUDA, or an
emulated fallback.

## Implemented foundation

- Standalone `SD_VULKAN` chip and `libnd4jvulkan` artifact.
- Managed shared LLVM/MLIR compiler stack with MLIR-to-SPIR-V pipeline creation.
- Vulkan device discovery, affinity, contexts, streams, memory pools, dual-buffer
  synchronization, and exact-device allocation ownership.
- Per-device constant, shape, and TAD cache ownership.
- Multi-device execution and concurrent device use.
- DSP graph capture/replay, persistent RNG state, and zero-input static shapes.
- JavaCPP platform classifiers selected by the Maven/POM tooling.
- Normal CUDA and Vulkan coexistence in one JVM without device masking, symbol
  hiding, `LD_PRELOAD`, or manual `java.library.path`.

## Validation boundary

The strongest retained evidence is Linux x86-64 on real NVIDIA hardware, including
an RTX 4090 and RTX 3070 Ti, with all enumerated Vulkan devices visible. Software
drivers can prove loader and correctness plumbing but cannot establish physical
GPU performance or mobile-driver qualification.

The complete strict emitter suite remains the coverage release gate. Integration
and eager numerical tests supplement that gate but do not replace its dispatched-
pipeline assertions.

## Remaining work

- Clear the complete strict real-kernel suite.
- Expand catalog and strict tests for genuinely missing operation families.
- Add BF16, narrow-integer, boolean-storage, and quantized coverage where the
  Vulkan capability model permits it.
- Replace correctness-first serial kernels with tiled/subgroup/cooperative
  implementations and retain benchmark evidence.
- Qualify Android, AMD, Intel, and Windows targets on their actual drivers and
  hardware.

## Authoritative references

- Build and test workflow: `VULKAN_BUILD_SETUP.md`
- Operation-coverage contract: `libnd4j/VULKAN_SPIRV_COVERAGE.md`
- Android target status: `VULKAN_ANDROID_BUILD.md`
- Emitter catalog: `libnd4j/include/graph/vulkan/VulkanKernelEmitterCatalog.cpp`
- Strict gate: `platform-tests/src/test/java/org/eclipse/deeplearning4j/nd4j/backends/VulkanKernelEmitterStrictReplayTest.java`
