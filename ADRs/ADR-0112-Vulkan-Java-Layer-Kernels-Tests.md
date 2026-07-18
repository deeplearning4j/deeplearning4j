# ADR-0112: Vulkan Java Layer, Kernel Registration, and Test Evidence

## Status

Accepted. The Java backend and kernel registration infrastructure are
implemented. Strict emitted-kernel coverage remains an active gate.

## Context

A loadable Java backend, a catalog entry, an eager numerical calculation, and a
successfully replayed Vulkan kernel are four different facts. Earlier Vulkan
documentation and tests blurred these boundaries, which made coverage appear
broader than the retained evidence.

ND4J operation traits are owned by the operations themselves. Vulkan must use
the same op-local metadata model rather than creating a second centralized
traits authority or maintaining semantic lists of hardcoded operation names.

## Decision

### Java backend

The Vulkan backend supplies its own backend, affinity, memory, workspace,
DataBuffer, constant, shape-info, and environment implementations. Backend
properties select these classes, and JavaCPP presets bind the Vulkan native
library. CPU and CUDA presets do not contain Vulkan-specific loading or
execution logic.

Native discovery is performed by the normal Maven/JavaCPP dependency model.
Tests must not require manual `java.library.path`, `LD_PRELOAD`, device
masking, or literal environment-variable passthrough.

### Traits and kernel eligibility

Each op declares its framework traits through its own `addTraits`
implementation. There is no centralized Vulkan traits table.

The Vulkan emitter catalog describes the combinations for which Vulkan has a
lowering recipe: descriptor identity, category, dtype, layout, rank, and
argument schema. Catalog eligibility is necessary for recording, but it is not
proof that MLIR lowering, SPIR-V creation, dispatch, and replay succeeded on
hardware.

Shared compiler utilities may be chip-neutral functions. Vulkan-specific
routing and state remain in the Vulkan backend and must not alter the CPU or
CUDA backend paths.

### Unsupported combinations

If an op or one of its dtype/layout/rank/argument combinations has no valid
Vulkan implementation, Vulkan recordability rejects it with a diagnostic. The
backend does not run the op on CPU, CUDA, an emulated Vulkan device, or an eager
side path while reporting Vulkan replay.

### Test evidence levels

1. **Strict emitted-kernel evidence**
   `VulkanKernelEmitterStrictReplayTest` builds the real op in SameDiff,
   requires Vulkan hardware and Triton-enabled DSP, verifies native Vulkan
   dispatch with no slot fallback, changes inputs, and compares against an
   independent reference. This suite is the coverage authority.

2. **DSP integration evidence**
   Tests that inspect DSP plan state and native replay prove capture/replay
   integration for their actual graphs. They establish coverage only for the
   operations present in those graphs.

3. **Eager numerical evidence**
   Repeated ND4J operations can validate Java/backend integration and numerical
   behavior. Repetition alone is not DSP replay evidence.

4. **Source or catalog evidence**
   A lowering, recipe, or registration present in source is implementation
   inventory, not executed coverage.

The legacy `VulkanReplayEquivalenceTest` class contains both DSP integration
and eager numerical tests. Its documentation and test names must identify that
distinction, and it must not synthesize an operation with Java loops and call
that an emitted-kernel test.

### Platform reporting

Build wiring and hardware qualification are reported separately. The Linux
NVIDIA host is the current retained hardware gate. Android profiles use API 24
and the platform Vulkan API, but Android vendor support requires real-device
runs. The same rule applies to AMD, Intel, and other desktop drivers.

## Consequences

- Op-local `addTraits` stays the single framework trait authority.
- Strict-test results, including failures, define current kernel coverage.
- Integration tests remain useful without being mislabeled as kernel replay.
- Unsupported work remains explicit instead of being concealed by fallback.
- JavaCPP and Maven remain responsible for native inclusion and loading.
