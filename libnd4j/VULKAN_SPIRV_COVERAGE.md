# Vulkan SPIR-V Coverage Contract

**Updated:** 2026-07-17

## Scope

This document defines how Vulkan operation support is measured. It intentionally
does not duplicate a hand-maintained list of operation names.

## Sources of truth

1. **Operation semantics and DSP traits** belong to each operation through
   `addTraits`. A centralized Vulkan-specific trait table is not authoritative.
2. **Vulkan eligibility** is declared by the typed entries in
   `VulkanKernelEmitterCatalog`. Each entry constrains descriptor identity,
   recipe, dtype, layout, rank, arguments, and output contract.
3. **Confirmed dispatched coverage** is established by
   `VulkanKernelEmitterStrictReplayTest`.

The catalog currently spans multiple families, including elementwise, reductions,
matmul, normalization, attention, data movement, constant generation, selected
fused LLM operations, and selected legacy operations. That breadth is eligibility,
not a claim that every entry or every argument combination has cleared hardware.

## Strict evidence standard

A nonconstant strict test must:

- build a one-operation SameDiff graph;
- reach the DSP replaying phase;
- compile the segment with `vulkan-native`;
- create and submit a real Vulkan compute pipeline;
- replay changed input data;
- compare against an independent Java oracle;
- assert that no slot-by-slot or fallback path executed.

Zero-input constant/stateful cases must additionally prove real capture before
the framework freezes or evolves the value.

An unsupported descriptor, dtype, layout, rank, or argument combination is
rejected during recordability validation. It is not sent to CPU, CUDA, or
emulated replay. Catalog acceptance can still fail later during shader
compilation or dispatch, which is why the strict gate is required.

## Tests that do not establish emitter coverage

The following are useful but are not substitutes for the strict gate:

- loader/SPI smoke tests;
- catalog enumeration or source inspection;
- eager numerical tests that repeat an ND4J operation without inspecting a DSP
  plan;
- Java loops that manually construct the expected tensor;
- software-driver tests used as physical-hardware or performance evidence.

`VulkanReplayEquivalenceTest` is an integration suite. Only its methods that
explicitly assert DSP plan phase, native compiler identity, replay counts, and
absence of fallback count as replay evidence.

## Current qualification gaps

- The complete strict catalog matrix must be green.
- Convolution/pooling, recurrent, loss, image, signal, sparse, and quantized
  families still require catalog/emitter work where no entry exists.
- FP16 and FP64 are capability-gated; BF16, narrow-integer payloads, and boolean
  storage remain incomplete.
- Correctness-first kernels still need tiled/subgroup/cooperative performance
  implementations.
- Linux NVIDIA has the strongest retained real-hardware evidence. Android, AMD,
  Intel, and Windows require equivalent retained gates.

## Reporting rule

Every coverage result records the Vulkan implementation and driver, physical
device, dtype/layout/rank/arguments, commit, exact Maven command, and retained
log. Never report a registration count as an execution pass rate.
