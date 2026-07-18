# ADR-0111: Vulkan Device Management and Multi-Device Semantics

## Status

Accepted. The device-management foundation is implemented; platform and driver
qualification remain ongoing.

## Context

A Vulkan backend owes ND4J the same device-backend capabilities that CUDA
provides: device identity and selection, device-owned memory, primary/special
buffer synchronization, execution contexts, constant and shape metadata,
resource lifetime management, and explicit multi-device behavior.

Those capabilities must be implemented with Vulkan device abstractions. Reusing
CPU backend classes or entering CUDA execution paths would violate backend
isolation even if a test happened to pass.

## Decision

### Device ownership

`VulkanDeviceManager` owns Vulkan instance and physical-device enumeration.
ND4J device ids map to enumerated Vulkan physical devices, and thread-local
selection is implemented by the Vulkan backend. Failure to enumerate or create
a required Vulkan device is reported as a Vulkan initialization failure; it
does not cause the Vulkan backend to impersonate CPU.

Each device has its own `VulkanDeviceContext`, queues, command pools,
synchronization state, descriptor resources, capability record, and pipeline
state. Objects whose validity depends on a device are keyed by device id.

### Memory and buffers

`VulkanMemoryPool` suballocates Vulkan memory and owns buffer attribution and
lifetime tracking. Destruction observes submitted-work completion. ND4J
DataBuffers retain the normal primary/special actuality contract, with special
storage backed by the selected Vulkan device and transfers performed through
Vulkan commands.

Host-visible and device-local memory are Vulkan memory types, not backend
fallbacks. Unified-memory devices may expose a placement that satisfies both
roles. Discrete devices use explicit staging where required by the Vulkan API.

### Constant, shape, TAD, and RNG state

Constant buffers, shape-info buffers, TAD metadata, and RNG state are
device-owned and device-keyed. A cached pointer from one Vulkan device is never
reused on another. The Java providers delegate allocation and synchronization
to Vulkan NativeOps rather than CPU providers.

### Multi-device behavior

Multiple physical devices and mixed ICDs in one process are normal. Every
context, pool, cache, constant/shape store, and replay resource is associated
with its device.

Vulkan does not imply CUDA-style unified virtual addressing or peer access.
Cross-device copies are explicit transfers coordinated by the two Vulkan
contexts. Optional external-memory or device-group capabilities may optimize a
supported pair only after capability checks; correctness never depends on them.

Device loss is a reported device error. It does not silently select a different
backend or device.

### DSP integration

The DSP platform hooks for a Vulkan build resolve to Vulkan implementations.
They bind the selected device, manage Vulkan resources and transfers, and
coordinate replay lifetime. Shared DSP utilities remain chip-neutral functions;
they do not make Vulkan execute CPU or CUDA platform hooks.

### Java layer

The Vulkan affinity, memory, workspace, buffer, constant, and shape providers
expose the native Vulkan implementation through the backend properties and
JavaCPP-generated bindings. Native loading is controlled by Maven dependencies
and presets. Applications and tests do not construct a backend by manually
setting `java.library.path`.

## Verification

Device-tier tests cover enumeration, selection, allocation, buffer
synchronization, constant/shape/TAD ownership, and multi-device isolation.
Replay tests additionally verify that device-resident buffers survive capture
and repeated execution.

The retained real-hardware evidence currently applies to the Linux NVIDIA test
host used by the project. Android, AMD, Intel, and other driver matrices must be
qualified independently; source compatibility or successful cross-compilation
is not hardware evidence.

## Consequences

- Vulkan remains independent from CPU and CUDA backends.
- Per-device ownership is part of correctness, not an optional optimization.
- Explicit Vulkan transfers are the normal portable multi-device mechanism.
- No environment masking, software-device substitution, or cross-backend
  execution is accepted as a device-management fix.
