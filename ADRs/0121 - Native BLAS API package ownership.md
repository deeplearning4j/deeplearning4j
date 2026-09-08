# ADR 0121: Native BLAS API package ownership

## Status

Implemented in source; remote build and Javadoc qualification pending.

## Context

Release run 34169797114 failed `nd4j-native-api` central-javadoc because `org.nd4j.nativeblas` was split between the JPMS modules `nd4j.api` and `nd4j.cpu.api`. The core native interfaces already belong to `nd4j-api`; their remaining shared helpers must have the same package owner.

## Decision

- Move `BaseNativeNDArrayFactory`, `LongPointerWrapper`, `PointerPointerWrapper`, `ResultWrapperAbstraction`, `Nd4jBlas`, `NativeLapack`, and `NativeOpsGPUInfoProvider` unchanged from `nd4j-native-api` to `nd4j-api`.
- Export `org.nd4j.nativeblas` only from `nd4j.api`. Retain all package and class names, without forwarding classes.
- Move both the classpath service registration and JPMS `GPUInfoProvider` provision to `nd4j-api`.
- Make `nd4j.cpu.api` require `nd4j.api` transitively so its consumers retain module readability for the relocated public API.
- Keep existing Maven dependencies and strict Javadoc settings unchanged; do not introduce a delombok pipeline or module-validation suppression.

## Consequences

The shared native BLAS API has one module owner, removing the split package at its source. Class names and service-provider behavior remain unchanged, but consumers must use aligned versions of the two artifacts rather than mixing old and new package layouts. Build, Javadoc, and service-discovery qualification must run remotely.

## Backend-neutral device and cache package migration

Release run 34181626837 (`f1f2607db7`) exposed two further split packages between `nd4j.api` and `nd4j.cuda.backend.common`: `org.nd4j.jita.constant` and `org.nd4j.linalg.jcublas`. Their shared classes already reside in the API artifact but must not retain CUDA-owned package names.

Relocate the following classes within `nd4j-api`, changing only package declarations and imports:

| Class | Previous package | API-owned package |
| --- | --- | --- |
| `DefaultDeviceIDProvider` | `org.nd4j.jita.constant` | `org.nd4j.linalg.api.concurrency` |
| `DeviceIDProvider` | `org.nd4j.jita.constant` | `org.nd4j.linalg.api.concurrency` |
| `ConstantProtector` | `org.nd4j.jita.constant` | `org.nd4j.linalg.cache` |
| `ProtectedCachedShapeInfoProvider` | `org.nd4j.jita.constant` | `org.nd4j.linalg.api.ndarray` |
| `CachedShapeInfoProvider` | `org.nd4j.linalg.jcublas` | `org.nd4j.linalg.api.ndarray` |

These destination packages are already exported by `nd4j.api`; the descriptor documents their ownership. Do not export the old packages from the API or retain forwarding classes, which would recreate the split. CUDA handlers and other CUDA implementations retain their backend packages. Update Java imports (including the previously implicit same-package `ConstantProtector` reference in `ProtectedCudaConstantHandler`), native-image reflection registrations, and CPU/minimizer/TPU/Vulkan/CUDA/ZLUDA backend properties together.

The singleton instances, device-keyed caches, synchronization, workspace exclusion, constant-buffer lifetime protection, shape handling, and purge behavior are unchanged. Unlike the native BLAS artifact relocation above, this package migration changes public binary names: downstream imports, reflection registrations, and custom provider configuration must migrate, and consumers must rebuild against aligned API/backend versions. No compatibility shim or alternate execution path is introduced.

Validation for this follow-up is static only locally; consolidated build, JPMS/Javadoc, native-image, and backend execution qualification remains remote.

## CUDA backend entrypoint ownership

The CUDA artifact also declared `JCublasBackend` in `org.nd4j.linalg.jcublas`, which is owned and exported by `nd4j.cuda.backend.common`. Move only this entrypoint to `org.nd4j.linalg.jcublas.backend` in `nd4j-cuda`, and export that distinct package from `nd4j.cuda`. Keep the implementation classes (`CudaEnvironment`, `JCublasNDArray`, and the other CUDA-common classes) in their existing packages and import the formerly same-package dependencies explicitly.

Update the JPMS `Nd4jBackend` provision, classpath service registration, native-image reflection entry, and the CUDA/Vulkan coexistence test's reflective class name together. No forwarding entrypoint remains in the common package. Availability checks, priority, device discovery, configuration resource lookup, and backend ID are unchanged. Simple-name and `jcublas`/`cublas` substring checks continue to match. External consumers referencing the old entrypoint binary name must migrate and use aligned artifacts.

This follow-up receives static source/reference review only locally; no local build, test, or code generation is run. The parent full remote matrix workflow is dry-run-only (`dryRun: true`); execution qualification is not claimed.

## References

- ADR 0016: Java 9+ Support
- ADR 0120: Canonical full Maven repository assembly
