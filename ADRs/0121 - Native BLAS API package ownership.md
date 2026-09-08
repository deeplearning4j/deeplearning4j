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

## References

- ADR 0016: Java 9+ Support
- ADR 0120: Canonical full Maven repository assembly
