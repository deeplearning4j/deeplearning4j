# ADR 0099 - GraalVM Native Image Support

## Status
Implemented

Proposed by: Adam Gibson (May 2026)

## Context

GraalVM Native Image compiles Java applications ahead-of-time into standalone executables, eliminating JVM startup time and reducing memory footprint. ND4J's extensive use of reflection (SameDiff op registry, `DifferentialFunction` hierarchy) and JNI (JavaCPP native bindings) requires explicit configuration for Native Image to compile correctly.

## Decision

Provide `META-INF/native-image/` configuration for nd4j-api, nd4j-native, and nd4j-cuda backends.

### Configuration per Module

**nd4j-api** — `reflect-config.json` covers the SameDiff/op class hierarchy (`DifferentialFunction`, `SameDiff`, all SD op namespaces). `native-image.properties` defers `org.nd4j` and `org.bytedeco` to runtime init (avoids Android class checks and `UnsafeUtil` at build time), eagerly initializes `org.nd4j.shade.protobuf`.

**nd4j-native** — Adds JNI support (`-H:+JNI`), defers the full JavaCPP native chain (`javacpp.Loader`, `Pointer`, OpenBLAS, MKL, oneDNN). `reflect-config.json` covers CPU backend classes (`NDArray`, `CpuMemoryManager`, `Nd4jCpu`). `jni-config.json` covers `NativeOps`/`NativeOpsHolder`. `resource-config.json` bundles native `.so` for 6 platforms (linux-x86_64, linux-aarch64, windows-x86_64, macosx-x86_64, macosx-arm64, android-arm64).

**nd4j-cuda** — Parallel config for CUDA-specific classes (`CudaMemoryManager`, `CudaAffinityManager`, `Nd4jCuda`).

### Init Strategy

Build-time (safe): `org.nd4j.shade.protobuf.*`. Runtime (needs native libs): everything under `org.nd4j.**`, `org.bytedeco.**`, and backend-specific classes.

## Consequences

- Enables single-executable deployment with millisecond startup
- Every new reflectively-accessed class must be added to the config
- Native Image executables must be built per-platform

## Related ADRs

- [0073](0073%20-%20DSP%20Self-Contained%20Runtime%20SDK%20and%20SDZ%20Deployment.md) — SDX runtime may use Native Image for single-binary deployment
