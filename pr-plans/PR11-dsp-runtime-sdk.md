# PR11: DSP Runtime SDK

**Estimated files:** ~43
**Merge layer:** 3
**Complexity:** Low (SDK binding stubs, not core execution)
**Reviewers:** SDK/API team

## Description

Multi-language SDK bindings for the DSP/SDX runtime C API. These are binding
stubs for C#, Java, Kotlin, Python, Rust, and Swift that wrap the native
`dsp_runtime_c.h` C API. This is the deployment/serving interface, separate
from the core execution engine.

## Files (43)

### C API header (1)
- `libnd4j/include/dsp/NativeOpsDsp.h`
- `libnd4j/include/dsp/runtime/dsp_runtime_c.h`

### Documentation (2)
- `libnd4j/include/dsp/runtime/README.md`
- `libnd4j/include/dsp/runtime/bindings/README.md`

### C# bindings (4)
- `libnd4j/include/dsp/runtime/bindings/csharp/SdxRuntime.cs`
- `libnd4j/include/dsp/runtime/bindings/csharp/SdxRuntime.csproj`
- `libnd4j/include/dsp/runtime/bindings/csharp/examples/BasicUsage.cs`
- `libnd4j/include/dsp/runtime/bindings/csharp/examples/BasicUsage.csproj`

### Java bindings (5)
- `libnd4j/include/dsp/runtime/bindings/java/SdxRuntime.java`
- `libnd4j/include/dsp/runtime/bindings/java/pom.xml`
- `libnd4j/include/dsp/runtime/bindings/java/README.md`
- `libnd4j/include/dsp/runtime/bindings/java/examples/BasicUsage.java`

### Kotlin bindings (6)
- `libnd4j/include/dsp/runtime/bindings/kotlin/SdxRuntime.kt`
- `libnd4j/include/dsp/runtime/bindings/kotlin/build.gradle.kts`
- `libnd4j/include/dsp/runtime/bindings/kotlin/settings.gradle.kts`
- `libnd4j/include/dsp/runtime/bindings/kotlin/README.md`
- `libnd4j/include/dsp/runtime/bindings/kotlin/examples/BasicUsage.kt`

### Python bindings (15)
- `libnd4j/include/dsp/runtime/bindings/python/sdx_runtime.py`
- `libnd4j/include/dsp/runtime/bindings/python/sdx_sdk_runner.py`
- `libnd4j/include/dsp/runtime/bindings/python/sdx_tensor_transport.py`
- `libnd4j/include/dsp/runtime/bindings/python/sdx_serving.proto`
- `libnd4j/include/dsp/runtime/bindings/python/sdx_serving_pb2*.py`
- `libnd4j/include/dsp/runtime/bindings/python/generate_proto.py`
- `libnd4j/include/dsp/runtime/bindings/python/__init__.py`
- `libnd4j/include/dsp/runtime/bindings/python/pyproject.toml`
- `libnd4j/include/dsp/runtime/bindings/python/requirements-runner.txt`
- `libnd4j/include/dsp/runtime/bindings/python/README.md`
- `libnd4j/include/dsp/runtime/bindings/python/examples/basic_usage.py`
- `libnd4j/include/dsp/runtime/bindings/python/tests/` (3 files)

### Rust bindings (6)
- `libnd4j/include/dsp/runtime/bindings/rust/src/lib.rs`
- `libnd4j/include/dsp/runtime/bindings/rust/build.rs`
- `libnd4j/include/dsp/runtime/bindings/rust/Cargo.toml`
- `libnd4j/include/dsp/runtime/bindings/rust/README.md`
- `libnd4j/include/dsp/runtime/bindings/rust/examples/basic_usage.rs`

### ADRs (2)
- `ADRs/0073 - DSP Self-Contained Runtime SDK and SDZ Deployment.md` — Stable native ABI for per-platform DSP runtime with multi-language SDK bindings
- `ADRs/0074 - SDX Runtime Serving Protocol (REST + gRPC).md` — gRPC primary / REST secondary serving protocol matching the C ABI

### Swift bindings (6)
- `libnd4j/include/dsp/runtime/bindings/swift/SdxRuntime.swift`
- `libnd4j/include/dsp/runtime/bindings/swift/Package.swift`
- `libnd4j/include/dsp/runtime/bindings/swift/README.md`
- `libnd4j/include/dsp/runtime/bindings/swift/module.modulemap`
- `libnd4j/include/dsp/runtime/bindings/swift/shim.h`
- `libnd4j/include/dsp/runtime/bindings/swift/examples/BasicUsage.swift`
