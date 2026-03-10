# SDX Runtime Language Bindings

This directory provides wrapper APIs for the SDX C runtime ABI defined in:

- `../dsp_runtime_c.h`

Available wrappers:

- `java/` (JNA-based Java API)
- `kotlin/` (Kotlin facade on top of Java wrapper)
- `python/` (`ctypes` API, supports `numpy.ndarray`)
- `rust/` (FFI + safe wrapper)
- `csharp/` (.NET P/Invoke API)
- `swift/` (Swift wrapper for C ABI / XCFramework usage)

Serving runner:

- `python/sdx_sdk_runner.py` exposes SDX runtime over REST and gRPC with binary ndarray transport (`NPZ` for REST and `bytes+shape+dtype` tensors for gRPC).

Notes:

- These wrappers target ABI version `1`.
- Runtime library names vary by platform/build (`nd4jcpu`, `nd4jcuda`, etc).
- Python runtime loader auto-detects host platform/arch and probes SDK packaged library layouts before linker fallback.
- CUDA/AMD wrappers use the same C ABI and select target via model/run options.
- Java wrapper can run with ND4J `INDArray` via `runNd4j(...)` when ND4J is present on classpath.
