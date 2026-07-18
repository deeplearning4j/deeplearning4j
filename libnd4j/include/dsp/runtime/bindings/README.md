# SDX Runtime Language Bindings

This directory provides wrapper APIs for the SDX C runtime ABI defined in:

- `../dsp_runtime_c.h`

Available wrappers:

- `java/` (JNA-based Java API — canonical source lives in the `nd4j-sdx` Maven
  module and is copied into staged SDK packages; see `java/README.md`)
- `kotlin/` (Kotlin facade on top of Java wrapper)
- `python/` (`ctypes` API, supports `numpy.ndarray`)
- `rust/` (FFI + safe wrapper; `standalone` feature links `libsdx_*`)
- `csharp/` (.NET P/Invoke API)
- `swift/` (Swift wrapper for C ABI / XCFramework usage)

All six wrappers cover the full ABI surface, including `sdxMarkInputVariable`,
`sdxMarkInputPlaceholder`, `sdxFreezeShapes`, `sdxGetPlanPhase`,
`sdxGetExecutionCount`, the input-contract discovery calls (`sdxGetNumInputs`,
`sdxGetNumOutputs`, `sdxGetInputName`), and the extended
`sdx_execution_report_t` (`plan_phase`, `execution_count`).

Run contract: a plan's external inputs cover the model's constants, variables,
AND placeholders — `sdxRun` binds one tensor per external, positionally in
plan order. Discover the order via `sdxGetNumInputs` + `sdxGetInputName`.

Serving runner:

- `python/sdx_sdk_runner.py` exposes SDX runtime over REST and gRPC with binary ndarray transport (`NPZ` for REST and `bytes+shape+dtype` tensors for gRPC).

Examples:

- Usage examples for each binding have been moved to the
  [deeplearning4j-examples](https://github.com/eclipse/deeplearning4j-examples)
  repository under `sdx-runtime-examples/`.

Notes:

- These wrappers target ABI version `1`.
- Loaders prefer the JVM-free standalone runtimes (`sdx_cpu`, `sdx_cuda`, built
  with `-DSD_BUILD_SDX_STANDALONE=ON`) and fall back to the monolithic backend
  libraries (`nd4jcpu`, `nd4jcuda`, `nd4jamd`) — both export the same `sdx*` ABI.
- Python runtime loader auto-detects host platform/arch and probes SDK packaged library layouts before linker fallback.
- CUDA/AMD wrappers use the same C ABI and select target via model/run options.
- Backend ordinals mirror `GraphExecutionMode`: HIP graph replay is `9` and
  Vulkan command-buffer replay is `11`; Level Zero, Metal, TPU, and Hexagon
  remain available at ordinals `10`, `12`, `13`, and `14` respectively.
- Java wrapper can run with ND4J `INDArray` via `runNd4j(...)` when ND4J is present on classpath.
- `sdxGetPlanPhase` / `plan_phase` values: 0=SLOT_BY_SLOT (warmup),
  1=SHAPES_FROZEN, 2=REPLAYING, 3=REPLAY_BLOCKED.
