# ADR: DSP Self-Contained Runtime SDK and SDZ Deployment

## Status

Accepted

**Milestone status:** M1 (runtime + SDZ load path) implemented; M2-M4 (packaging and release gates) proposed.

Proposed by: Adam Gibson (March 4, 2026)

Discussed with: Development Team

Supersedes:
- `DSP_SELF_CONTAINED_DEPLOYMENT_RFC.md`
- `DSP_ARCHITECTURE_OVERVIEW.md`
- `DSP_CPU_BACKEND_READINESS.md`
- `libnd4j/include/dsp/runtime/README.md`

## Context

DSP graph execution is already available in libnd4j through `NativeDynamicShapePlan`, with backend routing for MLX, NNAPI, Arm Compute, oneDNN, MLIR CPU, Triton, NVRTC, PTX, and slot-by-slot fallback.

What was missing was a single deployment contract for:

1. shipping self-contained runtime binaries per platform,
2. exposing a stable native ABI that does not depend on Java graph construction,
3. loading SameDiff model artifacts (`.sdz`/`.sdnb`) directly from the SDK/runtime,
4. ensuring consistent backend policy behavior across Apple, Android, CUDA, AMD, and Intel targets.

The prior documentation was spread across multiple markdown files with partial overlap and drift risk.

## Decision

We standardize deployment on a C runtime SDK centered on SDZ-first model loading, with optional bundle manifests for packaging metadata.

### 1. Canonical Runtime Model Input

The runtime must load `.sdz` and `.sdnb` directly.

`sdxLoadBundle(...)` accepts:
- direct `.sdz` or `.sdnb` model files,
- unpacked bundle directories containing `manifest.json` that resolves `modelPath`,
- manifest JSON files (`.json` or `.dspb` manifest path) that point to an underlying model file.

Current scope is unpacked bundles and direct model files. Packed `.dspb` archive extraction is deferred.

### 2. Public C ABI Contract

The public SDK header is:
- `libnd4j/include/dsp/runtime/dsp_runtime_c.h`

ABI principles:
- Opaque handles (`sdx_runtime_t`, `sdx_model_t`, `sdx_context_t`)
- Fixed runtime ABI constant (`SDX_RUNTIME_ABI_VERSION`)
- Additive struct evolution with `struct_size` checks
- No STL/C++ types in the public interface

Core API:
- `sdxGetRuntimeAbiVersion`
- `sdxCreateRuntime` / `sdxDestroyRuntime`
- `sdxLoadBundle` / `sdxUnloadModel`
- `sdxCreateContext` / `sdxDestroyContext`
- `sdxRun`
- `sdxGetLastError`
- `sdxGetExecutionReport`

### 3. C Runtime Reuses Existing C++ Execution Path (Parity)

The C runtime is a thin wrapper over existing native graph APIs:
- `loadModelFromFile(...)`
- `compileModelPlan(...)`
- `setPlanGraphExecutionMode(...)`
- `executeDynamicShapePlan(...)`

This guarantees parity with the C++ runtime path because both routes use the same native plan compilation and execution engine (`NativeDynamicShapePlan`) instead of maintaining duplicate execution implementations.

### 4. Backend Policy and Platform Matrix

Backend selection is policy-based with `AUTO` and forced modes. Forced mode may be strict (`strict_backend`) or allow fallback depending on runtime options.

Primary target expectations:

| Platform target | Preferred backend order (`AUTO`) | Notes |
|---|---|---|
| macOS/iOS arm64 | `MLX -> oneDNN -> ACL -> NNAPI -> ARM_HYBRID -> MLIR CPU -> slot` | MLX is the first-class Apple path |
| Android arm64 | `NNAPI -> ACL -> ARM_HYBRID -> MLIR CPU -> slot` | NNAPI + Arm Compute + MLIR are expected |
| Linux/Windows + NVIDIA | `TRITON -> NVRTC -> PTX -> CUDA graphs/slot` | CUDA graph replay and JIT paths supported |
| Linux + AMD GPU | `TRITON(AMD target) -> compatible fallback` | Runtime gpu target policy must be explicit |
| Intel CPU optimized | `oneDNN Graph -> MLIR CPU -> slot` | oneDNN is the optimized Intel CPU path |
| Intel GPU (where available) | `TRITON(Intel target) -> fallback` | Level Zero-based target path |

### 5. CUDA and AMD Tensor Support in C ABI

`sdx_tensor_view_t` supports:
- `SDX_DEVICE_HOST`
- `SDX_DEVICE_CUDA`
- `SDX_DEVICE_AMD`

`sdx_run_options_t` and `sdx_model_options_t` support:
- `gpu_target = AUTO | CUDA | AMD`

Current runtime constraints per `sdxRun(...)` invocation:
- all GPU tensors must share a single `device_id`,
- mixed CUDA and AMD tensors are rejected in one call,
- `AUTO` gpu target is inferred from tensor device types.

### 6. SDZ Compression Handling

SDZ loading supports ZIP entries with:
- `STORED` (no compression),
- `DEFLATE` when compiled with `HAVE_ZLIB`.

If zlib support is absent and model entries require DEFLATE, load fails with an explicit error.

### 7. SDK Packaging Output

A staging SDK target is provided:
- `cmake --build <build-dir> --target sdx_runtime_sdk`
- `cmake --build <build-dir> --target sdx_runtime_bindings`

Staged artifacts:
- `include/dsp/runtime/dsp_runtime_c.h`
- runtime library (`lib/`)
- DSP manifest schema (`share/dsp/manifest.schema.json`, when present)

Platform binding packages are emitted under:
- `<build-dir>/sdx-runtime-sdk/bindings/<platform>/<variant>/...`
- `<build-dir>/sdx-runtime-sdk/dist/sdx-runtime-<platform>-<variant>.zip`
- Android builds also emit `.aar` packages.
- Apple builds emit `.xcframework` when `xcodebuild` is available.

Language wrapper templates are staged in each package under:
- `wrappers/` with APIs for Swift, Kotlin, Java, Python, Rust, and C#.

Helper script for per-platform generation:
- `libnd4j/tools/sdx-generate-bindings.sh --platform <id> --backend <cpu|cuda|amd>`

Platform packaging goals remain:
- Apple: `.xcframework`
- Android: `.aar`
- Linux/Windows: native runtime package

### 8. Observability and Execution Reporting

`sdx_execution_report_t` provides:
- requested/applied backend,
- requested/applied GPU target,
- status code,
- fallback marker (reserved for richer telemetry),
- execution duration.

### 9. Disk Plan Cache Integration

Serialized DSP plan bytes are now persisted to `~/.kompile/cache/dsp/dsp_plan_cache/` (see ADR 0061, *Disk Plan Persistence* section). For SDK deployment:

- SDZ/SDNB models produce deterministic plan bytes on every load — the disk cache eliminates recompilation on process restart.
- Pre-compiled plans can be distributed via the override directory (`dsp_plan_override/`), analogous to the existing Triton override mechanism.
- The C runtime (`sdxLoadBundle` → `compileModelPlan`) benefits automatically when the Java plan cache populates the disk cache on first run.
- Future: SDZ archives can optionally embed plan `.bin`/`.meta` files alongside the model, enabling single-artifact deployment with pre-warmed plan cache.

Configuration: `ND4J_DSP_PLAN_CACHE_DIR`, `ND4J_DSP_PLAN_CACHE_DISK_ENABLED`, `ND4J_DSP_PLAN_CACHE_FORCE_RECOMPILE`, `ND4J_DSP_PLAN_CACHE_OVERRIDE_DIR`.

## Consequences

### Advantages

- One runtime contract for C/C++/JNI/Swift bindings.
- Direct SDZ loading from SDK runtime (no Java graph build requirement).
- Reuse of native plan compiler/executor preserves behavioral parity.
- Backend policy is explicit per platform and debuggable via execution reports.
- CUDA and AMD execution policies are represented in the public ABI.
- Disk plan persistence eliminates multi-second plan recompilation on process restart.

### Disadvantages

- Packed `.dspb` archive support is not complete yet (manifest/unpacked path today).
- AMD path depends on available GPU runtime plumbing in build/deploy environment.
- Fallback telemetry is currently coarse (`used_fallback` reserved, not full reason graph yet).

## Milestones

1. `M1` (implemented): public C ABI, SDZ/SDNB loading, runtime/context/run lifecycle, SDK staging target.
2. `M2` (next): stricter segment-level backend capability checks and richer fallback reasons.
3. `M3` (next): production packaging (`.xcframework`, `.aar`, Linux/Windows distributables).
4. `M4` (next): CI/release gates across backend matrix with forced-mode conformance tests.

## References

- [ADRs/0061 - DynamicShapePlan Execution.md](./0061%20-%20DynamicShapePlan%20Execution.md)
- [ADRs/0074 - SDX Runtime Serving Protocol (REST + gRPC).md](./0074%20-%20SDX%20Runtime%20Serving%20Protocol%20(REST%20+%20gRPC).md)
- `libnd4j/include/dsp/runtime/dsp_runtime_c.h`
- `libnd4j/include/legacy/impl/DspRuntimeC.cpp`
- `libnd4j/include/graph/impl/NativeDynamicShapePlan.cpp`
- `libnd4j/include/graph/impl/SdzReader.cpp`
- `nd4j/.../samediff/execution/DspPlanDiskCache.java`
