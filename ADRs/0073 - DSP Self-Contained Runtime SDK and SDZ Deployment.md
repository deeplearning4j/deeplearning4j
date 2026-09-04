# ADR: DSP Self-Contained Runtime SDK and SDZ Deployment

## Status

Accepted

**Milestone status:** M1 runtime loading, M3 packaging, and the source-SDZ target cache are implemented; remaining M2/M4 work is backend conformance and release coverage.

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

The runtime must load `.sdz` and `.sdnb` directly. The application and deployment tooling expose `.sdz` as the single public model contract.

An enriched SDZ may contain immutable target objects under `META-INF/sdx-cache/`. These objects are derived from the canonical SameDiff source during host compilation; their physical formats are private to the target compiler and runtime provider.

`sdxLoadBundle(...)` continues to accept direct SDZ/SDNB files, unpacked manifest directories, manifest JSON, and legacy packed `.dspb` archives for compatibility. New tooling does not produce DSPB. The mobile accelerator runtime resolves the selected target from the SDZ into an app-owned cache and passes only the opaque runtime path to the provider. That runtime never compiles target metadata or falls back on-device.

The optional raw-GGUF mobile preparation service is a staging process, not the
accelerator runtime. It runs in an isolated CPU importer before the provider
process starts, creates the canonical SDZ, invokes the same `TargetCompiler`
contract as host tooling, and retires before accelerator loading. The provider
runtime remains extraction-only and cannot synthesize target metadata or fall
back after commitment.

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
- Plan lifecycle: `sdxMarkInputVariable` / `sdxMarkInputPlaceholder` /
  `sdxFreezeShapes` / `sdxGetPlanPhase` / `sdxGetExecutionCount`
- Input contract discovery: `sdxGetNumInputs` / `sdxGetNumOutputs` /
  `sdxGetInputName`

Run contract: a plan's external inputs cover the model's **constants,
variables, and placeholders** — `sdxRun` binds one tensor per external,
positionally in plan order. Clients discover that order via
`sdxGetNumInputs` + `sdxGetInputName` (the serving layer reorders named
request tensors onto it automatically).

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

ZIP64 archives are supported (EOCD locator/record + per-entry 0x0001 extended
info), so `.sdz` models over 4GB — which `java.util.zip` writes as ZIP64
automatically — load correctly. The reader is in-memory; archives must fit in
host RAM.

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
- The Java wrapper source is copied at staging time from its canonical home
  (the `nd4j-sdx` Maven module) into `wrappers/java/src/main/java/`, keeping a
  single source of truth while making the staged package self-contained. The
  Kotlin facade compiles against that staged sibling (or, in the source tree,
  directly against the `nd4j-sdx` module path).

Helper script for per-platform generation:
- `libnd4j/tools/sdx-generate-bindings.sh --platform <id> --backend <cpu|cuda|amd>`

### 7a. Standalone (Minimal) Runtime Packaging

`-DSD_BUILD_SDX_STANDALONE=ON` (Maven: `-Dlibnd4j.sdx.standalone=ON`) builds
`libsdx_cpu.so` / `libsdx_cuda.so` from the already-compiled object set — a
JVM-free runtime whose exported symbol surface is restricted to `sdx*` via
`cmake/sdx_exports.lds` (or `-exported_symbol` on Apple). When enabled:

- `sdx_runtime_sdk` and every per-variant binding package/dist zip ship the
  standalone library instead of the monolithic backend library.
- `binding.json` records `"standalone": true|false` so loaders can detect
  which runtime a package carries.
- The packaged library copy (never the build-tree original) is stripped with
  the toolchain `strip --strip-unneeded` when available.
- `sdx-generate-bindings.sh` enables standalone mode by default
  (`--no-standalone` opts out). The default Maven/CI builds keep it OFF to
  avoid the extra link; enabling it in the release workflows is the remaining
  M3 step.

Language binding loaders (Python/Java/C#/Rust) probe the standalone library
names (`sdx_cpu`, `sdx_cuda`) first and fall back to the monolithic backend
names — both export the same `sdx*` ABI.

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
- execution duration,
- plan phase (0=SLOT_BY_SLOT/warmup, 1=SHAPES_FROZEN, 2=REPLAYING,
  3=REPLAY_BLOCKED — the `PlanLifecycle::toLegacyCode()` scale),
- execution count for the context.

All language bindings expose the full report plus `sdxMarkInputVariable`,
`sdxMarkInputPlaceholder`, `sdxFreezeShapes`, `sdxGetPlanPhase`, and
`sdxGetExecutionCount`. The C layer serializes concurrent `sdxRun` calls on a
context and guards error-string access, so bindings may share a context across
threads without corrupting cached wrapper state.

### 9. Disk Plan Cache Integration

Serialized DSP plan bytes are now persisted to `~/.kompile/cache/dsp/dsp_plan_cache/` (see ADR 0061, *Disk Plan Persistence* section). For SDK deployment:

- SDZ/SDNB models produce deterministic plan bytes on every load — the disk cache eliminates recompilation on process restart.
- Pre-compiled plans can be distributed via the override directory (`dsp_plan_override/`), analogous to the existing Triton override mechanism.
- The C runtime (`sdxLoadBundle` → `compileModelPlan`) benefits automatically when the Java plan cache populates the disk cache on first run.
- Implemented: enriched SDZ archives embed target/compiler cache objects under `META-INF/sdx-cache/`, enabling single-artifact deployment without exposing plan, SPIR-V, Hexagon, Metal, or vendor-package formats to applications.

Configuration: `ND4J_DSP_PLAN_CACHE_DIR`, `ND4J_DSP_PLAN_CACHE_DISK_ENABLED`, `ND4J_DSP_PLAN_CACHE_FORCE_RECOMPILE`, `ND4J_DSP_PLAN_CACHE_OVERRIDE_DIR`.

### 10. Source SDZ Target Compilation Cache

The backend-neutral `nd4j-sdx-model` module owns:

- logical source identity over sorted SDZ entries, excluding `META-INF/sdx-cache/`,
- target/compiler/version/config/auxiliary-digest cache keys,
- immutable per-file-hashed objects with locked atomic publication,
- a `TargetCompiler` SPI plus a direct-process host adapter,
- packaging selected cached targets back into one enriched SDZ,
- extraction-only fail-closed resolution on mobile.

The original and enriched SDZ therefore share the same logical source identity. Changing compiler version, command fingerprint, target profile, quantization metadata, tokenizer, generation configuration, or explicit cache-key properties creates a new object. The application never chooses or parses a provider extension.

Tensor G3 Q4 activation calibration follows the same boundary. Calibration is
generated only after the canonical SDZ identity and final SameDiff operation
names exist. The target compiler executes a deterministic representative corpus
through that exact graph, records scalar per-operation activation/output
envelopes, and stores the source-bound contract as immutable compiler metadata.
Applications do not supply a provider-specific calibration sidecar. Raw mobile
preparation performs this step internally in the isolated CPU importer; host
staging may precompute and embed the same target object. The accelerator runtime
accepts only the finalized metadata embedded in the derived SDZ and remains
fail-closed when it is absent or invalid.

`libnd4j/tools/sdx-compile.sh` is only a launcher for this Java API. The former shell/Python DSPB construction logic is retired.

## Consequences

### Advantages

- One runtime contract for C/C++/JNI/Swift bindings.
- Direct SDZ loading from SDK runtime (no Java graph build requirement).
- Reuse of native plan compiler/executor preserves behavioral parity.
- Backend policy is explicit per platform and debuggable via execution reports.
- CUDA and AMD execution policies are represented in the public ABI.
- Disk plan persistence eliminates multi-second plan recompilation on process restart.
- One logical SDZ identity and compiler-owned cache replace provider-format branching in applications.

### Disadvantages

- AMD path depends on available GPU runtime plumbing in build/deploy environment.
- Fallback telemetry is coarse: `applied_backend` is the plan's in-force mode
  and `used_fallback` derives from segment capture failures / REPLAY_BLOCKED —
  a boolean, not a full per-segment reason graph.
- The native SDZ and legacy packed-DSPB compatibility readers are in-memory; archives must fit in host RAM.

## Milestones

1. `M1` (implemented): public C ABI, SDZ/SDNB loading, runtime/context/run lifecycle, SDK staging target.
2. `M2` (next): stricter segment-level backend capability checks and richer fallback reasons.
3. `M3` (mostly implemented): production packaging (`.xcframework`, `.aar`, Linux/Windows distributables, standalone `libsdx_*` flow incl. per-variant zips + stripped copies + Java wrapper staging). Remaining: enable `-Dlibnd4j.sdx.standalone=ON` in the release CI workflows.
4. `M4` (partial): `publish-sdx-runtime-sdk` action + `publish-sdk-release.yml` exist; `SdxCApiEndToEndTest` (platform-tests) exercises the full C ABI lifecycle against a real `.sdz`. Remaining: forced-mode conformance matrix across backends.

## References

- [ADRs/0061 - DynamicShapePlan Execution.md](./0061%20-%20DynamicShapePlan%20Execution.md)
- [ADRs/0074 - SDX Runtime Serving Protocol (REST + gRPC).md](./0074%20-%20SDX%20Runtime%20Serving%20Protocol%20(REST%20+%20gRPC).md)
- `libnd4j/include/dsp/runtime/dsp_runtime_c.h`
- `libnd4j/include/legacy/impl/DspRuntimeC.cpp`
- `libnd4j/include/graph/impl/NativeDynamicShapePlan.cpp`
- `libnd4j/include/graph/impl/SdzReader.cpp`
- `nd4j/.../samediff/execution/DspPlanDiskCache.java`
- `nd4j/nd4j-backends/nd4j-backend-impls/nd4j-sdx-model/`
