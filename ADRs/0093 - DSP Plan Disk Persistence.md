# ADR 0093 - DSP Plan Disk Persistence

## Status

Implemented

Proposed by: Adam Gibson (May 24, 2026)

Discussed with: Development Team

## Context

DynamicShapePlan (DSP) compilation produces serialized plan bytes (`DSP1` v5 binary format) that are deterministic for a given SameDiff graph structure. For GGML-imported models, the graph topology is fully determined by the GGUF metadata — the same model file always produces byte-for-byte identical serialized plans across loads. Yet every JVM start recompiles the plan from scratch through the full pipeline: Java DAG builder, `DynamicShapePlanCompiler.compile()`, `DynamicShapePlan.serialize()`, JNI dispatch, and `NativeDynamicShapePlan::fromSerializedPlan()`.

Meanwhile, Triton kernel compilation results are already persisted to disk (`~/.kompile/cache/triton/triton_cache/`) with FNV-1a keyed `.ptx`/`.meta` files, atomic writes, an override directory for pre-seeded deployments, and LRU reload-on-eviction. The Triton cache pattern is proven, thread-safe, and cross-process safe. DSP plans should follow the same pattern.

The per-thread `NativePlanCache` key (`{outputSetHash, phShapeContentHash, phCount, graphExecutionMode, threadId}`) — which handles shape-keyed dispatch and thread isolation — is an orthogonal layer. Disk persistence operates above it, caching the serialized plan bytes (graph structure) that all threads and all shape signatures share.

## Decision

We introduce a Java-side disk cache for serialized DSP plan bytes, modeled directly after `TritonGraphBackend_cache.cpp`.

### Cache Key

FNV-1a 64-bit hash of the `DynamicShapePlan.serialize()` output bytes. Computed by `DynamicShapePlan.computeStructureHash(byte[])`. The hash captures the full graph structure: op names, wiring, iArgs/tArgs/bArgs/dArgs/sArgs, control flow metadata, external input names, release schedule, and requested output ordering.

File naming: `dsp_<16-hex-lowercase>.bin` + `dsp_<16-hex-lowercase>.meta`.

A model identity index (`dsp_model_<16-hex>.idx`) maps a hash of `(sorted output names + external input names)` to the structure hash. This enables cross-JVM plan lookup without recompiling the plan first — the index can be read using only information available from the `SameDiff` graph before DAG compilation.

### File Format

| File | Contents |
|------|----------|
| `.bin` | Raw `DynamicShapePlan.serialize()` output (DSP1 v5 binary). `fromSerializedPlan()` reads this format directly. |
| `.meta` | Key=value text sidecar: `dspVersion`, `numSlots`, `numExternalInputs`, `numRequestedOutputs`, `outputSet`, `planBytes`, `structureHash`, `nativeBuildFingerprint`, `createdAt`. |
| `.idx` | Model identity index: single line containing the 16-hex structure hash. |

### Cache Directory

| Directory | Purpose | Default |
|-----------|---------|---------|
| Cache | Read/write cache for compiled plans | `~/.kompile/cache/dsp/dsp_plan_cache/` |
| Override | Read-only directory for pre-seeded plans (highest priority) | `~/.kompile/cache/dsp/dsp_plan_override/` |

Resolution priority (mirrors `configuredOrDefaultTritonDir()`):
1. Java system property (`-Dnd4j.dsp.planCache.diskDir` / `-Dnd4j.dsp.planCache.overrideDir`)
2. Environment variable (`ND4J_DSP_PLAN_CACHE_DIR` / `ND4J_DSP_PLAN_CACHE_OVERRIDE_DIR`)
3. Default path

### Write Path

In `DynamicShapePlanExecutor.compileNativePlan()`, immediately after `cachedSerializedPlan = serialized` is assigned. Atomic write follows the Triton pattern:

1. Write to `<file>.tmp.<pid>.<tid>` (unique per writer)
2. `Files.move()` atomic rename to final path
3. IOException → log WARN, continue (disk cache is a pure optimization)

Thread safety: concurrent writers to the same hash produce identical bytes (serialization is deterministic), so the last-writer-wins rename is idempotent.

### Read Path

In `DynamicShapePlanExecutor.compileNativePlan()`, before `plan.serialize()`:

1. Try model-identity lookup: `DspPlanDiskCache.tryLoadByModelIdentity(requestedOutputs, externalInputKeys)` — reads `.idx` file, resolves to structure hash, loads `.bin`
2. If miss, fall through to `plan.serialize()` (fresh compilation)
3. Override directory is checked before regular cache directory

Validation: `.meta` `dspVersion` must match runtime `DSP_VERSION` (currently 5). `.bin` must pass `DynamicShapePlan.isValidSerializedPlan()` (magic + version check). The `.meta` `nativeBuildFingerprint` must equal the current `buildInfo()` — this scopes entries per native `.so` build (and therefore per backend: a Vulkan JVM and a CUDA JVM keep separate valid entries and never reuse each other's). The fingerprint MUST be canonicalized to a single line: `buildInfo()` is multi-line, the sidecar is line-oriented, and an un-flattened value truncates on read so the comparison never passes — a bug that silently made every disk read miss until fixed (July 16, 2026; regression guard: `DspPlanDiskCacheRoundTripTest`). Canonical source: native `buildInfoFingerprint()` (`build_info.h/.cpp` — newlines→spaces, trimmed), exposed on `NativeOps` with a throwing default; `DspPlanDiskCache.getNativeBuildFingerprint()` prefers it and falls back to flattening `buildInfo()` byte-identically on backend bindings generated before the API existed, so the two paths always agree for the same `.so`.

### Cache Invalidation

| Trigger | Behavior |
|---------|----------|
| `dspVersion` mismatch | Entry ignored, plan recompiled, files overwritten |
| Graph structure change | Different hash → new file. Old files become orphans (same as Triton). |
| `-Dnd4j.dsp.planCache.forceRecompile=true` | Bypass read, still write. Rebuilds cache with fresh plans. |
| Delete cache directory | Recreated on next write. |

### Per-Thread Handling

The disk cache stores one `.bin` per graph structure — no thread dimension. `NativePlanCache` creates per-thread `NativeDynamicShapePlan*` instances from the same `cachedSerializedPlan` bytes via `fromSerializedPlan()`. Whether those bytes came from disk or from `plan.serialize()` is transparent to the C++ layer. The donor-plan logging in `NativePlanCache::getOrInsert()` continues to track cross-thread plan lineage regardless of byte source.

### Configuration

| Java Property | Environment Variable | Default | Purpose |
|---------------|---------------------|---------|---------|
| `nd4j.dsp.planCache.diskEnabled` | `ND4J_DSP_PLAN_CACHE_DISK_ENABLED` | `true` | Enable/disable |
| `nd4j.dsp.planCache.diskDir` | `ND4J_DSP_PLAN_CACHE_DIR` | `~/.kompile/cache/dsp/dsp_plan_cache/` | Cache directory |
| `nd4j.dsp.planCache.overrideDir` | `ND4J_DSP_PLAN_CACHE_OVERRIDE_DIR` | `~/.kompile/cache/dsp/dsp_plan_override/` | Override directory |
| `nd4j.dsp.planCache.forceRecompile` | `ND4J_DSP_PLAN_CACHE_FORCE_RECOMPILE` | `false` | Force recompile |

C++ side: `DspConfig.h` exposes `planCacheDiskEnabled()`, `planCacheDiskDir()`, `planCacheDiskForceRecompile()`, `planCacheOverrideDir()`. Read from environment via `DspConfig::initFromEnvironment()`.

## Consequences

### Advantages

- Eliminates plan recompilation on JVM restart for unchanged models.
- Mirrors the proven Triton disk cache pattern — atomic writes, override directory, version-based invalidation.
- Transparent to C++ layer — no changes to the hot execution path.
- Model identity index enables cross-JVM plan reuse without recompilation.
- Override directory supports pre-seeded deployment (ship plans alongside model artifacts).

### Tradeoffs

- Orphan files accumulate when graph structures change. No automatic garbage collection (matches Triton behavior).
- Disk I/O on first compile adds ~milliseconds (amortized over many JVM starts).
- Model identity index assumes deterministic graph construction — non-deterministic graph builders would produce cache misses (but no correctness issues).

## Relationship to Other ADRs

- **ADR 0061** (DynamicShapePlan Execution): Disk persistence section added. Plan serialization format (`DSP1` v5) is the persistence unit. Shape-keyed `NativePlanCache` operates below the disk layer.
- **ADR 0073** (DSP Self-Contained Runtime SDK): SDK deployments benefit from disk plan cache automatically. Override directory enables pre-warmed single-artifact deployment.
- **ADR 0079** (NativeDynamicShapePlan Structural Refactoring): Disk persistence is step 10, building on the shape-keyed plan cache (step 9) and slot immutability guarantees.
- **ADR 0071** (Triton Graph Backend): Disk cache design mirrors `TritonGraphBackend_cache.cpp` architecture — FNV-1a keying, atomic writes, override directory, `.meta` sidecar.

## Key Files

| File | Purpose |
|------|---------|
| `nd4j/.../samediff/execution/DspPlanDiskCache.java` | Disk cache manager — read, write, model identity index, directory resolution |
| `nd4j/.../samediff/execution/DynamicShapePlan.java` | `computeStructureHash(byte[])` — FNV-1a 64-bit |
| `nd4j/.../samediff/execution/DynamicShapePlanExecutor.java` | Disk cache integration in `compileNativePlan()` |
| `nd4j/.../common/config/ND4JSystemProperties.java` | 4 property constants |
| `libnd4j/include/system/config/DspConfig.h` | C++ config fields + accessors |
| `libnd4j/include/system/config/impl/DspConfig.cpp` | Environment variable reading |
| `platform-tests/pom.xml` | Surefire environment variable wiring |
