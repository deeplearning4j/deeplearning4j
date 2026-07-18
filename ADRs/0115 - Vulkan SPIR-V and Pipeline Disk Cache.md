# ADR 0115 - Vulkan SPIR-V and Pipeline Disk Cache

## Status

Implemented (July 15, 2026) — Tiers 0/1/2 and configuration are present.
"Implemented" here describes the cache source and its focused tests; it does not
claim that every Vulkan kernel, driver, or platform matrix has passed. A skipped
DSP-replay cache case is not validation evidence. Hardware claims require a
retained, non-skipped Vulkan dispatch/replay run on that hardware.

Proposed by: Adam Gibson (July 15, 2026)

Discussed with: Development Team

## Context

The Vulkan backend (ADR-0110/0111/0112) is 100% runtime-JIT. Every unique
(opName, MLIR module text) pair triggers, per process:

1. MLIR parse of the textual module
2. A 4-stage pass pipeline (`createVulkanOpLoweringPass` → GPU outlining +
   SPIR-V entry-point ABI + `createConvertGPUToSPIRVPass` →
   `createSPIRVLowerABIAttributesPass`/`createSPIRVUpdateVCEPass` →
   canonicalize/CSE) in `VulkanPipelineCache::mlirToSpirv()`
   (`libnd4j/include/graph/vulkan/VulkanPipelineCache.cpp:348-538`)
3. `mlir::spirv::serialize()` to SPIR-V words
4. `vkCreateShaderModule` + `vkCreateComputePipelines` — the driver's own
   SPIR-V→ISA backend compile

No `.spv` files are precompiled at build time; nothing is persisted. The only
cache is the in-memory `std::map` in `VulkanPipelineCache` keyed by
`opName + "|push=" + pushConstantBytes + "|" + mlirModuleStr`
(`VulkanPipelineCache.cpp:191-196`, lookup in `getOrCompile()` at `:238-281`).
Two disconnected instances exist per device — one device-lifetime
(`VulkanDeviceContext::shaderPipelineCache_`, `VulkanDeviceContext.cpp:586`)
and one per replay handle (`VulkanReplayHandle.h:288`) — so identical kernels
can be lowered twice in one process and always re-lowered on every process
start.

A driver-level `VkPipelineCache` object is already created per device
(`VulkanDeviceContext.cpp:576-581`) but is dead weight:
`vkCreateComputePipelines` passes `VK_NULL_HANDLE` instead of it
(`VulkanPipelineCache.cpp:655`), it is never seeded via `pInitialData`, never
read back via `vkGetPipelineCacheData`, and
`VkPhysicalDeviceProperties::pipelineCacheUUID` is never queried anywhere in
libnd4j.

This cold-start cost lands exactly where the backend is aimed: Android.
Adreno/Mali driver shader compilation is notoriously slow, and mobile
processes are cold-started (and killed) far more often than desktop JVMs.

The repository already has two proven disk-cache patterns to mirror:

- **Triton kernel disk cache** (`TritonGraphBackend_cache.cpp`, ADR 0071):
  FNV-1a-64 key over an ABI literal + `buildInfo()` (BuildStamp → automatic
  invalidation on rebuild) + compile parameters + full TTIR text + target
  arch; `ttir_<16hex>.ptx` + `.meta` sidecar; atomic `tmp.<pid>.<tid>` +
  rename writes; read-only override directory for pre-seeded deployment; LRU
  module eviction with reload-from-disk; batch preload.
- **DSP plan disk cache** (`DspPlanDiskCache.java`, ADR 0093): backend-neutral
  `DSP1` v5 plan bytes, model identity index, `dspVersion` +
  `nativeBuildFingerprint` validation.

**Tier 0 comes for free.** The serialized DSP plan bytes contain no backend,
device, or execution-mode fields (`DynamicShapePlan.serialize()`;
`graphExecutionMode` is applied at `compileNativePlan()` time and never
persisted), so `DspPlanDiskCache` already serves the Vulkan backend unchanged.
The `nativeBuildFingerprint` check keys entries to `libnd4jvulkan.so`'s
`buildInfo()` automatically — Vulkan-JVM plans get their own entries and
correct invalidation with zero code change.

What Vulkan is missing is the *kernel-artifact* layers: the equivalent of the
Triton PTX disk cache (skip the MLIR pipeline) plus the Vulkan-native driver
blob (skip the driver's SPIR-V→ISA compile).

## Decision

Introduce a three-tier disk cache for the Vulkan backend. Tier 0 is inherited;
Tier 1 mirrors `TritonGraphBackend_cache.cpp` for SPIR-V; Tier 2 persists the
driver's `VkPipelineCache` blob, which has no CUDA analogue and uses the
mechanism the Vulkan spec provides for exactly this purpose.

### Tier 0 — DSP plan bytes (inherited, no code change)

`DspPlanDiskCache` is used as-is. Required action: a smoke assertion in the
Vulkan test tier that a second plan compile under the Vulkan backend reports a
disk hit (guards against a future backend-specific field silently entering
`serialize()` without a `DSP_VERSION` bump).

### Tier 1 — SPIR-V module disk cache (`VulkanSpirvDiskCache`, C++)

New native component modeled line-for-line on the Triton cache. It sits inside
`VulkanPipelineCache::getOrCompile()` between the in-memory miss and
`mlirToSpirv()`. Because it is a process-wide singleton (like the Triton
cache statics), it also unifies the two in-memory `VulkanPipelineCache`
instances: the replay-handle instance and the device-context instance share
disk entries even though their in-memory maps stay separate.

#### Cache key

FNV-1a 64-bit (`graph/DspHashUtils.h` primitives, same as Triton), mixed in
this order:

1. ABI literal `"vulkan-spirv-disk-cache-v1"` (bump to invalidate all entries)
2. `buildInfo()` string — includes `BuildStamp` from
   `cmake/GenerateBuildStamp.cmake`, so every `libnd4jvulkan.so` rebuild
   invalidates by cache miss, identical to Triton
3. Device-caps tuple that alters codegen: `apiVersion`, `fp16`, `storage16`,
   `fp64`, `int64`, `int8` — the exact fields `VulkanPipelineCache` captures
   at construction (`VulkanPipelineCache.cpp:202-211`); they select the SPIR-V
   target environment and capability list, so they are the Vulkan analogue of
   Triton's `targetArch`
4. `pushConstantBytes` (currently always 0; part of pipeline-layout identity)
5. The full MLIR module text (analogue of the TTIR text — shapes, dtypes, and
   op arguments are baked into it; no specialization constants exist)

Excluded, with the same rationale as Triton (`TritonGraphBackend_cache.cpp:122-129`):
`startSlot`, `endSlot`, segment shape keys, and `deviceId`. Device *identity*
is deliberately not in the Tier-1 key: SPIR-V is portable across devices that
share the caps tuple. Driver-specific state lives in Tier 2.

Key formatted as 16 lowercase hex digits.

#### On-disk layout

| File | Contents |
|------|----------|
| `spv_<16hex>.spv` | Raw SPIR-V words exactly as produced by `mlir::spirv::serialize()` |
| `spv_<16hex>.meta` | `key=value` text sidecar (below) |

`.meta` fields: `cacheAbi`, `nativeBuildInfoHash` (informational, as in
Triton), `opName`, `entryPoint` (currently always `main`),
`pushConstantBytes`, `apiVersion`, `capsFp16/capsStorage16/capsFp64/capsInt64/capsInt8`,
`descriptorBindings` (semicolon-separated entries in argument order,
serializing exactly the fields of `SpirvModule::descriptorBindings` that
`createComputePipeline()` consumes), `spirvBytes`, `createdAt`.

`descriptorBindings` is load-bearing: on a disk hit `mlirToSpirv()` is
skipped, and that is where bindings are extracted today
(`VulkanPipelineCache.cpp:490-511`). Reflecting them back out of the `.spv`
was rejected — it duplicates a parser for data we had at write time. If the
`SpirvModule` binding struct ever changes shape, bump the ABI literal.
Implementation audit item: if any dispatch parameter is currently derived
inside `mlirToSpirv()` rather than computed by `VulkanSegmentRecorder` at
record time, it must be added to the sidecar too.

#### Directory resolution (mirrors `configuredOrDefaultTritonDir()`)

| Directory | Purpose | Default |
|-----------|---------|---------|
| Cache | Read/write | `~/.kompile/cache/vulkan/spirv_cache/` |
| Override | Read-only pre-seed, checked first, never written | `~/.kompile/cache/vulkan/spirv_override/` |

Both directories resolve through the four-level chain defined in
**Configuration** below (property/`Environment` setter → environment variable
→ `$HOME` default → relative fallback) — the same contract as
`DspPlanDiskCache.getCacheDir()` and `configuredOrDefaultTritonDir()`. On
Android there is no usable `$HOME`; applications point both directories at
app storage (e.g. `Context.getCacheDir()`) via the system properties or the
`Environment` setters before first graph execution.
The override directory is the APK pre-seed path: ship `.spv`+`.meta` pairs as
assets, unpack once, point the override property at them. Pre-seeded artifacts
are per caps-profile by construction (the caps tuple is in the key), so an APK
ships one set per supported profile (e.g. fp16-on and fp16-off).

#### Read path

In `getOrCompile()` after in-memory miss, before `mlirToSpirv()`:
override dir → cache dir. Validation: files open and non-empty; first word of
`.spv` equals the SPIR-V magic `0x07230203`; `.meta` `cacheAbi` matches;
`entryPoint` and `descriptorBindings` present. Any failure → treat as miss,
fall through to JIT, overwrite. On hit: reconstruct `SpirvModule` from blob +
sidecar and proceed directly to `createComputePipeline()`.

#### Write path

After `vkCreateComputePipelines` succeeds (end-to-end validated artifact —
we never persist SPIR-V no driver has accepted; tradeoff: a device that fails
pipeline creation caches nothing, which is the safe direction). Atomic write
per the Triton sequence: write `<file>.tmp.<pid>.<tid-hash>`, `std::rename`;
`.spv` first, `.meta` only if the `.spv` rename succeeded. Concurrent writers
produce identical bytes (deterministic serialization), so last-writer-wins is
idempotent. IO failure → warn and continue; the cache is a pure optimization.

An `alwaysCompile` knob bypasses reads and skips writes (Triton
`ND4J_TRITON_ALWAYS_COMPILE` semantics).

### Tier 2 — `VkPipelineCache` driver-blob persistence

Tier 1 removes the MLIR pipeline; the driver's SPIR-V→ISA compile — dominant
on mobile — is only removable through the driver's own cache.

1. **Wire the existing handle.** Pass `VulkanDeviceContext::pipelineCache()`
   as the second argument of `vkCreateComputePipelines`
   (`VulkanPipelineCache.cpp:655`, currently `VK_NULL_HANDLE`). Plumb the
   handle into both `VulkanPipelineCache` owners; a default-flags
   `VkPipelineCache` is internally synchronized, so sharing one handle across
   the device-context and replay-handle instances is legal without extra
   locking.
2. **Device identity.** Query `VkPhysicalDeviceProperties` (vendorID,
   deviceID, driverVersion, 16-byte `pipelineCacheUUID`) directly where the
   blob path is computed and validated (`VulkanDeviceContext.cpp` helpers) —
   the context already holds `physicalDevice_`, so widening
   `VulkanDeviceInfo` is unnecessary.
3. **Blob file.** One blob per physical device:
   `vkpc_<16hex>.bin` under `~/.kompile/cache/vulkan/pipeline_cache/` (the
   directory is configurable through the same four-level chain as the Tier-1
   directories — see Configuration), where
   `<16hex>` = FNV-1a over ABI literal `"vulkan-driver-pipeline-cache-v1"` +
   `vendorID` + `deviceID` + `driverVersion` + `pipelineCacheUUID`. Driver
   updates change the filename; old blobs orphan, matching Triton/DSP
   behavior. The blob is never shared across devices — the lesson from
   `JitSegmentCacheKey` initially lacking `deviceId` (cross-device CUmodule
   reuse → CUDA error 700) is enforced here by construction.
4. **Load.** In `VulkanDeviceContext::initialize()`
   (`VulkanDeviceContext.cpp:576`): read the blob, validate the spec-defined
   32-byte `VK_PIPELINE_CACHE_HEADER_VERSION_ONE` header (length ≥ 32, header
   version, `vendorID`/`deviceID` match, UUID `memcmp`) before setting
   `pInitialData`. On mismatch or `vkCreatePipelineCache` failure with data,
   retry once with an empty create info. (The spec obliges drivers to
   tolerate invalid data; Android driver quality makes pre-validation plus
   retry mandatory anyway.)
5. **Save.** `vkGetPipelineCacheData` → atomic tmp+rename write at two
   points: (a) `VulkanDeviceContext::destroy()` before
   `vkDestroyPipelineCache` (`VulkanDeviceContext.cpp:661-663`); (b) a
   kill-safety flush after `VulkanReplayHandle` capture finalization — mobile
   processes rarely exit cleanly, and end-of-capture is exactly when new
   pipelines were just created. Skip the flush when the data size is
   unchanged since the last write.
6. **Growth control.** `VkPipelineCache` has no eviction API. If the blob
   exceeds a budget (default 64 MB, configurable), skip `pInitialData` on the
   next start and let the cache regenerate — stale-entry GC by regeneration.

### Configuration

Every directory and knob is configurable through the same surfaces as the DSP
plan disk cache and the Triton cache. The DSP contract
(`DspPlanDiskCache.getCacheDir()`, `DspPlanDiskCache.java:118-139`) is
property → env var → default; Triton's `configuredOrDefaultTritonDir()` adds
a `$HOME`-empty relative fallback. The Vulkan tiers use the union — every
directory resolves:

1. **Explicit configuration** — the native config value, settable two
   equivalent ways: the Java system property, or the `Environment` setter
   (`Nd4j.getEnvironment().setVulkanSpirvCacheDir(...)`). The setter is the
   first-class path on Android, where setting system properties before
   `Nd4j` init is awkward.
2. **Environment variable** — read natively at config-init time (the
   `TritonConfig`/`DspConfig::initFromEnvironment()` pattern), so caches work
   in pure-native/AOT contexts with no JVM.
3. **Default** — `~/.kompile/cache/vulkan/<leaf>`.
4. **Relative fallback** — `.kompile/cache/vulkan/<leaf>` when `$HOME` is
   empty (Triton behavior; load-bearing on Android).

Wiring follows the AGENTS.md configuration rule via the Triton chain, whose
Vulkan legs are already generated and working end-to-end (the existing
`tritonCacheDir()` round-trips through `VulkanEnvironment` today):

- Native fields + accessors on `Environment`/`VulkanConfig`
  (`libnd4j/include/system/`), mirroring `TritonConfig`; env vars read at
  init.
- A new `VulkanEnvironmentConfig` interface in
  `org.nd4j.linalg.factory.config` mirroring `TritonEnvironmentConfig.java`
  (default no-op getters/setters so non-Vulkan backends need no changes),
  surfaced on `Environment.java` the same way `tritonCacheDir()` is
  (`Environment.java:1072`).
- `VulkanEnvironment.java` overrides delegating to the generated `Nd4jVulkan`
  bindings — the exact pattern of its existing
  `tritonCacheDir()`/`setTritonCacheDir()` overrides
  (`VulkanEnvironment.java:805-806`). Regenerate the preset to expose the new
  native accessors; never edit generated files.
- Property constants in `ND4JSystemProperties.java` using the
  `nd4j.environment.*` convention for native-backed config (matching
  `ENV_TRITON_CACHE_DIR = "nd4j.environment.tritonCacheDir"`,
  `ND4JSystemProperties.java:1692`); env-var constants in
  `ND4JEnvironmentVars.java` (matching `ND4J_TRITON_CACHE_DIR`,
  `ND4JEnvironmentVars.java:302`). Constants always — never raw strings.
- `platform-tests/pom.xml` surefire `<environmentVariables>` entries for the
  env vars (Surefire forks a new JVM; `export` does not propagate).

| Java property (`ND4JSystemProperties`) | Env var (`ND4JEnvironmentVars`) | `Environment` accessor | Default | Purpose |
|----------------------------------------|--------------------------------|------------------------|---------|---------|
| `nd4j.environment.vulkanSpirvCacheEnabled` | `ND4J_VULKAN_SPIRV_CACHE_ENABLE` | `vulkanSpirvCacheEnabled()` | `true` | Tier 1 on/off |
| `nd4j.environment.vulkanSpirvCacheDir` | `ND4J_VULKAN_SPIRV_CACHE_DIR` | `vulkanSpirvCacheDir()` | `~/.kompile/cache/vulkan/spirv_cache/` | Tier 1 directory |
| `nd4j.environment.vulkanSpirvOverrideDir` | `ND4J_VULKAN_SPIRV_OVERRIDE_DIR` | `vulkanSpirvOverrideDir()` | `~/.kompile/cache/vulkan/spirv_override/` | Pre-seed directory |
| `nd4j.environment.vulkanAlwaysCompile` | `ND4J_VULKAN_ALWAYS_COMPILE` | `vulkanAlwaysCompile()` | `false` | Bypass Tier 1 read+write |
| `nd4j.environment.vulkanPipelineCacheEnabled` | `ND4J_VULKAN_PIPELINE_CACHE_ENABLE` | `vulkanPipelineCacheEnabled()` | `true` | Tier 2 on/off |
| `nd4j.environment.vulkanPipelineCacheDir` | `ND4J_VULKAN_PIPELINE_CACHE_DIR` | `vulkanPipelineCacheDir()` | `~/.kompile/cache/vulkan/pipeline_cache/` | Tier 2 directory |
| `nd4j.environment.vulkanPipelineCacheMaxBytes` | `ND4J_VULKAN_PIPELINE_CACHE_MAX_BYTES` | `vulkanPipelineCacheMaxBytes()` | `67108864` | Blob budget |
| `nd4j.environment.vulkanKernelDump` | `ND4J_VULKAN_KERNEL_DUMP` | `vulkanKernelDump()` | `false` | Dump MLIR/.spv artifacts (Triton `KERNEL_DUMP` parity) |

Each accessor has a matching setter (`setVulkanSpirvCacheDir(...)`, etc.).

### Diagnostics

DSP_DIAG `COMPILE`/`JIT` events for Tier-1 disk hit/miss/store and Tier-2
blob load/save (byte counts), extending the existing hit/miss logging in
`getOrCompile()`. Counters live on the native `VulkanConfig`
(`spirvDiskHits/Misses/Stores`, `pipelineBlobLoads/Saves`,
`clearCacheCounters()`) and are surfaced on the Java `Environment`
(`vulkanSpirvDiskHits()` etc.) — the TritonConfig
`moduleResidencyWarnFireCount` precedent: native `sd_printf` output cannot be
intercepted from Java, so tests observe cache behavior through counters.

### Invalidation

| Trigger | Tier 1 | Tier 2 |
|---------|--------|--------|
| `libnd4jvulkan.so` rebuild | `buildInfo()` in key → miss, new entries; orphans accumulate (accepted, matches Triton/DSP) | Unaffected (driver-keyed); superseded driver-internal entries sit unused |
| GPU driver update | Unaffected (SPIR-V is driver-independent) | `driverVersion`/UUID → new filename; old blob orphaned |
| ABI literal bump | Full invalidation | Full invalidation |
| Corrupt/truncated file | Validation fail → recompile + overwrite | Header check fail → start with empty cache |
| Caps profile change (e.g. fp16 toggled) | Different key → separate entries | Same blob (driver keys internally) |
| Delete cache directory | Recreated on next write | Recreated on next save |

### Phasing

- **Phase 1 — Tier 2 wiring.** Smallest diff (pass the handle, add identity
  fields, load/save the blob) and an immediate mobile win. All changes in
  `.cpp` files; no header cascade beyond `VulkanDeviceInfo`/plumbing.
- **Phase 2 — Tier 1.** New `VulkanSpirvDiskCache.{h,cpp}` (new header, low
  blast radius) + `getOrCompile()` integration + config + tests. The big
  cold-start win: warm start becomes file read + pipeline creation from a
  warm driver cache.
- **Phase 3 — Triton LRU/preload parity (deferred).** Pipeline residency
  budget with destroy-and-reload-from-Tier-1 eviction and a batch preload
  pass after capture, mirroring `TritonGraphBackend_lru.cpp`/`_preload.cpp`.
  Only if mobile memory data shows resident `VkPipeline` pressure; unlike
  512 MB CUmodules, pipeline host/driver memory is unlikely to dominate.

### Testing

All in `platform-tests/` (lavapipe-compatible; lavapipe implements pipeline
cache data, and Mesa's `pipelineCacheUUID` changes per Mesa build — which
exercises the invalidation contract rather than breaking it). Directories
pointed at temp paths via `-D` properties wired through the surefire
`<environmentVariables>` block — never `export`.

`VulkanDiskCacheTest` (as implemented):
1. `testConfigRoundTrip` — setters/getters round-trip Java `Environment` →
   generated bindings → native `VulkanConfig`.
2. `testDiskCachePhase` — cross-JVM phase harness (one phase per surefire
   fork, fixed dir wired through the pom env-var mapping, which it also
   validates): **cold** = misses then stores, valid `.spv`/`.meta` pairs;
   **warm** = fresh JVM Tier-1 disk hits with zero re-stores (key stability
   across processes) plus Tier-2 blob present, header-valid, and loaded at
   device-context init (counter read before clearing — the blob loads during
   backend init, before test bodies run); **corrupt** = flipped magic →
   miss → JIT fallback → entries re-stored and healed; **bypass** =
   `alwaysCompile` via the property→env-var chain → zero reads and writes.
   Runs on the eager kernel path, which is independent of the in-flight DSP
   input-staging frontier.
3. `testDspReplayPathPopulatesCache` — per-plan pipeline caches give true
   in-JVM fresh-instance disk hits through the VULKAN_REPLAY path;
   self-skips on the known in-flight staging error and arms automatically
   when that lands.

Still open (follow-ups): override-dir pre-seed test, tampered-UUID
empty-start test, and the Tier-0 `DspPlanDiskCache` smoke under the Vulkan
backend — all blocked on or naturally paired with the DSP replay path
becoming executable.

## Consequences

### Advantages

- Warm process start skips the entire MLIR parse/lower/serialize pipeline
  (Tier 1) and the driver's SPIR-V→ISA compile (Tier 2) — the two dominant
  cold-start costs, addressed independently and composably.
- Android APKs can ship pre-seeded SPIR-V via the override directory; first
  launch on a device still warms Tier 2 once and is fast thereafter.
- Reuses the proven Triton mechanics verbatim (keying, atomic writes,
  override, invalidation-by-build-stamp) and the Vulkan-spec-native blob
  mechanism — no novel cache machinery.
- Tier 0 already works; this ADR closes the gap without touching the DSP
  plan layer.

### Tradeoffs

- Orphan files accumulate on rebuilds/driver updates; no GC (accepted,
  matches Triton and ADR 0093).
- `descriptorBindings` metadata is duplicated into the sidecar; struct
  changes require ABI-literal discipline.
- Tier-2 blobs are opaque driver data; budget-and-regenerate is the only
  growth control.
- Writing Tier 1 only after full pipeline success means a driver that
  rejects a pipeline never populates the cache for that kernel (safe but
  conservative).

## Relationship to Other ADRs

- **ADR 0093** (DSP Plan Disk Persistence): Tier 0 inherited unchanged;
  directory/property naming and atomic-write conventions copied.
- **ADR 0071** (Triton Graph Backend): Tier 1 is the direct analogue of
  `TritonGraphBackend_cache.cpp`; Phase 3 mirrors `_lru`/`_preload`.
- **ADR-0110/0111/0112** (Vulkan Backend / Device Management / Java Layer):
  this ADR fills the "no disk persistence" gap those ADRs left out of scope;
  Tier 2 completes the per-device `VkPipelineCache` component ADR-0111
  declared but left disconnected.
- **ADR 0085** (MLIR JIT Compilation Backend): Tier 1 caches the output of
  the MLIR→SPIR-V leg only; NVRTC/PTX in-memory JIT caches remain
  process-local and out of scope.

## Key Files

| File | Role |
|------|------|
| `libnd4j/include/graph/vulkan/VulkanSpirvDiskCache.{h,cpp}` | **New** — Tier 1 (key, read/write, dirs, validation) |
| `libnd4j/include/graph/vulkan/VulkanPipelineCache.cpp:238` (`getOrCompile`) | Tier-1 read/write insertion point |
| `libnd4j/include/graph/vulkan/VulkanPipelineCache.cpp:655` | Pass device `VkPipelineCache` instead of `VK_NULL_HANDLE` |
| `libnd4j/include/graph/vulkan/VulkanDeviceContext.cpp:576-581` | Tier-2 blob load (`pInitialData`) |
| `libnd4j/include/graph/vulkan/VulkanDeviceContext.cpp:661-663` | Tier-2 blob save before destroy |
| `libnd4j/include/graph/vulkan/VulkanDeviceManager.cpp:333-337` | Store `deviceID`, `driverVersion`, `pipelineCacheUUID` |
| `libnd4j/include/graph/vulkan/VulkanReplayHandle.cpp` | Post-capture Tier-2 flush hook |
| `nd4j/.../common/config/ND4JSystemProperties.java` | 8 new `nd4j.environment.vulkan*` property constants |
| `nd4j/.../common/config/ND4JEnvironmentVars.java` | 8 new `ND4J_VULKAN_*` env-var constants |
| `nd4j/.../linalg/factory/config/VulkanEnvironmentConfig.java` | **New** — config interface, mirror of `TritonEnvironmentConfig.java` |
| `nd4j/.../linalg/factory/Environment.java` | Surface the Vulkan cache accessors/setters |
| `nd4j/.../linalg/vulkan/VulkanEnvironment.java` | Delegating overrides via generated `Nd4jVulkan` bindings (preset regen, never hand-edited) |
| `libnd4j/include/system/` (`Environment`/`VulkanConfig`) | Native config fields, env-var init, JNI-exposed accessors |
| `platform-tests/pom.xml` | Surefire environment-variable wiring |
| `platform-tests/.../VulkanDiskCacheTest.java` | **New** — test matrix above |
