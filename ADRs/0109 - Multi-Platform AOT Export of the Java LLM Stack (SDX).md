# ADR 0109 - Multi-Platform AOT Export of the Java LLM Stack (SDX)

## Status

Implemented (linux-x86_64 cpu first; other platforms wired progressively)

Proposed by: Adam Gibson (July 2026)

## Context

The SDX Runtime SDK (ADR 0073) ships a JVM-free C runtime (`libsdx_cpu` /
`libsdx_cuda`, `dsp_runtime_c.h`) plus six language wrappers per platform/variant.
That covers **graph execution** without a JVM — but the highest-level capabilities
of the stack live in Java and were previously JVM-only:

- **samediff-llm** — the full generation pipeline (prefill/decode, in-graph KV cache,
  INT8 KV, sampling, chat templates, sessions), tokenization, fine-tuning
- **nd4j-ggml** — GGUF/GGML import (pure Java) that produces the SameDiff graphs the
  C runtime executes
- **tokenizers-native** — Rust HF tokenizers via JavaCPP

ADR 0099 established GraalVM Native Image metadata for nd4j-api / nd4j-common /
nd4j-native, and the `graalvm-native-image-example` in deeplearning4j-examples
proved an ND4J CPU-backend binary builds and runs (eager ops + SameDiff) under
native-image. This ADR extends that groundwork from "example" to "shipped SDK
artifact".

## Decision

AOT-compile the Java LLM stack with GraalVM native-image and export it per
platform as an SDX release asset: `sdx-llm-aot-<platform>-<variant>.zip`.

### Artifacts

One Maven module — `nd4j/sdx-aot` (profile `sdx-aot`, images with `-Pnative`) —
produces two images from the same classpath and metadata:

1. **`bin/sdx-llm`** — CLI executable
   (`org.eclipse.deeplearning4j.sdx.aot.SdxLlmCli`): `generate`, `import`
   (GGUF → SDZ for the C runtime — previously impossible without a JVM),
   `tokenize`, `info`.
2. **`lib/libsdx_llm.{so,dylib,dll}`** — shared library exposing the `sdxLlm*`
   C ABI declared in `nd4j/sdx-aot/include/sdx_llm_c.h`
   (`SdxLlmCApi`, `@CEntryPoint`). Conventions mirror `dsp_runtime_c.h`
   (status codes, out-params, last-error query) so the existing six language
   wrappers can bind it the same way they bind the C runtime.

Both are thin shells over `SdxLlmCore`, which reuses the exact benchmark-proven
pipeline path (`GGMLModelImport.importModel` → `HuggingFaceTokenizer` →
`GenerationPipelineConfig` → `GenerationPipeline`). Options cross the ABI as JSON
parsed with the shaded Jackson **tree API only** — no databind reflection.

### Platform matrix

GraalVM native-image targets: **linux-x86_64, linux-arm64, windows-x86_64,
macos-x86_64, macos-arm64** — cpu variant first, cuda as phase 2
(`-Psdx-aot-cuda` swaps the backend dependency; untested).

**Android and iOS are explicitly out of scope** — native-image does not target
them; the JVM-free C runtime (`libsdx`) remains the mobile path (AAR /
XCFramework packaging, ADR 0073).

### Native Image configuration strategy

- Each producing module owns its metadata in `META-INF/native-image/…` inside its
  jar (auto-discovered): nd4j-api, nd4j-common, nd4j-native (ADR 0099),
  tokenizers-native (JavaCPP-generated per platform), and now samediff-llm
  (defers the tokenizer wrapper + native binding packages to runtime init).
- The `sdx-aot` module's own `native-image.properties` carries the flag set proven
  by the example build (runtime-init list for JavaCPP/Nd4j statics, `-H:+JNI`,
  charset/URL enablement, `--no-fallback`).
- **Resource bundling goes through `resource-config.json` only, never
  properties-file `-H:IncludeResources` args** — the JSON channel is what actually
  bundled every working resource (nd4j-api's root
  `nd4j-op-def.pbtxt`/`functions.properties`/onnx descriptors, JavaCPP's generated
  config for the tokenizers bindings, this module's `logback.xml`); the `-H:` args
  in properties files are experimental and were observed not to apply.
- **Native binaries are side-loaded, not embedded** — the established kompile
  pattern (`NativeProfileBuilder` / `NativeLibraryResolver` in the kompile repo).
  `-H:ExcludeResources=.*\.(so|so\..*|dylib|dll|a|lib)$` build args (CLI-arg
  channel — excludes beat the `resource-config.json` includes shipped in the
  nd4j-native/openblas/tokenizers jars) keep every native library out of the
  image; `maven-dependency-plugin:unpack-dependencies` collects them into
  `target/native-libs` at build time and the SDK zip ships them flattened in
  `lib/`. At startup `SdxNativeLibs.bootstrap()` (called before any ND4J/tokenizer
  init from both the CLI main and `sdxLlmLoadModel`) resolves
  `SDX_NATIVE_LIB_DIR` → `<binary-dir>/lib` → `<binary-dir>/../lib` →
  `~/.javacpp/cache`, then sets `org.bytedeco.javacpp.pathsFirst=true`, prepends
  `java.library.path`, and pins `org.bytedeco.javacpp.cachedir` — the exact
  property set kompile ships. On a JVM with platform-classifier jars on the
  classpath the bootstrap is a no-op.
- The validated in-image failure ladder and its fixes, all upstreamed to the owning
  module's metadata:
  1. `HuggingFaceTokenizer` reaches the JavaCPP tokenizer bindings **reflectively** →
     samediff-llm `reflect-config.json` (TokenizersNative + Opaque handles + javacpp
     pointer classes).
  2. Shaded-protobuf `FieldAccessorTable` reflection when `OpDescriptorHolder` parses
     `nd4j-op-def.pbtxt` → all `org.nd4j.ir.*` classes registered in nd4j-api's
     `reflect-config.json`.
  3. `SameDiff.dup()` (GraphOptimizer) round-trips shards whose manifest uses **Java
     serialization** → nd4j-api `serialization-config.json` (`LinkedHashMap`,
     boxed types; `Pair` was already registered in nd4j-common).
- Remaining gaps are closed iteratively with the tracing agent
  (`-agentlib:native-image-agent`) run over `sdx-llm generate`; merged output is
  checked into the owning module's config dir.

### Build interface, packaging and CI (all Maven-native)

- **`-Pnative` is the single AOT switch**; **`-Pcuda` composes** for the CUDA
  variant (the module declares a local profile with the same id as the repo-wide
  cuda profile, flipping `sdx.aot.variant`/backend artifact).
- **Optimized-math spins compose through `javacpp.platform.extension`** (the
  repo-wide leading-dash composites: `-avx2`, `-avx512`, `-onednn-avx2`,
  `-armcompute`, `-cudnn`, …): the extension selects the backend classifier jar
  whose native math library gets side-loaded, and becomes part of the package
  identity — `sdx-aot-<version>-<platform><extension>-<variant>-aot.zip`, with
  the spin recorded in `aot-manifest.json`. The CI workflows expose
  `aotAllSpins=1` to build one AOT package per optimized matrix combo (each
  combo installs its spin's backend jar immediately before the AOT step); the
  base spin stays the default. Wrappers need nothing per-spin — a package IS a
  spin, and `SDX_NATIVE_LIB_DIR` can always point at a different spin's `lib/`.
  All spins/variants coexist in one `target/` via per-spin directories
  (`target/native/<variant><extension>` etc.).
- **Toolchain follows the `javacpp.platform.compiler` convention** (same property
  nd4j-backend-impls and tokenizers-native use): an OS-activated profile defaults
  it to `/usr/bin/gcc` on Linux and feeds it to `-H:CCompilerPath`. It MUST be a
  distro toolchain — a linuxbrew gcc embeds the brew dynamic linker as ELF
  interpreter, whose `ld.so.cache` cannot see `/usr/lib64`; the CUDA driver then
  cannot dlopen `libnvidia-ptxjitcompiler.so.1` (error 221 on every Triton module
  load, graceful fallback) and the binary only runs where brew glibc exists.
  Override on such machines: `-Djavacpp.platform.compiler=/usr/bin/x86_64-redhat-linux-gcc`.
- **Packaging is part of the module build**: an antrun execution stages
  `bin/ lib/ include/ share/sdx` (images renamed to convention, side-loaded
  natives flattened, the JavaCPP-fabricated `libopenblas_nolapack.so.0` alias
  materialized as a copy, AWT/jsound JDK natives taken from `${java.home}` — the
  GraalVM running the build), and maven-assembly attaches
  `sdx-aot-<version>-<javacpp.platform>-<variant>-aot.zip`, which flows through
  install/deploy like the platform-classified backend jars.
- Composite action `.github/actions/build-sdx-aot` (setup-graalvm 21 →
  `mvn -Psdx-aot,native[,cuda] -pl :sdx-aot package` → CLI smoke test → upload
  the attached zip + optional `sdk-v<version>` release upload). Wired into
  `build-deploy-linux-x86_64.yml` behind the `buildAot` input (default on), base
  matrix combo only, `continue-on-error` so SDK builds never block on AOT.
  Remaining platform workflows adopt the same step as they are validated.
- AOT zips are **separate release assets** rather than folded into the SDK zips:
  not every SDK consumer wants them, and the CUDA variant is large. Embedding
  into the SDK zip via `SdxRuntimePackage.cmake` stays possible later without
  ABI changes.

### Threading model (v1)

A `sdx_llm_runtime_t*` is a GraalVM isolate thread: create, use, destroy from one
OS thread; one runtime per thread for concurrency. Multi-thread attach
(`ATTACH_THREAD` builtin) is a v2 extension.

## Consequences

- Non-JVM SDX consumers gain GGUF import, tokenization, and full-pipeline
  generation; `sdx-llm import` closes the "SDZ models must be produced on a JVM"
  gap in the pure-C SDK story.
- Side-loading keeps the image itself small and dedupes natives with the rest of
  the SDK package, at the cost of the binary no longer being single-file: the
  `lib/` directory (or `SDX_NATIVE_LIB_DIR`) must travel with it. Measured on
  linux-x86_64 cpu: each image dropped from 1.4 GB (embedded natives) to 252 MB;
  the SDK zip from 755 MB to 435 MB while now serving both artifacts from one
  `lib/`. One packaging subtlety inherited from kompile's `build-dist.sh`:
  JavaCPP fabricates alias sonames at cache-extraction time (e.g.
  `libopenblas_nolapack.so.0`), so the packager recreates them as symlinks and
  the zip is built with `-y`. kompile's additional image-tuning flags
  (`-H:LargeArrayThreshold=8192`, `-Dorg.bytedeco.javacpp.nopointergc=true`,
  build-time-initialized logback) are documented prior art but not yet adopted
  here — evaluate separately.
- Every new reflectively-reached class in the LLM path must land in module
  metadata; the CI smoke test only covers CLI startup — a model-based generate
  smoke test (small GGUF fixture) is the needed follow-up gate.
- CUDA images multiply size and add driver coupling; deferred until the CPU
  images are validated in the release flow.

## Related ADRs

- [0073](0073%20-%20DSP%20Self-Contained%20Runtime%20SDK%20and%20SDZ%20Deployment.md) — SDX runtime SDK and SDZ deployment
- [0099](0099%20-%20GraalVM%20Native%20Image%20Support.md) — Native Image metadata for ND4J modules
- [0096](0096%20-%20LLM%20Generation%20Pipeline.md) — the pipeline being exported
