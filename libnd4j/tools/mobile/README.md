# Mobile accelerator builds

This directory owns reproducible Android ARM64 accelerator builds for the offline chat stack. Profiles are declarative. The wrappers discover Android NDK r28, JDK 17, Maven, bounded parallelism, and a shared `/tmp/sdx-android-build` root; `SDX_*` environment variables and command-line options remain available as explicit overrides. Licensed vendor paths remain caller inputs.

Core rules:

- The application and public SDK accept one canonical SameDiff model format: `.sdz`.
- Hardware objects such as SPIR-V, Hexagon kernels, Metal libraries, and LiteRT-LM packages are compiler/cache internals, never formats selected by the app.
- Accelerator profiles disable BLAS and reject implicit slot-by-slot or application-level host fallback. Tensor G3 alone uses an explicit mixed-backend policy: EdgeTPU through NNAPI first, ACL/NEON for supported CPU segments, then bounded functional replay.
- Vulkan, Qualcomm, and Tensor G3 lowering/replay stay in SDX/libnd4j. Tensor G5 uses Google's direct LiteRT-LM NPU ABI.
- JavaCPP is transport only. Execution, tokenizer/KV state, sampling, streaming, compilation, and cache ownership remain in SDX or the selected native provider.
- Offline mode requires all source, Maven, Cargo, npm/Bazel, LFS, SDK, and vendor inputs to be pre-cached.
- AAR verifiers check AArch64 ELF closure, provider symbols, JavaCPP APIs, the common SDZ cache API, AOT policy, and forbidden host/BLAS dependencies.

## One SDZ and the compile cache

`nd4j-sdx-model` owns logical SDZ identity, target/compiler/config cache keys, immutable object publication, embedded cache extraction, and single-SDZ packaging. The logical source digest ignores `META-INF/sdx-cache/`, so an original SDZ and an enriched SDZ keep the same identity.

`tools/sdx-compile.sh` is a thin launcher over that Java API. A target compiler is invoked only on a cache miss. It receives `--input`, `--target`, and `--output` directly, without a shell. Existing prepared artifacts can be imported during migration:

    tools/sdx-compile.sh compile \
      --source model.sdz \
      --target android-arm64-vulkan \
      --cache .sdx-cache \
      --compiler-id libnd4j-vulkan-aot \
      --compiler-version 1 \
      --prepared-artifact compiled-spirv \
      --tokenizer tokenizer.json \
      --llm-config generation.json

    tools/sdx-compile.sh compile \
      --source model.sdz \
      --target android-arm64-hexagon-htp \
      --cache .sdx-cache \
      --compiler-id qualcomm-hexagon-aot \
      --compiler-version 1 \
      --prepared-artifact compiled-kernels \
      --tokenizer tokenizer.json \
      --llm-config generation.json

    tools/sdx-compile.sh compile \
      --source model.sdz \
      --target android-arm64-google-tensor-g5 \
      --cache .sdx-cache \
      --compiler-id google-litertlm-aot \
      --compiler-version 0.14.0 \
      --prepared-artifact vendor-model-package \
      --quantization-config tools/mobile/profiles/int8-per-channel-example.json

    tools/sdx-compile.sh package \
      --source model.sdz \
      --cache .sdx-cache \
      --targets android-arm64-vulkan,android-arm64-hexagon-htp,android-arm64-google-tensor-g5 \
      --output model-mobile-aot.sdz

For an integrated compiler, replace `--prepared-artifact` with `--compiler-command`, `--compiler-fingerprint`, and any repeated `--compiler-arg`. Compiler identity, version, command material, auxiliary-file digests, and `--cache-key-option key=value` values all participate in the cache key.

Mobile resolution is extraction-only and fail-closed. It never runs a compiler or silently selects a CPU path. A missing target object means the SDZ is not valid for that APK flavor.

## Android Vulkan GPU

Build the BLAS-free device-only Vulkan SDX AAR with the NDK:

    tools/mobile/build-android-accelerator.sh vulkan

The AAR links Android's system Vulkan loader and packages libnd4jvulkan, libjnisdx, functional replay, and the Java/JavaCPP runtime. Production sessions require cached AOT SPIR-V, capture the lowered command sequence once, and replay it for later tokens. Unsupported or unrecordable operations fail closed.

## Google Tensor G3 NNAPI

Build and publish the canonical Pixel Tensor G3 provider from the checkout:

    tools/mobile/build-android-accelerator.sh

The default command builds and installs the tokenizer preset, generated tokenizer bindings, SDX preset, model compiler/cache API, native runtime, and JavaCPP bridge in dependency order. The promoted AAR includes `binding.json`, `provider.json`, the complete Java class contract, and the exact AArch64 native dependency closure. Its verifier cross-checks the provider ID, artifact format, target SoC, runtime library, execution policy, Java APIs, and ELF dependencies before writing the SHA-256 sidecar.

`--skip-native` is only for an incremental provider rebuild. It requires the native AAR bytes, variant, canonical path, and SHA-256 to match the atomic `.build-receipt` produced by a completed non-skipped build. A missing or stale receipt fails closed; rerun without `--skip-native` instead of selecting an intermediate AAR manually.

Tensor G3 segment placement is explicit and observable: EdgeTPU-capable NNAPI segments have first precedence, ACL/NEON handles supported CPU segments, and functional replay is the bounded final degradation path. There is no implicit whole-graph slot-by-slot fallback.

## Qualcomm Hexagon/HTP

Build the runtime contract:

    tools/mobile/build-android-accelerator.sh hexagon

For a device-ready artifact, inject the licensed adapter:

    HEXAGON_ADAPTER_LIBRARY=/path/to/libhexagon_mlir_runtime.so \
      tools/mobile/build-android-accelerator.sh hexagon --device-ready

Functional/emulated replay produces the vendor compile request. Finalized kernels use inclusive segment bounds, exact shape keys, cache/adapter ABI, SoC, byte size, and SHA-256:

    tools/mobile/hexagon-aot.py plan \
      --segments-json replay-segments.json --soc SM8650 \
      --model-id local-chat --output hexagon-request.json

    # Run the licensed vendor compiler for every request segment.

    tools/mobile/hexagon-aot.py finalize \
      --request hexagon-request.json --kernel-dir compiled-kernels

The resulting directory is supplied to the SDX compiler SPI or the temporary `--prepared-artifact` adapter above.

## Google Tensor G5 TPU

The Tensor G5 provider is built from pinned LiteRT-LM 0.14.0 and Google's direct Tensor dispatch library. It fixes the backend to `npu`, validates `Build.SOC_MODEL`, and exposes no CPU/GPU/NNAPI selector:

    tools/mobile/build-google-tensor-g5.sh

The script builds the direct-NPU runtime and JavaCPP bridge, packages the exact AArch64 dependency closure, embeds the common SDZ cache API, and runs `verify-google-tensor-g5-aar.sh`. It does not download or bundle a gated model.

The provider's native boundary still consumes a vendor object, but Android receives only the enriched SDZ. SDX validates and extracts the Tensor target into an app-owned immutable cache before constructing the LiteRT-LM session.

## INT8 contract

`validate-int8-quantization.py` currently validates the fail-closed research metadata used by the Tensor compiler. It requires symmetric INT8 per-channel weights with FLOAT32 scales, vendor AOT execution, and disabled float fallback. Fully INT8 activations additionally require representative calibration metadata and a dataset digest.

    tools/mobile/validate-int8-quantization.py \
      tools/mobile/profiles/int8-per-channel-example.json

Pass the validated file to `sdx-compile.sh compile --quantization-config ...`; its digest becomes part of the compile key and the metadata is stored inside the immutable target object. Moving this validator into the backend-neutral Java compiler API is the remaining Python cleanup item.

## Outputs

The default build root is `/tmp/sdx-android-build/accelerator/<variant>` (or `$SDX_ANDROID_BUILD_ROOT/accelerator/<variant>`). Its `dist` directory contains the complete JavaCPP AAR and SHA-256 sidecar:

- `sdx-runtime-android-arm64-vulkan.aar`
- `sdx-runtime-android-arm64-hexagon.aar`
- `sdx-runtime-android-arm64-tensor-g3.aar`
- `sdx-chat-runtime-android-arm64-google-tensor-g5.aar`

Models remain separate `.sdz` inputs; licensed vendor adapters remain external inputs.
