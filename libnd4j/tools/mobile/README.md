# Mobile accelerator builds

This directory owns reproducible Android ARM64 accelerator builds for the offline chat stack. Profiles are declarative and all machine-specific SDK, NDK, Maven, and vendor-library paths are caller inputs.

Core rules:

- Accelerator profiles are device-only. They set BLAS off and reject CPU, slot-by-slot, NNAPI, or host fallback.
- Vulkan and Qualcomm graph/replay execution stay in SDX/libnd4j. The Google Tensor G5 chat provider uses Google's supported LiteRT-LM direct-NPU C ABI; it does not masquerade generic NNAPI as TPU support.
- JavaCPP is transport only. Tensor execution, tokenization, conversation/KV state, sampling, and streaming remain in the selected native provider.
- Each hardware variant is a separate AAR with truthful metadata and a strict native dependency closure.
- Offline mode disables dependency downloads. Source, LFS objects, Maven artifacts, Cargo crates, npm/Bazel repositories, and vendor SDK files must already be cached.
- Verifiers check Android AArch64 ELF headers, provider ABI symbols, transitive dependencies, JavaCPP classes, AOT/device policy, and reject OpenBLAS, MKL, GFortran, glibc, and Linux loader leakage.

Android Vulkan GPU

Build the BLAS-free, device-only Vulkan SDX AAR with the NDK:

    tools/mobile/build-android-accelerator.sh \
      --profile tools/mobile/profiles/vulkan.env \
      --android-ndk /path/to/android-ndk-r28b-or-newer \
      --offline

The AAR links Android's system libvulkan loader and packages libnd4jvulkan, libjnisdx, and the JavaCPP runtime surface. It does not package an NNAPI or CPU execution path. Production sessions use SdxRuntime.ModelOptions.mobileVulkan(), require bundle-owned AOT SPIR-V, capture the lowered command sequence once, and replay it for subsequent tokens. Unsupported or unrecordable operations fail closed.

The same canonical SameDiff model is used by Vulkan and Hexagon. Package target-specific artifacts beside it:

    tools/sdx-compile.sh --input model.sdz --output model.dspb-dir \
      --packed-output model.dspb \
      --targets android-arm64 --backends VULKAN,HEXAGON,TPU \
      --gpu-target vulkan --vulkan-spirv-dir compiled-spirv \
      --hexagon-kernel-dir compiled-kernels \
      --tensor-g5-model model.litertlm --tokenizer tokenizer.json \
      --llm-config generation.json --overwrite

The unpacked directory is useful for inspection and desktop development. The optional packed output is a deterministic ZIP_STORED archive with fixed member ordering, timestamps, modes, and a SHA-256 sidecar; it is the file imported by the Android app.

Qualcomm Hexagon/HTP

Build the runtime-contract artifact (usable for integration and packaging tests, but not represented as hardware-ready without the vendor adapter):

    tools/mobile/build-android-accelerator.sh \
      --profile tools/mobile/profiles/hexagon.env \
      --android-ndk /path/to/android-ndk \
      --offline

For a device-ready artifact, provide the adapter implementing the hexmlir ABI used by HexagonRuntimeManager:

    HEXAGON_ADAPTER_LIBRARY=/path/to/libhexagon_mlir_runtime.so \
    tools/mobile/build-android-accelerator.sh \
      --profile tools/mobile/profiles/hexagon.env \
      --android-ndk /path/to/android-ndk \
      --device-ready --offline

The adapter is packaged as an explicit runtime dependency and is checked for device, memory, DMA, dispatch, completion, kernel-release, and load-or-compile symbols. Qualcomm SDK binaries are not committed here and must be supplied under their own license.

Hexagon production sessions are AOT-only: model bundles carry exact shape-keyed files named hexagon_<inclusive-start>_<inclusive-end>_<16-hex-shape-key>.bin with matching metadata. Runtime compilation is a development-only adapter capability. A missing artifact or unavailable device fails closed.

Use the functional/emulated replay summary to create the vendor compile request, then finalize the returned binaries before bundling them:

    tools/mobile/hexagon-aot.py plan \
      --segments-json replay-segments.json --soc SM8650 \
      --model-id local-chat --output hexagon-request.json

    # Run the licensed vendor compiler for every request segment.

    tools/mobile/hexagon-aot.py finalize \
      --request hexagon-request.json --kernel-dir compiled-kernels

    tools/sdx-compile.sh --input model.sdz --output model.dspb-dir \
      --hexagon-kernel-dir compiled-kernels --tokenizer tokenizer.json \
      --llm-config generation.json --overwrite

The request and metadata declare inclusive segment bounds, cache/adapter ABI, SoC, exact shape key, byte size, and SHA-256. The finalizer rejects missing, extra, empty, stale, or tampered kernels and emits a deterministic artifact manifest.

Google Tensor G5 TPU

The direct TPU package is an optional SDX chat provider built from the pinned LiteRT-LM 0.14.0 public C API and Google's Tensor dispatch library. Current public support is AOT-only and specific to Tensor G5. The Java facade fixes the backend to `npu`, validates `Build.SOC_MODEL`, requires a `.litertlm` model, and exposes no CPU/GPU/NNAPI backend selector.

    tools/mobile/build-google-tensor-g5.sh \
      --android-ndk /path/to/android-ndk-r28b-or-newer \
      --maven /path/to/mvn

The script pins the LiteRT-LM tag and commit, Bazel/Bazelisk versions, and a SHA-256-verified user-space Git LFS bootstrap. It builds `liblitert-lm.so` plus `libLiteRtDispatch_GoogleTensor.so`, compiles the JavaCPP bridge, packages the exact dependency closure, and runs `verify-google-tensor-g5-aar.sh`. No gated model is downloaded or placed in the AAR.

For a canonical multi-target model, pass the Tensor G5 AOT derivative to `sdx-compile.sh --tensor-g5-model model.litertlm`. The bundler stores it at a fixed checksummed path inside the same `.dspb` that carries the canonical SameDiff graph and the Vulkan/Hexagon artifacts. The Tensor app also accepts a raw `.litertlm` for official vendor packages.

The primary acceptance model is the official Gemma 3 1B instruction-tuned Tensor G5 package with 8-bit per-channel quantization. See `models.yaml` for the official Google/Qualcomm baselines, smaller research candidates, embedding candidates for graph search, and parity thresholds. Each research chat candidate lists its exact app flavor and output extension. Qualcomm `.litertlm` packages are marked vendor baselines only; the Hexagon app accepts SDX `.dspb` bundles. Research entries are not claimed device-ready until vendor AOT compilation and on-device parity pass.

INT8 bundle contract

`validate-int8-quantization.py` defines the shared, fail-closed metadata contract for SDX graph bundles and LiteRT-LM packages. It requires symmetric INT8 per-channel weights with FLOAT32 scales, device-only vendor AOT execution, and explicitly disables float fallback. FLOAT16 activations are valid for the official Tensor G5 weight-quantized package; fully INT8 activations additionally require a calibration method, at least 32 representative samples, and a dataset SHA-256.

    tools/mobile/validate-int8-quantization.py \
      tools/mobile/profiles/int8-per-channel-example.json

    tools/sdx-compile.sh \
      --input model.sdz \
      --output model.dspb-dir \
      --quantization-config tools/mobile/profiles/int8-per-channel-example.json \
      --overwrite

The bundler validates the contract before copying it to `metadata/` and records its path in `manifest.json`. Invalid per-tensor, asymmetric, fallback-enabled, uncalibrated activation-INT8, or non-AOT configurations stop packaging.

Outputs

The default output root is `libnd4j/build/mobile/<variant>`. Its `dist` directory contains the full JavaCPP AAR and a SHA-256 sidecar. The three primary deliverables are `sdx-runtime-android-arm64-vulkan.aar`, `sdx-runtime-android-arm64-hexagon.aar`, and `sdx-chat-runtime-android-arm64-google-tensor-g5.aar`; models and licensed vendor adapters remain separate inputs.
