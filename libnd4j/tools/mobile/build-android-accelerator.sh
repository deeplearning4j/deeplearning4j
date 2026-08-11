#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  build-android-accelerator.sh --profile <profile.env> --android-ndk <path> [options]

Options:
  --output-root <path>  Build and distribution root
  --jobs <n>            Native/Cargo parallelism (default: 4)
  --offline             Forbid Cargo and Maven network access and disconnect CMake FetchContent
  --device-ready        Require and validate the profile's vendor adapter library
  --skip-tokenizers     Reuse already-built Rust/JavaCPP tokenizer artifacts
  --skip-native         Reuse an already-built native SDX Android AAR
  --skip-java           Reuse an already-built full JavaCPP SDX Android AAR
  -h, --help            Show this help

The profile selects an accelerator-only variant. CPU and BLAS-backed profiles are
rejected. Set MVN_CMD to choose a Maven executable. Qualcomm device-ready builds
also require HEXAGON_ADAPTER_LIBRARY to point at libhexagon_mlir_runtime.so.
Vulkan uses Android's system loader and requires no bundled vendor adapter.
Set SDX_NATIVE_OOM_MEMORY_THRESHOLD to override the serialized native-build
host-memory threshold (default: 90).
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIBND4J_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$LIBND4J_DIR/.." && pwd)"
TOKENIZER_ROOT="$REPO_ROOT/nd4j/nd4j-tokenizers"
TOKENIZER_BUILD="$TOKENIZER_ROOT/libtokenizers/build-mobile-tokenizers.sh"
TOKENIZER_PRESET_MODULE="$TOKENIZER_ROOT/tokenizers-native-preset"
TOKENIZER_MODULE="$TOKENIZER_ROOT/tokenizers-native"
SDX_MODULE="$REPO_ROOT/nd4j/nd4j-backends/nd4j-backend-impls/nd4j-sdx"
SDX_MODEL_MODULE="$REPO_ROOT/nd4j/nd4j-backends/nd4j-backend-impls/nd4j-sdx-model"
SDX_PRESET_MODULE="$REPO_ROOT/nd4j/nd4j-backends/nd4j-backend-impls/nd4j-sdx-preset"
VERIFY_SCRIPT="$SCRIPT_DIR/verify-android-accelerator-aar.sh"

PROFILE=""
ANDROID_NDK_ARG="${ANDROID_NDK:-${ANDROID_NDK_ROOT:-${ANDROID_NDK_HOME:-}}}"
OUTPUT_ROOT=""
JOBS="${JOBS:-4}"
OFFLINE=0
DEVICE_READY=0
SKIP_TOKENIZERS=0
SKIP_NATIVE=0
SKIP_JAVA=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile)
            PROFILE="${2:?missing value for --profile}"
            shift 2
            ;;
        --android-ndk)
            ANDROID_NDK_ARG="${2:?missing value for --android-ndk}"
            shift 2
            ;;
        --output-root)
            OUTPUT_ROOT="${2:?missing value for --output-root}"
            shift 2
            ;;
        --jobs)
            JOBS="${2:?missing value for --jobs}"
            shift 2
            ;;
        --offline)
            OFFLINE=1
            shift
            ;;
        --device-ready)
            DEVICE_READY=1
            shift
            ;;
        --skip-tokenizers)
            SKIP_TOKENIZERS=1
            shift
            ;;
        --skip-native)
            SKIP_NATIVE=1
            shift
            ;;
        --skip-java)
            SKIP_JAVA=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "$PROFILE" ]]; then
    echo "--profile is required" >&2
    exit 2
fi
if [[ ! -f "$PROFILE" ]]; then
    echo "Profile not found: $PROFILE" >&2
    exit 1
fi
if [[ -z "$ANDROID_NDK_ARG" || ! -f "$ANDROID_NDK_ARG/build/cmake/android.toolchain.cmake" ]]; then
    echo "--android-ndk must point at an installed Android NDK" >&2
    exit 1
fi
if [[ ! "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
    echo "--jobs must be a positive integer" >&2
    exit 2
fi

# Profiles are declarative inputs checked into tools/mobile/profiles.
# shellcheck disable=SC1090
source "$PROFILE"

SDX_GPU_TARGET="${SDX_GPU_TARGET:-AUTO}"

for required in SDX_VARIANT SDX_CHIP SDX_NATIVE_LIBRARY SDX_ACCELERATOR \
                SDX_ANDROID_API SDX_ANDROID_ABI SDX_DEVICE_ONLY \
                SDX_AOT_ONLY SDX_ALLOW_HOST_FALLBACK SDX_EXPECT_BLAS; do
    if [[ -z "${!required:-}" ]]; then
        echo "Profile is missing $required: $PROFILE" >&2
        exit 1
    fi
done

if [[ "$SDX_DEVICE_ONLY" != "1" || "$SDX_EXPECT_BLAS" != "0" ||
      "$SDX_ALLOW_HOST_FALLBACK" != "0" ]]; then
    echo "Accelerator profiles must be device-only, BLAS-free, and fail closed" >&2
    exit 1
fi
case "$SDX_CHIP" in
    vulkan|hexagon|google-tpu|nnapi) ;;
    *)
        echo "Unsupported Android device accelerator chip: $SDX_CHIP" >&2
        exit 1
        ;;
esac
if [[ "$SDX_ANDROID_ABI" != "arm64-v8a" ]]; then
    echo "Only Android arm64-v8a is supported by this mobile pipeline" >&2
    exit 1
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-$LIBND4J_DIR/build/mobile/$SDX_VARIANT}"
NATIVE_BUILD_DIR="$OUTPUT_ROOT/native"
DIST_DIR="$OUTPUT_ROOT/dist"
mkdir -p "$NATIVE_BUILD_DIR" "$DIST_DIR"

MVN_CMD="${MVN_CMD:-mvn}"
if ! command -v "$MVN_CMD" >/dev/null 2>&1; then
    echo "Maven executable not found: $MVN_CMD" >&2
    exit 1
fi
for command_name in cmake cargo rustup unzip zip sha256sum realpath mktemp; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "Required build command not found: $command_name" >&2
        exit 1
    fi
done

MAVEN_OFFLINE=()
TOKENIZER_OFFLINE=()
if [[ "$OFFLINE" == "1" ]]; then
    MAVEN_OFFLINE=(-o)
    TOKENIZER_OFFLINE=(--offline)
    export SDX_OFFLINE=ON
fi

ADAPTER_LIBRARY=""
if [[ "$SDX_VARIANT" == "hexagon" ]]; then
    ADAPTER_LIBRARY="${HEXAGON_ADAPTER_LIBRARY:-}"
    if [[ -n "$ADAPTER_LIBRARY" && ! -f "$ADAPTER_LIBRARY" ]]; then
        echo "HEXAGON_ADAPTER_LIBRARY does not exist: $ADAPTER_LIBRARY" >&2
        exit 1
    fi
    if [[ "$DEVICE_READY" == "1" && -z "$ADAPTER_LIBRARY" ]]; then
        echo "--device-ready requires HEXAGON_ADAPTER_LIBRARY" >&2
        exit 1
    fi
fi
if [[ -n "$ADAPTER_LIBRARY" ]]; then
    export SDX_EXTRA_RUNTIME_DEPENDENCY_FILES="$ADAPTER_LIBRARY"
    DEVICE_READY=1
else
    unset SDX_EXTRA_RUNTIME_DEPENDENCY_FILES || true
fi
if [[ "$SDX_VARIANT" == "vulkan" ]]; then
    DEVICE_READY=1
fi
if [[ "$SDX_VARIANT" == "tensor-g3" ]]; then
    export SD_NNAPI_REQUIRED_DEVICE_NAME="${SDX_REQUIRED_ACCELERATOR_DEVICE:-google-edgetpu}"
    # Tensor G3 uses the prevalidated Android/NDK ACL DSO. The generic
    # armcompute_install artifact is the Linux/libstdc++ package and cannot
    # satisfy an Android libc++ consumer. Keep the pipe escaped because
    # buildnativeoperations.sh evaluates CMAKE_ARGUMENTS as shell words.
    export CMAKE_ARGUMENTS="${CMAKE_ARGUMENTS:-} -DHELPERS_armcompute=ON -DSDX_NNAPI_TENSOR_G3_HYBRID=ON -DSDX_EXTRA_RUNTIME_DEPENDENCY_FILES:STRING=$NATIVE_BUILD_DIR/tensor_g3_armcompute_install/lib/armv8a-neon/libarm_compute.so"
fi

if [[ "$SKIP_TOKENIZERS" != "1" ]]; then
    "$TOKENIZER_BUILD" \
        --platform android-arm64 \
        --android-ndk "$ANDROID_NDK_ARG" \
        --android-api "$SDX_ANDROID_API" \
        --jobs "$JOBS" \
        "${TOKENIZER_OFFLINE[@]}"

    # The parser runs from tokenizers-native but loads its preset as a Maven
    # dependency. Install the preset from this checkout first so generated Java
    # cannot silently lag the header/Rust ABI behind a stale mavenLocal copy.
    "$MVN_CMD" "${MAVEN_OFFLINE[@]}" \
        -f "$TOKENIZER_PRESET_MODULE/pom.xml" \
        -DskipTests clean install

    "$MVN_CMD" "${MAVEN_OFFLINE[@]}" \
        -f "$TOKENIZER_MODULE/pom.xml" \
        -Pandroid-arm64 clean install \
        -DskipTests \
        -Dandroid.ndk="$ANDROID_NDK_ARG" \
        -Dandroid.api="$SDX_ANDROID_API"
fi

NATIVE_AAR="$NATIVE_BUILD_DIR/sdx-runtime-sdk/dist/sdx-runtime-android-arm64-$SDX_VARIANT.aar"
NATIVE_RECEIPT="$NATIVE_AAR.build-receipt"

if [[ "$SKIP_NATIVE" != "1" ]]; then
    # A failed or interrupted native rebuild must not leave an older receipt
    # authorizing whatever bytes happen to remain at the stable output path.
    rm -f -- "$NATIVE_RECEIPT"
    NATIVE_ARGS=(
        --platform android-arm64
        --android-abi "$SDX_ANDROID_ABI"
        --android-api "$SDX_ANDROID_API"
        --chip "$SDX_CHIP"
        --build-type release
        --output-path "$NATIVE_BUILD_DIR"
        --preprocess OFF
        # Mobile provider builds are deliberately serialized. The generic 80%
        # host-wide threshold can otherwise terminate a cache-hit-only build
        # because of unrelated workloads while tens of GiB remain available.
        --oom-memory-threshold "${SDX_NATIVE_OOM_MEMORY_THRESHOLD:-90}"
        -j "$JOBS"
    )
    if [[ "$SDX_VARIANT" == "tensor-g3" ]]; then
        NATIVE_ARGS+=(--helpers nnapi,armcompute)
    fi
    if [[ -n "${SDX_DATATYPES:-}" ]]; then
        NATIVE_ARGS+=(--datatypes "$SDX_DATATYPES")
    fi
    if [[ -n "${SDX_OPERATIONS:-}" ]]; then
        NATIVE_ARGS+=(--operations "$SDX_OPERATIONS")
    fi

    ANDROID_NDK="$ANDROID_NDK_ARG" \
    ANDROID_API="$SDX_ANDROID_API" \
    BUILD_WITH_JAVA=OFF \
    PREPROCESS=OFF \
        "$LIBND4J_DIR/buildnativeoperations.sh" "${NATIVE_ARGS[@]}"

    cmake --build "$NATIVE_BUILD_DIR" \
        --target sdx_runtime_bindings \
        --parallel "$JOBS"
fi

if [[ ! -s "$NATIVE_AAR" ]]; then
    echo "Native SDX AAR was not produced: $NATIVE_AAR" >&2
    exit 1
fi

NATIVE_AAR_REAL="$(realpath -e -- "$NATIVE_AAR")"
NATIVE_AAR_SHA256="$(sha256sum "$NATIVE_AAR_REAL" | cut -d ' ' -f 1)"
if [[ "$SKIP_NATIVE" != "1" ]]; then
    NATIVE_RECEIPT_TMP="$(mktemp "$NATIVE_RECEIPT.tmp.XXXXXX")"
    {
        printf 'format=1\n'
        printf 'variant=%s\n' "$SDX_VARIANT"
        printf 'artifact=%s\n' "$NATIVE_AAR_REAL"
        printf 'sha256=%s\n' "$NATIVE_AAR_SHA256"
    } >"$NATIVE_RECEIPT_TMP"
    mv -f -- "$NATIVE_RECEIPT_TMP" "$NATIVE_RECEIPT"
    echo "Native build receipt: $NATIVE_RECEIPT"
else
    if [[ ! -s "$NATIVE_RECEIPT" ]]; then
        echo "--skip-native requires a completed native build receipt: $NATIVE_RECEIPT" >&2
        echo "Rerun without --skip-native to rebuild and authorize the native AAR." >&2
        exit 1
    fi
    RECEIPT_FORMAT=""
    RECEIPT_VARIANT=""
    RECEIPT_ARTIFACT=""
    RECEIPT_SHA256=""
    while IFS='=' read -r receipt_key receipt_value; do
        case "$receipt_key" in
            format) RECEIPT_FORMAT="$receipt_value" ;;
            variant) RECEIPT_VARIANT="$receipt_value" ;;
            artifact) RECEIPT_ARTIFACT="$receipt_value" ;;
            sha256) RECEIPT_SHA256="$receipt_value" ;;
        esac
    done <"$NATIVE_RECEIPT"
    if [[ "$RECEIPT_FORMAT" != "1" ||
          "$RECEIPT_VARIANT" != "$SDX_VARIANT" ||
          "$RECEIPT_ARTIFACT" != "$NATIVE_AAR_REAL" ||
          "$RECEIPT_SHA256" != "$NATIVE_AAR_SHA256" ]]; then
        echo "Native build receipt does not authorize the current AAR: $NATIVE_RECEIPT" >&2
        echo "Rerun without --skip-native to rebuild and refresh the receipt." >&2
        exit 1
    fi
    echo "Verified native build receipt: $NATIVE_RECEIPT"
fi

if [[ "$SKIP_JAVA" != "1" ]]; then
    # JavaCPP-generated bindings name their preset classes directly. Build the
    # preset from this checkout and package it into the AAR so R8 and Android do
    # not depend on whatever happens to be present in Maven local.
    "$MVN_CMD" "${MAVEN_OFFLINE[@]}" \
        -f "$SDX_PRESET_MODULE/pom.xml" \
        -DskipTests clean install

    # Mainline the source-SDZ compile/cache API into every provider AAR. Clean
    # first so removed classes cannot linger in the stable target JAR.
    "$MVN_CMD" "${MAVEN_OFFLINE[@]}" \
        -f "$SDX_MODEL_MODULE/pom.xml" \
        -DskipTests clean install

    "$MVN_CMD" "${MAVEN_OFFLINE[@]}" \
        -f "$SDX_MODULE/pom.xml" \
        -Pandroid-arm64 clean install \
        -DskipTests \
        -Dandroid.ndk="$ANDROID_NDK_ARG" \
        -Dandroid.api="$SDX_ANDROID_API" \
        -Dlibnd4j.outputPath="$NATIVE_BUILD_DIR" \
        -Dsdx.android.variant="$SDX_VARIANT" \
        -Dsdx.native.library="$SDX_NATIVE_LIBRARY"
fi

shopt -s nullglob
AAR_CANDIDATES=("$SDX_MODULE/target"/sdx-runtime-*-android-arm64-"$SDX_VARIANT".aar)
shopt -u nullglob
if [[ ${#AAR_CANDIDATES[@]} -ne 1 || ! -s "${AAR_CANDIDATES[0]}" ]]; then
    echo "Expected one full JavaCPP AAR for $SDX_VARIANT, found ${#AAR_CANDIDATES[@]}" >&2
    exit 1
fi

FINAL_AAR="$DIST_DIR/sdx-runtime-android-arm64-$SDX_VARIANT.aar"
cp "${AAR_CANDIDATES[0]}" "$FINAL_AAR"

VERIFY_ARGS=(
    --aar "$FINAL_AAR"
    --variant "$SDX_VARIANT"
    --native-library "$SDX_NATIVE_LIBRARY"
    --accelerator "$SDX_ACCELERATOR"
    --gpu-target "$SDX_GPU_TARGET"
    --android-ndk "$ANDROID_NDK_ARG"
)
if [[ "$DEVICE_READY" == "1" ]]; then
    VERIFY_ARGS+=(--device-ready)
    if [[ -n "${SDX_ADAPTER_LIBRARY_NAME:-}" ]]; then
        VERIFY_ARGS+=(--adapter-name "$SDX_ADAPTER_LIBRARY_NAME")
    fi
fi
"$VERIFY_SCRIPT" "${VERIFY_ARGS[@]}"

sha256sum "$FINAL_AAR" > "$FINAL_AAR.sha256"
echo "Android accelerator AAR: $FINAL_AAR"
echo "SHA-256 manifest: $FINAL_AAR.sha256"
if [[ "$SDX_VARIANT" == "vulkan" ]]; then
    echo "Capability: device-ready Vulkan runtime; an AOT SPIR-V model and physical Vulkan GPU are required"
elif [[ "$DEVICE_READY" == "1" ]]; then
    echo "Capability: device-ready vendor adapter validated"
else
    echo "Capability: runtime contract only; inject the vendor adapter for device execution"
fi
