#!/usr/bin/env bash
#
# Cross-build the Rust tokenizer C ABI for the mobile SDX runtime.
#
# Android emits both libtokenizers_ffi.so (AAR/JNI packaging) and
# libtokenizers_ffi.a (single-library/static integration). iOS emits the
# static archive used by the SDX XCFramework build.

set -euo pipefail

usage() {
    echo "Usage: $0 --platform android-arm64|ios-arm64 [options]"
    echo
    echo "Options:"
    echo "  --android-ndk PATH   Android NDK root (required for Android)"
    echo "  --android-api N      Android minimum API (default: 24)"
    echo "  --ios-min VERSION    iOS deployment target (default: 15.0)"
    echo "  --output-dir PATH    Staging directory"
    echo "  --build-dir PATH     Cargo target directory"
    echo "  --jobs N             Cargo parallelism (default: 4)"
    echo "  --offline            Forbid Cargo network access"
    echo "  -h, --help           Show this help"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
# shellcheck source=../../../build-scripts/android-compiler-cache.sh
source "${REPO_ROOT}/build-scripts/android-compiler-cache.sh"
FFI_DIR="${SCRIPT_DIR}/tokenizers-ffi"
PLATFORM=""
ANDROID_NDK="${ANDROID_NDK:-}"
ANDROID_API="${ANDROID_API:-24}"
IOS_MIN="${IOS_MIN:-15.0}"
OUTPUT_DIR=""
BUILD_DIR=""
JOBS="${JOBS:-4}"
OFFLINE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --platform)
            PLATFORM="${2:?missing value for --platform}"
            shift 2
            ;;
        --android-ndk)
            ANDROID_NDK="${2:?missing value for --android-ndk}"
            shift 2
            ;;
        --android-api)
            ANDROID_API="${2:?missing value for --android-api}"
            shift 2
            ;;
        --ios-min)
            IOS_MIN="${2:?missing value for --ios-min}"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="${2:?missing value for --output-dir}"
            shift 2
            ;;
        --build-dir)
            BUILD_DIR="${2:?missing value for --build-dir}"
            shift 2
            ;;
        --jobs)
            JOBS="${2:?missing value for --jobs}"
            shift 2
            ;;
        --offline)
            OFFLINE=true
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

if [[ -z "${PLATFORM}" ]]; then
    echo "--platform is required" >&2
    usage >&2
    exit 2
fi

case "${PLATFORM}" in
    android-arm64)
        RUST_TARGET="aarch64-linux-android"
        DEFAULT_OUTPUT="${SCRIPT_DIR}/target/mobile/android-arm64-v8a"
        ;;
    ios-arm64)
        RUST_TARGET="aarch64-apple-ios"
        DEFAULT_OUTPUT="${SCRIPT_DIR}/target/mobile/ios-arm64"
        ;;
    *)
        echo "Unsupported mobile platform: ${PLATFORM}" >&2
        exit 2
        ;;
esac

OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUTPUT}}"
if [[ -z "${BUILD_DIR}" ]]; then
    if [[ "${PLATFORM}" == "android-arm64" ]]; then
        # CMake and Cargo both cache the selected linker/toolchain. Isolate those
        # caches by NDK and API so switching r27/r28 can never silently reuse a
        # wrapper configured against another NDK.
        NDK_CACHE_ID="$(basename "${ANDROID_NDK}")"
        BUILD_DIR="${SCRIPT_DIR}/build/mobile-tokenizers/${PLATFORM}-ndk${NDK_CACHE_ID}-api${ANDROID_API}"
    else
        BUILD_DIR="${SCRIPT_DIR}/build/mobile-tokenizers/${PLATFORM}-ios${IOS_MIN}"
    fi
fi

command -v cargo >/dev/null 2>&1 || {
    echo "cargo is required" >&2
    exit 1
}
command -v rustup >/dev/null 2>&1 || {
    echo "rustup is required to validate the cross target" >&2
    exit 1
}
if ! rustup target list --installed | grep -qx "${RUST_TARGET}"; then
    echo "Rust target ${RUST_TARGET} is not installed." >&2
    echo "Install it before entering an offline build: rustup target add ${RUST_TARGET}" >&2
    exit 1
fi

if [[ "${PLATFORM}" == "android-arm64" ]]; then
    if [[ -z "${ANDROID_NDK}" || ! -d "${ANDROID_NDK}" ]]; then
        echo "--android-ndk must point to an installed Android NDK" >&2
        exit 1
    fi

    case "$(uname -s)" in
        Linux) NDK_HOST_TAG="linux-x86_64" ;;
        Darwin) NDK_HOST_TAG="darwin-x86_64" ;;
        *)
            echo "Unsupported Android build host: $(uname -s)" >&2
            exit 1
            ;;
    esac

    NDK_TOOLCHAIN="${ANDROID_NDK}/toolchains/llvm/prebuilt/${NDK_HOST_TAG}"
    ANDROID_CC="${NDK_TOOLCHAIN}/bin/aarch64-linux-android${ANDROID_API}-clang"
    ANDROID_CXX="${NDK_TOOLCHAIN}/bin/aarch64-linux-android${ANDROID_API}-clang++"
    ANDROID_AR="${NDK_TOOLCHAIN}/bin/llvm-ar"
    if [[ ! -x "${ANDROID_CC}" || ! -x "${ANDROID_CXX}" || ! -x "${ANDROID_AR}" ]]; then
        echo "Android ARM64 toolchain is incomplete under ${NDK_TOOLCHAIN}" >&2
        exit 1
    fi

    dl4j_enable_android_compiler_cache_environment "Android tokenizer builds"
    export DL4J_COMPILER_CACHE
    CACHE_WRAPPER_DIR="${BUILD_DIR}/compiler-cache"
    mkdir -p "${CACHE_WRAPPER_DIR}"
    ANDROID_CACHED_CC="${CACHE_WRAPPER_DIR}/aarch64-linux-android-clang"
    ANDROID_CACHED_CXX="${CACHE_WRAPPER_DIR}/aarch64-linux-android-clang++"
    dl4j_write_android_compiler_cache_wrapper "${ANDROID_CACHED_CC}" "${ANDROID_CC}"
    dl4j_write_android_compiler_cache_wrapper "${ANDROID_CACHED_CXX}" "${ANDROID_CXX}"
    export DL4J_ANDROID_COMPILER="${ANDROID_CACHED_CXX}"
    export DL4J_ANDROID_REAL_COMPILER="${ANDROID_CXX}"
    if [[ -z "${RUSTC_WRAPPER:-}" &&
          "$(basename "${DL4J_COMPILER_CACHE}")" == sccache* ]]; then
        export RUSTC_WRAPPER="${DL4J_COMPILER_CACHE}"
    fi

    export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="${ANDROID_CC}"
    export CARGO_TARGET_AARCH64_LINUX_ANDROID_AR="${ANDROID_AR}"
    export CARGO_TARGET_AARCH64_LINUX_ANDROID_RUSTFLAGS="${CARGO_TARGET_AARCH64_LINUX_ANDROID_RUSTFLAGS:-} -C link-arg=-Wl,-soname,libtokenizers_ffi.so"
    export CC_aarch64_linux_android="${ANDROID_CACHED_CC}"
    export CXX_aarch64_linux_android="${ANDROID_CACHED_CXX}"
    export AR_aarch64_linux_android="${ANDROID_AR}"
else
    if [[ "$(uname -s)" != "Darwin" ]]; then
        echo "ios-arm64 builds require macOS with Xcode's iPhoneOS SDK" >&2
        exit 1
    fi
    command -v xcrun >/dev/null 2>&1 || {
        echo "xcrun is required for ios-arm64" >&2
        exit 1
    }
    export SDKROOT
    SDKROOT="$(xcrun --sdk iphoneos --show-sdk-path)"
    export IPHONEOS_DEPLOYMENT_TARGET="${IOS_MIN}"
    IOS_CLANG="$(xcrun --sdk iphoneos --find clang)"
    export CARGO_TARGET_AARCH64_APPLE_IOS_LINKER="${IOS_CLANG}"
    export CARGO_TARGET_AARCH64_APPLE_IOS_RUSTFLAGS="-C link-arg=-miphoneos-version-min=${IOS_MIN}"
fi

CARGO_ARGS=(
    build
    --release
    --locked
    --manifest-path "${FFI_DIR}/Cargo.toml"
    --target "${RUST_TARGET}"
    --no-default-features
    --features portable-regex
    --jobs "${JOBS}"
)
if [[ "${OFFLINE}" == "true" ]]; then
    CARGO_ARGS+=(--offline)
fi

echo "Building tokenizers-ffi for ${PLATFORM} (${RUST_TARGET})"
CARGO_TARGET_DIR="${BUILD_DIR}" cargo "${CARGO_ARGS[@]}"

ARTIFACT_DIR="${BUILD_DIR}/${RUST_TARGET}/release"
mkdir -p "${OUTPUT_DIR}/include" "${OUTPUT_DIR}/lib"
cp "${SCRIPT_DIR}/include/tokenizers_ffi.h" "${OUTPUT_DIR}/include/"
cp "${SCRIPT_DIR}/include/tokenizers_c.h" "${OUTPUT_DIR}/include/"
cp "${SCRIPT_DIR}/include/tokenizer_wrapper.h" "${OUTPUT_DIR}/include/"
cp "${SCRIPT_DIR}/include/model_manager.h" "${OUTPUT_DIR}/include/"

STATIC_LIB="${ARTIFACT_DIR}/libtokenizers_ffi.a"
if [[ ! -s "${STATIC_LIB}" ]]; then
    echo "Missing static tokenizer archive: ${STATIC_LIB}" >&2
    exit 1
fi
cp "${STATIC_LIB}" "${OUTPUT_DIR}/lib/"

if [[ "${PLATFORM}" == "android-arm64" ]]; then
    SHARED_LIB="${ARTIFACT_DIR}/libtokenizers_ffi.so"
    if [[ ! -s "${SHARED_LIB}" ]]; then
        echo "Missing Android tokenizer shared library: ${SHARED_LIB}" >&2
        exit 1
    fi
    cp "${SHARED_LIB}" "${OUTPUT_DIR}/lib/"

    # Build the existing stable C wrapper against the same Rust archive. This is
    # the library consumed by TokenizersNative's JavaCPP-generated JNI shim.
    WRAPPER_BUILD_DIR="${BUILD_DIR}/wrapper"
    cmake -S "${SCRIPT_DIR}" -B "${WRAPPER_BUILD_DIR}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" \
        -DANDROID_NDK="${ANDROID_NDK}" \
        -DANDROID_ABI=arm64-v8a \
        -DANDROID_PLATFORM="android-${ANDROID_API}" \
        -DANDROID_NATIVE_API_LEVEL="${ANDROID_API}" \
        -DANDROID_STL=c++_static \
        -DCMAKE_C_COMPILER_LAUNCHER:FILEPATH="${DL4J_COMPILER_CACHE}" \
        -DCMAKE_CXX_COMPILER_LAUNCHER:FILEPATH="${DL4J_COMPILER_CACHE}" \
        -DPLATFORM=android \
        -DARCH=arm64 \
        -DJAVACPP_PLATFORM=android-arm64 \
        -DTOKENIZERS_FFI_LIB="${STATIC_LIB}"
    cmake --build "${WRAPPER_BUILD_DIR}" --target tokenizers_wrapper --parallel "${JOBS}"

    WRAPPER_LIB="${WRAPPER_BUILD_DIR}/libtokenizers_wrapper.so"
    if [[ ! -s "${WRAPPER_LIB}" ]]; then
        echo "Missing Android tokenizer wrapper: ${WRAPPER_LIB}" >&2
        exit 1
    fi
    cp -L "${WRAPPER_LIB}" "${OUTPUT_DIR}/lib/libtokenizers_wrapper.so"
fi

echo "Mobile tokenizer artifacts staged at ${OUTPUT_DIR}"
