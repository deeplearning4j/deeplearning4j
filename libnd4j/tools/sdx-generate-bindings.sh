#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Generate SDX runtime bindings for a specific platform/backend.

Usage:
  sdx-generate-bindings.sh --platform <id> [--backend <cpu|cuda|amd|vulkan>] [options]

Platforms:
  linux-x86_64
  linux-arm64
  windows-x86_64
  macos-x86_64
  macos-arm64
  android-arm64
  android-x86_64
  ios-arm64
  ios-x86_64

Options:
  --platform <id>          Target platform ID (required)
  --backend <id>           cpu (default), cuda, amd, vulkan
  --build-dir <path>       Build directory (default: libnd4j/blasbuild/<platform>-<backend>)
  --generator <name>       CMake generator (default: Ninja if available, otherwise Unix Makefiles)
  --android-ndk <path>     Android NDK root (required for android-* targets)
  --android-api <level>    Android API level (default: 24; minimum for Vulkan)
  --cmake <path>           CMake executable (default: cmake)
  --jobs <n>               Build parallelism (default: nproc/sysctl)
  --ops <csv>              Compile only these libnd4j operations (mobile size optimization)
  --dtypes <csv>           Compile only these data types (for example float,int64)
  --build-tokenizer        Cross-build and package the Rust tokenizer for mobile targets
  --tokenizer-dir <path>   Package prebuilt tokenizer include/ and lib/ artifacts
  --no-tokenizer           Do not build or package the tokenizer (mobile defaults to build)
  --tokenizer-offline      Pass --offline to the mobile tokenizer Cargo build
  --extra-cmake "<flags>"  Additional CMake flags
  --no-standalone          Package the monolithic backend library instead of the
                           JVM-free standalone runtime (libsdx_cpu/libsdx_cuda).
                           Standalone is the default for SDK packaging.
  -h, --help               Show this help

Examples:
  ./libnd4j/tools/sdx-generate-bindings.sh --platform linux-x86_64 --backend cpu
  ./libnd4j/tools/sdx-generate-bindings.sh --platform linux-x86_64 --backend cuda
  ./libnd4j/tools/sdx-generate-bindings.sh --platform linux-x86_64 --backend amd
  ./libnd4j/tools/sdx-generate-bindings.sh --platform android-arm64 --backend vulkan --android-ndk "$ANDROID_NDK"
  ./libnd4j/tools/sdx-generate-bindings.sh --platform ios-arm64
EOF
}

PLATFORM=""
BACKEND="cpu"
BUILD_DIR=""
GENERATOR=""
ANDROID_NDK=""
ANDROID_API="24"
CMAKE_BIN="cmake"
JOBS=""
EXTRA_CMAKE=""
OPS_LIST=""
DTYPES_LIST=""
TOKENIZER_MODE="auto"
TOKENIZER_DIR=""
TOKENIZER_OFFLINE="OFF"
STANDALONE="ON"
STANDALONE_EXPLICIT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --platform)
      PLATFORM="${2:-}"
      shift 2
      ;;
    --backend)
      BACKEND="${2:-}"
      shift 2
      ;;
    --build-dir)
      BUILD_DIR="${2:-}"
      shift 2
      ;;
    --generator)
      GENERATOR="${2:-}"
      shift 2
      ;;
    --android-ndk)
      ANDROID_NDK="${2:-}"
      shift 2
      ;;
    --android-api)
      ANDROID_API="${2:-}"
      shift 2
      ;;
    --cmake)
      CMAKE_BIN="${2:-}"
      shift 2
      ;;
    --jobs)
      JOBS="${2:-}"
      shift 2
      ;;
    --ops)
      OPS_LIST="${2:-}"
      shift 2
      ;;
    --dtypes)
      DTYPES_LIST="${2:-}"
      shift 2
      ;;
    --build-tokenizer)
      TOKENIZER_MODE="build"
      shift
      ;;
    --tokenizer-dir)
      TOKENIZER_MODE="prebuilt"
      TOKENIZER_DIR="${2:-}"
      shift 2
      ;;
    --no-tokenizer)
      TOKENIZER_MODE="off"
      shift
      ;;
    --tokenizer-offline)
      TOKENIZER_OFFLINE="ON"
      shift
      ;;
    --extra-cmake)
      EXTRA_CMAKE="${2:-}"
      shift 2
      ;;
    --no-standalone)
      STANDALONE="OFF"
      STANDALONE_EXPLICIT="OFF"
      shift
      ;;
    --standalone)
      STANDALONE="ON"
      STANDALONE_EXPLICIT="ON"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
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

case "${BACKEND}" in
  cpu|cuda|amd|vulkan)
    ;;
  *)
    echo "Invalid --backend value: ${BACKEND}" >&2
    exit 2
    ;;
esac

if [[ "${BACKEND}" == "vulkan" ]]; then
  if [[ "${STANDALONE_EXPLICIT}" == "ON" ]]; then
    echo "The Vulkan backend is packaged from the monolithic runtime; --standalone is not supported" >&2
    exit 2
  fi
  if [[ "${PLATFORM}" == ios-* ]]; then
    echo "Vulkan is not an iOS backend; use the Metal runtime package for iOS" >&2
    exit 2
  fi
  if [[ "${PLATFORM}" == android-* ]]; then
    if ! [[ "${ANDROID_API}" =~ ^[0-9]+$ ]]; then
      echo "--android-api must be an integer" >&2
      exit 2
    fi
    if (( ANDROID_API < 24 )); then
      echo "Android Vulkan requires --android-api 24 or newer" >&2
      exit 2
    fi
  fi
  STANDALONE="OFF"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIBND4J_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -z "${BUILD_DIR}" ]]; then
  BUILD_DIR="${LIBND4J_DIR}/blasbuild/${PLATFORM}-${BACKEND}"
fi

if [[ -z "${GENERATOR}" ]]; then
  if command -v ninja >/dev/null 2>&1; then
    GENERATOR="Ninja"
  else
    GENERATOR="Unix Makefiles"
  fi
fi

if [[ -z "${JOBS}" ]]; then
  if command -v nproc >/dev/null 2>&1; then
    JOBS="$(nproc)"
  elif command -v sysctl >/dev/null 2>&1; then
    JOBS="$(sysctl -n hw.ncpu)"
  else
    JOBS="4"
  fi
fi

COMMON_FLAGS=(
  -DSD_BUILD_SDX_STANDALONE="${STANDALONE}"
  -DSD_BUILD_WITH_JAVA=OFF
)

if [[ -n "${OPS_LIST}" ]]; then
  OPS_LIST="${OPS_LIST//,/;}"
  COMMON_FLAGS+=(
    -DSD_ALL_OPS=OFF
    "-DSD_OPS_LIST=${OPS_LIST}"
  )
else
  COMMON_FLAGS+=(-DSD_ALL_OPS=ON)
fi

if [[ -n "${DTYPES_LIST}" ]]; then
  DTYPES_LIST="${DTYPES_LIST//,/;}"
  COMMON_FLAGS+=("-DSD_TYPES_LIST=${DTYPES_LIST}")
fi

case "${BACKEND}" in
  cpu)
    COMMON_FLAGS+=(-DSD_CPU=ON -DSD_CUDA=OFF -DSD_VULKAN=OFF)
    ;;
  cuda)
    COMMON_FLAGS+=(-DSD_CPU=ON -DSD_CUDA=ON -DSD_VULKAN=OFF)
    ;;
  amd)
    COMMON_FLAGS+=(-DSD_CPU=ON -DSD_CUDA=ON -DSD_VULKAN=OFF -DSD_ZLUDA=ON -DSD_ZLUDA_TARGET=AMD)
    ;;
  vulkan)
    COMMON_FLAGS+=(
      -DSD_CPU=OFF
      -DSD_CUDA=OFF
      -DSD_VULKAN=ON
      -DSD_TRITON=OFF
      -DHELPERS_mlir=OFF
      -DMLIR_ENABLE_VULKAN=OFF
    )
    ;;
esac

TOOLCHAIN_FLAGS=()
case "${PLATFORM}" in
  linux-x86_64|linux-arm64|windows-x86_64|macos-x86_64|macos-arm64)
    ;;
  android-arm64)
    if [[ -z "${ANDROID_NDK}" ]]; then
      ANDROID_NDK="${ANDROID_NDK_ROOT:-${ANDROID_NDK_HOME:-${ANDROID_NDK:-}}}"
    fi
    if [[ -z "${ANDROID_NDK}" ]]; then
      echo "android-arm64 target requires --android-ndk (or ANDROID_NDK_ROOT/ANDROID_NDK_HOME)" >&2
      exit 2
    fi
    ANDROID_TOOLCHAIN="${ANDROID_NDK}/build/cmake/android.toolchain.cmake"
    if [[ ! -f "${ANDROID_TOOLCHAIN}" ]]; then
      echo "Android NDK toolchain not found: ${ANDROID_TOOLCHAIN}" >&2
      exit 2
    fi
    TOOLCHAIN_FLAGS+=(
      -DCMAKE_TOOLCHAIN_FILE="${ANDROID_TOOLCHAIN}"
      -DANDROID_NDK="${ANDROID_NDK}"
      -DANDROID_ABI=arm64-v8a
      -DANDROID_PLATFORM="android-${ANDROID_API}"
      -DANDROID_NATIVE_API_LEVEL="${ANDROID_API}"
      -DANDROID_STL=c++_static
      -DSD_ANDROID_BUILD=ON
    )
    ;;
  android-x86_64)
    if [[ -z "${ANDROID_NDK}" ]]; then
      ANDROID_NDK="${ANDROID_NDK_ROOT:-${ANDROID_NDK_HOME:-${ANDROID_NDK:-}}}"
    fi
    if [[ -z "${ANDROID_NDK}" ]]; then
      echo "android-x86_64 target requires --android-ndk (or ANDROID_NDK_ROOT/ANDROID_NDK_HOME)" >&2
      exit 2
    fi
    ANDROID_TOOLCHAIN="${ANDROID_NDK}/build/cmake/android.toolchain.cmake"
    if [[ ! -f "${ANDROID_TOOLCHAIN}" ]]; then
      echo "Android NDK toolchain not found: ${ANDROID_TOOLCHAIN}" >&2
      exit 2
    fi
    TOOLCHAIN_FLAGS+=(
      -DCMAKE_TOOLCHAIN_FILE="${ANDROID_TOOLCHAIN}"
      -DANDROID_NDK="${ANDROID_NDK}"
      -DANDROID_ABI=x86_64
      -DANDROID_PLATFORM="android-${ANDROID_API}"
      -DANDROID_NATIVE_API_LEVEL="${ANDROID_API}"
      -DANDROID_STL=c++_static
      -DSD_ANDROID_BUILD=ON
    )
    ;;
  ios-arm64)
    TOOLCHAIN_FLAGS+=(-DCMAKE_TOOLCHAIN_FILE="${LIBND4J_DIR}/cmake/ios-arm64.cmake")
    ;;
  ios-x86_64)
    TOOLCHAIN_FLAGS+=(-DCMAKE_TOOLCHAIN_FILE="${LIBND4J_DIR}/cmake/ios-x86_64.cmake")
    ;;
  *)
    echo "Unsupported --platform: ${PLATFORM}" >&2
    exit 2
    ;;
esac

if [[ "${TOKENIZER_MODE}" == "auto" ]]; then
  case "${PLATFORM}" in
    android-arm64|ios-arm64)
      TOKENIZER_MODE="build"
      ;;
    *)
      TOKENIZER_MODE="off"
      ;;
  esac
fi

if [[ "${TOKENIZER_MODE}" == "build" ]]; then
  TOKENIZER_ROOT="${LIBND4J_DIR}/../nd4j/nd4j-tokenizers/libtokenizers"
  TOKENIZER_BUILD_SCRIPT="${TOKENIZER_ROOT}/build-mobile-tokenizers.sh"
  if [[ ! -x "${TOKENIZER_BUILD_SCRIPT}" ]]; then
    echo "Mobile tokenizer build script not found or not executable: ${TOKENIZER_BUILD_SCRIPT}" >&2
    exit 2
  fi
  if [[ -z "${TOKENIZER_DIR}" ]]; then
    TOKENIZER_DIR="${BUILD_DIR}/sdx-tokenizer"
  fi

  TOKENIZER_BUILD_ARGS=(
    --platform "${PLATFORM}"
    --jobs "${JOBS}"
    --output-dir "${TOKENIZER_DIR}"
  )
  if [[ "${PLATFORM}" == android-* ]]; then
    TOKENIZER_BUILD_ARGS+=(
      --android-ndk "${ANDROID_NDK}"
      --android-api "${ANDROID_API}"
    )
  fi
  if [[ "${TOKENIZER_OFFLINE}" == "ON" ]]; then
    TOKENIZER_BUILD_ARGS+=(--offline)
  fi

  echo "Building portable mobile tokenizer"
  "${TOKENIZER_BUILD_SCRIPT}" "${TOKENIZER_BUILD_ARGS[@]}"
elif [[ "${TOKENIZER_MODE}" == "prebuilt" && -z "${TOKENIZER_DIR}" ]]; then
  echo "--tokenizer-dir requires a non-empty path" >&2
  exit 2
fi

if [[ "${TOKENIZER_MODE}" != "off" ]]; then
  TOKENIZER_HEADER="${TOKENIZER_DIR}/include/tokenizers_ffi.h"
  case "${PLATFORM}" in
    android-*)
      TOKENIZER_LIBRARY="${TOKENIZER_DIR}/lib/libtokenizers_ffi.so"
      ;;
    ios-*)
      TOKENIZER_LIBRARY="${TOKENIZER_DIR}/lib/libtokenizers_ffi.a"
      ;;
    *)
      echo "Tokenizer packaging is only supported for Android and iOS targets" >&2
      exit 2
      ;;
  esac
  if [[ ! -f "${TOKENIZER_HEADER}" || ! -f "${TOKENIZER_LIBRARY}" ]]; then
    echo "Tokenizer artifacts are incomplete under ${TOKENIZER_DIR}" >&2
    echo "Expected: ${TOKENIZER_HEADER}" >&2
    echo "Expected: ${TOKENIZER_LIBRARY}" >&2
    exit 2
  fi
  COMMON_FLAGS+=(
    "-DSDX_TOKENIZER_HEADER_FILE=${TOKENIZER_HEADER}"
    "-DSDX_TOKENIZER_LIBRARY_FILE=${TOKENIZER_LIBRARY}"
  )
fi

mkdir -p "${BUILD_DIR}"

CONFIGURE_CMD=(
  "${CMAKE_BIN}"
  -S "${LIBND4J_DIR}"
  -B "${BUILD_DIR}"
  -G "${GENERATOR}"
  "${COMMON_FLAGS[@]}"
  "${TOOLCHAIN_FLAGS[@]}"
)

if [[ -n "${EXTRA_CMAKE}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_CMAKE_ARR=(${EXTRA_CMAKE})
  CONFIGURE_CMD+=("${EXTRA_CMAKE_ARR[@]}")
fi

echo "Configuring build in: ${BUILD_DIR}"
printf '  %q' "${CONFIGURE_CMD[@]}"
echo
"${CONFIGURE_CMD[@]}"

BUILD_CMD=(
  "${CMAKE_BIN}"
  --build "${BUILD_DIR}"
  --target sdx_runtime_bindings
  --parallel "${JOBS}"
)

echo "Building runtime bindings"
printf '  %q' "${BUILD_CMD[@]}"
echo
"${BUILD_CMD[@]}"

echo
echo "SDX runtime bindings generated:"
echo "  ${BUILD_DIR}/sdx-runtime-sdk/dist"
echo "Language wrappers are staged under:"
echo "  ${BUILD_DIR}/sdx-runtime-sdk/bindings/<platform>/<variant>/wrappers"
