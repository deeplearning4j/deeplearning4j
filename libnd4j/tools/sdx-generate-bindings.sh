#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Generate SDX runtime bindings for a specific platform/backend.

Usage:
  sdx-generate-bindings.sh --platform <id> [--backend <cpu|cuda|amd>] [options]

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
  --backend <id>           cpu (default), cuda, amd
  --build-dir <path>       Build directory (default: libnd4j/blasbuild/<platform>-<backend>)
  --generator <name>       CMake generator (default: Ninja if available, otherwise Unix Makefiles)
  --android-ndk <path>     Android NDK root (required for android-* targets)
  --android-api <level>    Android API level (default: 27)
  --cmake <path>           CMake executable (default: cmake)
  --jobs <n>               Build parallelism (default: nproc/sysctl)
  --extra-cmake "<flags>"  Additional CMake flags
  -h, --help               Show this help

Examples:
  ./libnd4j/tools/sdx-generate-bindings.sh --platform linux-x86_64 --backend cpu
  ./libnd4j/tools/sdx-generate-bindings.sh --platform linux-x86_64 --backend cuda
  ./libnd4j/tools/sdx-generate-bindings.sh --platform linux-x86_64 --backend amd
  ./libnd4j/tools/sdx-generate-bindings.sh --platform android-arm64 --android-ndk "$ANDROID_NDK"
  ./libnd4j/tools/sdx-generate-bindings.sh --platform ios-arm64
EOF
}

PLATFORM=""
BACKEND="cpu"
BUILD_DIR=""
GENERATOR=""
ANDROID_NDK=""
ANDROID_API="27"
CMAKE_BIN="cmake"
JOBS=""
EXTRA_CMAKE=""

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
    --extra-cmake)
      EXTRA_CMAKE="${2:-}"
      shift 2
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
  cpu|cuda|amd)
    ;;
  *)
    echo "Invalid --backend value: ${BACKEND}" >&2
    exit 2
    ;;
esac

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
  -DSD_CPU=ON
)

case "${BACKEND}" in
  cpu)
    COMMON_FLAGS+=(-DSD_CUDA=OFF)
    ;;
  cuda)
    COMMON_FLAGS+=(-DSD_CUDA=ON)
    ;;
  amd)
    COMMON_FLAGS+=(-DSD_CUDA=ON -DSD_ZLUDA=ON -DSD_ZLUDA_TARGET=AMD)
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
    TOOLCHAIN_FLAGS+=(
      -DCMAKE_TOOLCHAIN_FILE="${LIBND4J_DIR}/cmake/android-arm64.cmake"
      -DANDROID_NDK="${ANDROID_NDK}"
      -DANDROID_NATIVE_API_LEVEL="${ANDROID_API}"
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
    TOOLCHAIN_FLAGS+=(
      -DCMAKE_TOOLCHAIN_FILE="${LIBND4J_DIR}/cmake/android-x86_64.cmake"
      -DANDROID_NDK="${ANDROID_NDK}"
      -DANDROID_NATIVE_API_LEVEL="${ANDROID_API}"
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
