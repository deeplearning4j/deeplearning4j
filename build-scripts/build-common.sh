#!/usr/bin/env bash
# build-common.sh — Shared build infrastructure for deeplearning4j
# Source this file; do not execute directly.
#
# Usage:
#   source "$(dirname "${BASH_SOURCE[0]}")/build-common.sh"
#   build_platform linux-x86_64
#   build_platform linux-x86_64-cuda-12.9-cudnn
#   build_platform android-arm64

# ============================================================
# Section 1: Double-source guard and configuration defaults
# ============================================================

# Guard against double-sourcing
[[ -n "${_BUILD_COMMON_LOADED:-}" ]] && return 0
_BUILD_COMMON_LOADED=1

# Resolve the directory this file lives in, following symlinks
_bc_source="${BASH_SOURCE[0]}"
while [[ -L "${_bc_source}" ]]; do
    _bc_dir="$(cd -P "$(dirname "${_bc_source}")" >/dev/null 2>&1 && pwd)"
    _bc_source="$(readlink "${_bc_source}")"
    [[ "${_bc_source}" != /* ]] && _bc_source="${_bc_dir}/${_bc_source}"
done
SCRIPT_DIR="$(cd -P "$(dirname "${_bc_source}")" >/dev/null 2>&1 && pwd)"

# Project root is the parent of build-scripts/
PROJECT_ROOT="$(cd -P "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"

# Maven: prefer MVN env var, fall back to the user's known path, then PATH
if [[ -z "${MVN:-}" ]]; then
    if [[ -x "/home/agibsonccc/dev-apps/mvn/bin/mvn" ]]; then
        MVN="/home/agibsonccc/dev-apps/mvn/bin/mvn"
    else
        MVN="mvn"
    fi
fi
export MVN

# Thread count for libnd4j builds (matches -Dlibnd4j.buildthreads)
BUILD_THREADS="${BUILD_THREADS:-12}"

# CUDA versions supported by this project
CUDA_VERSIONS=("12.6" "12.9" "13.1")

# Compute targets per CUDA generation
CUDA_COMPUTE="8.6 9.0"             # CUDA 12.x: Ampere + Hopper
CUDA_COMPUTE_13="8.6 9.0 10.0 12.0" # CUDA 13.x: + Blackwell sm_100/sm_120

# SDX bindings output directory (overridable)
SDX_OUTPUT_DIR="${SDX_OUTPUT_DIR:-${PROJECT_ROOT}/build-output/sdx-sdk}"

# All supported platform names (49 entries)
ALL_PLATFORMS=(
    # Linux x86_64 — CPU variants
    linux-x86_64
    linux-x86_64-onednn
    linux-x86_64-avx2
    linux-x86_64-avx512

    # Linux x86_64 — CUDA variants (base, cudnn, mlir/compile)
    linux-x86_64-cuda-12.6
    linux-x86_64-cuda-12.6-cudnn
    linux-x86_64-cuda-12.6-mlir
    linux-x86_64-cuda-12.9
    linux-x86_64-cuda-12.9-cudnn
    linux-x86_64-cuda-12.9-mlir
    linux-x86_64-cuda-13.1
    linux-x86_64-cuda-13.1-cudnn
    linux-x86_64-cuda-13.1-mlir

    # Linux x86_64 — ROCm/ZLUDA and TPU
    linux-x86_64-rocm-zluda
    linux-x86_64-tpu-xla

    # Linux ARM64 — CPU variants
    linux-arm64
    linux-arm64-armcompute

    # Linux ARM64 — CUDA variants
    linux-arm64-cuda-12.6
    linux-arm64-cuda-12.6-cudnn
    linux-arm64-cuda-12.9
    linux-arm64-cuda-12.9-cudnn
    linux-arm64-cuda-13.1
    linux-arm64-cuda-13.1-cudnn

    # Android
    android-arm64
    android-x86_64

    # macOS
    macosx-arm64
    macosx-arm64-mps
    macosx-x86_64

    # Windows x86_64 — CPU variants
    windows-x86_64
    windows-x86_64-onednn
    windows-x86_64-avx2
    windows-x86_64-avx512

    # Windows x86_64 — CUDA variants (base, cudnn, mlir)
    windows-x86_64-cuda-12.6
    windows-x86_64-cuda-12.6-cudnn
    windows-x86_64-cuda-12.6-mlir
    windows-x86_64-cuda-12.9
    windows-x86_64-cuda-12.9-cudnn
    windows-x86_64-cuda-12.9-mlir
    windows-x86_64-cuda-13.1
    windows-x86_64-cuda-13.1-cudnn
    windows-x86_64-cuda-13.1-mlir

    # Extra combos (onednn+avx, mixed helpers)
    linux-x86_64-onednn-avx2
    linux-x86_64-onednn-avx512
    linux-x86_64-cuda-12.9-onednn
    linux-x86_64-cuda-13.1-onednn
    windows-x86_64-onednn-avx2
    windows-x86_64-onednn-avx512
    macosx-arm64-mps-accelerate
    linux-arm64-cuda-12.6-mlir
)

# ============================================================
# Section 2: Utility functions
# ============================================================

# Timestamped log line to stderr
log() {
    echo "[$(date '+%Y-%m-%dT%H:%M:%S')] $*" >&2
}
log_info() { log "INFO: $*"; }
log_warn() { log "WARN: $*"; }
log_error() { log "ERROR: $*"; }

# list_platforms — print all known platform names
list_platforms() {
    echo "Available platforms (${#ALL_PLATFORMS[@]} total):"
    for p in "${ALL_PLATFORMS[@]}"; do
        echo "  ${p}"
    done
}

# run_with_retry <max_attempts> <sleep_seconds> <command...>
run_with_retry() {
    local max_attempts="$1"
    local sleep_sec="$2"
    shift 2
    local attempt=1
    while true; do
        if "$@"; then
            return 0
        fi
        if [[ "${attempt}" -ge "${max_attempts}" ]]; then
            log "ERROR: command failed after ${max_attempts} attempts: $*"
            return 1
        fi
        log "Attempt ${attempt}/${max_attempts} failed; retrying in ${sleep_sec}s..."
        sleep "${sleep_sec}"
        attempt=$((attempt + 1))
    done
}

# Returns: linux / macos / windows
detect_host_os() {
    local kernel
    kernel="$(uname -s | tr '[:upper:]' '[:lower:]')"
    case "${kernel}" in
        linux*)   echo "linux" ;;
        darwin*)  echo "macos" ;;
        mingw*|msys*|cygwin*) echo "windows" ;;
        *)        echo "linux" ;;  # best guess
    esac
}

# Returns: x86_64 / arm64
detect_host_arch() {
    local machine
    machine="$(uname -m)"
    case "${machine}" in
        x86_64|amd64)          echo "x86_64" ;;
        aarch64|arm64)         echo "arm64" ;;
        *)                     echo "x86_64" ;;
    esac
}

# Returns: linux-x86_64 / linux-arm64 / macos / windows
detect_host_type() {
    local os arch
    os="$(detect_host_os)"
    arch="$(detect_host_arch)"
    case "${os}" in
        linux)   echo "linux-${arch}" ;;
        macos)   echo "macos" ;;
        windows) echo "windows" ;;
    esac
}

# can_build_on_host <platform>  — returns 0 if this host can build it, 1 otherwise
can_build_on_host() {
    local platform="$1"
    local host_type
    host_type="$(detect_host_type)"

    # Cross-compilation constraints:
    # Android can be built from Linux x86_64 or macOS (NDK supports both)
    # macOS targets require a macOS host (no cross-compile support here)
    # Windows targets require a Windows host (MSYS2)
    # Linux targets require a Linux host

    case "${platform}" in
        android-*)
            # NDK-based cross-compile: requires Linux x86_64 or macOS host
            [[ "${host_type}" == "linux-x86_64" || "${host_type}" == "macos" ]] && return 0 || return 1
            ;;
        macosx-*)
            [[ "${host_type}" == "macos" ]] && return 0 || return 1
            ;;
        windows-*)
            [[ "${host_type}" == "windows" ]] && return 0 || return 1
            ;;
        linux-x86_64*)
            [[ "${host_type}" == "linux-x86_64" ]] && return 0 || return 1
            ;;
        linux-arm64*)
            [[ "${host_type}" == "linux-arm64" ]] && return 0 || return 1
            ;;
        *)
            return 1
            ;;
    esac
}

# get_host_platforms — prints all platforms buildable on this host
get_host_platforms() {
    local platform
    for platform in "${ALL_PLATFORMS[@]}"; do
        can_build_on_host "${platform}" && echo "${platform}"
    done
}

# resolve_platform_group <group>  — expands a group name to individual platforms
# Groups: cpu, cuda, cuda-12.6, cuda-12.9, cuda-13.1, android, macos, windows, rocm, tpu, host, all
resolve_platform_group() {
    local group="$1"
    local host_type
    host_type="$(detect_host_type)"

    case "${group}" in
        all)
            printf '%s\n' "${ALL_PLATFORMS[@]}"
            ;;
        host)
            get_host_platforms
            ;;
        cpu)
            printf '%s\n' "${ALL_PLATFORMS[@]}" | grep -v 'cuda\|rocm\|tpu\|android\|mps'
            ;;
        cuda)
            printf '%s\n' "${ALL_PLATFORMS[@]}" | grep 'cuda'
            ;;
        cuda-12.6)
            printf '%s\n' "${ALL_PLATFORMS[@]}" | grep 'cuda-12\.6'
            ;;
        cuda-12.9)
            printf '%s\n' "${ALL_PLATFORMS[@]}" | grep 'cuda-12\.9'
            ;;
        cuda-13.1)
            printf '%s\n' "${ALL_PLATFORMS[@]}" | grep 'cuda-13\.1'
            ;;
        android)
            printf '%s\n' "${ALL_PLATFORMS[@]}" | grep '^android-'
            ;;
        macos)
            printf '%s\n' "${ALL_PLATFORMS[@]}" | grep '^macosx-'
            ;;
        windows)
            printf '%s\n' "${ALL_PLATFORMS[@]}" | grep '^windows-'
            ;;
        rocm)
            printf '%s\n' "${ALL_PLATFORMS[@]}" | grep 'rocm\|zluda'
            ;;
        tpu)
            printf '%s\n' "${ALL_PLATFORMS[@]}" | grep 'tpu'
            ;;
        linux)
            printf '%s\n' "${ALL_PLATFORMS[@]}" | grep '^linux-'
            ;;
        linux-x86_64)
            printf '%s\n' "${ALL_PLATFORMS[@]}" | grep '^linux-x86_64'
            ;;
        linux-arm64)
            printf '%s\n' "${ALL_PLATFORMS[@]}" | grep '^linux-arm64'
            ;;
        # Single platform — echo as-is
        *)
            echo "${group}"
            ;;
    esac
}

# preflight_check <platform>  — verifies required tools are present
preflight_check() {
    local platform="${1:-}"
    local missing=()
    local tool

    # Always required
    for tool in java cmake gcc g++; do
        command -v "${tool}" >/dev/null 2>&1 || missing+=("${tool}")
    done

    # Maven (could be a full path)
    if ! command -v "${MVN}" >/dev/null 2>&1 && [[ ! -x "${MVN}" ]]; then
        missing+=("mvn (MVN=${MVN})")
    fi

    # Platform-specific
    case "${platform}" in
        *cuda*)
            command -v nvcc >/dev/null 2>&1 || missing+=("nvcc (CUDA toolkit)")
            ;;
        *rocm*|*zluda*)
            [[ -d "/opt/rocm" || -d "${ROCM_PATH:-/opt/rocm}" ]] || missing+=("ROCm (/opt/rocm)")
            command -v cargo >/dev/null 2>&1 || missing+=("cargo (Rust, needed for ZLUDA)")
            ;;
        *tpu*)
            [[ -n "${LIBTPU_PATH:-}" && -f "${LIBTPU_PATH}" ]] || missing+=("libtpu.so (set LIBTPU_PATH)")
            ;;
        android-*)
            [[ -n "${ANDROID_NDK:-}" && -d "${ANDROID_NDK}" ]] || missing+=("ANDROID_NDK (set ANDROID_NDK env var)")
            ;;
        macosx-*-mps)
            [[ "$(detect_host_os)" == "macos" ]] || missing+=("macOS host (required for MPS)")
            ;;
        windows-*)
            command -v pacman >/dev/null 2>&1 || missing+=("MSYS2 pacman")
            ;;
    esac

    if [[ ${#missing[@]} -gt 0 ]]; then
        log "ERROR: Missing required tools/deps for '${platform}':"
        for m in "${missing[@]}"; do
            log "  - ${m}"
        done
        return 1
    fi

    log "Preflight OK for '${platform}'"
    return 0
}

# has_build_deps_for <platform>  — lightweight dependency probe (no fatal error)
has_build_deps_for() {
    local platform="$1"
    case "${platform}" in
        *cuda*)
            # Accept nvcc on PATH or a CUDA_HOME with nvcc in it
            command -v nvcc >/dev/null 2>&1 && return 0
            [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]] && return 0
            return 1
            ;;
        *rocm*|*zluda*)
            [[ -d "${ROCM_PATH:-/opt/rocm}" ]] && command -v cargo >/dev/null 2>&1 && return 0
            return 1
            ;;
        *tpu*)
            [[ -n "${LIBTPU_PATH:-}" && -f "${LIBTPU_PATH}" ]] && return 0
            return 1
            ;;
        android-*)
            [[ -n "${ANDROID_NDK:-}" && -d "${ANDROID_NDK}" ]] && return 0
            return 1
            ;;
        macosx-*-mps)
            [[ "$(detect_host_os)" == "macos" ]] && return 0
            return 1
            ;;
        *)
            return 0
            ;;
    esac
}

# ============================================================
# Section 3: CUDA version management
# ============================================================

# cuda_cudnn_version <cuda_ver>  — prints cuDNN version
cuda_cudnn_version() {
    local cuda_ver="$1"
    case "${cuda_ver}" in
        13.1) echo "9.19" ;;
        12.9) echo "9.10" ;;
        12.6) echo "9.5"  ;;
        *)
            log "ERROR: Unknown CUDA version '${cuda_ver}' (supported: 12.6, 12.9, 13.1)"
            return 1
            ;;
    esac
}

# cuda_javacpp_version <cuda_ver>  — prints JavaCPP version
cuda_javacpp_version() {
    local cuda_ver="$1"
    case "${cuda_ver}" in
        13.1) echo "1.5.13" ;;
        12.9) echo "1.5.12" ;;
        12.6) echo "1.5.11" ;;
        *)
            log "ERROR: Unknown CUDA version '${cuda_ver}' (supported: 12.6, 12.9, 13.1)"
            return 1
            ;;
    esac
}

# change_cuda_version <cuda_ver>  — runs the project's change-cuda-versions.sh
change_cuda_version() {
    local cuda_ver="$1"
    local script="${PROJECT_ROOT}/change-cuda-versions.sh"
    if [[ ! -x "${script}" ]]; then
        log "ERROR: ${script} not found or not executable"
        return 1
    fi
    log "Switching project to CUDA ${cuda_ver}..."
    (cd "${PROJECT_ROOT}" && bash "${script}" "${cuda_ver}")
}

# _extract_cuda_versions <platform_string>  — echoes the CUDA version embedded in a platform name
# e.g. "linux-x86_64-cuda-12.9-cudnn" → "12.9"
_extract_cuda_versions() {
    local platform="$1"
    # Match cuda-XX.Y substring
    if [[ "${platform}" =~ cuda-([0-9]+\.[0-9]+) ]]; then
        echo "${BASH_REMATCH[1]}"
    fi
}

# ============================================================
# Section 4: Environment setup functions
# ============================================================

setup_env_linux_x86_64() {
    export CC="${CC:-gcc}"
    export CXX="${CXX:-g++}"
    log "Environment: Linux x86_64 (CC=${CC}, CXX=${CXX})"
}

setup_env_linux_arm64() {
    export CC="${CC:-gcc}"
    export CXX="${CXX:-g++}"
    log "Environment: Linux ARM64 (CC=${CC}, CXX=${CXX})"
}

# setup_env_linux_cuda <cuda_version>
setup_env_linux_cuda() {
    local cuda_ver="$1"
    # Common CUDA installation paths (network installs, runfile, and distro packages)
    local candidates=(
        "${CUDA_HOME:-}"
        "/usr/local/cuda-${cuda_ver}"
        "/usr/local/cuda"
        "/opt/cuda"
    )
    local found=""
    for cand in "${candidates[@]}"; do
        [[ -z "${cand}" ]] && continue
        [[ -d "${cand}" ]] && { found="${cand}"; break; }
    done

    if [[ -n "${found}" ]]; then
        export CUDA_HOME="${found}"
        export PATH="${CUDA_HOME}/bin:${PATH}"
        export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
        log "CUDA ${cuda_ver}: CUDA_HOME=${CUDA_HOME}"
    else
        log "WARNING: Could not find CUDA ${cuda_ver} installation; assuming nvcc is on PATH"
    fi
}

setup_env_linux_rocm() {
    local rocm_path="${ROCM_PATH:-${ROCM_HOME:-/opt/rocm}}"
    if [[ -d "${rocm_path}" ]]; then
        export ROCM_PATH="${rocm_path}"
        export PATH="${rocm_path}/bin:${PATH}"
        export LD_LIBRARY_PATH="${rocm_path}/lib:${LD_LIBRARY_PATH:-}"
        log "ROCm: ROCM_PATH=${ROCM_PATH}"
    else
        log "WARNING: ROCm not found at ${rocm_path}; ZLUDA build may fail"
    fi
}

setup_env_linux_tpu() {
    if [[ -n "${LIBTPU_PATH:-}" ]]; then
        export LD_LIBRARY_PATH="$(dirname "${LIBTPU_PATH}"):${LD_LIBRARY_PATH:-}"
        log "TPU: LIBTPU_PATH=${LIBTPU_PATH}"
    else
        log "WARNING: LIBTPU_PATH is not set; TPU build may fail"
    fi
    # PJRT plugin path for OpenXLA
    if [[ -n "${PJRT_PLUGIN_PATH:-}" ]]; then
        export PJRT_PLUGIN_PATH
        log "TPU: PJRT_PLUGIN_PATH=${PJRT_PLUGIN_PATH}"
    fi
}

setup_env_android() {
    # Prefer explicit ANDROID_NDK, then common install locations
    if [[ -z "${ANDROID_NDK:-}" ]]; then
        local candidates=(
            "${ANDROID_NDK_ROOT:-}"
            "${ANDROID_NDK_HOME:-}"
            "${HOME}/Android/Sdk/ndk-bundle"
            "${HOME}/Android/android-ndk"
        )
        for cand in "${candidates[@]}"; do
            [[ -z "${cand}" ]] && continue
            [[ -d "${cand}" ]] && { ANDROID_NDK="${cand}"; break; }
        done
    fi

    if [[ -z "${ANDROID_NDK:-}" ]]; then
        log "ERROR: ANDROID_NDK not set and no default location found"
        return 1
    fi

    export ANDROID_NDK
    export ANDROID_VERSION="${ANDROID_VERSION:-21}"
    log "Android NDK: ANDROID_NDK=${ANDROID_NDK}, API=${ANDROID_VERSION}"
}

setup_env_macos() {
    # Prefer Homebrew clang when it provides a newer toolchain
    if command -v brew >/dev/null 2>&1; then
        local llvm_prefix
        llvm_prefix="$(brew --prefix llvm 2>/dev/null || true)"
        if [[ -d "${llvm_prefix}/bin" ]]; then
            export CC="${CC:-${llvm_prefix}/bin/clang}"
            export CXX="${CXX:-${llvm_prefix}/bin/clang++}"
        fi
    fi
    export CC="${CC:-clang}"
    export CXX="${CXX:-clang++}"
    log "Environment: macOS (CC=${CC}, CXX=${CXX})"
}

setup_env_windows() {
    # MSYS2 environment; assume MinGW64 toolchain
    export CC="${CC:-gcc}"
    export CXX="${CXX:-g++}"
    log "Environment: Windows/MSYS2 (CC=${CC}, CXX=${CXX})"
}

setup_env_windows_cuda() {
    local cuda_ver="$1"
    # CUDA on Windows is typically installed under Program Files
    local cuda_base="${CUDA_HOME:-/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${cuda_ver}}"
    if [[ -d "${cuda_base}" ]]; then
        export CUDA_HOME="${cuda_base}"
        export PATH="${CUDA_HOME}/bin:${PATH}"
        log "CUDA ${cuda_ver} (Windows): CUDA_HOME=${CUDA_HOME}"
    else
        log "WARNING: CUDA not found at '${cuda_base}'; assuming nvcc is on PATH"
    fi
}

# setup_environment <platform>  — dispatches to the right setup function
setup_environment() {
    local platform="$1"
    local cuda_ver
    cuda_ver="$(_extract_cuda_versions "${platform}")"

    case "${platform}" in
        linux-x86_64*)
            setup_env_linux_x86_64
            [[ -n "${cuda_ver}" ]] && setup_env_linux_cuda "${cuda_ver}"
            [[ "${platform}" == *rocm* || "${platform}" == *zluda* ]] && setup_env_linux_rocm
            [[ "${platform}" == *tpu* ]] && setup_env_linux_tpu
            ;;
        linux-arm64*)
            setup_env_linux_arm64
            [[ -n "${cuda_ver}" ]] && setup_env_linux_cuda "${cuda_ver}"
            ;;
        android-*)
            setup_env_android
            ;;
        macosx-*)
            setup_env_macos
            ;;
        windows-x86_64*)
            setup_env_windows
            [[ -n "${cuda_ver}" ]] && setup_env_windows_cuda "${cuda_ver}"
            ;;
        *)
            log "WARNING: No specific environment setup for '${platform}'"
            ;;
    esac
}

# ============================================================
# Section 5: Dependency installation (cross-platform)
# ============================================================

install_openblas() {
    local os
    os="$(detect_host_os)"
    case "${os}" in
        linux)
            if command -v apt-get >/dev/null 2>&1; then
                sudo apt-get install -y libopenblas-dev
            elif command -v yum >/dev/null 2>&1; then
                sudo yum install -y openblas-devel
            elif command -v dnf >/dev/null 2>&1; then
                sudo dnf install -y openblas-devel
            else
                log "WARNING: Cannot install OpenBLAS: no known package manager"
            fi
            ;;
        macos)
            brew install openblas
            ;;
        windows)
            pacman -S --noconfirm mingw-w64-x86_64-openblas
            ;;
    esac
}

# Download and build ARM Compute Library if not already present
install_armcompute() {
    local dest="${ARMCOMPUTE_DIR:-${PROJECT_ROOT}/third_party/arm_compute}"
    if [[ -f "${dest}/build/libarm_compute-static.a" ]]; then
        log "ARM Compute Library already present at ${dest}"
        export ARMCOMPUTE_DIR="${dest}"
        return 0
    fi

    local version="${ARMCOMPUTE_VERSION:-23.08}"
    local url="https://github.com/ARM-software/ComputeLibrary/archive/refs/tags/v${version}.tar.gz"
    local tmpdir
    tmpdir="$(mktemp -d)"
    log "Downloading ARM Compute Library ${version}..."
    curl -fsSL "${url}" -o "${tmpdir}/acl.tar.gz"
    mkdir -p "${dest}"
    tar -xzf "${tmpdir}/acl.tar.gz" -C "${dest}" --strip-components=1
    rm -rf "${tmpdir}"

    log "Building ARM Compute Library..."
    (
        cd "${dest}"
        scons Werror=0 debug=0 asserts=0 neon=1 opencl=0 os=linux arch=arm64-v8a build=native \
              -j"${BUILD_THREADS}" build_dir=build
    )
    export ARMCOMPUTE_DIR="${dest}"
    log "ARM Compute Library installed at ${dest}"
}

# Download Android NDK if ANDROID_NDK is not already set / present
install_android_ndk() {
    local version="${NDK_VERSION:-r26d}"
    local dest="${ANDROID_NDK:-${HOME}/Android/android-ndk-${version}}"

    if [[ -d "${dest}" ]]; then
        log "Android NDK already present at ${dest}"
        export ANDROID_NDK="${dest}"
        return 0
    fi

    local os_tag
    case "$(detect_host_os)" in
        linux)  os_tag="linux" ;;
        macos)  os_tag="darwin" ;;
        *)      log "ERROR: NDK auto-install only supported on Linux/macOS"; return 1 ;;
    esac

    local url="https://dl.google.com/android/repository/android-ndk-${version}-${os_tag}.zip"
    local tmpdir
    tmpdir="$(mktemp -d)"
    log "Downloading Android NDK ${version}..."
    curl -fsSL "${url}" -o "${tmpdir}/ndk.zip"
    mkdir -p "$(dirname "${dest}")"
    unzip -q "${tmpdir}/ndk.zip" -d "$(dirname "${dest}")"
    rm -rf "${tmpdir}"
    # The zip unpacks as android-ndk-<version>/; rename if needed
    local extracted="$(dirname "${dest}")/android-ndk-${version}"
    [[ -d "${extracted}" && "${extracted}" != "${dest}" ]] && mv "${extracted}" "${dest}"
    export ANDROID_NDK="${dest}"
    log "Android NDK ${version} installed at ${ANDROID_NDK}"
}

# Build ZLUDA from source using Cargo
install_zluda() {
    local dest="${ZLUDA_DIR:-${PROJECT_ROOT}/third_party/zluda}"
    if [[ -f "${dest}/target/release/libzluda.so" ]]; then
        log "ZLUDA already built at ${dest}"
        export ZLUDA_DIR="${dest}"
        return 0
    fi

    if ! command -v cargo >/dev/null 2>&1; then
        log "ERROR: cargo not found; install Rust >= 1.89 first"
        return 1
    fi

    log "Cloning and building ZLUDA..."
    [[ -d "${dest}" ]] || git clone --depth 1 https://github.com/vosen/ZLUDA.git "${dest}"
    (
        cd "${dest}"
        cargo build --release
    )
    export ZLUDA_DIR="${dest}"
    log "ZLUDA built at ${ZLUDA_DIR}"
}

_install_java() {
    local os
    os="$(detect_host_os)"
    case "${os}" in
        linux)
            sudo apt-get install -y openjdk-17-jdk 2>/dev/null \
                || sudo dnf install -y java-17-openjdk-devel 2>/dev/null \
                || log "WARNING: Could not install Java automatically"
            ;;
        macos) brew install openjdk@17 ;;
        windows) pacman -S --noconfirm mingw-w64-x86_64-jdk17-openjdk ;;
    esac
}

_install_maven() {
    local os
    os="$(detect_host_os)"
    case "${os}" in
        linux)
            sudo apt-get install -y maven 2>/dev/null \
                || sudo dnf install -y maven 2>/dev/null \
                || log "WARNING: Could not install Maven automatically"
            ;;
        macos) brew install maven ;;
        windows) pacman -S --noconfirm mingw-w64-x86_64-maven ;;
    esac
}

_install_cmake() {
    local os
    os="$(detect_host_os)"
    case "${os}" in
        linux)
            sudo apt-get install -y cmake 2>/dev/null \
                || sudo dnf install -y cmake 2>/dev/null \
                || log "WARNING: Could not install CMake automatically"
            ;;
        macos) brew install cmake ;;
        windows) pacman -S --noconfirm mingw-w64-x86_64-cmake ;;
    esac
}

_install_protobuf() {
    local os
    os="$(detect_host_os)"
    case "${os}" in
        linux)
            sudo apt-get install -y protobuf-compiler libprotobuf-dev 2>/dev/null \
                || sudo dnf install -y protobuf-compiler protobuf-devel 2>/dev/null \
                || log "WARNING: Could not install protobuf automatically"
            ;;
        macos) brew install protobuf ;;
        windows) pacman -S --noconfirm mingw-w64-x86_64-protobuf ;;
    esac
}

_install_sccache() {
    if command -v sccache >/dev/null 2>&1; then
        log "sccache already installed"
        return 0
    fi
    if command -v cargo >/dev/null 2>&1; then
        cargo install sccache
    else
        log "WARNING: cargo not found; cannot install sccache"
    fi
}

_install_rust() {
    if command -v rustup >/dev/null 2>&1; then
        log "Rust already installed"
        return 0
    fi
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
    # shellcheck source=/dev/null
    source "${HOME}/.cargo/env"
}

# Install CUDA toolkit for a given version on Linux (network installer path)
_install_cuda_linux() {
    local cuda_ver="$1"
    local cuda_major="${cuda_ver%%.*}"
    local cuda_minor="${cuda_ver##*.}"

    if [[ -d "/usr/local/cuda-${cuda_ver}" ]]; then
        log "CUDA ${cuda_ver} already installed at /usr/local/cuda-${cuda_ver}"
        return 0
    fi

    log "Installing CUDA ${cuda_ver} (requires sudo and network access)..."
    # Use the CUDA keyring approach; distro-agnostic instructions vary by version.
    # Provide a best-effort install via runfile for portability.
    local url="https://developer.download.nvidia.com/compute/cuda/${cuda_ver}/local_installers/cuda_${cuda_ver}.0_550.54.14_linux.run"
    local tmpfile
    tmpfile="$(mktemp --suffix=.run)"
    log "Downloading CUDA installer from ${url} ..."
    curl -fsSL "${url}" -o "${tmpfile}"
    sudo sh "${tmpfile}" --toolkit --silent --override
    rm -f "${tmpfile}"
}

# Install CUDA toolkit for a given version on Windows (network installer)
_install_cuda_windows() {
    local cuda_ver="$1"
    log "CUDA ${cuda_ver} on Windows: please download from https://developer.nvidia.com/cuda-downloads"
    log "After installing, ensure nvcc.exe is on PATH and re-run."
    return 1
}

# Install all deps needed for platforms this host can build
auto_setup_host() {
    local os
    os="$(detect_host_os)"
    log "Auto-setting up host for ${os}..."

    _install_java
    _install_maven
    _install_cmake
    _install_protobuf
    _install_rust
    _install_sccache
    install_openblas

    log "Host setup complete. Platform-specific deps (CUDA, NDK, ROCm) must be configured separately."
}

# ============================================================
# Section 6: Build functions
# ============================================================

# Common flags injected into every Maven build per user rules
_common_flags() {
    local flags="-DskipTests -Dlibnd4j.triton=ON -Dlibnd4j.buildthreads=${BUILD_THREADS}"
    [[ "${DEPLOY:-0}" == "1" ]] && flags="${flags} -DperformRelease"
    echo "${flags}"
}

# build_libnd4j_cpu [options...]
# Options (pass as arguments):
#   onednn       — enable oneDNN helper
#   armcompute   — enable ARM Compute Library helper
#   avx2         — set AVX2 extension
#   avx512       — set AVX-512 extension
build_libnd4j_cpu() {
    local helper=""
    local extension=""
    local extra_flags=()

    for opt in "$@"; do
        case "${opt}" in
            onednn)
                helper="${helper:+${helper},}onednn"
                ;;
            armcompute)
                helper="${helper:+${helper},}armcompute"
                ;;
            avx2)
                extension="avx2"
                ;;
            avx512)
                extension="avx512"
                ;;
            onednn-avx2)
                helper="${helper:+${helper},}onednn"
                extension="avx2"
                ;;
            onednn-avx512)
                helper="${helper:+${helper},}onednn"
                extension="avx512"
                ;;
        esac
    done

    [[ -n "${helper}" ]]    && extra_flags+=("-Dlibnd4j.helper=${helper}")
    [[ -n "${extension}" ]] && extra_flags+=("-Dlibnd4j.extension=${extension}")

    log "Building CPU backend (helper=${helper:-none}, extension=${extension:-none})..."
    (
        cd "${PROJECT_ROOT}"
        # shellcheck disable=SC2086
        "${MVN}" -Pcpu clean install $(_common_flags) "${extra_flags[@]}" \
            -pl :libnd4j,:nd4j-native-preset,:nd4j-native
    )
}

# build_libnd4j_cuda <cuda_version> [options...]
# Options:
#   cudnn      — enable cuDNN helper
#   mlir       — enable MLIR helper
#   onednn     — enable oneDNN helper alongside CUDA
build_libnd4j_cuda() {
    local cuda_ver="$1"
    shift

    # Select compute targets based on CUDA generation
    local compute_targets
    case "${cuda_ver}" in
        13.*) compute_targets="${CUDA_COMPUTE_13}" ;;
        *)    compute_targets="${CUDA_COMPUTE}" ;;
    esac

    local helper=""
    local extra_flags=()

    for opt in "$@"; do
        case "${opt}" in
            cudnn)  helper="${helper:+${helper},}cudnn" ;;
            mlir)   helper="${helper:+${helper},}mlir" ;;
            onednn) helper="${helper:+${helper},}onednn" ;;
        esac
    done

    [[ -n "${helper}" ]] && extra_flags+=("-Dlibnd4j.helper=${helper}")

    # Ensure the project pom files reference the right CUDA version
    change_cuda_version "${cuda_ver}"

    log "Building CUDA ${cuda_ver} backend (compute='${compute_targets}', helper=${helper:-none})..."
    (
        cd "${PROJECT_ROOT}"
        # shellcheck disable=SC2086
        "${MVN}" -Pcuda -Dlibnd4j.chip=cuda \
            -Dlibnd4j.compute="${compute_targets}" \
            clean install $(_common_flags) "${extra_flags[@]}" \
            -pl :libnd4j,:nd4j-cuda-${cuda_ver}-preset,:nd4j-cuda-${cuda_ver}
    )
}

# build_libnd4j_rocm — ZLUDA/ROCm variant (AMD GPU)
build_libnd4j_rocm() {
    local zluda_target="${ZLUDA_TARGET:-AMD}"

    log "Building ROCm/ZLUDA backend (target=${zluda_target})..."
    (
        cd "${PROJECT_ROOT}"
        # shellcheck disable=SC2086
        "${MVN}" -Pcuda -Pzluda-amd \
            -Dlibnd4j.chip=cuda \
            -DSD_ZLUDA=ON \
            -DZLUDA_TARGET="${zluda_target}" \
            clean install $(_common_flags) \
            -pl :libnd4j,:nd4j-zluda
    )
}

# build_libnd4j_tpu — TPU/OpenXLA PJRT variant
build_libnd4j_tpu() {
    log "Building TPU/XLA backend..."
    (
        cd "${PROJECT_ROOT}"
        # shellcheck disable=SC2086
        "${MVN}" -Dlibnd4j.tpu=true \
            -Dlibnd4j.chip=tpu \
            clean install $(_common_flags) \
            -pl :libnd4j,:nd4j-tpu-preset,:nd4j-tpu
    )
}

# build_libnd4j_android <arch>
# arch: arm64 | x86_64
build_libnd4j_android() {
    local arch="${1:-arm64}"
    local platform_flag
    case "${arch}" in
        arm64)   platform_flag="android-arm64" ;;
        x86_64)  platform_flag="android-x86_64" ;;
        *)
            log "ERROR: Unknown Android arch '${arch}' (use arm64 or x86_64)"
            return 1
            ;;
    esac

    setup_env_android || return 1

    log "Building Android ${arch} backend..."
    (
        cd "${PROJECT_ROOT}"
        # shellcheck disable=SC2086
        "${MVN}" -Pcpu \
            -Dlibnd4j.platform="${platform_flag}" \
            -DANDROID_NDK="${ANDROID_NDK}" \
            clean install $(_common_flags) \
            -pl :libnd4j,:nd4j-native-preset,:nd4j-native
    )
}

# build_libnd4j_macos [mps]
# Options:
#   mps  — enable Metal Performance Shaders backend
build_libnd4j_macos() {
    local helper=""
    local extra_flags=()

    for opt in "$@"; do
        case "${opt}" in
            mps) helper="${helper:+${helper},}mps" ;;
        esac
    done

    [[ -n "${helper}" ]] && extra_flags+=("-Dlibnd4j.helper=${helper}")

    log "Building macOS backend (helper=${helper:-none})..."
    (
        cd "${PROJECT_ROOT}"
        # shellcheck disable=SC2086
        "${MVN}" -Pcpu \
            clean install $(_common_flags) "${extra_flags[@]}" \
            -pl :libnd4j,:nd4j-native-preset,:nd4j-native
    )
}

# ============================================================
# Section 7: SDX bindings packaging
# ============================================================

# package_sdx_bindings <platform>
# Finds the sdx-runtime-sdk/dist/ directory produced by the Maven/CMake build
# and copies artifacts to SDX_OUTPUT_DIR.
package_sdx_bindings() {
    local platform="$1"

    # Map platform name to the blasbuild subdirectory CMake uses
    local blasbuild_dir
    case "${platform}" in
        *cuda*)   blasbuild_dir="${PROJECT_ROOT}/libnd4j/blasbuild/cuda" ;;
        *rocm*|*zluda*) blasbuild_dir="${PROJECT_ROOT}/libnd4j/blasbuild/cuda" ;;
        *tpu*)    blasbuild_dir="${PROJECT_ROOT}/libnd4j/blasbuild/tpu" ;;
        android-arm64)   blasbuild_dir="${PROJECT_ROOT}/libnd4j/blasbuild/cpu" ;;
        android-x86_64)  blasbuild_dir="${PROJECT_ROOT}/libnd4j/blasbuild/cpu" ;;
        *)        blasbuild_dir="${PROJECT_ROOT}/libnd4j/blasbuild/cpu" ;;
    esac

    local dist_dir="${blasbuild_dir}/sdx-runtime-sdk/dist"

    if [[ ! -d "${dist_dir}" ]]; then
        log "SDX dist directory not found at ${dist_dir} — skipping packaging"
        return 0
    fi

    local dest="${SDX_OUTPUT_DIR}/${platform}"
    mkdir -p "${dest}"

    local found_any=0

    # Collect zip bundles
    while IFS= read -r -d '' f; do
        log "Packaging SDX artifact: $(basename "${f}") → ${dest}/"
        cp -p "${f}" "${dest}/"
        found_any=1
    done < <(find "${dist_dir}" -maxdepth 2 -type f -name "*.zip" -print0 2>/dev/null)

    # Android: also collect .aar files
    if [[ "${platform}" == android-* ]]; then
        while IFS= read -r -d '' f; do
            log "Packaging Android AAR: $(basename "${f}") → ${dest}/"
            cp -p "${f}" "${dest}/"
            found_any=1
        done < <(find "${dist_dir}" -maxdepth 2 -type f -name "*.aar" -print0 2>/dev/null)
    fi

    # macOS: also collect .xcframework bundles (directories)
    if [[ "${platform}" == macosx-* ]]; then
        while IFS= read -r -d '' d; do
            log "Packaging XCFramework: $(basename "${d}") → ${dest}/"
            cp -rp "${d}" "${dest}/"
            found_any=1
        done < <(find "${dist_dir}" -maxdepth 2 -type d -name "*.xcframework" -print0 2>/dev/null)
    fi

    if [[ "${found_any}" -eq 0 ]]; then
        log "WARNING: No SDX artifacts found in ${dist_dir}"
    else
        log "SDX bindings packaged to ${dest}"
    fi
}

# ============================================================
# Section 8: Platform dispatch
# ============================================================

# build_platform <platform_name>
# Returns 0 on success, 1 on failure.
build_platform() {
    local platform="$1"

    log "=== Building platform: ${platform} ==="
    setup_environment "${platform}"

    local rc=0
    case "${platform}" in

        # ---- Linux x86_64 CPU variants ----
        linux-x86_64)
            build_libnd4j_cpu || rc=1 ;;

        linux-x86_64-onednn)
            build_libnd4j_cpu onednn || rc=1 ;;

        linux-x86_64-avx2)
            build_libnd4j_cpu avx2 || rc=1 ;;

        linux-x86_64-avx512)
            build_libnd4j_cpu avx512 || rc=1 ;;

        linux-x86_64-onednn-avx2)
            build_libnd4j_cpu onednn-avx2 || rc=1 ;;

        linux-x86_64-onednn-avx512)
            build_libnd4j_cpu onednn-avx512 || rc=1 ;;

        # ---- Linux x86_64 CUDA 12.6 ----
        linux-x86_64-cuda-12.6)
            build_libnd4j_cuda "12.6" || rc=1 ;;

        linux-x86_64-cuda-12.6-cudnn)
            build_libnd4j_cuda "12.6" cudnn || rc=1 ;;

        linux-x86_64-cuda-12.6-mlir)
            build_libnd4j_cuda "12.6" mlir || rc=1 ;;

        # ---- Linux x86_64 CUDA 12.9 ----
        linux-x86_64-cuda-12.9)
            build_libnd4j_cuda "12.9" || rc=1 ;;

        linux-x86_64-cuda-12.9-cudnn)
            build_libnd4j_cuda "12.9" cudnn || rc=1 ;;

        linux-x86_64-cuda-12.9-mlir)
            build_libnd4j_cuda "12.9" mlir || rc=1 ;;

        linux-x86_64-cuda-12.9-onednn)
            build_libnd4j_cuda "12.9" onednn || rc=1 ;;

        # ---- Linux x86_64 CUDA 13.1 ----
        linux-x86_64-cuda-13.1)
            build_libnd4j_cuda "13.1" || rc=1 ;;

        linux-x86_64-cuda-13.1-cudnn)
            build_libnd4j_cuda "13.1" cudnn || rc=1 ;;

        linux-x86_64-cuda-13.1-mlir)
            build_libnd4j_cuda "13.1" mlir || rc=1 ;;

        # ---- Linux x86_64 ROCm/TPU ----
        linux-x86_64-rocm-zluda)
            build_libnd4j_rocm || rc=1 ;;

        linux-x86_64-tpu-xla)
            build_libnd4j_tpu || rc=1 ;;

        # ---- Linux ARM64 CPU variants ----
        linux-arm64)
            build_libnd4j_cpu || rc=1 ;;

        linux-arm64-armcompute)
            build_libnd4j_cpu armcompute || rc=1 ;;

        # ---- Linux ARM64 CUDA 12.6 ----
        linux-arm64-cuda-12.6)
            build_libnd4j_cuda "12.6" || rc=1 ;;

        linux-arm64-cuda-12.6-cudnn)
            build_libnd4j_cuda "12.6" cudnn || rc=1 ;;

        # ---- Linux ARM64 CUDA 12.9 ----
        linux-arm64-cuda-12.9)
            build_libnd4j_cuda "12.9" || rc=1 ;;

        linux-arm64-cuda-12.9-cudnn)
            build_libnd4j_cuda "12.9" cudnn || rc=1 ;;

        # ---- Linux ARM64 CUDA 13.1 ----
        linux-arm64-cuda-13.1)
            build_libnd4j_cuda "13.1" || rc=1 ;;

        linux-arm64-cuda-13.1-cudnn)
            build_libnd4j_cuda "13.1" cudnn || rc=1 ;;

        # ---- Android ----
        android-arm64)
            build_libnd4j_android arm64 || rc=1 ;;

        android-x86_64)
            build_libnd4j_android x86_64 || rc=1 ;;

        # ---- macOS ----
        macosx-arm64)
            build_libnd4j_macos || rc=1 ;;

        macosx-arm64-mps)
            build_libnd4j_macos mps || rc=1 ;;

        macosx-x86_64)
            build_libnd4j_macos || rc=1 ;;

        # ---- Windows x86_64 CPU variants ----
        windows-x86_64)
            build_libnd4j_cpu || rc=1 ;;

        windows-x86_64-onednn)
            build_libnd4j_cpu onednn || rc=1 ;;

        windows-x86_64-avx2)
            build_libnd4j_cpu avx2 || rc=1 ;;

        windows-x86_64-avx512)
            build_libnd4j_cpu avx512 || rc=1 ;;

        windows-x86_64-onednn-avx2)
            build_libnd4j_cpu onednn-avx2 || rc=1 ;;

        windows-x86_64-onednn-avx512)
            build_libnd4j_cpu onednn-avx512 || rc=1 ;;

        # ---- Windows x86_64 CUDA 12.6 ----
        windows-x86_64-cuda-12.6)
            build_libnd4j_cuda "12.6" || rc=1 ;;

        windows-x86_64-cuda-12.6-cudnn)
            build_libnd4j_cuda "12.6" cudnn || rc=1 ;;

        windows-x86_64-cuda-12.6-mlir)
            build_libnd4j_cuda "12.6" mlir || rc=1 ;;

        # ---- Windows x86_64 CUDA 12.9 ----
        windows-x86_64-cuda-12.9)
            build_libnd4j_cuda "12.9" || rc=1 ;;

        windows-x86_64-cuda-12.9-cudnn)
            build_libnd4j_cuda "12.9" cudnn || rc=1 ;;

        windows-x86_64-cuda-12.9-mlir)
            build_libnd4j_cuda "12.9" mlir || rc=1 ;;

        # ---- Windows x86_64 CUDA 13.1 ----
        windows-x86_64-cuda-13.1)
            build_libnd4j_cuda "13.1" || rc=1 ;;

        windows-x86_64-cuda-13.1-cudnn)
            build_libnd4j_cuda "13.1" cudnn || rc=1 ;;

        windows-x86_64-cuda-13.1-mlir)
            build_libnd4j_cuda "13.1" mlir || rc=1 ;;

        # ---- Extra combos (onednn+avx, mixed helpers) ----
        linux-x86_64-onednn-avx2)
            build_libnd4j_cpu onednn-avx2 || rc=1 ;;

        linux-x86_64-onednn-avx512)
            build_libnd4j_cpu onednn-avx512 || rc=1 ;;

        linux-x86_64-cuda-12.9-onednn)
            build_libnd4j_cuda "12.9" onednn || rc=1 ;;

        linux-x86_64-cuda-13.1-onednn)
            build_libnd4j_cuda "13.1" onednn || rc=1 ;;

        windows-x86_64-onednn-avx2)
            build_libnd4j_cpu onednn-avx2 || rc=1 ;;

        windows-x86_64-onednn-avx512)
            build_libnd4j_cpu onednn-avx512 || rc=1 ;;

        # MPS + Accelerate combo (macOS ARM64)
        macosx-arm64-mps-accelerate)
            build_libnd4j_macos mps || rc=1 ;;

        # ARM64 MLIR (cross-compile)
        linux-arm64-cuda-12.6-mlir)
            build_libnd4j_cuda "12.6" mlir || rc=1 ;;

        *)
            log "ERROR: Unknown platform '${platform}'"
            log "Known platforms: ${ALL_PLATFORMS[*]}"
            return 1
            ;;
    esac

    if [[ "${rc}" -ne 0 ]]; then
        log "ERROR: Build failed for '${platform}' (exit code ${rc})"
        return 1
    fi

    # Package SDX bindings regardless of whether they were updated
    package_sdx_bindings "${platform}"

    log "=== Build complete: ${platform} ==="
    return 0
}
