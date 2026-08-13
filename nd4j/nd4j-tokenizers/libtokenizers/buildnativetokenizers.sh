#!/bin/bash

################################################################################
# buildnativetokenizers.sh
# Native build script for tokenizers Rust library with C++ wrapper
# Based on the nd4j buildnativeoperations.sh pattern
# Updated to work with externalized HuggingFace tokenizers project
# Updated to use JavaCPP directory structure
################################################################################

set -e  # Exit on any error

# Build configuration
BUILD_TYPE="${BUILD_TYPE:-Release}"
TOKENIZERS_PLATFORM_RAW="${TOKENIZERS_PLATFORM:-$(uname -s | tr '[:upper:]' '[:lower:]')}"
TOKENIZERS_ARCH="${TOKENIZERS_ARCH:-$(uname -m)}"
VERBOSE="${VERBOSE:-false}"

# Visual Studio defines Platform=x64. Environment variable names are
# case-insensitive on Windows, so using the generic PLATFORM name here makes
# MSYS inherit x64 instead of the host OS reported by uname. Keep host
# detection namespaced and reduce it to the three values used by this script.
case "${TOKENIZERS_PLATFORM_RAW}" in
    linux*)
        HOST_PLATFORM="linux"
        ;;
    darwin*)
        HOST_PLATFORM="darwin"
        ;;
    mingw*|cygwin*|msys*|windows*)
        HOST_PLATFORM="windows"
        ;;
    *)
        echo "Unsupported platform: ${TOKENIZERS_PLATFORM_RAW}"
        exit 1
        ;;
esac

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${SCRIPT_DIR}/build"
OUTPUT_DIR="${SCRIPT_DIR}/target/native"

# JavaCPP platform naming - use environment variable if set by Maven
if [[ -n "${JAVACPP_PLATFORM:-}" ]]; then
    echo "Using JAVACPP_PLATFORM from environment: ${JAVACPP_PLATFORM}"
else
    case "${HOST_PLATFORM}" in
        linux)
            if [[ "${TOKENIZERS_ARCH}" == "aarch64" || "${TOKENIZERS_ARCH}" == "arm64" ]]; then
                JAVACPP_PLATFORM="linux-arm64"
            else
                JAVACPP_PLATFORM="linux-x86_64"
            fi
            ;;
        darwin)
            if [[ "${TOKENIZERS_ARCH}" == "arm64" ]]; then
                JAVACPP_PLATFORM="macosx-arm64"
            else
                JAVACPP_PLATFORM="macosx-x86_64"
            fi
            ;;
        windows)
            JAVACPP_PLATFORM="windows-x86_64"
            ;;
    esac
fi

echo "==============================================="
echo "Building tokenizers native library"
echo "==============================================="
echo "Platform: ${HOST_PLATFORM} (detected as ${TOKENIZERS_PLATFORM_RAW})"
echo "Architecture: ${TOKENIZERS_ARCH}"
echo "JavaCPP Platform: ${JAVACPP_PLATFORM}"
echo "Build type: ${BUILD_TYPE}"
echo "Build directory: ${BUILD_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "==============================================="

# Side-effect-free diagnostics used by release preflight and regression tests.
if [[ "${1:-}" == "--print-build-config" ]]; then
    exit 0
fi

# Create directories using JavaCPP structure
mkdir -p "${BUILD_DIR}"
mkdir -p "${OUTPUT_DIR}/org/eclipse/deeplearning4j/tokenizers/${JAVACPP_PLATFORM}"
mkdir -p "${OUTPUT_DIR}/org/eclipse/deeplearning4j/tokenizers/include"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check dependencies
echo "Checking dependencies..."

if ! command_exists cargo; then
    echo "Error: Rust cargo not found. Please install Rust: https://rustup.rs/"
    exit 1
fi

if ! command_exists cmake; then
    echo "Error: CMake not found. Please install CMake."
    exit 1
fi

if ! command_exists git; then
    echo "Error: Git not found. Please install Git."
    exit 1
fi

echo "All dependencies found."

# CRITICAL: Detect and use linuxbrew compiler if available to match JavaCPP's ABI
# JavaCPP uses linuxbrew's GCC on Linux, so we must use the same compiler
# to avoid C++ ABI mismatches that cause SIGSEGV crashes
if [[ "${HOST_PLATFORM}" == "linux" ]]; then
    LINUXBREW_GCC="/home/linuxbrew/.linuxbrew/bin/gcc"
    LINUXBREW_GXX="/home/linuxbrew/.linuxbrew/bin/g++"

    if [[ -x "${LINUXBREW_GXX}" ]]; then
        echo "Found linuxbrew compiler, using it to match JavaCPP ABI..."
        export CC="${LINUXBREW_GCC}"
        export CXX="${LINUXBREW_GXX}"
        echo "CC=${CC}"
        echo "CXX=${CXX}"
    else
        echo "WARNING: Linuxbrew compiler not found at ${LINUXBREW_GXX}"
        echo "WARNING: Using system compiler may cause ABI mismatch with JavaCPP!"
        echo "WARNING: This can lead to SIGSEGV crashes at runtime!"
    fi
fi

# Platform-specific configuration
case "${HOST_PLATFORM}" in
    linux)
        TARGET="x86_64-unknown-linux-gnu"
        LIB_EXT="so"
        if [[ "${TOKENIZERS_ARCH}" == "aarch64" || "${TOKENIZERS_ARCH}" == "arm64" ]]; then
            TARGET="aarch64-unknown-linux-gnu"
        fi
        ;;
    darwin)
        TARGET="x86_64-apple-darwin"
        LIB_EXT="dylib"
        if [[ "${TOKENIZERS_ARCH}" == "arm64" ]]; then
            TARGET="aarch64-apple-darwin"
        fi
        ;;
    windows)
        TARGET="x86_64-pc-windows-gnu"
        LIB_EXT="dll"
        ;;
esac

CMAKE_PLATFORM="${HOST_PLATFORM}"

echo "Building for target: ${TARGET}"

# First, build the tokenizers-ffi Rust crate that provides C FFI bindings
echo "==============================================="
echo "Building tokenizers-ffi Rust crate..."
echo "==============================================="

FFI_DIR="${SCRIPT_DIR}/tokenizers-ffi"
FFI_TARGET_DIR="${BUILD_DIR}/tokenizers-ffi"

if [[ -d "${FFI_DIR}" ]]; then
    mkdir -p "${FFI_TARGET_DIR}"

    # Build the FFI crate. The Windows C++ wrapper uses MinGW, so Rust must
    # emit a GNU archive rather than the default MSVC tokenizers_ffi.lib.
    if [[ "${HOST_PLATFORM}" == "windows" ]]; then
        CARGO_BUILD_TARGET="${CARGO_BUILD_TARGET:-${TARGET}}"
        if command_exists rustup; then
            rustup target add "${CARGO_BUILD_TARGET}"
        fi
        CARGO_TARGET_DIR="${FFI_TARGET_DIR}" cargo build --release \
            --target "${CARGO_BUILD_TARGET}" --manifest-path="${FFI_DIR}/Cargo.toml"
    else
        CARGO_TARGET_DIR="${FFI_TARGET_DIR}" cargo build --release --manifest-path="${FFI_DIR}/Cargo.toml"
    fi

    # When CARGO_BUILD_TARGET is set, cargo outputs to target/<triple>/release/ instead of target/release/
    # Copy the library to the expected location so CMake can find it
    if [[ -n "${CARGO_BUILD_TARGET}" && -d "${FFI_TARGET_DIR}/${CARGO_BUILD_TARGET}/release" ]]; then
        echo "Copying FFI library from target-specific directory: ${CARGO_BUILD_TARGET}"
        mkdir -p "${FFI_TARGET_DIR}/release"
        cp "${FFI_TARGET_DIR}/${CARGO_BUILD_TARGET}/release"/libtokenizers_ffi.* "${FFI_TARGET_DIR}/release/" 2>/dev/null || true
        cp "${FFI_TARGET_DIR}/${CARGO_BUILD_TARGET}/release"/tokenizers_ffi.* "${FFI_TARGET_DIR}/release/" 2>/dev/null || true
    fi

    # Generate C header with cbindgen
    if command_exists cbindgen; then
        echo "Generating C header with cbindgen..."
        cbindgen --lang c --output "${SCRIPT_DIR}/include/tokenizers_ffi.h" "${FFI_DIR}"
    else
        echo "Installing cbindgen..."
        cargo install cbindgen
        cbindgen --lang c --output "${SCRIPT_DIR}/include/tokenizers_ffi.h" "${FFI_DIR}"
    fi

    echo "tokenizers-ffi built successfully"
    echo "Static library: ${FFI_TARGET_DIR}/release/libtokenizers_ffi.a"
    echo "Dynamic library: ${FFI_TARGET_DIR}/release/libtokenizers_ffi.so"
else
    echo "ERROR: tokenizers-ffi directory not found at ${FFI_DIR}"
    echo "Please ensure the tokenizers-ffi crate exists"
    exit 1
fi

# Build using CMake (which will handle the external project)
echo "==============================================="
echo "Building with CMake..."
echo "==============================================="
cd "${BUILD_DIR}"

# Check if we need to reconfigure due to compiler change
CMAKE_CACHE="${BUILD_DIR}/CMakeCache.txt"
NEED_RECONFIGURE=false

if [[ -f "${CMAKE_CACHE}" ]]; then
    CACHED_CXX=$(grep "CMAKE_CXX_COMPILER:FILEPATH=" "${CMAKE_CACHE}" 2>/dev/null | cut -d= -f2)
    if [[ -n "${CXX}" && "${CACHED_CXX}" != "${CXX}" ]]; then
        echo "Compiler changed from ${CACHED_CXX} to ${CXX}, forcing reconfigure..."
        rm -f "${CMAKE_CACHE}"
        NEED_RECONFIGURE=true
    fi
fi

# Build CMake command with compiler settings
CMAKE_ARGS=(
    "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}"
    "-DCMAKE_INSTALL_PREFIX=${OUTPUT_DIR}/org/eclipse/deeplearning4j/tokenizers"
    "-DPLATFORM=${CMAKE_PLATFORM}"
    "-DARCH=${TOKENIZERS_ARCH}"
    "-DJAVACPP_PLATFORM=${JAVACPP_PLATFORM}"
)

# On Windows/MSYS2, force MinGW Makefiles generator (otherwise cmake picks Visual Studio)
if [[ "${HOST_PLATFORM}" == "windows" ]]; then
    CMAKE_ARGS+=("-G" "MinGW Makefiles")
fi

# Explicitly pass compiler if set (critical for ABI compatibility)
if [[ -n "${CC}" ]]; then
    CMAKE_ARGS+=("-DCMAKE_C_COMPILER=${CC}")
fi
if [[ -n "${CXX}" ]]; then
    CMAKE_ARGS+=("-DCMAKE_CXX_COMPILER=${CXX}")
fi

# Configure CMake with JavaCPP-aware install prefix
cmake "${SCRIPT_DIR}" "${CMAKE_ARGS[@]}"

# Build everything (CMake will handle cloning, building Rust, generating bindings, and building C++ wrapper)
cmake --build . --config "${BUILD_TYPE}" -j$(nproc 2>/dev/null || echo 4)

# Install to JavaCPP structure
cmake --install . --config "${BUILD_TYPE}"

# Platform-specific library organization in JavaCPP structure
echo "Organizing platform-specific libraries in JavaCPP structure..."

JAVACPP_PLATFORM_DIR="${OUTPUT_DIR}/org/eclipse/deeplearning4j/tokenizers/${JAVACPP_PLATFORM}"

# Function to copy essential runtime libraries only
copy_essential_libraries() {
    local source_dir="$1"
    local dest_dir="$2"

    echo "Copying essential libraries from ${source_dir} to ${dest_dir}"

    # Copy our wrapper library (the main one we built)
    case "${HOST_PLATFORM}" in
        linux)
            find "${source_dir}" -maxdepth 1 -name "libtokenizers_wrapper.so*" -exec cp {} "${dest_dir}/" \; 2>/dev/null || true
            # Also copy the FFI library if present (needed if dynamically linked)
            if [[ -f "${FFI_TARGET_DIR}/release/libtokenizers_ffi.so" ]]; then
                cp "${FFI_TARGET_DIR}/release/libtokenizers_ffi.so" "${dest_dir}/" 2>/dev/null || true
            fi
            ;;
        darwin)
            find "${source_dir}" -maxdepth 1 -name "libtokenizers_wrapper.dylib*" -exec cp {} "${dest_dir}/" \; 2>/dev/null || true
            find "${source_dir}" -maxdepth 1 -name "libtokenizers_wrapper.jnilib*" -exec cp {} "${dest_dir}/" \; 2>/dev/null || true
            if [[ -f "${FFI_TARGET_DIR}/release/libtokenizers_ffi.dylib" ]]; then
                cp "${FFI_TARGET_DIR}/release/libtokenizers_ffi.dylib" "${dest_dir}/" 2>/dev/null || true
            fi
            ;;
        windows)
            find "${source_dir}" -maxdepth 1 -name "libtokenizers_wrapper.dll*" -exec cp {} "${dest_dir}/" \; 2>/dev/null || true
            if [[ -f "${FFI_TARGET_DIR}/release/tokenizers_ffi.dll" ]]; then
                cp "${FFI_TARGET_DIR}/release/tokenizers_ffi.dll" "${dest_dir}/" 2>/dev/null || true
            fi
            ;;
    esac

    # Note: The tokenizers-ffi library contains the actual HuggingFace tokenizers implementation
    # It is either statically linked into the wrapper, or needs to be present at runtime
}

# Copy only essential runtime libraries
copy_essential_libraries "${BUILD_DIR}" "${JAVACPP_PLATFORM_DIR}"

# Copy generated headers to JavaCPP include directory
JAVACPP_INCLUDE_DIR="${OUTPUT_DIR}/org/eclipse/deeplearning4j/tokenizers/include"

# Copy generated C headers
if [[ -f "${BUILD_DIR}/tokenizers-build/bindings/include/tokenizers.h" ]]; then
    cp "${BUILD_DIR}/tokenizers-build/bindings/include/tokenizers.h" "${JAVACPP_INCLUDE_DIR}/tokenizers_c.h"
fi

# Copy wrapper headers
if [[ -d "${SCRIPT_DIR}/include" ]]; then
    cp "${SCRIPT_DIR}/include"/*.h "${JAVACPP_INCLUDE_DIR}/" 2>/dev/null || true
fi

# Create platform-specific manifest file
MANIFEST_FILE="${JAVACPP_PLATFORM_DIR}/manifest.properties"
cat > "${MANIFEST_FILE}" << EOF
# Tokenizers Native Library Manifest
build.timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
build.platform=${HOST_PLATFORM}
build.arch=${TOKENIZERS_ARCH}
build.javacpp.platform=${JAVACPP_PLATFORM}
build.type=${BUILD_TYPE}
build.target=${TARGET}
rust.version=$(cargo --version)
cmake.version=$(cmake --version | head -n1)
tokenizers.repo=https://github.com/huggingface/tokenizers.git
library.names=libtokenizers_wrapper.${LIB_EXT}
javacpp.platform.root=org/eclipse/deeplearning4j/tokenizers/${JAVACPP_PLATFORM}
# Note: Rust static libraries (.rlib) are statically linked into the wrapper
# Note: Proc-macro libraries are not included as they are compile-time only
EOF

# For backward compatibility, also create legacy structure (but only with essential libraries)
echo "Creating backward compatibility structure..."
LEGACY_PLATFORM_DIR="${OUTPUT_DIR}/${JAVACPP_PLATFORM}"
mkdir -p "${LEGACY_PLATFORM_DIR}"

# Copy essential libraries to legacy structure
copy_essential_libraries "${BUILD_DIR}" "${LEGACY_PLATFORM_DIR}"

# Copy manifest to legacy location
cp "${MANIFEST_FILE}" "${LEGACY_PLATFORM_DIR}/" 2>/dev/null || true

echo "==============================================="
echo "Build completed successfully!"
echo "JavaCPP Output directory: ${OUTPUT_DIR}/org/eclipse/deeplearning4j/tokenizers"
echo "JavaCPP Platform directory: ${JAVACPP_PLATFORM_DIR}"
echo "JavaCPP Include directory: ${JAVACPP_INCLUDE_DIR}"
echo "Legacy Platform directory: ${LEGACY_PLATFORM_DIR}"
echo "==============================================="

# List built artifacts
echo "Built artifacts:"
echo "JavaCPP Platform Libraries:"
find "${JAVACPP_PLATFORM_DIR}" -type f \( -name "*.so" -o -name "*.dll" -o -name "*.dylib" -o -name "*.jnilib" \) | sort

echo "JavaCPP Headers:"
find "${JAVACPP_INCLUDE_DIR}" -type f -name "*.h" | sort

echo "Legacy Libraries:"
find "${LEGACY_PLATFORM_DIR}" -type f \( -name "*.so" -o -name "*.dll" -o -name "*.dylib" -o -name "*.jnilib" \) | sort

echo ""
echo "Note: Rust proc-macro libraries are not included in the final package"
echo "as they are only needed at compile time. The tokenizers Rust library"
echo "is statically linked into the C++ wrapper (libtokenizers_wrapper)."

echo "Build script completed."
