#!/bin/bash
# Custom install for Triton v3.2.0 — it uses OBJECT libraries, not install() rules.
# Usage: install_triton.sh <BINARY_DIR> <SOURCE_DIR> <INSTALL_DIR>
set -e

BINARY_DIR="$1"
SOURCE_DIR="$2"
INSTALL_DIR="$3"

echo "Triton custom install: BINARY_DIR=$BINARY_DIR SOURCE_DIR=$SOURCE_DIR INSTALL_DIR=$INSTALL_DIR"

# Archive all .o files into libtriton.a
mkdir -p "$INSTALL_DIR/lib"
find "$BINARY_DIR/lib" "$BINARY_DIR/third_party" -name '*.o' | xargs ar rcs "$INSTALL_DIR/lib/libtriton.a"

# Copy core Triton headers
mkdir -p "$INSTALL_DIR/include"
cp -a "$SOURCE_DIR/include/triton" "$INSTALL_DIR/include/"
cp -a "$BINARY_DIR/include/triton/." "$INSTALL_DIR/include/triton/" 2>/dev/null || true

# Copy NVIDIA backend headers (TritonNVIDIAGPUToLLVM, NVGPUToLLVM, Dialect/NVGPU)
# First from source dir, then overlay generated headers from build dir.
# Use cp -a with /. suffix to merge directories properly.
if [ -d "$BINARY_DIR/third_party/nvidia/include" ]; then
    if [ -d "$SOURCE_DIR/third_party/nvidia/include" ]; then
        cp -a "$SOURCE_DIR/third_party/nvidia/include/." "$INSTALL_DIR/include/"
        echo "Triton: installed NVIDIA source headers"
    fi
    cp -a "$BINARY_DIR/third_party/nvidia/include/." "$INSTALL_DIR/include/"
    echo "Triton: installed NVIDIA generated headers"
fi

# Create nvidia/include/ mirror for internal relative includes.
# Triton's NVIDIA headers use #include "nvidia/include/.../Passes.h.inc"
# which expects the Triton source tree layout. We mirror via symlinks.
if [ -d "$INSTALL_DIR/include/TritonNVIDIAGPUToLLVM" ]; then
    mkdir -p "$INSTALL_DIR/include/nvidia/include"
    ln -sfn "../../TritonNVIDIAGPUToLLVM" "$INSTALL_DIR/include/nvidia/include/TritonNVIDIAGPUToLLVM"
    ln -sfn "../../NVGPUToLLVM" "$INSTALL_DIR/include/nvidia/include/NVGPUToLLVM" 2>/dev/null || true
    ln -sfn "../../Dialect" "$INSTALL_DIR/include/nvidia/include/Dialect" 2>/dev/null || true
    echo "Triton: created nvidia/include/ symlinks for relative includes"
fi

# Copy AMD backend headers (TritonAMDGPUToLLVM, TritonAMDGPUTransforms, Dialect/TritonAMDGPU)
if [ -d "$BINARY_DIR/third_party/amd/include" ]; then
    if [ -d "$SOURCE_DIR/third_party/amd/include" ]; then
        cp -a "$SOURCE_DIR/third_party/amd/include/." "$INSTALL_DIR/include/"
        echo "Triton: installed AMD source headers"
    fi
    cp -a "$BINARY_DIR/third_party/amd/include/." "$INSTALL_DIR/include/"
    echo "Triton: installed AMD generated headers"
fi

# AMD headers contain relative includes rooted at third_party/amd/include/...
# Mirror that source-layout prefix for include resolution.
if [ -d "$INSTALL_DIR/include/TritonAMDGPUToLLVM" ]; then
    mkdir -p "$INSTALL_DIR/include/third_party/amd"
    ln -sfn ../../ "$INSTALL_DIR/include/third_party/amd/include"
    echo "Triton: created third_party/amd/include symlink mirror"
fi

# Optional Intel backend headers (if present in future Triton revisions).
if [ -d "$BINARY_DIR/third_party/intel/include" ]; then
    if [ -d "$SOURCE_DIR/third_party/intel/include" ]; then
        cp -a "$SOURCE_DIR/third_party/intel/include/." "$INSTALL_DIR/include/"
        echo "Triton: installed Intel source headers"
    fi
    cp -a "$BINARY_DIR/third_party/intel/include/." "$INSTALL_DIR/include/"
    echo "Triton: installed Intel generated headers"
fi

echo "Triton custom install complete: $(ls -lh "$INSTALL_DIR/lib/libtriton.a")"
