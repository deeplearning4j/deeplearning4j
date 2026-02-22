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
if [ -d "$SOURCE_DIR/third_party/nvidia/include" ]; then
    cp -a "$SOURCE_DIR/third_party/nvidia/include/." "$INSTALL_DIR/include/"
    echo "Triton: installed NVIDIA source headers"
fi
if [ -d "$BINARY_DIR/third_party/nvidia/include" ]; then
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

echo "Triton custom install complete: $(ls -lh "$INSTALL_DIR/lib/libtriton.a")"
