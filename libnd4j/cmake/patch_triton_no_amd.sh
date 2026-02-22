#!/bin/bash
# Patch RegisterTritonDialects.h to remove AMD backend references
SRC="$1"
H="$SRC/bin/RegisterTritonDialects.h"
BIN_CMAKE="$SRC/bin/CMakeLists.txt"
if [ -f "$H" ]; then
    sed -i '/amd\/include\//d' "$H"
    sed -i '/TritonAMDGPUToLLVM/d' "$H"
    sed -i '/TritonAMDGPUTransforms/d' "$H"
    sed -i '/registerConvertTritonAMDGPUToLLVM/d' "$H"
    sed -i '/registerConvertBuiltinFuncToLLVM/d' "$H"
    sed -i '/registerDecomposeUnsupportedAMDConversions/d' "$H"
    sed -i '/registerOptimizeAMDLDSUsage/d' "$H"
    sed -i '/registerTritonAMDGPU/d' "$H"
    sed -i '/TritonAMDGPUDialect/d' "$H"
fi
if [ -f "$BIN_CMAKE" ]; then
    sed -i '/MLIRGPUToROCDLTransforms/d' "$BIN_CMAKE"
fi
