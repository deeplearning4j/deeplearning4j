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

# Patch TritonToTritonGPUPass.cpp to add NegFOp and TanhOp as legal ops.
# Triton upstream comments out NegFOp and omits TanhOp; we need both for our IR.
TTGPU_PASS="$SRC/lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp"
if [ -f "$TTGPU_PASS" ]; then
    sed -i 's/GenericOpPattern<arith::ShRSIOp>, \/\/ NegFOp/GenericOpPattern<arith::ShRSIOp>, GenericOpPattern<arith::NegFOp>,/' "$TTGPU_PASS"
    sed -i 's/GenericOpPattern<math::FmaOp>>/GenericOpPattern<math::FmaOp>, GenericOpPattern<math::TanhOp>>/' "$TTGPU_PASS"
fi

# Patch ElementwiseOpToLLVM.cpp to add NegFOp and TanhOp LLVM lowering patterns.
ELEM_LLVM="$SRC/lib/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVM.cpp"
if [ -f "$ELEM_LLVM" ]; then
    sed -i '/POPULATE_UNARY_OP(arith::UIToFPOp, LLVM::UIToFPOp)/a\  POPULATE_UNARY_OP(arith::NegFOp, LLVM::FNegOp)' "$ELEM_LLVM"
    sed -i '/POPULATE_UNARY_OP(math::ExpOp, math::ExpOp)/a\  POPULATE_UNARY_OP(math::TanhOp, math::TanhOp)' "$ELEM_LLVM"
fi
