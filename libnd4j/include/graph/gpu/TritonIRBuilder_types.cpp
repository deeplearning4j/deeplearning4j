/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// TritonIRBuilder — Type system: getMLIRType, splatConstant*, type helpers
//

#include <config.h>

#if HAVE_TRITON

#include <graph/gpu/TritonIRBuilder.h>
#include <graph/gpu/TritonIRBuilder_internal.h>

// MLIR core
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>

// Triton MLIR dialect
#include <triton/Dialect/Triton/IR/Dialect.h>
#include <triton/Dialect/Triton/IR/Types.h>

// Standard MLIR dialects
#include <mlir/Dialect/Arith/IR/Arith.h>

namespace sd {
namespace graph {

using namespace ir_builder_internal;

// ─── MLIR emission helpers ──────────────────────────────────────────────────

mlir::Type TritonIRBuilder::getMLIRType(mlir::OpBuilder& builder, DataType dtype) {
  switch (dtype) {
    case FLOAT32:  return builder.getF32Type();
    case HALF:     return builder.getF16Type();
    case BFLOAT16: return builder.getBF16Type();
    case DOUBLE:   return builder.getF64Type();
    case INT8:     return builder.getIntegerType(8);
    case UINT8:    return builder.getIntegerType(8);
    case INT16:    return builder.getIntegerType(16);
    case UINT16:   return builder.getIntegerType(16);
    case INT32:    return builder.getI32Type();
    case UINT32:   return builder.getI32Type();
    case INT64:    return builder.getI64Type();
    case UINT64:   return builder.getI64Type();
    case BOOL:     return builder.getIntegerType(8);  // Use i8, not i1: Triton's LLVM lowering
                   // generates invalid bitcast (i8 to vector<1xi1>) for i1 ptr args.
                   // BOOL is stored as 1 byte in memory. castTo() handles i8→i1 when needed.
    default:       return builder.getF32Type();
  }
}

mlir::Value TritonIRBuilder::splatConstantF32(mlir::OpBuilder& builder, mlir::Location loc,
                                               mlir::RankedTensorType tensorType, float val) {
  auto elemType = tensorType.getElementType();
  if (mlir::isa<mlir::FloatType>(elemType)) {
    auto scalarAttr = builder.getFloatAttr(elemType, static_cast<double>(val));
    auto scalar = builder.create<mlir::arith::ConstantOp>(loc, elemType, scalarAttr);
    return builder.create<mlir::triton::SplatOp>(loc, tensorType, scalar);
  } else if (elemType.isSignlessInteger()) {
    auto scalarAttr = builder.getIntegerAttr(elemType, static_cast<int64_t>(val));
    auto scalar = builder.create<mlir::arith::ConstantOp>(loc, elemType, scalarAttr);
    return builder.create<mlir::triton::SplatOp>(loc, tensorType, scalar);
  }
  // Fallback: cast from f32 scalar to actual element type to avoid tt.splat type mismatch.
  // The tensor element type must match the scalar type exactly.
  auto scalarAttr = builder.getFloatAttr(builder.getF32Type(), static_cast<double>(val));
  mlir::Value scalarVal = builder.create<mlir::arith::ConstantOp>(loc, builder.getF32Type(), scalarAttr);
  // If tensorType element type isn't f32, we need to cast
  if (elemType != builder.getF32Type()) {
    if (mlir::isa<mlir::FloatType>(elemType)) {
      unsigned srcBits = builder.getF32Type().getWidth();
      unsigned dstBits = mlir::cast<mlir::FloatType>(elemType).getWidth();
      if (dstBits > srcBits) {
        scalarVal = builder.create<mlir::arith::ExtFOp>(loc, elemType, scalarVal);
      } else {
        scalarVal = builder.create<mlir::arith::TruncFOp>(loc, elemType, scalarVal);
      }
    } else if (elemType.isSignlessInteger()) {
      scalarVal = builder.create<mlir::arith::FPToSIOp>(loc, elemType, scalarVal);
    } else {
      // Last resort: rebuild tensorType with f32 element type
      tensorType = mlir::RankedTensorType::get(tensorType.getShape(), builder.getF32Type());
    }
  }
  return builder.create<mlir::triton::SplatOp>(loc, tensorType, scalarVal);
}

mlir::Value TritonIRBuilder::splatConstantI32(mlir::OpBuilder& builder, mlir::Location loc,
                                               mlir::RankedTensorType tensorType, int val) {
  auto elemType = tensorType.getElementType();
  if (elemType.isSignlessInteger()) {
    // Create scalar matching the tensor's actual integer bit width
    int bitWidth = elemType.getIntOrFloatBitWidth();
    auto scalar = builder.create<mlir::arith::ConstantIntOp>(loc, val, bitWidth);
    return builder.create<mlir::triton::SplatOp>(loc, tensorType, scalar);
  }
  // Fallback: i32 (and rebuild tensorType to match)
  auto scalar = builder.create<mlir::arith::ConstantIntOp>(loc, val, 32);
  tensorType = mlir::RankedTensorType::get(tensorType.getShape(), builder.getI32Type());
  return builder.create<mlir::triton::SplatOp>(loc, tensorType, scalar);
}

}  // namespace graph
}  // namespace sd

#endif  // HAVE_TRITON
