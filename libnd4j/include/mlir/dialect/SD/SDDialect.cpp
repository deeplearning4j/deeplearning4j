/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * See the NOTICE file distributed with this work for additional
 * information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include "SDDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
namespace sd {

//===----------------------------------------------------------------------===//
// SD Dialect
//===----------------------------------------------------------------------===//

SDDialect::SDDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<SDDialect>()) {
    // Register operations
    // addOperations<
    //     #define GET_OP_LIST
    //     #include "mlir/dialect/SD/SDOps.cpp.inc"
    // >();
}

Operation *SDDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                          Type type, Location loc) {
    // TODO: Implement constant materialization
    return nullptr;
}

Type SDDialect::parseType(DialectAsmParser &parser) const {
    // TODO: Implement custom type parsing if needed
    return Type();
}

void SDDialect::printType(Type type, DialectAsmPrinter &os) const {
    // TODO: Implement custom type printing if needed
}

//===----------------------------------------------------------------------===//
// Type Conversion Utilities
//===----------------------------------------------------------------------===//

SDDataType mlirTypeToSDDataType(Type type) {
    if (type.isF32())
        return SDDataType::FLOAT32;
    if (type.isF64())
        return SDDataType::DOUBLE;
    if (type.isF16())
        return SDDataType::HALF;
    if (type.isBF16())
        return SDDataType::BFLOAT16;
    if (type.isInteger(8))
        return SDDataType::INT8;
    if (type.isInteger(16))
        return SDDataType::INT16;
    if (type.isInteger(32))
        return SDDataType::INT32;
    if (type.isInteger(64))
        return SDDataType::INT64;
    if (type.isUnsignedInteger(8))
        return SDDataType::UINT8;
    if (type.isUnsignedInteger(16))
        return SDDataType::UINT16;
    if (type.isUnsignedInteger(32))
        return SDDataType::UINT32;
    if (type.isUnsignedInteger(64))
        return SDDataType::UINT64;
    if (type.isInteger(1))
        return SDDataType::BOOL;

    return SDDataType::ANY;
}

Type sdDataTypeToMLIRType(MLIRContext *ctx, SDDataType dtype) {
    switch (dtype) {
        case SDDataType::FLOAT32:
            return FloatType::getF32(ctx);
        case SDDataType::DOUBLE:
            return FloatType::getF64(ctx);
        case SDDataType::HALF:
        case SDDataType::FLOAT16:
            return FloatType::getF16(ctx);
        case SDDataType::BHALF:
        case SDDataType::BFLOAT16:
            return FloatType::getBF16(ctx);
        case SDDataType::INT8:
            return IntegerType::get(ctx, 8);
        case SDDataType::INT16:
            return IntegerType::get(ctx, 16);
        case SDDataType::INT32:
            return IntegerType::get(ctx, 32);
        case SDDataType::INT64:
            return IntegerType::get(ctx, 64);
        case SDDataType::UINT8:
            return IntegerType::get(ctx, 8, IntegerType::Unsigned);
        case SDDataType::UINT16:
            return IntegerType::get(ctx, 16, IntegerType::Unsigned);
        case SDDataType::UINT32:
            return IntegerType::get(ctx, 32, IntegerType::Unsigned);
        case SDDataType::UINT64:
            return IntegerType::get(ctx, 64, IntegerType::Unsigned);
        case SDDataType::BOOL:
            return IntegerType::get(ctx, 1);
        default:
            return FloatType::getF32(ctx);  // Default to float32
    }
}

} // namespace sd
} // namespace mlir

// Include the generated operation definitions
// #define GET_OP_CLASSES
// #include "mlir/dialect/SD/SDOps.cpp.inc"
