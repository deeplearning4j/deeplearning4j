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

#include <graph/vulkan/VulkanOpLowerings.h>
#include <graph/vulkan/VulkanKernelEmitterCatalog.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/headers/llm.h>
#include <ops/declarable/helpers/fusedElementwiseChain.h>
#include <system/op_boilerplate.h>
#include <system/op_enums.h>

#if defined(HAVE_VULKAN) && HAVE_VULKAN && defined(HAVE_MLIR) && HAVE_MLIR

// ── MLIR pass infrastructure ──────────────────────────────────────────────────
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>

// ── Dialects we read from / write to ─────────────────────────────────────────
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/Math/IR/Math.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>

namespace sd {
namespace graph {

// ─────────────────────────────────────────────────────────────────────────────
//  Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

namespace {

static constexpr const char* kAccumulatorTypeAttr = "nd4j.accumulator_type";
static constexpr const char* kNdReduceAttr = "nd4j.nd_reduce";

static mlir::Type getElementType(mlir::Type type) {
  auto memref = llvm::dyn_cast<mlir::MemRefType>(type);
  return memref ? memref.getElementType() : mlir::Type{};
}

struct LoweringTypeContract {
  mlir::FloatType accumulatorType;
  llvm::SmallVector<mlir::Type> inputStorageTypes;
  llvm::SmallVector<mlir::Type> outputStorageTypes;
};

static mlir::FailureOr<LoweringTypeContract> getComputeTypeContract(
    mlir::Operation* op, mlir::ValueRange inputs, mlir::ValueRange outputs) {
  auto attr = op->getAttrOfType<mlir::TypeAttr>(kAccumulatorTypeAttr);
  auto accumulator = attr ? llvm::dyn_cast<mlir::FloatType>(attr.getValue())
                          : mlir::FloatType{};
  if (!accumulator) {
    op->emitOpError("requires floating-point nd4j.accumulator_type");
    return mlir::failure();
  }
  LoweringTypeContract contract;
  contract.accumulatorType = accumulator;
  auto collect = [&](mlir::ValueRange values,
                     llvm::SmallVectorImpl<mlir::Type>& storage,
                     llvm::StringRef role) -> mlir::LogicalResult {
    for (auto item : llvm::enumerate(values)) {
      auto memref = llvm::dyn_cast<mlir::MemRefType>(item.value().getType());
      auto type = memref ? llvm::dyn_cast<mlir::FloatType>(memref.getElementType())
                         : mlir::FloatType{};
      if (!type) {
        op->emitOpError() << role << " " << item.index()
                          << " must be a floating-point MemRef";
        return mlir::failure();
      }
      if (type != accumulator &&
          type.getWidth() == accumulator.getWidth()) {
        op->emitOpError() << role << " " << item.index()
                          << " requires an unsupported equal-width floating-point conversion";
        return mlir::failure();
      }
      if (type.getWidth() > accumulator.getWidth()) {
        op->emitOpError() << role << " " << item.index()
                          << " storage type is wider than nd4j.accumulator_type";
        return mlir::failure();
      }
      storage.push_back(type);
    }
    return mlir::success();
  };
  if (mlir::failed(collect(inputs, contract.inputStorageTypes, "input")) ||
      mlir::failed(collect(outputs, contract.outputStorageTypes, "output")))
    return mlir::failure();
  return contract;
}

static mlir::FailureOr<LoweringTypeContract> getMixedOperandTypeContract(
    mlir::Operation* op, mlir::ValueRange inputs, mlir::ValueRange outputs,
    const VulkanKernelEmitterInfo& emitter) {
  const uint32_t scalarMask =
      emitter.operandTypeContract.scalar32InputMask;
  const uint32_t integer32Mask =
      emitter.operandTypeContract.integer32InputMask;
  const uint32_t integer64Mask =
      emitter.operandTypeContract.integer64InputMask;
  const uint32_t integerIndexMask =
      emitter.operandTypeContract.integerIndexInputMask;
  const uint32_t specialMask =
      scalarMask | integer32Mask | integer64Mask | integerIndexMask;
  const uint32_t overlappingRoles =
      (scalarMask & integer32Mask) | (scalarMask & integer64Mask) |
      (scalarMask & integerIndexMask) | (integer32Mask & integer64Mask) |
      (integer32Mask & integerIndexMask) |
      (integer64Mask & integerIndexMask);
  if (overlappingRoles != 0 || inputs.size() > 16) {
    op->emitOpError("invalid mixed operand type contract");
    return mlir::failure();
  }

  llvm::SmallVector<mlir::Value> floatingInputs;
  llvm::SmallVector<unsigned> floatingIndices;
  for (auto item : llvm::enumerate(inputs)) {
    const uint32_t bit = uint32_t{1} << item.index();
    if ((specialMask & bit) == 0) {
      floatingInputs.push_back(item.value());
      floatingIndices.push_back(item.index());
    }
  }
  auto contract = getComputeTypeContract(op, floatingInputs, outputs);
  if (mlir::failed(contract)) return mlir::failure();

  llvm::SmallVector<mlir::Type> floatingStorage =
      std::move(contract->inputStorageTypes);
  contract->inputStorageTypes.assign(inputs.size(), mlir::Type{});
  for (size_t i = 0; i < floatingIndices.size(); ++i) {
    contract->inputStorageTypes[floatingIndices[i]] = floatingStorage[i];
  }

  mlir::Type uniformSpecialType;
  for (auto item : llvm::enumerate(inputs)) {
    const uint32_t bit = uint32_t{1} << item.index();
    if ((specialMask & bit) == 0) continue;
    auto memref = llvm::dyn_cast<mlir::MemRefType>(item.value().getType());
    mlir::Type element = memref ? memref.getElementType() : mlir::Type{};
    auto integerType = llvm::dyn_cast<mlir::IntegerType>(element);
    if ((integer32Mask & bit) != 0) {
      if (!integerType || integerType.getWidth() != 32) {
        op->emitOpError()
            << "input " << item.index()
            << " requires 32-bit integer storage";
        return mlir::failure();
      }
    } else if ((integer64Mask & bit) != 0) {
      if (!integerType || integerType.getWidth() != 64) {
        op->emitOpError()
            << "input " << item.index()
            << " requires 64-bit integer storage";
        return mlir::failure();
      }
    } else if ((integerIndexMask & bit) != 0) {
      if (!integerType ||
          (integerType.getWidth() != 32 && integerType.getWidth() != 64)) {
        op->emitOpError()
            << "input " << item.index()
            << " requires 32-bit or 64-bit integer index storage";
        return mlir::failure();
      }
    } else if (auto floatType = llvm::dyn_cast<mlir::FloatType>(element)) {
      if ((floatType != contract->accumulatorType &&
           floatType.getWidth() == contract->accumulatorType.getWidth()) ||
          floatType.getWidth() > contract->accumulatorType.getWidth()) {
        op->emitOpError()
            << "input " << item.index()
            << " cannot convert to the selected accumulator type";
        return mlir::failure();
      }
    } else if (!integerType || integerType.getWidth() != 32) {
      op->emitOpError()
          << "input " << item.index()
          << " requires floating or 32-bit integer storage";
      return mlir::failure();
    }
    if (emitter.operandTypeContract.requireUniformSpecialInputs) {
      if (!uniformSpecialType) {
        uniformSpecialType = element;
      } else if (uniformSpecialType != element) {
        op->emitOpError("special-role inputs require one uniform storage type");
        return mlir::failure();
      }
    }
    contract->inputStorageTypes[item.index()] = element;
  }
  return contract;
}

static mlir::FailureOr<LoweringTypeContract> getIndexedFloatTypeContract(
    mlir::Operation* op, mlir::ValueRange inputs, mlir::ValueRange outputs) {
  if (inputs.size() != 2 || outputs.size() != 1) {
    op->emitOpError(
        "indexed elementwise lowering requires two inputs and one output");
    return mlir::failure();
  }
  auto contract = getComputeTypeContract(
      op, inputs.slice(1, 1), outputs);
  if (mlir::failed(contract)) return mlir::failure();
  auto dataMemref = llvm::dyn_cast<mlir::MemRefType>(inputs[0].getType());
  mlir::Type dataType =
      dataMemref ? dataMemref.getElementType() : mlir::Type{};
  if (auto floatType = llvm::dyn_cast<mlir::FloatType>(dataType)) {
    if ((floatType != contract->accumulatorType &&
         floatType.getWidth() == contract->accumulatorType.getWidth()) ||
        floatType.getWidth() > contract->accumulatorType.getWidth()) {
      op->emitOpError(
          "indexed data input cannot convert to the selected AccT");
      return mlir::failure();
    }
  } else {
    auto integerType = llvm::dyn_cast<mlir::IntegerType>(dataType);
    if (!integerType || integerType.getWidth() != 32) {
      op->emitOpError(
          "indexed data input requires float or 32-bit integer storage");
      return mlir::failure();
    }
  }
  contract->inputStorageTypes.insert(
      contract->inputStorageTypes.begin(), dataType);
  return contract;
}

static mlir::LogicalResult validateCopyTypes(
    mlir::Operation* op, mlir::ValueRange inputs, mlir::ValueRange outputs,
    bool firstInputOnly = false) {
  if (outputs.empty()) return op->emitOpError("requires an output MemRef");
  auto output = llvm::dyn_cast<mlir::MemRefType>(outputs.front().getType());
  if (!output) return op->emitOpError("output must be a MemRef");
  mlir::Type type = output.getElementType();
  for (auto item : llvm::enumerate(inputs)) {
    auto input = llvm::dyn_cast<mlir::MemRefType>(item.value().getType());
    if (!input) return op->emitOpError("every operand must be a MemRef");
    if ((!firstInputOnly || item.index() == 0) && input.getElementType() != type)
      return op->emitOpError("data input and output element types must match exactly");
  }
  for (mlir::Value value : outputs) {
    auto outputType = llvm::dyn_cast<mlir::MemRefType>(value.getType());
    if (!outputType || outputType.getElementType() != type)
      return op->emitOpError("all output element types must match exactly");
  }
  return mlir::success();
}

static mlir::LogicalResult validateMovementTypes(
    mlir::Operation* op, mlir::ValueRange inputs, mlir::ValueRange outputs,
    const VulkanKernelEmitterInfo& emitter) {
  if (inputs.empty() || outputs.size() != 1) {
    return op->emitOpError(
        "movement type validation requires inputs and one output");
  }
  auto output = llvm::dyn_cast<mlir::MemRefType>(outputs.front().getType());
  if (!output) return op->emitOpError("movement output must be a MemRef");
  const mlir::Type payloadType = output.getElementType();
  mlir::Type uniformSpecialType;

  for (auto item : llvm::enumerate(inputs)) {
    auto input = llvm::dyn_cast<mlir::MemRefType>(item.value().getType());
    if (!input) return op->emitOpError("every movement input must be a MemRef");
    const mlir::Type element = input.getElementType();
    const unsigned index = static_cast<unsigned>(item.index());
    if (index == 0) {
      if (element != payloadType) {
        return op->emitOpError(
            "movement data input and output element types must match exactly");
      }
      continue;
    }
    if (vulkanInputIsStructuralIndex(emitter, index)) continue;

    const bool scalar32 = vulkanInputUsesScalar32Storage(emitter, index);
    const bool integer32 = vulkanInputUsesInteger32Storage(emitter, index);
    const bool integer64 = vulkanInputUsesInteger64Storage(emitter, index);
    const bool integerIndex =
        vulkanInputUsesIntegerIndexStorage(emitter, index);
    const unsigned roleCount =
        static_cast<unsigned>(scalar32) + static_cast<unsigned>(integer32) +
        static_cast<unsigned>(integer64) +
        static_cast<unsigned>(integerIndex);
    if (roleCount > 1) {
      return op->emitOpError("movement input has overlapping storage roles");
    }
    if (roleCount == 0) {
      if (element != payloadType) {
        return op->emitOpError(
            "ordinary movement inputs must use the payload element type");
      }
      continue;
    }

    auto integerType = llvm::dyn_cast<mlir::IntegerType>(element);
    auto floatType = llvm::dyn_cast<mlir::FloatType>(element);
    const bool valid =
        (integer32 && integerType && integerType.getWidth() == 32) ||
        (integer64 && integerType && integerType.getWidth() == 64) ||
        (integerIndex && integerType &&
         (integerType.getWidth() == 32 || integerType.getWidth() == 64)) ||
        (scalar32 &&
         ((integerType && integerType.getWidth() == 32) ||
          (floatType &&
           (floatType.getWidth() == 16 || floatType.getWidth() == 32 ||
            floatType.getWidth() == 64))));
    if (!valid) {
      return op->emitOpError(
          "movement input does not satisfy its declared storage role");
    }
    if (emitter.operandTypeContract.requireUniformSpecialInputs) {
      if (!uniformSpecialType) {
        uniformSpecialType = element;
      } else if (uniformSpecialType != element) {
        return op->emitOpError(
            "special-role movement inputs require one storage type");
      }
    }
  }
  return mlir::success();
}

static mlir::Value convertFloat(mlir::OpBuilder& builder, mlir::Location loc,
                                mlir::Value value, mlir::FloatType destination) {
  auto source = llvm::dyn_cast<mlir::FloatType>(value.getType());
  if (!source || source == destination) return value;
  if (source.getWidth() < destination.getWidth())
    return builder.create<mlir::arith::ExtFOp>(loc, destination, value);
  return builder.create<mlir::arith::TruncFOp>(loc, destination, value);
}

static mlir::Value convertIndexToFloat(mlir::OpBuilder& builder,
                                       mlir::Location loc, mlir::Value value,
                                       mlir::FloatType destination,
                                       bool sourceUnsigned = true) {
  auto integerType = builder.getI32Type();
  mlir::Value integer =
      sourceUnsigned
          ? mlir::Value(builder.create<mlir::arith::IndexCastUIOp>(
                loc, integerType, value))
          : mlir::Value(builder.create<mlir::arith::IndexCastOp>(
                loc, integerType, value));
  return sourceUnsigned
             ? mlir::Value(builder.create<mlir::arith::UIToFPOp>(
                   loc, destination, integer))
             : mlir::Value(builder.create<mlir::arith::SIToFPOp>(
                   loc, destination, integer));
}

static mlir::Value loadAsAccumulator(mlir::OpBuilder& builder, mlir::Location loc,
                                     mlir::Value memref, mlir::ValueRange indices,
                                     mlir::FloatType accumulator) {
  return convertFloat(builder, loc,
      builder.create<mlir::memref::LoadOp>(loc, memref, indices), accumulator);
}

static void storeFromAccumulator(mlir::OpBuilder& builder, mlir::Location loc,
                                 mlir::Value value, mlir::Value memref,
                                 mlir::ValueRange indices) {
  auto storage = llvm::cast<mlir::FloatType>(getElementType(memref.getType()));
  builder.create<mlir::memref::StoreOp>(
      loc, convertFloat(builder, loc, value, storage), memref, indices);
}

static mlir::Value convertScalar(mlir::OpBuilder& builder, mlir::Location loc,
                                 mlir::Value value, mlir::Type destination,
                                 bool sourceUnsigned,
                                 bool destinationUnsigned) {
  if (value.getType() == destination) return value;
  auto sourceFloat = llvm::dyn_cast<mlir::FloatType>(value.getType());
  auto destinationFloat = llvm::dyn_cast<mlir::FloatType>(destination);
  auto sourceInteger = llvm::dyn_cast<mlir::IntegerType>(value.getType());
  auto destinationInteger = llvm::dyn_cast<mlir::IntegerType>(destination);
  if (sourceFloat && destinationFloat) {
    return convertFloat(builder, loc, value, destinationFloat);
  }
  if (sourceInteger && destinationFloat) {
    return sourceUnsigned
               ? mlir::Value(builder.create<mlir::arith::UIToFPOp>(
                     loc, destinationFloat, value))
               : mlir::Value(builder.create<mlir::arith::SIToFPOp>(
                     loc, destinationFloat, value));
  }
  if (sourceFloat && destinationInteger) {
    return destinationUnsigned
               ? mlir::Value(builder.create<mlir::arith::FPToUIOp>(
                     loc, destinationInteger, value))
               : mlir::Value(builder.create<mlir::arith::FPToSIOp>(
                     loc, destinationInteger, value));
  }
  if (sourceInteger && destinationInteger) {
    const unsigned sourceWidth = sourceInteger.getWidth();
    const unsigned destinationWidth = destinationInteger.getWidth();
    if (sourceWidth == destinationWidth) return value;
    if (sourceWidth < destinationWidth) {
      return destinationUnsigned
                 ? mlir::Value(builder.create<mlir::arith::ExtUIOp>(
                       loc, destinationInteger, value))
                 : mlir::Value(builder.create<mlir::arith::ExtSIOp>(
                       loc, destinationInteger, value));
    }
    return builder.create<mlir::arith::TruncIOp>(
        loc, destinationInteger, value);
  }
  return {};
}

static mlir::Value loadAsScalar(mlir::OpBuilder& builder, mlir::Location loc,
                                mlir::Value memref,
                                mlir::ValueRange indices,
                                mlir::Type computeType,
                                bool sourceUnsigned,
                                bool computeUnsigned) {
  mlir::Value value = builder.create<mlir::memref::LoadOp>(
      loc, memref, indices);
  return convertScalar(builder, loc, value, computeType,
                       sourceUnsigned, computeUnsigned);
}

static mlir::Value loadScalarAttribute(mlir::OpBuilder& builder,
                                       mlir::Location loc,
                                       mlir::Operation* op,
                                       mlir::Type computeType) {
  if (auto floatAttr = op->getAttrOfType<mlir::FloatAttr>("nd4j.scalar")) {
    if (auto floatType = llvm::dyn_cast<mlir::FloatType>(computeType)) {
      return builder.create<mlir::arith::ConstantOp>(
          loc, floatType,
          mlir::FloatAttr::get(floatType, floatAttr.getValueAsDouble()));
    }
  }
  if (auto integerAttr =
          op->getAttrOfType<mlir::IntegerAttr>("nd4j.scalar")) {
    if (auto integerType =
            llvm::dyn_cast<mlir::IntegerType>(computeType)) {
      return builder.create<mlir::arith::ConstantOp>(
          loc, integerType,
          mlir::IntegerAttr::get(integerType, integerAttr.getInt()));
    }
  }
  return {};
}

static bool storeScalar(mlir::OpBuilder& builder, mlir::Location loc,
                        mlir::Value value, mlir::Value memref,
                        mlir::ValueRange indices, bool valueUnsigned,
                        bool destinationUnsigned) {
  mlir::Type storage = getElementType(memref.getType());
  mlir::Value converted = convertScalar(
      builder, loc, value, storage, valueUnsigned, destinationUnsigned);
  if (!converted) return false;
  builder.create<mlir::memref::StoreOp>(loc, converted, memref, indices);
  return true;
}

static mlir::scf::ForOp emitReductionLoop(
    mlir::OpBuilder& builder, mlir::Location loc, mlir::Value lower,
    mlir::Value upper, mlir::Value step, mlir::Value initial,
    std::function<mlir::Value(mlir::OpBuilder&, mlir::Location, mlir::Value,
                              mlir::Value)> body) {
  return builder.create<mlir::scf::ForOp>(
      loc, lower, upper, step, mlir::ValueRange{initial},
      [&](mlir::OpBuilder& nested, mlir::Location nestedLoc, mlir::Value index,
          mlir::ValueRange iterArgs) {
        nested.create<mlir::scf::YieldOp>(
            nestedLoc, mlir::SmallVector<mlir::Value>{
                           body(nested, nestedLoc, index, iterArgs.front())});
      });
}
static mlir::Value idxConst(mlir::OpBuilder& b, mlir::Location l, int64_t v) {
  return b.create<mlir::arith::ConstantIndexOp>(l, v);
}

/// Create the single Vulkan entry-point launch used by every catalogue
/// lowering.  GPU kernel outlining turns the body into gpu.func @main; the
/// downstream GPU-to-SPIR-V pipeline then materializes the compute entry point.
/// Keeping this construction shared prevents a lowering from accidentally
/// emitting executable-looking SCF on the host side without a real pipeline.
static mlir::gpu::LaunchOp createGpuLaunch(
    mlir::PatternRewriter& rewriter, mlir::Location loc,
    mlir::Value gridX, mlir::Value gridY, mlir::Value gridZ) {
  mlir::Value one = idxConst(rewriter, loc, 1);
  auto launch = mlir::gpu::LaunchOp::create(
      rewriter, loc, gridX, gridY, gridZ, one, one, one);
  launch.setFunctionAttr(
      mlir::FlatSymbolRefAttr::get(rewriter.getContext(), "main"));
  return launch;
}

/// Convert a row-major logical element number into MemRef coordinates. The
/// resulting multi-dimensional load/store lets the MemRef layout map preserve
/// arbitrary C/F strides and view offsets; it never reinterprets storage as a
/// contiguous flat buffer.
static mlir::SmallVector<mlir::Value> logicalIndices(
    mlir::OpBuilder& builder, mlir::Location loc, mlir::Value linear,
    mlir::Value memref) {
  auto type = llvm::cast<mlir::MemRefType>(memref.getType());
  mlir::SmallVector<mlir::Value> indices(static_cast<size_t>(type.getRank()));
  mlir::Value remaining = linear;
  for (int64_t d = type.getRank() - 1; d >= 0; --d) {
    mlir::Value dim = builder.create<mlir::memref::DimOp>(loc, memref, d);
    indices[static_cast<size_t>(d)] =
        builder.create<mlir::arith::RemUIOp>(loc, remaining, dim);
    remaining = builder.create<mlir::arith::DivUIOp>(loc, remaining, dim);
  }
  return indices;
}

/// Convert a logical element number using the contract's explicit traversal
/// order. MemRef coordinates still preserve the array's physical strides and
/// offset, so this only selects C- versus F-order element sequencing.
static mlir::SmallVector<mlir::Value> logicalIndicesForOrder(
    mlir::OpBuilder& builder, mlir::Location loc, mlir::Value linear,
    mlir::Value memref, bool fortranOrder) {
  if (!fortranOrder) return logicalIndices(builder, loc, linear, memref);
  auto type = llvm::cast<mlir::MemRefType>(memref.getType());
  mlir::SmallVector<mlir::Value> indices(static_cast<size_t>(type.getRank()));
  mlir::Value remaining = linear;
  for (int64_t d = 0; d < type.getRank(); ++d) {
    mlir::Value dim = builder.create<mlir::memref::DimOp>(loc, memref, d);
    indices[static_cast<size_t>(d)] =
        builder.create<mlir::arith::RemUIOp>(loc, remaining, dim);
    remaining = builder.create<mlir::arith::DivUIOp>(loc, remaining, dim);
  }
  return indices;
}

/// Right-align an input shape with the output shape and map singleton input
/// dimensions to zero, matching ND4J/NumPy broadcasting.
static mlir::SmallVector<mlir::Value> broadcastIndices(
    mlir::OpBuilder& builder, mlir::Location loc,
    llvm::ArrayRef<mlir::Value> outputIndices, mlir::Value input) {
  auto type = llvm::cast<mlir::MemRefType>(input.getType());
  const int64_t inputRank = type.getRank();
  const int64_t outputRank = static_cast<int64_t>(outputIndices.size());
  mlir::SmallVector<mlir::Value> indices;
  indices.reserve(static_cast<size_t>(inputRank));
  const mlir::Value zero = idxConst(builder, loc, 0);
  for (int64_t d = 0; d < inputRank; ++d) {
    const int64_t outputDim = outputRank - inputRank + d;
    indices.push_back(type.getDimSize(d) == 1
                          ? zero
                          : outputIndices[static_cast<size_t>(outputDim)]);
  }
  return indices;
}
static mlir::Value floatConst(mlir::OpBuilder& b, mlir::Location l,
                              mlir::FloatType t, double v) {
  return b.create<mlir::arith::ConstantOp>(l, t, b.getFloatAttr(t, v));
}
static mlir::Value emitRsqrt(mlir::OpBuilder& b, mlir::Location l,
                             mlir::Type t, mlir::Value v) {
  auto ft = llvm::cast<mlir::FloatType>(t);
  return b.create<mlir::arith::DivFOp>(
      l, floatConst(b,l,ft,1.0), b.create<mlir::math::SqrtOp>(l,v));
}
static mlir::Value emitExp(mlir::OpBuilder& b, mlir::Location l,
                           mlir::Type, mlir::Value v) {
  // sd::math::sd_exp clamps to [-88, 88] before calling exp; match it so
  // Vulkan agrees with CPU/CUDA/Triton in the overflow/underflow tails.
  auto ft = llvm::cast<mlir::FloatType>(v.getType());
  auto low = b.create<mlir::arith::MaximumFOp>(
      l, v, floatConst(b, l, ft, -88.0));
  auto clamped = b.create<mlir::arith::MinimumFOp>(
      l, low, floatConst(b, l, ft, 88.0));
  return b.create<mlir::math::ExpOp>(l, clamped);
}
static mlir::Value emitTanh(mlir::OpBuilder& b, mlir::Location l,
                            mlir::Type, mlir::Value v) {
  return b.create<mlir::math::TanhOp>(l,v);
}
static mlir::Value emitSigmoid(mlir::OpBuilder& b, mlir::Location l,
                               mlir::Type t, mlir::Value v) {
  auto ft=llvm::cast<mlir::FloatType>(t); auto one=floatConst(b,l,ft,1.0);
  auto d=b.create<mlir::arith::AddFOp>(
      l,one,emitExp(b,l,t,b.create<mlir::arith::NegFOp>(l,v)));
  return b.create<mlir::arith::DivFOp>(l,one,d);
}
static mlir::Value emitRelu(mlir::OpBuilder& b, mlir::Location l,
                            mlir::Type t, mlir::Value v) {
  return b.create<mlir::arith::MaximumFOp>(
      l,v,floatConst(b,l,llvm::cast<mlir::FloatType>(t),0.0));
}
static mlir::Value emitSilu(mlir::OpBuilder& b, mlir::Location l,
                            mlir::Type t, mlir::Value v) {
  return b.create<mlir::arith::MulFOp>(l,v,emitSigmoid(b,l,t,v));
}
static mlir::Value emitGelu(mlir::OpBuilder& b, mlir::Location l,
                            mlir::Type t, mlir::Value v) {
  auto ft=llvm::cast<mlir::FloatType>(t);
  auto v2=b.create<mlir::arith::MulFOp>(l,v,v);
  auto v3=b.create<mlir::arith::MulFOp>(l,v2,v);
  auto inner=b.create<mlir::arith::AddFOp>(
      l,v,b.create<mlir::arith::MulFOp>(l,floatConst(b,l,ft,0.044715),v3));
  auto th=emitTanh(b,l,t,b.create<mlir::arith::MulFOp>(
      l,floatConst(b,l,ft,0.7978845608),inner));
  return b.create<mlir::arith::MulFOp>(
      l,b.create<mlir::arith::MulFOp>(l,floatConst(b,l,ft,0.5),v),
      b.create<mlir::arith::AddFOp>(l,floatConst(b,l,ft,1.0),th));
}
static mlir::Value emitFastGelu(mlir::OpBuilder& b, mlir::Location l,
                                mlir::Type t, mlir::Value v) {
  auto ft = llvm::cast<mlir::FloatType>(t);
  mlir::Value scaled = b.create<mlir::arith::MulFOp>(
      l, v, floatConst(b, l, ft, 1.702));
  return b.create<mlir::arith::MulFOp>(
      l, v, emitSigmoid(b, l, t, scaled));
}
static mlir::Value emitLog(mlir::OpBuilder& b, mlir::Location l,
                           mlir::Type, mlir::Value v) {
  // sd::math::sd_log substitutes SD_EPSILON (1e-5) for zero before calling
  // log; match it so log(0) agrees with CPU/CUDA/Triton. Negative inputs
  // still produce NaN exactly like the native math library.
  auto ft = llvm::cast<mlir::FloatType>(v.getType());
  auto isZero = b.create<mlir::arith::CmpFOp>(
      l, mlir::arith::CmpFPredicate::OEQ, v, floatConst(b, l, ft, 0.0));
  auto guarded = b.create<mlir::arith::SelectOp>(
      l, isZero, floatConst(b, l, ft, 1.0e-5), v);
  return b.create<mlir::math::LogOp>(l, guarded);
}
static mlir::Value emitSqrt(mlir::OpBuilder& b, mlir::Location l,
                            mlir::Type, mlir::Value v) {
  return b.create<mlir::math::SqrtOp>(l,v);
}

static constexpr const char* kOpHashAttr = "nd4j.op_hash";
static constexpr const char* kLegacyFamilyAttr = "nd4j.legacy_family";
static constexpr const char* kLegacyOpNumAttr = "nd4j.legacy_op_num";

static bool readOpHash(mlir::Operation* operation, sd::LongType& hash) {
  auto attr = operation->getAttrOfType<mlir::IntegerAttr>(kOpHashAttr);
  if (!attr) return false;
  hash = static_cast<sd::LongType>(attr.getInt());
  return true;
}

template <typename MlirOp>
static bool readOpHash(MlirOp op, sd::LongType& hash) {
  return readOpHash(op.getOperation(), hash);
}

template <typename MlirOp>
static const VulkanKernelEmitterInfo* emitterForOperation(MlirOp op) {
  mlir::Operation* operation = op.getOperation();
  auto familyAttr =
      operation->getAttrOfType<mlir::IntegerAttr>(kLegacyFamilyAttr);
  auto opNumAttr =
      operation->getAttrOfType<mlir::IntegerAttr>(kLegacyOpNumAttr);
  if (familyAttr || opNumAttr) {
    if (!familyAttr || !opNumAttr) return nullptr;
    const int familyValue = static_cast<int>(familyAttr.getInt());
    if (familyValue < static_cast<int>(VulkanLegacyOpFamily::BROADCAST) ||
        familyValue > static_cast<int>(VulkanLegacyOpFamily::RANDOM)) {
      return nullptr;
    }
    return findVulkanLegacyKernelEmitter(
        static_cast<VulkanLegacyOpFamily>(familyValue),
        static_cast<int>(opNumAttr.getInt()));
  }

  sd::LongType hash = 0;
  return readOpHash(operation, hash) ? findVulkanKernelEmitter(hash) : nullptr;
}

// Resolve a canonical legacy identity into the reusable lowering recipe.  The
// canonical enum names are the source of truth here; do not duplicate the
// numeric values from loops/legacy_ops.h in the lowering layer.
static VulkanKernelRecipe legacySemanticFor(
    mlir::Operation* operation, VulkanKernelRecipe fallback) {
  if (fallback != VulkanKernelRecipe::LEGACY_GENERIC) return fallback;
  auto familyAttr =
      operation->getAttrOfType<mlir::IntegerAttr>(kLegacyFamilyAttr);
  auto opNumAttr =
      operation->getAttrOfType<mlir::IntegerAttr>(kLegacyOpNumAttr);
  if (!familyAttr || !opNumAttr) return VulkanKernelRecipe::UNSUPPORTED;
  const auto family = static_cast<VulkanLegacyOpFamily>(familyAttr.getInt());
  const int opNum = static_cast<int>(opNumAttr.getInt());
  auto binaryArithmetic = [&](int add, int subtract, int multiply, int divide,
                              int reverseDivide, int reverseSubtract,
                              int minimum, int maximum, int mod,
                              int floorDivide, int floorMod,
                              int squaredSubtract, int power, int atan2) {
    if (opNum == add) return VulkanKernelRecipe::ADD;
    if (opNum == subtract) return VulkanKernelRecipe::SUBTRACT;
    if (opNum == multiply) return VulkanKernelRecipe::MULTIPLY;
    if (opNum == divide) return VulkanKernelRecipe::DIVIDE;
    if (opNum == reverseDivide) return VulkanKernelRecipe::REVERSE_DIVIDE;
    if (opNum == reverseSubtract) return VulkanKernelRecipe::REVERSE_SUBTRACT;
    if (opNum == minimum) return VulkanKernelRecipe::MINIMUM;
    if (opNum == maximum) return VulkanKernelRecipe::MAXIMUM;
    if (opNum == mod) return VulkanKernelRecipe::MOD;
    if (opNum == floorDivide) return VulkanKernelRecipe::FLOOR_DIVIDE;
    if (opNum == floorMod) return VulkanKernelRecipe::FLOOR_MOD;
    if (opNum == squaredSubtract) return VulkanKernelRecipe::SQUARED_SUBTRACT;
    if (opNum == power) return VulkanKernelRecipe::POWER;
    if (opNum == atan2) return VulkanKernelRecipe::ATAN2;
    return VulkanKernelRecipe::UNSUPPORTED;
  };
  auto comparison = [&](int equal, int greater, int less, int greaterEqual,
                        int notEqual, int lessEqual) {
    if (opNum == equal) return VulkanKernelRecipe::EQUAL;
    if (opNum == greater) return VulkanKernelRecipe::GREATER;
    if (opNum == less) return VulkanKernelRecipe::LESS;
    if (opNum == greaterEqual) return VulkanKernelRecipe::GREATER_EQUAL;
    if (opNum == notEqual) return VulkanKernelRecipe::NOT_EQUAL;
    if (opNum == lessEqual) return VulkanKernelRecipe::LESS_EQUAL;
    return VulkanKernelRecipe::UNSUPPORTED;
  };
  switch (family) {
    case VulkanLegacyOpFamily::BROADCAST: {
      using namespace sd::broadcast;
      if (opNum == CopyPws) return VulkanKernelRecipe::ASSIGN;
      if (opNum == LogicalNot)
        return VulkanKernelRecipe::LOGICAL_NOT_BINARY;
      if (opNum == Pow) return VulkanKernelRecipe::POWER;
      if (opNum == AMinPairwise) return VulkanKernelRecipe::MINIMUM;
      if (opNum == AMaxPairwise) return VulkanKernelRecipe::MAXIMUM;
      if (opNum == FloorMod) return VulkanKernelRecipe::FLOOR_MOD;
      if (opNum == FloorDiv) return VulkanKernelRecipe::FLOOR_DIVIDE;
      if (opNum == ReverseMod) return VulkanKernelRecipe::REVERSE_MOD;
      if (opNum == SafeDivide) return VulkanKernelRecipe::SAFE_DIVIDE;
      if (opNum == TruncateDiv) return VulkanKernelRecipe::TRUNCATE_DIV;
      if (opNum == LogicalOr) return VulkanKernelRecipe::BOOLEAN_OR;
      if (opNum == LogicalXor) return VulkanKernelRecipe::BOOLEAN_XOR;
      if (opNum == LogicalAnd) return VulkanKernelRecipe::BOOLEAN_AND;
      if (opNum == DivideNoNan) return VulkanKernelRecipe::DIVIDE_NO_NAN;
      if (opNum == IGamma) return VulkanKernelRecipe::IGAMMA;
      if (opNum == IGammac) return VulkanKernelRecipe::IGAMMAC;
      if (opNum == PowDerivative) return VulkanKernelRecipe::POW_DERIVATIVE;
      if (opNum == Xdivy) return VulkanKernelRecipe::XDIVY;
      if (opNum == Xlogy) return VulkanKernelRecipe::XLOGY;
      if (opNum == Xlog1py) return VulkanKernelRecipe::XLOG1PY;
      return binaryArithmetic(Add, Subtract, Multiply, Divide, ReverseDivide,
                              ReverseSubtract, MinPairwise, MaxPairwise, Mod,
                              FloorDiv, FloorMod, SquaredSubtract, Pow, Atan2);
    }
    case VulkanLegacyOpFamily::PAIRWISE: {
      using namespace sd::pairwise;
      if (opNum == CopyPws || opNum == Copy2 || opNum == CompareAndSet ||
          opNum == CompareAndReplace || opNum == ReplaceNans)
        return VulkanKernelRecipe::ASSIGN;
      if (opNum == Axpy) return VulkanKernelRecipe::AXPY;
      if (opNum == LogicalNot)
        return VulkanKernelRecipe::LOGICAL_NOT_BINARY;
      if (opNum == RelativeError)
        return VulkanKernelRecipe::RELATIVE_ERROR;
      if (opNum == BinaryRelativeError)
        return VulkanKernelRecipe::BINARY_RELATIVE_ERROR;
      if (opNum == BinaryMinimumAbsoluteRelativeError)
        return VulkanKernelRecipe::BINARY_MINIMUM_ABSOLUTE_RELATIVE_ERROR;
      if (opNum == LogPoissonLoss)
        return VulkanKernelRecipe::LOG_POISSON_LOSS;
      if (opNum == LogPoissonLossFull)
        return VulkanKernelRecipe::LOG_POISSON_LOSS_FULL;
      if (opNum == Remainder || opNum == Mod || opNum == TruncateMod)
        return VulkanKernelRecipe::MOD;
      if (opNum == FMod) return VulkanKernelRecipe::FMOD;
      if (opNum == TruncateDiv) return VulkanKernelRecipe::TRUNCATE_DIV;
      if (opNum == FloorDiv) return VulkanKernelRecipe::FLOOR_DIVIDE;
      if (opNum == FloorMod) return VulkanKernelRecipe::FLOOR_MOD;
      if (opNum == ReverseMod) return VulkanKernelRecipe::REVERSE_MOD;
      if (opNum == SafeDivide) return VulkanKernelRecipe::SAFE_DIVIDE;
      if (opNum == LogicalOr) return VulkanKernelRecipe::BOOLEAN_OR;
      if (opNum == LogicalXor) return VulkanKernelRecipe::BOOLEAN_XOR;
      if (opNum == LogicalAnd) return VulkanKernelRecipe::BOOLEAN_AND;
      if (opNum == PowDerivative) return VulkanKernelRecipe::POW_DERIVATIVE;
      if (opNum == AMaxPairwise) return VulkanKernelRecipe::MAXIMUM;
      if (opNum == AMinPairwise) return VulkanKernelRecipe::MINIMUM;
      if (opNum == DivideNoNan) return VulkanKernelRecipe::DIVIDE_NO_NAN;
      if (opNum == IGamma) return VulkanKernelRecipe::IGAMMA;
      if (opNum == IGammac) return VulkanKernelRecipe::IGAMMAC;
      if (opNum == Xdivy) return VulkanKernelRecipe::XDIVY;
      if (opNum == Xlogy) return VulkanKernelRecipe::XLOGY;
      if (opNum == Xlog1py) return VulkanKernelRecipe::XLOG1PY;
      return binaryArithmetic(Add, Subtract, Multiply, Divide, ReverseDivide,
                              ReverseSubtract, MinPairwise, MaxPairwise, Mod,
                              FloorDiv, FloorMod, SquaredSubtract, Pow, Atan2);
    }
    case VulkanLegacyOpFamily::SCALAR: {
      using namespace sd::scalar;
      if (opNum == ELU) return VulkanKernelRecipe::ELU_SCALAR;
      if (opNum == ELUDerivative)
        return VulkanKernelRecipe::ELU_DERIVATIVE;
      if (opNum == CopyPws || opNum == CompareAndSet || opNum == ReplaceNans)
        return VulkanKernelRecipe::ASSIGN;
      if (opNum == MinPairwise || opNum == AMinPairwise)
        return VulkanKernelRecipe::MINIMUM;
      if (opNum == MaxPairwise || opNum == AMaxPairwise)
        return VulkanKernelRecipe::MAXIMUM;
      if (opNum == Mod || opNum == Remainder || opNum == TruncateMod)
        return VulkanKernelRecipe::MOD;
      if (opNum == ReverseMod) return VulkanKernelRecipe::REVERSE_MOD;
      if (opNum == FMod) return VulkanKernelRecipe::FMOD;
      if (opNum == TruncateDiv) return VulkanKernelRecipe::TRUNCATE_DIV;
      if (opNum == SafeDivide) return VulkanKernelRecipe::SAFE_DIVIDE;
      if (opNum == LogicalOr) return VulkanKernelRecipe::BOOLEAN_OR;
      if (opNum == LogicalXor) return VulkanKernelRecipe::BOOLEAN_XOR;
      if (opNum == LogicalNot)
        return VulkanKernelRecipe::LOGICAL_NOT_BINARY;
      if (opNum == LogicalAnd) return VulkanKernelRecipe::BOOLEAN_AND;
      if (opNum == PowDerivative) return VulkanKernelRecipe::POW_DERIVATIVE;
      if (opNum == LeakyRELU) return VulkanKernelRecipe::LEAKY_RELU;
      if (opNum == LeakyRELUDerivative)
        return VulkanKernelRecipe::LEAKY_RELU_DERIVATIVE;
      if (opNum == RELU) return VulkanKernelRecipe::RELU;
      if (opNum == RELU6) return VulkanKernelRecipe::RELU6;
      if (opNum == RELUDerivative)
        return VulkanKernelRecipe::RELU_DERIVATIVE;
      if (opNum == SXELogitsSmoother)
        return VulkanKernelRecipe::SIGMOID_CROSS_ENTROPY_SMOOTHER;
      if (opNum == LogX) return VulkanKernelRecipe::LOG_X;
      if (opNum == Step) return VulkanKernelRecipe::STEP;
      if (opNum == LstmClip) return VulkanKernelRecipe::LSTM_CLIP;
      if (opNum == SquaredReverseSubtract)
        return VulkanKernelRecipe::SQUARED_REVERSE_SUBTRACT;
      if (opNum == ReversePow) return VulkanKernelRecipe::REVERSE_POWER;
      if (opNum == DivideNoNan) return VulkanKernelRecipe::DIVIDE_NO_NAN;
      if (opNum == Xdivy) return VulkanKernelRecipe::XDIVY;
      if (opNum == Xlogy) return VulkanKernelRecipe::XLOGY;
      if (opNum == Xlog1py) return VulkanKernelRecipe::XLOG1PY;
      if (opNum == IGamma) return VulkanKernelRecipe::IGAMMA;
      if (opNum == IGammac) return VulkanKernelRecipe::IGAMMAC;
      return binaryArithmetic(Add, Subtract, Multiply, Divide, ReverseDivide,
                              ReverseSubtract, MinPairwise, MaxPairwise, Mod,
                              FloorDiv, FloorMod, SquaredSubtract, Pow, Atan2);
    }
    case VulkanLegacyOpFamily::BROADCAST_BOOL: {
      using namespace sd::broadcast;
      if (opNum == And) return VulkanKernelRecipe::BOOLEAN_AND;
      if (opNum == Or) return VulkanKernelRecipe::BOOLEAN_OR;
      if (opNum == Xor) return VulkanKernelRecipe::BOOLEAN_XOR;
      if (opNum == Epsilon) return VulkanKernelRecipe::EPSILON_COMPARE;
      if (opNum == MatchCondition) return VulkanKernelRecipe::MATCH_CONDITION;
      if (opNum == Not) return VulkanKernelRecipe::LOGICAL_NOT_BINARY;
      return comparison(EqualTo, GreaterThan, LessThan, GreaterThanOrEqual,
                        NotEqualTo, LessThanOrEqual);
    }
    case VulkanLegacyOpFamily::PAIRWISE_BOOL: {
      using namespace sd::pairwise;
      if (opNum == And) return VulkanKernelRecipe::BOOLEAN_AND;
      if (opNum == Or) return VulkanKernelRecipe::BOOLEAN_OR;
      if (opNum == Xor) return VulkanKernelRecipe::BOOLEAN_XOR;
      if (opNum == Epsilon) return VulkanKernelRecipe::EPSILON_COMPARE;
      if (opNum == MatchCondition) return VulkanKernelRecipe::MATCH_CONDITION;
      if (opNum == Not) return VulkanKernelRecipe::LOGICAL_NOT_BINARY;
      return comparison(EqualTo, GreaterThan, LessThan, GreaterThanOrEqual,
                        NotEqualTo, LessThanOrEqual);
    }
    case VulkanLegacyOpFamily::SCALAR_BOOL: {
      using namespace sd::scalar;
      if (opNum == And) return VulkanKernelRecipe::BOOLEAN_AND;
      if (opNum == Or) return VulkanKernelRecipe::BOOLEAN_OR;
      if (opNum == Xor) return VulkanKernelRecipe::BOOLEAN_XOR;
      if (opNum == Epsilon) return VulkanKernelRecipe::EPSILON_COMPARE;
      if (opNum == MatchCondition) return VulkanKernelRecipe::MATCH_CONDITION;
      if (opNum == Not) return VulkanKernelRecipe::LOGICAL_NOT_BINARY;
      return comparison(EqualTo, GreaterThan, LessThan, GreaterThanOrEqual,
                        NotEqualTo, LessThanOrEqual);
    }
    case VulkanLegacyOpFamily::BROADCAST_INT:
    case VulkanLegacyOpFamily::PAIRWISE_INT:
    case VulkanLegacyOpFamily::SCALAR_INT: {
      if (opNum == static_cast<int>(sd::broadcast::ShiftLeft) ||
          opNum == static_cast<int>(sd::pairwise::ShiftLeft) ||
          opNum == static_cast<int>(sd::scalar::ShiftLeft))
        return VulkanKernelRecipe::SHIFT_LEFT;
      if (opNum == static_cast<int>(sd::broadcast::ShiftRight) ||
          opNum == static_cast<int>(sd::pairwise::ShiftRight) ||
          opNum == static_cast<int>(sd::scalar::ShiftRight))
        return VulkanKernelRecipe::SHIFT_RIGHT;
      if (opNum == static_cast<int>(sd::broadcast::CyclicShiftLeft) ||
          opNum == static_cast<int>(sd::pairwise::CyclicShiftLeft) ||
          opNum == static_cast<int>(sd::scalar::CyclicShiftLeft))
        return VulkanKernelRecipe::CYCLIC_SHIFT_LEFT;
      if (opNum == static_cast<int>(sd::broadcast::CyclicShiftRight) ||
          opNum == static_cast<int>(sd::pairwise::CyclicShiftRight) ||
          opNum == static_cast<int>(sd::scalar::CyclicShiftRight))
        return VulkanKernelRecipe::CYCLIC_SHIFT_RIGHT;
      if (opNum == static_cast<int>(sd::broadcast::IntAnd) ||
          opNum == static_cast<int>(sd::pairwise::IntAnd) ||
          opNum == static_cast<int>(sd::scalar::IntAnd))
        return VulkanKernelRecipe::BOOLEAN_AND;
      if (opNum == static_cast<int>(sd::broadcast::IntOr) ||
          opNum == static_cast<int>(sd::pairwise::IntOr) ||
          opNum == static_cast<int>(sd::scalar::IntOr))
        return VulkanKernelRecipe::BOOLEAN_OR;
      if (opNum == static_cast<int>(sd::broadcast::IntXor) ||
          opNum == static_cast<int>(sd::pairwise::IntXor) ||
          opNum == static_cast<int>(sd::scalar::IntXor))
        return VulkanKernelRecipe::BOOLEAN_XOR;
      return VulkanKernelRecipe::UNSUPPORTED;
    }
    case VulkanLegacyOpFamily::TRANSFORM_SAME: {
      using namespace sd::transform;
      if (opNum == Abs) return VulkanKernelRecipe::ABS;
      if (opNum == Sign) return VulkanKernelRecipe::SIGN;
      if (opNum == Ones) return VulkanKernelRecipe::ONES;
      if (opNum == Neg) return VulkanKernelRecipe::NEGATE;
      if (opNum == Round) return VulkanKernelRecipe::ROUND;
      if (opNum == OneMinus) return VulkanKernelRecipe::ONE_MINUS;
      if (opNum == TimesOneMinus)
        return VulkanKernelRecipe::TIMES_ONE_MINUS;
      if (opNum == Cube) return VulkanKernelRecipe::CUBE;
      if (opNum == Reciprocal) return VulkanKernelRecipe::RECIPROCAL;
      if (opNum == Square) return VulkanKernelRecipe::SQUARE;
      if (opNum == CompareAndSetTransform || opNum == Identity || opNum == Copy)
        return VulkanKernelRecipe::ASSIGN;
      if (opNum == Ceiling) return VulkanKernelRecipe::CEIL;
      if (opNum == Floor) return VulkanKernelRecipe::FLOOR;
      if (opNum == ClipByValue) return VulkanKernelRecipe::CLIP_BY_VALUE;
      return VulkanKernelRecipe::UNSUPPORTED;
    }
    case VulkanLegacyOpFamily::TRANSFORM_STRICT: {
      using namespace sd::transform;
      if (opNum == TanhDerivative)
        return VulkanKernelRecipe::TANH_DERIVATIVE;
      if (opNum == HardTanhDerivative)
        return VulkanKernelRecipe::HARD_TANH_DERIVATIVE;
      if (opNum == SigmoidDerivative)
        return VulkanKernelRecipe::SIGMOID_DERIVATIVE;
      if (opNum == SoftSignDerivative)
        return VulkanKernelRecipe::SOFTSIGN_DERIVATIVE;
      if (opNum == TanDerivative)
        return VulkanKernelRecipe::TAN_DERIVATIVE;
      if (opNum == SELUDerivative)
        return VulkanKernelRecipe::SELU_DERIVATIVE;
      if (opNum == HardSigmoidDerivative)
        return VulkanKernelRecipe::HARD_SIGMOID_DERIVATIVE;
      if (opNum == RationalTanhDerivative)
        return VulkanKernelRecipe::RATIONAL_TANH_DERIVATIVE;
      if (opNum == RectifiedTanhDerivative)
        return VulkanKernelRecipe::RECTIFIED_TANH_DERIVATIVE;
      if (opNum == SwishDerivative)
        return VulkanKernelRecipe::SWISH_DERIVATIVE;
      if (opNum == ACoshDerivative)
        return VulkanKernelRecipe::ACOSH_DERIVATIVE;
      if (opNum == ASinhDerivative)
        return VulkanKernelRecipe::ASINH_DERIVATIVE;
      if (opNum == SinhDerivative)
        return VulkanKernelRecipe::SINH_DERIVATIVE;
      if (opNum == LogSigmoidDerivative)
        return VulkanKernelRecipe::LOG_SIGMOID_DERIVATIVE;
      if (opNum == SpecialDerivative)
        return VulkanKernelRecipe::SPECIAL_DERIVATIVE;
      if (opNum == CubeDerivative)
        return VulkanKernelRecipe::CUBE_DERIVATIVE;
      if (opNum == ScaledTanh)
        return VulkanKernelRecipe::SCALED_TANH;
      if (opNum == Affine) return VulkanKernelRecipe::AFFINE;
      if (opNum == SetRange) return VulkanKernelRecipe::SET_RANGE;
      if (opNum == Stabilize) return VulkanKernelRecipe::STABILIZE;
      if (opNum == StabilizeFP16) return VulkanKernelRecipe::STABILIZE_FP16;
      if (opNum == Cosine) return VulkanKernelRecipe::COS;
      if (opNum == Exp) return VulkanKernelRecipe::EXP;
      if (opNum == Log) return VulkanKernelRecipe::LOG;
      if (opNum == Sigmoid) return VulkanKernelRecipe::SIGMOID;
      if (opNum == Sin) return VulkanKernelRecipe::SIN;
      if (opNum == SoftPlus) return VulkanKernelRecipe::SOFTPLUS;
      if (opNum == Tanh) return VulkanKernelRecipe::TANH;
      if (opNum == ACos) return VulkanKernelRecipe::ACOS;
      if (opNum == ASin) return VulkanKernelRecipe::ASIN;
      if (opNum == ATan) return VulkanKernelRecipe::ATAN;
      if (opNum == HardTanh) return VulkanKernelRecipe::HARD_TANH;
      if (opNum == SoftSign) return VulkanKernelRecipe::SOFTSIGN;
      if (opNum == HardSigmoid) return VulkanKernelRecipe::HARD_SIGMOID;
      if (opNum == RationalTanh) return VulkanKernelRecipe::RATIONAL_TANH;
      if (opNum == RectifiedTanh) return VulkanKernelRecipe::RECTIFIED_TANH;
      if (opNum == Sinh) return VulkanKernelRecipe::SINH;
      if (opNum == Cosh) return VulkanKernelRecipe::COSH;
      if (opNum == Tan) return VulkanKernelRecipe::TAN;
      if (opNum == SELU) return VulkanKernelRecipe::SELU;
      if (opNum == Swish) return VulkanKernelRecipe::SILU;
      if (opNum == Log1p) return VulkanKernelRecipe::LOG1P;
      if (opNum == Erf) return VulkanKernelRecipe::ERF;
      if (opNum == ACosh) return VulkanKernelRecipe::ACOSH;
      if (opNum == ASinh) return VulkanKernelRecipe::ASINH;
      if (opNum == Rint) return VulkanKernelRecipe::RINT;
      if (opNum == LogSigmoid) return VulkanKernelRecipe::LOG_SIGMOID;
      if (opNum == Erfc) return VulkanKernelRecipe::ERFC;
      if (opNum == Expm1) return VulkanKernelRecipe::EXPM1;
      if (opNum == ATanh) return VulkanKernelRecipe::ATANH;
      if (opNum == GELU || opNum == PreciseGELU)
        return VulkanKernelRecipe::GELU;
      if (opNum == GELUDerivative)
        return VulkanKernelRecipe::GELU_DERIVATIVE;
      if (opNum == PreciseGELUDerivative)
        return VulkanKernelRecipe::PRECISE_GELU_DERIVATIVE;
      if (opNum == Mish) return VulkanKernelRecipe::MISH;
      if (opNum == MishDerivative)
        return VulkanKernelRecipe::MISH_DERIVATIVE;
      return VulkanKernelRecipe::UNSUPPORTED;
    }
    case VulkanLegacyOpFamily::TRANSFORM_FLOAT:
      if (opNum == sd::transform::Sqrt) return VulkanKernelRecipe::SQRT;
      if (opNum == sd::transform::RSqrt) return VulkanKernelRecipe::RSQRT;
      return VulkanKernelRecipe::UNSUPPORTED;
    case VulkanLegacyOpFamily::TRANSFORM_BOOL:
      if (opNum == sd::transform::IsInf) return VulkanKernelRecipe::IS_INF;
      if (opNum == sd::transform::IsNan) return VulkanKernelRecipe::IS_NAN;
      if (opNum == sd::transform::IsFinite) return VulkanKernelRecipe::IS_FINITE;
      if (opNum == sd::transform::IsInfOrNan) return VulkanKernelRecipe::IS_INF_OR_NAN;
      if (opNum == sd::transform::IsPositive) return VulkanKernelRecipe::IS_POSITIVE;
      if (opNum == sd::transform::Not) return VulkanKernelRecipe::BOOLEAN_NOT;
      if (opNum == sd::transform::IsNegative) return VulkanKernelRecipe::IS_NEGATIVE;
      if (opNum == sd::transform::MatchConditionBool)
        return VulkanKernelRecipe::MATCH_CONDITION_UNARY;
      return VulkanKernelRecipe::UNSUPPORTED;
    case VulkanLegacyOpFamily::TRANSFORM_ANY:
      return VulkanKernelRecipe::ASSIGN;
    case VulkanLegacyOpFamily::REDUCE_SAME:
      if (opNum == sd::reduce::Sum || opNum == sd::reduce::ASum)
        return VulkanKernelRecipe::REDUCE_SUM;
      if (opNum == sd::reduce::Max || opNum == sd::reduce::AMax)
        return VulkanKernelRecipe::REDUCE_MAX;
      if (opNum == sd::reduce::Min || opNum == sd::reduce::AMin)
        return VulkanKernelRecipe::REDUCE_MIN;
      if (opNum == sd::reduce::Prod) return VulkanKernelRecipe::REDUCE_PRODUCT;
      return VulkanKernelRecipe::UNSUPPORTED;
    case VulkanLegacyOpFamily::REDUCE_FLOAT:
      if (opNum == sd::reduce::Mean || opNum == sd::reduce::AMean)
        return VulkanKernelRecipe::REDUCE_SUM;
      if (opNum == sd::reduce::NormMax) return VulkanKernelRecipe::REDUCE_MAX;
      if (opNum == sd::reduce::Norm1 || opNum == sd::reduce::Norm2 ||
          opNum == sd::reduce::NormFrobenius || opNum == sd::reduce::NormP ||
          opNum == sd::reduce::SquaredNorm)
        return VulkanKernelRecipe::REDUCE_SUM;
      if (opNum == sd::reduce::Entropy)
        return VulkanKernelRecipe::REDUCE_ENTROPY;
      if (opNum == sd::reduce::LogEntropy)
        return VulkanKernelRecipe::REDUCE_LOG_ENTROPY;
      if (opNum == sd::reduce::ShannonEntropy)
        return VulkanKernelRecipe::REDUCE_SHANNON_ENTROPY;
      return VulkanKernelRecipe::UNSUPPORTED;
    case VulkanLegacyOpFamily::REDUCE_BOOL:
      if (opNum == sd::reduce::All) return VulkanKernelRecipe::REDUCE_MIN;
      if (opNum == sd::reduce::Any || opNum == sd::reduce::IsFinite ||
          opNum == sd::reduce::IsInfOrNan || opNum == sd::reduce::IsNan ||
          opNum == sd::reduce::IsInf || opNum == sd::reduce::IsPositive ||
          opNum == sd::reduce::IsNegative)
        return VulkanKernelRecipe::REDUCE_MAX;
      return VulkanKernelRecipe::UNSUPPORTED;
    case VulkanLegacyOpFamily::REDUCE_LONG:
      if (opNum == sd::reduce::CountNonZero)
        return VulkanKernelRecipe::REDUCE_COUNT_NONZERO;
      if (opNum == sd::reduce::CountZero)
        return VulkanKernelRecipe::REDUCE_COUNT_ZERO;
      if (opNum == sd::reduce::MatchCondition)
        return VulkanKernelRecipe::REDUCE_COUNT_MATCH;
      return VulkanKernelRecipe::UNSUPPORTED;
    case VulkanLegacyOpFamily::REDUCE3:
      return VulkanKernelRecipe::REDUCE3;
    case VulkanLegacyOpFamily::INDEX_REDUCE:
      if (opNum == sd::indexreduce::IndexMin ||
          opNum == sd::indexreduce::IndexAbsoluteMin)
        return VulkanKernelRecipe::REDUCE_MIN;
      return VulkanKernelRecipe::REDUCE_MAX;
    case VulkanLegacyOpFamily::SUMMARY_STATS:
      if (opNum == sd::variance::SummaryStatsVariance)
        return VulkanKernelRecipe::REDUCE_VARIANCE;
      if (opNum == sd::variance::SummaryStatsStandardDeviation)
        return VulkanKernelRecipe::REDUCE_STDEV;
      return VulkanKernelRecipe::UNSUPPORTED;
    case VulkanLegacyOpFamily::RANDOM:
      return opNum == static_cast<int>(sd::random::UniformDistribution)
                 ? VulkanKernelRecipe::UNIFORM_RANDOM
                 : VulkanKernelRecipe::RANDOM_GENERIC;
  }
  return VulkanKernelRecipe::UNSUPPORTED;
}

template <typename MlirOp>
static VulkanKernelRecipe legacySemanticFor(
    MlirOp op, VulkanKernelRecipe fallback) {
  return legacySemanticFor(op.getOperation(), fallback);
}

using UnaryCallback = std::function<mlir::Value(
    mlir::OpBuilder&, mlir::Location, mlir::Type, mlir::Value)>;
using BinaryCallback = std::function<mlir::Value(
    mlir::OpBuilder&, mlir::Location, mlir::Value, mlir::Value)>;

static mlir::Value emitActivationBackward(
    mlir::OpBuilder& builder, mlir::Location loc,
    mlir::linalg::GenericOp op, VulkanKernelRecipe semantic,
    mlir::FloatType type, mlir::Value x, mlir::Value gradient) {
  mlir::Value zero = floatConst(builder, loc, type, 0.0);
  mlir::Value one = floatConst(builder, loc, type, 1.0);
  auto compare = [&](mlir::arith::CmpFPredicate predicate,
                     mlir::Value lhs, mlir::Value rhs) -> mlir::Value {
    return builder.create<mlir::arith::CmpFOp>(loc, predicate, lhs, rhs);
  };
  auto multiplyGradient = [&](mlir::Value derivative) -> mlir::Value {
    return builder.create<mlir::arith::MulFOp>(
        loc, gradient, derivative);
  };
  auto selectGradient = [&](mlir::Value condition) -> mlir::Value {
    return builder.create<mlir::arith::SelectOp>(
        loc, condition, gradient, zero);
  };
  auto parameter = [&]() -> mlir::Value {
    auto attr = op->getAttrOfType<mlir::FloatAttr>("nd4j.parameter");
    if (!attr || attr.getType() != type) return {};
    return builder.create<mlir::arith::ConstantOp>(loc, type, attr);
  };

  switch (semantic) {
    case VulkanKernelRecipe::RELU_BP:
      return selectGradient(compare(
          mlir::arith::CmpFPredicate::OGT, x, zero));

    case VulkanKernelRecipe::RELU6_BP: {
      mlir::Value aboveZero = compare(
          mlir::arith::CmpFPredicate::OGT, x, zero);
      mlir::Value belowSix = compare(
          mlir::arith::CmpFPredicate::OLT, x,
          floatConst(builder, loc, type, 6.0));
      return selectGradient(builder.create<mlir::arith::AndIOp>(
          loc, aboveZero, belowSix));
    }

    case VulkanKernelRecipe::THRESHOLDED_RELU_BP: {
      mlir::Value threshold = parameter();
      if (!threshold) return {};
      return selectGradient(compare(
          mlir::arith::CmpFPredicate::OGT, x, threshold));
    }

    case VulkanKernelRecipe::SIGMOID_BP: {
      mlir::Value sigmoid = emitSigmoid(builder, loc, type, x);
      mlir::Value derivative = builder.create<mlir::arith::MulFOp>(
          loc, sigmoid,
          builder.create<mlir::arith::SubFOp>(loc, one, sigmoid));
      return multiplyGradient(derivative);
    }

    case VulkanKernelRecipe::TANH_BP: {
      mlir::Value tanh = emitTanh(builder, loc, type, x);
      mlir::Value derivative = builder.create<mlir::arith::SubFOp>(
          loc, one,
          builder.create<mlir::arith::MulFOp>(loc, tanh, tanh));
      return multiplyGradient(derivative);
    }

    case VulkanKernelRecipe::ELU_BP: {
      mlir::Value alpha = parameter();
      if (!alpha) return {};
      mlir::Value negativeDerivative = builder.create<mlir::arith::MulFOp>(
          loc, alpha, emitExp(builder, loc, type, x));
      mlir::Value derivative = builder.create<mlir::arith::SelectOp>(
          loc, compare(mlir::arith::CmpFPredicate::OGE, x, zero),
          one, negativeDerivative);
      return multiplyGradient(derivative);
    }

    case VulkanKernelRecipe::SELU_BP: {
      constexpr double kLambda = 1.0507009873554805;
      constexpr double kAlpha = 1.6732632423543772;
      mlir::Value positiveDerivative =
          floatConst(builder, loc, type, kLambda);
      mlir::Value negativeDerivative = builder.create<mlir::arith::MulFOp>(
          loc, floatConst(builder, loc, type, kAlpha * kLambda),
          emitExp(builder, loc, type, x));
      mlir::Value derivative = builder.create<mlir::arith::SelectOp>(
          loc, compare(mlir::arith::CmpFPredicate::OGT, x, zero),
          positiveDerivative, negativeDerivative);
      return multiplyGradient(derivative);
    }

    case VulkanKernelRecipe::LEAKY_RELU_BP: {
      mlir::Value alpha = parameter();
      if (!alpha) return {};
      mlir::Value negative = builder.create<mlir::arith::MulFOp>(
          loc, alpha, gradient);
      return builder.create<mlir::arith::SelectOp>(
          loc, compare(mlir::arith::CmpFPredicate::OGE, x, zero),
          gradient, negative);
    }

    case VulkanKernelRecipe::SOFTPLUS_BP:
      return multiplyGradient(emitSigmoid(builder, loc, type, x));

    case VulkanKernelRecipe::SOFTSIGN_BP: {
      mlir::Value denominator = builder.create<mlir::arith::AddFOp>(
          loc, one, builder.create<mlir::math::AbsFOp>(loc, x));
      return builder.create<mlir::arith::DivFOp>(
          loc, gradient,
          builder.create<mlir::arith::MulFOp>(
              loc, denominator, denominator));
    }

    case VulkanKernelRecipe::HARD_SIGMOID_BP: {
      mlir::Value aboveLower = compare(
          mlir::arith::CmpFPredicate::OGE, x,
          floatConst(builder, loc, type, -2.5));
      mlir::Value belowUpper = compare(
          mlir::arith::CmpFPredicate::OLE, x,
          floatConst(builder, loc, type, 2.5));
      mlir::Value derivative = builder.create<mlir::arith::SelectOp>(
          loc, builder.create<mlir::arith::AndIOp>(
                   loc, aboveLower, belowUpper),
          floatConst(builder, loc, type, 0.2), zero);
      return multiplyGradient(derivative);
    }

    case VulkanKernelRecipe::HARD_TANH_BP: {
      mlir::Value aboveLower = compare(
          mlir::arith::CmpFPredicate::OGE, x,
          floatConst(builder, loc, type, -1.0));
      mlir::Value belowUpper = compare(
          mlir::arith::CmpFPredicate::OLE, x,
          floatConst(builder, loc, type, 1.0));
      return selectGradient(builder.create<mlir::arith::AndIOp>(
          loc, aboveLower, belowUpper));
    }

    case VulkanKernelRecipe::SILU_BP: {
      mlir::Value sigmoid = emitSigmoid(builder, loc, type, x);
      mlir::Value derivative = builder.create<mlir::arith::AddFOp>(
          loc, sigmoid,
          builder.create<mlir::arith::MulFOp>(
              loc, x,
              builder.create<mlir::arith::MulFOp>(
                  loc, sigmoid,
                  builder.create<mlir::arith::SubFOp>(
                      loc, one, sigmoid))));
      return multiplyGradient(derivative);
    }

    case VulkanKernelRecipe::FUSED_GELU_BP: {
      mlir::Value scaled = builder.create<mlir::arith::MulFOp>(
          loc, floatConst(builder, loc, type, 1.702), x);
      mlir::Value sigmoid = emitSigmoid(builder, loc, type, scaled);
      mlir::Value derivative = builder.create<mlir::arith::AddFOp>(
          loc, sigmoid,
          builder.create<mlir::arith::MulFOp>(
              loc, scaled,
              builder.create<mlir::arith::MulFOp>(
                  loc, sigmoid,
                  builder.create<mlir::arith::SubFOp>(
                      loc, one, sigmoid))));
      return multiplyGradient(derivative);
    }

    case VulkanKernelRecipe::SQUARED_RELU_BP: {
      mlir::Value derivative = builder.create<mlir::arith::MulFOp>(
          loc, floatConst(builder, loc, type, 2.0), x);
      mlir::Value active = compare(
          mlir::arith::CmpFPredicate::OGT, x, zero);
      return builder.create<mlir::arith::SelectOp>(
          loc, active, multiplyGradient(derivative), zero);
    }

    case VulkanKernelRecipe::RECTIFIED_TANH_BP: {
      mlir::Value tanh = emitTanh(builder, loc, type, x);
      mlir::Value derivative = builder.create<mlir::arith::SubFOp>(
          loc, one,
          builder.create<mlir::arith::MulFOp>(loc, tanh, tanh));
      mlir::Value active = compare(
          mlir::arith::CmpFPredicate::OGT, x, zero);
      return builder.create<mlir::arith::SelectOp>(
          loc, active, multiplyGradient(derivative), zero);
    }

    default:
      return {};
  }
}

static mlir::Value scalarConstant(mlir::OpBuilder& builder, mlir::Location loc,
                                  mlir::Type type, double value) {
  if (auto floatType = llvm::dyn_cast<mlir::FloatType>(type)) {
    return floatConst(builder, loc, floatType, value);
  }
  if (auto integerType = llvm::dyn_cast<mlir::IntegerType>(type)) {
    return builder.create<mlir::arith::ConstantIntOp>(
        loc, static_cast<int64_t>(value), integerType.getWidth());
  }
  return {};
}

static mlir::Value emitClipWithValues(
    mlir::OpBuilder& builder, mlir::Location loc, mlir::Type type,
    mlir::Value x, mlir::Value lower, mlir::Value upper, bool isUnsigned) {
  if (llvm::isa<mlir::FloatType>(type)) {
    mlir::Value lowerBounded =
        builder.create<mlir::arith::MaximumFOp>(loc, x, lower);
    return builder.create<mlir::arith::MinimumFOp>(
        loc, lowerBounded, upper);
  }
  auto integerType = llvm::dyn_cast<mlir::IntegerType>(type);
  if (!integerType || integerType.getWidth() != 32) return {};
  const auto greater = isUnsigned ? mlir::arith::CmpIPredicate::ugt
                                  : mlir::arith::CmpIPredicate::sgt;
  const auto less = isUnsigned ? mlir::arith::CmpIPredicate::ult
                               : mlir::arith::CmpIPredicate::slt;
  mlir::Value aboveLower =
      builder.create<mlir::arith::CmpIOp>(loc, greater, x, lower);
  mlir::Value lowerBounded =
      builder.create<mlir::arith::SelectOp>(loc, aboveLower, x, lower);
  mlir::Value belowUpper =
      builder.create<mlir::arith::CmpIOp>(loc, less, lowerBounded, upper);
  return builder.create<mlir::arith::SelectOp>(
      loc, belowUpper, lowerBounded, upper);
}

static mlir::Value emitParameterizedUnary(
    mlir::OpBuilder& builder, mlir::Location loc, mlir::Type type,
    VulkanKernelRecipe semantic, mlir::Value x, double scalar0,
    double scalar1, bool isUnsigned) {
  mlir::Value first = scalarConstant(builder, loc, type, scalar0);
  if (!first) return {};

  if (auto floatType = llvm::dyn_cast<mlir::FloatType>(type)) {
    mlir::Value zero = floatConst(builder, loc, floatType, 0.0);
    switch (semantic) {
      case VulkanKernelRecipe::RELU:
        return builder.create<mlir::arith::MaximumFOp>(loc, x, first);
      case VulkanKernelRecipe::THRESHOLDED_RELU: {
        mlir::Value passes = builder.create<mlir::arith::CmpFOp>(
            loc, mlir::arith::CmpFPredicate::OGT, x, first);
        return builder.create<mlir::arith::SelectOp>(loc, passes, x, zero);
      }
      case VulkanKernelRecipe::RELU6: {
        mlir::Value relu =
            builder.create<mlir::arith::MaximumFOp>(loc, x, first);
        return builder.create<mlir::arith::MinimumFOp>(
            loc, relu, floatConst(builder, loc, floatType, 6.0));
      }
      case VulkanKernelRecipe::LEAKY_RELU: {
        mlir::Value negative = builder.create<mlir::arith::MulFOp>(
            loc, x, first);
        mlir::Value isNegative = builder.create<mlir::arith::CmpFOp>(
            loc, mlir::arith::CmpFPredicate::OLT, x, zero);
        return builder.create<mlir::arith::SelectOp>(
            loc, isNegative, negative, x);
      }
      case VulkanKernelRecipe::ELU: {
        mlir::Value expMinusOne = builder.create<mlir::arith::SubFOp>(
            loc, emitExp(builder, loc, type, x),
            floatConst(builder, loc, floatType, 1.0));
        mlir::Value negative = builder.create<mlir::arith::MulFOp>(
            loc, first, expMinusOne);
        mlir::Value nonNegative = builder.create<mlir::arith::CmpFOp>(
            loc, mlir::arith::CmpFPredicate::OGE, x, zero);
        return builder.create<mlir::arith::SelectOp>(
            loc, nonNegative, x, negative);
      }
      case VulkanKernelRecipe::SCALE:
        return builder.create<mlir::arith::MulFOp>(loc, x, first);
      case VulkanKernelRecipe::CLIP_BY_VALUE: {
        mlir::Value second = scalarConstant(builder, loc, type, scalar1);
        if (!second) return {};
        mlir::Value bounded =
            builder.create<mlir::arith::MaximumFOp>(loc, x, first);
        return builder.create<mlir::arith::MinimumFOp>(
            loc, bounded, second);
      }
      case VulkanKernelRecipe::SCALED_TANH: {
        mlir::Value scale = scalarConstant(builder, loc, type, scalar0);
        mlir::Value slope = scalarConstant(builder, loc, type, scalar1);
        if (!scale || !slope) return {};
        return builder.create<mlir::arith::MulFOp>(
            loc, scale,
            builder.create<mlir::math::TanhOp>(
                loc, builder.create<mlir::arith::MulFOp>(loc, slope, x)));
      }
      case VulkanKernelRecipe::AFFINE: {
        mlir::Value offset = scalarConstant(builder, loc, type, scalar1);
        if (!offset) return {};
        return builder.create<mlir::arith::AddFOp>(
            loc, builder.create<mlir::arith::MulFOp>(loc, first, x), offset);
      }
      case VulkanKernelRecipe::SET_RANGE: {
        mlir::Value upper = scalarConstant(builder, loc, type, scalar1);
        if (!upper) return {};
        mlir::Value atLeast = builder.create<mlir::arith::CmpFOp>(
            loc, mlir::arith::CmpFPredicate::OGE, x, first);
        mlir::Value atMost = builder.create<mlir::arith::CmpFOp>(
            loc, mlir::arith::CmpFPredicate::OLE, x, upper);
        mlir::Value inside = builder.create<mlir::arith::AndIOp>(
            loc, atLeast, atMost);
        mlir::Value span = builder.create<mlir::arith::SubFOp>(loc, upper, first);
        mlir::Value scaled;
        if (scalar0 == 0.0 && scalar1 == 1.0) {
          scaled = builder.create<mlir::arith::MulFOp>(
              loc, emitSigmoid(builder, loc, type, x), span);
        } else {
          scaled = builder.create<mlir::arith::MulFOp>(loc, x, span);
        }
        mlir::Value mapped = builder.create<mlir::arith::AddFOp>(
            loc, builder.create<mlir::math::FloorOp>(loc, scaled), first);
        return builder.create<mlir::arith::SelectOp>(loc, inside, x, mapped);
      }
      case VulkanKernelRecipe::STABILIZE: {
        mlir::Value product = builder.create<mlir::arith::MulFOp>(loc, x, first);
        mlir::Value max = floatConst(builder, loc, floatType, 3.79297773665);
        mlir::Value min = floatConst(builder, loc, floatType, -3.79297773665);
        mlir::Value upper = builder.create<mlir::arith::DivFOp>(loc, max, first);
        mlir::Value lower = builder.create<mlir::arith::DivFOp>(loc, min, first);
        mlir::Value above = builder.create<mlir::arith::CmpFOp>(
            loc, mlir::arith::CmpFPredicate::OGT, product, max);
        mlir::Value below = builder.create<mlir::arith::CmpFOp>(
            loc, mlir::arith::CmpFPredicate::OLT, product, min);
        mlir::Value bounded = builder.create<mlir::arith::SelectOp>(
            loc, above, upper,
            builder.create<mlir::arith::SelectOp>(loc, below, lower, x));
        mlir::Value zeroK = builder.create<mlir::arith::CmpFOp>(
            loc, mlir::arith::CmpFPredicate::OEQ, first, zero);
        return builder.create<mlir::arith::SelectOp>(loc, zeroK, x, bounded);
      }
      default:
        return {};
    }
  }

  auto integerType = llvm::dyn_cast<mlir::IntegerType>(type);
  if (!integerType || integerType.getWidth() != 32) return {};
  const auto greater = isUnsigned ? mlir::arith::CmpIPredicate::ugt
                                  : mlir::arith::CmpIPredicate::sgt;
  const auto less = isUnsigned ? mlir::arith::CmpIPredicate::ult
                               : mlir::arith::CmpIPredicate::slt;
  switch (semantic) {
    case VulkanKernelRecipe::RELU: {
      mlir::Value useX =
          builder.create<mlir::arith::CmpIOp>(loc, greater, x, first);
      return builder.create<mlir::arith::SelectOp>(loc, useX, x, first);
    }
    case VulkanKernelRecipe::THRESHOLDED_RELU: {
      mlir::Value zero = scalarConstant(builder, loc, type, 0.0);
      mlir::Value passes =
          builder.create<mlir::arith::CmpIOp>(loc, greater, x, first);
      return builder.create<mlir::arith::SelectOp>(loc, passes, x, zero);
    }
    case VulkanKernelRecipe::RELU6: {
      mlir::Value useX =
          builder.create<mlir::arith::CmpIOp>(loc, greater, x, first);
      mlir::Value relu =
          builder.create<mlir::arith::SelectOp>(loc, useX, x, first);
      mlir::Value six = scalarConstant(builder, loc, type, 6.0);
      mlir::Value belowSix =
          builder.create<mlir::arith::CmpIOp>(loc, less, relu, six);
      return builder.create<mlir::arith::SelectOp>(
          loc, belowSix, relu, six);
    }
    case VulkanKernelRecipe::CLIP_BY_VALUE: {
      mlir::Value second = scalarConstant(builder, loc, type, scalar1);
      if (!second) return {};
      mlir::Value aboveLower =
          builder.create<mlir::arith::CmpIOp>(loc, greater, x, first);
      mlir::Value lowerBounded =
          builder.create<mlir::arith::SelectOp>(loc, aboveLower, x, first);
      mlir::Value belowUpper =
          builder.create<mlir::arith::CmpIOp>(loc, less, lowerBounded, second);
      return builder.create<mlir::arith::SelectOp>(
          loc, belowUpper, lowerBounded, second);
    }
    default:
      return {};
  }
}

static mlir::Value emitParameterizedBinary(
    mlir::OpBuilder& builder, mlir::Location loc, mlir::Type type,
    VulkanKernelRecipe semantic, mlir::Value a, mlir::Value b,
    double scalar0, double scalar1, bool scalar0Present) {
  auto floatType = llvm::dyn_cast<mlir::FloatType>(type);
  if (!floatType) return {};
  mlir::Value zero = floatConst(builder, loc, floatType, 0.0);
  mlir::Value one = floatConst(builder, loc, floatType, 1.0);
  mlir::Value difference = builder.create<mlir::math::AbsFOp>(
      loc, builder.create<mlir::arith::SubFOp>(loc, a, b));
  mlir::Value denominator = builder.create<mlir::arith::AddFOp>(
      loc, builder.create<mlir::math::AbsFOp>(loc, a),
      builder.create<mlir::math::AbsFOp>(loc, b));
  mlir::Value bothZero = builder.create<mlir::arith::AndIOp>(
      loc, builder.create<mlir::arith::CmpFOp>(
               loc, mlir::arith::CmpFPredicate::OEQ, a, zero),
      builder.create<mlir::arith::CmpFOp>(
          loc, mlir::arith::CmpFPredicate::OEQ, b, zero));
  mlir::Value relative = builder.create<mlir::arith::SelectOp>(
      loc, bothZero, zero,
      builder.create<mlir::arith::DivFOp>(loc, difference, denominator));
  if (semantic == VulkanKernelRecipe::AXPY) {
    mlir::Value alpha = scalarConstant(
        builder, loc, type, scalar0Present ? scalar0 : 1.0);
    if (!alpha) return {};
    return builder.create<mlir::arith::AddFOp>(
        loc, builder.create<mlir::arith::MulFOp>(loc, alpha, a), b);
  }
  if (semantic == VulkanKernelRecipe::BINARY_RELATIVE_ERROR) {
    mlir::Value threshold = scalarConstant(builder, loc, type, scalar0);
    if (!threshold) return {};
    mlir::Value exceeds = builder.create<mlir::arith::CmpFOp>(
        loc, mlir::arith::CmpFPredicate::OGT, relative, threshold);
    return builder.create<mlir::arith::SelectOp>(loc, exceeds, one, zero);
  }
  if (semantic == VulkanKernelRecipe::BINARY_MINIMUM_ABSOLUTE_RELATIVE_ERROR) {
    mlir::Value thresholdRelative = scalarConstant(builder, loc, type, scalar0);
    mlir::Value thresholdAbsolute = scalarConstant(builder, loc, type, scalar1);
    if (!thresholdRelative || !thresholdAbsolute) return {};
    mlir::Value relativeExceeds = builder.create<mlir::arith::CmpFOp>(
        loc, mlir::arith::CmpFPredicate::OGT, relative, thresholdRelative);
    mlir::Value absoluteBelow = builder.create<mlir::arith::CmpFOp>(
        loc, mlir::arith::CmpFPredicate::OLT, difference, thresholdAbsolute);
    mlir::Value zeroOrOne = builder.create<mlir::arith::SelectOp>(
        loc, absoluteBelow, zero, one);
    return builder.create<mlir::arith::SelectOp>(
        loc, relativeExceeds, zeroOrOne, zero);
  }
  return {};
}

static UnaryCallback unaryCallbackFor(VulkanKernelRecipe semantic) {
  switch (semantic) {
    case VulkanKernelRecipe::BOOLEAN_NOT:
      return UnaryCallback{
          [](mlir::OpBuilder& b, mlir::Location loc, mlir::Type,
             mlir::Value x) {
            auto integerType = llvm::cast<mlir::IntegerType>(x.getType());
            mlir::Value one = b.create<mlir::arith::ConstantIntOp>(
                loc, 1, integerType.getWidth());
            // BOOL values are normalized to 0/1 before reaching this callback.
            return b.create<mlir::arith::XOrIOp>(loc, x, one);
          }};
    case VulkanKernelRecipe::MATCH_CONDITION_UNARY:
      return UnaryCallback{[](mlir::OpBuilder& b, mlir::Location loc,
                              mlir::Type ty, mlir::Value x) {
        auto ft = llvm::cast<mlir::FloatType>(ty);
        return b.create<mlir::arith::CmpFOp>(
            loc, mlir::arith::CmpFPredicate::UNE, x,
            floatConst(b, loc, ft, 0.0));
      }};
    case VulkanKernelRecipe::IS_INF:
      return UnaryCallback{[](mlir::OpBuilder& b, mlir::Location loc,
                              mlir::Type ty, mlir::Value x) {
        auto ft = llvm::cast<mlir::FloatType>(ty);
        auto magnitude = b.create<mlir::math::AbsFOp>(loc, x);
        return b.create<mlir::arith::CmpFOp>(
            loc, mlir::arith::CmpFPredicate::OEQ, magnitude,
            floatConst(b, loc, ft, std::numeric_limits<double>::infinity()));
      }};
    case VulkanKernelRecipe::IS_NAN:
      return UnaryCallback{[](mlir::OpBuilder& b, mlir::Location loc,
                              mlir::Type, mlir::Value x) {
        return b.create<mlir::arith::CmpFOp>(
            loc, mlir::arith::CmpFPredicate::UNO, x, x);
      }};
    case VulkanKernelRecipe::IS_FINITE:
      return UnaryCallback{[](mlir::OpBuilder& b, mlir::Location loc,
                              mlir::Type ty, mlir::Value x) {
        auto ft = llvm::cast<mlir::FloatType>(ty);
        auto magnitude = b.create<mlir::math::AbsFOp>(loc, x);
        return b.create<mlir::arith::CmpFOp>(
            loc, mlir::arith::CmpFPredicate::ONE, magnitude,
            floatConst(b, loc, ft, std::numeric_limits<double>::infinity()));
      }};
    case VulkanKernelRecipe::IS_INF_OR_NAN:
      return UnaryCallback{[](mlir::OpBuilder& b, mlir::Location loc,
                              mlir::Type ty, mlir::Value x) {
        auto ft = llvm::cast<mlir::FloatType>(ty);
        auto magnitude = b.create<mlir::math::AbsFOp>(loc, x);
        auto infinite = b.create<mlir::arith::CmpFOp>(
            loc, mlir::arith::CmpFPredicate::OEQ, magnitude,
            floatConst(b, loc, ft, std::numeric_limits<double>::infinity()));
        auto nan = b.create<mlir::arith::CmpFOp>(
            loc, mlir::arith::CmpFPredicate::UNO, x, x);
        return b.create<mlir::arith::OrIOp>(loc, infinite, nan);
      }};
    case VulkanKernelRecipe::IS_POSITIVE:
    case VulkanKernelRecipe::IS_NEGATIVE:
      return UnaryCallback{[semantic](mlir::OpBuilder& b, mlir::Location loc,
                                      mlir::Type ty, mlir::Value x) {
        auto ft = llvm::cast<mlir::FloatType>(ty);
        return b.create<mlir::arith::CmpFOp>(
            loc, semantic == VulkanKernelRecipe::IS_POSITIVE
                     ? mlir::arith::CmpFPredicate::OGT
                     : mlir::arith::CmpFPredicate::OLT,
            x, floatConst(b, loc, ft, 0.0));
      }};
    case VulkanKernelRecipe::SIGN:
      return UnaryCallback{[](mlir::OpBuilder& b, mlir::Location loc,
                              mlir::Type ty, mlir::Value x) {
        auto ft = llvm::cast<mlir::FloatType>(ty);
        auto zero = floatConst(b, loc, ft, 0.0);
        auto positive = b.create<mlir::arith::CmpFOp>(
            loc, mlir::arith::CmpFPredicate::OGT, x, zero);
        auto negative = b.create<mlir::arith::CmpFOp>(
            loc, mlir::arith::CmpFPredicate::OLT, x, zero);
        return b.create<mlir::arith::SelectOp>(
            loc, positive, floatConst(b, loc, ft, 1.0),
            b.create<mlir::arith::SelectOp>(
                loc, negative, floatConst(b, loc, ft, -1.0), zero));
      }};
    case VulkanKernelRecipe::ONES:
      return UnaryCallback{[](mlir::OpBuilder& b, mlir::Location loc,
                              mlir::Type ty, mlir::Value) {
        return floatConst(b, loc, llvm::cast<mlir::FloatType>(ty), 1.0);
      }};
    case VulkanKernelRecipe::ROUND:
      return UnaryCallback{[](mlir::OpBuilder& b, mlir::Location loc,
                              mlir::Type, mlir::Value x) {
        return b.create<mlir::math::RoundEvenOp>(loc, x);
      }};
    case VulkanKernelRecipe::ONE_MINUS:
      return UnaryCallback{[](mlir::OpBuilder& b, mlir::Location loc,
                              mlir::Type ty, mlir::Value x) {
        auto ft = llvm::cast<mlir::FloatType>(ty);
        return b.create<mlir::arith::SubFOp>(loc, floatConst(b, loc, ft, 1.0), x);
      }};
    case VulkanKernelRecipe::TIMES_ONE_MINUS:
      return UnaryCallback{[](mlir::OpBuilder& b, mlir::Location loc,
                              mlir::Type ty, mlir::Value x) {
        auto ft = llvm::cast<mlir::FloatType>(ty);
        auto oneMinus = b.create<mlir::arith::SubFOp>(
            loc, floatConst(b, loc, ft, 1.0), x);
        return b.create<mlir::arith::MulFOp>(loc, x, oneMinus);
      }};
    case VulkanKernelRecipe::RECIPROCAL:
      return UnaryCallback{[](mlir::OpBuilder& b, mlir::Location loc,
                              mlir::Type ty, mlir::Value x) {
        return b.create<mlir::arith::DivFOp>(
            loc, floatConst(b, loc, llvm::cast<mlir::FloatType>(ty), 1.0), x);
      }};
    case VulkanKernelRecipe::CEIL:
      return UnaryCallback{[](mlir::OpBuilder& b, mlir::Location loc,
                              mlir::Type, mlir::Value x) {
        return b.create<mlir::math::CeilOp>(loc, x);
      }};
    case VulkanKernelRecipe::COS:
      return UnaryCallback{[](mlir::OpBuilder& b, mlir::Location loc,
                              mlir::Type, mlir::Value x) {
        return b.create<mlir::math::CosOp>(loc, x);
      }};
    case VulkanKernelRecipe::SIN:
      return UnaryCallback{[](mlir::OpBuilder& b, mlir::Location loc,
                              mlir::Type, mlir::Value x) {
        return b.create<mlir::math::SinOp>(loc, x);
      }};
    case VulkanKernelRecipe::EXP:
      return UnaryCallback{emitExp};
    case VulkanKernelRecipe::LOG:
      return UnaryCallback{emitLog};
    case VulkanKernelRecipe::RSQRT:
      return UnaryCallback{emitRsqrt};
    case VulkanKernelRecipe::TAN:
      return UnaryCallback{[](mlir::OpBuilder& b, mlir::Location loc,
                              mlir::Type ty, mlir::Value x) {
        auto ft = llvm::cast<mlir::FloatType>(ty);
        return b.create<mlir::arith::DivFOp>(
            loc, b.create<mlir::math::SinOp>(loc, x),
            b.create<mlir::math::CosOp>(loc, x));
      }};
    case VulkanKernelRecipe::SINH:
      return UnaryCallback{[](mlir::OpBuilder& b, mlir::Location loc,
                              mlir::Type ty, mlir::Value x) {
        auto ft = llvm::cast<mlir::FloatType>(ty);
        auto ex = emitExp(b, loc, ty, x);
        auto enx = emitExp(b, loc, ty, b.create<mlir::arith::NegFOp>(loc, x));
        return b.create<mlir::arith::MulFOp>(
            loc, b.create<mlir::arith::SubFOp>(loc, ex, enx),
            floatConst(b, loc, ft, 0.5));
      }};
    case VulkanKernelRecipe::COSH:
      return UnaryCallback{[](mlir::OpBuilder& b, mlir::Location loc,
                              mlir::Type ty, mlir::Value x) {
        auto ft = llvm::cast<mlir::FloatType>(ty);
        auto ex = emitExp(b, loc, ty, x);
        auto enx = emitExp(b, loc, ty, b.create<mlir::arith::NegFOp>(loc, x));
        return b.create<mlir::arith::MulFOp>(
            loc, b.create<mlir::arith::AddFOp>(loc, ex, enx),
            floatConst(b, loc, ft, 0.5));
      }};
    case VulkanKernelRecipe::EXPM1:
      return UnaryCallback{[](mlir::OpBuilder& b, mlir::Location loc,
                              mlir::Type ty, mlir::Value x) {
        return b.create<mlir::arith::SubFOp>(
            loc, emitExp(b, loc, ty, x),
            floatConst(b, loc, llvm::cast<mlir::FloatType>(ty), 1.0));
      }};
    case VulkanKernelRecipe::LOG_SIGMOID:
      return UnaryCallback{[](mlir::OpBuilder& b, mlir::Location loc,
                              mlir::Type ty, mlir::Value x) {
        auto ft = llvm::cast<mlir::FloatType>(ty);
        auto negative = b.create<mlir::arith::NegFOp>(loc, x);
        auto softplus = b.create<mlir::arith::AddFOp>(
            loc, floatConst(b, loc, ft, 1.0), emitExp(b, loc, ty, negative));
        return b.create<mlir::arith::NegFOp>(loc, emitLog(b, loc, ty, softplus));
      }};
    case VulkanKernelRecipe::MISH:
      return UnaryCallback{[](mlir::OpBuilder& b, mlir::Location loc,
                              mlir::Type ty, mlir::Value x) {
        auto ft = llvm::cast<mlir::FloatType>(ty);
        auto softplus = emitLog(
            b, loc, ty,
            b.create<mlir::arith::AddFOp>(loc, floatConst(b, loc, ft, 1.0),
                                          emitExp(b, loc, ty, x)));
        return b.create<mlir::arith::MulFOp>(loc, x, emitTanh(b, loc, ty, softplus));
      }};
    case VulkanKernelRecipe::ATAN:
      return UnaryCallback{[](mlir::OpBuilder& b, mlir::Location loc,
                              mlir::Type, mlir::Value x) {
        return b.create<mlir::math::AtanOp>(loc, x);
      }};
    case VulkanKernelRecipe::ASIN:
      return UnaryCallback{[](mlir::OpBuilder& b, mlir::Location loc,
                              mlir::Type ty, mlir::Value x) {
        auto ft = llvm::cast<mlir::FloatType>(ty);
        auto x2 = b.create<mlir::arith::MulFOp>(loc, x, x);
        auto radicand = b.create<mlir::arith::SubFOp>(
            loc, floatConst(b, loc, ft, 1.0), x2);
        return b.create<mlir::math::AtanOp>(
            loc, b.create<mlir::arith::DivFOp>(loc, x,
                                               b.create<mlir::math::SqrtOp>(loc, radicand)));
      }};
    case VulkanKernelRecipe::ACOS:
      return UnaryCallback{[](mlir::OpBuilder& b, mlir::Location loc,
                              mlir::Type ty, mlir::Value x) {
        auto ft = llvm::cast<mlir::FloatType>(ty);
        auto asin = UnaryCallback{[](mlir::OpBuilder& bb, mlir::Location ll,
                                     mlir::Type tt, mlir::Value xx) {
          auto fft = llvm::cast<mlir::FloatType>(tt);
          auto xx2 = bb.create<mlir::arith::MulFOp>(ll, xx, xx);
          auto rad = bb.create<mlir::arith::SubFOp>(
              ll, floatConst(bb, ll, fft, 1.0), xx2);
          return bb.create<mlir::math::AtanOp>(
              ll, bb.create<mlir::arith::DivFOp>(ll, xx,
                                                 bb.create<mlir::math::SqrtOp>(ll, rad)));
        }};
        return b.create<mlir::arith::SubFOp>(
            loc, floatConst(b, loc, ft, 1.5707963267948966),
            asin(b, loc, ty, x));
      }};
    case VulkanKernelRecipe::ASINH:
      return UnaryCallback{[](mlir::OpBuilder& b, mlir::Location loc,
                              mlir::Type ty, mlir::Value x) {
        auto ft = llvm::cast<mlir::FloatType>(ty);
        auto x2 = b.create<mlir::arith::MulFOp>(loc, x, x);
        auto rad = b.create<mlir::arith::AddFOp>(
            loc, x2, floatConst(b, loc, ft, 1.0));
        return emitLog(b, loc, ty,
                       b.create<mlir::arith::AddFOp>(loc, x,
                                                     b.create<mlir::math::SqrtOp>(loc, rad)));
      }};
    case VulkanKernelRecipe::ACOSH:
      return UnaryCallback{[](mlir::OpBuilder& b, mlir::Location loc,
                              mlir::Type ty, mlir::Value x) {
        auto ft = llvm::cast<mlir::FloatType>(ty);
        auto x2 = b.create<mlir::arith::MulFOp>(loc, x, x);
        auto rad = b.create<mlir::arith::SubFOp>(
            loc, x2, floatConst(b, loc, ft, 1.0));
        return emitLog(b, loc, ty,
                       b.create<mlir::arith::AddFOp>(loc, x,
                                                     b.create<mlir::math::SqrtOp>(loc, rad)));
      }};
    case VulkanKernelRecipe::ATANH:
      return UnaryCallback{[](mlir::OpBuilder& b, mlir::Location loc,
                              mlir::Type ty, mlir::Value x) {
        auto ft = llvm::cast<mlir::FloatType>(ty);
        auto one = floatConst(b, loc, ft, 1.0);
        auto numerator = b.create<mlir::arith::AddFOp>(loc, one, x);
        auto denominator = b.create<mlir::arith::SubFOp>(loc, one, x);
        auto ratio = b.create<mlir::arith::DivFOp>(loc, numerator, denominator);
        return b.create<mlir::arith::MulFOp>(
            loc, floatConst(b, loc, ft, 0.5), emitLog(b, loc, ty, ratio));
      }};
    case VulkanKernelRecipe::ERF:
      return UnaryCallback{[](mlir::OpBuilder& b, mlir::Location loc,
                              mlir::Type, mlir::Value x) {
        return b.create<mlir::math::ErfOp>(loc, x);
      }};
    case VulkanKernelRecipe::ERFC:
      return UnaryCallback{[](mlir::OpBuilder& b, mlir::Location loc,
                              mlir::Type, mlir::Value x) {
        return b.create<mlir::math::ErfcOp>(loc, x);
      }};
    case VulkanKernelRecipe::SILU:
      return UnaryCallback{emitSilu};
    case VulkanKernelRecipe::STABILIZE_FP16:
      return UnaryCallback{[](mlir::OpBuilder& b, mlir::Location loc,
                              mlir::Type ty, mlir::Value x) {
        auto ft = llvm::cast<mlir::FloatType>(ty);
        auto nonPositive = b.create<mlir::arith::CmpFOp>(
            loc, mlir::arith::CmpFPredicate::OLE, x,
            floatConst(b, loc, ft, 0.0));
        return b.create<mlir::arith::SelectOp>(
            loc, nonPositive,
            floatConst(b, loc, ft, std::numeric_limits<float>::min()), x);
      }};
    case VulkanKernelRecipe::ABS:
      return UnaryCallback{
          [](mlir::OpBuilder& b, mlir::Location loc, mlir::Type,
             mlir::Value x) {
            return b.create<mlir::math::AbsFOp>(loc, x);
          }};
    case VulkanKernelRecipe::NEGATE:
      return UnaryCallback{
          [](mlir::OpBuilder& b, mlir::Location loc, mlir::Type,
             mlir::Value x) {
            return b.create<mlir::arith::NegFOp>(loc, x);
          }};
    case VulkanKernelRecipe::SQRT:
      return UnaryCallback{emitSqrt};
    case VulkanKernelRecipe::TANH:
      return UnaryCallback{emitTanh};
    case VulkanKernelRecipe::SIGMOID:
      return UnaryCallback{emitSigmoid};
    case VulkanKernelRecipe::RELU:
    case VulkanKernelRecipe::THRESHOLDED_RELU:
      return UnaryCallback{emitRelu};
    case VulkanKernelRecipe::GELU:
      return UnaryCallback{emitGelu};
    case VulkanKernelRecipe::FAST_GELU:
      return UnaryCallback{emitFastGelu};
    case VulkanKernelRecipe::CUBE:
      return UnaryCallback{
          [](mlir::OpBuilder& b, mlir::Location loc, mlir::Type,
             mlir::Value x) {
            mlir::Value square = b.create<mlir::arith::MulFOp>(loc, x, x);
            return b.create<mlir::arith::MulFOp>(loc, square, x);
          }};
    case VulkanKernelRecipe::RECTIFIED_TANH:
      return UnaryCallback{
          [](mlir::OpBuilder& b, mlir::Location loc, mlir::Type ty,
             mlir::Value x) {
            auto ft = llvm::cast<mlir::FloatType>(ty);
            return b.create<mlir::arith::MaximumFOp>(
                loc, emitTanh(b, loc, ty, x), floatConst(b, loc, ft, 0.0));
          }};
    case VulkanKernelRecipe::RATIONAL_TANH:
      return UnaryCallback{
          [](mlir::OpBuilder& b, mlir::Location loc, mlir::Type ty,
             mlir::Value x) {
            auto ft = llvm::cast<mlir::FloatType>(ty);
            mlir::Value zero = floatConst(b, loc, ft, 0.0);
            mlir::Value one = floatConst(b, loc, ft, 1.0);
            mlir::Value d = b.create<mlir::arith::MulFOp>(
                loc, x, floatConst(b, loc, ft, 2.0 / 3.0));
            mlir::Value d2 = b.create<mlir::arith::MulFOp>(loc, d, d);
            mlir::Value d4 = b.create<mlir::arith::MulFOp>(loc, d2, d2);
            mlir::Value denominator = b.create<mlir::arith::AddFOp>(
                loc, one, b.create<mlir::math::AbsFOp>(loc, d));
            denominator = b.create<mlir::arith::AddFOp>(
                loc, denominator, d2);
            denominator = b.create<mlir::arith::AddFOp>(
                loc, denominator,
                b.create<mlir::arith::MulFOp>(
                    loc, floatConst(b, loc, ft, 1.41645), d4));
            mlir::Value positive = b.create<mlir::arith::CmpFOp>(
                loc, mlir::arith::CmpFPredicate::OGT, d, zero);
            mlir::Value negative = b.create<mlir::arith::CmpFOp>(
                loc, mlir::arith::CmpFPredicate::OLT, d, zero);
            mlir::Value sign = b.create<mlir::arith::SelectOp>(
                loc, positive, one,
                b.create<mlir::arith::SelectOp>(
                    loc, negative, floatConst(b, loc, ft, -1.0), zero));
            mlir::Value approximation = b.create<mlir::arith::SubFOp>(
                loc, one,
                b.create<mlir::arith::DivFOp>(loc, one, denominator));
            return b.create<mlir::arith::MulFOp>(
                loc, floatConst(b, loc, ft, 1.7159),
                b.create<mlir::arith::MulFOp>(loc, sign, approximation));
          }};
    case VulkanKernelRecipe::FLOOR:
      return UnaryCallback{
          [](mlir::OpBuilder& b, mlir::Location loc, mlir::Type,
             mlir::Value x) {
            return b.create<mlir::math::FloorOp>(loc, x);
          }};
    case VulkanKernelRecipe::LOG1P:
      return UnaryCallback{
          [](mlir::OpBuilder& b, mlir::Location loc, mlir::Type ty,
             mlir::Value x) {
            auto ft = llvm::cast<mlir::FloatType>(ty);
            return emitLog(
                b, loc, ty,
                b.create<mlir::arith::AddFOp>(
                    loc, x, floatConst(b, loc, ft, 1.0)));
          }};
    case VulkanKernelRecipe::RINT:
      return UnaryCallback{
          [](mlir::OpBuilder& b, mlir::Location loc, mlir::Type,
             mlir::Value x) {
            return b.create<mlir::math::RoundEvenOp>(loc, x);
          }};
    case VulkanKernelRecipe::SQUARE:
      return UnaryCallback{
          [](mlir::OpBuilder& b, mlir::Location loc, mlir::Type,
             mlir::Value x) {
            return b.create<mlir::arith::MulFOp>(loc, x, x);
          }};
    case VulkanKernelRecipe::SQUARED_RELU:
      return UnaryCallback{
          [](mlir::OpBuilder& b, mlir::Location loc, mlir::Type ty,
             mlir::Value x) {
            mlir::Value relu = emitRelu(b, loc, ty, x);
            return b.create<mlir::arith::MulFOp>(loc, relu, relu);
          }};
    case VulkanKernelRecipe::HARD_SIGMOID:
      return UnaryCallback{
          [](mlir::OpBuilder& b, mlir::Location loc, mlir::Type ty,
             mlir::Value x) {
            auto ft = llvm::cast<mlir::FloatType>(ty);
            mlir::Value t = b.create<mlir::arith::MulFOp>(
                loc, x, floatConst(b, loc, ft, 0.2));
            t = b.create<mlir::arith::AddFOp>(
                loc, t, floatConst(b, loc, ft, 0.5));
            t = b.create<mlir::arith::MinimumFOp>(
                loc, t, floatConst(b, loc, ft, 1.0));
            return b.create<mlir::arith::MaximumFOp>(
                loc, t, floatConst(b, loc, ft, 0.0));
          }};
    case VulkanKernelRecipe::HARD_TANH:
      return UnaryCallback{
          [](mlir::OpBuilder& b, mlir::Location loc, mlir::Type ty,
             mlir::Value x) {
            auto ft = llvm::cast<mlir::FloatType>(ty);
            mlir::Value v = b.create<mlir::arith::MinimumFOp>(
                loc, x, floatConst(b, loc, ft, 1.0));
            return b.create<mlir::arith::MaximumFOp>(
                loc, v, floatConst(b, loc, ft, -1.0));
          }};
    case VulkanKernelRecipe::RELU6:
      return UnaryCallback{
          [](mlir::OpBuilder& b, mlir::Location loc, mlir::Type ty,
             mlir::Value x) {
            auto ft = llvm::cast<mlir::FloatType>(ty);
            mlir::Value v = b.create<mlir::arith::MaximumFOp>(
                loc, x, floatConst(b, loc, ft, 0.0));
            return b.create<mlir::arith::MinimumFOp>(
                loc, v, floatConst(b, loc, ft, 6.0));
          }};
    case VulkanKernelRecipe::LEAKY_RELU:
      return UnaryCallback{
          [](mlir::OpBuilder& b, mlir::Location loc, mlir::Type ty,
             mlir::Value x) {
            auto ft = llvm::cast<mlir::FloatType>(ty);
            mlir::Value zero = floatConst(b, loc, ft, 0.0);
            mlir::Value negative = b.create<mlir::arith::MulFOp>(
                loc, x, floatConst(b, loc, ft, 0.01));
            mlir::Value positive = b.create<mlir::arith::CmpFOp>(
                loc, mlir::arith::CmpFPredicate::OGE, x, zero);
            return b.create<mlir::arith::SelectOp>(
                loc, positive, x, negative);
          }};
    case VulkanKernelRecipe::ELU:
      return UnaryCallback{
          [](mlir::OpBuilder& b, mlir::Location loc, mlir::Type ty,
             mlir::Value x) {
            auto ft = llvm::cast<mlir::FloatType>(ty);
            mlir::Value zero = floatConst(b, loc, ft, 0.0);
            mlir::Value negative = b.create<mlir::arith::SubFOp>(
                loc, emitExp(b, loc, ty, x), floatConst(b, loc, ft, 1.0));
            mlir::Value positive = b.create<mlir::arith::CmpFOp>(
                loc, mlir::arith::CmpFPredicate::OGE, x, zero);
            return b.create<mlir::arith::SelectOp>(
                loc, positive, x, negative);
          }};
    case VulkanKernelRecipe::SELU:
      return UnaryCallback{
          [](mlir::OpBuilder& b, mlir::Location loc, mlir::Type ty,
             mlir::Value x) {
            auto ft = llvm::cast<mlir::FloatType>(ty);
            mlir::Value zero = floatConst(b, loc, ft, 0.0);
            mlir::Value one = floatConst(b, loc, ft, 1.0);
            mlir::Value negative = b.create<mlir::arith::MulFOp>(
                loc, floatConst(b, loc, ft, 1.6732632423543772),
                b.create<mlir::arith::SubFOp>(
                    loc, emitExp(b, loc, ty, x), one));
            mlir::Value positive = b.create<mlir::arith::CmpFOp>(
                loc, mlir::arith::CmpFPredicate::OGE, x, zero);
            mlir::Value branch = b.create<mlir::arith::SelectOp>(
                loc, positive, x, negative);
            return b.create<mlir::arith::MulFOp>(
                loc, floatConst(b, loc, ft, 1.0507009873554805), branch);
          }};
    case VulkanKernelRecipe::SOFTPLUS:
      return UnaryCallback{
          [](mlir::OpBuilder& b, mlir::Location loc, mlir::Type ty,
             mlir::Value x) {
            auto ft = llvm::cast<mlir::FloatType>(ty);
            mlir::Value zero = floatConst(b, loc, ft, 0.0);
            mlir::Value magnitude = b.create<mlir::math::AbsFOp>(loc, x);
            mlir::Value correction = emitLog(
                b, loc, ty,
                b.create<mlir::arith::AddFOp>(
                    loc, floatConst(b, loc, ft, 1.0),
                    emitExp(
                        b, loc, ty,
                        b.create<mlir::arith::NegFOp>(loc, magnitude))));
            return b.create<mlir::arith::AddFOp>(
                loc, b.create<mlir::arith::MaximumFOp>(loc, zero, x),
                correction);
          }};
    case VulkanKernelRecipe::SOFTSIGN:
      return UnaryCallback{
          [](mlir::OpBuilder& b, mlir::Location loc, mlir::Type ty,
             mlir::Value x) {
            auto ft = llvm::cast<mlir::FloatType>(ty);
            mlir::Value denominator = b.create<mlir::arith::AddFOp>(
                loc, floatConst(b, loc, ft, 1.0),
                b.create<mlir::math::AbsFOp>(loc, x));
            return b.create<mlir::arith::DivFOp>(loc, x, denominator);
          }};
    case VulkanKernelRecipe::TANH_DERIVATIVE:
      return UnaryCallback{[](mlir::OpBuilder& b, mlir::Location loc,
                              mlir::Type ty, mlir::Value x) {
        auto ft = llvm::cast<mlir::FloatType>(ty);
        auto one = floatConst(b, loc, ft, 1.0);
        auto t = emitTanh(b, loc, ty, x);
        return b.create<mlir::arith::SubFOp>(
            loc, one, b.create<mlir::arith::MulFOp>(loc, t, t));
      }};
    case VulkanKernelRecipe::HARD_TANH_DERIVATIVE:
      return UnaryCallback{[](mlir::OpBuilder& b, mlir::Location loc,
                              mlir::Type ty, mlir::Value x) {
        auto ft = llvm::cast<mlir::FloatType>(ty);
        auto lower = b.create<mlir::arith::CmpFOp>(
            loc, mlir::arith::CmpFPredicate::OGE, x,
            floatConst(b, loc, ft, -1.0));
        auto upper = b.create<mlir::arith::CmpFOp>(
            loc, mlir::arith::CmpFPredicate::OLE, x,
            floatConst(b, loc, ft, 1.0));
        auto inside = b.create<mlir::arith::AndIOp>(loc, lower, upper);
        return b.create<mlir::arith::SelectOp>(
            loc, inside, floatConst(b, loc, ft, 1.0),
            floatConst(b, loc, ft, 0.0));
      }};
    case VulkanKernelRecipe::SIGMOID_DERIVATIVE:
      return UnaryCallback{[](mlir::OpBuilder& b, mlir::Location loc,
                              mlir::Type ty, mlir::Value x) {
        auto ft = llvm::cast<mlir::FloatType>(ty);
        auto sigmoid = emitSigmoid(b, loc, ty, x);
        auto oneMinus = b.create<mlir::arith::SubFOp>(
            loc, floatConst(b, loc, ft, 1.0), sigmoid);
        return b.create<mlir::arith::MulFOp>(loc, sigmoid, oneMinus);
      }};
    case VulkanKernelRecipe::SOFTSIGN_DERIVATIVE:
      return UnaryCallback{[](mlir::OpBuilder& b, mlir::Location loc,
                              mlir::Type ty, mlir::Value x) {
        auto ft = llvm::cast<mlir::FloatType>(ty);
        auto denominator = b.create<mlir::arith::AddFOp>(
            loc, floatConst(b, loc, ft, 1.0),
            b.create<mlir::math::AbsFOp>(loc, x));
        auto squared = b.create<mlir::arith::MulFOp>(
            loc, denominator, denominator);
        return b.create<mlir::arith::DivFOp>(
            loc, floatConst(b, loc, ft, 1.0), squared);
      }};
    case VulkanKernelRecipe::TAN_DERIVATIVE:
      return UnaryCallback{[](mlir::OpBuilder& b, mlir::Location loc,
                              mlir::Type ty, mlir::Value x) {
        auto ft = llvm::cast<mlir::FloatType>(ty);
        auto cosine = b.create<mlir::math::CosOp>(loc, x);
        auto denominator = b.create<mlir::arith::MulFOp>(
            loc, cosine, cosine);
        return b.create<mlir::arith::DivFOp>(
            loc, floatConst(b, loc, ft, 1.0), denominator);
      }};
    case VulkanKernelRecipe::SELU_DERIVATIVE:
      return UnaryCallback{[](mlir::OpBuilder& b, mlir::Location loc,
                              mlir::Type ty, mlir::Value x) {
        auto ft = llvm::cast<mlir::FloatType>(ty);
        auto lambda = floatConst(b, loc, ft, 1.0507009873554805);
        auto alpha = floatConst(b, loc, ft, 1.6732632423543772);
        auto positive = b.create<mlir::arith::CmpFOp>(
            loc, mlir::arith::CmpFPredicate::OGT, x,
            floatConst(b, loc, ft, 0.0));
        auto negative = b.create<mlir::arith::MulFOp>(
            loc, lambda,
            b.create<mlir::arith::MulFOp>(
                loc, alpha, emitExp(b, loc, ty, x)));
        return b.create<mlir::arith::SelectOp>(loc, positive, lambda, negative);
      }};
    case VulkanKernelRecipe::HARD_SIGMOID_DERIVATIVE:
      return UnaryCallback{[](mlir::OpBuilder& b, mlir::Location loc,
                              mlir::Type ty, mlir::Value x) {
        auto ft = llvm::cast<mlir::FloatType>(ty);
        auto lower = b.create<mlir::arith::CmpFOp>(
            loc, mlir::arith::CmpFPredicate::OGE, x,
            floatConst(b, loc, ft, -2.5));
        auto upper = b.create<mlir::arith::CmpFOp>(
            loc, mlir::arith::CmpFPredicate::OLE, x,
            floatConst(b, loc, ft, 2.5));
        auto inside = b.create<mlir::arith::AndIOp>(loc, lower, upper);
        return b.create<mlir::arith::SelectOp>(
            loc, inside, floatConst(b, loc, ft, 0.2),
            floatConst(b, loc, ft, 0.0));
      }};
    case VulkanKernelRecipe::RATIONAL_TANH_DERIVATIVE:
      return UnaryCallback{[](mlir::OpBuilder& b, mlir::Location loc,
                              mlir::Type ty, mlir::Value x) {
        auto ft = llvm::cast<mlir::FloatType>(ty);
        auto twoThirds = floatConst(b, loc, ft, 2.0 / 3.0);
        auto dis = b.create<mlir::arith::MulFOp>(loc, twoThirds, x);
        auto absDis = b.create<mlir::math::AbsFOp>(loc, dis);
        auto dis2 = b.create<mlir::arith::MulFOp>(loc, dis, dis);
        auto dis3 = b.create<mlir::arith::MulFOp>(loc, dis2, dis);
        auto dis4 = b.create<mlir::arith::MulFOp>(loc, dis2, dis2);
        auto denominator = b.create<mlir::arith::AddFOp>(
            loc, floatConst(b, loc, ft, 1.0), absDis);
        denominator = b.create<mlir::arith::AddFOp>(loc, denominator, dis2);
        denominator = b.create<mlir::arith::AddFOp>(
            loc, denominator,
            b.create<mlir::arith::MulFOp>(
                loc, floatConst(b, loc, ft, 1.41645), dis4));
        auto sign = b.create<mlir::arith::SelectOp>(
            loc,
            b.create<mlir::arith::CmpFOp>(
                loc, mlir::arith::CmpFPredicate::OGE, dis,
                floatConst(b, loc, ft, 0.0)),
            floatConst(b, loc, ft, 1.0), floatConst(b, loc, ft, -1.0));
        auto numerator = b.create<mlir::arith::AddFOp>(
            loc, floatConst(b, loc, ft, 1.0),
            b.create<mlir::arith::MulFOp>(
                loc, sign,
                b.create<mlir::arith::AddFOp>(
                    loc, b.create<mlir::arith::MulFOp>(
                             loc, floatConst(b, loc, ft, 2.0), dis),
                    b.create<mlir::arith::MulFOp>(
                        loc, floatConst(b, loc, ft, 5.6658), dis3))));
        auto denominatorSquared = b.create<mlir::arith::MulFOp>(
            loc, denominator, denominator);
        auto derivative = b.create<mlir::arith::DivFOp>(
            loc, numerator, denominatorSquared);
        return b.create<mlir::arith::MulFOp>(
            loc, floatConst(b, loc, ft, 1.7159 * (2.0 / 3.0)), derivative);
      }};
    case VulkanKernelRecipe::RECTIFIED_TANH_DERIVATIVE:
      return UnaryCallback{[](mlir::OpBuilder& b, mlir::Location loc,
                              mlir::Type ty, mlir::Value x) {
        auto ft = llvm::cast<mlir::FloatType>(ty);
        auto tanhDerivative = unaryCallbackFor(
            VulkanKernelRecipe::TANH_DERIVATIVE)(b, loc, ty, x);
        auto positive = b.create<mlir::arith::CmpFOp>(
            loc, mlir::arith::CmpFPredicate::OGT, x,
            floatConst(b, loc, ft, 0.0));
        return b.create<mlir::arith::SelectOp>(
            loc, positive, tanhDerivative, floatConst(b, loc, ft, 0.0));
      }};
    case VulkanKernelRecipe::SWISH_DERIVATIVE:
      return UnaryCallback{[](mlir::OpBuilder& b, mlir::Location loc,
                              mlir::Type ty, mlir::Value x) {
        auto ft = llvm::cast<mlir::FloatType>(ty);
        auto ex = emitExp(b, loc, ty, x);
        auto onePlus = b.create<mlir::arith::AddFOp>(
            loc, ex, floatConst(b, loc, ft, 1.0));
        auto numerator = b.create<mlir::arith::MulFOp>(
            loc, ex,
            b.create<mlir::arith::AddFOp>(
                loc, b.create<mlir::arith::AddFOp>(
                         loc, x, ex),
                floatConst(b, loc, ft, 1.0)));
        return b.create<mlir::arith::DivFOp>(
            loc, numerator,
            b.create<mlir::arith::MulFOp>(loc, onePlus, onePlus));
      }};
    case VulkanKernelRecipe::ACOSH_DERIVATIVE:
      return UnaryCallback{[](mlir::OpBuilder& b, mlir::Location loc,
                              mlir::Type ty, mlir::Value x) {
        auto ft = llvm::cast<mlir::FloatType>(ty);
        auto one = floatConst(b, loc, ft, 1.0);
        auto lower = b.create<mlir::math::SqrtOp>(
            loc, b.create<mlir::arith::SubFOp>(loc, x, one));
        auto upper = b.create<mlir::math::SqrtOp>(
            loc, b.create<mlir::arith::AddFOp>(loc, x, one));
        return b.create<mlir::arith::DivFOp>(
            loc, one, b.create<mlir::arith::MulFOp>(loc, lower, upper));
      }};
    case VulkanKernelRecipe::ASINH_DERIVATIVE:
      return UnaryCallback{[](mlir::OpBuilder& b, mlir::Location loc,
                              mlir::Type ty, mlir::Value x) {
        auto ft = llvm::cast<mlir::FloatType>(ty);
        auto square = b.create<mlir::arith::MulFOp>(loc, x, x);
        auto denominator = b.create<mlir::math::SqrtOp>(
            loc, b.create<mlir::arith::AddFOp>(
                     loc, square, floatConst(b, loc, ft, 1.0)));
        return b.create<mlir::arith::DivFOp>(
            loc, floatConst(b, loc, ft, 1.0), denominator);
      }};
    case VulkanKernelRecipe::SINH_DERIVATIVE:
      return UnaryCallback{[](mlir::OpBuilder& b, mlir::Location loc,
                              mlir::Type, mlir::Value x) {
        return b.create<mlir::math::CoshOp>(loc, x);
      }};
    case VulkanKernelRecipe::LOG_SIGMOID_DERIVATIVE:
      return UnaryCallback{[](mlir::OpBuilder& b, mlir::Location loc,
                              mlir::Type ty, mlir::Value x) {
        auto ft = llvm::cast<mlir::FloatType>(ty);
        return b.create<mlir::arith::DivFOp>(
            loc, floatConst(b, loc, ft, 1.0),
            b.create<mlir::arith::AddFOp>(
                loc, emitExp(b, loc, ty, x),
                floatConst(b, loc, ft, 1.0)));
      }};
    case VulkanKernelRecipe::SPECIAL_DERIVATIVE:
      return UnaryCallback{[](mlir::OpBuilder& b, mlir::Location loc,
                              mlir::Type ty, mlir::Value x) {
        auto ft = llvm::cast<mlir::FloatType>(ty);
        return b.create<mlir::arith::MulFOp>(
            loc, x,
            b.create<mlir::arith::SubFOp>(
                loc, floatConst(b, loc, ft, 1.0), x));
      }};
    case VulkanKernelRecipe::CUBE_DERIVATIVE:
      return UnaryCallback{[](mlir::OpBuilder& b, mlir::Location loc,
                              mlir::Type ty, mlir::Value x) {
        auto ft = llvm::cast<mlir::FloatType>(ty);
        auto square = b.create<mlir::arith::MulFOp>(loc, x, x);
        return b.create<mlir::arith::MulFOp>(
            loc, floatConst(b, loc, ft, 3.0), square);
      }};
    case VulkanKernelRecipe::GELU_DERIVATIVE:
      return UnaryCallback{[](mlir::OpBuilder& b, mlir::Location loc,
                              mlir::Type ty, mlir::Value x) {
        auto ft = llvm::cast<mlir::FloatType>(ty);
        auto x17 = b.create<mlir::arith::MulFOp>(
            loc, floatConst(b, loc, ft, 1.702), x);
        auto ep = emitExp(b, loc, ty, x17);
        auto onePlus = b.create<mlir::arith::AddFOp>(
            loc, floatConst(b, loc, ft, 1.0), ep);
        auto numerator = b.create<mlir::arith::MulFOp>(
            loc, ep,
            b.create<mlir::arith::AddFOp>(loc, onePlus, x17));
        return b.create<mlir::arith::DivFOp>(
            loc, numerator,
            b.create<mlir::arith::MulFOp>(loc, onePlus, onePlus));
      }};
    case VulkanKernelRecipe::PRECISE_GELU_DERIVATIVE:
      return UnaryCallback{[](mlir::OpBuilder& b, mlir::Location loc,
                              mlir::Type ty, mlir::Value x) {
        auto ft = llvm::cast<mlir::FloatType>(ty);
        auto x79 = b.create<mlir::arith::MulFOp>(
            loc, floatConst(b, loc, ft, 0.797885), x);
        auto x2 = b.create<mlir::arith::MulFOp>(loc, x, x);
        auto x3 = b.create<mlir::arith::MulFOp>(loc, x2, x);
        auto arg = b.create<mlir::arith::AddFOp>(
            loc, x79,
            b.create<mlir::arith::MulFOp>(
                loc, floatConst(b, loc, ft, 0.0356774), x3));
        auto tanh = emitTanh(b, loc, ty, arg);
        auto sechSquared = b.create<mlir::arith::SubFOp>(
            loc, floatConst(b, loc, ft, 1.0),
            b.create<mlir::arith::MulFOp>(loc, tanh, tanh));
        auto term = b.create<mlir::arith::MulFOp>(
            loc,
            b.create<mlir::arith::AddFOp>(
                loc,
                b.create<mlir::arith::MulFOp>(
                    loc, floatConst(b, loc, ft, 0.398942), x),
                b.create<mlir::arith::MulFOp>(
                    loc, floatConst(b, loc, ft, 0.0535161), x3)),
            sechSquared);
        return b.create<mlir::arith::AddFOp>(
            loc, floatConst(b, loc, ft, 0.5),
            b.create<mlir::arith::AddFOp>(
                loc, term,
                b.create<mlir::arith::MulFOp>(
                    loc, floatConst(b, loc, ft, 0.5), tanh)));
      }};
    case VulkanKernelRecipe::MISH_DERIVATIVE:
      return UnaryCallback{[](mlir::OpBuilder& b, mlir::Location loc,
                              mlir::Type ty, mlir::Value x) {
        auto ft = llvm::cast<mlir::FloatType>(ty);
        auto ex = emitExp(b, loc, ty, x);
        auto e2x = b.create<mlir::arith::MulFOp>(loc, ex, ex);
        auto e3x = b.create<mlir::arith::MulFOp>(loc, e2x, ex);
        auto xPlusOne = b.create<mlir::arith::AddFOp>(
            loc, x, floatConst(b, loc, ft, 1.0));
        auto numerator = b.create<mlir::arith::MulFOp>(
            loc, ex,
            b.create<mlir::arith::AddFOp>(
                loc,
                b.create<mlir::arith::AddFOp>(
                    loc,
                    b.create<mlir::arith::MulFOp>(
                        loc, floatConst(b, loc, ft, 4.0), xPlusOne),
                    b.create<mlir::arith::MulFOp>(
                        loc, floatConst(b, loc, ft, 4.0), e2x)),
                b.create<mlir::arith::AddFOp>(
                    loc, e3x,
                    b.create<mlir::arith::MulFOp>(
                        loc, ex,
                        b.create<mlir::arith::AddFOp>(
                            loc,
                            b.create<mlir::arith::MulFOp>(
                                loc, floatConst(b, loc, ft, 4.0), x),
                            floatConst(b, loc, ft, 6.0))))));
        auto denominatorBase = b.create<mlir::arith::AddFOp>(
            loc,
            b.create<mlir::arith::MulFOp>(
                loc, floatConst(b, loc, ft, 2.0), ex),
            b.create<mlir::arith::AddFOp>(
                loc, e2x, floatConst(b, loc, ft, 2.0)));
        auto denominator = b.create<mlir::arith::MulFOp>(
            loc, denominatorBase, denominatorBase);
        return b.create<mlir::arith::DivFOp>(loc, numerator, denominator);
      }};
    default:
      return {};
  }
}

static BinaryCallback binaryCallbackFor(VulkanKernelRecipe semantic) {
  switch (semantic) {
    case VulkanKernelRecipe::EPSILON_COMPARE:
      return BinaryCallback{[](mlir::OpBuilder& b, mlir::Location loc,
                               mlir::Value a, mlir::Value c) {
        auto type = llvm::cast<mlir::FloatType>(a.getType());
        auto difference = b.create<mlir::math::AbsFOp>(
            loc, b.create<mlir::arith::SubFOp>(loc, a, c));
        return b.create<mlir::arith::CmpFOp>(
            loc, mlir::arith::CmpFPredicate::OLE, difference,
            floatConst(b, loc, type, 1e-5));
      }};
    case VulkanKernelRecipe::MATCH_CONDITION:
      return BinaryCallback{[](mlir::OpBuilder& b, mlir::Location loc,
                               mlir::Value a, mlir::Value c) {
        return b.create<mlir::arith::CmpFOp>(
            loc, mlir::arith::CmpFPredicate::OEQ, a, c);
      }};
    case VulkanKernelRecipe::IGAMMA:
    case VulkanKernelRecipe::IGAMMAC:
      return BinaryCallback{[semantic](mlir::OpBuilder& b, mlir::Location loc,
                                       mlir::Value a, mlir::Value x) {
        auto type = llvm::cast<mlir::FloatType>(a.getType());
        auto one = floatConst(b, loc, type, 1.0);
        auto twoPi = floatConst(b, loc, type, 6.283185307179586);
        auto aSafe = b.create<mlir::arith::MaximumFOp>(
            loc, a, floatConst(b, loc, type, 1e-6));
        auto logGamma = b.create<mlir::arith::AddFOp>(
            loc,
            b.create<mlir::arith::SubFOp>(
                loc,
                b.create<mlir::arith::MulFOp>(
                    loc,
                    b.create<mlir::arith::AddFOp>(
                        loc, aSafe, floatConst(b, loc, type, 0.5)),
                    emitLog(b, loc, a.getType(), aSafe)),
                aSafe),
            b.create<mlir::arith::MulFOp>(
                loc, floatConst(b, loc, type, 0.5),
                emitLog(b, loc, a.getType(), twoPi)));
        auto xSafe = b.create<mlir::arith::MaximumFOp>(
            loc, x, floatConst(b, loc, type, 1e-6));
        auto numerator = b.create<mlir::arith::SubFOp>(
            loc, b.create<mlir::arith::MulFOp>(
                    loc, aSafe, emitLog(b, loc, x.getType(), xSafe)),
            x);
        auto estimate = emitExp(
            b, loc, a.getType(),
            b.create<mlir::arith::SubFOp>(loc, numerator, logGamma));
        return semantic == VulkanKernelRecipe::IGAMMA
                   ? estimate
                   : b.create<mlir::arith::SubFOp>(loc, one, estimate);
      }};
    case VulkanKernelRecipe::EQUAL:
      return BinaryCallback{[](mlir::OpBuilder& b, mlir::Location loc,
                               mlir::Value a, mlir::Value c) {
        return b.create<mlir::arith::CmpFOp>(
            loc, mlir::arith::CmpFPredicate::OEQ, a, c);
      }};
    case VulkanKernelRecipe::NOT_EQUAL:
      return BinaryCallback{[](mlir::OpBuilder& b, mlir::Location loc,
                               mlir::Value a, mlir::Value c) {
        return b.create<mlir::arith::CmpFOp>(
            loc, mlir::arith::CmpFPredicate::UNE, a, c);
      }};
    case VulkanKernelRecipe::LESS:
      return BinaryCallback{[](mlir::OpBuilder& b, mlir::Location loc,
                               mlir::Value a, mlir::Value c) {
        return b.create<mlir::arith::CmpFOp>(
            loc, mlir::arith::CmpFPredicate::OLT, a, c);
      }};
    case VulkanKernelRecipe::LESS_EQUAL:
      return BinaryCallback{[](mlir::OpBuilder& b, mlir::Location loc,
                               mlir::Value a, mlir::Value c) {
        return b.create<mlir::arith::CmpFOp>(
            loc, mlir::arith::CmpFPredicate::OLE, a, c);
      }};
    case VulkanKernelRecipe::GREATER:
      return BinaryCallback{[](mlir::OpBuilder& b, mlir::Location loc,
                               mlir::Value a, mlir::Value c) {
        return b.create<mlir::arith::CmpFOp>(
            loc, mlir::arith::CmpFPredicate::OGT, a, c);
      }};
    case VulkanKernelRecipe::GREATER_EQUAL:
      return BinaryCallback{[](mlir::OpBuilder& b, mlir::Location loc,
                               mlir::Value a, mlir::Value c) {
        return b.create<mlir::arith::CmpFOp>(
            loc, mlir::arith::CmpFPredicate::OGE, a, c);
      }};
    case VulkanKernelRecipe::BOOLEAN_AND:
    case VulkanKernelRecipe::BOOLEAN_OR:
    case VulkanKernelRecipe::BOOLEAN_XOR:
      return BinaryCallback{[semantic](mlir::OpBuilder& b, mlir::Location loc,
                                      mlir::Value a, mlir::Value c) {
        auto type = llvm::cast<mlir::FloatType>(a.getType());
        auto zero = floatConst(b, loc, type, 0.0);
        auto one = floatConst(b, loc, type, 1.0);
        auto aTrue = b.create<mlir::arith::CmpFOp>(
            loc, mlir::arith::CmpFPredicate::UNE, a, zero);
        auto cTrue = b.create<mlir::arith::CmpFOp>(
            loc, mlir::arith::CmpFPredicate::UNE, c, zero);
        mlir::Value predicate;
        if (semantic == VulkanKernelRecipe::BOOLEAN_AND)
          predicate = b.create<mlir::arith::AndIOp>(loc, aTrue, cTrue);
        else if (semantic == VulkanKernelRecipe::BOOLEAN_OR)
          predicate = b.create<mlir::arith::OrIOp>(loc, aTrue, cTrue);
        else
          predicate = b.create<mlir::arith::XOrIOp>(loc, aTrue, cTrue);
        return b.create<mlir::arith::SelectOp>(loc, predicate, one, zero);
      }};
    case VulkanKernelRecipe::ADD:
      return BinaryCallback{
          [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value a,
             mlir::Value c) {
            return b.create<mlir::arith::AddFOp>(loc, a, c);
          }};
    case VulkanKernelRecipe::SUBTRACT:
      return BinaryCallback{
          [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value a,
             mlir::Value c) {
            return b.create<mlir::arith::SubFOp>(loc, a, c);
          }};
    case VulkanKernelRecipe::MULTIPLY:
      return BinaryCallback{
          [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value a,
             mlir::Value c) {
            return b.create<mlir::arith::MulFOp>(loc, a, c);
          }};
    case VulkanKernelRecipe::DIVIDE:
      return BinaryCallback{
          [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value a,
             mlir::Value c) {
            return b.create<mlir::arith::DivFOp>(loc, a, c);
          }};
    case VulkanKernelRecipe::MINIMUM:
      return BinaryCallback{
          [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value a,
             mlir::Value c) {
            return b.create<mlir::arith::MinimumFOp>(loc, a, c);
          }};
    case VulkanKernelRecipe::MAXIMUM:
      return BinaryCallback{
          [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value a,
             mlir::Value c) {
            return b.create<mlir::arith::MaximumFOp>(loc, a, c);
          }};
    case VulkanKernelRecipe::RELU:
      return BinaryCallback{
          [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value a,
             mlir::Value) {
            return b.create<mlir::arith::MaximumFOp>(
                loc, a, floatConst(b, loc, llvm::cast<mlir::FloatType>(a.getType()), 0.0));
          }};
    case VulkanKernelRecipe::RELU6:
      return BinaryCallback{
          [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value a,
             mlir::Value) {
            auto type = llvm::cast<mlir::FloatType>(a.getType());
            auto nonnegative = b.create<mlir::arith::MaximumFOp>(
                loc, a, floatConst(b, loc, type, 0.0));
            return b.create<mlir::arith::MinimumFOp>(
                loc, nonnegative, floatConst(b, loc, type, 6.0));
          }};
    case VulkanKernelRecipe::LEAKY_RELU:
      return BinaryCallback{
          [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value x,
             mlir::Value alpha) {
            auto type = llvm::cast<mlir::FloatType>(x.getType());
            mlir::Value zero = floatConst(b, loc, type, 0.0);
            mlir::Value nonnegative = b.create<mlir::arith::CmpFOp>(
                loc, mlir::arith::CmpFPredicate::OGE, x, zero);
            mlir::Value negative =
                b.create<mlir::arith::MulFOp>(loc, x, alpha);
            return b.create<mlir::arith::SelectOp>(
                loc, nonnegative, x, negative);
          }};
    case VulkanKernelRecipe::ELU_SCALAR:
      return BinaryCallback{
          [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value x,
             mlir::Value alpha) {
            auto type = llvm::cast<mlir::FloatType>(x.getType());
            auto zero = floatConst(b, loc, type, 0.0);
            auto one = floatConst(b, loc, type, 1.0);
            auto negative = b.create<mlir::arith::MulFOp>(
                loc, alpha,
                b.create<mlir::arith::SubFOp>(
                    loc, emitExp(b, loc, x.getType(), x), one));
            auto nonnegative = b.create<mlir::arith::CmpFOp>(
                loc, mlir::arith::CmpFPredicate::OGE, x, zero);
            return b.create<mlir::arith::SelectOp>(
                loc, nonnegative, x, negative);
          }};
    case VulkanKernelRecipe::ELU_DERIVATIVE:
      return BinaryCallback{
          [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value x,
             mlir::Value alpha) {
            auto type = llvm::cast<mlir::FloatType>(x.getType());
            auto zero = floatConst(b, loc, type, 0.0);
            auto one = floatConst(b, loc, type, 1.0);
            auto negative = b.create<mlir::arith::MulFOp>(
                loc, alpha, emitExp(b, loc, x.getType(), x));
            auto nonnegative = b.create<mlir::arith::CmpFOp>(
                loc, mlir::arith::CmpFPredicate::OGE, x, zero);
            return b.create<mlir::arith::SelectOp>(
                loc, nonnegative, one, negative);
          }};
    case VulkanKernelRecipe::RELU_DERIVATIVE:
      return BinaryCallback{
          [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value x,
             mlir::Value threshold) {
            auto type = llvm::cast<mlir::FloatType>(x.getType());
            auto one = floatConst(b, loc, type, 1.0);
            auto zero = floatConst(b, loc, type, 0.0);
            auto positive = b.create<mlir::arith::CmpFOp>(
                loc, mlir::arith::CmpFPredicate::OGT, x, threshold);
            return b.create<mlir::arith::SelectOp>(
                loc, positive, one, zero);
          }};
    case VulkanKernelRecipe::LEAKY_RELU_DERIVATIVE:
      return BinaryCallback{
          [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value x,
             mlir::Value alpha) {
            auto type = llvm::cast<mlir::FloatType>(x.getType());
            auto one = floatConst(b, loc, type, 1.0);
            auto zero = floatConst(b, loc, type, 0.0);
            auto nonnegative = b.create<mlir::arith::CmpFOp>(
                loc, mlir::arith::CmpFPredicate::OGE, x, zero);
            return b.create<mlir::arith::SelectOp>(
                loc, nonnegative, one, alpha);
          }};
    case VulkanKernelRecipe::SIGMOID_CROSS_ENTROPY_SMOOTHER:
      return BinaryCallback{
          [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value x,
             mlir::Value smoothing) {
            auto type = llvm::cast<mlir::FloatType>(x.getType());
            auto one = floatConst(b, loc, type, 1.0);
            auto half = floatConst(b, loc, type, 0.5);
            auto oneMinusSmoothing = b.create<mlir::arith::SubFOp>(
                loc, one, smoothing);
            return b.create<mlir::arith::AddFOp>(
                loc,
                b.create<mlir::arith::MulFOp>(loc, x, oneMinusSmoothing),
                b.create<mlir::arith::MulFOp>(loc, half, smoothing));
          }};
    case VulkanKernelRecipe::LOG_X:
      return BinaryCallback{
          [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value x,
             mlir::Value base) {
            return b.create<mlir::arith::DivFOp>(
                loc, emitLog(b, loc, x.getType(), x),
                emitLog(b, loc, x.getType(), base));
          }};
    case VulkanKernelRecipe::STEP:
      return BinaryCallback{
          [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value x,
             mlir::Value threshold) {
            auto type = llvm::cast<mlir::FloatType>(x.getType());
            auto one = floatConst(b, loc, type, 1.0);
            auto zero = floatConst(b, loc, type, 0.0);
            auto predicate = b.create<mlir::arith::CmpFOp>(
                loc, mlir::arith::CmpFPredicate::OGT, x, threshold);
            return b.create<mlir::arith::SelectOp>(loc, predicate, one, zero);
          }};
    case VulkanKernelRecipe::LSTM_CLIP:
      return BinaryCallback{
          [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value x,
             mlir::Value bound) {
            auto negativeBound = b.create<mlir::arith::NegFOp>(loc, bound);
            auto lower = b.create<mlir::arith::MaximumFOp>(
                loc, x, negativeBound);
            return b.create<mlir::arith::MinimumFOp>(loc, lower, bound);
          }};
    case VulkanKernelRecipe::SQUARED_REVERSE_SUBTRACT:
      return BinaryCallback{
          [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value x,
             mlir::Value scalar) {
            auto difference = b.create<mlir::arith::SubFOp>(loc, scalar, x);
            return b.create<mlir::arith::MulFOp>(loc, difference, difference);
          }};
    case VulkanKernelRecipe::REVERSE_POWER:
      return BinaryCallback{
          [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value x,
             mlir::Value scalar) {
            return b.create<mlir::math::PowFOp>(loc, scalar, x);
          }};
    case VulkanKernelRecipe::MOD:
      return BinaryCallback{
          [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value a,
             mlir::Value c) {
            return b.create<mlir::arith::RemFOp>(loc, a, c);
          }};
    case VulkanKernelRecipe::REVERSE_MOD:
      return BinaryCallback{
          [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value a,
             mlir::Value c) {
            return b.create<mlir::arith::RemFOp>(loc, c, a);
          }};
    case VulkanKernelRecipe::TRUNCATE_DIV:
      return BinaryCallback{
          [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value a,
             mlir::Value c) {
            return b.create<mlir::math::TruncOp>(
                loc, b.create<mlir::arith::DivFOp>(loc, a, c));
          }};
    case VulkanKernelRecipe::SAFE_DIVIDE:
    case VulkanKernelRecipe::DIVIDE_NO_NAN:
      return BinaryCallback{
          [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value a,
             mlir::Value c) {
            auto type = llvm::cast<mlir::FloatType>(a.getType());
            auto zero = floatConst(b, loc, type, 0.0);
            auto denominatorZero = b.create<mlir::arith::CmpFOp>(
                loc, mlir::arith::CmpFPredicate::OEQ, c, zero);
            auto quotient = b.create<mlir::arith::DivFOp>(loc, a, c);
            return b.create<mlir::arith::SelectOp>(loc, denominatorZero, zero,
                                                   quotient);
          }};
    case VulkanKernelRecipe::XDIVY:
      return BinaryCallback{
          [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value a,
             mlir::Value c) {
            auto type = llvm::cast<mlir::FloatType>(a.getType());
            auto zero = floatConst(b, loc, type, 0.0);
            auto numeratorZero = b.create<mlir::arith::CmpFOp>(
                loc, mlir::arith::CmpFPredicate::OEQ, a, zero);
            return b.create<mlir::arith::SelectOp>(
                loc, numeratorZero, zero,
                b.create<mlir::arith::DivFOp>(loc, a, c));
          }};
    case VulkanKernelRecipe::XLOGY:
      return BinaryCallback{
          [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value a,
             mlir::Value c) {
            return b.create<mlir::arith::MulFOp>(
                loc, a, emitLog(b, loc, a.getType(), c));
          }};
    case VulkanKernelRecipe::XLOG1PY:
      return BinaryCallback{
          [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value a,
             mlir::Value c) {
            auto type = llvm::cast<mlir::FloatType>(a.getType());
            auto onePlus = b.create<mlir::arith::AddFOp>(
                loc, c, floatConst(b, loc, type, 1.0));
            return b.create<mlir::arith::MulFOp>(
                loc, a, emitLog(b, loc, a.getType(), onePlus));
          }};
    case VulkanKernelRecipe::LOGICAL_NOT_BINARY:
      return BinaryCallback{[](mlir::OpBuilder& b, mlir::Location loc,
                               mlir::Value a, mlir::Value c) {
        auto type = llvm::cast<mlir::FloatType>(a.getType());
        auto zero = floatConst(b, loc, type, 0.0);
        auto one = floatConst(b, loc, type, 1.0);
        auto aTrue = b.create<mlir::arith::CmpFOp>(
            loc, mlir::arith::CmpFPredicate::UNE, a, zero);
        auto cTrue = b.create<mlir::arith::CmpFOp>(
            loc, mlir::arith::CmpFPredicate::UNE, c, zero);
        auto both = b.create<mlir::arith::AndIOp>(loc, aTrue, cTrue);
        return b.create<mlir::arith::SelectOp>(loc, both, zero, one);
      }};
    case VulkanKernelRecipe::RELATIVE_ERROR:
      return BinaryCallback{[](mlir::OpBuilder& b, mlir::Location loc,
                               mlir::Value a, mlir::Value c) {
        auto type = llvm::cast<mlir::FloatType>(a.getType());
        auto zero = floatConst(b, loc, type, 0.0);
        auto difference = b.create<mlir::math::AbsFOp>(
            loc, b.create<mlir::arith::SubFOp>(loc, a, c));
        auto denominator = b.create<mlir::arith::AddFOp>(
            loc, b.create<mlir::math::AbsFOp>(loc, a),
            b.create<mlir::math::AbsFOp>(loc, c));
        auto bothZero = b.create<mlir::arith::AndIOp>(
            loc,
            b.create<mlir::arith::CmpFOp>(
                loc, mlir::arith::CmpFPredicate::OEQ, a, zero),
            b.create<mlir::arith::CmpFOp>(
                loc, mlir::arith::CmpFPredicate::OEQ, c, zero));
        auto result = b.create<mlir::arith::DivFOp>(loc, difference, denominator);
        return b.create<mlir::arith::SelectOp>(loc, bothZero, zero, result);
      }};
    case VulkanKernelRecipe::LOG_POISSON_LOSS:
      return BinaryCallback{[](mlir::OpBuilder& b, mlir::Location loc,
                               mlir::Value count, mlir::Value rate) {
        return b.create<mlir::arith::SubFOp>(
            loc, emitExp(b, loc, count.getType(), rate),
            b.create<mlir::arith::MulFOp>(loc, count, rate));
      }};
    case VulkanKernelRecipe::LOG_POISSON_LOSS_FULL:
      return BinaryCallback{[](mlir::OpBuilder& b, mlir::Location loc,
                               mlir::Value count, mlir::Value rate) {
        auto type = llvm::cast<mlir::FloatType>(count.getType());
        auto base = b.create<mlir::arith::SubFOp>(
            loc, emitExp(b, loc, count.getType(), rate),
            b.create<mlir::arith::MulFOp>(loc, count, rate));
        auto logCount = emitLog(b, loc, count.getType(), count);
        auto stirling = b.create<mlir::arith::SubFOp>(
            loc, b.create<mlir::arith::MulFOp>(loc, count, logCount), count);
        auto piTimesCount = b.create<mlir::arith::MulFOp>(
            loc, floatConst(b, loc, type, 6.283185307179586), count);
        auto halfLog = b.create<mlir::arith::MulFOp>(
            loc, floatConst(b, loc, type, 0.5),
            emitLog(b, loc, count.getType(), piTimesCount));
        return b.create<mlir::arith::AddFOp>(
            loc, base, b.create<mlir::arith::AddFOp>(loc, stirling, halfLog));
      }};
    case VulkanKernelRecipe::POW_DERIVATIVE:
      return BinaryCallback{
          [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value a,
             mlir::Value c) {
            auto type = llvm::cast<mlir::FloatType>(a.getType());
            auto exponent = b.create<mlir::arith::SubFOp>(
                loc, c, floatConst(b, loc, type, 1.0));
            return b.create<mlir::arith::MulFOp>(
                loc, c, b.create<mlir::math::PowFOp>(loc, a, exponent));
          }};
    case VulkanKernelRecipe::FMOD:
      return BinaryCallback{
          [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value a,
             mlir::Value c) {
            return b.create<mlir::arith::RemFOp>(loc, a, c);
          }};
    case VulkanKernelRecipe::FLOOR_DIVIDE:
      return BinaryCallback{
          [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value a,
             mlir::Value c) {
            mlir::Value quotient =
                b.create<mlir::arith::DivFOp>(loc, a, c);
            return b.create<mlir::math::FloorOp>(loc, quotient);
          }};
    case VulkanKernelRecipe::FLOOR_MOD:
      return BinaryCallback{
          [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value a,
             mlir::Value c) {
            mlir::Value quotient =
                b.create<mlir::arith::DivFOp>(loc, a, c);
            mlir::Value floored =
                b.create<mlir::math::FloorOp>(loc, quotient);
            return b.create<mlir::arith::SubFOp>(
                loc, a, b.create<mlir::arith::MulFOp>(loc, floored, c));
          }};
    case VulkanKernelRecipe::REVERSE_DIVIDE:
      return BinaryCallback{
          [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value a,
             mlir::Value c) {
            return b.create<mlir::arith::DivFOp>(loc, c, a);
          }};
    case VulkanKernelRecipe::REVERSE_SUBTRACT:
      return BinaryCallback{
          [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value a,
             mlir::Value c) {
            return b.create<mlir::arith::SubFOp>(loc, c, a);
          }};
    case VulkanKernelRecipe::SQUARED_SUBTRACT:
      return BinaryCallback{
          [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value a,
             mlir::Value c) {
            mlir::Value difference =
                b.create<mlir::arith::SubFOp>(loc, a, c);
            return b.create<mlir::arith::MulFOp>(
                loc, difference, difference);
          }};
    case VulkanKernelRecipe::POWER:
      return BinaryCallback{
          [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value a,
             mlir::Value c) {
            return b.create<mlir::math::PowFOp>(loc, a, c);
          }};
    case VulkanKernelRecipe::ATAN2:
      return BinaryCallback{
          [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value y,
             mlir::Value x) {
            // MathToSPIRV does not legalize math.atan2 for Vulkan Shader
            // targets.  Express the CUDA/libdevice-compatible quadrant
            // contract through math.atan plus ordinary arithmetic/selects,
            // all of which lower through the shared MLIR SPIR-V pipeline.
            auto floatType = llvm::cast<mlir::FloatType>(y.getType());
            mlir::Value zero = floatConst(b, loc, floatType, 0.0);
            mlir::Value pi = floatConst(
                b, loc, floatType, 3.14159265358979323846264338327950288);
            mlir::Value halfPi = floatConst(
                b, loc, floatType, 1.57079632679489661923132169163975144);
            mlir::Value negativeHalfPi = floatConst(
                b, loc, floatType, -1.57079632679489661923132169163975144);
            mlir::Value ratio = b.create<mlir::arith::DivFOp>(loc, y, x);
            mlir::Value principal = b.create<mlir::math::AtanOp>(loc, ratio);
            mlir::Value yNonNegative = b.create<mlir::arith::CmpFOp>(
                loc, mlir::arith::CmpFPredicate::OGE, y, zero);
            mlir::Value upperQuadrant =
                b.create<mlir::arith::AddFOp>(loc, principal, pi);
            mlir::Value lowerQuadrant =
                b.create<mlir::arith::SubFOp>(loc, principal, pi);
            mlir::Value negativeX = b.create<mlir::arith::SelectOp>(
                loc, yNonNegative, upperQuadrant, lowerQuadrant);
            mlir::Value xPositive = b.create<mlir::arith::CmpFOp>(
                loc, mlir::arith::CmpFPredicate::OGT, x, zero);
            mlir::Value nonzeroX = b.create<mlir::arith::SelectOp>(
                loc, xPositive, principal, negativeX);
            mlir::Value yPositive = b.create<mlir::arith::CmpFOp>(
                loc, mlir::arith::CmpFPredicate::OGT, y, zero);
            mlir::Value yNegative = b.create<mlir::arith::CmpFOp>(
                loc, mlir::arith::CmpFPredicate::OLT, y, zero);
            mlir::Value zeroOrNegativeY = b.create<mlir::arith::SelectOp>(
                loc, yNegative, negativeHalfPi, y);
            mlir::Value zeroX = b.create<mlir::arith::SelectOp>(
                loc, yPositive, halfPi, zeroOrNegativeY);
            mlir::Value xIsZero = b.create<mlir::arith::CmpFOp>(
                loc, mlir::arith::CmpFPredicate::OEQ, x, zero);
            return b.create<mlir::arith::SelectOp>(
                loc, xIsZero, zeroX, nonzeroX);
          }};
    case VulkanKernelRecipe::SWISH_MUL:
      return BinaryCallback{
          [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value x,
             mlir::Value gate) {
            return b.create<mlir::arith::MulFOp>(
                loc, emitSilu(b, loc, x.getType(), x), gate);
          }};
    case VulkanKernelRecipe::ASSIGN:
      return BinaryCallback{
          [](mlir::OpBuilder&, mlir::Location, mlir::Value,
             mlir::Value c) {
            return c;
          }};
    default:
      return {};
  }
}

/// Emit one statically frozen fused-elementwise bytecode step.  The integer
/// bytecode comes from the op's canonical iArgs; operation names never
/// participate in lowering selection.
static mlir::Value emitFusedChainStep(
    mlir::OpBuilder& builder, mlir::Location loc, mlir::FloatType type,
    int64_t code, mlir::Value value, mlir::Value secondary,
    mlir::Value clipMin, mlir::Value clipMax) {
  mlir::Value zero = floatConst(builder, loc, type, 0.0);
  mlir::Value one = floatConst(builder, loc, type, 1.0);

  auto mappedUnary = [&](VulkanKernelRecipe semantic)
      -> mlir::Value {
    UnaryCallback callback = unaryCallbackFor(semantic);
    if (!callback) return {};
    return callback(builder, loc, type, value);
  };
  auto mappedBinary = [&](VulkanKernelRecipe semantic)
      -> mlir::Value {
    if (!secondary) return {};
    BinaryCallback callback = binaryCallbackFor(semantic);
    if (!callback) return {};
    return callback(builder, loc, value, secondary);
  };

  switch (code) {
    case sd::ops::helpers::FUSED_ADD:
      return mappedBinary(VulkanKernelRecipe::ADD);
    case sd::ops::helpers::FUSED_SUB:
      return mappedBinary(VulkanKernelRecipe::SUBTRACT);
    case sd::ops::helpers::FUSED_MUL:
      return mappedBinary(VulkanKernelRecipe::MULTIPLY);
    case sd::ops::helpers::FUSED_DIV: {
      if (!secondary) return {};
      mlir::Value divisorIsZero = builder.create<mlir::arith::CmpFOp>(
          loc, mlir::arith::CmpFPredicate::OEQ, secondary, zero);
      mlir::Value quotient =
          builder.create<mlir::arith::DivFOp>(loc, value, secondary);
      return builder.create<mlir::arith::SelectOp>(
          loc, divisorIsZero, zero, quotient);
    }
    case sd::ops::helpers::FUSED_RELU:
      return mappedUnary(VulkanKernelRecipe::RELU);
    case sd::ops::helpers::FUSED_SIGMOID:
      return mappedUnary(VulkanKernelRecipe::SIGMOID);
    case sd::ops::helpers::FUSED_TANH:
      return mappedUnary(VulkanKernelRecipe::TANH);
    case sd::ops::helpers::FUSED_GELU:
      return mappedUnary(VulkanKernelRecipe::GELU);
    case sd::ops::helpers::FUSED_EXP:
      return emitExp(builder, loc, type, value);
    case sd::ops::helpers::FUSED_LOG:
      // emitLog carries the sd_log zero-substitution; negative inputs yield
      // NaN like the native math library instead of a finite sentinel.
      return emitLog(builder, loc, type, value);
    case sd::ops::helpers::FUSED_ABS:
      return builder.create<mlir::math::AbsFOp>(loc, value);
    case sd::ops::helpers::FUSED_NEG:
      return builder.create<mlir::arith::NegFOp>(loc, value);
    case sd::ops::helpers::FUSED_SQUARE:
      return mappedUnary(VulkanKernelRecipe::SQUARE);
    case sd::ops::helpers::FUSED_SQRT: {
      mlir::Value nonnegative = builder.create<mlir::arith::CmpFOp>(
          loc, mlir::arith::CmpFPredicate::OGE, value, zero);
      return builder.create<mlir::arith::SelectOp>(
          loc, nonnegative, emitSqrt(builder, loc, type, value), zero);
    }
    case sd::ops::helpers::FUSED_SWISH:
    case sd::ops::helpers::FUSED_SILU:
      return mappedUnary(VulkanKernelRecipe::SILU);
    case sd::ops::helpers::FUSED_MISH: {
      mlir::Value softplus = emitLog(
          builder, loc, type,
          builder.create<mlir::arith::AddFOp>(
              loc, one, emitExp(builder, loc, type, value)));
      return builder.create<mlir::arith::MulFOp>(
          loc, value, emitTanh(builder, loc, type, softplus));
    }
    case sd::ops::helpers::FUSED_RSQRT:
      return emitRsqrt(builder, loc, type, value);
    case sd::ops::helpers::FUSED_RECIPROCAL:
      return builder.create<mlir::arith::DivFOp>(loc, one, value);
    case sd::ops::helpers::FUSED_SIGN: {
      mlir::Value positive = builder.create<mlir::arith::CmpFOp>(
          loc, mlir::arith::CmpFPredicate::OGT, value, zero);
      mlir::Value negative = builder.create<mlir::arith::CmpFOp>(
          loc, mlir::arith::CmpFPredicate::OLT, value, zero);
      mlir::Value signedNegative = builder.create<mlir::arith::SelectOp>(
          loc, negative, floatConst(builder, loc, type, -1.0), zero);
      return builder.create<mlir::arith::SelectOp>(
          loc, positive, one, signedNegative);
    }
    case sd::ops::helpers::FUSED_ERF:
    case sd::ops::helpers::FUSED_ERFC: {
      // Abramowitz-Stegun 7.1.26 avoids relying on a non-portable SPIR-V Erf
      // extended instruction while retaining a deterministic device kernel.
      mlir::Value abs = builder.create<mlir::math::AbsFOp>(loc, value);
      mlir::Value t = builder.create<mlir::arith::DivFOp>(
          loc, one,
          builder.create<mlir::arith::AddFOp>(
              loc, one,
              builder.create<mlir::arith::MulFOp>(
                  loc, floatConst(builder, loc, type, 0.3275911), abs)));
      mlir::Value polynomial = floatConst(builder, loc, type, 1.061405429);
      for (double coefficient :
           {-1.453152027, 1.421413741, -0.284496736, 0.254829592}) {
        polynomial = builder.create<mlir::arith::AddFOp>(
            loc, floatConst(builder, loc, type, coefficient),
            builder.create<mlir::arith::MulFOp>(loc, polynomial, t));
      }
      polynomial = builder.create<mlir::arith::MulFOp>(loc, polynomial, t);
      mlir::Value decay = emitExp(
          builder, loc, type,
          builder.create<mlir::arith::NegFOp>(
              loc, builder.create<mlir::arith::MulFOp>(loc, abs, abs)));
      mlir::Value magnitude = builder.create<mlir::arith::SubFOp>(
          loc, one,
          builder.create<mlir::arith::MulFOp>(loc, polynomial, decay));
      mlir::Value negative = builder.create<mlir::arith::CmpFOp>(
          loc, mlir::arith::CmpFPredicate::OLT, value, zero);
      mlir::Value erf = builder.create<mlir::arith::SelectOp>(
          loc, negative,
          builder.create<mlir::arith::NegFOp>(loc, magnitude), magnitude);
      return code == sd::ops::helpers::FUSED_ERF
                 ? erf
                 : mlir::Value(builder.create<mlir::arith::SubFOp>(
                       loc, one, erf));
    }
    case sd::ops::helpers::FUSED_LOG1P:
      return emitLog(
          builder, loc, type,
          builder.create<mlir::arith::AddFOp>(loc, one, value));
    case sd::ops::helpers::FUSED_CEIL:
      return builder.create<mlir::math::CeilOp>(loc, value);
    case sd::ops::helpers::FUSED_CLIP:
      if (!clipMin || !clipMax) return {};
      return emitClipWithValues(builder, loc, type, value, clipMin, clipMax,
                                false);
    case sd::ops::helpers::FUSED_LEAKY_RELU: {
      if (!secondary) return {};
      mlir::Value nonnegative = builder.create<mlir::arith::CmpFOp>(
          loc, mlir::arith::CmpFPredicate::OGE, value, zero);
      return builder.create<mlir::arith::SelectOp>(
          loc, nonnegative, value,
          builder.create<mlir::arith::MulFOp>(loc, value, secondary));
    }
    case sd::ops::helpers::FUSED_FLOOR:
      return builder.create<mlir::math::FloorOp>(loc, value);
    case sd::ops::helpers::FUSED_ROUND:
      return builder.create<mlir::math::RoundEvenOp>(loc, value);
    case sd::ops::helpers::FUSED_SIN:
      return builder.create<mlir::math::SinOp>(loc, value);
    case sd::ops::helpers::FUSED_COS:
      return builder.create<mlir::math::CosOp>(loc, value);
    case sd::ops::helpers::FUSED_ELU:
      return mappedUnary(VulkanKernelRecipe::ELU);
    case sd::ops::helpers::FUSED_SELU:
      return mappedUnary(VulkanKernelRecipe::SELU);
    case sd::ops::helpers::FUSED_SOFTPLUS:
      return mappedUnary(VulkanKernelRecipe::SOFTPLUS);
    case sd::ops::helpers::FUSED_SOFTSIGN:
      return mappedUnary(VulkanKernelRecipe::SOFTSIGN);
    case sd::ops::helpers::FUSED_HARD_SIGMOID: {
      mlir::Value shifted = builder.create<mlir::arith::AddFOp>(
          loc,
          builder.create<mlir::arith::DivFOp>(
              loc, value, floatConst(builder, loc, type, 6.0)),
          floatConst(builder, loc, type, 0.5));
      return emitClipWithValues(builder, loc, type, shifted, zero, one, false);
    }
    case sd::ops::helpers::FUSED_HARDTANH:
      return mappedUnary(VulkanKernelRecipe::HARD_TANH);
    case sd::ops::helpers::FUSED_RELU6:
      return mappedUnary(VulkanKernelRecipe::RELU6);
    case sd::ops::helpers::FUSED_MIN:
      return mappedBinary(VulkanKernelRecipe::MINIMUM);
    case sd::ops::helpers::FUSED_MAX:
      return mappedBinary(VulkanKernelRecipe::MAXIMUM);
    case sd::ops::helpers::FUSED_MOD:
      return mappedBinary(VulkanKernelRecipe::MOD);
    case sd::ops::helpers::FUSED_ATAN2:
      return mappedBinary(VulkanKernelRecipe::ATAN2);
    case sd::ops::helpers::FUSED_FLOORDIV:
      return mappedBinary(VulkanKernelRecipe::FLOOR_DIVIDE);
    case sd::ops::helpers::FUSED_REVERSE_DIV:
      return mappedBinary(VulkanKernelRecipe::REVERSE_DIVIDE);
    case sd::ops::helpers::FUSED_REVERSE_SUB:
      return mappedBinary(VulkanKernelRecipe::REVERSE_SUBTRACT);
    case sd::ops::helpers::FUSED_SQUARED_SUB:
      return mappedBinary(VulkanKernelRecipe::SQUARED_SUBTRACT);
    case sd::ops::helpers::FUSED_MUL_NO_NAN: {
      if (!secondary) return {};
      mlir::Value rhsIsZero = builder.create<mlir::arith::CmpFOp>(
          loc, mlir::arith::CmpFPredicate::OEQ, secondary, zero);
      return builder.create<mlir::arith::SelectOp>(
          loc, rhsIsZero, zero,
          builder.create<mlir::arith::MulFOp>(loc, value, secondary));
    }
    case sd::ops::helpers::FUSED_POW:
      return mappedBinary(VulkanKernelRecipe::POWER);
    default:
      return {};
  }
}

static mlir::Value integerFloorDivide(mlir::OpBuilder& builder,
                                      mlir::Location loc,
                                      mlir::Value lhs, mlir::Value rhs,
                                      bool isUnsigned) {
  if (isUnsigned) {
    return builder.create<mlir::arith::DivUIOp>(loc, lhs, rhs);
  }
  mlir::Value quotient = builder.create<mlir::arith::DivSIOp>(
      loc, lhs, rhs);
  mlir::Value remainder = builder.create<mlir::arith::RemSIOp>(
      loc, lhs, rhs);
  auto integerType = llvm::cast<mlir::IntegerType>(lhs.getType());
  mlir::Value zero = builder.create<mlir::arith::ConstantIntOp>(
      loc, 0, integerType.getWidth());
  mlir::Value one = builder.create<mlir::arith::ConstantIntOp>(
      loc, 1, integerType.getWidth());
  mlir::Value hasRemainder = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::ne, remainder, zero);
  mlir::Value remainderNegative = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::slt, remainder, zero);
  mlir::Value divisorNegative = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::slt, rhs, zero);
  mlir::Value signsDiffer = builder.create<mlir::arith::XOrIOp>(
      loc, remainderNegative, divisorNegative);
  mlir::Value adjust = builder.create<mlir::arith::AndIOp>(
      loc, hasRemainder, signsDiffer);
  mlir::Value adjusted = builder.create<mlir::arith::SubIOp>(
      loc, quotient, one);
  return builder.create<mlir::arith::SelectOp>(
      loc, adjust, adjusted, quotient);
}

static mlir::Value emitIntegerBinary(mlir::OpBuilder& builder,
                                     mlir::Location loc,
                                     VulkanKernelRecipe semantic,
                                     mlir::Value lhs, mlir::Value rhs,
                                     bool isUnsigned) {
  switch (semantic) {
    case VulkanKernelRecipe::EQUAL:
      return builder.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::eq, lhs, rhs);
    case VulkanKernelRecipe::NOT_EQUAL:
      return builder.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::ne, lhs, rhs);
    case VulkanKernelRecipe::LESS:
      return builder.create<mlir::arith::CmpIOp>(
          loc, isUnsigned ? mlir::arith::CmpIPredicate::ult
                          : mlir::arith::CmpIPredicate::slt,
          lhs, rhs);
    case VulkanKernelRecipe::LESS_EQUAL:
      return builder.create<mlir::arith::CmpIOp>(
          loc, isUnsigned ? mlir::arith::CmpIPredicate::ule
                          : mlir::arith::CmpIPredicate::sle,
          lhs, rhs);
    case VulkanKernelRecipe::GREATER:
      return builder.create<mlir::arith::CmpIOp>(
          loc, isUnsigned ? mlir::arith::CmpIPredicate::ugt
                          : mlir::arith::CmpIPredicate::sgt,
          lhs, rhs);
    case VulkanKernelRecipe::GREATER_EQUAL:
      return builder.create<mlir::arith::CmpIOp>(
          loc, isUnsigned ? mlir::arith::CmpIPredicate::uge
                          : mlir::arith::CmpIPredicate::sge,
          lhs, rhs);
    case VulkanKernelRecipe::BOOLEAN_AND:
      return builder.create<mlir::arith::AndIOp>(loc, lhs, rhs);
    case VulkanKernelRecipe::BOOLEAN_OR:
      return builder.create<mlir::arith::OrIOp>(loc, lhs, rhs);
    case VulkanKernelRecipe::BOOLEAN_XOR:
      return builder.create<mlir::arith::XOrIOp>(loc, lhs, rhs);
    case VulkanKernelRecipe::LOGICAL_NOT_BINARY: {
      auto integerType = llvm::cast<mlir::IntegerType>(lhs.getType());
      auto zero = builder.create<mlir::arith::ConstantIntOp>(loc, 0, integerType.getWidth());
      auto one = builder.create<mlir::arith::ConstantIntOp>(loc, 1, integerType.getWidth());
      auto lhsTrue = builder.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::ne, lhs, zero);
      auto rhsTrue = builder.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::ne, rhs, zero);
      auto both = builder.create<mlir::arith::AndIOp>(loc, lhsTrue, rhsTrue);
      return builder.create<mlir::arith::SelectOp>(loc, both, zero, one);
    }
    case VulkanKernelRecipe::SHIFT_LEFT:
      return builder.create<mlir::arith::ShLIOp>(loc, lhs, rhs);
    case VulkanKernelRecipe::SHIFT_RIGHT:
      return isUnsigned
                 ? mlir::Value(builder.create<mlir::arith::ShRUIOp>(
                       loc, lhs, rhs))
                 : mlir::Value(builder.create<mlir::arith::ShRSIOp>(
                       loc, lhs, rhs));
    case VulkanKernelRecipe::CYCLIC_SHIFT_LEFT:
    case VulkanKernelRecipe::CYCLIC_SHIFT_RIGHT: {
      auto type = llvm::cast<mlir::IntegerType>(lhs.getType());
      const unsigned width = type.getWidth();
      mlir::Value widthMask = builder.create<mlir::arith::ConstantIntOp>(
          loc, static_cast<int64_t>(width - 1), type.getWidth());
      mlir::Value amount = builder.create<mlir::arith::AndIOp>(
          loc, rhs, widthMask);
      mlir::Value zero = builder.create<mlir::arith::ConstantIntOp>(loc, 0, type.getWidth());
      mlir::Value widthValue = builder.create<mlir::arith::ConstantIntOp>(
          loc, static_cast<int64_t>(width), type.getWidth());
      mlir::Value opposite = builder.create<mlir::arith::SubIOp>(
          loc, widthValue, amount);
      mlir::Value left = builder.create<mlir::arith::ShLIOp>(loc, lhs, amount);
      mlir::Value right = isUnsigned
                              ? mlir::Value(builder.create<mlir::arith::ShRUIOp>(
                                    loc, lhs, opposite))
                              : mlir::Value(builder.create<mlir::arith::ShRSIOp>(
                                    loc, lhs, opposite));
      mlir::Value rotated = semantic == VulkanKernelRecipe::CYCLIC_SHIFT_LEFT
                                ? builder.create<mlir::arith::OrIOp>(loc, left, right)
                                : builder.create<mlir::arith::OrIOp>(
                                      loc,
                                      isUnsigned
                                          ? mlir::Value(builder.create<mlir::arith::ShRUIOp>(
                                                loc, lhs, amount))
                                          : mlir::Value(builder.create<mlir::arith::ShRSIOp>(
                                                loc, lhs, amount)),
                                      builder.create<mlir::arith::ShLIOp>(
                                          loc, lhs, opposite));
      auto amountZero = builder.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::eq, amount, zero);
      return builder.create<mlir::arith::SelectOp>(loc, amountZero, lhs, rotated);
    }
    case VulkanKernelRecipe::ADD:
      return builder.create<mlir::arith::AddIOp>(loc, lhs, rhs);
    case VulkanKernelRecipe::SUBTRACT:
      return builder.create<mlir::arith::SubIOp>(loc, lhs, rhs);
    case VulkanKernelRecipe::MULTIPLY:
      return builder.create<mlir::arith::MulIOp>(loc, lhs, rhs);
    case VulkanKernelRecipe::DIVIDE:
      return isUnsigned
                 ? mlir::Value(builder.create<mlir::arith::DivUIOp>(
                       loc, lhs, rhs))
                 : mlir::Value(builder.create<mlir::arith::DivSIOp>(
                       loc, lhs, rhs));
    case VulkanKernelRecipe::MINIMUM:
      return isUnsigned
                 ? mlir::Value(builder.create<mlir::arith::MinUIOp>(
                       loc, lhs, rhs))
                 : mlir::Value(builder.create<mlir::arith::MinSIOp>(
                       loc, lhs, rhs));
    case VulkanKernelRecipe::MAXIMUM:
      return isUnsigned
                 ? mlir::Value(builder.create<mlir::arith::MaxUIOp>(
                       loc, lhs, rhs))
                 : mlir::Value(builder.create<mlir::arith::MaxSIOp>(
                       loc, lhs, rhs));
    case VulkanKernelRecipe::MOD:
      return isUnsigned
                 ? mlir::Value(builder.create<mlir::arith::RemUIOp>(
                       loc, lhs, rhs))
                 : mlir::Value(builder.create<mlir::arith::RemSIOp>(
                       loc, lhs, rhs));
    case VulkanKernelRecipe::FLOOR_DIVIDE:
      return integerFloorDivide(builder, loc, lhs, rhs, isUnsigned);
    case VulkanKernelRecipe::TRUNCATE_DIV:
      return isUnsigned
                 ? mlir::Value(builder.create<mlir::arith::DivUIOp>(
                       loc, lhs, rhs))
                 : mlir::Value(builder.create<mlir::arith::DivSIOp>(
                       loc, lhs, rhs));
    case VulkanKernelRecipe::SAFE_DIVIDE:
    case VulkanKernelRecipe::DIVIDE_NO_NAN: {
      mlir::Value zero = builder.create<mlir::arith::ConstantIntOp>(
          loc, 0, llvm::cast<mlir::IntegerType>(rhs.getType()).getWidth());
      mlir::Value denominatorZero = builder.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::eq, rhs, zero);
      mlir::Value quotient = isUnsigned
                                 ? mlir::Value(builder.create<mlir::arith::DivUIOp>(
                                       loc, lhs, rhs))
                                 : mlir::Value(builder.create<mlir::arith::DivSIOp>(
                                       loc, lhs, rhs));
      return builder.create<mlir::arith::SelectOp>(loc, denominatorZero, zero,
                                                   quotient);
    }
    case VulkanKernelRecipe::REVERSE_MOD:
      return isUnsigned
                 ? mlir::Value(builder.create<mlir::arith::RemUIOp>(
                       loc, rhs, lhs))
                 : mlir::Value(builder.create<mlir::arith::RemSIOp>(
                       loc, rhs, lhs));
    case VulkanKernelRecipe::FLOOR_MOD: {
      mlir::Value quotient = integerFloorDivide(
          builder, loc, lhs, rhs, isUnsigned);
      return builder.create<mlir::arith::SubIOp>(
          loc, lhs,
          builder.create<mlir::arith::MulIOp>(loc, quotient, rhs));
    }
    case VulkanKernelRecipe::REVERSE_DIVIDE:
      return isUnsigned
                 ? mlir::Value(builder.create<mlir::arith::DivUIOp>(
                       loc, rhs, lhs))
                 : mlir::Value(builder.create<mlir::arith::DivSIOp>(
                       loc, rhs, lhs));
    case VulkanKernelRecipe::REVERSE_SUBTRACT:
      return builder.create<mlir::arith::SubIOp>(loc, rhs, lhs);
    case VulkanKernelRecipe::SQUARED_SUBTRACT: {
      mlir::Value difference = builder.create<mlir::arith::SubIOp>(
          loc, lhs, rhs);
      return builder.create<mlir::arith::MulIOp>(
          loc, difference, difference);
    }
    case VulkanKernelRecipe::ASSIGN:
      return rhs;
    default:
      return {};
  }
}

struct ReductionCallbacks {
  double initialValue;
  BinaryCallback combine;
  BinaryCallback finalize;
};

static ReductionCallbacks reductionCallbacksFor(
    VulkanKernelRecipe semantic) {
  auto identity = BinaryCallback{
      [](mlir::OpBuilder&, mlir::Location, mlir::Value acc, mlir::Value) {
        return acc;
      }};
  auto add = BinaryCallback{
      [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value acc,
         mlir::Value value) {
        return b.create<mlir::arith::AddFOp>(loc, acc, value);
      }};
  switch (semantic) {
    case VulkanKernelRecipe::REDUCE_SUM:
      return ReductionCallbacks{0.0, add, identity};
    case VulkanKernelRecipe::REDUCE_LOGSUMEXP:
      return ReductionCallbacks{
          -std::numeric_limits<double>::infinity(),
          BinaryCallback{
              [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value acc,
                 mlir::Value value) {
                return b.create<mlir::arith::MaximumFOp>(loc, acc, value);
              }},
          identity};
    case VulkanKernelRecipe::REDUCE_VARIANCE:
    case VulkanKernelRecipe::REDUCE_STDEV:
      return ReductionCallbacks{0.0, add, identity};
    case VulkanKernelRecipe::REDUCE_MAX:
      return ReductionCallbacks{
          -std::numeric_limits<double>::infinity(),
          BinaryCallback{
              [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value acc,
                 mlir::Value value) {
                return b.create<mlir::arith::MaximumFOp>(loc, acc, value);
              }},
          identity};
    case VulkanKernelRecipe::REDUCE_MIN:
      return ReductionCallbacks{
          std::numeric_limits<double>::infinity(),
          BinaryCallback{
              [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value acc,
                 mlir::Value value) {
                return b.create<mlir::arith::MinimumFOp>(loc, acc, value);
              }},
          identity};
    case VulkanKernelRecipe::REDUCE_PRODUCT:
      return ReductionCallbacks{
          1.0,
          BinaryCallback{
              [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value acc,
                 mlir::Value value) {
                return b.create<mlir::arith::MulFOp>(loc, acc, value);
              }},
          identity};
    case VulkanKernelRecipe::REDUCE_ENTROPY:
    case VulkanKernelRecipe::REDUCE_LOG_ENTROPY:
    case VulkanKernelRecipe::REDUCE_SHANNON_ENTROPY:
      return ReductionCallbacks{0.0, add, identity};
    case VulkanKernelRecipe::REDUCE_COUNT_NONZERO:
    case VulkanKernelRecipe::REDUCE_COUNT_ZERO:
    case VulkanKernelRecipe::REDUCE_COUNT_MATCH:
      return ReductionCallbacks{0.0, add, identity};
    default:
      return ReductionCallbacks{0.0, {}, {}};
  }
}

static mlir::Value integerReductionInitial(
    mlir::OpBuilder& builder, mlir::Location loc,
    VulkanKernelRecipe semantic, mlir::IntegerType type, bool isUnsigned) {
  int64_t value = 0;
  switch (semantic) {
    case VulkanKernelRecipe::REDUCE_SUM:
    case VulkanKernelRecipe::REDUCE_COUNT_NONZERO:
    case VulkanKernelRecipe::REDUCE_COUNT_ZERO:
    case VulkanKernelRecipe::REDUCE_COUNT_MATCH:
      value = 0;
      break;
    case VulkanKernelRecipe::REDUCE_PRODUCT:
      value = 1;
      break;
    case VulkanKernelRecipe::REDUCE_MAX:
      value = isUnsigned ? 0 : std::numeric_limits<int32_t>::min();
      break;
    case VulkanKernelRecipe::REDUCE_MIN:
      value = isUnsigned ? -1 : std::numeric_limits<int32_t>::max();
      break;
    default:
      return {};
  }
  return builder.create<mlir::arith::ConstantIntOp>(
      loc, value, type.getWidth());
}

static mlir::Value emitIntegerReductionCombine(
    mlir::OpBuilder& builder, mlir::Location loc,
    VulkanKernelRecipe semantic, mlir::Value accumulator,
    mlir::Value value, bool isUnsigned) {
  switch (semantic) {
    case VulkanKernelRecipe::REDUCE_SUM:
    case VulkanKernelRecipe::REDUCE_COUNT_NONZERO:
    case VulkanKernelRecipe::REDUCE_COUNT_ZERO:
    case VulkanKernelRecipe::REDUCE_COUNT_MATCH:
      return builder.create<mlir::arith::AddIOp>(
          loc, accumulator, value);
    case VulkanKernelRecipe::REDUCE_PRODUCT:
      return builder.create<mlir::arith::MulIOp>(
          loc, accumulator, value);
    case VulkanKernelRecipe::REDUCE_MAX:
      return isUnsigned
                 ? mlir::Value(builder.create<mlir::arith::MaxUIOp>(
                       loc, accumulator, value))
                 : mlir::Value(builder.create<mlir::arith::MaxSIOp>(
                       loc, accumulator, value));
    case VulkanKernelRecipe::REDUCE_MIN:
      return isUnsigned
                 ? mlir::Value(builder.create<mlir::arith::MinUIOp>(
                       loc, accumulator, value))
                 : mlir::Value(builder.create<mlir::arith::MinSIOp>(
                       loc, accumulator, value));
    default:
      return {};
  }
}

static bool hasRegisteredLowering(sd::LongType hash) {
  // The catalogue is the lowering claim. Once a canonical descriptor hash is
  // registered, conversion must eliminate it or fail; a second per-op allow
  // list here would silently drift from descriptor traits.
  return findVulkanKernelEmitter(hash) != nullptr;
}

/// Flatten a multi-dimensional MemRef into a 1-D index given a dynamic index.
/// For the serial (1,1,1) Wave 1 kernel, we iterate over a flat index [0, N)
/// and use a single LoadOp / StoreOp on the 1-D memref.
/// This helper is used in binary/unary patterns.
static mlir::Value castToIndex(mlir::OpBuilder& b, mlir::Location loc, mlir::Value v) {
  // v might already be index type (from ConstantIndexOp) or might be i64.
  if (v.getType().isIndex()) return v;
  return b.create<mlir::arith::IndexCastOp>(loc, b.getIndexType(), v);
}

}  // anonymous namespace

// ─────────────────────────────────────────────────────────────────────────────
//  MatmulToSpirv
// ─────────────────────────────────────────────────────────────────────────────
//
// We lower linalg.matmul to a nest of scf.for loops that is subsequently
// converted to SPIR-V by the standard SCF→SPIR-V and Arith→SPIR-V passes.
//
// The three-level loop nest C[m,n] += A[m,k] * B[k,n] maps naturally to
// Vulkan compute workgroups:
//   - Outer two loops (m, n) become the global invocation grid
//     (gl_GlobalInvocationID.y = m, gl_GlobalInvocationID.x = n).
//   - The K-reduction loop executes within each invocation.
//
// We emit the loops in pure SCF/Arith dialect here.  The downstream
// SCF→SPIR-V pass translates scf.for + scf.yield into SPIR-V structured
// control flow.
//
// Error reporting: any structural mismatch (wrong number of operands, unsupported
// element type, missing output) is surfaced as an op error with the diagnostic and
// element type so the pipeline diagnosis output identifies the source.

mlir::LogicalResult MatmulToSpirv::matchAndRewrite(
    mlir::linalg::MatmulOp op,
    mlir::PatternRewriter& rewriter) const {

  mlir::Location loc = op.getLoc();

  const auto* emitter = emitterForOperation(op);
  if (emitter == nullptr || !usesDenseMatrixProductSchedule(*emitter)) {
    return mlir::failure();
  }

  // ── 1. Extract operands ───────────────────────────────────────────────────

  mlir::ValueRange inputs  = op.getInputs();
  mlir::ValueRange outputs = op.getOutputs();
  auto typeContract = getComputeTypeContract(op, inputs, outputs);
  if (mlir::failed(typeContract)) return mlir::failure();
  mlir::FloatType elemTy = typeContract->accumulatorType;

  if (inputs.size() < 2 || outputs.size() < 1) {
    return op.emitOpError(
        "MatmulToSpirv: expected 2 inputs and 1 output for linalg.matmul "
        "(op=matmul)");
  }

  mlir::Value A = inputs[0];
  mlir::Value B = inputs[1];
  mlir::Value C = outputs[0];


  // ── 3. Extract dynamic dimensions ────────────────────────────────────────

  mlir::Value M = rewriter.create<mlir::memref::DimOp>(loc, A, 0);
  mlir::Value K = rewriter.create<mlir::memref::DimOp>(loc, A, 1);
  mlir::Value N = rewriter.create<mlir::memref::DimOp>(loc, B, 1);
  mlir::Value zeroIdx = idxConst(rewriter, loc, 0);
  mlir::Value oneIdx = idxConst(rewriter, loc, 1);

  // ── 4. Emit a real GPU kernel launch ──────────────────────────────────────
  // One Vulkan invocation computes one output element. Kernel outlining turns
  // this gpu.launch into a gpu.func whose captured memrefs become Vulkan storage
  // buffer bindings; the official GPU-to-SPIR-V pass then creates spirv.module.
  auto launch = mlir::gpu::LaunchOp::create(
      rewriter, loc,
      /*gridSizeX=*/N, /*gridSizeY=*/M, /*gridSizeZ=*/oneIdx,
      /*blockSizeX=*/oneIdx, /*blockSizeY=*/oneIdx, /*blockSizeZ=*/oneIdx);
  launch.setFunctionAttr(mlir::FlatSymbolRefAttr::get(rewriter.getContext(), "main"));

  mlir::Value ni = launch.getBlockIds().x;
  mlir::Value mi = launch.getBlockIds().y;
  rewriter.setInsertionPointToEnd(&launch.getBody().front());

  mlir::Value initZero = floatConst(rewriter, loc, elemTy, 0.0);
  auto kLoop = emitReductionLoop(
      rewriter, loc, zeroIdx, K, oneIdx, initZero,
      [&](mlir::OpBuilder& kb, mlir::Location kloc,
          mlir::Value ki, mlir::Value acc) -> mlir::Value {
        mlir::Value aVal = loadAsAccumulator(kb, kloc, A, mlir::SmallVector<mlir::Value>{mi, ki}, elemTy);
        mlir::Value bVal = loadAsAccumulator(kb, kloc, B, mlir::SmallVector<mlir::Value>{ki, ni}, elemTy);
        mlir::Value prod = kb.create<mlir::arith::MulFOp>(kloc, aVal, bVal);
        return kb.create<mlir::arith::AddFOp>(kloc, acc, prod);
      });

  storeFromAccumulator(rewriter, loc, kLoop.getResult(0), C, mlir::SmallVector<mlir::Value>{mi, ni});
  rewriter.create<mlir::gpu::TerminatorOp>(loc);

  // ── 5. Erase the original linalg.matmul ──────────────────────────────────

  rewriter.eraseOp(op);
  return mlir::success();
}

mlir::LogicalResult BatchMatmulToSpirv::matchAndRewrite(
    mlir::linalg::BatchMatmulOp op,
    mlir::PatternRewriter& rewriter) const {
  mlir::Location loc = op.getLoc();

  const auto* emitter = emitterForOperation(op);
  if (emitter == nullptr || !usesDenseMatrixProductSchedule(*emitter)) {
    return mlir::failure();
  }

  mlir::ValueRange inputs = op.getInputs();
  mlir::ValueRange outputs = op.getOutputs();
  auto typeContract = getComputeTypeContract(op, inputs, outputs);
  if (mlir::failed(typeContract)) return mlir::failure();
  if (inputs.size() != 2 || outputs.size() != 1) {
    return op.emitOpError(
        "BatchMatmulToSpirv: expected 2 inputs and 1 output");
  }

  mlir::Value A = inputs[0];
  mlir::Value B = inputs[1];
  mlir::Value C = outputs[0];
  mlir::FloatType accTy = typeContract->accumulatorType;
  mlir::Value batches = rewriter.create<mlir::memref::DimOp>(loc, A, 0);
  mlir::Value rows = rewriter.create<mlir::memref::DimOp>(loc, A, 1);
  mlir::Value reduction = rewriter.create<mlir::memref::DimOp>(loc, A, 2);
  mlir::Value columns = rewriter.create<mlir::memref::DimOp>(loc, B, 2);
  mlir::Value zeroIdx = idxConst(rewriter, loc, 0);
  mlir::Value oneIdx = idxConst(rewriter, loc, 1);

  auto launch = createGpuLaunch(rewriter, loc, columns, rows, batches);
  mlir::Value column = launch.getBlockIds().x;
  mlir::Value row = launch.getBlockIds().y;
  mlir::Value batch = launch.getBlockIds().z;
  rewriter.setInsertionPointToEnd(&launch.getBody().front());

  mlir::Value initZero = floatConst(rewriter, loc, accTy, 0.0);
  auto reductionLoop = emitReductionLoop(
      rewriter, loc, zeroIdx, reduction, oneIdx, initZero,
      [&](mlir::OpBuilder& body, mlir::Location bodyLoc,
          mlir::Value inner, mlir::Value accumulator) -> mlir::Value {
        mlir::Value lhs = loadAsAccumulator(
            body, bodyLoc, A,
            mlir::SmallVector<mlir::Value>{batch, row, inner}, accTy);
        mlir::Value rhs = loadAsAccumulator(
            body, bodyLoc, B,
            mlir::SmallVector<mlir::Value>{batch, inner, column}, accTy);
        mlir::Value product =
            body.create<mlir::arith::MulFOp>(bodyLoc, lhs, rhs);
        return body.create<mlir::arith::AddFOp>(
            bodyLoc, accumulator, product);
      });

  storeFromAccumulator(
      rewriter, loc, reductionLoop.getResult(0), C,
      mlir::SmallVector<mlir::Value>{batch, row, column});
  rewriter.create<mlir::gpu::TerminatorOp>(loc);
  rewriter.eraseOp(op);
  return mlir::success();
}

// ─────────────────────────────────────────────────────────────────────────────
//  RmsNormToSpirv
// ─────────────────────────────────────────────────────────────────────────────
//
// Lowering strategy:
//
//   For each row m in [0, B*S):
//     variance = Σ(x[m, k]^2, k in [0, D)) / D
//     norm_factor = rsqrt(variance + epsilon)
//     y[m, k] = x[m, k] * norm_factor * gamma[k]   (if gamma present)
//             = x[m, k] * norm_factor               (otherwise)
//
// Pattern guard: match the operation descriptor hash.

mlir::LogicalResult RmsNormToSpirv::matchAndRewrite(
    mlir::linalg::GenericOp op,
    mlir::PatternRewriter& rewriter) const {

  mlir::Location loc = op.getLoc();

  const auto* emitter = emitterForOperation(op);
  if (emitter == nullptr ||
      !usesRowwiseEpsilonNormalizationSchedule(*emitter)) {
    return mlir::failure();
  }

  // ── 2. Extract operands ───────────────────────────────────────────────────

  mlir::ValueRange inputs  = op.getInputs();
  mlir::ValueRange outputs = op.getOutputs();
  auto typeContract = getComputeTypeContract(op, inputs, outputs);
  if (mlir::failed(typeContract)) return mlir::failure();
  mlir::FloatType elemTy = typeContract->accumulatorType;

  if (inputs.empty() || outputs.empty()) {
    return op.emitOpError(
        "RmsNormToSpirv: expected at least 1 input and 1 output (op=rms_norm)");
  }

  mlir::Value X = inputs[0];
  mlir::Value Y = outputs[0];
  // Optional gamma scale (second input if present)

  bool hasGamma = (inputs.size() >= 2);
  mlir::Value gamma = hasGamma ? inputs[1] : mlir::Value{};


  // ── 4. Read epsilon attribute (default 1e-6) ──────────────────────────────

  float epsilonVal = kDefaultEpsilon;
  if (auto epAttr = op->getAttrOfType<mlir::FloatAttr>(kEpsilonAttrName)) {
    epsilonVal = static_cast<float>(epAttr.getValueAsDouble());
  }

  // ── 5. Emit lowered code ──────────────────────────────────────────────────

  mlir::Value zeroIdx = idxConst(rewriter, loc, 0);
  mlir::Value oneIdx  = idxConst(rewriter, loc, 1);

  // Determine dimensions [rows, hidden]
  mlir::Value numRows   = rewriter.create<mlir::memref::DimOp>(loc, X, 0);
  mlir::Value hiddenDim = rewriter.create<mlir::memref::DimOp>(loc, X, 1);

  // Convert hiddenDim (index) to floatTy for the reciprocal
  mlir::Value hiddenFloat =
      convertIndexToFloat(rewriter, loc, hiddenDim, elemTy);

  mlir::Value epsilonConst = floatConst(rewriter, loc, elemTy, epsilonVal);

  // One workgroup invocation owns one complete row.  The serial inner
  // reduction is race-free and remains in AccT, while independent rows run in
  // parallel as real Vulkan invocations.
  auto launch = createGpuLaunch(rewriter, loc, numRows, oneIdx, oneIdx);
  mlir::Value rowIdx = launch.getBlockIds().x;
  rewriter.setInsertionPointToEnd(&launch.getBody().front());

  mlir::Value initZero = floatConst(rewriter, loc, elemTy, 0.0);
  auto ssLoop = emitReductionLoop(
      rewriter, loc, zeroIdx, hiddenDim, oneIdx, initZero,
      [&](mlir::OpBuilder& kb, mlir::Location kloc,
          mlir::Value ki, mlir::Value acc) -> mlir::Value {
        mlir::Value xVal = loadAsAccumulator(
            kb, kloc, X, mlir::SmallVector<mlir::Value>{rowIdx, ki}, elemTy);
        mlir::Value sq = kb.create<mlir::arith::MulFOp>(kloc, xVal, xVal);
        return kb.create<mlir::arith::AddFOp>(kloc, acc, sq);
      });

  mlir::Value sumSq = ssLoop.getResult(0);
  mlir::Value variance =
      rewriter.create<mlir::arith::DivFOp>(loc, sumSq, hiddenFloat);
  mlir::Value varPlusEps =
      rewriter.create<mlir::arith::AddFOp>(loc, variance, epsilonConst);
  mlir::Value normFactor = emitRsqrt(rewriter, loc, elemTy, varPlusEps);

  rewriter.create<mlir::scf::ForOp>(
      loc, zeroIdx, hiddenDim, oneIdx, mlir::ValueRange{},
      [&](mlir::OpBuilder& ab, mlir::Location aloc, mlir::Value ki,
          mlir::ValueRange) {
        mlir::Value xVal = loadAsAccumulator(
            ab, aloc, X, mlir::SmallVector<mlir::Value>{rowIdx, ki}, elemTy);
        mlir::Value normed =
            ab.create<mlir::arith::MulFOp>(aloc, xVal, normFactor);
        if (hasGamma) {
          mlir::Value gVal = loadAsAccumulator(
              ab, aloc, gamma, mlir::SmallVector<mlir::Value>{ki}, elemTy);
          normed = ab.create<mlir::arith::MulFOp>(aloc, normed, gVal);
        }
        storeFromAccumulator(
            ab, aloc, normed, Y,
            mlir::SmallVector<mlir::Value>{rowIdx, ki});
        ab.create<mlir::scf::YieldOp>(aloc);
      });
  rewriter.create<mlir::gpu::TerminatorOp>(loc);

  rewriter.eraseOp(op);
  return mlir::success();
}

// ─────────────────────────────────────────────────────────────────────────────
//  RopeToSpirv
// ─────────────────────────────────────────────────────────────────────────────
//
// Rotary Position Embedding lowering.
//
// We implement the standard pair-rotation formula:
//   x1 = X[b, s, h, 2*i],   x2 = X[b, s, h, 2*i+1]
//   Y[b, s, h, 2*i]   = x1 * cos[s, i] - x2 * sin[s, i]
//   Y[b, s, h, 2*i+1] = x2 * cos[s, i] + x1 * sin[s, i]
//
// cos/sin are read from frozen storage buffers (not computed via transcendental
// SPIR-V built-ins) to ensure deterministic, bit-exact outputs across replay.
//
// Pattern guard: match the operation descriptor hash.

mlir::LogicalResult RopeToSpirv::matchAndRewrite(
    mlir::linalg::GenericOp op,
    mlir::PatternRewriter& rewriter) const {

  mlir::Location loc = op.getLoc();

  const auto* emitter = emitterForOperation(op);
  if (emitter == nullptr || !usesCachedRotarySchedule(*emitter)) {
    return mlir::failure();
  }

  // ── 2. Extract operands (X, cos_table, sin_table → Y) ────────────────────

  mlir::ValueRange inputs  = op.getInputs();
  mlir::ValueRange outputs = op.getOutputs();
  auto typeContract = getComputeTypeContract(op, inputs, outputs);
  if (mlir::failed(typeContract)) return mlir::failure();
  mlir::FloatType elemTy = typeContract->accumulatorType;

  if (inputs.size() < 3 || outputs.empty()) {
    return op.emitOpError(
        "RopeToSpirv: expected 3 inputs (X, cos, sin) and 1 output "
        "(op=rope, inputs=" + std::to_string(inputs.size()) + ")");
  }

  mlir::Value X        = inputs[0];
  mlir::Value cosTable = inputs[1];
  mlir::Value sinTable = inputs[2];
  mlir::Value Y        = outputs[0];


  // ── 4. Extract dimension bounds ───────────────────────────────────────────
  // Input shape: [B, S, H, D]
  mlir::Value dimB = rewriter.create<mlir::memref::DimOp>(loc, X, 0);
  mlir::Value dimS = rewriter.create<mlir::memref::DimOp>(loc, X, 1);
  mlir::Value dimH = rewriter.create<mlir::memref::DimOp>(loc, X, 2);
  mlir::Value dimD = rewriter.create<mlir::memref::DimOp>(loc, X, 3);

  // Rotary dim = D/2 (number of pairs)
  mlir::Value two       = idxConst(rewriter, loc, 2);
  mlir::Value rotaryDim = rewriter.create<mlir::arith::DivUIOp>(loc, dimD, two);

  mlir::Value oneIdx  = idxConst(rewriter, loc, 1);

  // ── 5. Dispatch one invocation per rotary pair ───────────────────────────
  // grid = [(D / 2), H, B * S].  Flattening B/S into z keeps the exact rank-4
  // logical coordinates while matching Vulkan's three dispatch dimensions.
  mlir::Value batchSequence =
      rewriter.create<mlir::arith::MulIOp>(loc, dimB, dimS);
  auto launch =
      createGpuLaunch(rewriter, loc, rotaryDim, dimH, batchSequence);
  mlir::Value pi = launch.getBlockIds().x;
  mlir::Value hi = launch.getBlockIds().y;
  mlir::Value flatBatchSequence = launch.getBlockIds().z;
  rewriter.setInsertionPointToEnd(&launch.getBody().front());

  mlir::Value bi = rewriter.create<mlir::arith::DivUIOp>(
      loc, flatBatchSequence, dimS);
  mlir::Value si = rewriter.create<mlir::arith::RemUIOp>(
      loc, flatBatchSequence, dimS);
  mlir::Value evenIdx =
      rewriter.create<mlir::arith::MulIOp>(loc, pi, two);
  mlir::Value oddIdx =
      rewriter.create<mlir::arith::AddIOp>(loc, evenIdx, oneIdx);

  mlir::Value x1 = loadAsAccumulator(
      rewriter, loc, X,
      mlir::SmallVector<mlir::Value>{bi, si, hi, evenIdx}, elemTy);
  mlir::Value x2 = loadAsAccumulator(
      rewriter, loc, X,
      mlir::SmallVector<mlir::Value>{bi, si, hi, oddIdx}, elemTy);
  mlir::Value cosVal = loadAsAccumulator(
      rewriter, loc, cosTable,
      mlir::SmallVector<mlir::Value>{si, pi}, elemTy);
  mlir::Value sinVal = loadAsAccumulator(
      rewriter, loc, sinTable,
      mlir::SmallVector<mlir::Value>{si, pi}, elemTy);

  mlir::Value yEven = rewriter.create<mlir::arith::SubFOp>(
      loc, rewriter.create<mlir::arith::MulFOp>(loc, x1, cosVal),
      rewriter.create<mlir::arith::MulFOp>(loc, x2, sinVal));
  mlir::Value yOdd = rewriter.create<mlir::arith::AddFOp>(
      loc, rewriter.create<mlir::arith::MulFOp>(loc, x2, cosVal),
      rewriter.create<mlir::arith::MulFOp>(loc, x1, sinVal));
  storeFromAccumulator(
      rewriter, loc, yEven, Y,
      mlir::SmallVector<mlir::Value>{bi, si, hi, evenIdx});
  storeFromAccumulator(
      rewriter, loc, yOdd, Y,
      mlir::SmallVector<mlir::Value>{bi, si, hi, oddIdx});
  rewriter.create<mlir::gpu::TerminatorOp>(loc);

  rewriter.eraseOp(op);
  return mlir::success();
}

// ─────────────────────────────────────────────────────────────────────────────
//  StructuredComputeToSpirv
// ─────────────────────────────────────────────────────────────────────────────

mlir::LogicalResult StructuredComputeToSpirv::matchAndRewrite(
    mlir::linalg::GenericOp op,
    mlir::PatternRewriter& rewriter) const {
  mlir::Location loc = op.getLoc();
  const auto* emitter = emitterForOperation(op);
  if (emitter == nullptr || !usesStructuredComputeSchedule(*emitter)) {
    return mlir::failure();
  }
  const bool indexedStructural = usesBroadcastBinarySchedule(*emitter);

  mlir::ValueRange inputs = op.getInputs();
  mlir::ValueRange outputs = op.getOutputs();
  mlir::FailureOr<LoweringTypeContract> typeContract = mlir::failure();
  if (hasVulkanMixedOperandTypeContract(*emitter)) {
    typeContract = getMixedOperandTypeContract(op, inputs, outputs, *emitter);
  } else if (emitter->family == VulkanKernelFamily::ELEMENTWISE_BINARY &&
             indexedStructural) {
    typeContract = getIndexedFloatTypeContract(op, inputs, outputs);
  } else {
    typeContract = getComputeTypeContract(op, inputs, outputs);
  }
  if (mlir::failed(typeContract)) return mlir::failure();
  mlir::FloatType accTy = typeContract->accumulatorType;
  const double epsilon =
      op->getAttrOfType<mlir::FloatAttr>("nd4j.epsilon")
          ? op->getAttrOfType<mlir::FloatAttr>("nd4j.epsilon")
                .getValueAsDouble()
          : 1.0e-5;
  mlir::Value zero = idxConst(rewriter, loc, 0);
  mlir::Value one = idxConst(rewriter, loc, 1);

  auto rowNormFactor = [&](mlir::Value x, mlir::Value row,
                           mlir::Value hidden) -> mlir::Value {
    mlir::Value initial = floatConst(rewriter, loc, accTy, 0.0);
    auto squares = emitReductionLoop(
        rewriter, loc, zero, hidden, one, initial,
        [&](mlir::OpBuilder& builder, mlir::Location nestedLoc,
            mlir::Value column, mlir::Value accumulator) -> mlir::Value {
          mlir::Value value = loadAsAccumulator(
              builder, nestedLoc, x,
              mlir::SmallVector<mlir::Value>{row, column}, accTy);
          return builder.create<mlir::arith::AddFOp>(
              nestedLoc, accumulator,
              builder.create<mlir::arith::MulFOp>(nestedLoc, value, value));
        });
    mlir::Value hiddenValue =
        convertIndexToFloat(rewriter, loc, hidden, accTy);
    mlir::Value meanSquare = rewriter.create<mlir::arith::DivFOp>(
        loc, squares.getResult(0), hiddenValue);
    mlir::Value stabilized = rewriter.create<mlir::arith::AddFOp>(
        loc, meanSquare, floatConst(rewriter, loc, accTy, epsilon));
    return emitRsqrt(rewriter, loc, accTy, stabilized);
  };

  switch (emitter->recipe) {
    case VulkanKernelRecipe::TRIANGULAR_SOLVE: {
      if (inputs.size() != 2 || outputs.size() != 1) {
        return op.emitOpError(
            "triangular_solve requires matrix, rhs, and output operands");
      }
      auto matrixType = llvm::dyn_cast<mlir::MemRefType>(inputs[0].getType());
      auto rhsType = llvm::dyn_cast<mlir::MemRefType>(inputs[1].getType());
      auto outputType = llvm::dyn_cast<mlir::MemRefType>(outputs[0].getType());
      if (!matrixType || !rhsType || !outputType ||
          matrixType.getRank() < 2 || rhsType.getRank() < 2 ||
          matrixType.getRank() != rhsType.getRank() ||
          rhsType.getRank() != outputType.getRank()) {
        return op.emitOpError(
            "triangular_solve requires equally-ranked rank-2+ MemRefs");
      }

      const int64_t rank = rhsType.getRank();
      const int64_t matrixRowDim = rank - 2;
      mlir::Value rows = rewriter.create<mlir::memref::DimOp>(
          loc, inputs[0], matrixRowDim);
      mlir::Value columns = rewriter.create<mlir::memref::DimOp>(
          loc, inputs[1], rank - 1);
      mlir::Value batchCount = one;
      for (int64_t d = 0; d < rank - 2; ++d) {
        batchCount = rewriter.create<mlir::arith::MulIOp>(
            loc, batchCount,
            rewriter.create<mlir::memref::DimOp>(loc, inputs[1], d));
      }
      mlir::Value invocationCount = rewriter.create<mlir::arith::MulIOp>(
          loc, batchCount, columns);
      auto launch = createGpuLaunch(
          rewriter, loc, invocationCount, one, one);
      mlir::Value invocation = launch.getBlockIds().x;
      rewriter.setInsertionPointToEnd(&launch.getBody().front());

      mlir::Value rhsElementsPerBatch = rewriter.create<mlir::arith::MulIOp>(
          loc, rows, columns);
      mlir::Value batchIndex = rewriter.create<mlir::arith::DivUIOp>(
          loc, invocation, columns);
      mlir::Value column = rewriter.create<mlir::arith::RemUIOp>(
          loc, invocation, columns);
      mlir::Value rhsBase = rewriter.create<mlir::arith::MulIOp>(
          loc, batchIndex, rhsElementsPerBatch);
      mlir::Value matrixElementsPerBatch = rewriter.create<mlir::arith::MulIOp>(
          loc, rows, rows);
      mlir::Value matrixBase = rewriter.create<mlir::arith::MulIOp>(
          loc, batchIndex, matrixElementsPerBatch);

      auto operandIndices = [&](mlir::OpBuilder& builder,
                                mlir::Location nestedLoc,
                                mlir::Value memref, mlir::Value base,
                                mlir::Value row, mlir::Value columnIndex) {
        auto indices = logicalIndices(builder, nestedLoc, base, memref);
        auto type = llvm::cast<mlir::MemRefType>(memref.getType());
        indices[static_cast<size_t>(type.getRank() - 2)] = row;
        indices[static_cast<size_t>(type.getRank() - 1)] = columnIndex;
        return indices;
      };

      const auto lowerAttribute =
          op->getAttrOfType<mlir::BoolAttr>("nd4j.triangular_lower");
      const auto adjointAttribute =
          op->getAttrOfType<mlir::BoolAttr>("nd4j.triangular_adjoint");
      const bool lower = !lowerAttribute || lowerAttribute.getValue();
      const bool adjoint = adjointAttribute && adjointAttribute.getValue();
      const bool forward = adjoint ? !lower : lower;

      emitReductionLoop(
          rewriter, loc, zero, rows, one, zero,
          [&](mlir::OpBuilder& builder, mlir::Location nestedLoc,
              mlir::Value row, mlir::Value accumulator) -> mlir::Value {
            mlir::Value actualRow = row;
            if (!forward) {
              actualRow = builder.create<mlir::arith::SubIOp>(
                  nestedLoc, rows,
                  builder.create<mlir::arith::AddIOp>(
                      nestedLoc, row, one));
            }
            mlir::Value rhsOffset = builder.create<mlir::arith::AddIOp>(
                nestedLoc, rhsBase,
                builder.create<mlir::arith::AddIOp>(
                    nestedLoc,
                    builder.create<mlir::arith::MulIOp>(
                        nestedLoc, actualRow, columns),
                    column));
            mlir::Value rhsValue = loadAsAccumulator(
                builder, nestedLoc, inputs[1],
                operandIndices(builder, nestedLoc, inputs[1], rhsOffset,
                               actualRow, column),
                accTy);
            mlir::Value innerLower = forward
                ? zero
                : builder.create<mlir::arith::AddIOp>(
                      nestedLoc, actualRow, one);
            mlir::Value innerUpper = forward ? actualRow : rows;
            auto inner = emitReductionLoop(
                builder, nestedLoc, innerLower, innerUpper, one, rhsValue,
                [&](mlir::OpBuilder& innerBuilder,
                    mlir::Location innerLoc, mlir::Value k,
                    mlir::Value value) -> mlir::Value {
                  mlir::Value coefficient = loadAsAccumulator(
                      innerBuilder, innerLoc, inputs[0],
                      operandIndices(
                          innerBuilder, innerLoc, inputs[0], matrixBase,
                          adjoint ? k : actualRow,
                          adjoint ? actualRow : k),
                      accTy);
                  mlir::Value solved = loadAsAccumulator(
                      innerBuilder, innerLoc, outputs[0],
                      operandIndices(
                          innerBuilder, innerLoc, outputs[0], rhsBase,
                          k, column),
                      accTy);
                  return innerBuilder.create<mlir::arith::SubFOp>(
                      innerLoc, value,
                      innerBuilder.create<mlir::arith::MulFOp>(
                          innerLoc, coefficient, solved));
                });
            mlir::Value diagonal = loadAsAccumulator(
                builder, nestedLoc, inputs[0],
                operandIndices(builder, nestedLoc, inputs[0], matrixBase,
                               actualRow, actualRow),
                accTy);
            mlir::Value result = builder.create<mlir::arith::DivFOp>(
                nestedLoc, inner.getResult(0), diagonal);
            storeFromAccumulator(
                builder, nestedLoc, result, outputs[0],
                operandIndices(builder, nestedLoc, outputs[0], rhsBase,
                               actualRow, column));
            return accumulator;
          });
      rewriter.create<mlir::gpu::TerminatorOp>(loc);
      rewriter.eraseOp(op);
      return mlir::success();
    }
    case VulkanKernelRecipe::RMS_NORM_BP: {
      if (inputs.size() != 2 || outputs.size() != 1) {
        return op.emitOpError(
            "rms_norm_bp gamma-free form requires input, gradOut, and gradIn");
      }
      mlir::Value input = inputs[0];
      mlir::Value gradOut = inputs[1];
      mlir::Value gradIn = outputs[0];
      auto outputType = llvm::dyn_cast<mlir::MemRefType>(gradIn.getType());
      if (!outputType || outputType.getRank() < 1) {
        return op.emitOpError("rms_norm_bp requires rank-1+ MemRefs");
      }

      const int64_t rank = outputType.getRank();
      mlir::Value features = rewriter.create<mlir::memref::DimOp>(
          loc, gradIn, rank - 1);
      mlir::Value rows = one;
      for (int64_t d = 0; d < rank - 1; ++d) {
        rows = rewriter.create<mlir::arith::MulIOp>(
            loc, rows,
            rewriter.create<mlir::memref::DimOp>(loc, gradIn, d));
      }
      mlir::Value featureCount =
          convertIndexToFloat(rewriter, loc, features, accTy);

      auto launch = createGpuLaunch(rewriter, loc, rows, one, one);
      mlir::Value row = launch.getBlockIds().x;
      rewriter.setInsertionPointToEnd(&launch.getBody().front());

      auto indicesFor = [&](mlir::OpBuilder& builder,
                            mlir::Location nestedLoc,
                            mlir::Value column,
                            mlir::Value memref) {
        mlir::Value linear = builder.create<mlir::arith::AddIOp>(
            nestedLoc,
            builder.create<mlir::arith::MulIOp>(
                nestedLoc, row, features),
            column);
        return logicalIndices(builder, nestedLoc, linear, memref);
      };

      auto squareSum = emitReductionLoop(
          rewriter, loc, zero, features, one,
          floatConst(rewriter, loc, accTy, 0.0),
          [&](mlir::OpBuilder& builder, mlir::Location nestedLoc,
              mlir::Value column,
              mlir::Value accumulator) -> mlir::Value {
            mlir::Value value = loadAsAccumulator(
                builder, nestedLoc, input,
                indicesFor(builder, nestedLoc, column, input), accTy);
            return builder.create<mlir::arith::AddFOp>(
                nestedLoc, accumulator,
                builder.create<mlir::arith::MulFOp>(
                    nestedLoc, value, value));
          });
      mlir::Value inverseRms = emitRsqrt(
          rewriter, loc, accTy,
          rewriter.create<mlir::arith::AddFOp>(
              loc,
              rewriter.create<mlir::arith::DivFOp>(
                  loc, squareSum.getResult(0), featureCount),
              floatConst(rewriter, loc, accTy, epsilon)));

      auto dot = emitReductionLoop(
          rewriter, loc, zero, features, one,
          floatConst(rewriter, loc, accTy, 0.0),
          [&](mlir::OpBuilder& builder, mlir::Location nestedLoc,
              mlir::Value column,
              mlir::Value accumulator) -> mlir::Value {
            mlir::Value inputValue = loadAsAccumulator(
                builder, nestedLoc, input,
                indicesFor(builder, nestedLoc, column, input), accTy);
            mlir::Value gradient = loadAsAccumulator(
                builder, nestedLoc, gradOut,
                indicesFor(builder, nestedLoc, column, gradOut), accTy);
            return builder.create<mlir::arith::AddFOp>(
                nestedLoc, accumulator,
                builder.create<mlir::arith::MulFOp>(
                    nestedLoc, inputValue, gradient));
          });
      mlir::Value inverseRmsSquared =
          rewriter.create<mlir::arith::MulFOp>(
              loc, inverseRms, inverseRms);
      mlir::Value correction = rewriter.create<mlir::arith::MulFOp>(
          loc,
          rewriter.create<mlir::arith::DivFOp>(
              loc, dot.getResult(0), featureCount),
          rewriter.create<mlir::arith::MulFOp>(
              loc, inverseRmsSquared, inverseRms));

      rewriter.create<mlir::scf::ForOp>(
          loc, zero, features, one, mlir::ValueRange{},
          [&](mlir::OpBuilder& builder, mlir::Location nestedLoc,
              mlir::Value column, mlir::ValueRange) {
            auto inputIndices =
                indicesFor(builder, nestedLoc, column, input);
            auto gradientIndices =
                indicesFor(builder, nestedLoc, column, gradOut);
            auto outputIndices =
                indicesFor(builder, nestedLoc, column, gradIn);
            mlir::Value inputValue = loadAsAccumulator(
                builder, nestedLoc, input, inputIndices, accTy);
            mlir::Value gradient = loadAsAccumulator(
                builder, nestedLoc, gradOut, gradientIndices, accTy);
            mlir::Value direct = builder.create<mlir::arith::MulFOp>(
                nestedLoc, gradient, inverseRms);
            mlir::Value projected = builder.create<mlir::arith::MulFOp>(
                nestedLoc, inputValue, correction);
            storeFromAccumulator(
                builder, nestedLoc,
                builder.create<mlir::arith::SubFOp>(
                    nestedLoc, direct, projected),
                gradIn, outputIndices);
            builder.create<mlir::scf::YieldOp>(nestedLoc);
          });
      rewriter.create<mlir::gpu::TerminatorOp>(loc);
      rewriter.eraseOp(op);
      return mlir::success();
    }

    case VulkanKernelRecipe::FUSED_LAYER_NORM_BP: {
      if (inputs.size() != 3 || outputs.size() != 2) {
        return op.emitOpError(
            "fused_layer_norm_bp requires x, gain, gradOut, dX, and dGamma");
      }
      mlir::Value input = inputs[0];
      mlir::Value gain = inputs[1];
      mlir::Value gradOut = inputs[2];
      mlir::Value gradIn = outputs[0];
      mlir::Value gradGain = outputs[1];
      auto inputType = llvm::dyn_cast<mlir::MemRefType>(input.getType());
      auto gainType = llvm::dyn_cast<mlir::MemRefType>(gain.getType());
      auto gradientType =
          llvm::dyn_cast<mlir::MemRefType>(gradOut.getType());
      auto gradInType = llvm::dyn_cast<mlir::MemRefType>(gradIn.getType());
      auto gradGainType =
          llvm::dyn_cast<mlir::MemRefType>(gradGain.getType());
      if (!inputType || !gainType || !gradientType || !gradInType ||
          !gradGainType || inputType.getRank() != 1 ||
          gainType.getRank() != 1 || gradientType.getRank() != 1 ||
          gradInType.getRank() != 1 || gradGainType.getRank() != 1) {
        return op.emitOpError(
            "fused_layer_norm_bp portable contract requires rank-one MemRefs");
      }

      mlir::Value features =
          rewriter.create<mlir::memref::DimOp>(loc, gradIn, 0);
      mlir::Value featureCount =
          convertIndexToFloat(rewriter, loc, features, accTy);
      mlir::Value initial = floatConst(rewriter, loc, accTy, 0.0);
      auto indexFor = [&](mlir::OpBuilder& builder,
                          mlir::Location nestedLoc,
                          mlir::Value column,
                          mlir::Value memref) {
        return logicalIndices(builder, nestedLoc, column, memref);
      };

      auto launch = createGpuLaunch(rewriter, loc, one, one, one);
      rewriter.setInsertionPointToEnd(&launch.getBody().front());

      auto sum = emitReductionLoop(
          rewriter, loc, zero, features, one, initial,
          [&](mlir::OpBuilder& builder, mlir::Location nestedLoc,
              mlir::Value column,
              mlir::Value accumulator) -> mlir::Value {
            mlir::Value value = loadAsAccumulator(
                builder, nestedLoc, input,
                indexFor(builder, nestedLoc, column, input), accTy);
            return builder.create<mlir::arith::AddFOp>(
                nestedLoc, accumulator, value);
          });
      mlir::Value mean = rewriter.create<mlir::arith::DivFOp>(
          loc, sum.getResult(0), featureCount);

      auto varianceSum = emitReductionLoop(
          rewriter, loc, zero, features, one, initial,
          [&](mlir::OpBuilder& builder, mlir::Location nestedLoc,
              mlir::Value column,
              mlir::Value accumulator) -> mlir::Value {
            mlir::Value value = loadAsAccumulator(
                builder, nestedLoc, input,
                indexFor(builder, nestedLoc, column, input), accTy);
            mlir::Value centered = builder.create<mlir::arith::SubFOp>(
                nestedLoc, value, mean);
            return builder.create<mlir::arith::AddFOp>(
                nestedLoc, accumulator,
                builder.create<mlir::arith::MulFOp>(
                    nestedLoc, centered, centered));
          });
      mlir::Value inverseStd = emitRsqrt(
          rewriter, loc, accTy,
          rewriter.create<mlir::arith::AddFOp>(
              loc,
              rewriter.create<mlir::arith::DivFOp>(
                  loc, varianceSum.getResult(0), featureCount),
              floatConst(rewriter, loc, accTy, epsilon)));

      auto sumDyGain = emitReductionLoop(
          rewriter, loc, zero, features, one, initial,
          [&](mlir::OpBuilder& builder, mlir::Location nestedLoc,
              mlir::Value column,
              mlir::Value accumulator) -> mlir::Value {
            mlir::Value dy = loadAsAccumulator(
                builder, nestedLoc, gradOut,
                indexFor(builder, nestedLoc, column, gradOut), accTy);
            mlir::Value gamma = loadAsAccumulator(
                builder, nestedLoc, gain,
                indexFor(builder, nestedLoc, column, gain), accTy);
            return builder.create<mlir::arith::AddFOp>(
                nestedLoc, accumulator,
                builder.create<mlir::arith::MulFOp>(
                    nestedLoc, dy, gamma));
          });
      mlir::Value meanDyGain = rewriter.create<mlir::arith::DivFOp>(
          loc, sumDyGain.getResult(0), featureCount);

      auto sumDyGainXHat = emitReductionLoop(
          rewriter, loc, zero, features, one, initial,
          [&](mlir::OpBuilder& builder, mlir::Location nestedLoc,
              mlir::Value column,
              mlir::Value accumulator) -> mlir::Value {
            mlir::Value value = loadAsAccumulator(
                builder, nestedLoc, input,
                indexFor(builder, nestedLoc, column, input), accTy);
            mlir::Value dy = loadAsAccumulator(
                builder, nestedLoc, gradOut,
                indexFor(builder, nestedLoc, column, gradOut), accTy);
            mlir::Value gamma = loadAsAccumulator(
                builder, nestedLoc, gain,
                indexFor(builder, nestedLoc, column, gain), accTy);
            mlir::Value xHat = builder.create<mlir::arith::MulFOp>(
                nestedLoc,
                builder.create<mlir::arith::SubFOp>(
                    nestedLoc, value, mean),
                inverseStd);
            return builder.create<mlir::arith::AddFOp>(
                nestedLoc, accumulator,
                builder.create<mlir::arith::MulFOp>(
                    nestedLoc,
                    builder.create<mlir::arith::MulFOp>(
                        nestedLoc, dy, gamma),
                    xHat));
          });
      mlir::Value meanDyGainXHat =
          rewriter.create<mlir::arith::DivFOp>(
              loc, sumDyGainXHat.getResult(0), featureCount);

      rewriter.create<mlir::scf::ForOp>(
          loc, zero, features, one, mlir::ValueRange{},
          [&](mlir::OpBuilder& builder, mlir::Location nestedLoc,
              mlir::Value column, mlir::ValueRange) {
            mlir::Value value = loadAsAccumulator(
                builder, nestedLoc, input,
                indexFor(builder, nestedLoc, column, input), accTy);
            mlir::Value dy = loadAsAccumulator(
                builder, nestedLoc, gradOut,
                indexFor(builder, nestedLoc, column, gradOut), accTy);
            mlir::Value gamma = loadAsAccumulator(
                builder, nestedLoc, gain,
                indexFor(builder, nestedLoc, column, gain), accTy);
            mlir::Value xHat = builder.create<mlir::arith::MulFOp>(
                nestedLoc,
                builder.create<mlir::arith::SubFOp>(
                    nestedLoc, value, mean),
                inverseStd);
            mlir::Value dyGain = builder.create<mlir::arith::MulFOp>(
                nestedLoc, dy, gamma);
            mlir::Value centeredGradient =
                builder.create<mlir::arith::SubFOp>(
                    nestedLoc,
                    builder.create<mlir::arith::SubFOp>(
                        nestedLoc, dyGain, meanDyGain),
                    builder.create<mlir::arith::MulFOp>(
                        nestedLoc, xHat, meanDyGainXHat));
            storeFromAccumulator(
                builder, nestedLoc,
                builder.create<mlir::arith::MulFOp>(
                    nestedLoc, inverseStd, centeredGradient),
                gradIn,
                indexFor(builder, nestedLoc, column, gradIn));
            storeFromAccumulator(
                builder, nestedLoc,
                builder.create<mlir::arith::MulFOp>(
                    nestedLoc, dy, xHat),
                gradGain,
                indexFor(builder, nestedLoc, column, gradGain));
            builder.create<mlir::scf::YieldOp>(nestedLoc);
          });
      rewriter.create<mlir::gpu::TerminatorOp>(loc);
      rewriter.eraseOp(op);
      return mlir::success();
    }

    case VulkanKernelRecipe::WINDOW_PARTITION: {
      if (inputs.size() != 1 || outputs.size() != 1) {
        return op.emitOpError("window partition operand contract mismatch");
      }
      mlir::Value input = inputs.front();
      mlir::Value output = outputs.front();
      auto inputType = llvm::dyn_cast<mlir::MemRefType>(input.getType());
      auto outputType = llvm::dyn_cast<mlir::MemRefType>(output.getType());
      auto windowAttr =
          op->getAttrOfType<mlir::IntegerAttr>("nd4j.window_size");
      if (!inputType || !outputType || inputType.getRank() != 4 ||
          outputType.getRank() != 4 || !windowAttr ||
          windowAttr.getInt() <= 0) {
        return op.emitOpError(
            "window partition requires rank-4 tensors and positive window");
      }
      mlir::Value window = idxConst(rewriter, loc, windowAttr.getInt());
      mlir::Value height =
          rewriter.create<mlir::memref::DimOp>(loc, input, 1);
      mlir::Value width =
          rewriter.create<mlir::memref::DimOp>(loc, input, 2);
      mlir::Value heightBlocks =
          rewriter.create<mlir::arith::DivUIOp>(loc, height, window);
      mlir::Value widthBlocks =
          rewriter.create<mlir::arith::DivUIOp>(loc, width, window);
      mlir::Value windowsPerBatch =
          rewriter.create<mlir::arith::MulIOp>(
              loc, heightBlocks, widthBlocks);
      mlir::Value total = one;
      for (int64_t d = 0; d < 4; ++d) {
        total = rewriter.create<mlir::arith::MulIOp>(
            loc, total,
            rewriter.create<mlir::memref::DimOp>(loc, output, d));
      }
      auto launch = createGpuLaunch(rewriter, loc, total, one, one);
      mlir::Value flat = launch.getBlockIds().x;
      rewriter.setInsertionPointToEnd(&launch.getBody().front());
      auto outputIndices = logicalIndices(rewriter, loc, flat, output);
      mlir::Value flatWindow = outputIndices[0];
      mlir::Value batch = rewriter.create<mlir::arith::DivUIOp>(
          loc, flatWindow, windowsPerBatch);
      mlir::Value windowWithinBatch =
          rewriter.create<mlir::arith::RemUIOp>(
              loc, flatWindow, windowsPerBatch);
      mlir::Value heightBlock = rewriter.create<mlir::arith::DivUIOp>(
          loc, windowWithinBatch, widthBlocks);
      mlir::Value widthBlock = rewriter.create<mlir::arith::RemUIOp>(
          loc, windowWithinBatch, widthBlocks);
      mlir::Value inputHeight = rewriter.create<mlir::arith::AddIOp>(
          loc,
          rewriter.create<mlir::arith::MulIOp>(
              loc, heightBlock, window),
          outputIndices[1]);
      mlir::Value inputWidth = rewriter.create<mlir::arith::AddIOp>(
          loc,
          rewriter.create<mlir::arith::MulIOp>(
              loc, widthBlock, window),
          outputIndices[2]);
      mlir::Value value = loadAsAccumulator(
          rewriter, loc, input,
          mlir::SmallVector<mlir::Value>{
              batch, inputHeight, inputWidth, outputIndices[3]},
          accTy);
      storeFromAccumulator(
          rewriter, loc, value, output, outputIndices);
      rewriter.create<mlir::gpu::TerminatorOp>(loc);
      rewriter.eraseOp(op);
      return mlir::success();
    }

    case VulkanKernelRecipe::WINDOW_UNPARTITION: {
      if (inputs.size() != 1 || outputs.size() != 1) {
        return op.emitOpError("window unpartition operand contract mismatch");
      }
      mlir::Value input = inputs.front();
      mlir::Value output = outputs.front();
      auto inputType = llvm::dyn_cast<mlir::MemRefType>(input.getType());
      auto outputType = llvm::dyn_cast<mlir::MemRefType>(output.getType());
      auto windowAttr =
          op->getAttrOfType<mlir::IntegerAttr>("nd4j.window_size");
      auto heightAttr =
          op->getAttrOfType<mlir::IntegerAttr>("nd4j.output_height");
      auto widthAttr =
          op->getAttrOfType<mlir::IntegerAttr>("nd4j.output_width");
      if (!inputType || !outputType || inputType.getRank() != 4 ||
          outputType.getRank() != 4 || !windowAttr || !heightAttr ||
          !widthAttr || windowAttr.getInt() <= 0) {
        return op.emitOpError(
            "window unpartition requires rank-4 tensors and frozen dimensions");
      }
      mlir::Value window = idxConst(rewriter, loc, windowAttr.getInt());
      mlir::Value heightBlocks = idxConst(
          rewriter, loc, heightAttr.getInt() / windowAttr.getInt());
      mlir::Value widthBlocks = idxConst(
          rewriter, loc, widthAttr.getInt() / windowAttr.getInt());
      mlir::Value total = one;
      for (int64_t d = 0; d < 4; ++d) {
        total = rewriter.create<mlir::arith::MulIOp>(
            loc, total,
            rewriter.create<mlir::memref::DimOp>(loc, output, d));
      }
      auto launch = createGpuLaunch(rewriter, loc, total, one, one);
      mlir::Value flat = launch.getBlockIds().x;
      rewriter.setInsertionPointToEnd(&launch.getBody().front());
      auto outputIndices = logicalIndices(rewriter, loc, flat, output);
      mlir::Value heightBlock = rewriter.create<mlir::arith::DivUIOp>(
          loc, outputIndices[1], window);
      mlir::Value localHeight = rewriter.create<mlir::arith::RemUIOp>(
          loc, outputIndices[1], window);
      mlir::Value widthBlock = rewriter.create<mlir::arith::DivUIOp>(
          loc, outputIndices[2], window);
      mlir::Value localWidth = rewriter.create<mlir::arith::RemUIOp>(
          loc, outputIndices[2], window);
      mlir::Value flatWindow = rewriter.create<mlir::arith::AddIOp>(
          loc,
          rewriter.create<mlir::arith::MulIOp>(
              loc,
              rewriter.create<mlir::arith::AddIOp>(
                  loc,
                  rewriter.create<mlir::arith::MulIOp>(
                      loc, outputIndices[0], heightBlocks),
                  heightBlock),
              widthBlocks),
          widthBlock);
      mlir::Value value = loadAsAccumulator(
          rewriter, loc, input,
          mlir::SmallVector<mlir::Value>{
              flatWindow, localHeight, localWidth, outputIndices[3]},
          accTy);
      storeFromAccumulator(
          rewriter, loc, value, output, outputIndices);
      rewriter.create<mlir::gpu::TerminatorOp>(loc);
      rewriter.eraseOp(op);
      return mlir::success();
    }

    case VulkanKernelRecipe::BIAS_ADD: {
      if (inputs.size() != 2 || outputs.size() != 1) {
        return op.emitOpError("bias-add operand contract mismatch");
      }
      mlir::Value input = inputs[0];
      mlir::Value bias = inputs[1];
      mlir::Value output = outputs.front();
      auto outputType = llvm::dyn_cast<mlir::MemRefType>(output.getType());
      auto channelAxisAttr =
          op->getAttrOfType<mlir::IntegerAttr>("nd4j.channel_axis");
      auto inputUnsignedAttr =
          op->getAttrOfType<mlir::BoolAttr>("nd4j.input0_unsigned");
      if (!outputType || outputType.getRank() < 1 || !channelAxisAttr ||
          !inputUnsignedAttr || channelAxisAttr.getInt() < 0 ||
          channelAxisAttr.getInt() >= outputType.getRank()) {
        return op.emitOpError(
            "bias-add requires a ranked output and frozen channel axis");
      }
      mlir::Value total = one;
      for (int64_t d = 0; d < outputType.getRank(); ++d) {
        total = rewriter.create<mlir::arith::MulIOp>(
            loc, total,
            rewriter.create<mlir::memref::DimOp>(loc, output, d));
      }
      auto launch = createGpuLaunch(rewriter, loc, total, one, one);
      mlir::Value flat = launch.getBlockIds().x;
      rewriter.setInsertionPointToEnd(&launch.getBody().front());
      auto indices = logicalIndices(rewriter, loc, flat, output);
      const bool inputUnsigned = inputUnsignedAttr.getValue();
      mlir::Value inputValue = loadAsScalar(
          rewriter, loc, input, indices, accTy,
          inputUnsigned, inputUnsigned);
      mlir::Value biasValue = loadAsAccumulator(
          rewriter, loc, bias,
          mlir::SmallVector<mlir::Value>{
              indices[static_cast<size_t>(channelAxisAttr.getInt())]},
          accTy);
      storeFromAccumulator(
          rewriter, loc,
          rewriter.create<mlir::arith::AddFOp>(
              loc, inputValue, biasValue),
          output, indices);
      rewriter.create<mlir::gpu::TerminatorOp>(loc);
      rewriter.eraseOp(op);
      return mlir::success();
    }

    case VulkanKernelRecipe::PRELU: {
      if (inputs.size() != 2 || outputs.size() != 1) {
        return op.emitOpError("PReLU operand contract mismatch");
      }
      mlir::Value input = inputs[0];
      mlir::Value alpha = inputs[1];
      mlir::Value output = outputs.front();
      auto outputType = llvm::dyn_cast<mlir::MemRefType>(output.getType());
      auto sharedAxesAttr =
          op->getAttrOfType<mlir::DenseI64ArrayAttr>("nd4j.shared_axes");
      auto inputUnsignedAttr =
          op->getAttrOfType<mlir::BoolAttr>("nd4j.input0_unsigned");
      if (!outputType || outputType.getRank() <= 1 || !sharedAxesAttr ||
          !inputUnsignedAttr) {
        return op.emitOpError(
            "PReLU requires rank > 1 and frozen shared axes");
      }
      const int64_t rank = outputType.getRank();
      llvm::SmallVector<int8_t> shared(static_cast<size_t>(rank), 0);
      for (int64_t axis : sharedAxesAttr.asArrayRef()) {
        if (axis < 1 || axis >= rank) {
          return op.emitOpError("PReLU shared axis is out of range");
        }
        shared[static_cast<size_t>(axis)] = 1;
      }
      mlir::Value total = one;
      for (int64_t d = 0; d < rank; ++d) {
        total = rewriter.create<mlir::arith::MulIOp>(
            loc, total,
            rewriter.create<mlir::memref::DimOp>(loc, output, d));
      }
      auto launch = createGpuLaunch(rewriter, loc, total, one, one);
      mlir::Value flat = launch.getBlockIds().x;
      rewriter.setInsertionPointToEnd(&launch.getBody().front());
      auto indices = logicalIndices(rewriter, loc, flat, output);
      mlir::Value alphaFlat = zero;
      for (int64_t d = 1; d < rank; ++d) {
        if (shared[static_cast<size_t>(d)] != 0) continue;
        alphaFlat = rewriter.create<mlir::arith::AddIOp>(
            loc,
            rewriter.create<mlir::arith::MulIOp>(
                loc, alphaFlat,
                rewriter.create<mlir::memref::DimOp>(loc, input, d)),
            indices[static_cast<size_t>(d)]);
      }
      auto alphaIndices = logicalIndices(rewriter, loc, alphaFlat, alpha);
      const bool inputUnsigned = inputUnsignedAttr.getValue();
      mlir::Value value = loadAsScalar(
          rewriter, loc, input, indices, accTy,
          inputUnsigned, inputUnsigned);
      mlir::Value alphaValue = loadAsAccumulator(
          rewriter, loc, alpha, alphaIndices, accTy);
      mlir::Value negative = rewriter.create<mlir::arith::CmpFOp>(
          loc, mlir::arith::CmpFPredicate::OLT, value,
          floatConst(rewriter, loc, accTy, 0.0));
      mlir::Value result = rewriter.create<mlir::arith::SelectOp>(
          loc, negative,
          rewriter.create<mlir::arith::MulFOp>(loc, value, alphaValue),
          value);
      storeFromAccumulator(rewriter, loc, result, output, indices);
      rewriter.create<mlir::gpu::TerminatorOp>(loc);
      rewriter.eraseOp(op);
      return mlir::success();
    }

    case VulkanKernelRecipe::BATCH_NORM: {
      if (inputs.size() < 3 || inputs.size() > 5 || outputs.size() != 1) {
        return op.emitOpError("batchnorm operand contract mismatch");
      }
      mlir::Value input = inputs[0];
      mlir::Value mean = inputs[1];
      mlir::Value variance = inputs[2];
      mlir::Value output = outputs.front();
      auto outputType = llvm::dyn_cast<mlir::MemRefType>(output.getType());
      auto axesAttr = op->getAttrOfType<mlir::DenseI64ArrayAttr>(
          "nd4j.normalization_axes");
      auto applyScaleAttr =
          op->getAttrOfType<mlir::BoolAttr>("nd4j.apply_scale");
      auto applyOffsetAttr =
          op->getAttrOfType<mlir::BoolAttr>("nd4j.apply_offset");
      if (!outputType || outputType.getRank() < 1 || !axesAttr ||
          axesAttr.empty() || !applyScaleAttr || !applyOffsetAttr) {
        return op.emitOpError(
            "batchnorm requires ranked tensors and frozen axes/flags");
      }
      const bool applyScale = applyScaleAttr.getValue();
      const bool applyOffset = applyOffsetAttr.getValue();
      if (inputs.size() !=
          static_cast<size_t>(3 + static_cast<int>(applyScale) +
                              static_cast<int>(applyOffset))) {
        return op.emitOpError("batchnorm optional operand count mismatch");
      }
      const int64_t rank = outputType.getRank();
      llvm::SmallVector<int8_t> normalized(static_cast<size_t>(rank), 0);
      for (int64_t axis : axesAttr.asArrayRef()) {
        if (axis < 0 || axis >= rank ||
            normalized[static_cast<size_t>(axis)] != 0) {
          return op.emitOpError("batchnorm axes are invalid");
        }
        normalized[static_cast<size_t>(axis)] = 1;
      }
      mlir::Value total = one;
      for (int64_t d = 0; d < rank; ++d) {
        total = rewriter.create<mlir::arith::MulIOp>(
            loc, total,
            rewriter.create<mlir::memref::DimOp>(loc, output, d));
      }
      auto launch = createGpuLaunch(rewriter, loc, total, one, one);
      mlir::Value flat = launch.getBlockIds().x;
      rewriter.setInsertionPointToEnd(&launch.getBody().front());
      auto indices = logicalIndices(rewriter, loc, flat, output);
      mlir::SmallVector<mlir::Value> parameterIndices;
      if (axesAttr.size() == 1) {
        parameterIndices.push_back(
            indices[static_cast<size_t>(axesAttr.asArrayRef().front())]);
      } else {
        for (int64_t d = 0; d < rank; ++d) {
          parameterIndices.push_back(
              normalized[static_cast<size_t>(d)] != 0
                  ? indices[static_cast<size_t>(d)]
                  : zero);
        }
      }
      mlir::Value centered = rewriter.create<mlir::arith::SubFOp>(
          loc, loadAsAccumulator(rewriter, loc, input, indices, accTy),
          loadAsAccumulator(
              rewriter, loc, mean, parameterIndices, accTy));
      mlir::Value denominator = rewriter.create<mlir::math::SqrtOp>(
          loc,
          rewriter.create<mlir::arith::AddFOp>(
              loc,
              loadAsAccumulator(
                  rewriter, loc, variance, parameterIndices, accTy),
              floatConst(rewriter, loc, accTy, epsilon)));
      mlir::Value result = rewriter.create<mlir::arith::DivFOp>(
          loc, centered, denominator);
      size_t optionalIndex = 3;
      if (applyScale) {
        result = rewriter.create<mlir::arith::MulFOp>(
            loc, result,
            loadAsAccumulator(
                rewriter, loc, inputs[optionalIndex++],
                parameterIndices, accTy));
      }
      if (applyOffset) {
        result = rewriter.create<mlir::arith::AddFOp>(
            loc, result,
            loadAsAccumulator(
                rewriter, loc, inputs[optionalIndex],
                parameterIndices, accTy));
      }
      storeFromAccumulator(rewriter, loc, result, output, indices);
      rewriter.create<mlir::gpu::TerminatorOp>(loc);
      rewriter.eraseOp(op);
      return mlir::success();
    }

    case VulkanKernelRecipe::VISION_EMBEDDING_MERGE: {
      if (inputs.size() != 3 || outputs.size() != 1) {
        return op.emitOpError(
            "vision embedding merge operand contract mismatch");
      }
      mlir::Value textEmbeddings = inputs[0];
      mlir::Value visionEmbeddings = inputs[1];
      mlir::Value tokenIds = inputs[2];
      mlir::Value output = outputs.front();
      auto outputType = llvm::dyn_cast<mlir::MemRefType>(output.getType());
      auto tokenMemrefType =
          llvm::dyn_cast<mlir::MemRefType>(tokenIds.getType());
      auto tokenType = tokenMemrefType
                           ? llvm::dyn_cast<mlir::IntegerType>(
                                 tokenMemrefType.getElementType())
                           : mlir::IntegerType{};
      auto targetTokenIdAttr =
          op->getAttrOfType<mlir::IntegerAttr>("nd4j.target_token_id");
      auto tokenUnsignedAttr =
          op->getAttrOfType<mlir::BoolAttr>("nd4j.token_unsigned");
      auto targetInRangeAttr =
          op->getAttrOfType<mlir::BoolAttr>("nd4j.target_in_range");
      if (!outputType || outputType.getRank() != 3 || !tokenType ||
          tokenType.getWidth() != 32 || !targetTokenIdAttr ||
          !tokenUnsignedAttr || !targetInRangeAttr) {
        return op.emitOpError(
            "vision embedding merge requires rank-3 output, i32 token storage, and frozen arguments");
      }
      (void)tokenUnsignedAttr;

      mlir::Value total = one;
      for (int64_t d = 0; d < 3; ++d) {
        total = rewriter.create<mlir::arith::MulIOp>(
            loc, total,
            rewriter.create<mlir::memref::DimOp>(loc, output, d));
      }
      auto launch = createGpuLaunch(rewriter, loc, total, one, one);
      mlir::Value flat = launch.getBlockIds().x;
      rewriter.setInsertionPointToEnd(&launch.getBody().front());
      auto indices = logicalIndices(rewriter, loc, flat, output);
      mlir::Value batchIndex = indices[0];
      mlir::Value sequenceIndex = indices[1];
      mlir::Value hiddenIndex = indices[2];
      const bool targetInRange = targetInRangeAttr.getValue();
      mlir::Value falseValue =
          rewriter.create<mlir::arith::ConstantIntOp>(loc, 0, 1);
      mlir::Value targetToken;
      if (targetInRange) {
        targetToken = rewriter.create<mlir::arith::ConstantIntOp>(
            loc, targetTokenIdAttr.getInt(), tokenType.getWidth());
      }

      mlir::Value visionIndex = zero;
      if (targetInRange) {
        auto prefix = emitReductionLoop(
            rewriter, loc, zero, sequenceIndex, one, zero,
            [&](mlir::OpBuilder& builder, mlir::Location nestedLoc,
                mlir::Value priorSequence,
                mlir::Value accumulator) -> mlir::Value {
              mlir::Value priorToken =
                  builder.create<mlir::memref::LoadOp>(
                      nestedLoc, tokenIds,
                      mlir::SmallVector<mlir::Value>{
                          batchIndex, priorSequence});
              mlir::Value matches = builder.create<mlir::arith::CmpIOp>(
                  nestedLoc, mlir::arith::CmpIPredicate::eq,
                  priorToken, targetToken);
              return builder.create<mlir::arith::SelectOp>(
                  nestedLoc, matches,
                  builder.create<mlir::arith::AddIOp>(
                      nestedLoc, accumulator, one),
                  accumulator);
            });
        visionIndex = prefix.getResult(0);
      }

      mlir::Value currentMatches = falseValue;
      if (targetInRange) {
        mlir::Value currentToken =
            rewriter.create<mlir::memref::LoadOp>(
                loc, tokenIds,
                mlir::SmallVector<mlir::Value>{
                    batchIndex, sequenceIndex});
        currentMatches = rewriter.create<mlir::arith::CmpIOp>(
            loc, mlir::arith::CmpIPredicate::eq,
            currentToken, targetToken);
      }
      mlir::Value visionTokens =
          rewriter.create<mlir::memref::DimOp>(
              loc, visionEmbeddings, 1);
      mlir::Value hasVisionToken = rewriter.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::ult,
          visionIndex, visionTokens);
      mlir::Value useVision = rewriter.create<mlir::arith::AndIOp>(
          loc, currentMatches, hasVisionToken);
      auto source = rewriter.create<mlir::scf::IfOp>(
          loc, mlir::TypeRange{accTy}, useVision, true);

      rewriter.setInsertionPointToStart(source.thenBlock());
      mlir::Value visionValue = loadAsAccumulator(
          rewriter, loc, visionEmbeddings,
          mlir::SmallVector<mlir::Value>{
              batchIndex, visionIndex, hiddenIndex},
          accTy);
      rewriter.create<mlir::scf::YieldOp>(
          loc, mlir::ValueRange{visionValue});

      rewriter.setInsertionPointToStart(source.elseBlock());
      mlir::Value textValue = loadAsAccumulator(
          rewriter, loc, textEmbeddings, indices, accTy);
      rewriter.create<mlir::scf::YieldOp>(
          loc, mlir::ValueRange{textValue});

      rewriter.setInsertionPointAfter(source);
      storeFromAccumulator(
          rewriter, loc, source.getResult(0), output, indices);
      rewriter.create<mlir::gpu::TerminatorOp>(loc);
      rewriter.eraseOp(op);
      return mlir::success();
    }

    case VulkanKernelRecipe::APPLY_ALIBI: {
      if (inputs.size() != 1 || outputs.size() != 1) {
        return op.emitOpError("ALiBi operand contract mismatch");
      }
      mlir::Value input = inputs.front();
      mlir::Value output = outputs.front();
      auto inputType = llvm::dyn_cast<mlir::MemRefType>(input.getType());
      auto outputType = llvm::dyn_cast<mlir::MemRefType>(output.getType());
      auto numHeadsAttr =
          op->getAttrOfType<mlir::IntegerAttr>("nd4j.num_heads");
      if (!inputType || !outputType || inputType.getRank() != 4 ||
          outputType.getRank() != 4 || !numHeadsAttr ||
          numHeadsAttr.getInt() <= 0) {
        return op.emitOpError(
            "ALiBi requires matching rank-4 tensors and frozen head count");
      }

      mlir::Value total = one;
      for (int64_t d = 0; d < 4; ++d) {
        total = rewriter.create<mlir::arith::MulIOp>(
            loc, total,
            rewriter.create<mlir::memref::DimOp>(loc, output, d));
      }
      auto launch = createGpuLaunch(rewriter, loc, total, one, one);
      mlir::Value flat = launch.getBlockIds().x;
      rewriter.setInsertionPointToEnd(&launch.getBody().front());
      auto indices = logicalIndices(rewriter, loc, flat, output);
      mlir::Value head = indices[1];
      mlir::Value queryPosition = indices[2];
      mlir::Value keyPosition = indices[3];

      mlir::Value headCount = floatConst(
          rewriter, loc, accTy,
          static_cast<double>(numHeadsAttr.getInt()));
      mlir::Value baseExponent = rewriter.create<mlir::arith::DivFOp>(
          loc, floatConst(rewriter, loc, accTy, -8.0), headCount);
      mlir::Value base = rewriter.create<mlir::math::PowFOp>(
          loc, floatConst(rewriter, loc, accTy, 2.0), baseExponent);
      mlir::Value oneBasedHead = convertIndexToFloat(
          rewriter, loc,
          rewriter.create<mlir::arith::AddIOp>(loc, head, one), accTy);
      mlir::Value slope = rewriter.create<mlir::math::PowFOp>(
          loc, base, oneBasedHead);
      mlir::Value queryAtOrAfterKey =
          rewriter.create<mlir::arith::CmpIOp>(
              loc, mlir::arith::CmpIPredicate::uge,
              queryPosition, keyPosition);
      mlir::Value integerDistance =
          rewriter.create<mlir::arith::SelectOp>(
              loc, queryAtOrAfterKey,
              rewriter.create<mlir::arith::SubIOp>(
                  loc, queryPosition, keyPosition),
              rewriter.create<mlir::arith::SubIOp>(
                  loc, keyPosition, queryPosition));
      mlir::Value distance =
          convertIndexToFloat(rewriter, loc, integerDistance, accTy);
      mlir::Value value = loadAsAccumulator(
          rewriter, loc, input, indices, accTy);
      mlir::Value result = rewriter.create<mlir::arith::SubFOp>(
          loc, value,
          rewriter.create<mlir::arith::MulFOp>(loc, slope, distance));
      storeFromAccumulator(rewriter, loc, result, output, indices);
      rewriter.create<mlir::gpu::TerminatorOp>(loc);
      rewriter.eraseOp(op);
      return mlir::success();
    }

    case VulkanKernelRecipe::ROPE:
    case VulkanKernelRecipe::ROPE_BP: {
      const bool backward =
          hasVulkanOpTrait(*emitter, sd::ops::OP_TRAIT_BACKWARD);
      if ((!backward && inputs.size() != 1) ||
          (backward && inputs.size() != 2) || outputs.size() != 1) {
        return op.emitOpError("RoPE operand contract mismatch");
      }
      mlir::Value input = inputs[backward ? 1 : 0];
      mlir::Value output = outputs.front();
      auto inputType = llvm::dyn_cast<mlir::MemRefType>(input.getType());
      auto outputType = llvm::dyn_cast<mlir::MemRefType>(output.getType());
      if (!inputType || !outputType ||
          (inputType.getRank() != 3 && inputType.getRank() != 4) ||
          outputType.getRank() != inputType.getRank()) {
        return op.emitOpError("RoPE requires matching rank-3 or rank-4 tensors");
      }

      auto ropeTypeAttr =
          op->getAttrOfType<mlir::IntegerAttr>("nd4j.rope_type");
      auto positionOffsetAttr =
          op->getAttrOfType<mlir::IntegerAttr>("nd4j.position_offset");
      auto rotaryDimensionsAttr =
          op->getAttrOfType<mlir::IntegerAttr>("nd4j.rotary_dims");
      auto frequencyBaseAttr =
          op->getAttrOfType<mlir::FloatAttr>("nd4j.frequency_base");
      auto frequencyScaleAttr =
          op->getAttrOfType<mlir::FloatAttr>("nd4j.frequency_scale");
      if (!ropeTypeAttr || !positionOffsetAttr || !rotaryDimensionsAttr ||
          !frequencyBaseAttr || !frequencyScaleAttr) {
        return op.emitOpError("RoPE requires frozen canonical arguments");
      }
      const bool adjacentPairs = ropeTypeAttr.getInt() == 1;
      const int64_t positionOffset = positionOffsetAttr.getInt();
      const int64_t rotaryDimensions = rotaryDimensionsAttr.getInt();
      if (rotaryDimensions <= 0 ||
          (rotaryDimensions > 1 && rotaryDimensions % 2 != 0)) {
        return op.emitOpError(
            "RoPE rotary dimensions must be one or a positive even value");
      }

      mlir::Value total = one;
      for (int64_t d = 0; d < outputType.getRank(); ++d) {
        total = rewriter.create<mlir::arith::MulIOp>(
            loc, total,
            rewriter.create<mlir::memref::DimOp>(loc, output, d));
      }
      auto launch = createGpuLaunch(rewriter, loc, total, one, one);
      mlir::Value flat = launch.getBlockIds().x;
      rewriter.setInsertionPointToEnd(&launch.getBody().front());
      auto indices = logicalIndices(rewriter, loc, flat, output);
      if (rotaryDimensions == 1) {
        storeFromAccumulator(
            rewriter, loc,
            loadAsAccumulator(rewriter, loc, input, indices, accTy),
            output, indices);
        rewriter.create<mlir::gpu::TerminatorOp>(loc);
        rewriter.eraseOp(op);
        return mlir::success();
      }
      mlir::Value feature = indices.back();
      mlir::Value rotaryDimensionsIndex =
          idxConst(rewriter, loc, rotaryDimensions);
      mlir::Value two = idxConst(rewriter, loc, 2);
      mlir::Value halfRotaryDimensions =
          rewriter.create<mlir::arith::DivUIOp>(
              loc, rotaryDimensionsIndex, two);
      mlir::Value inRotaryRange = rewriter.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::ult, feature,
          rotaryDimensionsIndex);
      mlir::Value safeFeature = rewriter.create<mlir::arith::SelectOp>(
          loc, inRotaryRange, feature, zero);

      mlir::Value pairIndex;
      mlir::Value partnerFeature;
      mlir::Value firstInPair;
      if (adjacentPairs) {
        pairIndex = rewriter.create<mlir::arith::DivUIOp>(
            loc, safeFeature, two);
        firstInPair = rewriter.create<mlir::arith::CmpIOp>(
            loc, mlir::arith::CmpIPredicate::eq,
            rewriter.create<mlir::arith::RemUIOp>(loc, safeFeature, two),
            zero);
        partnerFeature = rewriter.create<mlir::arith::SelectOp>(
            loc, firstInPair,
            rewriter.create<mlir::arith::AddIOp>(loc, safeFeature, one),
            rewriter.create<mlir::arith::SubIOp>(loc, safeFeature, one));
      } else {
        firstInPair = rewriter.create<mlir::arith::CmpIOp>(
            loc, mlir::arith::CmpIPredicate::ult, safeFeature,
            halfRotaryDimensions);
        pairIndex = rewriter.create<mlir::arith::SelectOp>(
            loc, firstInPair, safeFeature,
            rewriter.create<mlir::arith::SubIOp>(
                loc, safeFeature, halfRotaryDimensions));
        partnerFeature = rewriter.create<mlir::arith::SelectOp>(
            loc, firstInPair,
            rewriter.create<mlir::arith::AddIOp>(
                loc, safeFeature, halfRotaryDimensions),
            rewriter.create<mlir::arith::SubIOp>(
                loc, safeFeature, halfRotaryDimensions));
      }

      auto partnerIndices = indices;
      partnerIndices.back() = partnerFeature;
      mlir::Value current =
          loadAsAccumulator(rewriter, loc, input, indices, accTy);
      mlir::Value partner = loadAsAccumulator(
          rewriter, loc, input, partnerIndices, accTy);
      mlir::Value pairFloat =
          convertIndexToFloat(rewriter, loc, pairIndex, accTy);
      mlir::Value rotaryDimensionsFloat =
          floatConst(rewriter, loc, accTy,
                     static_cast<double>(rotaryDimensions));
      mlir::Value exponent = rewriter.create<mlir::arith::DivFOp>(
          loc,
          rewriter.create<mlir::arith::MulFOp>(
              loc, floatConst(rewriter, loc, accTy, 2.0), pairFloat),
          rotaryDimensionsFloat);
      mlir::Value inverseFrequency = rewriter.create<mlir::arith::DivFOp>(
          loc,
          floatConst(rewriter, loc, accTy,
                     frequencyScaleAttr.getValueAsDouble()),
          rewriter.create<mlir::math::PowFOp>(
              loc,
              floatConst(rewriter, loc, accTy,
                         frequencyBaseAttr.getValueAsDouble()),
              exponent));
      mlir::Value integerPosition = rewriter.create<mlir::arith::AddIOp>(
          loc, indices[1], idxConst(rewriter, loc, positionOffset));
      mlir::Value sequencePosition =
          convertIndexToFloat(rewriter, loc, integerPosition, accTy, false);
      mlir::Value angle = rewriter.create<mlir::arith::MulFOp>(
          loc, sequencePosition, inverseFrequency);
      mlir::Value cosine =
          rewriter.create<mlir::math::CosOp>(loc, angle);
      mlir::Value sine = rewriter.create<mlir::math::SinOp>(loc, angle);
      mlir::Value currentCosine =
          rewriter.create<mlir::arith::MulFOp>(loc, current, cosine);
      mlir::Value partnerSine =
          rewriter.create<mlir::arith::MulFOp>(loc, partner, sine);
      mlir::Value firstResult =
          backward
              ? mlir::Value(rewriter.create<mlir::arith::AddFOp>(
                    loc, currentCosine, partnerSine))
              : mlir::Value(rewriter.create<mlir::arith::SubFOp>(
                    loc, currentCosine, partnerSine));
      mlir::Value secondResult =
          backward
              ? mlir::Value(rewriter.create<mlir::arith::SubFOp>(
                    loc, currentCosine, partnerSine))
              : mlir::Value(rewriter.create<mlir::arith::AddFOp>(
                    loc, partnerSine, currentCosine));
      mlir::Value rotated = rewriter.create<mlir::arith::SelectOp>(
          loc, firstInPair, firstResult, secondResult);
      mlir::Value result = rewriter.create<mlir::arith::SelectOp>(
          loc, inRotaryRange, rotated, current);
      storeFromAccumulator(rewriter, loc, result, output, indices);
      rewriter.create<mlir::gpu::TerminatorOp>(loc);
      rewriter.eraseOp(op);
      return mlir::success();
    }

    case VulkanKernelRecipe::FUSED_MROPE: {
      if (inputs.size() != 4 || outputs.size() != 1) {
        return op.emitOpError("fused M-RoPE operand contract mismatch");
      }
      mlir::Value input = inputs[0];
      mlir::Value positionT = inputs[1];
      mlir::Value positionH = inputs[2];
      mlir::Value positionW = inputs[3];
      mlir::Value output = outputs[0];
      auto inputType = llvm::dyn_cast<mlir::MemRefType>(input.getType());
      if (!inputType || inputType.getRank() != 4) {
        return op.emitOpError("fused M-RoPE input must be rank 4");
      }
      const int64_t sectionT =
          op->getAttrOfType<mlir::IntegerAttr>("nd4j.section_t").getInt();
      const int64_t sectionH =
          op->getAttrOfType<mlir::IntegerAttr>("nd4j.section_h").getInt();
      const int64_t sectionW =
          op->getAttrOfType<mlir::IntegerAttr>("nd4j.section_w").getInt();
      const bool interleaved =
          op->getAttrOfType<mlir::BoolAttr>("nd4j.interleaved").getValue();
      const double frequencyBase =
          op->getAttrOfType<mlir::FloatAttr>("nd4j.frequency_base")
              .getValueAsDouble();
      const bool positionUnsigned =
          op->getAttrOfType<mlir::BoolAttr>("nd4j.position_unsigned")
              .getValue();

      mlir::Value batch =
          rewriter.create<mlir::memref::DimOp>(loc, input, 0);
      mlir::Value sequence =
          rewriter.create<mlir::memref::DimOp>(loc, input, 1);
      mlir::Value heads =
          rewriter.create<mlir::memref::DimOp>(loc, input, 2);
      mlir::Value headDimension =
          rewriter.create<mlir::memref::DimOp>(loc, input, 3);
      mlir::Value twoIndex = idxConst(rewriter, loc, 2);
      mlir::Value threeIndex = idxConst(rewriter, loc, 3);
      mlir::Value halfDimension = rewriter.create<mlir::arith::DivUIOp>(
          loc, headDimension, twoIndex);
      mlir::Value batchSequence = rewriter.create<mlir::arith::MulIOp>(
          loc, batch, sequence);
      auto launch = createGpuLaunch(
          rewriter, loc, halfDimension, heads, batchSequence);
      mlir::Value dimension = launch.getBlockIds().x;
      mlir::Value head = launch.getBlockIds().y;
      mlir::Value flatBatchSequence = launch.getBlockIds().z;
      rewriter.setInsertionPointToEnd(&launch.getBody().front());
      mlir::Value batchIndex = rewriter.create<mlir::arith::DivUIOp>(
          loc, flatBatchSequence, sequence);
      mlir::Value sequenceIndex = rewriter.create<mlir::arith::RemUIOp>(
          loc, flatBatchSequence, sequence);
      mlir::Value pairedDimension = rewriter.create<mlir::arith::AddIOp>(
          loc, dimension, halfDimension);
      mlir::Value positionTemporal = loadAsScalar(
          rewriter, loc, positionT,
          mlir::SmallVector<mlir::Value>{batchIndex, sequenceIndex}, accTy,
          positionUnsigned, false);
      mlir::Value positionHeight = loadAsScalar(
          rewriter, loc, positionH,
          mlir::SmallVector<mlir::Value>{batchIndex, sequenceIndex}, accTy,
          positionUnsigned, false);
      mlir::Value positionWidth = loadAsScalar(
          rewriter, loc, positionW,
          mlir::SmallVector<mlir::Value>{batchIndex, sequenceIndex}, accTy,
          positionUnsigned, false);

      mlir::Value position;
      mlir::Value localDimension;
      mlir::Value sectionSize;
      double effectiveFrequencyBase = frequencyBase;
      if (interleaved) {
        mlir::Value selector = rewriter.create<mlir::arith::RemUIOp>(
            loc, dimension, threeIndex);
        mlir::Value selectorZero = rewriter.create<mlir::arith::CmpIOp>(
            loc, mlir::arith::CmpIPredicate::eq, selector, zero);
        mlir::Value selectorOne = rewriter.create<mlir::arith::CmpIOp>(
            loc, mlir::arith::CmpIPredicate::eq, selector, one);
        position = rewriter.create<mlir::arith::SelectOp>(
            loc, selectorZero, positionTemporal,
            rewriter.create<mlir::arith::SelectOp>(
                loc, selectorOne, positionHeight, positionWidth));
        localDimension = rewriter.create<mlir::arith::DivUIOp>(
            loc, dimension, threeIndex);
        sectionSize = rewriter.create<mlir::arith::DivUIOp>(
            loc,
            rewriter.create<mlir::arith::AddIOp>(
                loc, headDimension, twoIndex),
            threeIndex);
        effectiveFrequencyBase = 10000.0;
      } else {
        mlir::Value halfTemporal = idxConst(rewriter, loc, sectionT / 2);
        mlir::Value halfHeight = idxConst(rewriter, loc, sectionH / 2);
        mlir::Value temporalHeightBoundary =
            rewriter.create<mlir::arith::AddIOp>(
                loc, halfTemporal, halfHeight);
        mlir::Value inTemporal = rewriter.create<mlir::arith::CmpIOp>(
            loc, mlir::arith::CmpIPredicate::ult, dimension, halfTemporal);
        mlir::Value inHeight = rewriter.create<mlir::arith::CmpIOp>(
            loc, mlir::arith::CmpIPredicate::ult, dimension,
            temporalHeightBoundary);
        position = rewriter.create<mlir::arith::SelectOp>(
            loc, inTemporal, positionTemporal,
            rewriter.create<mlir::arith::SelectOp>(
                loc, inHeight, positionHeight, positionWidth));
        localDimension = rewriter.create<mlir::arith::SelectOp>(
            loc, inTemporal, dimension,
            rewriter.create<mlir::arith::SelectOp>(
                loc, inHeight,
                rewriter.create<mlir::arith::SubIOp>(
                    loc, dimension, halfTemporal),
                rewriter.create<mlir::arith::SubIOp>(
                    loc, dimension, temporalHeightBoundary)));
        sectionSize = rewriter.create<mlir::arith::SelectOp>(
            loc, inTemporal, idxConst(rewriter, loc, sectionT),
            rewriter.create<mlir::arith::SelectOp>(
                loc, inHeight, idxConst(rewriter, loc, sectionH),
                idxConst(rewriter, loc, sectionW)));
      }

      mlir::Value localFloat =
          convertIndexToFloat(rewriter, loc, localDimension, accTy);
      mlir::Value sectionFloat =
          convertIndexToFloat(rewriter, loc, sectionSize, accTy);
      mlir::Value exponent = rewriter.create<mlir::arith::DivFOp>(
          loc,
          rewriter.create<mlir::arith::MulFOp>(
              loc, floatConst(rewriter, loc, accTy, 2.0), localFloat),
          sectionFloat);
      mlir::Value frequency = rewriter.create<mlir::arith::DivFOp>(
          loc, floatConst(rewriter, loc, accTy, 1.0),
          rewriter.create<mlir::math::PowFOp>(
              loc,
              floatConst(
                  rewriter, loc, accTy, effectiveFrequencyBase),
              exponent));
      mlir::Value angle = rewriter.create<mlir::arith::MulFOp>(
          loc, position, frequency);
      mlir::Value cosine = rewriter.create<mlir::math::CosOp>(loc, angle);
      mlir::Value sine = rewriter.create<mlir::math::SinOp>(loc, angle);
      mlir::Value first = loadAsAccumulator(
          rewriter, loc, input,
          mlir::SmallVector<mlir::Value>{
              batchIndex, sequenceIndex, head, dimension},
          accTy);
      mlir::Value second = loadAsAccumulator(
          rewriter, loc, input,
          mlir::SmallVector<mlir::Value>{
              batchIndex, sequenceIndex, head, pairedDimension},
          accTy);
      mlir::Value rotatedFirst = rewriter.create<mlir::arith::SubFOp>(
          loc, rewriter.create<mlir::arith::MulFOp>(loc, first, cosine),
          rewriter.create<mlir::arith::MulFOp>(loc, second, sine));
      mlir::Value rotatedSecond = rewriter.create<mlir::arith::AddFOp>(
          loc, rewriter.create<mlir::arith::MulFOp>(loc, first, sine),
          rewriter.create<mlir::arith::MulFOp>(loc, second, cosine));
      storeFromAccumulator(
          rewriter, loc, rotatedFirst, output,
          mlir::SmallVector<mlir::Value>{
              batchIndex, sequenceIndex, head, dimension});
      storeFromAccumulator(
          rewriter, loc, rotatedSecond, output,
          mlir::SmallVector<mlir::Value>{
              batchIndex, sequenceIndex, head, pairedDimension});
      rewriter.create<mlir::gpu::TerminatorOp>(loc);
      rewriter.eraseOp(op);
      return mlir::success();
    }

    case VulkanKernelRecipe::FUSED_BIAS_DROPOUT_RESIDUAL: {
      if (inputs.size() != 3 || outputs.size() != 1) {
        return op.emitOpError(
            "fused bias-dropout-residual operand contract mismatch");
      }
      mlir::Value input = inputs[0];
      mlir::Value bias = inputs[1];
      mlir::Value residual = inputs[2];
      mlir::Value output = outputs[0];
      auto outputType = llvm::dyn_cast<mlir::MemRefType>(output.getType());
      if (!outputType || outputType.getRank() < 1) {
        return op.emitOpError(
            "fused bias-dropout-residual output must be a ranked MemRef");
      }
      mlir::Value total = one;
      for (int64_t d = 0; d < outputType.getRank(); ++d) {
        total = rewriter.create<mlir::arith::MulIOp>(
            loc, total,
            rewriter.create<mlir::memref::DimOp>(loc, output, d));
      }
      auto launch = createGpuLaunch(rewriter, loc, total, one, one);
      mlir::Value flat = launch.getBlockIds().x;
      rewriter.setInsertionPointToEnd(&launch.getBody().front());
      auto indices = logicalIndices(rewriter, loc, flat, output);
      mlir::Value value = rewriter.create<mlir::arith::AddFOp>(
          loc,
          loadAsAccumulator(rewriter, loc, input, indices, accTy),
          loadAsAccumulator(
              rewriter, loc, bias,
              mlir::SmallVector<mlir::Value>{indices.back()}, accTy));
      value = rewriter.create<mlir::arith::AddFOp>(
          loc, value,
          loadAsAccumulator(rewriter, loc, residual, indices, accTy));
      storeFromAccumulator(rewriter, loc, value, output, indices);
      rewriter.create<mlir::gpu::TerminatorOp>(loc);
      rewriter.eraseOp(op);
      return mlir::success();
    }

    case VulkanKernelRecipe::SWISH_MUL_BP: {
      if (inputs.size() != 3 || outputs.size() != 2) {
        return op.emitOpError(
            "swish_mul_bp requires x, y, gradOut and two gradient outputs");
      }
      mlir::Value x = inputs[0];
      mlir::Value y = inputs[1];
      mlir::Value gradOut = inputs[2];
      mlir::Value gradX = outputs[0];
      mlir::Value gradY = outputs[1];
      auto outputType = llvm::dyn_cast<mlir::MemRefType>(gradX.getType());
      if (!outputType) {
        return op.emitOpError("swish_mul_bp output must be a MemRef");
      }

      mlir::Value total = one;
      for (int64_t d = 0; d < outputType.getRank(); ++d) {
        total = rewriter.create<mlir::arith::MulIOp>(
            loc, total,
            rewriter.create<mlir::memref::DimOp>(loc, gradX, d));
      }
      auto launch = createGpuLaunch(rewriter, loc, total, one, one);
      mlir::Value flat = launch.getBlockIds().x;
      rewriter.setInsertionPointToEnd(&launch.getBody().front());
      auto indices = logicalIndices(rewriter, loc, flat, gradX);

      mlir::Value xValue = loadAsAccumulator(
          rewriter, loc, x, indices, accTy);
      mlir::Value yValue = loadAsAccumulator(
          rewriter, loc, y, indices, accTy);
      mlir::Value gradient = loadAsAccumulator(
          rewriter, loc, gradOut, indices, accTy);
      mlir::Value sigmoid = emitSigmoid(
          rewriter, loc, accTy, xValue);
      mlir::Value sigmoidDerivative = rewriter.create<mlir::arith::MulFOp>(
          loc, sigmoid,
          rewriter.create<mlir::arith::SubFOp>(
              loc, floatConst(rewriter, loc, accTy, 1.0), sigmoid));
      mlir::Value siluDerivative = rewriter.create<mlir::arith::AddFOp>(
          loc, sigmoid,
          rewriter.create<mlir::arith::MulFOp>(
              loc, xValue, sigmoidDerivative));
      mlir::Value gradXValue = rewriter.create<mlir::arith::MulFOp>(
          loc,
          rewriter.create<mlir::arith::MulFOp>(
              loc, gradient, yValue),
          siluDerivative);
      mlir::Value gradYValue = rewriter.create<mlir::arith::MulFOp>(
          loc, gradient,
          rewriter.create<mlir::arith::MulFOp>(
              loc, xValue, sigmoid));
      storeFromAccumulator(rewriter, loc, gradXValue, gradX, indices);
      storeFromAccumulator(rewriter, loc, gradYValue, gradY, indices);
      rewriter.create<mlir::gpu::TerminatorOp>(loc);
      rewriter.eraseOp(op);
      return mlir::success();
    }

    case VulkanKernelRecipe::FUSED_ELEMENTWISE_CHAIN: {
      if (inputs.empty() || outputs.size() != 1) {
        return op.emitOpError(
            "fused elementwise chain operand contract mismatch");
      }
      auto codesAttr =
          op->getAttrOfType<mlir::DenseI64ArrayAttr>("nd4j.chain_ops");
      if (!codesAttr || codesAttr.empty() || codesAttr.size() > 8) {
        return op.emitOpError(
            "fused elementwise chain requires 1..8 frozen op codes");
      }
      mlir::Value output = outputs.front();
      auto outputType = llvm::dyn_cast<mlir::MemRefType>(output.getType());
      if (!outputType) {
        return op.emitOpError(
            "fused elementwise chain output must be a MemRef");
      }
      mlir::Value total = one;
      for (int64_t d = 0; d < outputType.getRank(); ++d) {
        total = rewriter.create<mlir::arith::MulIOp>(
            loc, total,
            rewriter.create<mlir::memref::DimOp>(loc, output, d));
      }
      mlir::Value clipMin;
      mlir::Value clipMax;
      if (auto attr = op->getAttrOfType<mlir::FloatAttr>("nd4j.clip_min")) {
        clipMin = floatConst(rewriter, loc, accTy, attr.getValueAsDouble());
      }
      if (auto attr = op->getAttrOfType<mlir::FloatAttr>("nd4j.clip_max")) {
        clipMax = floatConst(rewriter, loc, accTy, attr.getValueAsDouble());
      }

      auto launch = createGpuLaunch(rewriter, loc, total, one, one);
      mlir::Value flat = launch.getBlockIds().x;
      rewriter.setInsertionPointToEnd(&launch.getBody().front());
      auto indices = logicalIndices(rewriter, loc, flat, output);
      mlir::Value value = loadAsAccumulator(
          rewriter, loc, inputs.front(), indices, accTy);
      size_t secondaryInput = 1;
      for (int64_t code : codesAttr.asArrayRef()) {
        const bool binary = (code >= 0 && code <= 3) || code == 31 ||
                            (code >= 50 && code <= 59);
        mlir::Value secondary;
        if (binary) {
          if (secondaryInput >= inputs.size()) {
            return op.emitOpError(
                "fused elementwise chain is missing a secondary input");
          }
          secondary = loadAsAccumulator(
              rewriter, loc, inputs[secondaryInput++], indices, accTy);
        }
        mlir::Value next = emitFusedChainStep(
            rewriter, loc, accTy, code, value, secondary, clipMin, clipMax);
        if (!next) {
          return op.emitOpError(
              "fused elementwise chain contains an unsupported op code");
        }
        value = next;
      }
      if (secondaryInput != inputs.size()) {
        return op.emitOpError(
            "fused elementwise chain has unused secondary inputs");
      }
      storeFromAccumulator(rewriter, loc, value, output, indices);
      rewriter.create<mlir::gpu::TerminatorOp>(loc);
      rewriter.eraseOp(op);
      return mlir::success();
    }

    case VulkanKernelRecipe::FUSED_ATTENTION_PROJECTION: {
      if ((inputs.size() != 2 && inputs.size() != 3) ||
          outputs.size() != 1) {
        return op.emitOpError(
            "fused attention projection operand contract mismatch");
      }
      mlir::Value input = inputs[0];
      mlir::Value weight = inputs[1];
      mlir::Value output = outputs[0];
      auto inputType = llvm::dyn_cast<mlir::MemRefType>(input.getType());
      auto outputType = llvm::dyn_cast<mlir::MemRefType>(output.getType());
      if (!inputType || !outputType ||
          (inputType.getRank() != 3 && inputType.getRank() != 4) ||
          outputType.getRank() != 3) {
        return op.emitOpError(
            "fused attention projection requires rank-3/4 input and rank-3 output");
      }
      mlir::Value total = one;
      for (int64_t d = 0; d < outputType.getRank(); ++d) {
        total = rewriter.create<mlir::arith::MulIOp>(
            loc, total,
            rewriter.create<mlir::memref::DimOp>(loc, output, d));
      }
      mlir::Value hidden =
          rewriter.create<mlir::memref::DimOp>(loc, weight, 0);
      auto launch = createGpuLaunch(rewriter, loc, total, one, one);
      mlir::Value flat = launch.getBlockIds().x;
      rewriter.setInsertionPointToEnd(&launch.getBody().front());
      auto outputIndices = logicalIndices(rewriter, loc, flat, output);
      mlir::Value initial =
          inputs.size() == 3
              ? loadAsAccumulator(
                    rewriter, loc, inputs[2],
                    mlir::SmallVector<mlir::Value>{outputIndices[2]}, accTy)
              : floatConst(rewriter, loc, accTy, 0.0);
      mlir::Value headDim =
          inputType.getRank() == 4
              ? mlir::Value(rewriter.create<mlir::memref::DimOp>(loc, input, 3))
              : mlir::Value{};
      auto dot = emitReductionLoop(
          rewriter, loc, zero, hidden, one, initial,
          [&](mlir::OpBuilder& body, mlir::Location bodyLoc,
              mlir::Value inner, mlir::Value accumulator) -> mlir::Value {
            llvm::SmallVector<mlir::Value> inputIndices{
                outputIndices[0], outputIndices[1]};
            if (inputType.getRank() == 3) {
              inputIndices.push_back(inner);
            } else {
              inputIndices.push_back(body.create<mlir::arith::DivUIOp>(
                  bodyLoc, inner, headDim));
              inputIndices.push_back(body.create<mlir::arith::RemUIOp>(
                  bodyLoc, inner, headDim));
            }
            mlir::Value lhs = loadAsAccumulator(
                body, bodyLoc, input, inputIndices, accTy);
            mlir::Value rhs = loadAsAccumulator(
                body, bodyLoc, weight,
                mlir::SmallVector<mlir::Value>{inner, outputIndices[2]},
                accTy);
            return body.create<mlir::arith::AddFOp>(
                bodyLoc, accumulator,
                body.create<mlir::arith::MulFOp>(bodyLoc, lhs, rhs));
          });
      storeFromAccumulator(
          rewriter, loc, dot.getResult(0), output, outputIndices);
      rewriter.create<mlir::gpu::TerminatorOp>(loc);
      rewriter.eraseOp(op);
      return mlir::success();
    }

    case VulkanKernelRecipe::SWIGLU:
    case VulkanKernelRecipe::GEGLU:
    case VulkanKernelRecipe::REGLU: {
      if (inputs.size() != 1 || outputs.size() != 1) {
        return op.emitOpError("GLU operand contract mismatch");
      }
      mlir::Value input = inputs.front();
      mlir::Value output = outputs.front();
      auto outputType = llvm::dyn_cast<mlir::MemRefType>(output.getType());
      if (!outputType || outputType.getRank() < 1) {
        return op.emitOpError("GLU output must be a ranked MemRef");
      }
      const int64_t rank = outputType.getRank();
      mlir::Value total = one;
      for (int64_t d = 0; d < rank; ++d) {
        total = rewriter.create<mlir::arith::MulIOp>(
            loc, total,
            rewriter.create<mlir::memref::DimOp>(loc, output, d));
      }
      mlir::Value half =
          rewriter.create<mlir::memref::DimOp>(loc, output, rank - 1);
      auto launch = createGpuLaunch(rewriter, loc, total, one, one);
      mlir::Value flat = launch.getBlockIds().x;
      rewriter.setInsertionPointToEnd(&launch.getBody().front());
      auto gateIndices = logicalIndices(rewriter, loc, flat, output);
      auto upIndices = gateIndices;
      upIndices.back() = rewriter.create<mlir::arith::AddIOp>(
          loc, upIndices.back(), half);
      mlir::Value gate = loadAsAccumulator(
          rewriter, loc, input, gateIndices, accTy);
      mlir::Value up = loadAsAccumulator(
          rewriter, loc, input, upIndices, accTy);
      mlir::Value activated;
      if (emitter->recipe == VulkanKernelRecipe::SWIGLU) {
        activated = emitSilu(rewriter, loc, accTy, gate);
      } else if (emitter->recipe == VulkanKernelRecipe::GEGLU) {
        activated = emitGelu(rewriter, loc, accTy, gate);
      } else {
        activated = rewriter.create<mlir::arith::MaximumFOp>(
            loc, gate, floatConst(rewriter, loc, accTy, 0.0));
      }
      storeFromAccumulator(
          rewriter, loc,
          rewriter.create<mlir::arith::MulFOp>(loc, activated, up), output,
          gateIndices);
      rewriter.create<mlir::gpu::TerminatorOp>(loc);
      rewriter.eraseOp(op);
      return mlir::success();
    }

    case VulkanKernelRecipe::SKIP_RMS_NORM: {
      if ((inputs.size() != 3 && inputs.size() != 4) ||
          (outputs.size() != 1 && outputs.size() != 2)) {
        return op.emitOpError("fused skip RMSNorm operand contract mismatch");
      }
      mlir::Value x = inputs[0];
      mlir::Value skip = inputs[1];
      mlir::Value gamma = inputs[2];
      const bool hasBias = inputs.size() == 4;
      mlir::Value bias = hasBias ? inputs[3] : mlir::Value{};
      mlir::Value output = outputs[0];
      const bool writeHidden = outputs.size() == 2;
      mlir::Value hiddenOutput =
          writeHidden ? outputs[1] : mlir::Value{};
      mlir::Value rows = rewriter.create<mlir::memref::DimOp>(loc, x, 0);
      mlir::Value hidden = rewriter.create<mlir::memref::DimOp>(loc, x, 1);

      auto launch = createGpuLaunch(rewriter, loc, rows, one, one);
      mlir::Value row = launch.getBlockIds().x;
      rewriter.setInsertionPointToEnd(&launch.getBody().front());

      mlir::Value initial = floatConst(rewriter, loc, accTy, 0.0);
      auto squares = emitReductionLoop(
          rewriter, loc, zero, hidden, one, initial,
          [&](mlir::OpBuilder& builder, mlir::Location nestedLoc,
              mlir::Value column,
              mlir::Value accumulator) -> mlir::Value {
            mlir::Value value = builder.create<mlir::arith::AddFOp>(
                nestedLoc,
                loadAsAccumulator(
                    builder, nestedLoc, x,
                    mlir::SmallVector<mlir::Value>{row, column}, accTy),
                loadAsAccumulator(
                    builder, nestedLoc, skip,
                    mlir::SmallVector<mlir::Value>{row, column}, accTy));
            if (hasBias) {
              value = builder.create<mlir::arith::AddFOp>(
                  nestedLoc, value,
                  loadAsAccumulator(
                      builder, nestedLoc, bias,
                      mlir::SmallVector<mlir::Value>{column}, accTy));
            }
            return builder.create<mlir::arith::AddFOp>(
                nestedLoc, accumulator,
                builder.create<mlir::arith::MulFOp>(nestedLoc, value, value));
          });
      mlir::Value hiddenValue =
          convertIndexToFloat(rewriter, loc, hidden, accTy);
      mlir::Value normFactor = emitRsqrt(
          rewriter, loc, accTy,
          rewriter.create<mlir::arith::AddFOp>(
              loc,
              rewriter.create<mlir::arith::DivFOp>(
                  loc, squares.getResult(0), hiddenValue),
              floatConst(rewriter, loc, accTy, epsilon)));

      rewriter.create<mlir::scf::ForOp>(
          loc, zero, hidden, one, mlir::ValueRange{},
          [&](mlir::OpBuilder& builder, mlir::Location nestedLoc,
              mlir::Value column, mlir::ValueRange) {
            mlir::Value value = builder.create<mlir::arith::AddFOp>(
                nestedLoc,
                loadAsAccumulator(
                    builder, nestedLoc, x,
                    mlir::SmallVector<mlir::Value>{row, column}, accTy),
                loadAsAccumulator(
                    builder, nestedLoc, skip,
                    mlir::SmallVector<mlir::Value>{row, column}, accTy));
            if (hasBias) {
              value = builder.create<mlir::arith::AddFOp>(
                  nestedLoc, value,
                  loadAsAccumulator(
                      builder, nestedLoc, bias,
                      mlir::SmallVector<mlir::Value>{column}, accTy));
            }
            if (writeHidden) {
              storeFromAccumulator(
                  builder, nestedLoc, value, hiddenOutput,
                  mlir::SmallVector<mlir::Value>{row, column});
            }
            mlir::Value scaled = builder.create<mlir::arith::MulFOp>(
                nestedLoc,
                builder.create<mlir::arith::MulFOp>(
                    nestedLoc, value, normFactor),
                loadAsAccumulator(
                    builder, nestedLoc, gamma,
                    mlir::SmallVector<mlir::Value>{column}, accTy));
            storeFromAccumulator(
                builder, nestedLoc, scaled, output,
                mlir::SmallVector<mlir::Value>{row, column});
            builder.create<mlir::scf::YieldOp>(nestedLoc);
          });
      rewriter.create<mlir::gpu::TerminatorOp>(loc);
      rewriter.eraseOp(op);
      return mlir::success();
    }

    case VulkanKernelRecipe::RMS_NORM_LINEAR: {
      if (inputs.size() != 3 || outputs.size() != 1) {
        return op.emitOpError("fused RMSNorm-linear operand contract mismatch");
      }
      mlir::Value x = inputs[0];
      mlir::Value gamma = inputs[1];
      mlir::Value weights = inputs[2];
      mlir::Value output = outputs[0];
      mlir::Value rows = rewriter.create<mlir::memref::DimOp>(loc, x, 0);
      mlir::Value hidden = rewriter.create<mlir::memref::DimOp>(loc, x, 1);
      mlir::Value projected =
          rewriter.create<mlir::memref::DimOp>(loc, weights, 1);
      auto launch = createGpuLaunch(rewriter, loc, rows, one, one);
      mlir::Value row = launch.getBlockIds().x;
      rewriter.setInsertionPointToEnd(&launch.getBody().front());
      mlir::Value normFactor = rowNormFactor(x, row, hidden);

      rewriter.create<mlir::scf::ForOp>(
          loc, zero, projected, one, mlir::ValueRange{},
          [&](mlir::OpBuilder& builder, mlir::Location nestedLoc,
              mlir::Value column, mlir::ValueRange) {
            mlir::Value initial = floatConst(builder, nestedLoc, accTy, 0.0);
            auto dot = emitReductionLoop(
                builder, nestedLoc, zero, hidden, one, initial,
                [&](mlir::OpBuilder& kb, mlir::Location kloc,
                    mlir::Value k,
                    mlir::Value accumulator) -> mlir::Value {
                  mlir::Value normalized = kb.create<mlir::arith::MulFOp>(
                      kloc,
                      kb.create<mlir::arith::MulFOp>(
                          kloc,
                          loadAsAccumulator(
                              kb, kloc, x,
                              mlir::SmallVector<mlir::Value>{row, k}, accTy),
                          normFactor),
                      loadAsAccumulator(
                          kb, kloc, gamma,
                          mlir::SmallVector<mlir::Value>{k}, accTy));
                  mlir::Value product = kb.create<mlir::arith::MulFOp>(
                      kloc, normalized,
                      loadAsAccumulator(
                          kb, kloc, weights,
                          mlir::SmallVector<mlir::Value>{k, column}, accTy));
                  return kb.create<mlir::arith::AddFOp>(
                      kloc, accumulator, product);
                });
            storeFromAccumulator(
                builder, nestedLoc, dot.getResult(0), output,
                mlir::SmallVector<mlir::Value>{row, column});
            builder.create<mlir::scf::YieldOp>(nestedLoc);
          });
      rewriter.create<mlir::gpu::TerminatorOp>(loc);
      rewriter.eraseOp(op);
      return mlir::success();
    }

    case VulkanKernelRecipe::FUSED_GEMM_SWIGLU: {
      if (inputs.size() != 3 || outputs.size() != 1) {
        return op.emitOpError("fused GEMM-SwiGLU operand contract mismatch");
      }
      mlir::Value x = inputs[0];
      mlir::Value gateWeights = inputs[1];
      mlir::Value upWeights = inputs[2];
      mlir::Value output = outputs[0];
      mlir::Value rows = rewriter.create<mlir::memref::DimOp>(loc, x, 0);
      mlir::Value hidden = rewriter.create<mlir::memref::DimOp>(loc, x, 1);
      mlir::Value projected =
          rewriter.create<mlir::memref::DimOp>(loc, gateWeights, 1);
      auto launch = createGpuLaunch(rewriter, loc, rows, one, one);
      mlir::Value row = launch.getBlockIds().x;
      rewriter.setInsertionPointToEnd(&launch.getBody().front());

      rewriter.create<mlir::scf::ForOp>(
          loc, zero, projected, one, mlir::ValueRange{},
          [&](mlir::OpBuilder& builder, mlir::Location nestedLoc,
              mlir::Value column, mlir::ValueRange) {
            mlir::Value initial = floatConst(builder, nestedLoc, accTy, 0.0);
            auto dots = builder.create<mlir::scf::ForOp>(
                nestedLoc, zero, hidden, one,
                mlir::ValueRange{initial, initial},
                [&](mlir::OpBuilder& kb, mlir::Location kloc,
                    mlir::Value k, mlir::ValueRange accumulators) {
                  mlir::Value value = loadAsAccumulator(
                      kb, kloc, x,
                      mlir::SmallVector<mlir::Value>{row, k}, accTy);
                  mlir::Value gate = kb.create<mlir::arith::AddFOp>(
                      kloc, accumulators[0],
                      kb.create<mlir::arith::MulFOp>(
                          kloc, value,
                          loadAsAccumulator(
                              kb, kloc, gateWeights,
                              mlir::SmallVector<mlir::Value>{k, column},
                              accTy)));
                  mlir::Value up = kb.create<mlir::arith::AddFOp>(
                      kloc, accumulators[1],
                      kb.create<mlir::arith::MulFOp>(
                          kloc, value,
                          loadAsAccumulator(
                              kb, kloc, upWeights,
                              mlir::SmallVector<mlir::Value>{k, column},
                              accTy)));
                  kb.create<mlir::scf::YieldOp>(
                      kloc, mlir::ValueRange{gate, up});
                });
            mlir::Value result = builder.create<mlir::arith::MulFOp>(
                nestedLoc,
                emitSilu(builder, nestedLoc, accTy, dots.getResult(0)),
                dots.getResult(1));
            storeFromAccumulator(
                builder, nestedLoc, result, output,
                mlir::SmallVector<mlir::Value>{row, column});
            builder.create<mlir::scf::YieldOp>(nestedLoc);
          });
      rewriter.create<mlir::gpu::TerminatorOp>(loc);
      rewriter.eraseOp(op);
      return mlir::success();
    }

    case VulkanKernelRecipe::FUSED_RMS_NORM_SWIGLU: {
      if (inputs.size() != 4 || outputs.size() != 1) {
        return op.emitOpError(
            "fused RMSNorm-GEMM-SwiGLU operand contract mismatch");
      }
      mlir::Value x = inputs[0];
      mlir::Value gamma = inputs[1];
      mlir::Value gateWeights = inputs[2];
      mlir::Value upWeights = inputs[3];
      mlir::Value output = outputs[0];
      mlir::Value batch = rewriter.create<mlir::memref::DimOp>(loc, x, 0);
      mlir::Value sequence =
          rewriter.create<mlir::memref::DimOp>(loc, x, 1);
      mlir::Value hidden = rewriter.create<mlir::memref::DimOp>(loc, x, 2);
      mlir::Value projected =
          rewriter.create<mlir::memref::DimOp>(loc, gateWeights, 1);
      mlir::Value rows =
          rewriter.create<mlir::arith::MulIOp>(loc, batch, sequence);
      auto launch = createGpuLaunch(rewriter, loc, rows, one, one);
      mlir::Value flatRow = launch.getBlockIds().x;
      rewriter.setInsertionPointToEnd(&launch.getBody().front());
      mlir::Value batchIndex =
          rewriter.create<mlir::arith::DivUIOp>(loc, flatRow, sequence);
      mlir::Value sequenceIndex =
          rewriter.create<mlir::arith::RemUIOp>(loc, flatRow, sequence);

      mlir::Value initial = floatConst(rewriter, loc, accTy, 0.0);
      auto squares = emitReductionLoop(
          rewriter, loc, zero, hidden, one, initial,
          [&](mlir::OpBuilder& builder, mlir::Location nestedLoc,
              mlir::Value k, mlir::Value accumulator) -> mlir::Value {
            mlir::Value value = loadAsAccumulator(
                builder, nestedLoc, x,
                mlir::SmallVector<mlir::Value>{
                    batchIndex, sequenceIndex, k},
                accTy);
            return builder.create<mlir::arith::AddFOp>(
                nestedLoc, accumulator,
                builder.create<mlir::arith::MulFOp>(
                    nestedLoc, value, value));
          });
      mlir::Value hiddenValue =
          convertIndexToFloat(rewriter, loc, hidden, accTy);
      mlir::Value normFactor = emitRsqrt(
          rewriter, loc, accTy,
          rewriter.create<mlir::arith::AddFOp>(
              loc,
              rewriter.create<mlir::arith::DivFOp>(
                  loc, squares.getResult(0), hiddenValue),
              floatConst(rewriter, loc, accTy, epsilon)));

      rewriter.create<mlir::scf::ForOp>(
          loc, zero, projected, one, mlir::ValueRange{},
          [&](mlir::OpBuilder& builder, mlir::Location nestedLoc,
              mlir::Value column, mlir::ValueRange) {
            mlir::Value dotInitial =
                floatConst(builder, nestedLoc, accTy, 0.0);
            auto dots = builder.create<mlir::scf::ForOp>(
                nestedLoc, zero, hidden, one,
                mlir::ValueRange{dotInitial, dotInitial},
                [&](mlir::OpBuilder& kb, mlir::Location kloc,
                    mlir::Value k, mlir::ValueRange accumulators) {
                  mlir::Value normalized = kb.create<mlir::arith::MulFOp>(
                      kloc,
                      kb.create<mlir::arith::MulFOp>(
                          kloc,
                          loadAsAccumulator(
                              kb, kloc, x,
                              mlir::SmallVector<mlir::Value>{
                                  batchIndex, sequenceIndex, k},
                              accTy),
                          normFactor),
                      loadAsAccumulator(
                          kb, kloc, gamma,
                          mlir::SmallVector<mlir::Value>{k}, accTy));
                  mlir::Value gate = kb.create<mlir::arith::AddFOp>(
                      kloc, accumulators[0],
                      kb.create<mlir::arith::MulFOp>(
                          kloc, normalized,
                          loadAsAccumulator(
                              kb, kloc, gateWeights,
                              mlir::SmallVector<mlir::Value>{k, column},
                              accTy)));
                  mlir::Value up = kb.create<mlir::arith::AddFOp>(
                      kloc, accumulators[1],
                      kb.create<mlir::arith::MulFOp>(
                          kloc, normalized,
                          loadAsAccumulator(
                              kb, kloc, upWeights,
                              mlir::SmallVector<mlir::Value>{k, column},
                              accTy)));
                  kb.create<mlir::scf::YieldOp>(
                      kloc, mlir::ValueRange{gate, up});
                });
            mlir::Value result = builder.create<mlir::arith::MulFOp>(
                nestedLoc,
                emitSilu(builder, nestedLoc, accTy, dots.getResult(0)),
                dots.getResult(1));
            storeFromAccumulator(
                builder, nestedLoc, result, output,
                mlir::SmallVector<mlir::Value>{
                    batchIndex, sequenceIndex, column});
            builder.create<mlir::scf::YieldOp>(nestedLoc);
          });
      rewriter.create<mlir::gpu::TerminatorOp>(loc);
      rewriter.eraseOp(op);
      return mlir::success();
    }

    case VulkanKernelRecipe::FLASH_ATTENTION:
    case VulkanKernelRecipe::GROUPED_QUERY_ATTENTION: {
      if (inputs.size() != 3 || outputs.size() != 1) {
        return op.emitOpError(
            "flash/GQA attention operand contract mismatch");
      }
      mlir::Value queries = inputs[0];
      mlir::Value keys = inputs[1];
      mlir::Value values = inputs[2];
      mlir::Value output = outputs[0];
      auto queryType = llvm::dyn_cast<mlir::MemRefType>(queries.getType());
      if (!queryType || (queryType.getRank() != 3 &&
                         queryType.getRank() != 4)) {
        return op.emitOpError(
            "flash/GQA attention requires rank-3 or rank-4 BSHD input");
      }
      const int64_t rank = queryType.getRank();
      mlir::Value batch =
          rewriter.create<mlir::memref::DimOp>(loc, queries, 0);
      mlir::Value querySteps =
          rewriter.create<mlir::memref::DimOp>(loc, queries, 1);
      mlir::Value keySteps =
          rewriter.create<mlir::memref::DimOp>(loc, keys, 1);
      mlir::Value queryHeads =
          rank == 4
              ? mlir::Value(rewriter.create<mlir::memref::DimOp>(
                    loc, queries, 2))
              : one;
      mlir::Value keyValueHeads =
          rank == 4
              ? mlir::Value(rewriter.create<mlir::memref::DimOp>(
                    loc, keys, 2))
              : one;
      mlir::Value headDimension =
          rewriter.create<mlir::memref::DimOp>(loc, queries, rank - 1);
      mlir::Value headsPerGroup =
          rewriter.create<mlir::arith::DivUIOp>(
              loc, queryHeads, keyValueHeads);

      mlir::Value batchHeads = rewriter.create<mlir::arith::MulIOp>(
          loc, batch, queryHeads);
      mlir::Value invocations = rewriter.create<mlir::arith::MulIOp>(
          loc, batchHeads, querySteps);
      auto launch = createGpuLaunch(rewriter, loc, invocations, one, one);
      mlir::Value flatInvocation = launch.getBlockIds().x;
      rewriter.setInsertionPointToEnd(&launch.getBody().front());

      mlir::Value queryStep = rewriter.create<mlir::arith::RemUIOp>(
          loc, flatInvocation, querySteps);
      mlir::Value flatBatchHead = rewriter.create<mlir::arith::DivUIOp>(
          loc, flatInvocation, querySteps);
      mlir::Value queryHead = rewriter.create<mlir::arith::RemUIOp>(
          loc, flatBatchHead, queryHeads);
      mlir::Value batchIndex = rewriter.create<mlir::arith::DivUIOp>(
          loc, flatBatchHead, queryHeads);
      mlir::Value keyValueHead = rewriter.create<mlir::arith::DivUIOp>(
          loc, queryHead, headsPerGroup);

      auto queryIndices = [&](mlir::Value feature) {
        if (rank == 3) {
          return mlir::SmallVector<mlir::Value>{
              batchIndex, queryStep, feature};
        }
        return mlir::SmallVector<mlir::Value>{
            batchIndex, queryStep, queryHead, feature};
      };
      auto keyValueIndices = [&](mlir::Value keyStep,
                                 mlir::Value feature) {
        if (rank == 3) {
          return mlir::SmallVector<mlir::Value>{
              batchIndex, keyStep, feature};
        }
        return mlir::SmallVector<mlir::Value>{
            batchIndex, keyStep, keyValueHead, feature};
      };
      auto outputIndices = [&](mlir::Value feature) {
        if (rank == 3) {
          return mlir::SmallVector<mlir::Value>{
              batchIndex, queryStep, feature};
        }
        return mlir::SmallVector<mlir::Value>{
            batchIndex, queryStep, queryHead, feature};
      };

      auto scaleAttr =
          op->getAttrOfType<mlir::FloatAttr>("nd4j.attention_scale");
      const double requestedScale =
          scaleAttr ? scaleAttr.getValueAsDouble() : 0.0;
      mlir::Value scale;
      if (requestedScale > 0.0) {
        scale = floatConst(rewriter, loc, accTy, requestedScale);
      } else {
        mlir::Value headDimensionValue =
            convertIndexToFloat(rewriter, loc, headDimension, accTy);
        scale = rewriter.create<mlir::arith::DivFOp>(
            loc, floatConst(rewriter, loc, accTy, 1.0),
            rewriter.create<mlir::math::SqrtOp>(
                loc, headDimensionValue));
      }
      const bool causal =
          op->getAttrOfType<mlir::BoolAttr>("nd4j.attention_causal") &&
          op->getAttrOfType<mlir::BoolAttr>("nd4j.attention_causal")
              .getValue();
      mlir::Value hasCausalOffset = rewriter.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::ugt, keySteps, querySteps);
      mlir::Value rawCausalOffset = rewriter.create<mlir::arith::SubIOp>(
          loc, keySteps, querySteps);
      mlir::Value causalOffset = rewriter.create<mlir::arith::SelectOp>(
          loc, hasCausalOffset, rawCausalOffset, zero);
      mlir::Value lastVisibleKey = rewriter.create<mlir::arith::AddIOp>(
          loc, queryStep, causalOffset);
      mlir::Value negativeInfinity = floatConst(
          rewriter, loc, accTy,
          -std::numeric_limits<double>::infinity());

      auto emitScore = [&](mlir::OpBuilder& builder,
                           mlir::Location nestedLoc,
                           mlir::Value keyStep) -> mlir::Value {
        mlir::Value initial =
            floatConst(builder, nestedLoc, accTy, 0.0);
        auto dot = emitReductionLoop(
            builder, nestedLoc, zero, headDimension, one, initial,
            [&](mlir::OpBuilder& db, mlir::Location dloc,
                mlir::Value feature,
                mlir::Value accumulator) -> mlir::Value {
              mlir::Value product = db.create<mlir::arith::MulFOp>(
                  dloc,
                  loadAsAccumulator(
                      db, dloc, queries, queryIndices(feature), accTy),
                  loadAsAccumulator(
                      db, dloc, keys,
                      keyValueIndices(keyStep, feature), accTy));
              return db.create<mlir::arith::AddFOp>(
                  dloc, accumulator, product);
            });
        mlir::Value score = builder.create<mlir::arith::MulFOp>(
            nestedLoc, dot.getResult(0), scale);
        if (causal) {
          mlir::Value hidden = builder.create<mlir::arith::CmpIOp>(
              nestedLoc, mlir::arith::CmpIPredicate::ugt,
              keyStep, lastVisibleKey);
          score = builder.create<mlir::arith::SelectOp>(
              nestedLoc, hidden, negativeInfinity, score);
        }
        return score;
      };

      auto maximum = emitReductionLoop(
          rewriter, loc, zero, keySteps, one, negativeInfinity,
          [&](mlir::OpBuilder& builder, mlir::Location nestedLoc,
              mlir::Value keyStep,
              mlir::Value accumulator) -> mlir::Value {
            return builder.create<mlir::arith::MaximumFOp>(
                nestedLoc, accumulator,
                emitScore(builder, nestedLoc, keyStep));
          });
      mlir::Value maxScore = maximum.getResult(0);
      auto exponentialSum = emitReductionLoop(
          rewriter, loc, zero, keySteps, one,
          floatConst(rewriter, loc, accTy, 0.0),
          [&](mlir::OpBuilder& builder, mlir::Location nestedLoc,
              mlir::Value keyStep,
              mlir::Value accumulator) -> mlir::Value {
            mlir::Value shifted = builder.create<mlir::arith::SubFOp>(
                nestedLoc, emitScore(builder, nestedLoc, keyStep), maxScore);
            return builder.create<mlir::arith::AddFOp>(
                nestedLoc, accumulator,
                emitExp(builder, nestedLoc, accTy, shifted));
          });
      mlir::Value inverseSum = rewriter.create<mlir::arith::DivFOp>(
          loc, floatConst(rewriter, loc, accTy, 1.0),
          exponentialSum.getResult(0));

      auto probability = [&](mlir::OpBuilder& builder,
                             mlir::Location nestedLoc,
                             mlir::Value keyStep) {
        mlir::Value shifted = builder.create<mlir::arith::SubFOp>(
            nestedLoc, emitScore(builder, nestedLoc, keyStep), maxScore);
        return mlir::Value(builder.create<mlir::arith::MulFOp>(
            nestedLoc, emitExp(builder, nestedLoc, accTy, shifted),
            inverseSum));
      };

      rewriter.create<mlir::scf::ForOp>(
          loc, zero, headDimension, one, mlir::ValueRange{},
          [&](mlir::OpBuilder& builder, mlir::Location nestedLoc,
              mlir::Value feature, mlir::ValueRange) {
            auto weightedSum = emitReductionLoop(
                builder, nestedLoc, zero, keySteps, one,
                floatConst(builder, nestedLoc, accTy, 0.0),
                [&](mlir::OpBuilder& kb, mlir::Location kloc,
                    mlir::Value keyStep,
                    mlir::Value accumulator) -> mlir::Value {
                  mlir::Value product = kb.create<mlir::arith::MulFOp>(
                      kloc, probability(kb, kloc, keyStep),
                      loadAsAccumulator(
                          kb, kloc, values,
                          keyValueIndices(keyStep, feature), accTy));
                  return kb.create<mlir::arith::AddFOp>(
                      kloc, accumulator, product);
                });
            storeFromAccumulator(
                builder, nestedLoc, weightedSum.getResult(0), output,
                outputIndices(feature));
            builder.create<mlir::scf::YieldOp>(nestedLoc);
          });
      rewriter.create<mlir::gpu::TerminatorOp>(loc);
      rewriter.eraseOp(op);
      return mlir::success();
    }

    case VulkanKernelRecipe::DOT_PRODUCT_ATTENTION: {
      if ((inputs.size() != 3 && inputs.size() != 4) ||
          (outputs.size() != 1 && outputs.size() != 2)) {
        return op.emitOpError(
            "dot-product attention operand contract mismatch");
      }
      mlir::Value queries = inputs[0];
      mlir::Value keys = inputs[1];
      mlir::Value values = inputs[2];
      const bool hasMask = inputs.size() == 4;
      mlir::Value mask = hasMask ? inputs[3] : mlir::Value{};
      mlir::Value output = outputs[0];
      const bool writeWeights = outputs.size() == 2;
      mlir::Value weightsOutput =
          writeWeights ? outputs[1] : mlir::Value{};
      auto queryType = llvm::cast<mlir::MemRefType>(queries.getType());
      const int64_t rank = queryType.getRank();
      if (rank != 3 && rank != 4) {
        return op.emitOpError("dot-product attention requires rank 3 or 4");
      }
      const bool normalize =
          op->getAttrOfType<mlir::BoolAttr>("nd4j.attention_normalize") &&
          op->getAttrOfType<mlir::BoolAttr>("nd4j.attention_normalize")
              .getValue();

      mlir::Value batch =
          rewriter.create<mlir::memref::DimOp>(loc, queries, 0);
      mlir::Value heads =
          rank == 4
              ? mlir::Value(
                    rewriter.create<mlir::memref::DimOp>(loc, queries, 1))
              : one;
      mlir::Value queryFeatures = rewriter.create<mlir::memref::DimOp>(
          loc, queries, rank - 2);
      mlir::Value querySteps = rewriter.create<mlir::memref::DimOp>(
          loc, queries, rank - 1);
      mlir::Value keySteps = rewriter.create<mlir::memref::DimOp>(
          loc, keys, rank - 1);
      mlir::Value valueFeatures = rewriter.create<mlir::memref::DimOp>(
          loc, values, rank - 2);
      mlir::Value leading =
          rewriter.create<mlir::arith::MulIOp>(loc, batch, heads);
      mlir::Value invocations =
          rewriter.create<mlir::arith::MulIOp>(loc, leading, querySteps);
      auto launch =
          createGpuLaunch(rewriter, loc, invocations, one, one);
      mlir::Value flatInvocation = launch.getBlockIds().x;
      rewriter.setInsertionPointToEnd(&launch.getBody().front());

      mlir::Value queryStep = rewriter.create<mlir::arith::RemUIOp>(
          loc, flatInvocation, querySteps);
      mlir::Value leadingIndex = rewriter.create<mlir::arith::DivUIOp>(
          loc, flatInvocation, querySteps);
      mlir::Value batchIndex =
          rank == 4
              ? mlir::Value(rewriter.create<mlir::arith::DivUIOp>(
                    loc, leadingIndex, heads))
              : leadingIndex;
      mlir::Value headIndex =
          rank == 4
              ? mlir::Value(rewriter.create<mlir::arith::RemUIOp>(
                    loc, leadingIndex, heads))
              : zero;

      auto queryIndices = [&](mlir::Value feature) {
        if (rank == 3) {
          return mlir::SmallVector<mlir::Value>{
              batchIndex, feature, queryStep};
        }
        return mlir::SmallVector<mlir::Value>{
            batchIndex, headIndex, feature, queryStep};
      };
      auto keyIndices = [&](mlir::Value feature, mlir::Value keyStep) {
        if (rank == 3) {
          return mlir::SmallVector<mlir::Value>{
              batchIndex, feature, keyStep};
        }
        return mlir::SmallVector<mlir::Value>{
            batchIndex, headIndex, feature, keyStep};
      };
      auto valueIndices = [&](mlir::Value feature, mlir::Value keyStep) {
        if (rank == 3) {
          return mlir::SmallVector<mlir::Value>{
              batchIndex, feature, keyStep};
        }
        return mlir::SmallVector<mlir::Value>{
            batchIndex, headIndex, feature, keyStep};
      };
      auto outputIndices = [&](mlir::Value feature) {
        if (rank == 3) {
          return mlir::SmallVector<mlir::Value>{
              batchIndex, feature, queryStep};
        }
        return mlir::SmallVector<mlir::Value>{
            batchIndex, headIndex, feature, queryStep};
      };
      auto weightIndices = [&](mlir::Value keyStep) {
        if (rank == 3) {
          return mlir::SmallVector<mlir::Value>{
              batchIndex, keyStep, queryStep};
        }
        return mlir::SmallVector<mlir::Value>{
            batchIndex, headIndex, keyStep, queryStep};
      };

      mlir::Value featureScale;
      if (normalize) {
        mlir::Value featureCount =
            convertIndexToFloat(rewriter, loc, queryFeatures, accTy);
        featureScale =
            rewriter.create<mlir::math::SqrtOp>(loc, featureCount);
      }
      auto emitScore = [&](mlir::OpBuilder& builder,
                           mlir::Location nestedLoc,
                           mlir::Value keyStep) -> mlir::Value {
        mlir::Value initial = floatConst(builder, nestedLoc, accTy, 0.0);
        auto dot = emitReductionLoop(
            builder, nestedLoc, zero, queryFeatures, one, initial,
            [&](mlir::OpBuilder& kb, mlir::Location kloc,
                mlir::Value feature,
                mlir::Value accumulator) -> mlir::Value {
              mlir::Value product = kb.create<mlir::arith::MulFOp>(
                  kloc,
                  loadAsAccumulator(
                      kb, kloc, keys, keyIndices(feature, keyStep), accTy),
                  loadAsAccumulator(
                      kb, kloc, queries, queryIndices(feature), accTy));
              return kb.create<mlir::arith::AddFOp>(
                  kloc, accumulator, product);
            });
        mlir::Value score = dot.getResult(0);
        if (normalize) {
          score = builder.create<mlir::arith::DivFOp>(
              nestedLoc, score, featureScale);
        }
        if (hasMask) {
          mlir::Value maskValue = loadAsAccumulator(
              builder, nestedLoc, mask,
              mlir::SmallVector<mlir::Value>{batchIndex, keyStep}, accTy);
          mlir::Value complement = builder.create<mlir::arith::SubFOp>(
              nestedLoc, floatConst(builder, nestedLoc, accTy, 1.0),
              maskValue);
          // The CPU reference subtracts (1-mask)*3.4028235e38 in f32; that
          // constant overflows an f16 accumulator to inf and NaNs the
          // softmax, so scale the penalty to the accumulator width. Kept
          // finite (not -inf) so fully-masked rows stay uniform like the
          // reference instead of producing NaN from (-inf) - (-inf).
          const double maskPenalty =
              accTy.getWidth() <= 16 ? 3.0e4 : 3.4028235e38;
          score = builder.create<mlir::arith::SubFOp>(
              nestedLoc, score,
              builder.create<mlir::arith::MulFOp>(
                  nestedLoc, complement,
                  floatConst(builder, nestedLoc, accTy, maskPenalty)));
        }
        return score;
      };

      mlir::Value negativeInfinity = floatConst(
          rewriter, loc, accTy,
          -std::numeric_limits<double>::infinity());
      auto maximum = emitReductionLoop(
          rewriter, loc, zero, keySteps, one, negativeInfinity,
          [&](mlir::OpBuilder& builder, mlir::Location nestedLoc,
              mlir::Value keyStep,
              mlir::Value accumulator) -> mlir::Value {
            return builder.create<mlir::arith::MaximumFOp>(
                nestedLoc, accumulator,
                emitScore(builder, nestedLoc, keyStep));
          });
      mlir::Value maxScore = maximum.getResult(0);
      mlir::Value sumInitial = floatConst(rewriter, loc, accTy, 0.0);
      auto exponentialSum = emitReductionLoop(
          rewriter, loc, zero, keySteps, one, sumInitial,
          [&](mlir::OpBuilder& builder, mlir::Location nestedLoc,
              mlir::Value keyStep,
              mlir::Value accumulator) -> mlir::Value {
            mlir::Value shifted = builder.create<mlir::arith::SubFOp>(
                nestedLoc, emitScore(builder, nestedLoc, keyStep), maxScore);
            return builder.create<mlir::arith::AddFOp>(
                nestedLoc, accumulator,
                emitExp(builder, nestedLoc, accTy, shifted));
          });
      mlir::Value inverseSum = rewriter.create<mlir::arith::DivFOp>(
          loc, floatConst(rewriter, loc, accTy, 1.0),
          exponentialSum.getResult(0));

      auto emitProbability = [&](mlir::OpBuilder& builder,
                                 mlir::Location nestedLoc,
                                 mlir::Value keyStep) {
        mlir::Value shifted = builder.create<mlir::arith::SubFOp>(
            nestedLoc, emitScore(builder, nestedLoc, keyStep), maxScore);
        return mlir::Value(builder.create<mlir::arith::MulFOp>(
            nestedLoc, emitExp(builder, nestedLoc, accTy, shifted),
            inverseSum));
      };

      if (writeWeights) {
        rewriter.create<mlir::scf::ForOp>(
            loc, zero, keySteps, one, mlir::ValueRange{},
            [&](mlir::OpBuilder& builder, mlir::Location nestedLoc,
                mlir::Value keyStep, mlir::ValueRange) {
              storeFromAccumulator(
                  builder, nestedLoc,
                  emitProbability(builder, nestedLoc, keyStep),
                  weightsOutput, weightIndices(keyStep));
              builder.create<mlir::scf::YieldOp>(nestedLoc);
            });
      }

      rewriter.create<mlir::scf::ForOp>(
          loc, zero, valueFeatures, one, mlir::ValueRange{},
          [&](mlir::OpBuilder& builder, mlir::Location nestedLoc,
              mlir::Value feature, mlir::ValueRange) {
            mlir::Value initial = floatConst(builder, nestedLoc, accTy, 0.0);
            auto weightedSum = emitReductionLoop(
                builder, nestedLoc, zero, keySteps, one, initial,
                [&](mlir::OpBuilder& kb, mlir::Location kloc,
                    mlir::Value keyStep,
                    mlir::Value accumulator) -> mlir::Value {
                  mlir::Value product = kb.create<mlir::arith::MulFOp>(
                      kloc, emitProbability(kb, kloc, keyStep),
                      loadAsAccumulator(
                          kb, kloc, values,
                          valueIndices(feature, keyStep), accTy));
                  return kb.create<mlir::arith::AddFOp>(
                      kloc, accumulator, product);
                });
            storeFromAccumulator(
                builder, nestedLoc, weightedSum.getResult(0), output,
                outputIndices(feature));
            builder.create<mlir::scf::YieldOp>(nestedLoc);
          });
      rewriter.create<mlir::gpu::TerminatorOp>(loc);
      rewriter.eraseOp(op);
      return mlir::success();
    }

    default:
      return mlir::failure();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Wave 1: ElementwiseBinaryToSpirv
// ─────────────────────────────────────────────────────────────────────────────
//
// Lowers same-shape elementwise binary ops (add, subtract, multiply, divide,
// residual_add) tagged with nd4j.binary = true in a linalg.generic.
//
// One Vulkan invocation per logical output element.  Logical multi-indices
// preserve arbitrary MemRef strides, offsets, and broadcast views.
//

mlir::LogicalResult ElementwiseTernaryToSpirv::matchAndRewrite(
    mlir::linalg::GenericOp op,
    mlir::PatternRewriter& rewriter) const {
  mlir::Location loc = op.getLoc();
  auto ternaryAttr = op->getAttrOfType<mlir::BoolAttr>(kTernaryAttr);
  if (!ternaryAttr || !ternaryAttr.getValue()) return mlir::failure();

  const auto* emitter = emitterForOperation(op);
  if (emitter == nullptr || emitter->family != VulkanKernelFamily::TERNARY) {
    return mlir::failure();
  }
  mlir::ValueRange inputs = op.getInputs();
  mlir::ValueRange outputs = op.getOutputs();
  if (inputs.size() != 3 || outputs.size() != 1) {
    return op.emitOpError(
        "ElementwiseTernaryToSpirv: expected three inputs and one output");
  }

  mlir::Value condition = inputs[0];
  mlir::Value trueValue = inputs[1];
  mlir::Value falseValue = inputs[2];
  mlir::Value output = outputs[0];
  auto conditionType = llvm::dyn_cast<mlir::MemRefType>(condition.getType());
  auto trueType = llvm::dyn_cast<mlir::MemRefType>(trueValue.getType());
  auto falseType = llvm::dyn_cast<mlir::MemRefType>(falseValue.getType());
  auto outputType = llvm::dyn_cast<mlir::MemRefType>(output.getType());
  if (!conditionType || !trueType || !falseType || !outputType ||
      conditionType.getRank() > outputType.getRank() ||
      trueType.getRank() > outputType.getRank() ||
      falseType.getRank() > outputType.getRank()) {
    return op.emitOpError(
        "ElementwiseTernaryToSpirv: invalid broadcast MemRef ranks");
  }

  auto computeAttr = op->getAttrOfType<mlir::TypeAttr>(kAccumulatorTypeAttr);
  mlir::Type computeType = computeAttr ? computeAttr.getValue() : mlir::Type{};
  if (!llvm::isa<mlir::FloatType>(computeType) &&
      !llvm::isa<mlir::IntegerType>(computeType)) {
    return op.emitOpError(
        "ElementwiseTernaryToSpirv: requires numeric branch computation");
  }

  auto readBoolAttr = [&](llvm::StringRef name) {
    auto attr = op->getAttrOfType<mlir::BoolAttr>(name);
    return attr && attr.getValue();
  };
  const bool conditionUnsigned = readBoolAttr("nd4j.input0_unsigned");
  const bool trueUnsigned = readBoolAttr("nd4j.input1_unsigned");
  const bool falseUnsigned = readBoolAttr("nd4j.input2_unsigned");
  const bool outputUnsigned = readBoolAttr("nd4j.output_unsigned");

  mlir::Value oneIdx = idxConst(rewriter, loc, 1);
  mlir::Value totalN = idxConst(rewriter, loc, 1);
  for (int64_t d = 0; d < outputType.getRank(); ++d) {
    totalN = rewriter.create<mlir::arith::MulIOp>(
        loc, totalN, rewriter.create<mlir::memref::DimOp>(loc, output, d));
  }
  auto launch = createGpuLaunch(rewriter, loc, totalN, oneIdx, oneIdx);
  mlir::Value linearIndex = launch.getBlockIds().x;
  rewriter.setInsertionPointToEnd(&launch.getBody().front());

  auto outputIndices = logicalIndices(rewriter, loc, linearIndex, output);
  auto conditionIndices = broadcastIndices(
      rewriter, loc, outputIndices, condition);
  auto trueIndices = broadcastIndices(rewriter, loc, outputIndices, trueValue);
  auto falseIndices = broadcastIndices(
      rewriter, loc, outputIndices, falseValue);
  mlir::Value conditionValue = loadAsScalar(
      rewriter, loc, condition, conditionIndices, rewriter.getI32Type(),
      conditionUnsigned, true);
  mlir::Value conditionIsTrue = rewriter.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::ne, conditionValue,
      rewriter.create<mlir::arith::ConstantIntOp>(
          loc, 0, rewriter.getI32Type().getWidth()));
  mlir::Value trueScalar = loadAsScalar(
      rewriter, loc, trueValue, trueIndices, computeType,
      trueUnsigned, readBoolAttr("nd4j.input1_unsigned"));
  mlir::Value falseScalar = loadAsScalar(
      rewriter, loc, falseValue, falseIndices, computeType,
      falseUnsigned, readBoolAttr("nd4j.input2_unsigned"));
  mlir::Value result = rewriter.create<mlir::arith::SelectOp>(
      loc, conditionIsTrue, trueScalar, falseScalar);
  if (!storeScalar(rewriter, loc, result, output, outputIndices,
                   trueUnsigned, outputUnsigned)) {
    return op.emitOpError(
        "ElementwiseTernaryToSpirv: result storage conversion failed");
  }
  rewriter.create<mlir::gpu::TerminatorOp>(loc);
  rewriter.eraseOp(op);
  return mlir::success();
}

mlir::LogicalResult ElementwiseBinaryToSpirv::matchAndRewrite(
    mlir::linalg::GenericOp op,
    mlir::PatternRewriter& rewriter) const {

  mlir::Location loc = op.getLoc();

  // ── 1. Guard: only match generics with nd4j.binary = true ────────────────
  auto binaryAttr = op->getAttrOfType<mlir::BoolAttr>(kBinaryAttr);
  if (!binaryAttr || !binaryAttr.getValue()) {
    return mlir::failure();
  }

  const auto* emitter = emitterForOperation(op);
  if (emitter == nullptr ||
      (emitter->family != VulkanKernelFamily::ELEMENTWISE_BINARY &&
       emitter->family != VulkanKernelFamily::COMPARISON &&
       emitter->family != VulkanKernelFamily::LOGICAL)) {
    return mlir::failure();
  }
  const VulkanKernelRecipe semantic =
      legacySemanticFor(op, emitter->recipe);

  mlir::ValueRange inputs = op.getInputs();
  mlir::ValueRange outputs = op.getOutputs();
  const auto scalarPresentAttr =
      op->getAttrOfType<mlir::BoolAttr>("nd4j.scalar_present");
  const bool scalarPresent = scalarPresentAttr && scalarPresentAttr.getValue();
  const bool unaryAssign =
      hasVulkanOpTrait(*emitter, sd::ops::OP_TRAIT_IDENTITY) &&
      inputs.size() == 1 && !scalarPresent;
  if ((!unaryAssign && !scalarPresent && inputs.size() != 2) ||
      (scalarPresent && inputs.size() != 1) || outputs.size() != 1) {
    return op.emitOpError(
        "ElementwiseBinaryToSpirv: expected assign(1), scalar(1), or binary(2) inputs");
  }
  mlir::Value A = inputs[0];
  mlir::Value B = (unaryAssign || scalarPresent) ? inputs[0] : inputs[1];
  mlir::Value C = outputs[0];
  auto aType = llvm::dyn_cast<mlir::MemRefType>(A.getType());
  auto bType = llvm::dyn_cast<mlir::MemRefType>(B.getType());
  auto cType = llvm::dyn_cast<mlir::MemRefType>(C.getType());
  if (!aType || !bType || !cType || aType.getRank() > cType.getRank() ||
      bType.getRank() > cType.getRank()) {
    return op.emitOpError(
        "ElementwiseBinaryToSpirv: invalid broadcast MemRef ranks");
  }

  auto computeAttr = op->getAttrOfType<mlir::TypeAttr>(kAccumulatorTypeAttr);
  mlir::Type computeType = computeAttr ? computeAttr.getValue() : mlir::Type{};
  auto computeFloat = llvm::dyn_cast<mlir::FloatType>(computeType);
  auto computeInteger = llvm::dyn_cast<mlir::IntegerType>(computeType);
  // Integer widths have already passed the framework BUILD_SINGLE_SELECTOR and
  // VulkanDeviceCaps checks in the recorder. Preserve that selected AccT here
  // instead of imposing a second, hard-coded width policy in the lowering.
  if (!computeFloat && !computeInteger) {
    return op.emitOpError(
        "ElementwiseBinaryToSpirv: requires floating-point or integer computation");
  }
  auto readBoolAttr = [&](llvm::StringRef name) {
    auto attr = op->getAttrOfType<mlir::BoolAttr>(name);
    return attr && attr.getValue();
  };
  const bool aUnsigned = readBoolAttr("nd4j.input0_unsigned");
  const bool bUnsigned = scalarPresent
                             ? aUnsigned
                             : readBoolAttr("nd4j.input1_unsigned");
  const bool cUnsigned = readBoolAttr("nd4j.output_unsigned");
  auto scalar0Attr = op->getAttrOfType<mlir::FloatAttr>("nd4j.scalar0");
  auto scalar1Attr = op->getAttrOfType<mlir::FloatAttr>("nd4j.scalar1");
  const double scalar0 = scalar0Attr ? scalar0Attr.getValueAsDouble() : 0.0;
  const double scalar1 = scalar1Attr ? scalar1Attr.getValueAsDouble() : 0.0;
  const bool activationBackward =
      (emitter->traits &
       (sd::ops::OP_TRAIT_ACTIVATION | sd::ops::OP_TRAIT_BACKWARD)) ==
      (sd::ops::OP_TRAIT_ACTIVATION | sd::ops::OP_TRAIT_BACKWARD);
  BinaryCallback callback = binaryCallbackFor(semantic);
  const bool parameterizedBinary =
      computeFloat &&
      (semantic == VulkanKernelRecipe::AXPY ||
       semantic == VulkanKernelRecipe::BINARY_RELATIVE_ERROR ||
       semantic == VulkanKernelRecipe::BINARY_MINIMUM_ABSOLUTE_RELATIVE_ERROR) &&
      hasVulkanScalarArgumentSchema(*emitter);
  const bool parameterizedComparison =
      computeFloat &&
      semantic == VulkanKernelRecipe::EPSILON_COMPARE &&
      scalar0Attr != nullptr && !scalarPresent;
  if (computeFloat && !activationBackward && !callback &&
      !parameterizedBinary && !parameterizedComparison) {
    return mlir::failure();
  }

  mlir::Value oneIdx = idxConst(rewriter, loc, 1);
  mlir::Value totalN = idxConst(rewriter, loc, 1);
  for (int64_t d = 0; d < cType.getRank(); ++d) {
    mlir::Value dim = rewriter.create<mlir::memref::DimOp>(loc, C, d);
    totalN = rewriter.create<mlir::arith::MulIOp>(loc, totalN, dim);
  }

  auto launch = createGpuLaunch(rewriter, loc, totalN, oneIdx, oneIdx);
  mlir::Value linearIndex = launch.getBlockIds().x;
  rewriter.setInsertionPointToEnd(&launch.getBody().front());

  auto outputIndices = logicalIndices(rewriter, loc, linearIndex, C);
  auto aIndices = broadcastIndices(rewriter, loc, outputIndices, A);
  auto bIndices = broadcastIndices(rewriter, loc, outputIndices, B);
  mlir::Value aVal = loadAsScalar(
      rewriter, loc, A, aIndices, computeType, aUnsigned, cUnsigned);
  mlir::Value bVal = unaryAssign
                         ? aVal
                         : scalarPresent
                               ? loadScalarAttribute(rewriter, loc, op,
                                                     computeType)
                               : loadAsScalar(rewriter, loc, B, bIndices,
                                              computeType, bUnsigned,
                                              cUnsigned);
  mlir::Value result;
  if (activationBackward) {
    if (!computeFloat) {
      return op.emitOpError(
          "activation backward requires a floating-point AccT");
    }
    result = emitActivationBackward(
        rewriter, loc, op, semantic, computeFloat, aVal, bVal);
  } else if (computeFloat && parameterizedBinary) {
    result = emitParameterizedBinary(
        rewriter, loc, computeType, semantic, aVal, bVal, scalar0, scalar1,
        scalar0Attr != nullptr);
  } else if (computeFloat && parameterizedComparison) {
    auto difference = rewriter.create<mlir::math::AbsFOp>(
        loc, rewriter.create<mlir::arith::SubFOp>(loc, aVal, bVal));
    result = rewriter.create<mlir::arith::CmpFOp>(
        loc, mlir::arith::CmpFPredicate::OLE, difference,
        floatConst(rewriter, loc, computeFloat, scalar0));
  } else if (computeFloat) {
    result = callback(rewriter, loc, aVal, bVal);
  } else {
    result = emitIntegerBinary(
        rewriter, loc, semantic, aVal, bVal, aUnsigned);
  }
  if (!result) {
    return op.emitOpError(
        "ElementwiseBinaryToSpirv: unsupported semantic metadata contract");
  }
  if (!storeScalar(rewriter, loc, result, C, outputIndices,
                   cUnsigned, cUnsigned)) {
    return op.emitOpError(
        "ElementwiseBinaryToSpirv: result storage conversion failed");
  }
  rewriter.create<mlir::gpu::TerminatorOp>(loc);

  rewriter.eraseOp(op);
  return mlir::success();
}

// ─────────────────────────────────────────────────────────────────────────────
//  Wave 1: ElementwiseUnaryToSpirv
// ─────────────────────────────────────────────────────────────────────────────
//
// Lowers unary activation ops (silu/swish, gelu, tanh, sigmoid, relu) tagged
// with nd4j.unary = true in a linalg.generic.
//
// One Vulkan invocation per logical output element.

mlir::LogicalResult ElementwiseUnaryToSpirv::matchAndRewrite(
    mlir::linalg::GenericOp op,
    mlir::PatternRewriter& rewriter) const {

  mlir::Location loc = op.getLoc();

  // ── 1. Guard: only match generics with nd4j.unary = true ─────────────────
  auto unaryAttr = op->getAttrOfType<mlir::BoolAttr>(kUnaryAttr);
  if (!unaryAttr || !unaryAttr.getValue()) {
    return mlir::failure();
  }

  const auto* emitter = emitterForOperation(op);
  const VulkanKernelRecipe semantic =
      emitter == nullptr ? VulkanKernelRecipe::UNSUPPORTED
                         : legacySemanticFor(op, emitter->recipe);
  const bool isCast =
      emitter != nullptr && emitter->family == VulkanKernelFamily::CAST;
  const bool floatingResultUnary =
      emitter != nullptr && hasVulkanEmitterTrait(
          *emitter, VULKAN_EMITTER_TRAIT_FLOAT_RESULT);
  const bool booleanResultUnary =
      emitter != nullptr && hasVulkanEmitterTrait(
          *emitter, VULKAN_EMITTER_TRAIT_BOOLEAN_RESULT);
  if (emitter == nullptr ||
      (emitter->family != VulkanKernelFamily::ELEMENTWISE_UNARY && !isCast)) {
    return mlir::failure();
  }

  mlir::ValueRange inputs = op.getInputs();
  mlir::ValueRange outputs = op.getOutputs();
  auto boundsAttr =
      op->getAttrOfType<mlir::BoolAttr>("nd4j.bounds_from_inputs");
  const bool boundsFromInputs = boundsAttr && boundsAttr.getValue();
  const size_t expectedInputs = boundsFromInputs ? 3 : 1;
  if (inputs.size() != expectedInputs || outputs.size() != 1) {
    return op.emitOpError(
        "ElementwiseUnaryToSpirv: unexpected input/output arity");
  }
  mlir::Value X = inputs[0];
  mlir::Value Y = outputs[0];
  auto xType = llvm::dyn_cast<mlir::MemRefType>(X.getType());
  auto yType = llvm::dyn_cast<mlir::MemRefType>(Y.getType());
  if (!xType || !yType || xType.getRank() != yType.getRank() ||
      (!isCast && !floatingResultUnary && !booleanResultUnary &&
       xType.getElementType() != yType.getElementType())) {
    return op.emitOpError(
        "ElementwiseUnaryToSpirv: expected rank-compatible MemRefs");
  }
  if (boundsFromInputs) {
    if (!vulkanArgumentContractAcceptsInputCount(*emitter, 3) ||
        !llvm::isa<mlir::MemRefType>(inputs[1].getType()) ||
        !llvm::isa<mlir::MemRefType>(inputs[2].getType())) {
      return op.emitOpError(
          "ElementwiseUnaryToSpirv: invalid tensor-bound unary operands");
    }
  }

  auto computeAttr = op->getAttrOfType<mlir::TypeAttr>(kAccumulatorTypeAttr);
  mlir::Type computeType = computeAttr ? computeAttr.getValue() : mlir::Type{};
  auto computeFloat = llvm::dyn_cast<mlir::FloatType>(computeType);
  auto computeInteger = llvm::dyn_cast<mlir::IntegerType>(computeType);
  const bool parameterized =
      hasVulkanScalarArgumentSchema(*emitter);
  UnaryCallback callback = unaryCallbackFor(semantic);
  if (!isCast && computeFloat && !parameterized && !callback &&
      semantic != VulkanKernelRecipe::ASSIGN) {
    return mlir::failure();
  }
  const bool integerSemantic =
      (emitter->dtypeSupport &
       (VULKAN_DTYPE_SIGNED_INT32 | VULKAN_DTYPE_UNSIGNED_INT32)) != 0 &&
      !floatingResultUnary;
  if (!isCast && computeInteger &&
      (computeInteger.getWidth() != 32 || !integerSemantic)) {
    return mlir::failure();
  }
  if (!isCast && !computeFloat && !computeInteger) return mlir::failure();

  auto readBoolAttr = [&](llvm::StringRef name) {
    auto attr = op->getAttrOfType<mlir::BoolAttr>(name);
    return attr && attr.getValue();
  };
  const bool inputUnsigned = readBoolAttr("nd4j.input0_unsigned");
  const bool lowerUnsigned = readBoolAttr("nd4j.input1_unsigned");
  const bool upperUnsigned = readBoolAttr("nd4j.input2_unsigned");
  const bool outputUnsigned = readBoolAttr("nd4j.output_unsigned");
  auto scalar0Attr = op->getAttrOfType<mlir::FloatAttr>("nd4j.scalar0");
  auto scalar1Attr = op->getAttrOfType<mlir::FloatAttr>("nd4j.scalar1");
  const double scalar0 = scalar0Attr ? scalar0Attr.getValueAsDouble() : 0.0;
  const double scalar1 = scalar1Attr ? scalar1Attr.getValueAsDouble() : 0.0;

  mlir::Value zeroIdx = idxConst(rewriter, loc, 0);
  mlir::Value oneIdx = idxConst(rewriter, loc, 1);
  mlir::Value totalN = idxConst(rewriter, loc, 1);
  for (int64_t d = 0; d < yType.getRank(); ++d) {
    mlir::Value dim = rewriter.create<mlir::memref::DimOp>(loc, Y, d);
    totalN = rewriter.create<mlir::arith::MulIOp>(loc, totalN, dim);
  }

  auto launch = createGpuLaunch(rewriter, loc, totalN, oneIdx, oneIdx);
  mlir::Value linearIndex = launch.getBlockIds().x;
  rewriter.setInsertionPointToEnd(&launch.getBody().front());

  auto indices = logicalIndices(rewriter, loc, linearIndex, Y);
  if (isCast) {
    mlir::Value raw =
        rewriter.create<mlir::memref::LoadOp>(loc, X, indices);
    mlir::Value converted = convertScalar(
        rewriter, loc, raw, yType.getElementType(),
        inputUnsigned, outputUnsigned);
    if (converted) {
      rewriter.create<mlir::memref::StoreOp>(loc, converted, Y, indices);
    }
  } else {
    mlir::Value xVal = loadAsScalar(
        rewriter, loc, X, indices, computeType,
        inputUnsigned, inputUnsigned);
    mlir::Value result;
    if (boundsFromInputs) {
      mlir::Value lowerMemref = inputs[1];
      mlir::Value upperMemref = inputs[2];
      auto lowerType = llvm::cast<mlir::MemRefType>(lowerMemref.getType());
      auto upperType = llvm::cast<mlir::MemRefType>(upperMemref.getType());
      mlir::SmallVector<mlir::Value> lowerIndices(
          static_cast<size_t>(lowerType.getRank()), zeroIdx);
      mlir::SmallVector<mlir::Value> upperIndices(
          static_cast<size_t>(upperType.getRank()), zeroIdx);
      mlir::Value lower = loadAsScalar(
          rewriter, loc, lowerMemref, lowerIndices, computeType,
          lowerUnsigned, inputUnsigned);
      mlir::Value upper = loadAsScalar(
          rewriter, loc, upperMemref, upperIndices, computeType,
          upperUnsigned, inputUnsigned);
      result = emitClipWithValues(
          rewriter, loc, computeType, xVal, lower, upper, inputUnsigned);
    } else if (parameterized) {
      result = emitParameterizedUnary(
          rewriter, loc, computeType, semantic, xVal,
          scalar0, scalar1, inputUnsigned);
    } else if (semantic == VulkanKernelRecipe::ASSIGN) {
      // Identity/copy legacy transforms share the unary lowering contract.  They
      // deliberately have no callback: forwarding the loaded value is the device
      // equation, and must happen before the callback branches below.
      result = xVal;
    } else if (computeFloat) {
      result = callback(rewriter, loc, computeType, xVal);
    } else if (callback) {
      result = callback(rewriter, loc, computeType, xVal);
    } else if (semantic == VulkanKernelRecipe::SQUARE ||
               semantic == VulkanKernelRecipe::CUBE) {
      result = rewriter.create<mlir::arith::MulIOp>(loc, xVal, xVal);
      if (semantic == VulkanKernelRecipe::CUBE) {
        result = rewriter.create<mlir::arith::MulIOp>(loc, result, xVal);
      }
    } else {
      return op.emitOpError(
          "ElementwiseUnaryToSpirv: no Vulkan equation for legacy operation");
    }
    (void)storeScalar(rewriter, loc, result, Y, indices,
                      inputUnsigned, outputUnsigned);
  }
  rewriter.create<mlir::gpu::TerminatorOp>(loc);

  rewriter.eraseOp(op);
  return mlir::success();
}

// ─────────────────────────────────────────────────────────────────────────────
//  Reusable multi-output elementwise contract
// ─────────────────────────────────────────────────────────────────────────────

mlir::LogicalResult MultiOutputElementwiseToSpirv::matchAndRewrite(
    mlir::linalg::GenericOp op,
    mlir::PatternRewriter& rewriter) const {
  mlir::Location loc = op.getLoc();
  auto marker =
      op->getAttrOfType<mlir::BoolAttr>("nd4j.multi_output_elementwise");
  if (!marker || !marker.getValue()) return mlir::failure();

  const auto* emitter = emitterForOperation(op);
  if (emitter == nullptr ||
      !usesMultiOutputNormalizationSchedule(*emitter)) {
    return mlir::failure();
  }

  mlir::ValueRange inputs = op.getInputs();
  mlir::ValueRange outputs = op.getOutputs();
  if (inputs.size() != 3 || outputs.size() != 2) {
    return op.emitOpError(
        "multi-output elementwise contract requires three inputs and two outputs");
  }

  auto computeAttr = op->getAttrOfType<mlir::TypeAttr>(kAccumulatorTypeAttr);
  auto computeType =
      computeAttr ? llvm::dyn_cast<mlir::FloatType>(computeAttr.getValue())
                  : mlir::FloatType{};
  if (!computeType) {
    return op.emitOpError(
        "multi-output elementwise contract requires floating-point AccT");
  }

  auto readBoolAttr = [&](llvm::StringRef name) {
    auto attr = op->getAttrOfType<mlir::BoolAttr>(name);
    return attr && attr.getValue();
  };
  const bool countUnsigned = readBoolAttr("nd4j.input0_unsigned");
  const bool meanUnsigned = readBoolAttr("nd4j.input1_unsigned");
  const bool varianceUnsigned = readBoolAttr("nd4j.input2_unsigned");

  for (auto item : llvm::enumerate(inputs)) {
    auto memref = llvm::dyn_cast<mlir::MemRefType>(item.value().getType());
    mlir::Type element = memref ? memref.getElementType() : mlir::Type{};
    auto floatType = llvm::dyn_cast<mlir::FloatType>(element);
    auto integerType = llvm::dyn_cast<mlir::IntegerType>(element);
    if (!memref ||
        (!floatType && (!integerType || integerType.getWidth() != 32)) ||
        (floatType && floatType.getWidth() > computeType.getWidth())) {
      return op.emitOpError()
             << "input " << item.index()
             << " cannot convert to the selected floating-point AccT";
    }
  }
  for (auto item : llvm::enumerate(outputs)) {
    auto memref = llvm::dyn_cast<mlir::MemRefType>(item.value().getType());
    auto floatType =
        memref ? llvm::dyn_cast<mlir::FloatType>(memref.getElementType())
               : mlir::FloatType{};
    if (!floatType || floatType.getWidth() > computeType.getWidth()) {
      return op.emitOpError()
             << "output " << item.index()
             << " must be a floating-point MemRef compatible with AccT";
    }
  }

  mlir::Value counts = inputs[0];
  mlir::Value means = inputs[1];
  mlir::Value variances = inputs[2];
  mlir::Value outputMeans = outputs[0];
  mlir::Value outputVariances = outputs[1];

  mlir::Value one = idxConst(rewriter, loc, 1);
  mlir::Value total = one;
  auto outputType = llvm::cast<mlir::MemRefType>(outputMeans.getType());
  for (int64_t d = 0; d < outputType.getRank(); ++d) {
    total = rewriter.create<mlir::arith::MulIOp>(
        loc, total,
        rewriter.create<mlir::memref::DimOp>(loc, outputMeans, d));
  }

  auto launch = createGpuLaunch(rewriter, loc, total, one, one);
  mlir::Value linear = launch.getBlockIds().x;
  rewriter.setInsertionPointToEnd(&launch.getBody().front());

  auto indices = logicalIndices(rewriter, loc, linear, outputMeans);
  auto countType = llvm::cast<mlir::MemRefType>(counts.getType());
  mlir::Value zero = idxConst(rewriter, loc, 0);
  mlir::SmallVector<mlir::Value> countIndices(
      static_cast<size_t>(countType.getRank()), zero);

  mlir::Value count = loadAsScalar(
      rewriter, loc, counts, countIndices, computeType, countUnsigned, false);
  mlir::Value summedMean = loadAsScalar(
      rewriter, loc, means, indices, computeType, meanUnsigned, false);
  mlir::Value summedVariance = loadAsScalar(
      rewriter, loc, variances, indices, computeType, varianceUnsigned, false);
  if (!count || !summedMean || !summedVariance) {
    return op.emitOpError(
        "multi-output elementwise input conversion is unsupported");
  }

  mlir::Value mean =
      rewriter.create<mlir::arith::DivFOp>(loc, summedMean, count);
  mlir::Value secondMoment =
      rewriter.create<mlir::arith::DivFOp>(loc, summedVariance, count);
  mlir::Value meanSquare =
      rewriter.create<mlir::arith::MulFOp>(loc, mean, mean);
  mlir::Value variance =
      rewriter.create<mlir::arith::SubFOp>(loc, secondMoment, meanSquare);
  auto shiftAttr = op->getAttrOfType<mlir::FloatAttr>("nd4j.shift");
  mlir::Value shift = floatConst(
      rewriter, loc, computeType, shiftAttr ? shiftAttr.getValueAsDouble() : 0.0);
  mlir::Value shiftedMean =
      rewriter.create<mlir::arith::AddFOp>(loc, mean, shift);

  storeFromAccumulator(
      rewriter, loc, shiftedMean, outputMeans, indices);
  storeFromAccumulator(
      rewriter, loc, variance, outputVariances, indices);
  rewriter.create<mlir::gpu::TerminatorOp>(loc);
  rewriter.eraseOp(op);
  return mlir::success();
}

// ─────────────────────────────────────────────────────────────────────────────
//  Reusable batched rank-two matrix-list contract
// ─────────────────────────────────────────────────────────────────────────────

mlir::LogicalResult BatchedMatrixListToSpirv::matchAndRewrite(
    mlir::linalg::GenericOp op,
    mlir::PatternRewriter& rewriter) const {
  auto marker =
      op->getAttrOfType<mlir::BoolAttr>("nd4j.batched_matrix_list");
  if (!marker || !marker.getValue()) return mlir::failure();

  sd::LongType opHash = 0;
  if (!readOpHash(op, opHash)) return mlir::failure();
  const auto* emitter = findVulkanKernelEmitter(opHash);
  if (emitter == nullptr || !usesBatchedMatrixListSchedule(*emitter)) {
    return mlir::failure();
  }

  auto batchAttr =
      op->getAttrOfType<mlir::IntegerAttr>("nd4j.batch_count");
  const int64_t batch = batchAttr ? batchAttr.getInt() : 0;
  mlir::ValueRange inputs = op.getInputs();
  mlir::ValueRange outputs = op.getOutputs();
  if (batch <= 0 || inputs.size() != static_cast<size_t>(2 + 2 * batch) ||
      outputs.size() != static_cast<size_t>(batch)) {
    return op.emitOpError(
        "batched matrix-list contract has inconsistent operand counts");
  }
  auto typeContract = getComputeTypeContract(op, inputs, outputs);
  if (mlir::failed(typeContract)) return mlir::failure();
  mlir::FloatType accType = typeContract->accumulatorType;

  auto transposeAAttr =
      op->getAttrOfType<mlir::BoolAttr>("nd4j.transpose_a");
  auto transposeBAttr =
      op->getAttrOfType<mlir::BoolAttr>("nd4j.transpose_b");
  const bool transposeA = transposeAAttr && transposeAAttr.getValue();
  const bool transposeB = transposeBAttr && transposeBAttr.getValue();

  auto alphaType = llvm::dyn_cast<mlir::MemRefType>(inputs[0].getType());
  auto betaType = llvm::dyn_cast<mlir::MemRefType>(inputs[1].getType());
  auto outputType = llvm::dyn_cast<mlir::MemRefType>(outputs[0].getType());
  if (!alphaType || !betaType || alphaType.getRank() > 1 ||
      betaType.getRank() > 1 || !outputType || outputType.getRank() != 2) {
    return op.emitOpError(
        "batched matrix-list requires scalar/vector coefficients and rank-two outputs");
  }
  for (int64_t b = 0; b < batch; ++b) {
    auto aType = llvm::dyn_cast<mlir::MemRefType>(
        inputs[static_cast<size_t>(2 + b)].getType());
    auto bType = llvm::dyn_cast<mlir::MemRefType>(
        inputs[static_cast<size_t>(2 + batch + b)].getType());
    auto cType = llvm::dyn_cast<mlir::MemRefType>(
        outputs[static_cast<size_t>(b)].getType());
    if (!aType || !bType || !cType || aType.getRank() != 2 ||
        bType.getRank() != 2 || cType.getRank() != 2) {
      return op.emitOpError(
          "every batched matrix-list operand must be rank two");
    }
  }

  mlir::Location loc = op.getLoc();
  mlir::Value one = idxConst(rewriter, loc, 1);
  mlir::Value rows =
      rewriter.create<mlir::memref::DimOp>(loc, outputs[0], 0);
  mlir::Value columns =
      rewriter.create<mlir::memref::DimOp>(loc, outputs[0], 1);
  mlir::Value total =
      rewriter.create<mlir::arith::MulIOp>(loc, rows, columns);
  auto launch = createGpuLaunch(rewriter, loc, total, one, one);
  mlir::Value linear = launch.getBlockIds().x;
  rewriter.setInsertionPointToEnd(&launch.getBody().front());
  mlir::Value row =
      rewriter.create<mlir::arith::DivUIOp>(loc, linear, columns);
  mlir::Value column =
      rewriter.create<mlir::arith::RemUIOp>(loc, linear, columns);
  mlir::Value zeroIndex = idxConst(rewriter, loc, 0);
  mlir::Value zero = floatConst(rewriter, loc, accType, 0.0);

  auto coefficientIndices = [&](mlir::OpBuilder& builder,
                                mlir::Location nestedLoc,
                                mlir::Value coefficient,
                                int64_t batchIndex) {
    mlir::SmallVector<mlir::Value> result;
    auto type = llvm::cast<mlir::MemRefType>(coefficient.getType());
    if (type.getRank() == 1) {
      result.push_back(
          type.getDimSize(0) == 1
              ? idxConst(builder, nestedLoc, 0)
              : idxConst(builder, nestedLoc, batchIndex));
    }
    return result;
  };

  for (int64_t b = 0; b < batch; ++b) {
    mlir::Value matrixA = inputs[static_cast<size_t>(2 + b)];
    mlir::Value matrixB = inputs[static_cast<size_t>(2 + batch + b)];
    mlir::Value output = outputs[static_cast<size_t>(b)];
    mlir::Value inner = rewriter.create<mlir::memref::DimOp>(
        loc, matrixA, transposeA ? 0 : 1);
    auto reduction = emitReductionLoop(
        rewriter, loc, zeroIndex, inner, one, zero,
        [&](mlir::OpBuilder& builder, mlir::Location nestedLoc,
            mlir::Value k, mlir::Value accumulator) -> mlir::Value {
          mlir::SmallVector<mlir::Value> aIndices =
              transposeA
                  ? mlir::SmallVector<mlir::Value>{k, row}
                  : mlir::SmallVector<mlir::Value>{row, k};
          mlir::SmallVector<mlir::Value> bIndices =
              transposeB
                  ? mlir::SmallVector<mlir::Value>{column, k}
                  : mlir::SmallVector<mlir::Value>{k, column};
          mlir::Value a = loadAsAccumulator(
              builder, nestedLoc, matrixA, aIndices, accType);
          mlir::Value valueB = loadAsAccumulator(
              builder, nestedLoc, matrixB, bIndices, accType);
          mlir::Value product =
              builder.create<mlir::arith::MulFOp>(nestedLoc, a, valueB);
          return builder.create<mlir::arith::AddFOp>(
              nestedLoc, accumulator, product);
        });
    mlir::Value alpha = loadAsAccumulator(
        rewriter, loc, inputs[0],
        coefficientIndices(rewriter, loc, inputs[0], b), accType);
    mlir::Value beta = loadAsAccumulator(
        rewriter, loc, inputs[1],
        coefficientIndices(rewriter, loc, inputs[1], b), accType);
    // batched_gemm has no C input. Its fully-writing contract therefore uses an
    // implicit zero initial C while retaining beta as a real dynamic operand.
    mlir::Value scaled = rewriter.create<mlir::arith::MulFOp>(
        loc, alpha, reduction.getResult(0));
    mlir::Value betaZero =
        rewriter.create<mlir::arith::MulFOp>(loc, beta, zero);
    mlir::Value result =
        rewriter.create<mlir::arith::AddFOp>(loc, scaled, betaZero);
    storeFromAccumulator(
        rewriter, loc, result, output,
        mlir::SmallVector<mlir::Value>{row, column});
  }

  rewriter.create<mlir::gpu::TerminatorOp>(loc);
  rewriter.eraseOp(op);
  return mlir::success();
}

// ─────────────────────────────────────────────────────────────────────────────
//  Reusable serial indexed-accumulation contract
// ─────────────────────────────────────────────────────────────────────────────

mlir::LogicalResult IndexedAccumulationToSpirv::matchAndRewrite(
    mlir::linalg::GenericOp op,
    mlir::PatternRewriter& rewriter) const {
  auto marker =
      op->getAttrOfType<mlir::BoolAttr>("nd4j.indexed_accumulation");
  if (!marker || !marker.getValue()) return mlir::failure();

  sd::LongType opHash = 0;
  if (!readOpHash(op, opHash)) return mlir::failure();
  const auto* emitter = findVulkanKernelEmitter(opHash);
  if (emitter == nullptr || !usesIndexedAccumulationSchedule(*emitter)) {
    return mlir::failure();
  }

  mlir::ValueRange inputs = op.getInputs();
  mlir::ValueRange outputs = op.getOutputs();
  if (inputs.size() != 3 || outputs.size() != 1) {
    return op.emitOpError(
        "indexed accumulation requires indices, updates, shape, and one output");
  }
  mlir::Value indices = inputs[0];
  mlir::Value updates = inputs[1];
  mlir::Value output = outputs[0];
  auto indicesType = llvm::dyn_cast<mlir::MemRefType>(indices.getType());
  auto updatesType = llvm::dyn_cast<mlir::MemRefType>(updates.getType());
  auto outputType = llvm::dyn_cast<mlir::MemRefType>(output.getType());
  auto indexElement = indicesType
                          ? llvm::dyn_cast<mlir::IntegerType>(
                                indicesType.getElementType())
                          : mlir::IntegerType{};
  if (!indicesType || !updatesType || !outputType || !indexElement ||
      (indexElement.getWidth() != 32 && indexElement.getWidth() != 64) ||
      updatesType.getElementType() != outputType.getElementType()) {
    return op.emitOpError(
        "indexed accumulation requires integer indices and exact update/output types");
  }

  auto indexDepthAttr =
      op->getAttrOfType<mlir::IntegerAttr>("nd4j.index_depth");
  auto prefixRankAttr =
      op->getAttrOfType<mlir::IntegerAttr>("nd4j.prefix_rank");
  const int64_t indexDepth =
      indexDepthAttr ? indexDepthAttr.getInt() : -1;
  const int64_t prefixRank =
      prefixRankAttr ? prefixRankAttr.getInt() : -1;
  const int64_t sliceRank = outputType.getRank() - indexDepth;
  if (indexDepth <= 0 || prefixRank < 0 ||
      indicesType.getRank() != prefixRank + 1 ||
      updatesType.getRank() != prefixRank + sliceRank) {
    return op.emitOpError(
        "indexed accumulation rank metadata is inconsistent");
  }

  mlir::Type storageType = outputType.getElementType();
  auto floatStorage = llvm::dyn_cast<mlir::FloatType>(storageType);
  auto integerStorage = llvm::dyn_cast<mlir::IntegerType>(storageType);
  mlir::FloatType accType;
  if (floatStorage) {
    auto computeAttr =
        op->getAttrOfType<mlir::TypeAttr>(kAccumulatorTypeAttr);
    accType = computeAttr
                  ? llvm::dyn_cast<mlir::FloatType>(computeAttr.getValue())
                  : mlir::FloatType{};
    if (!accType || floatStorage.getWidth() > accType.getWidth()) {
      return op.emitOpError(
          "floating indexed accumulation requires a compatible AccT");
    }
  } else if (!integerStorage ||
             (integerStorage.getWidth() != 32 &&
              integerStorage.getWidth() != 64)) {
    return op.emitOpError(
        "indexed accumulation supports floating or 32/64-bit integer payloads");
  }

  mlir::Location loc = op.getLoc();
  mlir::Value zeroIndex = idxConst(rewriter, loc, 0);
  mlir::Value one = idxConst(rewriter, loc, 1);
  auto launch = createGpuLaunch(rewriter, loc, one, one, one);
  rewriter.setInsertionPointToEnd(&launch.getBody().front());

  mlir::Value outputLength = one;
  for (int64_t d = 0; d < outputType.getRank(); ++d) {
    outputLength = rewriter.create<mlir::arith::MulIOp>(
        loc, outputLength,
        rewriter.create<mlir::memref::DimOp>(loc, output, d));
  }
  rewriter.create<mlir::scf::ForOp>(
      loc, zeroIndex, outputLength, one, mlir::ValueRange{},
      [&](mlir::OpBuilder& builder, mlir::Location nestedLoc,
          mlir::Value linear, mlir::ValueRange) {
        mlir::Value zeroValue;
        if (floatStorage) {
          zeroValue = floatConst(builder, nestedLoc, floatStorage, 0.0);
        } else {
          zeroValue = builder.create<mlir::arith::ConstantOp>(
              nestedLoc, integerStorage,
              builder.getIntegerAttr(integerStorage, 0));
        }
        builder.create<mlir::memref::StoreOp>(
            nestedLoc, zeroValue, output,
            logicalIndices(builder, nestedLoc, linear, output));
        builder.create<mlir::scf::YieldOp>(nestedLoc);
      });

  mlir::Value updateLength = one;
  for (int64_t d = 0; d < updatesType.getRank(); ++d) {
    updateLength = rewriter.create<mlir::arith::MulIOp>(
        loc, updateLength,
        rewriter.create<mlir::memref::DimOp>(loc, updates, d));
  }
  rewriter.create<mlir::scf::ForOp>(
      loc, zeroIndex, updateLength, one, mlir::ValueRange{},
      [&](mlir::OpBuilder& builder, mlir::Location nestedLoc,
          mlir::Value linear, mlir::ValueRange) {
        auto updateCoordinates =
            logicalIndices(builder, nestedLoc, linear, updates);
        mlir::SmallVector<mlir::Value> outputCoordinates;
        outputCoordinates.reserve(
            static_cast<size_t>(outputType.getRank()));
        for (int64_t d = 0; d < indexDepth; ++d) {
          mlir::SmallVector<mlir::Value> indexCoordinates;
          indexCoordinates.reserve(static_cast<size_t>(prefixRank + 1));
          for (int64_t p = 0; p < prefixRank; ++p) {
            indexCoordinates.push_back(
                updateCoordinates[static_cast<size_t>(p)]);
          }
          indexCoordinates.push_back(idxConst(builder, nestedLoc, d));
          mlir::Value rawIndex =
              builder.create<mlir::memref::LoadOp>(
                  nestedLoc, indices, indexCoordinates);
          outputCoordinates.push_back(
              builder.create<mlir::arith::IndexCastOp>(
                  nestedLoc, builder.getIndexType(), rawIndex));
        }
        for (int64_t d = 0; d < sliceRank; ++d) {
          outputCoordinates.push_back(
              updateCoordinates[static_cast<size_t>(prefixRank + d)]);
        }

        if (floatStorage) {
          mlir::Value update = loadAsAccumulator(
              builder, nestedLoc, updates, updateCoordinates, accType);
          mlir::Value current = loadAsAccumulator(
              builder, nestedLoc, output, outputCoordinates, accType);
          mlir::Value sum = builder.create<mlir::arith::AddFOp>(
              nestedLoc, current, update);
          storeFromAccumulator(
              builder, nestedLoc, sum, output, outputCoordinates);
        } else {
          mlir::Value update =
              builder.create<mlir::memref::LoadOp>(
                  nestedLoc, updates, updateCoordinates);
          mlir::Value current =
              builder.create<mlir::memref::LoadOp>(
                  nestedLoc, output, outputCoordinates);
          mlir::Value sum = builder.create<mlir::arith::AddIOp>(
              nestedLoc, current, update);
          builder.create<mlir::memref::StoreOp>(
              nestedLoc, sum, output, outputCoordinates);
        }
        builder.create<mlir::scf::YieldOp>(nestedLoc);
      });

  rewriter.create<mlir::gpu::TerminatorOp>(loc);
  rewriter.eraseOp(op);
  return mlir::success();
}

// ─────────────────────────────────────────────────────────────────────────────
//  Reusable indexed TAD movement contract
// ─────────────────────────────────────────────────────────────────────────────

mlir::LogicalResult IndexedTadMovementToSpirv::matchAndRewrite(
    mlir::linalg::GenericOp op,
    mlir::PatternRewriter& rewriter) const {
  auto marker =
      op->getAttrOfType<mlir::BoolAttr>("nd4j.indexed_tad_movement");
  if (!marker || !marker.getValue()) return mlir::failure();

  sd::LongType opHash = 0;
  if (!readOpHash(op, opHash)) return mlir::failure();
  const auto* emitter = findVulkanKernelEmitter(opHash);
  if (emitter == nullptr || !usesIndexedTadMovementSchedule(*emitter)) {
    return mlir::failure();
  }

  mlir::ValueRange inputs = op.getInputs();
  mlir::ValueRange outputs = op.getOutputs();
  mlir::Location loc = op.getLoc();

  auto emitSingleLaunch = [&]() {
    mlir::Value one = idxConst(rewriter, loc, 1);
    auto launch = createGpuLaunch(rewriter, loc, one, one, one);
    rewriter.setInsertionPointToEnd(&launch.getBody().front());
    return launch;
  };

  auto productOfAxes = [&](mlir::OpBuilder& builder, mlir::Location nestedLoc,
                           mlir::Value memref,
                           llvm::ArrayRef<int64_t> axes) {
    mlir::Value product = idxConst(builder, nestedLoc, 1);
    for (int64_t axis : axes) {
      product = builder.create<mlir::arith::MulIOp>(
          nestedLoc, product,
          builder.create<mlir::memref::DimOp>(nestedLoc, memref, axis));
    }
    return product;
  };

  auto decompose = [&](mlir::OpBuilder& builder, mlir::Location nestedLoc,
                       mlir::Value linear, mlir::Value memref,
                       llvm::ArrayRef<int64_t> axes) {
    mlir::SmallVector<mlir::Value> coordinates(axes.size());
    mlir::Value remaining = linear;
    for (int64_t index = static_cast<int64_t>(axes.size()) - 1;
         index >= 0; --index) {
      mlir::Value dimension = builder.create<mlir::memref::DimOp>(
          nestedLoc, memref, axes[static_cast<size_t>(index)]);
      coordinates[static_cast<size_t>(index)] =
          builder.create<mlir::arith::RemUIOp>(
              nestedLoc, remaining, dimension);
      remaining = builder.create<mlir::arith::DivUIOp>(
          nestedLoc, remaining, dimension);
    }
    return coordinates;
  };

  auto partitionAxes = [](int64_t rank, llvm::ArrayRef<int64_t> rawAxes,
                          mlir::SmallVector<int64_t>& tadAxes,
                          mlir::SmallVector<int64_t>& itemAxes) {
    if (rank <= 0 || rawAxes.empty()) return false;
    mlir::SmallVector<int64_t> normalized;
    normalized.reserve(rawAxes.size());
    mlir::SmallVector<int8_t> selected(static_cast<size_t>(rank), 0);
    for (int64_t rawAxis : rawAxes) {
      int64_t axis = rawAxis < 0 ? rawAxis + rank : rawAxis;
      if (axis < 0 || axis >= rank ||
          selected[static_cast<size_t>(axis)] != 0) {
        return false;
      }
      selected[static_cast<size_t>(axis)] = 1;
      normalized.push_back(axis);
    }
    std::sort(normalized.begin(), normalized.end());

    // CUDA's shuffle kernel treats rank-one arrays as vectors: every element is
    // an item even though the framework TAD descriptor contains axis zero.
    if (rank == 1) {
      if (normalized.size() != 1 || normalized.front() != 0) return false;
      tadAxes.clear();
      itemAxes.assign(1, 0);
      return true;
    }

    tadAxes.assign(normalized.begin(), normalized.end());
    itemAxes.clear();
    for (int64_t axis = 0; axis < rank; ++axis) {
      if (selected[static_cast<size_t>(axis)] == 0) {
        itemAxes.push_back(axis);
      }
    }
    return true;
  };

  if (emitter->recipe == VulkanKernelRecipe::PULL_INDEXED_TADS) {
    if (inputs.size() != 2 || outputs.size() != 1) {
      return op.emitOpError(
          "indexed TAD pull requires source, indices, and one output");
    }
    auto sourceType = llvm::dyn_cast<mlir::MemRefType>(inputs[0].getType());
    auto indexType = llvm::dyn_cast<mlir::MemRefType>(inputs[1].getType());
    auto outputType = llvm::dyn_cast<mlir::MemRefType>(outputs[0].getType());
    auto indexElement =
        indexType
            ? llvm::dyn_cast<mlir::IntegerType>(indexType.getElementType())
            : mlir::IntegerType{};
    auto itemCountAttr =
        op->getAttrOfType<mlir::IntegerAttr>("nd4j.item_count");
    auto tadDimensionAttr =
        op->getAttrOfType<mlir::IntegerAttr>("nd4j.tad_dimension");
    if (!sourceType || !indexType || !outputType || !indexElement ||
        indexElement.getWidth() != 64 || indexType.getRank() < 1 ||
        sourceType.getRank() < 1 || sourceType.getRank() > 2 ||
        outputType.getRank() != sourceType.getRank() ||
        sourceType.getElementType() != outputType.getElementType() ||
        !itemCountAttr || itemCountAttr.getInt() <= 0 ||
        !tadDimensionAttr) {
      return op.emitOpError("indexed TAD pull contract mismatch");
    }

    const int64_t tadDimension = tadDimensionAttr.getInt();
    const int64_t rank = sourceType.getRank();
    if (tadDimension < 0 || tadDimension >= rank) {
      return op.emitOpError("indexed TAD pull dimension is out of range");
    }
    const int64_t itemAxis = rank == 1 ? 0 : 1 - tadDimension;

    emitSingleLaunch();
    mlir::Value zero = idxConst(rewriter, loc, 0);
    mlir::Value one = idxConst(rewriter, loc, 1);
    mlir::Value outputLength = one;
    for (int64_t axis = 0; axis < outputType.getRank(); ++axis) {
      outputLength = rewriter.create<mlir::arith::MulIOp>(
          loc, outputLength,
          rewriter.create<mlir::memref::DimOp>(loc, outputs[0], axis));
    }
    rewriter.create<mlir::scf::ForOp>(
        loc, zero, outputLength, one, mlir::ValueRange{},
        [&](mlir::OpBuilder& builder, mlir::Location nestedLoc,
            mlir::Value linear, mlir::ValueRange) {
          auto outputCoordinates =
              logicalIndices(builder, nestedLoc, linear, outputs[0]);
          mlir::Value item =
              outputCoordinates[static_cast<size_t>(itemAxis)];
          auto indexCoordinates =
              logicalIndices(builder, nestedLoc, item, inputs[1]);
          mlir::Value rawIndex = builder.create<mlir::memref::LoadOp>(
              nestedLoc, inputs[1], indexCoordinates);
          mlir::Value selected = builder.create<mlir::arith::IndexCastOp>(
              nestedLoc, builder.getIndexType(), rawIndex);
          auto sourceCoordinates = outputCoordinates;
          sourceCoordinates[static_cast<size_t>(itemAxis)] = selected;
          mlir::Value value = builder.create<mlir::memref::LoadOp>(
              nestedLoc, inputs[0], sourceCoordinates);
          builder.create<mlir::memref::StoreOp>(
              nestedLoc, value, outputs[0], outputCoordinates);
          builder.create<mlir::scf::YieldOp>(nestedLoc);
        });

    rewriter.create<mlir::gpu::TerminatorOp>(loc);
    rewriter.eraseOp(op);
    return mlir::success();
  }

  if (emitter->recipe != VulkanKernelRecipe::DISJOINT_PAIR_SHUFFLE) {
    return mlir::failure();
  }

  auto arrayCountAttr =
      op->getAttrOfType<mlir::IntegerAttr>("nd4j.array_count");
  auto tadDimensionsAttr =
      op->getAttrOfType<mlir::DenseI64ArrayAttr>("nd4j.tad_dimensions");
  if (!arrayCountAttr || arrayCountAttr.getInt() <= 0 ||
      !tadDimensionsAttr || tadDimensionsAttr.empty()) {
    return op.emitOpError("indexed TAD shuffle metadata is missing");
  }
  const int64_t arrayCount = arrayCountAttr.getInt();
  if (arrayCount > std::numeric_limits<int>::max()) {
    return op.emitOpError("indexed TAD shuffle array count is too large");
  }
  const size_t arrayCountSize = static_cast<size_t>(arrayCount);
  // The linalg shell owns one destination. Additional destinations are inert
  // inputs so differently-shaped TAD payloads do not impose one loop domain.
  if (inputs.size() != arrayCountSize * 2 || outputs.size() != 1) {
    return op.emitOpError("indexed TAD shuffle operand count mismatch");
  }

  mlir::Value shuffleMap = inputs[arrayCountSize];
  auto mapType = llvm::dyn_cast<mlir::MemRefType>(shuffleMap.getType());
  auto mapElement =
      mapType ? llvm::dyn_cast<mlir::IntegerType>(mapType.getElementType())
              : mlir::IntegerType{};
  if (!mapType || mapType.getRank() < 1 || !mapElement ||
      mapElement.getWidth() != 32) {
    return op.emitOpError("indexed TAD shuffle requires a signed i32 map");
  }

  mlir::SmallVector<mlir::Value> sources;
  mlir::SmallVector<mlir::Value> destinations;
  mlir::SmallVector<mlir::SmallVector<int64_t>> tadAxesByArray;
  mlir::SmallVector<mlir::SmallVector<int64_t>> itemAxesByArray;
  sources.reserve(static_cast<size_t>(arrayCount));
  destinations.reserve(static_cast<size_t>(arrayCount));
  tadAxesByArray.reserve(static_cast<size_t>(arrayCount));
  itemAxesByArray.reserve(static_cast<size_t>(arrayCount));

  llvm::ArrayRef<int64_t> rawTadAxes = tadDimensionsAttr.asArrayRef();
  for (int64_t array = 0; array < arrayCount; ++array) {
    const size_t arrayIndex = static_cast<size_t>(array);
    mlir::Value source = inputs[arrayIndex];
    mlir::Value destination =
        array == 0 ? outputs.front() : inputs[arrayCountSize + arrayIndex];
    auto sourceType = llvm::dyn_cast<mlir::MemRefType>(source.getType());
    auto destinationType =
        llvm::dyn_cast<mlir::MemRefType>(destination.getType());
    if (!sourceType || !destinationType || sourceType.getRank() < 1 ||
        destinationType.getRank() != sourceType.getRank() ||
        destinationType.getShape() != sourceType.getShape() ||
        destinationType.getElementType() != sourceType.getElementType()) {
      return op.emitOpError("indexed TAD shuffle payload contract mismatch");
    }

    mlir::SmallVector<int64_t> tadAxes;
    mlir::SmallVector<int64_t> itemAxes;
    if (!partitionAxes(sourceType.getRank(), rawTadAxes, tadAxes, itemAxes)) {
      return op.emitOpError("indexed TAD shuffle axes are invalid");
    }
    sources.push_back(source);
    destinations.push_back(destination);
    tadAxesByArray.push_back(std::move(tadAxes));
    itemAxesByArray.push_back(std::move(itemAxes));
  }

  emitSingleLaunch();
  mlir::Value zero = idxConst(rewriter, loc, 0);
  mlir::Value one = idxConst(rewriter, loc, 1);
  mlir::Value itemCount = productOfAxes(
      rewriter, loc, sources.front(), itemAxesByArray.front());

  rewriter.create<mlir::scf::ForOp>(
      loc, zero, itemCount, one, mlir::ValueRange{},
      [&](mlir::OpBuilder& builder, mlir::Location nestedLoc,
          mlir::Value sourceItem, mlir::ValueRange) {
        auto mapCoordinates =
            logicalIndices(builder, nestedLoc, sourceItem, shuffleMap);
        mlir::Value rawTarget = builder.create<mlir::memref::LoadOp>(
            nestedLoc, shuffleMap, mapCoordinates);
        mlir::Value mapZero = builder.create<mlir::arith::ConstantOp>(
            nestedLoc, mapElement, builder.getIntegerAttr(mapElement, 0));
        mlir::Value validTarget = builder.create<mlir::arith::CmpIOp>(
            nestedLoc, mlir::arith::CmpIPredicate::sge, rawTarget, mapZero);
        auto guarded = builder.create<mlir::scf::IfOp>(
            nestedLoc, mlir::TypeRange{}, validTarget, false);
        builder.setInsertionPointToStart(guarded.thenBlock());

        mlir::Value targetItem =
            builder.create<mlir::arith::IndexCastOp>(
                nestedLoc, builder.getIndexType(), rawTarget);
        for (int64_t array = 0; array < arrayCount; ++array) {
          const size_t arrayIndex = static_cast<size_t>(array);
          mlir::Value tadLength = productOfAxes(
              builder, nestedLoc, sources[arrayIndex],
              tadAxesByArray[arrayIndex]);
          builder.create<mlir::scf::ForOp>(
              nestedLoc, zero, tadLength, one, mlir::ValueRange{},
              [&](mlir::OpBuilder& innerBuilder, mlir::Location innerLoc,
                  mlir::Value tadElement, mlir::ValueRange) {
                auto sourceItemCoordinates = decompose(
                    innerBuilder, innerLoc, sourceItem, sources[arrayIndex],
                    itemAxesByArray[arrayIndex]);
                auto targetItemCoordinates = decompose(
                    innerBuilder, innerLoc, targetItem, sources[arrayIndex],
                    itemAxesByArray[arrayIndex]);
                auto tadCoordinates = decompose(
                    innerBuilder, innerLoc, tadElement, sources[arrayIndex],
                    tadAxesByArray[arrayIndex]);

                auto sourceType = llvm::cast<mlir::MemRefType>(
                    sources[arrayIndex].getType());
                mlir::SmallVector<mlir::Value> sourceCoordinates(
                    static_cast<size_t>(sourceType.getRank()));
                mlir::SmallVector<mlir::Value> targetCoordinates(
                    static_cast<size_t>(sourceType.getRank()));
                for (size_t axis = 0;
                     axis < itemAxesByArray[arrayIndex].size(); ++axis) {
                  const size_t coordinate = static_cast<size_t>(
                      itemAxesByArray[arrayIndex][axis]);
                  sourceCoordinates[coordinate] =
                      sourceItemCoordinates[axis];
                  targetCoordinates[coordinate] =
                      targetItemCoordinates[axis];
                }
                for (size_t axis = 0;
                     axis < tadAxesByArray[arrayIndex].size(); ++axis) {
                  const size_t coordinate = static_cast<size_t>(
                      tadAxesByArray[arrayIndex][axis]);
                  sourceCoordinates[coordinate] = tadCoordinates[axis];
                  targetCoordinates[coordinate] = tadCoordinates[axis];
                }

                // Load both values before either store. Sources and destinations
                // are exact aliases by the recorder contract, matching CUDA's
                // in-place disjoint-pair swap semantics.
                mlir::Value sourceValue =
                    innerBuilder.create<mlir::memref::LoadOp>(
                        innerLoc, sources[arrayIndex], sourceCoordinates);
                mlir::Value targetValue =
                    innerBuilder.create<mlir::memref::LoadOp>(
                        innerLoc, sources[arrayIndex], targetCoordinates);
                innerBuilder.create<mlir::memref::StoreOp>(
                    innerLoc, targetValue, destinations[arrayIndex],
                    sourceCoordinates);
                innerBuilder.create<mlir::memref::StoreOp>(
                    innerLoc, sourceValue, destinations[arrayIndex],
                    targetCoordinates);
                innerBuilder.create<mlir::scf::YieldOp>(innerLoc);
              });
        }
        // scf.if creates its zero-result terminator automatically.
        builder.setInsertionPointAfter(guarded);
        builder.create<mlir::scf::YieldOp>(nestedLoc);
      });

  rewriter.create<mlir::gpu::TerminatorOp>(loc);
  rewriter.eraseOp(op);
  return mlir::success();
}

// ─────────────────────────────────────────────────────────────────────────────
//  Wave 1: SoftmaxToSpirv
// ─────────────────────────────────────────────────────────────────────────────
//
// Row-wise softmax (last-dim reduce) using a three-pass serial algorithm:
//   Pass 1: max reduction per row (numerically stable shift)
//   Pass 2: exp-sum reduction per row
//   Pass 3: normalize each element
//
// Input shape: [rows, dim] (2-D; higher ranks flattened by the emitter).
// Pattern guard: match the operation descriptor hash.

mlir::LogicalResult SoftmaxToSpirv::matchAndRewrite(
    mlir::linalg::GenericOp op,
    mlir::PatternRewriter& rewriter) const {

  mlir::Location loc = op.getLoc();

  sd::LongType opHash = 0;
  if (!readOpHash(op, opHash)) return mlir::failure();
  const auto* emitter = findVulkanKernelEmitter(opHash);
  if (emitter == nullptr ||
      emitter->family != VulkanKernelFamily::NORMALIZATION ||
      emitter->loweringContract != VulkanLoweringContract::SOFTMAX) {
    return mlir::failure();
  }
  const bool logSoftmax =
      emitter->recipe == VulkanKernelRecipe::LOG_SOFTMAX;

  // ── 2. Extract operands ───────────────────────────────────────────────────
  mlir::ValueRange inputs  = op.getInputs();
  mlir::ValueRange outputs = op.getOutputs();
  auto typeContract = getComputeTypeContract(op, inputs, outputs);
  if (mlir::failed(typeContract)) return mlir::failure();
  mlir::FloatType elemTy = typeContract->accumulatorType;

  if (inputs.empty() || outputs.empty()) {
    return op.emitOpError("SoftmaxToSpirv: expected 1 input and 1 output");
  }

  mlir::Value X = inputs[0];
  mlir::Value Y = outputs[0];


  // ── 4. Validate shape: exactly 2-D ───────────────────────────────────────
  auto xType = llvm::dyn_cast<mlir::MemRefType>(X.getType());
  if (!xType || xType.getRank() != 2) {
    return op.emitOpError("SoftmaxToSpirv: expected 2-D input [rows, dim]");
  }

  mlir::Value zeroIdx = idxConst(rewriter, loc, 0);
  mlir::Value oneIdx  = idxConst(rewriter, loc, 1);
  mlir::Value numRows = rewriter.create<mlir::memref::DimOp>(loc, X, 0);
  mlir::Value numCols = rewriter.create<mlir::memref::DimOp>(loc, X, 1);

  // Large negative float for max init.
  mlir::Value negInf = floatConst(rewriter, loc, elemTy,
                 -std::numeric_limits<double>::infinity());

  // ── 5. One real Vulkan invocation per row ────────────────────────────────
  auto launch = createGpuLaunch(rewriter, loc, numRows, oneIdx, oneIdx);
  mlir::Value row = launch.getBlockIds().x;
  rewriter.setInsertionPointToEnd(&launch.getBody().front());

  auto maxLoop = emitReductionLoop(
      rewriter, loc, zeroIdx, numCols, oneIdx, negInf,
      [&](mlir::OpBuilder& kb, mlir::Location kloc,
          mlir::Value column, mlir::Value acc) -> mlir::Value {
        mlir::Value value = loadAsAccumulator(
            kb, kloc, X, mlir::SmallVector<mlir::Value>{row, column}, elemTy);
        return kb.create<mlir::arith::MaximumFOp>(kloc, acc, value);
      });
  mlir::Value rowMax = maxLoop.getResult(0);

  mlir::Value initZero = floatConst(rewriter, loc, elemTy, 0.0);
  auto sumLoop = emitReductionLoop(
      rewriter, loc, zeroIdx, numCols, oneIdx, initZero,
      [&](mlir::OpBuilder& kb, mlir::Location kloc,
          mlir::Value column, mlir::Value acc) -> mlir::Value {
        mlir::Value value = loadAsAccumulator(
            kb, kloc, X, mlir::SmallVector<mlir::Value>{row, column}, elemTy);
        mlir::Value shifted =
            kb.create<mlir::arith::SubFOp>(kloc, value, rowMax);
        mlir::Value exponential = emitExp(kb, kloc, elemTy, shifted);
        return kb.create<mlir::arith::AddFOp>(kloc, acc, exponential);
      });
  mlir::Value inverseSum;
  mlir::Value logSum;
  if (logSoftmax) {
    logSum = emitLog(rewriter, loc, elemTy, sumLoop.getResult(0));
  } else {
    inverseSum = rewriter.create<mlir::arith::DivFOp>(
        loc, floatConst(rewriter, loc, elemTy, 1.0), sumLoop.getResult(0));
  }

  rewriter.create<mlir::scf::ForOp>(
      loc, zeroIdx, numCols, oneIdx, mlir::ValueRange{},
      [&](mlir::OpBuilder& ab, mlir::Location aloc, mlir::Value column,
          mlir::ValueRange) {
        mlir::Value value = loadAsAccumulator(
            ab, aloc, X,
            mlir::SmallVector<mlir::Value>{row, column}, elemTy);
        mlir::Value shifted =
            ab.create<mlir::arith::SubFOp>(aloc, value, rowMax);
        mlir::Value output =
            logSoftmax
                ? mlir::Value(ab.create<mlir::arith::SubFOp>(
                      aloc, shifted, logSum))
                : mlir::Value(ab.create<mlir::arith::MulFOp>(
                      aloc, emitExp(ab, aloc, elemTy, shifted), inverseSum));
        storeFromAccumulator(
            ab, aloc, output, Y,
            mlir::SmallVector<mlir::Value>{row, column});
        ab.create<mlir::scf::YieldOp>(aloc);
      });
  rewriter.create<mlir::gpu::TerminatorOp>(loc);

  rewriter.eraseOp(op);
  return mlir::success();
}

// ─────────────────────────────────────────────────────────────────────────────
//  Wave 1: LayerNormToSpirv
// ─────────────────────────────────────────────────────────────────────────────
//
//   pass 1: mean = Σ x[m, k] / D
//   pass 2: variance = Σ (x[m, k] - mean)^2 / D
//   apply:  y[m, k] = (x[m, k] - mean) / sqrt(variance + epsilon)
//                      * gamma[k] + beta[k]   (if present)
//
// Pattern guard: match the operation descriptor hash.

mlir::LogicalResult LayerNormToSpirv::matchAndRewrite(
    mlir::linalg::GenericOp op,

    mlir::PatternRewriter& rewriter) const {

  mlir::Location loc = op.getLoc();

  sd::LongType opHash = 0;
  if (!readOpHash(op, opHash)) return mlir::failure();
  const auto* emitter = findVulkanKernelEmitter(opHash);
  if (emitter == nullptr ||
      emitter->family != VulkanKernelFamily::NORMALIZATION ||
      emitter->loweringContract != VulkanLoweringContract::LAYER_NORM) {
    return mlir::failure();
  }

  // ── 2. Extract operands (X, [gamma, beta] → Y) ───────────────────────────
  mlir::ValueRange inputs  = op.getInputs();
  mlir::ValueRange outputs = op.getOutputs();
  auto typeContract = getComputeTypeContract(op, inputs, outputs);
  if (mlir::failed(typeContract)) return mlir::failure();
  mlir::FloatType elemTy = typeContract->accumulatorType;

  if (inputs.empty() || outputs.empty()) {
    return op.emitOpError("LayerNormToSpirv: expected ≥1 input and 1 output");
  }

  mlir::Value X     = inputs[0];
  mlir::Value Y     = outputs[0];
  bool hasGamma = (inputs.size() >= 2 && inputs[1]);
  bool hasBeta  = (inputs.size() >= 3 && inputs[2]);
  mlir::Value gamma = hasGamma ? inputs[1] : mlir::Value{};
  mlir::Value beta  = hasBeta  ? inputs[2] : mlir::Value{};


  // ── 4. Validate shape: exactly 2-D ───────────────────────────────────────
  auto xType = llvm::dyn_cast<mlir::MemRefType>(X.getType());
  if (!xType || xType.getRank() != 2) {
    return op.emitOpError("LayerNormToSpirv: expected 2-D input [rows, hidden]");
  }

  // ── 5. Read epsilon ───────────────────────────────────────────────────────
  float epsilonVal = kDefaultEpsilon;
  if (auto epAttr = op->getAttrOfType<mlir::FloatAttr>(kEpsilonAttr)) {
    epsilonVal = static_cast<float>(epAttr.getValueAsDouble());
  }

  mlir::Value zeroIdx   = idxConst(rewriter, loc, 0);
  mlir::Value oneIdx    = idxConst(rewriter, loc, 1);
  mlir::Value numRows   = rewriter.create<mlir::memref::DimOp>(loc, X, 0);
  mlir::Value hiddenDim = rewriter.create<mlir::memref::DimOp>(loc, X, 1);

  mlir::Value hiddenFloat =
      convertIndexToFloat(rewriter, loc, hiddenDim, elemTy);
  mlir::Value epsilonConst = floatConst(rewriter, loc, elemTy, epsilonVal);

  // ── 6. One real Vulkan invocation per row ────────────────────────────────
  auto launch = createGpuLaunch(rewriter, loc, numRows, oneIdx, oneIdx);
  mlir::Value row = launch.getBlockIds().x;
  rewriter.setInsertionPointToEnd(&launch.getBody().front());

  mlir::Value initZero = floatConst(rewriter, loc, elemTy, 0.0);
  auto meanLoop = emitReductionLoop(
      rewriter, loc, zeroIdx, hiddenDim, oneIdx, initZero,
      [&](mlir::OpBuilder& kb, mlir::Location kloc,
          mlir::Value column, mlir::Value acc) -> mlir::Value {
        mlir::Value value = loadAsAccumulator(
            kb, kloc, X, mlir::SmallVector<mlir::Value>{row, column}, elemTy);
        return kb.create<mlir::arith::AddFOp>(kloc, acc, value);
      });
  mlir::Value mean = rewriter.create<mlir::arith::DivFOp>(
      loc, meanLoop.getResult(0), hiddenFloat);

  auto varianceLoop = emitReductionLoop(
      rewriter, loc, zeroIdx, hiddenDim, oneIdx, initZero,
      [&](mlir::OpBuilder& kb, mlir::Location kloc,
          mlir::Value column, mlir::Value acc) -> mlir::Value {
        mlir::Value value = loadAsAccumulator(
            kb, kloc, X, mlir::SmallVector<mlir::Value>{row, column}, elemTy);
        mlir::Value difference =
            kb.create<mlir::arith::SubFOp>(kloc, value, mean);
        mlir::Value square =
            kb.create<mlir::arith::MulFOp>(kloc, difference, difference);
        return kb.create<mlir::arith::AddFOp>(kloc, acc, square);
      });
  mlir::Value variance = rewriter.create<mlir::arith::DivFOp>(
      loc, varianceLoop.getResult(0), hiddenFloat);
  mlir::Value normFactor = emitRsqrt(
      rewriter, loc, elemTy,
      rewriter.create<mlir::arith::AddFOp>(loc, variance, epsilonConst));

  rewriter.create<mlir::scf::ForOp>(
      loc, zeroIdx, hiddenDim, oneIdx, mlir::ValueRange{},
      [&](mlir::OpBuilder& ab, mlir::Location aloc, mlir::Value column,
          mlir::ValueRange) {
        mlir::Value value = loadAsAccumulator(
            ab, aloc, X,
            mlir::SmallVector<mlir::Value>{row, column}, elemTy);
        mlir::Value normalized = ab.create<mlir::arith::MulFOp>(
            aloc, ab.create<mlir::arith::SubFOp>(aloc, value, mean),
            normFactor);
        if (hasGamma) {
          mlir::Value scale = loadAsAccumulator(
              ab, aloc, gamma, mlir::SmallVector<mlir::Value>{column}, elemTy);
          normalized =
              ab.create<mlir::arith::MulFOp>(aloc, normalized, scale);
        }
        if (hasBeta) {
          mlir::Value bias = loadAsAccumulator(
              ab, aloc, beta, mlir::SmallVector<mlir::Value>{column}, elemTy);
          normalized =
              ab.create<mlir::arith::AddFOp>(aloc, normalized, bias);
        }
        storeFromAccumulator(
            ab, aloc, normalized, Y,
            mlir::SmallVector<mlir::Value>{row, column});
        ab.create<mlir::scf::YieldOp>(aloc);
      });
  rewriter.create<mlir::gpu::TerminatorOp>(loc);

  rewriter.eraseOp(op);
  return mlir::success();
}

// ─────────────────────────────────────────────────────────────────────────────
//  Wave 2: GatherToSpirv
// ─────────────────────────────────────────────────────────────────────────────
//
// Gather: output[i, d] = table[indices[i], d]
//
// Constraints enforced by opIsRecordable:
//   - axis == 0 (only axis-0 gather is lowered here)
//   - table rank == 2, indices rank == 1
//
// We emit a flat double loop: for i in [0, I) for d in [0, D).
// The integer index is loaded from the indices buffer and cast to index type.

mlir::LogicalResult GatherToSpirv::matchAndRewrite(
    mlir::linalg::GenericOp op,
    mlir::PatternRewriter& rewriter) const {

  mlir::Location loc = op.getLoc();

  sd::LongType opHash = 0;
  if (!readOpHash(op, opHash)) return mlir::failure();
  const auto* emitter = findVulkanKernelEmitter(opHash);
  if (emitter == nullptr || !usesIndexedLookupSchedule(*emitter)) {
    return mlir::failure();
  }

  // Guard: axis must be 0
  auto axisAttr = op->getAttrOfType<mlir::IntegerAttr>(kAxisAttr);
  if (!axisAttr || axisAttr.getInt() != 0) {
    return mlir::failure();
  }

  mlir::ValueRange inputs  = op.getInputs();
  mlir::ValueRange outputs = op.getOutputs();
  if (mlir::failed(validateCopyTypes(op, inputs, outputs, true)))
    return mlir::failure();

  if (inputs.size() != 2 || outputs.size() != 1) {
    return op.emitOpError(
        "GatherToSpirv: expected 2 inputs (table, indices) and 1 output");
  }

  mlir::Value table = inputs[0];
  mlir::Value indices = inputs[1];
  mlir::Value out = outputs[0];
  auto tableType = llvm::dyn_cast<mlir::MemRefType>(table.getType());
  auto indicesType = llvm::dyn_cast<mlir::MemRefType>(indices.getType());
  auto outputType = llvm::dyn_cast<mlir::MemRefType>(out.getType());
  auto indexType = indicesType
                       ? llvm::dyn_cast<mlir::IntegerType>(
                             indicesType.getElementType())
                       : mlir::IntegerType{};
  auto unsignedAttr =
      op->getAttrOfType<mlir::BoolAttr>("nd4j.index_unsigned");
  if (!tableType || !indicesType || !outputType || !indexType ||
      indexType.getWidth() != 32 || indicesType.getRank() != 1 ||
      tableType.getRank() < 1 ||
      outputType.getRank() != tableType.getRank() || !unsignedAttr) {
    return op.emitOpError("gather rank/index metadata contract mismatch");
  }
  mlir::Value oneIdx = idxConst(rewriter, loc, 1);
  mlir::Value totalN = oneIdx;
  for (int64_t d = 0; d < outputType.getRank(); ++d) {
    totalN = rewriter.create<mlir::arith::MulIOp>(
        loc, totalN, rewriter.create<mlir::memref::DimOp>(loc, out, d));
  }
  auto launch = createGpuLaunch(rewriter, loc, totalN, oneIdx, oneIdx);
  mlir::Value linearIndex = launch.getBlockIds().x;
  rewriter.setInsertionPointToEnd(&launch.getBody().front());
  auto outputIndices = logicalIndices(rewriter, loc, linearIndex, out);
  mlir::Value outputRow = outputIndices.front();
  mlir::Value rawIndex = rewriter.create<mlir::memref::LoadOp>(
      loc, indices, mlir::SmallVector<mlir::Value>{outputRow});
  mlir::Value zeroIndexValue = rewriter.create<mlir::arith::ConstantIntOp>(
      loc, 0, indexType.getWidth());
  mlir::Value nonnegative = rawIndex;
  if (!unsignedAttr.getValue()) {
    nonnegative = rewriter.create<mlir::arith::SelectOp>(
        loc,
        rewriter.create<mlir::arith::CmpIOp>(
            loc, mlir::arith::CmpIPredicate::slt, rawIndex, zeroIndexValue),
        zeroIndexValue, rawIndex);
  }
  mlir::Value tableRows =
      rewriter.create<mlir::memref::DimOp>(loc, table, 0);
  mlir::Value lastTableRow = rewriter.create<mlir::arith::SubIOp>(
      loc, tableRows, oneIdx);
  mlir::Value lastTableRowValue = rewriter.create<mlir::arith::IndexCastOp>(
      loc, rawIndex.getType(), lastTableRow);
  mlir::Value bounded = rewriter.create<mlir::arith::SelectOp>(
      loc,
      rewriter.create<mlir::arith::CmpIOp>(
          loc,
          unsignedAttr.getValue() ? mlir::arith::CmpIPredicate::ugt
                                  : mlir::arith::CmpIPredicate::sgt,
          nonnegative,
          lastTableRowValue),
      lastTableRowValue, nonnegative);
  mlir::Value tableRow = rewriter.create<mlir::arith::IndexCastOp>(
      loc, rewriter.getIndexType(), bounded);
  mlir::SmallVector<mlir::Value> sourceIndices;
  sourceIndices.reserve(static_cast<size_t>(tableType.getRank()));
  sourceIndices.push_back(tableRow);
  sourceIndices.append(outputIndices.begin() + 1, outputIndices.end());
  mlir::Value value = rewriter.create<mlir::memref::LoadOp>(
      loc, table, sourceIndices);
  rewriter.create<mlir::memref::StoreOp>(
      loc, value, out, outputIndices);
  rewriter.create<mlir::gpu::TerminatorOp>(loc);

  rewriter.eraseOp(op);
  return mlir::success();
}

// ─────────────────────────────────────────────────────────────────────────────
//  Wave 2: ConcatToSpirv
// ─────────────────────────────────────────────────────────────────────────────
//
// Concat: output = cat([A0, A1, ..., AN-1], axis)
//
// We iterate over each input and copy its elements into the output, tracking
// the axis-dimension offset with a running counter.
//
// Uses one real Vulkan invocation containing a deterministic nested copy loop:
// outer = input index, inner = flat iteration.
// For axis K with ranks beyond 2, we do a full rank-N index decomposition.
// For simplicity (Wave 2 serial contract) we decompose flat index → ND indices.

mlir::LogicalResult ConcatToSpirv::matchAndRewrite(
    mlir::linalg::GenericOp op,
    mlir::PatternRewriter& rewriter) const {

  mlir::Location loc = op.getLoc();

  const auto* emitter = emitterForOperation(op);
  if (emitter == nullptr || !usesVariadicAxisConcatSchedule(*emitter)) {
    return mlir::failure();
  }

  auto axisAttr = op->getAttrOfType<mlir::IntegerAttr>(kAxisAttr);
  if (!axisAttr) {
    return mlir::failure();
  }
  int64_t axis = axisAttr.getInt();

  auto numInputsAttr = op->getAttrOfType<mlir::IntegerAttr>(kNumInputsAttr);
  if (!numInputsAttr) {
    return mlir::failure();
  }
  int64_t numInputs = numInputsAttr.getInt();

  mlir::ValueRange inputs  = op.getInputs();
  mlir::ValueRange outputs = op.getOutputs();
  if (mlir::failed(validateCopyTypes(op, inputs.take_front(numInputs), outputs)))
    return mlir::failure();

  if ((int64_t)inputs.size() < numInputs || outputs.empty()) {
    return op.emitOpError("ConcatToSpirv: input count mismatch");
  }

  mlir::Value outBuf = outputs[0];
  mlir::Value zeroIdx = idxConst(rewriter, loc, 0);
  mlir::Value oneIdx  = idxConst(rewriter, loc, 1);

  // Determine rank from first input
  auto firstType = llvm::dyn_cast<mlir::MemRefType>(inputs[0].getType());
  if (!firstType) {
    return op.emitOpError("ConcatToSpirv: input 0 is not a MemRef");
  }
  int64_t rank = firstType.getRank();
  if (axis < 0 || axis >= rank) {
    return op.emitOpError("ConcatToSpirv: axis out of range");
  }

  // Concat has a variable number of differently shaped input buffers.  A
  // single invocation keeps this as one compute pipeline and avoids atomics or
  // multiple entry points while all actual calculation remains device-side.
  auto launch = createGpuLaunch(rewriter, loc, oneIdx, oneIdx, oneIdx);
  rewriter.setInsertionPointToEnd(&launch.getBody().front());

  // Running output offset along the concat axis
  mlir::Value axisOffset = zeroIdx;

  for (int64_t inp = 0; inp < numInputs; ++inp) {
    mlir::Value inBuf = inputs[inp];
    auto inType = llvm::dyn_cast<mlir::MemRefType>(inBuf.getType());
    if (!inType) {
      return op.emitOpError("ConcatToSpirv: input is not a MemRef");
    }

    // Get dynamic dims of this input
    mlir::SmallVector<mlir::Value> dims;
    for (int64_t d = 0; d < rank; ++d) {
      dims.push_back(rewriter.create<mlir::memref::DimOp>(loc, inBuf, d));
    }

    // Build total element count for this input: N = d0 * d1 * ... * d(rank-1)
    mlir::Value totalN = oneIdx;
    for (int64_t d = 0; d < rank; ++d) {
      totalN = rewriter.create<mlir::arith::MulIOp>(loc, totalN, dims[d]);
    }

    // Flatten input and output to 1-D for serial iteration, then reconstruct ND indices.
    // Strides of input (row-major): stride[rank-1]=1, stride[d]=stride[d+1]*dim[d+1]
    mlir::SmallVector<mlir::Value> strides(rank);
    strides[rank - 1] = oneIdx;
    for (int64_t d = rank - 2; d >= 0; --d) {
      strides[d] = rewriter.create<mlir::arith::MulIOp>(loc, strides[d + 1], dims[d + 1]);
    }

    // Output dims along non-concat axes are same as input.
    // Along concat axis: use axisOffset to offset the output index.
    rewriter.create<mlir::scf::ForOp>(
        loc, zeroIdx, totalN, oneIdx,
        mlir::ValueRange{},
        [&](mlir::OpBuilder& fb, mlir::Location floc, mlir::Value fi, mlir::ValueRange) {
          // Decompose flat index fi into ND indices for the input
          mlir::SmallVector<mlir::Value> inIdx(rank);
          mlir::Value rem = fi;
          for (int64_t d = 0; d < rank; ++d) {
            inIdx[d] = fb.create<mlir::arith::DivUIOp>(floc, rem, strides[d]);
            rem = fb.create<mlir::arith::RemUIOp>(floc, rem, strides[d]);
          }

          // Build output indices: same as input indices but axis dimension is
          // shifted by axisOffset
          mlir::SmallVector<mlir::Value> outIdx(rank);
          for (int64_t d = 0; d < rank; ++d) {
            if (d == axis) {
              outIdx[d] = fb.create<mlir::arith::AddIOp>(floc, inIdx[d], axisOffset);
            } else {
              outIdx[d] = inIdx[d];
            }
          }

          mlir::Value val = fb.create<mlir::memref::LoadOp>(floc, inBuf,
                                mlir::SmallVector<mlir::Value>(inIdx.begin(), inIdx.end()));
          fb.create<mlir::memref::StoreOp>(floc, val, outBuf,
                                mlir::SmallVector<mlir::Value>(outIdx.begin(), outIdx.end()));
          fb.create<mlir::scf::YieldOp>(floc);
        });

    // Advance axisOffset by this input's axis-dim size
    axisOffset = rewriter.create<mlir::arith::AddIOp>(loc, axisOffset, dims[axis]);
  }

  rewriter.create<mlir::gpu::TerminatorOp>(loc);

  rewriter.eraseOp(op);
  return mlir::success();
}

// ─────────────────────────────────────────────────────────────────────────────
//  Wave 2: TransposeToSpirv
// ─────────────────────────────────────────────────────────────────────────────
//
// Transpose: output[perm(i)] = input[i]  for all index tuples i.
//
// The permutation is read from nd4j.permutation as a DenseI64ArrayAttr.
// One Vulkan invocation per logical input element with ND decomposition.

mlir::LogicalResult TransposeToSpirv::matchAndRewrite(
    mlir::linalg::GenericOp op,

    mlir::PatternRewriter& rewriter) const {

  mlir::Location loc = op.getLoc();

  const auto* emitter = emitterForOperation(op);
  if (emitter == nullptr || !usesRankPermutationSchedule(*emitter)) {
    return mlir::failure();
  }

  // Read permutation — stored as DenseI64ArrayAttr
  auto permAttr = op->getAttrOfType<mlir::DenseI64ArrayAttr>(kPermutationAttr);
  if (!permAttr) {
    return op.emitOpError("TransposeToSpirv: missing nd4j.permutation attribute");
  }
  llvm::ArrayRef<int64_t> perm = permAttr.asArrayRef();

  mlir::ValueRange inputs  = op.getInputs();
  mlir::ValueRange outputs = op.getOutputs();
  if (mlir::failed(validateCopyTypes(op, inputs, outputs)))
    return mlir::failure();

  if (inputs.empty() || outputs.empty()) {
    return op.emitOpError("TransposeToSpirv: expected 1 input and 1 output");
  }

  mlir::Value X = inputs[0];
  mlir::Value Y = outputs[0];

  auto xType = llvm::dyn_cast<mlir::MemRefType>(X.getType());
  if (!xType) {
    return op.emitOpError("TransposeToSpirv: input is not a MemRef");
  }
  int64_t rank = xType.getRank();
  if (rank < 2 || rank > 4 || (int64_t)perm.size() != rank) {
    return op.emitOpError("TransposeToSpirv: rank must be 2–4 and perm length must match rank");
  }

  mlir::Value oneIdx  = idxConst(rewriter, loc, 1);

  // Input dims
  mlir::SmallVector<mlir::Value> inDims(rank);
  for (int64_t d = 0; d < rank; ++d) {
    inDims[d] = rewriter.create<mlir::memref::DimOp>(loc, X, d);
  }

  // Input strides (row-major)
  mlir::SmallVector<mlir::Value> inStrides(rank);
  inStrides[rank - 1] = oneIdx;
  for (int64_t d = rank - 2; d >= 0; --d) {
    inStrides[d] = rewriter.create<mlir::arith::MulIOp>(loc, inStrides[d + 1], inDims[d + 1]);
  }

  // Output dims = input dims permuted
  mlir::SmallVector<mlir::Value> outDims(rank);
  for (int64_t d = 0; d < rank; ++d) {
    outDims[d] = inDims[perm[d]];
  }

  // Output strides (row-major over output dims)
  mlir::SmallVector<mlir::Value> outStrides(rank);
  outStrides[rank - 1] = oneIdx;
  for (int64_t d = rank - 2; d >= 0; --d) {
    outStrides[d] = rewriter.create<mlir::arith::MulIOp>(loc, outStrides[d + 1], outDims[d + 1]);
  }

  // Total element count (same for both shapes)
  mlir::Value totalN = oneIdx;
  for (int64_t d = 0; d < rank; ++d) {
    totalN = rewriter.create<mlir::arith::MulIOp>(loc, totalN, inDims[d]);
  }

  auto launch = createGpuLaunch(rewriter, loc, totalN, oneIdx, oneIdx);
  mlir::Value flatIndex = launch.getBlockIds().x;
  rewriter.setInsertionPointToEnd(&launch.getBody().front());

  mlir::SmallVector<mlir::Value> inIdx(rank);
  mlir::Value remainder = flatIndex;
  for (int64_t d = 0; d < rank; ++d) {
    inIdx[d] = rewriter.create<mlir::arith::DivUIOp>(
        loc, remainder, inStrides[d]);
    remainder = rewriter.create<mlir::arith::RemUIOp>(
        loc, remainder, inStrides[d]);
  }

  mlir::SmallVector<mlir::Value> outIdx(rank);
  for (int64_t d = 0; d < rank; ++d) {
    outIdx[d] = inIdx[perm[d]];
  }

  mlir::Value value = rewriter.create<mlir::memref::LoadOp>(
      loc, X, mlir::SmallVector<mlir::Value>(inIdx.begin(), inIdx.end()));
  rewriter.create<mlir::memref::StoreOp>(
      loc, value, Y,
      mlir::SmallVector<mlir::Value>(outIdx.begin(), outIdx.end()));
  rewriter.create<mlir::gpu::TerminatorOp>(loc);

  rewriter.eraseOp(op);
  return mlir::success();
}

// ─────────────────────────────────────────────────────────────────────────────
//  Static/replay-safe rank-N data movement
// ─────────────────────────────────────────────────────────────────────────────

static mlir::LogicalResult lowerContractMovement(
    mlir::linalg::GenericOp op, mlir::PatternRewriter& rewriter,
    const VulkanKernelEmitterInfo& emitter) {
  const mlir::Location loc = op.getLoc();
  const mlir::ValueRange inputs = op.getInputs();
  const mlir::ValueRange outputs = op.getOutputs();
  auto contractAttr =
      op->getAttrOfType<mlir::BoolAttr>("nd4j.contract_movement");
  if (!contractAttr || !contractAttr.getValue()) {
    return op.emitOpError("contract data movement metadata is missing");
  }

  auto memrefType = [&](mlir::Value value) {
    return llvm::dyn_cast<mlir::MemRefType>(value.getType());
  };
  auto elementCount = [&](mlir::Value value) {
    auto type = llvm::cast<mlir::MemRefType>(value.getType());
    mlir::Value count = idxConst(rewriter, loc, 1);
    for (int64_t d = 0; d < type.getRank(); ++d) {
      count = rewriter.create<mlir::arith::MulIOp>(
          loc, count,
          rewriter.create<mlir::memref::DimOp>(loc, value, d));
    }
    return count;
  };
  auto sameElementType = [&](mlir::Value lhs, mlir::Value rhs) {
    auto lhsType = memrefType(lhs);
    auto rhsType = memrefType(rhs);
    return lhsType && rhsType &&
           lhsType.getElementType() == rhsType.getElementType();
  };
  auto finishLaunch = [&]() {
    rewriter.create<mlir::gpu::TerminatorOp>(loc);
    rewriter.eraseOp(op);
    return mlir::success();
  };
  const mlir::Value one = idxConst(rewriter, loc, 1);

  if (emitter.loweringContract == VulkanLoweringContract::LINEAR_COPY) {
      if (usesSameShapeCopySchedule(emitter)) {
        auto payloadCountAttr =
            op->getAttrOfType<mlir::IntegerAttr>("nd4j.num_payload_inputs");
        auto destinationCountAttr =
            op->getAttrOfType<mlir::IntegerAttr>("nd4j.num_outputs");
        if (!payloadCountAttr || !destinationCountAttr ||
            payloadCountAttr.getInt() < 1 || destinationCountAttr.getInt() < 1 ||
            outputs.size() != 1 ||
            inputs.size() != static_cast<size_t>(
                                 payloadCountAttr.getInt() +
                                 destinationCountAttr.getInt() - 1)) {
          return op.emitOpError("same-shape copy metadata mismatch");
        }

        const size_t payloadCount =
            static_cast<size_t>(payloadCountAttr.getInt());
        const size_t destinationCount =
            static_cast<size_t>(destinationCountAttr.getInt());
        mlir::SmallVector<mlir::Value> sources;
        mlir::SmallVector<mlir::Value> destinations;
        destinations.reserve(destinationCount);
        destinations.push_back(outputs.front());
        for (size_t i = 1; i < destinationCount; ++i) {
          destinations.push_back(inputs[payloadCount + i - 1]);
        }
        if (usesTrailingPayloadCopySchedule(emitter)) {
          if (destinationCount != 1) {
            return op.emitOpError(
                "trailing-payload copy requires one destination");
          }
          sources.push_back(inputs[payloadCount - 1]);
        } else {
          if (payloadCount != destinationCount) {
            return op.emitOpError(
                "pairwise copy requires matching payload/destination counts");
          }
          sources.reserve(payloadCount);
          for (size_t i = 0; i < payloadCount; ++i) {
            sources.push_back(inputs[i]);
          }
        }
        for (size_t i = 0; i < destinations.size(); ++i) {
          auto sourceType = memrefType(sources[i]);
          auto destinationType = memrefType(destinations[i]);
          if (!sourceType || !destinationType ||
              sourceType.getRank() != destinationType.getRank() ||
              sourceType.getShape() != destinationType.getShape() ||
              sourceType.getElementType() !=
                  destinationType.getElementType()) {
            return op.emitOpError("same-shape copy operand mismatch");
          }
        }

        mlir::SmallVector<mlir::Value> starts;
        mlir::SmallVector<mlir::Value> ends;
        starts.reserve(destinations.size());
        ends.reserve(destinations.size());
        mlir::Value total = idxConst(rewriter, loc, 0);
        for (mlir::Value destination : destinations) {
          starts.push_back(total);
          total = rewriter.create<mlir::arith::AddIOp>(
              loc, total, elementCount(destination));
          ends.push_back(total);
        }

        auto launch = createGpuLaunch(rewriter, loc, total, one, one);
        mlir::Value flat = launch.getBlockIds().x;
        rewriter.setInsertionPointToEnd(&launch.getBody().front());
        auto emitPair = [&](size_t selected) {
          mlir::Value local = rewriter.create<mlir::arith::SubIOp>(
              loc, flat, starts[selected]);
          auto destinationIndices = logicalIndices(
              rewriter, loc, local, destinations[selected]);
          mlir::SmallVector<mlir::Value> sourceIndices(
              destinationIndices.begin(), destinationIndices.end());
          mlir::Value value = rewriter.create<mlir::memref::LoadOp>(
              loc, sources[selected], sourceIndices);
          rewriter.create<mlir::memref::StoreOp>(
              loc, value, destinations[selected], destinationIndices);
        };
        std::function<void(size_t)> selectPair =
            [&](size_t selected) {
          if (selected + 1 == destinations.size()) {
            emitPair(selected);
            return;
          }
          mlir::Value beforeEnd = rewriter.create<mlir::arith::CmpIOp>(
              loc, mlir::arith::CmpIPredicate::ult, flat, ends[selected]);
          auto branch = rewriter.create<mlir::scf::IfOp>(
              loc, mlir::TypeRange{}, beforeEnd, true);
          // Zero-result scf.if regions already own scf.yield terminators.
          // Insert each branch body at the block start so it remains before the
          // existing terminator.
          rewriter.setInsertionPointToStart(branch.thenBlock());
          emitPair(selected);
          rewriter.setInsertionPointToStart(branch.elseBlock());
          selectPair(selected + 1);
          rewriter.setInsertionPointAfter(branch);
        };
        selectPair(0);
        return finishLaunch();
      }

      const bool structuralShapeCopy =
          usesStructuralShapeCopySchedule(emitter);
      const bool reshapeCopy = usesReshapeCopySchedule(emitter);
      const bool validInputCount =
          structuralShapeCopy ? inputs.size() == 2
                              : (reshapeCopy ? (inputs.size() == 1 ||
                                                inputs.size() == 2)
                                             : inputs.size() == 1);
      if (!validInputCount || outputs.size() != 1 ||
          !sameElementType(inputs.front(), outputs.front())) {
        return op.emitOpError("linear copy operand contract mismatch");
      }
      // A structural shape operand remains in the function and descriptor ABI,
      // but output MemRef metadata completely determines the replay schedule.
      auto inputFortranAttr =
          op->getAttrOfType<mlir::BoolAttr>("nd4j.input_fortran_order");
      auto outputFortranAttr =
          op->getAttrOfType<mlir::BoolAttr>("nd4j.output_fortran_order");
      auto scalarExpandAttr =
          op->getAttrOfType<mlir::BoolAttr>("nd4j.scalar_expand");
      if (!inputFortranAttr || !outputFortranAttr || !scalarExpandAttr) {
        return op.emitOpError("linear copy traversal metadata is missing");
      }
      mlir::Value input = inputs.front();
      mlir::Value output = outputs.front();
      auto launch = createGpuLaunch(
          rewriter, loc, elementCount(output), one, one);
      mlir::Value flat = launch.getBlockIds().x;
      rewriter.setInsertionPointToEnd(&launch.getBody().front());
      auto outputIndices = logicalIndicesForOrder(
          rewriter, loc, flat, output, outputFortranAttr.getValue());
      mlir::SmallVector<mlir::Value> inputIndices;
      if (scalarExpandAttr.getValue()) {
        const int64_t rank = memrefType(input).getRank();
        inputIndices.assign(static_cast<size_t>(rank),
                            idxConst(rewriter, loc, 0));
      } else {
        inputIndices = logicalIndicesForOrder(
            rewriter, loc, flat, input, inputFortranAttr.getValue());
      }
      mlir::Value value = rewriter.create<mlir::memref::LoadOp>(
          loc, input, inputIndices);
      rewriter.create<mlir::memref::StoreOp>(
          loc, value, output, outputIndices);
      return finishLaunch();
    }

  if (usesLinearConcatSchedule(emitter)) {
      if (inputs.empty() || outputs.size() != 1) {
        return op.emitOpError("linear concat operand contract mismatch");
      }
      mlir::Value output = outputs.front();
      auto outputType = memrefType(output);
      auto inputFortranAttr =
          op->getAttrOfType<mlir::BoolAttr>("nd4j.input_fortran_order");
      if (!outputType || !inputFortranAttr) {
        return op.emitOpError("linear concat metadata is missing");
      }
      for (mlir::Value input : inputs) {
        if (!sameElementType(input, output)) {
          return op.emitOpError(
              "linear concat requires identical payload element types");
        }
      }

      // Follow CUDA flatten's proven schedule: each input owns a contiguous
      // output interval, while coordinates inside that interval follow the
      // requested C/F traversal and the input MemRef supplies physical strides.
      // Keep the loops inside one launch so this remains one replay dispatch.
      auto launch = createGpuLaunch(rewriter, loc, one, one, one);
      rewriter.setInsertionPointToEnd(&launch.getBody().front());
      mlir::Value outputOffset = idxConst(rewriter, loc, 0);
      for (mlir::Value input : inputs) {
        mlir::Value inputCount = elementCount(input);
        rewriter.create<mlir::scf::ForOp>(
            loc, idxConst(rewriter, loc, 0), inputCount, one,
            mlir::ValueRange{},
            [&](mlir::OpBuilder& fb, mlir::Location floc, mlir::Value local,
                mlir::ValueRange) {
              auto inputIndices = logicalIndicesForOrder(
                  fb, floc, local, input, inputFortranAttr.getValue());
              mlir::Value outputFlat = fb.create<mlir::arith::AddIOp>(
                  floc, outputOffset, local);
              auto outputIndices =
                  logicalIndices(fb, floc, outputFlat, output);
              mlir::Value value = fb.create<mlir::memref::LoadOp>(
                  floc, input, inputIndices);
              fb.create<mlir::memref::StoreOp>(
                  floc, value, output, outputIndices);
              fb.create<mlir::scf::YieldOp>(floc);
            });
        outputOffset = rewriter.create<mlir::arith::AddIOp>(
            loc, outputOffset, inputCount);
      }
      return finishLaunch();
    }

  if (usesOutputShapeBroadcastSchedule(emitter)) {
      if (inputs.size() != 2 || outputs.size() != 1 ||
          !sameElementType(inputs.front(), outputs.front())) {
        return op.emitOpError("output-shape broadcast operand mismatch");
      }
      // inputs[1] is the structural output-shape tensor. The descriptor's
      // frozen output MemRef is authoritative; the device kernel never loads it.
      mlir::Value data = inputs.front();
      mlir::Value output = outputs.front();
      auto dataType = memrefType(data);
      auto outputType = memrefType(output);
      if (!dataType || !outputType ||
          dataType.getRank() > outputType.getRank()) {
        return op.emitOpError("output-shape broadcast rank mismatch");
      }
      auto launch = createGpuLaunch(
          rewriter, loc, elementCount(output), one, one);
      mlir::Value flat = launch.getBlockIds().x;
      rewriter.setInsertionPointToEnd(&launch.getBody().front());
      auto outputIndices = logicalIndices(rewriter, loc, flat, output);
      auto inputIndices =
          broadcastIndices(rewriter, loc, outputIndices, data);
      mlir::Value value = rewriter.create<mlir::memref::LoadOp>(
          loc, data, inputIndices);
      rewriter.create<mlir::memref::StoreOp>(
          loc, value, output, outputIndices);
      return finishLaunch();
    }

  if (usesAxisPartitionSchedule(emitter)) {
      auto axisAttr = op->getAttrOfType<mlir::IntegerAttr>("nd4j.axis");
      auto countAttr =
          op->getAttrOfType<mlir::IntegerAttr>("nd4j.num_outputs");
      if (!axisAttr || !countAttr || outputs.size() != 1 ||
          countAttr.getInt() < 1 ||
          inputs.size() != static_cast<size_t>(countAttr.getInt() + 1)) {
        return op.emitOpError("axis partition metadata mismatch");
      }
      // Generic inputs are [payload, structural sizes, destination1, ...].
      // The structural tensor stays bound for ABI correctness but is never read.
      mlir::Value data = inputs[0];
      auto dataType = memrefType(data);
      const int64_t axis = axisAttr.getInt();
      if (!dataType || axis < 0 || axis >= dataType.getRank()) {
        return op.emitOpError("axis partition rank mismatch");
      }
      mlir::SmallVector<mlir::Value> destinations;
      destinations.reserve(static_cast<size_t>(countAttr.getInt()));
      destinations.push_back(outputs.front());
      for (size_t i = 2; i < inputs.size(); ++i) {
        destinations.push_back(inputs[i]);
      }
      for (mlir::Value destination : destinations) {
        auto destinationType = memrefType(destination);
        if (!destinationType ||
            destinationType.getRank() != dataType.getRank() ||
            destinationType.getElementType() != dataType.getElementType()) {
          return op.emitOpError("axis partition destination mismatch");
        }
      }

      auto launch = createGpuLaunch(
          rewriter, loc, elementCount(data), one, one);
      mlir::Value flat = launch.getBlockIds().x;
      rewriter.setInsertionPointToEnd(&launch.getBody().front());
      auto sourceIndices = logicalIndices(rewriter, loc, flat, data);
      mlir::Value value = rewriter.create<mlir::memref::LoadOp>(
          loc, data, sourceIndices);
      mlir::Value sourceAxis = sourceIndices[static_cast<size_t>(axis)];
      mlir::Value offset = idxConst(rewriter, loc, 0);
      for (mlir::Value destination : destinations) {
        mlir::Value width = rewriter.create<mlir::memref::DimOp>(
            loc, destination, axis);
        mlir::Value end = rewriter.create<mlir::arith::AddIOp>(
            loc, offset, width);
        mlir::Value atOrAfterStart =
            rewriter.create<mlir::arith::CmpIOp>(
                loc, mlir::arith::CmpIPredicate::uge, sourceAxis, offset);
        mlir::Value beforeEnd = rewriter.create<mlir::arith::CmpIOp>(
            loc, mlir::arith::CmpIPredicate::ult, sourceAxis, end);
        mlir::Value selected = rewriter.create<mlir::arith::AndIOp>(
            loc, atOrAfterStart, beforeEnd);
        auto guarded = rewriter.create<mlir::scf::IfOp>(
            loc, mlir::TypeRange{}, selected, false);
        rewriter.setInsertionPointToStart(guarded.thenBlock());
        auto destinationIndices = sourceIndices;
        destinationIndices[static_cast<size_t>(axis)] =
            rewriter.create<mlir::arith::SubIOp>(
                loc, sourceAxis, offset);
        rewriter.create<mlir::memref::StoreOp>(
            loc, value, destination, destinationIndices);
        rewriter.setInsertionPointAfter(guarded);
        offset = end;
      }
      return finishLaunch();
    }

  return mlir::failure();
}

mlir::LogicalResult DataMovementToSpirv::matchAndRewrite(
    mlir::linalg::GenericOp op,
    mlir::PatternRewriter& rewriter) const {
  mlir::Location loc = op.getLoc();
  const auto* emitter = emitterForOperation(op);
  if (emitter == nullptr) return mlir::failure();
  if (usesContractMovementSchedule(*emitter)) {
    return lowerContractMovement(op, rewriter, *emitter);
  }
  if (emitter->family != VulkanKernelFamily::DATA_MOVEMENT &&
      emitter->family != VulkanKernelFamily::CONSTANT_GENERATION) {
    return mlir::failure();
  }
  const bool constantFill =
      emitter->family == VulkanKernelFamily::CONSTANT_GENERATION;
  if (!constantFill &&
      (usesStructuredComputeSchedule(*emitter) ||
       usesIndexedAccumulationSchedule(*emitter) ||
       usesIndexedTadMovementSchedule(*emitter) ||
       usesIndexedLookupSchedule(*emitter) ||
       usesVariadicAxisConcatSchedule(*emitter) ||
       usesRankPermutationSchedule(*emitter))) {
    return mlir::failure();
  }

  mlir::ValueRange inputs = op.getInputs();
  mlir::ValueRange outputs = op.getOutputs();
  const bool hasRuntimeState = hasVulkanEmitterTrait(
      *emitter, VULKAN_EMITTER_TRAIT_RANDOM_STATE);
  const int semanticInputCount =
      static_cast<int>(inputs.size()) - (hasRuntimeState ? 1 : 0);
  const bool multiOutput =
      usesMultiOutputPartitionSchedule(*emitter);
  if (semanticInputCount < 0 ||
      !vulkanArgumentContractAcceptsTensorCounts(
          *emitter, semanticInputCount, static_cast<int>(outputs.size())) ||
      outputs.empty() || (!constantFill && semanticInputCount == 0) ||
      (!multiOutput && outputs.size() != 1)) {
    return op.emitOpError(
        "data movement lowering has an invalid input/output contract");
  }
  if (constantFill) {
    const bool zeroInput = hasVulkanEmitterTrait(
        *emitter, VULKAN_EMITTER_TRAIT_ARGUMENT_GENERATED);
    const bool genericRandom =
        hasRuntimeState &&
        emitter->recipe == VulkanKernelRecipe::RANDOM_GENERIC;
    const size_t expectedFunctionInputs =
        hasRuntimeState
            ? (genericRandom ? static_cast<size_t>(semanticInputCount + 1) : 1)
            : (zeroInput ? 0 : 1);
    if (inputs.size() != expectedFunctionInputs ||
        (hasRuntimeState && zeroInput && !genericRandom)) {
      return op.emitOpError(
          "constant generation input count mismatch");
    }
    auto outputType = llvm::dyn_cast<mlir::MemRefType>(outputs.front().getType());
    auto inputType =
        inputs.empty()
            ? mlir::MemRefType{}
            : llvm::dyn_cast<mlir::MemRefType>(inputs.front().getType());
    if (!outputType || (!inputs.empty() && !inputType)) {
      return op.emitOpError("constant generation operands must be MemRefs");
    }
    if (hasRuntimeState) {
      auto stateElement =
          llvm::dyn_cast<mlir::IntegerType>(inputType.getElementType());
      if (inputType.getRank() != 1 ||
          inputType.getDimSize(0) !=
              static_cast<int64_t>(kVulkanRandomStateWordCount) ||
          !stateElement || stateElement.getWidth() != 32) {
        return op.emitOpError(
            "runtime state must be a four-word i32 MemRef");
      }
    }
    mlir::Value output = outputs.front();
    mlir::Value one = idxConst(rewriter, loc, 1);
    auto launchOutput = [&]() {
      mlir::Value total = one;
      for (int64_t d = 0; d < outputType.getRank(); ++d) {
        total = rewriter.create<mlir::arith::MulIOp>(
            loc, total,
            rewriter.create<mlir::memref::DimOp>(loc, output, d));
      }
      auto launch = createGpuLaunch(rewriter, loc, total, one, one);
      mlir::Value flat = launch.getBlockIds().x;
      rewriter.setInsertionPointToEnd(&launch.getBody().front());
      return std::make_pair(
          flat, logicalIndices(rewriter, loc, flat, output));
    };

    if (emitter->recipe == VulkanKernelRecipe::EYE) {
      auto eyeAttr = op->getAttrOfType<mlir::BoolAttr>("nd4j.eye");
      auto outputUnsignedAttr =
          op->getAttrOfType<mlir::BoolAttr>("nd4j.output_unsigned");
      auto outputElement =
          llvm::dyn_cast<mlir::FloatType>(outputType.getElementType());
      if (!eyeAttr || !eyeAttr.getValue() || !outputUnsignedAttr ||
          !outputElement || outputType.getRank() < 2) {
        return op.emitOpError("eye metadata contract mismatch");
      }
      auto [flat, outputIndices] = launchOutput();
      (void)flat;
      const size_t rank = outputIndices.size();
      mlir::Value diagonal = rewriter.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::eq,
          outputIndices[rank - 2], outputIndices[rank - 1]);
      mlir::Value zeroValue =
          scalarConstant(rewriter, loc, outputElement, 0.0);
      mlir::Value oneValue =
          scalarConstant(rewriter, loc, outputElement, 1.0);
      mlir::Value value = rewriter.create<mlir::arith::SelectOp>(
          loc, diagonal, oneValue, zeroValue);
      rewriter.create<mlir::memref::StoreOp>(
          loc, value, output, outputIndices);
      rewriter.create<mlir::gpu::TerminatorOp>(loc);
      rewriter.eraseOp(op);
      return mlir::success();
    }

    if (emitter->recipe == VulkanKernelRecipe::MIN_MAX_DATATYPE) {
      auto valueAttr =
          op->getAttrOfType<mlir::FloatAttr>("nd4j.constant_value");
      auto outputUnsignedAttr =
          op->getAttrOfType<mlir::BoolAttr>("nd4j.output_unsigned");
      auto outputFloat =
          llvm::dyn_cast<mlir::FloatType>(outputType.getElementType());
      auto outputInteger =
          llvm::dyn_cast<mlir::IntegerType>(outputType.getElementType());
      if (!valueAttr || !outputUnsignedAttr || outputType.getRank() != 0 ||
          (!outputFloat &&
           (!outputInteger || outputInteger.getWidth() != 32))) {
        return op.emitOpError("min_max_datatype metadata contract mismatch");
      }
      auto [flat, outputIndices] = launchOutput();
      (void)flat;
      mlir::Value value = scalarConstant(
          rewriter, loc, outputType.getElementType(),
          valueAttr.getValueAsDouble());
      if (!value) {
        return op.emitOpError("min_max_datatype constant conversion failed");
      }
      rewriter.create<mlir::memref::StoreOp>(
          loc, value, output, outputIndices);
      rewriter.create<mlir::gpu::TerminatorOp>(loc);
      rewriter.eraseOp(op);
      return mlir::success();
    }

    if (emitter->recipe == VulkanKernelRecipe::UNIFORM_RANDOM ||
        emitter->recipe == VulkanKernelRecipe::RANDOM_GENERIC) {
      const bool genericRandom =
          emitter->recipe == VulkanKernelRecipe::RANDOM_GENERIC;
      auto accumulatorAttr =
          op->getAttrOfType<mlir::TypeAttr>(kAccumulatorTypeAttr);
      auto accumulator =
          accumulatorAttr
              ? llvm::dyn_cast<mlir::FloatType>(accumulatorAttr.getValue())
              : mlir::FloatType{};
      auto fromAttr =
          op->getAttrOfType<mlir::FloatAttr>("nd4j.random_from");
      auto toAttr =
          op->getAttrOfType<mlir::FloatAttr>("nd4j.random_to");
      auto outputFloat =
          llvm::dyn_cast<mlir::FloatType>(outputType.getElementType());
      if (!hasRuntimeState || inputs.empty() || !accumulator ||
          !outputFloat ||
          (!genericRandom &&
           (!fromAttr || !toAttr ||
            fromAttr.getType() != accumulator ||
            toAttr.getType() != accumulator))) {
        return op.emitOpError("random metadata contract mismatch");
      }

      auto launchResult = launchOutput();
      mlir::Value flat = launchResult.first;
      auto outputIndices = std::move(launchResult.second);
      auto i32 = rewriter.getI32Type();
      auto i32Constant = [&](uint32_t bits) -> mlir::Value {
        auto value = static_cast<int64_t>(static_cast<int32_t>(bits));
        return rewriter.create<mlir::arith::ConstantOp>(
            loc, i32, rewriter.getIntegerAttr(i32, value));
      };

      mlir::Value index = rewriter.create<mlir::arith::IndexCastUIOp>(
          loc, i32, flat);
      mlir::Value indexPlusTwo = rewriter.create<mlir::arith::AddIOp>(
          loc, index, i32Constant(2u));
      mlir::SmallVector<mlir::Value> rootIndex{
          idxConst(rewriter, loc,
                   static_cast<int64_t>(kVulkanRandomRootLowWord))};
      mlir::SmallVector<mlir::Value> nodeIndex{
          idxConst(rewriter, loc,
                   static_cast<int64_t>(kVulkanRandomNodeLowWord))};
      mlir::Value root = rewriter.create<mlir::memref::LoadOp>(
          loc, inputs.front(), rootIndex);
      mlir::Value node = rewriter.create<mlir::memref::LoadOp>(
          loc, inputs.front(), nodeIndex);
      mlir::Value s0 = rewriter.create<mlir::arith::XOrIOp>(
          loc, root,
          rewriter.create<mlir::arith::MulIOp>(
              loc, indexPlusTwo,
              rewriter.create<mlir::arith::AddIOp>(
                  loc, node, i32Constant(24243287u))));
      mlir::Value s1 = rewriter.create<mlir::arith::XOrIOp>(
          loc, node,
          rewriter.create<mlir::arith::MulIOp>(
              loc, indexPlusTwo,
              rewriter.create<mlir::arith::AddIOp>(
                  loc, s0, i32Constant(723829u))));
      mlir::Value product = rewriter.create<mlir::arith::MulIOp>(
          loc,
          rewriter.create<mlir::arith::XOrIOp>(loc, s1, s0),
          i32Constant(0x9E3779BBu));
      mlir::Value rotated = rewriter.create<mlir::arith::OrIOp>(
          loc,
          rewriter.create<mlir::arith::ShLIOp>(
              loc, product, i32Constant(5u)),
          rewriter.create<mlir::arith::ShRUIOp>(
              loc, product, i32Constant(27u)));
      mlir::Value randomBits = rewriter.create<mlir::arith::MulIOp>(
          loc, rotated, i32Constant(5u));
      mlir::Value unitBits = rewriter.create<mlir::arith::OrIOp>(
          loc, i32Constant(0x3f800000u),
          rewriter.create<mlir::arith::ShRUIOp>(
              loc, randomBits, i32Constant(9u)));
      auto f32 = rewriter.getF32Type();
      mlir::Value unit = rewriter.create<mlir::arith::SubFOp>(
          loc,
          rewriter.create<mlir::arith::BitcastOp>(loc, f32, unitBits),
          scalarConstant(rewriter, loc, f32, 1.0));
      mlir::Value unitAccumulator =
          convertFloat(rewriter, loc, unit, accumulator);
      if (genericRandom) {
        auto randomArgument = [&](int ordinal, double fallback) {
          auto attr = op->getAttrOfType<mlir::FloatAttr>(
              "nd4j.random_arg" + std::to_string(ordinal));
          return attr && attr.getType() == accumulator
                     ? mlir::Value(rewriter.create<mlir::arith::ConstantOp>(
                           loc, accumulator, attr))
                     : floatConst(rewriter, loc, accumulator, fallback);
        };
        auto loadRandomInput = [&](int ordinal) {
          if (inputs.size() <= static_cast<size_t>(ordinal + 1)) {
            return unitAccumulator;
          }
          mlir::Value source = inputs[static_cast<size_t>(ordinal + 1)];
          auto sourceType =
              llvm::cast<mlir::MemRefType>(source.getType());
          mlir::SmallVector<mlir::Value> sourceIndices;
          bool sameShape = sourceType.getRank() == outputType.getRank();
          for (int64_t d = 0; sameShape && d < outputType.getRank(); ++d) {
            if (sourceType.isDynamicDim(d) || outputType.isDynamicDim(d)) {
              continue;
            }
            sameShape = sourceType.getDimSize(d) == outputType.getDimSize(d);
          }
          if (sameShape) {
            sourceIndices = logicalIndices(rewriter, loc, flat, source);
          } else {
            sourceIndices.assign(static_cast<size_t>(sourceType.getRank()),
                                 idxConst(rewriter, loc, 0));
          }
          return loadAsAccumulator(rewriter, loc, source, sourceIndices,
                                   accumulator);
        };
        auto normalValue = [&]() {
          auto safeUnit = rewriter.create<mlir::arith::MaximumFOp>(
              loc, unitAccumulator,
              floatConst(rewriter, loc, accumulator, 1.0e-5));
          auto radius = rewriter.create<mlir::math::SqrtOp>(
              loc,
              rewriter.create<mlir::arith::MulFOp>(
                  loc, floatConst(rewriter, loc, accumulator, -2.0),
                  emitLog(rewriter, loc, accumulator, safeUnit)));
          auto angle = rewriter.create<mlir::arith::MulFOp>(
              loc, floatConst(rewriter, loc, accumulator, 6.283185307179586),
              unitAccumulator);
          return rewriter.create<mlir::arith::MulFOp>(
              loc, radius,
              rewriter.create<mlir::math::CosOp>(loc, angle));
        };
        auto opNumberAttr = op->getAttrOfType<mlir::IntegerAttr>(
            "nd4j.legacy_op_num");
        const int opNumber =
            opNumberAttr ? static_cast<int>(opNumberAttr.getInt()) : -1;
        mlir::Value value = unitAccumulator;
        switch (opNumber) {
          case 1: {  // DropOut: retain with probability p.
            auto inputValue = loadRandomInput(0);
            auto probability = randomArgument(0, 1.0);
            auto keep = rewriter.create<mlir::arith::CmpFOp>(
                loc, mlir::arith::CmpFPredicate::OLT, unitAccumulator,
                probability);
            value = rewriter.create<mlir::arith::MulFOp>(
                loc, inputValue,
                rewriter.create<mlir::arith::SelectOp>(
                    loc, keep, floatConst(rewriter, loc, accumulator, 1.0),
                    floatConst(rewriter, loc, accumulator, 0.0)));
            break;
          }
          case 2: {  // Inverted dropout.
            auto inputValue = loadRandomInput(0);
            auto probability = randomArgument(0, 1.0);
            auto keep = rewriter.create<mlir::arith::CmpFOp>(
                loc, mlir::arith::CmpFPredicate::OLT, unitAccumulator,
                probability);
            auto scale = rewriter.create<mlir::arith::DivFOp>(
                loc, rewriter.create<mlir::arith::SelectOp>(
                         loc, keep,
                         floatConst(rewriter, loc, accumulator, 1.0),
                         floatConst(rewriter, loc, accumulator, 0.0)),
                rewriter.create<mlir::arith::MaximumFOp>(
                    loc, probability,
                    floatConst(rewriter, loc, accumulator, 1.0e-5)));
            value = rewriter.create<mlir::arith::MulFOp>(loc, inputValue, scale);
            break;
          }
          case 3: {  // Probabilistic merge of two input tensors.
            auto first = loadRandomInput(0);
            auto second = loadRandomInput(1);
            auto chooseFirst = rewriter.create<mlir::arith::CmpFOp>(
                loc, mlir::arith::CmpFPredicate::OLT, unitAccumulator,
                randomArgument(0, 0.5));
            value = rewriter.create<mlir::arith::SelectOp>(
                loc, chooseFirst, first, second);
            break;
          }
          case 4: {  // Linspace.
            auto start = loadRandomInput(0);
            auto finish = loadRandomInput(1);
            auto count = loadRandomInput(2);
            auto total = one;
            for (int64_t d = 0; d < outputType.getRank(); ++d) {
              total = rewriter.create<mlir::arith::MulIOp>(
                  loc, total, rewriter.create<mlir::memref::DimOp>(
                                  loc, output, d));
            }
            auto position = convertIndexToFloat(
                rewriter, loc, flat, accumulator);
            auto denominator = convertIndexToFloat(
                rewriter, loc,
                rewriter.create<mlir::arith::SubIOp>(
                    loc, total, idxConst(rewriter, loc, 1)), accumulator);
            auto fraction = rewriter.create<mlir::arith::SelectOp>(
                loc,
                rewriter.create<mlir::arith::CmpFOp>(
                    loc, mlir::arith::CmpFPredicate::OEQ, count,
                    floatConst(rewriter, loc, accumulator, 0.0)),
                rewriter.create<mlir::arith::DivFOp>(
                    loc, position,
                    rewriter.create<mlir::arith::MaximumFOp>(
                        loc, denominator,
                        floatConst(rewriter, loc, accumulator, 1.0))),
                count);
            value = rewriter.create<mlir::arith::AddFOp>(
                loc, start,
                rewriter.create<mlir::arith::MulFOp>(
                    loc, fraction,
                    rewriter.create<mlir::arith::SubFOp>(loc, finish, start)));
            break;
          }
          case 5: {  // Choice: categorical sample from source/probability vectors.
            if (inputs.size() < 3) {
              return op.emitOpError("choice requires source and probability inputs");
            }
            auto source = inputs[1];
            auto probabilities = inputs[2];
            auto sourceType = llvm::dyn_cast<mlir::MemRefType>(source.getType());
            auto probabilityType =
                llvm::dyn_cast<mlir::MemRefType>(probabilities.getType());
            if (!sourceType || !probabilityType || sourceType.getRank() != 1 ||
                probabilityType.getRank() != 1) {
              return op.emitOpError("choice requires rank-1 source/probability inputs");
            }
            auto sourceLength = rewriter.create<mlir::memref::DimOp>(loc, source, 0);
            auto probabilityLength =
                rewriter.create<mlir::memref::DimOp>(loc, probabilities, 0);
            auto sameLength = rewriter.create<mlir::arith::CmpIOp>(
                loc, mlir::arith::CmpIPredicate::eq, sourceLength,
                probabilityLength);
            auto guarded = rewriter.create<mlir::scf::IfOp>(
                loc, mlir::TypeRange{accumulator}, sameLength, true);
            rewriter.setInsertionPointToStart(guarded.thenBlock());
            auto zero = floatConst(rewriter, loc, accumulator, 0.0);
            auto sourceZero = mlir::SmallVector<mlir::Value>{idxConst(rewriter, loc, 0)};
            auto initialValue = loadAsAccumulator(
                rewriter, loc, source, sourceZero, accumulator);
            if (!initialValue) {
              return op.emitOpError("choice source type cannot be converted");
            }
            auto found = rewriter.create<mlir::arith::ConstantOp>(
                loc, rewriter.getI1Type(), rewriter.getBoolAttr(false));
            auto loop = rewriter.create<mlir::scf::ForOp>(
                loc, idxConst(rewriter, loc, 0), sourceLength,
                idxConst(rewriter, loc, 1),
                mlir::ValueRange{initialValue, zero, found},
                [&](mlir::OpBuilder& nested, mlir::Location nestedLoc,
                    mlir::Value item, mlir::ValueRange iterArgs) {
                  auto itemIndices = mlir::SmallVector<mlir::Value>{item};
                  auto probability = loadAsAccumulator(
                      nested, nestedLoc, probabilities, itemIndices, accumulator);
                  auto sourceValue = loadAsAccumulator(
                      nested, nestedLoc, source, itemIndices, accumulator);
                  auto cumulative = nested.create<mlir::arith::AddFOp>(
                      nestedLoc, iterArgs[1], probability);
                  auto below = nested.create<mlir::arith::CmpFOp>(
                      nestedLoc, mlir::arith::CmpFPredicate::OLT,
                      unitAccumulator, cumulative);
                  auto notFound = nested.create<mlir::arith::XOrIOp>(
                      nestedLoc, iterArgs[2],
                      nested.create<mlir::arith::ConstantOp>(
                          nestedLoc, nested.getI1Type(),
                          nested.getBoolAttr(true)));
                  auto take = nested.create<mlir::arith::AndIOp>(
                      nestedLoc, below, notFound);
                  auto selected = nested.create<mlir::arith::SelectOp>(
                      nestedLoc, take, sourceValue, iterArgs[0]);
                  auto selectedFound = nested.create<mlir::arith::OrIOp>(
                      nestedLoc, iterArgs[2], take);
                  nested.create<mlir::scf::YieldOp>(
                      nestedLoc, mlir::ValueRange{selected, cumulative, selectedFound});
                });
            rewriter.setInsertionPointToEnd(guarded.thenBlock());
            auto lastIndex = rewriter.create<mlir::arith::SubIOp>(
                loc, sourceLength, idxConst(rewriter, loc, 1));
            auto lastValue = loadAsAccumulator(
                rewriter, loc, source,
                mlir::SmallVector<mlir::Value>{lastIndex}, accumulator);
            auto chosen = rewriter.create<mlir::arith::SelectOp>(
                loc, loop.getResult(2), loop.getResult(0), lastValue);
            rewriter.create<mlir::scf::YieldOp>(loc, chosen.getResult());
            rewriter.setInsertionPointToStart(guarded.elseBlock());
            rewriter.create<mlir::scf::YieldOp>(loc, unitAccumulator);
            rewriter.setInsertionPointAfter(guarded);
            value = guarded.getResult(0);
            break;
          }
          case 6: {  // Gaussian.
            value = rewriter.create<mlir::arith::AddFOp>(
                loc, randomArgument(0, 0.0),
                rewriter.create<mlir::arith::MulFOp>(
                    loc, randomArgument(1, 1.0), normalValue()));
            break;
          }
          case 7: {  // Bernoulli.
            auto probability = randomArgument(0, 0.5);
            auto keep = rewriter.create<mlir::arith::CmpFOp>(
                loc, mlir::arith::CmpFPredicate::OLT, unitAccumulator,
                probability);
            value = rewriter.create<mlir::arith::SelectOp>(
                loc, keep, floatConst(rewriter, loc, accumulator, 1.0),
                floatConst(rewriter, loc, accumulator, 0.0));
            break;
          }
          case 8:
          case 9: {  // Binomial and BinomialEx.
            auto trials = randomArgument(0, 1.0);
            auto probability = randomArgument(1, 0.5);
            value = rewriter.create<mlir::math::FloorOp>(
                loc, rewriter.create<mlir::arith::MulFOp>(
                         loc, trials,
                         rewriter.create<mlir::arith::AddFOp>(
                             loc, probability,
                             rewriter.create<mlir::arith::MulFOp>(
                                 loc, unitAccumulator,
                                 rewriter.create<mlir::arith::SubFOp>(
                                     loc, floatConst(rewriter, loc, accumulator, 1.0),
                                     probability)))));
            break;
          }
          case 10: {  // Log-normal.
            value = emitExp(
                rewriter, loc, accumulator,
                rewriter.create<mlir::arith::AddFOp>(
                    loc, randomArgument(0, 0.0),
                    rewriter.create<mlir::arith::MulFOp>(
                        loc, randomArgument(1, 1.0), normalValue())));
            break;
          }
          case 11: {  // Truncated normal.
            auto normal = normalValue();
            value = rewriter.create<mlir::arith::AddFOp>(
                loc, randomArgument(0, 0.0),
                rewriter.create<mlir::arith::MulFOp>(
                    loc, randomArgument(1, 1.0),
                    rewriter.create<mlir::arith::MinimumFOp>(
                        loc,
                        rewriter.create<mlir::arith::MaximumFOp>(
                            loc, normal,
                            floatConst(rewriter, loc, accumulator, -2.0)),
                        floatConst(rewriter, loc, accumulator, 2.0))));
            break;
          }
          case 12: {  // Alpha dropout.
            auto inputValue = loadRandomInput(0);
            auto probability = randomArgument(0, 1.0);
            auto keep = rewriter.create<mlir::arith::CmpFOp>(
                loc, mlir::arith::CmpFPredicate::OLT, unitAccumulator,
                probability);
            auto retained = rewriter.create<mlir::arith::MulFOp>(
                loc, randomArgument(1, 1.0), inputValue);
            auto dropped = rewriter.create<mlir::arith::AddFOp>(
                loc,
                rewriter.create<mlir::arith::MulFOp>(
                    loc, randomArgument(1, 1.0), randomArgument(3, 0.0)),
                randomArgument(2, 0.0));
            value = rewriter.create<mlir::arith::SelectOp>(
                loc, keep, retained, dropped);
            break;
          }
          case 13:
          case 14: {  // Exponential and inverse exponential.
            auto lambda = rewriter.create<mlir::arith::MaximumFOp>(
                loc, randomArgument(0, 1.0),
                floatConst(rewriter, loc, accumulator, 1.0e-5));
            value = rewriter.create<mlir::arith::DivFOp>(
                loc,
                rewriter.create<mlir::arith::NegFOp>(
                    loc, emitLog(rewriter, loc, accumulator,
                                 rewriter.create<mlir::arith::SubFOp>(
                                     loc, floatConst(rewriter, loc, accumulator, 1.0),
                                     unitAccumulator))),
                lambda);
            break;
          }
          case 15: {  // Poisson approximation.
            auto lambda = randomArgument(0, 1.0);
            value = rewriter.create<mlir::math::FloorOp>(
                loc, rewriter.create<mlir::arith::MaximumFOp>(
                         loc, floatConst(rewriter, loc, accumulator, 0.0),
                         rewriter.create<mlir::arith::AddFOp>(
                             loc, lambda,
                             rewriter.create<mlir::arith::MulFOp>(
                                 loc, rewriter.create<mlir::math::SqrtOp>(
                                          loc, lambda),
                                 normalValue()))));
            break;
          }
          case 16: {  // Gamma approximation.
            auto shape = rewriter.create<mlir::arith::MaximumFOp>(
                loc, randomArgument(0, 1.0),
                floatConst(rewriter, loc, accumulator, 1.0e-5));
            auto scale = randomArgument(1, 1.0);
            value = emitExp(
                rewriter, loc, accumulator,
                rewriter.create<mlir::arith::AddFOp>(
                    loc, rewriter.create<mlir::math::LogOp>(
                             loc, rewriter.create<mlir::arith::MulFOp>(
                                      loc, shape, scale)),
                    rewriter.create<mlir::arith::MulFOp>(
                        loc,
                        rewriter.create<mlir::arith::DivFOp>(
                            loc, normalValue(),
                            rewriter.create<mlir::math::SqrtOp>(
                                loc, shape)),
                        floatConst(rewriter, loc, accumulator, 0.5))));
            break;
          }
          default:
            break;
        }
        storeFromAccumulator(rewriter, loc, value, output, outputIndices);
        rewriter.create<mlir::gpu::TerminatorOp>(loc);
        rewriter.eraseOp(op);
        return mlir::success();
      }
      mlir::Value from = rewriter.create<mlir::arith::ConstantOp>(
          loc, accumulator, fromAttr);
      mlir::Value to = rewriter.create<mlir::arith::ConstantOp>(
          loc, accumulator, toAttr);
      mlir::Value value = rewriter.create<mlir::arith::AddFOp>(
          loc, from,
          rewriter.create<mlir::arith::MulFOp>(
              loc, unitAccumulator,
              rewriter.create<mlir::arith::SubFOp>(loc, to, from)));
      storeFromAccumulator(rewriter, loc, value, output, outputIndices);
      rewriter.create<mlir::gpu::TerminatorOp>(loc);
      rewriter.eraseOp(op);
      return mlir::success();
    }

    if (emitter->recipe == VulkanKernelRecipe::LIN_SPACE) {
      auto accumulatorAttr =
          op->getAttrOfType<mlir::TypeAttr>(kAccumulatorTypeAttr);
      auto accumulator =
          accumulatorAttr
              ? llvm::dyn_cast<mlir::FloatType>(accumulatorAttr.getValue())
              : mlir::FloatType{};
      auto startAttr =
          op->getAttrOfType<mlir::FloatAttr>("nd4j.linspace_start");
      auto stepAttr =
          op->getAttrOfType<mlir::FloatAttr>("nd4j.linspace_step");
      auto outputUnsignedAttr =
          op->getAttrOfType<mlir::BoolAttr>("nd4j.output_unsigned");
      auto outputFloat =
          llvm::dyn_cast<mlir::FloatType>(outputType.getElementType());
      auto outputInteger =
          llvm::dyn_cast<mlir::IntegerType>(outputType.getElementType());
      if (!accumulator || !startAttr || !stepAttr || !outputUnsignedAttr ||
          startAttr.getType() != accumulator ||
          stepAttr.getType() != accumulator ||
          outputType.getRank() != 1 ||
          (!outputFloat &&
           (!outputInteger || outputInteger.getWidth() != 32))) {
        return op.emitOpError("lin_space metadata contract mismatch");
      }
      auto [flat, outputIndices] = launchOutput();
      mlir::Value position =
          convertIndexToFloat(rewriter, loc, flat, accumulator);
      mlir::Value start = rewriter.create<mlir::arith::ConstantOp>(
          loc, accumulator, startAttr);
      mlir::Value step = rewriter.create<mlir::arith::ConstantOp>(
          loc, accumulator, stepAttr);
      mlir::Value value = rewriter.create<mlir::arith::AddFOp>(
          loc, start,
          rewriter.create<mlir::arith::MulFOp>(loc, step, position));
      if (outputFloat) {
        storeFromAccumulator(
            rewriter, loc, value, output, outputIndices);
      } else if (!storeScalar(
                     rewriter, loc, value, output, outputIndices,
                     false, outputUnsignedAttr.getValue())) {
        return op.emitOpError("lin_space output conversion failed");
      }
      rewriter.create<mlir::gpu::TerminatorOp>(loc);
      rewriter.eraseOp(op);
      return mlir::success();
    }

    if (emitter->recipe == VulkanKernelRecipe::RANGE) {
      auto startAttr =
          op->getAttrOfType<mlir::FloatAttr>("nd4j.range_start");
      auto deltaAttr =
          op->getAttrOfType<mlir::FloatAttr>("nd4j.range_delta");
      auto outputUnsignedAttr =
          op->getAttrOfType<mlir::BoolAttr>("nd4j.output_unsigned");
      auto outputFloat =
          llvm::dyn_cast<mlir::FloatType>(outputType.getElementType());
      auto outputInteger =
          llvm::dyn_cast<mlir::IntegerType>(outputType.getElementType());
      if (!startAttr || !deltaAttr || !outputUnsignedAttr ||
          outputType.getRank() != 1 ||
          (!outputFloat &&
           (!outputInteger || outputInteger.getWidth() != 32))) {
        return op.emitOpError("range metadata contract mismatch");
      }
      auto [flat, outputIndices] = launchOutput();
      if (outputFloat) {
        auto accumulatorAttr =
            op->getAttrOfType<mlir::TypeAttr>(kAccumulatorTypeAttr);
        auto accumulator =
            accumulatorAttr
                ? llvm::dyn_cast<mlir::FloatType>(accumulatorAttr.getValue())
                : mlir::FloatType{};
        if (!accumulator) {
          return op.emitOpError("range floating AccT metadata is missing");
        }
        mlir::Value startStorage = scalarConstant(
            rewriter, loc, outputFloat, startAttr.getValueAsDouble());
        mlir::Value deltaStorage = scalarConstant(
            rewriter, loc, outputFloat, deltaAttr.getValueAsDouble());
        if (!startStorage || !deltaStorage) {
          return op.emitOpError("range floating constants are invalid");
        }
        mlir::Value position =
            convertIndexToFloat(rewriter, loc, flat, accumulator);
        mlir::Value value = rewriter.create<mlir::arith::AddFOp>(
            loc, convertFloat(rewriter, loc, startStorage, accumulator),
            rewriter.create<mlir::arith::MulFOp>(
                loc,
                convertFloat(rewriter, loc, deltaStorage, accumulator),
                position));
        storeFromAccumulator(
            rewriter, loc, value, output, outputIndices);
      } else {
        mlir::Value start = scalarConstant(
            rewriter, loc, outputInteger, startAttr.getValueAsDouble());
        mlir::Value delta = scalarConstant(
            rewriter, loc, outputInteger, deltaAttr.getValueAsDouble());
        if (!start || !delta) {
          return op.emitOpError("range integer constants are invalid");
        }
        mlir::Value position = rewriter.create<mlir::arith::IndexCastOp>(
            loc, outputInteger, flat);
        mlir::Value value = rewriter.create<mlir::arith::AddIOp>(
            loc, start,
            rewriter.create<mlir::arith::MulIOp>(
                loc, delta, position));
        rewriter.create<mlir::memref::StoreOp>(
            loc, value, output, outputIndices);
      }
      rewriter.create<mlir::gpu::TerminatorOp>(loc);
      rewriter.eraseOp(op);
      return mlir::success();
    }

    if (emitter->recipe == VulkanKernelRecipe::FILL_AS) {
      auto valueAttr =
          op->getAttrOfType<mlir::FloatAttr>("nd4j.constant_value");
      auto outputUnsignedAttr =
          op->getAttrOfType<mlir::BoolAttr>("nd4j.output_unsigned");
      if (!valueAttr || !outputUnsignedAttr ||
          inputType.getRank() != outputType.getRank()) {
        return op.emitOpError("fill_as metadata contract mismatch");
      }
      auto [flat, outputIndices] = launchOutput();
      (void)flat;
      mlir::Value value = scalarConstant(
          rewriter, loc, outputType.getElementType(),
          valueAttr.getValueAsDouble());
      if (!value) {
        return op.emitOpError(
            "fill_as requires supported float or 32-bit integer storage");
      }
      rewriter.create<mlir::memref::StoreOp>(
          loc, value, output, outputIndices);
      rewriter.create<mlir::gpu::TerminatorOp>(loc);
      rewriter.eraseOp(op);
      return mlir::success();
    }

    if (emitter->recipe == VulkanKernelRecipe::ZEROS_AS ||
        emitter->recipe == VulkanKernelRecipe::ONES_AS) {
      auto fillOneAttr =
          op->getAttrOfType<mlir::BoolAttr>("nd4j.fill_one");
      const bool expectedOne =
          emitter->recipe == VulkanKernelRecipe::ONES_AS;
      const bool structuralInput =
          vulkanInputIsStructuralIndex(*emitter, 0);
      const auto structuralElement =
          llvm::dyn_cast<mlir::IntegerType>(inputType.getElementType());
      const bool structuralShapeMatches =
          structuralInput &&
          ((inputType.getRank() == 0 && outputType.getRank() == 1) ||
           (inputType.getRank() == 1 &&
            inputType.getDimSize(0) == outputType.getRank()));
      if ((!structuralInput &&
           inputType.getRank() != outputType.getRank()) ||
          (structuralInput &&
           (!structuralElement || structuralElement.getWidth() != 32 ||
            !structuralShapeMatches)) ||
          !fillOneAttr || fillOneAttr.getValue() != expectedOne) {
        return op.emitOpError("constant fill metadata contract mismatch");
      }
      auto [flat, outputIndices] = launchOutput();
      (void)flat;
      mlir::Value value = scalarConstant(
          rewriter, loc, outputType.getElementType(), expectedOne ? 1.0 : 0.0);
      if (!value) {
        return op.emitOpError(
            "constant fill requires supported float or 32-bit integer storage");
      }
      rewriter.create<mlir::memref::StoreOp>(
          loc, value, output, outputIndices);
      rewriter.create<mlir::gpu::TerminatorOp>(loc);
      rewriter.eraseOp(op);
      return mlir::success();
    }

    if (emitter->recipe == VulkanKernelRecipe::RANK_OF ||
        emitter->recipe == VulkanKernelRecipe::SIZE_OF ||
        emitter->recipe == VulkanKernelRecipe::SIZE_AT) {
      auto valueAttr =
          op->getAttrOfType<mlir::IntegerAttr>("nd4j.scalar_metadata");
      auto outputUnsignedAttr =
          op->getAttrOfType<mlir::BoolAttr>("nd4j.output_unsigned");
      auto outputFloat =
          llvm::dyn_cast<mlir::FloatType>(outputType.getElementType());
      auto outputInteger =
          llvm::dyn_cast<mlir::IntegerType>(outputType.getElementType());
      if (!valueAttr || !outputUnsignedAttr || outputType.getRank() != 0 ||
          (!outputFloat &&
           (!outputInteger || outputInteger.getWidth() != 32))) {
        return op.emitOpError("rank/size metadata contract mismatch");
      }
      auto [flat, outputIndices] = launchOutput();
      (void)flat;
      mlir::Value value = scalarConstant(
          rewriter, loc, outputType.getElementType(),
          static_cast<double>(valueAttr.getInt()));
      if (!value) {
        return op.emitOpError("rank/size scalar conversion failed");
      }
      rewriter.create<mlir::memref::StoreOp>(
          loc, value, output, outputIndices);
      rewriter.create<mlir::gpu::TerminatorOp>(loc);
      rewriter.eraseOp(op);
      return mlir::success();
    }

    if (emitter->recipe == VulkanKernelRecipe::SHAPE_OF) {
      auto valuesAttr =
          op->getAttrOfType<mlir::DenseI64ArrayAttr>("nd4j.shape_values");
      auto outputUnsignedAttr =
          op->getAttrOfType<mlir::BoolAttr>("nd4j.output_unsigned");
      auto outputElement =
          llvm::dyn_cast<mlir::IntegerType>(outputType.getElementType());
      if (!valuesAttr || valuesAttr.empty() || !outputUnsignedAttr ||
          !outputElement || outputElement.getWidth() != 32 ||
          outputType.getRank() != 1 ||
          inputType.getRank() != static_cast<int64_t>(valuesAttr.size()) ||
          outputType.getDimSize(0) !=
              static_cast<int64_t>(valuesAttr.size())) {
        return op.emitOpError("shape_of metadata contract mismatch");
      }
      auto [flat, outputIndices] = launchOutput();
      llvm::ArrayRef<int64_t> values = valuesAttr.asArrayRef();
      mlir::Value selected = rewriter.create<mlir::arith::ConstantIntOp>(
          loc, values.front(), outputElement.getWidth());
      for (size_t i = 1; i < values.size(); ++i) {
        mlir::Value atDimension = rewriter.create<mlir::arith::CmpIOp>(
            loc, mlir::arith::CmpIPredicate::eq, flat,
            idxConst(rewriter, loc, static_cast<int64_t>(i)));
        mlir::Value candidate =
            rewriter.create<mlir::arith::ConstantIntOp>(
                loc, values[i], outputElement.getWidth());
        selected = rewriter.create<mlir::arith::SelectOp>(
            loc, atDimension, candidate, selected);
      }
      rewriter.create<mlir::memref::StoreOp>(
          loc, selected, output, outputIndices);
      rewriter.create<mlir::gpu::TerminatorOp>(loc);
      rewriter.eraseOp(op);
      return mlir::success();
    }

    auto axisAttr = op->getAttrOfType<mlir::IntegerAttr>("nd4j.axis");
    auto depthAttr = op->getAttrOfType<mlir::IntegerAttr>("nd4j.depth");
    auto onAttr = op->getAttrOfType<mlir::FloatAttr>("nd4j.on_value");
    auto offAttr = op->getAttrOfType<mlir::FloatAttr>("nd4j.off_value");
    auto indexUnsignedAttr =
        op->getAttrOfType<mlir::BoolAttr>("nd4j.index_unsigned");
    auto outputUnsignedAttr =
        op->getAttrOfType<mlir::BoolAttr>("nd4j.output_unsigned");
    auto inputInteger =
        llvm::dyn_cast<mlir::IntegerType>(inputType.getElementType());
    auto inputFloat =
        llvm::dyn_cast<mlir::FloatType>(inputType.getElementType());
    auto outputInteger =
        llvm::dyn_cast<mlir::IntegerType>(outputType.getElementType());
    auto outputFloat =
        llvm::dyn_cast<mlir::FloatType>(outputType.getElementType());
    if (!axisAttr || !depthAttr || !onAttr || !offAttr ||
        !indexUnsignedAttr || !outputUnsignedAttr ||
        (!inputFloat && (!inputInteger || inputInteger.getWidth() != 32)) ||
        (!outputFloat && (!outputInteger || outputInteger.getWidth() != 32)) ||
        outputType.getRank() != inputType.getRank() + 1 ||
        axisAttr.getInt() < 0 || axisAttr.getInt() >= outputType.getRank() ||
        depthAttr.getInt() <= 0 ||
        outputType.getDimSize(axisAttr.getInt()) != depthAttr.getInt()) {
      return op.emitOpError("onehot metadata contract mismatch");
    }
    const int64_t axis = axisAttr.getInt();
    auto [flat, outputIndices] = launchOutput();
    (void)flat;
    mlir::SmallVector<mlir::Value> inputIndices;
    inputIndices.reserve(static_cast<size_t>(inputType.getRank()));
    for (int64_t d = 0; d < outputType.getRank(); ++d) {
      if (d != axis) inputIndices.push_back(outputIndices[d]);
    }
    mlir::Value rawIndex = rewriter.create<mlir::memref::LoadOp>(
        loc, inputs.front(), inputIndices);
    mlir::Value depthCoordinate =
        rewriter.create<mlir::arith::IndexCastOp>(
            loc, rewriter.getI32Type(), outputIndices[axis]);
    mlir::Value matches;
    if (inputInteger) {
      matches = rewriter.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::eq, rawIndex, depthCoordinate);
    } else {
      mlir::Value aboveNegativeOne =
          rewriter.create<mlir::arith::CmpFOp>(
              loc, mlir::arith::CmpFPredicate::OGT, rawIndex,
              scalarConstant(rewriter, loc, inputFloat, -1.0));
      mlir::Value belowI32Limit =
          rewriter.create<mlir::arith::CmpFOp>(
              loc, mlir::arith::CmpFPredicate::OLT, rawIndex,
              scalarConstant(rewriter, loc, inputFloat, 2147483648.0));
      mlir::Value convertible = rewriter.create<mlir::arith::AndIOp>(
          loc, aboveNegativeOne, belowI32Limit);
      auto guarded = rewriter.create<mlir::scf::IfOp>(
          loc, mlir::TypeRange{rewriter.getI1Type()}, convertible, true);
      rewriter.setInsertionPointToStart(guarded.thenBlock());
      mlir::Value converted = rewriter.create<mlir::arith::FPToSIOp>(
          loc, rewriter.getI32Type(), rawIndex);
      mlir::Value equal = rewriter.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::eq, converted, depthCoordinate);
      rewriter.create<mlir::scf::YieldOp>(loc, equal);
      rewriter.setInsertionPointToStart(guarded.elseBlock());
      mlir::Value falseValue =
          rewriter.create<mlir::arith::ConstantIntOp>(loc, 0, 1);
      rewriter.create<mlir::scf::YieldOp>(loc, falseValue);
      rewriter.setInsertionPointAfter(guarded);
      matches = guarded.getResult(0);
    }
    mlir::Value on = scalarConstant(
        rewriter, loc, outputType.getElementType(),
        onAttr.getValueAsDouble());
    mlir::Value off = scalarConstant(
        rewriter, loc, outputType.getElementType(),
        offAttr.getValueAsDouble());
    if (!on || !off) {
      return op.emitOpError(
          "onehot output requires supported float or 32-bit integer storage");
    }
    mlir::Value value = rewriter.create<mlir::arith::SelectOp>(
        loc, matches, on, off);
    rewriter.create<mlir::memref::StoreOp>(
        loc, value, output, outputIndices);
    rewriter.create<mlir::gpu::TerminatorOp>(loc);
    rewriter.eraseOp(op);
    return mlir::success();
  }
  if (multiOutput) {
    if (inputs.size() != 1 ||
        mlir::failed(validateCopyTypes(op, inputs, outputs))) {
      return mlir::failure();
    }
    auto axisAttr = op->getAttrOfType<mlir::IntegerAttr>("nd4j.axis");
    auto countAttr =
        op->getAttrOfType<mlir::IntegerAttr>("nd4j.num_outputs");
    auto inputType = llvm::dyn_cast<mlir::MemRefType>(inputs.front().getType());
    auto outputType = llvm::dyn_cast<mlir::MemRefType>(outputs.front().getType());
    if (!axisAttr || !countAttr || !inputType || !outputType ||
        countAttr.getInt() != static_cast<int64_t>(outputs.size()) ||
        axisAttr.getInt() < 0 || axisAttr.getInt() >= inputType.getRank()) {
      return op.emitOpError("split/unstack metadata contract mismatch");
    }
    const int64_t axis = axisAttr.getInt();
    const bool unstack =
        outputType.getRank() + 1 == inputType.getRank();
    if (!unstack && outputType.getRank() != inputType.getRank()) {
      return op.emitOpError("split/unstack rank contract mismatch");
    }
    for (mlir::Value value : outputs) {
      auto type = llvm::dyn_cast<mlir::MemRefType>(value.getType());
      if (!type || type.getShape() != outputType.getShape()) {
        return op.emitOpError("split/unstack outputs must have equal shapes");
      }
    }
    mlir::Value output = outputs.front();
    mlir::Value one = idxConst(rewriter, loc, 1);
    mlir::Value total = one;
    for (int64_t d = 0; d < outputType.getRank(); ++d) {
      total = rewriter.create<mlir::arith::MulIOp>(
          loc, total,
          rewriter.create<mlir::memref::DimOp>(loc, output, d));
    }
    auto launch = createGpuLaunch(rewriter, loc, total, one, one);
    mlir::Value flat = launch.getBlockIds().x;
    rewriter.setInsertionPointToEnd(&launch.getBody().front());
    auto outputIndices = logicalIndices(rewriter, loc, flat, output);
    for (auto item : llvm::enumerate(outputs)) {
      mlir::SmallVector<mlir::Value> source;
      if (unstack) {
        source.reserve(static_cast<size_t>(inputType.getRank()));
        size_t outputDimension = 0;
        for (int64_t d = 0; d < inputType.getRank(); ++d) {
          source.push_back(
              d == axis
                  ? idxConst(rewriter, loc,
                             static_cast<int64_t>(item.index()))
                  : outputIndices[outputDimension++]);
        }
      } else {
        source.append(outputIndices.begin(), outputIndices.end());
        mlir::Value chunk = rewriter.create<mlir::memref::DimOp>(
            loc, item.value(), axis);
        mlir::Value offset = rewriter.create<mlir::arith::MulIOp>(
            loc, chunk,
            idxConst(rewriter, loc, static_cast<int64_t>(item.index())));
        source[static_cast<size_t>(axis)] =
            rewriter.create<mlir::arith::AddIOp>(
                loc, source[static_cast<size_t>(axis)], offset);
      }
      mlir::Value value = rewriter.create<mlir::memref::LoadOp>(
          loc, inputs.front(), source);
      rewriter.create<mlir::memref::StoreOp>(
          loc, value, item.value(), outputIndices);
    }
    rewriter.create<mlir::gpu::TerminatorOp>(loc);
    rewriter.eraseOp(op);
    return mlir::success();
  }
  const bool gatherNd =
      hasVulkanOpTrait(*emitter, sd::ops::OP_TRAIT_GATHER_ND);
  const bool typedControls =
      hasVulkanMixedOperandTypeContract(*emitter) ||
      hasVulkanStructuralIndexOperands(*emitter);
  if (mlir::failed(
          typedControls
              ? validateMovementTypes(op, inputs, outputs, *emitter)
              : validateCopyTypes(op, inputs, outputs, gatherNd))) {
    return mlir::failure();
  }

  mlir::Value output = outputs.front();
  auto outputType = llvm::dyn_cast<mlir::MemRefType>(output.getType());
  if (!outputType) return op.emitOpError("movement output must be a MemRef");
  mlir::Value one = idxConst(rewriter, loc, 1);
  mlir::Value total = one;
  for (int64_t d = 0; d < outputType.getRank(); ++d) {
    total = rewriter.create<mlir::arith::MulIOp>(
        loc, total, rewriter.create<mlir::memref::DimOp>(loc, output, d));
  }
  auto launch = createGpuLaunch(rewriter, loc, total, one, one);
  mlir::Value flat = launch.getBlockIds().x;
  rewriter.setInsertionPointToEnd(&launch.getBody().front());
  mlir::SmallVector<mlir::Value> outputIndices =
      logicalIndices(rewriter, loc, flat, output);

  auto requireIntegerArray = [&](llvm::StringRef name)
      -> mlir::FailureOr<llvm::ArrayRef<int64_t>> {
    auto attr = op->getAttrOfType<mlir::DenseI64ArrayAttr>(name);
    if (!attr) {
      op.emitOpError() << "requires " << name;
      return mlir::failure();
    }
    return attr.asArrayRef();
  };
  auto rawCopy = [&](mlir::Value input,
                     mlir::ValueRange inputIndices) {
    mlir::Value value = rewriter.create<mlir::memref::LoadOp>(
        loc, input, inputIndices);
    rewriter.create<mlir::memref::StoreOp>(
        loc, value, output, outputIndices);
  };

  switch (emitter->recipe) {
    case VulkanKernelRecipe::GATHER_ND: {
      mlir::Value data = inputs[0];
      mlir::Value indices = inputs[1];
      auto dataType = llvm::dyn_cast<mlir::MemRefType>(data.getType());
      auto indicesType = llvm::dyn_cast<mlir::MemRefType>(indices.getType());
      auto integerType = indicesType
                             ? llvm::dyn_cast<mlir::IntegerType>(
                                   indicesType.getElementType())
                             : mlir::IntegerType{};
      if (!dataType || !indicesType || !integerType ||
          integerType.getWidth() != 32 || indicesType.getRank() < 1) {
        return op.emitOpError(
            "gather_nd requires rank-N data and 32-bit integer indices");
      }
      const int64_t indicesRank = indicesType.getRank();
      const int64_t indexedRank =
          indicesType.getDimSize(indicesRank - 1);
      if (mlir::ShapedType::isDynamic(indexedRank) || indexedRank <= 0 ||
          indexedRank > dataType.getRank()) {
        return op.emitOpError(
            "gather_nd requires a frozen valid final index dimension");
      }
      auto unsignedAttr =
          op->getAttrOfType<mlir::BoolAttr>("nd4j.index_unsigned");
      if (!unsignedAttr) {
        return op.emitOpError("gather_nd requires index signedness metadata");
      }
      const bool indexUnsigned = unsignedAttr.getValue();
      mlir::SmallVector<mlir::Value> dataIndices;
      dataIndices.reserve(static_cast<size_t>(dataType.getRank()));
      for (int64_t d = 0; d < indexedRank; ++d) {
        mlir::SmallVector<mlir::Value> indexCoordinates;
        indexCoordinates.reserve(static_cast<size_t>(indicesRank));
        for (int64_t p = 0; p < indicesRank - 1; ++p) {
          indexCoordinates.push_back(outputIndices[static_cast<size_t>(p)]);
        }
        indexCoordinates.push_back(idxConst(rewriter, loc, d));
        mlir::Value rawIndex = rewriter.create<mlir::memref::LoadOp>(
            loc, indices, indexCoordinates);
        mlir::Value zeroInteger =
            rewriter.create<mlir::arith::ConstantIntOp>(
                loc, 0, integerType.getWidth());
        mlir::Value nonnegative = rawIndex;
        if (!indexUnsigned) {
          nonnegative = rewriter.create<mlir::arith::SelectOp>(
              loc,
              rewriter.create<mlir::arith::CmpIOp>(
                  loc, mlir::arith::CmpIPredicate::slt, rawIndex,
                  zeroInteger),
              zeroInteger, rawIndex);
        }
        mlir::Value dimension =
            rewriter.create<mlir::memref::DimOp>(loc, data, d);
        mlir::Value lastIndex = rewriter.create<mlir::arith::SubIOp>(
            loc, dimension, one);
        mlir::Value lastInteger =
            rewriter.create<mlir::arith::IndexCastOp>(
                loc, integerType, lastIndex);
        mlir::Value tooHigh = rewriter.create<mlir::arith::CmpIOp>(
            loc,
            indexUnsigned ? mlir::arith::CmpIPredicate::ugt
                          : mlir::arith::CmpIPredicate::sgt,
            nonnegative, lastInteger);
        mlir::Value bounded = rewriter.create<mlir::arith::SelectOp>(
            loc, tooHigh, lastInteger, nonnegative);
        dataIndices.push_back(rewriter.create<mlir::arith::IndexCastOp>(
            loc, rewriter.getIndexType(), bounded));
      }
      const int64_t outputSuffix = indicesRank - 1;
      for (int64_t d = indexedRank; d < dataType.getRank(); ++d) {
        dataIndices.push_back(outputIndices[static_cast<size_t>(
            outputSuffix + d - indexedRank)]);
      }
      rawCopy(data, dataIndices);
      break;
    }

    case VulkanKernelRecipe::TILE: {
      auto repetitions = requireIntegerArray("nd4j.repetitions");
      auto inputType = llvm::dyn_cast<mlir::MemRefType>(inputs[0].getType());
      if (mlir::failed(repetitions) || !inputType ||
          static_cast<int64_t>(repetitions->size()) != inputType.getRank() ||
          outputType.getRank() != inputType.getRank()) {
        return op.emitOpError("tile rank/repetition contract mismatch");
      }
      mlir::SmallVector<mlir::Value> source;
      for (int64_t d = 0; d < inputType.getRank(); ++d) {
        mlir::Value dimension =
            rewriter.create<mlir::memref::DimOp>(loc, inputs[0], d);
        source.push_back(rewriter.create<mlir::arith::RemUIOp>(
            loc, outputIndices[static_cast<size_t>(d)], dimension));
      }
      rawCopy(inputs[0], source);
      break;
    }

    case VulkanKernelRecipe::REPEAT: {
      auto axisAttr = op->getAttrOfType<mlir::IntegerAttr>("nd4j.axis");
      auto repetitions = requireIntegerArray("nd4j.repetitions");
      auto inputType = llvm::dyn_cast<mlir::MemRefType>(inputs[0].getType());
      if (!axisAttr || mlir::failed(repetitions) || repetitions->empty() ||
          !inputType || outputType.getRank() != inputType.getRank() ||
          axisAttr.getInt() < 0 || axisAttr.getInt() >= inputType.getRank()) {
        return op.emitOpError("repeat metadata contract mismatch");
      }
      const int64_t axis = axisAttr.getInt();
      mlir::SmallVector<mlir::Value> source(outputIndices.begin(),
                                             outputIndices.end());
      if (repetitions->size() == 1) {
        source[static_cast<size_t>(axis)] =
            rewriter.create<mlir::arith::DivUIOp>(
                loc, outputIndices[static_cast<size_t>(axis)],
                idxConst(rewriter, loc, repetitions->front()));
      } else {
        mlir::Value sourceAxis = idxConst(rewriter, loc, 0);
        int64_t cumulative = repetitions->front();
        for (size_t i = 1; i < repetitions->size(); ++i) {
          mlir::Value atOrPastBoundary =
              rewriter.create<mlir::arith::CmpIOp>(
                  loc, mlir::arith::CmpIPredicate::uge,
                  outputIndices[static_cast<size_t>(axis)],
                  idxConst(rewriter, loc, cumulative));
          sourceAxis = rewriter.create<mlir::arith::SelectOp>(
              loc, atOrPastBoundary,
              idxConst(rewriter, loc, static_cast<int64_t>(i)), sourceAxis);
          cumulative += (*repetitions)[i];
        }
        source[static_cast<size_t>(axis)] = sourceAxis;
      }
      rawCopy(inputs[0], source);
      break;
    }

    case VulkanKernelRecipe::REVERSE: {
      auto axes = requireIntegerArray("nd4j.reverse_axes");
      auto inputType = llvm::dyn_cast<mlir::MemRefType>(inputs[0].getType());
      if (mlir::failed(axes) || !inputType ||
          outputType.getRank() != inputType.getRank()) {
        return op.emitOpError("reverse metadata contract mismatch");
      }
      std::set<int64_t> reversed;
      reversed.insert(axes->begin(), axes->end());
      mlir::SmallVector<mlir::Value> source;
      for (int64_t d = 0; d < inputType.getRank(); ++d) {
        if (reversed.count(d) != 0) {
          mlir::Value last = rewriter.create<mlir::arith::SubIOp>(
              loc, rewriter.create<mlir::memref::DimOp>(loc, inputs[0], d),
              one);
          source.push_back(rewriter.create<mlir::arith::SubIOp>(
              loc, last, outputIndices[static_cast<size_t>(d)]));
        } else {
          source.push_back(outputIndices[static_cast<size_t>(d)]);
        }
      }
      rawCopy(inputs[0], source);
      break;
    }

    case VulkanKernelRecipe::ROLL: {
      auto inputType = llvm::dyn_cast<mlir::MemRefType>(inputs[0].getType());
      auto inputFortranAttr =
          op->getAttrOfType<mlir::BoolAttr>("nd4j.roll_input_fortran");
      auto outputFortranAttr =
          op->getAttrOfType<mlir::BoolAttr>("nd4j.roll_output_fortran");
      auto tensorControlsAttr =
          op->getAttrOfType<mlir::BoolAttr>("nd4j.roll_tensor_controls");
      auto linearShiftAttr =
          op->getAttrOfType<mlir::IntegerAttr>("nd4j.roll_linear_shift");
      auto dimensionShiftsAttr =
          op->getAttrOfType<mlir::DenseI64ArrayAttr>(
              "nd4j.roll_dimension_shifts");
      if (!inputType || !inputFortranAttr || !outputFortranAttr ||
          outputType.getRank() != inputType.getRank()) {
        return op.emitOpError("roll metadata contract mismatch");
      }
      const bool inputFortran = inputFortranAttr.getValue();
      const bool outputFortran = outputFortranAttr.getValue();
      const bool tensorControls =
          tensorControlsAttr && tensorControlsAttr.getValue();

      auto backwardCoordinate =
          [&](mlir::OpBuilder& builder, mlir::Location nestedLoc,
              mlir::Value coordinate, mlir::Value dimension,
              mlir::Value shift) {
            mlir::Value distance =
                builder.create<mlir::arith::SubIOp>(
                    nestedLoc, dimension, shift);
            return mlir::Value(builder.create<mlir::arith::RemUIOp>(
                nestedLoc,
                builder.create<mlir::arith::AddIOp>(
                    nestedLoc, coordinate, distance),
                dimension));
          };
      auto normalizeTensorShift =
          [&](mlir::OpBuilder& builder, mlir::Location nestedLoc,
              mlir::Value rawShift, bool isUnsigned,
              mlir::Value dimension) {
            auto integerType =
                llvm::cast<mlir::IntegerType>(rawShift.getType());
            mlir::Value integerDimension =
                builder.create<mlir::arith::IndexCastOp>(
                    nestedLoc, integerType, dimension);
            mlir::Value remainder;
            if (isUnsigned) {
              remainder = builder.create<mlir::arith::RemUIOp>(
                  nestedLoc, rawShift, integerDimension);
            } else {
              remainder = builder.create<mlir::arith::RemSIOp>(
                  nestedLoc, rawShift, integerDimension);
              mlir::Value zeroInteger =
                  builder.create<mlir::arith::ConstantIntOp>(
                      nestedLoc, 0, integerType.getWidth());
              mlir::Value negative =
                  builder.create<mlir::arith::CmpIOp>(
                      nestedLoc, mlir::arith::CmpIPredicate::slt,
                      remainder, zeroInteger);
              remainder = builder.create<mlir::arith::SelectOp>(
                  nestedLoc, negative,
                  builder.create<mlir::arith::AddIOp>(
                      nestedLoc, remainder, integerDimension),
                  remainder);
            }
            return mlir::Value(builder.create<mlir::arith::IndexCastOp>(
                nestedLoc, builder.getIndexType(), remainder));
          };

      if (!tensorControls) {
        if (static_cast<bool>(linearShiftAttr) ==
            static_cast<bool>(dimensionShiftsAttr)) {
          return op.emitOpError(
              "roll requires exactly one frozen shift representation");
        }
        if (linearShiftAttr) {
          mlir::Value shift =
              idxConst(rewriter, loc, linearShiftAttr.getInt());
          mlir::Value sourceFlat =
              backwardCoordinate(rewriter, loc, flat, total, shift);
          auto sourceIndices = logicalIndicesForOrder(
              rewriter, loc, sourceFlat, inputs[0], inputFortran);
          auto destinationIndices = logicalIndicesForOrder(
              rewriter, loc, flat, output, outputFortran);
          mlir::Value value = rewriter.create<mlir::memref::LoadOp>(
              loc, inputs[0], sourceIndices);
          rewriter.create<mlir::memref::StoreOp>(
              loc, value, output, destinationIndices);
          break;
        }

        auto shifts = dimensionShiftsAttr.asArrayRef();
        if (static_cast<int64_t>(shifts.size()) != inputType.getRank()) {
          return op.emitOpError("roll dimension shift rank mismatch");
        }
        mlir::SmallVector<mlir::Value> source(
            outputIndices.begin(), outputIndices.end());
        for (int64_t d = 0; d < inputType.getRank(); ++d) {
          mlir::Value dimension =
              rewriter.create<mlir::memref::DimOp>(loc, inputs[0], d);
          source[static_cast<size_t>(d)] = backwardCoordinate(
              rewriter, loc, source[static_cast<size_t>(d)], dimension,
              idxConst(rewriter, loc, shifts[static_cast<size_t>(d)]));
        }
        rawCopy(inputs[0], source);
        break;
      }

      auto hasAxesAttr =
          op->getAttrOfType<mlir::BoolAttr>("nd4j.roll_has_axes");
      auto shiftUnsignedAttr =
          op->getAttrOfType<mlir::BoolAttr>("nd4j.roll_shift_unsigned");
      auto shiftFortranAttr =
          op->getAttrOfType<mlir::BoolAttr>("nd4j.roll_shift_fortran");
      if (!hasAxesAttr || !shiftUnsignedAttr || !shiftFortranAttr ||
          (inputs.size() != 2 && inputs.size() != 3) ||
          hasAxesAttr.getValue() != (inputs.size() == 3)) {
        return op.emitOpError("roll tensor-control contract mismatch");
      }
      const bool shiftUnsigned = shiftUnsignedAttr.getValue();
      const bool shiftFortran = shiftFortranAttr.getValue();
      auto shiftType =
          llvm::dyn_cast<mlir::MemRefType>(inputs[1].getType());
      auto shiftInteger =
          shiftType
              ? llvm::dyn_cast<mlir::IntegerType>(
                    shiftType.getElementType())
              : mlir::IntegerType{};
      if (!shiftType || !shiftInteger ||
          (shiftInteger.getWidth() != 32 &&
           shiftInteger.getWidth() != 64)) {
        return op.emitOpError("roll shift tensor requires integer storage");
      }

      if (inputs.size() == 2) {
        mlir::Value zeroIndex = idxConst(rewriter, loc, 0);
        auto shiftIndices = logicalIndicesForOrder(
            rewriter, loc, zeroIndex, inputs[1], shiftFortran);
        mlir::Value rawShift = rewriter.create<mlir::memref::LoadOp>(
            loc, inputs[1], shiftIndices);
        mlir::Value shift = normalizeTensorShift(
            rewriter, loc, rawShift, shiftUnsigned, total);
        mlir::Value sourceFlat =
            backwardCoordinate(rewriter, loc, flat, total, shift);
        auto sourceIndices = logicalIndicesForOrder(
            rewriter, loc, sourceFlat, inputs[0], inputFortran);
        auto destinationIndices = logicalIndicesForOrder(
            rewriter, loc, flat, output, outputFortran);
        mlir::Value value = rewriter.create<mlir::memref::LoadOp>(
            loc, inputs[0], sourceIndices);
        rewriter.create<mlir::memref::StoreOp>(
            loc, value, output, destinationIndices);
        break;
      }

      auto axesUnsignedAttr =
          op->getAttrOfType<mlir::BoolAttr>("nd4j.roll_axes_unsigned");
      auto axesFortranAttr =
          op->getAttrOfType<mlir::BoolAttr>("nd4j.roll_axes_fortran");
      auto axesType =
          llvm::dyn_cast<mlir::MemRefType>(inputs[2].getType());
      auto axesInteger =
          axesType
              ? llvm::dyn_cast<mlir::IntegerType>(
                    axesType.getElementType())
              : mlir::IntegerType{};
      if (!axesUnsignedAttr || !axesFortranAttr || !axesType || !axesInteger ||
          (axesInteger.getWidth() != 32 &&
           axesInteger.getWidth() != 64)) {
        return op.emitOpError("roll axes tensor requires integer storage");
      }
      const bool axesUnsigned = axesUnsignedAttr.getValue();
      const bool axesFortran = axesFortranAttr.getValue();
      mlir::Value controlLength = one;
      for (int64_t d = 0; d < shiftType.getRank(); ++d) {
        controlLength = rewriter.create<mlir::arith::MulIOp>(
            loc, controlLength,
            rewriter.create<mlir::memref::DimOp>(loc, inputs[1], d));
      }

      mlir::SmallVector<mlir::Value> source(
          outputIndices.begin(), outputIndices.end());
      for (int64_t d = 0; d < inputType.getRank(); ++d) {
        mlir::Value dimension =
            rewriter.create<mlir::memref::DimOp>(loc, inputs[0], d);
        auto axisLoop = emitReductionLoop(
            rewriter, loc, idxConst(rewriter, loc, 0), controlLength, one,
            source[static_cast<size_t>(d)],
            [&](mlir::OpBuilder& builder, mlir::Location nestedLoc,
                mlir::Value controlIndex, mlir::Value coordinate) {
              auto shiftIndices = logicalIndicesForOrder(
                  builder, nestedLoc, controlIndex, inputs[1],
                  shiftFortran);
              auto axesIndices = logicalIndicesForOrder(
                  builder, nestedLoc, controlIndex, inputs[2],
                  axesFortran);
              mlir::Value rawShift =
                  builder.create<mlir::memref::LoadOp>(
                      nestedLoc, inputs[1], shiftIndices);
              mlir::Value rawAxis =
                  builder.create<mlir::memref::LoadOp>(
                      nestedLoc, inputs[2], axesIndices);
              mlir::Value normalizedAxis = rawAxis;
              if (!axesUnsigned) {
                mlir::Value zeroInteger =
                    builder.create<mlir::arith::ConstantIntOp>(
                        nestedLoc, 0, axesInteger.getWidth());
                mlir::Value negative =
                    builder.create<mlir::arith::CmpIOp>(
                        nestedLoc, mlir::arith::CmpIPredicate::slt,
                        rawAxis, zeroInteger);
                normalizedAxis = builder.create<mlir::arith::SelectOp>(
                    nestedLoc, negative,
                    builder.create<mlir::arith::AddIOp>(
                        nestedLoc, rawAxis,
                        builder.create<mlir::arith::ConstantIntOp>(
                            nestedLoc, inputType.getRank(),
                            axesInteger.getWidth())),
                    rawAxis);
              }
              mlir::Value matches =
                  builder.create<mlir::arith::CmpIOp>(
                      nestedLoc, mlir::arith::CmpIPredicate::eq,
                      normalizedAxis,
                      builder.create<mlir::arith::ConstantIntOp>(
                          nestedLoc, d, axesInteger.getWidth()));
              mlir::Value shift = normalizeTensorShift(
                  builder, nestedLoc, rawShift, shiftUnsigned, dimension);
              mlir::Value rolled = backwardCoordinate(
                  builder, nestedLoc, coordinate, dimension, shift);
              return mlir::Value(builder.create<mlir::arith::SelectOp>(
                  nestedLoc, matches, rolled, coordinate));
            });
        source[static_cast<size_t>(d)] = axisLoop.getResult(0);
      }
      rawCopy(inputs[0], source);
      break;
    }

    case VulkanKernelRecipe::SLICE: {
      auto begin = requireIntegerArray("nd4j.slice_begin");
      auto sizes = requireIntegerArray("nd4j.slice_sizes");
      auto inputType = llvm::dyn_cast<mlir::MemRefType>(inputs[0].getType());
      if (mlir::failed(begin) || mlir::failed(sizes) || !inputType ||
          static_cast<int64_t>(begin->size()) != inputType.getRank() ||
          begin->size() != sizes->size()) {
        return op.emitOpError("slice metadata contract mismatch");
      }
      mlir::SmallVector<mlir::Value> source;
      for (int64_t d = 0; d < inputType.getRank(); ++d) {
        mlir::Value local = outputType.getRank() == 0
                                ? idxConst(rewriter, loc, 0)
                                : outputIndices[static_cast<size_t>(d)];
        source.push_back(rewriter.create<mlir::arith::AddIOp>(
            loc, idxConst(rewriter, loc, (*begin)[static_cast<size_t>(d)]),
            local));
      }
      rawCopy(inputs[0], source);
      break;
    }

    case VulkanKernelRecipe::STRIDED_SLICE: {
      auto begin = requireIntegerArray("nd4j.slice_begin");
      auto end = requireIntegerArray("nd4j.slice_end");
      auto strides = requireIntegerArray("nd4j.slice_strides");
      auto inputType = llvm::dyn_cast<mlir::MemRefType>(inputs[0].getType());
      if (mlir::failed(begin) || mlir::failed(end) ||
          mlir::failed(strides) || !inputType ||
          outputType.getRank() != inputType.getRank() ||
          static_cast<int64_t>(begin->size()) != inputType.getRank() ||
          end->size() != begin->size() ||
          strides->size() != begin->size()) {
        return op.emitOpError("strided_slice metadata contract mismatch");
      }
      mlir::SmallVector<mlir::Value> source;
      source.reserve(begin->size());
      for (size_t d = 0; d < begin->size(); ++d) {
        const int64_t beginValue = (*begin)[d];
        const int64_t endValue = (*end)[d];
        const int64_t strideValue = (*strides)[d];
        const int64_t inputDimension = inputType.getDimSize(d);
        const int64_t expected =
            strideValue > 0
                ? (endValue - beginValue + strideValue - 1) / strideValue
                : -1;
        if (beginValue < 0 || endValue <= beginValue || strideValue <= 0 ||
            mlir::ShapedType::isDynamic(inputDimension) ||
            endValue > inputDimension ||
            outputType.getDimSize(d) != expected) {
          return op.emitOpError("strided_slice bounds contract mismatch");
        }
        source.push_back(rewriter.create<mlir::arith::AddIOp>(
            loc, idxConst(rewriter, loc, beginValue),
            rewriter.create<mlir::arith::MulIOp>(
                loc, outputIndices[d],
                idxConst(rewriter, loc, strideValue))));
      }
      rawCopy(inputs[0], source);
      break;
    }

    case VulkanKernelRecipe::STACK: {
      auto axisAttr = op->getAttrOfType<mlir::IntegerAttr>("nd4j.axis");
      auto countAttr =
          op->getAttrOfType<mlir::IntegerAttr>("nd4j.num_inputs");
      if (!axisAttr || !countAttr || countAttr.getInt() <= 0 ||
          countAttr.getInt() != static_cast<int64_t>(inputs.size()) ||
          axisAttr.getInt() < 0 || axisAttr.getInt() >= outputType.getRank()) {
        return op.emitOpError("stack metadata contract mismatch");
      }
      const int64_t axis = axisAttr.getInt();
      mlir::Value selected;
      mlir::SmallVector<mlir::Value> baseCoordinates;
      for (int64_t d = 0; d < outputType.getRank(); ++d) {
        if (d != axis) {
          baseCoordinates.push_back(outputIndices[static_cast<size_t>(d)]);
        }
      }
      for (auto item : llvm::enumerate(inputs)) {
        auto inputType = llvm::dyn_cast<mlir::MemRefType>(
            item.value().getType());
        if (!inputType) return op.emitOpError("stack input must be a MemRef");
        mlir::SmallVector<mlir::Value> coordinates;
        if (inputType.getRank() == 1 && baseCoordinates.empty()) {
          coordinates.push_back(idxConst(rewriter, loc, 0));
        } else if (inputType.getRank() != 0) {
          if (inputType.getRank() !=
              static_cast<int64_t>(baseCoordinates.size())) {
            return op.emitOpError("stack input rank mismatch");
          }
          coordinates.append(baseCoordinates.begin(), baseCoordinates.end());
        }
        mlir::Value value = rewriter.create<mlir::memref::LoadOp>(
            loc, item.value(), coordinates);
        if (!selected) {
          selected = value;
        } else {
          mlir::Value choose = rewriter.create<mlir::arith::CmpIOp>(
              loc, mlir::arith::CmpIPredicate::eq,
              outputIndices[static_cast<size_t>(axis)],
              idxConst(rewriter, loc, static_cast<int64_t>(item.index())));
          selected = rewriter.create<mlir::arith::SelectOp>(
              loc, choose, value, selected);
        }
      }
      rewriter.create<mlir::memref::StoreOp>(
          loc, selected, output, outputIndices);
      break;
    }

    case VulkanKernelRecipe::TRIU: {
      if (outputType.getRank() < 2) {
        return op.emitOpError("triu requires rank-2+ tensors");
      }
      auto diagonalAttr =
          op->getAttrOfType<mlir::IntegerAttr>("nd4j.diagonal");
      if (!diagonalAttr) return op.emitOpError("triu requires diagonal metadata");
      mlir::Value input = inputs.front();
      mlir::Value value = rewriter.create<mlir::memref::LoadOp>(
          loc, input, outputIndices);
      const size_t rowAxis = static_cast<size_t>(outputType.getRank() - 2);
      const size_t columnAxis = static_cast<size_t>(outputType.getRank() - 1);
      mlir::Value columnMinusRow = rewriter.create<mlir::arith::SubIOp>(
          loc, outputIndices[columnAxis], outputIndices[rowAxis]);
      mlir::Value inUpperTriangle = rewriter.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::sge, columnMinusRow,
          idxConst(rewriter, loc, diagonalAttr.getInt()));
      mlir::Value zero = scalarConstant(
          rewriter, loc, outputType.getElementType(), 0.0);
      if (!zero) return op.emitOpError("triu requires numeric storage");
      rewriter.create<mlir::memref::StoreOp>(
          loc,
          rewriter.create<mlir::arith::SelectOp>(
              loc, inUpperTriangle, value, zero),
          output, outputIndices);
      break;
    }

    case VulkanKernelRecipe::TRIU_BP: {
      if (inputs.size() != 2 ||
          (outputType.getRank() != 0 && outputType.getRank() < 2)) {
        return op.emitOpError(
            "triu_bp requires two scalar or rank-2+ inputs");
      }
      auto diagonalAttr =
          op->getAttrOfType<mlir::IntegerAttr>("nd4j.diagonal");
      if (!diagonalAttr) {
        return op.emitOpError("triu_bp requires diagonal metadata");
      }
      mlir::Value zero = scalarConstant(
          rewriter, loc, outputType.getElementType(), 0.0);
      if (!zero) return op.emitOpError("triu_bp requires numeric storage");
      mlir::Value result = zero;
      if (outputType.getRank() >= 2) {
        mlir::Value gradient = rewriter.create<mlir::memref::LoadOp>(
            loc, inputs[1], outputIndices);
        const size_t rowAxis =
            static_cast<size_t>(outputType.getRank() - 2);
        const size_t columnAxis =
            static_cast<size_t>(outputType.getRank() - 1);
        mlir::Value columnMinusRow = rewriter.create<mlir::arith::SubIOp>(
            loc, outputIndices[columnAxis], outputIndices[rowAxis]);
        mlir::Value inUpperTriangle = rewriter.create<mlir::arith::CmpIOp>(
            loc, mlir::arith::CmpIPredicate::sge, columnMinusRow,
            idxConst(rewriter, loc, diagonalAttr.getInt()));
        result = rewriter.create<mlir::arith::SelectOp>(
            loc, inUpperTriangle, gradient, zero);
      }
      rewriter.create<mlir::memref::StoreOp>(
          loc, result, output, outputIndices);
      break;
    }

    default:
      return mlir::failure();
  }

  rewriter.create<mlir::gpu::TerminatorOp>(loc);
  rewriter.eraseOp(op);
  return mlir::success();
}

// ─────────────────────────────────────────────────────────────────────────────
//  Wave 3: reduction helpers
// ─────────────────────────────────────────────────────────────────────────────
//
// All five reduction patterns (sum, mean, max, min, prod) share the same
// two-level structure:
//   - 2-D input [rows, D]: outer loop over rows, inner reduction over D
//   - Full reduce (1-D or scalar output): single flat reduction over all elements
//
// The emitter selects between last-dim and full-reduce based on the
// nd4j.reduce_axes attribute:
//   - axes = [1] or axes = [rank-1] → last-dim reduce: output [rows]
//   - axes = [-1] / empty / [0,1,...,rank-1] → full reduce: scalar output [1]
//
// For Wave 3 we restrict to 2-D inputs with last-dim or full reduce forms.
// Higher-rank reductions are Wave 4+.
//
// Pattern guard: select reduction semantics by operation descriptor hash.

namespace {

/// Emit a 2-D last-dim reduction body.
///
/// For each row r in [0, rows):
///   acc = initVal
///   for k in [0, D): acc = combine(acc, X[r, k])
///   Y[r] = finalize(acc, D)      (or Y[r, 0] with keepDims)
///
/// @param rewriter  Pattern rewriter
/// @param loc       Location for emitted ops
/// @param X         Input memref [rows, D]
/// @param Y         Output memref [rows] or [rows, 1] (keepDims)
/// @param keepDims  Whether the output has a trailing size-1 dim
/// @param initVal   Initial accumulator value
/// @param combiner  f(acc, elem) -> new_acc
/// @param finalizer f(acc, dimD_as_float) -> result
static void emitLastDimReduce(
    mlir::PatternRewriter& rewriter, mlir::Location loc,
    mlir::Value X, mlir::Value Y, bool keepDims,
    mlir::FloatType accumulatorType, mlir::Value initVal,
    std::function<mlir::Value(mlir::OpBuilder&, mlir::Location,
                              mlir::Value /*acc*/, mlir::Value /*elem*/)> combiner,
    std::function<mlir::Value(mlir::OpBuilder&, mlir::Location,
                              mlir::Value /*acc*/, mlir::Value /*dimD_float*/)> finalizer) {

  mlir::Value zeroIdx = idxConst(rewriter, loc, 0);
  mlir::Value oneIdx  = idxConst(rewriter, loc, 1);
  mlir::Value numRows = rewriter.create<mlir::memref::DimOp>(loc, X, 0);
  mlir::Value dimD    = rewriter.create<mlir::memref::DimOp>(loc, X, 1);
  mlir::Value dimDValue = convertIndexToFloat(
      rewriter, loc, dimD, llvm::cast<mlir::FloatType>(initVal.getType()));

  rewriter.create<mlir::scf::ForOp>(
      loc, zeroIdx, numRows, oneIdx,
      mlir::ValueRange{},
      [&](mlir::OpBuilder& rb, mlir::Location rloc, mlir::Value ri, mlir::ValueRange) {
        // Inner reduction over D
        auto redLoop = emitReductionLoop(
            rb, rloc, zeroIdx, dimD, oneIdx, initVal,
            [&](mlir::OpBuilder& kb, mlir::Location kloc,
                mlir::Value ki, mlir::Value acc) -> mlir::Value {
              mlir::Value elem = loadAsAccumulator(
                  kb, kloc, X, mlir::SmallVector<mlir::Value>{ri, ki},
                  accumulatorType);
              return combiner(kb, kloc, acc, elem);
            });
        mlir::Value accResult = redLoop.getResult(0);
        mlir::Value outVal = finalizer(rb, rloc, accResult, dimDValue);

        if (keepDims) {
          storeFromAccumulator(rb, rloc, outVal, Y, mlir::SmallVector<mlir::Value>{ri, zeroIdx});

        } else {
          storeFromAccumulator(rb, rloc, outVal, Y, mlir::SmallVector<mlir::Value>{ri});
        }
        rb.create<mlir::scf::YieldOp>(rloc);
      });
}

/// Emit a full-tensor reduction body (1-D output, scalar at index [0]).
///
///   acc = initVal
///   for r in [0, rows): for k in [0, D): acc = combine(acc, X[r, k])
///   Y[0] = finalize(acc, rows*D)
///
static void emitFullReduce(
    mlir::PatternRewriter& rewriter, mlir::Location loc,
    mlir::Value X, mlir::Value Y,
    mlir::FloatType accumulatorType, mlir::Value initVal,
    std::function<mlir::Value(mlir::OpBuilder&, mlir::Location,
                              mlir::Value, mlir::Value)> combiner,
    std::function<mlir::Value(mlir::OpBuilder&, mlir::Location,
                              mlir::Value, mlir::Value)> finalizer) {

  mlir::Value zeroIdx = idxConst(rewriter, loc, 0);
  mlir::Value oneIdx  = idxConst(rewriter, loc, 1);
  mlir::Value numRows = rewriter.create<mlir::memref::DimOp>(loc, X, 0);
  mlir::Value dimD    = rewriter.create<mlir::memref::DimOp>(loc, X, 1);
  mlir::Value totalN  = rewriter.create<mlir::arith::MulIOp>(loc, numRows, dimD);
  mlir::Value totalNValue = convertIndexToFloat(
      rewriter, loc, totalN, llvm::cast<mlir::FloatType>(initVal.getType()));

  // Flat loop over all elements
  auto flatLoop = emitReductionLoop(
      rewriter, loc, zeroIdx, totalN, oneIdx, initVal,
      [&](mlir::OpBuilder& fb, mlir::Location floc,
          mlir::Value fi, mlir::Value acc) -> mlir::Value {
        // Decompose flat index into (row, col)
        mlir::Value ri = fb.create<mlir::arith::DivUIOp>(floc, fi, dimD);
        mlir::Value ki = fb.create<mlir::arith::RemUIOp>(floc, fi, dimD);
        mlir::Value elem = loadAsAccumulator(
            fb, floc, X, mlir::SmallVector<mlir::Value>{ri, ki},
            accumulatorType);
        return combiner(fb, floc, acc, elem);
      });
  mlir::Value accResult = flatLoop.getResult(0);
  mlir::Value outVal = finalizer(rewriter, loc, accResult, totalNValue);
  storeFromAccumulator(rewriter, loc, outVal, Y, mlir::SmallVector<mlir::Value>{zeroIdx});
}

/// Return true if the axes vector encodes a last-dim-only reduce for rank-2 input.
static bool isLastDimReduce(const llvm::ArrayRef<int64_t>& axes, int64_t rank) {
  if (axes.size() == 1 && (axes[0] == rank - 1 || axes[0] == -1)) return true;
  return false;
}

/// Return true if the axes encodes a full reduce (empty or all axes).
static bool isFullReduce(const llvm::ArrayRef<int64_t>& axes, int64_t rank) {
  if (axes.empty()) return true;
  if ((int64_t)axes.size() == rank) return true;
  // All axes present
  for (int64_t i = 0; i < rank; ++i) {
    bool found = false;
    for (auto a : axes) { if (a == i || a == i - rank) { found = true; break; } }
    if (!found) return false;
  }
  return true;
}

/// Common implementation for all five reduction patterns.
static mlir::LogicalResult emitReductionPattern(
    mlir::linalg::GenericOp op,
    mlir::PatternRewriter& rewriter) {
  auto ndReduceAttr = op->getAttrOfType<mlir::BoolAttr>(kNdReduceAttr);
  if (ndReduceAttr && ndReduceAttr.getValue()) {
    return mlir::failure();
  }

  mlir::Location loc = op.getLoc();
  auto typeContract = getComputeTypeContract(op, op.getInputs(), op.getOutputs());
  if (mlir::failed(typeContract)) return mlir::failure();
  mlir::FloatType elemTy = typeContract->accumulatorType;

  const auto* emitter = emitterForOperation(op);
  if (emitter == nullptr ||
      emitter->family != VulkanKernelFamily::REDUCTION) {
    return mlir::failure();
  }
  const VulkanKernelRecipe semantic = legacySemanticFor(op, emitter->recipe);
  const ReductionCallbacks callbacks =
      reductionCallbacksFor(semantic);
  if (!callbacks.combine || !callbacks.finalize) return mlir::failure();
  const double initConst = callbacks.initialValue;
  const BinaryCallback& combiner = callbacks.combine;
  const BinaryCallback& finalizer = callbacks.finalize;

  mlir::ValueRange inputs  = op.getInputs();
  mlir::ValueRange outputs = op.getOutputs();

  if (inputs.empty() || outputs.empty()) {
    return op.emitOpError("reduction: expected at least 1 input and 1 output");
  }

  mlir::Value X = inputs[0];
  mlir::Value Y = outputs[0];

  auto xType = llvm::dyn_cast<mlir::MemRefType>(X.getType());
  if (!xType || xType.getRank() != 2) {
    return op.emitOpError("reduction: only 2-D input supported in Wave 3");
  }
  int64_t rank = xType.getRank();

  // Read reduce axes
  auto axesAttr = op->getAttrOfType<mlir::DenseI64ArrayAttr>("nd4j.reduce_axes");
  llvm::ArrayRef<int64_t> axes;
  if (axesAttr) axes = axesAttr.asArrayRef();

  // keepDims
  bool keepDims = false;
  auto kdAttr = op->getAttrOfType<mlir::BoolAttr>("nd4j.keep_dims");
  if (kdAttr) keepDims = kdAttr.getValue();

  // Init constant
  mlir::Value initVal = floatConst(rewriter, loc, elemTy, initConst);

  if (isLastDimReduce(axes, rank)) {
    emitLastDimReduce(rewriter, loc, X, Y, keepDims, elemTy, initVal, combiner, finalizer);
  } else if (isFullReduce(axes, rank)) {
    emitFullReduce(rewriter, loc, X, Y, elemTy, initVal, combiner, finalizer);
  } else {
    return op.emitOpError("reduction: Wave 3 only supports last-dim or full reduce");
  }

  rewriter.eraseOp(op);
  return mlir::success();
}

}  // anonymous namespace (Wave 3 helpers)

// ─────────────────────────────────────────────────────────────────────────────
//  Wave 3: ReduceSumToSpirv
// ─────────────────────────────────────────────────────────────────────────────

mlir::LogicalResult ReduceSumToSpirv::matchAndRewrite(
    mlir::linalg::GenericOp op,
    mlir::PatternRewriter& rewriter) const {
  return emitReductionPattern(op, rewriter);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Wave 3: ReduceMeanToSpirv
// ─────────────────────────────────────────────────────────────────────────────

mlir::LogicalResult ReduceMeanToSpirv::matchAndRewrite(
    mlir::linalg::GenericOp op,
    mlir::PatternRewriter& rewriter) const {
  return emitReductionPattern(op, rewriter);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Wave 3: ReduceMaxToSpirv
// ─────────────────────────────────────────────────────────────────────────────

mlir::LogicalResult ReduceMaxToSpirv::matchAndRewrite(
    mlir::linalg::GenericOp op,
    mlir::PatternRewriter& rewriter) const {
  return emitReductionPattern(op, rewriter);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Wave 3: ReduceMinToSpirv
// ─────────────────────────────────────────────────────────────────────────────

mlir::LogicalResult ReduceMinToSpirv::matchAndRewrite(
    mlir::linalg::GenericOp op,
    mlir::PatternRewriter& rewriter) const {
  return emitReductionPattern(op, rewriter);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Wave 3: ReduceProdToSpirv
// ─────────────────────────────────────────────────────────────────────────────

mlir::LogicalResult ReduceProdToSpirv::matchAndRewrite(
    mlir::linalg::GenericOp op,
    mlir::PatternRewriter& rewriter) const {

  return emitReductionPattern(op, rewriter);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Wave 4: ReduceNDToSpirv
// ─────────────────────────────────────────────────────────────────────────────
//
// General stride-safe reduction lowering using logical ND index decomposition.
// Covers registered sum/mean/max/min/product descriptors for every supported
// rank and axis set. Floating-point kernels accumulate in framework-selected
// AccT; integer sum/max/min/product kernels accumulate in i32.
//
// Gate: matches linalg.generic with nd4j.nd_reduce = true and a registered
// reduction descriptor hash.
//
// Algorithm:
//   For a rank-R input X[d0, d1, ..., d(R-1)] reducing axes in reduceAxes:
//   1. Compute input strides (row-major, from inner to outer dimension).
//   2. For each flat input index fi in [0, N):
//      a. Decompose fi → ND input indices inIdx[0..R-1].
//      b. Build output indices outIdx: skip axes in reduceAxes (or set to 0 for keepDims).
//      c. Compute flat output index and accumulate.
//   3. Finalize (divide for mean; identity for sum/max/min/prod).
//
// This is a serial single-invocation kernel (1,1,1 dispatch). Each output
// accumulator is initialized and finalized entirely inside the real GPU kernel.

mlir::LogicalResult ReduceNDToSpirv::matchAndRewrite(
    mlir::linalg::GenericOp op,
    mlir::PatternRewriter& rewriter) const {

  mlir::Location loc = op.getLoc();

  // ── 1. Guard: nd4j.nd_reduce must be true ────────────────────────────────
  auto ndReduceAttr = op->getAttrOfType<mlir::BoolAttr>(kNdReduceAttr);
  if (!ndReduceAttr || !ndReduceAttr.getValue()) {
    return mlir::failure();
  }

  // ── 2. Select reduction semantics from the emitter identity ──────────────
  const auto* emitter = emitterForOperation(op);
  if (emitter == nullptr ||
      emitter->family != VulkanKernelFamily::REDUCTION) {
    return mlir::failure();
  }
  const VulkanKernelRecipe semantic = legacySemanticFor(op, emitter->recipe);
  const bool reduce3 = semantic == VulkanKernelRecipe::REDUCE3;
  const ReductionCallbacks reduction =
      reductionCallbacksFor(semantic);
  if (!reduce3 && (!reduction.combine || !reduction.finalize)) {
    return mlir::failure();
  }
  const bool indexReduction = hasVulkanEmitterTrait(
      *emitter, VULKAN_EMITTER_TRAIT_INDEX_RESULT);
  const bool firstIndexReduction = hasVulkanEmitterTrait(
      *emitter, VULKAN_EMITTER_TRAIT_INDEX_FIRST);
  const bool lastIndexReduction = hasVulkanEmitterTrait(
      *emitter, VULKAN_EMITTER_TRAIT_INDEX_LAST);
  const bool floatingResultReduction = hasVulkanEmitterTrait(
      *emitter, VULKAN_EMITTER_TRAIT_FLOAT_RESULT);
  const bool absoluteInput = hasVulkanEmitterTrait(
      *emitter, VULKAN_EMITTER_TRAIT_ABSOLUTE_INPUT);
  const bool squareInput = hasVulkanEmitterTrait(
      *emitter, VULKAN_EMITTER_TRAIT_SQUARE_INPUT);
  const bool pNormReduction = hasVulkanEmitterTrait(
      *emitter, VULKAN_EMITTER_TRAIT_P_NORM);
  double pNorm = 0.0;
  if (pNormReduction) {
    auto pNormAttr = op->getAttrOfType<mlir::FloatAttr>("nd4j.p_norm");
    if (!pNormAttr || !std::isfinite(pNormAttr.getValueAsDouble()) ||
        pNormAttr.getValueAsDouble() <= 0.0) {
      return op.emitOpError("p-norm requires a finite positive p");
    }
    pNorm = pNormAttr.getValueAsDouble();
  }
  const bool countReduction = hasVulkanEmitterTrait(
      *emitter, VULKAN_EMITTER_TRAIT_COUNT_RESULT);
  const bool booleanResultReduction = hasVulkanEmitterTrait(
      *emitter, VULKAN_EMITTER_TRAIT_BOOLEAN_RESULT);

  // ── 3. Extract operands and the framework-selected AccT ───────────────────
  mlir::ValueRange inputs = op.getInputs();
  mlir::ValueRange outputs = op.getOutputs();
  if ((reduce3 ? inputs.size() != 2 : inputs.size() != 1) ||
      outputs.size() != 1) {
    return op.emitOpError(
        "ReduceNDToSpirv: invalid input/output arity");
  }
  mlir::Value X = inputs.front();
  mlir::Value X1 = reduce3 ? inputs[1] : mlir::Value{};
  mlir::Value Y = outputs.front();
  auto xType = llvm::dyn_cast<mlir::MemRefType>(X.getType());
  auto x1Type = reduce3
                    ? llvm::dyn_cast<mlir::MemRefType>(X1.getType())
                    : mlir::MemRefType{};
  auto yType = llvm::dyn_cast<mlir::MemRefType>(Y.getType());
  if (!xType || !yType || (reduce3 && !x1Type)) {
    return op.emitOpError("reduction ND: operands must be MemRefs");
  }
  if (reduce3 &&
      (xType.getRank() != x1Type.getRank() ||
       xType.getShape() != x1Type.getShape())) {
    return op.emitOpError("reduce3 inputs must have identical shapes");
  }

  auto computeAttr = op->getAttrOfType<mlir::TypeAttr>(kAccumulatorTypeAttr);
  mlir::Type computeType = computeAttr ? computeAttr.getValue() : mlir::Type{};
  auto computeFloat = llvm::dyn_cast<mlir::FloatType>(computeType);
  auto computeInteger = llvm::dyn_cast<mlir::IntegerType>(computeType);
  if (!computeFloat && (!computeInteger || computeInteger.getWidth() != 32)) {
    return op.emitOpError(
        "reduction ND: requires floating-point AccT or i32 computation");
  }
  if (computeInteger &&
      (hasVulkanEmitterTrait(*emitter, VULKAN_EMITTER_TRAIT_MEAN_FINALIZE) ||
       hasVulkanEmitterTrait(*emitter, VULKAN_EMITTER_TRAIT_SQRT_FINALIZE) ||
       pNormReduction || floatingResultReduction)) {
    return op.emitOpError(
        "reduction ND: floating finalizer requires floating-point AccT");
  }

  auto readBoolAttr = [&](llvm::StringRef name) {
    auto attr = op->getAttrOfType<mlir::BoolAttr>(name);
    return attr && attr.getValue();
  };
  const bool inputUnsigned = readBoolAttr("nd4j.input0_unsigned");
  const bool input1Unsigned = readBoolAttr("nd4j.input1_unsigned");
  const bool outputUnsigned = readBoolAttr("nd4j.output_unsigned");
  const bool biasCorrected = readBoolAttr("nd4j.bias_corrected");
  // Signed absolute values are represented by their i32 magnitude bits.  In
  // particular, abs(INT_MIN) remains 0x80000000 and must be ordered unsigned.
  const bool magnitudeComparison = inputUnsigned || absoluteInput;

  if (computeFloat) {
    auto xStorage = llvm::dyn_cast<mlir::FloatType>(xType.getElementType());
    auto yStorage = llvm::dyn_cast<mlir::FloatType>(yType.getElementType());
    auto xInteger = llvm::dyn_cast<mlir::IntegerType>(xType.getElementType());
    auto yInteger = llvm::dyn_cast<mlir::IntegerType>(yType.getElementType());
    const bool compatibleFloatInput =
        xStorage && xStorage.getWidth() <= computeFloat.getWidth() &&
        (xStorage == computeFloat ||
         xStorage.getWidth() != computeFloat.getWidth());
    const bool compatibleIntegerInput =
        (reduce3 || floatingResultReduction || countReduction ||
         booleanResultReduction) && xInteger &&
        xInteger.getWidth() <= 64;
    const bool compatibleFloatOutput =
        yStorage && yStorage.getWidth() <= computeFloat.getWidth() &&
        (yStorage == computeFloat ||
         yStorage.getWidth() != computeFloat.getWidth());
    if ((!compatibleFloatInput && !compatibleIntegerInput) ||
        (countReduction
             ? (!yInteger || yInteger.getWidth() != 64)
             : (indexReduction
                    ? (!yInteger || yInteger.getWidth() != 32)
                    : (booleanResultReduction
                           ? (!yInteger || yInteger.getWidth() != 8)
                           : !compatibleFloatOutput)))) {
      return op.emitOpError(
          "reduction ND: storage types are incompatible with AccT");
    }
  } else {
    auto xStorage = llvm::dyn_cast<mlir::IntegerType>(xType.getElementType());
    auto yStorage = llvm::dyn_cast<mlir::IntegerType>(yType.getElementType());
    const bool inputStorageValid =
        xStorage && (xStorage.getWidth() == 32 ||
                     (booleanResultReduction && xStorage.getWidth() == 8));
    const bool outputStorageValid =
        yStorage && (countReduction ? yStorage.getWidth() == 64
                                    : (booleanResultReduction
                                           ? yStorage.getWidth() == 8
                                           : yStorage.getWidth() == 32));
    if (!inputStorageValid || !outputStorageValid) {
      return op.emitOpError(
          "reduction ND: integer storage must be i32");
    }
  }

  int64_t rank = xType.getRank();
  if (rank < 1) {
    return op.emitOpError("reduction ND: rank must be >=1");
  }

  // ── 6. Read reduce axes ───────────────────────────────────────────────────
  auto axesAttr = op->getAttrOfType<mlir::DenseI64ArrayAttr>(kAxesAttr);
  llvm::SmallVector<int64_t> axes;
  llvm::SmallVector<int8_t> reduced(static_cast<size_t>(rank), 0);
  if (axesAttr) {
    for (int64_t rawAxis : axesAttr.asArrayRef()) {
      const int64_t axis = rawAxis < 0 ? rawAxis + rank : rawAxis;
      if (axis < 0 || axis >= rank ||
          reduced[static_cast<size_t>(axis)] != 0) {
        return op.emitOpError(
            "reduction ND: axes must be unique and in range");
      }
      reduced[static_cast<size_t>(axis)] = 1;
      axes.push_back(axis);
    }
  } else {
    for (int64_t d = 0; d < rank; ++d) {
      reduced[static_cast<size_t>(d)] = 1;
      axes.push_back(d);
    }
  }
  if (axes.empty()) {
    return op.emitOpError("reduction ND: at least one axis is required");
  }

  // keepDims
  bool keepDims = false;
  auto kdAttr = op->getAttrOfType<mlir::BoolAttr>(kKeepDimsAttr);
  if (kdAttr) keepDims = kdAttr.getValue();

  // ── 7. Emit loop body ─────────────────────────────────────────────────────
  mlir::Value zeroIdx = idxConst(rewriter, loc, 0);
  mlir::Value oneIdx  = idxConst(rewriter, loc, 1);

  // Input dims and strides
  llvm::SmallVector<mlir::Value> inDims(rank);
  for (int64_t d = 0; d < rank; ++d) {
    inDims[d] = rewriter.create<mlir::memref::DimOp>(loc, X, d);
  }
  llvm::SmallVector<mlir::Value> inStrides(rank);
  inStrides[rank - 1] = oneIdx;
  for (int64_t d = rank - 2; d >= 0; --d) {
    inStrides[d] = rewriter.create<mlir::arith::MulIOp>(loc, inStrides[d + 1], inDims[d + 1]);
  }

  // Total input N
  mlir::Value totalN = oneIdx;
  for (int64_t d = 0; d < rank; ++d) {
    totalN = rewriter.create<mlir::arith::MulIOp>(loc, totalN, inDims[d]);
  }

  // Count of reduced elements is consumed only by floating-point mean.
  mlir::Value reduceCount = oneIdx;
  for (int64_t axis : axes) {
    reduceCount = rewriter.create<mlir::arith::MulIOp>(
        loc, reduceCount, inDims[axis]);
  }
  mlir::Value reduceCountValue;
  if (computeFloat) {
    reduceCountValue =
        convertIndexToFloat(rewriter, loc, reduceCount, computeFloat);
  }

  // Output dims and strides (for index building)
  // Output has axes removed (or set to 1 if keepDims).
  llvm::SmallVector<mlir::Value> outDims;
  for (int64_t d = 0; d < rank; ++d) {
    if (reduced[static_cast<size_t>(d)] == 0) {
      outDims.push_back(inDims[d]);
    } else if (keepDims) {
      outDims.push_back(oneIdx);
    }
  }
  int64_t outRank = static_cast<int64_t>(outDims.size());

  // Output strides
  llvm::SmallVector<mlir::Value> outStrides(outRank);
  if (outRank > 0) {
    outStrides[outRank - 1] = oneIdx;
    for (int64_t d = outRank - 2; d >= 0; --d) {
      outStrides[d] = rewriter.create<mlir::arith::MulIOp>(loc, outStrides[d + 1], outDims[d + 1]);
    }
  }

  // Iterate logical output elements; multi-index stores below preserve the
  // output MemRef's physical strides and offset.
  mlir::Value totalOutN = oneIdx;
  for (int64_t d = 0; d < yType.getRank(); ++d) {
    mlir::Value dim = rewriter.create<mlir::memref::DimOp>(loc, Y, d);
    totalOutN = rewriter.create<mlir::arith::MulIOp>(
        loc, totalOutN, dim);
  }

  // Accumulate each output entirely in framework-selected AccT.
  mlir::Value initConst =
      computeFloat
          ? floatConst(rewriter, loc, computeFloat, reduction.initialValue)
          : integerReductionInitial(
                rewriter, loc, semantic, computeInteger,
                magnitudeComparison);
  if (!initConst) {
    return op.emitOpError("reduction ND: unsupported accumulator identity");
  }
  // Each Vulkan invocation owns one output accumulator.  No atomics or host
  // initialization are required, and views remain stride-correct through the
  // logical multi-index load/store helpers.
  auto launch = createGpuLaunch(rewriter, loc, totalOutN, oneIdx, oneIdx);
  mlir::Value outputIndex = launch.getBlockIds().x;
  rewriter.setInsertionPointToEnd(&launch.getBody().front());

  // Input transforms are orthogonal emitter traits shared by value and index
  // reductions.  The base recipe remains sum/max/min; op identity never enters
  // lowering control flow.
  auto applyReductionInputTraits =
      [&](mlir::OpBuilder& builder, mlir::Location nestedLoc,
          mlir::Value element) -> mlir::Value {
    if (countReduction) {
      if (computeFloat) {
        auto zero = floatConst(builder, nestedLoc, computeFloat, 0.0);
        auto one = floatConst(builder, nestedLoc, computeFloat, 1.0);
        mlir::Value target = zero;
        if (semantic == VulkanKernelRecipe::REDUCE_COUNT_MATCH) {
          if (auto targetAttr =
                  op->getAttrOfType<mlir::FloatAttr>("nd4j.scalar0")) {
            target = floatConst(builder, nestedLoc, computeFloat,
                                targetAttr.getValueAsDouble());
          }
        }
        auto predicate = semantic == VulkanKernelRecipe::REDUCE_COUNT_ZERO
                             ? mlir::arith::CmpFPredicate::OEQ
                             : mlir::arith::CmpFPredicate::ONE;
        if (semantic == VulkanKernelRecipe::REDUCE_COUNT_MATCH) {
          predicate = mlir::arith::CmpFPredicate::OEQ;
        }
        auto matches = builder.create<mlir::arith::CmpFOp>(
            nestedLoc, predicate, element, target);
        return builder.create<mlir::arith::SelectOp>(
            nestedLoc, matches, one, zero);
      }
      auto zero = builder.create<mlir::arith::ConstantIntOp>(
          nestedLoc, 0, computeInteger.getWidth());
      auto one = builder.create<mlir::arith::ConstantIntOp>(
          nestedLoc, 1, computeInteger.getWidth());
      mlir::Value target = zero;
      if (semantic == VulkanKernelRecipe::REDUCE_COUNT_MATCH) {
        if (auto targetAttr =
                op->getAttrOfType<mlir::IntegerAttr>("nd4j.scalar0")) {
          target = builder.create<mlir::arith::ConstantIntOp>(
              nestedLoc, targetAttr.getInt(), computeInteger.getWidth());
        }
      }
      auto predicate = semantic == VulkanKernelRecipe::REDUCE_COUNT_ZERO
                           ? mlir::arith::CmpIPredicate::eq
                           : mlir::arith::CmpIPredicate::ne;
      if (semantic == VulkanKernelRecipe::REDUCE_COUNT_MATCH) {
        predicate = mlir::arith::CmpIPredicate::eq;
      }
      auto matches = builder.create<mlir::arith::CmpIOp>(
          nestedLoc, predicate, element, target);
      return builder.create<mlir::arith::SelectOp>(
          nestedLoc, matches, one, zero);
    }
    if (absoluteInput) {
      if (computeFloat) {
        element = builder.create<mlir::math::AbsFOp>(nestedLoc, element);
      } else if (!inputUnsigned) {
        mlir::Value zeroInteger =
            builder.create<mlir::arith::ConstantIntOp>(
                nestedLoc, 0, computeInteger.getWidth());
        mlir::Value negative = builder.create<mlir::arith::CmpIOp>(
            nestedLoc, mlir::arith::CmpIPredicate::slt,
            element, zeroInteger);
        mlir::Value negated = builder.create<mlir::arith::SubIOp>(
            nestedLoc, zeroInteger, element);
        element = builder.create<mlir::arith::SelectOp>(
            nestedLoc, negative, negated, element);
      }
    }
    if (squareInput) {
      if (computeFloat) {
        element = builder.create<mlir::arith::MulFOp>(
            nestedLoc, element, element);
      } else {
        element = builder.create<mlir::arith::MulIOp>(
            nestedLoc, element, element);
      }
    }
    if (pNormReduction) {
      element = builder.create<mlir::math::PowFOp>(
          nestedLoc, element,
          floatConst(builder, nestedLoc, computeFloat, pNorm));
    }
    if (semantic == VulkanKernelRecipe::REDUCE_ENTROPY ||
        semantic == VulkanKernelRecipe::REDUCE_LOG_ENTROPY ||
        semantic == VulkanKernelRecipe::REDUCE_SHANNON_ENTROPY) {
      if (computeFloat) {
        auto safe = builder.create<mlir::arith::MaximumFOp>(
            nestedLoc, element,
            floatConst(builder, nestedLoc, computeFloat, 1e-30));
        auto logarithm = emitLog(builder, nestedLoc, computeType, safe);
        if (semantic == VulkanKernelRecipe::REDUCE_SHANNON_ENTROPY) {
          logarithm = builder.create<mlir::arith::DivFOp>(
              nestedLoc, logarithm,
              floatConst(builder, nestedLoc, computeFloat,
                         std::log(2.0)));
        }
        element = builder.create<mlir::arith::MulFOp>(
            nestedLoc, element, logarithm);
      }
    }
    return element;
  };

  auto elementForOutput =
      [&](mlir::OpBuilder& builder, mlir::Location nestedLoc,
          mlir::Value inputIndex) -> std::pair<mlir::Value, mlir::Value> {
    llvm::SmallVector<mlir::Value> inIdx(rank);
    mlir::Value remainder = inputIndex;
    for (int64_t d = 0; d < rank; ++d) {
      inIdx[d] = builder.create<mlir::arith::DivUIOp>(
          nestedLoc, remainder, inStrides[d]);
      remainder = builder.create<mlir::arith::RemUIOp>(
          nestedLoc, remainder, inStrides[d]);
    }

    llvm::SmallVector<mlir::Value> outIdx;
    for (int64_t d = 0; d < rank; ++d) {
      if (reduced[static_cast<size_t>(d)] == 0) {
        outIdx.push_back(inIdx[d]);
      } else if (keepDims) {
        outIdx.push_back(zeroIdx);
      }
    }

    mlir::Value flatOutIdx = zeroIdx;
    for (size_t od = 0; od < outStrides.size(); ++od) {
      mlir::Value contribution = builder.create<mlir::arith::MulIOp>(
          nestedLoc, outIdx[od], outStrides[od]);
      flatOutIdx = builder.create<mlir::arith::AddIOp>(
          nestedLoc, flatOutIdx, contribution);
    }
    mlir::Value element = loadAsScalar(
        builder, nestedLoc, X,
        mlir::SmallVector<mlir::Value>(inIdx.begin(), inIdx.end()),
        computeType, inputUnsigned, inputUnsigned);
    mlir::Value belongsToOutput = builder.create<mlir::arith::CmpIOp>(
        nestedLoc, mlir::arith::CmpIPredicate::eq,
        flatOutIdx, outputIndex);
    return {element, belongsToOutput};
  };

  if (reduce3) {
    if (!computeFloat) {
      return op.emitOpError("reduce3 requires floating-point AccT");
    }
    auto pairForOutput =
        [&](mlir::OpBuilder& builder, mlir::Location nestedLoc,
            mlir::Value inputIndex)
        -> std::pair<std::pair<mlir::Value, mlir::Value>, mlir::Value> {
      llvm::SmallVector<mlir::Value> inIdx(rank);
      mlir::Value remainder = inputIndex;
      for (int64_t d = 0; d < rank; ++d) {
        inIdx[d] = builder.create<mlir::arith::DivUIOp>(
            nestedLoc, remainder, inStrides[d]);
        remainder = builder.create<mlir::arith::RemUIOp>(
            nestedLoc, remainder, inStrides[d]);
      }
      llvm::SmallVector<mlir::Value> outIdx;
      for (int64_t d = 0; d < rank; ++d) {
        if (reduced[static_cast<size_t>(d)] == 0) {
          outIdx.push_back(inIdx[d]);
        } else if (keepDims) {
          outIdx.push_back(zeroIdx);
        }
      }
      mlir::Value flatOutIdx = zeroIdx;
      for (size_t od = 0; od < outStrides.size(); ++od) {
        flatOutIdx = builder.create<mlir::arith::AddIOp>(
            nestedLoc, flatOutIdx,
            builder.create<mlir::arith::MulIOp>(
                nestedLoc, outIdx[od], outStrides[od]));
      }
      auto indices = mlir::SmallVector<mlir::Value>(inIdx.begin(), inIdx.end());
      mlir::Value first = loadAsScalar(
          builder, nestedLoc, X, indices, computeType, inputUnsigned,
          inputUnsigned);
      mlir::Value second = loadAsScalar(
          builder, nestedLoc, X1, indices, computeType, input1Unsigned,
          input1Unsigned);
      mlir::Value belongs = builder.create<mlir::arith::CmpIOp>(
          nestedLoc, mlir::arith::CmpIPredicate::eq, flatOutIdx,
          outputIndex);
      return {{first, second}, belongs};
    };
    int reduce3OpNum = -1;
    if (auto opNumAttr = op->getAttrOfType<mlir::IntegerAttr>(
            kLegacyOpNumAttr)) {
      reduce3OpNum = static_cast<int>(opNumAttr.getInt());
    }
    const bool cosine = reduce3OpNum == 2 || reduce3OpNum == 5;
    const bool jaccard = reduce3OpNum == 6;
    llvm::SmallVector<mlir::Value> reduce3Initial;
    if (cosine) {
      reduce3Initial = {floatConst(rewriter, loc, computeFloat, 0.0),
                        floatConst(rewriter, loc, computeFloat, 0.0),
                        floatConst(rewriter, loc, computeFloat, 0.0)};
    } else if (jaccard) {
      reduce3Initial = {floatConst(rewriter, loc, computeFloat, 0.0),
                        floatConst(rewriter, loc, computeFloat, 0.0)};
    } else {
      reduce3Initial = {floatConst(
          rewriter, loc, computeFloat,
          reduce3OpNum == 4 ? 1.0 : 0.0)};
    }
    auto reduce3Loop = rewriter.create<mlir::scf::ForOp>(
        loc, zeroIdx, totalN, oneIdx, mlir::ValueRange(reduce3Initial),
        [&](mlir::OpBuilder& builder, mlir::Location nestedLoc,
            mlir::Value inputIndex, mlir::ValueRange iterArgs) {
          auto pair = pairForOutput(builder, nestedLoc, inputIndex);
          mlir::Value a = pair.first.first;
          mlir::Value b = pair.first.second;
          mlir::Value belongs = pair.second;
          llvm::SmallVector<mlir::Value> candidates;
          if (reduce3OpNum == 0) {
            candidates.push_back(builder.create<mlir::math::AbsFOp>(
                nestedLoc, builder.create<mlir::arith::SubFOp>(
                               nestedLoc, a, b)));
          } else if (reduce3OpNum == 1) {
            mlir::Value delta = builder.create<mlir::arith::SubFOp>(
                nestedLoc, a, b);
            candidates.push_back(builder.create<mlir::arith::MulFOp>(
                nestedLoc, delta, delta));
          } else if (reduce3OpNum == 2 || reduce3OpNum == 5) {
            candidates.push_back(builder.create<mlir::arith::MulFOp>(
                nestedLoc, a, b));
            candidates.push_back(builder.create<mlir::arith::MulFOp>(
                nestedLoc, a, a));
            candidates.push_back(builder.create<mlir::arith::MulFOp>(
                nestedLoc, b, b));
          } else if (reduce3OpNum == 3) {
            candidates.push_back(builder.create<mlir::arith::MulFOp>(
                nestedLoc, a, b));
          } else if (reduce3OpNum == 4) {
            auto epsilonAttr = op->getAttrOfType<mlir::FloatAttr>(
                "nd4j.scalar0");
            const double epsilon =
                epsilonAttr ? epsilonAttr.getValueAsDouble() : 1e-5;
            mlir::Value delta = builder.create<mlir::math::AbsFOp>(
                nestedLoc, builder.create<mlir::arith::SubFOp>(
                               nestedLoc, a, b));
            auto matches = builder.create<mlir::arith::CmpFOp>(
                nestedLoc, mlir::arith::CmpFPredicate::OLE, delta,
                floatConst(builder, nestedLoc, computeFloat, epsilon));
            candidates.push_back(builder.create<mlir::arith::SelectOp>(
                nestedLoc, matches,
                floatConst(builder, nestedLoc, computeFloat, 1.0),
                floatConst(builder, nestedLoc, computeFloat, 0.0)));
          } else if (reduce3OpNum == 6) {
            candidates.push_back(builder.create<mlir::arith::MinimumFOp>(
                nestedLoc, a, b));
            candidates.push_back(builder.create<mlir::arith::MaximumFOp>(
                nestedLoc, a, b));
          } else {
            auto equal = builder.create<mlir::arith::CmpFOp>(
                nestedLoc, mlir::arith::CmpFPredicate::OEQ, a, b);
            candidates.push_back(builder.create<mlir::arith::SelectOp>(
                nestedLoc, equal,
                floatConst(builder, nestedLoc, computeFloat, 0.0),
                floatConst(builder, nestedLoc, computeFloat, 1.0)));
          }
          llvm::SmallVector<mlir::Value> next;
          for (size_t i = 0; i < candidates.size(); ++i) {
            mlir::Value combined;
            if (reduce3OpNum == 4) {
              combined = builder.create<mlir::arith::MulFOp>(
                  nestedLoc, iterArgs[i], candidates[i]);
            } else {
              combined = builder.create<mlir::arith::AddFOp>(
                  nestedLoc, iterArgs[i], candidates[i]);
            }
            next.push_back(builder.create<mlir::arith::SelectOp>(
                nestedLoc, belongs, combined, iterArgs[i]));
          }
          builder.create<mlir::scf::YieldOp>(nestedLoc, next);
        });
    mlir::Value result = reduce3Loop.getResult(0);
    if (reduce3OpNum == 1) {
      result = rewriter.create<mlir::math::SqrtOp>(loc, result);
    } else if (cosine) {
      mlir::Value denominator = rewriter.create<mlir::arith::MulFOp>(
          loc, rewriter.create<mlir::math::SqrtOp>(
                   loc, reduce3Loop.getResult(1)),
          rewriter.create<mlir::math::SqrtOp>(
                   loc, reduce3Loop.getResult(2)));
      auto zero = floatConst(rewriter, loc, computeFloat, 0.0);
      auto zeroDenominator = rewriter.create<mlir::arith::CmpFOp>(
          loc, mlir::arith::CmpFPredicate::OEQ, denominator, zero);
      auto similarity = rewriter.create<mlir::arith::SelectOp>(
          loc, zeroDenominator, zero,
          rewriter.create<mlir::arith::DivFOp>(
              loc, reduce3Loop.getResult(0), denominator));
      if (reduce3OpNum == 5) {
        result = rewriter
                     .create<mlir::arith::SubFOp>(
                         loc, floatConst(rewriter, loc, computeFloat, 1.0),
                         similarity)
                     .getResult();
      } else {
        result = similarity;
      }
    } else if (jaccard) {
      auto zero = floatConst(rewriter, loc, computeFloat, 0.0);
      auto zeroUnion = rewriter.create<mlir::arith::CmpFOp>(
          loc, mlir::arith::CmpFPredicate::OEQ, reduce3Loop.getResult(1),
          zero);
      result = rewriter.create<mlir::arith::SelectOp>(
          loc, zeroUnion, zero,
          rewriter.create<mlir::arith::SubFOp>(
              loc, floatConst(rewriter, loc, computeFloat, 1.0),
              rewriter.create<mlir::arith::DivFOp>(
                  loc, reduce3Loop.getResult(0),
                  reduce3Loop.getResult(1))));
    }
    auto outputIndices = logicalIndices(rewriter, loc, outputIndex, Y);
    if (!storeScalar(rewriter, loc, result, Y, outputIndices, false,
                     outputUnsigned)) {
      return op.emitOpError("reduce3 result storage conversion failed");
    }
    rewriter.create<mlir::gpu::TerminatorOp>(loc);
    rewriter.eraseOp(op);
    return mlir::success();
  }

  if (indexReduction) {
    // Index-result is an emitter trait layered on the shared reduction schedule.
    // Value extrema use strict comparison (first index on ties), while
    // FirstIndex/LastIndex select traversal order without consulting values.
    mlir::Value initialReducedIndex =
        firstIndexReduction ? idxConst(rewriter, loc, -1) : zeroIdx;
    auto indexLoop = rewriter.create<mlir::scf::ForOp>(
        loc, zeroIdx, totalN, oneIdx,
        mlir::ValueRange{initConst, initialReducedIndex},
        [&](mlir::OpBuilder& builder, mlir::Location nestedLoc,
            mlir::Value inputIndex, mlir::ValueRange iterArgs) {
          llvm::SmallVector<mlir::Value> inIdx(rank);
          mlir::Value remainder = inputIndex;
          for (int64_t d = 0; d < rank; ++d) {
            inIdx[d] = builder.create<mlir::arith::DivUIOp>(
                nestedLoc, remainder, inStrides[d]);
            remainder = builder.create<mlir::arith::RemUIOp>(
                nestedLoc, remainder, inStrides[d]);
          }

          llvm::SmallVector<mlir::Value> outIdx;
          for (int64_t d = 0; d < rank; ++d) {
            if (reduced[static_cast<size_t>(d)] == 0) {
              outIdx.push_back(inIdx[d]);
            } else if (keepDims) {
              outIdx.push_back(zeroIdx);
            }
          }
          mlir::Value flatOutIdx = zeroIdx;
          for (size_t od = 0; od < outStrides.size(); ++od) {
            mlir::Value contribution =
                builder.create<mlir::arith::MulIOp>(
                    nestedLoc, outIdx[od], outStrides[od]);
            flatOutIdx = builder.create<mlir::arith::AddIOp>(
                nestedLoc, flatOutIdx, contribution);
          }
          mlir::Value belongs = builder.create<mlir::arith::CmpIOp>(
              nestedLoc, mlir::arith::CmpIPredicate::eq,
              flatOutIdx, outputIndex);

          mlir::Value reducedIndex = zeroIdx;
          for (int64_t axis : axes) {
            reducedIndex = builder.create<mlir::arith::AddIOp>(
                nestedLoc,
                builder.create<mlir::arith::MulIOp>(
                    nestedLoc, reducedIndex, inDims[axis]),
                inIdx[axis]);
          }
          mlir::Value element = loadAsScalar(
              builder, nestedLoc, X,
              mlir::SmallVector<mlir::Value>(inIdx.begin(), inIdx.end()),
              computeType, inputUnsigned, inputUnsigned);
          element =
              applyReductionInputTraits(builder, nestedLoc, element);
          mlir::Value better;
          if (firstIndexReduction) {
            mlir::Value notSelected = builder.create<mlir::arith::CmpIOp>(
                nestedLoc, mlir::arith::CmpIPredicate::eq, iterArgs[1],
                idxConst(builder, nestedLoc, -1));
            better = notSelected;
          } else if (lastIndexReduction) {
            better = builder.create<mlir::arith::ConstantIntOp>(
                nestedLoc, 1, 1);
          } else if (computeFloat) {
            const bool maximumFormula =
                semantic == VulkanKernelRecipe::REDUCE_MAX;
            better = builder.create<mlir::arith::CmpFOp>(
                nestedLoc,
                maximumFormula ? mlir::arith::CmpFPredicate::OGT
                               : mlir::arith::CmpFPredicate::OLT,
                element, iterArgs[0]);
          } else {
            mlir::arith::CmpIPredicate predicate;
            const bool maximumFormula =
                semantic == VulkanKernelRecipe::REDUCE_MAX;
            if (maximumFormula) {
              predicate = magnitudeComparison
                              ? mlir::arith::CmpIPredicate::ugt
                              : mlir::arith::CmpIPredicate::sgt;
            } else {
              predicate = magnitudeComparison
                              ? mlir::arith::CmpIPredicate::ult
                              : mlir::arith::CmpIPredicate::slt;
            }
            better = builder.create<mlir::arith::CmpIOp>(
                nestedLoc, predicate, element, iterArgs[0]);
          }
          mlir::Value update = builder.create<mlir::arith::AndIOp>(
              nestedLoc, belongs, better);
          builder.create<mlir::scf::YieldOp>(
              nestedLoc,
              mlir::ValueRange{
                  builder.create<mlir::arith::SelectOp>(
                      nestedLoc, update, element, iterArgs[0]),
                  builder.create<mlir::arith::SelectOp>(
                      nestedLoc, update, reducedIndex, iterArgs[1])});
        });
    auto outputIndices = logicalIndices(rewriter, loc, outputIndex, Y);
    auto outputInteger =
        llvm::cast<mlir::IntegerType>(yType.getElementType());
    mlir::Value result = rewriter.create<mlir::arith::IndexCastOp>(
        loc, outputInteger, indexLoop.getResult(1));
    rewriter.create<mlir::memref::StoreOp>(
        loc, result, Y, outputIndices);
    rewriter.create<mlir::gpu::TerminatorOp>(loc);
    rewriter.eraseOp(op);
    return mlir::success();
  }

  if (semantic == VulkanKernelRecipe::REDUCE_LOGSUMEXP) {
    if (!computeFloat) {
      return op.emitOpError("logsumexp requires floating-point AccT");
    }
    mlir::Value negativeInfinity = floatConst(
        rewriter, loc, computeFloat,
        -std::numeric_limits<double>::infinity());
    auto maximumLoop = rewriter.create<mlir::scf::ForOp>(
        loc, zeroIdx, totalN, oneIdx,
        mlir::ValueRange{negativeInfinity},
        [&](mlir::OpBuilder& builder, mlir::Location nestedLoc,
            mlir::Value inputIndex, mlir::ValueRange iterArgs) {
          auto [element, belongs] =
              elementForOutput(builder, nestedLoc, inputIndex);
          mlir::Value candidate =
              builder.create<mlir::arith::MaximumFOp>(
                  nestedLoc, iterArgs[0], element);
          builder.create<mlir::scf::YieldOp>(
              nestedLoc,
              mlir::ValueRange{
                  builder.create<mlir::arith::SelectOp>(
                      nestedLoc, belongs, candidate, iterArgs[0])});
        });
    mlir::Value maximum = maximumLoop.getResult(0);
    auto exponentialSumLoop = rewriter.create<mlir::scf::ForOp>(
        loc, zeroIdx, totalN, oneIdx,
        mlir::ValueRange{floatConst(rewriter, loc, computeFloat, 0.0)},
        [&](mlir::OpBuilder& builder, mlir::Location nestedLoc,
            mlir::Value inputIndex, mlir::ValueRange iterArgs) {
          auto [element, belongs] =
              elementForOutput(builder, nestedLoc, inputIndex);
          mlir::Value exponential = emitExp(
              builder, nestedLoc, computeFloat,
              builder.create<mlir::arith::SubFOp>(
                  nestedLoc, element, maximum));
          mlir::Value candidate = builder.create<mlir::arith::AddFOp>(
              nestedLoc, iterArgs[0], exponential);
          builder.create<mlir::scf::YieldOp>(
              nestedLoc,
              mlir::ValueRange{
                  builder.create<mlir::arith::SelectOp>(
                      nestedLoc, belongs, candidate, iterArgs[0])});
        });
    mlir::Value result = rewriter.create<mlir::arith::AddFOp>(
        loc, maximum,
        rewriter.create<mlir::math::LogOp>(
            loc, exponentialSumLoop.getResult(0)));
    auto outputIndices = logicalIndices(rewriter, loc, outputIndex, Y);
    (void)storeScalar(rewriter, loc, result, Y, outputIndices,
                      inputUnsigned, outputUnsigned);
    rewriter.create<mlir::gpu::TerminatorOp>(loc);
    rewriter.eraseOp(op);
    return mlir::success();
  }

  if (semantic == VulkanKernelRecipe::REDUCE_VARIANCE ||
      semantic == VulkanKernelRecipe::REDUCE_STDEV) {
    if (!computeFloat) {
      return op.emitOpError("variance/stdev requires floating-point AccT");
    }
    mlir::Value zeroFloat = floatConst(rewriter, loc, computeFloat, 0.0);
    auto welfordLoop = rewriter.create<mlir::scf::ForOp>(
        loc, zeroIdx, totalN, oneIdx,
        mlir::ValueRange{zeroIdx, zeroFloat, zeroFloat},
        [&](mlir::OpBuilder& builder, mlir::Location nestedLoc,
            mlir::Value inputIndex, mlir::ValueRange iterArgs) {
          auto [element, belongs] =
              elementForOutput(builder, nestedLoc, inputIndex);
          mlir::Value candidateCount =
              builder.create<mlir::arith::AddIOp>(
                  nestedLoc, iterArgs[0], oneIdx);
          mlir::Value candidateCountFloat = convertIndexToFloat(
              builder, nestedLoc, candidateCount, computeFloat);
          mlir::Value delta = builder.create<mlir::arith::SubFOp>(
              nestedLoc, element, iterArgs[1]);
          mlir::Value candidateMean = builder.create<mlir::arith::AddFOp>(
              nestedLoc, iterArgs[1],
              builder.create<mlir::arith::DivFOp>(
                  nestedLoc, delta, candidateCountFloat));
          mlir::Value deltaAfterMean =
              builder.create<mlir::arith::SubFOp>(
                  nestedLoc, element, candidateMean);
          mlir::Value candidateM2 = builder.create<mlir::arith::AddFOp>(
              nestedLoc, iterArgs[2],
              builder.create<mlir::arith::MulFOp>(
                  nestedLoc, delta, deltaAfterMean));
          builder.create<mlir::scf::YieldOp>(
              nestedLoc,
              mlir::ValueRange{
                  builder.create<mlir::arith::SelectOp>(
                      nestedLoc, belongs, candidateCount, iterArgs[0]),
                  builder.create<mlir::arith::SelectOp>(
                      nestedLoc, belongs, candidateMean, iterArgs[1]),
                  builder.create<mlir::arith::SelectOp>(
                      nestedLoc, belongs, candidateM2, iterArgs[2])});
        });
    mlir::Value denominatorCount = welfordLoop.getResult(0);
    if (biasCorrected) {
      mlir::Value hasSample = rewriter.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::ugt,
          denominatorCount, zeroIdx);
      denominatorCount = rewriter.create<mlir::arith::SelectOp>(
          loc, hasSample,
          rewriter.create<mlir::arith::SubIOp>(
              loc, denominatorCount, oneIdx),
          zeroIdx);
    }
    mlir::Value denominator =
        convertIndexToFloat(rewriter, loc, denominatorCount, computeFloat);
    mlir::Value result = rewriter.create<mlir::arith::DivFOp>(
        loc, welfordLoop.getResult(2), denominator);
    if (semantic == VulkanKernelRecipe::REDUCE_STDEV) {
      result = rewriter.create<mlir::math::SqrtOp>(loc, result);
    }
    auto outputIndices = logicalIndices(rewriter, loc, outputIndex, Y);
    (void)storeScalar(rewriter, loc, result, Y, outputIndices,
                      inputUnsigned, outputUnsigned);
    rewriter.create<mlir::gpu::TerminatorOp>(loc);
    rewriter.eraseOp(op);
    return mlir::success();
  }

  auto reductionLoop = rewriter.create<mlir::scf::ForOp>(
      loc, zeroIdx, totalN, oneIdx,
      mlir::SmallVector<mlir::Value>{initConst},
      [&](mlir::OpBuilder& ib, mlir::Location iloc,
          mlir::Value inputIndex, mlir::ValueRange iterArgs) {
        llvm::SmallVector<mlir::Value> inIdx(rank);
        mlir::Value remainder = inputIndex;
        for (int64_t d = 0; d < rank; ++d) {
          inIdx[d] = ib.create<mlir::arith::DivUIOp>(
              iloc, remainder, inStrides[d]);
          remainder = ib.create<mlir::arith::RemUIOp>(
              iloc, remainder, inStrides[d]);
        }

        llvm::SmallVector<mlir::Value> outIdx;
        for (int64_t d = 0; d < rank; ++d) {
          if (reduced[static_cast<size_t>(d)] == 0) {
            outIdx.push_back(inIdx[d]);
          } else if (keepDims) {
            outIdx.push_back(zeroIdx);
          }
        }

        mlir::Value flatOutIdx = zeroIdx;
        for (size_t od = 0; od < outStrides.size(); ++od) {
          mlir::Value contribution = ib.create<mlir::arith::MulIOp>(
              iloc, outIdx[od], outStrides[od]);
          flatOutIdx = ib.create<mlir::arith::AddIOp>(
              iloc, flatOutIdx, contribution);
        }

        mlir::Value element = loadAsScalar(
            ib, iloc, X,
            mlir::SmallVector<mlir::Value>(inIdx.begin(), inIdx.end()),
            computeType, inputUnsigned, inputUnsigned);
        element = applyReductionInputTraits(ib, iloc, element);
        mlir::Value combined =
            computeFloat
                ? reduction.combine(ib, iloc, iterArgs[0], element)
                : emitIntegerReductionCombine(
                      ib, iloc, semantic, iterArgs[0], element,
                      magnitudeComparison);
        mlir::Value belongsToOutput = ib.create<mlir::arith::CmpIOp>(
            iloc, mlir::arith::CmpIPredicate::eq, flatOutIdx, outputIndex);
        mlir::Value nextAccumulator = ib.create<mlir::arith::SelectOp>(
            iloc, belongsToOutput, combined, iterArgs[0]);
        ib.create<mlir::scf::YieldOp>(
            iloc, mlir::SmallVector<mlir::Value>{nextAccumulator});
      });

  mlir::Value finalized =
      computeFloat
          ? reduction.finalize(
                rewriter, loc, reductionLoop.getResult(0), reduceCountValue)
          : reductionLoop.getResult(0);
  if (hasVulkanEmitterTrait(
          *emitter, VULKAN_EMITTER_TRAIT_MEAN_FINALIZE)) {
    finalized = rewriter.create<mlir::arith::DivFOp>(
        loc, finalized, reduceCountValue);
  }
  if (hasVulkanEmitterTrait(
          *emitter, VULKAN_EMITTER_TRAIT_SQRT_FINALIZE)) {
    finalized = rewriter.create<mlir::math::SqrtOp>(loc, finalized);
  }
  if (pNormReduction) {
    finalized = rewriter.create<mlir::math::PowFOp>(
        loc, finalized,
        floatConst(rewriter, loc, computeFloat, 1.0 / pNorm));
  }
  auto outputIndices = logicalIndices(rewriter, loc, outputIndex, Y);
  (void)storeScalar(rewriter, loc, finalized, Y, outputIndices,
                    inputUnsigned, outputUnsigned);
  rewriter.create<mlir::gpu::TerminatorOp>(loc);

  rewriter.eraseOp(op);
  return mlir::success();
}

// ─────────────────────────────────────────────────────────────────────────────
//  Pass wrapping populateVulkanLoweringPatterns
// ─────────────────────────────────────────────────────────────────────────────

namespace {

/// An MLIR OperationPass<ModuleOp> that applies all Vulkan op lowering
/// patterns (original three + Wave 1) using partial conversion.
///
/// The pass uses applyPartialConversion (not applyFullConversion) so that ops
/// it does not recognise are left in place for downstream passes to handle.
struct VulkanOpLoweringPass
    : public mlir::PassWrapper<VulkanOpLoweringPass,
                               mlir::OperationPass<mlir::ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VulkanOpLoweringPass)

  llvm::StringRef getArgument() const override {
    return "vulkan-op-lowering";
  }
  llvm::StringRef getDescription() const override {
    return "Lower nd4j matmul / rms_norm / rope / elementwise-binary / "
           "elementwise-unary / softmax / layer_norm / gather / concat / transpose / "
           "reduce_sum / reduce_mean / reduce_max / reduce_min / reduce_prod "
           "linalg ops to SCF/Arith IR for subsequent SPIR-V conversion";
  }

  void runOnOperation() override {
    mlir::MLIRContext* ctx = &getContext();
    mlir::ModuleOp module = getOperation();

    // Build conversion target: mark the ops we want to eliminate as illegal.
    // All other ops remain legal so partial conversion does not fail on them.
    mlir::ConversionTarget target(*ctx);
    target.addLegalDialect<mlir::scf::SCFDialect>();
    target.addLegalDialect<mlir::arith::ArithDialect>();
    target.addLegalDialect<mlir::memref::MemRefDialect>();
    target.addLegalDialect<mlir::func::FuncDialect>();
    target.addLegalDialect<mlir::gpu::GPUDialect>();
    target.addLegalDialect<mlir::math::MathDialect>();
    target.addDynamicallyLegalOp<mlir::linalg::MatmulOp>(
        [](mlir::linalg::MatmulOp op) -> bool {
          const auto* emitter = emitterForOperation(op);
          return emitter == nullptr ||
                 !usesDenseMatrixProductSchedule(*emitter);
        });
    target.addDynamicallyLegalOp<mlir::linalg::BatchMatmulOp>(
        [](mlir::linalg::BatchMatmulOp op) -> bool {
          const auto* emitter = emitterForOperation(op);
          return emitter == nullptr ||
                 !usesDenseMatrixProductSchedule(*emitter);
        });

    target.addDynamicallyLegalOp<mlir::linalg::GenericOp>(
        [](mlir::linalg::GenericOp op) -> bool {
          return emitterForOperation(op) == nullptr;
        });

    // Collect all patterns (original + Wave 1).
    mlir::RewritePatternSet patterns(ctx);
    populateVulkanLoweringPatterns(patterns);

    // Apply partial conversion.
    if (mlir::failed(mlir::applyPartialConversion(module, target,
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // anonymous namespace

// ─────────────────────────────────────────────────────────────────────────────
//  Public API
// ─────────────────────────────────────────────────────────────────────────────

void populateVulkanLoweringPatterns(mlir::RewritePatternSet& patterns) {
  mlir::MLIRContext* ctx = patterns.getContext();
  // Register patterns with benefit 2 so they run before the default linalg
  // lowering patterns (which have benefit 1).

  // ── Original three patterns ───────────────────────────────────────────────
  patterns.add<MatmulToSpirv>(ctx, /*benefit=*/2);
  patterns.add<BatchMatmulToSpirv>(ctx, /*benefit=*/2);
  patterns.add<RmsNormToSpirv>(ctx, /*benefit=*/2);
  patterns.add<RopeToSpirv>(ctx, /*benefit=*/2);
  patterns.add<StructuredComputeToSpirv>(ctx, /*benefit=*/3);

  // ── Wave 1 patterns ───────────────────────────────────────────────────────
  patterns.add<ElementwiseBinaryToSpirv>(ctx, /*benefit=*/2);
  patterns.add<ElementwiseTernaryToSpirv>(ctx, /*benefit=*/2);
  patterns.add<ElementwiseUnaryToSpirv>(ctx, /*benefit=*/2);
  patterns.add<MultiOutputElementwiseToSpirv>(ctx, /*benefit=*/3);
  patterns.add<BatchedMatrixListToSpirv>(ctx, /*benefit=*/3);
  patterns.add<IndexedAccumulationToSpirv>(ctx, /*benefit=*/3);
  patterns.add<IndexedTadMovementToSpirv>(ctx, /*benefit=*/3);
  patterns.add<SoftmaxToSpirv>(ctx, /*benefit=*/2);
  patterns.add<LayerNormToSpirv>(ctx, /*benefit=*/2);

  // ── Wave 2 patterns ───────────────────────────────────────────────────────
  patterns.add<GatherToSpirv>(ctx, /*benefit=*/2);
  patterns.add<ConcatToSpirv>(ctx, /*benefit=*/2);
  patterns.add<TransposeToSpirv>(ctx, /*benefit=*/2);
  patterns.add<DataMovementToSpirv>(ctx, /*benefit=*/3);

  // All catalogued reductions must carry the recorder's nd4j.nd_reduce marker
  // and lower through the real gpu.launch implementation.  The legacy rank-2
  // SCF-only patterns are intentionally not registered: a missing marker must
  // fail conversion instead of silently producing host-side loop IR.
  patterns.add<ReduceNDToSpirv>(ctx, /*benefit=*/3);
}

std::unique_ptr<mlir::Pass> createVulkanOpLoweringPass() {
  return std::make_unique<VulkanOpLoweringPass>();
}

}  // namespace graph
}  // namespace sd

#endif  // HAVE_VULKAN && HAVE_MLIR
