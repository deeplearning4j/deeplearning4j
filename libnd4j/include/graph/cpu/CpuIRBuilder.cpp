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

#if HAVE_MLIR

#include <graph/cpu/CpuIRBuilder.h>
#include <graph/GraphAnalysisUtils.h>
#include <graph/DspDiagnostics.h>
#include <helpers/logger.h>
#include <system/Environment.h>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

#if __has_include("mlir/Dialect/Math/IR/Math.h")
#include "mlir/Dialect/Math/IR/Math.h"
#define SD_CPUIR_HAS_MATH 1
#endif

#include <algorithm>
#include <cctype>
#include <cmath>
#include <sstream>

namespace sd {
namespace graph {

namespace {

std::string toLower(const std::string& s) {
  std::string result = s;
  std::transform(result.begin(), result.end(), result.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return result;
}

}  // namespace

// ═══════════════════════════════════════════════════════════════════════════════
// Static analysis methods
// ═══════════════════════════════════════════════════════════════════════════════

bool CpuIRBuilder::isMlirMappable(const std::string& opName) {
  const auto& table = getOpCategoryTable();
  auto it = table.find(opName);
  if (it == table.end()) {
    DSP_DIAG(BACKEND, "CpuIRBuilder::isMlirMappable: op '%s' not in category table", opName.c_str());
    return false;
  }
  // Accept ALL categories except UNSUPPORTED
  return it->second != TritonOpCategory::UNSUPPORTED;
}

SegmentProfile CpuIRBuilder::profileSegment(NativeSlot* slots, int startSlot, int endSlot,
                                             NDArray** outputSlots, int totalOutputSlots) {
  return GraphAnalysisUtils::profileSegment(slots, startSlot, endSlot, outputSlots, totalOutputSlots);
}

std::unordered_set<int> CpuIRBuilder::computeExternallyVisibleOutputs(
    NativeSlot* slots, int startSlot, int endSlot, int totalSlots) {
  return GraphAnalysisUtils::computeExternallyVisibleOutputs(slots, startSlot, endSlot, totalSlots);
}

SegmentAnalysis CpuIRBuilder::analyzeSegment(NativeSlot* slots, int startSlot, int endSlot,
                                              int totalSlots,
                                              NDArray** externalInputs, int numExternalInputs,
                                              NDArray** outputSlots, int totalOutputSlots) {
  auto profile = profileSegment(slots, startSlot, endSlot, outputSlots, totalOutputSlots);
  SegmentAnalysis analysis;

  // Fill category counts
  analysis.numElementwise = profile.categoryCounts[static_cast<int>(TritonOpCategory::BINARY_ELEMENTWISE)] +
                             profile.categoryCounts[static_cast<int>(TritonOpCategory::UNARY_ELEMENTWISE)];
  analysis.numMatmul = profile.categoryCounts[static_cast<int>(TritonOpCategory::MATMUL)];
  analysis.numReduction = profile.categoryCounts[static_cast<int>(TritonOpCategory::REDUCTION)];
  analysis.numNormalization = profile.categoryCounts[static_cast<int>(TritonOpCategory::NORMALIZATION)];
  analysis.numAttention = profile.categoryCounts[static_cast<int>(TritonOpCategory::FUSED_ATTENTION)];
  analysis.numShapeManip = profile.categoryCounts[static_cast<int>(TritonOpCategory::SHAPE_MANIPULATION)];
  analysis.numDataMovement = profile.categoryCounts[static_cast<int>(TritonOpCategory::DATA_MOVEMENT)];
  analysis.numConstGen = profile.categoryCounts[static_cast<int>(TritonOpCategory::CONSTANT_GENERATION)];
  analysis.numIdentity = profile.categoryCounts[static_cast<int>(TritonOpCategory::IDENTITY)];
  analysis.numCast = profile.categoryCounts[static_cast<int>(TritonOpCategory::CAST)];

  // Count unique input args
  std::unordered_set<int> internalSlotOutputs;
  for (int i = startSlot; i <= endSlot; i++) {
    for (int o = 0; o < slots[i].wiring.numOutputs; o++) {
      internalSlotOutputs.insert(slots[i].wiring.outputSlotIndices[o]);
    }
  }

  std::unordered_set<int> seenInputs;
  int inputArgCount = 0;
  for (int i = startSlot; i <= endSlot; i++) {
    for (int inp = 0; inp < slots[i].wiring.numInputs; inp++) {
      int srcIdx = slots[i].wiring.inputSourceIndices[inp];
      if (seenInputs.count(srcIdx)) continue;
      seenInputs.insert(srcIdx);
      if (srcIdx < 0) {
        int extIdx = -(srcIdx + 1);
        if (extIdx < numExternalInputs && externalInputs && externalInputs[extIdx]) inputArgCount++;
      } else if (!internalSlotOutputs.count(srcIdx)) {
        if (srcIdx < totalOutputSlots && outputSlots && outputSlots[srcIdx]) inputArgCount++;
      }
    }
  }

  auto externalOutputs = computeExternallyVisibleOutputs(slots, startSlot, endSlot, totalSlots);
  int outputArgCount = static_cast<int>(externalOutputs.size());

  analysis.totalInputArgs = inputArgCount;
  analysis.totalOutputArgs = outputArgCount;
  analysis.totalArgs = inputArgCount + outputArgCount + 1;

  // Classify pattern — accept everything that has no UNSUPPORTED ops
  bool hasUnsupported = profile.categoryCounts[static_cast<int>(TritonOpCategory::UNSUPPORTED)] > 0;

  if (hasUnsupported) {
    analysis.canCompile = false;
    analysis.failureReason = "Segment contains unsupported ops";
  } else if (analysis.numAttention > 0) {
    // Fused attention requires specialized kernel structure — skip for now
    analysis.canCompile = false;
    analysis.failureReason = "Fused attention not yet supported on CPU MLIR";
  } else {
    analysis.canCompile = true;

    // Classify the pattern
    if (analysis.numMatmul > 0 && analysis.numElementwise > 0) {
      analysis.pattern = SegmentKernelPattern::MATMUL_EPILOGUE;
    } else if (analysis.numMatmul > 0) {
      analysis.pattern = SegmentKernelPattern::MATMUL_2D;
    } else if (analysis.numNormalization > 0) {
      analysis.pattern = SegmentKernelPattern::NORMALIZATION;
    } else if (analysis.numReduction > 0) {
      analysis.pattern = SegmentKernelPattern::REDUCTION_1D;
    } else if (analysis.numDataMovement > 0 || analysis.numShapeManip > 0 ||
               analysis.numConstGen > 0) {
      analysis.pattern = SegmentKernelPattern::WHOLE_GRAPH;
    } else {
      analysis.pattern = SegmentKernelPattern::ELEMENTWISE_1D;
    }
  }

  DSP_DIAG(COMPILE, "CpuIRBuilder::analyzeSegment [%d-%d]: %d ops, canCompile=%s, "
            "elem=%d matmul=%d reduce=%d norm=%d shape=%d data=%d const=%d",
            startSlot, endSlot, profile.totalOps,
            analysis.canCompile ? "true" : "false",
            analysis.numElementwise, analysis.numMatmul, analysis.numReduction,
            analysis.numNormalization, analysis.numShapeManip, analysis.numDataMovement,
            analysis.numConstGen);

  return analysis;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Scalar emit helpers (inside scf.for body)
// ═══════════════════════════════════════════════════════════════════════════════

mlir::Value CpuIRBuilder::emitBinaryElementwise(mlir::OpBuilder& builder, mlir::Location loc,
                                                  const std::string& opName,
                                                  mlir::Value lhs, mlir::Value rhs,
                                                  mlir::Type resultType) {
  std::string lower = toLower(opName);

  if (lower == "add") return builder.create<mlir::arith::AddFOp>(loc, lhs, rhs);
  if (lower == "subtract" || lower == "sub") return builder.create<mlir::arith::SubFOp>(loc, lhs, rhs);
  if (lower == "multiply" || lower == "mul") return builder.create<mlir::arith::MulFOp>(loc, lhs, rhs);
  if (lower == "divide" || lower == "div" || lower == "realdiv")
    return builder.create<mlir::arith::DivFOp>(loc, lhs, rhs);
  if (lower == "minimum" || lower == "min" || lower == "min_pairwise" || lower == "minpairwise")
    return builder.create<mlir::arith::MinimumFOp>(loc, lhs, rhs);
  if (lower == "maximum" || lower == "max" || lower == "max_pairwise" || lower == "maxpairwise")
    return builder.create<mlir::arith::MaximumFOp>(loc, lhs, rhs);
#ifdef SD_CPUIR_HAS_MATH
  if (lower == "pow") return builder.create<mlir::math::PowFOp>(loc, lhs, rhs);
  if (lower == "atan2") return builder.create<mlir::math::Atan2Op>(loc, lhs, rhs);
#endif
  if (lower == "floormod" || lower == "mod") {
    auto div = builder.create<mlir::arith::DivFOp>(loc, lhs, rhs);
#ifdef SD_CPUIR_HAS_MATH
    auto floorDiv = builder.create<mlir::math::FloorOp>(loc, div);
#else
    auto floorDiv = div;  // Approximate if no math dialect
#endif
    auto mul = builder.create<mlir::arith::MulFOp>(loc, floorDiv, rhs);
    return builder.create<mlir::arith::SubFOp>(loc, lhs, mul);
  }
  if (lower == "floordiv") {
    auto div = builder.create<mlir::arith::DivFOp>(loc, lhs, rhs);
#ifdef SD_CPUIR_HAS_MATH
    return builder.create<mlir::math::FloorOp>(loc, div);
#else
    return div;
#endif
  }
  if (lower == "reversedivide") return builder.create<mlir::arith::DivFOp>(loc, rhs, lhs);
  if (lower == "reversesubtract") return builder.create<mlir::arith::SubFOp>(loc, rhs, lhs);
  if (lower == "squaredsubtract") {
    auto diff = builder.create<mlir::arith::SubFOp>(loc, lhs, rhs);
    return builder.create<mlir::arith::MulFOp>(loc, diff, diff);
  }
  if (lower == "multiply_no_nan" || lower == "multiplynonan") {
    auto product = builder.create<mlir::arith::MulFOp>(loc, lhs, rhs);
    auto zero = builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(resultType, 0.0));
    auto isZero = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OEQ, rhs, zero);
    return builder.create<mlir::arith::SelectOp>(loc, isZero, zero, product);
  }
  if (lower == "swish_mul" || lower == "swishmul") {
    auto negX = builder.create<mlir::arith::NegFOp>(loc, lhs);
#ifdef SD_CPUIR_HAS_MATH
    auto expNeg = builder.create<mlir::math::ExpOp>(loc, negX);
#else
    auto expNeg = negX;
#endif
    auto one = builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(resultType, 1.0));
    auto denom = builder.create<mlir::arith::AddFOp>(loc, one, expNeg);
    auto sigmoid = builder.create<mlir::arith::DivFOp>(loc, one, denom);
    auto xSig = builder.create<mlir::arith::MulFOp>(loc, lhs, sigmoid);
    return builder.create<mlir::arith::MulFOp>(loc, xSig, rhs);
  }

  DSP_DIAG(FALLBACK, "CpuIRBuilder: unhandled binary op '%s', falling back to add", opName.c_str());
  return builder.create<mlir::arith::AddFOp>(loc, lhs, rhs);
}

mlir::Value CpuIRBuilder::emitUnaryElementwise(mlir::OpBuilder& builder, mlir::Location loc,
                                                 const std::string& opName,
                                                 mlir::Value input, mlir::Type resultType,
                                                 const double* tArgs, int numTArgs) {
  std::string lower = toLower(opName);
  auto constF = [&](double val) -> mlir::Value {
    return builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(resultType, val));
  };

  if (lower == "relu") {
    return builder.create<mlir::arith::MaximumFOp>(loc, input, constF(0.0));
  }
  if (lower == "sigmoid") {
    auto negX = builder.create<mlir::arith::NegFOp>(loc, input);
#ifdef SD_CPUIR_HAS_MATH
    auto expNeg = builder.create<mlir::math::ExpOp>(loc, negX);
#else
    auto expNeg = negX;
#endif
    auto one = constF(1.0);
    auto denom = builder.create<mlir::arith::AddFOp>(loc, one, expNeg);
    return builder.create<mlir::arith::DivFOp>(loc, one, denom);
  }
#ifdef SD_CPUIR_HAS_MATH
  if (lower == "tanh") return builder.create<mlir::math::TanhOp>(loc, input);
  if (lower == "exp") return builder.create<mlir::math::ExpOp>(loc, input);
  if (lower == "log") return builder.create<mlir::math::LogOp>(loc, input);
  if (lower == "log1p") return builder.create<mlir::math::Log1pOp>(loc, input);
  if (lower == "abs") return builder.create<mlir::math::AbsFOp>(loc, input);
  if (lower == "sqrt") return builder.create<mlir::math::SqrtOp>(loc, input);
  if (lower == "rsqrt") return builder.create<mlir::math::RsqrtOp>(loc, input);
  if (lower == "sin") return builder.create<mlir::math::SinOp>(loc, input);
  if (lower == "cos") return builder.create<mlir::math::CosOp>(loc, input);
  if (lower == "ceil") return builder.create<mlir::math::CeilOp>(loc, input);
  if (lower == "floor") return builder.create<mlir::math::FloorOp>(loc, input);
  if (lower == "round") return builder.create<mlir::math::RoundOp>(loc, input);
  if (lower == "erf") return builder.create<mlir::math::ErfOp>(loc, input);
#endif
  if (lower == "neg") return builder.create<mlir::arith::NegFOp>(loc, input);
  if (lower == "square") return builder.create<mlir::arith::MulFOp>(loc, input, input);
  if (lower == "reciprocal") {
    return builder.create<mlir::arith::DivFOp>(loc, constF(1.0), input);
  }
  if (lower == "sign") {
    auto zero = constF(0.0);
    auto pos = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OGT, input, zero);
    auto neg = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OLT, input, zero);
    auto posVal = builder.create<mlir::arith::SelectOp>(loc, pos, constF(1.0), zero);
    return builder.create<mlir::arith::SelectOp>(loc, neg, constF(-1.0), posVal);
  }
  if (lower == "gelu") {
#ifdef SD_CPUIR_HAS_MATH
    auto invSqrt2 = constF(0.7071067811865476);
    auto scaled = builder.create<mlir::arith::MulFOp>(loc, input, invSqrt2);
    auto erfVal = builder.create<mlir::math::ErfOp>(loc, scaled);
    auto erfPlus1 = builder.create<mlir::arith::AddFOp>(loc, constF(1.0), erfVal);
    auto halfX = builder.create<mlir::arith::MulFOp>(loc, constF(0.5), input);
    return builder.create<mlir::arith::MulFOp>(loc, halfX, erfPlus1);
#else
    return input;
#endif
  }
  if (lower == "silu" || lower == "swish") {
    auto negX = builder.create<mlir::arith::NegFOp>(loc, input);
#ifdef SD_CPUIR_HAS_MATH
    auto expNeg = builder.create<mlir::math::ExpOp>(loc, negX);
#else
    auto expNeg = negX;
#endif
    auto one = constF(1.0);
    auto denom = builder.create<mlir::arith::AddFOp>(loc, one, expNeg);
    auto sigmoid = builder.create<mlir::arith::DivFOp>(loc, one, denom);
    return builder.create<mlir::arith::MulFOp>(loc, input, sigmoid);
  }
  if (lower == "mish") {
#ifdef SD_CPUIR_HAS_MATH
    auto expX = builder.create<mlir::math::ExpOp>(loc, input);
    auto sp = builder.create<mlir::arith::AddFOp>(loc, constF(1.0), expX);
    auto logSp = builder.create<mlir::math::LogOp>(loc, sp);
    auto tanhSp = builder.create<mlir::math::TanhOp>(loc, logSp);
    return builder.create<mlir::arith::MulFOp>(loc, input, tanhSp);
#else
    return input;
#endif
  }
  if (lower == "softplus") {
#ifdef SD_CPUIR_HAS_MATH
    auto expX = builder.create<mlir::math::ExpOp>(loc, input);
    auto sum = builder.create<mlir::arith::AddFOp>(loc, constF(1.0), expX);
    return builder.create<mlir::math::LogOp>(loc, sum);
#else
    return input;
#endif
  }
  if (lower == "softsign") {
#ifdef SD_CPUIR_HAS_MATH
    auto absX = builder.create<mlir::math::AbsFOp>(loc, input);
#else
    auto absX = input;
#endif
    auto denom = builder.create<mlir::arith::AddFOp>(loc, constF(1.0), absX);
    return builder.create<mlir::arith::DivFOp>(loc, input, denom);
  }
  if (lower == "elu") {
    double alpha = (numTArgs > 0) ? tArgs[0] : 1.0;
    auto zero = constF(0.0);
#ifdef SD_CPUIR_HAS_MATH
    auto expX = builder.create<mlir::math::ExpOp>(loc, input);
#else
    auto expX = input;
#endif
    auto expMinus1 = builder.create<mlir::arith::SubFOp>(loc, expX, constF(1.0));
    auto negBranch = builder.create<mlir::arith::MulFOp>(loc, constF(alpha), expMinus1);
    auto cmp = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OGE, input, zero);
    return builder.create<mlir::arith::SelectOp>(loc, cmp, input, negBranch);
  }
  if (lower == "selu") {
    double lambda = 1.0507009873554804934193349852946;
    double alpha = 1.6732632423543772848170429916717;
    auto zero = constF(0.0);
#ifdef SD_CPUIR_HAS_MATH
    auto expX = builder.create<mlir::math::ExpOp>(loc, input);
#else
    auto expX = input;
#endif
    auto expMinus1 = builder.create<mlir::arith::SubFOp>(loc, expX, constF(1.0));
    auto negBranch = builder.create<mlir::arith::MulFOp>(loc, constF(alpha), expMinus1);
    auto cmp = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OGE, input, zero);
    auto selected = builder.create<mlir::arith::SelectOp>(loc, cmp, input, negBranch);
    return builder.create<mlir::arith::MulFOp>(loc, constF(lambda), selected);
  }
  if (lower == "leakyrelu") {
    double alpha = (numTArgs > 0) ? tArgs[0] : 0.01;
    auto zero = constF(0.0);
    auto scaled = builder.create<mlir::arith::MulFOp>(loc, constF(alpha), input);
    auto cmp = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OGE, input, zero);
    return builder.create<mlir::arith::SelectOp>(loc, cmp, input, scaled);
  }
  if (lower == "relu6") {
    auto clampLow = builder.create<mlir::arith::MaximumFOp>(loc, input, constF(0.0));
    return builder.create<mlir::arith::MinimumFOp>(loc, clampLow, constF(6.0));
  }
  if (lower == "hard_sigmoid" || lower == "hardsigmoid") {
    auto scaled = builder.create<mlir::arith::MulFOp>(loc, input, constF(0.2));
    auto shifted = builder.create<mlir::arith::AddFOp>(loc, scaled, constF(0.5));
    auto clampLow = builder.create<mlir::arith::MaximumFOp>(loc, shifted, constF(0.0));
    return builder.create<mlir::arith::MinimumFOp>(loc, clampLow, constF(1.0));
  }
  if (lower == "hardtanh") {
    auto clampLow = builder.create<mlir::arith::MaximumFOp>(loc, input, constF(-1.0));
    return builder.create<mlir::arith::MinimumFOp>(loc, clampLow, constF(1.0));
  }
  if (lower == "clamp" || lower == "clipbyvalue" || lower == "clip_by_value") {
    double lo = (numTArgs > 0) ? tArgs[0] : -3.4028235e+38;
    double hi = (numTArgs > 1) ? tArgs[1] : 3.4028235e+38;
    auto clampLow = builder.create<mlir::arith::MaximumFOp>(loc, input, constF(lo));
    return builder.create<mlir::arith::MinimumFOp>(loc, clampLow, constF(hi));
  }
  if (lower == "erfc") {
#ifdef SD_CPUIR_HAS_MATH
    auto erfVal = builder.create<mlir::math::ErfOp>(loc, input);
    return builder.create<mlir::arith::SubFOp>(loc, constF(1.0), erfVal);
#else
    return input;
#endif
  }
  // Scalar ops
  if (lower == "add_scalar") {
    return builder.create<mlir::arith::AddFOp>(loc, input, constF(numTArgs > 0 ? tArgs[0] : 0.0));
  }
  if (lower == "subtract_scalar") {
    return builder.create<mlir::arith::SubFOp>(loc, input, constF(numTArgs > 0 ? tArgs[0] : 0.0));
  }
  if (lower == "multiply_scalar") {
    return builder.create<mlir::arith::MulFOp>(loc, input, constF(numTArgs > 0 ? tArgs[0] : 1.0));
  }
  if (lower == "divide_scalar") {
    return builder.create<mlir::arith::DivFOp>(loc, input, constF(numTArgs > 0 ? tArgs[0] : 1.0));
  }
  // Identity
  if (lower == "identity" || lower == "assign") return input;
  // Cast (f32→f32 is identity for now)
  if (lower == "cast") return input;

  DSP_DIAG(FALLBACK, "CpuIRBuilder: unhandled unary op '%s', returning input", opName.c_str());
  return input;
}

mlir::Value CpuIRBuilder::emitComparisonOp(mlir::OpBuilder& builder, mlir::Location loc,
                                            const std::string& opName,
                                            mlir::Value lhs, mlir::Value rhs) {
  std::string lower = toLower(opName);
  mlir::arith::CmpFPredicate pred;
  if (lower == "greater") pred = mlir::arith::CmpFPredicate::OGT;
  else if (lower == "greater_equal" || lower == "greaterequal") pred = mlir::arith::CmpFPredicate::OGE;
  else if (lower == "less") pred = mlir::arith::CmpFPredicate::OLT;
  else if (lower == "less_equal" || lower == "lessequal") pred = mlir::arith::CmpFPredicate::OLE;
  else if (lower == "equals") pred = mlir::arith::CmpFPredicate::OEQ;
  else if (lower == "not_equals" || lower == "notequals") pred = mlir::arith::CmpFPredicate::ONE;
  else pred = mlir::arith::CmpFPredicate::OEQ;
  return builder.create<mlir::arith::CmpFOp>(loc, pred, lhs, rhs);
}

mlir::Value CpuIRBuilder::emitTernaryOp(mlir::OpBuilder& builder, mlir::Location loc,
                                          const std::string& opName,
                                          mlir::Value cond, mlir::Value ifTrue, mlir::Value ifFalse) {
  return builder.create<mlir::arith::SelectOp>(loc, cond, ifTrue, ifFalse);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Full-array emit helpers (emit their own loop nests)
// ═══════════════════════════════════════════════════════════════════════════════

void CpuIRBuilder::emitReductionOp(mlir::OpBuilder& builder, mlir::Location loc,
                                    const std::string& opName,
                                    mlir::Value inputMemref, mlir::Value outputMemref,
                                    int64_t nElements, mlir::Type elemType) {
  std::string lower = toLower(opName);
  auto indexType = mlir::IndexType::get(builder.getContext());
  auto zero = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
  auto one = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
  auto n = builder.create<mlir::arith::ConstantIndexOp>(loc, nElements);

  // Determine initial accumulator value and combiner
  double initVal = 0.0;
  bool isMax = false, isMin = false, isProd = false;
  if (lower == "reduce_max" || lower == "max" || lower == "normmax") {
    initVal = -3.4028235e+38;
    isMax = true;
  } else if (lower == "reduce_min" || lower == "min") {
    initVal = 3.4028235e+38;
    isMin = true;
  } else if (lower == "reduce_prod" || lower == "prod") {
    initVal = 1.0;
    isProd = true;
  }
  // All others (sum, mean, norm1, norm2, variance, stdev, logsumexp) init to 0

  auto initAcc = builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(elemType, initVal));

  // scf.for with carried accumulator
  auto forOp = builder.create<mlir::scf::ForOp>(
      loc, zero, n, one, mlir::ValueRange{initAcc.getResult()});
  {
    auto& body = forOp.getRegion().front();
    builder.setInsertionPointToStart(&body);
    auto iv = body.getArgument(0);
    auto acc = body.getArgument(1);

    auto elem = builder.create<mlir::memref::LoadOp>(loc, inputMemref, mlir::ValueRange{iv});

    // Pre-process element based on op type
    mlir::Value processed = elem;
    if (lower == "reduce_norm1" || lower == "norm1") {
#ifdef SD_CPUIR_HAS_MATH
      processed = builder.create<mlir::math::AbsFOp>(loc, elem);
#endif
    } else if (lower == "reduce_norm2" || lower == "norm2") {
      processed = builder.create<mlir::arith::MulFOp>(loc, elem, elem);
    }

    // Combine
    mlir::Value newAcc;
    if (isMax) {
      newAcc = builder.create<mlir::arith::MaximumFOp>(loc, acc, processed);
    } else if (isMin) {
      newAcc = builder.create<mlir::arith::MinimumFOp>(loc, acc, processed);
    } else if (isProd) {
      newAcc = builder.create<mlir::arith::MulFOp>(loc, acc, processed);
    } else {
      newAcc = builder.create<mlir::arith::AddFOp>(loc, acc, processed);
    }

    builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{newAcc});
  }

  builder.setInsertionPointAfter(forOp);
  mlir::Value result = forOp.getResult(0);

  // Post-process
  if (lower == "reduce_mean" || lower == "mean") {
    auto nf = builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(elemType, static_cast<double>(nElements)));
    result = builder.create<mlir::arith::DivFOp>(loc, result, nf);
  } else if (lower == "reduce_norm2" || lower == "norm2") {
#ifdef SD_CPUIR_HAS_MATH
    result = builder.create<mlir::math::SqrtOp>(loc, result);
#endif
  } else if (lower == "reduce_stdev") {
#ifdef SD_CPUIR_HAS_MATH
    result = builder.create<mlir::math::SqrtOp>(loc, result);
#endif
  } else if (lower == "reduce_logsumexp" || lower == "reducelogsumexp") {
#ifdef SD_CPUIR_HAS_MATH
    result = builder.create<mlir::math::LogOp>(loc, result);
#endif
  }

  // Store scalar result
  builder.create<mlir::memref::StoreOp>(loc, result, outputMemref, mlir::ValueRange{zero});
}

void CpuIRBuilder::emitNormalizationOp(mlir::OpBuilder& builder, mlir::Location loc,
                                         const std::string& opName,
                                         mlir::Value inputMemref, mlir::Value outputMemref,
                                         int64_t nElements, mlir::Type elemType) {
  std::string lower = toLower(opName);
  auto indexType = mlir::IndexType::get(builder.getContext());
  auto zero = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
  auto one = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
  auto n = builder.create<mlir::arith::ConstantIndexOp>(loc, nElements);
  auto nf = builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(elemType, static_cast<double>(nElements)));

  auto makeReduce = [&](mlir::Value initVal, bool useMax) -> mlir::Value {
    auto forOp = builder.create<mlir::scf::ForOp>(
        loc, zero, n, one, mlir::ValueRange{initVal});
    {
      auto& body = forOp.getRegion().front();
      builder.setInsertionPointToStart(&body);
      auto iv = body.getArgument(0);
      auto acc = body.getArgument(1);
      auto elem = builder.create<mlir::memref::LoadOp>(loc, inputMemref, mlir::ValueRange{iv});
      mlir::Value newAcc;
      if (useMax) {
        newAcc = builder.create<mlir::arith::MaximumFOp>(loc, acc, elem);
      } else {
        newAcc = builder.create<mlir::arith::AddFOp>(loc, acc, elem);
      }
      builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{newAcc});
    }
    builder.setInsertionPointAfter(forOp);
    return forOp.getResult(0);
  };

  if (lower == "softmax") {
    // Pass 1: max
    auto negInf = builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(elemType, -3.4028235e+38));
    auto maxVal = makeReduce(negInf, true);

    // Pass 2: exp(x - max) and sum
    auto zeroF = builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(elemType, 0.0));
    auto sumForOp = builder.create<mlir::scf::ForOp>(
        loc, zero, n, one, mlir::ValueRange{zeroF.getResult()});
    {
      auto& body = sumForOp.getRegion().front();
      builder.setInsertionPointToStart(&body);
      auto iv = body.getArgument(0);
      auto acc = body.getArgument(1);
      auto elem = builder.create<mlir::memref::LoadOp>(loc, inputMemref, mlir::ValueRange{iv});
      auto shifted = builder.create<mlir::arith::SubFOp>(loc, elem, maxVal);
#ifdef SD_CPUIR_HAS_MATH
      auto expVal = builder.create<mlir::math::ExpOp>(loc, shifted);
#else
      auto expVal = shifted;
#endif
      // Store exp to output as temp
      builder.create<mlir::memref::StoreOp>(loc, expVal, outputMemref, mlir::ValueRange{iv});
      auto newAcc = builder.create<mlir::arith::AddFOp>(loc, acc, expVal);
      builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{newAcc});
    }
    builder.setInsertionPointAfter(sumForOp);
    auto sumVal = sumForOp.getResult(0);

    // Pass 3: normalize
    auto normForOp = builder.create<mlir::scf::ForOp>(loc, zero, n, one);
    {
      auto& body = normForOp.getRegion().front();
      builder.setInsertionPointToStart(&body);
      auto iv = body.getArgument(0);
      auto expVal = builder.create<mlir::memref::LoadOp>(loc, outputMemref, mlir::ValueRange{iv});
      auto result = builder.create<mlir::arith::DivFOp>(loc, expVal, sumVal);
      builder.create<mlir::memref::StoreOp>(loc, result, outputMemref, mlir::ValueRange{iv});
      builder.create<mlir::scf::YieldOp>(loc);
    }
    builder.setInsertionPointAfter(normForOp);

  } else if (lower == "log_softmax" || lower == "logsoftmax") {
    auto negInf = builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(elemType, -3.4028235e+38));
    auto maxVal = makeReduce(negInf, true);

    auto zeroF = builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(elemType, 0.0));
    auto sumForOp = builder.create<mlir::scf::ForOp>(
        loc, zero, n, one, mlir::ValueRange{zeroF.getResult()});
    {
      auto& body = sumForOp.getRegion().front();
      builder.setInsertionPointToStart(&body);
      auto iv = body.getArgument(0);
      auto acc = body.getArgument(1);
      auto elem = builder.create<mlir::memref::LoadOp>(loc, inputMemref, mlir::ValueRange{iv});
      auto shifted = builder.create<mlir::arith::SubFOp>(loc, elem, maxVal);
#ifdef SD_CPUIR_HAS_MATH
      auto expVal = builder.create<mlir::math::ExpOp>(loc, shifted);
#else
      auto expVal = shifted;
#endif
      auto newAcc = builder.create<mlir::arith::AddFOp>(loc, acc, expVal);
      builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{newAcc});
    }
    builder.setInsertionPointAfter(sumForOp);
    auto sumVal = sumForOp.getResult(0);
#ifdef SD_CPUIR_HAS_MATH
    auto logSum = builder.create<mlir::math::LogOp>(loc, sumVal);
#else
    auto logSum = sumVal;
#endif

    auto outForOp = builder.create<mlir::scf::ForOp>(loc, zero, n, one);
    {
      auto& body = outForOp.getRegion().front();
      builder.setInsertionPointToStart(&body);
      auto iv = body.getArgument(0);
      auto elem = builder.create<mlir::memref::LoadOp>(loc, inputMemref, mlir::ValueRange{iv});
      auto shifted = builder.create<mlir::arith::SubFOp>(loc, elem, maxVal);
      auto result = builder.create<mlir::arith::SubFOp>(loc, shifted, logSum);
      builder.create<mlir::memref::StoreOp>(loc, result, outputMemref, mlir::ValueRange{iv});
      builder.create<mlir::scf::YieldOp>(loc);
    }
    builder.setInsertionPointAfter(outForOp);

  } else if (lower == "rms_norm" || lower == "rmsnorm") {
    // mean_sq = sum(x^2) / N; result = x * rsqrt(mean_sq + eps)
    auto zeroF = builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(elemType, 0.0));
    auto sumSqFor = builder.create<mlir::scf::ForOp>(
        loc, zero, n, one, mlir::ValueRange{zeroF.getResult()});
    {
      auto& body = sumSqFor.getRegion().front();
      builder.setInsertionPointToStart(&body);
      auto iv = body.getArgument(0);
      auto acc = body.getArgument(1);
      auto elem = builder.create<mlir::memref::LoadOp>(loc, inputMemref, mlir::ValueRange{iv});
      auto sq = builder.create<mlir::arith::MulFOp>(loc, elem, elem);
      auto newAcc = builder.create<mlir::arith::AddFOp>(loc, acc, sq);
      builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{newAcc});
    }
    builder.setInsertionPointAfter(sumSqFor);
    auto sumSq = sumSqFor.getResult(0);
    auto meanSq = builder.create<mlir::arith::DivFOp>(loc, sumSq, nf);
    auto eps = builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(elemType, 1e-6));
    auto meanSqEps = builder.create<mlir::arith::AddFOp>(loc, meanSq, eps);
#ifdef SD_CPUIR_HAS_MATH
    auto rsqrtVal = builder.create<mlir::math::RsqrtOp>(loc, meanSqEps);
#else
    auto rsqrtVal = meanSqEps;
#endif

    auto outFor = builder.create<mlir::scf::ForOp>(loc, zero, n, one);
    {
      auto& body = outFor.getRegion().front();
      builder.setInsertionPointToStart(&body);
      auto iv = body.getArgument(0);
      auto elem = builder.create<mlir::memref::LoadOp>(loc, inputMemref, mlir::ValueRange{iv});
      auto result = builder.create<mlir::arith::MulFOp>(loc, elem, rsqrtVal);
      builder.create<mlir::memref::StoreOp>(loc, result, outputMemref, mlir::ValueRange{iv});
      builder.create<mlir::scf::YieldOp>(loc);
    }
    builder.setInsertionPointAfter(outFor);

  } else if (lower == "layer_norm" || lower == "layernorm") {
    // mean; centered; var; result = centered * rsqrt(var + eps)
    auto zeroF = builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(elemType, 0.0));
    auto sumFor = builder.create<mlir::scf::ForOp>(
        loc, zero, n, one, mlir::ValueRange{zeroF.getResult()});
    {
      auto& body = sumFor.getRegion().front();
      builder.setInsertionPointToStart(&body);
      auto iv = body.getArgument(0);
      auto acc = body.getArgument(1);
      auto elem = builder.create<mlir::memref::LoadOp>(loc, inputMemref, mlir::ValueRange{iv});
      auto newAcc = builder.create<mlir::arith::AddFOp>(loc, acc, elem);
      builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{newAcc});
    }
    builder.setInsertionPointAfter(sumFor);
    auto mean = builder.create<mlir::arith::DivFOp>(loc, sumFor.getResult(0), nf);

    // Variance
    auto varFor = builder.create<mlir::scf::ForOp>(
        loc, zero, n, one, mlir::ValueRange{zeroF.getResult()});
    {
      auto& body = varFor.getRegion().front();
      builder.setInsertionPointToStart(&body);
      auto iv = body.getArgument(0);
      auto acc = body.getArgument(1);
      auto elem = builder.create<mlir::memref::LoadOp>(loc, inputMemref, mlir::ValueRange{iv});
      auto centered = builder.create<mlir::arith::SubFOp>(loc, elem, mean);
      auto sq = builder.create<mlir::arith::MulFOp>(loc, centered, centered);
      auto newAcc = builder.create<mlir::arith::AddFOp>(loc, acc, sq);
      builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{newAcc});
    }
    builder.setInsertionPointAfter(varFor);
    auto variance = builder.create<mlir::arith::DivFOp>(loc, varFor.getResult(0), nf);
    auto eps = builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(elemType, 1e-5));
    auto varEps = builder.create<mlir::arith::AddFOp>(loc, variance, eps);
#ifdef SD_CPUIR_HAS_MATH
    auto rsqrtVal = builder.create<mlir::math::RsqrtOp>(loc, varEps);
#else
    auto rsqrtVal = varEps;
#endif

    auto outFor = builder.create<mlir::scf::ForOp>(loc, zero, n, one);
    {
      auto& body = outFor.getRegion().front();
      builder.setInsertionPointToStart(&body);
      auto iv = body.getArgument(0);
      auto elem = builder.create<mlir::memref::LoadOp>(loc, inputMemref, mlir::ValueRange{iv});
      auto centered = builder.create<mlir::arith::SubFOp>(loc, elem, mean);
      auto result = builder.create<mlir::arith::MulFOp>(loc, centered, rsqrtVal);
      builder.create<mlir::memref::StoreOp>(loc, result, outputMemref, mlir::ValueRange{iv});
      builder.create<mlir::scf::YieldOp>(loc);
    }
    builder.setInsertionPointAfter(outFor);

  } else {
    // batch_norm, normalize_moments: just copy input to output
    auto copyFor = builder.create<mlir::scf::ForOp>(loc, zero, n, one);
    {
      auto& body = copyFor.getRegion().front();
      builder.setInsertionPointToStart(&body);
      auto iv = body.getArgument(0);
      auto elem = builder.create<mlir::memref::LoadOp>(loc, inputMemref, mlir::ValueRange{iv});
      builder.create<mlir::memref::StoreOp>(loc, elem, outputMemref, mlir::ValueRange{iv});
      builder.create<mlir::scf::YieldOp>(loc);
    }
    builder.setInsertionPointAfter(copyFor);
  }
}

void CpuIRBuilder::emitMatmulOp(mlir::OpBuilder& builder, mlir::Location loc,
                                  mlir::Value aMemref, mlir::Value bMemref, mlir::Value cMemref,
                                  int64_t M, int64_t N, int64_t K, mlir::Type elemType) {
  auto zero = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
  auto one = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
  auto mBound = builder.create<mlir::arith::ConstantIndexOp>(loc, M);
  auto nBound = builder.create<mlir::arith::ConstantIndexOp>(loc, N);
  auto kBound = builder.create<mlir::arith::ConstantIndexOp>(loc, K);
  auto zeroF = builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(elemType, 0.0));

  // for i in M
  auto iLoop = builder.create<mlir::scf::ForOp>(loc, zero, mBound, one);
  {
    auto& iBody = iLoop.getRegion().front();
    builder.setInsertionPointToStart(&iBody);
    auto i = iBody.getArgument(0);

    // for j in N
    auto jLoop = builder.create<mlir::scf::ForOp>(loc, zero, nBound, one);
    {
      auto& jBody = jLoop.getRegion().front();
      builder.setInsertionPointToStart(&jBody);
      auto j = jBody.getArgument(0);

      // for k in K, accumulate
      auto kLoop = builder.create<mlir::scf::ForOp>(
          loc, zero, kBound, one, mlir::ValueRange{zeroF.getResult()});
      {
        auto& kBody = kLoop.getRegion().front();
        builder.setInsertionPointToStart(&kBody);
        auto k = kBody.getArgument(0);
        auto acc = kBody.getArgument(1);

        // A is [M, K] stored row-major: A[i*K + k]
        auto aIdx = builder.create<mlir::arith::AddIOp>(loc,
            builder.create<mlir::arith::MulIOp>(loc, i, kBound), k);
        auto aVal = builder.create<mlir::memref::LoadOp>(loc, aMemref, mlir::ValueRange{aIdx});

        // B is [K, N] stored row-major: B[k*N + j]
        auto bIdx = builder.create<mlir::arith::AddIOp>(loc,
            builder.create<mlir::arith::MulIOp>(loc, k, nBound), j);
        auto bVal = builder.create<mlir::memref::LoadOp>(loc, bMemref, mlir::ValueRange{bIdx});

        auto prod = builder.create<mlir::arith::MulFOp>(loc, aVal, bVal);
        auto newAcc = builder.create<mlir::arith::AddFOp>(loc, acc, prod);
        builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{newAcc});
      }
      builder.setInsertionPointAfter(kLoop);

      // C[i*N + j] = accumulator
      auto cIdx = builder.create<mlir::arith::AddIOp>(loc,
          builder.create<mlir::arith::MulIOp>(loc, i, nBound), j);
      builder.create<mlir::memref::StoreOp>(loc, kLoop.getResult(0), cMemref, mlir::ValueRange{cIdx});

      builder.create<mlir::scf::YieldOp>(loc);
    }
    builder.setInsertionPointAfter(jLoop);
    builder.create<mlir::scf::YieldOp>(loc);
  }
  builder.setInsertionPointAfter(iLoop);
}

void CpuIRBuilder::emitGatherOp(mlir::OpBuilder& builder, mlir::Location loc,
                                  mlir::Value dataMemref, mlir::Value indicesMemref,
                                  mlir::Value outputMemref,
                                  int64_t nElements, int gatherAxis,
                                  const std::vector<LongType>& dataShape,
                                  mlir::Type elemType) {
  auto zero = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
  auto one = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
  auto n = builder.create<mlir::arith::ConstantIndexOp>(loc, nElements);

  // Simplified 1D gather: output[i] = data[indices[i]]
  auto loop = builder.create<mlir::scf::ForOp>(loc, zero, n, one);
  {
    auto& body = loop.getRegion().front();
    builder.setInsertionPointToStart(&body);
    auto iv = body.getArgument(0);

    // Load index (as f32, truncate to index)
    auto idxF = builder.create<mlir::memref::LoadOp>(loc, indicesMemref, mlir::ValueRange{iv});
    auto idxI = builder.create<mlir::arith::FPToSIOp>(loc, builder.getI64Type(), idxF);
    auto idxIdx = builder.create<mlir::arith::IndexCastOp>(loc, mlir::IndexType::get(builder.getContext()), idxI);

    auto gathered = builder.create<mlir::memref::LoadOp>(loc, dataMemref, mlir::ValueRange{idxIdx});
    builder.create<mlir::memref::StoreOp>(loc, gathered, outputMemref, mlir::ValueRange{iv});
    builder.create<mlir::scf::YieldOp>(loc);
  }
  builder.setInsertionPointAfter(loop);
}

void CpuIRBuilder::emitConcatOp(mlir::OpBuilder& builder, mlir::Location loc,
                                  const std::vector<mlir::Value>& inputMemrefs,
                                  mlir::Value outputMemref,
                                  const std::vector<int64_t>& inputLengths,
                                  mlir::Type elemType) {
  auto zero = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
  auto one = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);

  // Sequential copy from each input
  int64_t offset = 0;
  for (size_t i = 0; i < inputMemrefs.size(); i++) {
    auto len = builder.create<mlir::arith::ConstantIndexOp>(loc, inputLengths[i]);
    auto baseOffset = builder.create<mlir::arith::ConstantIndexOp>(loc, offset);

    auto loop = builder.create<mlir::scf::ForOp>(loc, zero, len, one);
    {
      auto& body = loop.getRegion().front();
      builder.setInsertionPointToStart(&body);
      auto iv = body.getArgument(0);
      auto elem = builder.create<mlir::memref::LoadOp>(loc, inputMemrefs[i], mlir::ValueRange{iv});
      auto outIdx = builder.create<mlir::arith::AddIOp>(loc, iv, baseOffset);
      builder.create<mlir::memref::StoreOp>(loc, elem, outputMemref, mlir::ValueRange{outIdx});
      builder.create<mlir::scf::YieldOp>(loc);
    }
    builder.setInsertionPointAfter(loop);
    offset += inputLengths[i];
  }
}

void CpuIRBuilder::emitTileOp(mlir::OpBuilder& builder, mlir::Location loc,
                                mlir::Value inputMemref, mlir::Value outputMemref,
                                int64_t inputLen, int64_t outputLen, mlir::Type elemType) {
  auto zero = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
  auto one = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
  auto n = builder.create<mlir::arith::ConstantIndexOp>(loc, outputLen);
  auto inLen = builder.create<mlir::arith::ConstantIndexOp>(loc, inputLen);

  auto loop = builder.create<mlir::scf::ForOp>(loc, zero, n, one);
  {
    auto& body = loop.getRegion().front();
    builder.setInsertionPointToStart(&body);
    auto iv = body.getArgument(0);
    // Wrap with modulo: input[iv % inputLen]
    auto inIdx = builder.create<mlir::arith::RemUIOp>(loc, iv, inLen);
    auto elem = builder.create<mlir::memref::LoadOp>(loc, inputMemref, mlir::ValueRange{inIdx});
    builder.create<mlir::memref::StoreOp>(loc, elem, outputMemref, mlir::ValueRange{iv});
    builder.create<mlir::scf::YieldOp>(loc);
  }
  builder.setInsertionPointAfter(loop);
}

void CpuIRBuilder::emitScatterNdOp(mlir::OpBuilder& builder, mlir::Location loc,
                                     mlir::Value dataMemref, mlir::Value indicesMemref,
                                     mlir::Value updatesMemref, mlir::Value outputMemref,
                                     int64_t dataLen, int64_t numUpdates, int64_t sliceSize,
                                     mlir::Type elemType) {
  auto zero = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
  auto one = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);

  // Phase 1: Copy data → output
  auto dLen = builder.create<mlir::arith::ConstantIndexOp>(loc, dataLen);
  auto copyLoop = builder.create<mlir::scf::ForOp>(loc, zero, dLen, one);
  {
    auto& body = copyLoop.getRegion().front();
    builder.setInsertionPointToStart(&body);
    auto iv = body.getArgument(0);
    auto elem = builder.create<mlir::memref::LoadOp>(loc, dataMemref, mlir::ValueRange{iv});
    builder.create<mlir::memref::StoreOp>(loc, elem, outputMemref, mlir::ValueRange{iv});
    builder.create<mlir::scf::YieldOp>(loc);
  }
  builder.setInsertionPointAfter(copyLoop);

  // Phase 2: Scatter updates
  auto nUp = builder.create<mlir::arith::ConstantIndexOp>(loc, numUpdates * sliceSize);
  auto sliceSz = builder.create<mlir::arith::ConstantIndexOp>(loc, sliceSize);
  auto scatterLoop = builder.create<mlir::scf::ForOp>(loc, zero, nUp, one);
  {
    auto& body = scatterLoop.getRegion().front();
    builder.setInsertionPointToStart(&body);
    auto iv = body.getArgument(0);
    // updateIdx = iv / sliceSize
    auto updateIdx = builder.create<mlir::arith::DivUIOp>(loc, iv, sliceSz);
    // slicePos = iv % sliceSize
    auto slicePos = builder.create<mlir::arith::RemUIOp>(loc, iv, sliceSz);
    // Load index
    auto idxF = builder.create<mlir::memref::LoadOp>(loc, indicesMemref, mlir::ValueRange{updateIdx});
    auto idxI = builder.create<mlir::arith::FPToSIOp>(loc, builder.getI64Type(), idxF);
    auto idxIdx = builder.create<mlir::arith::IndexCastOp>(loc, mlir::IndexType::get(builder.getContext()), idxI);
    // outPos = index * sliceSize + slicePos
    auto base = builder.create<mlir::arith::MulIOp>(loc, idxIdx, sliceSz);
    auto outPos = builder.create<mlir::arith::AddIOp>(loc, base, slicePos);
    auto update = builder.create<mlir::memref::LoadOp>(loc, updatesMemref, mlir::ValueRange{iv});
    builder.create<mlir::memref::StoreOp>(loc, update, outputMemref, mlir::ValueRange{outPos});
    builder.create<mlir::scf::YieldOp>(loc);
  }
  builder.setInsertionPointAfter(scatterLoop);
}

void CpuIRBuilder::emitShapeManipOp(mlir::OpBuilder& builder, mlir::Location loc,
                                      const std::string& opName,
                                      mlir::Value inputMemref, mlir::Value outputMemref,
                                      int64_t nElements,
                                      const std::vector<LongType>& inputShape,
                                      const std::vector<LongType>& outputShape,
                                      const LongType* iArgs, int numIArgs,
                                      mlir::Type elemType) {
  std::string lower = toLower(opName);
  auto zero = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
  auto one = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
  auto n = builder.create<mlir::arith::ConstantIndexOp>(loc, nElements);

  bool isPermute = (lower == "permute" || lower == "transpose") && !inputShape.empty() && numIArgs > 0;

  if (isPermute && inputShape.size() >= 2 && static_cast<int>(inputShape.size()) <= numIArgs) {
    // Permute: remap indices
    int rank = static_cast<int>(inputShape.size());
    // Compute input strides (row-major)
    std::vector<int64_t> inStrides(rank), outStrides(rank);
    inStrides[rank - 1] = 1;
    outStrides[rank - 1] = 1;
    for (int d = rank - 2; d >= 0; d--) {
      inStrides[d] = inStrides[d + 1] * inputShape[d + 1];
      outStrides[d] = outStrides[d + 1] * outputShape[d + 1];
    }

    auto loop = builder.create<mlir::scf::ForOp>(loc, zero, n, one);
    {
      auto& body = loop.getRegion().front();
      builder.setInsertionPointToStart(&body);
      auto flatIdx = body.getArgument(0);

      // Unravel flatIdx → output coords, then remap to input coords
      mlir::Value remaining = flatIdx;
      mlir::Value inFlatIdx = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);

      for (int d = 0; d < rank; d++) {
        auto stride = builder.create<mlir::arith::ConstantIndexOp>(loc, outStrides[d]);
        auto coord = builder.create<mlir::arith::DivSIOp>(loc, remaining, stride);
        remaining = builder.create<mlir::arith::RemSIOp>(loc, remaining, stride);

        // This output dim d corresponds to input dim iArgs[d]
        int srcDim = static_cast<int>(iArgs[d]);
        auto inStride = builder.create<mlir::arith::ConstantIndexOp>(loc, inStrides[srcDim]);
        auto contrib = builder.create<mlir::arith::MulIOp>(loc, coord, inStride);
        inFlatIdx = builder.create<mlir::arith::AddIOp>(loc, inFlatIdx, contrib);
      }

      auto elem = builder.create<mlir::memref::LoadOp>(loc, inputMemref, mlir::ValueRange{inFlatIdx});
      builder.create<mlir::memref::StoreOp>(loc, elem, outputMemref, mlir::ValueRange{flatIdx});
      builder.create<mlir::scf::YieldOp>(loc);
    }
    builder.setInsertionPointAfter(loop);
  } else {
    // Reshape, flatten, expand_dims, squeeze: straight memcpy (same buffer layout)
    auto loop = builder.create<mlir::scf::ForOp>(loc, zero, n, one);
    {
      auto& body = loop.getRegion().front();
      builder.setInsertionPointToStart(&body);
      auto iv = body.getArgument(0);
      auto elem = builder.create<mlir::memref::LoadOp>(loc, inputMemref, mlir::ValueRange{iv});
      builder.create<mlir::memref::StoreOp>(loc, elem, outputMemref, mlir::ValueRange{iv});
      builder.create<mlir::scf::YieldOp>(loc);
    }
    builder.setInsertionPointAfter(loop);
  }
}

void CpuIRBuilder::emitConstantGenOp(mlir::OpBuilder& builder, mlir::Location loc,
                                       const std::string& opName,
                                       mlir::Value outputMemref,
                                       int64_t nElements, mlir::Type elemType,
                                       const double* tArgs, int numTArgs) {
  std::string lower = toLower(opName);
  auto zero = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
  auto one = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
  auto n = builder.create<mlir::arith::ConstantIndexOp>(loc, nElements);

  if (lower == "range") {
    // range(start, stop, step) — tArgs[0]=start, tArgs[1]=stop, tArgs[2]=step
    double start = (numTArgs > 0) ? tArgs[0] : 0.0;
    double step = (numTArgs > 2) ? tArgs[2] : 1.0;
    auto startVal = builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(elemType, start));
    auto stepVal = builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(elemType, step));

    auto loop = builder.create<mlir::scf::ForOp>(loc, zero, n, one);
    {
      auto& body = loop.getRegion().front();
      builder.setInsertionPointToStart(&body);
      auto iv = body.getArgument(0);
      auto ivI64 = builder.create<mlir::arith::IndexCastOp>(loc, builder.getI64Type(), iv);
      auto ivF = builder.create<mlir::arith::SIToFPOp>(loc, elemType, ivI64);
      auto val = builder.create<mlir::arith::MulFOp>(loc, ivF, stepVal);
      auto result = builder.create<mlir::arith::AddFOp>(loc, startVal, val);
      builder.create<mlir::memref::StoreOp>(loc, result, outputMemref, mlir::ValueRange{iv});
      builder.create<mlir::scf::YieldOp>(loc);
    }
    builder.setInsertionPointAfter(loop);
  } else {
    // zeros_like, ones_like, create, set_scalar, shape_of, min_max_datatype
    double fillVal = 0.0;
    if (lower == "ones_as" || lower == "ones_like" || lower == "oneslike") fillVal = 1.0;
    if (lower == "set_scalar" && numTArgs > 0) fillVal = tArgs[0];
    auto fillConst = builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(elemType, fillVal));

    auto loop = builder.create<mlir::scf::ForOp>(loc, zero, n, one);
    {
      auto& body = loop.getRegion().front();
      builder.setInsertionPointToStart(&body);
      auto iv = body.getArgument(0);
      builder.create<mlir::memref::StoreOp>(loc, fillConst, outputMemref, mlir::ValueRange{iv});
      builder.create<mlir::scf::YieldOp>(loc);
    }
    builder.setInsertionPointAfter(loop);
  }
}

void CpuIRBuilder::emitConvolutionOp(mlir::OpBuilder& builder, mlir::Location loc,
                                       mlir::Value inputMemref, mlir::Value filterMemref,
                                       mlir::Value outputMemref,
                                       const std::vector<LongType>& inputShape,
                                       const std::vector<LongType>& filterShape,
                                       const std::vector<LongType>& outputShape,
                                       const LongType* iArgs, int numIArgs,
                                       mlir::Type elemType) {
  // Direct conv2d: NCHW input, OIHW filter (or deduced from shapes)
  // iArgs: [kH, kW, sH, sW, pH, pW, dH, dW, ...]
  if (inputShape.size() < 4 || filterShape.size() < 4 || outputShape.size() < 4) {
    // Fallback: identity copy
    auto zero = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto one = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
    int64_t total = 1;
    for (auto d : outputShape) total *= d;
    auto n = builder.create<mlir::arith::ConstantIndexOp>(loc, total);
    auto loop = builder.create<mlir::scf::ForOp>(loc, zero, n, one);
    {
      auto& body = loop.getRegion().front();
      builder.setInsertionPointToStart(&body);
      auto iv = body.getArgument(0);
      auto elem = builder.create<mlir::memref::LoadOp>(loc, inputMemref, mlir::ValueRange{iv});
      builder.create<mlir::memref::StoreOp>(loc, elem, outputMemref, mlir::ValueRange{iv});
      builder.create<mlir::scf::YieldOp>(loc);
    }
    builder.setInsertionPointAfter(loop);
    return;
  }

  int64_t N = outputShape[0], OC = outputShape[1], OH = outputShape[2], OW = outputShape[3];
  int64_t IC = inputShape[1], IH = inputShape[2], IW = inputShape[3];
  int64_t KH = filterShape[2], KW = filterShape[3];
  int64_t sH = (numIArgs > 2) ? iArgs[2] : 1;
  int64_t sW = (numIArgs > 3) ? iArgs[3] : 1;
  int64_t pH = (numIArgs > 4) ? iArgs[4] : 0;
  int64_t pW = (numIArgs > 5) ? iArgs[5] : 0;

  auto zero = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
  auto one = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
  auto zeroF = builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(elemType, 0.0));

  // Flatten output as 1D loop over N*OC*OH*OW
  int64_t totalOut = N * OC * OH * OW;
  auto nTotal = builder.create<mlir::arith::ConstantIndexOp>(loc, totalOut);
  auto ocBound = builder.create<mlir::arith::ConstantIndexOp>(loc, OC);
  auto ohBound = builder.create<mlir::arith::ConstantIndexOp>(loc, OH);
  auto owBound = builder.create<mlir::arith::ConstantIndexOp>(loc, OW);
  auto icBound = builder.create<mlir::arith::ConstantIndexOp>(loc, IC);
  auto khBound = builder.create<mlir::arith::ConstantIndexOp>(loc, KH);
  auto kwBound = builder.create<mlir::arith::ConstantIndexOp>(loc, KW);
  auto sHval = builder.create<mlir::arith::ConstantIndexOp>(loc, sH);
  auto sWval = builder.create<mlir::arith::ConstantIndexOp>(loc, sW);
  auto pHval = builder.create<mlir::arith::ConstantIndexOp>(loc, pH);
  auto pWval = builder.create<mlir::arith::ConstantIndexOp>(loc, pW);
  auto iwBound = builder.create<mlir::arith::ConstantIndexOp>(loc, IW);
  auto ihBound = builder.create<mlir::arith::ConstantIndexOp>(loc, IH);

  auto outLoop = builder.create<mlir::scf::ForOp>(loc, zero, nTotal, one);
  {
    auto& body = outLoop.getRegion().front();
    builder.setInsertionPointToStart(&body);
    auto flatIdx = body.getArgument(0);

    // Decompose flatIdx → (n, oc, oh, ow)
    auto ohow = builder.create<mlir::arith::ConstantIndexOp>(loc, OH * OW);
    auto ocohow = builder.create<mlir::arith::ConstantIndexOp>(loc, OC * OH * OW);
    auto nIdx = builder.create<mlir::arith::DivUIOp>(loc, flatIdx, ocohow);
    auto rem1 = builder.create<mlir::arith::RemUIOp>(loc, flatIdx, ocohow);
    auto ocIdx = builder.create<mlir::arith::DivUIOp>(loc, rem1, ohow);
    auto rem2 = builder.create<mlir::arith::RemUIOp>(loc, rem1, ohow);
    auto ohIdx = builder.create<mlir::arith::DivUIOp>(loc, rem2, owBound);
    auto owIdx = builder.create<mlir::arith::RemUIOp>(loc, rem2, owBound);

    // Accumulate over IC, KH, KW
    auto ickhkw = builder.create<mlir::arith::ConstantIndexOp>(loc, IC * KH * KW);
    auto khkw = builder.create<mlir::arith::ConstantIndexOp>(loc, KH * KW);
    auto innerLoop = builder.create<mlir::scf::ForOp>(
        loc, zero, ickhkw, one, mlir::ValueRange{zeroF.getResult()});
    {
      auto& innerBody = innerLoop.getRegion().front();
      builder.setInsertionPointToStart(&innerBody);
      auto innerIv = innerBody.getArgument(0);
      auto acc = innerBody.getArgument(1);

      auto icIdx = builder.create<mlir::arith::DivUIOp>(loc, innerIv, khkw);
      auto rem3 = builder.create<mlir::arith::RemUIOp>(loc, innerIv, khkw);
      auto khIdx = builder.create<mlir::arith::DivUIOp>(loc, rem3, kwBound);
      auto kwIdx = builder.create<mlir::arith::RemUIOp>(loc, rem3, kwBound);

      // h_in = oh * sH - pH + kh
      auto hBase = builder.create<mlir::arith::MulIOp>(loc, ohIdx, sHval);
      auto hNoPad = builder.create<mlir::arith::SubIOp>(loc, hBase, pHval);
      auto hIn = builder.create<mlir::arith::AddIOp>(loc, hNoPad, khIdx);
      // w_in = ow * sW - pW + kw
      auto wBase = builder.create<mlir::arith::MulIOp>(loc, owIdx, sWval);
      auto wNoPad = builder.create<mlir::arith::SubIOp>(loc, wBase, pWval);
      auto wIn = builder.create<mlir::arith::AddIOp>(loc, wNoPad, kwIdx);

      // Bounds check: 0 <= h_in < IH && 0 <= w_in < IW
      // For simplicity, use zero-padding: out-of-bounds → 0
      auto hInI64 = builder.create<mlir::arith::IndexCastOp>(loc, builder.getI64Type(), hIn);
      auto wInI64 = builder.create<mlir::arith::IndexCastOp>(loc, builder.getI64Type(), wIn);
      auto ihI64 = builder.create<mlir::arith::ConstantOp>(loc, builder.getI64IntegerAttr(IH));
      auto iwI64 = builder.create<mlir::arith::ConstantOp>(loc, builder.getI64IntegerAttr(IW));
      auto zeroI64 = builder.create<mlir::arith::ConstantOp>(loc, builder.getI64IntegerAttr(0));

      auto hGe0 = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sge, hInI64, zeroI64);
      auto hLtH = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, hInI64, ihI64);
      auto wGe0 = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sge, wInI64, zeroI64);
      auto wLtW = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, wInI64, iwI64);
      auto hOk = builder.create<mlir::arith::AndIOp>(loc, hGe0, hLtH);
      auto wOk = builder.create<mlir::arith::AndIOp>(loc, wGe0, wLtW);
      auto inBounds = builder.create<mlir::arith::AndIOp>(loc, hOk, wOk);

      // input[n, ic, h_in, w_in] = n * IC*IH*IW + ic * IH*IW + h_in * IW + w_in
      auto icihiw = builder.create<mlir::arith::ConstantIndexOp>(loc, IC * IH * IW);
      auto ihiw = builder.create<mlir::arith::ConstantIndexOp>(loc, IH * IW);
      auto inIdx = builder.create<mlir::arith::AddIOp>(loc,
          builder.create<mlir::arith::AddIOp>(loc,
              builder.create<mlir::arith::AddIOp>(loc,
                  builder.create<mlir::arith::MulIOp>(loc, nIdx, icihiw),
                  builder.create<mlir::arith::MulIOp>(loc, icIdx, ihiw)),
              builder.create<mlir::arith::MulIOp>(loc, hIn, iwBound)),
          wIn);

      // filter[oc, ic, kh, kw] = oc * IC*KH*KW + ic * KH*KW + kh * KW + kw
      auto filtIdx = builder.create<mlir::arith::AddIOp>(loc,
          builder.create<mlir::arith::AddIOp>(loc,
              builder.create<mlir::arith::AddIOp>(loc,
                  builder.create<mlir::arith::MulIOp>(loc, ocIdx, ickhkw),
                  builder.create<mlir::arith::MulIOp>(loc, icIdx, khkw)),
              builder.create<mlir::arith::MulIOp>(loc, khIdx, kwBound)),
          kwIdx);

      auto inVal = builder.create<mlir::memref::LoadOp>(loc, inputMemref, mlir::ValueRange{inIdx});
      auto filtVal = builder.create<mlir::memref::LoadOp>(loc, filterMemref, mlir::ValueRange{filtIdx});
      auto prod = builder.create<mlir::arith::MulFOp>(loc, inVal, filtVal);
      // Zero out-of-bounds contributions
      auto maskedProd = builder.create<mlir::arith::SelectOp>(loc, inBounds, prod, zeroF);
      auto newAcc = builder.create<mlir::arith::AddFOp>(loc, acc, maskedProd);
      builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{newAcc});
    }
    builder.setInsertionPointAfter(innerLoop);

    builder.create<mlir::memref::StoreOp>(loc, innerLoop.getResult(0), outputMemref, mlir::ValueRange{flatIdx});
    builder.create<mlir::scf::YieldOp>(loc);
  }
  builder.setInsertionPointAfter(outLoop);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Module building — handles ALL op categories
// ═══════════════════════════════════════════════════════════════════════════════

mlir::OwningOpRef<mlir::ModuleOp> CpuIRBuilder::buildModule(
    mlir::MLIRContext* context,
    NativeSlot* slots, int startSlot, int endSlot,
    int totalSlots,
    NDArray** externalInputs, int numExternalInputs,
    NDArray** outputSlots, int totalOutputSlots) {

  int segSize = endSlot - startSlot + 1;
  mlir::OpBuilder builder(context);
  auto loc = builder.getUnknownLoc();

  // ── Step 1: Collect unique buffer arguments ────────────────────────────

  struct BufferArg {
    int sourceIndex;
    bool isOutput;
    NDArray* array;
  };

  std::unordered_map<int, int> sourceToArgIdx;
  std::vector<BufferArg> bufferArgs;

  std::unordered_set<int> internalOutputs;
  for (int i = startSlot; i <= endSlot; i++) {
    for (int o = 0; o < slots[i].wiring.numOutputs; o++) {
      internalOutputs.insert(slots[i].wiring.outputSlotIndices[o]);
    }
  }

  // Collect input buffer args
  for (int i = startSlot; i <= endSlot; i++) {
    for (int inp = 0; inp < slots[i].wiring.numInputs; inp++) {
      int srcIdx = slots[i].wiring.inputSourceIndices[inp];
      if (sourceToArgIdx.count(srcIdx)) continue;

      NDArray* arr = nullptr;
      if (srcIdx < 0) {
        int extIdx = -(srcIdx + 1);
        if (extIdx < numExternalInputs && externalInputs) arr = externalInputs[extIdx];
      } else if (!internalOutputs.count(srcIdx)) {
        if (srcIdx < totalOutputSlots && outputSlots) arr = outputSlots[srcIdx];
      } else {
        continue;
      }
      if (!arr) continue;

      int argIdx = static_cast<int>(bufferArgs.size());
      sourceToArgIdx[srcIdx] = argIdx;
      bufferArgs.push_back({srcIdx, false, arr});
    }
  }

  int numInputArgs = static_cast<int>(bufferArgs.size());

  // Collect output buffer args
  auto externalOutputSet = computeExternallyVisibleOutputs(slots, startSlot, endSlot, totalSlots);
  for (int i = startSlot; i <= endSlot; i++) {
    for (int o = 0; o < slots[i].wiring.numOutputs; o++) {
      int outIdx = slots[i].wiring.outputSlotIndices[o];
      if (!externalOutputSet.count(outIdx)) continue;
      if (sourceToArgIdx.count(outIdx)) continue;

      NDArray* arr = nullptr;
      if (outIdx >= 0 && outIdx < totalOutputSlots && outputSlots) arr = outputSlots[outIdx];
      if (!arr) continue;

      int argIdx = static_cast<int>(bufferArgs.size());
      sourceToArgIdx[outIdx] = argIdx;
      bufferArgs.push_back({outIdx, true, arr});
    }
  }

  // Also register internal intermediate outputs that have consumer ops needing
  // their memref (for non-element-wise ops that use full-array emitters)
  for (int i = startSlot; i <= endSlot; i++) {
    for (int o = 0; o < slots[i].wiring.numOutputs; o++) {
      int outIdx = slots[i].wiring.outputSlotIndices[o];
      if (sourceToArgIdx.count(outIdx)) continue;
      // Check if any non-trivial consumer needs this as a full memref
      NDArray* arr = nullptr;
      if (outIdx >= 0 && outIdx < totalOutputSlots && outputSlots) arr = outputSlots[outIdx];
      if (!arr) continue;

      int argIdx = static_cast<int>(bufferArgs.size());
      sourceToArgIdx[outIdx] = argIdx;
      bufferArgs.push_back({outIdx, true, arr});
    }
  }

  if (bufferArgs.empty()) {
    DSP_DIAG(COMPILE, "CpuIRBuilder::buildModule: no buffer args for segment [%d-%d]",
              startSlot, endSlot);
    return mlir::OwningOpRef<mlir::ModuleOp>();
  }

  // ── Step 2: Create module and function ─────────────────────────────────
  auto module = mlir::ModuleOp::create(loc);
  builder.setInsertionPointToEnd(module.getBody());

  auto f32Type = mlir::FloatType::getF32(context);
  auto memrefType = mlir::MemRefType::get({mlir::ShapedType::kDynamic}, f32Type);
  auto indexType = mlir::IndexType::get(context);

  llvm::SmallVector<mlir::Type, 16> argTypes;
  for (size_t i = 0; i < bufferArgs.size(); i++) {
    argTypes.push_back(memrefType);
  }
  argTypes.push_back(indexType);  // n_elements

  auto funcType = builder.getFunctionType(argTypes, {});
  auto func = builder.create<mlir::func::FuncOp>(loc, "fused_kernel", funcType);
  func.setVisibility(mlir::SymbolTable::Visibility::Public);

  auto* entryBlock = func.addEntryBlock();
  builder.setInsertionPointToEnd(entryBlock);

  auto nElements = entryBlock->getArgument(static_cast<int>(bufferArgs.size()));

  // ── Step 3: Emit ops sequentially ──────────────────────────────────────
  // Element-wise ops are batched into a single scf.for loop.
  // Non-element-wise ops break the element-wise batch and emit their own structures.

  // Accumulate element-wise ops to batch them
  std::vector<int> currentEwBatch;
  std::unordered_map<int, mlir::Value> slotOutputToValue;

  auto flushElementwiseBatch = [&]() {
    if (currentEwBatch.empty()) return;

    auto zeroIdx = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto oneIdx = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);

    auto forOp = builder.create<mlir::scf::ForOp>(loc, zeroIdx, nElements, oneIdx);
    builder.setInsertionPointToStart(forOp.getBody());
    auto iv = forOp.getInductionVar();

    // SSA value map: when an op's output feeds another op in the same batch,
    // use the SSA Value directly instead of storing+loading through the memref.
    // This is the CPU equivalent of Triton's SSA intermediate elimination —
    // data stays in registers between fused ops, no global memory round-trip.
    std::unordered_map<int, mlir::Value> ssaValues;

    for (int slotIdx : currentEwBatch) {
      auto& slot = slots[slotIdx];
      std::string lower = toLower(slot.ident.opName);

      // Gather input values — prefer SSA values from previous ops in this batch
      std::vector<mlir::Value> inputValues;
      for (int inp = 0; inp < slot.wiring.numInputs; inp++) {
        int srcIdx = slot.wiring.inputSourceIndices[inp];
        // First: check if a previous op in this batch produced this value
        auto ssaIt = ssaValues.find(srcIdx);
        if (ssaIt != ssaValues.end()) {
          inputValues.push_back(ssaIt->second);
        } else {
          // Fall back to loading from memref
          auto argIt = sourceToArgIdx.find(srcIdx);
          if (argIt != sourceToArgIdx.end()) {
            auto memref = entryBlock->getArgument(argIt->second);
            auto loaded = builder.create<mlir::memref::LoadOp>(loc, memref, mlir::ValueRange{iv});
            inputValues.push_back(loaded);
          } else {
            inputValues.push_back(
                builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(f32Type, 0.0)));
          }
        }
      }

      // Emit the op
      mlir::Value result;
      const auto& table = getOpCategoryTable();
      auto catIt = table.find(slot.ident.opName);
      TritonOpCategory category = (catIt != table.end()) ? catIt->second : TritonOpCategory::UNSUPPORTED;

      switch (category) {
        case TritonOpCategory::BINARY_ELEMENTWISE:
          if (inputValues.size() >= 2)
            result = emitBinaryElementwise(builder, loc, lower, inputValues[0], inputValues[1], f32Type);
          break;
        case TritonOpCategory::UNARY_ELEMENTWISE:
          if (!inputValues.empty())
            result = emitUnaryElementwise(builder, loc, lower, inputValues[0], f32Type, slot.args.tArgs, slot.args.numTArgs);
          break;
        case TritonOpCategory::COMPARISON:
          if (inputValues.size() >= 2) {
            auto cmpResult = emitComparisonOp(builder, loc, lower, inputValues[0], inputValues[1]);
            auto oneF = builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(f32Type, 1.0));
            auto zeroF = builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(f32Type, 0.0));
            result = builder.create<mlir::arith::SelectOp>(loc, cmpResult, oneF, zeroF);
          }
          break;
        case TritonOpCategory::TERNARY:
          if (inputValues.size() >= 3) {
            auto zeroF = builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(f32Type, 0.0));
            auto condBool = builder.create<mlir::arith::CmpFOp>(
                loc, mlir::arith::CmpFPredicate::ONE, inputValues[0], zeroF);
            result = emitTernaryOp(builder, loc, lower, condBool, inputValues[1], inputValues[2]);
          }
          break;
        case TritonOpCategory::LOGICAL:
          if (inputValues.size() >= 2) {
            auto zeroF = builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(f32Type, 0.0));
            auto lhsBool = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::ONE, inputValues[0], zeroF);
            auto rhsBool = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::ONE, inputValues[1], zeroF);
            mlir::Value logicResult;
            if (lower == "boolean_and" || lower == "logical_and")
              logicResult = builder.create<mlir::arith::AndIOp>(loc, lhsBool, rhsBool);
            else if (lower == "boolean_or" || lower == "logical_or")
              logicResult = builder.create<mlir::arith::OrIOp>(loc, lhsBool, rhsBool);
            else if (lower == "boolean_xor")
              logicResult = builder.create<mlir::arith::XOrIOp>(loc, lhsBool, rhsBool);
            else
              logicResult = lhsBool;
            auto oneF = builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(f32Type, 1.0));
            result = builder.create<mlir::arith::SelectOp>(loc, logicResult, oneF, zeroF);
          } else if (inputValues.size() == 1) {
            auto zeroF = builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(f32Type, 0.0));
            auto oneF = builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(f32Type, 1.0));
            auto inBool = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OEQ, inputValues[0], zeroF);
            result = builder.create<mlir::arith::SelectOp>(loc, inBool, oneF, zeroF);
          }
          break;
        case TritonOpCategory::IDENTITY:
        case TritonOpCategory::CAST:
          if (!inputValues.empty()) result = inputValues[0];
          break;
        default:
          if (!inputValues.empty()) result = inputValues[0];
          break;
      }

      if (!result) {
        result = builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(f32Type, 0.0));
      }

      // Register in SSA value map for downstream ops in this batch
      if (slot.wiring.numOutputs > 0) {
        int outIdx = slot.wiring.outputSlotIndices[0];
        ssaValues[outIdx] = result;

        // Only store to memref if externally visible (consumed outside this batch).
        // Internal intermediates stay in SSA registers — no memory round-trip.
        if (externalOutputSet.count(outIdx) || !internalOutputs.count(outIdx)) {
          if (sourceToArgIdx.count(outIdx)) {
            auto memref = entryBlock->getArgument(sourceToArgIdx[outIdx]);
            builder.create<mlir::memref::StoreOp>(loc, result, memref, mlir::ValueRange{iv});
          }
        }
      }
    }

    builder.setInsertionPointAfter(forOp);
    currentEwBatch.clear();
  };

  // Process each slot
  for (int i = 0; i < segSize; i++) {
    int absSlot = startSlot + i;
    auto& slot = slots[absSlot];

    const auto& table = getOpCategoryTable();
    auto catIt = table.find(slot.ident.opName);
    TritonOpCategory category = (catIt != table.end()) ? catIt->second : TritonOpCategory::UNSUPPORTED;

    // Check if this is an element-wise-compatible op.
    // ROPE is classified as ELEMENTWISE in the Triton path because paired access
    // fits in a single block. On CPU it's also element-wise (no reduction/sync).
    bool isEw = (category == TritonOpCategory::BINARY_ELEMENTWISE ||
                 category == TritonOpCategory::UNARY_ELEMENTWISE ||
                 category == TritonOpCategory::COMPARISON ||
                 category == TritonOpCategory::LOGICAL ||
                 category == TritonOpCategory::TERNARY ||
                 category == TritonOpCategory::IDENTITY ||
                 category == TritonOpCategory::CAST ||
                 category == TritonOpCategory::ROPE);

    if (isEw) {
      currentEwBatch.push_back(absSlot);
      continue;
    }

    // Non-element-wise op: flush the current batch, then emit specialized code
    flushElementwiseBatch();

    // Resolve memref args for this op's inputs and outputs
    auto getMemref = [&](int srcIdx) -> mlir::Value {
      auto it = sourceToArgIdx.find(srcIdx);
      if (it != sourceToArgIdx.end()) return entryBlock->getArgument(it->second);
      return mlir::Value();
    };

    switch (category) {
      case TritonOpCategory::REDUCTION: {
        if (slot.wiring.numInputs > 0 && slot.wiring.numOutputs > 0) {
          int inSrc = slot.wiring.inputSourceIndices[0];
          int outIdx = slot.wiring.outputSlotIndices[0];
          auto inMem = getMemref(inSrc);
          auto outMem = getMemref(outIdx);
          if (inMem && outMem) {
            int64_t nElem = bufferArgs[sourceToArgIdx[inSrc]].array->lengthOf();
            emitReductionOp(builder, loc, slot.ident.opName, inMem, outMem, nElem, f32Type);
          }
        }
        break;
      }
      case TritonOpCategory::NORMALIZATION: {
        if (slot.wiring.numInputs > 0 && slot.wiring.numOutputs > 0) {
          int inSrc = slot.wiring.inputSourceIndices[0];
          int outIdx = slot.wiring.outputSlotIndices[0];
          auto inMem = getMemref(inSrc);
          auto outMem = getMemref(outIdx);
          if (inMem && outMem) {
            int64_t nElem = bufferArgs[sourceToArgIdx[inSrc]].array->lengthOf();
            emitNormalizationOp(builder, loc, slot.ident.opName, inMem, outMem, nElem, f32Type);
          }
        }
        break;
      }
      case TritonOpCategory::MATMUL: {
        if (slot.wiring.numInputs >= 2 && slot.wiring.numOutputs > 0) {
          int aSrc = slot.wiring.inputSourceIndices[0];
          int bSrc = slot.wiring.inputSourceIndices[1];
          int outIdx = slot.wiring.outputSlotIndices[0];
          auto aMem = getMemref(aSrc);
          auto bMem = getMemref(bSrc);
          auto cMem = getMemref(outIdx);
          if (aMem && bMem && cMem) {
            auto* aArr = bufferArgs[sourceToArgIdx[aSrc]].array;
            auto* bArr = bufferArgs[sourceToArgIdx[bSrc]].array;
            int64_t M = (aArr->rankOf() >= 2) ? aArr->sizeAt(0) : 1;
            int64_t K = (aArr->rankOf() >= 2) ? aArr->sizeAt(1) : aArr->lengthOf();
            int64_t N = (bArr->rankOf() >= 2) ? bArr->sizeAt(1) : 1;
            emitMatmulOp(builder, loc, aMem, bMem, cMem, M, N, K, f32Type);
          }
        }
        break;
      }
      case TritonOpCategory::DATA_MOVEMENT: {
        std::string lower = toLower(slot.ident.opName);
        if (lower == "gather" || lower == "gather_nd" || lower == "gathernd") {
          if (slot.wiring.numInputs >= 2 && slot.wiring.numOutputs > 0) {
            int dataSrc = slot.wiring.inputSourceIndices[0];
            int idxSrc = slot.wiring.inputSourceIndices[1];
            int outIdx = slot.wiring.outputSlotIndices[0];
            auto dataMem = getMemref(dataSrc);
            auto idxMem = getMemref(idxSrc);
            auto outMem = getMemref(outIdx);
            if (dataMem && idxMem && outMem) {
              auto* outArr = bufferArgs[sourceToArgIdx[outIdx]].array;
              auto* dataArr = bufferArgs[sourceToArgIdx[dataSrc]].array;
              int axis = (slot.args.numIArgs > 0) ? static_cast<int>(slot.args.iArgs[0]) : 0;
              std::vector<LongType> dataShape(dataArr->rankOf());
              for (int d = 0; d < dataArr->rankOf(); d++) dataShape[d] = dataArr->sizeAt(d);
              emitGatherOp(builder, loc, dataMem, idxMem, outMem, outArr->lengthOf(), axis, dataShape, f32Type);
            }
          }
        } else if (lower == "concat") {
          if (slot.wiring.numInputs >= 2 && slot.wiring.numOutputs > 0) {
            std::vector<mlir::Value> inMemrefs;
            std::vector<int64_t> inLengths;
            for (int inp = 0; inp < slot.wiring.numInputs; inp++) {
              int src = slot.wiring.inputSourceIndices[inp];
              auto mem = getMemref(src);
              if (mem) {
                inMemrefs.push_back(mem);
                inLengths.push_back(bufferArgs[sourceToArgIdx[src]].array->lengthOf());
              }
            }
            int outIdx = slot.wiring.outputSlotIndices[0];
            auto outMem = getMemref(outIdx);
            if (outMem && !inMemrefs.empty()) {
              emitConcatOp(builder, loc, inMemrefs, outMem, inLengths, f32Type);
            }
          }
        } else if (lower == "tile") {
          if (slot.wiring.numInputs > 0 && slot.wiring.numOutputs > 0) {
            int inSrc = slot.wiring.inputSourceIndices[0];
            int outIdx = slot.wiring.outputSlotIndices[0];
            auto inMem = getMemref(inSrc);
            auto outMem = getMemref(outIdx);
            if (inMem && outMem) {
              auto* inArr = bufferArgs[sourceToArgIdx[inSrc]].array;
              auto* outArr = bufferArgs[sourceToArgIdx[outIdx]].array;
              emitTileOp(builder, loc, inMem, outMem, inArr->lengthOf(), outArr->lengthOf(), f32Type);
            }
          }
        } else if (lower == "scatter_nd" || lower == "scatter_nd_update" ||
                   lower == "scatternd" || lower == "scatterndupdate") {
          if (slot.wiring.numInputs >= 3 && slot.wiring.numOutputs > 0) {
            int dataSrc = slot.wiring.inputSourceIndices[0];
            int idxSrc = slot.wiring.inputSourceIndices[1];
            int updSrc = slot.wiring.inputSourceIndices[2];
            int outIdx = slot.wiring.outputSlotIndices[0];
            auto dataMem = getMemref(dataSrc);
            auto idxMem = getMemref(idxSrc);
            auto updMem = getMemref(updSrc);
            auto outMem = getMemref(outIdx);
            if (dataMem && idxMem && updMem && outMem) {
              auto* dataArr = bufferArgs[sourceToArgIdx[dataSrc]].array;
              auto* idxArr = bufferArgs[sourceToArgIdx[idxSrc]].array;
              emitScatterNdOp(builder, loc, dataMem, idxMem, updMem, outMem,
                              dataArr->lengthOf(), idxArr->lengthOf(), 1, f32Type);
            }
          }
        } else {
          // split, split_v, stack, strided_slice: copy for now
          if (slot.wiring.numInputs > 0 && slot.wiring.numOutputs > 0) {
            int inSrc = slot.wiring.inputSourceIndices[0];
            int outIdx = slot.wiring.outputSlotIndices[0];
            auto inMem = getMemref(inSrc);
            auto outMem = getMemref(outIdx);
            if (inMem && outMem) {
              auto* outArr = bufferArgs[sourceToArgIdx[outIdx]].array;
              int64_t nElem = outArr->lengthOf();
              auto zeroIdx = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
              auto oneIdx = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
              auto nVal = builder.create<mlir::arith::ConstantIndexOp>(loc, nElem);
              auto copyLoop = builder.create<mlir::scf::ForOp>(loc, zeroIdx, nVal, oneIdx);
              {
                auto& body = copyLoop.getRegion().front();
                builder.setInsertionPointToStart(&body);
                auto iv = body.getArgument(0);
                auto elem = builder.create<mlir::memref::LoadOp>(loc, inMem, mlir::ValueRange{iv});
                builder.create<mlir::memref::StoreOp>(loc, elem, outMem, mlir::ValueRange{iv});
                builder.create<mlir::scf::YieldOp>(loc);
              }
              builder.setInsertionPointAfter(copyLoop);
            }
          }
        }
        break;
      }
      case TritonOpCategory::SHAPE_MANIPULATION: {
        if (slot.wiring.numInputs > 0 && slot.wiring.numOutputs > 0) {
          int inSrc = slot.wiring.inputSourceIndices[0];
          int outIdx = slot.wiring.outputSlotIndices[0];
          auto inMem = getMemref(inSrc);
          auto outMem = getMemref(outIdx);
          if (inMem && outMem) {
            auto* inArr = bufferArgs[sourceToArgIdx[inSrc]].array;
            auto* outArr = bufferArgs[sourceToArgIdx[outIdx]].array;
            std::vector<LongType> inShape(inArr->rankOf()), outShape(outArr->rankOf());
            for (int d = 0; d < inArr->rankOf(); d++) inShape[d] = inArr->sizeAt(d);
            for (int d = 0; d < outArr->rankOf(); d++) outShape[d] = outArr->sizeAt(d);
            emitShapeManipOp(builder, loc, slot.ident.opName, inMem, outMem,
                             outArr->lengthOf(), inShape, outShape,
                             slot.args.iArgs, slot.args.numIArgs, f32Type);
          }
        }
        break;
      }
      case TritonOpCategory::CONSTANT_GENERATION: {
        if (slot.wiring.numOutputs > 0) {
          int outIdx = slot.wiring.outputSlotIndices[0];
          auto outMem = getMemref(outIdx);
          if (outMem) {
            auto* outArr = bufferArgs[sourceToArgIdx[outIdx]].array;
            emitConstantGenOp(builder, loc, slot.ident.opName, outMem,
                              outArr->lengthOf(), f32Type,
                              slot.args.tArgs, slot.args.numTArgs);
          }
        }
        break;
      }
      case TritonOpCategory::CONVOLUTION: {
        if (slot.wiring.numInputs >= 2 && slot.wiring.numOutputs > 0) {
          int inSrc = slot.wiring.inputSourceIndices[0];
          int filtSrc = slot.wiring.inputSourceIndices[1];
          int outIdx = slot.wiring.outputSlotIndices[0];
          auto inMem = getMemref(inSrc);
          auto filtMem = getMemref(filtSrc);
          auto outMem = getMemref(outIdx);
          if (inMem && filtMem && outMem) {
            auto* inArr = bufferArgs[sourceToArgIdx[inSrc]].array;
            auto* filtArr = bufferArgs[sourceToArgIdx[filtSrc]].array;
            auto* outArr = bufferArgs[sourceToArgIdx[outIdx]].array;
            std::vector<LongType> inShape(inArr->rankOf()), filtShape(filtArr->rankOf()), outShape(outArr->rankOf());
            for (int d = 0; d < inArr->rankOf(); d++) inShape[d] = inArr->sizeAt(d);
            for (int d = 0; d < filtArr->rankOf(); d++) filtShape[d] = filtArr->sizeAt(d);
            for (int d = 0; d < outArr->rankOf(); d++) outShape[d] = outArr->sizeAt(d);
            emitConvolutionOp(builder, loc, inMem, filtMem, outMem,
                              inShape, filtShape, outShape,
                              slot.args.iArgs, slot.args.numIArgs, f32Type);
          }
        }
        break;
      }
      case TritonOpCategory::FUSED_LLM: {
        // Fused LLM ops: decompose into constituent ops for CPU emission.
        std::string lower = toLower(slot.ident.opName);
        if (lower == "rms_norm_linear" || lower == "rmsnormlinear") {
          // rms_norm_linear(x, gamma, W) = matmul(rms_norm(x, gamma, eps), W)
          // Decomposed: normalize x, then matmul with W
          if (slot.wiring.numInputs >= 3 && slot.wiring.numOutputs > 0) {
            int xSrc = slot.wiring.inputSourceIndices[0];
            int gammaSrc = slot.wiring.inputSourceIndices[1];
            int wSrc = slot.wiring.inputSourceIndices[2];
            int outIdx = slot.wiring.outputSlotIndices[0];
            auto xMem = getMemref(xSrc);
            auto gammaMem = getMemref(gammaSrc);
            auto wMem = getMemref(wSrc);
            auto outMem = getMemref(outIdx);
            if (xMem && wMem && outMem) {
              auto* xArr = bufferArgs[sourceToArgIdx[xSrc]].array;
              auto* wArr = bufferArgs[sourceToArgIdx[wSrc]].array;
              int64_t M = (xArr->rankOf() >= 2) ? xArr->sizeAt(0) : 1;
              int64_t K = (xArr->rankOf() >= 2) ? xArr->sizeAt(1) : xArr->lengthOf();
              int64_t N = (wArr->rankOf() >= 2) ? wArr->sizeAt(1) : 1;

              // Step 1: RMS normalize x into a temporary (reuse output buffer as scratch)
              // We'll normalize into outMem temporarily, then matmul into outMem properly.
              // Actually, we need a scratch buffer. Since we only have memrefs, normalize
              // x in-place into output buffer row by row, then do matmul.
              // For correctness, emit rms_norm of x into a portion of outMem, then matmul.
              // Simpler approach: emit as two separate ops using existing emitters.
              emitNormalizationOp(builder, loc, "rms_norm", xMem, xMem, K, f32Type);
              // Now xMem has the normalized values; matmul into outMem
              emitMatmulOp(builder, loc, xMem, wMem, outMem, M, N, K, f32Type);
            }
          }
        } else if (lower == "fused_gemm_swiglu" || lower == "fusedgemmswiglu") {
          // fused_gemm_swiglu(x, W_gate, W_up) = silu(x @ W_gate) * (x @ W_up)
          if (slot.wiring.numInputs >= 3 && slot.wiring.numOutputs > 0) {
            int xSrc = slot.wiring.inputSourceIndices[0];
            int wGateSrc = slot.wiring.inputSourceIndices[1];
            int wUpSrc = slot.wiring.inputSourceIndices[2];
            int outIdx = slot.wiring.outputSlotIndices[0];
            auto xMem = getMemref(xSrc);
            auto wGateMem = getMemref(wGateSrc);
            auto wUpMem = getMemref(wUpSrc);
            auto outMem = getMemref(outIdx);
            if (xMem && wGateMem && wUpMem && outMem) {
              auto* xArr = bufferArgs[sourceToArgIdx[xSrc]].array;
              auto* wGateArr = bufferArgs[sourceToArgIdx[wGateSrc]].array;
              int64_t M = (xArr->rankOf() >= 2) ? xArr->sizeAt(0) : 1;
              int64_t K = (xArr->rankOf() >= 2) ? xArr->sizeAt(1) : xArr->lengthOf();
              int64_t N = (wGateArr->rankOf() >= 2) ? wGateArr->sizeAt(1) : 1;

              // gate = x @ W_gate (into output buffer)
              emitMatmulOp(builder, loc, xMem, wGateMem, outMem, M, N, K, f32Type);

              // Apply silu in-place on outMem (gate result), then multiply by up result.
              // This requires a second matmul for up. We'll use outMem for gate,
              // need a temporary for up. Since we don't have extra memrefs, we'll
              // emit the fused SwiGLU loop that computes both GEMMs and combines.
              // For now, use the sequential approach:
              // 1. gate = x @ W_gate → outMem
              // 2. Apply silu to outMem in-place
              // 3. Compute up = x @ W_up element-by-element and multiply with outMem

              // Step 2: Apply silu in-place on outMem
              int64_t outLen = M * N;
              auto zeroIdx2 = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
              auto oneIdx2 = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
              auto nVal2 = builder.create<mlir::arith::ConstantIndexOp>(loc, outLen);
              auto siluLoop = builder.create<mlir::scf::ForOp>(loc, zeroIdx2, nVal2, oneIdx2);
              {
                auto& body = siluLoop.getRegion().front();
                builder.setInsertionPointToStart(&body);
                auto iv = body.getArgument(0);
                auto val = builder.create<mlir::memref::LoadOp>(loc, outMem, mlir::ValueRange{iv});
                auto siluVal = emitUnaryElementwise(builder, loc, "silu", val, f32Type, nullptr, 0);
                builder.create<mlir::memref::StoreOp>(loc, siluVal, outMem, mlir::ValueRange{iv});
                builder.create<mlir::scf::YieldOp>(loc);
              }
              builder.setInsertionPointAfter(siluLoop);

              // Step 3: For each element, compute up_j = sum_k(x_ik * W_up_kj) and
              // multiply with silu(gate)_ij in outMem. This is a fused matmul+multiply.
              // Emit as a double loop: for i in M, for j in N
              auto iLoop = builder.create<mlir::scf::ForOp>(loc, zeroIdx2,
                  builder.create<mlir::arith::ConstantIndexOp>(loc, M), oneIdx2);
              builder.setInsertionPointToStart(iLoop.getBody());
              auto iIv = iLoop.getInductionVar();

              auto jLoop = builder.create<mlir::scf::ForOp>(loc, zeroIdx2,
                  builder.create<mlir::arith::ConstantIndexOp>(loc, N), oneIdx2);
              builder.setInsertionPointToStart(jLoop.getBody());
              auto jIv = jLoop.getInductionVar();

              // Accumulate up_ij = sum_k(x_ik * W_up_kj)
              auto zeroF = builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(f32Type, 0.0));
              auto kConst = builder.create<mlir::arith::ConstantIndexOp>(loc, K);
              auto kLoop = builder.create<mlir::scf::ForOp>(loc, zeroIdx2, kConst, oneIdx2,
                  mlir::ValueRange{zeroF.getResult()});
              {
                auto& kBody = kLoop.getRegion().front();
                builder.setInsertionPointToStart(&kBody);
                auto kIv = kBody.getArgument(0);
                auto acc = kBody.getArgument(1);
                // x[i*K + k]
                auto iTimesK = builder.create<mlir::arith::MulIOp>(loc, iIv,
                    builder.create<mlir::arith::ConstantIndexOp>(loc, K));
                auto xIdx = builder.create<mlir::arith::AddIOp>(loc, iTimesK, kIv);
                auto xVal = builder.create<mlir::memref::LoadOp>(loc, xMem, mlir::ValueRange{xIdx});
                // W_up[k*N + j]
                auto kTimesN = builder.create<mlir::arith::MulIOp>(loc, kIv,
                    builder.create<mlir::arith::ConstantIndexOp>(loc, N));
                auto wIdx = builder.create<mlir::arith::AddIOp>(loc, kTimesN, jIv);
                auto wVal = builder.create<mlir::memref::LoadOp>(loc, wUpMem, mlir::ValueRange{wIdx});
                auto prod = builder.create<mlir::arith::MulFOp>(loc, xVal, wVal);
                auto newAcc = builder.create<mlir::arith::AddFOp>(loc, acc, prod);
                builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{newAcc});
              }
              builder.setInsertionPointAfter(kLoop);
              auto upIJ = kLoop.getResult(0);

              // Multiply silu(gate)_ij * up_ij
              auto outIdxCalc = builder.create<mlir::arith::MulIOp>(loc, iIv,
                  builder.create<mlir::arith::ConstantIndexOp>(loc, N));
              auto outIdxFinal = builder.create<mlir::arith::AddIOp>(loc, outIdxCalc, jIv);
              auto gateVal = builder.create<mlir::memref::LoadOp>(loc, outMem, mlir::ValueRange{outIdxFinal});
              auto result = builder.create<mlir::arith::MulFOp>(loc, gateVal, upIJ);
              builder.create<mlir::memref::StoreOp>(loc, result, outMem, mlir::ValueRange{outIdxFinal});

              builder.create<mlir::scf::YieldOp>(loc);  // end j loop
              builder.setInsertionPointAfter(jLoop);
              builder.create<mlir::scf::YieldOp>(loc);  // end i loop
              builder.setInsertionPointAfter(iLoop);
            }
          }
        }
        break;
      }
      default:
        break;
    }
  }

  // Flush any remaining element-wise batch
  flushElementwiseBatch();

  builder.create<mlir::func::ReturnOp>(loc);

  DSP_DIAG(COMPILE, "CpuIRBuilder::buildModule: built module for segment [%d-%d] with %d buffer args "
            "and %d fused ops",
            startSlot, endSlot, static_cast<int>(bufferArgs.size()), segSize);

  return module;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Additional emitters: pooling, embedding, loss ops
// ═══════════════════════════════════════════════════════════════════════════════

void CpuIRBuilder::emitPooling2dOp(mlir::OpBuilder& builder, mlir::Location loc,
                                     const std::string& opName,
                                     mlir::Value inputMemref, mlir::Value outputMemref,
                                     const std::vector<LongType>& inputShape,
                                     const std::vector<LongType>& outputShape,
                                     int kH, int kW, int sH, int sW,
                                     int pH, int pW, int dH, int dW,
                                     mlir::Type elemType) {
  // Pooling 2D: NCHW format
  // input:  [N, C, iH, iW]
  // output: [N, C, oH, oW]
  if (inputShape.size() < 4 || outputShape.size() < 4) return;

  int64_t batchN = inputShape[0], C = inputShape[1];
  int64_t iH = inputShape[2], iW = inputShape[3];
  int64_t oH = outputShape[2], oW = outputShape[3];

  std::string lowerOp = opName;
  std::transform(lowerOp.begin(), lowerOp.end(), lowerOp.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  bool isMax = (lowerOp.find("max") != std::string::npos);

  auto zeroIdx = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
  auto oneIdx = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);

  // Loop over N, C, oH, oW
  auto nLoop = builder.create<mlir::scf::ForOp>(loc, zeroIdx,
      builder.create<mlir::arith::ConstantIndexOp>(loc, batchN), oneIdx);
  builder.setInsertionPointToStart(nLoop.getBody());
  auto n = nLoop.getInductionVar();

  auto cLoop = builder.create<mlir::scf::ForOp>(loc, zeroIdx,
      builder.create<mlir::arith::ConstantIndexOp>(loc, C), oneIdx);
  builder.setInsertionPointToStart(cLoop.getBody());
  auto c = cLoop.getInductionVar();

  auto ohLoop = builder.create<mlir::scf::ForOp>(loc, zeroIdx,
      builder.create<mlir::arith::ConstantIndexOp>(loc, oH), oneIdx);
  builder.setInsertionPointToStart(ohLoop.getBody());
  auto oh = ohLoop.getInductionVar();

  auto owLoop = builder.create<mlir::scf::ForOp>(loc, zeroIdx,
      builder.create<mlir::arith::ConstantIndexOp>(loc, oW), oneIdx);
  builder.setInsertionPointToStart(owLoop.getBody());
  auto ow = owLoop.getInductionVar();

  // Initialize accumulator
  mlir::Value initVal;
  if (isMax) {
    initVal = builder.create<mlir::arith::ConstantOp>(
        loc, builder.getFloatAttr(elemType, -1e38));
  } else {
    initVal = builder.create<mlir::arith::ConstantOp>(
        loc, builder.getFloatAttr(elemType, 0.0));
  }

  // Kernel loops
  auto khLoop = builder.create<mlir::scf::ForOp>(loc, zeroIdx,
      builder.create<mlir::arith::ConstantIndexOp>(loc, kH), oneIdx,
      mlir::ValueRange{initVal});
  builder.setInsertionPointToStart(khLoop.getBody());
  auto kh = khLoop.getInductionVar();
  auto accOuter = khLoop.getRegionIterArg(0);

  auto kwLoop = builder.create<mlir::scf::ForOp>(loc, zeroIdx,
      builder.create<mlir::arith::ConstantIndexOp>(loc, kW), oneIdx,
      mlir::ValueRange{accOuter});
  builder.setInsertionPointToStart(kwLoop.getBody());
  auto kw = kwLoop.getInductionVar();
  auto acc = kwLoop.getRegionIterArg(0);

  // ih = oh * sH - pH + kh * dH
  auto sHConst = builder.create<mlir::arith::ConstantIndexOp>(loc, sH);
  auto pHConst = builder.create<mlir::arith::ConstantIndexOp>(loc, pH);
  auto dHConst = builder.create<mlir::arith::ConstantIndexOp>(loc, dH);
  auto sWConst = builder.create<mlir::arith::ConstantIndexOp>(loc, sW);
  auto pWConst = builder.create<mlir::arith::ConstantIndexOp>(loc, pW);
  auto dWConst = builder.create<mlir::arith::ConstantIndexOp>(loc, dW);

  auto ih = builder.create<mlir::arith::AddIOp>(loc,
      builder.create<mlir::arith::SubIOp>(loc,
          builder.create<mlir::arith::MulIOp>(loc, oh, sHConst), pHConst),
      builder.create<mlir::arith::MulIOp>(loc, kh, dHConst));
  auto iw = builder.create<mlir::arith::AddIOp>(loc,
      builder.create<mlir::arith::SubIOp>(loc,
          builder.create<mlir::arith::MulIOp>(loc, ow, sWConst), pWConst),
      builder.create<mlir::arith::MulIOp>(loc, kw, dWConst));

  // Bounds check: ih >= 0 && ih < iH && iw >= 0 && iw < iW
  auto iHConst = builder.create<mlir::arith::ConstantIndexOp>(loc, iH);
  auto iWConst = builder.create<mlir::arith::ConstantIndexOp>(loc, iW);
  auto ihValid = builder.create<mlir::arith::AndIOp>(loc,
      builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sge, ih, zeroIdx),
      builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, ih, iHConst));
  auto iwValid = builder.create<mlir::arith::AndIOp>(loc,
      builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sge, iw, zeroIdx),
      builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, iw, iWConst));
  auto valid = builder.create<mlir::arith::AndIOp>(loc, ihValid, iwValid);

  // flat input index = n*C*iH*iW + c*iH*iW + ih*iW + iw
  auto cihiw = builder.create<mlir::arith::ConstantIndexOp>(loc, C * iH * iW);
  auto ihiw = builder.create<mlir::arith::ConstantIndexOp>(loc, iH * iW);

  auto inIdx = builder.create<mlir::arith::AddIOp>(loc,
      builder.create<mlir::arith::AddIOp>(loc,
          builder.create<mlir::arith::MulIOp>(loc, n, cihiw),
          builder.create<mlir::arith::MulIOp>(loc, c, ihiw)),
      builder.create<mlir::arith::AddIOp>(loc,
          builder.create<mlir::arith::MulIOp>(loc, ih, iWConst), iw));

  // Conditional load and accumulate
  auto ifOp = builder.create<mlir::scf::IfOp>(loc, mlir::TypeRange{elemType}, valid, true);
  // Then block: valid input position
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  auto val = builder.create<mlir::memref::LoadOp>(loc, inputMemref, mlir::ValueRange{inIdx});
  mlir::Value newAcc;
  if (isMax) {
    newAcc = builder.create<mlir::arith::MaximumFOp>(loc, acc, val);
  } else {
    newAcc = builder.create<mlir::arith::AddFOp>(loc, acc, val);
  }
  builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{newAcc});

  // Else block: out of bounds, keep accumulator
  builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
  builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{acc});

  builder.setInsertionPointAfter(ifOp);
  builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{ifOp.getResult(0)});

  // End kw loop
  builder.setInsertionPointAfter(kwLoop);
  builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{kwLoop.getResult(0)});

  // End kh loop
  builder.setInsertionPointAfter(khLoop);
  mlir::Value poolResult = khLoop.getResult(0);

  // For avg pooling, divide by kernel size
  if (!isMax) {
    auto kernelSize = builder.create<mlir::arith::ConstantOp>(
        loc, builder.getFloatAttr(elemType, static_cast<double>(kH * kW)));
    poolResult = builder.create<mlir::arith::DivFOp>(loc, poolResult, kernelSize);
  }

  // Store result: flat output index = n*C*oH*oW + c*oH*oW + oh*oW + ow
  auto cohow = builder.create<mlir::arith::ConstantIndexOp>(loc, C * oH * oW);
  auto ohow = builder.create<mlir::arith::ConstantIndexOp>(loc, oH * oW);
  auto oWConst = builder.create<mlir::arith::ConstantIndexOp>(loc, oW);

  auto outIdx = builder.create<mlir::arith::AddIOp>(loc,
      builder.create<mlir::arith::AddIOp>(loc,
          builder.create<mlir::arith::MulIOp>(loc, n, cohow),
          builder.create<mlir::arith::MulIOp>(loc, c, ohow)),
      builder.create<mlir::arith::AddIOp>(loc,
          builder.create<mlir::arith::MulIOp>(loc, oh, oWConst), ow));

  builder.create<mlir::memref::StoreOp>(loc, poolResult, outputMemref, mlir::ValueRange{outIdx});

  // Close all loops
  builder.setInsertionPointAfter(owLoop);
  builder.setInsertionPointAfter(ohLoop);
  builder.setInsertionPointAfter(cLoop);
  builder.setInsertionPointAfter(nLoop);
}

void CpuIRBuilder::emitEmbeddingLookup(mlir::OpBuilder& builder, mlir::Location loc,
                                          mlir::Value tableMemref, mlir::Value indicesMemref,
                                          mlir::Value outputMemref,
                                          int64_t numLookups, int64_t embeddingDim,
                                          mlir::Type elemType) {
  // output[i*D+d] = table[indices[i]*D+d] for i in numLookups, d in D
  auto zeroIdx = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
  auto oneIdx = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
  auto nConst = builder.create<mlir::arith::ConstantIndexOp>(loc, numLookups);
  auto dConst = builder.create<mlir::arith::ConstantIndexOp>(loc, embeddingDim);

  auto iLoop = builder.create<mlir::scf::ForOp>(loc, zeroIdx, nConst, oneIdx);
  builder.setInsertionPointToStart(iLoop.getBody());
  auto i = iLoop.getInductionVar();

  // Load index (as integer, then cast to index)
  auto idxVal = builder.create<mlir::memref::LoadOp>(loc, indicesMemref, mlir::ValueRange{i});
  auto idxIndex = builder.create<mlir::arith::IndexCastOp>(
      loc, builder.getIndexType(), idxVal);

  auto dLoop = builder.create<mlir::scf::ForOp>(loc, zeroIdx, dConst, oneIdx);
  builder.setInsertionPointToStart(dLoop.getBody());
  auto d = dLoop.getInductionVar();

  // table[idx*D + d]
  auto tableIdx = builder.create<mlir::arith::AddIOp>(loc,
      builder.create<mlir::arith::MulIOp>(loc, idxIndex, dConst), d);
  auto val = builder.create<mlir::memref::LoadOp>(loc, tableMemref, mlir::ValueRange{tableIdx});

  // output[i*D + d]
  auto outIdx = builder.create<mlir::arith::AddIOp>(loc,
      builder.create<mlir::arith::MulIOp>(loc, i, dConst), d);
  builder.create<mlir::memref::StoreOp>(loc, val, outputMemref, mlir::ValueRange{outIdx});

  builder.setInsertionPointAfter(dLoop);
  builder.setInsertionPointAfter(iLoop);
}

void CpuIRBuilder::emitSoftmaxCrossEntropy(mlir::OpBuilder& builder, mlir::Location loc,
                                              mlir::Value logitsMemref, mlir::Value labelsMemref,
                                              mlir::Value outputMemref,
                                              int64_t nElements, mlir::Type elemType) {
  // loss = -sum(labels * log(softmax(logits)))
  // = -sum(labels * (logits - log(sum(exp(logits)))))
  auto zeroIdx = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
  auto oneIdx = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
  auto nConst = builder.create<mlir::arith::ConstantIndexOp>(loc, nElements);
  auto negInf = builder.create<mlir::arith::ConstantOp>(
      loc, builder.getFloatAttr(elemType, -1e38));
  auto zeroF = builder.create<mlir::arith::ConstantOp>(
      loc, builder.getFloatAttr(elemType, 0.0));

  // Pass 1: find max for numerical stability
  auto maxLoop = builder.create<mlir::scf::ForOp>(
      loc, zeroIdx, nConst, oneIdx, mlir::ValueRange{negInf.getResult()});
  {
    builder.setInsertionPointToStart(maxLoop.getBody());
    auto iv = maxLoop.getInductionVar();
    auto acc = maxLoop.getRegionIterArg(0);
    auto val = builder.create<mlir::memref::LoadOp>(loc, logitsMemref, mlir::ValueRange{iv});
    auto newMax = builder.create<mlir::arith::MaximumFOp>(loc, acc, val);
    builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{newMax});
  }
  builder.setInsertionPointAfter(maxLoop);
  auto maxVal = maxLoop.getResult(0);

  // Pass 2: log(sum(exp(logits - max)))
  auto sumLoop = builder.create<mlir::scf::ForOp>(
      loc, zeroIdx, nConst, oneIdx, mlir::ValueRange{zeroF.getResult()});
  {
    builder.setInsertionPointToStart(sumLoop.getBody());
    auto iv = sumLoop.getInductionVar();
    auto acc = sumLoop.getRegionIterArg(0);
    auto val = builder.create<mlir::memref::LoadOp>(loc, logitsMemref, mlir::ValueRange{iv});
    auto shifted = builder.create<mlir::arith::SubFOp>(loc, val, maxVal);
    auto expVal = builder.create<mlir::math::ExpOp>(loc, shifted);
    auto newSum = builder.create<mlir::arith::AddFOp>(loc, acc, expVal);
    builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{newSum});
  }
  builder.setInsertionPointAfter(sumLoop);
  auto logSumExp = builder.create<mlir::math::LogOp>(loc, sumLoop.getResult(0));

  // Pass 3: loss = -sum(labels * (logits - max - logSumExp))
  auto lossLoop = builder.create<mlir::scf::ForOp>(
      loc, zeroIdx, nConst, oneIdx, mlir::ValueRange{zeroF.getResult()});
  {
    builder.setInsertionPointToStart(lossLoop.getBody());
    auto iv = lossLoop.getInductionVar();
    auto acc = lossLoop.getRegionIterArg(0);
    auto logit = builder.create<mlir::memref::LoadOp>(loc, logitsMemref, mlir::ValueRange{iv});
    auto label = builder.create<mlir::memref::LoadOp>(loc, labelsMemref, mlir::ValueRange{iv});
    auto logSoftmax = builder.create<mlir::arith::SubFOp>(loc,
        builder.create<mlir::arith::SubFOp>(loc, logit, maxVal), logSumExp);
    auto term = builder.create<mlir::arith::MulFOp>(loc, label, logSoftmax);
    auto newAcc = builder.create<mlir::arith::AddFOp>(loc, acc, term);
    builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{newAcc});
  }
  builder.setInsertionPointAfter(lossLoop);

  // Negate the sum
  auto loss = builder.create<mlir::arith::NegFOp>(loc, lossLoop.getResult(0));

  // Store scalar output
  builder.create<mlir::memref::StoreOp>(loc, loss, outputMemref, mlir::ValueRange{zeroIdx});
}

void CpuIRBuilder::emitMSELoss(mlir::OpBuilder& builder, mlir::Location loc,
                                  mlir::Value predictionsMemref, mlir::Value labelsMemref,
                                  mlir::Value outputMemref,
                                  int64_t nElements, mlir::Type elemType) {
  // MSE = mean((predictions - labels)^2)
  auto zeroIdx = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
  auto oneIdx = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
  auto nConst = builder.create<mlir::arith::ConstantIndexOp>(loc, nElements);
  auto zeroF = builder.create<mlir::arith::ConstantOp>(
      loc, builder.getFloatAttr(elemType, 0.0));

  auto loop = builder.create<mlir::scf::ForOp>(
      loc, zeroIdx, nConst, oneIdx, mlir::ValueRange{zeroF.getResult()});
  {
    builder.setInsertionPointToStart(loop.getBody());
    auto iv = loop.getInductionVar();
    auto acc = loop.getRegionIterArg(0);
    auto pred = builder.create<mlir::memref::LoadOp>(loc, predictionsMemref, mlir::ValueRange{iv});
    auto label = builder.create<mlir::memref::LoadOp>(loc, labelsMemref, mlir::ValueRange{iv});
    auto diff = builder.create<mlir::arith::SubFOp>(loc, pred, label);
    auto sq = builder.create<mlir::arith::MulFOp>(loc, diff, diff);
    auto newAcc = builder.create<mlir::arith::AddFOp>(loc, acc, sq);
    builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{newAcc});
  }
  builder.setInsertionPointAfter(loop);

  auto countF = builder.create<mlir::arith::ConstantOp>(
      loc, builder.getFloatAttr(elemType, static_cast<double>(nElements)));
  auto mse = builder.create<mlir::arith::DivFOp>(loc, loop.getResult(0), countF);
  builder.create<mlir::memref::StoreOp>(loc, mse, outputMemref, mlir::ValueRange{zeroIdx});
}

// ─── New emitters ─────────────────────────────────────────────────────────

void CpuIRBuilder::emitSliceOp(mlir::OpBuilder& builder, mlir::Location loc,
                                mlir::Value inputMemref, mlir::Value outputMemref,
                                int64_t begin, int64_t length, mlir::Type elemType) {
  auto zeroIdx = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
  auto oneIdx = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
  auto beginConst = builder.create<mlir::arith::ConstantIndexOp>(loc, begin);
  auto lenConst = builder.create<mlir::arith::ConstantIndexOp>(loc, length);

  auto loop = builder.create<mlir::scf::ForOp>(loc, zeroIdx, lenConst, oneIdx);
  builder.setInsertionPointToStart(loop.getBody());
  auto iv = loop.getInductionVar();
  auto srcIdx = builder.create<mlir::arith::AddIOp>(loc, iv, beginConst);
  auto val = builder.create<mlir::memref::LoadOp>(loc, inputMemref, mlir::ValueRange{srcIdx});
  builder.create<mlir::memref::StoreOp>(loc, val, outputMemref, mlir::ValueRange{iv});
  builder.setInsertionPointAfter(loop);
}

void CpuIRBuilder::emitLogicalOp(mlir::OpBuilder& builder, mlir::Location loc,
                                   const std::string& opName,
                                   mlir::Value lhsMemref, mlir::Value rhsMemref,
                                   mlir::Value outputMemref,
                                   int64_t nElements, mlir::Type elemType) {
  auto zeroIdx = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
  auto oneIdx = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
  auto nConst = builder.create<mlir::arith::ConstantIndexOp>(loc, nElements);
  auto zeroF = builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(elemType, 0.0));
  auto oneF = builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(elemType, 1.0));

  auto loop = builder.create<mlir::scf::ForOp>(loc, zeroIdx, nConst, oneIdx);
  builder.setInsertionPointToStart(loop.getBody());
  auto iv = loop.getInductionVar();
  auto lhs = builder.create<mlir::memref::LoadOp>(loc, lhsMemref, mlir::ValueRange{iv});
  auto rhs = builder.create<mlir::memref::LoadOp>(loc, rhsMemref, mlir::ValueRange{iv});
  auto lhsBool = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::ONE, lhs, zeroF);
  auto rhsBool = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::ONE, rhs, zeroF);

  mlir::Value boolResult;
  if (opName.find("and") != std::string::npos) {
    boolResult = builder.create<mlir::arith::AndIOp>(loc, lhsBool, rhsBool);
  } else if (opName.find("or") != std::string::npos) {
    boolResult = builder.create<mlir::arith::OrIOp>(loc, lhsBool, rhsBool);
  } else {
    boolResult = builder.create<mlir::arith::XOrIOp>(loc, lhsBool, rhsBool);
  }

  auto result = builder.create<mlir::arith::SelectOp>(loc, boolResult, oneF, zeroF);
  builder.create<mlir::memref::StoreOp>(loc, result, outputMemref, mlir::ValueRange{iv});
  builder.setInsertionPointAfter(loop);
}

void CpuIRBuilder::emitHuberLoss(mlir::OpBuilder& builder, mlir::Location loc,
                                   mlir::Value predictionsMemref, mlir::Value labelsMemref,
                                   mlir::Value outputMemref,
                                   int64_t nElements, double delta, mlir::Type elemType) {
  auto zeroIdx = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
  auto oneIdx = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
  auto nConst = builder.create<mlir::arith::ConstantIndexOp>(loc, nElements);
  auto zeroF = builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(elemType, 0.0));
  auto deltaF = builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(elemType, delta));
  auto halfF = builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(elemType, 0.5));

  auto loop = builder.create<mlir::scf::ForOp>(
      loc, zeroIdx, nConst, oneIdx, mlir::ValueRange{zeroF.getResult()});
  {
    builder.setInsertionPointToStart(loop.getBody());
    auto iv = loop.getInductionVar();
    auto acc = loop.getRegionIterArg(0);
    auto pred = builder.create<mlir::memref::LoadOp>(loc, predictionsMemref, mlir::ValueRange{iv});
    auto label = builder.create<mlir::memref::LoadOp>(loc, labelsMemref, mlir::ValueRange{iv});
    auto diff = builder.create<mlir::arith::SubFOp>(loc, pred, label);
    auto absDiff = builder.create<mlir::math::AbsFOp>(loc, diff);
    auto isSmall = builder.create<mlir::arith::CmpFOp>(
        loc, mlir::arith::CmpFPredicate::OLE, absDiff, deltaF);
    // MSE path: 0.5 * diff^2
    auto sq = builder.create<mlir::arith::MulFOp>(loc, diff, diff);
    auto msePath = builder.create<mlir::arith::MulFOp>(loc, halfF, sq);
    // MAE path: delta * (|diff| - 0.5 * delta)
    auto halfDelta = builder.create<mlir::arith::MulFOp>(loc, halfF, deltaF);
    auto maePath = builder.create<mlir::arith::MulFOp>(loc, deltaF,
        builder.create<mlir::arith::SubFOp>(loc, absDiff, halfDelta));
    auto loss = builder.create<mlir::arith::SelectOp>(loc, isSmall, msePath, maePath);
    auto newAcc = builder.create<mlir::arith::AddFOp>(loc, acc, loss);
    builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{newAcc});
  }
  builder.setInsertionPointAfter(loop);

  auto countF = builder.create<mlir::arith::ConstantOp>(
      loc, builder.getFloatAttr(elemType, static_cast<double>(nElements)));
  auto result = builder.create<mlir::arith::DivFOp>(loc, loop.getResult(0), countF);
  builder.create<mlir::memref::StoreOp>(loc, result, outputMemref, mlir::ValueRange{zeroIdx});
}

void CpuIRBuilder::emitHingeLoss(mlir::OpBuilder& builder, mlir::Location loc,
                                   mlir::Value predictionsMemref, mlir::Value labelsMemref,
                                   mlir::Value outputMemref,
                                   int64_t nElements, mlir::Type elemType) {
  auto zeroIdx = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
  auto oneIdx = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
  auto nConst = builder.create<mlir::arith::ConstantIndexOp>(loc, nElements);
  auto zeroF = builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(elemType, 0.0));
  auto oneF = builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(elemType, 1.0));

  auto loop = builder.create<mlir::scf::ForOp>(
      loc, zeroIdx, nConst, oneIdx, mlir::ValueRange{zeroF.getResult()});
  {
    builder.setInsertionPointToStart(loop.getBody());
    auto iv = loop.getInductionVar();
    auto acc = loop.getRegionIterArg(0);
    auto pred = builder.create<mlir::memref::LoadOp>(loc, predictionsMemref, mlir::ValueRange{iv});
    auto label = builder.create<mlir::memref::LoadOp>(loc, labelsMemref, mlir::ValueRange{iv});
    // max(0, 1 - y*t)
    auto yt = builder.create<mlir::arith::MulFOp>(loc, label, pred);
    auto margin = builder.create<mlir::arith::SubFOp>(loc, oneF, yt);
    auto loss = builder.create<mlir::arith::MaximumFOp>(loc, zeroF, margin);
    auto newAcc = builder.create<mlir::arith::AddFOp>(loc, acc, loss);
    builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{newAcc});
  }
  builder.setInsertionPointAfter(loop);

  auto countF = builder.create<mlir::arith::ConstantOp>(
      loc, builder.getFloatAttr(elemType, static_cast<double>(nElements)));
  auto result = builder.create<mlir::arith::DivFOp>(loc, loop.getResult(0), countF);
  builder.create<mlir::memref::StoreOp>(loc, result, outputMemref, mlir::ValueRange{zeroIdx});
}

void CpuIRBuilder::emitLogLoss(mlir::OpBuilder& builder, mlir::Location loc,
                                 mlir::Value predictionsMemref, mlir::Value labelsMemref,
                                 mlir::Value outputMemref,
                                 int64_t nElements, double epsilon, mlir::Type elemType) {
  auto zeroIdx = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
  auto oneIdx = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
  auto nConst = builder.create<mlir::arith::ConstantIndexOp>(loc, nElements);
  auto zeroF = builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(elemType, 0.0));
  auto oneF = builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(elemType, 1.0));
  auto epsF = builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(elemType, epsilon));

  auto loop = builder.create<mlir::scf::ForOp>(
      loc, zeroIdx, nConst, oneIdx, mlir::ValueRange{zeroF.getResult()});
  {
    builder.setInsertionPointToStart(loop.getBody());
    auto iv = loop.getInductionVar();
    auto acc = loop.getRegionIterArg(0);
    auto pred = builder.create<mlir::memref::LoadOp>(loc, predictionsMemref, mlir::ValueRange{iv});
    auto label = builder.create<mlir::memref::LoadOp>(loc, labelsMemref, mlir::ValueRange{iv});
    // -[y*log(p+eps) + (1-y)*log(1-p+eps)]
    auto pEps = builder.create<mlir::arith::AddFOp>(loc, pred, epsF);
    auto logP = builder.create<mlir::math::LogOp>(loc, pEps);
    auto term1 = builder.create<mlir::arith::MulFOp>(loc, label, logP);
    auto oneMinusY = builder.create<mlir::arith::SubFOp>(loc, oneF, label);
    auto oneMinusPEps = builder.create<mlir::arith::AddFOp>(loc,
        builder.create<mlir::arith::SubFOp>(loc, oneF, pred), epsF);
    auto logOneMinusP = builder.create<mlir::math::LogOp>(loc, oneMinusPEps);
    auto term2 = builder.create<mlir::arith::MulFOp>(loc, oneMinusY, logOneMinusP);
    auto loss = builder.create<mlir::arith::AddFOp>(loc, term1, term2);
    auto negLoss = builder.create<mlir::arith::NegFOp>(loc, loss);
    auto newAcc = builder.create<mlir::arith::AddFOp>(loc, acc, negLoss);
    builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{newAcc});
  }
  builder.setInsertionPointAfter(loop);

  auto countF = builder.create<mlir::arith::ConstantOp>(
      loc, builder.getFloatAttr(elemType, static_cast<double>(nElements)));
  auto result = builder.create<mlir::arith::DivFOp>(loc, loop.getResult(0), countF);
  builder.create<mlir::memref::StoreOp>(loc, result, outputMemref, mlir::ValueRange{zeroIdx});
}

void CpuIRBuilder::emitOneHot(mlir::OpBuilder& builder, mlir::Location loc,
                                mlir::Value indicesMemref, mlir::Value outputMemref,
                                int64_t numIndices, int64_t depth,
                                double onValue, double offValue, mlir::Type elemType) {
  auto zeroIdx = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
  auto oneIdx = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
  auto depthConst = builder.create<mlir::arith::ConstantIndexOp>(loc, depth);
  auto numConst = builder.create<mlir::arith::ConstantIndexOp>(loc, numIndices);
  auto onF = builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(elemType, onValue));
  auto offF = builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(elemType, offValue));

  // Fill with off-value
  auto totalLen = builder.create<mlir::arith::ConstantIndexOp>(loc, numIndices * depth);
  auto fillLoop = builder.create<mlir::scf::ForOp>(loc, zeroIdx, totalLen, oneIdx);
  {
    builder.setInsertionPointToStart(fillLoop.getBody());
    builder.create<mlir::memref::StoreOp>(loc, offF, outputMemref,
        mlir::ValueRange{fillLoop.getInductionVar()});
  }
  builder.setInsertionPointAfter(fillLoop);

  // Set on-values at indices
  auto iLoop = builder.create<mlir::scf::ForOp>(loc, zeroIdx, numConst, oneIdx);
  {
    builder.setInsertionPointToStart(iLoop.getBody());
    auto i = iLoop.getInductionVar();
    auto idxVal = builder.create<mlir::memref::LoadOp>(loc, indicesMemref, mlir::ValueRange{i});
    // Convert float index to integer index
    auto idxI64 = builder.create<mlir::arith::FPToSIOp>(loc, builder.getI64Type(), idxVal);
    auto idx = builder.create<mlir::arith::IndexCastOp>(loc, builder.getIndexType(), idxI64);
    auto outIdx = builder.create<mlir::arith::AddIOp>(loc,
        builder.create<mlir::arith::MulIOp>(loc, i, depthConst), idx);
    builder.create<mlir::memref::StoreOp>(loc, onF, outputMemref, mlir::ValueRange{outIdx});
  }
  builder.setInsertionPointAfter(iLoop);
}

void CpuIRBuilder::emitXWPlusB(mlir::OpBuilder& builder, mlir::Location loc,
                                  mlir::Value xMemref, mlir::Value wMemref,
                                  mlir::Value biasMemref, mlir::Value outputMemref,
                                  int64_t batch, int64_t inFeatures, int64_t outFeatures,
                                  mlir::Type elemType) {
  auto zeroIdx = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
  auto oneIdx = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
  auto batchConst = builder.create<mlir::arith::ConstantIndexOp>(loc, batch);
  auto inConst = builder.create<mlir::arith::ConstantIndexOp>(loc, inFeatures);
  auto outConst = builder.create<mlir::arith::ConstantIndexOp>(loc, outFeatures);

  auto iLoop = builder.create<mlir::scf::ForOp>(loc, zeroIdx, batchConst, oneIdx);
  {
    builder.setInsertionPointToStart(iLoop.getBody());
    auto i = iLoop.getInductionVar();

    auto jLoop = builder.create<mlir::scf::ForOp>(loc, zeroIdx, outConst, oneIdx);
    {
      builder.setInsertionPointToStart(jLoop.getBody());
      auto j = jLoop.getInductionVar();

      // Load bias[j]
      auto biasVal = builder.create<mlir::memref::LoadOp>(loc, biasMemref, mlir::ValueRange{j});

      // Accumulate x[i,k] * w[k,j] for k in 0..inFeatures
      auto kLoop = builder.create<mlir::scf::ForOp>(
          loc, zeroIdx, inConst, oneIdx, mlir::ValueRange{biasVal.getResult()});
      {
        builder.setInsertionPointToStart(kLoop.getBody());
        auto k = kLoop.getInductionVar();
        auto acc = kLoop.getRegionIterArg(0);
        // x[i*inFeatures + k]
        auto xIdx = builder.create<mlir::arith::AddIOp>(loc,
            builder.create<mlir::arith::MulIOp>(loc, i, inConst), k);
        auto xVal = builder.create<mlir::memref::LoadOp>(loc, xMemref, mlir::ValueRange{xIdx});
        // w[k*outFeatures + j]
        auto wIdx = builder.create<mlir::arith::AddIOp>(loc,
            builder.create<mlir::arith::MulIOp>(loc, k, outConst), j);
        auto wVal = builder.create<mlir::memref::LoadOp>(loc, wMemref, mlir::ValueRange{wIdx});
        auto prod = builder.create<mlir::arith::MulFOp>(loc, xVal, wVal);
        auto newAcc = builder.create<mlir::arith::AddFOp>(loc, acc, prod);
        builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{newAcc});
      }
      builder.setInsertionPointAfter(kLoop);

      // Store output[i*outFeatures + j]
      auto outIdx = builder.create<mlir::arith::AddIOp>(loc,
          builder.create<mlir::arith::MulIOp>(loc, i, outConst), j);
      builder.create<mlir::memref::StoreOp>(loc, kLoop.getResult(0), outputMemref,
          mlir::ValueRange{outIdx});
    }
    builder.setInsertionPointAfter(jLoop);
  }
  builder.setInsertionPointAfter(iLoop);
}

}  // namespace graph
}  // namespace sd

#endif  // HAVE_MLIR
