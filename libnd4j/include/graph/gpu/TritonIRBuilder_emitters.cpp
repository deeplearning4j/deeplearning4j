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
// TritonIRBuilder — Scalar op emission:
//   emitBinaryElementwise, emitUnaryElementwise, emitComparisonOp,
//   emitLogicalOp, emitTernaryOp, emitReductionOp, emitNormalizationOp
//

#include <config.h>

#if HAVE_TRITON

#include <graph/gpu/TritonIRBuilder.h>
#include <graph/gpu/TritonIRBuilder_internal.h>
#include <graph/DspDiagnostics.h>

#include <algorithm>
#include <cmath>
#include <string>

// MLIR core
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>

// Triton MLIR dialect
#include <triton/Dialect/Triton/IR/Dialect.h>
#include <triton/Dialect/Triton/IR/Types.h>

// Standard MLIR dialects
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Math/IR/Math.h>

namespace sd {
namespace graph {

using namespace ir_builder_internal;

mlir::Value TritonIRBuilder::emitNativeCudaExp(mlir::OpBuilder& builder,
                                                mlir::Location loc,
                                                mlir::Value input) {
  auto tensorType = mlir::cast<mlir::RankedTensorType>(input.getType());
  auto elementType = tensorType.getElementType();
  if (!elementType.isF32() && !elementType.isF64()) {
    return builder.create<mlir::math::ExpOp>(loc, input);
  }

  // sd::math::sd_exp clamps to [-88, 88] and then calls CUDA expf/exp.
  // Triton's generic math.exp uses the approximate ex2 instruction, which can
  // differ by one ULP and perturb recurrent state. Call the same libdevice
  // entry points as native CUDA so slot-by-slot and replay are raw-bit equal.
  auto scalar = [&](double value) -> mlir::Value {
    auto constant = builder.create<mlir::arith::ConstantOp>(
        loc, elementType, builder.getFloatAttr(elementType, value));
    return builder.create<mlir::triton::SplatOp>(loc, tensorType, constant);
  };
  auto clampedLow = builder.create<mlir::arith::MaximumFOp>(loc, input, scalar(-88.0));
  auto clamped = builder.create<mlir::arith::MinimumFOp>(loc, clampedLow, scalar(88.0));
  const char* symbol = elementType.isF32() ? "__nv_expf" : "__nv_exp";
  return builder.create<mlir::triton::ExternElementwiseOp>(
      loc, tensorType, mlir::ValueRange{clamped.getResult()},
      /*libname=*/"", /*libpath=*/"", symbol, /*pure=*/true).getResult();
}

mlir::Value TritonIRBuilder::emitNativeCudaLog(mlir::OpBuilder& builder,
                                                mlir::Location loc,
                                                mlir::Value input) {
  auto tensorType = mlir::cast<mlir::RankedTensorType>(input.getType());
  auto elementType = tensorType.getElementType();
  if (!elementType.isF32() && !elementType.isF64()) {
    return builder.create<mlir::math::LogOp>(loc, input);
  }

  // sd::math::sd_log substitutes SD_EPSILON for zero, then calls CUDA
  // logf/log. Triton's generic math.log uses an approximation that differs
  // by several ULP in softplus and perturbs recurrent state at the first
  // compiled execution.
  auto scalar = [&](double value) -> mlir::Value {
    auto constant = builder.create<mlir::arith::ConstantOp>(
        loc, elementType, builder.getFloatAttr(elementType, value));
    return builder.create<mlir::triton::SplatOp>(loc, tensorType, constant);
  };
  auto isZero = builder.create<mlir::arith::CmpFOp>(
      loc, mlir::arith::CmpFPredicate::OEQ, input, scalar(0.0));
  auto guarded = builder.create<mlir::arith::SelectOp>(
      loc, isZero, scalar(1.0e-5), input);
  const char* symbol = elementType.isF32() ? "__nv_logf" : "__nv_log";
  return builder.create<mlir::triton::ExternElementwiseOp>(
      loc, tensorType, mlir::ValueRange{guarded.getResult()},
      /*libname=*/"", /*libpath=*/"", symbol, /*pure=*/true).getResult();
}

mlir::Value TritonIRBuilder::emitNativeCudaPow(mlir::OpBuilder& builder,
                                                mlir::Location loc,
                                                mlir::Value base,
                                                mlir::Value exponent) {
  auto tensorType = mlir::cast<mlir::RankedTensorType>(base.getType());
  auto elementType = tensorType.getElementType();
  if (!elementType.isF32() && !elementType.isF64()) {
    return builder.create<mlir::math::PowFOp>(loc, base, exponent);
  }

  // Match native CUDA powf/pow instead of reconstructing power through
  // approximate exp/log operations.
  const char* symbol = elementType.isF32() ? "__nv_powf" : "__nv_pow";
  return builder.create<mlir::triton::ExternElementwiseOp>(
      loc, tensorType, mlir::ValueRange{base, exponent},
      /*libname=*/"", /*libpath=*/"", symbol, /*pure=*/true).getResult();
}

mlir::Value TritonIRBuilder::emitNativeCudaCos(mlir::OpBuilder& builder,
                                                mlir::Location loc,
                                                mlir::Value input) {
  auto tensorType = mlir::cast<mlir::RankedTensorType>(input.getType());
  auto elementType = tensorType.getElementType();
  if (!elementType.isF32() && !elementType.isF64()) {
    return builder.create<mlir::math::CosOp>(loc, input);
  }

  const char* symbol = elementType.isF32() ? "__nv_cosf" : "__nv_cos";
  return builder.create<mlir::triton::ExternElementwiseOp>(
      loc, tensorType, mlir::ValueRange{input},
      /*libname=*/"", /*libpath=*/"", symbol, /*pure=*/true).getResult();
}

mlir::Value TritonIRBuilder::emitNativeCudaSin(mlir::OpBuilder& builder,
                                                mlir::Location loc,
                                                mlir::Value input) {
  auto tensorType = mlir::cast<mlir::RankedTensorType>(input.getType());
  auto elementType = tensorType.getElementType();
  if (!elementType.isF32() && !elementType.isF64()) {
    return builder.create<mlir::math::SinOp>(loc, input);
  }

  const char* symbol = elementType.isF32() ? "__nv_sinf" : "__nv_sin";
  return builder.create<mlir::triton::ExternElementwiseOp>(
      loc, tensorType, mlir::ValueRange{input},
      /*libname=*/"", /*libpath=*/"", symbol, /*pure=*/true).getResult();
}

mlir::Value TritonIRBuilder::emitNativeCudaMulRn(mlir::OpBuilder& builder,
                                                  mlir::Location loc,
                                                  mlir::Value lhs,
                                                  mlir::Value rhs) {
  auto tensorType = mlir::cast<mlir::RankedTensorType>(lhs.getType());
  auto elementType = tensorType.getElementType();
  if (!elementType.isF32() && !elementType.isF64()) {
    return builder.create<mlir::arith::MulFOp>(loc, lhs, rhs);
  }

  const char* symbol = elementType.isF32() ? "__nv_fmul_rn" : "__nv_dmul_rn";
  return builder.create<mlir::triton::ExternElementwiseOp>(
      loc, tensorType, mlir::ValueRange{lhs, rhs},
      /*libname=*/"", /*libpath=*/"", symbol, /*pure=*/true).getResult();
}

mlir::Value TritonIRBuilder::emitNativeCudaFmaRn(mlir::OpBuilder& builder,
                                                  mlir::Location loc,
                                                  mlir::Value lhs,
                                                  mlir::Value rhs,
                                                  mlir::Value addend) {
  auto tensorType = mlir::cast<mlir::RankedTensorType>(lhs.getType());
  auto elementType = tensorType.getElementType();
  if (!elementType.isF32() && !elementType.isF64()) {
    auto product = builder.create<mlir::arith::MulFOp>(loc, lhs, rhs);
    return builder.create<mlir::arith::AddFOp>(loc, product, addend);
  }

  const char* symbol = elementType.isF32() ? "__nv_fmaf_rn" : "__nv_fma_rn";
  return builder.create<mlir::triton::ExternElementwiseOp>(
      loc, tensorType, mlir::ValueRange{lhs, rhs, addend},
      /*libname=*/"", /*libpath=*/"", symbol, /*pure=*/true).getResult();
}

mlir::Value TritonIRBuilder::emitNativeCudaDiv(mlir::OpBuilder& builder,
                                                mlir::Location loc,
                                                mlir::Value lhs,
                                                mlir::Value rhs) {
  // precise_divf is FP32-only. FP64 arith.divf already retains IEEE division.
  if (getElementType(lhs).isF32()) {
    return builder.create<mlir::triton::PreciseDivFOp>(loc, lhs, rhs);
  }
  return builder.create<mlir::arith::DivFOp>(loc, lhs, rhs);
}

mlir::Value TritonIRBuilder::emitBinaryElementwise(mlir::OpBuilder& builder, mlir::Location loc,
                                                    const TritonOpMapping& mapping,
                                                    const NativeSlot& slot,
                                                    mlir::Value lhs, mlir::Value rhs) {
  auto opIr = mapping.tritonIrOp;
  bool lhsIsFloat = isFloatType(lhs.getType());
  bool rhsIsFloat = isFloatType(rhs.getType());
  bool bothInt = !lhsIsFloat && !rhsIsFloat;
  bool lhsIsBool = isBoolType(lhs.getType());
  bool rhsIsBool = isBoolType(rhs.getType());

  // Integer/bool path: stay in integer domain when both operands are integer
  if (bothInt) {
    // Coerce to same integer width (widen narrower operand)
    if (!lhsIsBool && !rhsIsBool) {
      auto intTy = commonIntType(lhs, rhs);
      lhs = castTo(builder, loc, lhs, intTy);
      rhs = castTo(builder, loc, rhs, intTy);
    } else if (lhsIsBool && rhsIsBool) {
      // Both bool — use logical ops
      if (opIr == "arith.mulf") return builder.create<mlir::arith::AndIOp>(loc, lhs, rhs);
      if (opIr == "arith.addf") return builder.create<mlir::arith::OrIOp>(loc, lhs, rhs);
      if (opIr == "arith.maximumf") return builder.create<mlir::arith::OrIOp>(loc, lhs, rhs);
      if (opIr == "arith.minimumf") return builder.create<mlir::arith::AndIOp>(loc, lhs, rhs);
      // For sub/div on bools, promote to i32
      lhs = castTo(builder, loc, lhs, builder.getI32Type());
      rhs = castTo(builder, loc, rhs, builder.getI32Type());
    } else {
      // Mixed bool + int: promote bool to the int type
      auto intTy = lhsIsBool ? getElementType(rhs) : getElementType(lhs);
      lhs = castTo(builder, loc, lhs, intTy);
      rhs = castTo(builder, loc, rhs, intTy);
    }

    // Integer arithmetic (skip if we already returned for bool ops above)
    if (opIr == "arith.addf") return builder.create<mlir::arith::AddIOp>(loc, lhs, rhs);
    if (opIr == "arith.subf") return builder.create<mlir::arith::SubIOp>(loc, lhs, rhs);
    if (opIr == "arith.mulf") return builder.create<mlir::arith::MulIOp>(loc, lhs, rhs);
    if (opIr == "arith.divf") return builder.create<mlir::arith::DivSIOp>(loc, lhs, rhs);
    if (opIr == "arith.maximumf") {
      auto cmp = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sgt, lhs, rhs);
      return builder.create<mlir::arith::SelectOp>(loc, cmp, lhs, rhs);
    }
    if (opIr == "arith.minimumf") {
      auto cmp = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, lhs, rhs);
      return builder.create<mlir::arith::SelectOp>(loc, cmp, lhs, rhs);
    }
    if (opIr == "arith.remf") return builder.create<mlir::arith::RemSIOp>(loc, lhs, rhs);
  }

  // Float path: promote both operands to a common float type
  auto floatTy = commonFloatType(builder, lhs, rhs);
  lhs = castTo(builder, loc, lhs, floatTy);
  rhs = castTo(builder, loc, rhs, floatTy);

  if (opIr == "arith.addf") return builder.create<mlir::arith::AddFOp>(loc, lhs, rhs);
  if (opIr == "arith.subf") return builder.create<mlir::arith::SubFOp>(loc, lhs, rhs);
  if (opIr == "arith.mulf") return builder.create<mlir::arith::MulFOp>(loc, lhs, rhs);
  if (opIr == "arith.divf") return emitNativeCudaDiv(builder, loc, lhs, rhs);
  if (opIr == "arith.maximumf") return builder.create<mlir::arith::MaximumFOp>(loc, lhs, rhs);
  if (opIr == "arith.minimumf") return builder.create<mlir::arith::MinimumFOp>(loc, lhs, rhs);
  if (opIr == "arith.remf") return builder.create<mlir::arith::RemFOp>(loc, lhs, rhs);
  if (opIr == "math.atan2") return builder.create<mlir::math::Atan2Op>(loc, lhs, rhs);

  // Custom compound binary ops
  if (opIr == "custom.floordiv") {
    // floordiv(a, b) = floor(a / b)
    auto div = builder.create<mlir::arith::DivFOp>(loc, lhs, rhs);
    return builder.create<mlir::math::FloorOp>(loc, div);
  }
  if (opIr == "custom.reversediv") {
    // reversedivide(a, b) = b / a (swapped operands)
    return builder.create<mlir::arith::DivFOp>(loc, rhs, lhs);
  }
  if (opIr == "custom.reversesub") {
    // reversesubtract(a, b) = b - a (swapped operands)
    return builder.create<mlir::arith::SubFOp>(loc, rhs, lhs);
  }
  if (opIr == "custom.squaredsub") {
    // squaredsubtract(a, b) = (a - b)^2
    auto diff = builder.create<mlir::arith::SubFOp>(loc, lhs, rhs);
    return builder.create<mlir::arith::MulFOp>(loc, diff, diff);
  }
  if (opIr == "custom.swish_mul") {
    // swish_mul(x, y) = x * sigmoid(x) * y  (SwiGLU activation)
    auto negX = builder.create<mlir::arith::NegFOp>(loc, lhs);
    auto expNegX = emitNativeCudaExp(builder, loc, negX);
    auto tensorTy = mlir::cast<mlir::RankedTensorType>(lhs.getType());
    auto one = splatConstantF32(builder, loc, tensorTy, 1.0f);
    auto onePlusExp = builder.create<mlir::arith::AddFOp>(loc, one, expNegX);
    auto sigmoid = emitNativeCudaDiv(builder, loc, one, onePlusExp);
    auto xTimesSigmoid = builder.create<mlir::arith::MulFOp>(loc, lhs, sigmoid);
    return builder.create<mlir::arith::MulFOp>(loc, xTimesSigmoid, rhs);
  }
  if (opIr == "custom.mul_no_nan") {
    // multiply_no_nan(a, b) = b == 0 ? 0 : a * b
    auto tensorTy = mlir::cast<mlir::RankedTensorType>(lhs.getType());
    auto zero = splatConstantF32(builder, loc, tensorTy, 0.0f);
    auto product = builder.create<mlir::arith::MulFOp>(loc, lhs, rhs);
    auto isZero = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OEQ, rhs, zero);
    return builder.create<mlir::arith::SelectOp>(loc, isZero, zero, product);
  }
  if (opIr == "custom.pow") {
    // pow(base, exponent) = exp(exponent * log(|base|))
    // lhs = base, rhs = exponent (both tensors)
    // Use abs(base) before log to handle negative bases safely.
    // For even integer exponents (the common case, e.g. pow(x-mean, 2) in LayerNorm),
    // |x|^e == x^e.  For odd integer exponents on negative inputs this loses the sign,
    // but Triton's exp-log path already can't represent that (log of negative = NaN).
    // The abs-based formula is strictly better than the raw log path.
    auto absBase = builder.create<mlir::math::AbsFOp>(loc, lhs);
    auto logBase = builder.create<mlir::math::LogOp>(loc, absBase);
    auto eLogBase = builder.create<mlir::arith::MulFOp>(loc, rhs, logBase);
    return builder.create<mlir::math::ExpOp>(loc, eLogBase);
  }

  // ─── Activation backward ops ──────────────────────────────────────────────
  // All _bp ops: lhs = x (forward input), rhs = dy (upstream gradient)
  // Output: dx = dy * f'(x)

  if (opIr == "custom.relu_bp") {
    // relu_bp: dx = x > 0 ? dy : 0
    auto tensorTy = mlir::cast<mlir::RankedTensorType>(lhs.getType());
    auto zero = splatConstantF32(builder, loc, tensorTy, 0.0f);
    auto cmp = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OGT, lhs, zero);
    return builder.create<mlir::arith::SelectOp>(loc, cmp, rhs, zero);
  }

  if (opIr == "custom.relu6_bp") {
    // relu6_bp: dx = (x > 0 && x < 6) ? dy : 0
    auto tensorTy = mlir::cast<mlir::RankedTensorType>(lhs.getType());
    auto zero = splatConstantF32(builder, loc, tensorTy, 0.0f);
    auto six = splatConstantF32(builder, loc, tensorTy, 6.0f);
    auto gtZero = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OGT, lhs, zero);
    auto ltSix = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OLT, lhs, six);
    auto inRange = builder.create<mlir::arith::AndIOp>(loc, gtZero, ltSix);
    return builder.create<mlir::arith::SelectOp>(loc, inRange, rhs, zero);
  }

  if (opIr == "custom.thresholdedrelu_bp") {
    // thresholdedrelu_bp: dx = x > threshold ? dy : 0
    float threshold = 0.0f;
    if (slot.args.numTArgs > 0 && slot.args.tArgs) {
      threshold = static_cast<float>(slot.args.tArgs[0]);
    }
    auto tensorTy = mlir::cast<mlir::RankedTensorType>(lhs.getType());
    auto zero = splatConstantF32(builder, loc, tensorTy, 0.0f);
    auto threshSplat = splatConstantF32(builder, loc, tensorTy, threshold);
    auto cmp = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OGT, lhs, threshSplat);
    return builder.create<mlir::arith::SelectOp>(loc, cmp, rhs, zero);
  }

  if (opIr == "custom.sigmoid_bp") {
    // sigmoid_bp: s = sigmoid(x), dx = dy * s * (1 - s)
    auto tensorTy = mlir::cast<mlir::RankedTensorType>(lhs.getType());
    auto one = splatConstantF32(builder, loc, tensorTy, 1.0f);
    auto negX = builder.create<mlir::arith::NegFOp>(loc, lhs);
    auto expNegX = builder.create<mlir::math::ExpOp>(loc, negX);
    auto onePlusExp = builder.create<mlir::arith::AddFOp>(loc, one, expNegX);
    auto sig = builder.create<mlir::arith::DivFOp>(loc, one, onePlusExp);
    auto oneMinusSig = builder.create<mlir::arith::SubFOp>(loc, one, sig);
    auto sigDeriv = builder.create<mlir::arith::MulFOp>(loc, sig, oneMinusSig);
    return builder.create<mlir::arith::MulFOp>(loc, rhs, sigDeriv);
  }

  if (opIr == "custom.tanh_bp") {
    // tanh_bp: t = tanh(x), dx = dy * (1 - t^2)
    // Numerically stable: clamp input to [-10,10]
    auto tensorTy = mlir::cast<mlir::RankedTensorType>(lhs.getType());
    auto clampLo = splatConstantF32(builder, loc, tensorTy, -10.0f);
    auto clampHi = splatConstantF32(builder, loc, tensorTy, 10.0f);
    auto clampedLo = builder.create<mlir::arith::MaximumFOp>(loc, lhs, clampLo);
    auto clamped = builder.create<mlir::arith::MinimumFOp>(loc, clampedLo, clampHi);
    auto two = splatConstantF32(builder, loc, tensorTy, 2.0f);
    auto one = splatConstantF32(builder, loc, tensorTy, 1.0f);
    auto twoX = builder.create<mlir::arith::MulFOp>(loc, two, clamped);
    auto exp2x = builder.create<mlir::math::ExpOp>(loc, twoX);
    auto num = builder.create<mlir::arith::SubFOp>(loc, exp2x, one);
    auto den = builder.create<mlir::arith::AddFOp>(loc, exp2x, one);
    auto tanhX = builder.create<mlir::arith::DivFOp>(loc, num, den);
    auto tanhSq = builder.create<mlir::arith::MulFOp>(loc, tanhX, tanhX);
    auto deriv = builder.create<mlir::arith::SubFOp>(loc, one, tanhSq);
    return builder.create<mlir::arith::MulFOp>(loc, rhs, deriv);
  }

  if (opIr == "custom.elu_bp") {
    // elu_bp: dx = dy * (x >= 0 ? 1 : alpha * exp(x)), default alpha = 1.0
    float alpha = 1.0f;
    if (slot.args.numTArgs > 0 && slot.args.tArgs) {
      alpha = static_cast<float>(slot.args.tArgs[0]);
    }
    auto tensorTy = mlir::cast<mlir::RankedTensorType>(lhs.getType());
    auto zero = splatConstantF32(builder, loc, tensorTy, 0.0f);
    auto one = splatConstantF32(builder, loc, tensorTy, 1.0f);
    auto alphaSplat = splatConstantF32(builder, loc, tensorTy, alpha);
    auto expX = builder.create<mlir::math::ExpOp>(loc, lhs);
    auto negDeriv = builder.create<mlir::arith::MulFOp>(loc, alphaSplat, expX);
    auto cmp = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OGE, lhs, zero);
    auto deriv = builder.create<mlir::arith::SelectOp>(loc, cmp, one, negDeriv);
    return builder.create<mlir::arith::MulFOp>(loc, rhs, deriv);
  }

  if (opIr == "custom.selu_bp") {
    // selu_bp: dx = dy * (x > 0 ? lambda : alpha * lambda * exp(x))
    float lambda = 1.0507009873554804934193349852946f;
    float alphaLambda = 1.6732632423543772848170429916717f * lambda;
    auto tensorTy = mlir::cast<mlir::RankedTensorType>(lhs.getType());
    auto zero = splatConstantF32(builder, loc, tensorTy, 0.0f);
    auto lambdaSplat = splatConstantF32(builder, loc, tensorTy, lambda);
    auto alphaLambdaSplat = splatConstantF32(builder, loc, tensorTy, alphaLambda);
    auto expX = builder.create<mlir::math::ExpOp>(loc, lhs);
    auto negDeriv = builder.create<mlir::arith::MulFOp>(loc, alphaLambdaSplat, expX);
    auto cmp = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OGT, lhs, zero);
    auto deriv = builder.create<mlir::arith::SelectOp>(loc, cmp, lambdaSplat, negDeriv);
    return builder.create<mlir::arith::MulFOp>(loc, rhs, deriv);
  }

  if (opIr == "custom.lrelu_bp") {
    // lrelu_bp: dx = x < 0 ? alpha * dy : dy
    float alpha = 0.01f;
    if (slot.args.numTArgs > 0 && slot.args.tArgs) {
      alpha = static_cast<float>(slot.args.tArgs[0]);
    }
    auto tensorTy = mlir::cast<mlir::RankedTensorType>(lhs.getType());
    auto zero = splatConstantF32(builder, loc, tensorTy, 0.0f);
    auto alphaSplat = splatConstantF32(builder, loc, tensorTy, alpha);
    auto alphaDy = builder.create<mlir::arith::MulFOp>(loc, alphaSplat, rhs);
    auto cmp = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OLT, lhs, zero);
    return builder.create<mlir::arith::SelectOp>(loc, cmp, alphaDy, rhs);
  }

  if (opIr == "custom.softplus_bp") {
    // softplus_bp: dx = dy * sigmoid(x)
    auto tensorTy = mlir::cast<mlir::RankedTensorType>(lhs.getType());
    auto one = splatConstantF32(builder, loc, tensorTy, 1.0f);
    auto negX = builder.create<mlir::arith::NegFOp>(loc, lhs);
    auto expNegX = builder.create<mlir::math::ExpOp>(loc, negX);
    auto onePlusExp = builder.create<mlir::arith::AddFOp>(loc, one, expNegX);
    auto sig = builder.create<mlir::arith::DivFOp>(loc, one, onePlusExp);
    return builder.create<mlir::arith::MulFOp>(loc, rhs, sig);
  }

  if (opIr == "custom.softsign_bp") {
    // softsign_bp: dx = dy / (1 + |x|)^2
    auto tensorTy = mlir::cast<mlir::RankedTensorType>(lhs.getType());
    auto one = splatConstantF32(builder, loc, tensorTy, 1.0f);
    auto absX = builder.create<mlir::math::AbsFOp>(loc, lhs);
    auto denom = builder.create<mlir::arith::AddFOp>(loc, one, absX);
    auto denomSq = builder.create<mlir::arith::MulFOp>(loc, denom, denom);
    return builder.create<mlir::arith::DivFOp>(loc, rhs, denomSq);
  }

  if (opIr == "custom.hardsigmoid_bp") {
    // hardsigmoid_bp: dx = dy * (0.2 if -2.5 <= x <= 2.5 else 0)
    auto tensorTy = mlir::cast<mlir::RankedTensorType>(lhs.getType());
    auto zero = splatConstantF32(builder, loc, tensorTy, 0.0f);
    auto fifth = splatConstantF32(builder, loc, tensorTy, 0.2f);
    auto lo = splatConstantF32(builder, loc, tensorTy, -2.5f);
    auto hi = splatConstantF32(builder, loc, tensorTy, 2.5f);
    auto geLo = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OGE, lhs, lo);
    auto leHi = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OLE, lhs, hi);
    auto inRange = builder.create<mlir::arith::AndIOp>(loc, geLo, leHi);
    auto scaledDy = builder.create<mlir::arith::MulFOp>(loc, rhs, fifth);
    return builder.create<mlir::arith::SelectOp>(loc, inRange, scaledDy, zero);
  }

  if (opIr == "custom.hardtanh_bp") {
    // hardtanh_bp: dx = dy * (1 if -1 <= x <= 1 else 0)
    auto tensorTy = mlir::cast<mlir::RankedTensorType>(lhs.getType());
    auto zero = splatConstantF32(builder, loc, tensorTy, 0.0f);
    auto negOne = splatConstantF32(builder, loc, tensorTy, -1.0f);
    auto one = splatConstantF32(builder, loc, tensorTy, 1.0f);
    auto geLo = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OGE, lhs, negOne);
    auto leHi = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OLE, lhs, one);
    auto inRange = builder.create<mlir::arith::AndIOp>(loc, geLo, leHi);
    return builder.create<mlir::arith::SelectOp>(loc, inRange, rhs, zero);
  }

  if (opIr == "custom.silu_bp") {
    // silu_bp: s = sigmoid(x), dx = dy * (s + x * s * (1 - s))
    auto tensorTy = mlir::cast<mlir::RankedTensorType>(lhs.getType());
    auto one = splatConstantF32(builder, loc, tensorTy, 1.0f);
    auto negX = builder.create<mlir::arith::NegFOp>(loc, lhs);
    auto expNegX = builder.create<mlir::math::ExpOp>(loc, negX);
    auto onePlusExp = builder.create<mlir::arith::AddFOp>(loc, one, expNegX);
    auto sig = builder.create<mlir::arith::DivFOp>(loc, one, onePlusExp);
    auto oneMinusSig = builder.create<mlir::arith::SubFOp>(loc, one, sig);
    auto xTimesSig = builder.create<mlir::arith::MulFOp>(loc, lhs, sig);
    auto xSig1mS = builder.create<mlir::arith::MulFOp>(loc, xTimesSig, oneMinusSig);
    auto deriv = builder.create<mlir::arith::AddFOp>(loc, sig, xSig1mS);
    return builder.create<mlir::arith::MulFOp>(loc, rhs, deriv);
  }

  if (opIr == "custom.fused_gelu_bp") {
    // fused_gelu_bp: s = sigmoid(1.702*x), dx = dy * (s + x * 1.702 * s * (1-s))
    auto tensorTy = mlir::cast<mlir::RankedTensorType>(lhs.getType());
    auto one = splatConstantF32(builder, loc, tensorTy, 1.0f);
    auto coeff = splatConstantF32(builder, loc, tensorTy, 1.702f);
    auto scaled = builder.create<mlir::arith::MulFOp>(loc, coeff, lhs);
    auto negScaled = builder.create<mlir::arith::NegFOp>(loc, scaled);
    auto expNeg = builder.create<mlir::math::ExpOp>(loc, negScaled);
    auto onePlusExp = builder.create<mlir::arith::AddFOp>(loc, one, expNeg);
    auto sig = builder.create<mlir::arith::DivFOp>(loc, one, onePlusExp);
    auto oneMinusSig = builder.create<mlir::arith::SubFOp>(loc, one, sig);
    auto xCoeff = builder.create<mlir::arith::MulFOp>(loc, lhs, coeff);
    auto xCoeffSig = builder.create<mlir::arith::MulFOp>(loc, xCoeff, sig);
    auto xCoeffSig1mS = builder.create<mlir::arith::MulFOp>(loc, xCoeffSig, oneMinusSig);
    auto deriv = builder.create<mlir::arith::AddFOp>(loc, sig, xCoeffSig1mS);
    return builder.create<mlir::arith::MulFOp>(loc, rhs, deriv);
  }

  if (opIr == "custom.squared_relu_bp") {
    // squared_relu_bp: dx = dy * 2 * x * (x > 0)
    auto tensorTy = mlir::cast<mlir::RankedTensorType>(lhs.getType());
    auto zero = splatConstantF32(builder, loc, tensorTy, 0.0f);
    auto two = splatConstantF32(builder, loc, tensorTy, 2.0f);
    auto cmp = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OGT, lhs, zero);
    auto twoX = builder.create<mlir::arith::MulFOp>(loc, two, lhs);
    auto dyTwoX = builder.create<mlir::arith::MulFOp>(loc, rhs, twoX);
    return builder.create<mlir::arith::SelectOp>(loc, cmp, dyTwoX, zero);
  }

  if (opIr == "custom.rectifiedtanh_bp") {
    // rectifiedtanh_bp: dx = x > 0 ? dy * (1 - tanh(x)^2) : 0
    auto tensorTy = mlir::cast<mlir::RankedTensorType>(lhs.getType());
    auto zero = splatConstantF32(builder, loc, tensorTy, 0.0f);
    auto one = splatConstantF32(builder, loc, tensorTy, 1.0f);
    auto clampLo = splatConstantF32(builder, loc, tensorTy, -10.0f);
    auto clampHi = splatConstantF32(builder, loc, tensorTy, 10.0f);
    auto clampedLo = builder.create<mlir::arith::MaximumFOp>(loc, lhs, clampLo);
    auto clamped = builder.create<mlir::arith::MinimumFOp>(loc, clampedLo, clampHi);
    auto two = splatConstantF32(builder, loc, tensorTy, 2.0f);
    auto twoX = builder.create<mlir::arith::MulFOp>(loc, two, clamped);
    auto exp2x = builder.create<mlir::math::ExpOp>(loc, twoX);
    auto num = builder.create<mlir::arith::SubFOp>(loc, exp2x, one);
    auto den = builder.create<mlir::arith::AddFOp>(loc, exp2x, one);
    auto tanhX = builder.create<mlir::arith::DivFOp>(loc, num, den);
    auto tanhSq = builder.create<mlir::arith::MulFOp>(loc, tanhX, tanhX);
    auto deriv = builder.create<mlir::arith::SubFOp>(loc, one, tanhSq);
    auto dyDeriv = builder.create<mlir::arith::MulFOp>(loc, rhs, deriv);
    auto gtZero = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OGT, lhs, zero);
    return builder.create<mlir::arith::SelectOp>(loc, gtZero, dyDeriv, zero);
  }

  if (opIr == "custom.swish_mul_bp") {
    // swish_mul_bp: forward = silu(x) * y, gradient w.r.t. x only
    // s = sigmoid(x), dx = dy * y * (s + x * s * (1 - s))
    // Note: lhs = x, rhs = dy. This produces gradient w.r.t. x only.
    auto tensorTy = mlir::cast<mlir::RankedTensorType>(lhs.getType());
    auto one = splatConstantF32(builder, loc, tensorTy, 1.0f);
    auto negX = builder.create<mlir::arith::NegFOp>(loc, lhs);
    auto expNegX = builder.create<mlir::math::ExpOp>(loc, negX);
    auto onePlusExp = builder.create<mlir::arith::AddFOp>(loc, one, expNegX);
    auto sig = builder.create<mlir::arith::DivFOp>(loc, one, onePlusExp);
    auto oneMinusSig = builder.create<mlir::arith::SubFOp>(loc, one, sig);
    auto xTimesSig = builder.create<mlir::arith::MulFOp>(loc, lhs, sig);
    auto xSig1mS = builder.create<mlir::arith::MulFOp>(loc, xTimesSig, oneMinusSig);
    auto deriv = builder.create<mlir::arith::AddFOp>(loc, sig, xSig1mS);
    return builder.create<mlir::arith::MulFOp>(loc, rhs, deriv);
  }

  std::string msg = "TritonIRBuilder::emitBinaryElementwise: unknown op '" + opIr +
                    "' — categorized as BINARY_ELEMENTWISE but has no emitter. "
                    "Either add an emitter or recategorize this op.";
  THROW_EXCEPTION(msg.c_str());
  return lhs;  // unreachable
}

mlir::Value TritonIRBuilder::emitUnaryElementwise(mlir::OpBuilder& builder, mlir::Location loc,
                                                   const TritonOpMapping& mapping,
                                                   const NativeSlot& slot, mlir::Value input,
                                                   int blockSize) {
  auto tensorType = mlir::cast<mlir::RankedTensorType>(input.getType());
  auto opName = mapping.opName;

  // Math ops require float inputs — promote integer/bool/f16/bf16 to at least f32
  input = promoteToFloat(builder, loc, input);
  tensorType = mlir::cast<mlir::RankedTensorType>(input.getType());

  // Convert to lowercase for matching
  std::string opLower = opName;
  std::transform(opLower.begin(), opLower.end(), opLower.begin(), ::tolower);

  if (opLower == "relu") {
    // relu(x) = max(x, 0.0)
    auto zero = splatConstantF32(builder, loc, tensorType, 0.0f);
    return builder.create<mlir::arith::MaximumFOp>(loc, input, zero);
  }

  if (opLower == "sigmoid") {
    // Match native sd_sigmoid: 1 / (1 + expf(-x)), including libdevice exp
    // and round-to-nearest FP32 division.
    auto negX = builder.create<mlir::arith::NegFOp>(loc, input);
    auto expNegX = emitNativeCudaExp(builder, loc, negX);
    auto one = splatConstantF32(builder, loc, tensorType, 1.0f);
    auto onePlusExp = builder.create<mlir::arith::AddFOp>(loc, one, expNegX);
    return emitNativeCudaDiv(builder, loc, one, onePlusExp);
  }

  if (opLower == "tanh") {
    // Numerically stable tanh: clamp input to [-10,10] before exp(2x).
    // For |x| > 10, tanh(x) = ±1.0 to float32 precision (error < 1e-9).
    // exp(20) ≈ 4.8e8 — safely within float32 range (~3.4e38).
    // Without clamp: exp(2*45) overflows to Inf → (Inf-1)/(Inf+1) = NaN.
    auto clampLo = splatConstantF32(builder, loc, tensorType, -10.0f);
    auto clampHi = splatConstantF32(builder, loc, tensorType, 10.0f);
    auto clampedLo = builder.create<mlir::arith::MaximumFOp>(loc, input, clampLo);
    auto clamped = builder.create<mlir::arith::MinimumFOp>(loc, clampedLo, clampHi);
    auto two = splatConstantF32(builder, loc, tensorType, 2.0f);
    auto one = splatConstantF32(builder, loc, tensorType, 1.0f);
    auto twoX = builder.create<mlir::arith::MulFOp>(loc, two, clamped);
    auto exp2x = builder.create<mlir::math::ExpOp>(loc, twoX);
    auto num = builder.create<mlir::arith::SubFOp>(loc, exp2x, one);
    auto den = builder.create<mlir::arith::AddFOp>(loc, exp2x, one);
    return builder.create<mlir::arith::DivFOp>(loc, num, den);
  }

  if (opLower == "gelu") {
    // Match native GELU: x * sigmoid(1.702 * x) = x / (1 + exp(-1.702 * x))
    auto coeff = splatConstantF32(builder, loc, tensorType, 1.702f);
    auto scaled = builder.create<mlir::arith::MulFOp>(loc, coeff, input);
    auto neg = builder.create<mlir::arith::NegFOp>(loc, scaled);
    auto expNeg = builder.create<mlir::math::ExpOp>(loc, neg);
    auto one = splatConstantF32(builder, loc, tensorType, 1.0f);
    auto denom = builder.create<mlir::arith::AddFOp>(loc, one, expNeg);
    auto sigmoid = builder.create<mlir::arith::DivFOp>(loc, one, denom);
    return builder.create<mlir::arith::MulFOp>(loc, input, sigmoid);
  }

  if (opLower == "exp") {
    return emitNativeCudaExp(builder, loc, input);
  }

  if (opLower == "log") {
    return emitNativeCudaLog(builder, loc, input);
  }

  if (opLower == "abs") {
    return builder.create<mlir::math::AbsFOp>(loc, input);
  }

  if (opLower == "sqrt") {
    // Native CUDA's sqrtf is round-to-nearest. Triton's generic FP32 math.sqrt
    // lowers to the approximate instruction and can differ by one ULP, which
    // is enough to perturb recurrent decode state across execution paths.
    // precise_sqrt is FP32-only; retain generic sqrt for FP64.
    if (getElementType(input).isF32()) {
      return builder.create<mlir::triton::PreciseSqrtOp>(loc, input);
    }
    return builder.create<mlir::math::SqrtOp>(loc, input);
  }

  if (opLower == "square") {
    // square(x) = x * x
    return builder.create<mlir::arith::MulFOp>(loc, input, input);
  }

  if (opLower == "pow") {
    // pow(x, exponent) — avoid math.PowFOp because Triton's NVIDIA backend
    // fails to legalize it during TTGIR→LLVM lowering.
    // Instead, use special cases for common exponents and exp(e*log(x)) for general case.
    float exponent = 2.0f;
    if (slot.args.numTArgs > 0 && slot.args.tArgs) {
      exponent = static_cast<float>(slot.args.tArgs[0]);
    }
    // Special cases that avoid log/exp entirely
    if (exponent == 0.0f) {
      return splatConstantF32(builder, loc, tensorType, 1.0f);
    }
    if (exponent == 1.0f) {
      return input;
    }
    if (exponent == 2.0f) {
      return builder.create<mlir::arith::MulFOp>(loc, input, input);
    }
    if (exponent == 0.5f) {
      return builder.create<mlir::math::SqrtOp>(loc, input);
    }
    if (exponent == -0.5f) {
      auto sq = builder.create<mlir::math::SqrtOp>(loc, input);
      auto one = splatConstantF32(builder, loc, tensorType, 1.0f);
      return builder.create<mlir::arith::DivFOp>(loc, one, sq);
    }
    if (exponent == -1.0f) {
      auto one = splatConstantF32(builder, loc, tensorType, 1.0f);
      return builder.create<mlir::arith::DivFOp>(loc, one, input);
    }
    if (exponent == 3.0f) {
      auto x2 = builder.create<mlir::arith::MulFOp>(loc, input, input);
      return builder.create<mlir::arith::MulFOp>(loc, x2, input);
    }
    // General case: pow(x, e) = exp(e * log(x))
    auto logX = builder.create<mlir::math::LogOp>(loc, input);
    auto expVal = splatConstantF32(builder, loc, tensorType, exponent);
    auto eLogX = builder.create<mlir::arith::MulFOp>(loc, expVal, logX);
    return builder.create<mlir::math::ExpOp>(loc, eLogX);
  }

  if (opLower == "clamp" || opLower == "clipbyvalue") {
    // clamp(x, min, max) = min(max(x, minVal), maxVal)
    float minVal = -3.4028235e+38f;
    float maxVal = 3.4028235e+38f;
    if (slot.args.numTArgs >= 2 && slot.args.tArgs) {
      minVal = static_cast<float>(slot.args.tArgs[0]);
      maxVal = static_cast<float>(slot.args.tArgs[1]);
    }
    auto minSplat = splatConstantF32(builder, loc, tensorType, minVal);
    auto maxSplat = splatConstantF32(builder, loc, tensorType, maxVal);
    auto clamped = builder.create<mlir::arith::MaximumFOp>(loc, input, minSplat);
    return builder.create<mlir::arith::MinimumFOp>(loc, clamped, maxSplat);
  }

  if (opLower == "neg") {
    return builder.create<mlir::arith::NegFOp>(loc, input);
  }

  if (opLower == "reciprocal") {
    // reciprocal(x) = 1.0 / x
    auto one = splatConstantF32(builder, loc, tensorType, 1.0f);
    return builder.create<mlir::arith::DivFOp>(loc, one, input);
  }

  if (opLower == "rsqrt") {
    // rsqrt(x) = 1.0 / sqrt(x)
    auto sq = builder.create<mlir::math::SqrtOp>(loc, input);
    auto one = splatConstantF32(builder, loc, tensorType, 1.0f);
    return builder.create<mlir::arith::DivFOp>(loc, one, sq);
  }

  if (opLower == "sign") {
    // sign(x) = x > 0 ? 1 : (x < 0 ? -1 : 0)
    auto zero = splatConstantF32(builder, loc, tensorType, 0.0f);
    auto one = splatConstantF32(builder, loc, tensorType, 1.0f);
    auto negOne = splatConstantF32(builder, loc, tensorType, -1.0f);
    auto gtZero = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OGT, input, zero);
    auto ltZero = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OLT, input, zero);
    auto negPart = builder.create<mlir::arith::SelectOp>(loc, ltZero, negOne, zero);
    return builder.create<mlir::arith::SelectOp>(loc, gtZero, one, negPart);
  }

  if (opLower == "erf") {
    return builder.create<mlir::math::ErfOp>(loc, input);
  }

  if (opLower == "erfc") {
    // erfc(x) = 1.0 - erf(x)
    auto erfVal = builder.create<mlir::math::ErfOp>(loc, input);
    auto one = splatConstantF32(builder, loc, tensorType, 1.0f);
    return builder.create<mlir::arith::SubFOp>(loc, one, erfVal);
  }

  if (opLower == "clip_by_value") {
    // clip_by_value(x, min, max) — alias of clipbyvalue
    float minVal = -3.4028235e+38f;
    float maxVal = 3.4028235e+38f;
    if (slot.args.numTArgs >= 2 && slot.args.tArgs) {
      minVal = static_cast<float>(slot.args.tArgs[0]);
      maxVal = static_cast<float>(slot.args.tArgs[1]);
    }
    auto minSplat = splatConstantF32(builder, loc, tensorType, minVal);
    auto maxSplat = splatConstantF32(builder, loc, tensorType, maxVal);
    auto clamped = builder.create<mlir::arith::MaximumFOp>(loc, input, minSplat);
    return builder.create<mlir::arith::MinimumFOp>(loc, clamped, maxSplat);
  }

  if (opLower == "log1p") {
    return builder.create<mlir::math::Log1pOp>(loc, input);
  }

  if (opLower == "ceil") {
    return builder.create<mlir::math::CeilOp>(loc, input);
  }

  if (opLower == "floor") {
    return builder.create<mlir::math::FloorOp>(loc, input);
  }

  if (opLower == "round") {
    return builder.create<mlir::math::RoundEvenOp>(loc, input);
  }

  if (opLower == "sin") {
    return builder.create<mlir::math::SinOp>(loc, input);
  }

  if (opLower == "cos") {
    return builder.create<mlir::math::CosOp>(loc, input);
  }

  if (opLower == "leakyrelu") {
    // leakyrelu(x) = x > 0 ? x : alpha * x, default alpha = 0.01
    float alpha = 0.01f;
    if (slot.args.numTArgs > 0 && slot.args.tArgs) {
      alpha = static_cast<float>(slot.args.tArgs[0]);
    }
    auto zero = splatConstantF32(builder, loc, tensorType, 0.0f);
    auto alphaSplat = splatConstantF32(builder, loc, tensorType, alpha);
    auto alphaX = builder.create<mlir::arith::MulFOp>(loc, alphaSplat, input);
    auto cmp = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OGT, input, zero);
    return builder.create<mlir::arith::SelectOp>(loc, cmp, input, alphaX);
  }

  if (opLower == "silu" || opLower == "swish") {
    // Match native Swish exactly: x * sd_sigmoid(x). Native evaluates the
    // reciprocal and final multiply as separate rounded operations, so folding
    // them into x / denominator changes recurrent-model outputs by one ULP.
    auto negX = builder.create<mlir::arith::NegFOp>(loc, input);
    auto expNegX = emitNativeCudaExp(builder, loc, negX);
    auto one = splatConstantF32(builder, loc, tensorType, 1.0f);
    auto onePlusExp = builder.create<mlir::arith::AddFOp>(loc, one, expNegX);
    auto sigmoid = emitNativeCudaDiv(builder, loc, one, onePlusExp);
    return builder.create<mlir::arith::MulFOp>(loc, input, sigmoid);
  }

  if (opLower == "mish") {
    // mish(x) = x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))
    // Clamp softplus to [0,10] before tanh to avoid exp overflow (same as tanh fix)
    auto expX = builder.create<mlir::math::ExpOp>(loc, input);
    auto one = splatConstantF32(builder, loc, tensorType, 1.0f);
    auto onePlusExp = builder.create<mlir::arith::AddFOp>(loc, one, expX);
    auto sp = builder.create<mlir::math::LogOp>(loc, onePlusExp);
    // Clamp softplus to [0,10] — softplus >= 0 always, tanh(10) = 1.0 to float32
    auto clampHi = splatConstantF32(builder, loc, tensorType, 10.0f);
    auto spClamped = builder.create<mlir::arith::MinimumFOp>(loc, sp, clampHi);
    auto two = splatConstantF32(builder, loc, tensorType, 2.0f);
    auto twoSp = builder.create<mlir::arith::MulFOp>(loc, two, spClamped);
    auto exp2sp = builder.create<mlir::math::ExpOp>(loc, twoSp);
    auto numMish = builder.create<mlir::arith::SubFOp>(loc, exp2sp, one);
    auto denMish = builder.create<mlir::arith::AddFOp>(loc, exp2sp, one);
    auto tanhSp = builder.create<mlir::arith::DivFOp>(loc, numMish, denMish);
    return builder.create<mlir::arith::MulFOp>(loc, input, tanhSp);
  }

  if (opLower == "elu") {
    // elu(x) = x > 0 ? x : alpha * (exp(x) - 1), default alpha = 1.0
    float alpha = 1.0f;
    if (slot.args.numTArgs > 0 && slot.args.tArgs) {
      alpha = static_cast<float>(slot.args.tArgs[0]);
    }
    auto zero = splatConstantF32(builder, loc, tensorType, 0.0f);
    auto one = splatConstantF32(builder, loc, tensorType, 1.0f);
    auto alphaSplat = splatConstantF32(builder, loc, tensorType, alpha);
    auto expX = builder.create<mlir::math::ExpOp>(loc, input);
    auto expXMinusOne = builder.create<mlir::arith::SubFOp>(loc, expX, one);
    auto negPart = builder.create<mlir::arith::MulFOp>(loc, alphaSplat, expXMinusOne);
    auto cmp = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OGT, input, zero);
    return builder.create<mlir::arith::SelectOp>(loc, cmp, input, negPart);
  }

  if (opLower == "selu") {
    // selu(x) = lambda * (x > 0 ? x : alpha * (exp(x) - 1))
    // lambda = 1.0507, alpha = 1.67326
    float lambda = 1.0507009873554804934193349852946f;
    float alpha = 1.6732632423543772848170429916717f;
    auto zero = splatConstantF32(builder, loc, tensorType, 0.0f);
    auto one = splatConstantF32(builder, loc, tensorType, 1.0f);
    auto alphaSplat = splatConstantF32(builder, loc, tensorType, alpha);
    auto lambdaSplat = splatConstantF32(builder, loc, tensorType, lambda);
    auto expX = builder.create<mlir::math::ExpOp>(loc, input);
    auto expXMinusOne = builder.create<mlir::arith::SubFOp>(loc, expX, one);
    auto negPart = builder.create<mlir::arith::MulFOp>(loc, alphaSplat, expXMinusOne);
    auto cmp = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OGT, input, zero);
    auto selected = builder.create<mlir::arith::SelectOp>(loc, cmp, input, negPart);
    return builder.create<mlir::arith::MulFOp>(loc, lambdaSplat, selected);
  }

  if (opLower == "softplus") {
    // softplus(x) = max(0, x) + log(1 + exp(-|x|))  [overflow-safe]
    auto zero = splatConstantF32(builder, loc, tensorType, 0.0f);
    auto one = splatConstantF32(builder, loc, tensorType, 1.0f);
    auto absX = builder.create<mlir::math::AbsFOp>(loc, input);
    auto negAbsX = builder.create<mlir::arith::NegFOp>(loc, absX);
    auto maxPart = builder.create<mlir::arith::MaximumFOp>(loc, input, zero);
    auto expNegAbs = emitNativeCudaExp(builder, loc, negAbsX);
    auto onePlusExp = builder.create<mlir::arith::AddFOp>(loc, one, expNegAbs);
    auto logPart = emitNativeCudaLog(builder, loc, onePlusExp);
    return builder.create<mlir::arith::AddFOp>(loc, maxPart, logPart);
  }

  if (opLower == "softsign") {
    // softsign(x) = x / (1 + |x|)
    auto absX = builder.create<mlir::math::AbsFOp>(loc, input);
    auto one = splatConstantF32(builder, loc, tensorType, 1.0f);
    auto denom = builder.create<mlir::arith::AddFOp>(loc, one, absX);
    return builder.create<mlir::arith::DivFOp>(loc, input, denom);
  }

  if (opLower == "hard_sigmoid") {
    // hard_sigmoid(x) = clip(x/6 + 0.5, 0, 1)
    auto sixth = splatConstantF32(builder, loc, tensorType, 1.0f / 6.0f);
    auto half = splatConstantF32(builder, loc, tensorType, 0.5f);
    auto zero = splatConstantF32(builder, loc, tensorType, 0.0f);
    auto one = splatConstantF32(builder, loc, tensorType, 1.0f);
    auto scaled = builder.create<mlir::arith::MulFOp>(loc, input, sixth);
    auto shifted = builder.create<mlir::arith::AddFOp>(loc, scaled, half);
    auto clamped = builder.create<mlir::arith::MaximumFOp>(loc, shifted, zero);
    return builder.create<mlir::arith::MinimumFOp>(loc, clamped, one);
  }

  if (opLower == "hardtanh") {
    // hardtanh(x) = clip(x, -1, 1)
    auto negOne = splatConstantF32(builder, loc, tensorType, -1.0f);
    auto one = splatConstantF32(builder, loc, tensorType, 1.0f);
    auto clamped = builder.create<mlir::arith::MaximumFOp>(loc, input, negOne);
    return builder.create<mlir::arith::MinimumFOp>(loc, clamped, one);
  }

  if (opLower == "relu6") {
    // relu6(x) = clip(x, 0, 6)
    auto zero = splatConstantF32(builder, loc, tensorType, 0.0f);
    auto six = splatConstantF32(builder, loc, tensorType, 6.0f);
    auto clamped = builder.create<mlir::arith::MaximumFOp>(loc, input, zero);
    return builder.create<mlir::arith::MinimumFOp>(loc, clamped, six);
  }

  // Scalar binary ops: second operand comes from tArgs[0]
  if (opLower == "add_scalar") {
    float scalar = (slot.args.numTArgs > 0 && slot.args.tArgs) ? static_cast<float>(slot.args.tArgs[0]) : 0.0f;
    auto scalarSplat = splatConstantF32(builder, loc, tensorType, scalar);
    return builder.create<mlir::arith::AddFOp>(loc, input, scalarSplat);
  }

  if (opLower == "subtract_scalar" || opLower == "sub_scalar") {
    float scalar = (slot.args.numTArgs > 0 && slot.args.tArgs) ? static_cast<float>(slot.args.tArgs[0]) : 0.0f;
    auto scalarSplat = splatConstantF32(builder, loc, tensorType, scalar);
    return builder.create<mlir::arith::SubFOp>(loc, input, scalarSplat);
  }

  if (opLower == "multiply_scalar" || opLower == "mul_scalar") {
    float scalar = (slot.args.numTArgs > 0 && slot.args.tArgs) ? static_cast<float>(slot.args.tArgs[0]) : 1.0f;
    auto scalarSplat = splatConstantF32(builder, loc, tensorType, scalar);
    return builder.create<mlir::arith::MulFOp>(loc, input, scalarSplat);
  }

  if (opLower == "divide_scalar" || opLower == "div_scalar") {
    float scalar = (slot.args.numTArgs > 0 && slot.args.tArgs) ? static_cast<float>(slot.args.tArgs[0]) : 1.0f;
    auto scalarSplat = splatConstantF32(builder, loc, tensorType, scalar);
    return builder.create<mlir::arith::DivFOp>(loc, input, scalarSplat);
  }

  if (opLower == "rsub_scalar") {
    float scalar = (slot.args.numTArgs > 0 && slot.args.tArgs) ? static_cast<float>(slot.args.tArgs[0]) : 0.0f;
    auto scalarSplat = splatConstantF32(builder, loc, tensorType, scalar);
    return builder.create<mlir::arith::SubFOp>(loc, scalarSplat, input);
  }

  if (opLower == "rdiv_scalar") {
    float scalar = (slot.args.numTArgs > 0 && slot.args.tArgs) ? static_cast<float>(slot.args.tArgs[0]) : 1.0f;
    auto scalarSplat = splatConstantF32(builder, loc, tensorType, scalar);
    return builder.create<mlir::arith::DivFOp>(loc, scalarSplat, input);
  }

  std::string msg = "TritonIRBuilder::emitUnaryElementwise: unknown op '" + opName +
                    "' — categorized as UNARY_ELEMENTWISE but has no emitter. "
                    "Either add an emitter or recategorize this op.";
  THROW_EXCEPTION(msg.c_str());
  return input;  // unreachable
}

// ─── Comparison op emission ─────────────────────────────────────────────────

mlir::Value TritonIRBuilder::emitComparisonOp(mlir::OpBuilder& builder, mlir::Location loc,
                                               const std::string& opName,
                                               mlir::Value lhs, mlir::Value rhs, int blockSize) {
  std::string opKey = normalizeOpToken(opName);

  bool lhsIsFloat = isFloatType(lhs.getType());
  bool rhsIsFloat = isFloatType(rhs.getType());

  // If either operand is float, promote both to a common float type
  if (lhsIsFloat || rhsIsFloat) {
    auto floatTy = commonFloatType(builder, lhs, rhs);
    lhs = castTo(builder, loc, lhs, floatTy);
    rhs = castTo(builder, loc, rhs, floatTy);

    mlir::arith::CmpFPredicate pred;
    if (opKey == "greater" || opKey == "greaterthanscalar") pred = mlir::arith::CmpFPredicate::OGT;
    else if (opKey == "greaterequal" || opKey == "greaterthanorequalscalar") pred = mlir::arith::CmpFPredicate::OGE;
    else if (opKey == "less" || opKey == "lessthanscalar") pred = mlir::arith::CmpFPredicate::OLT;
    else if (opKey == "lessequal" || opKey == "lessthanorequalscalar") pred = mlir::arith::CmpFPredicate::OLE;
    else if (opKey == "equals" || opKey == "equal" || opKey == "eq" || opKey == "equalsscalar") pred = mlir::arith::CmpFPredicate::OEQ;
    else if (opKey == "notequals" || opKey == "notequal" || opKey == "neq" || opKey == "notequalsscalar") pred = mlir::arith::CmpFPredicate::ONE;
    else {
      std::string msg = "TritonIRBuilder::emitComparisonOp: unknown float comparison '" + opName +
                        "' — categorized as COMPARISON but has no emitter.";
      THROW_EXCEPTION(msg.c_str());
      pred = mlir::arith::CmpFPredicate::OEQ;  // unreachable
    }
    return builder.create<mlir::arith::CmpFOp>(loc, pred, lhs, rhs);
  } else {
    // Both integer — coerce to same width
    auto intTy = commonIntType(lhs, rhs);
    lhs = castTo(builder, loc, lhs, intTy);
    rhs = castTo(builder, loc, rhs, intTy);

    mlir::arith::CmpIPredicate pred;
    if (opKey == "greater" || opKey == "greaterthanscalar") pred = mlir::arith::CmpIPredicate::sgt;
    else if (opKey == "greaterequal" || opKey == "greaterthanorequalscalar") pred = mlir::arith::CmpIPredicate::sge;
    else if (opKey == "less" || opKey == "lessthanscalar") pred = mlir::arith::CmpIPredicate::slt;
    else if (opKey == "lessequal" || opKey == "lessthanorequalscalar") pred = mlir::arith::CmpIPredicate::sle;
    else if (opKey == "equals" || opKey == "equal" || opKey == "eq" || opKey == "equalsscalar") pred = mlir::arith::CmpIPredicate::eq;
    else if (opKey == "notequals" || opKey == "notequal" || opKey == "neq" || opKey == "notequalsscalar") pred = mlir::arith::CmpIPredicate::ne;
    else {
      std::string msg = "TritonIRBuilder::emitComparisonOp: unknown int comparison '" + opName +
                        "' — categorized as COMPARISON but has no emitter.";
      THROW_EXCEPTION(msg.c_str());
      pred = mlir::arith::CmpIPredicate::eq;  // unreachable
    }
    return builder.create<mlir::arith::CmpIOp>(loc, pred, lhs, rhs);
  }
}

// ─── Logical op emission ────────────────────────────────────────────────────

mlir::Value TritonIRBuilder::emitLogicalOp(mlir::OpBuilder& builder, mlir::Location loc,
                                            const std::string& opName,
                                            mlir::Value lhs, mlir::Value rhs, int blockSize) {
  std::string opKey = normalizeOpToken(opName);

  // Coerce both inputs to i1 (bool)
  lhs = castTo(builder, loc, lhs, builder.getI1Type());

  // Unary logical_not / boolean_not — single input XOR with all-ones
  if (opKey == "booleannot" || opKey == "logicalnot" || opKey == "boolnot") {
    auto tensorTy = mlir::cast<mlir::RankedTensorType>(lhs.getType());
    auto trueAttr = builder.getIntegerAttr(builder.getI1Type(), 1);
    auto trueScalar = builder.create<mlir::arith::ConstantOp>(loc, builder.getI1Type(), trueAttr);
    auto allOnes = builder.create<mlir::triton::SplatOp>(loc, tensorTy, trueScalar);
    return builder.create<mlir::arith::XOrIOp>(loc, lhs, allOnes);
  }

  // Binary logical ops
  rhs = castTo(builder, loc, rhs, builder.getI1Type());

  if (opKey == "booleanand" || opKey == "logicaland") {
    return builder.create<mlir::arith::AndIOp>(loc, lhs, rhs);
  }
  if (opKey == "booleanor" || opKey == "logicalor") {
    return builder.create<mlir::arith::OrIOp>(loc, lhs, rhs);
  }
  if (opKey == "booleanxor" || opKey == "logicalxor") {
    return builder.create<mlir::arith::XOrIOp>(loc, lhs, rhs);
  }

  std::string msg = "TritonIRBuilder::emitLogicalOp: unknown logical op '" + opName +
                    "' — categorized as LOGICAL but has no emitter. "
                    "Either add an emitter or recategorize this op.";
  THROW_EXCEPTION(msg.c_str());
  return lhs;  // unreachable
}

// ─── Ternary select/where emission ──────────────────────────────────────────

mlir::Value TritonIRBuilder::emitTernaryOp(mlir::OpBuilder& builder, mlir::Location loc,
                                            mlir::Value condition, mlir::Value trueVal,
                                            mlir::Value falseVal, int blockSize) {
  // Condition must be i1 (bool)
  condition = castTo(builder, loc, condition, builder.getI1Type());

  // trueVal and falseVal must have same type — promote to common type
  auto trueElem = getElementType(trueVal);
  auto falseElem = getElementType(falseVal);

  if (trueElem != falseElem) {
    bool trueIsFloat = mlir::isa<mlir::FloatType>(trueElem);
    bool falseIsFloat = mlir::isa<mlir::FloatType>(falseElem);
    if (trueIsFloat || falseIsFloat) {
      auto floatTy = commonFloatType(builder, trueVal, falseVal);
      trueVal = castTo(builder, loc, trueVal, floatTy);
      falseVal = castTo(builder, loc, falseVal, floatTy);
    } else {
      auto intTy = commonIntType(trueVal, falseVal);
      trueVal = castTo(builder, loc, trueVal, intTy);
      falseVal = castTo(builder, loc, falseVal, intTy);
    }
  }

  return builder.create<mlir::arith::SelectOp>(loc, condition, trueVal, falseVal);
}

// ─── Reduction op emission ───────────────────────────────────────────────────

mlir::Value TritonIRBuilder::emitReductionOp(mlir::OpBuilder& builder, mlir::Location loc,
                                              const std::string& opName,
                                              mlir::Value input, int reductionAxis,
                                              mlir::RankedTensorType outputType) {
  std::string opKey = normalizeOpToken(opName);

  // Promote input to float for math ops
  input = promoteToFloat(builder, loc, input);
  auto tensorTy = mlir::cast<mlir::RankedTensorType>(input.getType());
  auto elemType = tensorTy.getElementType();

  // For reduce_norm2: square input first, then reduce_sum, then sqrt
  if (opKey == "reducenorm2" || opKey == "norm2") {
    input = builder.create<mlir::arith::MulFOp>(loc, input, input);
  }
  // For reduce_norm1: abs input first, then reduce_sum
  if (opKey == "reducenorm1" || opKey == "norm1") {
    input = builder.create<mlir::math::AbsFOp>(loc, input);
  }

  // Clamp reduction axis to valid range for the tensor's actual rank.
  // In the 1D kernel skeleton, tensors are rank-1 so axis must be 0.
  int64_t rank = tensorTy.getRank();
  if (reductionAxis < 0) reductionAxis += static_cast<int>(rank);
  if (reductionAxis < 0 || reductionAxis >= static_cast<int>(rank)) reductionAxis = 0;

  if (opKey == "argmax" || opKey == "argmin" ||
      opKey == "reducevariance" || opKey == "reducestdev" ||
      opKey == "reducelogsumexp") {
    // Multi-pass reductions (index tracking, mean-centered deviations,
    // max-shifted exp sums) are implemented in the ordered-reduction module
    // path (emitOrderedReductionValue in TritonIRBuilder_module.cpp). This
    // single-tt.reduce path cannot express them — refuse loudly instead of
    // emitting a plain sum.
    std::string msg = "TritonIRBuilder::emitReductionOp: reduction '" + opName +
                      "' requires a multi-pass algorithm and is not implemented in this path";
    THROW_EXCEPTION(msg.c_str());
    return input;
  }

  // Create tt.reduce op with combiner region
  // tt.reduce takes a tensor and reduces along one axis using a combiner
  auto reduceOp = builder.create<mlir::triton::ReduceOp>(loc,
      mlir::ValueRange{input}, reductionAxis);

  // Build combiner region — two block arguments (accumulator, element)
  auto& combinerRegion = reduceOp.getCombineOp();
  auto* combinerBlock = builder.createBlock(&combinerRegion, {}, {elemType, elemType},
                                             {loc, loc});
  auto acc = combinerBlock->getArgument(0);
  auto elem = combinerBlock->getArgument(1);

  builder.setInsertionPointToEnd(combinerBlock);

  mlir::Value combined;
  if (opKey == "reducesum" || opKey == "sum" ||
      opKey == "reducemean" || opKey == "mean" ||
      opKey == "reducenorm1" || opKey == "norm1" ||
      opKey == "reducenorm2" || opKey == "norm2") {
    combined = builder.create<mlir::arith::AddFOp>(loc, acc, elem);
  } else if (opKey == "reducemax" || opKey == "max" || opKey == "normmax") {
    combined = builder.create<mlir::arith::MaximumFOp>(loc, acc, elem);
  } else if (opKey == "reducemin" || opKey == "min") {
    combined = builder.create<mlir::arith::MinimumFOp>(loc, acc, elem);
  } else if (opKey == "reduceprod" || opKey == "prod") {
    combined = builder.create<mlir::arith::MulFOp>(loc, acc, elem);
  } else {
    std::string msg = "TritonIRBuilder::emitReductionOp: unsupported reduction '" + opName + "'";
    THROW_EXCEPTION(msg.c_str());
    combined = acc;
  }

  builder.create<mlir::triton::ReduceReturnOp>(loc, mlir::ValueRange{combined});

  // Restore insertion point to after the reduce op (was inside combiner region)
  builder.setInsertionPointAfter(reduceOp);

  // Get the reduction result
  mlir::Value result = reduceOp->getResult(0);

  // Post-processing for compound reductions
  if (opKey == "reducemean" || opKey == "mean") {
    // Divide by reduction dimension size
    int64_t reductionSize = tensorTy.getShape()[reductionAxis];
    auto countVal = builder.create<mlir::arith::ConstantOp>(
        loc, elemType, builder.getFloatAttr(elemType, static_cast<double>(reductionSize)));
    result = builder.create<mlir::arith::DivFOp>(loc, result, countVal);
  } else if (opKey == "reducenorm2" || opKey == "norm2") {
    result = builder.create<mlir::math::SqrtOp>(loc, result);
  }

  return result;
}

// ─── Normalization op emission ───────────────────────────────────────────────

mlir::Value TritonIRBuilder::emitNormalizationOp(mlir::OpBuilder& builder, mlir::Location loc,
                                                 const std::string& opName,
                                                 mlir::Value input, int axis,
                                                 mlir::RankedTensorType outputType,
                                                 mlir::Value scaleInput,
                                                 mlir::Value biasInput,
                                                 mlir::Value meanInput,
                                                 mlir::Value varianceInput,
                                                 float epsilon,
                                                 int64_t logicalReductionSize) {
  std::string opKey = normalizeOpToken(opName);

  // Promote primary input to float for math-heavy normalization arithmetic.
  input = promoteToFloat(builder, loc, input);
  auto tensorTy = mlir::cast<mlir::RankedTensorType>(input.getType());
  auto elemType = tensorTy.getElementType();

  // Optional affine/statistics tensors are consumed in float domain as well.
  if (scaleInput) scaleInput = promoteToFloat(builder, loc, scaleInput);
  if (biasInput) biasInput = promoteToFloat(builder, loc, biasInput);
  if (meanInput) meanInput = promoteToFloat(builder, loc, meanInput);
  if (varianceInput) varianceInput = promoteToFloat(builder, loc, varianceInput);

  int64_t normRank = tensorTy.getRank();
  if (axis < 0) axis += static_cast<int>(normRank);
  if (axis < 0 || axis >= static_cast<int>(normRank)) axis = 0;
  // Use logical reduction size (may differ from tensor shape if padded)
  int64_t reductionSize = logicalReductionSize > 0 ? logicalReductionSize : tensorTy.getShape()[axis];
  if (reductionSize <= 0) reductionSize = 1;

  auto makeReduce = [&](mlir::Value src, int reduceAxis, auto combinerFn) -> mlir::Value {
    auto op = builder.create<mlir::triton::ReduceOp>(loc, mlir::ValueRange{src}, reduceAxis);
    {
      auto& region = op.getCombineOp();
      auto* block = builder.createBlock(&region, {}, {elemType, elemType}, {loc, loc});
      builder.setInsertionPointToEnd(block);
      auto result = combinerFn(builder, loc, block->getArgument(0), block->getArgument(1));
      builder.create<mlir::triton::ReduceReturnOp>(loc, mlir::ValueRange{result});
    }
    builder.setInsertionPointAfter(op);
    return op->getResult(0);
  };

  auto addCombiner = [](mlir::OpBuilder& b, mlir::Location l, mlir::Value a, mlir::Value e) {
    return b.create<mlir::arith::AddFOp>(l, a, e).getResult();
  };
  auto maxCombiner = [](mlir::OpBuilder& b, mlir::Location l, mlir::Value a, mlir::Value e) {
    return b.create<mlir::arith::MaximumFOp>(l, a, e).getResult();
  };
  auto maybeApplyAffine = [&](mlir::Value val) -> mlir::Value {
    if (scaleInput) {
      auto scale = castTo(builder, loc, scaleInput, elemType);
      val = builder.create<mlir::arith::MulFOp>(loc, val, scale);
    }
    if (biasInput) {
      auto bias = castTo(builder, loc, biasInput, elemType);
      val = builder.create<mlir::arith::AddFOp>(loc, val, bias);
    }
    return val;
  };

  mlir::Value result;
  if (opKey == "softmax") {
    auto maxResult = makeReduce(input, axis, maxCombiner);
    auto maxSplat = builder.create<mlir::triton::SplatOp>(loc, tensorTy, maxResult);
    auto shifted = builder.create<mlir::arith::SubFOp>(loc, input, maxSplat);
    auto expShifted = builder.create<mlir::math::ExpOp>(loc, shifted);
    auto sumResult = makeReduce(expShifted, axis, addCombiner);
    auto sumSplat = builder.create<mlir::triton::SplatOp>(loc, tensorTy, sumResult);
    result = builder.create<mlir::arith::DivFOp>(loc, expShifted, sumSplat);

  } else if (opKey == "logsoftmax") {
    auto maxResult = makeReduce(input, axis, maxCombiner);
    auto maxSplat = builder.create<mlir::triton::SplatOp>(loc, tensorTy, maxResult);
    auto shifted = builder.create<mlir::arith::SubFOp>(loc, input, maxSplat);
    auto expShifted = builder.create<mlir::math::ExpOp>(loc, shifted);
    auto sumResult = makeReduce(expShifted, axis, addCombiner);
    auto logSum = builder.create<mlir::math::LogOp>(loc, sumResult);
    auto logSumSplat = builder.create<mlir::triton::SplatOp>(loc, tensorTy, logSum);
    result = builder.create<mlir::arith::SubFOp>(loc, shifted, logSumSplat);

  } else if (opKey == "rmsnorm" || opKey == "skiprmsnorm") {
    // skip_rms_norm: the residual add (input + skip) is handled in the module
    // dispatch before calling this emitter, so `input` already contains `hidden`.
    // The RMS norm computation is identical to rms_norm.
    mlir::Value squared = builder.create<mlir::arith::MulFOp>(loc, input, input);
    mlir::Value sumSquared;
    // Match rmsBlockReduceSum exactly for the one-value-per-thread kernels:
    // each warp pairs lower and upper halves at offsets 16,8,4,2,1, then
    // warp 0 applies the same tree to the warp totals. Generic tt.reduce is
    // free to reassociate and differs by one ULP for production decode rows.
    bool nativeWarpTree = tensorTy.getRank() == 1 && axis == 0 &&
                          tensorTy.getShape()[0] == reductionSize &&
                          reductionSize >= 32 && reductionSize <= 1024 &&
                          (reductionSize & (reductionSize - 1)) == 0;
    if (nativeWarpTree) {
      auto roundedBinary = [&](mlir::Value lhs, mlir::Value rhs,
                               const char* symbol) -> mlir::Value {
        return builder.create<mlir::triton::ExternElementwiseOp>(
            loc, lhs.getType(), mlir::ValueRange{lhs, rhs},
            /*libname=*/"", /*libpath=*/"", symbol, /*pure=*/true).getResult();
      };
      // These intrinsics require an FP32 rounding boundary. In particular,
      // __nv_fmul_rn prevents LLVM from contracting x*x + y*y into an FMA.
      squared = roundedBinary(input, input, "__nv_fmul_rn");
      int64_t numWarps = reductionSize / 32;
      mlir::Value partials = builder.create<mlir::triton::ReshapeOp>(
          loc, mlir::RankedTensorType::get({numWarps, 32}, elemType),
          squared, /*allowReorder=*/false);
      for (int64_t width = 32; width > 1; width /= 2) {
        auto paired = builder.create<mlir::triton::ReshapeOp>(
            loc, mlir::RankedTensorType::get({numWarps, 2, width / 2}, elemType),
            partials, /*allowReorder=*/false);
        auto order = builder.getDenseI32ArrayAttr({0, 2, 1});
        auto transposed = builder.create<mlir::triton::TransOp>(loc, paired, order);
        auto split = builder.create<mlir::triton::SplitOp>(loc, transposed);
        partials = roundedBinary(
            split.getResult(0), split.getResult(1), "__nv_fadd_rn");
      }
      partials = builder.create<mlir::triton::ReshapeOp>(
          loc, mlir::RankedTensorType::get({numWarps}, elemType),
          partials, /*allowReorder=*/false);
      for (int64_t width = numWarps; width > 1; width /= 2) {
        auto paired = builder.create<mlir::triton::ReshapeOp>(
            loc, mlir::RankedTensorType::get({2, width / 2}, elemType),
            partials, /*allowReorder=*/false);
        auto order = builder.getDenseI32ArrayAttr({1, 0});
        auto transposed = builder.create<mlir::triton::TransOp>(loc, paired, order);
        auto split = builder.create<mlir::triton::SplitOp>(loc, transposed);
        partials = roundedBinary(
            split.getResult(0), split.getResult(1), "__nv_fadd_rn");
      }
      sumSquared = makeReduce(partials, 0, addCombiner);
    } else {
      sumSquared = makeReduce(squared, axis, addCombiner);
    }
    auto countVal = builder.create<mlir::arith::ConstantOp>(
        loc, elemType, builder.getFloatAttr(elemType, static_cast<double>(reductionSize)));
    auto meanSquared = builder.create<mlir::arith::DivFOp>(loc, sumSquared, countVal);
    auto epsVal = builder.create<mlir::arith::ConstantOp>(
        loc, elemType, builder.getFloatAttr(elemType, static_cast<double>(epsilon)));
    auto meanPlusEps = builder.create<mlir::arith::AddFOp>(loc, meanSquared, epsVal);
    auto rsqrtVal = builder.create<mlir::math::RsqrtOp>(loc, meanPlusEps);
    auto rsqrtSplat = builder.create<mlir::triton::SplatOp>(loc, tensorTy, rsqrtVal);
    result = builder.create<mlir::arith::MulFOp>(loc, input, rsqrtSplat);
    result = maybeApplyAffine(result);

  } else if (opKey == "layernorm" || opKey == "layernormalization") {
    auto countVal = builder.create<mlir::arith::ConstantOp>(
        loc, elemType, builder.getFloatAttr(elemType, static_cast<double>(reductionSize)));

    auto sumResult = makeReduce(input, axis, addCombiner);
    auto meanVal = builder.create<mlir::arith::DivFOp>(loc, sumResult, countVal);
    auto meanSplat = builder.create<mlir::triton::SplatOp>(loc, tensorTy, meanVal);
    auto centered = builder.create<mlir::arith::SubFOp>(loc, input, meanSplat);

    auto centeredSq = builder.create<mlir::arith::MulFOp>(loc, centered, centered);
    auto varSum = makeReduce(centeredSq, axis, addCombiner);
    auto varianceVal = builder.create<mlir::arith::DivFOp>(loc, varSum, countVal);
    auto epsVal = builder.create<mlir::arith::ConstantOp>(
        loc, elemType, builder.getFloatAttr(elemType, static_cast<double>(epsilon)));
    auto varPlusEps = builder.create<mlir::arith::AddFOp>(loc, varianceVal, epsVal);
    auto rsqrtVal = builder.create<mlir::math::RsqrtOp>(loc, varPlusEps);
    auto rsqrtSplat = builder.create<mlir::triton::SplatOp>(loc, tensorTy, rsqrtVal);
    result = builder.create<mlir::arith::MulFOp>(loc, centered, rsqrtSplat);
    result = maybeApplyAffine(result);

  } else if (opKey == "batchnorm") {
    if (!meanInput || !varianceInput) {
      std::string msg = "TritonIRBuilder::emitNormalizationOp: batch_norm requires mean and variance tensors";
      THROW_EXCEPTION(msg.c_str());
    }
    auto meanVal = castTo(builder, loc, meanInput, elemType);
    auto varVal = castTo(builder, loc, varianceInput, elemType);
    auto centered = builder.create<mlir::arith::SubFOp>(loc, input, meanVal);
    auto epsVal = builder.create<mlir::arith::ConstantOp>(
        loc, elemType, builder.getFloatAttr(elemType, static_cast<double>(epsilon)));
    auto varPlusEps = builder.create<mlir::arith::AddFOp>(loc, varVal, epsVal);
    auto invStd = builder.create<mlir::math::RsqrtOp>(loc, varPlusEps);
    result = builder.create<mlir::arith::MulFOp>(loc, centered, invStd);
    result = maybeApplyAffine(result);

  } else if (opKey == "normalizemoments") {
    std::string msg = "TritonIRBuilder::emitNormalizationOp: normalize_moments has two outputs and is unsupported in "
                      "single-output normalization emission";
    THROW_EXCEPTION(msg.c_str());
    return input;

  } else {
    std::string msg = "TritonIRBuilder::emitNormalizationOp: unsupported normalization '" + opName + "'";
    THROW_EXCEPTION(msg.c_str());
    return input;
  }

  if (!result) return input;
  if (outputType) {
    result = castTo(builder, loc, result, outputType.getElementType());
  }
  return result;
}

// ─── Normalization backward emitter ─────────────────────────────────────────
//
// emitNormalizationBackwardSection
//   Emits the backward pass for rms_norm_bp and layer_norm_bp / fused_layer_norm_bp
//   into the current insertion point of `builder`.
//
// All tensor-valued inputs (dy, x, gamma) are padded-row tensors of element
// type f32 (already promoted by the caller via loadRowFromBuffer).
// The emitter writes dx, dgamma (and dbeta for layer-norm) directly to global
// memory using tt.store so that each row block handles all its gradient updates.
//
// Algorithm — rms_norm_bp
//   rms   = sqrt(mean(x^2, axis=last) + eps)        # scalar per row
//   invRms = 1 / rms
//   x_hat  = x * invRms                              # [paddedRowLen]
//   dgamma = sum(dy * x_hat, axis=0)                 # [paddedRowLen] — accumulated over rows
//   coeff  = mean(dy * gamma * x_hat, axis=last)     # scalar per row
//   dx     = invRms * (dy * gamma - x_hat * coeff)   # [paddedRowLen]
//
// Algorithm — layer_norm_bp (uses pre-computed invStd = 1/sqrt(var+eps))
//   x_hat   = (x - mean) * invStd
//   dbeta   = sum(dy, axis=0)                         # accumulated over rows
//   dgamma  = sum(dy * x_hat, axis=0)                 # accumulated over rows
//   dx_hat  = dy * gamma                              # [paddedRowLen]
//   coeff1  = mean(dx_hat, axis=last)                 # scalar per row
//   coeff2  = mean(dx_hat * x_hat, axis=last)         # scalar per row
//   dx      = invStd * (dx_hat - coeff1 - x_hat * coeff2)
//
// Note: dgamma and dbeta are ACCUMULATED across rows (each row's block adds its
// contribution).  The caller must initialise these buffers to zero before the
// kernel launch.  Because parallel blocks write to the same dgamma/dbeta
// locations we use tt.atomic_rmw ADD to avoid data races.

bool emitNormalizationBackwardSection(
    mlir::OpBuilder& builder, mlir::Location loc,
    const std::string& opName,
    mlir::Value dy, mlir::Value x,
    mlir::Value gamma,
    mlir::Value mean, mlir::Value invStd,
    float epsilon,
    int64_t logicalRowLen,
    mlir::Value rowBase,
    mlir::Value dxPtr,
    mlir::Value dgammaPtr,
    mlir::Value dbetaPtr) {

  using namespace ir_builder_internal;

  // normalizeOpToken strips all non-alphanumeric chars and lowercases:
  //   rms_norm_bp -> rmsnormbp, layer_norm_bp -> layernormbp, fused_layer_norm_bp -> fusedlayernormbp
  std::string opKey = normalizeOpToken(opName);
  bool isRmsNormBp    = (opKey == "rmsnormbp");
  bool isLayerNormBp  = (opKey == "layernormbp"          ||
                          opKey == "layernormalizationbp" ||
                          opKey == "fusedlayernormbp");

  if (!isRmsNormBp && !isLayerNormBp) {
    return false;
  }

  // Compute paddedRowLen = next power-of-2 >= logicalRowLen
  int64_t paddedRowLen = 1;
  while (paddedRowLen < logicalRowLen) paddedRowLen <<= 1;

  auto f32Type    = builder.getF32Type();
  auto i32Type    = builder.getI32Type();

  auto rowTensorType   = mlir::RankedTensorType::get({paddedRowLen}, f32Type);
  auto rowIdxType      = mlir::RankedTensorType::get({paddedRowLen}, i32Type);

  // ── Build per-row range, global offsets, and mask ──────────────────────────
  auto rowRange  = builder.create<mlir::triton::MakeRangeOp>(loc, rowIdxType, 0, paddedRowLen);
  auto rowBaseSplat = builder.create<mlir::triton::SplatOp>(loc, rowIdxType, rowBase);
  auto rowOffsets = builder.create<mlir::arith::AddIOp>(loc, rowBaseSplat, rowRange);

  auto logRowLenConst = builder.create<mlir::arith::ConstantIntOp>(loc, logicalRowLen, 32);
  auto logRowLenSplat = builder.create<mlir::triton::SplatOp>(loc, rowIdxType, logRowLenConst);
  auto rowMask = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::slt, rowRange, logRowLenSplat);

  // Helper: promote tensor to f32 (the arithmetic domain for norm math)
  auto promoteF32 = [&](mlir::Value v) -> mlir::Value {
    return castTo(builder, loc, v, f32Type);
  };

  // Ensure dy and x are f32 row tensors
  dy = promoteF32(dy);
  x  = promoteF32(x);
  if (gamma) gamma = promoteF32(gamma);
  if (mean)  mean  = promoteF32(mean);
  if (invStd) invStd = promoteF32(invStd);

  // Helper: row-wide reduce-sum → scalar f32
  auto rowReduceSum = [&](mlir::Value rowTensor) -> mlir::Value {
    auto op = builder.create<mlir::triton::ReduceOp>(loc, mlir::ValueRange{rowTensor}, /*axis=*/0);
    {
      auto& region = op.getCombineOp();
      auto* block = builder.createBlock(&region, {}, {f32Type, f32Type}, {loc, loc});
      builder.setInsertionPointToEnd(block);
      auto s = builder.create<mlir::arith::AddFOp>(loc, block->getArgument(0), block->getArgument(1));
      builder.create<mlir::triton::ReduceReturnOp>(loc, mlir::ValueRange{s});
    }
    builder.setInsertionPointAfter(op);
    return op->getResult(0);  // scalar f32
  };

  // Helper: splat a scalar f32 to the row tensor type
  auto splatF32 = [&](mlir::Value scalar) -> mlir::Value {
    return builder.create<mlir::triton::SplatOp>(loc, rowTensorType, scalar);
  };

  // Helper: scalar constant f32
  auto constF32 = [&](float v) -> mlir::Value {
    return builder.create<mlir::arith::ConstantOp>(
        loc, f32Type, builder.getFloatAttr(f32Type, static_cast<double>(v)));
  };

  // ── Compute invNorm (1/rms or pre-computed invStd) ──────────────────────────
  // For rms_norm_bp: compute RMS from x on the fly.
  // For layer_norm_bp: use pre-computed invStd + mean passed from caller.

  mlir::Value invNorm;   // scalar f32: 1/rms or 1/std
  mlir::Value xHat;      // [paddedRowLen] f32: normalized x

  // Logically correct count = logicalRowLen (not paddedRowLen, which may include padding zeros)
  auto countF = constF32(static_cast<float>(logicalRowLen));

  if (isRmsNormBp) {
    // rms = sqrt(mean(x^2) + eps)
    auto xSq     = builder.create<mlir::arith::MulFOp>(loc, x, x);
    auto sumXSq  = rowReduceSum(xSq);
    auto meanXSq = builder.create<mlir::arith::DivFOp>(loc, sumXSq, countF);
    auto epsVal  = constF32(epsilon);
    auto varPlusEps = builder.create<mlir::arith::AddFOp>(loc, meanXSq, epsVal);
    invNorm = builder.create<mlir::math::RsqrtOp>(loc, varPlusEps);   // scalar: 1/rms
    // x_hat = x * invRms
    xHat = builder.create<mlir::arith::MulFOp>(loc, x, splatF32(invNorm));
  } else {
    // layer_norm_bp: use pre-computed invStd and mean
    if (invStd && mean) {
      // invStd and mean are already scalar f32 values loaded by the caller.
      // Use directly — no row-tensor path needed (callers always pass scalars here).
      invNorm = invStd;
      // x_hat = (x - mean) * invStd
      auto meanSplat = splatF32(mean);
      auto xCentered = builder.create<mlir::arith::SubFOp>(loc, x, meanSplat);
      xHat = builder.create<mlir::arith::MulFOp>(loc, xCentered, splatF32(invStd));
    } else {
      // Fallback: compute mean and variance from x directly
      auto sumX  = rowReduceSum(x);
      auto meanX = builder.create<mlir::arith::DivFOp>(loc, sumX, countF);
      auto meanSplat = splatF32(meanX);
      auto xCentered = builder.create<mlir::arith::SubFOp>(loc, x, meanSplat);
      auto xCentSq = builder.create<mlir::arith::MulFOp>(loc, xCentered, xCentered);
      auto sumVar = rowReduceSum(xCentSq);
      auto varVal = builder.create<mlir::arith::DivFOp>(loc, sumVar, countF);
      auto epsVal = constF32(epsilon);
      auto varPlusEps = builder.create<mlir::arith::AddFOp>(loc, varVal, epsVal);
      invNorm = builder.create<mlir::math::RsqrtOp>(loc, varPlusEps);
      xHat = builder.create<mlir::arith::MulFOp>(loc, xCentered, splatF32(invNorm));
    }
  }

  // ── Compute dx ───────────────────────────────────────────────────────────────

  // Apply gamma: dyGamma = dy * gamma (element-wise) — or just dy if no gamma
  mlir::Value dyGamma = dy;
  if (gamma) {
    dyGamma = builder.create<mlir::arith::MulFOp>(loc, dy, gamma);
  }

  // coeff = mean(dyGamma * xHat, axis=last)  → scalar
  auto dyGammaXHat = builder.create<mlir::arith::MulFOp>(loc, dyGamma, xHat);
  auto sumDyGammaXHat = rowReduceSum(dyGammaXHat);
  auto coeffScalar = builder.create<mlir::arith::DivFOp>(loc, sumDyGammaXHat, countF);

  mlir::Value dx;
  if (isRmsNormBp) {
    // dx = invRms * (dyGamma - xHat * coeff)
    auto xHatCoeff = builder.create<mlir::arith::MulFOp>(loc, xHat, splatF32(coeffScalar));
    auto dyMinusCoeff = builder.create<mlir::arith::SubFOp>(loc, dyGamma, xHatCoeff);
    dx = builder.create<mlir::arith::MulFOp>(loc, splatF32(invNorm), dyMinusCoeff);
  } else {
    // layer_norm_bp:
    //   dx_hat = dy * gamma  (= dyGamma)
    //   coeff1 = mean(dx_hat)
    //   coeff2 = mean(dx_hat * x_hat) = coeffScalar (computed above)
    //   dx = invStd * (dx_hat - coeff1 - x_hat * coeff2)
    auto sumDyGamma = rowReduceSum(dyGamma);
    auto coeff1 = builder.create<mlir::arith::DivFOp>(loc, sumDyGamma, countF);
    auto coeff2 = coeffScalar;
    auto term1 = builder.create<mlir::arith::SubFOp>(loc, dyGamma, splatF32(coeff1));
    auto xHatCoeff2 = builder.create<mlir::arith::MulFOp>(loc, xHat, splatF32(coeff2));
    auto dxHat = builder.create<mlir::arith::SubFOp>(loc, term1, xHatCoeff2);
    dx = builder.create<mlir::arith::MulFOp>(loc, splatF32(invNorm), dxHat);
  }

  // ── Store dx via tt.store ────────────────────────────────────────────────────
  if (dxPtr) {
    auto ptrType = mlir::cast<mlir::triton::PointerType>(dxPtr.getType());
    auto ptrTensorType = mlir::RankedTensorType::get({paddedRowLen}, ptrType);
    auto splatDxPtr = builder.create<mlir::triton::SplatOp>(loc, ptrTensorType, dxPtr);
    auto dxPtrs = builder.create<mlir::triton::AddPtrOp>(loc, ptrTensorType, splatDxPtr, rowOffsets);
    auto dxStore = castTo(builder, loc, dx, ptrType.getPointeeType());
    builder.create<mlir::triton::StoreOp>(loc, dxPtrs, dxStore, rowMask,
        mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL);
  }

  // ── Atomic accumulate dgamma: dgamma[j] += dy[j] * x_hat[j] ─────────────────
  // dgamma is accumulated over all rows (one per block) using atomic add.
  // Use a lane-local range (0..paddedRowLen) not offset by rowBase.
  if (dgammaPtr) {
    auto dgammaElem = builder.create<mlir::arith::MulFOp>(loc, dy, xHat);
    auto gammaPtrType = mlir::cast<mlir::triton::PointerType>(dgammaPtr.getType());
    auto gammaElemType = gammaPtrType.getPointeeType();
    auto gammaDataTensorType = mlir::RankedTensorType::get({paddedRowLen}, gammaElemType);
    auto gammaPtrTensorType = mlir::RankedTensorType::get({paddedRowLen}, gammaPtrType);
    // dgamma is indexed by column (j = 0..logicalRowLen-1), not by row+col
    auto splatGammaPtr = builder.create<mlir::triton::SplatOp>(loc, gammaPtrTensorType, dgammaPtr);
    auto dgammaPtrs = builder.create<mlir::triton::AddPtrOp>(loc, gammaPtrTensorType, splatGammaPtr, rowRange);
    auto dgammaStoreVal = castTo(builder, loc, dgammaElem, gammaElemType);
    builder.create<mlir::triton::AtomicRMWOp>(loc, gammaDataTensorType,
        mlir::triton::RMWOp::FADD,
        dgammaPtrs, dgammaStoreVal, rowMask,
        mlir::triton::MemSemantic::RELAXED, mlir::triton::MemSyncScope::GPU);
  }

  // ── Atomic accumulate dbeta: dbeta[j] += dy[j] ───────────────────────────────
  // Only for layer_norm_bp (rms_norm has no beta).
  if (dbetaPtr && isLayerNormBp) {
    auto betaPtrType = mlir::cast<mlir::triton::PointerType>(dbetaPtr.getType());
    auto betaElemType = betaPtrType.getPointeeType();
    auto betaDataTensorType = mlir::RankedTensorType::get({paddedRowLen}, betaElemType);
    auto betaPtrTensorType = mlir::RankedTensorType::get({paddedRowLen}, betaPtrType);
    auto splatBetaPtr = builder.create<mlir::triton::SplatOp>(loc, betaPtrTensorType, dbetaPtr);
    auto dbetaPtrs = builder.create<mlir::triton::AddPtrOp>(loc, betaPtrTensorType, splatBetaPtr, rowRange);
    auto dbetaStoreVal = castTo(builder, loc, dy, betaElemType);
    builder.create<mlir::triton::AtomicRMWOp>(loc, betaDataTensorType,
        mlir::triton::RMWOp::FADD,
        dbetaPtrs, dbetaStoreVal, rowMask,
        mlir::triton::MemSemantic::RELAXED, mlir::triton::MemSyncScope::GPU);
  }

  return true;
}

}  // namespace graph
}  // namespace sd

#endif  // HAVE_TRITON
