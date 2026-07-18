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

#ifndef LIBND4J_KERNELSPEC_KERNELEXPRMLIREMITTER_H
#define LIBND4J_KERNELSPEC_KERNELEXPRMLIREMITTER_H

// Shared MLIR interpreter for KernelExpr graphs (ADR-0116). One emission
// walk serves every MLIR-value-based backend; precision-sensitive primitives
// (exp/log/div/pow/...) and constant materialization go through a per-backend
// policy so Triton can substitute its __nv_* libdevice emitters and Vulkan its
// linalg-body constants while CPU uses the stock math dialect.
//
// This header requires MLIR headers on the include path. Include it only from
// translation units already compiled against MLIR (HAVE_MLIR builds, or a
// Triton build's vendored MLIR). It is intentionally header-only so it adds no
// unconditioned translation unit to non-MLIR builds.
//
// Type contract (mirrors the existing category emitters): all `inputs` share
// one float compute type (scalar or statically-shaped tensor); the caller
// performs promotion beforehand. Comparisons produce i1(-shaped) values.
// SCALAR_PARAM / CONST_F are materialized as splat constants of
// `typeExemplar`'s type — baked, matching today's Triton/Vulkan behavior.

#include <graph/kernelspec/KernelExpr.h>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#if __has_include("mlir/Dialect/Math/IR/Math.h")
#include "mlir/Dialect/Math/IR/Math.h"
#define SD_KSPEC_HAS_MATH 1
#endif

#include <functional>
#include <stdexcept>
#include <string>

namespace sd {
namespace kernelspec {

// Per-backend emission hooks. Everything not covered here (add/sub/mul/
// min/max/neg/compares/select/logic) is emitted directly as arith ops.
struct MlirEmitPolicy {
  using UnaryFn = std::function<mlir::Value(mlir::OpBuilder&, mlir::Location, mlir::Value)>;
  using BinaryFn = std::function<mlir::Value(mlir::OpBuilder&, mlir::Location, mlir::Value, mlir::Value)>;
  using SplatFn = std::function<mlir::Value(mlir::OpBuilder&, mlir::Location, mlir::Value /*typeExemplar*/, double)>;

  UnaryFn expFn, logFn, sqrtFn, tanhFn, sinFn, cosFn, erfFn, absFn, floorFn, ceilFn, roundFn;
  BinaryFn divFn, powFn;
  SplatFn splatFn;
};

// Splat/scalar float constant of the exemplar's type. Shaped exemplars must be
// statically shaped (true for Triton block tensors and Vulkan loop bodies).
inline mlir::Value kspecSplatConstant(mlir::OpBuilder& builder, mlir::Location loc,
                                      mlir::Value typeExemplar, double value) {
  auto type = typeExemplar.getType();
  if (auto shaped = mlir::dyn_cast<mlir::ShapedType>(type)) {
    auto floatType = mlir::dyn_cast<mlir::FloatType>(shaped.getElementType());
    if (!floatType) throw std::runtime_error("kernelspec: splat constant requires a float element type");
    auto elem = builder.getFloatAttr(floatType, value);
    auto dense = mlir::DenseElementsAttr::get(shaped, llvm::ArrayRef<mlir::Attribute>{elem});
    return builder.create<mlir::arith::ConstantOp>(loc, dense);
  }
  auto floatType = mlir::dyn_cast<mlir::FloatType>(type);
  if (!floatType) throw std::runtime_error("kernelspec: splat constant requires a float type");
  return builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(floatType, value));
}

// i1(-shaped) constant with the same shape as `boolExemplar` (an existing
// comparison result). Used for NOT via xor-with-true.
inline mlir::Value kspecBoolConstantLike(mlir::OpBuilder& builder, mlir::Location loc,
                                         mlir::Value boolExemplar, bool value) {
  auto type = boolExemplar.getType();
  if (auto shaped = mlir::dyn_cast<mlir::ShapedType>(type)) {
    auto elem = builder.getBoolAttr(value);
    auto dense = mlir::DenseElementsAttr::get(shaped, llvm::ArrayRef<mlir::Attribute>{elem});
    return builder.create<mlir::arith::ConstantOp>(loc, dense);
  }
  return builder.create<mlir::arith::ConstantOp>(loc, builder.getIntegerAttr(type, value ? 1 : 0));
}

// Default policy: stock math-dialect lowering, arith division, dense splats.
// Backends override individual hooks (e.g. Triton's emitNativeCudaExp for
// bit-exact libdevice math) without touching the interpreter.
inline MlirEmitPolicy makeDefaultMlirEmitPolicy() {
  MlirEmitPolicy p;
  p.splatFn = &kspecSplatConstant;
  p.divFn = [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value l, mlir::Value r) -> mlir::Value {
    return b.create<mlir::arith::DivFOp>(loc, l, r);
  };
#if defined(SD_KSPEC_HAS_MATH)
  p.expFn = [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value v) -> mlir::Value {
    return b.create<mlir::math::ExpOp>(loc, v);
  };
  p.logFn = [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value v) -> mlir::Value {
    return b.create<mlir::math::LogOp>(loc, v);
  };
  p.sqrtFn = [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value v) -> mlir::Value {
    return b.create<mlir::math::SqrtOp>(loc, v);
  };
  p.tanhFn = [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value v) -> mlir::Value {
    return b.create<mlir::math::TanhOp>(loc, v);
  };
  p.sinFn = [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value v) -> mlir::Value {
    return b.create<mlir::math::SinOp>(loc, v);
  };
  p.cosFn = [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value v) -> mlir::Value {
    return b.create<mlir::math::CosOp>(loc, v);
  };
  p.erfFn = [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value v) -> mlir::Value {
    return b.create<mlir::math::ErfOp>(loc, v);
  };
  p.absFn = [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value v) -> mlir::Value {
    return b.create<mlir::math::AbsFOp>(loc, v);
  };
  p.floorFn = [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value v) -> mlir::Value {
    return b.create<mlir::math::FloorOp>(loc, v);
  };
  p.ceilFn = [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value v) -> mlir::Value {
    return b.create<mlir::math::CeilOp>(loc, v);
  };
  p.roundFn = [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value v) -> mlir::Value {
    return b.create<mlir::math::RoundOp>(loc, v);
  };
  p.powFn = [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value l, mlir::Value r) -> mlir::Value {
    return b.create<mlir::math::PowFOp>(loc, l, r);
  };
#endif
  return p;
}

inline mlir::Value emitKernelExpr(mlir::OpBuilder& builder, mlir::Location loc,
                                  const ExprGraph& graph,
                                  llvm::ArrayRef<mlir::Value> inputs,
                                  llvm::ArrayRef<double> scalarValues,
                                  mlir::Value typeExemplar,
                                  const MlirEmitPolicy& policy) {
  auto err = graph.validate();
  if (!err.empty()) throw std::invalid_argument("kernelspec: invalid expression: " + err);
  if (static_cast<int32_t>(inputs.size()) < graph.inputArity())
    throw std::invalid_argument("kernelspec: expression needs " + std::to_string(graph.inputArity()) +
                                " inputs, got " + std::to_string(inputs.size()));
  if (static_cast<int32_t>(scalarValues.size()) < graph.scalarArity())
    throw std::invalid_argument("kernelspec: expression needs " + std::to_string(graph.scalarArity()) +
                                " scalar values, got " + std::to_string(scalarValues.size()));

  auto requireHook = [](const auto& fn, const char* what) {
    if (!fn) throw std::runtime_error(std::string("kernelspec: emit policy has no hook for ") + what);
  };

  const auto& nodes = graph.nodes();
  llvm::SmallVector<mlir::Value, 32> memo(nodes.size());

  for (size_t i = 0; i < nodes.size(); i++) {
    const auto& n = nodes[i];
    auto A = [&]() { return memo[n.a]; };
    auto B = [&]() { return memo[n.b]; };
    auto C = [&]() { return memo[n.c]; };

    switch (n.op) {
      case ExprOp::INPUT:
        memo[i] = inputs[n.index];
        break;
      case ExprOp::SCALAR_PARAM:
        requireHook(policy.splatFn, "splat");
        memo[i] = policy.splatFn(builder, loc, typeExemplar, scalarValues[n.index]);
        break;
      case ExprOp::CONST_F:
        requireHook(policy.splatFn, "splat");
        memo[i] = policy.splatFn(builder, loc, typeExemplar, n.f);
        break;

      case ExprOp::NEG:
        memo[i] = builder.create<mlir::arith::NegFOp>(loc, A());
        break;
      case ExprOp::ABS:
        requireHook(policy.absFn, "abs");
        memo[i] = policy.absFn(builder, loc, A());
        break;
      case ExprOp::EXP:
        requireHook(policy.expFn, "exp");
        memo[i] = policy.expFn(builder, loc, A());
        break;
      case ExprOp::LOG:
        requireHook(policy.logFn, "log");
        memo[i] = policy.logFn(builder, loc, A());
        break;
      case ExprOp::SQRT:
        requireHook(policy.sqrtFn, "sqrt");
        memo[i] = policy.sqrtFn(builder, loc, A());
        break;
      case ExprOp::TANH:
        requireHook(policy.tanhFn, "tanh");
        memo[i] = policy.tanhFn(builder, loc, A());
        break;
      case ExprOp::SIN:
        requireHook(policy.sinFn, "sin");
        memo[i] = policy.sinFn(builder, loc, A());
        break;
      case ExprOp::COS:
        requireHook(policy.cosFn, "cos");
        memo[i] = policy.cosFn(builder, loc, A());
        break;
      case ExprOp::ERF:
        requireHook(policy.erfFn, "erf");
        memo[i] = policy.erfFn(builder, loc, A());
        break;
      case ExprOp::FLOOR:
        requireHook(policy.floorFn, "floor");
        memo[i] = policy.floorFn(builder, loc, A());
        break;
      case ExprOp::CEIL:
        requireHook(policy.ceilFn, "ceil");
        memo[i] = policy.ceilFn(builder, loc, A());
        break;
      case ExprOp::ROUND:
        requireHook(policy.roundFn, "round");
        memo[i] = policy.roundFn(builder, loc, A());
        break;
      case ExprOp::NOT:
        memo[i] = builder.create<mlir::arith::XOrIOp>(loc, A(), kspecBoolConstantLike(builder, loc, A(), true));
        break;

      case ExprOp::ADD:
        memo[i] = builder.create<mlir::arith::AddFOp>(loc, A(), B());
        break;
      case ExprOp::SUB:
        memo[i] = builder.create<mlir::arith::SubFOp>(loc, A(), B());
        break;
      case ExprOp::MUL:
        memo[i] = builder.create<mlir::arith::MulFOp>(loc, A(), B());
        break;
      case ExprOp::DIV:
        requireHook(policy.divFn, "div");
        memo[i] = policy.divFn(builder, loc, A(), B());
        break;
      case ExprOp::POW:
        requireHook(policy.powFn, "pow");
        memo[i] = policy.powFn(builder, loc, A(), B());
        break;
      case ExprOp::MIN:
        memo[i] = builder.create<mlir::arith::MinimumFOp>(loc, A(), B());
        break;
      case ExprOp::MAX:
        memo[i] = builder.create<mlir::arith::MaximumFOp>(loc, A(), B());
        break;

      case ExprOp::CMP_LT:
        memo[i] = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OLT, A(), B());
        break;
      case ExprOp::CMP_LE:
        memo[i] = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OLE, A(), B());
        break;
      case ExprOp::CMP_GT:
        memo[i] = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OGT, A(), B());
        break;
      case ExprOp::CMP_GE:
        memo[i] = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OGE, A(), B());
        break;
      case ExprOp::CMP_EQ:
        memo[i] = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OEQ, A(), B());
        break;
      case ExprOp::CMP_NE:
        memo[i] = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::UNE, A(), B());
        break;
      case ExprOp::AND:
        memo[i] = builder.create<mlir::arith::AndIOp>(loc, A(), B());
        break;
      case ExprOp::OR:
        memo[i] = builder.create<mlir::arith::OrIOp>(loc, A(), B());
        break;

      case ExprOp::SELECT:
        memo[i] = builder.create<mlir::arith::SelectOp>(loc, A(), B(), C());
        break;
    }
  }

  return memo[graph.rootIndex()];
}

}  // namespace kernelspec
}  // namespace sd

#endif  // LIBND4J_KERNELSPEC_KERNELEXPRMLIREMITTER_H
