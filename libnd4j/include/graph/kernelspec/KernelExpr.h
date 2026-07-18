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

#ifndef LIBND4J_KERNELSPEC_KERNELEXPR_H
#define LIBND4J_KERNELSPEC_KERNELEXPR_H

#include <cstdint>
#include <string>
#include <vector>

// Kernel expression DSL (ADR-0116). A tiny, dependency-free scalar expression
// AST used to author an op's elementwise math (or a reduction's
// init/combine/finalize triple) exactly once. Backend emitters interpret the
// AST into their own IR (Triton TTIR, CPU MLIR, Vulkan linalg bodies, MLX).
//
// NOT wired into any execution path yet: nothing in the emitters consults this
// module. It is additive foundation code.
//
// Semantics contract for interpreters:
// - All INPUT values are assumed pre-promoted to one common float compute type
//   by the caller (mirrors the existing category emitters' promoteToFloat /
//   commonFloatType behavior).
// - Comparison / logical nodes produce boolean values; they may only feed
//   AND / OR / NOT / SELECT-condition positions.
// - SCALAR_PARAM(i) is the op's i-th declared scalar (tArgs-backed); v1
//   interpreters bake the resolved value as a constant, matching today's
//   Triton and Vulkan behavior.

namespace sd {
namespace kernelspec {

enum class ExprOp : uint8_t {
  // leaves
  INPUT = 0,
  SCALAR_PARAM,
  CONST_F,
  // unary
  NEG,
  ABS,
  EXP,
  LOG,
  SQRT,
  TANH,
  SIN,
  COS,
  ERF,
  FLOOR,
  CEIL,
  ROUND,
  NOT,
  // binary
  ADD,
  SUB,
  MUL,
  DIV,
  POW,
  MIN,
  MAX,
  CMP_LT,
  CMP_LE,
  CMP_GT,
  CMP_GE,
  CMP_EQ,
  CMP_NE,
  AND,
  OR,
  // ternary
  SELECT
};

int exprOpArity(ExprOp op);
const char* exprOpName(ExprOp op);
// True for ops whose result is boolean (comparisons and logical ops).
bool exprOpIsBooleanProducing(ExprOp op);

struct ExprNode {
  ExprOp op;
  int32_t a = -1;      // first child index
  int32_t b = -1;      // second child index
  int32_t c = -1;      // third child index
  double f = 0.0;      // CONST_F payload
  int32_t index = -1;  // INPUT / SCALAR_PARAM ordinal
};

class ExprGraph;

// Lightweight authoring handle. Valid only while its ExprGraph is alive and
// has not been moved; handles are never stored in a KernelSpec.
class Expr {
 public:
  Expr() = default;
  Expr(ExprGraph* g, int32_t idx) : graph_(g), idx_(idx) {}
  int32_t index() const { return idx_; }
  ExprGraph* graph() const { return graph_; }
  bool valid() const { return graph_ != nullptr && idx_ >= 0; }

 private:
  ExprGraph* graph_ = nullptr;
  int32_t idx_ = -1;
};

// Arena of expression nodes. Nodes are appended in construction order, so
// every child index is strictly smaller than its parent's index (the graph is
// topologically ordered by construction).
class ExprGraph {
 public:
  Expr input(int32_t i);
  Expr scalarParam(int32_t i);
  Expr c(double value);

  Expr unary(ExprOp op, Expr a);
  Expr binary(ExprOp op, Expr a, Expr b);
  Expr ternary(ExprOp op, Expr a, Expr b, Expr c);

  void setRoot(Expr root);
  int32_t rootIndex() const { return root_; }
  const std::vector<ExprNode>& nodes() const { return nodes_; }
  bool empty() const { return nodes_.empty(); }

  // Highest referenced INPUT / SCALAR_PARAM ordinal + 1 (0 when none).
  int32_t inputArity() const;
  int32_t scalarArity() const;

  // Empty string when structurally valid, otherwise a description of the
  // first problem found (bad arity, forward reference, non-boolean operand in
  // a boolean position, NaN constant, unset root, ...).
  std::string validate() const;
  std::string toString() const;

 private:
  Expr push(ExprNode node);

  std::vector<ExprNode> nodes_;
  int32_t root_ = -1;
};

// ── arithmetic sugar ────────────────────────────────────────────────────────
Expr operator+(Expr a, Expr b);
Expr operator-(Expr a, Expr b);
Expr operator*(Expr a, Expr b);
Expr operator/(Expr a, Expr b);
Expr operator-(Expr a);

Expr operator+(Expr a, double b);
Expr operator+(double a, Expr b);
Expr operator-(Expr a, double b);
Expr operator-(double a, Expr b);
Expr operator*(Expr a, double b);
Expr operator*(double a, Expr b);
Expr operator/(Expr a, double b);
Expr operator/(double a, Expr b);

// ── comparisons (produce boolean values) ────────────────────────────────────
Expr operator<(Expr a, Expr b);
Expr operator<=(Expr a, Expr b);
Expr operator>(Expr a, Expr b);
Expr operator>=(Expr a, Expr b);
Expr eq(Expr a, Expr b);
Expr ne(Expr a, Expr b);

Expr operator<(Expr a, double b);
Expr operator<=(Expr a, double b);
Expr operator>(Expr a, double b);
Expr operator>=(Expr a, double b);

// ── named math helpers ──────────────────────────────────────────────────────
Expr exp(Expr x);
Expr log(Expr x);
Expr sqrt(Expr x);
Expr tanh(Expr x);
Expr sin(Expr x);
Expr cos(Expr x);
Expr erf(Expr x);
Expr abs(Expr x);
Expr floor(Expr x);
Expr ceil(Expr x);
Expr round(Expr x);
Expr pow(Expr base, Expr e);
Expr pow(Expr base, double e);
Expr min(Expr a, Expr b);
Expr max(Expr a, Expr b);
Expr min(Expr a, double b);
Expr max(Expr a, double b);

// ── logical / selection ─────────────────────────────────────────────────────
Expr logicalAnd(Expr a, Expr b);
Expr logicalOr(Expr a, Expr b);
Expr logicalNot(Expr a);
Expr select(Expr cond, Expr onTrue, Expr onFalse);

// ── composite helpers (expand into primitive nodes) ─────────────────────────
Expr sigmoid(Expr x);              // 1 / (1 + exp(-x))
Expr silu(Expr x);                 // x * sigmoid(x)
Expr relu(Expr x);                 // max(x, 0)
Expr softplus(Expr x);             // log(1 + exp(x))
Expr mish(Expr x);                 // x * tanh(softplus(x))
Expr clamp(Expr x, Expr lo, Expr hi);
Expr clamp(Expr x, double lo, double hi);
Expr hardSigmoid(Expr x);          // clamp(0.2 * x + 0.5, 0, 1)

}  // namespace kernelspec
}  // namespace sd

#endif  // LIBND4J_KERNELSPEC_KERNELEXPR_H
