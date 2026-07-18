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

#include <graph/kernelspec/KernelExpr.h>

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace sd {
namespace kernelspec {

int exprOpArity(ExprOp op) {
  switch (op) {
    case ExprOp::INPUT:
    case ExprOp::SCALAR_PARAM:
    case ExprOp::CONST_F:
      return 0;
    case ExprOp::NEG:
    case ExprOp::ABS:
    case ExprOp::EXP:
    case ExprOp::LOG:
    case ExprOp::SQRT:
    case ExprOp::TANH:
    case ExprOp::SIN:
    case ExprOp::COS:
    case ExprOp::ERF:
    case ExprOp::FLOOR:
    case ExprOp::CEIL:
    case ExprOp::ROUND:
    case ExprOp::NOT:
      return 1;
    case ExprOp::ADD:
    case ExprOp::SUB:
    case ExprOp::MUL:
    case ExprOp::DIV:
    case ExprOp::POW:
    case ExprOp::MIN:
    case ExprOp::MAX:
    case ExprOp::CMP_LT:
    case ExprOp::CMP_LE:
    case ExprOp::CMP_GT:
    case ExprOp::CMP_GE:
    case ExprOp::CMP_EQ:
    case ExprOp::CMP_NE:
    case ExprOp::AND:
    case ExprOp::OR:
      return 2;
    case ExprOp::SELECT:
      return 3;
  }
  return -1;
}

const char* exprOpName(ExprOp op) {
  switch (op) {
    case ExprOp::INPUT: return "input";
    case ExprOp::SCALAR_PARAM: return "scalar";
    case ExprOp::CONST_F: return "const";
    case ExprOp::NEG: return "neg";
    case ExprOp::ABS: return "abs";
    case ExprOp::EXP: return "exp";
    case ExprOp::LOG: return "log";
    case ExprOp::SQRT: return "sqrt";
    case ExprOp::TANH: return "tanh";
    case ExprOp::SIN: return "sin";
    case ExprOp::COS: return "cos";
    case ExprOp::ERF: return "erf";
    case ExprOp::FLOOR: return "floor";
    case ExprOp::CEIL: return "ceil";
    case ExprOp::ROUND: return "round";
    case ExprOp::NOT: return "not";
    case ExprOp::ADD: return "add";
    case ExprOp::SUB: return "sub";
    case ExprOp::MUL: return "mul";
    case ExprOp::DIV: return "div";
    case ExprOp::POW: return "pow";
    case ExprOp::MIN: return "min";
    case ExprOp::MAX: return "max";
    case ExprOp::CMP_LT: return "lt";
    case ExprOp::CMP_LE: return "le";
    case ExprOp::CMP_GT: return "gt";
    case ExprOp::CMP_GE: return "ge";
    case ExprOp::CMP_EQ: return "eq";
    case ExprOp::CMP_NE: return "ne";
    case ExprOp::AND: return "and";
    case ExprOp::OR: return "or";
    case ExprOp::SELECT: return "select";
  }
  return "?";
}

bool exprOpIsBooleanProducing(ExprOp op) {
  switch (op) {
    case ExprOp::CMP_LT:
    case ExprOp::CMP_LE:
    case ExprOp::CMP_GT:
    case ExprOp::CMP_GE:
    case ExprOp::CMP_EQ:
    case ExprOp::CMP_NE:
    case ExprOp::AND:
    case ExprOp::OR:
    case ExprOp::NOT:
      return true;
    default:
      return false;
  }
}

Expr ExprGraph::push(ExprNode node) {
  nodes_.push_back(node);
  return Expr(this, static_cast<int32_t>(nodes_.size()) - 1);
}

Expr ExprGraph::input(int32_t i) {
  ExprNode n;
  n.op = ExprOp::INPUT;
  n.index = i;
  return push(n);
}

Expr ExprGraph::scalarParam(int32_t i) {
  ExprNode n;
  n.op = ExprOp::SCALAR_PARAM;
  n.index = i;
  return push(n);
}

Expr ExprGraph::c(double value) {
  ExprNode n;
  n.op = ExprOp::CONST_F;
  n.f = value;
  return push(n);
}

Expr ExprGraph::unary(ExprOp op, Expr a) {
  if (a.graph() != this) throw std::invalid_argument("kernelspec: operand belongs to a different ExprGraph");
  ExprNode n;
  n.op = op;
  n.a = a.index();
  return push(n);
}

Expr ExprGraph::binary(ExprOp op, Expr a, Expr b) {
  if (a.graph() != this || b.graph() != this)
    throw std::invalid_argument("kernelspec: operand belongs to a different ExprGraph");
  ExprNode n;
  n.op = op;
  n.a = a.index();
  n.b = b.index();
  return push(n);
}

Expr ExprGraph::ternary(ExprOp op, Expr a, Expr b, Expr c) {
  if (a.graph() != this || b.graph() != this || c.graph() != this)
    throw std::invalid_argument("kernelspec: operand belongs to a different ExprGraph");
  ExprNode n;
  n.op = op;
  n.a = a.index();
  n.b = b.index();
  n.c = c.index();
  return push(n);
}

void ExprGraph::setRoot(Expr root) {
  if (root.graph() != this) throw std::invalid_argument("kernelspec: root belongs to a different ExprGraph");
  root_ = root.index();
}

int32_t ExprGraph::inputArity() const {
  int32_t arity = 0;
  for (const auto& n : nodes_)
    if (n.op == ExprOp::INPUT && n.index + 1 > arity) arity = n.index + 1;
  return arity;
}

int32_t ExprGraph::scalarArity() const {
  int32_t arity = 0;
  for (const auto& n : nodes_)
    if (n.op == ExprOp::SCALAR_PARAM && n.index + 1 > arity) arity = n.index + 1;
  return arity;
}

std::string ExprGraph::validate() const {
  if (nodes_.empty()) return "expression graph is empty";
  if (root_ < 0 || root_ >= static_cast<int32_t>(nodes_.size())) return "root is unset or out of range";

  for (size_t i = 0; i < nodes_.size(); i++) {
    const auto& n = nodes_[i];
    const int arity = exprOpArity(n.op);
    if (arity < 0) return "node " + std::to_string(i) + ": unknown op";

    const int32_t children[3] = {n.a, n.b, n.c};
    for (int k = 0; k < 3; k++) {
      if (k < arity) {
        if (children[k] < 0 || children[k] >= static_cast<int32_t>(i))
          return "node " + std::to_string(i) + " (" + exprOpName(n.op) +
                 "): child " + std::to_string(k) + " is unset or a forward reference";
      } else if (children[k] != -1) {
        return "node " + std::to_string(i) + " (" + exprOpName(n.op) + "): too many operands";
      }
    }

    if ((n.op == ExprOp::INPUT || n.op == ExprOp::SCALAR_PARAM) && n.index < 0)
      return "node " + std::to_string(i) + ": negative " + std::string(exprOpName(n.op)) + " ordinal";

    if (n.op == ExprOp::CONST_F && std::isnan(n.f))
      return "node " + std::to_string(i) + ": NaN constant";

    // Boolean-typed positions must be fed by boolean-producing nodes.
    if (n.op == ExprOp::AND || n.op == ExprOp::OR) {
      if (!exprOpIsBooleanProducing(nodes_[n.a].op) || !exprOpIsBooleanProducing(nodes_[n.b].op))
        return "node " + std::to_string(i) + " (" + exprOpName(n.op) + "): operands must be boolean";
    }
    if (n.op == ExprOp::NOT && !exprOpIsBooleanProducing(nodes_[n.a].op))
      return "node " + std::to_string(i) + " (not): operand must be boolean";
    if (n.op == ExprOp::SELECT && !exprOpIsBooleanProducing(nodes_[n.a].op))
      return "node " + std::to_string(i) + " (select): condition must be boolean";
  }
  return "";
}

namespace {
void printNode(const std::vector<ExprNode>& nodes, int32_t idx, std::ostringstream& out) {
  const auto& n = nodes[idx];
  switch (n.op) {
    case ExprOp::INPUT:
      out << "in" << n.index;
      return;
    case ExprOp::SCALAR_PARAM:
      out << "s" << n.index;
      return;
    case ExprOp::CONST_F:
      out << n.f;
      return;
    default:
      break;
  }
  out << exprOpName(n.op) << "(";
  const int arity = exprOpArity(n.op);
  const int32_t children[3] = {n.a, n.b, n.c};
  for (int k = 0; k < arity; k++) {
    if (k > 0) out << ", ";
    printNode(nodes, children[k], out);
  }
  out << ")";
}
}  // namespace

std::string ExprGraph::toString() const {
  if (nodes_.empty()) return "<empty>";
  std::ostringstream out;
  const int32_t idx = (root_ >= 0 && root_ < static_cast<int32_t>(nodes_.size()))
                          ? root_
                          : static_cast<int32_t>(nodes_.size()) - 1;
  printNode(nodes_, idx, out);
  return out.str();
}

// ── sugar implementations ───────────────────────────────────────────────────

namespace {
ExprGraph* graphOf(Expr a) {
  if (!a.valid()) throw std::invalid_argument("kernelspec: invalid Expr handle");
  return a.graph();
}
Expr lift(ExprGraph* g, double v) { return g->c(v); }
}  // namespace

Expr operator+(Expr a, Expr b) { return graphOf(a)->binary(ExprOp::ADD, a, b); }
Expr operator-(Expr a, Expr b) { return graphOf(a)->binary(ExprOp::SUB, a, b); }
Expr operator*(Expr a, Expr b) { return graphOf(a)->binary(ExprOp::MUL, a, b); }
Expr operator/(Expr a, Expr b) { return graphOf(a)->binary(ExprOp::DIV, a, b); }
Expr operator-(Expr a) { return graphOf(a)->unary(ExprOp::NEG, a); }

Expr operator+(Expr a, double b) { return a + lift(graphOf(a), b); }
Expr operator+(double a, Expr b) { return lift(graphOf(b), a) + b; }
Expr operator-(Expr a, double b) { return a - lift(graphOf(a), b); }
Expr operator-(double a, Expr b) { return lift(graphOf(b), a) - b; }
Expr operator*(Expr a, double b) { return a * lift(graphOf(a), b); }
Expr operator*(double a, Expr b) { return lift(graphOf(b), a) * b; }
Expr operator/(Expr a, double b) { return a / lift(graphOf(a), b); }
Expr operator/(double a, Expr b) { return lift(graphOf(b), a) / b; }

Expr operator<(Expr a, Expr b) { return graphOf(a)->binary(ExprOp::CMP_LT, a, b); }
Expr operator<=(Expr a, Expr b) { return graphOf(a)->binary(ExprOp::CMP_LE, a, b); }
Expr operator>(Expr a, Expr b) { return graphOf(a)->binary(ExprOp::CMP_GT, a, b); }
Expr operator>=(Expr a, Expr b) { return graphOf(a)->binary(ExprOp::CMP_GE, a, b); }
Expr eq(Expr a, Expr b) { return graphOf(a)->binary(ExprOp::CMP_EQ, a, b); }
Expr ne(Expr a, Expr b) { return graphOf(a)->binary(ExprOp::CMP_NE, a, b); }

Expr operator<(Expr a, double b) { return a < lift(graphOf(a), b); }
Expr operator<=(Expr a, double b) { return a <= lift(graphOf(a), b); }
Expr operator>(Expr a, double b) { return a > lift(graphOf(a), b); }
Expr operator>=(Expr a, double b) { return a >= lift(graphOf(a), b); }

Expr exp(Expr x) { return graphOf(x)->unary(ExprOp::EXP, x); }
Expr log(Expr x) { return graphOf(x)->unary(ExprOp::LOG, x); }
Expr sqrt(Expr x) { return graphOf(x)->unary(ExprOp::SQRT, x); }
Expr tanh(Expr x) { return graphOf(x)->unary(ExprOp::TANH, x); }
Expr sin(Expr x) { return graphOf(x)->unary(ExprOp::SIN, x); }
Expr cos(Expr x) { return graphOf(x)->unary(ExprOp::COS, x); }
Expr erf(Expr x) { return graphOf(x)->unary(ExprOp::ERF, x); }
Expr abs(Expr x) { return graphOf(x)->unary(ExprOp::ABS, x); }
Expr floor(Expr x) { return graphOf(x)->unary(ExprOp::FLOOR, x); }
Expr ceil(Expr x) { return graphOf(x)->unary(ExprOp::CEIL, x); }
Expr round(Expr x) { return graphOf(x)->unary(ExprOp::ROUND, x); }
Expr pow(Expr base, Expr e) { return graphOf(base)->binary(ExprOp::POW, base, e); }
Expr pow(Expr base, double e) { return pow(base, lift(graphOf(base), e)); }
Expr min(Expr a, Expr b) { return graphOf(a)->binary(ExprOp::MIN, a, b); }
Expr max(Expr a, Expr b) { return graphOf(a)->binary(ExprOp::MAX, a, b); }
Expr min(Expr a, double b) { return min(a, lift(graphOf(a), b)); }
Expr max(Expr a, double b) { return max(a, lift(graphOf(a), b)); }

Expr logicalAnd(Expr a, Expr b) { return graphOf(a)->binary(ExprOp::AND, a, b); }
Expr logicalOr(Expr a, Expr b) { return graphOf(a)->binary(ExprOp::OR, a, b); }
Expr logicalNot(Expr a) { return graphOf(a)->unary(ExprOp::NOT, a); }
Expr select(Expr cond, Expr onTrue, Expr onFalse) {
  return graphOf(cond)->ternary(ExprOp::SELECT, cond, onTrue, onFalse);
}

Expr sigmoid(Expr x) {
  auto* g = graphOf(x);
  return g->c(1.0) / (g->c(1.0) + exp(-x));
}
Expr silu(Expr x) { return x * sigmoid(x); }
Expr relu(Expr x) { return max(x, 0.0); }
Expr softplus(Expr x) { return log(exp(x) + 1.0); }
Expr mish(Expr x) { return x * tanh(softplus(x)); }
Expr clamp(Expr x, Expr lo, Expr hi) { return min(max(x, lo), hi); }
Expr clamp(Expr x, double lo, double hi) {
  auto* g = graphOf(x);
  return clamp(x, g->c(lo), g->c(hi));
}
Expr hardSigmoid(Expr x) { return clamp(x * 0.2 + 0.5, 0.0, 1.0); }

}  // namespace kernelspec
}  // namespace sd
