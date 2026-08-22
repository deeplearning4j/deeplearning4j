/* ******************************************************************************
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#ifdef SD_TPU

#include <graph/tpu/StableHloKernelExprEmitter.h>

#include <cmath>
#include <iomanip>
#include <limits>

namespace sd {
namespace graph {
namespace {

std::string valueName(int& nextValueId) {
  return "%v" + std::to_string(nextValueId++);
}

std::string floatLiteral(double value) {
  if (std::isinf(value)) return value < 0 ? "-0x7FF0000000000000" : "0x7FF0000000000000";
  std::ostringstream stream;
  stream << std::setprecision(17) << value;
  return stream.str();
}

const char* unaryOpcode(kernelspec::ExprOp op) {
  using kernelspec::ExprOp;
  switch (op) {
    case ExprOp::NEG: return "negate";
    case ExprOp::ABS: return "abs";
    case ExprOp::EXP: return "exponential";
    case ExprOp::LOG: return "log";
    case ExprOp::SQRT: return "sqrt";
    case ExprOp::TANH: return "tanh";
    case ExprOp::SIN: return "sine";
    case ExprOp::COS: return "cosine";
    case ExprOp::FLOOR: return "floor";
    case ExprOp::CEIL: return "ceil";
    case ExprOp::ROUND: return "round_nearest_afz";
    case ExprOp::NOT: return "not";
    default: return nullptr;
  }
}

const char* binaryOpcode(kernelspec::ExprOp op) {
  using kernelspec::ExprOp;
  switch (op) {
    case ExprOp::ADD: return "add";
    case ExprOp::SUB: return "subtract";
    case ExprOp::MUL: return "multiply";
    case ExprOp::DIV: return "divide";
    case ExprOp::POW: return "power";
    case ExprOp::MIN: return "minimum";
    case ExprOp::MAX: return "maximum";
    case ExprOp::AND: return "and";
    case ExprOp::OR: return "or";
    default: return nullptr;
  }
}

const char* compareDirection(kernelspec::ExprOp op) {
  using kernelspec::ExprOp;
  switch (op) {
    case ExprOp::CMP_LT: return "LT";
    case ExprOp::CMP_LE: return "LE";
    case ExprOp::CMP_GT: return "GT";
    case ExprOp::CMP_GE: return "GE";
    case ExprOp::CMP_EQ: return "EQ";
    case ExprOp::CMP_NE: return "NE";
    default: return nullptr;
  }
}

}  // namespace

StableHloExprResult StableHloKernelExprEmitter::emit(
    const kernelspec::ExprGraph& expression,
    const std::vector<std::string>& inputs,
    const std::vector<double>& scalarValues,
    const std::string& tensorType,
    const std::string& booleanTensorType,
    int& nextValueId,
    std::ostringstream& body) {
  StableHloExprResult result;
  const std::string validation = expression.validate();
  if (!validation.empty()) {
    result.error = "invalid KernelExpr: " + validation;
    return result;
  }
  if (static_cast<int>(inputs.size()) < expression.inputArity() ||
      static_cast<int>(scalarValues.size()) < expression.scalarArity()) {
    result.error = "KernelExpr invocation arity mismatch";
    return result;
  }

  struct EmittedValue {
    std::string name;
    bool booleanValue = false;
  };
  std::vector<EmittedValue> values(expression.nodes().size());

  for (size_t index = 0; index < expression.nodes().size(); ++index) {
    const auto& node = expression.nodes()[index];
    auto child = [&](int32_t childIndex) -> const EmittedValue& {
      return values[static_cast<size_t>(childIndex)];
    };

    if (node.op == kernelspec::ExprOp::INPUT) {
      values[index] = {inputs[static_cast<size_t>(node.index)], false};
      continue;
    }
    if (node.op == kernelspec::ExprOp::SCALAR_PARAM ||
        node.op == kernelspec::ExprOp::CONST_F) {
      const double literal = node.op == kernelspec::ExprOp::SCALAR_PARAM
                                 ? scalarValues[static_cast<size_t>(node.index)]
                                 : node.f;
      const std::string output = valueName(nextValueId);
      body << "    " << output << " = stablehlo.constant dense<"
           << floatLiteral(literal) << "> : " << tensorType << "\n";
      values[index] = {output, false};
      continue;
    }

    if (const char* opcode = unaryOpcode(node.op)) {
      const EmittedValue& operand = child(node.a);
      const bool booleanResult = node.op == kernelspec::ExprOp::NOT;
      if (booleanResult != operand.booleanValue) {
        result.error = "KernelExpr unary boolean/numeric type mismatch";
        return result;
      }
      const std::string output = valueName(nextValueId);
      body << "    " << output << " = stablehlo." << opcode << " "
           << operand.name << " : "
           << (booleanResult ? booleanTensorType : tensorType) << "\n";
      values[index] = {output, booleanResult};
      continue;
    }
    if (node.op == kernelspec::ExprOp::ERF) {
      result.error = "StableHLO has no portable erf primitive";
      return result;
    }

    if (const char* direction = compareDirection(node.op)) {
      const EmittedValue& left = child(node.a);
      const EmittedValue& right = child(node.b);
      if (left.booleanValue || right.booleanValue) {
        result.error = "KernelExpr comparison operands must be numeric";
        return result;
      }
      const std::string output = valueName(nextValueId);
      body << "    " << output << " = \"stablehlo.compare\"("
           << left.name << ", " << right.name
           << ") <{compare_type = #stablehlo<comparison_type FLOAT>, "
              "comparison_direction = #stablehlo<comparison_direction "
           << direction << ">}> : (" << tensorType << ", " << tensorType
           << ") -> " << booleanTensorType << "\n";
      values[index] = {output, true};
      continue;
    }

    if (const char* opcode = binaryOpcode(node.op)) {
      const EmittedValue& left = child(node.a);
      const EmittedValue& right = child(node.b);
      const bool booleanResult = node.op == kernelspec::ExprOp::AND ||
                                 node.op == kernelspec::ExprOp::OR;
      if (left.booleanValue != booleanResult || right.booleanValue != booleanResult) {
        result.error = "KernelExpr binary boolean/numeric type mismatch";
        return result;
      }
      const std::string output = valueName(nextValueId);
      body << "    " << output << " = stablehlo." << opcode << " "
           << left.name << ", " << right.name << " : "
           << (booleanResult ? booleanTensorType : tensorType) << "\n";
      values[index] = {output, booleanResult};
      continue;
    }

    if (node.op == kernelspec::ExprOp::SELECT) {
      const EmittedValue& condition = child(node.a);
      const EmittedValue& onTrue = child(node.b);
      const EmittedValue& onFalse = child(node.c);
      if (!condition.booleanValue || onTrue.booleanValue != onFalse.booleanValue) {
        result.error = "KernelExpr select type mismatch";
        return result;
      }
      const std::string output = valueName(nextValueId);
      const std::string& resultType = onTrue.booleanValue ? booleanTensorType : tensorType;
      body << "    " << output << " = stablehlo.select " << condition.name
           << ", " << onTrue.name << ", " << onFalse.name << " : "
           << resultType << "\n";
      values[index] = {output, onTrue.booleanValue};
      continue;
    }

    result.error = std::string("unsupported KernelExpr primitive ") +
                   kernelspec::exprOpName(node.op);
    return result;
  }

  const auto& root = values[static_cast<size_t>(expression.rootIndex())];
  result.success = true;
  result.value = root.name;
  result.booleanValue = root.booleanValue;
  return result;
}

}  // namespace graph
}  // namespace sd

#endif  // SD_TPU
