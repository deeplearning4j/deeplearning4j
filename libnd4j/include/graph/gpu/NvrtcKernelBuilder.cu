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


#include <graph/gpu/NvrtcKernelBuilder.h>
#include <graph/gpu/OpCategoryTable.h>
#include <system/common.h>

#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace sd {
namespace graph {

// ── canJitSegment ──────────────────────────────────────────────────────────

bool canJitSegment(const NativeSlot* slots, int startSlot, int endSlot) {
  if (startSlot > endSlot) return false;

  bool hasNonSkippedSlot = false;

  for (int i = startSlot; i <= endSlot; i++) {
    const auto& slot = slots[i];

    // Skip frozen constants and identity ops — their outputs are pre-set
    if (slot.frozenConstantSlot() || slot.isIdentityOp()) continue;
    // Skip fused chain tails — head dispatches the entire chain
    if (slot.fusedChain.isFusedChainTail) continue;

    // Use the unified op category system
    auto cat = getOpCategoryFromName(slot.ident.opName);
    if (!isNvrtcJittable(cat)) {
      return false;
    }

    hasNonSkippedSlot = true;
  }

  // Need at least one actual op to JIT
  return hasNonSkippedSlot;
}

// ── Expression generators per category ────────────────────────────────────

// Check if this op is a scalar op (second operand from tArgs[0])
static bool isScalarOp(const std::string& opName) {
  return opName == "add_scalar" || opName == "subtract_scalar" ||
         opName == "sub_scalar" || opName == "multiply_scalar" ||
         opName == "mul_scalar" || opName == "divide_scalar" ||
         opName == "div_scalar" || opName == "rsub_scalar" ||
         opName == "rdiv_scalar";
}

static std::string generateBinaryExpr(const std::string& opName,
                                       const std::string& a, const std::string& b) {
  if (opName == "add" || opName == "Add") return a + " + " + b;
  if (opName == "subtract" || opName == "Sub") return a + " - " + b;
  if (opName == "multiply" || opName == "Mul") return a + " * " + b;
  if (opName == "divide" || opName == "Div" || opName == "RealDiv")
    return "(" + b + " != 0.0f ? " + a + " / " + b + " : 0.0f)";
  if (opName == "minimum" || opName == "Min" || opName == "min_pairwise" || opName == "MinPairwise")
    return "fminf(" + a + ", " + b + ")";
  if (opName == "maximum" || opName == "Max" || opName == "max_pairwise" || opName == "MaxPairwise")
    return "fmaxf(" + a + ", " + b + ")";
  if (opName == "mod" || opName == "Mod" || opName == "floormod" || opName == "FloorMod")
    return "fmodf(" + a + ", " + b + ")";
  if (opName == "atan2" || opName == "Atan2")
    return "atan2f(" + a + ", " + b + ")";
  if (opName == "floordiv" || opName == "FloorDiv")
    return "floorf(" + a + " / " + b + ")";
  if (opName == "reversedivide" || opName == "ReverseDivide")
    return "(" + a + " != 0.0f ? " + b + " / " + a + " : 0.0f)";
  if (opName == "reversesubtract" || opName == "ReverseSubtract")
    return b + " - " + a;
  if (opName == "squaredsubtract" || opName == "SquaredSubtract")
    return "((" + a + " - " + b + ") * (" + a + " - " + b + "))";
  if (opName == "multiply_no_nan" || opName == "MultiplyNoNan")
    return "(" + b + " != 0.0f ? " + a + " * " + b + " : 0.0f)";
  if (opName == "pow" || opName == "Pow")
    return "powf(" + a + ", " + b + ")";
  // SwiGLU: swish_mul(x, y) = x * sigmoid(x) * y
  if (opName == "swish_mul" || opName == "SwishMul")
    return "(" + a + " / (1.0f + __expf(-" + a + ")) * " + b + ")";
  // Fallback
  return a + " + " + b;
}

static std::string generateUnaryExpr(const std::string& opName, const std::string& val,
                                      const NativeSlot& slot) {
  if (opName == "relu" || opName == "Relu")
    return "(" + val + " > 0.0f ? " + val + " : 0.0f)";
  if (opName == "sigmoid" || opName == "Sigmoid")
    return "(1.0f / (1.0f + __expf(-" + val + ")))";
  if (opName == "tanh" || opName == "Tanh")
    return "tanhf(" + val + ")";
  if (opName == "gelu" || opName == "Gelu")
    return "(0.5f * " + val + " * (1.0f + tanhf(0.7978845608f * ("
           + val + " + 0.044715f * " + val + " * " + val + " * " + val + "))))";
  if (opName == "exp" || opName == "Exp")
    return "__expf(" + val + ")";
  if (opName == "log" || opName == "Log")
    return "(" + val + " > 0.0f ? __logf(" + val + ") : -1e38f)";
  if (opName == "abs" || opName == "Abs")
    return "fabsf(" + val + ")";
  if (opName == "neg" || opName == "Neg")
    return "(-" + val + ")";
  if (opName == "square" || opName == "Square")
    return "(" + val + " * " + val + ")";
  if (opName == "sqrt" || opName == "Sqrt")
    return "(" + val + " >= 0.0f ? sqrtf(" + val + ") : 0.0f)";
  if (opName == "swish" || opName == "Swish" || opName == "silu" || opName == "Silu")
    return "(" + val + " / (1.0f + __expf(-" + val + ")))";
  if (opName == "mish" || opName == "Mish")
    return "(" + val + " * tanhf(__logf(1.0f + __expf(" + val + "))))";
  if (opName == "rsqrt" || opName == "Rsqrt")
    return "(1.0f / sqrtf(" + val + "))";
  if (opName == "reciprocal" || opName == "Reciprocal")
    return "(1.0f / " + val + ")";
  if (opName == "sign" || opName == "Sign")
    return "(" + val + " > 0.0f ? 1.0f : (" + val + " < 0.0f ? -1.0f : 0.0f))";
  if (opName == "erf" || opName == "Erf")
    return "erff(" + val + ")";
  if (opName == "erfc" || opName == "Erfc")
    return "erfcf(" + val + ")";
  if (opName == "log1p" || opName == "Log1p")
    return "log1pf(" + val + ")";
  if (opName == "ceil" || opName == "Ceil")
    return "ceilf(" + val + ")";
  if (opName == "floor" || opName == "Floor")
    return "floorf(" + val + ")";
  if (opName == "round" || opName == "Round")
    return "rintf(" + val + ")";
  if (opName == "sin" || opName == "Sin")
    return "__sinf(" + val + ")";
  if (opName == "cos" || opName == "Cos")
    return "__cosf(" + val + ")";
  if (opName == "elu" || opName == "Elu")
    return "(" + val + " >= 0.0f ? " + val + " : (__expf(" + val + ") - 1.0f))";
  if (opName == "selu" || opName == "Selu")
    return "(" + val + " >= 0.0f ? 1.0507009873554805f * " + val
           + " : 1.0507009873554805f * 1.6732632423543772f * (__expf(" + val + ") - 1.0f))";
  if (opName == "softplus" || opName == "Softplus")
    return "(fmaxf(0.0f, " + val + ") + __logf(1.0f + __expf(-fabsf(" + val + "))))";
  if (opName == "softsign" || opName == "Softsign")
    return "(" + val + " / (1.0f + fabsf(" + val + ")))";
  if (opName == "hard_sigmoid" || opName == "HardSigmoid")
    return "fminf(1.0f, fmaxf(0.0f, 0.2f * " + val + " + 0.5f))";
  if (opName == "hardtanh" || opName == "HardTanh")
    return "fminf(1.0f, fmaxf(-1.0f, " + val + "))";
  if (opName == "relu6" || opName == "Relu6")
    return "fminf(6.0f, fmaxf(0.0f, " + val + "))";
  if (opName == "clipbyvalue" || opName == "ClipByValue" || opName == "clip_by_value" || opName == "clamp")
    return "fminf(fmaxf(" + val + ", clipMin), clipMax)";
  if (opName == "leakyrelu" || opName == "LeakyRelu") {
    // Alpha from tArgs[0], default 0.01
    std::string alpha = "0.01f";
    if (slot.args.numTArgs > 0) {
      alpha = std::to_string(static_cast<float>(slot.args.tArgs[0])) + "f";
    }
    return "(" + val + " >= 0.0f ? " + val + " : " + val + " * " + alpha + ")";
  }
  // Scalar ops: second operand from tArgs[0]
  if (opName == "add_scalar") {
    std::string scalar = std::to_string(static_cast<float>(slot.args.tArgs[0])) + "f";
    return val + " + " + scalar;
  }
  if (opName == "subtract_scalar" || opName == "sub_scalar") {
    std::string scalar = std::to_string(static_cast<float>(slot.args.tArgs[0])) + "f";
    return val + " - " + scalar;
  }
  if (opName == "multiply_scalar" || opName == "mul_scalar") {
    std::string scalar = std::to_string(static_cast<float>(slot.args.tArgs[0])) + "f";
    return val + " * " + scalar;
  }
  if (opName == "divide_scalar" || opName == "div_scalar") {
    std::string scalar = std::to_string(static_cast<float>(slot.args.tArgs[0])) + "f";
    return "(" + val + " / " + scalar + ")";
  }
  if (opName == "rsub_scalar") {
    std::string scalar = std::to_string(static_cast<float>(slot.args.tArgs[0])) + "f";
    return scalar + " - " + val;
  }
  if (opName == "rdiv_scalar") {
    std::string scalar = std::to_string(static_cast<float>(slot.args.tArgs[0])) + "f";
    return "(" + scalar + " / " + val + ")";
  }
  // Fallback
  return val;
}

static std::string generateComparisonExpr(const std::string& opName,
                                           const std::string& a, const std::string& b) {
  if (opName == "greater" || opName == "Greater")
    return "(" + a + " > " + b + " ? 1.0f : 0.0f)";
  if (opName == "greater_equal" || opName == "GreaterEqual")
    return "(" + a + " >= " + b + " ? 1.0f : 0.0f)";
  if (opName == "less" || opName == "Less")
    return "(" + a + " < " + b + " ? 1.0f : 0.0f)";
  if (opName == "less_equal" || opName == "LessEqual")
    return "(" + a + " <= " + b + " ? 1.0f : 0.0f)";
  if (opName == "equals" || opName == "Equals")
    return "(" + a + " == " + b + " ? 1.0f : 0.0f)";
  if (opName == "not_equals" || opName == "NotEquals")
    return "(" + a + " != " + b + " ? 1.0f : 0.0f)";
  // Fallback
  return "(" + a + " > " + b + " ? 1.0f : 0.0f)";
}

static std::string generateLogicalExpr(const std::string& opName,
                                        const std::string& a, const std::string& b) {
  if (opName == "boolean_and" || opName == "BooleanAnd" ||
      opName == "logical_and" || opName == "LogicalAnd")
    return "((" + a + " != 0.0f && " + b + " != 0.0f) ? 1.0f : 0.0f)";
  if (opName == "boolean_or" || opName == "BooleanOr" ||
      opName == "logical_or" || opName == "LogicalOr")
    return "((" + a + " != 0.0f || " + b + " != 0.0f) ? 1.0f : 0.0f)";
  if (opName == "boolean_not" || opName == "BooleanNot" ||
      opName == "logical_not" || opName == "LogicalNot")
    return "(" + a + " == 0.0f ? 1.0f : 0.0f)";
  if (opName == "boolean_xor" || opName == "BooleanXor")
    return "(((" + a + " != 0.0f) != (" + b + " != 0.0f)) ? 1.0f : 0.0f)";
  // Fallback
  return "((" + a + " != 0.0f && " + b + " != 0.0f) ? 1.0f : 0.0f)";
}

static std::string generateTernaryExpr(const std::string& opName,
                                        const std::string& cond,
                                        const std::string& trueVal,
                                        const std::string& falseVal) {
  // where/select: cond != 0 ? trueVal : falseVal
  return "(" + cond + " != 0.0f ? " + trueVal + " : " + falseVal + ")";
}

// ── Category-based expression dispatch ────────────────────────────────────

static std::string generateOpExpression(TritonOpCategory cat, const std::string& opName,
                                         const std::string& primary,
                                         const std::string& secondary,
                                         const std::string& tertiary,
                                         const NativeSlot& slot) {
  switch (cat) {
    case TritonOpCategory::BINARY_ELEMENTWISE:
      return generateBinaryExpr(opName, primary, secondary);
    case TritonOpCategory::UNARY_ELEMENTWISE:
      return generateUnaryExpr(opName, primary, slot);
    case TritonOpCategory::COMPARISON:
      return generateComparisonExpr(opName, primary, secondary);
    case TritonOpCategory::LOGICAL:
      return generateLogicalExpr(opName, primary, secondary);
    case TritonOpCategory::TERNARY:
      return generateTernaryExpr(opName, primary, secondary, tertiary);
    case TritonOpCategory::CAST:
      return primary;  // Pass through — kernel operates in float
    case TritonOpCategory::IDENTITY:
      // assign(x, y) -> y (copies second input to output), unlike identity which forwards input[0]
      if (opName == "assign" || opName == "Assign") return secondary.empty() ? primary : secondary;
      return primary;  // SSA forwarding
    default:
      return primary;
  }
}

// ── Determine input count for a specific op ───────────────────────────────

static int getInputCount(TritonOpCategory cat, const std::string& opName) {
  // Logical NOT is unary despite LOGICAL category being nominally binary
  if (cat == TritonOpCategory::LOGICAL &&
      (opName == "boolean_not" || opName == "BooleanNot" ||
       opName == "logical_not" || opName == "LogicalNot")) {
    return 1;
  }
  // assign(x, y) -> y is IDENTITY category but has 2 inputs
  if (cat == TritonOpCategory::IDENTITY &&
      (opName == "assign" || opName == "Assign")) {
    return 2;
  }
  return categoryInputCount(cat);
}

// ── buildKernelSource ──────────────────────────────────────────────────────

JitKernelSource buildKernelSource(const NativeSlot* slots, int startSlot, int endSlot,
                                  NDArray** outputSlots, int totalOutputSlots,
                                  int segmentIndex) {
  JitKernelSource result;

  // Collect active (non-skipped) slots in order
  struct ActiveSlot {
    int slotIdx;                   // global slot index
    TritonOpCategory category;     // op category
    std::string opName;            // for expression dispatch
    int inputCount;                // 1, 2, or 3
    int primaryInputSource;        // inputSourceIndices[0]
    int secondaryInputSource;      // inputSourceIndices[1] for binary/comparison/logical (-1 if unused)
    int tertiaryInputSource;       // inputSourceIndices[2] for ternary (-1 if unused)
    int outputSlotIdx;             // outputSlotIndices[0]
    int slotArrayIdx;              // index into slots[] for tArgs access
  };
  std::vector<ActiveSlot> activeSlots;

  for (int i = startSlot; i <= endSlot; i++) {
    const auto& slot = slots[i];
    if (slot.frozenConstantSlot() || slot.isIdentityOp() || slot.fusedChain.isFusedChainTail) continue;

    auto cat = getOpCategoryFromName(slot.ident.opName);
    if (!isNvrtcJittable(cat)) {
      result.valid = false;
      return result;
    }

    ActiveSlot as;
    as.slotIdx = i;
    as.slotArrayIdx = i;
    as.category = cat;
    as.opName = slot.ident.opName;
    as.inputCount = getInputCount(cat, slot.ident.opName);
    as.primaryInputSource = (slot.wiring.numInputs > 0) ? slot.wiring.inputSourceIndices[0] : -1;
    as.secondaryInputSource = (slot.wiring.numInputs > 1 && as.inputCount >= 2) ? slot.wiring.inputSourceIndices[1] : -1;
    as.tertiaryInputSource = (slot.wiring.numInputs > 2 && as.inputCount >= 3) ? slot.wiring.inputSourceIndices[2] : -1;
    as.outputSlotIdx = (slot.wiring.numOutputs > 0) ? slot.wiring.outputSlotIndices[0] : -1;
    activeSlots.push_back(as);
  }

  if (activeSlots.empty()) {
    result.valid = false;
    return result;
  }

  // ── Analyze SSA graph: count consumers for each output slot ──
  std::unordered_map<int, int> outputConsumerCount;
  for (const auto& as : activeSlots) {
    if (as.primaryInputSource >= 0) outputConsumerCount[as.primaryInputSource]++;
    if (as.secondaryInputSource >= 0) outputConsumerCount[as.secondaryInputSource]++;
    if (as.tertiaryInputSource >= 0) outputConsumerCount[as.tertiaryInputSource]++;
  }

  // Determine which output slots need global memory pointers
  std::unordered_set<int> needsGlobalPtr;
  if (!activeSlots.empty()) {
    int lastOut = activeSlots.back().outputSlotIdx;
    if (lastOut >= 0) needsGlobalPtr.insert(lastOut);
  }
  for (const auto& as : activeSlots) {
    int outIdx = as.outputSlotIdx;
    if (outIdx >= 0 && outputConsumerCount.count(outIdx) && outputConsumerCount[outIdx] > 1) {
      needsGlobalPtr.insert(outIdx);
    }
  }

  // ── Collect all external input sources (negative inputSourceIndices) ──
  std::unordered_map<int, int> externalInputMap;
  std::vector<int> externalInputList;
  auto checkExt = [&](int src) {
    if (src < 0) {
      int extIdx = -(src + 1);
      if (externalInputMap.find(extIdx) == externalInputMap.end()) {
        externalInputMap[extIdx] = static_cast<int>(externalInputList.size());
        externalInputList.push_back(extIdx);
      }
    }
  };
  for (const auto& as : activeSlots) {
    checkExt(as.primaryInputSource);
    if (as.inputCount >= 2) checkExt(as.secondaryInputSource);
    if (as.inputCount >= 3) checkExt(as.tertiaryInputSource);
  }

  // ── Collect cross-segment input sources (>= 0, from prior segments) ──
  std::unordered_map<int, int> crossSegMap;
  std::vector<int> crossSegList;
  auto checkCross = [&](int src, int consumerSlotIdx) {
    if (src >= 0) {
      bool internal = false;
      for (const auto& other : activeSlots) {
        if (other.outputSlotIdx == src && other.slotIdx < consumerSlotIdx) {
          internal = true;
          break;
        }
      }
      if (!internal) {
        if (crossSegMap.find(src) == crossSegMap.end()) {
          crossSegMap[src] = static_cast<int>(crossSegList.size());
          crossSegList.push_back(src);
        }
      }
    }
  };
  for (const auto& as : activeSlots) {
    checkCross(as.primaryInputSource, as.slotIdx);
    if (as.inputCount >= 2) checkCross(as.secondaryInputSource, as.slotIdx);
    if (as.inputCount >= 3) checkCross(as.tertiaryInputSource, as.slotIdx);
  }

  // ── Collect output slots that need global memory store ──
  std::vector<int> outputPtrList;
  std::unordered_map<int, int> outputPtrMap;
  for (int outIdx : needsGlobalPtr) {
    outputPtrMap[outIdx] = static_cast<int>(outputPtrList.size());
    outputPtrList.push_back(outIdx);
  }

  // ── Check for ops that need extra params ──
  bool hasClip = false;
  for (const auto& as : activeSlots) {
    if (as.opName == "clipbyvalue" || as.opName == "ClipByValue" ||
        as.opName == "clip_by_value" || as.opName == "clamp") {
      hasClip = true;
      break;
    }
  }

  // ── Generate kernel source ──
  std::ostringstream src;
  std::string kernelName = "jit_seg_" + std::to_string(segmentIndex) + "_"
                           + std::to_string(startSlot) + "_" + std::to_string(endSlot);

  // Build parameter list
  src << "extern \"C\" __global__ void " << kernelName << "(\n";
  src << "    long long n";

  // External input pointers
  for (size_t i = 0; i < externalInputList.size(); i++) {
    src << ",\n    const float* __restrict__ ext" << i;
  }
  // Cross-segment input pointers
  for (size_t i = 0; i < crossSegList.size(); i++) {
    src << ",\n    const float* __restrict__ cross" << i;
  }
  // Output pointers
  for (size_t i = 0; i < outputPtrList.size(); i++) {
    src << ",\n    float* __restrict__ out" << i;
  }
  // Clip params
  if (hasClip) {
    src << ",\n    float clipMin, float clipMax";
  }
  src << ") {\n";

  // Thread index
  src << "  long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;\n";
  src << "  if (idx >= n) return;\n\n";

  // ── Helper to resolve an input source to an expression ──
  auto resolveInput = [&](int inputSrc) -> std::string {
    if (inputSrc < 0) {
      int extIdx = -(inputSrc + 1);
      int paramIdx = externalInputMap[extIdx];
      return "ext" + std::to_string(paramIdx) + "[idx]";
    }
    // Check internal SSA variable
    // (slotToVar is populated below as we generate code)
    return "";  // placeholder — resolved inline below
  };

  // Track which output slot index maps to which register variable name
  std::unordered_map<int, std::string> slotToVar;

  for (size_t ai = 0; ai < activeSlots.size(); ai++) {
    const auto& as = activeSlots[ai];
    std::string varName = "v" + std::to_string(ai);

    // Resolve primary input
    auto resolveSource = [&](int inputSrc) -> std::string {
      if (inputSrc < 0) {
        int extIdx = -(inputSrc + 1);
        int paramIdx = externalInputMap[extIdx];
        return "ext" + std::to_string(paramIdx) + "[idx]";
      } else if (slotToVar.count(inputSrc)) {
        return slotToVar[inputSrc];
      } else if (crossSegMap.count(inputSrc)) {
        int paramIdx = crossSegMap[inputSrc];
        return "cross" + std::to_string(paramIdx) + "[idx]";
      }
      return "0.0f";
    };

    std::string primaryExpr = resolveSource(as.primaryInputSource);
    std::string secondaryExpr = (as.inputCount >= 2 && as.secondaryInputSource != -1)
                                    ? resolveSource(as.secondaryInputSource) : "0.0f";
    std::string tertiaryExpr = (as.inputCount >= 3 && as.tertiaryInputSource != -1)
                                   ? resolveSource(as.tertiaryInputSource) : "0.0f";

    // Generate the computation expression
    std::string opExpr = generateOpExpression(as.category, as.opName,
                                              primaryExpr, secondaryExpr, tertiaryExpr,
                                              slots[as.slotArrayIdx]);

    src << "  float " << varName << " = " << opExpr << ";\n";

    // Track the variable for downstream slots
    if (as.outputSlotIdx >= 0) {
      slotToVar[as.outputSlotIdx] = varName;

      // Store to global memory if needed
      if (needsGlobalPtr.count(as.outputSlotIdx)) {
        int outParamIdx = outputPtrMap[as.outputSlotIdx];
        src << "  out" << outParamIdx << "[idx] = " << varName << ";\n";
      }
    }
  }

  src << "}\n";

  // ── Build param bindings ──
  result.paramBindings.clear();

  // First param: length (n)
  {
    JitParamBinding b;
    b.type = JitParamBinding::LENGTH;
    result.paramBindings.push_back(b);
  }

  // External input pointers
  for (int extIdx : externalInputList) {
    JitParamBinding b;
    b.type = JitParamBinding::INPUT_PTR;
    b.externalInputIdx = extIdx;
    result.paramBindings.push_back(b);
  }

  // Cross-segment input pointers
  for (int slotIdx : crossSegList) {
    JitParamBinding b;
    b.type = JitParamBinding::CROSS_SEG_PTR;
    b.slotIdx = slotIdx;
    result.paramBindings.push_back(b);
  }

  // Output pointers
  for (int slotIdx : outputPtrList) {
    JitParamBinding b;
    b.type = JitParamBinding::OUTPUT_PTR;
    b.slotIdx = slotIdx;
    result.paramBindings.push_back(b);
  }

  result.sourceCode = src.str();
  result.kernelName = kernelName;
  result.valid = true;
  return result;
}

}  // namespace graph
}  // namespace sd
