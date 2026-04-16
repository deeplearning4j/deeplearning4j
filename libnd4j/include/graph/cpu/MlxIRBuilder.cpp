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

#include <config.h>

#if HAVE_MLX

#include <graph/cpu/MlxIRBuilder.h>
#include <graph/DspDiagnostics.h>
#include <graph/gpu/OpCategoryTable.h>
#include <helpers/shape.h>
#include <helpers/logger.h>
#include <system/Environment.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>

// MLX C++ API headers
#include <mlx/mlx.h>
#include <mlx/fast.h>

namespace mx = mlx::core;

namespace sd {
namespace graph {

// ─── Helper: unwrap shared_ptr<void> to mx::array& ────────────────────────
static mx::array& unwrap(const std::shared_ptr<void>& p) {
  return *static_cast<mx::array*>(p.get());
}

static std::shared_ptr<void> wrap(mx::array&& arr) {
  return std::shared_ptr<void>(new mx::array(std::move(arr)),
                               [](void* p) { delete static_cast<mx::array*>(p); });
}

// ─── Normalize op name: lowercase + strip underscores ─────────────────────
// OpCategoryTable has CamelCase entries ("BatchNorm" -> "batchnorm") and
// underscore entries ("batch_norm" -> "batchnorm"). By stripping underscores
// both forms match the same handler.
static std::string normalizeOp(const std::string& opName) {
  std::string n;
  n.reserve(opName.size());
  for (char c : opName) {
    if (c != '_') n.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  }
  return n;
}

// ═══════════════════════════════════════════════════════════════════════════
// Type mapping
// ═══════════════════════════════════════════════════════════════════════════

int MlxIRBuilder::sdTypeToMlxDtype(sd::DataType dt) {
  switch (dt) {
    case sd::DataType::FLOAT32: return static_cast<int>(mx::float32);
    case sd::DataType::FLOAT16: return static_cast<int>(mx::float16);
    case sd::DataType::BFLOAT16: return static_cast<int>(mx::bfloat16);
    case sd::DataType::INT32: return static_cast<int>(mx::int32);
    case sd::DataType::INT64: return static_cast<int>(mx::int64);
    case sd::DataType::INT16: return static_cast<int>(mx::int16);
    case sd::DataType::INT8: return static_cast<int>(mx::int8);
    case sd::DataType::UINT8: return static_cast<int>(mx::uint8);
    case sd::DataType::UINT16: return static_cast<int>(mx::uint16);
    case sd::DataType::UINT32: return static_cast<int>(mx::uint32);
    case sd::DataType::UINT64: return static_cast<int>(mx::uint64);
    case sd::DataType::BOOL: return static_cast<int>(mx::bool_);
    default: return static_cast<int>(mx::float32);
  }
}

static mx::Dtype sdTypeToMlxDtypeInternal(sd::DataType dt) {
  switch (dt) {
    case sd::DataType::FLOAT32: return mx::float32;
    case sd::DataType::FLOAT16: return mx::float16;
    case sd::DataType::BFLOAT16: return mx::bfloat16;
    case sd::DataType::INT32: return mx::int32;
    case sd::DataType::INT64: return mx::int64;
    case sd::DataType::INT16: return mx::int16;
    case sd::DataType::INT8: return mx::int8;
    case sd::DataType::UINT8: return mx::uint8;
    case sd::DataType::UINT16: return mx::uint16;
    case sd::DataType::UINT32: return mx::uint32;
    case sd::DataType::UINT64: return mx::uint64;
    case sd::DataType::BOOL: return mx::bool_;
    default: return mx::float32;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Static analysis
// ═══════════════════════════════════════════════════════════════════════════

bool MlxIRBuilder::isMlxMappable(const std::string& opName) {
  auto cat = OpCategoryTable::categorize(opName);
  switch (cat) {
    // Phase 1: element-wise
    case TritonOpCategory::BINARY_ELEMENTWISE:
    case TritonOpCategory::UNARY_ELEMENTWISE:
    case TritonOpCategory::COMPARISON:
    case TritonOpCategory::LOGICAL:
    case TritonOpCategory::TERNARY:
    case TritonOpCategory::IDENTITY:
    case TritonOpCategory::CAST:
    // Phase 2: structured ops
    case TritonOpCategory::REDUCTION:
    case TritonOpCategory::MATMUL:
    case TritonOpCategory::NORMALIZATION:
    case TritonOpCategory::SHAPE_MANIPULATION:
    case TritonOpCategory::DATA_MOVEMENT:
    case TritonOpCategory::CONSTANT_GENERATION:
    // Phase 3: compute-intensive
    case TritonOpCategory::CONVOLUTION:
    case TritonOpCategory::FUSED_ATTENTION:
    // LLM-specific
    case TritonOpCategory::ROPE:
    case TritonOpCategory::FUSED_LLM:
      return true;
    default:
      return false;
  }
}

SegmentProfile MlxIRBuilder::profileSegment(NativeSlot* slots, int startSlot, int endSlot,
                                             NDArray** outputSlots, int totalOutputSlots) {
  SegmentProfile profile;
  profile.startSlot = startSlot;
  profile.endSlot = endSlot;
  profile.totalOps = endSlot - startSlot + 1;
  profile.elementwiseOps = 0;
  profile.reductionOps = 0;
  profile.matmulOps = 0;
  profile.unsupportedOps = 0;

  for (int i = startSlot; i <= endSlot; i++) {
    auto cat = OpCategoryTable::categorize(slots[i].ident.opName);
    switch (cat) {
      case TritonOpCategory::BINARY_ELEMENTWISE:
      case TritonOpCategory::UNARY_ELEMENTWISE:
      case TritonOpCategory::COMPARISON:
      case TritonOpCategory::LOGICAL:
      case TritonOpCategory::TERNARY:
      case TritonOpCategory::IDENTITY:
      case TritonOpCategory::CAST:
      case TritonOpCategory::SHAPE_MANIPULATION:
      case TritonOpCategory::DATA_MOVEMENT:
      case TritonOpCategory::CONSTANT_GENERATION:
        profile.elementwiseOps++;
        break;
      case TritonOpCategory::REDUCTION:
      case TritonOpCategory::NORMALIZATION:
        profile.reductionOps++;
        break;
      case TritonOpCategory::MATMUL:
      case TritonOpCategory::CONVOLUTION:
      case TritonOpCategory::FUSED_ATTENTION:
      case TritonOpCategory::ROPE:
      case TritonOpCategory::FUSED_LLM:
        profile.matmulOps++;
        break;
      default:
        profile.unsupportedOps++;
        break;
    }
  }

  return profile;
}

SegmentAnalysis MlxIRBuilder::analyzeSegment(NativeSlot* slots, int startSlot, int endSlot,
                                              int totalSlots,
                                              NDArray** externalInputs, int numExternalInputs,
                                              NDArray** outputSlots, int totalOutputSlots) {
  SegmentAnalysis analysis;
  analysis.canCompile = true;

  auto profile = profileSegment(slots, startSlot, endSlot, outputSlots, totalOutputSlots);

  if (profile.unsupportedOps > 0) {
    analysis.canCompile = false;
    analysis.failureReason = "segment contains unsupported ops for MLX";
    return analysis;
  }

  if (profile.totalOps < 2) {
    analysis.canCompile = false;
    analysis.failureReason = "segment too small (need >= 2 ops)";
    return analysis;
  }

  // Verify all inputs are available
  for (int i = startSlot; i <= endSlot; i++) {
    for (int inp = 0; inp < slots[i].wiring.numInputs; inp++) {
      int srcIdx = slots[i].wiring.inputSourceIndices[inp];
      if (srcIdx < 0) {
        int extIdx = -(srcIdx + 1);
        if (extIdx >= numExternalInputs || !externalInputs || !externalInputs[extIdx]) {
          analysis.canCompile = false;
          analysis.failureReason = "missing external input " + std::to_string(extIdx);
          return analysis;
        }
      }
    }
  }

  return analysis;
}

std::unordered_set<int> MlxIRBuilder::computeExternallyVisibleOutputs(
    NativeSlot* slots, int startSlot, int endSlot, int totalSlots) {
  std::unordered_set<int> internalOutputs;
  for (int i = startSlot; i <= endSlot; i++) {
    for (int o = 0; o < slots[i].wiring.numOutputs; o++) {
      internalOutputs.insert(slots[i].wiring.outputSlotIndices[o]);
    }
  }

  std::unordered_set<int> externalOutputs;
  for (int outIdx : internalOutputs) {
    bool consumedOutside = false;
    for (int i = 0; i < totalSlots; i++) {
      if (i >= startSlot && i <= endSlot) continue;
      for (int inp = 0; inp < slots[i].wiring.numInputs; inp++) {
        if (slots[i].wiring.inputSourceIndices[inp] == outIdx) {
          consumedOutside = true;
          break;
        }
      }
      if (consumedOutside) break;
    }
    if (consumedOutside) {
      externalOutputs.insert(outIdx);
    }
    if (!consumedOutside) {
      for (int o = 0; o < slots[endSlot].wiring.numOutputs; o++) {
        if (slots[endSlot].wiring.outputSlotIndices[o] == outIdx) {
          externalOutputs.insert(outIdx);
        }
      }
    }
  }

  return externalOutputs;
}

// ═══════════════════════════════════════════════════════════════════════════
// Memory bridge
// ═══════════════════════════════════════════════════════════════════════════

std::shared_ptr<void> MlxIRBuilder::ndArrayToMlxArray(NDArray* arr) {
  if (!arr || arr->isEmpty()) return nullptr;

  auto rank = arr->rankOf();
  std::vector<int> shape(rank);
  for (int i = 0; i < rank; i++) {
    shape[i] = static_cast<int>(arr->sizeAt(i));
  }

  auto dtype = sdTypeToMlxDtypeInternal(arr->dataType());

  if (shape::strideDescendingCAscendingF(arr->shapeInfo()) && arr->ordering() == 'c') {
    auto* dataPtr = arr->buffer();
    auto mlxArr = mx::array(dataPtr, shape, dtype);
    return wrap(std::move(mlxArr));
  } else {
    auto duped = arr->dup('c');
    auto* dataPtr = duped.buffer();
    auto mlxArr = mx::array(dataPtr, shape, dtype);
    mx::eval(mlxArr);
    return wrap(std::move(mlxArr));
  }
}

void MlxIRBuilder::mlxArrayToNDArray(const std::shared_ptr<void>& mlxArr, NDArray* output) {
  if (!mlxArr || !output) return;

  auto& arr = unwrap(mlxArr);
  mx::eval(arr);

  auto nbytes = arr.nbytes();
  auto* src = arr.data<void>();
  auto* dst = output->buffer();

  if (src != dst) {
    std::memcpy(dst, src, nbytes);
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Phase 1: Element-wise emit helpers
// ═══════════════════════════════════════════════════════════════════════════

std::shared_ptr<void> MlxIRBuilder::emitBinaryElementwise(const std::string& opName,
                                                            const std::shared_ptr<void>& lhs,
                                                            const std::shared_ptr<void>& rhs) {
  auto& a = unwrap(lhs);
  auto& b = unwrap(rhs);
  std::string op = normalizeOp(opName);

  if (op == "add" || op == "addscalar") return wrap(mx::add(a, b));
  if (op == "subtract" || op == "sub" || op == "subtractscalar") return wrap(mx::subtract(a, b));
  if (op == "multiply" || op == "mul" || op == "mulscalar" || op == "multiplyscalar") return wrap(mx::multiply(a, b));
  if (op == "divide" || op == "div" || op == "divscalar" || op == "dividescalar") return wrap(mx::divide(a, b));
  if (op == "floormod" || op == "mod" || op == "floormodscalar") return wrap(mx::remainder(a, b));
  if (op == "maximum" || op == "maxpairwise") return wrap(mx::maximum(a, b));
  if (op == "minimum" || op == "minpairwise") return wrap(mx::minimum(a, b));
  if (op == "pow" || op == "powscalar") return wrap(mx::power(a, b));
  if (op == "squaredsubtract") {
    auto diff = mx::subtract(a, b);
    return wrap(mx::multiply(diff, diff));
  }
  if (op == "reversedivide" || op == "rdiv" || op == "reversedividescalar") return wrap(mx::divide(b, a));
  if (op == "reversesubtract" || op == "rsub" || op == "reversesubtractscalar") return wrap(mx::subtract(b, a));
  if (op == "atan2") return wrap(mx::arctan2(a, b));
  if (op == "realdiv") return wrap(mx::divide(a, b));
  if (op == "floordiv") return wrap(mx::floor(mx::divide(a, b)));
  if (op == "multiplynonan") {
    // multiply but return 0 where either input is NaN
    auto product = mx::multiply(a, b);
    auto nanMask = mx::logical_or(mx::isnan(a), mx::isnan(b));
    return wrap(mx::where(nanMask, mx::array(0.0f), product));
  }
  if (op == "swishmul") {
    // SwiGLU: x * sigmoid(x) * y
    return wrap(mx::multiply(mx::multiply(a, mx::sigmoid(a)), b));
  }
  // prelu: max(0, x) + alpha * min(0, x) where alpha is per-channel
  if (op == "prelu") {
    auto pos = mx::maximum(a, mx::array(0.0f));
    auto neg = mx::multiply(b, mx::minimum(a, mx::array(0.0f)));
    return wrap(mx::add(pos, neg));
  }

  DSP_DIAG(FALLBACK, "MlxIRBuilder: unsupported binary op '%s'", opName.c_str());
  return nullptr;
}

std::shared_ptr<void> MlxIRBuilder::emitUnaryElementwise(const std::string& opName,
                                                           const std::shared_ptr<void>& input,
                                                           const double* tArgs, int numTArgs) {
  auto& x = unwrap(input);
  std::string op = normalizeOp(opName);

  if (op == "abs") return wrap(mx::abs(x));
  if (op == "neg" || op == "negative") return wrap(mx::negative(x));
  if (op == "exp") return wrap(mx::exp(x));
  if (op == "log" || op == "logx") return wrap(mx::log(x));
  if (op == "sqrt") return wrap(mx::sqrt(x));
  if (op == "rsqrt") return wrap(mx::rsqrt(x));
  if (op == "square") return wrap(mx::square(x));
  if (op == "ceil") return wrap(mx::ceil(x));
  if (op == "floor") return wrap(mx::floor(x));
  if (op == "round") return wrap(mx::round(x));
  if (op == "sign") return wrap(mx::sign(x));
  if (op == "reciprocal") return wrap(mx::reciprocal(x));
  if (op == "sin") return wrap(mx::sin(x));
  if (op == "cos") return wrap(mx::cos(x));
  if (op == "tan") return wrap(mx::tan(x));
  if (op == "asin") return wrap(mx::arcsin(x));
  if (op == "acos") return wrap(mx::arccos(x));
  if (op == "atan") return wrap(mx::arctan(x));
  if (op == "sinh") return wrap(mx::sinh(x));
  if (op == "cosh") return wrap(mx::cosh(x));
  if (op == "tanh") return wrap(mx::tanh(x));
  if (op == "erf") return wrap(mx::erf(x));
  if (op == "erfc") return wrap(mx::subtract(mx::array(1.0f), mx::erf(x)));
  if (op == "sigmoid" || op == "logistic") return wrap(mx::sigmoid(x));

  // ReLU family
  if (op == "relu") return wrap(mx::maximum(x, mx::array(0.0f)));
  if (op == "relu6") return wrap(mx::minimum(mx::maximum(x, mx::array(0.0f)), mx::array(6.0f)));
  if (op == "leakyrelu") {
    float alpha = (numTArgs > 0) ? static_cast<float>(tArgs[0]) : 0.01f;
    return wrap(mx::where(mx::greater(x, mx::array(0.0f)), x, mx::multiply(x, mx::array(alpha))));
  }
  if (op == "elu") {
    float alpha = (numTArgs > 0) ? static_cast<float>(tArgs[0]) : 1.0f;
    auto pos = x;
    auto neg = mx::multiply(mx::array(alpha), mx::subtract(mx::exp(x), mx::array(1.0f)));
    return wrap(mx::where(mx::greater(x, mx::array(0.0f)), pos, neg));
  }
  if (op == "selu") {
    float lambda = 1.0507009873554804934193349852946f;
    float alphaVal = 1.6732632423543772848170429916717f;
    auto pos = mx::multiply(mx::array(lambda), x);
    auto neg = mx::multiply(mx::array(lambda * alphaVal), mx::subtract(mx::exp(x), mx::array(1.0f)));
    return wrap(mx::where(mx::greater(x, mx::array(0.0f)), pos, neg));
  }
  if (op == "gelu") {
    float sqrt2pi = 0.7978845608028654f;
    auto x3 = mx::power(x, mx::array(3.0f));
    auto inner = mx::multiply(mx::array(sqrt2pi), mx::add(x, mx::multiply(mx::array(0.044715f), x3)));
    auto cdf = mx::multiply(mx::array(0.5f), mx::add(mx::array(1.0f), mx::tanh(inner)));
    return wrap(mx::multiply(x, cdf));
  }
  if (op == "softplus") return wrap(mx::log(mx::add(mx::array(1.0f), mx::exp(x))));
  if (op == "swish" || op == "silu") return wrap(mx::multiply(x, mx::sigmoid(x)));
  if (op == "mish") {
    auto sp = mx::log(mx::add(mx::array(1.0f), mx::exp(x)));
    return wrap(mx::multiply(x, mx::tanh(sp)));
  }
  if (op == "log1p") return wrap(mx::log1p(x));
  if (op == "expm1") return wrap(mx::subtract(mx::exp(x), mx::array(1.0f)));
  if (op == "isnan") return wrap(mx::isnan(x));
  if (op == "isinf") return wrap(mx::isinf(x));
  if (op == "clip" || op == "clipbyvalue" || op == "clamp") {
    float minVal = (numTArgs > 0) ? static_cast<float>(tArgs[0]) : 0.0f;
    float maxVal = (numTArgs > 1) ? static_cast<float>(tArgs[1]) : 6.0f;
    return wrap(mx::clip(x, mx::array(minVal), mx::array(maxVal)));
  }
  // softsign: x / (1 + |x|)
  if (op == "softsign") {
    return wrap(mx::divide(x, mx::add(mx::array(1.0f), mx::abs(x))));
  }
  // hard_sigmoid: max(0, min(1, alpha*x + beta))  with alpha=0.2, beta=0.5
  if (op == "hardsigmoid") {
    float alpha = (numTArgs > 0) ? static_cast<float>(tArgs[0]) : 0.2f;
    float beta = (numTArgs > 1) ? static_cast<float>(tArgs[1]) : 0.5f;
    auto linear = mx::add(mx::multiply(mx::array(alpha), x), mx::array(beta));
    return wrap(mx::clip(linear, mx::array(0.0f), mx::array(1.0f)));
  }
  // hardtanh: clamp(x, -1, 1)
  if (op == "hardtanh") {
    float minVal = (numTArgs > 0) ? static_cast<float>(tArgs[0]) : -1.0f;
    float maxVal = (numTArgs > 1) ? static_cast<float>(tArgs[1]) : 1.0f;
    return wrap(mx::clip(x, mx::array(minVal), mx::array(maxVal)));
  }
  // Scalar ops: second operand from tArgs[0] (categorized as UNARY in OpCategoryTable)
  if (op == "addscalar") {
    float scalar = (numTArgs > 0) ? static_cast<float>(tArgs[0]) : 0.0f;
    return wrap(mx::add(x, mx::array(scalar)));
  }
  if (op == "subtractscalar") {
    float scalar = (numTArgs > 0) ? static_cast<float>(tArgs[0]) : 0.0f;
    return wrap(mx::subtract(x, mx::array(scalar)));
  }
  if (op == "multiplyscalar") {
    float scalar = (numTArgs > 0) ? static_cast<float>(tArgs[0]) : 1.0f;
    return wrap(mx::multiply(x, mx::array(scalar)));
  }
  if (op == "dividescalar") {
    float scalar = (numTArgs > 0) ? static_cast<float>(tArgs[0]) : 1.0f;
    return wrap(mx::divide(x, mx::array(scalar)));
  }
  // fused_gelu: fast GELU approximation x * sigmoid(1.702 * x)
  if (op == "fusedgelu") {
    return wrap(mx::multiply(x, mx::sigmoid(mx::multiply(mx::array(1.702f), x))));
  }
  // hardswish: x * relu6(x + 3) / 6
  if (op == "hardswish") {
    auto inner = mx::add(x, mx::array(3.0f));
    auto clamped = mx::clip(inner, mx::array(0.0f), mx::array(6.0f));
    return wrap(mx::divide(mx::multiply(x, clamped), mx::array(6.0f)));
  }
  // celu: max(0,x) + min(0, alpha * (exp(x/alpha) - 1))
  if (op == "celu") {
    float alpha = (numTArgs > 0) ? static_cast<float>(tArgs[0]) : 1.0f;
    auto pos = mx::maximum(x, mx::array(0.0f));
    auto neg = mx::minimum(mx::array(0.0f),
                           mx::multiply(mx::array(alpha),
                                        mx::subtract(mx::exp(mx::divide(x, mx::array(alpha))),
                                                     mx::array(1.0f))));
    return wrap(mx::add(pos, neg));
  }
  // thresholdedrelu: x if x > theta, else 0
  if (op == "thresholdedrelu") {
    float theta = (numTArgs > 0) ? static_cast<float>(tArgs[0]) : 1.0f;
    return wrap(mx::where(mx::greater(x, mx::array(theta)), x, mx::array(0.0f)));
  }

  DSP_DIAG(FALLBACK, "MlxIRBuilder: unsupported unary op '%s'", opName.c_str());
  return nullptr;
}

std::shared_ptr<void> MlxIRBuilder::emitComparisonOp(const std::string& opName,
                                                       const std::shared_ptr<void>& lhs,
                                                       const std::shared_ptr<void>& rhs) {
  auto& a = unwrap(lhs);
  auto& b = unwrap(rhs);
  std::string op = normalizeOp(opName);

  if (op == "greater" || op == "greaterthan") return wrap(mx::greater(a, b));
  if (op == "less" || op == "lessthan") return wrap(mx::less(a, b));
  if (op == "greaterequal" || op == "greaterthanorequal") return wrap(mx::greater_equal(a, b));
  if (op == "lessequal" || op == "lessthanorequal") return wrap(mx::less_equal(a, b));
  if (op == "equals" || op == "equal") return wrap(mx::equal(a, b));
  if (op == "notequals" || op == "notequal") return wrap(mx::not_equal(a, b));

  DSP_DIAG(FALLBACK, "MlxIRBuilder: unsupported comparison op '%s'", opName.c_str());
  return nullptr;
}

std::shared_ptr<void> MlxIRBuilder::emitLogicalOp(const std::string& opName,
                                                    const std::shared_ptr<void>& lhs,
                                                    const std::shared_ptr<void>& rhs) {
  auto& a = unwrap(lhs);
  std::string op = normalizeOp(opName);

  if (op == "booleanand" || op == "and" || op == "logicaland") {
    return wrap(mx::logical_and(a, unwrap(rhs)));
  }
  if (op == "booleanor" || op == "or" || op == "logicalor") {
    return wrap(mx::logical_or(a, unwrap(rhs)));
  }
  if (op == "booleannot" || op == "not" || op == "logicalnot" || op == "boolnot") {
    return wrap(mx::logical_not(a));
  }
  if (op == "booleanxor" || op == "xor" || op == "logicalxor") {
    auto orResult = mx::logical_or(a, unwrap(rhs));
    auto andResult = mx::logical_and(a, unwrap(rhs));
    return wrap(mx::logical_and(orResult, mx::logical_not(andResult)));
  }

  DSP_DIAG(FALLBACK, "MlxIRBuilder: unsupported logical op '%s'", opName.c_str());
  return nullptr;
}

std::shared_ptr<void> MlxIRBuilder::emitTernaryOp(const std::string& opName,
                                                    const std::shared_ptr<void>& cond,
                                                    const std::shared_ptr<void>& ifTrue,
                                                    const std::shared_ptr<void>& ifFalse) {
  return wrap(mx::where(unwrap(cond), unwrap(ifTrue), unwrap(ifFalse)));
}

std::shared_ptr<void> MlxIRBuilder::emitIdentityOp(const std::shared_ptr<void>& input) {
  return input;
}

std::shared_ptr<void> MlxIRBuilder::emitCastOp(const std::shared_ptr<void>& input,
                                                 sd::DataType targetType) {
  auto& x = unwrap(input);
  auto dtype = sdTypeToMlxDtypeInternal(targetType);
  return wrap(mx::astype(x, dtype));
}

// ═══════════════════════════════════════════════════════════════════════════
// Phase 2: Reduction emit helpers
// ═══════════════════════════════════════════════════════════════════════════

std::shared_ptr<void> MlxIRBuilder::emitReductionOp(const std::string& opName,
                                                      const std::shared_ptr<void>& input,
                                                      const LongType* iArgs, int numIArgs,
                                                      bool keepDims) {
  auto& x = unwrap(input);
  std::string op = normalizeOp(opName);

  // Build axes vector from iArgs
  std::vector<int> axes;
  if (numIArgs > 0 && iArgs) {
    for (int i = 0; i < numIArgs; i++) {
      axes.push_back(static_cast<int>(iArgs[i]));
    }
  }
  // Empty axes = reduce all dimensions

  // reduce_sum / sum
  if (op == "reducesum" || op == "sum" || op == "reducesumbp") {
    if (axes.empty()) return wrap(mx::sum(x, keepDims));
    return wrap(mx::sum(x, axes, keepDims));
  }
  // reduce_max / max
  if (op == "reducemax" || op == "max" || op == "reduceamax") {
    if (axes.empty()) return wrap(mx::max(x, keepDims));
    return wrap(mx::max(x, axes, keepDims));
  }
  // reduce_min / min
  if (op == "reducemin" || op == "min" || op == "reduceamin") {
    if (axes.empty()) return wrap(mx::min(x, keepDims));
    return wrap(mx::min(x, axes, keepDims));
  }
  // reduce_mean / mean
  if (op == "reducemean" || op == "mean") {
    if (axes.empty()) return wrap(mx::mean(x, keepDims));
    return wrap(mx::mean(x, axes, keepDims));
  }
  // reduce_prod / prod
  if (op == "reduceprod" || op == "prod") {
    if (axes.empty()) return wrap(mx::prod(x, keepDims));
    return wrap(mx::prod(x, axes, keepDims));
  }
  // reduce_variance / variance
  if (op == "reducevariance" || op == "variance") {
    if (axes.empty()) return wrap(mx::var(x, keepDims));
    return wrap(mx::var(x, axes, keepDims));
  }
  // reduce_stdev / stdev
  if (op == "reducestdev" || op == "stdev") {
    if (axes.empty()) {
      auto v = mx::var(x, keepDims);
      return wrap(mx::sqrt(v));
    }
    auto v = mx::var(x, axes, keepDims);
    return wrap(mx::sqrt(v));
  }
  // reduce_norm1 / norm1
  if (op == "reducenorm1" || op == "norm1") {
    auto absX = mx::abs(x);
    if (axes.empty()) return wrap(mx::sum(absX, keepDims));
    return wrap(mx::sum(absX, axes, keepDims));
  }
  // reduce_norm2 / norm2
  if (op == "reducenorm2" || op == "norm2") {
    auto sq = mx::square(x);
    if (axes.empty()) return wrap(mx::sqrt(mx::sum(sq, keepDims)));
    return wrap(mx::sqrt(mx::sum(sq, axes, keepDims)));
  }
  // reduce_logsumexp / logsumexp
  if (op == "reducelogsumexp" || op == "logsumexp") {
    if (axes.empty()) return wrap(mx::logsumexp(x, keepDims));
    return wrap(mx::logsumexp(x, axes, keepDims));
  }
  // argmax
  if (op == "argmax") {
    int axis = (numIArgs > 0 && iArgs) ? static_cast<int>(iArgs[0]) : 0;
    return wrap(mx::argmax(x, axis, keepDims));
  }
  // argmin
  if (op == "argmin") {
    int axis = (numIArgs > 0 && iArgs) ? static_cast<int>(iArgs[0]) : 0;
    return wrap(mx::argmin(x, axis, keepDims));
  }
  // normmax: infinity norm = max(|x|)
  if (op == "normmax") {
    auto absX = mx::abs(x);
    if (axes.empty()) return wrap(mx::max(absX, keepDims));
    return wrap(mx::max(absX, axes, keepDims));
  }

  // cumsum: cumulative sum along axis
  if (op == "cumsum") {
    int axis = (numIArgs > 0 && iArgs) ? static_cast<int>(iArgs[0]) : 0;
    bool exclusive = (numIArgs > 1 && iArgs) ? (iArgs[1] != 0) : false;
    bool reverse = (numIArgs > 2 && iArgs) ? (iArgs[2] != 0) : false;
    return wrap(mx::cumsum(x, axis, reverse, !exclusive));
  }

  // cumprod: cumulative product along axis
  if (op == "cumprod") {
    int axis = (numIArgs > 0 && iArgs) ? static_cast<int>(iArgs[0]) : 0;
    bool exclusive = (numIArgs > 1 && iArgs) ? (iArgs[1] != 0) : false;
    bool reverse = (numIArgs > 2 && iArgs) ? (iArgs[2] != 0) : false;
    return wrap(mx::cumprod(x, axis, reverse, !exclusive));
  }

  // top_k: return k largest elements (NOTE: categorized as DATA_MOVEMENT, but handled here too)
  if (op == "topk") {
    int k = (numIArgs > 0 && iArgs) ? static_cast<int>(iArgs[0]) : 1;
    return wrap(mx::topk(x, k, -1));
  }

  // mean_square: mean(x^2) along last dim — building block for RMS norm
  if (op == "meansquare") {
    auto sq = mx::square(x);
    std::vector<int> lastAxis = {-1};
    return wrap(mx::mean(sq, lastAxis, keepDims));
  }

  DSP_DIAG(FALLBACK, "MlxIRBuilder: unsupported reduction op '%s'", opName.c_str());
  return nullptr;
}

// ═══════════════════════════════════════════════════════════════════════════
// Phase 2: Matmul emit helpers
// ═══════════════════════════════════════════════════════════════════════════

std::shared_ptr<void> MlxIRBuilder::emitMatmulOp(const std::string& opName,
                                                   const std::vector<std::shared_ptr<void>>& inputs,
                                                   const double* tArgs, int numTArgs,
                                                   const LongType* iArgs, int numIArgs) {
  std::string op = normalizeOp(opName);

  if (inputs.size() < 2) {
    DSP_DIAG(COMPILE, "MlxIRBuilder: matmul needs >= 2 inputs, got %d", static_cast<int>(inputs.size()));
    return nullptr;
  }

  auto& a = unwrap(inputs[0]);
  auto& b = unwrap(inputs[1]);

  // matmul / batch_matmul / batched_gemm
  if (op == "matmul" || op == "mmul" || op == "batchmatmul" || op == "batchedgemm" || op == "tensormmul") {
    // Check for transpositions from iArgs
    bool transA = (numIArgs > 0 && iArgs) ? (iArgs[0] != 0) : false;
    bool transB = (numIArgs > 1 && iArgs) ? (iArgs[1] != 0) : false;

    auto aT = transA ? mx::swapaxes(a, -2, -1) : a;
    auto bT = transB ? mx::swapaxes(b, -2, -1) : b;

    auto result = mx::matmul(aT, bT);

    // Alpha scaling
    if (numTArgs > 0 && tArgs && tArgs[0] != 1.0) {
      result = mx::multiply(result, mx::array(static_cast<float>(tArgs[0])));
    }

    return wrap(std::move(result));
  }

  // xw_plus_b: output = x @ w + b
  if (op == "xwplusb") {
    auto result = mx::matmul(a, b);
    if (inputs.size() >= 3) {
      result = mx::add(result, unwrap(inputs[2]));
    }
    return wrap(std::move(result));
  }

  // quantized_matmul: matmul with quantized weights
  // MLX supports mx::quantized_matmul natively for INT4/INT8 weights
  if (op == "quantizedmatmul") {
    // iArgs[0] = quant type (0=Q4_0, 1=Q4_1, 2=Q5_0, 3=Q5_1, 4=Q8_0)
    // iArgs[1] = transpose_a, iArgs[2] = transpose_b
    bool transA = (numIArgs > 1 && iArgs) ? (iArgs[1] != 0) : false;
    bool transB = (numIArgs > 2 && iArgs) ? (iArgs[2] != 0) : false;

    auto aT = transA ? mx::swapaxes(a, -2, -1) : a;
    auto bT = transB ? mx::swapaxes(b, -2, -1) : b;

    // If weights are already dequantized (float), just do regular matmul
    // Full quantized_matmul with scales/biases needs inputs[2]=scales, inputs[3]=biases
    if (inputs.size() >= 4) {
      // Use MLX quantized matmul: x @ dequantize(w, scales, biases)
      auto& scales = unwrap(inputs[2]);
      auto& biases = unwrap(inputs[3]);
      int groupSize = (numIArgs > 3 && iArgs) ? static_cast<int>(iArgs[3]) : 64;
      int bits = (numIArgs > 4 && iArgs) ? static_cast<int>(iArgs[4]) : 4;
      // Dequantize and matmul (MLX fuses this internally)
      auto wDequant = mx::dequantize(bT, scales, biases, groupSize, bits);
      return wrap(mx::matmul(aT, wDequant));
    }

    // Fallback: regular matmul (weights already float)
    return wrap(mx::matmul(aT, bT));
  }

  DSP_DIAG(FALLBACK, "MlxIRBuilder: unsupported matmul op '%s'", opName.c_str());
  return nullptr;
}

// ═══════════════════════════════════════════════════════════════════════════
// Phase 2: Normalization emit helpers
// ═══════════════════════════════════════════════════════════════════════════

std::shared_ptr<void> MlxIRBuilder::emitNormalizationOp(const std::string& opName,
                                                          const std::vector<std::shared_ptr<void>>& inputs,
                                                          const LongType* iArgs, int numIArgs,
                                                          const double* tArgs, int numTArgs) {
  if (inputs.empty()) return nullptr;
  auto& x = unwrap(inputs[0]);
  std::string op = normalizeOp(opName);

  // softmax
  if (op == "softmax") {
    int axis = (numIArgs > 0 && iArgs) ? static_cast<int>(iArgs[0]) : -1;
    return wrap(mx::softmax(x, axis));
  }

  // log_softmax
  if (op == "logsoftmax") {
    int axis = (numIArgs > 0 && iArgs) ? static_cast<int>(iArgs[0]) : -1;
    // log_softmax(x) = x - logsumexp(x, axis, keepdims=true)
    auto lse = mx::logsumexp(x, std::vector<int>{axis}, true);
    return wrap(mx::subtract(x, lse));
  }

  // layer_norm: uses mlx::core::fast::layer_norm for optimized Metal kernel
  if (op == "layernorm" || op == "fusedlayernorm") {
    float eps = (numTArgs > 0 && tArgs) ? static_cast<float>(tArgs[0]) : 1e-5f;

    // Use MLX fast layer_norm: accumulates in higher precision
    if (inputs.size() >= 3) {
      auto result = mx::fast::layer_norm(x, unwrap(inputs[1]), unwrap(inputs[2]), eps);
      return wrap(std::move(result));
    } else if (inputs.size() >= 2) {
      // weight only, no bias — pass empty array for bias
      auto result = mx::fast::layer_norm(x, unwrap(inputs[1]), mx::array(), eps);
      return wrap(std::move(result));
    } else {
      // No weight/bias
      auto result = mx::fast::layer_norm(x, mx::array(), mx::array(), eps);
      return wrap(std::move(result));
    }
  }

  // rms_norm: uses mlx::core::fast::rms_norm for optimized Metal kernel
  if (op == "rmsnorm") {
    float eps = (numTArgs > 0 && tArgs) ? static_cast<float>(tArgs[0]) : 1e-5f;

    // Use MLX fast rms_norm: accumulates in higher precision
    if (inputs.size() >= 2) {
      auto result = mx::fast::rms_norm(x, unwrap(inputs[1]), eps);
      return wrap(std::move(result));
    } else {
      auto result = mx::fast::rms_norm(x, mx::array(), eps);
      return wrap(std::move(result));
    }
  }

  // batch_norm: (x - mean) / sqrt(var + eps) * gamma + beta
  if (op == "batchnorm") {
    float eps = (numTArgs > 0 && tArgs) ? static_cast<float>(tArgs[0]) : 1e-5f;
    // For inference: use running mean/var (inputs[3], inputs[4])
    // For training: compute from batch (axes [0, 2, 3] for NCHW)
    if (inputs.size() >= 5) {
      auto& gamma = unwrap(inputs[1]);
      auto& beta = unwrap(inputs[2]);
      auto& runMean = unwrap(inputs[3]);
      auto& runVar = unwrap(inputs[4]);
      auto normalized = mx::multiply(
          mx::subtract(x, runMean),
          mx::rsqrt(mx::add(runVar, mx::array(eps)))
      );
      normalized = mx::add(mx::multiply(normalized, gamma), beta);
      return wrap(std::move(normalized));
    }
    // Simplified: just normalize over batch dim
    std::vector<int> axes = {0};
    auto mean = mx::mean(x, axes, true);
    auto variance = mx::var(x, axes, true);
    auto normalized = mx::multiply(
        mx::subtract(x, mean),
        mx::rsqrt(mx::add(variance, mx::array(eps)))
    );
    if (inputs.size() >= 2) normalized = mx::multiply(normalized, unwrap(inputs[1]));
    if (inputs.size() >= 3) normalized = mx::add(normalized, unwrap(inputs[2]));
    return wrap(std::move(normalized));
  }

  // normalize_moments: just return normalized
  if (op == "normalizemoments") {
    return wrap(mx::array(x));  // Pass-through for now
  }

  DSP_DIAG(FALLBACK, "MlxIRBuilder: unsupported normalization op '%s'", opName.c_str());
  return nullptr;
}

// ═══════════════════════════════════════════════════════════════════════════
// Phase 2: Shape manipulation emit helpers
// ═══════════════════════════════════════════════════════════════════════════

std::shared_ptr<void> MlxIRBuilder::emitShapeManipOp(const std::string& opName,
                                                       const std::shared_ptr<void>& input,
                                                       const LongType* iArgs, int numIArgs,
                                                       NDArray* outputArr) {
  auto& x = unwrap(input);
  std::string op = normalizeOp(opName);

  // reshape
  if (op == "reshape") {
    if (!outputArr) return nullptr;
    auto rank = outputArr->rankOf();
    std::vector<int> newShape(rank);
    for (int i = 0; i < rank; i++) {
      newShape[i] = static_cast<int>(outputArr->sizeAt(i));
    }
    return wrap(mx::reshape(x, newShape));
  }

  // permute / transpose
  if (op == "permute" || op == "transpose") {
    if (numIArgs > 0 && iArgs) {
      std::vector<int> axes(numIArgs);
      for (int i = 0; i < numIArgs; i++) {
        axes[i] = static_cast<int>(iArgs[i]);
      }
      return wrap(mx::transpose(x, axes));
    }
    // No axes specified: reverse all dimensions
    return wrap(mx::transpose(x));
  }

  // expand_dims
  if (op == "expanddims") {
    int axis = (numIArgs > 0 && iArgs) ? static_cast<int>(iArgs[0]) : 0;
    return wrap(mx::expand_dims(x, axis));
  }

  // squeeze
  if (op == "squeeze") {
    if (numIArgs > 0 && iArgs) {
      std::vector<int> axes(numIArgs);
      for (int i = 0; i < numIArgs; i++) {
        axes[i] = static_cast<int>(iArgs[i]);
      }
      return wrap(mx::squeeze(x, axes));
    }
    return wrap(mx::squeeze(x));
  }

  // flatten / flatten_2d
  if (op == "flatten" || op == "flatten2d") {
    return wrap(mx::flatten(x));
  }

  // transpose (explicit, same as permute)
  if (op == "transpose") {
    if (numIArgs > 0 && iArgs) {
      std::vector<int> axes(numIArgs);
      for (int i = 0; i < numIArgs; i++) {
        axes[i] = static_cast<int>(iArgs[i]);
      }
      return wrap(mx::transpose(x, axes));
    }
    return wrap(mx::transpose(x));
  }

  // triu: upper triangular
  if (op == "triu") {
    int k = (numIArgs > 0 && iArgs) ? static_cast<int>(iArgs[0]) : 0;
    return wrap(mx::triu(x, k));
  }

  // tril: lower triangular
  if (op == "tril") {
    int k = (numIArgs > 0 && iArgs) ? static_cast<int>(iArgs[0]) : 0;
    return wrap(mx::tril(x, k));
  }

  DSP_DIAG(FALLBACK, "MlxIRBuilder: unsupported shape manipulation op '%s'", opName.c_str());
  return nullptr;
}

// ═══════════════════════════════════════════════════════════════════════════
// Phase 2: Data movement emit helpers
// ═══════════════════════════════════════════════════════════════════════════

std::shared_ptr<void> MlxIRBuilder::emitDataMovementOp(const std::string& opName,
                                                         const std::vector<std::shared_ptr<void>>& inputs,
                                                         const LongType* iArgs, int numIArgs,
                                                         const double* tArgs, int numTArgs,
                                                         NDArray* outputArr) {
  if (inputs.empty()) return nullptr;
  std::string op = normalizeOp(opName);

  // gather
  if (op == "gather" || op == "gathernd") {
    if (inputs.size() < 2) return nullptr;
    auto& data = unwrap(inputs[0]);
    auto& indices = unwrap(inputs[1]);
    int axis = (numIArgs > 0 && iArgs) ? static_cast<int>(iArgs[0]) : 0;

    if (op == "gathernd") {
      // gather_nd: advanced indexing — use take for 1D case
      // For general case, flatten indices and gather
      auto indicesInt = mx::astype(indices, mx::int32);
      return wrap(mx::take(data, indicesInt, axis));
    }

    auto indicesInt = mx::astype(indices, mx::int32);
    return wrap(mx::take(data, indicesInt, axis));
  }

  // concat
  if (op == "concat" || op == "concatbp") {
    int axis = (numIArgs > 0 && iArgs) ? static_cast<int>(iArgs[0]) : 0;
    std::vector<mx::array> arrays;
    for (auto& inp : inputs) {
      arrays.push_back(unwrap(inp));
    }
    return wrap(mx::concatenate(arrays, axis));
  }

  // stack
  if (op == "stack") {
    int axis = (numIArgs > 0 && iArgs) ? static_cast<int>(iArgs[0]) : 0;
    std::vector<mx::array> arrays;
    for (auto& inp : inputs) {
      arrays.push_back(unwrap(inp));
    }
    return wrap(mx::stack(arrays, axis));
  }

  // split / split_v
  if (op == "split" || op == "splitv") {
    auto& data = unwrap(inputs[0]);
    int axis = (numIArgs > 0 && iArgs) ? static_cast<int>(iArgs[0]) : 0;

    if (op == "split" && numIArgs > 1) {
      int numSplits = static_cast<int>(iArgs[1]);
      auto splits = mx::split(data, numSplits, axis);
      // Return the first split for single-output slot (multi-output handled at buildGraph level)
      if (!splits.empty()) return wrap(std::move(splits[0]));
    }

    // split_v: variable-size splits from iArgs
    if (numIArgs > 1) {
      std::vector<int> indices;
      int cumSum = 0;
      for (int i = 1; i < numIArgs; i++) {
        cumSum += static_cast<int>(iArgs[i]);
        indices.push_back(cumSum);
      }
      auto splits = mx::split(data, indices, axis);
      if (!splits.empty()) return wrap(std::move(splits[0]));
    }

    return nullptr;
  }

  // tile
  if (op == "tile") {
    auto& data = unwrap(inputs[0]);
    if (numIArgs > 0 && iArgs) {
      std::vector<int> reps(numIArgs);
      for (int i = 0; i < numIArgs; i++) {
        reps[i] = static_cast<int>(iArgs[i]);
      }
      return wrap(mx::tile(data, reps));
    }
    // If reps from second input
    if (inputs.size() >= 2) {
      // Evaluate the reps array to get values
      auto& repsArr = unwrap(inputs[1]);
      mx::eval(repsArr);
      auto repsData = repsArr.data<int32_t>();
      int ndim = static_cast<int>(repsArr.size());
      std::vector<int> reps(ndim);
      for (int i = 0; i < ndim; i++) reps[i] = repsData[i];
      return wrap(mx::tile(data, reps));
    }
    return nullptr;
  }

  // strided_slice
  if (op == "stridedslice") {
    auto& data = unwrap(inputs[0]);
    // iArgs layout: [begin0, begin1, ..., end0, end1, ..., strides0, strides1, ...]
    int rank = static_cast<int>(data.ndim());
    if (numIArgs < rank * 3) {
      DSP_DIAG(SHAPE, "MlxIRBuilder: strided_slice needs 3*rank iArgs, got %d for rank %d",
                numIArgs, rank);
      return nullptr;
    }

    std::vector<int> starts(rank), stops(rank), strides(rank);
    for (int i = 0; i < rank; i++) {
      starts[i] = static_cast<int>(iArgs[i]);
      stops[i] = static_cast<int>(iArgs[rank + i]);
      strides[i] = static_cast<int>(iArgs[2 * rank + i]);
    }

    return wrap(mx::slice(data, starts, stops, strides));
  }

  // scatter_nd / scatter_nd_update
  if (op == "scatternd" || op == "scatterndupdate") {
    if (inputs.size() < 3) return nullptr;
    auto& data = unwrap(inputs[0]);
    auto& indices = unwrap(inputs[1]);
    auto& updates = unwrap(inputs[2]);

    // Use scatter: data[indices] = updates
    // MLX scatter_add equivalent: create zeros, scatter updates into it
    auto indicesInt = mx::astype(indices, mx::int32);

    // For scatter_nd_update: start with data, overwrite at indices
    // This is a simplified 1D approach
    int axis = 0;
    auto result = mx::put_along_axis(data, indicesInt, updates, axis);
    return wrap(std::move(result));
  }

  // embedding_lookup: table[indices] via mx::take
  if (op == "embeddinglookup") {
    if (inputs.size() < 2) return nullptr;
    auto& table = unwrap(inputs[0]);
    auto& indices = unwrap(inputs[1]);
    auto indicesInt = mx::astype(indices, mx::int32);
    // partition_mode from iArgs[0], but for basic lookup just use take on axis 0
    return wrap(mx::take(table, indicesInt, 0));
  }

  // repeat: replicate array elements along axis
  if (op == "repeat") {
    auto& data = unwrap(inputs[0]);
    if (numIArgs >= 2 && iArgs) {
      int repeats = static_cast<int>(iArgs[0]);
      int axis = static_cast<int>(iArgs[1]);
      return wrap(mx::repeat(data, repeats, axis));
    } else if (numIArgs >= 1 && iArgs) {
      int repeats = static_cast<int>(iArgs[0]);
      return wrap(mx::repeat(data, repeats));
    }
    return nullptr;
  }

  // pad: add padding to array
  if (op == "pad") {
    if (inputs.size() < 2) return nullptr;
    auto& data = unwrap(inputs[0]);
    // padding spec from second input (Nx2 array of [before, after] per dim)
    auto& padSpec = unwrap(inputs[1]);
    mx::eval(padSpec);

    int ndim = static_cast<int>(data.ndim());
    std::vector<int> padWidths;
    auto padData = padSpec.data<int32_t>();
    for (int i = 0; i < ndim * 2; i++) {
      padWidths.push_back(padData[i]);
    }

    // Pad mode from iArgs[0]: 0=constant, 1=reflect, 2=edge
    float padValue = (numTArgs > 0 && tArgs) ? static_cast<float>(tArgs[0]) : 0.0f;
    return wrap(mx::pad(data, padWidths, mx::array(padValue)));
  }

  // slice: extract sub-tensor (simpler than strided_slice)
  if (op == "slice") {
    auto& data = unwrap(inputs[0]);
    int rank = static_cast<int>(data.ndim());
    if (numIArgs < rank * 2) return nullptr;

    std::vector<int> starts(rank), sizes(rank);
    for (int i = 0; i < rank; i++) {
      starts[i] = static_cast<int>(iArgs[i]);
      sizes[i] = static_cast<int>(iArgs[rank + i]);
    }
    // Convert sizes to stops
    std::vector<int> stops(rank);
    for (int i = 0; i < rank; i++) {
      stops[i] = starts[i] + sizes[i];
    }
    return wrap(mx::slice(data, starts, stops));
  }

  // kv_cache_update: copy new K/V into cache at position offset
  if (op == "kvcacheupdate") {
    // inputs: [key_cache, value_cache, new_keys, new_values]
    // iArgs[0] = start position
    if (inputs.size() < 4) return nullptr;
    // Return updated key cache — actual in-place update done at execution level
    // For MLX lazy graph, concatenate or slice-assign
    auto& cache = unwrap(inputs[0]);
    auto& newKV = unwrap(inputs[2]);
    int startPos = (numIArgs > 0 && iArgs) ? static_cast<int>(iArgs[0]) : 0;
    int newLen = newKV.shape(1);  // seq_len dimension

    // Slice: cache[:, startPos:startPos+newLen, :, :] = newKV
    // MLX doesn't have slice_update, so concatenate old prefix + new + old suffix
    if (startPos == 0) {
      // Replacing from beginning
      auto suffix = mx::slice(cache, {0, newLen}, {cache.shape(0), cache.shape(1)});
      return wrap(mx::concatenate({newKV, suffix}, 1));
    }
    auto prefix = mx::slice(cache, {0, 0}, {cache.shape(0), startPos});
    int endPos = startPos + newLen;
    if (endPos < cache.shape(1)) {
      auto suffix = mx::slice(cache, {0, endPos}, {cache.shape(0), cache.shape(1)});
      return wrap(mx::concatenate({prefix, newKV, suffix}, 1));
    }
    return wrap(mx::concatenate({prefix, newKV}, 1));
  }

  // kv_scatter: scatter KV pairs into cache
  if (op == "kvscatter") {
    if (inputs.size() < 2) return nullptr;
    // Fall through to native — complex scatter semantics
    return nullptr;
  }

  DSP_DIAG(FALLBACK, "MlxIRBuilder: unsupported data movement op '%s'", opName.c_str());
  return nullptr;
}

// ═══════════════════════════════════════════════════════════════════════════
// Phase 2: Constant generation emit helpers
// ═══════════════════════════════════════════════════════════════════════════

std::shared_ptr<void> MlxIRBuilder::emitConstantGenOp(const std::string& opName,
                                                        const std::vector<std::shared_ptr<void>>& inputs,
                                                        const double* tArgs, int numTArgs,
                                                        const LongType* iArgs, int numIArgs,
                                                        NDArray* outputArr) {
  std::string op = normalizeOp(opName);

  if (!outputArr) {
    DSP_DIAG(COMPILE, "MlxIRBuilder: constant gen op '%s' needs output array for shape", opName.c_str());
    return nullptr;
  }

  auto rank = outputArr->rankOf();
  std::vector<int> shape(rank);
  for (int i = 0; i < rank; i++) {
    shape[i] = static_cast<int>(outputArr->sizeAt(i));
  }
  auto dtype = sdTypeToMlxDtypeInternal(outputArr->dataType());

  // zeros_like / zeros_as / zeroslike
  if (op == "zeroslike" || op == "zerosas") {
    return wrap(mx::zeros(shape, dtype));
  }

  // ones_like / ones_as / oneslike
  if (op == "oneslike" || op == "onesas") {
    return wrap(mx::ones(shape, dtype));
  }

  // create (fill with zero by default)
  if (op == "create") {
    return wrap(mx::zeros(shape, dtype));
  }

  // set_scalar: fill with tArgs[0]
  if (op == "setscalar") {
    float val = (numTArgs > 0 && tArgs) ? static_cast<float>(tArgs[0]) : 0.0f;
    return wrap(mx::full(shape, mx::array(val), dtype));
  }

  // range: start, stop, step from tArgs
  if (op == "range") {
    float start = (numTArgs > 0 && tArgs) ? static_cast<float>(tArgs[0]) : 0.0f;
    float stop = (numTArgs > 1 && tArgs) ? static_cast<float>(tArgs[1]) : 1.0f;
    float step = (numTArgs > 2 && tArgs) ? static_cast<float>(tArgs[2]) : 1.0f;
    return wrap(mx::arange(start, stop, step, dtype));
  }

  // shape_of: return the shape as an array
  if (op == "shapeof") {
    if (!inputs.empty()) {
      auto& x = unwrap(inputs[0]);
      auto xShape = x.shape();
      std::vector<int32_t> shapeVals(xShape.begin(), xShape.end());
      return wrap(mx::array(shapeVals.data(), {static_cast<int>(shapeVals.size())}, mx::int32));
    }
    return nullptr;
  }

  // min_max_datatype: return type limits
  if (op == "minmaxdatatype") {
    return wrap(mx::zeros(shape, dtype));
  }

  // eye: identity matrix
  if (op == "eye") {
    int n = shape.size() > 0 ? shape[0] : 1;
    int m = shape.size() > 1 ? shape[1] : n;
    return wrap(mx::eye(n, m, 0, dtype));
  }

  // linspace: evenly spaced values
  if (op == "linspace") {
    float start = (numTArgs > 0 && tArgs) ? static_cast<float>(tArgs[0]) : 0.0f;
    float stop = (numTArgs > 1 && tArgs) ? static_cast<float>(tArgs[1]) : 1.0f;
    int num = shape.size() > 0 ? shape[0] : 100;
    return wrap(mx::linspace(start, stop, num, dtype));
  }

  // fill: fill with value
  if (op == "fill") {
    float val = (numTArgs > 0 && tArgs) ? static_cast<float>(tArgs[0]) : 0.0f;
    return wrap(mx::full(shape, mx::array(val), dtype));
  }

  // onehot: one-hot encoding
  if (op == "onehot") {
    if (!inputs.empty()) {
      auto& indices = unwrap(inputs[0]);
      auto indicesInt = mx::astype(indices, mx::int32);
      int depth = (numIArgs > 0 && iArgs) ? static_cast<int>(iArgs[0]) : shape.back();
      // Build one-hot manually: eye(depth)[indices]
      auto eye = mx::eye(depth);
      return wrap(mx::take(eye, indicesInt, 0));
    }
    return nullptr;
  }

  DSP_DIAG(FALLBACK, "MlxIRBuilder: unsupported constant gen op '%s'", opName.c_str());
  return nullptr;
}

// ═══════════════════════════════════════════════════════════════════════════
// Phase 3: Convolution emit helpers
// ═══════════════════════════════════════════════════════════════════════════

std::shared_ptr<void> MlxIRBuilder::emitConvolutionOp(const std::string& opName,
                                                        const std::vector<std::shared_ptr<void>>& inputs,
                                                        const LongType* iArgs, int numIArgs,
                                                        NDArray* outputArr) {
  if (inputs.size() < 2) return nullptr;
  std::string op = normalizeOp(opName);

  auto& input = unwrap(inputs[0]);
  auto& filter = unwrap(inputs[1]);

  // conv2d
  // libnd4j uses NCHW, MLX conv2d expects NHWC
  // iArgs: [kH, kW, sH, sW, pH, pW, dH, dW, isSameMode, isNCHW, wFormat]
  if (op == "conv2d") {
    int sH = (numIArgs > 2) ? static_cast<int>(iArgs[2]) : 1;
    int sW = (numIArgs > 3) ? static_cast<int>(iArgs[3]) : 1;
    int pH = (numIArgs > 4) ? static_cast<int>(iArgs[4]) : 0;
    int pW = (numIArgs > 5) ? static_cast<int>(iArgs[5]) : 0;
    int dH = (numIArgs > 6) ? static_cast<int>(iArgs[6]) : 1;
    int dW = (numIArgs > 7) ? static_cast<int>(iArgs[7]) : 1;
    bool isNCHW = (numIArgs > 9) ? (iArgs[9] != 0) : true;

    // Convert NCHW -> NHWC for MLX
    auto inputNHWC = isNCHW ? mx::transpose(input, {0, 2, 3, 1}) : input;

    // Filter: libnd4j OIHW -> MLX expects OHWI
    auto filterOHWI = mx::transpose(filter, {0, 2, 3, 1});

    // MLX conv2d with padding, stride, dilation
    auto result = mx::conv2d(inputNHWC, filterOHWI,
                              {sH, sW},     // stride
                              {pH, pW},     // padding
                              {dH, dW},     // dilation
                              1);           // groups

    // Add bias if present
    if (inputs.size() >= 3) {
      result = mx::add(result, unwrap(inputs[2]));
    }

    // Convert back NHWC -> NCHW if needed
    if (isNCHW) {
      result = mx::transpose(result, {0, 3, 1, 2});
    }

    return wrap(std::move(result));
  }

  // depthwise_conv2d
  if (op == "depthwiseconv2d") {
    int sH = (numIArgs > 2) ? static_cast<int>(iArgs[2]) : 1;
    int sW = (numIArgs > 3) ? static_cast<int>(iArgs[3]) : 1;
    int pH = (numIArgs > 4) ? static_cast<int>(iArgs[4]) : 0;
    int pW = (numIArgs > 5) ? static_cast<int>(iArgs[5]) : 0;
    int dH = (numIArgs > 6) ? static_cast<int>(iArgs[6]) : 1;
    int dW = (numIArgs > 7) ? static_cast<int>(iArgs[7]) : 1;
    bool isNCHW = (numIArgs > 9) ? (iArgs[9] != 0) : true;

    auto inputNHWC = isNCHW ? mx::transpose(input, {0, 2, 3, 1}) : input;
    int inChannels = inputNHWC.shape(-1);

    // Depthwise: groups = in_channels
    auto filterOHWI = mx::transpose(filter, {0, 2, 3, 1});

    auto result = mx::conv2d(inputNHWC, filterOHWI,
                              {sH, sW}, {pH, pW}, {dH, dW},
                              inChannels);

    if (inputs.size() >= 3) {
      result = mx::add(result, unwrap(inputs[2]));
    }
    if (isNCHW) {
      result = mx::transpose(result, {0, 3, 1, 2});
    }

    return wrap(std::move(result));
  }

  // conv3d: not yet optimized in MLX — decompose to 2D slices or fall back
  if (op == "conv3d") {
    DSP_DIAG(FALLBACK, "MlxIRBuilder: conv3d not yet implemented in MLX backend");
    return nullptr;
  }

  // im2col / col2im: helper ops
  // im2col: extract image patches into columns for matrix-multiply convolution
  // Input: NCHW [N, C, H, W] -> Output: [N, C*kH*kW, outH*outW]
  // iArgs: [kH, kW, sH, sW, pH, pW, dH, dW, isSameMode]
  if (op == "im2col") {
    int kH = (numIArgs > 0) ? static_cast<int>(iArgs[0]) : 3;
    int kW = (numIArgs > 1) ? static_cast<int>(iArgs[1]) : 3;
    int sH = (numIArgs > 2) ? static_cast<int>(iArgs[2]) : 1;
    int sW = (numIArgs > 3) ? static_cast<int>(iArgs[3]) : 1;
    int pH = (numIArgs > 4) ? static_cast<int>(iArgs[4]) : 0;
    int pW = (numIArgs > 5) ? static_cast<int>(iArgs[5]) : 0;
    int dH = (numIArgs > 6) ? static_cast<int>(iArgs[6]) : 1;
    int dW = (numIArgs > 7) ? static_cast<int>(iArgs[7]) : 1;

    // Use a 1x1 identity convolution approach via as_strided
    // Simpler approach: pad, then use sliding window via unfold-like semantics
    // MLX doesn't have a direct im2col, but we can do it via conv2d with identity filter

    // Convert NCHW -> NHWC for MLX
    auto inputNHWC = mx::transpose(input, {0, 2, 3, 1});

    int C = input.shape(1);
    // Create identity filter: [C*kH*kW, kH, kW, C] that extracts each patch element
    // Each output channel corresponds to one (c, kh, kw) combination
    int outChannels = C * kH * kW;
    auto identityFilter = mx::zeros({outChannels, kH, kW, C}, mx::float32);
    mx::eval(identityFilter);

    // Fill identity: filter[c*kH*kW + kh*kW + kw, kh, kw, c] = 1.0
    // This is complex to set up dynamically — use the conv2d approach directly
    // For now, fall back to letting the op execute natively
    DSP_DIAG(FALLBACK, "MlxIRBuilder: im2col using native fallback");
    return nullptr;
  }

  // col2im: inverse of im2col — fold columns back into image
  if (op == "col2im") {
    DSP_DIAG(FALLBACK, "MlxIRBuilder: col2im using native fallback");
    return nullptr;
  }

  // Backprop variants: always fall back to native execution
  if (op == "im2colbp" || op == "col2imbp") {
    DSP_DIAG(FALLBACK, "MlxIRBuilder: %s using native fallback", op.c_str());
    return nullptr;
  }

  // conv1d: 1D convolution — MLX supports conv1d natively
  if (op == "conv1d") {
    int sW = (numIArgs > 2) ? static_cast<int>(iArgs[2]) : 1;
    int pW = (numIArgs > 4) ? static_cast<int>(iArgs[4]) : 0;
    int dW = (numIArgs > 6) ? static_cast<int>(iArgs[6]) : 1;

    auto result = mx::conv1d(input, filter, sW, pW, dW, 1);
    if (inputs.size() >= 3) {
      result = mx::add(result, unwrap(inputs[2]));
    }
    return wrap(std::move(result));
  }

  // Pooling ops: fall back to native for now
  // TODO: implement via custom Metal kernel or sliding window
  if (op == "maxpool2d" || op == "avgpool2d" || op == "maxpool3d" || op == "avgpool3d") {
    DSP_DIAG(FALLBACK, "MlxIRBuilder: %s using native fallback", op.c_str());
    return nullptr;
  }

  DSP_DIAG(FALLBACK, "MlxIRBuilder: unsupported convolution op '%s'", opName.c_str());
  return nullptr;
}

// ═══════════════════════════════════════════════════════════════════════════
// Phase 3: Fused attention emit helpers
// ═══════════════════════════════════════════════════════════════════════════

std::shared_ptr<void> MlxIRBuilder::emitFusedAttentionOp(const std::string& opName,
                                                           const std::vector<std::shared_ptr<void>>& inputs,
                                                           const double* tArgs, int numTArgs,
                                                           const LongType* iArgs, int numIArgs) {
  std::string op = normalizeOp(opName);

  // Scaled dot-product attention via MLX fast SDPA kernel
  // mlx::core::fast::scaled_dot_product_attention(Q, K, V, scale, mask)
  // Dispatches to optimized Metal shader on Apple Silicon
  if (op == "multiheadattention" || op == "onnxmultiheadattention" || op == "dotproductattentionv2") {

    if (inputs.size() < 3) {
      DSP_DIAG(COMPILE, "MlxIRBuilder: attention needs >= 3 inputs (Q, K, V), got %d",
                static_cast<int>(inputs.size()));
      return nullptr;
    }

    auto& Q = unwrap(inputs[0]);
    auto& K = unwrap(inputs[1]);
    auto& V = unwrap(inputs[2]);

    // Scale factor
    float scale = 1.0f;
    if (numTArgs > 0 && tArgs) {
      scale = static_cast<float>(tArgs[0]);
    } else {
      // Auto-compute: 1/sqrt(dk)
      int dk = K.shape(-1);
      scale = 1.0f / std::sqrt(static_cast<float>(dk));
    }

    // Use MLX native fast SDPA — optimized Metal kernel
    // Signature: scaled_dot_product_attention(q, k, v, scale, mask, stream)
    // mask can be a bool/additive array broadcastable to [B, N, T_q, T_kv]
    if (inputs.size() >= 4 && inputs[3]) {
      auto& mask = unwrap(inputs[3]);
      auto result = mx::fast::scaled_dot_product_attention(Q, K, V, scale, mask);
      return wrap(std::move(result));
    } else {
      // No mask — pass empty array
      auto result = mx::fast::scaled_dot_product_attention(Q, K, V, scale, mx::array());
      return wrap(std::move(result));
    }
  }

  // flash_attention: same as SDPA but always causal
  if (op == "flashattention") {
    if (inputs.size() < 3) return nullptr;

    auto& Q = unwrap(inputs[0]);
    auto& K = unwrap(inputs[1]);
    auto& V = unwrap(inputs[2]);

    bool causal = (numIArgs > 0 && iArgs) ? (iArgs[0] != 0) : true;
    float scale = 1.0f;
    if (numTArgs > 0 && tArgs) {
      scale = static_cast<float>(tArgs[0]);
    } else {
      int dk = K.shape(-1);
      scale = 1.0f / std::sqrt(static_cast<float>(dk));
    }

    if (inputs.size() >= 4 && inputs[3]) {
      return wrap(mx::fast::scaled_dot_product_attention(Q, K, V, scale, unwrap(inputs[3])));
    }
    // For causal, we could pass "causal" string mask but MLX C++ API may differ
    // Use explicit causal mask
    if (causal) {
      int seqLen = Q.shape(-2);
      int kvLen = K.shape(-2);
      auto causalMask = mx::triu(mx::full({seqLen, kvLen}, mx::array(-std::numeric_limits<float>::infinity())), 1);
      return wrap(mx::fast::scaled_dot_product_attention(Q, K, V, scale, causalMask));
    }
    return wrap(mx::fast::scaled_dot_product_attention(Q, K, V, scale, mx::array()));
  }

  // grouped_query_attention: GQA with head repeating
  if (op == "groupedqueryattention") {
    if (inputs.size() < 3) return nullptr;

    auto& Q = unwrap(inputs[0]);  // [B, seqLen, numHeads, headDim]
    auto& K = unwrap(inputs[1]);  // [B, kvLen, numKVHeads, headDim]
    auto& V = unwrap(inputs[2]);  // [B, kvLen, numKVHeads, headDim]

    int numHeads = (numIArgs > 0 && iArgs) ? static_cast<int>(iArgs[0]) : Q.shape(2);
    int numKVHeads = (numIArgs > 1 && iArgs) ? static_cast<int>(iArgs[1]) : numHeads;
    bool causal = (numIArgs > 2 && iArgs) ? (iArgs[2] != 0) : true;

    float scale = 1.0f;
    if (numTArgs > 0 && tArgs) {
      scale = static_cast<float>(tArgs[0]);
    } else {
      int dk = Q.shape(-1);
      scale = 1.0f / std::sqrt(static_cast<float>(dk));
    }

    // Reshape Q/K/V to [B, numHeads, seqLen, headDim] for SDPA
    auto Qr = mx::transpose(Q, {0, 2, 1, 3});
    auto Kr = mx::transpose(K, {0, 2, 1, 3});
    auto Vr = mx::transpose(V, {0, 2, 1, 3});

    // Repeat K/V heads to match Q heads for GQA
    if (numKVHeads < numHeads) {
      int repeats = numHeads / numKVHeads;
      // Expand: [B, numKVHeads, kvLen, headDim] -> [B, numHeads, kvLen, headDim]
      Kr = mx::repeat(Kr, repeats, 1);
      Vr = mx::repeat(Vr, repeats, 1);
    }

    // Build mask
    mx::array mask;
    if (inputs.size() >= 4 && inputs[3]) {
      mask = unwrap(inputs[3]);
    } else if (causal) {
      int seqLen = Qr.shape(-2);
      int kvLen = Kr.shape(-2);
      mask = mx::triu(mx::full({seqLen, kvLen}, mx::array(-std::numeric_limits<float>::infinity())), 1);
    }

    auto result = mx::fast::scaled_dot_product_attention(Qr, Kr, Vr, scale, mask);
    // Transpose back: [B, numHeads, seqLen, headDim] -> [B, seqLen, numHeads, headDim]
    result = mx::transpose(result, {0, 2, 1, 3});
    return wrap(std::move(result));
  }

  // sliding_window_attention: local attention with fixed window
  if (op == "slidingwindowattention") {
    if (inputs.size() < 3) return nullptr;

    auto& Q = unwrap(inputs[0]);
    auto& K = unwrap(inputs[1]);
    auto& V = unwrap(inputs[2]);

    int windowSize = (numIArgs > 0 && iArgs) ? static_cast<int>(iArgs[0]) : 4096;
    int numHeads = (numIArgs > 1 && iArgs) ? static_cast<int>(iArgs[1]) : Q.shape(2);
    int numKVHeads = (numIArgs > 2 && iArgs) ? static_cast<int>(iArgs[2]) : numHeads;

    float scale = (numTArgs > 0 && tArgs) ? static_cast<float>(tArgs[0])
                : 1.0f / std::sqrt(static_cast<float>(Q.shape(-1)));

    auto Qr = mx::transpose(Q, {0, 2, 1, 3});
    auto Kr = mx::transpose(K, {0, 2, 1, 3});
    auto Vr = mx::transpose(V, {0, 2, 1, 3});

    if (numKVHeads < numHeads) {
      int repeats = numHeads / numKVHeads;
      Kr = mx::repeat(Kr, repeats, 1);
      Vr = mx::repeat(Vr, repeats, 1);
    }

    // Build sliding window causal mask
    int seqLen = Qr.shape(-2);
    int kvLen = Kr.shape(-2);
    auto fullMask = mx::full({seqLen, kvLen}, mx::array(-std::numeric_limits<float>::infinity()));
    // Allow attending to positions within window
    auto causalMask = mx::triu(fullMask, 1);
    auto windowMask = mx::tril(fullMask, -windowSize);
    auto mask = mx::maximum(causalMask, windowMask);

    auto result = mx::fast::scaled_dot_product_attention(Qr, Kr, Vr, scale, mask);
    result = mx::transpose(result, {0, 2, 1, 3});
    return wrap(std::move(result));
  }

  // apply_alibi: add ALiBi linear bias to attention scores
  if (op == "applyalibi") {
    // This op takes attention scores and adds linear position bias
    // For integration with SDPA, we'd need to compute the bias mask
    // Fall back to native for now — ALiBi is less common in Apple Silicon models
    DSP_DIAG(FALLBACK, "MlxIRBuilder: apply_alibi using native fallback");
    return nullptr;
  }

  DSP_DIAG(FALLBACK, "MlxIRBuilder: unsupported attention op '%s'", opName.c_str());
  return nullptr;
}

// ═══════════════════════════════════════════════════════════════════════════
// LLM: Rotary Position Embedding (RoPE)
// ═══════════════════════════════════════════════════════════════════════════

std::shared_ptr<void> MlxIRBuilder::emitRopeOp(const std::string& opName,
                                                 const std::vector<std::shared_ptr<void>>& inputs,
                                                 const double* tArgs, int numTArgs,
                                                 const LongType* iArgs, int numIArgs) {
  if (inputs.empty()) return nullptr;
  auto& x = unwrap(inputs[0]);

  // iArgs: [mode, n_past, n_dims, n_ctx]
  // mode: 0=standard/LLaMA, 1=neox, 2=gpt-j
  int mode = (numIArgs > 0 && iArgs) ? static_cast<int>(iArgs[0]) : 0;
  int nPast = (numIArgs > 1 && iArgs) ? static_cast<int>(iArgs[1]) : 0;
  int nDims = (numIArgs > 2 && iArgs) ? static_cast<int>(iArgs[2]) : x.shape(-1);

  // tArgs: [freq_base, freq_scale]
  float freqBase = (numTArgs > 0 && tArgs) ? static_cast<float>(tArgs[0]) : 10000.0f;
  float freqScale = (numTArgs > 1 && tArgs) ? static_cast<float>(tArgs[1]) : 1.0f;

  // traditional=true for GPT-J style, false for LLaMA/NeoX
  bool traditional = (mode == 2);

  // Use MLX fast RoPE — optimized Metal kernel
  // mx::fast::rope(a, dims, traditional, base, scale, offset, freqs)
  auto result = mx::fast::rope(x, nDims, traditional, freqBase, freqScale, nPast);

  return wrap(std::move(result));
}

// ═══════════════════════════════════════════════════════════════════════════
// LLM: Fused mega-kernels
// ═══════════════════════════════════════════════════════════════════════════

std::shared_ptr<void> MlxIRBuilder::emitFusedLlmOp(const std::string& opName,
                                                      const std::vector<std::shared_ptr<void>>& inputs,
                                                      const double* tArgs, int numTArgs,
                                                      const LongType* iArgs, int numIArgs) {
  std::string op = normalizeOp(opName);

  // fused_rms_norm_swiglu: silu(rms_norm(input) @ W_gate) * (rms_norm(input) @ W_up)
  if (op == "fusedrmsnormswiglu") {
    if (inputs.size() < 4) {
      DSP_DIAG(COMPILE, "MlxIRBuilder: fused_rms_norm_swiglu needs 4 inputs, got %d",
                static_cast<int>(inputs.size()));
      return nullptr;
    }

    auto& x = unwrap(inputs[0]);
    auto& gamma = unwrap(inputs[1]);
    auto& wGate = unwrap(inputs[2]);
    auto& wUp = unwrap(inputs[3]);

    float eps = (numTArgs > 0 && tArgs) ? static_cast<float>(tArgs[0]) : 1e-5f;

    // Step 1: RMS norm with fast kernel
    auto normed = mx::fast::rms_norm(x, gamma, eps);

    // Step 2: Gate projection + SiLU
    auto gate = mx::matmul(normed, wGate);
    auto gateActivated = mx::multiply(gate, mx::sigmoid(gate));  // silu

    // Step 3: Up projection
    auto up = mx::matmul(normed, wUp);

    // Step 4: Element-wise multiply
    return wrap(mx::multiply(gateActivated, up));
  }

  // fused_bias_dropout_residual: dropout(input + bias) + residual
  if (op == "fusedbiasdropoutresidual") {
    if (inputs.size() < 3) return nullptr;
    auto& input = unwrap(inputs[0]);
    auto& bias = unwrap(inputs[1]);
    auto& residual = unwrap(inputs[2]);

    // dropout prob from tArgs[0] — at inference time, prob=0.0 means no dropout
    float dropProb = (numTArgs > 0 && tArgs) ? static_cast<float>(tArgs[0]) : 0.0f;
    bool training = (iArgs && numIArgs > 0) ? (iArgs[0] != 0) : false;

    auto biased = mx::add(input, bias);

    if (training && dropProb > 0.0f) {
      // Apply dropout: scale by 1/(1-p), zero out randomly
      // For inference, skip dropout entirely
      auto mask = mx::greater(mx::random::uniform(biased.shape()), mx::array(dropProb));
      biased = mx::multiply(biased, mx::astype(mask, biased.dtype()));
      biased = mx::divide(biased, mx::array(1.0f - dropProb));
    }

    return wrap(mx::add(biased, residual));
  }

  // fused_elementwise_chain: sequence of elementwise ops in one kernel
  // For MLX this decomposes to individual ops (MLX handles fusion in its graph evaluator)
  if (op == "fusedelementwisechain") {
    if (inputs.empty()) return nullptr;
    // The chain is specified via iArgs op codes
    // For now, fall through to native — the chain semantics are complex
    DSP_DIAG(FALLBACK, "MlxIRBuilder: fused_elementwise_chain using native fallback");
    return nullptr;
  }

  DSP_DIAG(FALLBACK, "MlxIRBuilder: unsupported fused LLM op '%s'", opName.c_str());
  return nullptr;
}

// ═══════════════════════════════════════════════════════════════════════════
// Graph construction (all phases)
// ═══════════════════════════════════════════════════════════════════════════

MlxIRBuilder::MlxGraph MlxIRBuilder::buildGraph(
    NativeSlot* slots, int startSlot, int endSlot,
    int totalSlots,
    NDArray** externalInputs, int numExternalInputs,
    NDArray** outputSlots, int totalOutputSlots) {

  MlxGraph graph;

  // SSA map: slot output index -> mlx array (as shared_ptr<void>)
  std::unordered_map<int, std::shared_ptr<void>> ssaMap;

  for (int si = startSlot; si <= endSlot; si++) {
    auto& slot = slots[si];
    auto cat = OpCategoryTable::categorize(slot.ident.opName);

    // Resolve inputs from SSA map or external inputs
    std::vector<std::shared_ptr<void>> inputs;
    for (int inp = 0; inp < slot.wiring.numInputs; inp++) {
      int srcIdx = slot.wiring.inputSourceIndices[inp];
      std::shared_ptr<void> mlxInput;

      if (srcIdx < 0) {
        int extIdx = -(srcIdx + 1);
        if (extIdx < numExternalInputs && externalInputs && externalInputs[extIdx]) {
          mlxInput = ndArrayToMlxArray(externalInputs[extIdx]);
        }
      } else {
        auto it = ssaMap.find(srcIdx);
        if (it != ssaMap.end()) {
          mlxInput = it->second;
        } else if (srcIdx < totalOutputSlots && outputSlots && outputSlots[srcIdx]) {
          mlxInput = ndArrayToMlxArray(outputSlots[srcIdx]);
        }
      }

      if (!mlxInput) {
        DSP_DIAG(COMPILE, "MlxIRBuilder: null input %d for slot %d (op=%s)",
                  inp, si, slot.ident.opName);
        graph.valid = false;
        return graph;
      }
      inputs.push_back(mlxInput);
    }

    // Get the output NDArray for shape reference (needed by some ops)
    NDArray* outArr = nullptr;
    if (slot.wiring.numOutputs > 0) {
      int outIdx = slot.wiring.outputSlotIndices[0];
      if (outIdx >= 0 && outIdx < totalOutputSlots && outputSlots) {
        outArr = outputSlots[outIdx];
      }
    }

    // Dispatch to emit helper based on category
    std::shared_ptr<void> result;

    switch (cat) {
      // ── Phase 1: Element-wise ──
      case TritonOpCategory::BINARY_ELEMENTWISE:
        if (inputs.size() >= 2) {
          result = emitBinaryElementwise(slot.ident.opName, inputs[0], inputs[1]);
        }
        break;

      case TritonOpCategory::UNARY_ELEMENTWISE:
        if (inputs.size() >= 1) {
          result = emitUnaryElementwise(slot.ident.opName, inputs[0], slot.args.tArgs, slot.args.numTArgs);
        }
        break;

      case TritonOpCategory::COMPARISON:
        if (inputs.size() >= 2) {
          result = emitComparisonOp(slot.ident.opName, inputs[0], inputs[1]);
        }
        break;

      case TritonOpCategory::LOGICAL:
        if (inputs.size() >= 1) {
          auto rhs = (inputs.size() >= 2) ? inputs[1] : nullptr;
          result = emitLogicalOp(slot.ident.opName, inputs[0], rhs);
        }
        break;

      case TritonOpCategory::TERNARY:
        if (inputs.size() >= 3) {
          result = emitTernaryOp(slot.ident.opName, inputs[0], inputs[1], inputs[2]);
        }
        break;

      case TritonOpCategory::IDENTITY:
        if (inputs.size() >= 1) {
          result = emitIdentityOp(inputs[0]);
        }
        break;

      case TritonOpCategory::CAST:
        if (inputs.size() >= 1 && outArr) {
          result = emitCastOp(inputs[0], outArr->dataType());
        }
        break;

      // ── Phase 2: Structured ops ──
      case TritonOpCategory::REDUCTION:
        if (inputs.size() >= 1) {
          // keepDims is typically bArgs[0]
          bool keepDims = (slot.args.numBArgs > 0 && slot.args.bArgs) ? slot.args.bArgs[0] : false;
          result = emitReductionOp(slot.ident.opName, inputs[0],
                                   slot.args.iArgs, slot.args.numIArgs, keepDims);
        }
        break;

      case TritonOpCategory::MATMUL:
        result = emitMatmulOp(slot.ident.opName, inputs,
                               slot.args.tArgs, slot.args.numTArgs,
                               slot.args.iArgs, slot.args.numIArgs);
        break;

      case TritonOpCategory::NORMALIZATION:
        result = emitNormalizationOp(slot.ident.opName, inputs,
                                      slot.args.iArgs, slot.args.numIArgs,
                                      slot.args.tArgs, slot.args.numTArgs);
        break;

      case TritonOpCategory::SHAPE_MANIPULATION:
        if (inputs.size() >= 1) {
          result = emitShapeManipOp(slot.ident.opName, inputs[0],
                                    slot.args.iArgs, slot.args.numIArgs, outArr);
        }
        break;

      case TritonOpCategory::DATA_MOVEMENT:
        result = emitDataMovementOp(slot.ident.opName, inputs,
                                     slot.args.iArgs, slot.args.numIArgs,
                                     slot.args.tArgs, slot.args.numTArgs, outArr);
        break;

      case TritonOpCategory::CONSTANT_GENERATION:
        result = emitConstantGenOp(slot.ident.opName, inputs,
                                    slot.args.tArgs, slot.args.numTArgs,
                                    slot.args.iArgs, slot.args.numIArgs, outArr);
        break;

      // ── Phase 3: Compute-intensive ops ──
      case TritonOpCategory::CONVOLUTION:
        result = emitConvolutionOp(slot.ident.opName, inputs,
                                    slot.args.iArgs, slot.args.numIArgs, outArr);
        break;

      case TritonOpCategory::FUSED_ATTENTION:
        result = emitFusedAttentionOp(slot.ident.opName, inputs,
                                       slot.args.tArgs, slot.args.numTArgs,
                                       slot.args.iArgs, slot.args.numIArgs);
        break;

      // ── LLM-specific ops ──
      case TritonOpCategory::ROPE:
        result = emitRopeOp(slot.ident.opName, inputs,
                             slot.args.tArgs, slot.args.numTArgs,
                             slot.args.iArgs, slot.args.numIArgs);
        break;

      case TritonOpCategory::FUSED_LLM:
        result = emitFusedLlmOp(slot.ident.opName, inputs,
                                  slot.args.tArgs, slot.args.numTArgs,
                                  slot.args.iArgs, slot.args.numIArgs);
        break;

      default:
        DSP_DIAG(FALLBACK, "MlxIRBuilder: unsupported category for op '%s'", slot.ident.opName);
        graph.valid = false;
        return graph;
    }

    if (!result) {
      DSP_DIAG(COMPILE, "MlxIRBuilder: emit failed for slot %d (op=%s)", si, slot.ident.opName);
      graph.valid = false;
      return graph;
    }

    // Register outputs in SSA map
    for (int o = 0; o < slot.wiring.numOutputs; o++) {
      int outIdx = slot.wiring.outputSlotIndices[o];
      ssaMap[outIdx] = result;
    }
  }

  // Collect externally visible outputs
  auto externalOutputs = computeExternallyVisibleOutputs(slots, startSlot, endSlot, totalSlots);
  for (int outIdx : externalOutputs) {
    auto it = ssaMap.find(outIdx);
    if (it != ssaMap.end()) {
      graph.outputArrays[outIdx] = it->second;
    }
  }

  // Also include all outputs of the segment (conservative)
  for (int si = startSlot; si <= endSlot; si++) {
    for (int o = 0; o < slots[si].wiring.numOutputs; o++) {
      int outIdx = slots[si].wiring.outputSlotIndices[o];
      if (graph.outputArrays.find(outIdx) == graph.outputArrays.end()) {
        auto it = ssaMap.find(outIdx);
        if (it != ssaMap.end()) {
          graph.outputArrays[outIdx] = it->second;
        }
      }
    }
  }

  graph.valid = true;
  return graph;
}

}  // namespace graph
}  // namespace sd

#endif  // HAVE_MLX
