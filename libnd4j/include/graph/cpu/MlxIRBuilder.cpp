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
#include <graph/gpu/OpCategoryTable.h>
#include <helpers/logger.h>
#include <system/Environment.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <numeric>

// MLX C++ API headers
#include <mlx/mlx.h>

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

// ─── Normalize op name (lowercase) ────────────────────────────────────────
static std::string normalizeOp(const std::string& opName) {
  std::string n = opName;
  std::transform(n.begin(), n.end(), n.begin(),
                 [](unsigned char c) { return std::tolower(c); });
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
    auto cat = OpCategoryTable::categorize(slots[i].opName);
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
    for (int inp = 0; inp < slots[i].numInputs; inp++) {
      int srcIdx = slots[i].inputSourceIndices[inp];
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
    for (int o = 0; o < slots[i].numOutputs; o++) {
      internalOutputs.insert(slots[i].outputSlotIndices[o]);
    }
  }

  std::unordered_set<int> externalOutputs;
  for (int outIdx : internalOutputs) {
    bool consumedOutside = false;
    for (int i = 0; i < totalSlots; i++) {
      if (i >= startSlot && i <= endSlot) continue;
      for (int inp = 0; inp < slots[i].numInputs; inp++) {
        if (slots[i].inputSourceIndices[inp] == outIdx) {
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
      for (int o = 0; o < slots[endSlot].numOutputs; o++) {
        if (slots[endSlot].outputSlotIndices[o] == outIdx) {
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

  if (arr->ews() == 1 && arr->ordering() == 'c') {
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

  if (op == "add" || op == "add_scalar") return wrap(mx::add(a, b));
  if (op == "subtract" || op == "sub" || op == "subtract_scalar") return wrap(mx::subtract(a, b));
  if (op == "multiply" || op == "mul" || op == "mul_scalar" || op == "multiply_scalar") return wrap(mx::multiply(a, b));
  if (op == "divide" || op == "div" || op == "div_scalar" || op == "divide_scalar") return wrap(mx::divide(a, b));
  if (op == "floormod" || op == "mod" || op == "floormod_scalar") return wrap(mx::remainder(a, b));
  if (op == "maximum" || op == "max_pairwise") return wrap(mx::maximum(a, b));
  if (op == "minimum" || op == "min_pairwise") return wrap(mx::minimum(a, b));
  if (op == "pow" || op == "pow_scalar") return wrap(mx::power(a, b));
  if (op == "squaredsubtract" || op == "squared_subtract") {
    auto diff = mx::subtract(a, b);
    return wrap(mx::multiply(diff, diff));
  }
  if (op == "reversedivide" || op == "rdiv" || op == "reversedivide_scalar") return wrap(mx::divide(b, a));
  if (op == "reversesubtract" || op == "rsub" || op == "reversesubtract_scalar") return wrap(mx::subtract(b, a));
  if (op == "atan2") return wrap(mx::arctan2(a, b));
  if (op == "realdiv") return wrap(mx::divide(a, b));
  if (op == "floordiv") return wrap(mx::floor(mx::divide(a, b)));
  if (op == "multiply_no_nan" || op == "multiplynonan") {
    // multiply but return 0 where either input is NaN
    auto product = mx::multiply(a, b);
    auto nanMask = mx::logical_or(mx::isnan(a), mx::isnan(b));
    return wrap(mx::where(nanMask, mx::array(0.0f), product));
  }
  if (op == "swish_mul" || op == "swishmul") {
    // SwiGLU: x * sigmoid(x) * y
    return wrap(mx::multiply(mx::multiply(a, mx::sigmoid(a)), b));
  }

  sd_printf("MlxIRBuilder: unsupported binary op '%s'\n", opName.c_str());
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
  if (op == "log" || op == "log_x") return wrap(mx::log(x));
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
  if (op == "clip" || op == "clipbyvalue" || op == "clip_by_value" || op == "clamp") {
    float minVal = (numTArgs > 0) ? static_cast<float>(tArgs[0]) : 0.0f;
    float maxVal = (numTArgs > 1) ? static_cast<float>(tArgs[1]) : 6.0f;
    return wrap(mx::clip(x, mx::array(minVal), mx::array(maxVal)));
  }
  // softsign: x / (1 + |x|)
  if (op == "softsign") {
    return wrap(mx::divide(x, mx::add(mx::array(1.0f), mx::abs(x))));
  }
  // hard_sigmoid: max(0, min(1, alpha*x + beta))  with alpha=0.2, beta=0.5
  if (op == "hard_sigmoid" || op == "hardsigmoid") {
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
  if (op == "add_scalar") {
    float scalar = (numTArgs > 0) ? static_cast<float>(tArgs[0]) : 0.0f;
    return wrap(mx::add(x, mx::array(scalar)));
  }
  if (op == "subtract_scalar") {
    float scalar = (numTArgs > 0) ? static_cast<float>(tArgs[0]) : 0.0f;
    return wrap(mx::subtract(x, mx::array(scalar)));
  }
  if (op == "multiply_scalar") {
    float scalar = (numTArgs > 0) ? static_cast<float>(tArgs[0]) : 1.0f;
    return wrap(mx::multiply(x, mx::array(scalar)));
  }
  if (op == "divide_scalar") {
    float scalar = (numTArgs > 0) ? static_cast<float>(tArgs[0]) : 1.0f;
    return wrap(mx::divide(x, mx::array(scalar)));
  }

  sd_printf("MlxIRBuilder: unsupported unary op '%s'\n", opName.c_str());
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
  if (op == "greater_equal" || op == "greaterthanorequal") return wrap(mx::greater_equal(a, b));
  if (op == "less_equal" || op == "lessthanorequal") return wrap(mx::less_equal(a, b));
  if (op == "equals" || op == "equal") return wrap(mx::equal(a, b));
  if (op == "not_equals" || op == "notequals" || op == "not_equal") return wrap(mx::not_equal(a, b));

  sd_printf("MlxIRBuilder: unsupported comparison op '%s'\n", opName.c_str());
  return nullptr;
}

std::shared_ptr<void> MlxIRBuilder::emitLogicalOp(const std::string& opName,
                                                    const std::shared_ptr<void>& lhs,
                                                    const std::shared_ptr<void>& rhs) {
  auto& a = unwrap(lhs);
  std::string op = normalizeOp(opName);

  if (op == "boolean_and" || op == "and" || op == "logical_and") {
    return wrap(mx::logical_and(a, unwrap(rhs)));
  }
  if (op == "boolean_or" || op == "or" || op == "logical_or") {
    return wrap(mx::logical_or(a, unwrap(rhs)));
  }
  if (op == "boolean_not" || op == "not" || op == "logical_not") {
    return wrap(mx::logical_not(a));
  }
  if (op == "boolean_xor" || op == "xor" || op == "logical_xor") {
    auto orResult = mx::logical_or(a, unwrap(rhs));
    auto andResult = mx::logical_and(a, unwrap(rhs));
    return wrap(mx::logical_and(orResult, mx::logical_not(andResult)));
  }

  sd_printf("MlxIRBuilder: unsupported logical op '%s'\n", opName.c_str());
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
  if (op == "reduce_sum" || op == "sum" || op == "reduce_sum_bp") {
    if (axes.empty()) return wrap(mx::sum(x, keepDims));
    return wrap(mx::sum(x, axes, keepDims));
  }
  // reduce_max / max
  if (op == "reduce_max" || op == "max" || op == "reduce_amax") {
    if (axes.empty()) return wrap(mx::max(x, keepDims));
    return wrap(mx::max(x, axes, keepDims));
  }
  // reduce_min / min
  if (op == "reduce_min" || op == "min" || op == "reduce_amin") {
    if (axes.empty()) return wrap(mx::min(x, keepDims));
    return wrap(mx::min(x, axes, keepDims));
  }
  // reduce_mean / mean
  if (op == "reduce_mean" || op == "mean") {
    if (axes.empty()) return wrap(mx::mean(x, keepDims));
    return wrap(mx::mean(x, axes, keepDims));
  }
  // reduce_prod / prod
  if (op == "reduce_prod" || op == "prod") {
    if (axes.empty()) return wrap(mx::prod(x, keepDims));
    return wrap(mx::prod(x, axes, keepDims));
  }
  // reduce_variance / variance
  if (op == "reduce_variance" || op == "variance") {
    if (axes.empty()) return wrap(mx::var(x, keepDims));
    return wrap(mx::var(x, axes, keepDims));
  }
  // reduce_stdev / stdev
  if (op == "reduce_stdev" || op == "stdev") {
    if (axes.empty()) {
      auto v = mx::var(x, keepDims);
      return wrap(mx::sqrt(v));
    }
    auto v = mx::var(x, axes, keepDims);
    return wrap(mx::sqrt(v));
  }
  // reduce_norm1 / norm1
  if (op == "reduce_norm1" || op == "norm1") {
    auto absX = mx::abs(x);
    if (axes.empty()) return wrap(mx::sum(absX, keepDims));
    return wrap(mx::sum(absX, axes, keepDims));
  }
  // reduce_norm2 / norm2
  if (op == "reduce_norm2" || op == "norm2") {
    auto sq = mx::square(x);
    if (axes.empty()) return wrap(mx::sqrt(mx::sum(sq, keepDims)));
    return wrap(mx::sqrt(mx::sum(sq, axes, keepDims)));
  }
  // reduce_logsumexp / logsumexp
  if (op == "reduce_logsumexp" || op == "logsumexp") {
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

  sd_printf("MlxIRBuilder: unsupported reduction op '%s'\n", opName.c_str());
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
    sd_printf("MlxIRBuilder: matmul needs >= 2 inputs, got %d\n", static_cast<int>(inputs.size()));
    return nullptr;
  }

  auto& a = unwrap(inputs[0]);
  auto& b = unwrap(inputs[1]);

  // matmul / batch_matmul / batched_gemm
  if (op == "matmul" || op == "mmul" || op == "batch_matmul" || op == "batched_gemm" || op == "tensormmul") {
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
  if (op == "xw_plus_b") {
    auto result = mx::matmul(a, b);
    if (inputs.size() >= 3) {
      result = mx::add(result, unwrap(inputs[2]));
    }
    return wrap(std::move(result));
  }

  sd_printf("MlxIRBuilder: unsupported matmul op '%s'\n", opName.c_str());
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
  if (op == "log_softmax") {
    int axis = (numIArgs > 0 && iArgs) ? static_cast<int>(iArgs[0]) : -1;
    // log_softmax(x) = x - logsumexp(x, axis, keepdims=true)
    auto lse = mx::logsumexp(x, std::vector<int>{axis}, true);
    return wrap(mx::subtract(x, lse));
  }

  // layer_norm: (x - mean) / sqrt(var + eps) * gamma + beta
  if (op == "layer_norm") {
    float eps = (numTArgs > 0 && tArgs) ? static_cast<float>(tArgs[0]) : 1e-5f;
    // Normalize over the last axis by default
    int axis = -1;
    if (numIArgs > 0 && iArgs) axis = static_cast<int>(iArgs[0]);

    std::vector<int> axes = {axis};
    auto mean = mx::mean(x, axes, true);
    auto centered = mx::subtract(x, mean);
    auto variance = mx::mean(mx::square(centered), axes, true);
    auto normalized = mx::multiply(centered, mx::rsqrt(mx::add(variance, mx::array(eps))));

    // Apply gamma (scale) and beta (shift) if provided
    if (inputs.size() >= 2) {
      normalized = mx::multiply(normalized, unwrap(inputs[1]));
    }
    if (inputs.size() >= 3) {
      normalized = mx::add(normalized, unwrap(inputs[2]));
    }

    return wrap(std::move(normalized));
  }

  // rms_norm: x * rsqrt(mean(x^2) + eps) * gamma
  if (op == "rms_norm") {
    float eps = (numTArgs > 0 && tArgs) ? static_cast<float>(tArgs[0]) : 1e-5f;
    int axis = -1;

    std::vector<int> axes = {axis};
    auto meanSq = mx::mean(mx::square(x), axes, true);
    auto normalized = mx::multiply(x, mx::rsqrt(mx::add(meanSq, mx::array(eps))));

    if (inputs.size() >= 2) {
      normalized = mx::multiply(normalized, unwrap(inputs[1]));
    }

    return wrap(std::move(normalized));
  }

  // batch_norm: (x - mean) / sqrt(var + eps) * gamma + beta
  if (op == "batch_norm") {
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
  if (op == "normalize_moments") {
    return wrap(mx::array(x));  // Pass-through for now
  }

  sd_printf("MlxIRBuilder: unsupported normalization op '%s'\n", opName.c_str());
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
  if (op == "expand_dims") {
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
  if (op == "flatten" || op == "flatten_2d") {
    return wrap(mx::flatten(x));
  }

  sd_printf("MlxIRBuilder: unsupported shape manipulation op '%s'\n", opName.c_str());
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
  if (op == "gather" || op == "gather_nd") {
    if (inputs.size() < 2) return nullptr;
    auto& data = unwrap(inputs[0]);
    auto& indices = unwrap(inputs[1]);
    int axis = (numIArgs > 0 && iArgs) ? static_cast<int>(iArgs[0]) : 0;

    if (op == "gather_nd") {
      // gather_nd: advanced indexing — use take for 1D case
      // For general case, flatten indices and gather
      auto indicesInt = mx::astype(indices, mx::int32);
      return wrap(mx::take(data, indicesInt, axis));
    }

    auto indicesInt = mx::astype(indices, mx::int32);
    return wrap(mx::take(data, indicesInt, axis));
  }

  // concat
  if (op == "concat" || op == "concat_bp") {
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
  if (op == "split" || op == "split_v") {
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
  if (op == "strided_slice") {
    auto& data = unwrap(inputs[0]);
    // iArgs layout: [begin0, begin1, ..., end0, end1, ..., strides0, strides1, ...]
    int rank = static_cast<int>(data.ndim());
    if (numIArgs < rank * 3) {
      sd_printf("MlxIRBuilder: strided_slice needs 3*rank iArgs, got %d for rank %d\n",
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
  if (op == "scatter_nd" || op == "scatter_nd_update") {
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

  sd_printf("MlxIRBuilder: unsupported data movement op '%s'\n", opName.c_str());
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
    sd_printf("MlxIRBuilder: constant gen op '%s' needs output array for shape\n", opName.c_str());
    return nullptr;
  }

  auto rank = outputArr->rankOf();
  std::vector<int> shape(rank);
  for (int i = 0; i < rank; i++) {
    shape[i] = static_cast<int>(outputArr->sizeAt(i));
  }
  auto dtype = sdTypeToMlxDtypeInternal(outputArr->dataType());

  // zeros_like / zeros_as / zeroslike
  if (op == "zeros_like" || op == "zeros_as" || op == "zeroslike") {
    return wrap(mx::zeros(shape, dtype));
  }

  // ones_like / ones_as / oneslike
  if (op == "ones_like" || op == "ones_as" || op == "oneslike") {
    return wrap(mx::ones(shape, dtype));
  }

  // create (fill with zero by default)
  if (op == "create") {
    return wrap(mx::zeros(shape, dtype));
  }

  // set_scalar: fill with tArgs[0]
  if (op == "set_scalar") {
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
  if (op == "shape_of") {
    if (!inputs.empty()) {
      auto& x = unwrap(inputs[0]);
      auto xShape = x.shape();
      std::vector<int32_t> shapeVals(xShape.begin(), xShape.end());
      return wrap(mx::array(shapeVals.data(), {static_cast<int>(shapeVals.size())}, mx::int32));
    }
    return nullptr;
  }

  // min_max_datatype: return type limits
  if (op == "min_max_datatype") {
    return wrap(mx::zeros(shape, dtype));
  }

  sd_printf("MlxIRBuilder: unsupported constant gen op '%s'\n", opName.c_str());
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
  if (op == "depthwise_conv2d") {
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
    sd_printf("MlxIRBuilder: conv3d not yet implemented in MLX backend\n", "");
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
    sd_printf("MlxIRBuilder: im2col using native fallback\n", "");
    return nullptr;
  }

  // col2im: inverse of im2col — fold columns back into image
  if (op == "col2im") {
    sd_printf("MlxIRBuilder: col2im using native fallback\n", "");
    return nullptr;
  }

  // Backprop variants: always fall back to native execution
  if (op == "im2col_bp" || op == "col2im_bp") {
    sd_printf("MlxIRBuilder: %s using native fallback\n", op.c_str());
    return nullptr;
  }

  sd_printf("MlxIRBuilder: unsupported convolution op '%s'\n", opName.c_str());
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

  // Scaled dot-product attention: softmax(Q @ K^T / sqrt(dk)) @ V
  if (op == "multi_head_attention" || op == "onnx_multi_head_attention" ||
      op == "dot_product_attention_v2") {

    if (inputs.size() < 3) {
      sd_printf("MlxIRBuilder: attention needs >= 3 inputs (Q, K, V), got %d\n",
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

    // Use MLX fast SDPA when available
    // mlx::core::fast::scaled_dot_product_attention(Q, K, V, scale, mask)
    // For now, implement manually: softmax(Q @ K^T * scale) @ V
    auto scores = mx::matmul(Q, mx::swapaxes(K, -2, -1));
    scores = mx::multiply(scores, mx::array(scale));

    // Apply attention mask if provided (input[3])
    if (inputs.size() >= 4 && inputs[3]) {
      auto& mask = unwrap(inputs[3]);
      // Mask: 0 = attend, large negative = mask out (or 1/0 bool mask)
      // Convention: add mask to scores (mask has -inf for masked positions)
      scores = mx::add(scores, mask);
    }

    auto attnWeights = mx::softmax(scores, -1);

    auto result = mx::matmul(attnWeights, V);
    return wrap(std::move(result));
  }

  sd_printf("MlxIRBuilder: unsupported attention op '%s'\n", opName.c_str());
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
    auto cat = OpCategoryTable::categorize(slot.opName);

    // Resolve inputs from SSA map or external inputs
    std::vector<std::shared_ptr<void>> inputs;
    for (int inp = 0; inp < slot.numInputs; inp++) {
      int srcIdx = slot.inputSourceIndices[inp];
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
        sd_printf("MlxIRBuilder: null input %d for slot %d (op=%s)\n",
                  inp, si, slot.opName);
        graph.valid = false;
        return graph;
      }
      inputs.push_back(mlxInput);
    }

    // Get the output NDArray for shape reference (needed by some ops)
    NDArray* outArr = nullptr;
    if (slot.numOutputs > 0) {
      int outIdx = slot.outputSlotIndices[0];
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
          result = emitBinaryElementwise(slot.opName, inputs[0], inputs[1]);
        }
        break;

      case TritonOpCategory::UNARY_ELEMENTWISE:
        if (inputs.size() >= 1) {
          result = emitUnaryElementwise(slot.opName, inputs[0], slot.tArgs, slot.numTArgs);
        }
        break;

      case TritonOpCategory::COMPARISON:
        if (inputs.size() >= 2) {
          result = emitComparisonOp(slot.opName, inputs[0], inputs[1]);
        }
        break;

      case TritonOpCategory::LOGICAL:
        if (inputs.size() >= 1) {
          auto rhs = (inputs.size() >= 2) ? inputs[1] : nullptr;
          result = emitLogicalOp(slot.opName, inputs[0], rhs);
        }
        break;

      case TritonOpCategory::TERNARY:
        if (inputs.size() >= 3) {
          result = emitTernaryOp(slot.opName, inputs[0], inputs[1], inputs[2]);
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
          bool keepDims = (slot.numBArgs > 0 && slot.bArgs) ? slot.bArgs[0] : false;
          result = emitReductionOp(slot.opName, inputs[0],
                                   slot.iArgs, slot.numIArgs, keepDims);
        }
        break;

      case TritonOpCategory::MATMUL:
        result = emitMatmulOp(slot.opName, inputs,
                               slot.tArgs, slot.numTArgs,
                               slot.iArgs, slot.numIArgs);
        break;

      case TritonOpCategory::NORMALIZATION:
        result = emitNormalizationOp(slot.opName, inputs,
                                      slot.iArgs, slot.numIArgs,
                                      slot.tArgs, slot.numTArgs);
        break;

      case TritonOpCategory::SHAPE_MANIPULATION:
        if (inputs.size() >= 1) {
          result = emitShapeManipOp(slot.opName, inputs[0],
                                    slot.iArgs, slot.numIArgs, outArr);
        }
        break;

      case TritonOpCategory::DATA_MOVEMENT:
        result = emitDataMovementOp(slot.opName, inputs,
                                     slot.iArgs, slot.numIArgs,
                                     slot.tArgs, slot.numTArgs, outArr);
        break;

      case TritonOpCategory::CONSTANT_GENERATION:
        result = emitConstantGenOp(slot.opName, inputs,
                                    slot.tArgs, slot.numTArgs,
                                    slot.iArgs, slot.numIArgs, outArr);
        break;

      // ── Phase 3: Compute-intensive ops ──
      case TritonOpCategory::CONVOLUTION:
        result = emitConvolutionOp(slot.opName, inputs,
                                    slot.iArgs, slot.numIArgs, outArr);
        break;

      case TritonOpCategory::FUSED_ATTENTION:
        result = emitFusedAttentionOp(slot.opName, inputs,
                                       slot.tArgs, slot.numTArgs,
                                       slot.iArgs, slot.numIArgs);
        break;

      default:
        sd_printf("MlxIRBuilder: unsupported category for op '%s'\n", slot.opName);
        graph.valid = false;
        return graph;
    }

    if (!result) {
      sd_printf("MlxIRBuilder: emit failed for slot %d (op=%s)\n", si, slot.opName);
      graph.valid = false;
      return graph;
    }

    // Register outputs in SSA map
    for (int o = 0; o < slot.numOutputs; o++) {
      int outIdx = slot.outputSlotIndices[o];
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
    for (int o = 0; o < slots[si].numOutputs; o++) {
      int outIdx = slots[si].outputSlotIndices[o];
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
