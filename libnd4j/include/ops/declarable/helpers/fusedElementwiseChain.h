/* ******************************************************************************
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
// @author Adam Gibson
//
// Fused element-wise chain kernel.
// Executes a chain of element-wise unary/binary ops in a single kernel pass,
// eliminating intermediate buffer allocations and global memory round-trips.
//

#ifndef LIBND4J_FUSED_ELEMENTWISE_CHAIN_H
#define LIBND4J_FUSED_ELEMENTWISE_CHAIN_H

#include <array/DataTypeUtils.h>
#include <ops/declarable/helpers/helpers.h>

#include <string>

namespace sd {
namespace ops {
namespace helpers {

/** Longest chain one fused kernel runs. */
constexpr int FUSED_CHAIN_MAX_OPS = 8;

/**
 * Op codes for the fused elementwise interpreter.
 * Each code maps to a simple element-wise operation applied sequentially.
 * The per-element math lives in fusedElementwiseChainMath.h.
 */
enum FusedElemOp : uint8_t {
    // Binary ops (use secondaryInput)
    FUSED_ADD = 0,
    FUSED_SUB = 1,
    FUSED_MUL = 2,
    FUSED_DIV = 3,

    // Unary ops
    FUSED_RELU = 10,
    FUSED_SIGMOID = 11,
    FUSED_TANH = 12,
    FUSED_GELU = 13,
    FUSED_EXP = 14,
    FUSED_LOG = 15,
    FUSED_ABS = 16,
    FUSED_NEG = 17,
    FUSED_SQUARE = 18,
    FUSED_SQRT = 19,
    FUSED_SWISH = 20,       // legacy swish transform: x * sigmoid(x), rounded once
    FUSED_SILU = 21,        // silu declarable: sigmoid(x) rounded, then multiplied by x
    FUSED_MISH = 22,

    // Additional unary ops
    FUSED_RSQRT = 23,
    FUSED_RECIPROCAL = 24,
    FUSED_SIGN = 25,
    FUSED_ERF = 26,
    FUSED_ERFC = 27,
    FUSED_LOG1P = 28,
    FUSED_CEIL = 29,

    // Parameterized ops
    FUSED_CLIP = 30,        // Uses clipMin/clipMax
    FUSED_LEAKY_RELU = 31,  // Binary: the secondary input is alpha

    // Additional unary ops (continued)
    FUSED_FLOOR = 32,
    FUSED_ROUND = 33,
    FUSED_SIN = 34,
    FUSED_COS = 35,
    FUSED_ELU = 36,         // alpha 1
    FUSED_SELU = 37,
    FUSED_SOFTPLUS = 38,
    FUSED_SOFTSIGN = 39,
    FUSED_HARD_SIGMOID = 40,
    FUSED_HARDTANH = 41,
    FUSED_RELU6 = 42,       // relu6 with threshold 0

    // Additional binary ops
    FUSED_MIN = 50,         // minimum/min_pairwise
    FUSED_MAX = 51,         // maximum/max_pairwise
    FUSED_MOD = 52,         // legacy Mod: x - floor(x / y) * y (not floormod)
    FUSED_ATAN2 = 53,
    FUSED_FLOORDIV = 54,
    FUSED_REVERSE_DIV = 55,
    FUSED_REVERSE_SUB = 56,
    FUSED_SQUARED_SUB = 57,
    FUSED_MUL_NO_NAN = 58,  // y == 0 ? 0 : x * y
    FUSED_POW = 59,
};

/**
 * Check if a FusedElemOp needs a secondary input value.
 * Binary arithmetic ops and leaky_relu (whose alpha is the secondary input).
 */
SD_HOST_DEVICE inline bool isBinaryFusedOp(FusedElemOp op) {
    return op <= FUSED_DIV || op == FUSED_LEAKY_RELU ||
           (op >= FUSED_MIN && op <= FUSED_POW);
}

/**
 * True for every code fusedChainStep() (fusedElementwiseChainMath.h) implements. Keep the two
 * lists identical: an unlisted code must fail here rather than pass through a kernel unchanged.
 */
inline bool isImplementedFusedOp(int code) {
  switch (code) {
    case FUSED_ADD: case FUSED_SUB: case FUSED_MUL: case FUSED_DIV:
    case FUSED_REVERSE_SUB: case FUSED_REVERSE_DIV: case FUSED_SQUARED_SUB:
    case FUSED_MIN: case FUSED_MAX: case FUSED_MOD: case FUSED_ATAN2: case FUSED_FLOORDIV:
    case FUSED_POW: case FUSED_MUL_NO_NAN: case FUSED_LEAKY_RELU:
    case FUSED_RELU: case FUSED_RELU6: case FUSED_ELU:
    case FUSED_SIGMOID: case FUSED_TANH: case FUSED_GELU: case FUSED_EXP: case FUSED_LOG:
    case FUSED_ABS: case FUSED_NEG: case FUSED_SQUARE: case FUSED_SQRT: case FUSED_RSQRT:
    case FUSED_RECIPROCAL: case FUSED_SIGN: case FUSED_ERF: case FUSED_ERFC: case FUSED_LOG1P:
    case FUSED_CEIL: case FUSED_FLOOR: case FUSED_ROUND: case FUSED_SIN: case FUSED_COS:
    case FUSED_SELU: case FUSED_SOFTPLUS: case FUSED_SOFTSIGN: case FUSED_HARD_SIGMOID:
    case FUSED_HARDTANH: case FUSED_SWISH: case FUSED_SILU: case FUSED_MISH:
    case FUSED_CLIP:
      return true;
    default:
      return false;
  }
}

/**
 * The code computing the same value when the chain value is the member's right-hand operand
 * (the kernels always pass the chain value first), or -1 when no such code exists. min/max
 * have none: sd_max/sd_min pick by operand position for NaN and signed zeros.
 */
inline int swappedBinaryFusedCode(int code) {
  switch (code) {
    case FUSED_ADD:
    case FUSED_MUL:
    case FUSED_SQUARED_SUB:
      return code;
    case FUSED_SUB:         return FUSED_REVERSE_SUB;
    case FUSED_REVERSE_SUB: return FUSED_SUB;
    case FUSED_DIV:         return FUSED_REVERSE_DIV;
    case FUSED_REVERSE_DIV: return FUSED_DIV;
    default:                return -1;
  }
}

/** Storage types the fused kernels are compiled for (SD_FLOAT_TYPES of this build). */
inline bool isFusedChainStorageType(DataType type) {
  switch (type) {
#if defined(HAS_FLOAT16)
    case DataType::HALF:
#endif
#if defined(HAS_BFLOAT16)
    case DataType::BFLOAT16:
#endif
#if defined(HAS_FLOAT32)
    case DataType::FLOAT32:
#endif
#if defined(HAS_DOUBLE)
    case DataType::DOUBLE:
#endif
      return true;
    default:
      return false;
  }
}

/**
 * True when every element sits at its C-order linear index. Size-1 dimensions may carry any
 * stride. Uses the host shape info.
 */
inline bool isFusedChainDenseC(NDArray* array) {
  const int rank = array->rankOf();
  const LongType* shape = array->shapeOf();
  const LongType* strides = array->stridesOf();
  LongType expected = 1;
  for (int d = rank - 1; d >= 0; d--) {
    if (shape[d] != 1 && strides[d] != expected) return false;
    expected *= shape[d];
  }
  return true;
}

/**
 * True when a secondary input can be read for every output element: a single element, or a
 * right-aligned broadcast whose dimensions are 1 or equal to the output's.
 */
inline bool isFusedChainBroadcastable(NDArray* secondary, NDArray* output) {
  if (output->isEmpty()) return true;
  if (secondary->isEmpty()) return false;
  if (secondary->lengthOf() == 1) return true;
  const int rank = output->rankOf();
  const int secondaryRank = secondary->rankOf();
  if (secondaryRank > rank) return false;
  const LongType* shape = output->shapeOf();
  const LongType* secondaryShape = secondary->shapeOf();
  const int lead = rank - secondaryRank;
  for (int d = 0; d < secondaryRank; d++) {
    if (secondaryShape[d] != 1 && secondaryShape[d] != shape[lead + d]) return false;
  }
  return true;
}

/**
 * Empty when fusedElementwiseChain() can run these operands, otherwise the reason it cannot.
 * Callers able to run the chain some other way (the DSP slot dispatch) check this first;
 * fusedElementwiseChain() throws with the same reason.
 */
inline std::string fusedChainUnsupportedReason(NDArray* input, NDArray* output, const FusedElemOp* ops, int numOps,
                                               NDArray** secondaryInputs, const double* clipMin,
                                               const double* clipMax) {
  if (input == nullptr || output == nullptr) return "input and output are required";
  if (ops == nullptr || numOps < 1 || numOps > FUSED_CHAIN_MAX_OPS)
    return "chain length " + std::to_string(numOps) + " is outside 1.." + std::to_string(FUSED_CHAIN_MAX_OPS);
  const DataType dtype = input->dataType();
  if (!isFusedChainStorageType(dtype))
    return "dtype " + DataTypeUtils::asString(dtype) + " is not a floating storage type of this build";
  if (output->dataType() != dtype)
    return "output dtype " + DataTypeUtils::asString(output->dataType()) + " differs from input dtype " +
           DataTypeUtils::asString(dtype);
  if (!input->isSameShape(output)) return "output shape differs from input shape";
  for (int m = 0; m < numOps; m++) {
    const int code = static_cast<int>(ops[m]);
    if (!isImplementedFusedOp(code))
      return "op code " + std::to_string(code) + " at member " + std::to_string(m) + " is not implemented";
    if (code == FUSED_CLIP && (clipMin == nullptr || clipMax == nullptr))
      return "FUSED_CLIP at member " + std::to_string(m) + " needs clipMin and clipMax";
    // clipbyvalue rejects these bounds (NaN included), so a fused clip must too.
    if (code == FUSED_CLIP && !(*clipMin < *clipMax))
      return "FUSED_CLIP bounds [" + std::to_string(*clipMin) + ", " + std::to_string(*clipMax) +
             "] need clipMin < clipMax";
    if (!isBinaryFusedOp(ops[m])) continue;
    NDArray* secondary = secondaryInputs == nullptr ? nullptr : secondaryInputs[m];
    if (secondary == nullptr) return "binary member " + std::to_string(m) + " has no secondary input";
    if (secondary->dataType() != dtype)
      return "secondary input of member " + std::to_string(m) + " has dtype " +
             DataTypeUtils::asString(secondary->dataType()) + ", chain dtype is " + DataTypeUtils::asString(dtype);
    if (!isFusedChainBroadcastable(secondary, output))
      return "secondary input of member " + std::to_string(m) + " does not broadcast into the output shape";
  }
  return {};
}

/**
 * Execute a chain of element-wise ops in a single kernel.
 *
 * The result equals running the members one by one: each member applies its eager op's functor
 * in the storage type and rounds to it, with the chain value as the first operand. Arrays may be
 * arbitrary views; output coordinates map through each array's own strides.
 *
 * @param input         Primary input tensor (HALF, BFLOAT16, FLOAT32 or DOUBLE)
 * @param output        Output tensor: same shape and dtype as input
 * @param ops           Array of op codes to apply sequentially (1..FUSED_CHAIN_MAX_OPS)
 * @param numOps        Number of ops in the chain
 * @param secondaryInputs  Secondary inputs, array of length numOps; entry i is required when
 *                         ops[i] is binary and ignored otherwise. Same dtype as input; a single
 *                         element or a right-aligned broadcast into the output shape.
 * @param clipMin       Lower bound for FUSED_CLIP (required when the chain has one)
 * @param clipMax       Upper bound for FUSED_CLIP (required when the chain has one)
 * @param context       Launch context (stream for CUDA)
 * @throws when fusedChainUnsupportedReason() reports a reason
 */
SD_LIB_HIDDEN void fusedElementwiseChain(
    NDArray* input,
    NDArray* output,
    const FusedElemOp* ops,
    int numOps,
    NDArray** secondaryInputs,
    const double* clipMin,
    const double* clipMax,
    LaunchContext* context);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_FUSED_ELEMENTWISE_CHAIN_H
