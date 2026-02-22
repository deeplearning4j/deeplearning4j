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

#include <ops/declarable/helpers/helpers.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * Op codes for the fused elementwise interpreter.
 * Each code maps to a simple element-wise operation applied sequentially.
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
    FUSED_SWISH = 20,
    FUSED_SILU = 21,
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
    FUSED_LEAKY_RELU = 31,  // Uses tArgs[0] as alpha

    // Additional unary ops (continued)
    FUSED_FLOOR = 32,
    FUSED_ROUND = 33,
    FUSED_SIN = 34,
    FUSED_COS = 35,
    FUSED_ELU = 36,
    FUSED_SELU = 37,
    FUSED_SOFTPLUS = 38,
    FUSED_SOFTSIGN = 39,
    FUSED_HARD_SIGMOID = 40,
    FUSED_HARDTANH = 41,
    FUSED_RELU6 = 42,

    // Additional binary ops
    FUSED_MIN = 50,         // minimum/min_pairwise
    FUSED_MAX = 51,         // maximum/max_pairwise
    FUSED_MOD = 52,         // mod/floormod
    FUSED_ATAN2 = 53,
    FUSED_FLOORDIV = 54,
    FUSED_REVERSE_DIV = 55,
    FUSED_REVERSE_SUB = 56,
    FUSED_SQUARED_SUB = 57,
    FUSED_MUL_NO_NAN = 58,
    FUSED_POW = 59,
};

/**
 * Check if a FusedElemOp needs a secondary input value.
 * Binary arithmetic ops (add/sub/mul/div) and leaky_relu (alpha parameter).
 */
SD_HOST_DEVICE inline bool isBinaryFusedOp(FusedElemOp op) {
    return op <= FUSED_DIV || op == FUSED_LEAKY_RELU ||
           (op >= FUSED_MIN && op <= FUSED_POW);
}

/**
 * Execute a chain of element-wise ops in a single kernel.
 *
 * The kernel processes all elements of the input tensor, applying the chain
 * of operations sequentially per-element. Binary ops use the corresponding
 * secondaryInput for the second operand (with broadcasting support).
 *
 * @param input         Primary input tensor
 * @param output        Output tensor (must be pre-allocated with correct shape)
 * @param ops           Array of op codes to apply sequentially
 * @param numOps        Number of ops in the chain
 * @param secondaryInputs  Secondary inputs for binary ops (nullptr for unary ops).
 *                         Array of length numOps; entry i is used when ops[i] is binary.
 * @param clipMin       Minimum value for FUSED_CLIP ops (nullptr if unused)
 * @param clipMax       Maximum value for FUSED_CLIP ops (nullptr if unused)
 * @param context       Launch context (stream for CUDA)
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

/**
 * Map op name string to FusedElemOp code.
 * Returns -1 if the op name is not supported for fused execution.
 */
inline int opNameToFusedCode(const std::string& opName) {
    if (opName == "add") return FUSED_ADD;
    if (opName == "subtract") return FUSED_SUB;
    if (opName == "multiply") return FUSED_MUL;
    if (opName == "divide") return FUSED_DIV;
    if (opName == "relu") return FUSED_RELU;
    if (opName == "sigmoid") return FUSED_SIGMOID;
    if (opName == "tanh") return FUSED_TANH;
    if (opName == "gelu") return FUSED_GELU;
    if (opName == "exp") return FUSED_EXP;
    if (opName == "log") return FUSED_LOG;
    if (opName == "abs") return FUSED_ABS;
    if (opName == "neg") return FUSED_NEG;
    if (opName == "square") return FUSED_SQUARE;
    if (opName == "sqrt") return FUSED_SQRT;
    if (opName == "swish") return FUSED_SWISH;
    if (opName == "silu") return FUSED_SILU;
    if (opName == "mish") return FUSED_MISH;
    if (opName == "rsqrt") return FUSED_RSQRT;
    if (opName == "reciprocal") return FUSED_RECIPROCAL;
    if (opName == "sign") return FUSED_SIGN;
    if (opName == "erf") return FUSED_ERF;
    if (opName == "erfc") return FUSED_ERFC;
    if (opName == "log1p") return FUSED_LOG1P;
    if (opName == "ceil") return FUSED_CEIL;
    if (opName == "floor") return FUSED_FLOOR;
    if (opName == "round") return FUSED_ROUND;
    if (opName == "sin") return FUSED_SIN;
    if (opName == "cos") return FUSED_COS;
    if (opName == "elu") return FUSED_ELU;
    if (opName == "selu") return FUSED_SELU;
    if (opName == "softplus") return FUSED_SOFTPLUS;
    if (opName == "softsign") return FUSED_SOFTSIGN;
    if (opName == "hard_sigmoid") return FUSED_HARD_SIGMOID;
    if (opName == "hardtanh") return FUSED_HARDTANH;
    if (opName == "relu6") return FUSED_RELU6;
    if (opName == "leakyrelu") return FUSED_LEAKY_RELU;
    if (opName == "clipbyvalue" || opName == "clip_by_value") return FUSED_CLIP;
    if (opName == "minimum" || opName == "min_pairwise") return FUSED_MIN;
    if (opName == "maximum" || opName == "max_pairwise") return FUSED_MAX;
    if (opName == "mod" || opName == "floormod") return FUSED_MOD;
    if (opName == "atan2") return FUSED_ATAN2;
    if (opName == "floordiv") return FUSED_FLOORDIV;
    if (opName == "reversedivide") return FUSED_REVERSE_DIV;
    if (opName == "reversesubtract") return FUSED_REVERSE_SUB;
    if (opName == "squaredsubtract") return FUSED_SQUARED_SUB;
    if (opName == "multiply_no_nan") return FUSED_MUL_NO_NAN;
    if (opName == "pow") return FUSED_POW;
    return -1;
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_FUSED_ELEMENTWISE_CHAIN_H
