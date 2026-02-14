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

    // Parameterized ops
    FUSED_CLIP = 30,        // Uses clipMin/clipMax
    FUSED_LEAKY_RELU = 31,  // Uses tArgs[0] as alpha
};

/**
 * Check if a FusedElemOp needs a secondary input value.
 * Binary arithmetic ops (add/sub/mul/div) and leaky_relu (alpha parameter).
 */
SD_HOST_DEVICE inline bool isBinaryFusedOp(FusedElemOp op) {
    return op <= FUSED_DIV || op == FUSED_LEAKY_RELU;
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

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_FUSED_ELEMENTWISE_CHAIN_H
