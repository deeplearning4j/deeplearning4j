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

#ifndef LIBND4J_INT8_GEMM_H
#define LIBND4J_INT8_GEMM_H

#include <array/NDArray.h>
#include <system/common.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * INT8 scaled GEMM via cublasLt for W8A8 quantized inference.
 *
 * Performs: output = (A_int8 * B_int8) * scaleA * scaleB
 *
 * Uses cublasLtMatmul with CUBLAS_COMPUTE_32I for native INT8 tensor core
 * support (SM75+). The output is dequantized to FP32 or FP16 using the
 * per-tensor or per-token scale factors.
 *
 * @param context       launch context
 * @param A             [M, K] INT8 input (quantized activations)
 * @param B             [K, N] INT8 input (quantized weights)
 * @param scaleA        [M] or [1] per-token or per-tensor scale for A
 * @param scaleB        [N] or [1] per-channel or per-tensor scale for B
 * @param output        [M, N] output in FP32 or FP16
 * @param bias          [N] optional bias to add (may be nullptr)
 */
SD_LIB_HIDDEN void int8ScaledGemm(LaunchContext* context,
                                    NDArray* A,
                                    NDArray* B,
                                    NDArray* scaleA,
                                    NDArray* scaleB,
                                    NDArray* output,
                                    NDArray* bias);

/**
 * Scaled GEMM over operands of any storage type: FP8 and wider floating
 * types, and integer codes (INT8 weights) whose values the scales dequantize.
 *
 * Performs: output = (op(A) * op(B)) * scaleA * scaleB + bias
 *
 * A and B keep their own storage types; each converts into the aggregate type
 * of the output (FP8 and INT8 exactly, a copy made only when the type or layout
 * differs) and is multiplied there, and the scaled, biased product converts
 * once into the output's type.
 *
 * @param context       launch context
 * @param A             [..., K] input whose leading axes flatten into the M rows
 *                      of the product, or [K, M] when transposeA
 * @param B             [K, N] input, or [N, K] when transposeB
 * @param scaleA        nullptr, one scale for the tensor, or one per row of the product ([M])
 * @param scaleB        nullptr, one scale for the tensor, or one per column of the product ([N])
 * @param bias          nullptr, or [N] added to every row
 * @param output        floating output holding the M x N product in c order ([M, N] or [..., N])
 * @param transposeA    multiply by the transpose of A
 * @param transposeB    multiply by the transpose of B
 */
SD_LIB_HIDDEN void scaledGemm(LaunchContext* context,
                              NDArray* A,
                              NDArray* B,
                              NDArray* scaleA,
                              NDArray* scaleB,
                              NDArray* bias,
                              NDArray* output,
                              bool transposeA,
                              bool transposeB);

/**
 * SmoothQuant smoothing: output = input / smoothScale along the last axis,
 * computed in the aggregate type of the output.
 *
 * @param context       launch context
 * @param input         [..., K] floating activations
 * @param smoothScale   [K] per-channel smoothing scale
 * @param output        floating output of input's shape
 */
SD_LIB_HIDDEN void smoothActivation(LaunchContext* context, NDArray* input, NDArray* smoothScale, NDArray* output);

/**
 * SmoothQuant GEMM: the smoothed activation quantizes onto the weight's grid
 * (A8 with W8, integer or FP8 codes) and the product dequantizes by both scales:
 *
 *   Xq = q(input / (smoothScale * actScale))
 *   output = ((Xq * actScale) @ op(weight)) * weightScale + bias
 *
 * @param context          launch context
 * @param input            [..., K] floating activations
 * @param weight           [N, K] signed integer or floating weight codes, or [K, N] when transposeWeight
 * @param smoothScale      [K] per-channel smoothing scale
 * @param actScale         one activation scale, or one per channel ([K])
 * @param weightScale      one weight scale, or one per output channel ([N])
 * @param bias             nullptr, or [N] added to every row
 * @param output           floating output [..., N]
 * @param transposeWeight  weight is stored as [K, N]
 */
SD_LIB_HIDDEN void smoothQuantGemm(LaunchContext* context,
                                   NDArray* input,
                                   NDArray* weight,
                                   NDArray* smoothScale,
                                   NDArray* actScale,
                                   NDArray* weightScale,
                                   NDArray* bias,
                                   NDArray* output,
                                   bool transposeWeight);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_INT8_GEMM_H
