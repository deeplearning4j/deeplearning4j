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
 * FP8 scaled GEMM for W8A8 FP8 inference pipelines.
 *
 * Performs: output = (A_fp8 * B_fp8) * scaleA * scaleB
 *
 * Uses CUTLASS FP8 tensor core GEMM (SM89+) with per-tensor scale
 * factors. Output is FP16 or FP32.
 *
 * @param context       launch context
 * @param A             [M, K] FP8 E4M3 input (stored as int8)
 * @param B             [K, N] FP8 E4M3 input (stored as int8)
 * @param scaleA        [1] per-tensor scale for A
 * @param scaleB        [1] per-tensor scale for B
 * @param output        [M, N] output in FP16 or FP32
 */
SD_LIB_HIDDEN void fp8ScaledGemm(LaunchContext* context,
                                   NDArray* A,
                                   NDArray* B,
                                   NDArray* scaleA,
                                   NDArray* scaleB,
                                   NDArray* output);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_INT8_GEMM_H
