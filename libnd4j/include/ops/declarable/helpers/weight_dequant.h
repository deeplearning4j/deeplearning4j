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

#ifndef LIBND4J_WEIGHT_DEQUANT_H
#define LIBND4J_WEIGHT_DEQUANT_H

#include <array/NDArray.h>
#include <system/common.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * AWQ-style INT4 group dequantization.
 *
 * AWQ (Activation-aware Weight Quantization) stores weights as INT4 with
 * per-group scale and zero-point:
 *   weight_fp = (packed_int4 - zero) * scale
 *
 * The packed format stores two INT4 values per byte (low nibble, high nibble).
 * Each group of groupSize consecutive elements shares one scale+zero pair.
 *
 * @param context       launch context
 * @param packedWeights [outFeatures, inFeatures / 2] packed INT4 pairs (uint8)
 * @param scales        [outFeatures, numGroups] per-group scale factors
 * @param zeros         [outFeatures, numGroups] per-group zero points
 * @param output        [outFeatures, inFeatures] dequantized FP16/FP32 weights
 * @param groupSize     elements per quantization group (typically 128)
 */
SD_LIB_HIDDEN void awqDequantize(LaunchContext* context,
                                  NDArray* packedWeights,
                                  NDArray* scales,
                                  NDArray* zeros,
                                  NDArray* output,
                                  int groupSize);

/**
 * GPTQ-style INT4 dequantization with permutation.
 *
 * GPTQ (Generative Pre-Training Quantization) uses INT4 with per-group
 * quantization and an optional column permutation (g_idx) that maps each
 * input column to its quantization group.
 *
 *   weight_fp = (packed_int4 - zero[g_idx[col]]) * scale[g_idx[col]]
 *
 * @param context       launch context
 * @param packedWeights [outFeatures, inFeatures / 2] packed INT4 pairs (uint8)
 * @param scales        [numGroups, outFeatures] per-group scales
 * @param zeros         [numGroups, outFeatures / 8] packed zero-points
 * @param gIdx          [inFeatures] group index mapping (may be nullptr for sequential)
 * @param output        [outFeatures, inFeatures] dequantized output
 * @param groupSize     elements per group (typically 128)
 * @param bits          quantization bits (4 or 8)
 */
SD_LIB_HIDDEN void gptqDequantize(LaunchContext* context,
                                   NDArray* packedWeights,
                                   NDArray* scales,
                                   NDArray* zeros,
                                   NDArray* gIdx,
                                   NDArray* output,
                                   int groupSize,
                                   int bits);

/**
 * Marlin-style INT4xFP16 fused dequantize + GEMM.
 *
 * Marlin format pre-permutes weights for efficient GPU memory access
 * and fuses dequantization with GEMM for maximum throughput.
 * Achieves near-FP16 speed at 4-bit precision by overlapping
 * dequant with tensor core math in a single kernel.
 *
 * output = input_fp16 @ dequant(marlinWeights, scales)
 *
 * @param context       launch context
 * @param input         [M, K] FP16 input activations
 * @param marlinWeights [K/16, N*16/8] Marlin-packed INT4 weights
 * @param scales        [numGroups, N] per-group scale factors
 * @param output        [M, N] FP16 output
 * @param workspace     [N/128 * 16] workspace for reduction (pre-allocated)
 * @param groupSize     elements per quantization group (-1 for per-channel)
 */
SD_LIB_HIDDEN void marlinGemm(LaunchContext* context,
                                NDArray* input,
                                NDArray* marlinWeights,
                                NDArray* scales,
                                NDArray* output,
                                NDArray* workspace,
                                int groupSize);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_WEIGHT_DEQUANT_H
