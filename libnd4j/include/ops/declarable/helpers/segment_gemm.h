/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

#ifndef LIBND4J_HELPERS_SEGMENT_GEMM_H
#define LIBND4J_HELPERS_SEGMENT_GEMM_H

#include <system/op_boilerplate.h>
#include <array/NDArray.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * Segmented GEMM — variable-length batched matrix multiply for MoE dispatch.
 *
 * Performs a batched matrix multiply where each segment (expert) processes
 * a variable number of tokens. This is the core operation for Mixture of
 * Experts (MoE) models where tokens are routed to different experts.
 *
 * Given:
 *   - input: [totalTokens, inDim] — all tokens concatenated
 *   - weights: [numExperts, inDim, outDim] — per-expert weight matrices
 *   - segmentOffsets: [numExperts] INT64 — start index of each expert's tokens
 *   - segmentSizes: [numExperts] INT64 — number of tokens for each expert
 *
 * For each expert e:
 *   output[offset_e : offset_e + size_e] =
 *       input[offset_e : offset_e + size_e] @ weights[e]
 *
 * @param context         Launch context
 * @param input           [totalTokens, inDim] — concatenated token embeddings
 * @param weights         [numExperts, inDim, outDim] — expert weight matrices
 * @param segmentOffsets  [numExperts] INT64 — start offsets per expert
 * @param segmentSizes    [numExperts] INT64 — token counts per expert
 * @param output          [totalTokens, outDim] — output embeddings
 */
SD_LIB_HIDDEN void segmentGemmCpu(LaunchContext* context,
                                   NDArray* input, NDArray* weights,
                                   NDArray* segmentOffsets, NDArray* segmentSizes,
                                   NDArray* output);

#if defined(SD_CUDA)
SD_LIB_HIDDEN void segmentGemmCuda(LaunchContext* context,
                                    NDArray* input, NDArray* weights,
                                    NDArray* segmentOffsets, NDArray* segmentSizes,
                                    NDArray* output);
#endif

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_HELPERS_SEGMENT_GEMM_H
