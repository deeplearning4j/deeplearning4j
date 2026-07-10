/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//
// @author Eclipse Deeplearning4j
//
// moe_weighted_sum — fused top-K expert output accumulation.
//
// Reduces the [T, topK, D] or sorted-flat [totalRows, D] + rowIndex layout produced by
// upstream expert GEMMs into [T, D] by computing:
//
//   out[t, d] = Σ_{k=0}^{topK-1}  w[t, k] · expertOut[t, k, d]
//
// without the caller materialising and holding the full [T, topK, D] tensor
// through subsequent ops.  fp32 accumulation is used regardless of the
// element type of the inputs.
//

#pragma once

#include <array/NDArray.h>
#include <execution/LaunchContext.h>
#include <system/common.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * Dense layout: expertOutputs is [T, topK, D].
 *
 * @param context       launch context (CPU: ignored for CUDA, required for CUDA)
 * @param expertOutputs [T, topK, D] — stacked expert outputs per token; fp32 accumulation
 * @param weights       [T, topK]    — router scores (already normalised; do NOT re-normalise)
 * @param output        [T, D]       — result; must be pre-allocated and zeroed by caller
 * @param T             number of tokens
 * @param topK          experts per token
 * @param D             hidden dimension
 */
SD_LIB_HIDDEN void moeWeightedSumDense(sd::LaunchContext* context,
                                        NDArray* expertOutputs,
                                        NDArray* weights,
                                        NDArray* output);

/**
 * Sorted-flat layout: expert outputs arrive in sorted order from segment_gemm
 * as [totalRows, D] with a companion rowIndex [totalRows, 2] (INT32/INT64) where
 * rowIndex[r] = {token_idx, k_idx} mapping row r back to (token, expert slot).
 *
 * @param context       launch context
 * @param expertOutputs [totalRows, D] sorted flat expert outputs
 * @param weights       [T, topK]      router scores
 * @param rowIndex      [totalRows, 2] INT32 or INT64 — each row is {token_idx, k_idx}
 * @param output        [T, D]         result; must be pre-allocated and zeroed by caller
 */
SD_LIB_HIDDEN void moeWeightedSumFlat(sd::LaunchContext* context,
                                       NDArray* expertOutputs,
                                       NDArray* weights,
                                       NDArray* rowIndex,
                                       NDArray* output);

}  // namespace helpers
}  // namespace ops
}  // namespace sd
