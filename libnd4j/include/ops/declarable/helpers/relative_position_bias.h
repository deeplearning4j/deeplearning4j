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
// Relative Position Bias for attention mechanisms (Swin, SAM, T5, ALiBi)
//

#ifndef LIBND4J_HELPERS_RELATIVE_POSITION_BIAS_H
#define LIBND4J_HELPERS_RELATIVE_POSITION_BIAS_H

#include <array/NDArray.h>
#include <graph/Context.h>
#include <ops/declarable/helpers/helpers.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * @brief Compute relative position bias for attention
 *
 * Supports two modes:
 * 1. Learned bias (Swin/SAM style): looks up bias values from a learned table
 * 2. ALiBi: computes linear position-based bias
 *
 * @param context Launch context for execution
 * @param biasTable Learned bias table [num_relative_positions, num_heads] (for learned mode)
 *                  or sequence length tensor (for ALiBi mode)
 * @param relativePositionIndex Precomputed relative position index [window_size^2, window_size^2] (optional)
 * @param output Output bias tensor [num_heads, seq_len, seq_len] or [num_heads, window_size^2, window_size^2]
 * @param numHeads Number of attention heads
 * @param windowSize Window size for 2D position encoding (ignored in ALiBi mode)
 * @param useAlibi Whether to use ALiBi mode
 */
SD_LIB_HIDDEN void relativePositionBias(sd::LaunchContext* context,
                                         NDArray* biasTable,
                                         NDArray* relativePositionIndex,
                                         NDArray* output,
                                         int numHeads,
                                         int windowSize,
                                         bool useAlibi);

/**
 * @brief Compute ALiBi slopes for each head
 *
 * @param numHeads Number of attention heads
 * @return Vector of slopes, one per head
 */
template <typename T>
std::vector<T> computeAlibiSlopes(int numHeads);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_HELPERS_RELATIVE_POSITION_BIAS_H
