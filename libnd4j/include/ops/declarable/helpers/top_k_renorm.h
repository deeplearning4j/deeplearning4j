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

#ifndef LIBND4J_HELPERS_TOP_K_RENORM_H
#define LIBND4J_HELPERS_TOP_K_RENORM_H

#include <system/op_boilerplate.h>
#include <array/NDArray.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * Top-K filtering with renormalization.
 *
 * Keeps only the top-K highest-probability tokens, zeros the rest,
 * then renormalizes so the kept probabilities sum to 1.
 *
 * Input is logits (pre-softmax). The op:
 *   1. Computes softmax probabilities
 *   2. Finds the K-th largest probability
 *   3. Zeros probabilities below that threshold
 *   4. Renormalizes remaining probabilities to sum to 1
 *   5. Writes renormalized probabilities to output
 *
 * @param logits  [batch, vocabSize] — input logits
 * @param output  [batch, vocabSize] — renormalized probabilities
 * @param k       number of top tokens to keep
 */
SD_LIB_HIDDEN void topKRenorm(LaunchContext* context, NDArray* logits, NDArray* output, int k);

/**
 * Top-P (nucleus) filtering with renormalization.
 *
 * Sorts tokens by descending probability, accumulates until cumulative
 * probability >= p, zeros the rest, then renormalizes.
 *
 * @param logits  [batch, vocabSize] — input logits
 * @param output  [batch, vocabSize] — renormalized probabilities
 * @param p       cumulative probability threshold (0.0-1.0)
 */
SD_LIB_HIDDEN void topPRenorm(LaunchContext* context, NDArray* logits, NDArray* output, double p);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_HELPERS_TOP_K_RENORM_H
