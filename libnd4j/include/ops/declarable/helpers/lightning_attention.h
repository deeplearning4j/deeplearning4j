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

//
// Lightning Attention: O(n) linear attention with intra/inter-chunk decomposition.
// Based on the algorithmic patterns from cuLA (inclusionAI/cuLA).
// Reference: https://arxiv.org/abs/2405.17381 (Lightning Attention-2)
//

#ifndef LIBND4J_HELPERS_LIGHTNING_ATTENTION_H
#define LIBND4J_HELPERS_LIGHTNING_ATTENTION_H

#include <ops/declarable/helpers/helpers.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * Lightning Attention forward pass.
 *
 * Implements linear attention with per-head exponential decay via
 * intra/inter-chunk decomposition:
 *   O(n) complexity vs O(n²) for standard softmax attention.
 *
 * Input/output layout: [batch, seqLen, numHeads, headDim]  (BSHD)
 * State layout:  [batch, numHeads, headDim, headDim] (float32, in/out)
 * Decay layout:  [numHeads]  (per-head scalar decay rate, float32)
 *
 * @param context     Launch context
 * @param query       Q tensor [batch, seqLen, numHeads, headDim]
 * @param key         K tensor [batch, seqLen, numHeads, headDim]
 * @param value       V tensor [batch, seqLen, numHeads, headDim]
 * @param decayRates  Per-head decay scalars [numHeads], float32
 * @param state       Recurrent state [batch, numHeads, headDim, headDim], float32, in/out
 * @param output      Output [batch, seqLen, numHeads, headDim], same dtype as Q/K/V
 * @param isCausal    Whether to apply causal (lower-triangular) masking within a chunk
 */
SD_LIB_HIDDEN void lightningAttention(LaunchContext* context,
                                       NDArray* query,
                                       NDArray* key,
                                       NDArray* value,
                                       NDArray* decayRates,
                                       NDArray* state,
                                       NDArray* output,
                                       bool isCausal);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_HELPERS_LIGHTNING_ATTENTION_H
