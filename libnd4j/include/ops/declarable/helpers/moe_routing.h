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

#ifndef LIBND4J_MOE_ROUTING_H
#define LIBND4J_MOE_ROUTING_H

#include <array/NDArray.h>
#include <system/common.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * MoE top-K routing with sigmoid gating (DeepSeek V3/R1 style).
 *
 * Unlike softmax routing (normalized), sigmoid routing computes independent
 * per-expert probabilities: score_e = sigmoid(logit_e + bias_e).
 * Then selects top-K experts per token.
 *
 * @param context       launch context
 * @param gatingLogits  [numTokens, numExperts] raw router logits
 * @param expertBias    [numExperts] optional per-expert bias (may be nullptr)
 * @param expertScores  [numTokens, topK] output routing weights
 * @param expertIndices [numTokens, topK] output selected expert indices
 * @param topK          number of experts per token
 * @param renormalize   if true, normalize selected scores to sum to 1
 * @param routingScale  multiplicative scale applied to final routing weights
 */
SD_LIB_HIDDEN void moeTopkSigmoid(LaunchContext* context,
                                    NDArray* gatingLogits,
                                    NDArray* expertBias,
                                    NDArray* expertScores,
                                    NDArray* expertIndices,
                                    int topK,
                                    bool renormalize,
                                    float routingScale);

/**
 * MoE top-K routing with softmax gating.
 *
 * Computes softmax over gating logits, then selects top-K experts.
 * Supports optional tanh softcapping for logit clamping.
 *
 * @param context       launch context
 * @param gatingLogits  [numTokens, numExperts] raw router logits
 * @param expertScores  [numTokens, topK] output routing weights
 * @param expertIndices [numTokens, topK] output selected expert indices
 * @param topK          number of experts per token
 * @param renormalize   if true, renormalize selected top-K scores
 * @param softcapScale  if > 0, apply tanh(logits/scale)*scale before softmax
 */
SD_LIB_HIDDEN void moeTopkSoftmax(LaunchContext* context,
                                    NDArray* gatingLogits,
                                    NDArray* expertScores,
                                    NDArray* expertIndices,
                                    int topK,
                                    bool renormalize,
                                    float softcapScale);

/**
 * MoE align block size: sort and pad tokens per expert for uniform batched GEMM.
 *
 * Takes expert assignments and produces:
 * - sorted_token_ids: token indices sorted by expert, padded to blockSize multiples
 * - expert_ids: which expert each sorted position belongs to
 * - num_tokens_per_expert: count of real (non-padding) tokens per expert
 *
 * @param context              launch context
 * @param expertIndices        [numTokens, topK] selected expert indices
 * @param numExperts           total number of experts
 * @param blockSize            pad each expert's tokens to this multiple
 * @param sortedTokenIds       [totalPaddedTokens] output sorted token indices
 * @param expertIds            [totalPaddedTokens] output expert id per position
 * @param numTokensPerExpert   [numExperts] output token count per expert
 */
SD_LIB_HIDDEN void moeAlignBlockSize(LaunchContext* context,
                                      NDArray* expertIndices,
                                      int numExperts,
                                      int blockSize,
                                      NDArray* sortedTokenIds,
                                      NDArray* expertIds,
                                      NDArray* numTokensPerExpert);

/**
 * MoE output summation: aggregate top-K expert outputs per token.
 *
 * input[numTokens, topK, hiddenDim] -> output[numTokens, hiddenDim]
 * output[t, d] = sum_k(input[t, k, d] * weights[t, k])
 *
 * @param context       launch context
 * @param expertOutputs [numTokens, topK, hiddenDim] per-expert outputs
 * @param routingWeights [numTokens, topK] routing weights
 * @param output        [numTokens, hiddenDim] aggregated output
 * @param topK          number of experts per token
 */
SD_LIB_HIDDEN void moeSumReduce(LaunchContext* context,
                                 NDArray* expertOutputs,
                                 NDArray* routingWeights,
                                 NDArray* output,
                                 int topK);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_MOE_ROUTING_H
