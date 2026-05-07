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

#ifndef LIBND4J_SPECULATIVE_DECODING_H
#define LIBND4J_SPECULATIVE_DECODING_H

#include <array/NDArray.h>
#include <system/common.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * Tree-based speculative sampling using rejection sampling.
 *
 * Given draft model probabilities and target model probabilities for a tree
 * of candidate tokens, performs rejection sampling to accept/reject each
 * draft token. Accepted tokens are output along with the first rejected
 * position where a new token is sampled from an adjusted distribution.
 *
 * For each candidate token at position i:
 *   r ~ Uniform(0,1)
 *   if r < target_prob[token_i] / draft_prob[token_i]:
 *     accept token_i
 *   else:
 *     reject: sample from max(0, target - draft) distribution
 *
 * @param context         launch context
 * @param targetProbs     [batchSize, maxSpecLen, vocabSize] target model probs
 * @param draftProbs      [batchSize, maxSpecLen, vocabSize] draft model probs
 * @param draftTokenIds   [batchSize, maxSpecLen] draft token IDs
 * @param randVals        [batchSize, maxSpecLen] uniform random values in [0,1)
 * @param outputTokenIds  [batchSize, maxSpecLen+1] accepted token IDs (padded with -1)
 * @param numAccepted     [batchSize] number of accepted tokens per sequence
 */
SD_LIB_HIDDEN void treeSpeculativeSampling(LaunchContext* context,
                                            NDArray* targetProbs,
                                            NDArray* draftProbs,
                                            NDArray* draftTokenIds,
                                            NDArray* randVals,
                                            NDArray* outputTokenIds,
                                            NDArray* numAccepted);

/**
 * Greedy tree verification for speculative decoding.
 *
 * Verifies draft tokens against target model argmax predictions.
 * Accepts draft tokens as long as they match the greedy target prediction.
 * At the first mismatch, outputs the target model's prediction instead.
 *
 * @param context         launch context
 * @param targetLogits    [batchSize, maxSpecLen, vocabSize] target model logits
 * @param draftTokenIds   [batchSize, maxSpecLen] proposed draft tokens
 * @param outputTokenIds  [batchSize, maxSpecLen+1] verified token sequence
 * @param numAccepted     [batchSize] number of accepted draft tokens per sequence
 */
SD_LIB_HIDDEN void treeVerifyGreedy(LaunchContext* context,
                                     NDArray* targetLogits,
                                     NDArray* draftTokenIds,
                                     NDArray* outputTokenIds,
                                     NDArray* numAccepted);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_SPECULATIVE_DECODING_H
