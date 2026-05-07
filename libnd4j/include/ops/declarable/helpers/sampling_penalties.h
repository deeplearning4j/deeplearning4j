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

#ifndef LIBND4J_HELPERS_SAMPLING_PENALTIES_H
#define LIBND4J_HELPERS_SAMPLING_PENALTIES_H

#include <system/op_boilerplate.h>
#include <array/NDArray.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * Apply repetition, frequency, and presence penalties to logits in-place.
 *
 * For each token ID that appears in inputIds:
 *   - Repetition penalty: logit = logit > 0 ? logit / repPenalty : logit * repPenalty
 *   - Frequency penalty:  logit -= count(tokenId) * freqPenalty
 *   - Presence penalty:   logit -= (count(tokenId) > 0 ? 1 : 0) * presPenalty
 *
 * @param logits       [batch, vocabSize] or [vocabSize] — modified in-place
 * @param inputIds     [batch, seqLen] or [seqLen] INT64 — previously generated token IDs
 * @param repPenalty   Repetition penalty factor (1.0 = no penalty, >1.0 penalizes repetition)
 * @param freqPenalty  Frequency penalty (0.0 = off, positive penalizes by count)
 * @param presPenalty  Presence penalty (0.0 = off, positive penalizes any presence)
 */
SD_LIB_HIDDEN void applyLogitPenaltiesCpu(NDArray* logits, NDArray* inputIds,
                                           double repPenalty, double freqPenalty,
                                           double presPenalty, LaunchContext* context);

/**
 * Apply min-p filtering to logits in-place.
 *
 * After computing probabilities via softmax, any token with probability < minP * max_prob
 * has its logit set to -infinity. This is an adaptive threshold that scales with the
 * model's confidence — keeps more tokens when uncertain, fewer when confident.
 *
 * @param logits  [batch, vocabSize] or [vocabSize] — modified in-place
 * @param minP    Minimum probability threshold relative to the top token (0.0 = off, 0.1 typical)
 */
SD_LIB_HIDDEN void applyMinPFilterCpu(NDArray* logits, double minP, LaunchContext* context);

#if defined(SD_CUDA)

SD_LIB_HIDDEN void applyLogitPenaltiesCuda(NDArray* logits, NDArray* inputIds,
                                            double repPenalty, double freqPenalty,
                                            double presPenalty, LaunchContext* context);

SD_LIB_HIDDEN void applyMinPFilterCuda(NDArray* logits, double minP, LaunchContext* context);

#endif

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_HELPERS_SAMPLING_PENALTIES_H
