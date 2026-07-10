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
 * @param logits       [batch, seqLen, vocabSize] or [batch, vocabSize] or [vocabSize] — modified in-place (last seq position)
 * @param inputIds     [batch, seqLen] or [seqLen] INT64 — previously generated token IDs
 * @param repPenalty   Repetition penalty factor (1.0 = no penalty, >1.0 penalizes repetition)
 * @param freqPenalty  Frequency penalty (0.0 = off, positive penalizes by count)
 * @param presPenalty  Presence penalty (0.0 = off, positive penalizes any presence)
 */
SD_LIB_HIDDEN void applyLogitPenalties(NDArray* logits, NDArray* inputIds,
                                        double repPenalty, double freqPenalty,
                                        double presPenalty, LaunchContext* context);

/**
 * Apply min-p filtering to logits in-place.
 *
 * After computing probabilities via softmax, any token with probability < minP * max_prob
 * has its logit set to -infinity. This is an adaptive threshold that scales with the
 * model's confidence — keeps more tokens when uncertain, fewer when confident.
 *
 * @param logits  [batch, seqLen, vocabSize] or [batch, vocabSize] or [vocabSize] — modified in-place (last seq position)
 * @param minP    Minimum probability threshold relative to the top token (0.0 = off, 0.1 typical)
 */
SD_LIB_HIDDEN void applyMinPFilter(NDArray* logits, double minP, LaunchContext* context);

/**
 * Apply typical-p (entropy-deviation) filtering to logits in-place.
 *
 * Algorithm (Meister et al. 2023, "Locally Typical Sampling"):
 *   1. Compute p_i = softmax(logits)_i
 *   2. H = -sum_i p_i * log(p_i)  (Shannon entropy of the distribution)
 *   3. deviation_i = |−log(p_i) − H|
 *   4. Sort tokens by deviation (ascending = most "typical" first)
 *   5. Keep tokens until their cumulative probability >= typicalP
 *   6. All remaining tokens (highest deviation) → logit = -inf
 *
 * typicalP in (0, 1) enables the filter; 1.0 is a no-op.
 * Operates on logits in-place on the last-sequence-position row.
 *
 * @param logits   [batch, seqLen, vocabSize] or [batch, vocabSize] or [vocabSize]
 * @param typicalP Cumulative probability mass to keep (1.0 = off)
 */
SD_LIB_HIDDEN void applyTypicalPFilter(NDArray* logits, double typicalP, LaunchContext* context);

/**
 * Apply XTC (Exclude Top Choices) sampling filter to logits in-place.
 *
 * With probability xtcProbability: among tokens whose softmax probability >= xtcThreshold,
 * if there are >= 2 such tokens, mask ALL of them EXCEPT the lowest-probability one.
 * This encourages diversity by removing the model's most confident choices.
 *
 * With probability (1 - xtcProbability) the logits are returned unchanged.
 *
 * RNG is driven by seed ^ (seed >> 17) to produce a per-call uniform scalar that
 * decides whether to apply. The same seed used for the final multinomial ensures
 * reproducibility across the full sampling pipeline.
 *
 * xtcProbability = 0.0 → always skip (no-op).
 * xtcThreshold = 0.0 → all tokens qualify; keep only the weakest one.
 * xtcThreshold >= 0.5 is rejected by validation before this is called.
 *
 * @param logits         [batch, vocabSize] or [vocabSize] — modified in-place
 * @param xtcProbability Probability of applying XTC (0 = off, 1 = always apply)
 * @param xtcThreshold   Per-token probability threshold to qualify for exclusion
 * @param seed           RNG seed (reproducible when > 0)
 */
SD_LIB_HIDDEN void applyXtcFilter(NDArray* logits, double xtcProbability, double xtcThreshold,
                                   LongType seed, LaunchContext* context);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_HELPERS_SAMPLING_PENALTIES_H
