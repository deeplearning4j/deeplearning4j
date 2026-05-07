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

#include <ops/declarable/helpers/sampling_penalties.h>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <vector>

namespace sd {
namespace ops {
namespace helpers {

void applyLogitPenaltiesCpu(NDArray* logits, NDArray* inputIds,
                            double repPenalty, double freqPenalty,
                            double presPenalty, LaunchContext* context) {
    if (repPenalty == 1.0 && freqPenalty == 0.0 && presPenalty == 0.0) return;

    auto logitsRank = logits->rankOf();
    auto idsRank = inputIds->rankOf();

    LongType batch = 1;
    LongType vocabSize;
    LongType seqLen;

    if (logitsRank == 1) {
        vocabSize = logits->sizeAt(0);
    } else {
        batch = logits->sizeAt(0);
        vocabSize = logits->sizeAt(1);
    }

    if (idsRank == 1) {
        seqLen = inputIds->sizeAt(0);
    } else {
        seqLen = inputIds->sizeAt(1);
    }

    float repF = static_cast<float>(repPenalty);
    float freqF = static_cast<float>(freqPenalty);
    float presF = static_cast<float>(presPenalty);

    for (LongType b = 0; b < batch; b++) {
        // Count token frequencies in input
        std::unordered_map<LongType, int> tokenCounts;
        for (LongType s = 0; s < seqLen; s++) {
            LongType tokenId;
            if (idsRank == 1) {
                tokenId = inputIds->e<LongType>(s);
            } else {
                tokenId = inputIds->e<LongType>(b, s);
            }
            if (tokenId >= 0 && tokenId < vocabSize) {
                tokenCounts[tokenId]++;
            }
        }

        // Apply penalties to each token that appeared
        for (auto& pair : tokenCounts) {
            LongType tokenId = pair.first;
            int count = pair.second;

            float logit;
            if (logitsRank == 1) {
                logit = logits->e<float>(tokenId);
            } else {
                logit = logits->e<float>(b, tokenId);
            }

            // Repetition penalty: multiplicative, direction-aware
            if (repF != 1.0f) {
                logit = logit > 0.0f ? logit / repF : logit * repF;
            }

            // Frequency penalty: proportional to count
            if (freqF != 0.0f) {
                logit -= static_cast<float>(count) * freqF;
            }

            // Presence penalty: flat penalty if token appeared at all
            if (presF != 0.0f) {
                logit -= presF;
            }

            if (logitsRank == 1) {
                logits->p(tokenId, logit);
            } else {
                logits->p(b, tokenId, logit);
            }
        }
    }
}

void applyMinPFilterCpu(NDArray* logits, double minP, LaunchContext* context) {
    if (minP <= 0.0) return;

    auto rank = logits->rankOf();
    LongType batch = 1;
    LongType vocabSize;

    if (rank == 1) {
        vocabSize = logits->sizeAt(0);
    } else {
        batch = logits->sizeAt(0);
        vocabSize = logits->sizeAt(1);
    }

    float minPF = static_cast<float>(minP);

    for (LongType b = 0; b < batch; b++) {
        // Step 1: Find max logit for numerical stability
        float maxLogit = -std::numeric_limits<float>::infinity();
        for (LongType v = 0; v < vocabSize; v++) {
            float val = (rank == 1) ? logits->e<float>(v) : logits->e<float>(b, v);
            if (val > maxLogit) maxLogit = val;
        }

        // Step 2: Compute softmax probabilities and find max probability
        std::vector<float> probs(vocabSize);
        float sumExp = 0.0f;
        for (LongType v = 0; v < vocabSize; v++) {
            float val = (rank == 1) ? logits->e<float>(v) : logits->e<float>(b, v);
            probs[v] = std::exp(val - maxLogit);
            sumExp += probs[v];
        }

        float maxProb = 0.0f;
        for (LongType v = 0; v < vocabSize; v++) {
            probs[v] /= sumExp;
            if (probs[v] > maxProb) maxProb = probs[v];
        }

        // Step 3: Filter — set logits to -inf where prob < minP * maxProb
        float threshold = minPF * maxProb;
        for (LongType v = 0; v < vocabSize; v++) {
            if (probs[v] < threshold) {
                float negInf = -std::numeric_limits<float>::infinity();
                if (rank == 1) {
                    logits->p(v, negInf);
                } else {
                    logits->p(b, v, negInf);
                }
            }
        }
    }
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
