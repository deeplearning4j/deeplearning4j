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
// Speculative decoding — CPU implementation.
// Tree-based rejection sampling and greedy verification.
//

#include <execution/Threads.h>
#include <math/templatemath.h>
#include <ops/declarable/helpers/speculative_decoding.h>

#include <cmath>
#include <limits>

namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////////
// Tree speculative sampling (CPU)
// Per-batch sequential rejection sampling.
//////////////////////////////////////////////////////////////////////////////
template <typename T>
static void treeSpeculativeSamplingCpu_(NDArray* targetProbs, NDArray* draftProbs,
                                         NDArray* draftTokenIds, NDArray* randVals,
                                         NDArray* outputTokenIds, NDArray* numAccepted) {
    const LongType batchSize = targetProbs->sizeAt(0);
    const LongType maxSpecLen = targetProbs->sizeAt(1);
    const LongType vocabSize = targetProbs->sizeAt(2);

    const T* tProbs = targetProbs->bufferAsT<T>();
    const T* dProbs = draftProbs->bufferAsT<T>();
    const int32_t* draftTokens = reinterpret_cast<const int32_t*>(draftTokenIds->buffer());
    const float* rands = reinterpret_cast<const float*>(randVals->buffer());
    int32_t* outTokens = reinterpret_cast<int32_t*>(outputTokenIds->buffer());
    int32_t* accepted = reinterpret_cast<int32_t*>(numAccepted->buffer());

    auto func = PRAGMA_THREADS_FOR {
        for (auto b = start; b < stop; ++b) {
            const T* batchTarget = tProbs + b * maxSpecLen * vocabSize;
            const T* batchDraft = dProbs + b * maxSpecLen * vocabSize;
            const int32_t* batchDraftTokens = draftTokens + b * maxSpecLen;
            const float* batchRand = rands + b * maxSpecLen;
            int32_t* batchOut = outTokens + b * (maxSpecLen + 1);

            // Initialize output to -1
            for (LongType i = 0; i <= maxSpecLen; ++i) {
                batchOut[i] = -1;
            }

            int numAcc = 0;

            for (LongType pos = 0; pos < maxSpecLen; ++pos) {
                int32_t draftToken = batchDraftTokens[pos];
                if (draftToken < 0) break;

                float tProb = static_cast<float>(batchTarget[pos * vocabSize + draftToken]);
                float dProb = static_cast<float>(batchDraft[pos * vocabSize + draftToken]);
                if (dProb <= 0.0f) dProb = 1e-10f;

                float acceptProb = sd::math::sd_min<float>(1.0f, tProb / dProb);
                float r = batchRand[pos];

                if (r < acceptProb) {
                    batchOut[numAcc] = draftToken;
                    numAcc++;
                } else {
                    // Reject: argmax of max(0, target - draft)
                    float bestVal = -1.0f;
                    int32_t bestToken = 0;

                    for (LongType v = 0; v < vocabSize; ++v) {
                        float tP = static_cast<float>(batchTarget[pos * vocabSize + v]);
                        float dP = static_cast<float>(batchDraft[pos * vocabSize + v]);
                        float adjusted = sd::math::sd_max<float>(0.0f, tP - dP);
                        if (adjusted > bestVal) {
                            bestVal = adjusted;
                            bestToken = static_cast<int32_t>(v);
                        }
                    }

                    batchOut[numAcc] = bestToken;
                    numAcc++;
                    break;
                }
            }

            // If all draft tokens accepted, add one more from target's last position
            if (numAcc == maxSpecLen) {
                float bestVal = -std::numeric_limits<float>::max();
                int32_t bestToken = 0;
                LongType lastPos = maxSpecLen - 1;

                for (LongType v = 0; v < vocabSize; ++v) {
                    float val = static_cast<float>(batchTarget[lastPos * vocabSize + v]);
                    if (val > bestVal) {
                        bestVal = val;
                        bestToken = static_cast<int32_t>(v);
                    }
                }
                batchOut[numAcc] = bestToken;
            }

            accepted[b] = numAcc;
        }
    };
    samediff::Threads::parallel_tad(func, 0, batchSize);
}

//////////////////////////////////////////////////////////////////////////////
// Greedy tree verification (CPU)
// Compares draft tokens against target argmax.
//////////////////////////////////////////////////////////////////////////////
template <typename T>
static void treeVerifyGreedyCpu_(NDArray* targetLogits, NDArray* draftTokenIds,
                                  NDArray* outputTokenIds, NDArray* numAccepted) {
    const LongType batchSize = targetLogits->sizeAt(0);
    const LongType maxSpecLen = targetLogits->sizeAt(1);
    const LongType vocabSize = targetLogits->sizeAt(2);

    const T* logits = targetLogits->bufferAsT<T>();
    const int32_t* draftTokens = reinterpret_cast<const int32_t*>(draftTokenIds->buffer());
    int32_t* outTokens = reinterpret_cast<int32_t*>(outputTokenIds->buffer());
    int32_t* accepted = reinterpret_cast<int32_t*>(numAccepted->buffer());

    auto func = PRAGMA_THREADS_FOR {
        for (auto b = start; b < stop; ++b) {
            const T* batchLogits = logits + b * maxSpecLen * vocabSize;
            const int32_t* batchDraft = draftTokens + b * maxSpecLen;
            int32_t* batchOut = outTokens + b * (maxSpecLen + 1);

            // Initialize output to -1
            for (LongType i = 0; i <= maxSpecLen; ++i) {
                batchOut[i] = -1;
            }

            int numAcc = 0;

            for (LongType pos = 0; pos < maxSpecLen; ++pos) {
                int32_t draftToken = batchDraft[pos];
                if (draftToken < 0) break;

                // Find argmax of target logits at this position
                float bestVal = -std::numeric_limits<float>::max();
                int32_t bestToken = 0;
                const T* posLogits = batchLogits + pos * vocabSize;

                for (LongType v = 0; v < vocabSize; ++v) {
                    float val = static_cast<float>(posLogits[v]);
                    if (val > bestVal) {
                        bestVal = val;
                        bestToken = static_cast<int32_t>(v);
                    }
                }

                if (bestToken == draftToken) {
                    batchOut[numAcc] = draftToken;
                    numAcc++;
                } else {
                    batchOut[numAcc] = bestToken;
                    numAcc++;
                    break;
                }
            }

            // If all accepted, add one more from last position
            if (numAcc == maxSpecLen) {
                const T* lastLogits = batchLogits + (maxSpecLen - 1) * vocabSize;
                float bestVal = -std::numeric_limits<float>::max();
                int32_t bestToken = 0;
                for (LongType v = 0; v < vocabSize; ++v) {
                    float val = static_cast<float>(lastLogits[v]);
                    if (val > bestVal) {
                        bestVal = val;
                        bestToken = static_cast<int32_t>(v);
                    }
                }
                batchOut[numAcc] = bestToken;
            }

            accepted[b] = numAcc;
        }
    };
    samediff::Threads::parallel_tad(func, 0, batchSize);
}

//////////////////////////////////////////////////////////////////////////////
// Public interface
//////////////////////////////////////////////////////////////////////////////
void treeSpeculativeSampling(LaunchContext* context,
                              NDArray* targetProbs,
                              NDArray* draftProbs,
                              NDArray* draftTokenIds,
                              NDArray* randVals,
                              NDArray* outputTokenIds,
                              NDArray* numAccepted) {
    BUILD_SINGLE_SELECTOR(targetProbs->dataType(), treeSpeculativeSamplingCpu_,
                          (targetProbs, draftProbs, draftTokenIds, randVals,
                           outputTokenIds, numAccepted), SD_FLOAT_TYPES);

    outputTokenIds->tickWriteHost();
    numAccepted->tickWriteHost();
}

void treeVerifyGreedy(LaunchContext* context,
                       NDArray* targetLogits,
                       NDArray* draftTokenIds,
                       NDArray* outputTokenIds,
                       NDArray* numAccepted) {
    BUILD_SINGLE_SELECTOR(targetLogits->dataType(), treeVerifyGreedyCpu_,
                          (targetLogits, draftTokenIds, outputTokenIds, numAccepted),
                          SD_FLOAT_TYPES);

    outputTokenIds->tickWriteHost();
    numAccepted->tickWriteHost();
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
