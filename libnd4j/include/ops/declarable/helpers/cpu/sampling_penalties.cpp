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
#include <execution/Threads.h>
#include <math/templatemath.h>
#include <system/openmp_pragmas.h>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <vector>
#include <cfloat>

namespace sd {
namespace ops {
namespace helpers {

// ─── Repetition / frequency / presence penalties ────────────────────────────

template <typename T>
static void applyLogitPenalties_(NDArray* logits, NDArray* inputIds,
                                  float repF, float freqF, float presF) {
    auto logitsRank = logits->rankOf();
    auto idsRank = inputIds->rankOf();

    LongType batch = 1;
    LongType vocabSize;
    LongType logitsSeqLen = 1;

    if (logitsRank == 1) {
        vocabSize = logits->sizeAt(0);
    } else if (logitsRank == 2) {
        batch = logits->sizeAt(0);
        vocabSize = logits->sizeAt(1);
    } else {
        batch = logits->sizeAt(0);
        logitsSeqLen = logits->sizeAt(1);
        vocabSize = logits->sizeAt(2);
    }

    LongType seqLen;
    if (idsRank == 0) {
        seqLen = 1;
    } else if (idsRank == 1) {
        seqLen = inputIds->sizeAt(0);
    } else {
        seqLen = inputIds->sizeAt(1);
    }

    // Raw buffer access
    T* logitsBuf = logits->bufferAsT<T>();
    auto logitsStrides = logits->stridesOf();

    auto idsBuf = inputIds->bufferAsT<LongType>();
    auto idsStrides = inputIds->stridesOf();

    // Precompute logits strides
    LongType logitsBatchStride, logitsElemStride, logitsRowOffset;
    if (logitsRank == 1) {
        logitsBatchStride = 0;
        logitsElemStride = logitsStrides[0];
        logitsRowOffset = 0;
    } else if (logitsRank == 2) {
        logitsBatchStride = logitsStrides[0];
        logitsElemStride = logitsStrides[1];
        logitsRowOffset = 0;
    } else {
        logitsBatchStride = logitsStrides[0];
        logitsElemStride = logitsStrides[2];
        logitsRowOffset = (logitsSeqLen - 1) * logitsStrides[1];
    }

    LongType idsBatchStride = 0;
    LongType idsElemStride = 0;
    if (idsRank == 1) {
        idsElemStride = idsStrides[0];
    } else if (idsRank == 2) {
        idsBatchStride = idsStrides[0];
        idsElemStride = idsStrides[1];
    }

    auto func = PRAGMA_THREADS_FOR {
        for (auto b = start; b < stop; b++) {
            LongType logitsBase = b * logitsBatchStride + logitsRowOffset;
            LongType idsBase = b * idsBatchStride;

            // Count token frequencies from input IDs
            std::unordered_map<LongType, int> tokenCounts;
            for (LongType s = 0; s < seqLen; s++) {
                LongType tokenId = idsBuf[idsBase + s * idsElemStride];
                if (tokenId >= 0 && tokenId < vocabSize) {
                    tokenCounts[tokenId]++;
                }
            }

            // Apply penalties to each seen token
            for (auto& pair : tokenCounts) {
                LongType tokenId = pair.first;
                int count = pair.second;

                LongType offset = logitsBase + tokenId * logitsElemStride;
                float logit = static_cast<float>(logitsBuf[offset]);

                if (repF != 1.0f) {
                    logit = logit > 0.0f ? logit / repF : logit * repF;
                }
                if (freqF != 0.0f) {
                    logit -= static_cast<float>(count) * freqF;
                }
                if (presF != 0.0f) {
                    logit -= presF;
                }

                logitsBuf[offset] = static_cast<T>(logit);
            }
        }
    };

    samediff::Threads::parallel_tad(func, 0, batch);
}

void applyLogitPenaltiesCpu(NDArray* logits, NDArray* inputIds,
                            double repPenalty, double freqPenalty,
                            double presPenalty, LaunchContext* context) {
    if (repPenalty == 1.0 && freqPenalty == 0.0 && presPenalty == 0.0) return;

    float repF = static_cast<float>(repPenalty);
    float freqF = static_cast<float>(freqPenalty);
    float presF = static_cast<float>(presPenalty);

    BUILD_SINGLE_SELECTOR(logits->dataType(), applyLogitPenalties_,
                          (logits, inputIds, repF, freqF, presF),
                          SD_FLOAT_TYPES);
}

// ─── Min-P filtering ────────────────────────────────────────────────────────

template <typename T>
static void applyMinPFilter_(NDArray* logits, float minPF) {
    auto rank = logits->rankOf();

    LongType batch = 1;
    LongType vocabSize;
    LongType logitsSeqLen = 1;

    if (rank == 1) {
        vocabSize = logits->sizeAt(0);
    } else if (rank == 2) {
        batch = logits->sizeAt(0);
        vocabSize = logits->sizeAt(1);
    } else {
        batch = logits->sizeAt(0);
        logitsSeqLen = logits->sizeAt(1);
        vocabSize = logits->sizeAt(2);
    }

    T* buf = logits->bufferAsT<T>();
    auto strides = logits->stridesOf();

    LongType batchStride, elemStride, rowOffset;
    if (rank == 1) {
        batchStride = 0;
        elemStride = strides[0];
        rowOffset = 0;
    } else if (rank == 2) {
        batchStride = strides[0];
        elemStride = strides[1];
        rowOffset = 0;
    } else {
        batchStride = strides[0];
        elemStride = strides[2];
        rowOffset = (logitsSeqLen - 1) * strides[1];
    }

    auto func = PRAGMA_THREADS_FOR {
        for (auto b = start; b < stop; b++) {
            LongType base = b * batchStride + rowOffset;

            // Pass 1: find max logit for numerical stability
            float maxLogit = -FLT_MAX;
            for (LongType v = 0; v < vocabSize; v++) {
                float val = static_cast<float>(buf[base + v * elemStride]);
                if (val > maxLogit) maxLogit = val;
            }

            // Pass 2: compute softmax probabilities and sum
            float sumExp = 0.0f;
            for (LongType v = 0; v < vocabSize; v++) {
                float val = static_cast<float>(buf[base + v * elemStride]);
                sumExp += sd::math::sd_exp<float, float>(val - maxLogit);
            }

            // maxProb = exp(maxLogit - maxLogit) / sumExp = 1.0 / sumExp
            // threshold in exp-space = minP * maxProb * sumExp = minP
            float threshold = minPF;

            // Pass 3: filter — set logits to -inf where prob < minP * maxProb
            for (LongType v = 0; v < vocabSize; v++) {
                LongType offset = base + v * elemStride;
                float val = static_cast<float>(buf[offset]);
                float expVal = sd::math::sd_exp<float, float>(val - maxLogit);
                if (expVal < threshold) {
                    buf[offset] = static_cast<T>(-FLT_MAX);
                }
            }
        }
    };

    samediff::Threads::parallel_tad(func, 0, batch);
}

void applyMinPFilterCpu(NDArray* logits, double minP, LaunchContext* context) {
    if (minP <= 0.0) return;

    BUILD_SINGLE_SELECTOR(logits->dataType(), applyMinPFilter_,
                          (logits, static_cast<float>(minP)),
                          SD_FLOAT_TYPES);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
