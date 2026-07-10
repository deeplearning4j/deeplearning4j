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
#include <cstdint>
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

void applyLogitPenalties(NDArray* logits, NDArray* inputIds,
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

void applyMinPFilter(NDArray* logits, double minP, LaunchContext* context) {
    if (minP <= 0.0) return;

    BUILD_SINGLE_SELECTOR(logits->dataType(), applyMinPFilter_,
                          (logits, static_cast<float>(minP)),
                          SD_FLOAT_TYPES);
}

// ─── Typical-p (entropy-deviation) filtering ────────────────────────────────
//
// Meister et al. 2023 "Locally Typical Sampling" semantics:
//   H = -sum_i p_i log(p_i)
//   deviation_i = |-log(p_i) - H|
//   Sort by deviation ascending; accumulate cumulative probability
//   until >= typicalP; mask remaining tokens to -inf.
//
// CPU: exact sort over (deviation, index) pairs — O(V log V).

template <typename T>
static void applyTypicalPFilter_(NDArray* logits, float typicalPF) {
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

            // Pass 1: softmax numerics (max-shift for stability)
            float maxLogit = -FLT_MAX;
            for (LongType v = 0; v < vocabSize; v++) {
                float val = static_cast<float>(buf[base + v * elemStride]);
                if (val > maxLogit) maxLogit = val;
            }
            float sumExp = 0.0f;
            for (LongType v = 0; v < vocabSize; v++) {
                sumExp += sd::math::sd_exp<float, float>(
                    static_cast<float>(buf[base + v * elemStride]) - maxLogit);
            }

            // Pass 2: compute entropy H = -sum p_i log(p_i)
            // and (deviation, index) pairs
            float entropy = 0.0f;
            std::vector<std::pair<float, LongType>> devs(vocabSize);
            for (LongType v = 0; v < vocabSize; v++) {
                float expVal = sd::math::sd_exp<float, float>(
                    static_cast<float>(buf[base + v * elemStride]) - maxLogit);
                float p = expVal / sumExp;
                float negLogP = -sd::math::sd_log<float, float>(p + 1e-10f);
                entropy += p * negLogP;
                devs[v] = {negLogP, v};  // store -log p; subtract entropy after
            }

            // Pass 3: deviation = |(-log p_i) - H|; sort ascending by deviation
            for (auto& dv : devs) {
                dv.first = std::abs(dv.first - entropy);
            }
            std::sort(devs.begin(), devs.end(),
                      [](const std::pair<float, LongType>& a,
                         const std::pair<float, LongType>& b) {
                          return a.first < b.first;
                      });

            // Pass 4: accumulate probability along sorted order; find cutoff
            float cumProb = 0.0f;
            LongType cutoff = vocabSize;
            for (LongType k = 0; k < vocabSize; k++) {
                LongType v = devs[k].second;
                float expVal = sd::math::sd_exp<float, float>(
                    static_cast<float>(buf[base + v * elemStride]) - maxLogit);
                cumProb += expVal / sumExp;
                if (cumProb >= typicalPF) {
                    cutoff = k + 1;
                    break;
                }
            }

            // Pass 4b: tie-inclusive cutoff. Tokens whose deviation is numerically equal
            // to the last included token's are equally typical — excluding them by sort
            // order would be arbitrary and diverge from the CUDA threshold kernel, which
            // always includes whole deviation classes. 1e-6 absorbs float noise in
            // -log(p) - H for identical probabilities.
            if (cutoff > 0 && cutoff < vocabSize) {
                float cutDev = devs[cutoff - 1].first;
                while (cutoff < vocabSize && devs[cutoff].first <= cutDev + 1e-6f) cutoff++;
            }

            // Pass 5: mask all tokens beyond the cutoff
            for (LongType k = cutoff; k < vocabSize; k++) {
                LongType v = devs[k].second;
                buf[base + v * elemStride] = static_cast<T>(-FLT_MAX);
            }
        }
    };

    samediff::Threads::parallel_tad(func, 0, batch);
}

void applyTypicalPFilter(NDArray* logits, double typicalP, LaunchContext* context) {
    // typicalP >= 1.0 is a no-op; <= 0 is invalid (caller guards)
    if (typicalP >= 1.0 || typicalP <= 0.0) return;

    BUILD_SINGLE_SELECTOR(logits->dataType(), applyTypicalPFilter_,
                          (logits, static_cast<float>(typicalP)),
                          SD_FLOAT_TYPES);
}

// ─── XTC (Exclude Top Choices) filtering ────────────────────────────────────
//
// Reference: omlx/utils/sampling.py apply_xtc (ztc_special_tokens not supported
// here — handled by callers suppressing special tokens before this filter).
//
// With probability xtcProbability: among tokens with softmax prob >= xtcThreshold,
// if there are >= 2 such tokens, keep only the LOWEST-probability one (mask the rest).
// Tie-breaking: lowest index among equal probabilities is kept (stable argmin).
//
// RNG for the apply/skip decision: LCG mixing of seed to produce a uniform scalar.

template <typename T>
static void applyXtcFilter_(NDArray* logits, float xtcProbF, float xtcThrF,
                              LongType seed, LongType batchOffset) {
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
            // Per-batch RNG decision: mix seed + batch index
            // Simple LCG + xorshift to get a float in [0, 1)
            uint64_t state = static_cast<uint64_t>(seed) ^ (static_cast<uint64_t>(b + batchOffset) * 6364136223846793005ULL + 1442695040888963407ULL);
            state ^= (state >> 30);
            state *= 0xbf58476d1ce4e5b9ULL;
            state ^= (state >> 27);
            state *= 0x94d049bb133111ebULL;
            state ^= (state >> 31);
            float u = static_cast<float>(state >> 11) / static_cast<float>(1ULL << 53);

            if (u > xtcProbF) {
                // Skip XTC for this batch element
                continue;
            }

            LongType base = b * batchStride + rowOffset;

            // Compute softmax probabilities for threshold comparison
            float maxLogit = -FLT_MAX;
            for (LongType v = 0; v < vocabSize; v++) {
                float val = static_cast<float>(buf[base + v * elemStride]);
                if (val > maxLogit) maxLogit = val;
            }
            float sumExp = 0.0f;
            for (LongType v = 0; v < vocabSize; v++) {
                sumExp += sd::math::sd_exp<float, float>(
                    static_cast<float>(buf[base + v * elemStride]) - maxLogit);
            }

            // Find tokens with p >= xtcThrF and track the argmin-probability one
            float minP = FLT_MAX;
            LongType minIdx = -1;
            int countAbove = 0;
            for (LongType v = 0; v < vocabSize; v++) {
                float expVal = sd::math::sd_exp<float, float>(
                    static_cast<float>(buf[base + v * elemStride]) - maxLogit);
                float p = expVal / sumExp;
                if (p >= xtcThrF) {
                    countAbove++;
                    if (p < minP) {
                        minP = p;
                        minIdx = v;
                    }
                }
            }

            // Need >= 2 qualifying tokens to apply XTC (otherwise no diversity to gain)
            if (countAbove < 2) continue;

            // Mask all qualifying tokens EXCEPT the lowest-probability one
            for (LongType v = 0; v < vocabSize; v++) {
                if (v == static_cast<LongType>(minIdx)) continue;
                float expVal = sd::math::sd_exp<float, float>(
                    static_cast<float>(buf[base + v * elemStride]) - maxLogit);
                float p = expVal / sumExp;
                if (p >= xtcThrF) {
                    buf[base + v * elemStride] = static_cast<T>(-FLT_MAX);
                }
            }
        }
    };

    samediff::Threads::parallel_tad(func, 0, batch);
}

void applyXtcFilter(NDArray* logits, double xtcProbability, double xtcThreshold,
                    LongType seed, LaunchContext* context) {
    if (xtcProbability <= 0.0) return;

    BUILD_SINGLE_SELECTOR(logits->dataType(), applyXtcFilter_,
                          (logits, static_cast<float>(xtcProbability),
                           static_cast<float>(xtcThreshold),
                           seed, static_cast<LongType>(0)),
                          SD_FLOAT_TYPES);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
