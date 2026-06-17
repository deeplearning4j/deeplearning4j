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
// MoE routing — CPU implementation.
// Sigmoid and softmax top-K routing, align block size, sum reduce.
//

#include <execution/Threads.h>
#include <math/templatemath.h>
#include <ops/declarable/helpers/moe_routing.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////////
// MoE top-K sigmoid routing (CPU)
// score_e = sigmoid(logit_e + bias_e), then select top-K.
//////////////////////////////////////////////////////////////////////////////
template <typename T>
static void moeTopkSigmoidCpu_(NDArray* gatingLogits, NDArray* expertBias,
                                 NDArray* expertScores, NDArray* expertIndices,
                                 int topK, bool renormalize, float routingScale) {
    const LongType numTokens = gatingLogits->sizeAt(0);
    const LongType numExperts = gatingLogits->sizeAt(1);

    const T* logits = gatingLogits->bufferAsT<T>();
    const T* bias = (expertBias != nullptr) ? expertBias->bufferAsT<T>() : nullptr;
    T* scores = expertScores->bufferAsT<T>();
    int32_t* indices = reinterpret_cast<int32_t*>(expertIndices->buffer());

    auto func = PRAGMA_THREADS_FOR {
        // Per-thread scratch for top-K selection
        std::vector<std::pair<float, int32_t>> expertVals(numExperts);

        for (auto t = start; t < stop; ++t) {
            const T* tokenLogits = logits + t * numExperts;

            // Compute sigmoid scores
            for (LongType e = 0; e < numExperts; ++e) {
                float logit = static_cast<float>(tokenLogits[e]);
                if (bias != nullptr) {
                    logit += static_cast<float>(bias[e]);
                }
                float score = 1.0f / (1.0f + sd::math::sd_exp<float>(-logit));
                expertVals[e] = {score, static_cast<int32_t>(e)};
            }

            // Partial sort to find top-K (descending by score)
            std::partial_sort(expertVals.begin(), expertVals.begin() + topK,
                             expertVals.end(),
                             [](const auto& a, const auto& b) { return a.first > b.first; });

            // Optional renormalization
            float sumScores = 0.0f;
            if (renormalize) {
                for (int k = 0; k < topK; ++k) {
                    sumScores += expertVals[k].first;
                }
                if (sumScores == 0.0f) sumScores = 1.0f;
            }

            // Write outputs
            T* tokenScores = scores + t * topK;
            int32_t* tokenIndices = indices + t * topK;
            for (int k = 0; k < topK; ++k) {
                float s = expertVals[k].first;
                if (renormalize) {
                    s /= sumScores;
                }
                s *= routingScale;
                tokenScores[k] = static_cast<T>(s);
                tokenIndices[k] = expertVals[k].second;
            }
        }
    };
    samediff::Threads::parallel_tad(func, 0, numTokens);
}

//////////////////////////////////////////////////////////////////////////////
// MoE top-K softmax routing (CPU)
// softmax(logits) with optional tanh softcapping, then select top-K.
//////////////////////////////////////////////////////////////////////////////
template <typename T>
static void moeTopkSoftmaxCpu_(NDArray* gatingLogits, NDArray* expertScores,
                                 NDArray* expertIndices, int topK,
                                 bool renormalize, float softcapScale) {
    const LongType numTokens = gatingLogits->sizeAt(0);
    const LongType numExperts = gatingLogits->sizeAt(1);

    const T* logits = gatingLogits->bufferAsT<T>();
    T* scores = expertScores->bufferAsT<T>();
    int32_t* indices = reinterpret_cast<int32_t*>(expertIndices->buffer());

    auto func = PRAGMA_THREADS_FOR {
        std::vector<float> softmaxVals(numExperts);
        std::vector<std::pair<float, int32_t>> expertVals(numExperts);

        for (auto t = start; t < stop; ++t) {
            const T* tokenLogits = logits + t * numExperts;

            // Optional softcapping: logits = tanh(logits / scale) * scale
            float maxLogit = -std::numeric_limits<float>::max();
            for (LongType e = 0; e < numExperts; ++e) {
                float val = static_cast<float>(tokenLogits[e]);
                if (softcapScale > 0.0f) {
                    val = std::tanh(val / softcapScale) * softcapScale;
                }
                softmaxVals[e] = val;
                if (val > maxLogit) maxLogit = val;
            }

            // Softmax: exp(x - max) / sum
            float sumExp = 0.0f;
            for (LongType e = 0; e < numExperts; ++e) {
                softmaxVals[e] = sd::math::sd_exp<float>(softmaxVals[e] - maxLogit);
                sumExp += softmaxVals[e];
            }
            float invSum = 1.0f / sumExp;
            for (LongType e = 0; e < numExperts; ++e) {
                softmaxVals[e] *= invSum;
                expertVals[e] = {softmaxVals[e], static_cast<int32_t>(e)};
            }

            // Partial sort for top-K
            std::partial_sort(expertVals.begin(), expertVals.begin() + topK,
                             expertVals.end(),
                             [](const auto& a, const auto& b) { return a.first > b.first; });

            // Optional renormalization of top-K
            float sumTopK = 0.0f;
            if (renormalize) {
                for (int k = 0; k < topK; ++k) {
                    sumTopK += expertVals[k].first;
                }
                if (sumTopK == 0.0f) sumTopK = 1.0f;
            }

            T* tokenScores = scores + t * topK;
            int32_t* tokenIndices = indices + t * topK;
            for (int k = 0; k < topK; ++k) {
                float s = expertVals[k].first;
                if (renormalize) {
                    s /= sumTopK;
                }
                tokenScores[k] = static_cast<T>(s);
                tokenIndices[k] = expertVals[k].second;
            }
        }
    };
    samediff::Threads::parallel_tad(func, 0, numTokens);
}

//////////////////////////////////////////////////////////////////////////////
// MoE align block size (CPU)
// Sort tokens by expert, pad to blockSize multiples.
//////////////////////////////////////////////////////////////////////////////
void moeAlignBlockSize(LaunchContext* context,
                        NDArray* expertIndices,
                        int numExperts,
                        int blockSize,
                        NDArray* sortedTokenIds,
                        NDArray* expertIds,
                        NDArray* numTokensPerExpert) {
    const LongType numTokens = expertIndices->sizeAt(0);
    const int topK = static_cast<int>(expertIndices->sizeAt(1));
    const int32_t* idxPtr = reinterpret_cast<const int32_t*>(expertIndices->buffer());

    // Count tokens per expert
    std::vector<int32_t> counts(numExperts, 0);
    for (LongType t = 0; t < numTokens; ++t) {
        for (int k = 0; k < topK; ++k) {
            int32_t expert = idxPtr[t * topK + k];
            if (expert >= 0 && expert < numExperts) {
                counts[expert]++;
            }
        }
    }

    // Write numTokensPerExpert
    int32_t* ntpe = reinterpret_cast<int32_t*>(numTokensPerExpert->buffer());
    for (int e = 0; e < numExperts; ++e) {
        ntpe[e] = counts[e];
    }

    // Compute padded counts and offsets
    std::vector<int32_t> paddedCounts(numExperts);
    std::vector<int32_t> offsets(numExperts);
    int32_t totalPadded = 0;
    for (int e = 0; e < numExperts; ++e) {
        paddedCounts[e] = ((counts[e] + blockSize - 1) / blockSize) * blockSize;
        offsets[e] = totalPadded;
        totalPadded += paddedCounts[e];
    }

    // Collect token IDs per expert
    std::vector<std::vector<int32_t>> perExpertTokens(numExperts);
    for (int e = 0; e < numExperts; ++e) {
        perExpertTokens[e].reserve(counts[e]);
    }

    for (LongType t = 0; t < numTokens; ++t) {
        for (int k = 0; k < topK; ++k) {
            int32_t expert = idxPtr[t * topK + k];
            if (expert >= 0 && expert < numExperts) {
                perExpertTokens[expert].push_back(static_cast<int32_t>(t));
            }
        }
    }

    // Fill sorted output arrays
    int32_t* sortedPtr = reinterpret_cast<int32_t*>(sortedTokenIds->buffer());
    int32_t* expertIdsPtr = reinterpret_cast<int32_t*>(expertIds->buffer());

    // Initialize to padding values
    LongType totalLen = sortedTokenIds->lengthOf();
    for (LongType i = 0; i < totalLen; ++i) {
        sortedPtr[i] = static_cast<int32_t>(numTokens);  // padding token ID
        expertIdsPtr[i] = numExperts;  // invalid expert
    }

    for (int e = 0; e < numExperts; ++e) {
        int32_t off = offsets[e];
        for (size_t i = 0; i < perExpertTokens[e].size(); ++i) {
            sortedPtr[off + i] = perExpertTokens[e][i];
            expertIdsPtr[off + i] = e;
        }
        // Padding positions keep default values
        for (int32_t i = static_cast<int32_t>(perExpertTokens[e].size()); i < paddedCounts[e]; ++i) {
            expertIdsPtr[off + i] = e;
        }
    }

    sortedTokenIds->tickWriteHost();
    expertIds->tickWriteHost();
    numTokensPerExpert->tickWriteHost();
}

//////////////////////////////////////////////////////////////////////////////
// MoE sum reduce (CPU)
// output[t, d] = sum_k(expertOutputs[t, k, d] * routingWeights[t, k])
//////////////////////////////////////////////////////////////////////////////
void moeSumReduce(LaunchContext* context,
                   NDArray* expertOutputs,
                   NDArray* routingWeights,
                   NDArray* output,
                   int topK) {
    const LongType numTokens = output->sizeAt(0);
    const LongType hiddenDim = output->sizeAt(1);

    auto dtype = expertOutputs->dataType();

    if (dtype == DataType::FLOAT32) {
        const float* expOut = expertOutputs->bufferAsT<float>();
        const float* weights = routingWeights->bufferAsT<float>();
        float* out = output->bufferAsT<float>();

        auto func = PRAGMA_THREADS_FOR {
            for (auto t = start; t < stop; ++t) {
                float* outRow = out + t * hiddenDim;
                const float* wRow = weights + t * topK;

                // Zero output
                for (LongType d = 0; d < hiddenDim; ++d) {
                    outRow[d] = 0.0f;
                }

                // Accumulate weighted expert outputs
                for (int k = 0; k < topK; ++k) {
                    float w = wRow[k];
                    const float* expRow = expOut + (t * topK + k) * hiddenDim;
                    for (LongType d = 0; d < hiddenDim; ++d) {
                        outRow[d] += expRow[d] * w;
                    }
                }
            }
        };
        samediff::Threads::parallel_tad(func, 0, numTokens);
    } else if (dtype == DataType::DOUBLE) {
        const double* expOut = expertOutputs->bufferAsT<double>();
        const double* weights = routingWeights->bufferAsT<double>();
        double* out = output->bufferAsT<double>();

        auto func = PRAGMA_THREADS_FOR {
            for (auto t = start; t < stop; ++t) {
                double* outRow = out + t * hiddenDim;
                const double* wRow = weights + t * topK;

                for (LongType d = 0; d < hiddenDim; ++d) {
                    outRow[d] = 0.0;
                }

                for (int k = 0; k < topK; ++k) {
                    double w = wRow[k];
                    const double* expRow = expOut + (t * topK + k) * hiddenDim;
                    for (LongType d = 0; d < hiddenDim; ++d) {
                        outRow[d] += expRow[d] * w;
                    }
                }
            }
        };
        samediff::Threads::parallel_tad(func, 0, numTokens);
    } else {
        THROW_EXCEPTION("moeSumReduce CPU: unsupported data type");
    }

    output->tickWriteHost();
}

//////////////////////////////////////////////////////////////////////////////
// Public: moeTopkSigmoid
//////////////////////////////////////////////////////////////////////////////
void moeTopkSigmoid(LaunchContext* context,
                      NDArray* gatingLogits,
                      NDArray* expertBias,
                      NDArray* expertScores,
                      NDArray* expertIndices,
                      int topK,
                      bool renormalize,
                      float routingScale) {
    BUILD_SINGLE_SELECTOR(gatingLogits->dataType(), moeTopkSigmoidCpu_,
                          (gatingLogits, expertBias, expertScores, expertIndices,
                           topK, renormalize, routingScale), SD_FLOAT_TYPES);

    expertScores->tickWriteHost();
    expertIndices->tickWriteHost();
}

//////////////////////////////////////////////////////////////////////////////
// Public: moeTopkSoftmax
//////////////////////////////////////////////////////////////////////////////
void moeTopkSoftmax(LaunchContext* context,
                      NDArray* gatingLogits,
                      NDArray* expertScores,
                      NDArray* expertIndices,
                      int topK,
                      bool renormalize,
                      float softcapScale) {
    BUILD_SINGLE_SELECTOR(gatingLogits->dataType(), moeTopkSoftmaxCpu_,
                          (gatingLogits, expertScores, expertIndices,
                           topK, renormalize, softcapScale), SD_FLOAT_TYPES);

    expertScores->tickWriteHost();
    expertIndices->tickWriteHost();
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
