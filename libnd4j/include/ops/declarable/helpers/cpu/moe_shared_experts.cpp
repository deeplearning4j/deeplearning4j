/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//
// @author Eclipse Deeplearning4j
//

#include <execution/Threads.h>
#include <helpers/LoopsCoordsHelper.h>
#include <math/templatemath.h>
#include <ops/declarable/helpers/moe_shared_experts.h>

#include <algorithm>
#include <cmath>
#include <vector>
#include <system/selective_rendering.h>

#if NOT_EXCLUDED(OP_moe_shared_experts)

namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static void softmaxInPlace(T* data, int size) {
    T maxVal = data[0];
    for (int i = 1; i < size; i++) {
        if (data[i] > maxVal) maxVal = data[i];
    }
    T sum = static_cast<T>(0);
    for (int i = 0; i < size; i++) {
        data[i] = sd::math::sd_exp<T, T>(data[i] - maxVal);
        sum += data[i];
    }
    for (int i = 0; i < size; i++) {
        data[i] /= sum;
    }
}

template <typename T>
static void topKSelect(const T* data, int size, int k,
                       std::vector<int>& indices, std::vector<T>& values) {
    std::vector<std::pair<T, int>> pairs(size);
    for (int i = 0; i < size; i++) {
        pairs[i] = {data[i], i};
    }
    std::partial_sort(pairs.begin(), pairs.begin() + k, pairs.end(),
                      [](const std::pair<T, int>& a, const std::pair<T, int>& b) {
                          return a.first > b.first;
                      });
    indices.resize(k);
    values.resize(k);
    for (int i = 0; i < k; i++) {
        indices[i] = pairs[i].second;
        values[i] = pairs[i].first;
    }
}

template <typename T>
static T siluActivation(T x) {
    return x / (static_cast<T>(1) + sd::math::sd_exp<T, T>(-x));
}

template <typename T>
static void moeSharedExperts_(sd::LaunchContext* context,
                               NDArray* input,
                               NDArray* routerWeights,
                               NDArray* routedExpertWeights,
                               NDArray* sharedGateProj,
                               NDArray* sharedUpProj,
                               NDArray* sharedDownProj,
                               NDArray* routedExpertBias,
                               NDArray* output,
                               NDArray* routerProbs,
                               NDArray* expertIndices,
                               int numRoutedExperts,
                               int topK,
                               bool normalizeProbs,
                               double capacityFactor) {

    const auto batchSize = input->sizeAt(0);
    const auto seqLen = input->sizeAt(1);
    const auto hiddenDim = input->sizeAt(2);
    const auto expertHidden = routedExpertWeights->sizeAt(2);
    const auto sharedIntermediate = sharedGateProj->sizeAt(1);

    const T* inputBuf = input->bufferAsT<T>();
    const T* routerBuf = routerWeights->bufferAsT<T>();
    const T* expertBuf = routedExpertWeights->bufferAsT<T>();
    const T* gateProjBuf = sharedGateProj->bufferAsT<T>();
    const T* upProjBuf = sharedUpProj->bufferAsT<T>();
    const T* downProjBuf = sharedDownProj->bufferAsT<T>();
    const T* biasBuf = routedExpertBias != nullptr ? routedExpertBias->bufferAsT<T>() : nullptr;
    T* outputBuf = output->bufferAsT<T>();
    T* routerProbsBuf = routerProbs != nullptr ? routerProbs->bufferAsT<T>() : nullptr;

    // Input strides
    const auto inBatchStride = input->strideAt(0);
    const auto inSeqStride = input->strideAt(1);
    const auto inDimStride = input->strideAt(2);

    // Router strides
    const auto rtrDimStride = routerWeights->strideAt(0);
    const auto rtrExpStride = routerWeights->strideAt(1);

    // Routed expert strides [num_experts, hidden_size, expert_hidden]
    const auto expExpStride = routedExpertWeights->strideAt(0);
    const auto expInStride = routedExpertWeights->strideAt(1);
    const auto expOutStride = routedExpertWeights->strideAt(2);

    // Shared expert strides
    const auto gateInStride = sharedGateProj->strideAt(0);
    const auto gateOutStride = sharedGateProj->strideAt(1);
    const auto upInStride = sharedUpProj->strideAt(0);
    const auto upOutStride = sharedUpProj->strideAt(1);
    const auto downInStride = sharedDownProj->strideAt(0);
    const auto downOutStride = sharedDownProj->strideAt(1);

    // Output strides
    const auto outBatchStride = output->strideAt(0);
    const auto outSeqStride = output->strideAt(1);
    const auto outDimStride = output->strideAt(2);

    // Router probs strides
    sd::LongType rpBatchStride = 0, rpSeqStride = 0, rpExpStride = 0;
    if (routerProbs != nullptr) {
        rpBatchStride = routerProbs->strideAt(0);
        rpSeqStride = routerProbs->strideAt(1);
        rpExpStride = routerProbs->strideAt(2);
    }

    output->nullify();

    auto func = [&](uint64_t thread_id, int64_t start, int64_t stop, int64_t increment) {
        std::vector<T> logits(numRoutedExperts);
        std::vector<int> selectedExperts(topK);
        std::vector<T> expertScores(topK);
        std::vector<T> gateVals(sharedIntermediate);
        std::vector<T> upVals(sharedIntermediate);
        std::vector<T> sharedOut(hiddenDim);
        std::vector<T> expertOutput(expertHidden);

        for (sd::LongType idx = start; idx < stop; idx += increment) {
            const sd::LongType b = idx / seqLen;
            const sd::LongType s = idx % seqLen;

            const T* tokenIn = inputBuf + b * inBatchStride + s * inSeqStride;
            T* tokenOut = outputBuf + b * outBatchStride + s * outSeqStride;

            // === Shared expert: SwiGLU ===
            // gate_proj: [hidden_size, shared_intermediate] -> gateVals
            for (sd::LongType j = 0; j < sharedIntermediate; j++) {
                T val = static_cast<T>(0);
                for (sd::LongType d = 0; d < hiddenDim; d++) {
                    val += tokenIn[d * inDimStride] * gateProjBuf[d * gateInStride + j * gateOutStride];
                }
                gateVals[j] = siluActivation(val);
            }

            // up_proj: [hidden_size, shared_intermediate] -> upVals
            for (sd::LongType j = 0; j < sharedIntermediate; j++) {
                T val = static_cast<T>(0);
                for (sd::LongType d = 0; d < hiddenDim; d++) {
                    val += tokenIn[d * inDimStride] * upProjBuf[d * upInStride + j * upOutStride];
                }
                upVals[j] = val;
            }

            // Element-wise multiply: gateVals * upVals
            for (sd::LongType j = 0; j < sharedIntermediate; j++) {
                gateVals[j] *= upVals[j];
            }

            // down_proj: [shared_intermediate, hidden_size] -> sharedOut
            for (sd::LongType d = 0; d < hiddenDim; d++) {
                T val = static_cast<T>(0);
                for (sd::LongType j = 0; j < sharedIntermediate; j++) {
                    val += gateVals[j] * downProjBuf[j * downInStride + d * downOutStride];
                }
                sharedOut[d] = val;
            }

            // Write shared expert output to token output
            for (sd::LongType d = 0; d < hiddenDim; d++) {
                tokenOut[d * outDimStride] = sharedOut[d];
            }

            // === Routed experts ===
            // Compute router logits: input @ routerWeights
            for (int e = 0; e < numRoutedExperts; e++) {
                T logit = static_cast<T>(0);
                for (sd::LongType d = 0; d < hiddenDim; d++) {
                    logit += tokenIn[d * inDimStride] * routerBuf[d * rtrDimStride + e * rtrExpStride];
                }
                logits[e] = logit;
            }

            // Softmax
            softmaxInPlace(logits.data(), numRoutedExperts);

            // Store router probs if requested
            if (routerProbsBuf != nullptr) {
                T* rpToken = routerProbsBuf + b * rpBatchStride + s * rpSeqStride;
                for (int e = 0; e < numRoutedExperts; e++) {
                    rpToken[e * rpExpStride] = logits[e];
                }
            }

            // Top-K selection
            topKSelect(logits.data(), numRoutedExperts, topK, selectedExperts, expertScores);

            // Store expert indices if requested
            if (expertIndices != nullptr) {
                for (int k = 0; k < topK; k++) {
                    expertIndices->p(b, s, k, static_cast<sd::LongType>(selectedExperts[k]));
                }
            }

            // Normalize selected expert scores
            if (normalizeProbs) {
                T scoreSum = static_cast<T>(0);
                for (int k = 0; k < topK; k++) {
                    scoreSum += expertScores[k];
                }
                if (scoreSum > static_cast<T>(0)) {
                    for (int k = 0; k < topK; k++) {
                        expertScores[k] /= scoreSum;
                    }
                }
            }

            // Apply each selected expert and accumulate onto shared output
            for (int k = 0; k < topK; k++) {
                int expertIdx = selectedExperts[k];
                T score = expertScores[k];
                const T* expertW = expertBuf + expertIdx * expExpStride;

                for (sd::LongType od = 0; od < expertHidden; od++) {
                    T sum = static_cast<T>(0);
                    for (sd::LongType id = 0; id < hiddenDim; id++) {
                        sum += tokenIn[id * inDimStride] * expertW[id * expInStride + od * expOutStride];
                    }
                    if (biasBuf != nullptr) {
                        sum += biasBuf[expertIdx * expertHidden + od];
                    }
                    expertOutput[od] = sum;
                }

                // Add weighted expert output (assuming expertHidden == hiddenDim)
                sd::LongType outDim = std::min(expertHidden, hiddenDim);
                for (sd::LongType d = 0; d < outDim; d++) {
                    tokenOut[d * outDimStride] += score * expertOutput[d];
                }
            }
        }
    };

    samediff::Threads::parallel_for(func, 0, batchSize * seqLen, 1);
}

void moeSharedExperts(sd::LaunchContext* context,
                       NDArray* input,
                       NDArray* routerWeights,
                       NDArray* routedExpertWeights,
                       NDArray* sharedGateProj,
                       NDArray* sharedUpProj,
                       NDArray* sharedDownProj,
                       NDArray* routedExpertBias,
                       NDArray* output,
                       NDArray* routerProbs,
                       NDArray* expertIndices,
                       int numRoutedExperts,
                       int topK,
                       bool normalizeProbs,
                       double capacityFactor) {

    BUILD_SINGLE_SELECTOR(input->dataType(), moeSharedExperts_,
                          (context, input, routerWeights, routedExpertWeights,
                           sharedGateProj, sharedUpProj, sharedDownProj, routedExpertBias,
                           output, routerProbs, expertIndices,
                           numRoutedExperts, topK, normalizeProbs, capacityFactor),
                          SD_FLOAT_TYPES);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif
