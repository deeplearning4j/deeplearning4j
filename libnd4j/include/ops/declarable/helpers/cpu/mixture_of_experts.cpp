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
#include <ops/declarable/helpers/mixture_of_experts.h>

#include <algorithm>
#include <cmath>
#include <vector>
#include <math/templatemath.h>
#include <system/selective_rendering.h>

#if NOT_EXCLUDED(OP_mixture_of_experts)

namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static void softmax(T* data, int size) {
    T maxVal = data[0];
    for (int i = 1; i < size; i++) {
        if (data[i] > maxVal) maxVal = data[i];
    }
    T sum = static_cast<T>(0);
    for (int i = 0; i < size; i++) {
        data[i] = sd::math::sd_exp<T>(data[i] - maxVal);
        sum += data[i];
    }
    for (int i = 0; i < size; i++) {
        data[i] /= sum;
    }
}

template <typename T>
static void topKIndices(const T* data, int size, int k, std::vector<int>& indices, std::vector<T>& values) {
    // Simple top-K selection
    std::vector<std::pair<T, int>> pairs(size);
    for (int i = 0; i < size; i++) {
        pairs[i] = {data[i], i};
    }

    // Partial sort to get top-K
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
static void mixtureOfExpertsSimple_(sd::LaunchContext* context,
                                     NDArray* input,
                                     NDArray* gatingWeights,
                                     NDArray* expertWeights,
                                     NDArray* expertBias,
                                     NDArray* output,
                                     int numExperts,
                                     int topK,
                                     bool normalizeProbs) {

    // Input: [batch, seq_len, hidden_dim]
    // Gating: [hidden_dim, num_experts]
    // Expert weights: [num_experts, hidden_dim, expert_dim] (assuming symmetric for simplicity)
    // Output: [batch, seq_len, hidden_dim]

    const auto batchSize = input->sizeAt(0);
    const auto seqLen = input->sizeAt(1);
    const auto hiddenDim = input->sizeAt(2);
    const auto expertDim = expertWeights->sizeAt(2);

    const T* inputBuffer = input->bufferAsT<T>();
    const T* gatingBuffer = gatingWeights->bufferAsT<T>();
    const T* expertBuffer = expertWeights->bufferAsT<T>();
    const T* biasBuffer = expertBias != nullptr ? expertBias->bufferAsT<T>() : nullptr;
    T* outputBuffer = output->bufferAsT<T>();

    // Strides
    const auto inBatchStride = input->strideAt(0);
    const auto inSeqStride = input->strideAt(1);
    const auto inDimStride = input->strideAt(2);

    const auto gatingDimStride = gatingWeights->strideAt(0);
    const auto gatingExpStride = gatingWeights->strideAt(1);

    const auto expExpStride = expertWeights->strideAt(0);
    const auto expInStride = expertWeights->strideAt(1);
    const auto expOutStride = expertWeights->strideAt(2);

    const auto outBatchStride = output->strideAt(0);
    const auto outSeqStride = output->strideAt(1);
    const auto outDimStride = output->strideAt(2);

    // Initialize output to zero
    output->nullify();

    // Process each token in parallel
    auto func = [&](uint64_t thread_id, int64_t start, int64_t stop, int64_t increment) {
        std::vector<T> routerLogits(numExperts);
        std::vector<int> selectedExperts(topK);
        std::vector<T> expertScores(topK);
        std::vector<T> expertOutput(expertDim);

        for (sd::LongType idx = start; idx < stop; idx += increment) {
            const sd::LongType b = idx / seqLen;
            const sd::LongType s = idx % seqLen;

            const T* tokenInput = inputBuffer + b * inBatchStride + s * inSeqStride;
            T* tokenOutput = outputBuffer + b * outBatchStride + s * outSeqStride;

            // Compute router logits: input @ gatingWeights
            for (int e = 0; e < numExperts; e++) {
                T logit = static_cast<T>(0);
                for (sd::LongType d = 0; d < hiddenDim; d++) {
                    logit += tokenInput[d * inDimStride] * gatingBuffer[d * gatingDimStride + e * gatingExpStride];
                }
                routerLogits[e] = logit;
            }

            // Apply softmax and select top-K
            softmax(routerLogits.data(), numExperts);
            topKIndices(routerLogits.data(), numExperts, topK, selectedExperts, expertScores);

            // Optionally renormalize selected expert scores.
            // WARNING: renormalization rescales gradients by numExperts/topK in the
            // backward pass (e.g. ~16x for numExperts=32, topK=2), causing the
            // "~30x too large gradients" bug.  Only enable when inference-only.
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

            // Apply each selected expert and combine
            for (int k = 0; k < topK; k++) {
                int expertIdx = selectedExperts[k];
                T score = expertScores[k];

                const T* expertW = expertBuffer + expertIdx * expExpStride;

                // Compute expert output: input @ expertWeights[expert]
                for (sd::LongType od = 0; od < expertDim; od++) {
                    T sum = static_cast<T>(0);
                    for (sd::LongType id = 0; id < hiddenDim; id++) {
                        sum += tokenInput[id * inDimStride] * expertW[id * expInStride + od * expOutStride];
                    }
                    if (biasBuffer != nullptr) {
                        sum += biasBuffer[expertIdx * expertDim + od];
                    }
                    expertOutput[od] = sum;
                }

                // Add weighted expert output to token output
                // Note: This assumes expert_dim == hidden_dim for simplicity
                // Real implementations often have expert_dim != hidden_dim with projection
                sd::LongType outDim = std::min(expertDim, hiddenDim);
                for (sd::LongType d = 0; d < outDim; d++) {
                    tokenOutput[d * outDimStride] += score * expertOutput[d];
                }
            }
        }
    };

    samediff::Threads::parallel_for(func, 0, batchSize * seqLen, 1);
}

void mixtureOfExpertsSimple(sd::LaunchContext* context,
                             NDArray* input,
                             NDArray* gatingWeights,
                             NDArray* expertWeights,
                             NDArray* expertBias,
                             NDArray* output,
                             int numExperts,
                             int topK,
                             bool normalizeProbs) {

    BUILD_SINGLE_SELECTOR(input->dataType(), mixtureOfExpertsSimple_,
                          (context, input, gatingWeights, expertWeights, expertBias, output, numExperts, topK, normalizeProbs),
                          SD_FLOAT_TYPES);
}

void mixtureOfExperts(sd::LaunchContext* context,
                       NDArray* input,
                       NDArray* gatingWeights,
                       const std::vector<NDArray*>& expertWeights,
                       const std::vector<NDArray*>& expertBias,
                       NDArray* output,
                       NDArray* routerLogits,
                       NDArray* expertIndices,
                       int numExperts,
                       int topK,
                       int expertCapacity,
                       bool useSoftmaxGating,
                       bool useAuxLoss) {

    // For simplicity, delegate to simple implementation if expert weights are provided as single tensor
    // This version handles the list of expert weights case
    // TODO: Implement full version with capacity constraints and aux loss

    if (expertWeights.size() == 1) {
        // Note: normalizeProbs=false to avoid ~numExperts/topK gradient amplification
        // in the backward pass. Pass true only for inference-only (non-training) use.
        mixtureOfExpertsSimple(context, input, gatingWeights, expertWeights[0],
                                expertBias.empty() ? nullptr : expertBias[0],
                                output, numExperts, topK, false);
    } else {
        // Handle multiple expert weight tensors - implement later
        THROW_EXCEPTION("mixtureOfExperts: Multiple expert weight tensors not yet supported");
    }
}


}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif
