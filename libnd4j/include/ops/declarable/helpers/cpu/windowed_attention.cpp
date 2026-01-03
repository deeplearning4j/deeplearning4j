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
#include <ops/declarable/helpers/windowed_attention.h>

#include <cmath>
#include <system/selective_rendering.h>

#if NOT_EXCLUDED(OP_windowed_attention)

namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static void windowedAttention_(sd::LaunchContext* context,
                                NDArray* query,
                                NDArray* key,
                                NDArray* value,
                                NDArray* relativePositionBias,
                                NDArray* attentionMask,
                                NDArray* output,
                                NDArray* attentionWeights,
                                int windowSize,
                                int numHeads,
                                int shiftSize,
                                double scale,
                                bool returnWeights) {

    // Input shape: [batch, seq_len, num_heads, head_dim]
    const auto batchSize = query->sizeAt(0);
    const auto seqLen = query->sizeAt(1);
    const auto headDim = query->sizeAt(3);

    // Auto-compute scale if not provided
    T scaleFactor = scale > 0.0 ? static_cast<T>(scale) : static_cast<T>(1.0 / std::sqrt(static_cast<double>(headDim)));

    // Half window for local attention
    const int halfWindow = windowSize / 2;

    // Get raw buffers
    const T* queryBuffer = query->bufferAsT<T>();
    const T* keyBuffer = key->bufferAsT<T>();
    const T* valueBuffer = value->bufferAsT<T>();
    const T* biasBuffer = relativePositionBias != nullptr ? relativePositionBias->bufferAsT<T>() : nullptr;
    const T* maskBuffer = attentionMask != nullptr ? attentionMask->bufferAsT<T>() : nullptr;
    T* outputBuffer = output->bufferAsT<T>();
    T* weightsBuffer = returnWeights && attentionWeights != nullptr ? attentionWeights->bufferAsT<T>() : nullptr;

    // Get strides for query/key/value [batch, seq, heads, dim]
    const auto batchStride = query->strideAt(0);
    const auto seqStride = query->strideAt(1);
    const auto headStride = query->strideAt(2);
    const auto dimStride = query->strideAt(3);

    // Output strides
    const auto outBatchStride = output->strideAt(0);
    const auto outSeqStride = output->strideAt(1);
    const auto outHeadStride = output->strideAt(2);
    const auto outDimStride = output->strideAt(3);

    // Parallel over batch and heads
    auto func = [&](uint64_t thread_id, int64_t start, int64_t stop, int64_t increment) {
        // Temporary buffers for attention scores
        std::vector<T> scores(windowSize);
        std::vector<T> softmaxScores(windowSize);

        for (sd::LongType idx = start; idx < stop; idx += increment) {
            const sd::LongType b = idx / numHeads;
            const sd::LongType h = idx % numHeads;

            const T* batchQuery = queryBuffer + b * batchStride + h * headStride;
            const T* batchKey = keyBuffer + b * batchStride + h * headStride;
            const T* batchValue = valueBuffer + b * batchStride + h * headStride;
            T* batchOutput = outputBuffer + b * outBatchStride + h * outHeadStride;

            // For each query position
            for (sd::LongType q = 0; q < seqLen; q++) {
                // Apply shift if specified
                sd::LongType shiftedQ = q;
                if (shiftSize > 0) {
                    shiftedQ = (q + shiftSize) % seqLen;
                }

                const T* queryVec = batchQuery + q * seqStride;
                T* outputVec = batchOutput + q * outSeqStride;

                // Determine window bounds
                sd::LongType windowStart = std::max(static_cast<sd::LongType>(0), shiftedQ - halfWindow);
                sd::LongType windowEnd = std::min(seqLen, shiftedQ + halfWindow + 1);
                int actualWindowSize = static_cast<int>(windowEnd - windowStart);

                // Compute attention scores within window
                T maxScore = -1e9f;
                for (int w = 0; w < actualWindowSize; w++) {
                    sd::LongType k = windowStart + w;
                    const T* keyVec = batchKey + k * seqStride;

                    // Dot product
                    T score = static_cast<T>(0);
                    for (sd::LongType d = 0; d < headDim; d++) {
                        score += queryVec[d * dimStride] * keyVec[d * dimStride];
                    }
                    score *= scaleFactor;

                    // Add relative position bias if provided
                    if (biasBuffer != nullptr) {
                        // Bias shape: [num_heads, window_size, window_size]
                        // Map positions to bias indices
                        int biasQ = static_cast<int>(q % windowSize);
                        int biasK = static_cast<int>(k % windowSize);
                        score += biasBuffer[h * windowSize * windowSize + biasQ * windowSize + biasK];
                    }

                    // Apply attention mask if provided
                    if (maskBuffer != nullptr) {
                        // Assume mask shape compatible with attention
                        T maskVal = maskBuffer[b * seqLen * seqLen + q * seqLen + k];
                        if (maskVal < static_cast<T>(-1e4)) {
                            score = static_cast<T>(-1e9);
                        }
                    }

                    scores[w] = score;
                    maxScore = std::max(maxScore, score);
                }

                // Softmax
                T sumExp = static_cast<T>(0);
                for (int w = 0; w < actualWindowSize; w++) {
                    softmaxScores[w] = std::exp(scores[w] - maxScore);
                    sumExp += softmaxScores[w];
                }
                for (int w = 0; w < actualWindowSize; w++) {
                    softmaxScores[w] /= sumExp;
                }

                // Store attention weights if requested
                if (weightsBuffer != nullptr) {
                    for (int w = 0; w < actualWindowSize; w++) {
                        sd::LongType k = windowStart + w;
                        weightsBuffer[b * numHeads * seqLen * seqLen + h * seqLen * seqLen + q * seqLen + k] = softmaxScores[w];
                    }
                }

                // Weighted sum of values
                for (sd::LongType d = 0; d < headDim; d++) {
                    T sum = static_cast<T>(0);
                    for (int w = 0; w < actualWindowSize; w++) {
                        sd::LongType v = windowStart + w;
                        const T* valueVec = batchValue + v * seqStride;
                        sum += softmaxScores[w] * valueVec[d * dimStride];
                    }
                    outputVec[d * outDimStride] = sum;
                }
            }
        }
    };

    samediff::Threads::parallel_for(func, 0, batchSize * numHeads, 1);
}

void windowedAttention(sd::LaunchContext* context,
                        NDArray* query,
                        NDArray* key,
                        NDArray* value,
                        NDArray* relativePositionBias,
                        NDArray* attentionMask,
                        NDArray* output,
                        NDArray* attentionWeights,
                        int windowSize,
                        int numHeads,
                        int shiftSize,
                        double scale,
                        bool returnWeights) {

    BUILD_SINGLE_SELECTOR(query->dataType(), windowedAttention_,
                          (context, query, key, value, relativePositionBias, attentionMask,
                           output, attentionWeights, windowSize, numHeads, shiftSize, scale, returnWeights),
                          SD_FLOAT_TYPES);
}

BUILD_SINGLE_TEMPLATE(void windowedAttention_,
                      (sd::LaunchContext* context, NDArray* query, NDArray* key,
                       NDArray* value, NDArray* relativePositionBias, NDArray* attentionMask,
                       NDArray* output, NDArray* attentionWeights, int windowSize, int numHeads,
                       int shiftSize, double scale, bool returnWeights),
                      SD_FLOAT_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif
