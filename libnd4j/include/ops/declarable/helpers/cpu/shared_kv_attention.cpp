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

//
// @author Eclipse Deeplearning4j
//

#include <execution/Threads.h>
#include <math/templatemath.h>
#include <ops/declarable/helpers/shared_kv_attention.h>

#include <cmath>
#include <algorithm>
#include <vector>

namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static void sharedKvAttention_(LaunchContext* context,
                                NDArray* query,
                                NDArray* sharedKey,
                                NDArray* sharedValue,
                                NDArray* mask,
                                NDArray* output,
                                int numHeads,
                                int numKvHeads,
                                bool causal,
                                int slidingWindowSize,
                                double scale) {

    // query:       [batch, numHeads,   qSeqLen,  headDim]
    // sharedKey:   [batch, numKvHeads, kvSeqLen, headDim]
    // sharedValue: [batch, numKvHeads, kvSeqLen, headDim]
    // mask:        [batch, 1, qSeqLen, kvSeqLen] or nullptr
    // output:      [batch, numHeads,   qSeqLen,  headDim]

    const auto batchSize = query->sizeAt(0);
    const auto qSeqLen   = query->sizeAt(2);
    const auto headDim   = query->sizeAt(3);
    const auto kvSeqLen  = sharedKey->sizeAt(2);

    const int headsPerGroup = numHeads / numKvHeads;

    const T scaleFactor = static_cast<T>(scale);
    const T negInf = static_cast<T>(-3.4028235e+38f);

    // Get raw buffers and strides
    if (mask != nullptr) {
        NDArray::preparePrimaryUse({output}, {query, sharedKey, sharedValue, mask});
    } else {
        NDArray::preparePrimaryUse({output}, {query, sharedKey, sharedValue});
    }

    const T* qBuf = query->bufferAsT<T>();
    const T* kBuf = sharedKey->bufferAsT<T>();
    const T* vBuf = sharedValue->bufferAsT<T>();
    const T* mBuf = (mask != nullptr) ? mask->bufferAsT<T>() : nullptr;
    T* oBuf       = output->bufferAsT<T>();

    // Query strides: [batch, numHeads, qSeqLen, headDim]
    const auto qBatchStride = query->strideAt(0);
    const auto qHeadStride  = query->strideAt(1);
    const auto qSeqStride   = query->strideAt(2);
    const auto qDimStride   = query->strideAt(3);

    // Key strides: [batch, numKvHeads, kvSeqLen, headDim]
    const auto kBatchStride = sharedKey->strideAt(0);
    const auto kHeadStride  = sharedKey->strideAt(1);
    const auto kSeqStride   = sharedKey->strideAt(2);
    const auto kDimStride   = sharedKey->strideAt(3);

    // Value strides: [batch, numKvHeads, kvSeqLen, headDim]
    const auto vBatchStride = sharedValue->strideAt(0);
    const auto vHeadStride  = sharedValue->strideAt(1);
    const auto vSeqStride   = sharedValue->strideAt(2);
    const auto vDimStride   = sharedValue->strideAt(3);

    // Mask strides: [batch, 1, qSeqLen, kvSeqLen]
    sd::LongType mBatchStride = 0, mSeqStride = 0, mKvStride = 0;
    if (mBuf != nullptr) {
        mBatchStride = mask->strideAt(0);
        // dim 1 is broadcast (size 1), skip
        mSeqStride   = mask->strideAt(2);
        mKvStride    = mask->strideAt(3);
    }

    // Output strides: same layout as query
    const auto oBatchStride = output->strideAt(0);
    const auto oHeadStride  = output->strideAt(1);
    const auto oSeqStride   = output->strideAt(2);
    const auto oDimStride   = output->strideAt(3);

    // Parallel over batch * numHeads
    const sd::LongType totalWork = batchSize * numHeads;

    auto func = [&](uint64_t thread_id, int64_t start, int64_t stop, int64_t increment) {
        // Per-thread scratch for attention scores
        std::vector<T> scores(kvSeqLen);

        for (sd::LongType idx = start; idx < stop; idx += increment) {
            const sd::LongType b = idx / numHeads;
            const sd::LongType h = idx % numHeads;
            const int kvHead = h / headsPerGroup;  // GQA mapping

            const T* queryBase = qBuf + b * qBatchStride + h * qHeadStride;
            const T* keyBase   = kBuf + b * kBatchStride + kvHead * kHeadStride;
            const T* valBase   = vBuf + b * vBatchStride + kvHead * vHeadStride;
            T* outBase         = oBuf + b * oBatchStride + h * oHeadStride;

            for (sd::LongType q = 0; q < qSeqLen; q++) {
                const T* queryVec = queryBase + q * qSeqStride;
                T* outputVec      = outBase + q * oSeqStride;

                // Compute Q @ K^T scores for all kv positions
                T maxScore = negInf;
                for (sd::LongType k = 0; k < kvSeqLen; k++) {
                    const T* keyVec = keyBase + k * kSeqStride;

                    // Dot product
                    T dot = static_cast<T>(0);
                    for (sd::LongType d = 0; d < headDim; d++) {
                        dot += queryVec[d * qDimStride] * keyVec[d * kDimStride];
                    }
                    dot *= scaleFactor;

                    // Apply causal mask: future positions get -inf
                    if (causal && k > q) {
                        dot = negInf;
                    }

                    // Apply sliding window mask
                    if (slidingWindowSize > 0 && (q - k) > slidingWindowSize) {
                        dot = negInf;
                    }

                    // Apply external additive mask
                    if (mBuf != nullptr) {
                        T maskVal = mBuf[b * mBatchStride + q * mSeqStride + k * mKvStride];
                        dot += maskVal;
                    }

                    scores[k] = dot;
                    maxScore = std::max(maxScore, dot);
                }

                // Softmax: exp and sum
                T sumExp = static_cast<T>(0);
                for (sd::LongType k = 0; k < kvSeqLen; k++) {
                    scores[k] = sd::math::sd_exp<T>(scores[k] - maxScore);
                    sumExp += scores[k];
                }
                // Normalize
                T invSum = static_cast<T>(1) / sumExp;
                for (sd::LongType k = 0; k < kvSeqLen; k++) {
                    scores[k] *= invSum;
                }

                // Weighted sum of values: output = scores @ V
                for (sd::LongType d = 0; d < headDim; d++) {
                    T acc = static_cast<T>(0);
                    for (sd::LongType k = 0; k < kvSeqLen; k++) {
                        acc += scores[k] * valBase[k * vSeqStride + d * vDimStride];
                    }
                    outputVec[d * oDimStride] = acc;
                }
            }
        }
    };

    samediff::Threads::parallel_for(func, 0, totalWork, 1);

    if (mask != nullptr) {
        NDArray::registerPrimaryUse({output}, {query, sharedKey, sharedValue, mask});
    } else {
        NDArray::registerPrimaryUse({output}, {query, sharedKey, sharedValue});
    }
}

void sharedKvAttention(LaunchContext* context,
                        NDArray* query,
                        NDArray* sharedKey,
                        NDArray* sharedValue,
                        NDArray* mask,
                        NDArray* output,
                        int numHeads,
                        int numKvHeads,
                        bool causal,
                        int slidingWindowSize,
                        double scale) {

    BUILD_SINGLE_SELECTOR(query->dataType(), sharedKvAttention_,
                          (context, query, sharedKey, sharedValue, mask, output,
                           numHeads, numKvHeads, causal, slidingWindowSize, scale),
                          SD_FLOAT_TYPES);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
