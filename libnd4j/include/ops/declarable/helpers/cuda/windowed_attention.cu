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

#include <cuda_runtime.h>
#include <helpers/PointersManager.h>
#include <ops/declarable/helpers/windowed_attention.h>
#include <system/selective_rendering.h>

#if NOT_EXCLUDED(OP_windowed_attention)

namespace sd {
namespace ops {
namespace helpers {

template <typename T>
__global__ static void windowedAttentionKernel(
    const T* query,
    const T* key,
    const T* value,
    const T* relativePositionBias,
    const T* attentionMask,
    T* output,
    T* attentionWeights,
    const sd::LongType batchSize,
    const sd::LongType seqLen,
    const sd::LongType numHeads,
    const sd::LongType headDim,
    const int windowSize,
    const int shiftSize,
    const T scaleFactor,
    const bool returnWeights,
    const sd::LongType batchStride,
    const sd::LongType seqStride,
    const sd::LongType headStride,
    const sd::LongType dimStride,
    const sd::LongType outBatchStride,
    const sd::LongType outSeqStride,
    const sd::LongType outHeadStride,
    const sd::LongType outDimStride) {

    // Each block handles one (batch, head, query_position)
    const sd::LongType idx = blockIdx.x;
    const sd::LongType totalPositions = batchSize * numHeads * seqLen;

    if (idx >= totalPositions) return;

    const sd::LongType b = idx / (numHeads * seqLen);
    const sd::LongType h = (idx / seqLen) % numHeads;
    const sd::LongType q = idx % seqLen;

    const int halfWindow = windowSize / 2;

    // Apply shift if specified
    sd::LongType shiftedQ = q;
    if (shiftSize > 0) {
        shiftedQ = (q + shiftSize) % seqLen;
    }

    // Window bounds
    sd::LongType windowStart = (shiftedQ > halfWindow) ? shiftedQ - halfWindow : 0;
    sd::LongType windowEnd = min(seqLen, shiftedQ + halfWindow + 1);
    int actualWindowSize = static_cast<int>(windowEnd - windowStart);

    // Pointers
    const T* queryVec = query + b * batchStride + h * headStride + q * seqStride;
    const T* batchKey = key + b * batchStride + h * headStride;
    const T* batchValue = value + b * batchStride + h * headStride;
    T* outputVec = output + b * outBatchStride + h * outHeadStride + q * outSeqStride;

    // Shared memory for scores and softmax
    extern __shared__ char sharedMem[];
    T* scores = reinterpret_cast<T*>(sharedMem);
    T* softmaxScores = scores + windowSize;

    // Compute attention scores - each thread handles part of the window
    T maxScore = static_cast<T>(-3.4028235e+38f);
    for (int w = threadIdx.x; w < actualWindowSize; w += blockDim.x) {
        sd::LongType k = windowStart + w;
        const T* keyVec = batchKey + k * seqStride;

        // Dot product
        T score = static_cast<T>(0);
        for (sd::LongType d = 0; d < headDim; d++) {
            score += queryVec[d * dimStride] * keyVec[d * dimStride];
        }
        score *= scaleFactor;

        // Add relative position bias if provided
        if (relativePositionBias != nullptr) {
            int biasQ = static_cast<int>(q % windowSize);
            int biasK = static_cast<int>(k % windowSize);
            score += relativePositionBias[h * windowSize * windowSize + biasQ * windowSize + biasK];
        }

        // Apply attention mask if provided
        if (attentionMask != nullptr) {
            T maskVal = attentionMask[b * seqLen * seqLen + q * seqLen + k];
            if (maskVal < static_cast<T>(-1e30)) {
                score = static_cast<T>(-3.4028235e+38f);
            }
        }

        scores[w] = score;
    }
    __syncthreads();

    // Find max for numerical stability
    if (threadIdx.x == 0) {
        maxScore = scores[0];
        for (int w = 1; w < actualWindowSize; w++) {
            if (scores[w] > maxScore) maxScore = scores[w];
        }
    }
    __syncthreads();

    // Compute softmax
    T sumExp = static_cast<T>(0);
    for (int w = threadIdx.x; w < actualWindowSize; w += blockDim.x) {
        softmaxScores[w] = exp(scores[w] - maxScore);
    }
    __syncthreads();

    // Sum reduction
    if (threadIdx.x == 0) {
        for (int w = 0; w < actualWindowSize; w++) {
            sumExp += softmaxScores[w];
        }
    }
    __syncthreads();

    // Normalize
    for (int w = threadIdx.x; w < actualWindowSize; w += blockDim.x) {
        softmaxScores[w] /= sumExp;
    }
    __syncthreads();

    // Store attention weights if requested
    if (returnWeights && attentionWeights != nullptr && threadIdx.x == 0) {
        for (int w = 0; w < actualWindowSize; w++) {
            sd::LongType k = windowStart + w;
            attentionWeights[b * numHeads * seqLen * seqLen + h * seqLen * seqLen + q * seqLen + k] = softmaxScores[w];
        }
    }

    // Weighted sum of values - each thread handles part of head_dim
    for (sd::LongType d = threadIdx.x; d < headDim; d += blockDim.x) {
        T sum = static_cast<T>(0);
        for (int w = 0; w < actualWindowSize; w++) {
            sd::LongType v = windowStart + w;
            const T* valueVec = batchValue + v * seqStride;
            sum += softmaxScores[w] * valueVec[d * dimStride];
        }
        outputVec[d * outDimStride] = sum;
    }
}

template <typename T>
static void windowedAttentionCuda_(sd::LaunchContext* context,
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

    const auto batchSize = query->sizeAt(0);
    const auto seqLen = query->sizeAt(1);
    const auto headDim = query->sizeAt(3);

    T scaleFactor = scale > 0.0 ? static_cast<T>(scale) : static_cast<T>(1.0 / std::sqrt(static_cast<double>(headDim)));

    // Strides
    const auto batchStride = query->strideAt(0);
    const auto seqStride = query->strideAt(1);
    const auto headStride = query->strideAt(2);
    const auto dimStride = query->strideAt(3);

    const auto outBatchStride = output->strideAt(0);
    const auto outSeqStride = output->strideAt(1);
    const auto outHeadStride = output->strideAt(2);
    const auto outDimStride = output->strideAt(3);

    const sd::LongType totalPositions = batchSize * numHeads * seqLen;
    const int blockSize = 64;
    const size_t sharedMemSize = 2 * windowSize * sizeof(T);

    PointersManager manager(context, "windowedAttention");

    const T* queryBuffer = query->specialBufferasT<T>();
    const T* keyBuffer = key->specialBufferasT<T>();
    const T* valueBuffer = value->specialBufferasT<T>();
    const T* biasBuffer = relativePositionBias != nullptr ? relativePositionBias->specialBufferasT<T>() : nullptr;
    const T* maskBuffer = attentionMask != nullptr ? attentionMask->specialBufferasT<T>() : nullptr;
    T* outputBuffer = output->specialBufferasT<T>();
    T* weightsBuffer = returnWeights && attentionWeights != nullptr ?
                       attentionWeights->specialBufferasT<T>() : nullptr;

    windowedAttentionKernel<T><<<totalPositions, blockSize, sharedMemSize, *context->getCudaStream()>>>(
        queryBuffer, keyBuffer, valueBuffer, biasBuffer, maskBuffer,
        outputBuffer, weightsBuffer,
        batchSize, seqLen, numHeads, headDim,
        windowSize, shiftSize, scaleFactor, returnWeights,
        batchStride, seqStride, headStride, dimStride,
        outBatchStride, outSeqStride, outHeadStride, outDimStride);

    manager.synchronize();
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

    BUILD_SINGLE_SELECTOR(query->dataType(), windowedAttentionCuda_,
                          (context, query, key, value, relativePositionBias, attentionMask,
                           output, attentionWeights, windowSize, numHeads, shiftSize, scale, returnWeights),
                          SD_FLOAT_TYPES);
}

BUILD_SINGLE_TEMPLATE(void windowedAttentionCuda_,
                      (sd::LaunchContext* context, NDArray* query, NDArray* key,
                       NDArray* value, NDArray* relativePositionBias, NDArray* attentionMask,
                       NDArray* output, NDArray* attentionWeights, int windowSize, int numHeads,
                       int shiftSize, double scale, bool returnWeights),
                      SD_FLOAT_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif
