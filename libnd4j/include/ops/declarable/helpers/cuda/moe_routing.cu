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
// MoE routing CUDA kernels:
// - Sigmoid top-K routing (DeepSeek V3/R1)
// - Softmax top-K routing with softcap
// - Align block size (sort + pad for batched GEMM)
// - Sum reduce (weighted expert output aggregation)
//

#include <cuda_runtime.h>
#include <helpers/DebugHelper.h>
#include <array/NDArray.h>
#include <array/NDArrayFactory.h>
#include <types/float16.h>
#include <ops/declarable/helpers/moe_routing.h>

namespace sd {
namespace ops {
namespace helpers {

constexpr int MOE_WARP_SIZE = 32;

//////////////////////////////////////////////////////////////////////////////
// Sigmoid top-K routing kernel
// One block per token. Computes sigmoid scores and selects top-K experts.
//////////////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL __launch_bounds__(128, 4) void moeTopkSigmoidKernel(
    const T* __restrict__ gatingLogits,     // [numTokens, numExperts]
    const T* __restrict__ expertBias,       // [numExperts] or nullptr
    T* __restrict__ expertScores,           // [numTokens, topK]
    int32_t* __restrict__ expertIndices,    // [numTokens, topK]
    const LongType numTokens,
    const int numExperts,
    const int topK,
    const bool renormalize,
    const float routingScale) {

    const LongType token = blockIdx.x;
    if (token >= numTokens) return;

    extern __shared__ char sharedMemRaw[];
    float* scores = reinterpret_cast<float*>(sharedMemRaw);
    int32_t* topKIdx = reinterpret_cast<int32_t*>(scores + numExperts);
    float* topKVal = reinterpret_cast<float*>(topKIdx + topK);

    const T* tokenLogits = gatingLogits + token * numExperts;

    // Step 1: Compute sigmoid scores
    for (int e = threadIdx.x; e < numExperts; e += blockDim.x) {
        float logit = static_cast<float>(tokenLogits[e]);
        if (expertBias != nullptr) {
            logit += static_cast<float>(expertBias[e]);
        }
        // sigmoid: 1 / (1 + exp(-x))
        scores[e] = 1.0f / (1.0f + __expf(-logit));
    }
    __syncthreads();

    // Step 2: Thread 0 performs top-K selection
    if (threadIdx.x == 0) {
        // Initialize top-K with -inf
        for (int k = 0; k < topK; k++) {
            topKVal[k] = -1e30f;
            topKIdx[k] = -1;
        }

        // Selection sort for top-K
        for (int e = 0; e < numExperts; e++) {
            float score = scores[e];

            // Check if this score should replace the smallest in top-K
            int minK = 0;
            float minVal = topKVal[0];
            for (int k = 1; k < topK; k++) {
                if (topKVal[k] < minVal) {
                    minVal = topKVal[k];
                    minK = k;
                }
            }

            if (score > minVal) {
                topKVal[minK] = score;
                topKIdx[minK] = e;
            }
        }

        // Optional renormalization
        float scoreSum = 0.0f;
        if (renormalize) {
            for (int k = 0; k < topK; k++) {
                scoreSum += topKVal[k];
            }
            if (scoreSum == 0.0f) scoreSum = 1.0f;
        }

        // Write output
        T* tokenScores = expertScores + token * topK;
        int32_t* tokenIndices = expertIndices + token * topK;
        for (int k = 0; k < topK; k++) {
            float finalScore = topKVal[k] * routingScale;
            if (renormalize) {
                finalScore = (topKVal[k] / scoreSum) * routingScale;
            }
            tokenScores[k] = static_cast<T>(finalScore);
            tokenIndices[k] = topKIdx[k];
        }
    }
}

//////////////////////////////////////////////////////////////////////////////
// Softmax top-K routing kernel with optional softcap
//////////////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL __launch_bounds__(128, 4) void moeTopkSoftmaxKernel(
    const T* __restrict__ gatingLogits,     // [numTokens, numExperts]
    T* __restrict__ expertScores,           // [numTokens, topK]
    int32_t* __restrict__ expertIndices,    // [numTokens, topK]
    const LongType numTokens,
    const int numExperts,
    const int topK,
    const bool renormalize,
    const float softcapScale) {

    const LongType token = blockIdx.x;
    if (token >= numTokens) return;

    extern __shared__ char sharedMemRaw[];
    float* scores = reinterpret_cast<float*>(sharedMemRaw);
    int32_t* topKIdx = reinterpret_cast<int32_t*>(scores + numExperts);
    float* topKVal = reinterpret_cast<float*>(topKIdx + topK);

    const T* tokenLogits = gatingLogits + token * numExperts;

    // Step 1: Load logits with optional softcap
    for (int e = threadIdx.x; e < numExperts; e += blockDim.x) {
        float logit = static_cast<float>(tokenLogits[e]);
        if (softcapScale > 0.0f) {
            logit = tanhf(logit / softcapScale) * softcapScale;
        }
        scores[e] = logit;
    }
    __syncthreads();

    // Step 2: Thread 0 performs softmax + top-K
    if (threadIdx.x == 0) {
        // Softmax: find max
        float maxVal = scores[0];
        for (int e = 1; e < numExperts; e++) {
            if (scores[e] > maxVal) maxVal = scores[e];
        }

        // Softmax: exp and sum
        float sumExp = 0.0f;
        for (int e = 0; e < numExperts; e++) {
            scores[e] = __expf(scores[e] - maxVal);
            sumExp += scores[e];
        }
        for (int e = 0; e < numExperts; e++) {
            scores[e] /= sumExp;
        }

        // Top-K selection
        for (int k = 0; k < topK; k++) {
            topKVal[k] = -1e30f;
            topKIdx[k] = -1;
        }

        for (int e = 0; e < numExperts; e++) {
            int minK = 0;
            float minVal = topKVal[0];
            for (int k = 1; k < topK; k++) {
                if (topKVal[k] < minVal) { minVal = topKVal[k]; minK = k; }
            }
            if (scores[e] > minVal) {
                topKVal[minK] = scores[e];
                topKIdx[minK] = e;
            }
        }

        // Optional renormalization of top-K scores
        if (renormalize) {
            float topSum = 0.0f;
            for (int k = 0; k < topK; k++) topSum += topKVal[k];
            if (topSum > 0.0f) {
                for (int k = 0; k < topK; k++) topKVal[k] /= topSum;
            }
        }

        // Write output
        T* tokenScores = expertScores + token * topK;
        int32_t* tokenIndices = expertIndices + token * topK;
        for (int k = 0; k < topK; k++) {
            tokenScores[k] = static_cast<T>(topKVal[k]);
            tokenIndices[k] = topKIdx[k];
        }
    }
}

//////////////////////////////////////////////////////////////////////////////
// MoE align block size kernel
// Phase 1: Count tokens per expert (atomicAdd)
// Phase 2: Compute expert offsets (prefix sum)
// Phase 3: Sort tokens into expert-major order
//////////////////////////////////////////////////////////////////////////////
SD_KERNEL __launch_bounds__(256, 2) void moeCountTokensKernel(
    const int32_t* __restrict__ expertIndices,  // [numTokens, topK]
    int32_t* __restrict__ tokenCounts,          // [numExperts] (atomically incremented)
    const LongType numTokens,
    const int topK) {

    const LongType idx = blockIdx.x * blockDim.x + threadIdx.x;
    const LongType totalAssignments = numTokens * topK;
    if (idx >= totalAssignments) return;

    int32_t expertId = expertIndices[idx];
    if (expertId >= 0) {
        atomicAdd(&tokenCounts[expertId], 1);
    }
}

SD_KERNEL __launch_bounds__(256, 2) void moeSortTokensKernel(
    const int32_t* __restrict__ expertIndices,   // [numTokens, topK]
    const int32_t* __restrict__ expertOffsets,    // [numExperts] prefix sum
    int32_t* __restrict__ sortedTokenIds,         // [totalPaddedTokens]
    int32_t* __restrict__ expertIds,              // [totalPaddedTokens]
    int32_t* __restrict__ writePos,               // [numExperts] (atomically incremented)
    const LongType numTokens,
    const int topK,
    const int totalPaddedTokens) {

    const LongType idx = blockIdx.x * blockDim.x + threadIdx.x;
    const LongType totalAssignments = numTokens * topK;
    if (idx >= totalAssignments) return;

    int32_t expertId = expertIndices[idx];
    if (expertId < 0) return;

    LongType tokenId = idx / topK;
    int32_t pos = atomicAdd(&writePos[expertId], 1);
    int32_t offset = expertOffsets[expertId];

    if (offset + pos < totalPaddedTokens) {
        sortedTokenIds[offset + pos] = static_cast<int32_t>(tokenId);
        expertIds[offset + pos] = expertId;
    }
}

//////////////////////////////////////////////////////////////////////////////
// MoE sum reduce kernel
// Each block handles one token's output dimension
//////////////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL __launch_bounds__(256, 2) void moeSumReduceKernel(
    const T* __restrict__ expertOutputs,   // [numTokens, topK, hiddenDim]
    const T* __restrict__ routingWeights,   // [numTokens, topK]
    T* __restrict__ output,                 // [numTokens, hiddenDim]
    const LongType numTokens,
    const LongType hiddenDim,
    const int topK) {

    const LongType token = blockIdx.x;
    if (token >= numTokens) return;

    const T* tokenExperts = expertOutputs + token * topK * hiddenDim;
    const T* tokenWeights = routingWeights + token * topK;
    T* tokenOut = output + token * hiddenDim;

    for (LongType d = threadIdx.x; d < hiddenDim; d += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < topK; k++) {
            float weight = static_cast<float>(tokenWeights[k]);
            float val = static_cast<float>(tokenExperts[k * hiddenDim + d]);
            acc += weight * val;
        }
        tokenOut[d] = static_cast<T>(acc);
    }
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
    const LongType numTokens = gatingLogits->sizeAt(0);
    const int numExperts = static_cast<int>(gatingLogits->sizeAt(1));

    NDArray::prepareSpecialUse({expertScores, expertIndices}, {gatingLogits});
    if (expertBias != nullptr) NDArray::prepareSpecialUse({}, {expertBias});

    auto stream = context->getCudaStream();
    auto dtype = gatingLogits->dataType();

    int threads = 128;
    size_t smem = numExperts * sizeof(float) + topK * sizeof(int32_t) + topK * sizeof(float);

    const void* biasPtr = (expertBias != nullptr) ? expertBias->specialBuffer() : nullptr;

    if (dtype == DataType::FLOAT32) {
        moeTopkSigmoidKernel<float><<<numTokens, threads, smem, *stream>>>(
            reinterpret_cast<const float*>(gatingLogits->specialBuffer()),
            reinterpret_cast<const float*>(biasPtr),
            reinterpret_cast<float*>(expertScores->specialBuffer()),
            reinterpret_cast<int32_t*>(expertIndices->specialBuffer()),
            numTokens, numExperts, topK, renormalize, routingScale);
    } else if (dtype == DataType::HALF) {
        moeTopkSigmoidKernel<float16><<<numTokens, threads, smem, *stream>>>(
            reinterpret_cast<const float16*>(gatingLogits->specialBuffer()),
            reinterpret_cast<const float16*>(biasPtr),
            reinterpret_cast<float16*>(expertScores->specialBuffer()),
            reinterpret_cast<int32_t*>(expertIndices->specialBuffer()),
            numTokens, numExperts, topK, renormalize, routingScale);
    } else {
        THROW_EXCEPTION("moeTopkSigmoid: unsupported data type");
    }

    DebugHelper::checkGlobalErrorCode("moeTopkSigmoid failed");
    NDArray::registerSpecialUse({expertScores, expertIndices}, {gatingLogits});
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
    const LongType numTokens = gatingLogits->sizeAt(0);
    const int numExperts = static_cast<int>(gatingLogits->sizeAt(1));

    NDArray::prepareSpecialUse({expertScores, expertIndices}, {gatingLogits});

    auto stream = context->getCudaStream();
    auto dtype = gatingLogits->dataType();

    int threads = 128;
    size_t smem = numExperts * sizeof(float) + topK * sizeof(int32_t) + topK * sizeof(float);

    if (dtype == DataType::FLOAT32) {
        moeTopkSoftmaxKernel<float><<<numTokens, threads, smem, *stream>>>(
            reinterpret_cast<const float*>(gatingLogits->specialBuffer()),
            reinterpret_cast<float*>(expertScores->specialBuffer()),
            reinterpret_cast<int32_t*>(expertIndices->specialBuffer()),
            numTokens, numExperts, topK, renormalize, softcapScale);
    } else if (dtype == DataType::HALF) {
        moeTopkSoftmaxKernel<float16><<<numTokens, threads, smem, *stream>>>(
            reinterpret_cast<const float16*>(gatingLogits->specialBuffer()),
            reinterpret_cast<float16*>(expertScores->specialBuffer()),
            reinterpret_cast<int32_t*>(expertIndices->specialBuffer()),
            numTokens, numExperts, topK, renormalize, softcapScale);
    } else {
        THROW_EXCEPTION("moeTopkSoftmax: unsupported data type");
    }

    DebugHelper::checkGlobalErrorCode("moeTopkSoftmax failed");
    NDArray::registerSpecialUse({expertScores, expertIndices}, {gatingLogits});
}

//////////////////////////////////////////////////////////////////////////////
// Public: moeAlignBlockSize
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
    const int totalPaddedTokens = static_cast<int>(sortedTokenIds->lengthOf());

    NDArray::prepareSpecialUse({sortedTokenIds, expertIds, numTokensPerExpert}, {expertIndices});

    auto stream = context->getCudaStream();

    // Initialize outputs
    cudaMemsetAsync(numTokensPerExpert->specialBuffer(), 0, numExperts * sizeof(int32_t), *stream);
    cudaMemsetAsync(sortedTokenIds->specialBuffer(), 0xFF, totalPaddedTokens * sizeof(int32_t), *stream);  // -1
    cudaMemsetAsync(expertIds->specialBuffer(), 0xFF, totalPaddedTokens * sizeof(int32_t), *stream);

    // Phase 1: Count tokens per expert
    int threads1 = 256;
    LongType totalAssignments = numTokens * topK;
    int blocks1 = (totalAssignments + threads1 - 1) / threads1;

    moeCountTokensKernel<<<blocks1, threads1, 0, *stream>>>(
        reinterpret_cast<const int32_t*>(expertIndices->specialBuffer()),
        reinterpret_cast<int32_t*>(numTokensPerExpert->specialBuffer()),
        numTokens, topK);

    // Phase 2: Compute padded expert offsets (CPU-side prefix sum for simplicity)
    // Sync to read counts
    cudaStreamSynchronize(*stream);

    std::vector<int32_t> counts(numExperts);
    cudaMemcpyAsync(counts.data(), numTokensPerExpert->specialBuffer(),
               numExperts * sizeof(int32_t), cudaMemcpyDeviceToHost, *stream);
    // Sync so CPU can read back counts before computing prefix sums
    cudaStreamSynchronize(*stream);

    std::vector<int32_t> offsets(numExperts);
    int32_t offset = 0;
    for (int e = 0; e < numExperts; e++) {
        offsets[e] = offset;
        int paddedCount = ((counts[e] + blockSize - 1) / blockSize) * blockSize;
        offset += paddedCount;
    }

    // Upload offsets
    std::vector<sd::LongType> expertOffsetsShape = {numExperts};
    auto expertOffsets = NDArrayFactory::create_<int32_t>('c', expertOffsetsShape, context);
    cudaMemcpyAsync(expertOffsets->specialBuffer(), offsets.data(),
               numExperts * sizeof(int32_t), cudaMemcpyHostToDevice, *stream);

    // Phase 3: Sort tokens into expert-major order
    std::vector<sd::LongType> writePosShape = {numExperts};
    auto writePos = NDArrayFactory::create_<int32_t>('c', writePosShape, context);
    cudaMemsetAsync(writePos->specialBuffer(), 0, numExperts * sizeof(int32_t), *stream);

    moeSortTokensKernel<<<blocks1, threads1, 0, *stream>>>(
        reinterpret_cast<const int32_t*>(expertIndices->specialBuffer()),
        reinterpret_cast<const int32_t*>(expertOffsets->specialBuffer()),
        reinterpret_cast<int32_t*>(sortedTokenIds->specialBuffer()),
        reinterpret_cast<int32_t*>(expertIds->specialBuffer()),
        reinterpret_cast<int32_t*>(writePos->specialBuffer()),
        numTokens, topK, totalPaddedTokens);

    delete expertOffsets;
    delete writePos;

    DebugHelper::checkGlobalErrorCode("moeAlignBlockSize failed");
    NDArray::registerSpecialUse({sortedTokenIds, expertIds, numTokensPerExpert}, {expertIndices});
}

//////////////////////////////////////////////////////////////////////////////
// Public: moeSumReduce
//////////////////////////////////////////////////////////////////////////////
void moeSumReduce(LaunchContext* context,
                   NDArray* expertOutputs,
                   NDArray* routingWeights,
                   NDArray* output,
                   int topK) {
    const LongType numTokens = output->sizeAt(0);
    const LongType hiddenDim = output->sizeAt(1);

    NDArray::prepareSpecialUse({output}, {expertOutputs, routingWeights});

    auto stream = context->getCudaStream();
    auto dtype = output->dataType();

    int threads = 256;
    if (hiddenDim <= 256) threads = 128;

    if (dtype == DataType::FLOAT32) {
        moeSumReduceKernel<float><<<numTokens, threads, 0, *stream>>>(
            reinterpret_cast<const float*>(expertOutputs->specialBuffer()),
            reinterpret_cast<const float*>(routingWeights->specialBuffer()),
            reinterpret_cast<float*>(output->specialBuffer()),
            numTokens, hiddenDim, topK);
    } else if (dtype == DataType::HALF) {
        moeSumReduceKernel<float16><<<numTokens, threads, 0, *stream>>>(
            reinterpret_cast<const float16*>(expertOutputs->specialBuffer()),
            reinterpret_cast<const float16*>(routingWeights->specialBuffer()),
            reinterpret_cast<float16*>(output->specialBuffer()),
            numTokens, hiddenDim, topK);
    } else {
        THROW_EXCEPTION("moeSumReduce: unsupported data type");
    }

    DebugHelper::checkGlobalErrorCode("moeSumReduce failed");
    NDArray::registerSpecialUse({output}, {expertOutputs, routingWeights});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
