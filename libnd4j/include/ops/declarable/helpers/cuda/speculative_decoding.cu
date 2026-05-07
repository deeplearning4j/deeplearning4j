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
// Speculative decoding CUDA kernels.
// Tree-based rejection sampling and greedy verification for draft+target model inference.
//

#include <cuda_runtime.h>
#include <helpers/DebugHelper.h>
#include <array/NDArray.h>
#include <array/NDArrayFactory.h>
#include <types/float16.h>
#include <ops/declarable/helpers/speculative_decoding.h>

namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////////
// Tree speculative sampling kernel
// One block per batch element. Sequential over speculative positions.
//
// For each draft position i:
//   r = randVals[batch][i]
//   acceptProb = targetProbs[batch][i][token] / draftProbs[batch][i][token]
//   if r < acceptProb: accept, advance
//   else: sample from max(0, target - draft), stop
//////////////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL void treeSpeculativeSamplingKernel(
    const T* __restrict__ targetProbs,    // [B, maxSpecLen, V]
    const T* __restrict__ draftProbs,     // [B, maxSpecLen, V]
    const int32_t* __restrict__ draftTokenIds, // [B, maxSpecLen]
    const float* __restrict__ randVals,   // [B, maxSpecLen]
    int32_t* __restrict__ outputTokenIds, // [B, maxSpecLen+1]
    int32_t* __restrict__ numAccepted,    // [B]
    const LongType batchSize,
    const LongType maxSpecLen,
    const LongType vocabSize) {

    const LongType b = blockIdx.x;
    if (b >= batchSize) return;

    // Shared memory for adjusted distribution when rejection sampling
    extern __shared__ char sharedMemRaw[];
    float* adjustedDist = reinterpret_cast<float*>(sharedMemRaw);

    const T* batchTargetProbs = targetProbs + b * maxSpecLen * vocabSize;
    const T* batchDraftProbs = draftProbs + b * maxSpecLen * vocabSize;
    const int32_t* batchDraftTokens = draftTokenIds + b * maxSpecLen;
    const float* batchRand = randVals + b * maxSpecLen;
    int32_t* batchOutput = outputTokenIds + b * (maxSpecLen + 1);

    // Initialize output to -1 (padding)
    for (LongType i = threadIdx.x; i <= maxSpecLen; i += blockDim.x) {
        batchOutput[i] = -1;
    }
    __syncthreads();

    if (threadIdx.x != 0) return;  // Sequential logic on thread 0

    int accepted = 0;

    for (LongType pos = 0; pos < maxSpecLen; pos++) {
        int32_t draftToken = batchDraftTokens[pos];
        if (draftToken < 0) break;  // End of draft sequence

        float tProb = static_cast<float>(batchTargetProbs[pos * vocabSize + draftToken]);
        float dProb = static_cast<float>(batchDraftProbs[pos * vocabSize + draftToken]);

        // Avoid division by zero
        if (dProb <= 0.0f) dProb = 1e-10f;

        float acceptProb = fminf(1.0f, tProb / dProb);
        float r = batchRand[pos];

        if (r < acceptProb) {
            // Accept this draft token
            batchOutput[accepted] = draftToken;
            accepted++;
        } else {
            // Reject: sample from adjusted distribution max(0, target - draft)
            // For simplicity, we take the argmax of the adjusted distribution
            float bestVal = -1.0f;
            int32_t bestToken = 0;

            for (LongType v = 0; v < vocabSize; v++) {
                float tP = static_cast<float>(batchTargetProbs[pos * vocabSize + v]);
                float dP = static_cast<float>(batchDraftProbs[pos * vocabSize + v]);
                float adjusted = fmaxf(0.0f, tP - dP);
                if (adjusted > bestVal) {
                    bestVal = adjusted;
                    bestToken = static_cast<int32_t>(v);
                }
            }

            batchOutput[accepted] = bestToken;
            accepted++;
            break;  // Stop after first rejection
        }
    }

    // If all draft tokens accepted, sample one more from target's last position
    if (accepted == maxSpecLen) {
        // Argmax of target distribution at last position
        float bestVal = -1e30f;
        int32_t bestToken = 0;
        LongType lastPos = maxSpecLen - 1;

        for (LongType v = 0; v < vocabSize; v++) {
            float val = static_cast<float>(batchTargetProbs[lastPos * vocabSize + v]);
            if (val > bestVal) {
                bestVal = val;
                bestToken = static_cast<int32_t>(v);
            }
        }
        batchOutput[accepted] = bestToken;
    }

    numAccepted[b] = accepted;
}

//////////////////////////////////////////////////////////////////////////////
// Greedy tree verification kernel
// One block per batch. Compares draft tokens against target argmax.
//////////////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL void treeVerifyGreedyKernel(
    const T* __restrict__ targetLogits,    // [B, maxSpecLen, V]
    const int32_t* __restrict__ draftTokenIds, // [B, maxSpecLen]
    int32_t* __restrict__ outputTokenIds,  // [B, maxSpecLen+1]
    int32_t* __restrict__ numAccepted,     // [B]
    const LongType batchSize,
    const LongType maxSpecLen,
    const LongType vocabSize) {

    const LongType b = blockIdx.x;
    if (b >= batchSize) return;

    // Initialize output to -1
    for (LongType i = threadIdx.x; i <= maxSpecLen; i += blockDim.x) {
        outputTokenIds[b * (maxSpecLen + 1) + i] = -1;
    }
    __syncthreads();

    if (threadIdx.x != 0) return;

    int32_t* batchOutput = outputTokenIds + b * (maxSpecLen + 1);
    const int32_t* batchDraft = draftTokenIds + b * maxSpecLen;
    const T* batchLogits = targetLogits + b * maxSpecLen * vocabSize;

    int accepted = 0;

    for (LongType pos = 0; pos < maxSpecLen; pos++) {
        int32_t draftToken = batchDraft[pos];
        if (draftToken < 0) break;

        // Find argmax of target logits at this position
        float bestVal = -1e30f;
        int32_t bestToken = 0;
        const T* posLogits = batchLogits + pos * vocabSize;

        for (LongType v = 0; v < vocabSize; v++) {
            float val = static_cast<float>(posLogits[v]);
            if (val > bestVal) {
                bestVal = val;
                bestToken = static_cast<int32_t>(v);
            }
        }

        if (bestToken == draftToken) {
            // Draft matches target argmax: accept
            batchOutput[accepted] = draftToken;
            accepted++;
        } else {
            // Mismatch: output target's prediction and stop
            batchOutput[accepted] = bestToken;
            accepted++;
            break;
        }
    }

    // If all draft tokens accepted, add one more from last target position
    if (accepted == maxSpecLen) {
        const T* lastLogits = batchLogits + (maxSpecLen - 1) * vocabSize;
        float bestVal = -1e30f;
        int32_t bestToken = 0;
        for (LongType v = 0; v < vocabSize; v++) {
            float val = static_cast<float>(lastLogits[v]);
            if (val > bestVal) {
                bestVal = val;
                bestToken = static_cast<int32_t>(v);
            }
        }
        batchOutput[accepted] = bestToken;
    }

    numAccepted[b] = accepted;
}

//////////////////////////////////////////////////////////////////////////////
// Parallel argmax helper for greedy verification (large vocab)
// Uses block-level reduction for efficient argmax across vocab
//////////////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL void treeVerifyGreedyParallelKernel(
    const T* __restrict__ targetLogits,    // [B, maxSpecLen, V]
    const int32_t* __restrict__ draftTokenIds, // [B, maxSpecLen]
    int32_t* __restrict__ argmaxTokens,    // [B, maxSpecLen] workspace
    const LongType batchSize,
    const LongType maxSpecLen,
    const LongType vocabSize) {

    // Grid: (B * maxSpecLen), each block finds argmax for one position
    const LongType idx = blockIdx.x;
    if (idx >= batchSize * maxSpecLen) return;

    const T* posLogits = targetLogits + idx * vocabSize;

    extern __shared__ char sharedMemRaw[];
    float* sVals = reinterpret_cast<float*>(sharedMemRaw);
    int32_t* sIdxs = reinterpret_cast<int32_t*>(sVals + blockDim.x);

    // Thread-local max
    float threadMax = -1e30f;
    int32_t threadIdx_ = 0;

    for (LongType v = threadIdx.x; v < vocabSize; v += blockDim.x) {
        float val = static_cast<float>(posLogits[v]);
        if (val > threadMax) {
            threadMax = val;
            threadIdx_ = static_cast<int32_t>(v);
        }
    }

    sVals[threadIdx.x] = threadMax;
    sIdxs[threadIdx.x] = threadIdx_;
    __syncthreads();

    // Block reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (sVals[threadIdx.x + stride] > sVals[threadIdx.x]) {
                sVals[threadIdx.x] = sVals[threadIdx.x + stride];
                sIdxs[threadIdx.x] = sIdxs[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        argmaxTokens[idx] = sIdxs[0];
    }
}

//////////////////////////////////////////////////////////////////////////////
// Public: treeSpeculativeSampling
//////////////////////////////////////////////////////////////////////////////
void treeSpeculativeSampling(LaunchContext* context,
                              NDArray* targetProbs,
                              NDArray* draftProbs,
                              NDArray* draftTokenIds,
                              NDArray* randVals,
                              NDArray* outputTokenIds,
                              NDArray* numAccepted) {
    const LongType batchSize = targetProbs->sizeAt(0);
    const LongType maxSpecLen = targetProbs->sizeAt(1);
    const LongType vocabSize = targetProbs->sizeAt(2);

    NDArray::prepareSpecialUse({outputTokenIds, numAccepted},
                               {targetProbs, draftProbs, draftTokenIds, randVals});

    auto stream = context->getCudaStream();
    auto dtype = targetProbs->dataType();

    // One block per batch element
    int threads = 1;  // Sequential per batch for correctness (rejection is sequential)
    size_t smem = vocabSize * sizeof(float);  // For adjusted distribution

    if (dtype == DataType::FLOAT32) {
        treeSpeculativeSamplingKernel<float><<<batchSize, threads, smem, *stream>>>(
            reinterpret_cast<const float*>(targetProbs->specialBuffer()),
            reinterpret_cast<const float*>(draftProbs->specialBuffer()),
            reinterpret_cast<const int32_t*>(draftTokenIds->specialBuffer()),
            reinterpret_cast<const float*>(randVals->specialBuffer()),
            reinterpret_cast<int32_t*>(outputTokenIds->specialBuffer()),
            reinterpret_cast<int32_t*>(numAccepted->specialBuffer()),
            batchSize, maxSpecLen, vocabSize);
    } else if (dtype == DataType::HALF) {
        treeSpeculativeSamplingKernel<float16><<<batchSize, threads, smem, *stream>>>(
            reinterpret_cast<const float16*>(targetProbs->specialBuffer()),
            reinterpret_cast<const float16*>(draftProbs->specialBuffer()),
            reinterpret_cast<const int32_t*>(draftTokenIds->specialBuffer()),
            reinterpret_cast<const float*>(randVals->specialBuffer()),
            reinterpret_cast<int32_t*>(outputTokenIds->specialBuffer()),
            reinterpret_cast<int32_t*>(numAccepted->specialBuffer()),
            batchSize, maxSpecLen, vocabSize);
    } else {
        THROW_EXCEPTION("treeSpeculativeSampling: unsupported data type");
    }

    DebugHelper::checkGlobalErrorCode("treeSpeculativeSampling failed");
    NDArray::registerSpecialUse({outputTokenIds, numAccepted},
                                {targetProbs, draftProbs, draftTokenIds, randVals});
}

//////////////////////////////////////////////////////////////////////////////
// Public: treeVerifyGreedy
//////////////////////////////////////////////////////////////////////////////
void treeVerifyGreedy(LaunchContext* context,
                       NDArray* targetLogits,
                       NDArray* draftTokenIds,
                       NDArray* outputTokenIds,
                       NDArray* numAccepted) {
    const LongType batchSize = targetLogits->sizeAt(0);
    const LongType maxSpecLen = targetLogits->sizeAt(1);
    const LongType vocabSize = targetLogits->sizeAt(2);

    NDArray::prepareSpecialUse({outputTokenIds, numAccepted}, {targetLogits, draftTokenIds});

    auto stream = context->getCudaStream();
    auto dtype = targetLogits->dataType();

    if (vocabSize <= 32768) {
        // Small vocab: single-thread sequential verification
        if (dtype == DataType::FLOAT32) {
            treeVerifyGreedyKernel<float><<<batchSize, 1, 0, *stream>>>(
                reinterpret_cast<const float*>(targetLogits->specialBuffer()),
                reinterpret_cast<const int32_t*>(draftTokenIds->specialBuffer()),
                reinterpret_cast<int32_t*>(outputTokenIds->specialBuffer()),
                reinterpret_cast<int32_t*>(numAccepted->specialBuffer()),
                batchSize, maxSpecLen, vocabSize);
        } else if (dtype == DataType::HALF) {
            treeVerifyGreedyKernel<float16><<<batchSize, 1, 0, *stream>>>(
                reinterpret_cast<const float16*>(targetLogits->specialBuffer()),
                reinterpret_cast<const int32_t*>(draftTokenIds->specialBuffer()),
                reinterpret_cast<int32_t*>(outputTokenIds->specialBuffer()),
                reinterpret_cast<int32_t*>(numAccepted->specialBuffer()),
                batchSize, maxSpecLen, vocabSize);
        } else {
            THROW_EXCEPTION("treeVerifyGreedy: unsupported data type");
        }
    } else {
        // Large vocab: parallel argmax first, then sequential verify
        // Step 1: parallel argmax for all positions
        std::vector<sd::LongType> argmaxShape = {batchSize, maxSpecLen};
        auto argmaxTokens = NDArrayFactory::create_<int32_t>('c', argmaxShape, context);
        NDArray::prepareSpecialUse({argmaxTokens}, {targetLogits});

        int threads = 256;
        size_t smem = threads * (sizeof(float) + sizeof(int32_t));
        LongType totalPositions = batchSize * maxSpecLen;

        if (dtype == DataType::FLOAT32) {
            treeVerifyGreedyParallelKernel<float><<<totalPositions, threads, smem, *stream>>>(
                reinterpret_cast<const float*>(targetLogits->specialBuffer()),
                reinterpret_cast<const int32_t*>(draftTokenIds->specialBuffer()),
                reinterpret_cast<int32_t*>(argmaxTokens->specialBuffer()),
                batchSize, maxSpecLen, vocabSize);
        } else if (dtype == DataType::HALF) {
            treeVerifyGreedyParallelKernel<float16><<<totalPositions, threads, smem, *stream>>>(
                reinterpret_cast<const float16*>(targetLogits->specialBuffer()),
                reinterpret_cast<const int32_t*>(draftTokenIds->specialBuffer()),
                reinterpret_cast<int32_t*>(argmaxTokens->specialBuffer()),
                batchSize, maxSpecLen, vocabSize);
        }

        NDArray::registerSpecialUse({argmaxTokens}, {targetLogits});

        // Step 2: sequential verify using precomputed argmax
        // Use the argmax tokens as "target logits" with a simple comparison kernel
        // For now, fall back to the simple kernel
        treeVerifyGreedyKernel<float><<<batchSize, 1, 0, *stream>>>(
            reinterpret_cast<const float*>(targetLogits->specialBuffer()),
            reinterpret_cast<const int32_t*>(draftTokenIds->specialBuffer()),
            reinterpret_cast<int32_t*>(outputTokenIds->specialBuffer()),
            reinterpret_cast<int32_t*>(numAccepted->specialBuffer()),
            batchSize, maxSpecLen, vocabSize);

        delete argmaxTokens;
    }

    DebugHelper::checkGlobalErrorCode("treeVerifyGreedy failed");
    NDArray::registerSpecialUse({outputTokenIds, numAccepted}, {targetLogits, draftTokenIds});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
