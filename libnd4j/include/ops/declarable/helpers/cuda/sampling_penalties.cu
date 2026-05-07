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
#include <array/NDArray.h>
#include <helpers/DebugHelper.h>
#include <cuda_runtime.h>
#include <cfloat>

#include "execution/cuda/LaunchDims.h"

namespace sd {
namespace ops {
namespace helpers {

// ────────────────────────────────────────────────────────────────────────────
// Repetition / frequency / presence penalty kernel
//
// Strategy: For each batch element, one thread-block iterates over inputIds
// to build per-token counts in shared memory (hash table), then applies
// penalties to the logits for those tokens.
//
// For large vocabularies and short sequences (typical decode), it's more
// efficient to iterate over input tokens (short) and scatter-update logits
// rather than iterate over the full vocabulary.
// ────────────────────────────────────────────────────────────────────────────

// Hash table size — must be power of 2. Handles up to ~3K unique tokens
// per batch element without excessive collisions.
#define PENALTY_HASH_SIZE 4096
#define PENALTY_HASH_MASK (PENALTY_HASH_SIZE - 1)

/**
 * Kernel: count token occurrences from inputIds and apply penalties to logits.
 *
 * One block per batch element. Thread 0 scans inputIds to build a hash table
 * of (tokenId, count), then all threads cooperate to apply penalties.
 *
 * Hash table layout in shared memory:
 *   LongType keys[PENALTY_HASH_SIZE]   — token IDs (-1 = empty)
 *   int     counts[PENALTY_HASH_SIZE]  — occurrence counts
 */
template <typename T>
static SD_KERNEL void applyPenaltiesKernel(void* vLogits,
                                            const LongType logitsRowStride,
                                            const LongType logitsElemStride,
                                            const void* vInputIds,
                                            const LongType idsRowStride,
                                            const LongType idsElemStride,
                                            const LongType seqLen,
                                            const LongType vocabSize,
                                            const float repPenalty,
                                            const float freqPenalty,
                                            const float presPenalty) {
    extern __shared__ char sharedMem[];
    auto sKeys = reinterpret_cast<LongType*>(sharedMem);
    auto sCounts = reinterpret_cast<int*>(sKeys + PENALTY_HASH_SIZE);

    auto logits = reinterpret_cast<T*>(vLogits);
    auto inputIds = reinterpret_cast<const LongType*>(vInputIds);
    LongType b = blockIdx.x;

    // Initialize hash table
    for (int i = threadIdx.x; i < PENALTY_HASH_SIZE; i += blockDim.x) {
        sKeys[i] = -1;
        sCounts[i] = 0;
    }
    __syncthreads();

    // Thread 0 builds the hash table by iterating over input tokens.
    // For typical decode sequences (< 4K tokens), this is fast enough single-threaded
    // and avoids atomic collisions.
    if (threadIdx.x == 0) {
        for (LongType s = 0; s < seqLen; s++) {
            LongType tokenId = inputIds[b * idsRowStride + s * idsElemStride];
            if (tokenId < 0 || tokenId >= vocabSize) continue;

            // Open addressing with linear probing
            int slot = static_cast<int>(tokenId & PENALTY_HASH_MASK);
            for (int probe = 0; probe < PENALTY_HASH_SIZE; probe++) {
                int idx = (slot + probe) & PENALTY_HASH_MASK;
                if (sKeys[idx] == tokenId) {
                    sCounts[idx]++;
                    break;
                }
                if (sKeys[idx] == -1) {
                    sKeys[idx] = tokenId;
                    sCounts[idx] = 1;
                    break;
                }
            }
        }
    }
    __syncthreads();

    // All threads cooperate to apply penalties from the hash table
    LongType logitsBase = b * logitsRowStride;
    for (int i = threadIdx.x; i < PENALTY_HASH_SIZE; i += blockDim.x) {
        if (sKeys[i] == -1) continue;

        LongType tokenId = sKeys[i];
        int count = sCounts[i];
        LongType offset = logitsBase + tokenId * logitsElemStride;

        float logit = static_cast<float>(logits[offset]);

        // Repetition penalty: multiplicative, direction-aware
        if (repPenalty != 1.0f) {
            logit = logit > 0.0f ? logit / repPenalty : logit * repPenalty;
        }

        // Frequency penalty: proportional to count
        if (freqPenalty != 0.0f) {
            logit -= static_cast<float>(count) * freqPenalty;
        }

        // Presence penalty: flat penalty if token appeared at all
        if (presPenalty != 0.0f) {
            logit -= presPenalty;
        }

        logits[offset] = static_cast<T>(logit);
    }
}

template <typename T>
static void applyPenaltiesLauncher(NDArray* logits, NDArray* inputIds,
                                    double repPenalty, double freqPenalty,
                                    double presPenalty, LaunchContext* context) {
    auto stream = context->getCudaStream();
    auto logitsRank = logits->rankOf();
    auto idsRank = inputIds->rankOf();

    LongType batch = 1;
    LongType vocabSize;
    LongType seqLen;

    if (logitsRank == 1) {
        vocabSize = logits->sizeAt(0);
    } else {
        batch = logits->sizeAt(0);
        vocabSize = logits->sizeAt(1);
    }

    if (idsRank == 1) {
        seqLen = inputIds->sizeAt(0);
    } else {
        seqLen = inputIds->sizeAt(1);
    }

    auto logitsStrides = logits->stridesOf();
    LongType logitsRowStride = (logitsRank == 1) ? 0 : logitsStrides[0];
    LongType logitsElemStride = (logitsRank == 1) ? logitsStrides[0] : logitsStrides[1];

    auto idsStrides = inputIds->stridesOf();
    LongType idsRowStride = (idsRank == 1) ? 0 : idsStrides[0];
    LongType idsElemStride = (idsRank == 1) ? idsStrides[0] : idsStrides[1];

    size_t sharedSize = PENALTY_HASH_SIZE * (sizeof(LongType) + sizeof(int));
    int blockSize = 256;

    applyPenaltiesKernel<T><<<batch, blockSize, sharedSize, *stream>>>(
        logits->specialBuffer(),
        logitsRowStride, logitsElemStride,
        inputIds->specialBuffer(),
        idsRowStride, idsElemStride,
        seqLen, vocabSize,
        static_cast<float>(repPenalty),
        static_cast<float>(freqPenalty),
        static_cast<float>(presPenalty));
    DebugHelper::checkGlobalErrorCode("applyPenaltiesKernel failed");
}

void applyLogitPenaltiesCuda(NDArray* logits, NDArray* inputIds,
                              double repPenalty, double freqPenalty,
                              double presPenalty, LaunchContext* context) {
    if (repPenalty == 1.0 && freqPenalty == 0.0 && presPenalty == 0.0) return;

    NDArray::prepareSpecialUse({logits}, {logits, inputIds});
    BUILD_SINGLE_SELECTOR(logits->dataType(), applyPenaltiesLauncher,
                          (logits, inputIds, repPenalty, freqPenalty, presPenalty, context),
                          SD_FLOAT_TYPES);
    NDArray::registerSpecialUse({logits}, {logits, inputIds});
}


// ────────────────────────────────────────────────────────────────────────────
// Min-P filtering kernel
//
// Two-pass approach:
//   Pass 1: Find max logit and compute log-sum-exp (for softmax normalization)
//   Pass 2: Set logits to -inf where softmax(logit) < minP * max_prob
// ────────────────────────────────────────────────────────────────────────────

template <typename T>
static SD_KERNEL void minPFilterKernel(void* vLogits,
                                        const LongType vocabSize,
                                        const LongType rowStride,
                                        const LongType elemStride,
                                        const float minP) {
    extern __shared__ char sharedMem[];
    float* sdata = reinterpret_cast<float*>(sharedMem);

    auto logits = reinterpret_cast<T*>(vLogits);
    LongType b = blockIdx.x;
    LongType base = b * rowStride;

    // Pass 1a: find max logit (parallel reduction)
    float localMax = -FLT_MAX;
    for (LongType v = threadIdx.x; v < vocabSize; v += blockDim.x) {
        float val = static_cast<float>(logits[base + v * elemStride]);
        if (val > localMax) localMax = val;
    }
    sdata[threadIdx.x] = localMax;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + stride]);
        __syncthreads();
    }
    float rowMax = sdata[0];
    __syncthreads();

    // Pass 1b: compute sum of exp (parallel reduction)
    float localSum = 0.0f;
    for (LongType v = threadIdx.x; v < vocabSize; v += blockDim.x) {
        float val = static_cast<float>(logits[base + v * elemStride]);
        localSum += expf(val - rowMax);
    }
    sdata[threadIdx.x] = localSum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        __syncthreads();
    }
    float sumExp = sdata[0];
    __syncthreads();

    // Pass 1c: find max probability (parallel reduction)
    // maxProb = exp(rowMax - rowMax) / sumExp = 1.0 / sumExp
    // Actually we need the max softmax probability which is the token with highest logit:
    // max_prob = exp(maxLogit - rowMax) / sumExp = 1.0 / sumExp
    // since rowMax IS the max logit. So threshold = minP / sumExp in exp space.
    float threshold = minP / sumExp;

    // Pass 2: mask logits where exp(logit - rowMax) / sumExp < threshold
    // i.e. exp(logit - rowMax) < minP
    for (LongType v = threadIdx.x; v < vocabSize; v += blockDim.x) {
        LongType offset = base + v * elemStride;
        float val = static_cast<float>(logits[offset]);
        float expVal = expf(val - rowMax);
        if (expVal < threshold * sumExp) {
            logits[offset] = static_cast<T>(-FLT_MAX);
        }
    }
}

template <typename T>
static void minPFilterLauncher(NDArray* logits, double minP, LaunchContext* context) {
    auto stream = context->getCudaStream();
    auto rank = logits->rankOf();

    LongType batch = 1;
    LongType vocabSize;

    if (rank == 1) {
        vocabSize = logits->sizeAt(0);
    } else {
        batch = logits->sizeAt(0);
        vocabSize = logits->sizeAt(1);
    }

    auto strides = logits->stridesOf();
    LongType rowStride = (rank == 1) ? 0 : strides[0];
    LongType elemStride = (rank == 1) ? strides[0] : strides[1];

    dim3 launchDims = getLaunchDims("token_sample");
    size_t sharedSize = launchDims.y * sizeof(float);

    minPFilterKernel<T><<<batch, launchDims.y, sharedSize, *stream>>>(
        logits->specialBuffer(),
        vocabSize, rowStride, elemStride,
        static_cast<float>(minP));
    DebugHelper::checkGlobalErrorCode("minPFilterKernel failed");
}

void applyMinPFilterCuda(NDArray* logits, double minP, LaunchContext* context) {
    if (minP <= 0.0) return;

    NDArray::prepareSpecialUse({logits}, {logits});
    BUILD_SINGLE_SELECTOR(logits->dataType(), minPFilterLauncher,
                          (logits, minP, context),
                          SD_FLOAT_TYPES);
    NDArray::registerSpecialUse({logits}, {logits});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
