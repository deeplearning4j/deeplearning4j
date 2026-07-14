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
#include <array/DataTypeUtils.h>
#include <math/templatemath.h>
#include <helpers/DebugHelper.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cfloat>

#include "execution/cuda/LaunchDims.h"

namespace sd {
namespace ops {
namespace helpers {

// Accumulator type: double when T=double for precision, float otherwise.
// Token-count reductions stay int (sdata is reinterpreted as int* — fits).
template <typename T>
struct AccType { using type = float; };
template <>
struct AccType<double> { using type = double; };

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
static SD_KERNEL __launch_bounds__(256, 2) void applyPenaltiesKernel(void* vLogits,
                                            const LongType logitsRowStride,
                                            const LongType logitsElemStride,
                                            const LongType logitsRowOffset,
                                            const void* vInputIds,
                                            const LongType idsRowStride,
                                            const LongType idsElemStride,
                                            const LongType seqLen,
                                            const LongType vocabSize,
                                            const float repPenalty,
                                            const float freqPenalty,
                                            const float presPenalty) {
    using AccT = typename AccType<T>::type;

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

    // Widen scalar penalty params once to AccT
    AccT repP  = static_cast<AccT>(repPenalty);
    AccT freqP = static_cast<AccT>(freqPenalty);
    AccT presP = static_cast<AccT>(presPenalty);

    // All threads cooperate to apply penalties from the hash table.
    // logitsRowOffset offsets into the last sequence position for rank-3 logits.
    LongType logitsBase = b * logitsRowStride + logitsRowOffset;
    for (int i = threadIdx.x; i < PENALTY_HASH_SIZE; i += blockDim.x) {
        if (sKeys[i] == -1) continue;

        LongType tokenId = sKeys[i];
        int count = sCounts[i];
        LongType offset = logitsBase + tokenId * logitsElemStride;

        AccT logit = static_cast<AccT>(logits[offset]);

        // Repetition penalty: multiplicative, direction-aware
        if (repP != static_cast<AccT>(1.0)) {
            logit = logit > static_cast<AccT>(0.0) ? logit / repP : logit * repP;
        }

        // Frequency penalty: proportional to count
        if (freqP != static_cast<AccT>(0.0)) {
            logit -= static_cast<AccT>(count) * freqP;
        }

        // Presence penalty: flat penalty if token appeared at all
        if (presP != static_cast<AccT>(0.0)) {
            logit -= presP;
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
    LongType logitsSeqLen = 1;

    if (logitsRank == 1) {
        vocabSize = logits->sizeAt(0);
    } else if (logitsRank == 2) {
        batch = logits->sizeAt(0);
        vocabSize = logits->sizeAt(1);
    } else {
        // Rank 3: [batch, seqLen, vocabSize] — typical plan output shape
        batch = logits->sizeAt(0);
        logitsSeqLen = logits->sizeAt(1);
        vocabSize = logits->sizeAt(2);
    }

    if (idsRank == 0) {
        seqLen = 1;
    } else if (idsRank == 1) {
        seqLen = inputIds->sizeAt(0);
    } else {
        seqLen = inputIds->sizeAt(1);
    }

    auto logitsStrides = logits->stridesOf();
    LongType logitsRowStride, logitsElemStride, logitsRowOffset;
    if (logitsRank == 1) {
        logitsRowStride = 0;
        logitsElemStride = logitsStrides[0];
        logitsRowOffset = 0;
    } else if (logitsRank == 2) {
        logitsRowStride = logitsStrides[0];
        logitsElemStride = logitsStrides[1];
        logitsRowOffset = 0;
    } else {
        // Rank 3: [batch, seqLen, vocabSize]. Row stride is batch stride,
        // element stride is vocab stride (strides[2]).
        // Offset to last sequence position so penalties apply to the last-position logits.
        logitsRowStride = logitsStrides[0];
        logitsElemStride = logitsStrides[2];
        logitsRowOffset = (logitsSeqLen - 1) * logitsStrides[1];
    }

    auto idsStrides = inputIds->stridesOf();
    LongType idsRowStride = 0;
    LongType idsElemStride = 0;
    if (idsRank == 1) {
        idsElemStride = idsStrides[0];
    } else if (idsRank == 2) {
        idsRowStride = idsStrides[0];
        idsElemStride = idsStrides[1];
    }

    size_t sharedSize = PENALTY_HASH_SIZE * (sizeof(LongType) + sizeof(int));
    int blockSize = 256;

    applyPenaltiesKernel<T><<<batch, blockSize, sharedSize, *stream>>>(
        logits->specialBuffer(),
        logitsRowStride, logitsElemStride, logitsRowOffset,
        inputIds->specialBuffer(),
        idsRowStride, idsElemStride,
        seqLen, vocabSize,
        static_cast<float>(repPenalty),
        static_cast<float>(freqPenalty),
        static_cast<float>(presPenalty));
    DebugHelper::checkGlobalErrorCode("applyPenaltiesKernel failed");
}

void applyLogitPenalties(NDArray* logits, NDArray* inputIds,
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
static SD_KERNEL __launch_bounds__(256, 2) void minPFilterKernel(void* vLogits,
                                        const LongType vocabSize,
                                        const LongType rowStride,
                                        const LongType elemStride,
                                        const float minP) {
    using AccT = typename AccType<T>::type;

    extern __shared__ char sharedMem[];
    AccT* sdata = reinterpret_cast<AccT*>(sharedMem);

    auto logits = reinterpret_cast<T*>(vLogits);
    LongType b = blockIdx.x;
    LongType base = b * rowStride;

    AccT minPAcc = static_cast<AccT>(minP);

    // Pass 1a: find max logit (parallel reduction)
    AccT localMax = -sd::DataTypeUtils::max<AccT>();
    for (LongType v = threadIdx.x; v < vocabSize; v += blockDim.x) {
        AccT val = static_cast<AccT>(logits[base + v * elemStride]);
        if (val > localMax) localMax = val;
    }
    sdata[threadIdx.x] = localMax;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            sdata[threadIdx.x] = sd::math::sd_max<AccT>(sdata[threadIdx.x], sdata[threadIdx.x + stride]);
        __syncthreads();
    }
    AccT rowMax = sdata[0];
    __syncthreads();

    // Pass 1b: compute sum of exp (parallel reduction)
    AccT localSum = static_cast<AccT>(0.0);
    for (LongType v = threadIdx.x; v < vocabSize; v += blockDim.x) {
        AccT val = static_cast<AccT>(logits[base + v * elemStride]);
        localSum += sd::math::sd_exp<AccT, AccT>(val - rowMax);
    }
    sdata[threadIdx.x] = localSum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        __syncthreads();
    }
    AccT sumExp = sdata[0];
    __syncthreads();

    // Pass 1c: find max probability (parallel reduction)
    // maxProb = exp(rowMax - rowMax) / sumExp = 1.0 / sumExp
    // Actually we need the max softmax probability which is the token with highest logit:
    // max_prob = exp(maxLogit - rowMax) / sumExp = 1.0 / sumExp
    // since rowMax IS the max logit. So threshold = minP / sumExp in exp space.
    AccT threshold = minPAcc / sumExp;

    // Pass 2: mask logits where exp(logit - rowMax) / sumExp < threshold
    // i.e. exp(logit - rowMax) < minP
    for (LongType v = threadIdx.x; v < vocabSize; v += blockDim.x) {
        LongType offset = base + v * elemStride;
        AccT val = static_cast<AccT>(logits[offset]);
        AccT expVal = sd::math::sd_exp<AccT, AccT>(val - rowMax);
        if (expVal < threshold * sumExp) {
            logits[offset] = static_cast<T>(-sd::DataTypeUtils::infOrMax<T>());
        }
    }
}

template <typename T>
static void minPFilterLauncher(NDArray* logits, double minP, LaunchContext* context) {
    auto stream = context->getCudaStream();
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
        // Rank 3: [batch, seqLen, vocabSize]
        batch = logits->sizeAt(0);
        logitsSeqLen = logits->sizeAt(1);
        vocabSize = logits->sizeAt(2);
    }

    auto strides = logits->stridesOf();
    LongType rowStride, elemStride;
    if (rank == 1) {
        rowStride = 0;
        elemStride = strides[0];
    } else if (rank == 2) {
        rowStride = strides[0];
        elemStride = strides[1];
    } else {
        // Rank 3: batch stride for row, vocab stride for element.
        // Offset the buffer pointer to the last sequence position so
        // the kernel operates on the correct logits row.
        rowStride = strides[0];
        elemStride = strides[2];
    }

    // For rank 3, offset the buffer to the last sequence position
    void* bufferPtr = logits->specialBuffer();
    if (rank >= 3 && logitsSeqLen > 1) {
        bufferPtr = static_cast<char*>(bufferPtr) +
                    (logitsSeqLen - 1) * strides[1] * logits->sizeOfT();
    }

    dim3 launchDims = getLaunchDims("token_sample");
    size_t sharedSize = launchDims.y * sizeof(typename AccType<T>::type);

    minPFilterKernel<T><<<batch, launchDims.y, sharedSize, *stream>>>(
        bufferPtr,
        vocabSize, rowStride, elemStride,
        static_cast<float>(minP));
    DebugHelper::checkGlobalErrorCode("minPFilterKernel failed");
}

void applyMinPFilter(NDArray* logits, double minP, LaunchContext* context) {
    if (minP <= 0.0) return;

    NDArray::prepareSpecialUse({logits}, {logits});
    BUILD_SINGLE_SELECTOR(logits->dataType(), minPFilterLauncher,
                          (logits, minP, context),
                          SD_FLOAT_TYPES);
    NDArray::registerSpecialUse({logits}, {logits});
}

// ────────────────────────────────────────────────────────────────────────────
// Typical-p (entropy-deviation) filtering kernel
//
// Meister et al. 2023 "Locally Typical Sampling":
//   H = -sum_i p_i log(p_i)
//   deviation_i = |(-log p_i) - H|
// Sort by deviation ascending; accumulate probability until >= typicalP;
// mask remaining tokens.
//
// CUDA implementation: No full sort — use binary search on the deviation
// threshold τ such that sum_{deviation_i <= τ} p_i >= typicalP.
// This mirrors the existing top-p binary-search style and avoids scratch
// allocations, keeping the kernel capture-safe.
//
// Binary search invariant: for threshold τ in [0, logV] (deviation range),
//   kept(τ) = sum_i {p_i : |(-log p_i) - H| <= τ}
// We want the smallest τ such that kept(τ) >= typicalP.
// 48 iterations of bisection on [0, log(vocabSize)+2] gives ~1e-14 resolution.
// ────────────────────────────────────────────────────────────────────────────

template <typename T>
static SD_KERNEL __launch_bounds__(256, 2) void typicalPFilterKernel(void* vLogits,
                                           const LongType vocabSize,
                                           const LongType rowStride,
                                           const LongType elemStride,
                                           const LongType rowOffset,
                                           const float typicalP) {
    using AccT = typename AccType<T>::type;

    extern __shared__ char sharedMem[];
    AccT* sdata = reinterpret_cast<AccT*>(sharedMem);

    auto logits = reinterpret_cast<T*>(vLogits);
    LongType b = blockIdx.x;
    LongType base = b * rowStride + rowOffset;

    AccT typicalPAcc = static_cast<AccT>(typicalP);

    // Phase 1a: row max (numerical stability)
    AccT localMax = -sd::DataTypeUtils::max<AccT>();
    for (LongType v = threadIdx.x; v < vocabSize; v += blockDim.x) {
        AccT val = static_cast<AccT>(logits[base + v * elemStride]);
        if (val > localMax) localMax = val;
    }
    sdata[threadIdx.x] = localMax;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            sdata[threadIdx.x] = sd::math::sd_max<AccT>(sdata[threadIdx.x], sdata[threadIdx.x + stride]);
        __syncthreads();
    }
    AccT rowMax = sdata[0];
    __syncthreads();

    // Phase 1b: sum-exp for softmax denominator
    AccT localSum = static_cast<AccT>(0.0);
    for (LongType v = threadIdx.x; v < vocabSize; v += blockDim.x) {
        localSum += sd::math::sd_exp<AccT, AccT>(static_cast<AccT>(logits[base + v * elemStride]) - rowMax);
    }
    sdata[threadIdx.x] = localSum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        __syncthreads();
    }
    AccT sumExp = sdata[0];
    __syncthreads();

    // Phase 2: compute entropy H = -sum p_i log(p_i)
    AccT localH = static_cast<AccT>(0.0);
    for (LongType v = threadIdx.x; v < vocabSize; v += blockDim.x) {
        AccT w = sd::math::sd_exp<AccT, AccT>(static_cast<AccT>(logits[base + v * elemStride]) - rowMax);
        AccT p = w / sumExp;
        if (p > static_cast<AccT>(0.0)) localH -= p * sd::math::sd_log<AccT, AccT>(p);
    }
    sdata[threadIdx.x] = localH;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        __syncthreads();
    }
    AccT entropy = sdata[0];
    __syncthreads();

    // Phase 3: binary search on deviation threshold τ
    // kept(τ) = sum_i { p_i : |(-log p_i) - H| <= τ }
    // We want the smallest τ such that kept(τ) >= typicalP.
    // Deviation range: [0, log(vocabSize) + 2] (generous upper bound).
    AccT lo = static_cast<AccT>(0.0);
    AccT hi = sd::math::sd_log<AccT, AccT>(static_cast<AccT>(vocabSize)) + static_cast<AccT>(2.0);
    AccT deviationThr = hi;

    for (int iter = 0; iter < 48; iter++) {
        AccT mid = (lo + hi) * static_cast<AccT>(0.5);
        AccT localKept = static_cast<AccT>(0.0);
        for (LongType v = threadIdx.x; v < vocabSize; v += blockDim.x) {
            AccT w = sd::math::sd_exp<AccT, AccT>(static_cast<AccT>(logits[base + v * elemStride]) - rowMax);
            AccT p = w / sumExp;
            AccT negLogP = (p > static_cast<AccT>(0.0)) ? -sd::math::sd_log<AccT, AccT>(p) : sd::DataTypeUtils::max<AccT>();
            AccT dev = sd::math::sd_abs<AccT, AccT>(negLogP - entropy);
            if (dev <= mid) localKept += p;
        }
        sdata[threadIdx.x] = localKept;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride)
                sdata[threadIdx.x] += sdata[threadIdx.x + stride];
            __syncthreads();
        }
        AccT kept = sdata[0];
        __syncthreads();

        if (kept >= typicalPAcc) {
            deviationThr = mid;
            hi = mid;
        } else {
            lo = mid;
        }
    }

    // Phase 4: mask tokens whose deviation > deviationThr (+ tie epsilon).
    // The 1e-6 epsilon keeps whole numeric tie classes together: tokens with
    // identical probability differ in deviation only by float noise, and the CPU
    // path (sort-based, same epsilon) must agree with this kernel on which
    // deviation class survives the cutoff.
    for (LongType v = threadIdx.x; v < vocabSize; v += blockDim.x) {
        AccT w = sd::math::sd_exp<AccT, AccT>(static_cast<AccT>(logits[base + v * elemStride]) - rowMax);
        AccT p = w / sumExp;
        AccT negLogP = (p > static_cast<AccT>(0.0)) ? -sd::math::sd_log<AccT, AccT>(p) : sd::DataTypeUtils::max<AccT>();
        AccT dev = sd::math::sd_abs<AccT, AccT>(negLogP - entropy);
        if (dev > deviationThr + static_cast<AccT>(1e-6)) {
            logits[base + v * elemStride] = static_cast<T>(-sd::DataTypeUtils::infOrMax<T>());
        }
    }
}

template <typename T>
static void typicalPFilterLauncher(NDArray* logits, double typicalP, LaunchContext* context) {
    auto stream = context->getCudaStream();
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

    auto strides = logits->stridesOf();
    LongType rowStride, elemStride, rowOffset;
    if (rank == 1) {
        rowStride = 0;
        elemStride = strides[0];
        rowOffset = 0;
    } else if (rank == 2) {
        rowStride = strides[0];
        elemStride = strides[1];
        rowOffset = 0;
    } else {
        rowStride = strides[0];
        elemStride = strides[2];
        rowOffset = (logitsSeqLen - 1) * strides[1];
    }

    dim3 launchDims = getLaunchDims("token_sample");
    size_t sharedSize = launchDims.y * sizeof(typename AccType<T>::type);

    typicalPFilterKernel<T><<<batch, launchDims.y, sharedSize, *stream>>>(
        logits->specialBuffer(),
        vocabSize, rowStride, elemStride, rowOffset,
        static_cast<float>(typicalP));
    DebugHelper::checkGlobalErrorCode("typicalPFilterKernel failed");
}

void applyTypicalPFilter(NDArray* logits, double typicalP, LaunchContext* context) {
    if (typicalP >= 1.0 || typicalP <= 0.0) return;

    NDArray::prepareSpecialUse({logits}, {logits});
    BUILD_SINGLE_SELECTOR(logits->dataType(), typicalPFilterLauncher,
                          (logits, typicalP, context),
                          SD_FLOAT_TYPES);
    NDArray::registerSpecialUse({logits}, {logits});
}

// ────────────────────────────────────────────────────────────────────────────
// XTC (Exclude Top Choices) filtering kernel
//
// With probability xtcProbability: among tokens with p_i >= xtcThreshold,
// if there are >= 2 such tokens, mask all EXCEPT the lowest-probability one.
//
// CUDA: one block per batch element.
//   - Thread 0 makes the apply/skip decision via curand uniform.
//   - All threads cooperate to find count and argmin-above-threshold.
//   - Thread 0 writes the mask.
// ────────────────────────────────────────────────────────────────────────────

template <typename T>
static SD_KERNEL __launch_bounds__(256, 2) void xtcFilterKernel(void* vLogits,
                                        const LongType vocabSize,
                                        const LongType rowStride,
                                        const LongType elemStride,
                                        const LongType rowOffset,
                                        const float xtcProbability,
                                        const float xtcThreshold,
                                        const LongType seed) {
    using AccT = typename AccType<T>::type;

    extern __shared__ char sharedMem[];
    AccT* sdata = reinterpret_cast<AccT*>(sharedMem);
    int*  sidata = reinterpret_cast<int*>(sharedMem);  // phase-exclusive alias (int count phase)

    auto logits = reinterpret_cast<T*>(vLogits);
    LongType b = blockIdx.x;
    LongType base = b * rowStride + rowOffset;

    AccT xtcThr = static_cast<AccT>(xtcThreshold);

    // Thread 0: curand uniform to decide apply/skip (curand stays float by API design)
    __shared__ bool doApply;
    __shared__ AccT sumExpShared;
    __shared__ AccT rowMaxShared;
    if (threadIdx.x == 0) {
        curandState state;
        curand_init(static_cast<unsigned long long>(seed + b * 1000007ULL), 0ULL, 0ULL, &state);
        float u = curand_uniform(&state);
        doApply = (u <= xtcProbability);  // xtcProbability stays float (RNG gate)
    }
    __syncthreads();
    if (!doApply) return;

    // Phase 1a: row max
    AccT localMax = -sd::DataTypeUtils::max<AccT>();
    for (LongType v = threadIdx.x; v < vocabSize; v += blockDim.x) {
        AccT val = static_cast<AccT>(logits[base + v * elemStride]);
        if (val > localMax) localMax = val;
    }
    sdata[threadIdx.x] = localMax;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            sdata[threadIdx.x] = sd::math::sd_max<AccT>(sdata[threadIdx.x], sdata[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) rowMaxShared = sdata[0];
    __syncthreads();
    AccT rowMax = rowMaxShared;

    // Phase 1b: sum-exp
    AccT localSum = static_cast<AccT>(0.0);
    for (LongType v = threadIdx.x; v < vocabSize; v += blockDim.x) {
        localSum += sd::math::sd_exp<AccT, AccT>(static_cast<AccT>(logits[base + v * elemStride]) - rowMax);
    }
    sdata[threadIdx.x] = localSum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) sumExpShared = sdata[0];
    __syncthreads();
    AccT sumExp = sumExpShared;

    // Phase 2: count above-threshold tokens (int count, sidata aliases sharedMem base)
    int localCount = 0;
    for (LongType v = threadIdx.x; v < vocabSize; v += blockDim.x) {
        AccT w = sd::math::sd_exp<AccT, AccT>(static_cast<AccT>(logits[base + v * elemStride]) - rowMax);
        AccT p = w / sumExp;
        if (p >= xtcThr) localCount++;
    }
    sidata[threadIdx.x] = localCount;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            sidata[threadIdx.x] += sidata[threadIdx.x + stride];
        __syncthreads();
    }
    int totalAbove = sidata[0];
    __syncthreads();

    // Need >= 2 to apply XTC
    if (totalAbove < 2) return;

    // Phase 3: find argmin-probability above threshold
    // Each thread tracks its local (minP, minIdx); then we reduce to global min.
    // Use AccT* sdata for minP values, int* sMinI for indices (offset by blockDim.x AccT slots).
    AccT* sMinP = reinterpret_cast<AccT*>(sharedMem);
    int*  sMinI = reinterpret_cast<int*>(sharedMem + blockDim.x * sizeof(AccT));

    AccT localMinP = sd::DataTypeUtils::max<AccT>();
    int  localMinI = -1;
    for (LongType v = threadIdx.x; v < vocabSize; v += blockDim.x) {
        AccT w = sd::math::sd_exp<AccT, AccT>(static_cast<AccT>(logits[base + v * elemStride]) - rowMax);
        AccT p = w / sumExp;
        if (p >= xtcThr && p < localMinP) {
            localMinP = p;
            localMinI = static_cast<int>(v);
        }
    }
    sMinP[threadIdx.x] = localMinP;
    sMinI[threadIdx.x] = localMinI;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (sMinP[threadIdx.x + stride] < sMinP[threadIdx.x]) {
                sMinP[threadIdx.x] = sMinP[threadIdx.x + stride];
                sMinI[threadIdx.x] = sMinI[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }
    int keepIdx = sMinI[0];
    __syncthreads();

    // Phase 4: mask all above-threshold tokens except keepIdx
    for (LongType v = threadIdx.x; v < vocabSize; v += blockDim.x) {
        if (static_cast<int>(v) == keepIdx) continue;
        AccT w = sd::math::sd_exp<AccT, AccT>(static_cast<AccT>(logits[base + v * elemStride]) - rowMax);
        AccT p = w / sumExp;
        if (p >= xtcThr) {
            logits[base + v * elemStride] = static_cast<T>(-sd::DataTypeUtils::infOrMax<T>());
        }
    }
}

template <typename T>
static void xtcFilterLauncher(NDArray* logits, double xtcProbability, double xtcThreshold,
                               LongType seed, LaunchContext* context) {
    auto stream = context->getCudaStream();
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

    auto strides = logits->stridesOf();
    LongType rowStride, elemStride, rowOffset;
    if (rank == 1) {
        rowStride = 0;
        elemStride = strides[0];
        rowOffset = 0;
    } else if (rank == 2) {
        rowStride = strides[0];
        elemStride = strides[1];
        rowOffset = 0;
    } else {
        rowStride = strides[0];
        elemStride = strides[2];
        rowOffset = (logitsSeqLen - 1) * strides[1];
    }

    dim3 launchDims = getLaunchDims("token_sample");
    // Shared mem: blockDim.x * (sizeof(AccT) + sizeof(int)) for minP/minI reduction
    size_t sharedSize = launchDims.y * (sizeof(typename AccType<T>::type) + sizeof(int));

    xtcFilterKernel<T><<<batch, launchDims.y, sharedSize, *stream>>>(
        logits->specialBuffer(),
        vocabSize, rowStride, elemStride, rowOffset,
        static_cast<float>(xtcProbability),
        static_cast<float>(xtcThreshold),
        seed);
    DebugHelper::checkGlobalErrorCode("xtcFilterKernel failed");
}

void applyXtcFilter(NDArray* logits, double xtcProbability, double xtcThreshold,
                    LongType seed, LaunchContext* context) {
    if (xtcProbability <= 0.0) return;

    NDArray::prepareSpecialUse({logits}, {logits});
    BUILD_SINGLE_SELECTOR(logits->dataType(), xtcFilterLauncher,
                          (logits, xtcProbability, xtcThreshold, seed, context),
                          SD_FLOAT_TYPES);
    NDArray::registerSpecialUse({logits}, {logits});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
