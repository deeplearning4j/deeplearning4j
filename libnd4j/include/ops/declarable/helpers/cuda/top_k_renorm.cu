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

#include <ops/declarable/helpers/top_k_renorm.h>
#include <array/NDArray.h>
#include <array/DataTypeUtils.h>
#include <math/templatemath.h>
#include <helpers/DebugHelper.h>
#include <cuda_runtime.h>
#include <cfloat>

#include "execution/cuda/LaunchDims.h"

namespace sd {
namespace ops {
namespace helpers {

// Accumulator type: double when T=double for precision, float otherwise. The softmax /
// renorm scratch (sdata) and reductions follow this so double logits are not narrowed to
// float. Token-count reductions stay int (sdata is reinterpreted as int*, which fits).
template <typename T>
struct AccType { using type = float; };
template <>
struct AccType<double> { using type = double; };

// ────────────────────────────────────────────────────────────────────────────
// Top-K Renormalization kernel
//
// Three-pass approach per batch element:
//   Pass 1: Compute softmax (find max, compute sum-exp, normalize)
//   Pass 2: Find the K-th largest probability via partial-sort in shared mem
//   Pass 3: Zero probs below threshold, compute renorm sum, normalize
//
// For large vocabs, we use a two-step approach:
//   Step A: Compute softmax probabilities in-place
//   Step B: Use nth_element-like approach to find threshold, then renorm
// ────────────────────────────────────────────────────────────────────────────

/**
 * Kernel: softmax + top-K filtering + renormalization
 *
 * One block per batch element. Threads cooperate via shared memory reductions.
 * For K <= 1024, the K-th largest is found by iterating K times with max-reductions.
 */
template <typename T>
static SD_KERNEL __launch_bounds__(256, 2) void topKRenormKernel(const void* vLogits,
                                        void* vOutput,
                                        const LongType vocabSize,
                                        const LongType inRowStride,
                                        const LongType inElemStride,
                                        const LongType outRowStride,
                                        const LongType outElemStride,
                                        const int k) {
    using AccT = typename AccType<T>::type;

    extern __shared__ char sharedMem[];
    AccT* sdata = reinterpret_cast<AccT*>(sharedMem);

    auto logits = reinterpret_cast<const T*>(vLogits);
    auto output = reinterpret_cast<T*>(vOutput);
    LongType b = blockIdx.x;
    LongType inBase = b * inRowStride;
    LongType outBase = b * outRowStride;

    // Pass 1a: find max logit (parallel reduction)
    AccT localMax = -sd::DataTypeUtils::max<AccT>();
    for (LongType v = threadIdx.x; v < vocabSize; v += blockDim.x) {
        AccT val = static_cast<AccT>(logits[inBase + v * inElemStride]);
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
    AccT localSum = static_cast<AccT>(0);
    for (LongType v = threadIdx.x; v < vocabSize; v += blockDim.x) {
        AccT val = static_cast<AccT>(logits[inBase + v * inElemStride]);
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

    // Pass 1c: write probabilities to output
    for (LongType v = threadIdx.x; v < vocabSize; v += blockDim.x) {
        AccT val = static_cast<AccT>(logits[inBase + v * inElemStride]);
        AccT prob = sd::math::sd_exp<AccT, AccT>(val - rowMax) / sumExp;
        output[outBase + v * outElemStride] = static_cast<T>(prob);
    }
    __syncthreads();

    // Pass 2: find K-th largest probability
    // Use iterative max-exclusion: find max K times, marking each found value
    // For efficiency, we use a threshold approach:
    //   - Count how many values exceed each candidate threshold
    //   - Binary-search style to find the right threshold
    // Simple approach: find threshold via counting
    // We iterate to find the K-th largest value by repeated max-finding

    // Use a simpler threshold-finding approach:
    // The K-th largest = smallest value among the top K
    // We find it by binary search on the probability value space

    // Step 1: Find min and max prob for binary search bounds
    AccT localMin2 = sd::DataTypeUtils::max<AccT>();
    AccT localMax2 = -sd::DataTypeUtils::max<AccT>();
    for (LongType v = threadIdx.x; v < vocabSize; v += blockDim.x) {
        AccT prob = static_cast<AccT>(output[outBase + v * outElemStride]);
        if (prob < localMin2) localMin2 = prob;
        if (prob > localMax2) localMax2 = prob;
    }
    sdata[threadIdx.x] = localMin2;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            sdata[threadIdx.x] = sd::math::sd_min<AccT>(sdata[threadIdx.x], sdata[threadIdx.x + stride]);
        __syncthreads();
    }
    AccT globalMin = sdata[0];
    __syncthreads();

    sdata[threadIdx.x] = localMax2;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            sdata[threadIdx.x] = sd::math::sd_max<AccT>(sdata[threadIdx.x], sdata[threadIdx.x + stride]);
        __syncthreads();
    }
    AccT globalMax = sdata[0];
    __syncthreads();

    // Binary search for threshold: find the largest threshold such that
    // count(prob >= threshold) >= k
    AccT lo = globalMin;
    AccT hi = globalMax;
    AccT threshold = globalMin;

    for (int iter = 0; iter < 32; iter++) {
        AccT mid = (lo + hi) * static_cast<AccT>(0.5);

        // Count values >= mid
        int localCount = 0;
        for (LongType v = threadIdx.x; v < vocabSize; v += blockDim.x) {
            AccT prob = static_cast<AccT>(output[outBase + v * outElemStride]);
            if (prob >= mid) localCount++;
        }

        // Parallel reduction of counts
        // Reuse sdata as int*
        reinterpret_cast<int*>(sdata)[threadIdx.x] = localCount;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride)
                reinterpret_cast<int*>(sdata)[threadIdx.x] += reinterpret_cast<int*>(sdata)[threadIdx.x + stride];
            __syncthreads();
        }
        int totalCount = reinterpret_cast<int*>(sdata)[0];
        __syncthreads();

        if (totalCount >= k) {
            threshold = mid;
            lo = mid;
        } else {
            hi = mid;
        }
    }

    // Pass 3: zero probs below threshold and compute renorm sum
    AccT localRenormSum = static_cast<AccT>(0);
    for (LongType v = threadIdx.x; v < vocabSize; v += blockDim.x) {
        AccT prob = static_cast<AccT>(output[outBase + v * outElemStride]);
        if (prob < threshold) {
            output[outBase + v * outElemStride] = static_cast<T>(0);
        } else {
            localRenormSum += prob;
        }
    }
    sdata[threadIdx.x] = localRenormSum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        __syncthreads();
    }
    AccT renormSum = sdata[0];
    __syncthreads();

    // Pass 4: renormalize
    if (renormSum > static_cast<AccT>(0)) {
        for (LongType v = threadIdx.x; v < vocabSize; v += blockDim.x) {
            LongType offset = outBase + v * outElemStride;
            AccT prob = static_cast<AccT>(output[offset]);
            if (prob > static_cast<AccT>(0)) {
                output[offset] = static_cast<T>(prob / renormSum);
            }
        }
    }
}

template <typename T>
static void topKRenormLauncher(LaunchContext* context, NDArray* logits, NDArray* output, int k) {
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

    auto inStrides = logits->stridesOf();
    LongType inRowStride = (rank == 1) ? 0 : inStrides[0];
    LongType inElemStride = (rank == 1) ? inStrides[0] : inStrides[1];

    auto outStrides = output->stridesOf();
    LongType outRowStride = (rank == 1) ? 0 : outStrides[0];
    LongType outElemStride = (rank == 1) ? outStrides[0] : outStrides[1];

    int blockSize = 256;
    size_t sharedSize = blockSize * sizeof(typename AccType<T>::type);

    topKRenormKernel<T><<<batch, blockSize, sharedSize, *stream>>>(
        logits->specialBuffer(),
        output->specialBuffer(),
        vocabSize,
        inRowStride, inElemStride,
        outRowStride, outElemStride,
        k);
    DebugHelper::checkGlobalErrorCode("topKRenormKernel failed");
}

void topKRenorm(LaunchContext* context, NDArray* logits, NDArray* output, int k) {
    NDArray::prepareSpecialUse({output}, {logits});
    BUILD_SINGLE_SELECTOR(logits->dataType(), topKRenormLauncher,
                          (context, logits, output, k),
                          SD_FLOAT_TYPES);
    NDArray::registerSpecialUse({output}, {logits});
}

// ────────────────────────────────────────────────────────────────────────────
// Top-P (Nucleus) Renormalization kernel
//
// Approach:
//   1. Compute softmax probabilities
//   2. Find cumulative probability threshold via binary search
//   3. Zero probs below threshold, renormalize
// ────────────────────────────────────────────────────────────────────────────

template <typename T>
static SD_KERNEL __launch_bounds__(256, 2) void topPRenormKernel(const void* vLogits,
                                        void* vOutput,
                                        const LongType vocabSize,
                                        const LongType inRowStride,
                                        const LongType inElemStride,
                                        const LongType outRowStride,
                                        const LongType outElemStride,
                                        const float p) {
    using AccT = typename AccType<T>::type;

    extern __shared__ char sharedMem[];
    AccT* sdata = reinterpret_cast<AccT*>(sharedMem);

    auto logits = reinterpret_cast<const T*>(vLogits);
    auto output = reinterpret_cast<T*>(vOutput);
    LongType b = blockIdx.x;
    LongType inBase = b * inRowStride;
    LongType outBase = b * outRowStride;

    // Pass 1a: find max logit
    AccT localMax = -sd::DataTypeUtils::max<AccT>();
    for (LongType v = threadIdx.x; v < vocabSize; v += blockDim.x) {
        AccT val = static_cast<AccT>(logits[inBase + v * inElemStride]);
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

    // Pass 1b: compute sum of exp
    AccT localSum = static_cast<AccT>(0);
    for (LongType v = threadIdx.x; v < vocabSize; v += blockDim.x) {
        AccT val = static_cast<AccT>(logits[inBase + v * inElemStride]);
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

    // Pass 1c: write probabilities to output
    for (LongType v = threadIdx.x; v < vocabSize; v += blockDim.x) {
        AccT val = static_cast<AccT>(logits[inBase + v * inElemStride]);
        AccT prob = sd::math::sd_exp<AccT, AccT>(val - rowMax) / sumExp;
        output[outBase + v * outElemStride] = static_cast<T>(prob);
    }
    __syncthreads();

    // Pass 2: Binary search for the probability threshold
    // We want to find the smallest threshold such that:
    //   sum(prob for prob >= threshold) >= p
    //
    // This is equivalent to: the kept probability mass >= p

    AccT lo = static_cast<AccT>(0);
    AccT hi = static_cast<AccT>(1);
    AccT threshold = static_cast<AccT>(0);

    for (int iter = 0; iter < 32; iter++) {
        AccT mid = (lo + hi) * static_cast<AccT>(0.5);

        // Sum probabilities >= mid
        AccT localKeptSum = static_cast<AccT>(0);
        for (LongType v = threadIdx.x; v < vocabSize; v += blockDim.x) {
            AccT prob = static_cast<AccT>(output[outBase + v * outElemStride]);
            if (prob >= mid) localKeptSum += prob;
        }
        sdata[threadIdx.x] = localKeptSum;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride)
                sdata[threadIdx.x] += sdata[threadIdx.x + stride];
            __syncthreads();
        }
        AccT totalKept = sdata[0];
        __syncthreads();

        if (totalKept >= static_cast<AccT>(p)) {
            threshold = mid;
            lo = mid;
        } else {
            hi = mid;
        }
    }

    // Pass 3: zero probs below threshold, compute renorm sum
    AccT localRenormSum = static_cast<AccT>(0);
    for (LongType v = threadIdx.x; v < vocabSize; v += blockDim.x) {
        AccT prob = static_cast<AccT>(output[outBase + v * outElemStride]);
        if (prob < threshold) {
            output[outBase + v * outElemStride] = static_cast<T>(0);
        } else {
            localRenormSum += prob;
        }
    }
    sdata[threadIdx.x] = localRenormSum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        __syncthreads();
    }
    AccT renormSum = sdata[0];
    __syncthreads();

    // Pass 4: renormalize
    if (renormSum > static_cast<AccT>(0)) {
        for (LongType v = threadIdx.x; v < vocabSize; v += blockDim.x) {
            LongType offset = outBase + v * outElemStride;
            AccT prob = static_cast<AccT>(output[offset]);
            if (prob > static_cast<AccT>(0)) {
                output[offset] = static_cast<T>(prob / renormSum);
            }
        }
    }
}

template <typename T>
static void topPRenormLauncher(LaunchContext* context, NDArray* logits, NDArray* output, double p) {
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

    auto inStrides = logits->stridesOf();
    LongType inRowStride = (rank == 1) ? 0 : inStrides[0];
    LongType inElemStride = (rank == 1) ? inStrides[0] : inStrides[1];

    auto outStrides = output->stridesOf();
    LongType outRowStride = (rank == 1) ? 0 : outStrides[0];
    LongType outElemStride = (rank == 1) ? outStrides[0] : outStrides[1];

    int blockSize = 256;
    size_t sharedSize = blockSize * sizeof(typename AccType<T>::type);

    topPRenormKernel<T><<<batch, blockSize, sharedSize, *stream>>>(
        logits->specialBuffer(),
        output->specialBuffer(),
        vocabSize,
        inRowStride, inElemStride,
        outRowStride, outElemStride,
        static_cast<float>(p));
    DebugHelper::checkGlobalErrorCode("topPRenormKernel failed");
}

void topPRenorm(LaunchContext* context, NDArray* logits, NDArray* output, double p) {
    NDArray::prepareSpecialUse({output}, {logits});
    BUILD_SINGLE_SELECTOR(logits->dataType(), topPRenormLauncher,
                          (context, logits, output, p),
                          SD_FLOAT_TYPES);
    NDArray::registerSpecialUse({output}, {logits});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
