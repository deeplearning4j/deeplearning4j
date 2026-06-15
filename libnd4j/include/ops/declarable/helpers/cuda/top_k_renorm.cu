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
#include <helpers/DebugHelper.h>
#include <cuda_runtime.h>
#include <cfloat>

#include "execution/cuda/LaunchDims.h"

namespace sd {
namespace ops {
namespace helpers {

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
    extern __shared__ char sharedMem[];
    float* sdata = reinterpret_cast<float*>(sharedMem);

    auto logits = reinterpret_cast<const T*>(vLogits);
    auto output = reinterpret_cast<T*>(vOutput);
    LongType b = blockIdx.x;
    LongType inBase = b * inRowStride;
    LongType outBase = b * outRowStride;

    // Pass 1a: find max logit (parallel reduction)
    float localMax = -FLT_MAX;
    for (LongType v = threadIdx.x; v < vocabSize; v += blockDim.x) {
        float val = static_cast<float>(logits[inBase + v * inElemStride]);
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
        float val = static_cast<float>(logits[inBase + v * inElemStride]);
        localSum += __expf(val - rowMax);
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

    // Pass 1c: write probabilities to output
    for (LongType v = threadIdx.x; v < vocabSize; v += blockDim.x) {
        float val = static_cast<float>(logits[inBase + v * inElemStride]);
        float prob = __expf(val - rowMax) / sumExp;
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
    float localMin2 = FLT_MAX;
    float localMax2 = -FLT_MAX;
    for (LongType v = threadIdx.x; v < vocabSize; v += blockDim.x) {
        float prob = static_cast<float>(output[outBase + v * outElemStride]);
        if (prob < localMin2) localMin2 = prob;
        if (prob > localMax2) localMax2 = prob;
    }
    sdata[threadIdx.x] = localMin2;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            sdata[threadIdx.x] = fminf(sdata[threadIdx.x], sdata[threadIdx.x + stride]);
        __syncthreads();
    }
    float globalMin = sdata[0];
    __syncthreads();

    sdata[threadIdx.x] = localMax2;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + stride]);
        __syncthreads();
    }
    float globalMax = sdata[0];
    __syncthreads();

    // Binary search for threshold: find the largest threshold such that
    // count(prob >= threshold) >= k
    float lo = globalMin;
    float hi = globalMax;
    float threshold = globalMin;

    for (int iter = 0; iter < 32; iter++) {
        float mid = (lo + hi) * 0.5f;

        // Count values >= mid
        int localCount = 0;
        for (LongType v = threadIdx.x; v < vocabSize; v += blockDim.x) {
            float prob = static_cast<float>(output[outBase + v * outElemStride]);
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
    float localRenormSum = 0.0f;
    for (LongType v = threadIdx.x; v < vocabSize; v += blockDim.x) {
        float prob = static_cast<float>(output[outBase + v * outElemStride]);
        if (prob < threshold) {
            output[outBase + v * outElemStride] = static_cast<T>(0.0f);
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
    float renormSum = sdata[0];
    __syncthreads();

    // Pass 4: renormalize
    if (renormSum > 0.0f) {
        for (LongType v = threadIdx.x; v < vocabSize; v += blockDim.x) {
            LongType offset = outBase + v * outElemStride;
            float prob = static_cast<float>(output[offset]);
            if (prob > 0.0f) {
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
    size_t sharedSize = blockSize * sizeof(float);

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
    extern __shared__ char sharedMem[];
    float* sdata = reinterpret_cast<float*>(sharedMem);

    auto logits = reinterpret_cast<const T*>(vLogits);
    auto output = reinterpret_cast<T*>(vOutput);
    LongType b = blockIdx.x;
    LongType inBase = b * inRowStride;
    LongType outBase = b * outRowStride;

    // Pass 1a: find max logit
    float localMax = -FLT_MAX;
    for (LongType v = threadIdx.x; v < vocabSize; v += blockDim.x) {
        float val = static_cast<float>(logits[inBase + v * inElemStride]);
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

    // Pass 1b: compute sum of exp
    float localSum = 0.0f;
    for (LongType v = threadIdx.x; v < vocabSize; v += blockDim.x) {
        float val = static_cast<float>(logits[inBase + v * inElemStride]);
        localSum += __expf(val - rowMax);
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

    // Pass 1c: write probabilities to output
    for (LongType v = threadIdx.x; v < vocabSize; v += blockDim.x) {
        float val = static_cast<float>(logits[inBase + v * inElemStride]);
        float prob = __expf(val - rowMax) / sumExp;
        output[outBase + v * outElemStride] = static_cast<T>(prob);
    }
    __syncthreads();

    // Pass 2: Binary search for the probability threshold
    // We want to find the smallest threshold such that:
    //   sum(prob for prob >= threshold) >= p
    //
    // This is equivalent to: the kept probability mass >= p

    float lo = 0.0f;
    float hi = 1.0f;
    float threshold = 0.0f;

    for (int iter = 0; iter < 32; iter++) {
        float mid = (lo + hi) * 0.5f;

        // Sum probabilities >= mid
        float localKeptSum = 0.0f;
        for (LongType v = threadIdx.x; v < vocabSize; v += blockDim.x) {
            float prob = static_cast<float>(output[outBase + v * outElemStride]);
            if (prob >= mid) localKeptSum += prob;
        }
        sdata[threadIdx.x] = localKeptSum;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride)
                sdata[threadIdx.x] += sdata[threadIdx.x + stride];
            __syncthreads();
        }
        float totalKept = sdata[0];
        __syncthreads();

        if (totalKept >= p) {
            threshold = mid;
            lo = mid;
        } else {
            hi = mid;
        }
    }

    // Pass 3: zero probs below threshold, compute renorm sum
    float localRenormSum = 0.0f;
    for (LongType v = threadIdx.x; v < vocabSize; v += blockDim.x) {
        float prob = static_cast<float>(output[outBase + v * outElemStride]);
        if (prob < threshold) {
            output[outBase + v * outElemStride] = static_cast<T>(0.0f);
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
    float renormSum = sdata[0];
    __syncthreads();

    // Pass 4: renormalize
    if (renormSum > 0.0f) {
        for (LongType v = threadIdx.x; v < vocabSize; v += blockDim.x) {
            LongType offset = outBase + v * outElemStride;
            float prob = static_cast<float>(output[offset]);
            if (prob > 0.0f) {
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
    size_t sharedSize = blockSize * sizeof(float);

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
