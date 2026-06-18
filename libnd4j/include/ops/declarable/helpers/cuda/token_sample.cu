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

#include <ops/declarable/helpers/token_sample.h>
#include <ops/declarable/helpers/sampling_penalties.h>
#include <array/NDArray.h>
#include <array/NDArrayFactory.h>
#include <helpers/DebugHelper.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cfloat>

#include "execution/cuda/LaunchDims.h"

namespace sd {
namespace ops {
namespace helpers {

// Kernel: greedy argmax — one block per batch element, threads cooperate via shared mem reduction
template <typename T>
static SD_KERNEL __launch_bounds__(256, 2) void greedyArgmaxKernel(const void* vlogits,
                                          void* voutput,
                                          const LongType vocabSize,
                                          const LongType rowStride,
                                          const LongType elemStride,
                                          const LongType rowOffset) {
    extern __shared__ char sharedMem[];
    auto sMaxVal = reinterpret_cast<float*>(sharedMem);
    auto sMaxIdx = reinterpret_cast<LongType*>(sMaxVal + blockDim.x);

    auto logits = reinterpret_cast<const T*>(vlogits);
    auto output = reinterpret_cast<LongType*>(voutput);

    LongType batchIdx = blockIdx.x;
    LongType baseOffset = batchIdx * rowStride + rowOffset;

    float localMax = -FLT_MAX;
    LongType localIdx = 0;

    for (LongType v = threadIdx.x; v < vocabSize; v += blockDim.x) {
        float val = static_cast<float>(logits[baseOffset + v * elemStride]);
        if (val > localMax) {
            localMax = val;
            localIdx = v;
        }
    }

    sMaxVal[threadIdx.x] = localMax;
    sMaxIdx[threadIdx.x] = localIdx;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (sMaxVal[threadIdx.x + stride] > sMaxVal[threadIdx.x]) {
                sMaxVal[threadIdx.x] = sMaxVal[threadIdx.x + stride];
                sMaxIdx[threadIdx.x] = sMaxIdx[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
        output[batchIdx] = sMaxIdx[0];
}

// Kernel: apply temperature, compute softmax probabilities, write to float probs buffer
// One block per batch element
template <typename T>
static SD_KERNEL __launch_bounds__(256, 2) void tempSoftmaxSampleKernel(const void* vlogits,
                                               void* voutput,
                                               const LongType vocabSize,
                                               const LongType rowStride,
                                               const LongType elemStride,
                                               const LongType rowOffset,
                                               const float invTemp,
                                               const LongType seed) {
    extern __shared__ char sharedMem[];
    float* sdata = reinterpret_cast<float*>(sharedMem);

    auto logits = reinterpret_cast<const T*>(vlogits);
    auto output = reinterpret_cast<LongType*>(voutput);

    LongType b = blockIdx.x;
    LongType baseOffset = b * rowStride + rowOffset;

    // Phase 1: find max (for numerical stability)
    float localMax = -FLT_MAX;
    for (LongType v = threadIdx.x; v < vocabSize; v += blockDim.x) {
        float val = static_cast<float>(logits[baseOffset + v * elemStride]) * invTemp;
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

    // Phase 2: compute sum of exp
    float localSum = 0.0f;
    for (LongType v = threadIdx.x; v < vocabSize; v += blockDim.x) {
        float val = static_cast<float>(logits[baseOffset + v * elemStride]) * invTemp;
        localSum += __expf(val - rowMax);
    }
    sdata[threadIdx.x] = localSum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        __syncthreads();
    }
    float rowSum = sdata[0];

    // Phase 3: thread 0 samples from the distribution using CDF scan
    if (threadIdx.x == 0) {
        curandState state;
        curand_init(static_cast<unsigned long long>(seed + b), 0ULL, 0ULL, &state);
        float u = curand_uniform(&state);
        float target = u * rowSum;

        float cumSum = 0.0f;
        LongType sampled = vocabSize - 1;
        for (LongType v = 0; v < vocabSize; v++) {
            float val = static_cast<float>(logits[baseOffset + v * elemStride]) * invTemp;
            cumSum += __expf(val - rowMax);
            if (cumSum >= target) {
                sampled = v;
                break;
            }
        }
        output[b] = sampled;
    }
}

template <typename T>
static void tokenSampleLauncher(NDArray* logits, NDArray* output,
                                 double temperature, int topK, double topP,
                                 LongType seed, LaunchContext* context) {
    auto stream = context->getCudaStream();
    auto rank = logits->rankOf();

    LongType batch = 1;
    LongType vocabSize;
    LongType seqLen = 1;

    if (rank == 1) {
        vocabSize = logits->sizeAt(0);
    } else if (rank == 2) {
        batch = logits->sizeAt(0);
        vocabSize = logits->sizeAt(1);
    } else {
        batch = logits->sizeAt(0);
        seqLen = logits->sizeAt(1);
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
        rowOffset = (seqLen - 1) * strides[1];
    }

    bool greedy = (temperature <= 0.0 && topK <= 0 && topP <= 0.0);

    dim3 launchDims = getLaunchDims("token_sample");
    size_t sharedSize = launchDims.y * (sizeof(float) + sizeof(LongType));

    if (greedy) {
        greedyArgmaxKernel<T><<<batch, launchDims.y, sharedSize, *stream>>>(
            logits->specialBuffer(), output->specialBuffer(),
            vocabSize, rowStride, elemStride, rowOffset);
        DebugHelper::checkGlobalErrorCode("greedyArgmax failed");
    } else {
        float invTemp = (temperature > 0.0) ? static_cast<float>(1.0 / temperature) : 1.0f;
        LongType actualSeed = (seed > 0) ? seed : static_cast<LongType>(clock());
        size_t smShared = launchDims.y * sizeof(float);

        tempSoftmaxSampleKernel<T><<<batch, launchDims.y, smShared, *stream>>>(
            logits->specialBuffer(), output->specialBuffer(),
            vocabSize, rowStride, elemStride, rowOffset,
            invTemp, actualSeed);
        DebugHelper::checkGlobalErrorCode("tempSoftmaxSample failed");
    }
}

void tokenSample(NDArray* logits, NDArray* output,
                     double temperature, int topK, double topP,
                     LongType seed, LaunchContext* context) {
    NDArray::prepareSpecialUse({output}, {logits});
    BUILD_SINGLE_SELECTOR(logits->dataType(), tokenSampleLauncher,
                          (logits, output, temperature, topK, topP, seed, context),
                          SD_FLOAT_TYPES);
    NDArray::registerSpecialUse({output}, {logits});
}

void tokenSampleWithPenalties(NDArray* logits, NDArray* output,
                                   NDArray* inputIds,
                                   double temperature, int topK,
                                   double topP, double minP,
                                   double repPenalty, double freqPenalty,
                                   double presPenalty,
                                   LongType seed, LaunchContext* context) {
    // Step 1: Apply penalties (in-place on device)
    if (inputIds != nullptr && (repPenalty != 1.0 || freqPenalty != 0.0 || presPenalty != 0.0)) {
        applyLogitPenalties(logits, inputIds, repPenalty, freqPenalty, presPenalty, context);
    }

    // Step 2: Apply min-p filtering (in-place on device)
    if (minP > 0.0) {
        applyMinPFilter(logits, minP, context);
    }

    // Step 3: Standard sampling (temperature, topK, topP)
    tokenSample(logits, output, temperature, topK, topP, seed, context);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
