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
// Token indices and int counts are unaffected.
template <typename T>
struct AccType { using type = float; };
template <>
struct AccType<double> { using type = double; };

// Kernel: greedy argmax — one block per batch element, threads cooperate via shared mem reduction
template <typename T>
static SD_KERNEL __launch_bounds__(256, 2) void greedyArgmaxKernel(const void* vlogits,
                                          void* voutput,
                                          const LongType vocabSize,
                                          const LongType rowStride,
                                          const LongType elemStride,
                                          const LongType rowOffset) {
    using AccT = typename AccType<T>::type;

    extern __shared__ char sharedMem[];
    auto sMaxVal = reinterpret_cast<AccT*>(sharedMem);
    auto sMaxIdx = reinterpret_cast<LongType*>(sMaxVal + blockDim.x);

    auto logits = reinterpret_cast<const T*>(vlogits);
    auto output = reinterpret_cast<LongType*>(voutput);

    LongType batchIdx = blockIdx.x;
    LongType baseOffset = batchIdx * rowStride + rowOffset;

    AccT localMax = -sd::DataTypeUtils::max<AccT>();
    LongType localIdx = 0;

    for (LongType v = threadIdx.x; v < vocabSize; v += blockDim.x) {
        AccT val = static_cast<AccT>(logits[baseOffset + v * elemStride]);
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

// Kernel: temperature + top-k + top-p (nucleus) filtered multinomial sampling.
// One block per batch element. Mirrors the CPU pipeline in cpu/token_sample.cpp exactly:
//   temp-scale -> top-k (keep k largest) -> softmax -> top-p (nucleus) -> renormalize -> sample.
// Operates in unnormalized softmax-weight space w(v)=exp(tv-rowMax), tv=logit*invTemp (so max w=1);
// the top-k and top-p thresholds are found by block-cooperative binary search, so no scratch
// buffer is required. (top-k / top-p were previously ignored on the CUDA path.)
template <typename T>
static SD_KERNEL __launch_bounds__(256, 2) void tempTopKTopPSampleKernel(const void* vlogits,
                                               void* voutput,
                                               const LongType vocabSize,
                                               const LongType rowStride,
                                               const LongType elemStride,
                                               const LongType rowOffset,
                                               const float invTemp,
                                               const int topK,
                                               const float topP,
                                               const LongType seed) {
    using AccT = typename AccType<T>::type;

    extern __shared__ char sharedMem[];
    AccT* sdata = reinterpret_cast<AccT*>(sharedMem);
    int* sidata = reinterpret_cast<int*>(sharedMem);  // phase-exclusive alias (int count phase)

    auto logits = reinterpret_cast<const T*>(vlogits);
    auto output = reinterpret_cast<LongType*>(voutput);

    LongType b = blockIdx.x;
    LongType baseOffset = b * rowStride + rowOffset;

    // Widen scalar params once
    AccT invTempAcc = static_cast<AccT>(invTemp);
    AccT topPAcc    = static_cast<AccT>(topP);

    // Phase 1: rowMax of temperature-scaled logits (numerical stability)
    AccT localMax = -sd::DataTypeUtils::max<AccT>();
    for (LongType v = threadIdx.x; v < vocabSize; v += blockDim.x) {
        AccT tv = static_cast<AccT>(logits[baseOffset + v * elemStride]) * invTempAcc;
        if (tv > localMax) localMax = tv;
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

    // Phase 2: top-k weight threshold wKthr (keep w >= wKthr) via binary search on count >= k.
    // w(v) = exp(tv - rowMax) lies in (0, 1].
    AccT wKthr = static_cast<AccT>(0.0);
    if (topK > 0 && topK < vocabSize) {
        AccT lo = static_cast<AccT>(0.0), hi = static_cast<AccT>(1.0);
        for (int iter = 0; iter < 32; iter++) {
            AccT mid = (lo + hi) * static_cast<AccT>(0.5);
            int localCount = 0;
            for (LongType v = threadIdx.x; v < vocabSize; v += blockDim.x) {
                AccT w = sd::math::sd_exp<AccT, AccT>(static_cast<AccT>(logits[baseOffset + v * elemStride]) * invTempAcc - rowMax);
                if (w >= mid) localCount++;
            }
            sidata[threadIdx.x] = localCount;
            __syncthreads();
            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (threadIdx.x < stride)
                    sidata[threadIdx.x] += sidata[threadIdx.x + stride];
                __syncthreads();
            }
            int total = sidata[0];
            __syncthreads();
            if (total >= topK) { wKthr = mid; lo = mid; } else { hi = mid; }
        }
    }

    // sumK = total weight kept by top-k (renormalization base for top-p)
    AccT localSumK = static_cast<AccT>(0.0);
    for (LongType v = threadIdx.x; v < vocabSize; v += blockDim.x) {
        AccT w = sd::math::sd_exp<AccT, AccT>(static_cast<AccT>(logits[baseOffset + v * elemStride]) * invTempAcc - rowMax);
        if (w >= wKthr) localSumK += w;
    }
    sdata[threadIdx.x] = localSumK;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        __syncthreads();
    }
    AccT sumK = sdata[0];
    __syncthreads();

    // Phase 3: top-p (nucleus) weight threshold over the top-k-renormalized distribution.
    AccT pThr = static_cast<AccT>(0.0);
    if (topPAcc > static_cast<AccT>(0.0) && topPAcc < static_cast<AccT>(1.0) && sumK > static_cast<AccT>(0.0)) {
        AccT lo = static_cast<AccT>(0.0), hi = static_cast<AccT>(1.0);
        for (int iter = 0; iter < 32; iter++) {
            AccT mid = (lo + hi) * static_cast<AccT>(0.5);
            AccT thr = sd::math::sd_max<AccT>(mid, wKthr);
            AccT localKept = static_cast<AccT>(0.0);
            for (LongType v = threadIdx.x; v < vocabSize; v += blockDim.x) {
                AccT w = sd::math::sd_exp<AccT, AccT>(static_cast<AccT>(logits[baseOffset + v * elemStride]) * invTempAcc - rowMax);
                if (w >= thr) localKept += w;
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
            if (kept >= topPAcc * sumK) { pThr = mid; lo = mid; } else { hi = mid; }
        }
    }

    AccT finalThr = sd::math::sd_max<AccT>(wKthr, pThr);

    // sumFinal = renormalization mass over the kept (top-k ∩ top-p) set
    AccT localFinal = static_cast<AccT>(0.0);
    for (LongType v = threadIdx.x; v < vocabSize; v += blockDim.x) {
        AccT w = sd::math::sd_exp<AccT, AccT>(static_cast<AccT>(logits[baseOffset + v * elemStride]) * invTempAcc - rowMax);
        if (w >= finalThr) localFinal += w;
    }
    sdata[threadIdx.x] = localFinal;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        __syncthreads();
    }
    AccT sumFinal = sdata[0];
    __syncthreads();

    // Phase 4: thread 0 samples from the kept, renormalized distribution via CDF scan.
    if (threadIdx.x == 0) {
        curandState state;
        curand_init(static_cast<unsigned long long>(seed + b), 0ULL, 0ULL, &state);
        // curand_uniform returns float by API design; product widens to AccT
        AccT target = static_cast<AccT>(curand_uniform(&state)) * sumFinal;

        AccT cumSum = static_cast<AccT>(0.0);
        LongType sampled = -1;
        for (LongType v = 0; v < vocabSize; v++) {
            AccT w = sd::math::sd_exp<AccT, AccT>(static_cast<AccT>(logits[baseOffset + v * elemStride]) * invTempAcc - rowMax);
            if (w >= finalThr) {
                cumSum += w;
                if (cumSum >= target) { sampled = v; break; }
            }
        }
        if (sampled < 0) {
            // Numerical guard: fall back to the argmax of the kept set.
            AccT bestW = static_cast<AccT>(-1.0);
            for (LongType v = 0; v < vocabSize; v++) {
                AccT w = sd::math::sd_exp<AccT, AccT>(static_cast<AccT>(logits[baseOffset + v * elemStride]) * invTempAcc - rowMax);
                if (w >= finalThr && w > bestW) { bestW = w; sampled = v; }
            }
            if (sampled < 0) sampled = 0;
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
    size_t sharedSize = launchDims.y * (sizeof(typename AccType<T>::type) + sizeof(LongType));

    if (greedy) {
        greedyArgmaxKernel<T><<<batch, launchDims.y, sharedSize, *stream>>>(
            logits->specialBuffer(), output->specialBuffer(),
            vocabSize, rowStride, elemStride, rowOffset);
        DebugHelper::checkGlobalErrorCode("greedyArgmax failed");
    } else {
        float invTemp = (temperature > 0.0) ? static_cast<float>(1.0 / temperature) : 1.0f;
        LongType actualSeed = (seed > 0) ? seed : static_cast<LongType>(clock());
        size_t smShared = launchDims.y * sizeof(typename AccType<T>::type);

        // Full pipeline: temperature + top-k + top-p (nucleus) filtered multinomial sampling,
        // matching CPU token_sample. (top-k / top-p were previously ignored on the CUDA path.)
        tempTopKTopPSampleKernel<T><<<batch, launchDims.y, smShared, *stream>>>(
            logits->specialBuffer(), output->specialBuffer(),
            vocabSize, rowStride, elemStride, rowOffset,
            invTemp, topK, static_cast<float>(topP), actualSeed);
        DebugHelper::checkGlobalErrorCode("tempTopKTopPSample failed");
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
                                   double typicalP,
                                   double xtcProbability, double xtcThreshold,
                                   LongType seed, LaunchContext* context) {
    // Sampling pipeline order (see token_sample.h for canonical documentation):
    // 1. Penalties (repetition / frequency / presence)
    // 2. Min-p filter
    // 3. Typical-p filter
    // 4. XTC filter
    // 5. Temperature + top-k + top-p + sample (tokenSample_)

    // Step 1: Apply penalties (in-place on device)
    if (inputIds != nullptr && (repPenalty != 1.0 || freqPenalty != 0.0 || presPenalty != 0.0)) {
        applyLogitPenalties(logits, inputIds, repPenalty, freqPenalty, presPenalty, context);
    }

    // Step 2: Apply min-p filtering (in-place on device)
    if (minP > 0.0) {
        applyMinPFilter(logits, minP, context);
    }

    // Step 3: Apply typical-p filtering (in-place on device)
    if (typicalP > 0.0 && typicalP < 1.0) {
        applyTypicalPFilter(logits, typicalP, context);
    }

    // Step 4: Apply XTC filter (in-place on device, stochastic)
    if (xtcProbability > 0.0) {
        applyXtcFilter(logits, xtcProbability, xtcThreshold, seed, context);
    }

    // Step 5: Standard sampling (temperature, topK, topP)
    tokenSample(logits, output, temperature, topK, topP, seed, context);
}

static bool shouldSuppressStopTokens(const TokenSampleConfig& config) {
    return config.minNewTokens > 0
        && config.generatedTokenOffset < config.minNewTokens
        && config.stopTokenIds != nullptr
        && config.stopTokenCount > 0;
}

template <typename T>
static SD_KERNEL void suppressOneStopTokenKernel(void* vlogits,
                                                 const int stopId,
                                                 const LongType batch,
                                                 const LongType vocabSize,
                                                 const LongType rowStride,
                                                 const LongType elemStride,
                                                 const LongType rowOffset) {
    if (stopId < 0 || stopId >= vocabSize) return;
    auto logits = reinterpret_cast<T*>(vlogits);
    LongType b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) return;
    LongType base = b * rowStride + rowOffset;
    // -inf sentinel (infOrMax = +inf for float/double, +max for half/bf16) to match the
    // generation subsystem's -infinity masking convention.
    logits[base + static_cast<LongType>(stopId) * elemStride] = static_cast<T>(-sd::DataTypeUtils::infOrMax<T>());
}

template <typename T>
static void suppressStopTokensLauncher(NDArray* logits, const TokenSampleConfig& config,
                                       LaunchContext* context) {
    if (!shouldSuppressStopTokens(config)) return;

    const int rank = logits->rankOf();
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
    LongType rowStride = 0;
    LongType elemStride;
    LongType rowOffset = 0;
    if (rank == 1) {
        elemStride = strides[0];
    } else if (rank == 2) {
        rowStride = strides[0];
        elemStride = strides[1];
    } else {
        rowStride = strides[0];
        elemStride = strides[2];
        rowOffset = (seqLen - 1) * strides[1];
    }

    auto stream = context->getCudaStream();
    dim3 launchDims = getLaunchDims("token_sample");
    int threads = static_cast<int>(launchDims.y);
    int blocks = static_cast<int>((batch + threads - 1) / threads);
    for (int i = 0; i < config.stopTokenCount; i++) {
        suppressOneStopTokenKernel<T><<<blocks, threads, 0, *stream>>>(
            logits->specialBuffer(), config.stopTokenIds[i], batch, vocabSize,
            rowStride, elemStride, rowOffset);
        DebugHelper::checkGlobalErrorCode("suppressOneStopToken failed");
    }
}

static void suppressStopTokens(NDArray* logits, const TokenSampleConfig& config,
                               LaunchContext* context) {
    if (!shouldSuppressStopTokens(config)) return;
    NDArray::prepareSpecialUse({logits}, {logits});
    BUILD_SINGLE_SELECTOR(logits->dataType(), suppressStopTokensLauncher,
                          (logits, config, context), SD_FLOAT_TYPES);
    NDArray::registerSpecialUse({logits}, {logits});
}

static int resolveScalarStrategy(const TokenSampleConfig& config) {
    if (config.strategy == TOKEN_SAMPLE_AUTO) {
        return (config.temperature <= 0.0 || (config.topK <= 1 && config.topP <= 0.0))
               ? TOKEN_SAMPLE_GREEDY : TOKEN_SAMPLE_SAMPLE;
    }
    return config.strategy;
}

void tokenSamplePolicy(NDArray* logits, NDArray* output,
                       NDArray* inputIds,
                       const TokenSampleConfig& config,
                       TokenSampleResult* result,
                       LaunchContext* context) {
    const int strategy = resolveScalarStrategy(config);
    const bool scalar = config.batchMax == 1 && config.windowMax == 1
                        && config.activeBatch == 1 && config.activeWindow == 1;
    if (!scalar || (strategy != TOKEN_SAMPLE_GREEDY && strategy != TOKEN_SAMPLE_SAMPLE)) {
        THROW_EXCEPTION("tokenSamplePolicy: only scalar GREEDY/SAMPLE is implemented until "
                        "autoregressive_decode exposes the ADR 0106 B/W substrate");
    }

    if (result != nullptr) {
        result->selectedToken = -1;
        result->selectedBatch = 0;
        result->selectedWindow = 0;
        result->acceptedCount = 1;
        result->parentBeam = 0;
        result->score = 0.0;
    }

    suppressStopTokens(logits, config, context);

    if (strategy == TOKEN_SAMPLE_GREEDY) {
        tokenSample(logits, output, 0.0, 0, 0.0, config.seed, context);
    } else {
        const bool hasPenalties = inputIds != nullptr
            && (config.repPenalty != 1.0 || config.freqPenalty != 0.0 || config.presPenalty != 0.0);
        const bool hasExtendedSamplers = config.minP > 0.0
            || (config.typicalP > 0.0 && config.typicalP < 1.0)
            || config.xtcProbability > 0.0;
        if (hasPenalties || hasExtendedSamplers) {
            tokenSampleWithPenalties(logits, output, inputIds,
                                     config.temperature, config.topK, config.topP, config.minP,
                                     config.repPenalty, config.freqPenalty, config.presPenalty,
                                     config.typicalP, config.xtcProbability, config.xtcThreshold,
                                     config.seed, context);
        } else {
            tokenSample(logits, output, config.temperature, config.topK, config.topP,
                        config.seed, context);
        }
    }
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
