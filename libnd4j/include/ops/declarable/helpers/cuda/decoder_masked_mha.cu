/* ******************************************************************************
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
// Fused Decoder Masked Multi-Head Attention CUDA implementation
// One block per (batch, queryHead) pair.
// Shared memory used for Q vector, partial attention scores, and warp reductions.
//

#include <cuda_runtime.h>
#include <math/templatemath.h>
#include <helpers/DebugHelper.h>
#include <array/NDArray.h>
#include <types/float16.h>
#include <ops/declarable/helpers/decoder_masked_mha.h>
#include <ops/declarable/helpers/cuda/device_primitives.cuh>

namespace sd {
namespace ops {
namespace helpers {

constexpr int MHA_WARP_SIZE = 32;

// Accumulator type: use double when T=double for full precision, float otherwise.
template <typename T>
struct AccType { using type = float; };
template <>
struct AccType<double> { using type = double; };

//////////////////////////////////////////////////////////////////////////////
// Warp/block sum+max reductions come from device_primitives.cuh
// (sd::device::warpReduce* / sd::device::blockReduce*).

//////////////////////////////////////////////////////////////////////////////
// Main fused decoder MHA kernel
// Grid: (batch * numHeads)
// Each block computes attention for one (batch, queryHead) pair
//
// Shared memory layout:
//   float qVec[headDim]           - Q vector for this head
//   float reductionBuf[numWarps]  - for block reductions
//////////////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL __launch_bounds__(256, 2) void decoderMaskedMhaKernel(
    const T* __restrict__ hiddenStates,   // [B, 1, H]
    const T* __restrict__ qkvWeight,      // [H, 3*H]
    const T* __restrict__ oWeight,        // [H, H]
    const T* __restrict__ keyCacheIn,     // [B, numKvHeads, maxSeq, headDim]
    const T* __restrict__ valueCacheIn,   // [B, numKvHeads, maxSeq, headDim]
    const T* __restrict__ maskBuf,        // [B, 1, 1, seqLen] or nullptr
    const T* __restrict__ cosBuf,         // [maxSeq, headDim/2] or nullptr
    const T* __restrict__ sinBuf,         // [maxSeq, headDim/2] or nullptr
    T* __restrict__ outputBuf,            // [B, 1, H]
    T* __restrict__ updatedKBuf,          // [B, numKvHeads, maxSeq, headDim]
    T* __restrict__ updatedVBuf,          // [B, numKvHeads, maxSeq, headDim]
    const int numHeads,
    const int numKvHeads,
    const int headDim,
    const int hiddenDim,
    const int cachePosition,
    const int kvCacheMaxSeq,
    const int ropeType,
    const float attScale,
    const LongType batch) {

    using AccT = typename AccType<T>::type;

    const int blockId = blockIdx.x;
    const int batchIdx = blockId / numHeads;
    const int headIdx = blockId % numHeads;

    if (batchIdx >= batch) return;

    // GQA mapping
    const int kvHeadIdx = headIdx * numKvHeads / numHeads;

    extern __shared__ char sharedMem[];
    AccT* qVec = reinterpret_cast<AccT*>(sharedMem);
    AccT* reductionBuf = qVec + headDim;

    const int seqLen = cachePosition + 1;
    const LongType totalQkvDim = 3 * hiddenDim;
    const LongType qSize = numHeads * headDim;

    // Step 1: Compute Q for this head via QKV projection
    // Q[d] = sum_k hidden[k] * qkvWeight[k, headIdx*headDim + d]
    const T* hRow = hiddenStates + batchIdx * hiddenDim;
    for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
        AccT sum = static_cast<AccT>(0);
        LongType colIdx = headIdx * headDim + d;
        for (int k = 0; k < hiddenDim; ++k) {
            sum += static_cast<AccT>(hRow[k]) * static_cast<AccT>(qkvWeight[k * totalQkvDim + colIdx]);
        }
        qVec[d] = sum;
    }
    __syncthreads();

    // Step 2: Compute K and V for this KV head (only one thread block per KV head group needs to do this)
    // We do this for the first query head in the GQA group
    if (headIdx % (numHeads / numKvHeads) == 0) {
        for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
            // K
            AccT kSum = static_cast<AccT>(0);
            LongType kColIdx = qSize + kvHeadIdx * headDim + d;
            for (int k = 0; k < hiddenDim; ++k) {
                kSum += static_cast<AccT>(hRow[k]) * static_cast<AccT>(qkvWeight[k * totalQkvDim + kColIdx]);
            }

            // V
            AccT vSum = static_cast<AccT>(0);
            LongType vColIdx = qSize + numKvHeads * headDim + kvHeadIdx * headDim + d;
            for (int k = 0; k < hiddenDim; ++k) {
                vSum += static_cast<AccT>(hRow[k]) * static_cast<AccT>(qkvWeight[k * totalQkvDim + vColIdx]);
            }

            // Apply RoPE to K if enabled
            if (ropeType > 0 && cosBuf != nullptr && sinBuf != nullptr) {
                int halfDim = headDim / 2;
                if (ropeType == 1 && d < halfDim) {
                    // Standard RoPE: pair (d, d + halfDim)
                    AccT kPartner = static_cast<AccT>(0);
                    LongType kPartnerCol = qSize + kvHeadIdx * headDim + d + halfDim;
                    for (int k2 = 0; k2 < hiddenDim; ++k2) {
                        kPartner += static_cast<AccT>(hRow[k2]) * static_cast<AccT>(qkvWeight[k2 * totalQkvDim + kPartnerCol]);
                    }
                    AccT c = static_cast<AccT>(cosBuf[cachePosition * halfDim + d]);
                    AccT s = static_cast<AccT>(sinBuf[cachePosition * halfDim + d]);
                    AccT kRotated = kSum * c - kPartner * s;
                    AccT kPartnerRotated = kSum * s + kPartner * c;

                    // Write to cache
                    LongType cacheOffset = ((batchIdx * numKvHeads + kvHeadIdx) * kvCacheMaxSeq + cachePosition) * headDim;
                    updatedKBuf[cacheOffset + d] = static_cast<T>(kRotated);
                    updatedKBuf[cacheOffset + d + halfDim] = static_cast<T>(kPartnerRotated);
                } else if (ropeType == 1 && d >= halfDim) {
                    // Already handled by the d < halfDim case
                } else if (ropeType == 2) {
                    // NeoX RoPE: pair (2i, 2i+1) - handle even indices only
                    if (d % 2 == 0 && d / 2 < headDim / 2) {
                        AccT kPartner = static_cast<AccT>(0);
                        LongType kPartnerCol = qSize + kvHeadIdx * headDim + d + 1;
                        for (int k2 = 0; k2 < hiddenDim; ++k2) {
                            kPartner += static_cast<AccT>(hRow[k2]) * static_cast<AccT>(qkvWeight[k2 * totalQkvDim + kPartnerCol]);
                        }
                        int ri = d / 2;
                        AccT c = static_cast<AccT>(cosBuf[cachePosition * (headDim / 2) + ri]);
                        AccT s = static_cast<AccT>(sinBuf[cachePosition * (headDim / 2) + ri]);

                        LongType cacheOffset = ((batchIdx * numKvHeads + kvHeadIdx) * kvCacheMaxSeq + cachePosition) * headDim;
                        updatedKBuf[cacheOffset + d] = static_cast<T>(kSum * c - kPartner * s);
                        updatedKBuf[cacheOffset + d + 1] = static_cast<T>(kSum * s + kPartner * c);
                    }
                }
            } else {
                // No RoPE: write K directly to cache
                LongType cacheOffset = ((batchIdx * numKvHeads + kvHeadIdx) * kvCacheMaxSeq + cachePosition) * headDim;
                updatedKBuf[cacheOffset + d] = static_cast<T>(kSum);
            }

            // Write V to cache (no RoPE on V)
            LongType vCacheOffset = ((batchIdx * numKvHeads + kvHeadIdx) * kvCacheMaxSeq + cachePosition) * headDim;
            updatedVBuf[vCacheOffset + d] = static_cast<T>(vSum);
        }
    }
    __syncthreads();

    // Step 3: Apply RoPE to Q in shared memory
    if (ropeType > 0 && cosBuf != nullptr && sinBuf != nullptr) {
        int halfDim = headDim / 2;
        if (ropeType == 1) {
            for (int i = threadIdx.x; i < halfDim; i += blockDim.x) {
                AccT c = static_cast<AccT>(cosBuf[cachePosition * halfDim + i]);
                AccT s = static_cast<AccT>(sinBuf[cachePosition * halfDim + i]);
                AccT x0 = qVec[i];
                AccT x1 = qVec[i + halfDim];
                qVec[i] = x0 * c - x1 * s;
                qVec[i + halfDim] = x0 * s + x1 * c;
            }
        } else if (ropeType == 2) {
            for (int i = threadIdx.x; i < halfDim; i += blockDim.x) {
                AccT c = static_cast<AccT>(cosBuf[cachePosition * halfDim + i]);
                AccT s = static_cast<AccT>(sinBuf[cachePosition * halfDim + i]);
                AccT x0 = qVec[2 * i];
                AccT x1 = qVec[2 * i + 1];
                qVec[2 * i] = x0 * c - x1 * s;
                qVec[2 * i + 1] = x0 * s + x1 * c;
            }
        }
        __syncthreads();
    }

    // Step 4: Compute attention scores Q @ K^T for all positions
    // First pass: find max score
    AccT localMax = static_cast<AccT>(-1e30);
    for (int pos = threadIdx.x; pos < seqLen; pos += blockDim.x) {
        LongType kOffset = ((batchIdx * numKvHeads + kvHeadIdx) * kvCacheMaxSeq + pos) * headDim;
        AccT dot = static_cast<AccT>(0);
        for (int d = 0; d < headDim; ++d) {
            dot += qVec[d] * static_cast<AccT>(updatedKBuf[kOffset + d]);
        }
        dot *= static_cast<AccT>(attScale);

        if (maskBuf != nullptr) {
            dot += static_cast<AccT>(maskBuf[batchIdx * seqLen + pos]);
        }

        if (dot > localMax) localMax = dot;
    }

    __shared__ AccT globalMax;
    AccT reducedMax = sd::device::blockReduceMax(localMax, reductionBuf);
    if (threadIdx.x == 0) globalMax = reducedMax;
    __syncthreads();

    // Second pass: compute exp(score - max) and sum
    AccT localSum = static_cast<AccT>(0);
    for (int pos = threadIdx.x; pos < seqLen; pos += blockDim.x) {
        LongType kOffset = ((batchIdx * numKvHeads + kvHeadIdx) * kvCacheMaxSeq + pos) * headDim;
        AccT dot = static_cast<AccT>(0);
        for (int d = 0; d < headDim; ++d) {
            dot += qVec[d] * static_cast<AccT>(updatedKBuf[kOffset + d]);
        }
        dot *= static_cast<AccT>(attScale);

        if (maskBuf != nullptr) {
            dot += static_cast<AccT>(maskBuf[batchIdx * seqLen + pos]);
        }

        AccT expVal = sd::math::sd_exp<AccT, AccT>(dot - globalMax);
        localSum += expVal;
    }

    __shared__ AccT globalSum;
    AccT reducedSum = sd::device::blockReduceSum(localSum, reductionBuf);
    if (threadIdx.x == 0) globalSum = reducedSum;
    __syncthreads();

    AccT invSum = static_cast<AccT>(1) / globalSum;

    // Step 5: Compute weighted sum of values
    for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
        AccT acc = static_cast<AccT>(0);
        for (int pos = 0; pos < seqLen; ++pos) {
            LongType kOffset = ((batchIdx * numKvHeads + kvHeadIdx) * kvCacheMaxSeq + pos) * headDim;

            // Recompute attention weight for this position
            AccT dot = static_cast<AccT>(0);
            for (int dd = 0; dd < headDim; ++dd) {
                dot += qVec[dd] * static_cast<AccT>(updatedKBuf[kOffset + dd]);
            }
            dot *= static_cast<AccT>(attScale);
            if (maskBuf != nullptr) {
                dot += static_cast<AccT>(maskBuf[batchIdx * seqLen + pos]);
            }
            AccT weight = sd::math::sd_exp<AccT, AccT>(dot - globalMax) * invSum;

            LongType vOffset = ((batchIdx * numKvHeads + kvHeadIdx) * kvCacheMaxSeq + pos) * headDim;
            acc += weight * static_cast<AccT>(updatedVBuf[vOffset + d]);
        }
        // Store in Q shared memory (reuse since Q is no longer needed)
        qVec[d] = acc;
    }
    __syncthreads();

    // Step 6: Output projection
    for (int j = threadIdx.x; j < hiddenDim; j += blockDim.x) {
        AccT sum = static_cast<AccT>(0);
        for (int d = 0; d < headDim; ++d) {
            LongType wRow = headIdx * headDim + d;
            sum += qVec[d] * static_cast<AccT>(oWeight[wRow * hiddenDim + j]);
        }
        // Atomically add since multiple heads write to the same output
        sd::math::atomics::sd_atomicAdd<T>(&outputBuf[batchIdx * hiddenDim + j], static_cast<T>(sum));
    }
}

//////////////////////////////////////////////////////////////////////////////
// Kernel to zero output buffer before accumulation
//////////////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL __launch_bounds__(256, 2) void zeroOutputKernel(T* output, LongType size) {
    LongType idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = static_cast<T>(0);
    }
}

//////////////////////////////////////////////////////////////////////////////
// Kernel to copy KV cache (existing entries, not the new one)
//////////////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL __launch_bounds__(256, 2) void copyKvCacheKernel(
    const T* __restrict__ src,
    T* __restrict__ dst,
    LongType size) {
    LongType idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] = src[idx];
    }
}

//////////////////////////////////////////////////////////////////////////////
// Launcher
//////////////////////////////////////////////////////////////////////////////
template <typename T>
static void launchDecoderMaskedMhaKernel(
    const T* hiddenStates,
    const T* qkvWeight,
    const T* oWeight,
    const T* keyCacheIn,
    const T* valueCacheIn,
    const T* maskBuf,
    const T* cosBuf,
    const T* sinBuf,
    T* outputBuf,
    T* updatedKBuf,
    T* updatedVBuf,
    int numHeads,
    int numKvHeads,
    int headDim,
    int hiddenDim,
    int cachePosition,
    int kvCacheMaxSeq,
    int ropeType,
    float attScale,
    LongType batch,
    LongType kvCacheSize,
    LongType outputSize,
    cudaStream_t stream) {

    // Zero output buffer for atomic accumulation
    {
        int threads = 256;
        int blocks = (outputSize + threads - 1) / threads;
        zeroOutputKernel<T><<<blocks, threads, 0, stream>>>(outputBuf, outputSize);
    }

    // Copy existing KV cache
    {
        int threads = 256;
        int blocks = (kvCacheSize + threads - 1) / threads;
        copyKvCacheKernel<T><<<blocks, threads, 0, stream>>>(keyCacheIn, updatedKBuf, kvCacheSize);
        copyKvCacheKernel<T><<<blocks, threads, 0, stream>>>(valueCacheIn, updatedVBuf, kvCacheSize);
    }

    // Main kernel: one block per (batch, head)
    int numBlocks = batch * numHeads;
    int threadsPerBlock = 128;
    if (headDim > 128) threadsPerBlock = 256;

    // Shared memory: qVec[headDim] + reductionBuf[numWarps]
    // Use sizeof AccType<T>::type: double when T=double, float otherwise.
    int numWarps = (threadsPerBlock + MHA_WARP_SIZE - 1) / MHA_WARP_SIZE;
    size_t sharedMemSize = (headDim + numWarps) * sizeof(typename AccType<T>::type);

    decoderMaskedMhaKernel<T><<<numBlocks, threadsPerBlock, sharedMemSize, stream>>>(
        hiddenStates, qkvWeight, oWeight,
        keyCacheIn, valueCacheIn,
        maskBuf, cosBuf, sinBuf,
        outputBuf, updatedKBuf, updatedVBuf,
        numHeads, numKvHeads, headDim, hiddenDim,
        cachePosition, kvCacheMaxSeq, ropeType, attScale, batch);

    DebugHelper::checkGlobalErrorCode("decoderMaskedMhaKernel failed");
}

// Explicit instantiations
template void launchDecoderMaskedMhaKernel<float>(
    const float*, const float*, const float*,
    const float*, const float*, const float*,
    const float*, const float*,
    float*, float*, float*,
    int, int, int, int, int, int, int, float,
    LongType, LongType, LongType, cudaStream_t);

template void launchDecoderMaskedMhaKernel<double>(
    const double*, const double*, const double*,
    const double*, const double*, const double*,
    const double*, const double*,
    double*, double*, double*,
    int, int, int, int, int, int, int, float,
    LongType, LongType, LongType, cudaStream_t);

template void launchDecoderMaskedMhaKernel<float16>(
    const float16*, const float16*, const float16*,
    const float16*, const float16*, const float16*,
    const float16*, const float16*,
    float16*, float16*, float16*,
    int, int, int, int, int, int, int, float,
    LongType, LongType, LongType, cudaStream_t);

//////////////////////////////////////////////////////////////////////////////
// Public interface
//////////////////////////////////////////////////////////////////////////////
void decoderMaskedMha(
    NDArray* hiddenStates,
    NDArray* qkvWeight,
    NDArray* oWeight,
    NDArray* keyCache,
    NDArray* valueCache,
    NDArray* mask,
    NDArray* cosCache,
    NDArray* sinCache,
    NDArray* output,
    NDArray* updatedKeyCache,
    NDArray* updatedValueCache,
    const DecoderMhaConfig& config,
    LaunchContext* context) {

    const LongType batch = hiddenStates->sizeAt(0);
    const int hiddenDim = static_cast<int>(hiddenStates->sizeAt(2));
    const int kvCacheMaxSeq = static_cast<int>(keyCache->sizeAt(2));
    const LongType kvCacheSize = keyCache->lengthOf();
    const LongType outputSize = output->lengthOf();

    float attScale = config.attScale;
    if (attScale == 0.0f) {
        attScale = 1.0f / sqrtf(static_cast<float>(config.headDim));
    }

    NDArray::prepareSpecialUse({output, updatedKeyCache, updatedValueCache},
                               {hiddenStates, qkvWeight, oWeight, keyCache, valueCache, mask, cosCache, sinCache});

    auto stream = context->getCudaStream();
    auto dtype = hiddenStates->dataType();

    if (dtype == DataType::FLOAT32) {
        launchDecoderMaskedMhaKernel<float>(
            reinterpret_cast<const float*>(hiddenStates->specialBuffer()),
            reinterpret_cast<const float*>(qkvWeight->specialBuffer()),
            reinterpret_cast<const float*>(oWeight->specialBuffer()),
            reinterpret_cast<const float*>(keyCache->specialBuffer()),
            reinterpret_cast<const float*>(valueCache->specialBuffer()),
            mask != nullptr ? reinterpret_cast<const float*>(mask->specialBuffer()) : nullptr,
            cosCache != nullptr ? reinterpret_cast<const float*>(cosCache->specialBuffer()) : nullptr,
            sinCache != nullptr ? reinterpret_cast<const float*>(sinCache->specialBuffer()) : nullptr,
            reinterpret_cast<float*>(output->specialBuffer()),
            reinterpret_cast<float*>(updatedKeyCache->specialBuffer()),
            reinterpret_cast<float*>(updatedValueCache->specialBuffer()),
            config.numHeads, config.numKvHeads, config.headDim, hiddenDim,
            config.cachePosition, kvCacheMaxSeq, config.ropeType, attScale,
            batch, kvCacheSize, outputSize, *stream);
    } else if (dtype == DataType::DOUBLE) {
        launchDecoderMaskedMhaKernel<double>(
            reinterpret_cast<const double*>(hiddenStates->specialBuffer()),
            reinterpret_cast<const double*>(qkvWeight->specialBuffer()),
            reinterpret_cast<const double*>(oWeight->specialBuffer()),
            reinterpret_cast<const double*>(keyCache->specialBuffer()),
            reinterpret_cast<const double*>(valueCache->specialBuffer()),
            mask != nullptr ? reinterpret_cast<const double*>(mask->specialBuffer()) : nullptr,
            cosCache != nullptr ? reinterpret_cast<const double*>(cosCache->specialBuffer()) : nullptr,
            sinCache != nullptr ? reinterpret_cast<const double*>(sinCache->specialBuffer()) : nullptr,
            reinterpret_cast<double*>(output->specialBuffer()),
            reinterpret_cast<double*>(updatedKeyCache->specialBuffer()),
            reinterpret_cast<double*>(updatedValueCache->specialBuffer()),
            config.numHeads, config.numKvHeads, config.headDim, hiddenDim,
            config.cachePosition, kvCacheMaxSeq, config.ropeType, attScale,
            batch, kvCacheSize, outputSize, *stream);
    } else if (dtype == DataType::HALF) {
        launchDecoderMaskedMhaKernel<float16>(
            reinterpret_cast<const float16*>(hiddenStates->specialBuffer()),
            reinterpret_cast<const float16*>(qkvWeight->specialBuffer()),
            reinterpret_cast<const float16*>(oWeight->specialBuffer()),
            reinterpret_cast<const float16*>(keyCache->specialBuffer()),
            reinterpret_cast<const float16*>(valueCache->specialBuffer()),
            mask != nullptr ? reinterpret_cast<const float16*>(mask->specialBuffer()) : nullptr,
            cosCache != nullptr ? reinterpret_cast<const float16*>(cosCache->specialBuffer()) : nullptr,
            sinCache != nullptr ? reinterpret_cast<const float16*>(sinCache->specialBuffer()) : nullptr,
            reinterpret_cast<float16*>(output->specialBuffer()),
            reinterpret_cast<float16*>(updatedKeyCache->specialBuffer()),
            reinterpret_cast<float16*>(updatedValueCache->specialBuffer()),
            config.numHeads, config.numKvHeads, config.headDim, hiddenDim,
            config.cachePosition, kvCacheMaxSeq, config.ropeType, attScale,
            batch, kvCacheSize, outputSize, *stream);
    } else {
        THROW_EXCEPTION("decoderMaskedMha: Unsupported data type");
    }

    NDArray::registerSpecialUse({output, updatedKeyCache, updatedValueCache},
                                {hiddenStates, qkvWeight, oWeight, keyCache, valueCache, mask, cosCache, sinCache});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
