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

namespace sd {
namespace ops {
namespace helpers {

constexpr int MHA_WARP_SIZE = 32;

//////////////////////////////////////////////////////////////////////////////
// Warp-level reduction: sum
//////////////////////////////////////////////////////////////////////////////
template <typename T>
SD_DEVICE SD_INLINE T mhaWarpReduceSum(T val) {
    for (int offset = MHA_WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

//////////////////////////////////////////////////////////////////////////////
// Warp-level reduction: max
//////////////////////////////////////////////////////////////////////////////
template <typename T>
SD_DEVICE SD_INLINE T mhaWarpReduceMax(T val) {
    for (int offset = MHA_WARP_SIZE / 2; offset > 0; offset /= 2) {
        T other = __shfl_down_sync(0xffffffff, val, offset);
        val = val > other ? val : other;
    }
    return val;
}

//////////////////////////////////////////////////////////////////////////////
// Block-level reduction: sum
//////////////////////////////////////////////////////////////////////////////
SD_DEVICE float mhaBlockReduceSum(float val, float* sharedMem) {
    const int lane = threadIdx.x % MHA_WARP_SIZE;
    const int wid = threadIdx.x / MHA_WARP_SIZE;
    const int numWarps = (blockDim.x + MHA_WARP_SIZE - 1) / MHA_WARP_SIZE;

    val = mhaWarpReduceSum(val);
    if (lane == 0) sharedMem[wid] = val;
    __syncthreads();

    val = (threadIdx.x < numWarps) ? sharedMem[threadIdx.x] : 0.0f;
    if (wid == 0) val = mhaWarpReduceSum(val);
    return val;
}

//////////////////////////////////////////////////////////////////////////////
// Block-level reduction: max
//////////////////////////////////////////////////////////////////////////////
SD_DEVICE float mhaBlockReduceMax(float val, float* sharedMem) {
    const int lane = threadIdx.x % MHA_WARP_SIZE;
    const int wid = threadIdx.x / MHA_WARP_SIZE;
    const int numWarps = (blockDim.x + MHA_WARP_SIZE - 1) / MHA_WARP_SIZE;

    val = mhaWarpReduceMax(val);
    if (lane == 0) sharedMem[wid] = val;
    __syncthreads();

    val = (threadIdx.x < numWarps) ? sharedMem[threadIdx.x] : -1e30f;
    if (wid == 0) val = mhaWarpReduceMax(val);
    return val;
}

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

    const int blockId = blockIdx.x;
    const int batchIdx = blockId / numHeads;
    const int headIdx = blockId % numHeads;

    if (batchIdx >= batch) return;

    // GQA mapping
    const int kvHeadIdx = headIdx * numKvHeads / numHeads;

    extern __shared__ char sharedMem[];
    float* qVec = reinterpret_cast<float*>(sharedMem);
    float* reductionBuf = qVec + headDim;

    const int seqLen = cachePosition + 1;
    const LongType totalQkvDim = 3 * hiddenDim;
    const LongType qSize = numHeads * headDim;

    // Step 1: Compute Q for this head via QKV projection
    // Q[d] = sum_k hidden[k] * qkvWeight[k, headIdx*headDim + d]
    const T* hRow = hiddenStates + batchIdx * hiddenDim;
    for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
        float sum = 0.0f;
        LongType colIdx = headIdx * headDim + d;
        for (int k = 0; k < hiddenDim; ++k) {
            sum += static_cast<float>(hRow[k]) * static_cast<float>(qkvWeight[k * totalQkvDim + colIdx]);
        }
        qVec[d] = sum;
    }
    __syncthreads();

    // Step 2: Compute K and V for this KV head (only one thread block per KV head group needs to do this)
    // We do this for the first query head in the GQA group
    if (headIdx % (numHeads / numKvHeads) == 0) {
        for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
            // K
            float kSum = 0.0f;
            LongType kColIdx = qSize + kvHeadIdx * headDim + d;
            for (int k = 0; k < hiddenDim; ++k) {
                kSum += static_cast<float>(hRow[k]) * static_cast<float>(qkvWeight[k * totalQkvDim + kColIdx]);
            }

            // V
            float vSum = 0.0f;
            LongType vColIdx = qSize + numKvHeads * headDim + kvHeadIdx * headDim + d;
            for (int k = 0; k < hiddenDim; ++k) {
                vSum += static_cast<float>(hRow[k]) * static_cast<float>(qkvWeight[k * totalQkvDim + vColIdx]);
            }

            // Apply RoPE to K if enabled
            if (ropeType > 0 && cosBuf != nullptr && sinBuf != nullptr) {
                int halfDim = headDim / 2;
                if (ropeType == 1 && d < halfDim) {
                    // Standard RoPE: pair (d, d + halfDim)
                    // We need the partner value, so we compute both
                    float kPartner = 0.0f;
                    LongType kPartnerCol = qSize + kvHeadIdx * headDim + d + halfDim;
                    for (int k2 = 0; k2 < hiddenDim; ++k2) {
                        kPartner += static_cast<float>(hRow[k2]) * static_cast<float>(qkvWeight[k2 * totalQkvDim + kPartnerCol]);
                    }
                    float c = static_cast<float>(cosBuf[cachePosition * halfDim + d]);
                    float s = static_cast<float>(sinBuf[cachePosition * halfDim + d]);
                    float kRotated = kSum * c - kPartner * s;
                    float kPartnerRotated = kSum * s + kPartner * c;

                    // Write to cache
                    LongType cacheOffset = ((batchIdx * numKvHeads + kvHeadIdx) * kvCacheMaxSeq + cachePosition) * headDim;
                    updatedKBuf[cacheOffset + d] = static_cast<T>(kRotated);
                    updatedKBuf[cacheOffset + d + halfDim] = static_cast<T>(kPartnerRotated);
                } else if (ropeType == 1 && d >= halfDim) {
                    // Already handled by the d < halfDim case
                } else if (ropeType == 2) {
                    // NeoX RoPE: pair (2i, 2i+1) - handle even indices only
                    if (d % 2 == 0 && d / 2 < headDim / 2) {
                        float kPartner = 0.0f;
                        LongType kPartnerCol = qSize + kvHeadIdx * headDim + d + 1;
                        for (int k2 = 0; k2 < hiddenDim; ++k2) {
                            kPartner += static_cast<float>(hRow[k2]) * static_cast<float>(qkvWeight[k2 * totalQkvDim + kPartnerCol]);
                        }
                        int ri = d / 2;
                        float c = static_cast<float>(cosBuf[cachePosition * (headDim / 2) + ri]);
                        float s = static_cast<float>(sinBuf[cachePosition * (headDim / 2) + ri]);

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
                float c = static_cast<float>(cosBuf[cachePosition * halfDim + i]);
                float s = static_cast<float>(sinBuf[cachePosition * halfDim + i]);
                float x0 = qVec[i];
                float x1 = qVec[i + halfDim];
                qVec[i] = x0 * c - x1 * s;
                qVec[i + halfDim] = x0 * s + x1 * c;
            }
        } else if (ropeType == 2) {
            for (int i = threadIdx.x; i < halfDim; i += blockDim.x) {
                float c = static_cast<float>(cosBuf[cachePosition * halfDim + i]);
                float s = static_cast<float>(sinBuf[cachePosition * halfDim + i]);
                float x0 = qVec[2 * i];
                float x1 = qVec[2 * i + 1];
                qVec[2 * i] = x0 * c - x1 * s;
                qVec[2 * i + 1] = x0 * s + x1 * c;
            }
        }
        __syncthreads();
    }

    // Step 4: Compute attention scores Q @ K^T for all positions
    // Each thread handles a subset of positions
    // First pass: find max score
    float localMax = -1e30f;
    for (int pos = threadIdx.x; pos < seqLen; pos += blockDim.x) {
        LongType kOffset = ((batchIdx * numKvHeads + kvHeadIdx) * kvCacheMaxSeq + pos) * headDim;
        float dot = 0.0f;
        for (int d = 0; d < headDim; ++d) {
            dot += qVec[d] * static_cast<float>(updatedKBuf[kOffset + d]);
        }
        dot *= attScale;

        if (maskBuf != nullptr) {
            dot += static_cast<float>(maskBuf[batchIdx * seqLen + pos]);
        }

        if (dot > localMax) localMax = dot;
    }

    __shared__ float globalMax;
    float reducedMax = mhaBlockReduceMax(localMax, reductionBuf);
    if (threadIdx.x == 0) globalMax = reducedMax;
    __syncthreads();

    // Second pass: compute exp(score - max) and sum
    float localSum = 0.0f;
    for (int pos = threadIdx.x; pos < seqLen; pos += blockDim.x) {
        LongType kOffset = ((batchIdx * numKvHeads + kvHeadIdx) * kvCacheMaxSeq + pos) * headDim;
        float dot = 0.0f;
        for (int d = 0; d < headDim; ++d) {
            dot += qVec[d] * static_cast<float>(updatedKBuf[kOffset + d]);
        }
        dot *= attScale;

        if (maskBuf != nullptr) {
            dot += static_cast<float>(maskBuf[batchIdx * seqLen + pos]);
        }

        float expVal = __expf(dot - globalMax);
        localSum += expVal;
    }

    __shared__ float globalSum;
    float reducedSum = mhaBlockReduceSum(localSum, reductionBuf);
    if (threadIdx.x == 0) globalSum = reducedSum;
    __syncthreads();

    float invSum = 1.0f / globalSum;

    // Step 5: Compute weighted sum of values
    // Each thread accumulates partial results for a subset of headDim
    // We iterate over all positions and accumulate
    for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
        float acc = 0.0f;
        for (int pos = 0; pos < seqLen; ++pos) {
            LongType kOffset = ((batchIdx * numKvHeads + kvHeadIdx) * kvCacheMaxSeq + pos) * headDim;

            // Recompute attention weight for this position
            float dot = 0.0f;
            for (int dd = 0; dd < headDim; ++dd) {
                dot += qVec[dd] * static_cast<float>(updatedKBuf[kOffset + dd]);
            }
            dot *= attScale;
            if (maskBuf != nullptr) {
                dot += static_cast<float>(maskBuf[batchIdx * seqLen + pos]);
            }
            float weight = __expf(dot - globalMax) * invSum;

            LongType vOffset = ((batchIdx * numKvHeads + kvHeadIdx) * kvCacheMaxSeq + pos) * headDim;
            acc += weight * static_cast<float>(updatedVBuf[vOffset + d]);
        }
        // Store in Q shared memory (reuse since Q is no longer needed)
        qVec[d] = acc;
    }
    __syncthreads();

    // Step 6: Output projection
    // output[b, 0, j] += attnHead[d] * oWeight[headIdx*headDim + d, j]
    // Each thread handles a subset of output dimensions
    for (int j = threadIdx.x; j < hiddenDim; j += blockDim.x) {
        float sum = 0.0f;
        for (int d = 0; d < headDim; ++d) {
            LongType wRow = headIdx * headDim + d;
            sum += qVec[d] * static_cast<float>(oWeight[wRow * hiddenDim + j]);
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
    int numWarps = (threadsPerBlock + MHA_WARP_SIZE - 1) / MHA_WARP_SIZE;
    size_t sharedMemSize = (headDim + numWarps) * sizeof(float);

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
void decoderMaskedMhaCuda(
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
        THROW_EXCEPTION("decoderMaskedMhaCuda: Unsupported data type");
    }

    NDArray::registerSpecialUse({output, updatedKeyCache, updatedValueCache},
                                {hiddenStates, qkvWeight, oWeight, keyCache, valueCache, mask, cosCache, sinCache});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
