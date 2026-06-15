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

#include <ops/declarable/helpers/cascade_attention.h>
#include <array/NDArray.h>
#include <helpers/DebugHelper.h>
#include <cuda_runtime.h>
#include <cfloat>

#include "execution/cuda/LaunchDims.h"

namespace sd {
namespace ops {
namespace helpers {

/**
 * Cascade Attention CUDA kernel.
 *
 * Each block handles one (batch, head, queryPos) triple.
 * The block iterates over KV chunks, computing per-chunk attention
 * and merging results via log-sum-exp.
 *
 * Threads cooperate on the dot product (Q @ K^T) and weighted sum (attn @ V)
 * using shared memory reductions.
 */
template <typename T>
static SD_KERNEL __launch_bounds__(128, 4) void cascadeAttentionKernel(const void* vQuery,
                                              const void* vKey,
                                              const void* vValue,
                                              void* vOutput,
                                              const LongType queryLen,
                                              const LongType kvLen,
                                              const LongType headDim,
                                              const LongType qBatchStride,
                                              const LongType qHeadStride,
                                              const LongType qSeqStride,
                                              const LongType qDimStride,
                                              const LongType kBatchStride,
                                              const LongType kHeadStride,
                                              const LongType kSeqStride,
                                              const LongType kDimStride,
                                              const LongType vBatchStride,
                                              const LongType vHeadStride,
                                              const LongType vSeqStride,
                                              const LongType vDimStride,
                                              const LongType oBatchStride,
                                              const LongType oHeadStride,
                                              const LongType oSeqStride,
                                              const LongType oDimStride,
                                              const int chunkSize,
                                              const float scale) {
    // Grid: (batch * heads, queryLen)
    LongType bh = blockIdx.x;
    LongType heads = gridDim.x / (oBatchStride > 0 ? 1 : 1);  // computed outside
    LongType qPos = blockIdx.y;

    auto query = reinterpret_cast<const T*>(vQuery);
    auto key = reinterpret_cast<const T*>(vKey);
    auto value = reinterpret_cast<const T*>(vValue);
    auto output = reinterpret_cast<T*>(vOutput);

    LongType b = blockIdx.x;
    LongType h = blockIdx.y;
    LongType q = blockIdx.z;

    extern __shared__ char sharedMem[];
    float* sdata = reinterpret_cast<float*>(sharedMem);

    LongType qBase = b * qBatchStride + h * qHeadStride + q * qSeqStride;
    LongType oBase = b * oBatchStride + h * oHeadStride + q * oSeqStride;

    // Per-query accumulators
    // outAccum[headDim] and globalMax, globalSumExp stored in registers/shared
    // We use shared memory for the output accumulator
    float* outAccum = sdata;  // [headDim]
    float* tempBuf = sdata + headDim;  // [blockDim.x] for reductions

    // Initialize output accumulator
    for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
        outAccum[d] = 0.0f;
    }
    __syncthreads();

    float globalMax = -FLT_MAX;
    float globalSumExp = 0.0f;

    // Process KV in chunks
    for (LongType chunkStart = 0; chunkStart < kvLen; chunkStart += chunkSize) {
        LongType chunkEnd = min(chunkStart + (LongType)chunkSize, kvLen);
        LongType chunkLen = chunkEnd - chunkStart;

        // Find chunk max score
        float chunkMax = -FLT_MAX;
        for (LongType k2 = 0; k2 < chunkLen; k2++) {
            // Compute dot product Q @ K[k2]
            float localDot = 0.0f;
            LongType kBase = b * kBatchStride + h * kHeadStride + (chunkStart + k2) * kSeqStride;
            for (LongType d = threadIdx.x; d < headDim; d += blockDim.x) {
                localDot += static_cast<float>(query[qBase + d * qDimStride]) *
                           static_cast<float>(key[kBase + d * kDimStride]);
            }
            // Reduce dot product
            tempBuf[threadIdx.x] = localDot;
            __syncthreads();
            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (threadIdx.x < stride)
                    tempBuf[threadIdx.x] += tempBuf[threadIdx.x + stride];
                __syncthreads();
            }
            float score = tempBuf[0] * scale;
            if (score > chunkMax) chunkMax = score;
            __syncthreads();
        }

        // Compute chunk outputs with softmax
        float chunkSumExp = 0.0f;

        // Temporary chunk output in registers (we accumulate into outAccum after)
        // Process each KV position in the chunk
        for (LongType k2 = 0; k2 < chunkLen; k2++) {
            // Recompute score (we didn't store them to save shared memory)
            float localDot = 0.0f;
            LongType kBase = b * kBatchStride + h * kHeadStride + (chunkStart + k2) * kSeqStride;
            for (LongType d = threadIdx.x; d < headDim; d += blockDim.x) {
                localDot += static_cast<float>(query[qBase + d * qDimStride]) *
                           static_cast<float>(key[kBase + d * kDimStride]);
            }
            tempBuf[threadIdx.x] = localDot;
            __syncthreads();
            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (threadIdx.x < stride)
                    tempBuf[threadIdx.x] += tempBuf[threadIdx.x + stride];
                __syncthreads();
            }
            float w = __expf(tempBuf[0] * scale - chunkMax);
            chunkSumExp += w;
            __syncthreads();

            // Accumulate weighted value into a temporary per-chunk buffer
            // We merge after the chunk loop
            LongType vBase = b * vBatchStride + h * vHeadStride + (chunkStart + k2) * vSeqStride;

            if (chunkStart == 0 && k2 == 0 && globalMax == -FLT_MAX) {
                // First KV position ever — just write
                for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
                    // Will accumulate, but we need to handle the merge at chunk boundaries
                }
            }
        }

        // Simpler approach: recompute and accumulate in one pass
        // Reset for actual computation
        float chunkSumExp2 = 0.0f;

        // First pass: compute all softmax weights for this chunk
        // Second pass: accumulate weighted values
        // For the merge, we need chunkOut[headDim]

        // Use per-thread local accumulators for chunk output
        // Then reduce into shared outAccum

        // Compute chunk output directly
        float* chunkOut = tempBuf + blockDim.x; // [headDim] after tempBuf

        for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
            chunkOut[d] = 0.0f;
        }
        __syncthreads();

        chunkSumExp2 = 0.0f;
        for (LongType k2 = 0; k2 < chunkLen; k2++) {
            float localDot = 0.0f;
            LongType kBase = b * kBatchStride + h * kHeadStride + (chunkStart + k2) * kSeqStride;
            for (LongType d = threadIdx.x; d < headDim; d += blockDim.x) {
                localDot += static_cast<float>(query[qBase + d * qDimStride]) *
                           static_cast<float>(key[kBase + d * kDimStride]);
            }
            tempBuf[threadIdx.x] = localDot;
            __syncthreads();
            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (threadIdx.x < stride)
                    tempBuf[threadIdx.x] += tempBuf[threadIdx.x + stride];
                __syncthreads();
            }
            float w = __expf(tempBuf[0] * scale - chunkMax);
            chunkSumExp2 += w;

            LongType vBase = b * vBatchStride + h * vHeadStride + (chunkStart + k2) * vSeqStride;
            for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
                chunkOut[d] += w * static_cast<float>(value[vBase + d * vDimStride]);
            }
            __syncthreads();
        }

        // Merge chunk result with global accumulator (log-sum-exp)
        if (globalMax == -FLT_MAX) {
            // First chunk
            globalMax = chunkMax;
            globalSumExp = chunkSumExp2;
            for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
                outAccum[d] = chunkOut[d];
            }
        } else {
            float newMax = fmaxf(globalMax, chunkMax);
            float globalRescale = __expf(globalMax - newMax);
            float chunkRescale = __expf(chunkMax - newMax);

            globalSumExp = globalSumExp * globalRescale + chunkSumExp2 * chunkRescale;

            for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
                outAccum[d] = outAccum[d] * globalRescale + chunkOut[d] * chunkRescale;
            }
            globalMax = newMax;
        }
        __syncthreads();
    }

    // Write normalized output
    if (globalSumExp > 0.0f) {
        for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
            output[oBase + d * oDimStride] = static_cast<T>(outAccum[d] / globalSumExp);
        }
    }
}

template <typename T>
static void cascadeAttentionLauncher(LaunchContext* context,
                                      NDArray* query, NDArray* key, NDArray* value,
                                      NDArray* output, int chunkSize, double scale) {
    auto stream = context->getCudaStream();

    auto batch = query->sizeAt(0);
    auto heads = query->sizeAt(1);
    auto queryLen = query->sizeAt(2);
    auto headDim = query->sizeAt(3);
    auto kvLen = key->sizeAt(2);

    auto qStrides = query->stridesOf();
    auto kStrides = key->stridesOf();
    auto vStrides = value->stridesOf();
    auto oStrides = output->stridesOf();

    // Grid: (batch, heads, queryLen)
    dim3 grid(batch, heads, queryLen);
    int blockSize = 128;
    // Shared memory: outAccum[headDim] + tempBuf[blockSize] + chunkOut[headDim]
    size_t sharedSize = (headDim * 2 + blockSize) * sizeof(float);

    cascadeAttentionKernel<T><<<grid, blockSize, sharedSize, *stream>>>(
        query->specialBuffer(),
        key->specialBuffer(),
        value->specialBuffer(),
        output->specialBuffer(),
        queryLen, kvLen, headDim,
        qStrides[0], qStrides[1], qStrides[2], qStrides[3],
        kStrides[0], kStrides[1], kStrides[2], kStrides[3],
        vStrides[0], vStrides[1], vStrides[2], vStrides[3],
        oStrides[0], oStrides[1], oStrides[2], oStrides[3],
        chunkSize,
        static_cast<float>(scale));
    DebugHelper::checkGlobalErrorCode("cascadeAttentionKernel failed");
}

void cascadeAttentionCuda(LaunchContext* context,
                           NDArray* query, NDArray* key, NDArray* value,
                           NDArray* output, int chunkSize, double scale) {
    NDArray::prepareSpecialUse({output}, {query, key, value});
    BUILD_SINGLE_SELECTOR(query->dataType(), cascadeAttentionLauncher,
                          (context, query, key, value, output, chunkSize, scale),
                          SD_FLOAT_TYPES);
    NDArray::registerSpecialUse({output}, {query, key, value});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
