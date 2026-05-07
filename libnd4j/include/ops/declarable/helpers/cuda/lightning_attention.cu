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

#include <ops/declarable/helpers/lightning_attention.h>
#include <array/NDArray.h>
#include <helpers/DebugHelper.h>
#include <cuda_runtime.h>
#include <cfloat>

#include "execution/cuda/LaunchDims.h"

namespace sd {
namespace ops {
namespace helpers {

/**
 * Lightning Attention CUDA kernel.
 *
 * One block per (batch, head) pair. The block processes the sequence
 * sequentially, maintaining the running state S[headDim, headDim] in
 * shared memory. For each position:
 *   1. Accumulate S += K[s]^T * V[s]  (outer product)
 *   2. Compute O[s] = scale * Q[s] @ S  (matrix-vector product)
 *
 * Threads within the block cooperate on the matrix-vector multiply.
 *
 * This kernel handles headDim <= 128 (typical for transformers).
 * For larger headDim, a tiled approach would be needed.
 */
template <typename T>
static SD_KERNEL void lightningAttentionKernel(const void* vQuery,
                                                const void* vKey,
                                                const void* vValue,
                                                void* vOutput,
                                                const LongType seqLen,
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
                                                const float scale) {
    // Each block handles one (batch, head) pair
    LongType batchHead = blockIdx.x;
    LongType heads = gridDim.x / (gridDim.y > 0 ? gridDim.y : 1);

    // Decode batch and head from linear block index
    // Grid is launched as (batch * heads) blocks
    auto query = reinterpret_cast<const T*>(vQuery);
    auto key = reinterpret_cast<const T*>(vKey);
    auto value = reinterpret_cast<const T*>(vValue);
    auto output = reinterpret_cast<T*>(vOutput);

    LongType b = blockIdx.x / (headDim > 0 ? 1 : 1);  // Will be set by grid config
    LongType h = blockIdx.y;
    b = blockIdx.x;

    // State matrix S[headDim][headDim] in shared memory
    extern __shared__ char sharedMem[];
    float* state = reinterpret_cast<float*>(sharedMem);

    // Initialize state to zero
    for (int i = threadIdx.x; i < headDim * headDim; i += blockDim.x) {
        state[i] = 0.0f;
    }
    __syncthreads();

    // Process sequence positions one at a time
    LongType qBase = b * qBatchStride + h * qHeadStride;
    LongType kBase = b * kBatchStride + h * kHeadStride;
    LongType vBase = b * vBatchStride + h * vHeadStride;
    LongType oBase = b * oBatchStride + h * oHeadStride;

    for (LongType s = 0; s < seqLen; s++) {
        // Step 1: Accumulate outer product S += K[s] outer V[s]
        // Each thread handles a subset of the headDim x headDim elements
        for (int idx = threadIdx.x; idx < headDim * headDim; idx += blockDim.x) {
            int i = idx / headDim;
            int j = idx % headDim;
            float kVal = static_cast<float>(key[kBase + s * kSeqStride + i * kDimStride]);
            float vVal = static_cast<float>(value[vBase + s * vSeqStride + j * vDimStride]);
            state[idx] += kVal * vVal;
        }
        __syncthreads();

        // Step 2: Compute O[s,d] = scale * sum_k(Q[s,k] * S[k,d])
        // Each thread computes one output dimension
        for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
            float sum = 0.0f;
            for (int k2 = 0; k2 < headDim; k2++) {
                float qVal = static_cast<float>(query[qBase + s * qSeqStride + k2 * qDimStride]);
                sum += qVal * state[k2 * headDim + d];
            }
            output[oBase + s * oSeqStride + d * oDimStride] = static_cast<T>(sum * scale);
        }
        __syncthreads();
    }
}

template <typename T>
static void lightningAttentionLauncher(LaunchContext* context,
                                        NDArray* query, NDArray* key, NDArray* value,
                                        NDArray* output, double scale) {
    auto stream = context->getCudaStream();

    auto batch = query->sizeAt(0);
    auto heads = query->sizeAt(1);
    auto seqLen = query->sizeAt(2);
    auto headDim = query->sizeAt(3);

    auto qStrides = query->stridesOf();
    auto kStrides = key->stridesOf();
    auto vStrides = value->stridesOf();
    auto oStrides = output->stridesOf();

    // Grid: one block per (batch, head) pair
    dim3 grid(batch, heads);
    int blockSize = 256;
    size_t sharedSize = headDim * headDim * sizeof(float);

    lightningAttentionKernel<T><<<grid, blockSize, sharedSize, *stream>>>(
        query->specialBuffer(),
        key->specialBuffer(),
        value->specialBuffer(),
        output->specialBuffer(),
        seqLen, headDim,
        qStrides[0], qStrides[1], qStrides[2], qStrides[3],
        kStrides[0], kStrides[1], kStrides[2], kStrides[3],
        vStrides[0], vStrides[1], vStrides[2], vStrides[3],
        oStrides[0], oStrides[1], oStrides[2], oStrides[3],
        static_cast<float>(scale));
    DebugHelper::checkGlobalErrorCode("lightningAttentionKernel failed");
}

void lightningAttentionCuda(LaunchContext* context,
                             NDArray* query, NDArray* key, NDArray* value,
                             NDArray* output, double scale) {
    NDArray::prepareSpecialUse({output}, {query, key, value});
    BUILD_SINGLE_SELECTOR(query->dataType(), lightningAttentionLauncher,
                          (context, query, key, value, output, scale),
                          SD_FLOAT_TYPES);
    NDArray::registerSpecialUse({output}, {query, key, value});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
