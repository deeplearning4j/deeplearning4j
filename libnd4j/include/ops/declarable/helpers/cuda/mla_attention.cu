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
// Multi-head Latent Attention (MLA) CUDA implementation
//
// Each CUDA block handles one (batch, head) pair.
// The kernel performs:
//   1. Latent KV decompression via kvDownProj
//   2. Scaled dot-product attention (Q @ K^T * scale, softmax, @ V)
//   3. GQA head mapping
//

#include <cuda_runtime.h>
#include <helpers/DebugHelper.h>
#include <array/NDArray.h>
#include <ops/declarable/helpers/mla_attention.h>
#include <types/float16.h>

namespace sd {
namespace ops {
namespace helpers {

constexpr int MLA_WARP_SIZE = 32;

//////////////////////////////////////////////////////////////////////////////
// Warp-level reduction for sum
//////////////////////////////////////////////////////////////////////////////
template <typename T>
__device__ __forceinline__ T mlaWarpReduceSum(T val) {
    for (int offset = MLA_WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

//////////////////////////////////////////////////////////////////////////////
// Warp-level reduction for max
//////////////////////////////////////////////////////////////////////////////
template <typename T>
__device__ __forceinline__ T mlaWarpReduceMax(T val) {
    for (int offset = MLA_WARP_SIZE / 2; offset > 0; offset /= 2) {
        T other = __shfl_down_sync(0xffffffff, val, offset);
        val = val > other ? val : other;
    }
    return val;
}

//////////////////////////////////////////////////////////////////////////////
// Block-level reduction for sum
//////////////////////////////////////////////////////////////////////////////
template <typename T>
__device__ T mlaBlockReduceSum(T val, T* sharedMem) {
    const int lane = threadIdx.x % MLA_WARP_SIZE;
    const int wid = threadIdx.x / MLA_WARP_SIZE;
    const int numWarps = (blockDim.x + MLA_WARP_SIZE - 1) / MLA_WARP_SIZE;

    val = mlaWarpReduceSum(val);
    if (lane == 0) sharedMem[wid] = val;
    __syncthreads();

    val = (threadIdx.x < numWarps) ? sharedMem[threadIdx.x] : static_cast<T>(0);
    if (wid == 0) val = mlaWarpReduceSum(val);
    return val;
}

//////////////////////////////////////////////////////////////////////////////
// Block-level reduction for max
//////////////////////////////////////////////////////////////////////////////
template <typename T>
__device__ T mlaBlockReduceMax(T val, T* sharedMem) {
    const int lane = threadIdx.x % MLA_WARP_SIZE;
    const int wid = threadIdx.x / MLA_WARP_SIZE;
    const int numWarps = (blockDim.x + MLA_WARP_SIZE - 1) / MLA_WARP_SIZE;

    val = mlaWarpReduceMax(val);
    if (lane == 0) sharedMem[wid] = val;
    __syncthreads();

    val = (threadIdx.x < numWarps) ? sharedMem[threadIdx.x] : -1e30f;
    if (wid == 0) val = mlaWarpReduceMax(val);
    return val;
}

//////////////////////////////////////////////////////////////////////////////
// MLA Attention Kernel
//
// One block per (batch, head) pair.
// Steps:
//   1. For each sequence position, decompress K and V from latent via matmul
//   2. Compute Q @ K^T * scale -> attention scores
//   3. Softmax over sequence positions
//   4. Weighted sum of V -> output
//
// Uses online softmax for numerical stability.
//////////////////////////////////////////////////////////////////////////////
template <typename T>
__global__ __launch_bounds__(512, 1) void mlaAttentionKernel(
    const T* __restrict__ query,          // [B, numHeads, headDim] (squeezed seq=1)
    const T* __restrict__ latentKVCache,  // [B, maxSeqLen, latentDim]
    const T* __restrict__ kvDownProj,     // [latentDim, 2 * numKvHeads * headDim]
    T* __restrict__ output,               // [B, numHeads, headDim] (squeezed seq=1)
    const int batchSize,
    const int numHeads,
    const int numKvHeads,
    const int headDim,
    const int latentDim,
    const int maxSeqLen,
    const int seqLen,
    const float scale) {

    // Block indices
    const int batchIdx = blockIdx.x / numHeads;
    const int headIdx = blockIdx.x % numHeads;
    if (batchIdx >= batchSize) return;

    const int kvHead = headIdx / (numHeads / numKvHeads);  // GQA mapping
    const int kvProjectedDim = numKvHeads * headDim;
    const int projStride = 2 * kvProjectedDim;

    extern __shared__ char sharedBytes[];
    float* sdata = reinterpret_cast<float*>(sharedBytes);

    // Pointer to query for this (batch, head)
    const T* qRow = query + (batchIdx * numHeads + headIdx) * headDim;

    // We use online softmax: maintain running max and sum
    // Each thread accumulates a partial weighted value vector
    // Thread-local accumulator for output (one element at a time via loop)

    // Phase 1: Compute attention scores for each sequence position
    // We process sequence positions in parallel across threads
    float threadMax = -1e30f;
    float threadSumExp = 0.0f;

    // Thread-local output accumulator
    // We need headDim values; each thread maintains partial sums
    // Since headDim can be large, we process one dim at a time later

    // First pass: compute all attention scores
    // Each thread handles a subset of sequence positions
    // We need to store scores for the softmax pass
    // Use a two-pass approach: (1) compute scores + max, (2) softmax + weighted V

    // Shared memory layout: after reduction scratch, store scores
    // sdata[0..numWarps-1] = reduction scratch
    const int numWarps = (blockDim.x + MLA_WARP_SIZE - 1) / MLA_WARP_SIZE;
    float* scores = sdata + numWarps;  // [seqLen] in shared memory

    // Pass 1: Compute Q @ K^T * scale for each sequence position
    for (int s = threadIdx.x; s < seqLen; s += blockDim.x) {
        const T* latentRow = latentKVCache + (batchIdx * maxSeqLen + s) * latentDim;

        float dot = 0.0f;
        for (int d = 0; d < headDim; ++d) {
            // Decompress K[s, d] = latentRow @ kvDownProj[:, kvHead * headDim + d]
            float kVal = 0.0f;
            const int kColIdx = kvHead * headDim + d;
            for (int l = 0; l < latentDim; ++l) {
                kVal += static_cast<float>(latentRow[l]) * static_cast<float>(kvDownProj[l * projStride + kColIdx]);
            }
            dot += static_cast<float>(qRow[d]) * kVal;
        }
        scores[s] = dot * scale;
    }
    __syncthreads();

    // Pass 2: Find max score (for numerical stability)
    float localMax = -1e30f;
    for (int s = threadIdx.x; s < seqLen; s += blockDim.x) {
        if (scores[s] > localMax) localMax = scores[s];
    }
    float globalMax = mlaBlockReduceMax(localMax, sdata);
    __shared__ float sharedMax;
    if (threadIdx.x == 0) sharedMax = globalMax;
    __syncthreads();

    // Pass 3: Compute exp(score - max) and sum
    float localSum = 0.0f;
    for (int s = threadIdx.x; s < seqLen; s += blockDim.x) {
        scores[s] = __expf(scores[s] - sharedMax);
        localSum += scores[s];
    }
    float globalSum = mlaBlockReduceSum(localSum, sdata);
    __shared__ float sharedInvSum;
    if (threadIdx.x == 0) sharedInvSum = 1.0f / globalSum;
    __syncthreads();

    // Normalize scores
    for (int s = threadIdx.x; s < seqLen; s += blockDim.x) {
        scores[s] *= sharedInvSum;
    }
    __syncthreads();

    // Pass 4: Compute weighted sum of V -> output
    // Each thread handles a subset of headDim dimensions
    T* outRow = output + (batchIdx * numHeads + headIdx) * headDim;

    for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
        float val = 0.0f;
        const int vColIdx = kvProjectedDim + kvHead * headDim + d;

        for (int s = 0; s < seqLen; ++s) {
            // Decompress V[s, d] on the fly
            const T* latentRow = latentKVCache + (batchIdx * maxSeqLen + s) * latentDim;
            float vVal = 0.0f;
            for (int l = 0; l < latentDim; ++l) {
                vVal += static_cast<float>(latentRow[l]) * static_cast<float>(kvDownProj[l * projStride + vColIdx]);
            }
            val += scores[s] * vVal;
        }
        outRow[d] = static_cast<T>(val);
    }
}

//////////////////////////////////////////////////////////////////////////////
// Launcher
//////////////////////////////////////////////////////////////////////////////
template <typename T>
void launchMlaAttentionKernel(
    const T* query,
    const T* latentKVCache,
    const T* kvDownProj,
    T* output,
    int batchSize,
    int numHeads,
    int numKvHeads,
    int headDim,
    int latentDim,
    int maxSeqLen,
    int seqLen,
    float scale,
    cudaStream_t stream) {

    int threadsPerBlock = 256;
    if (headDim > 256 || seqLen > 256) threadsPerBlock = 512;

    int numBlocks = batchSize * numHeads;

    // Shared memory: reduction scratch + scores array
    int numWarps = (threadsPerBlock + MLA_WARP_SIZE - 1) / MLA_WARP_SIZE;
    size_t sharedMemSize = (numWarps + seqLen) * sizeof(float);

    mlaAttentionKernel<T><<<numBlocks, threadsPerBlock, sharedMemSize, stream>>>(
        query, latentKVCache, kvDownProj, output,
        batchSize, numHeads, numKvHeads, headDim, latentDim, maxSeqLen, seqLen, scale);

    DebugHelper::checkGlobalErrorCode("mlaAttentionKernel failed");
}

// Explicit instantiations
template void launchMlaAttentionKernel<float>(
    const float*, const float*, const float*, float*,
    int, int, int, int, int, int, int, float, cudaStream_t);

template void launchMlaAttentionKernel<double>(
    const double*, const double*, const double*, double*,
    int, int, int, int, int, int, int, float, cudaStream_t);

template void launchMlaAttentionKernel<float16>(
    const float16*, const float16*, const float16*, float16*,
    int, int, int, int, int, int, int, float, cudaStream_t);

//////////////////////////////////////////////////////////////////////////////
// Public interface
//////////////////////////////////////////////////////////////////////////////
void mlaAttentionCuda(
    NDArray* query,
    NDArray* latentKVCache,
    NDArray* kvDownProj,
    NDArray* ropeCache,
    NDArray* output,
    int seqLen,
    const MLAConfig& config,
    LaunchContext* context) {

    const int batchSize = static_cast<int>(query->sizeAt(0));
    const int maxSeqLen = static_cast<int>(latentKVCache->sizeAt(1));
    const float scale = config.attScale > 0.0f
        ? config.attScale
        : 1.0f / sqrtf(static_cast<float>(config.headDim));

    NDArray::prepareSpecialUse({output}, {query, latentKVCache, kvDownProj, ropeCache});

    auto stream = context->getCudaStream();
    auto dtype = query->dataType();

    if (dtype == DataType::FLOAT32) {
        launchMlaAttentionKernel<float>(
            reinterpret_cast<const float*>(query->specialBuffer()),
            reinterpret_cast<const float*>(latentKVCache->specialBuffer()),
            reinterpret_cast<const float*>(kvDownProj->specialBuffer()),
            reinterpret_cast<float*>(output->specialBuffer()),
            batchSize, config.numHeads, config.numKvHeads, config.headDim,
            config.latentDim, maxSeqLen, seqLen, scale, *stream);
    } else if (dtype == DataType::DOUBLE) {
        launchMlaAttentionKernel<double>(
            reinterpret_cast<const double*>(query->specialBuffer()),
            reinterpret_cast<const double*>(latentKVCache->specialBuffer()),
            reinterpret_cast<const double*>(kvDownProj->specialBuffer()),
            reinterpret_cast<double*>(output->specialBuffer()),
            batchSize, config.numHeads, config.numKvHeads, config.headDim,
            config.latentDim, maxSeqLen, seqLen, scale, *stream);
    } else if (dtype == DataType::HALF) {
        launchMlaAttentionKernel<float16>(
            reinterpret_cast<const float16*>(query->specialBuffer()),
            reinterpret_cast<const float16*>(latentKVCache->specialBuffer()),
            reinterpret_cast<const float16*>(kvDownProj->specialBuffer()),
            reinterpret_cast<float16*>(output->specialBuffer()),
            batchSize, config.numHeads, config.numKvHeads, config.headDim,
            config.latentDim, maxSeqLen, seqLen, scale, *stream);
    } else {
        THROW_EXCEPTION("mlaAttentionCuda: Unsupported data type");
    }

    NDArray::registerSpecialUse({output}, {query, latentKVCache, kvDownProj, ropeCache});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
