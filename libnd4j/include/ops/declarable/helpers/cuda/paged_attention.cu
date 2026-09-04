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
// @author Adam Gibson
//
// CUDA kernels for paged attention with block-based KV cache.
//

#include <ops/declarable/helpers/paged_attention.h>
#include <helpers/PointersManager.h>
#include <helpers/DebugHelper.h>
#include <execution/cuda/LaunchDims.h>
#include <array/NDArrayFactory.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>

namespace sd {
namespace ops {
namespace helpers {

// ──────────────────────────────────────────────────────────────────────────────
// Kernel: Paged attention decode (single new token per sequence)
//
// Each thread block handles one (batch, head) pair.
// Iterates over blocks in the page table, loads K/V from the block pool,
// computes attention scores, and produces the output.
//
// This is optimized for the decode case: query seqLen = 1.
// For prefill (seqLen > 1), use standard FlashAttention with dense KV.
// ──────────────────────────────────────────────────────────────────────────────

template <typename T>
static SD_KERNEL __launch_bounds__(256, 2) void pagedAttentionDecodeKernel(
    const T* __restrict__ query,          // [batch, 1, numHeads, headDim]
    const T* __restrict__ keyPool,        // [numBlocks, blockSize, numKvHeads, headDim]
    const T* __restrict__ valuePool,      // [numBlocks, blockSize, numKvHeads, headDim]
    const int* __restrict__ pageTables,   // [batch, maxBlocksPerSeq]
    const int* __restrict__ contextLens,  // [batch]
    T* __restrict__ output,               // [batch, 1, numHeads, headDim]
    int numHeads,
    int numKvHeads,
    int headDim,
    int blockSize,
    int maxBlocksPerSeq,
    float scale) {

    // Block assignment: one CUDA block per (batch, queryHead)
    int batchIdx = blockIdx.x;
    int headIdx = blockIdx.y;
    int tid = threadIdx.x;

    // GQA: map query head to KV head
    int kvHeadIdx = (numKvHeads > 0 && numKvHeads < numHeads)
        ? (headIdx * numKvHeads / numHeads)
        : headIdx;

    int ctxLen = contextLens[batchIdx];
    if (ctxLen <= 0) return;

    // Load query vector into shared memory
    extern __shared__ char smem[];
    T* qVec = reinterpret_cast<T*>(smem);                        // [headDim]
    // scores region starts at the first 4-byte-aligned byte after qVec —
    // must match the host-side smemBytes computation in pagedAttentionForward.
    float* scores = reinterpret_cast<float*>(smem + ((headDim * sizeof(T) + 3) & ~3));

    // Cooperatively load query: query[batchIdx, 0, headIdx, :]
    int qOffset = ((batchIdx * 1 * numHeads) + headIdx) * headDim;
    for (int d = tid; d < headDim; d += blockDim.x) {
        qVec[d] = query[qOffset + d];
    }
    __syncthreads();

    // Phase 1: Compute attention scores for all context positions
    // Iterate through page table blocks
    int numCtxBlocks = (ctxLen + blockSize - 1) / blockSize;

    for (int pos = tid; pos < ctxLen; pos += blockDim.x) {
        int logicalBlock = pos / blockSize;
        int offsetInBlock = pos % blockSize;

        // Look up physical block from page table
        int physicalBlock = pageTables[batchIdx * maxBlocksPerSeq + logicalBlock];
        if (physicalBlock < 0) {
            scores[pos] = -FLT_MAX;
            continue;
        }

        // Key address: keyPool[physicalBlock, offsetInBlock, kvHeadIdx, :]
        int keyOffset = ((physicalBlock * blockSize + offsetInBlock) * numKvHeads + kvHeadIdx) * headDim;

        // Dot product: q · k
        float score = 0.0f;
        for (int d = 0; d < headDim; d++) {
            score += static_cast<float>(qVec[d]) * static_cast<float>(keyPool[keyOffset + d]);
        }
        scores[pos] = score * scale;
    }
    __syncthreads();

    // Phase 2: Softmax over scores
    // Find max (reduction)
    __shared__ float maxScore;
    __shared__ float sumExp;

    if (tid == 0) {
        maxScore = -FLT_MAX;
        for (int i = 0; i < ctxLen; i++) {
            if (scores[i] > maxScore) maxScore = scores[i];
        }
    }
    __syncthreads();

    // Compute exp and sum
    if (tid == 0) {
        sumExp = 0.0f;
        for (int i = 0; i < ctxLen; i++) {
            scores[i] = __expf(scores[i] - maxScore);
            sumExp += scores[i];
        }
        // Normalize
        float invSum = 1.0f / (sumExp + 1e-8f);
        for (int i = 0; i < ctxLen; i++) {
            scores[i] *= invSum;
        }
    }
    __syncthreads();

    // Phase 3: Weighted sum of values
    // output[batchIdx, 0, headIdx, d] = sum_pos( scores[pos] * V[pos, kvHeadIdx, d] )
    int outOffset = ((batchIdx * 1 * numHeads) + headIdx) * headDim;

    for (int d = tid; d < headDim; d += blockDim.x) {
        float acc = 0.0f;
        for (int pos = 0; pos < ctxLen; pos++) {
            int logicalBlock = pos / blockSize;
            int offsetInBlock = pos % blockSize;
            int physicalBlock = pageTables[batchIdx * maxBlocksPerSeq + logicalBlock];
            if (physicalBlock < 0) continue;

            int valOffset = ((physicalBlock * blockSize + offsetInBlock) * numKvHeads + kvHeadIdx) * headDim + d;
            acc += scores[pos] * static_cast<float>(valuePool[valOffset]);
        }
        output[outOffset + d] = static_cast<T>(acc);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Kernel: Append new KV entries to block pool via page table
// ──────────────────────────────────────────────────────────────────────────────

template <typename T>
static SD_KERNEL __launch_bounds__(256, 2) void pagedKvAppendKernel(
    T* __restrict__ keyPool,              // [numBlocks, blockSize, numKvHeads, headDim]
    T* __restrict__ valuePool,            // [numBlocks, blockSize, numKvHeads, headDim]
    const T* __restrict__ newKeys,        // [batch, newLen, numKvHeads, headDim]
    const T* __restrict__ newValues,      // [batch, newLen, numKvHeads, headDim]
    const int* __restrict__ pageTables,   // [batch, maxBlocksPerSeq]
    const int* __restrict__ contextLens,  // [batch] - lengths BEFORE append
    int numKvHeads,
    int headDim,
    int blockSize,
    int maxBlocksPerSeq,
    int newLen,
    int batch) {

    // Each thread handles one element: (batchIdx, tokenIdx, headIdx, dimIdx)
    int totalElements = batch * newLen * numKvHeads * headDim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalElements) return;

    // Decompose flat index
    int dimIdx = idx % headDim;
    int remainder = idx / headDim;
    int headIdx = remainder % numKvHeads;
    remainder = remainder / numKvHeads;
    int tokenIdx = remainder % newLen;
    int batchIdx = remainder / newLen;

    // Where in the sequence does this token go?
    int pos = contextLens[batchIdx] + tokenIdx;
    int logicalBlock = pos / blockSize;
    int offsetInBlock = pos % blockSize;

    if (logicalBlock >= maxBlocksPerSeq) return;

    int physicalBlock = pageTables[batchIdx * maxBlocksPerSeq + logicalBlock];
    if (physicalBlock < 0) return;

    // Source index in newKeys/newValues: [batchIdx, tokenIdx, headIdx, dimIdx]
    int srcIdx = ((batchIdx * newLen + tokenIdx) * numKvHeads + headIdx) * headDim + dimIdx;

    // Dest index in pool: [physicalBlock, offsetInBlock, headIdx, dimIdx]
    int dstIdx = ((physicalBlock * blockSize + offsetInBlock) * numKvHeads + headIdx) * headDim + dimIdx;

    keyPool[dstIdx] = newKeys[srcIdx];
    valuePool[dstIdx] = newValues[srcIdx];
}

// ──────────────────────────────────────────────────────────────────────────────
// Host-side dispatch
// ──────────────────────────────────────────────────────────────────────────────

void pagedAttentionForward(
    NDArray* query,
    NDArray* keyBlockPool,
    NDArray* valueBlockPool,
    NDArray* pageTables,
    NDArray* contextLens,
    NDArray* output,
    const PagedAttentionConfig& config,
    LaunchContext* context) {

    auto stream = context != nullptr ? context->getCudaStream() : nullptr;

    int batch = static_cast<int>(query->sizeAt(0));
    int numHeads = config.numHeads > 0 ? config.numHeads : static_cast<int>(query->sizeAt(2));
    int headDim = config.headDim > 0 ? config.headDim : static_cast<int>(query->sizeAt(3));
    int numKvHeads = config.numKvHeads > 0 ? config.numKvHeads : numHeads;
    int blockSize = config.blockSize;
    int maxBlocksPerSeq = static_cast<int>(pageTables->sizeAt(1));

    float scale = config.scale;
    if (scale <= 0.0f) {
        scale = 1.0f / sqrtf(static_cast<float>(headDim));
    }

    // Find max context length for shared memory sizing
    // (Use a fixed upper bound to avoid host sync; actual length checked in kernel)
    int maxCtxLen = maxBlocksPerSeq * blockSize;

    // Shared memory: qVec[headDim] + scores[maxCtxLen]
    // scores are always float (softmax computed in fp32); qVec uses T's size.
    // Alignment: qVec is T-typed, scores follow it — pad so the float* region
    // starts at a 4-byte boundary for all T sizes.
    int qVecBytes = headDim * static_cast<int>(query->sizeOfT());
    int qVecPadded = (qVecBytes + 3) & ~3;
    int smemBytes = qVecPadded + maxCtxLen * static_cast<int>(sizeof(float));
    // Cap at device limit, fall back to smaller ctxLen if needed
    if (smemBytes > 48 * 1024) {
        // Limit to 48KB shared memory; reduce maxCtxLen if needed
        smemBytes = 48 * 1024;
    }

    dim3 grid(batch, numHeads);
    int threadsPerBlock = std::min(256, headDim);

    auto dt = query->dataType();
    PointersManager pm(context, "pagedAttentionForward");

    if (dt == DataType::FLOAT32) {
        pagedAttentionDecodeKernel<float><<<grid, threadsPerBlock, smemBytes, *stream>>>(
            reinterpret_cast<const float*>(query->specialBuffer()),
            reinterpret_cast<const float*>(keyBlockPool->specialBuffer()),
            reinterpret_cast<const float*>(valueBlockPool->specialBuffer()),
            reinterpret_cast<const int*>(pageTables->specialBuffer()),
            reinterpret_cast<const int*>(contextLens->specialBuffer()),
            reinterpret_cast<float*>(output->specialBuffer()),
            numHeads, numKvHeads, headDim, blockSize, maxBlocksPerSeq, scale);
    } else if (dt == DataType::HALF) {
        pagedAttentionDecodeKernel<half><<<grid, threadsPerBlock, smemBytes, *stream>>>(
            reinterpret_cast<const half*>(query->specialBuffer()),
            reinterpret_cast<const half*>(keyBlockPool->specialBuffer()),
            reinterpret_cast<const half*>(valueBlockPool->specialBuffer()),
            reinterpret_cast<const int*>(pageTables->specialBuffer()),
            reinterpret_cast<const int*>(contextLens->specialBuffer()),
            reinterpret_cast<half*>(output->specialBuffer()),
            numHeads, numKvHeads, headDim, blockSize, maxBlocksPerSeq, scale);
    } else {
        // No dispatch = no output write = a silently wrong result. Fail loudly.
        // The op's DECLARE_TYPES restricts inputs to FLOAT32/HALF; reaching this
        // line means the descriptor contract was violated.
        THROW_EXCEPTION("pagedAttentionForward: unsupported data type (need FLOAT32 or HALF)");
    }

    pm.synchronize();
}

void pagedKvCacheAppend(
    NDArray* keyBlockPool,
    NDArray* valueBlockPool,
    NDArray* newKeys,
    NDArray* newValues,
    NDArray* pageTables,
    NDArray* contextLens,
    int blockSize,
    LaunchContext* context) {

    auto stream = context != nullptr ? context->getCudaStream() : nullptr;

    int batch = static_cast<int>(newKeys->sizeAt(0));
    int newLen = static_cast<int>(newKeys->sizeAt(1));
    int numKvHeads = static_cast<int>(newKeys->sizeAt(2));
    int headDim = static_cast<int>(newKeys->sizeAt(3));
    int maxBlocksPerSeq = static_cast<int>(pageTables->sizeAt(1));

    int totalElements = batch * newLen * numKvHeads * headDim;
    int threadsPerBlock = 256;
    int numBlocks = (totalElements + threadsPerBlock - 1) / threadsPerBlock;

    auto dt = newKeys->dataType();
    PointersManager pm(context, "pagedKvCacheAppend");

    if (dt == DataType::FLOAT32) {
        pagedKvAppendKernel<float><<<numBlocks, threadsPerBlock, 0, *stream>>>(
            reinterpret_cast<float*>(keyBlockPool->specialBuffer()),
            reinterpret_cast<float*>(valueBlockPool->specialBuffer()),
            reinterpret_cast<const float*>(newKeys->specialBuffer()),
            reinterpret_cast<const float*>(newValues->specialBuffer()),
            reinterpret_cast<const int*>(pageTables->specialBuffer()),
            reinterpret_cast<const int*>(contextLens->specialBuffer()),
            numKvHeads, headDim, blockSize, maxBlocksPerSeq, newLen, batch);
    } else if (dt == DataType::HALF) {
        pagedKvAppendKernel<half><<<numBlocks, threadsPerBlock, 0, *stream>>>(
            reinterpret_cast<half*>(keyBlockPool->specialBuffer()),
            reinterpret_cast<half*>(valueBlockPool->specialBuffer()),
            reinterpret_cast<const half*>(newKeys->specialBuffer()),
            reinterpret_cast<const half*>(newValues->specialBuffer()),
            reinterpret_cast<const int*>(pageTables->specialBuffer()),
            reinterpret_cast<const int*>(contextLens->specialBuffer()),
            numKvHeads, headDim, blockSize, maxBlocksPerSeq, newLen, batch);
    } else {
        // No dispatch = the pools are never written = silent corruption on later
        // reads of the appended positions. Fail loudly instead.
        THROW_EXCEPTION("pagedKvCacheAppend: unsupported data type (need FLOAT32 or HALF)");
    }

    pm.synchronize();
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
