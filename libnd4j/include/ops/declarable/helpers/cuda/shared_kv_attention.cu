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

//
// @author Eclipse Deeplearning4j
//

#include <cuda_runtime.h>
#include <helpers/PointersManager.h>
#include <helpers/DebugHelper.h>
#include <ops/declarable/helpers/shared_kv_attention.h>
#include <execution/cuda/LaunchDims.h>

#include <cfloat>
#include <ops/declarable/helpers/cuda/device_primitives.cuh>

namespace sd {
namespace ops {
namespace helpers {

// Accumulator type: double when T=double for precision, float otherwise.
template <typename T>
struct AccType { using type = float; };
template <>
struct AccType<double> { using type = double; };

static constexpr int SKV_WARP_SIZE = 32;

/**
 * CUDA kernel: shared KV attention.
 *
 * One block per (batch, queryHead, queryPosition).
 * Threads cooperate over the kvSeqLen dimension for the dot-product / softmax,
 * then cooperate over headDim for the weighted-sum output.
 *
 * Template parameter T is the element data type.
 */
template <typename T>
SD_KERNEL __launch_bounds__(256, 2) void sharedKvAttentionKernel(
    const T* __restrict__ query,        // [batch, numHeads,   qSeqLen,  headDim]
    const T* __restrict__ key,          // [batch, numKvHeads, kvSeqLen, headDim]
    const T* __restrict__ value,        // [batch, numKvHeads, kvSeqLen, headDim]
    const T* __restrict__ mask,         // [batch, 1, qSeqLen, kvSeqLen] or nullptr
    T* __restrict__ output,             // [batch, numHeads,   qSeqLen,  headDim]
    // shape info
    const sd::LongType batchSize,
    const int numHeads,
    const int numKvHeads,
    const sd::LongType qSeqLen,
    const sd::LongType kvSeqLen,
    const sd::LongType headDim,
    // flags
    const bool causal,
    const int slidingWindowSize,
    const float scale,
    // query strides
    const sd::LongType qBatchStride,
    const sd::LongType qHeadStride,
    const sd::LongType qSeqStride,
    const sd::LongType qDimStride,
    // key strides
    const sd::LongType kBatchStride,
    const sd::LongType kHeadStride,
    const sd::LongType kSeqStride,
    const sd::LongType kDimStride,
    // value strides
    const sd::LongType vBatchStride,
    const sd::LongType vHeadStride,
    const sd::LongType vSeqStride,
    const sd::LongType vDimStride,
    // mask strides (0 if mask == nullptr)
    const sd::LongType mBatchStride,
    const sd::LongType mSeqStride,
    const sd::LongType mKvStride,
    // output strides
    const sd::LongType oBatchStride,
    const sd::LongType oHeadStride,
    const sd::LongType oSeqStride,
    const sd::LongType oDimStride) {
    using AccT = typename AccType<T>::type;

    // Grid-stride: each block handles one (batch, head, queryPos) triple
    const sd::LongType totalPositions = batchSize * numHeads * qSeqLen;
    const sd::LongType blockId = blockIdx.x;
    if (blockId >= totalPositions) return;

    const sd::LongType b = blockId / (numHeads * qSeqLen);
    const sd::LongType h = (blockId / qSeqLen) % numHeads;
    const sd::LongType q = blockId % qSeqLen;

    const int headsPerGroup = numHeads / numKvHeads;
    const int kvHead = static_cast<int>(h) / headsPerGroup;

    // Base pointers
    const T* queryVec = query  + b * qBatchStride + h * qHeadStride + q * qSeqStride;
    const T* keyBase  = key    + b * kBatchStride + kvHead * kHeadStride;
    const T* valBase  = value  + b * vBatchStride + kvHead * vHeadStride;
    T* outputVec      = output + b * oBatchStride + h * oHeadStride + q * oSeqStride;

    // Shared memory layout: warpBuf for reductions
    extern __shared__ char sharedMem[];
    AccT* warpBuf = reinterpret_cast<AccT*>(sharedMem);

    const int numWarps = (blockDim.x + SKV_WARP_SIZE - 1) / SKV_WARP_SIZE;

    // ===== Pass 1: compute scores and find max =====
    // Each thread handles a subset of kv positions.
    // We do an online max across the thread's positions.
    AccT threadMax = -sd::DataTypeUtils::max<AccT>();

    // We need to store scores somewhere. Use a second region of shared memory
    // after warpBuf (numWarps AccT).  We also need kvSeqLen AccT for scores.
    AccT* scores = warpBuf + numWarps;

    for (sd::LongType k = threadIdx.x; k < kvSeqLen; k += blockDim.x) {
        const T* keyVec = keyBase + k * kSeqStride;

        // Dot product Q . K
        AccT dot = static_cast<AccT>(0);
        for (sd::LongType d = 0; d < headDim; d++) {
            dot += static_cast<AccT>(queryVec[d * qDimStride]) *
                   static_cast<AccT>(keyVec[d * kDimStride]);
        }
        dot *= static_cast<AccT>(scale);

        // Causal mask
        if (causal && k > q) {
            dot = -sd::DataTypeUtils::max<AccT>();
        }

        // Sliding window mask
        if (slidingWindowSize > 0 && (static_cast<sd::LongType>(q) - k) > slidingWindowSize) {
            dot = -sd::DataTypeUtils::max<AccT>();
        }

        // External additive mask
        if (mask != nullptr) {
            AccT maskVal = static_cast<AccT>(
                mask[b * mBatchStride + q * mSeqStride + k * mKvStride]);
            dot += maskVal;
        }

        scores[k] = dot;
        if (dot > threadMax) threadMax = dot;
    }
    __syncthreads();

    // Block-reduce max (result broadcast to all threads)
    AccT rowMax = sd::device::blockAllReduceMax(threadMax, warpBuf);

    // ===== Pass 2: exp(score - max) and sum =====
    AccT threadSum = static_cast<AccT>(0);
    for (sd::LongType k = threadIdx.x; k < kvSeqLen; k += blockDim.x) {
        AccT val = sd::math::sd_exp<AccT, AccT>(scores[k] - rowMax);
        scores[k] = val;
        threadSum += val;
    }
    __syncthreads();

    // Block-reduce sum (result broadcast to all threads)
    AccT blockSum = sd::device::blockAllReduceSum(threadSum, warpBuf);
    AccT sharedInvSum = static_cast<AccT>(1) / blockSum;

    // Normalize scores
    for (sd::LongType k = threadIdx.x; k < kvSeqLen; k += blockDim.x) {
        scores[k] *= sharedInvSum;
    }
    __syncthreads();

    // ===== Pass 3: weighted sum output = scores @ V =====
    // Threads cooperate over headDim
    for (sd::LongType d = threadIdx.x; d < headDim; d += blockDim.x) {
        AccT acc = static_cast<AccT>(0);
        for (sd::LongType k = 0; k < kvSeqLen; k++) {
            acc += scores[k] * static_cast<AccT>(valBase[k * vSeqStride + d * vDimStride]);
        }
        outputVec[d * oDimStride] = static_cast<T>(acc);
    }
}

template <typename T>
static void sharedKvAttentionCuda_(LaunchContext* context,
                                    NDArray* query,
                                    NDArray* sharedKey,
                                    NDArray* sharedValue,
                                    NDArray* mask,
                                    NDArray* output,
                                    int numHeads,
                                    int numKvHeads,
                                    bool causal,
                                    int slidingWindowSize,
                                    double scale) {

    const auto batchSize = query->sizeAt(0);
    const auto qSeqLen   = query->sizeAt(2);
    const auto headDim   = query->sizeAt(3);
    const auto kvSeqLen  = sharedKey->sizeAt(2);

    const sd::LongType totalPositions = batchSize * numHeads * qSeqLen;

    // Use the registered launch dims for default block size
    dim3 defaultDims = getLaunchDims("shared_kv_attention");
    int blockSize = defaultDims.y;  // block size from the map

    // Shared memory: numWarps AccT (warpBuf) + kvSeqLen AccT (scores)
    using AccT = typename AccType<T>::type;
    const int numWarps = (blockSize + SKV_WARP_SIZE - 1) / SKV_WARP_SIZE;
    const size_t sharedMemSize = (numWarps + kvSeqLen) * sizeof(AccT);

    PointersManager manager(context, "sharedKvAttention");

    // Strides
    const auto qBatchStride = query->strideAt(0);
    const auto qHeadStride  = query->strideAt(1);
    const auto qSeqStride   = query->strideAt(2);
    const auto qDimStride   = query->strideAt(3);

    const auto kBatchStride = sharedKey->strideAt(0);
    const auto kHeadStride  = sharedKey->strideAt(1);
    const auto kSeqStride   = sharedKey->strideAt(2);
    const auto kDimStride   = sharedKey->strideAt(3);

    const auto vBatchStride = sharedValue->strideAt(0);
    const auto vHeadStride  = sharedValue->strideAt(1);
    const auto vSeqStride   = sharedValue->strideAt(2);
    const auto vDimStride   = sharedValue->strideAt(3);

    sd::LongType mBatchStride = 0, mSeqStride = 0, mKvStride = 0;
    if (mask != nullptr) {
        mBatchStride = mask->strideAt(0);
        mSeqStride   = mask->strideAt(2);
        mKvStride    = mask->strideAt(3);
    }

    const auto oBatchStride = output->strideAt(0);
    const auto oHeadStride  = output->strideAt(1);
    const auto oSeqStride   = output->strideAt(2);
    const auto oDimStride   = output->strideAt(3);

    const T* qPtr = query->specialBufferasT<T>();
    const T* kPtr = sharedKey->specialBufferasT<T>();
    const T* vPtr = sharedValue->specialBufferasT<T>();
    const T* mPtr = (mask != nullptr) ? mask->specialBufferasT<T>() : nullptr;
    T* oPtr       = output->specialBufferasT<T>();

    sharedKvAttentionKernel<T><<<totalPositions, blockSize, sharedMemSize,
                                  *context->getCudaStream()>>>(
        qPtr, kPtr, vPtr, mPtr, oPtr,
        batchSize, numHeads, numKvHeads, qSeqLen, kvSeqLen, headDim,
        causal, slidingWindowSize, static_cast<float>(scale),
        qBatchStride, qHeadStride, qSeqStride, qDimStride,
        kBatchStride, kHeadStride, kSeqStride, kDimStride,
        vBatchStride, vHeadStride, vSeqStride, vDimStride,
        mBatchStride, mSeqStride, mKvStride,
        oBatchStride, oHeadStride, oSeqStride, oDimStride);

    DebugHelper::checkGlobalErrorCode("sharedKvAttentionKernel failed");

    manager.synchronize();
}

BUILD_SINGLE_TEMPLATE(void sharedKvAttentionCuda_,
                      (LaunchContext* context, NDArray* query, NDArray* sharedKey,
                       NDArray* sharedValue, NDArray* mask, NDArray* output,
                       int numHeads, int numKvHeads, bool causal,
                       int slidingWindowSize, double scale),
                      SD_FLOAT_TYPES);

void sharedKvAttention(LaunchContext* context,
                        NDArray* query,
                        NDArray* sharedKey,
                        NDArray* sharedValue,
                        NDArray* mask,
                        NDArray* output,
                        int numHeads,
                        int numKvHeads,
                        bool causal,
                        int slidingWindowSize,
                        double scale) {

    if (mask != nullptr) {
        NDArray::prepareSpecialUse({output}, {query, sharedKey, sharedValue, mask});
    } else {
        NDArray::prepareSpecialUse({output}, {query, sharedKey, sharedValue});
    }

    BUILD_SINGLE_SELECTOR(query->dataType(), sharedKvAttentionCuda_,
                          (context, query, sharedKey, sharedValue, mask, output,
                           numHeads, numKvHeads, causal, slidingWindowSize, scale),
                          SD_FLOAT_TYPES);

    if (mask != nullptr) {
        NDArray::registerSpecialUse({output}, {query, sharedKey, sharedValue, mask});
    } else {
        NDArray::registerSpecialUse({output}, {query, sharedKey, sharedValue});
    }
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
