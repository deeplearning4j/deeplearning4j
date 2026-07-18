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

#include <ops/declarable/helpers/two_way_cross_attention.h>
#include <array/NDArray.h>
#include <array/NDArrayFactory.h>
#include <helpers/DebugHelper.h>
#include <math/templatemath.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <ops/declarable/helpers/cuda/device_primitives.cuh>

namespace sd {
namespace ops {
namespace helpers {

static constexpr int TWA_WARP_SIZE = 32;

// Accumulator/scratch type: double when T=double for precision, float otherwise.
template <typename T>
struct AccType { using type = float; };
template <>
struct AccType<double> { using type = double; };

// Kernel: Compute attention logits: logits[i][j] = sum_k(Q[i][k] * K[j][k]) * scale
template <typename T>
SD_KERNEL void crossAttnLogitsKernel(
    const T* __restrict__ Q,
    const T* __restrict__ K,
    typename AccType<T>::type* __restrict__ logits,
    const LongType qLen,
    const LongType kLen,
    const LongType dim,
    const float scale) {

    using AccT = typename AccType<T>::type;

    const int i = blockIdx.y;
    const int j = blockIdx.x;
    if (i >= qLen || j >= kLen) return;

    extern __shared__ char sharedMem[];
    AccT* warpBuf = reinterpret_cast<AccT*>(sharedMem);

    const int lane = threadIdx.x % TWA_WARP_SIZE;
    const int wid = threadIdx.x / TWA_WARP_SIZE;
    const int numWarps = (blockDim.x + TWA_WARP_SIZE - 1) / TWA_WARP_SIZE;

    AccT threadDot = static_cast<AccT>(0);
    for (LongType k = threadIdx.x; k < dim; k += blockDim.x)
        threadDot += static_cast<AccT>(Q[i * dim + k]) * static_cast<AccT>(K[j * dim + k]);

    AccT dot = sd::device::blockReduceSum(threadDot, warpBuf);

    if (threadIdx.x == 0)
        logits[i * kLen + j] = dot * static_cast<AccT>(scale);
}

// Kernel: Row-wise softmax on logits matrix
template <typename AccT>
SD_KERNEL void crossAttnSoftmaxKernel(
    AccT* __restrict__ logits,
    const LongType rows,
    const LongType cols) {

    const int i = blockIdx.x;
    if (i >= rows) return;

    extern __shared__ char sharedMem[];
    AccT* warpBuf = reinterpret_cast<AccT*>(sharedMem);

    const int lane = threadIdx.x % TWA_WARP_SIZE;
    const int wid = threadIdx.x / TWA_WARP_SIZE;
    const int numWarps = (blockDim.x + TWA_WARP_SIZE - 1) / TWA_WARP_SIZE;

    AccT threadMax = -sd::DataTypeUtils::max<AccT>();
    for (LongType j = threadIdx.x; j < cols; j += blockDim.x)
        threadMax = sd::math::sd_max<AccT>(threadMax, logits[i * cols + j]);

    AccT rowMax = sd::device::blockAllReduceMax(threadMax, warpBuf);

    AccT threadSum = static_cast<AccT>(0);
    for (LongType j = threadIdx.x; j < cols; j += blockDim.x) {
        AccT val = sd::math::sd_exp<AccT, AccT>(logits[i * cols + j] - rowMax);
        logits[i * cols + j] = val;
        threadSum += val;
    }

    AccT rowSum = sd::device::blockAllReduceSum(threadSum, warpBuf);
    AccT sharedInvSum = static_cast<AccT>(1) / rowSum;

    for (LongType j = threadIdx.x; j < cols; j += blockDim.x)
        logits[i * cols + j] *= sharedInvSum;
}

// Kernel: Output = attnWeights @ V
// output[i][d] = sum_j(attnWeights[i][j] * V[j][d])
template <typename T>
SD_KERNEL void crossAttnOutputKernel(
    const typename AccType<T>::type* __restrict__ attnWeights,
    const T* __restrict__ V,
    T* __restrict__ output,
    const LongType qLen,
    const LongType kLen,
    const LongType dim) {

    using AccT = typename AccType<T>::type;

    const LongType total = qLen * dim;
    for (LongType idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += blockDim.x * gridDim.x) {
        const int i = idx / dim;
        const int d = idx % dim;

        AccT acc = static_cast<AccT>(0);
        for (LongType j = 0; j < kLen; j++) {
            acc += attnWeights[i * kLen + j] * static_cast<AccT>(V[j * dim + d]);
        }
        output[i * dim + d] = static_cast<T>(acc);
    }
}

template <typename T>
void twoWayCrossAttentionCudaLauncher(const cudaStream_t* stream,
                                       const void* vTokenQ, const void* vTokenK, const void* vTokenV,
                                       const void* vImageQ, const void* vImageK, const void* vImageV,
                                       void* vTokenOut, void* vImageOut,
                                       void* vLogits1, void* vLogits2,
                                       LongType tokenLen, LongType imageLen, LongType dim,
                                       float scale) {
    auto tokenQ = reinterpret_cast<const T*>(vTokenQ);
    auto tokenK = reinterpret_cast<const T*>(vTokenK);
    auto tokenV = reinterpret_cast<const T*>(vTokenV);
    auto imageQ = reinterpret_cast<const T*>(vImageQ);
    auto imageK = reinterpret_cast<const T*>(vImageK);
    auto imageV = reinterpret_cast<const T*>(vImageV);
    auto tokenOut = reinterpret_cast<T*>(vTokenOut);
    auto imageOut = reinterpret_cast<T*>(vImageOut);
    using AccT = typename AccType<T>::type;
    auto logits1 = reinterpret_cast<AccT*>(vLogits1);
    auto logits2 = reinterpret_cast<AccT*>(vLogits2);

    int dotThreads = 256;
    if (dim < 256) {
        dotThreads = ((dim + TWA_WARP_SIZE - 1) / TWA_WARP_SIZE) * TWA_WARP_SIZE;
        if (dotThreads < TWA_WARP_SIZE) dotThreads = TWA_WARP_SIZE;
    }
    int numWarps = (dotThreads + TWA_WARP_SIZE - 1) / TWA_WARP_SIZE;
    size_t sharedSize = numWarps * sizeof(AccT);

    // Direction 1: token attends to image
    // logits1 = tokenQ @ imageK^T * scale
    {
        dim3 grid1(imageLen, tokenLen);
        crossAttnLogitsKernel<T><<<grid1, dotThreads, sharedSize, *stream>>>(
            tokenQ, imageK, logits1, tokenLen, imageLen, dim, scale);
        DebugHelper::checkGlobalErrorCode("crossAttnLogits1 failed");

        int smThreads = 256;
        if (imageLen < 256) {
            smThreads = ((imageLen + TWA_WARP_SIZE - 1) / TWA_WARP_SIZE) * TWA_WARP_SIZE;
            if (smThreads < TWA_WARP_SIZE) smThreads = TWA_WARP_SIZE;
        }
        int smWarps = (smThreads + TWA_WARP_SIZE - 1) / TWA_WARP_SIZE;
        crossAttnSoftmaxKernel<AccT><<<tokenLen, smThreads, smWarps * sizeof(AccT), *stream>>>(
            logits1, tokenLen, imageLen);
        DebugHelper::checkGlobalErrorCode("crossAttnSoftmax1 failed");

        LongType outElems = tokenLen * dim;
        int outThreads = 256;
        int outBlocks = (outElems + outThreads - 1) / outThreads;
        crossAttnOutputKernel<T><<<outBlocks, outThreads, 0, *stream>>>(
            logits1, imageV, tokenOut, tokenLen, imageLen, dim);
        DebugHelper::checkGlobalErrorCode("crossAttnOutput1 failed");
    }

    // Direction 2: image attends to token
    // logits2 = imageQ @ tokenK^T * scale
    {
        dim3 grid2(tokenLen, imageLen);
        crossAttnLogitsKernel<T><<<grid2, dotThreads, sharedSize, *stream>>>(
            imageQ, tokenK, logits2, imageLen, tokenLen, dim, scale);
        DebugHelper::checkGlobalErrorCode("crossAttnLogits2 failed");

        int smThreads = 256;
        if (tokenLen < 256) {
            smThreads = ((tokenLen + TWA_WARP_SIZE - 1) / TWA_WARP_SIZE) * TWA_WARP_SIZE;
            if (smThreads < TWA_WARP_SIZE) smThreads = TWA_WARP_SIZE;
        }
        int smWarps = (smThreads + TWA_WARP_SIZE - 1) / TWA_WARP_SIZE;
        crossAttnSoftmaxKernel<AccT><<<imageLen, smThreads, smWarps * sizeof(AccT), *stream>>>(
            logits2, imageLen, tokenLen);
        DebugHelper::checkGlobalErrorCode("crossAttnSoftmax2 failed");

        LongType outElems = imageLen * dim;
        int outThreads = 256;
        int outBlocks = (outElems + outThreads - 1) / outThreads;
        crossAttnOutputKernel<T><<<outBlocks, outThreads, 0, *stream>>>(
            logits2, tokenV, imageOut, imageLen, tokenLen, dim);
        DebugHelper::checkGlobalErrorCode("crossAttnOutput2 failed");
    }
}

BUILD_SINGLE_TEMPLATE(void twoWayCrossAttentionCudaLauncher,
                      (const cudaStream_t* stream,
                       const void* vTokenQ, const void* vTokenK, const void* vTokenV,
                       const void* vImageQ, const void* vImageK, const void* vImageV,
                       void* vTokenOut, void* vImageOut,
                       void* vLogits1, void* vLogits2,
                       LongType tokenLen, LongType imageLen, LongType dim, float scale),
                      SD_FLOAT_TYPES);

void twoWayCrossAttention(NDArray* tokenQuery, NDArray* tokenKey,
                           NDArray* tokenValue, NDArray* imageQuery,
                           NDArray* imageKey, NDArray* imageValue,
                           NDArray* tokenOutput, NDArray* imageOutput,
                           double scale, LaunchContext* context) {
    auto rank = tokenQuery->rankOf();
    auto stream = context->getCudaStream();

    NDArray::prepareSpecialUse({tokenOutput, imageOutput},
                               {tokenQuery, tokenKey, tokenValue,
                                imageQuery, imageKey, imageValue});

    auto accDtype = tokenQuery->dataType() == DataType::DOUBLE ? DataType::DOUBLE : DataType::FLOAT32;

    if (rank == 2) {
        auto tokenLen = tokenQuery->sizeAt(0);
        auto imageLen = imageQuery->sizeAt(0);
        auto dim = tokenQuery->sizeAt(1);

        auto logits1 = NDArrayFactory::create('c', {tokenLen, imageLen}, accDtype, context);
        auto logits2 = NDArrayFactory::create('c', {imageLen, tokenLen}, accDtype, context);

        BUILD_SINGLE_SELECTOR(tokenQuery->dataType(), twoWayCrossAttentionCudaLauncher,
                              (stream,
                               tokenQuery->specialBuffer(), tokenKey->specialBuffer(),
                               tokenValue->specialBuffer(),
                               imageQuery->specialBuffer(), imageKey->specialBuffer(),
                               imageValue->specialBuffer(),
                               tokenOutput->specialBuffer(), imageOutput->specialBuffer(),
                               logits1->specialBuffer(), logits2->specialBuffer(),
                               tokenLen, imageLen, dim, static_cast<float>(scale)),
                              SD_FLOAT_TYPES);

        delete logits1;
        delete logits2;
    } else {
        // rank 3: [batch, seq, dim] — process each batch via pointer arithmetic
        auto batchSize = tokenQuery->sizeAt(0);
        auto tokenLen = tokenQuery->sizeAt(1);
        auto imageLen = imageQuery->sizeAt(1);
        auto dim = tokenQuery->sizeAt(2);

        auto elemSize = tokenQuery->sizeOfT();
        LongType tokenSliceSize = tokenLen * dim;
        LongType imageSliceSize = imageLen * dim;

        auto dt = tokenQuery->dataType();

        for (LongType b = 0; b < batchSize; b++) {
            LongType tokOff = b * tokenSliceSize * elemSize;
            LongType imgOff = b * imageSliceSize * elemSize;

            // Allocate temp logits inside loop so each batch gets fresh buffers
            auto logits1 = NDArrayFactory::create('c', {tokenLen, imageLen}, accDtype, context);
            auto logits2 = NDArrayFactory::create('c', {imageLen, tokenLen}, accDtype, context);

            const void* tQBuf = static_cast<const int8_t*>(tokenQuery->specialBuffer()) + tokOff;
            const void* tKBuf = static_cast<const int8_t*>(tokenKey->specialBuffer()) + tokOff;
            const void* tVBuf = static_cast<const int8_t*>(tokenValue->specialBuffer()) + tokOff;
            const void* iQBuf = static_cast<const int8_t*>(imageQuery->specialBuffer()) + imgOff;
            const void* iKBuf = static_cast<const int8_t*>(imageKey->specialBuffer()) + imgOff;
            const void* iVBuf = static_cast<const int8_t*>(imageValue->specialBuffer()) + imgOff;
            void* tOutBuf = static_cast<int8_t*>(tokenOutput->specialBuffer()) + tokOff;
            void* iOutBuf = static_cast<int8_t*>(imageOutput->specialBuffer()) + imgOff;

            BUILD_SINGLE_SELECTOR(dt, twoWayCrossAttentionCudaLauncher,
                                  (stream,
                                   tQBuf, tKBuf, tVBuf,
                                   iQBuf, iKBuf, iVBuf,
                                   tOutBuf, iOutBuf,
                                   logits1->specialBuffer(), logits2->specialBuffer(),
                                   tokenLen, imageLen, dim, static_cast<float>(scale)),
                                  SD_FLOAT_TYPES);

            delete logits1;
            delete logits2;
        }
    }

    NDArray::registerSpecialUse({tokenOutput, imageOutput},
                                {tokenQuery, tokenKey, tokenValue,
                                 imageQuery, imageKey, imageValue});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
