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

namespace sd {
namespace ops {
namespace helpers {

static constexpr int TWA_BP_WARP_SIZE = 32;

// Reuse logits kernel from forward (recompute attention weights)
template <typename T>
SD_KERNEL void bpCrossAttnLogitsKernel(
    const T* __restrict__ Q,
    const T* __restrict__ K,
    float* __restrict__ logits,
    const LongType qLen,
    const LongType kLen,
    const LongType dim,
    const float scale) {

    const int i = blockIdx.y;
    const int j = blockIdx.x;
    if (i >= qLen || j >= kLen) return;

    extern __shared__ char sharedMem[];
    float* warpBuf = reinterpret_cast<float*>(sharedMem);

    const int lane = threadIdx.x % TWA_BP_WARP_SIZE;
    const int wid = threadIdx.x / TWA_BP_WARP_SIZE;
    const int numWarps = (blockDim.x + TWA_BP_WARP_SIZE - 1) / TWA_BP_WARP_SIZE;

    float threadDot = 0.0f;
    for (LongType k = threadIdx.x; k < dim; k += blockDim.x)
        threadDot += static_cast<float>(Q[i * dim + k]) * static_cast<float>(K[j * dim + k]);

    for (int offset = TWA_BP_WARP_SIZE / 2; offset > 0; offset /= 2)
        threadDot += __shfl_down_sync(0xffffffff, threadDot, offset);
    if (lane == 0) warpBuf[wid] = threadDot;
    __syncthreads();

    float dot = 0.0f;
    if (threadIdx.x < numWarps) dot = warpBuf[threadIdx.x];
    for (int offset = TWA_BP_WARP_SIZE / 2; offset > 0; offset /= 2)
        dot += __shfl_down_sync(0xffffffff, dot, offset);

    if (threadIdx.x == 0)
        logits[i * kLen + j] = dot * scale;
}

// Row-wise softmax (in-place)
SD_KERNEL void bpCrossAttnSoftmaxKernel(
    float* __restrict__ logits,
    const LongType rows,
    const LongType cols) {

    const int i = blockIdx.x;
    if (i >= rows) return;

    extern __shared__ char sharedMem[];
    float* warpBuf = reinterpret_cast<float*>(sharedMem);

    const int lane = threadIdx.x % TWA_BP_WARP_SIZE;
    const int wid = threadIdx.x / TWA_BP_WARP_SIZE;
    const int numWarps = (blockDim.x + TWA_BP_WARP_SIZE - 1) / TWA_BP_WARP_SIZE;

    float threadMax = -FLT_MAX;
    for (LongType j = threadIdx.x; j < cols; j += blockDim.x)
        threadMax = sd::math::sd_max<float>(threadMax, logits[i * cols + j]);

    for (int offset = TWA_BP_WARP_SIZE / 2; offset > 0; offset /= 2)
        threadMax = sd::math::sd_max<float>(threadMax, __shfl_down_sync(0xffffffff, threadMax, offset));
    if (lane == 0) warpBuf[wid] = threadMax;
    __syncthreads();

    float rowMax = -FLT_MAX;
    if (threadIdx.x < numWarps) rowMax = warpBuf[threadIdx.x];
    for (int offset = TWA_BP_WARP_SIZE / 2; offset > 0; offset /= 2)
        rowMax = sd::math::sd_max<float>(rowMax, __shfl_down_sync(0xffffffff, rowMax, offset));
    __shared__ float sharedMax;
    if (threadIdx.x == 0) sharedMax = rowMax;
    __syncthreads();
    rowMax = sharedMax;

    float threadSum = 0.0f;
    for (LongType j = threadIdx.x; j < cols; j += blockDim.x) {
        float val = sd::math::sd_exp<float, float>(logits[i * cols + j] - rowMax);
        logits[i * cols + j] = val;
        threadSum += val;
    }

    for (int offset = TWA_BP_WARP_SIZE / 2; offset > 0; offset /= 2)
        threadSum += __shfl_down_sync(0xffffffff, threadSum, offset);
    if (lane == 0) warpBuf[wid] = threadSum;
    __syncthreads();

    float rowSum = 0.0f;
    if (threadIdx.x < numWarps) rowSum = warpBuf[threadIdx.x];
    for (int offset = TWA_BP_WARP_SIZE / 2; offset > 0; offset /= 2)
        rowSum += __shfl_down_sync(0xffffffff, rowSum, offset);
    __shared__ float sharedInvSum;
    if (threadIdx.x == 0) sharedInvSum = 1.0f / rowSum;
    __syncthreads();

    for (LongType j = threadIdx.x; j < cols; j += blockDim.x)
        logits[i * cols + j] *= sharedInvSum;
}

// dL/dAttnWeights = gradOut @ V^T => dLdAttn[i][j] = sum_d(gradOut[i][d] * V[j][d])
template <typename T>
SD_KERNEL void bpDLdAttnKernel(
    const T* __restrict__ gradOut,
    const T* __restrict__ V,
    float* __restrict__ dLdAttn,
    const LongType qLen,
    const LongType kLen,
    const LongType dim) {

    const int i = blockIdx.y;
    const int j = blockIdx.x;
    if (i >= qLen || j >= kLen) return;

    extern __shared__ char sharedMem[];
    float* warpBuf = reinterpret_cast<float*>(sharedMem);

    const int lane = threadIdx.x % TWA_BP_WARP_SIZE;
    const int wid = threadIdx.x / TWA_BP_WARP_SIZE;
    const int numWarps = (blockDim.x + TWA_BP_WARP_SIZE - 1) / TWA_BP_WARP_SIZE;

    float threadDot = 0.0f;
    for (LongType d = threadIdx.x; d < dim; d += blockDim.x)
        threadDot += static_cast<float>(gradOut[i * dim + d]) * static_cast<float>(V[j * dim + d]);

    for (int offset = TWA_BP_WARP_SIZE / 2; offset > 0; offset /= 2)
        threadDot += __shfl_down_sync(0xffffffff, threadDot, offset);
    if (lane == 0) warpBuf[wid] = threadDot;
    __syncthreads();

    float dot = 0.0f;
    if (threadIdx.x < numWarps) dot = warpBuf[threadIdx.x];
    for (int offset = TWA_BP_WARP_SIZE / 2; offset > 0; offset /= 2)
        dot += __shfl_down_sync(0xffffffff, dot, offset);

    if (threadIdx.x == 0)
        dLdAttn[i * kLen + j] = dot;
}

// dL/dV = attnWeights^T @ gradOut => dLdV[j][d] = sum_i(attnWeights[i][j] * gradOut[i][d])
template <typename T>
SD_KERNEL void bpDLdVKernel(
    const float* __restrict__ attnWeights,
    const T* __restrict__ gradOut,
    T* __restrict__ dLdV,
    const LongType qLen,
    const LongType kLen,
    const LongType dim) {

    const LongType total = kLen * dim;
    for (LongType idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += blockDim.x * gridDim.x) {
        const int j = idx / dim;
        const int d = idx % dim;

        float acc = 0.0f;
        for (LongType i = 0; i < qLen; i++) {
            acc += attnWeights[i * kLen + j] * static_cast<float>(gradOut[i * dim + d]);
        }
        dLdV[j * dim + d] = static_cast<T>(acc);
    }
}

// Softmax backward: dL/dLogits = attnWeights * (dLdAttn - rowwise_dot(dLdAttn, attnWeights))
SD_KERNEL void bpSoftmaxBackwardKernel(
    const float* __restrict__ attnWeights,
    float* __restrict__ dLdAttn,  // in: dL/dAttnWeights, out: dL/dLogits
    const LongType rows,
    const LongType cols) {

    const int i = blockIdx.x;
    if (i >= rows) return;

    extern __shared__ char sharedMem[];
    float* warpBuf = reinterpret_cast<float*>(sharedMem);

    const int lane = threadIdx.x % TWA_BP_WARP_SIZE;
    const int wid = threadIdx.x / TWA_BP_WARP_SIZE;
    const int numWarps = (blockDim.x + TWA_BP_WARP_SIZE - 1) / TWA_BP_WARP_SIZE;

    // Compute dot(dLdAttn[i], attnWeights[i])
    float threadDot = 0.0f;
    for (LongType j = threadIdx.x; j < cols; j += blockDim.x)
        threadDot += dLdAttn[i * cols + j] * attnWeights[i * cols + j];

    for (int offset = TWA_BP_WARP_SIZE / 2; offset > 0; offset /= 2)
        threadDot += __shfl_down_sync(0xffffffff, threadDot, offset);
    if (lane == 0) warpBuf[wid] = threadDot;
    __syncthreads();

    float rowDot = 0.0f;
    if (threadIdx.x < numWarps) rowDot = warpBuf[threadIdx.x];
    for (int offset = TWA_BP_WARP_SIZE / 2; offset > 0; offset /= 2)
        rowDot += __shfl_down_sync(0xffffffff, rowDot, offset);
    __shared__ float sharedDot;
    if (threadIdx.x == 0) sharedDot = rowDot;
    __syncthreads();

    // dL/dLogits[i][j] = attnWeights[i][j] * (dLdAttn[i][j] - dot)
    for (LongType j = threadIdx.x; j < cols; j += blockDim.x)
        dLdAttn[i * cols + j] = attnWeights[i * cols + j] * (dLdAttn[i * cols + j] - sharedDot);
}

// dL/dQ = dL/dLogits @ K * scale => dLdQ[i][d] = sum_j(dLdLogits[i][j] * K[j][d]) * scale
template <typename T>
SD_KERNEL void bpDLdQKernel(
    const float* __restrict__ dLdLogits,
    const T* __restrict__ K,
    T* __restrict__ dLdQ,
    const LongType qLen,
    const LongType kLen,
    const LongType dim,
    const float scale) {

    const LongType total = qLen * dim;
    for (LongType idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += blockDim.x * gridDim.x) {
        const int i = idx / dim;
        const int d = idx % dim;

        float acc = 0.0f;
        for (LongType j = 0; j < kLen; j++) {
            acc += dLdLogits[i * kLen + j] * static_cast<float>(K[j * dim + d]);
        }
        dLdQ[i * dim + d] = static_cast<T>(acc * scale);
    }
}

// dL/dK = dL/dLogits^T @ Q * scale => dLdK[j][d] = sum_i(dLdLogits[i][j] * Q[i][d]) * scale
template <typename T>
SD_KERNEL void bpDLdKKernel(
    const float* __restrict__ dLdLogits,
    const T* __restrict__ Q,
    T* __restrict__ dLdK,
    const LongType qLen,
    const LongType kLen,
    const LongType dim,
    const float scale) {

    const LongType total = kLen * dim;
    for (LongType idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += blockDim.x * gridDim.x) {
        const int j = idx / dim;
        const int d = idx % dim;

        float acc = 0.0f;
        for (LongType i = 0; i < qLen; i++) {
            acc += dLdLogits[i * kLen + j] * static_cast<float>(Q[i * dim + d]);
        }
        dLdK[j * dim + d] = static_cast<T>(acc * scale);
    }
}

template <typename T>
void twoWayCrossAttentionBpCudaLauncher(
    const cudaStream_t* stream,
    const void* vTokenQ, const void* vTokenK, const void* vTokenV,
    const void* vImageQ, const void* vImageK, const void* vImageV,
    const void* vGradTokenOut, const void* vGradImageOut,
    void* vDLdTokenQ, void* vDLdTokenK, void* vDLdTokenV,
    void* vDLdImageQ, void* vDLdImageK, void* vDLdImageV,
    void* vAttn1, void* vAttn2,
    LongType tokenLen, LongType imageLen, LongType dim, float scale) {

    auto tokenQ = reinterpret_cast<const T*>(vTokenQ);
    auto tokenK = reinterpret_cast<const T*>(vTokenK);
    auto tokenV = reinterpret_cast<const T*>(vTokenV);
    auto imageQ = reinterpret_cast<const T*>(vImageQ);
    auto imageK = reinterpret_cast<const T*>(vImageK);
    auto imageV = reinterpret_cast<const T*>(vImageV);
    auto gradTokenOut = reinterpret_cast<const T*>(vGradTokenOut);
    auto gradImageOut = reinterpret_cast<const T*>(vGradImageOut);
    auto dLdTokenQ = reinterpret_cast<T*>(vDLdTokenQ);
    auto dLdTokenK = reinterpret_cast<T*>(vDLdTokenK);
    auto dLdTokenV = reinterpret_cast<T*>(vDLdTokenV);
    auto dLdImageQ = reinterpret_cast<T*>(vDLdImageQ);
    auto dLdImageK = reinterpret_cast<T*>(vDLdImageK);
    auto dLdImageV = reinterpret_cast<T*>(vDLdImageV);
    auto attn1 = reinterpret_cast<float*>(vAttn1);  // [tokenLen, imageLen]
    auto attn2 = reinterpret_cast<float*>(vAttn2);  // [imageLen, tokenLen]

    int dotThreads = 256;
    if (dim < 256) {
        dotThreads = ((dim + TWA_BP_WARP_SIZE - 1) / TWA_BP_WARP_SIZE) * TWA_BP_WARP_SIZE;
        if (dotThreads < TWA_BP_WARP_SIZE) dotThreads = TWA_BP_WARP_SIZE;
    }
    int numWarps = (dotThreads + TWA_BP_WARP_SIZE - 1) / TWA_BP_WARP_SIZE;
    size_t sharedSize = numWarps * sizeof(float);

    // ---- Direction 1: tokenOut = softmax(tokenQ @ imageK^T * scale) @ imageV ----
    {
        // Recompute attention weights
        dim3 grid1(imageLen, tokenLen);
        bpCrossAttnLogitsKernel<T><<<grid1, dotThreads, sharedSize, *stream>>>(
            tokenQ, imageK, attn1, tokenLen, imageLen, dim, scale);
        DebugHelper::checkGlobalErrorCode("bpLogits1 failed");

        int smThreads = 256;
        if (imageLen < 256) {
            smThreads = ((imageLen + TWA_BP_WARP_SIZE - 1) / TWA_BP_WARP_SIZE) * TWA_BP_WARP_SIZE;
            if (smThreads < TWA_BP_WARP_SIZE) smThreads = TWA_BP_WARP_SIZE;
        }
        int smWarps = (smThreads + TWA_BP_WARP_SIZE - 1) / TWA_BP_WARP_SIZE;
        bpCrossAttnSoftmaxKernel<<<tokenLen, smThreads, smWarps * sizeof(float), *stream>>>(
            attn1, tokenLen, imageLen);
        DebugHelper::checkGlobalErrorCode("bpSoftmax1 failed");

        // dL/dImageValue = attn1^T @ gradTokenOut
        LongType vElems = imageLen * dim;
        int vThreads = 256;
        int vBlocks = (vElems + vThreads - 1) / vThreads;
        bpDLdVKernel<T><<<vBlocks, vThreads, 0, *stream>>>(
            attn1, gradTokenOut, dLdImageV, tokenLen, imageLen, dim);
        DebugHelper::checkGlobalErrorCode("bpDLdImageV failed");

        // dL/dAttn1 = gradTokenOut @ imageV^T (reuse attn1 buffer for dLdLogits after)
        // First compute dLdAttn into a temp buffer, then softmax backward
        // We can reuse attn1 space: first compute dLdAttn, store separately
        // Actually, we need attn1 (the weights) for softmax backward AND dLdAttn
        // So we need a separate buffer. We'll use attn2 temporarily since Dir2 hasn't started.
        // Wait - attn2 is needed for Dir2. Let's allocate dLdAttn inline.
        // Actually, the pattern is: compute dLdAttn into attn2, then softmaxBackward reads attn1+attn2
        // then overwrites attn2 with dLdLogits. Then we use dLdLogits (in attn2) for Q/K grads.
        // Then we recompute attn2 for Dir2.

        // dL/dAttn1 -> stored in attn2 temporarily
        dim3 gridAttn1(imageLen, tokenLen);
        bpDLdAttnKernel<T><<<gridAttn1, dotThreads, sharedSize, *stream>>>(
            gradTokenOut, imageV, attn2, tokenLen, imageLen, dim);
        DebugHelper::checkGlobalErrorCode("bpDLdAttn1 failed");

        // Softmax backward: attn2 = attn1 * (attn2 - rowDot) => dL/dLogits
        bpSoftmaxBackwardKernel<<<tokenLen, smThreads, smWarps * sizeof(float), *stream>>>(
            attn1, attn2, tokenLen, imageLen);
        DebugHelper::checkGlobalErrorCode("bpSoftmaxBack1 failed");

        // dL/dTokenQuery = dL/dLogits @ imageKey * scale
        LongType qElems = tokenLen * dim;
        int qThreads = 256;
        int qBlocks = (qElems + qThreads - 1) / qThreads;
        bpDLdQKernel<T><<<qBlocks, qThreads, 0, *stream>>>(
            attn2, imageK, dLdTokenQ, tokenLen, imageLen, dim, scale);
        DebugHelper::checkGlobalErrorCode("bpDLdTokenQ failed");

        // dL/dImageKey = dL/dLogits^T @ tokenQuery * scale
        LongType kElems = imageLen * dim;
        int kThreads = 256;
        int kBlocks = (kElems + kThreads - 1) / kThreads;
        bpDLdKKernel<T><<<kBlocks, kThreads, 0, *stream>>>(
            attn2, tokenQ, dLdImageK, tokenLen, imageLen, dim, scale);
        DebugHelper::checkGlobalErrorCode("bpDLdImageK failed");
    }

    // ---- Direction 2: imageOut = softmax(imageQ @ tokenK^T * scale) @ tokenV ----
    {
        // Recompute attention weights
        dim3 grid2(tokenLen, imageLen);
        bpCrossAttnLogitsKernel<T><<<grid2, dotThreads, sharedSize, *stream>>>(
            imageQ, tokenK, attn2, imageLen, tokenLen, dim, scale);
        DebugHelper::checkGlobalErrorCode("bpLogits2 failed");

        int smThreads = 256;
        if (tokenLen < 256) {
            smThreads = ((tokenLen + TWA_BP_WARP_SIZE - 1) / TWA_BP_WARP_SIZE) * TWA_BP_WARP_SIZE;
            if (smThreads < TWA_BP_WARP_SIZE) smThreads = TWA_BP_WARP_SIZE;
        }
        int smWarps = (smThreads + TWA_BP_WARP_SIZE - 1) / TWA_BP_WARP_SIZE;
        bpCrossAttnSoftmaxKernel<<<imageLen, smThreads, smWarps * sizeof(float), *stream>>>(
            attn2, imageLen, tokenLen);
        DebugHelper::checkGlobalErrorCode("bpSoftmax2 failed");

        // dL/dTokenValue = attn2^T @ gradImageOut
        LongType vElems = tokenLen * dim;
        int vThreads = 256;
        int vBlocks = (vElems + vThreads - 1) / vThreads;
        bpDLdVKernel<T><<<vBlocks, vThreads, 0, *stream>>>(
            attn2, gradImageOut, dLdTokenV, imageLen, tokenLen, dim);
        DebugHelper::checkGlobalErrorCode("bpDLdTokenV failed");

        // dL/dAttn2 -> stored in attn1 temporarily
        dim3 gridAttn2(tokenLen, imageLen);
        bpDLdAttnKernel<T><<<gridAttn2, dotThreads, sharedSize, *stream>>>(
            gradImageOut, tokenV, attn1, imageLen, tokenLen, dim);
        DebugHelper::checkGlobalErrorCode("bpDLdAttn2 failed");

        // Softmax backward
        bpSoftmaxBackwardKernel<<<imageLen, smThreads, smWarps * sizeof(float), *stream>>>(
            attn2, attn1, imageLen, tokenLen);
        DebugHelper::checkGlobalErrorCode("bpSoftmaxBack2 failed");

        // dL/dImageQuery = dL/dLogits @ tokenKey * scale
        LongType qElems = imageLen * dim;
        int qThreads = 256;
        int qBlocks = (qElems + qThreads - 1) / qThreads;
        bpDLdQKernel<T><<<qBlocks, qThreads, 0, *stream>>>(
            attn1, tokenK, dLdImageQ, imageLen, tokenLen, dim, scale);
        DebugHelper::checkGlobalErrorCode("bpDLdImageQ failed");

        // dL/dTokenKey = dL/dLogits^T @ imageQuery * scale
        LongType kElems = tokenLen * dim;
        int kThreads = 256;
        int kBlocks = (kElems + kThreads - 1) / kThreads;
        bpDLdKKernel<T><<<kBlocks, kThreads, 0, *stream>>>(
            attn1, imageQ, dLdTokenK, imageLen, tokenLen, dim, scale);
        DebugHelper::checkGlobalErrorCode("bpDLdTokenK failed");
    }
}

BUILD_SINGLE_TEMPLATE(void twoWayCrossAttentionBpCudaLauncher,
                      (const cudaStream_t* stream,
                       const void* vTokenQ, const void* vTokenK, const void* vTokenV,
                       const void* vImageQ, const void* vImageK, const void* vImageV,
                       const void* vGradTokenOut, const void* vGradImageOut,
                       void* vDLdTokenQ, void* vDLdTokenK, void* vDLdTokenV,
                       void* vDLdImageQ, void* vDLdImageK, void* vDLdImageV,
                       void* vAttn1, void* vAttn2,
                       LongType tokenLen, LongType imageLen, LongType dim, float scale),
                      SD_FLOAT_TYPES);

void twoWayCrossAttentionBp(
    NDArray* tokenQuery, NDArray* tokenKey, NDArray* tokenValue,
    NDArray* imageQuery, NDArray* imageKey, NDArray* imageValue,
    NDArray* gradTokenOut, NDArray* gradImageOut,
    NDArray* dLdTokenQuery, NDArray* dLdTokenKey, NDArray* dLdTokenValue,
    NDArray* dLdImageQuery, NDArray* dLdImageKey, NDArray* dLdImageValue,
    double scale, LaunchContext* context) {

    auto rank = tokenQuery->rankOf();
    auto stream = context->getCudaStream();

    NDArray::prepareSpecialUse(
        {dLdTokenQuery, dLdTokenKey, dLdTokenValue,
         dLdImageQuery, dLdImageKey, dLdImageValue},
        {tokenQuery, tokenKey, tokenValue,
         imageQuery, imageKey, imageValue,
         gradTokenOut, gradImageOut});

    if (rank == 2) {
        auto tokenLen = tokenQuery->sizeAt(0);
        auto imageLen = imageQuery->sizeAt(0);
        auto dim = tokenQuery->sizeAt(1);

        auto attn1 = NDArrayFactory::create('c', {tokenLen, imageLen}, DataType::FLOAT32, context);
        auto attn2 = NDArrayFactory::create('c', {imageLen, tokenLen}, DataType::FLOAT32, context);

        BUILD_SINGLE_SELECTOR(tokenQuery->dataType(), twoWayCrossAttentionBpCudaLauncher,
                              (stream,
                               tokenQuery->specialBuffer(), tokenKey->specialBuffer(),
                               tokenValue->specialBuffer(),
                               imageQuery->specialBuffer(), imageKey->specialBuffer(),
                               imageValue->specialBuffer(),
                               gradTokenOut->specialBuffer(), gradImageOut->specialBuffer(),
                               dLdTokenQuery->specialBuffer(), dLdTokenKey->specialBuffer(),
                               dLdTokenValue->specialBuffer(),
                               dLdImageQuery->specialBuffer(), dLdImageKey->specialBuffer(),
                               dLdImageValue->specialBuffer(),
                               attn1->specialBuffer(), attn2->specialBuffer(),
                               tokenLen, imageLen, dim, static_cast<float>(scale)),
                              SD_FLOAT_TYPES);

        delete attn1;
        delete attn2;
    } else {
        // rank 3: [batch, seq, dim] — process each batch element
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

            auto attn1 = NDArrayFactory::create('c', {tokenLen, imageLen}, DataType::FLOAT32, context);
            auto attn2 = NDArrayFactory::create('c', {imageLen, tokenLen}, DataType::FLOAT32, context);

            const void* tQBuf = static_cast<const int8_t*>(tokenQuery->specialBuffer()) + tokOff;
            const void* tKBuf = static_cast<const int8_t*>(tokenKey->specialBuffer()) + tokOff;
            const void* tVBuf = static_cast<const int8_t*>(tokenValue->specialBuffer()) + tokOff;
            const void* iQBuf = static_cast<const int8_t*>(imageQuery->specialBuffer()) + imgOff;
            const void* iKBuf = static_cast<const int8_t*>(imageKey->specialBuffer()) + imgOff;
            const void* iVBuf = static_cast<const int8_t*>(imageValue->specialBuffer()) + imgOff;
            const void* gTOBuf = static_cast<const int8_t*>(gradTokenOut->specialBuffer()) + tokOff;
            const void* gIOBuf = static_cast<const int8_t*>(gradImageOut->specialBuffer()) + imgOff;
            void* dTQBuf = static_cast<int8_t*>(dLdTokenQuery->specialBuffer()) + tokOff;
            void* dTKBuf = static_cast<int8_t*>(dLdTokenKey->specialBuffer()) + tokOff;
            void* dTVBuf = static_cast<int8_t*>(dLdTokenValue->specialBuffer()) + tokOff;
            void* dIQBuf = static_cast<int8_t*>(dLdImageQuery->specialBuffer()) + imgOff;
            void* dIKBuf = static_cast<int8_t*>(dLdImageKey->specialBuffer()) + imgOff;
            void* dIVBuf = static_cast<int8_t*>(dLdImageValue->specialBuffer()) + imgOff;

            BUILD_SINGLE_SELECTOR(dt, twoWayCrossAttentionBpCudaLauncher,
                                  (stream,
                                   tQBuf, tKBuf, tVBuf,
                                   iQBuf, iKBuf, iVBuf,
                                   gTOBuf, gIOBuf,
                                   dTQBuf, dTKBuf, dTVBuf,
                                   dIQBuf, dIKBuf, dIVBuf,
                                   attn1->specialBuffer(), attn2->specialBuffer(),
                                   tokenLen, imageLen, dim, static_cast<float>(scale)),
                                  SD_FLOAT_TYPES);

            delete attn1;
            delete attn2;
        }
    }

    NDArray::registerSpecialUse(
        {dLdTokenQuery, dLdTokenKey, dLdTokenValue,
         dLdImageQuery, dLdImageKey, dLdImageValue},
        {tokenQuery, tokenKey, tokenValue,
         imageQuery, imageKey, imageValue,
         gradTokenOut, gradImageOut});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
