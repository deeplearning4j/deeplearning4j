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

#include <ops/declarable/helpers/contrastive_loss.h>
#include <array/NDArray.h>
#include <array/NDArrayFactory.h>
#include <helpers/DebugHelper.h>
#include <math/templatemath.h>
#include <cuda_runtime.h>
#include <cfloat>

namespace sd {
namespace ops {
namespace helpers {

static constexpr int CL_BP_WARP_SIZE = 32;

// Kernel 1: Similarity matrix
template <typename T>
__global__ void contrastiveBpSimKernel(
    const T* __restrict__ imageEmb,
    const T* __restrict__ textEmb,
    float* __restrict__ sim,
    const LongType batch,
    const LongType embedDim,
    const float temperature) {

    const int i = blockIdx.y;
    const int j = blockIdx.x;
    if (i >= batch || j >= batch) return;

    extern __shared__ char sharedMem[];
    float* warpBuf = reinterpret_cast<float*>(sharedMem);

    const int lane = threadIdx.x % CL_BP_WARP_SIZE;
    const int wid = threadIdx.x / CL_BP_WARP_SIZE;
    const int numWarps = (blockDim.x + CL_BP_WARP_SIZE - 1) / CL_BP_WARP_SIZE;

    float threadDot = 0.0f;
    for (LongType k = threadIdx.x; k < embedDim; k += blockDim.x)
        threadDot += static_cast<float>(imageEmb[i * embedDim + k]) *
                     static_cast<float>(textEmb[j * embedDim + k]);

    for (int offset = CL_BP_WARP_SIZE / 2; offset > 0; offset /= 2)
        threadDot += __shfl_down_sync(0xffffffff, threadDot, offset);
    if (lane == 0) warpBuf[wid] = threadDot;
    __syncthreads();

    float dot = 0.0f;
    if (threadIdx.x < numWarps) dot = warpBuf[threadIdx.x];
    for (int offset = CL_BP_WARP_SIZE / 2; offset > 0; offset /= 2)
        dot += __shfl_down_sync(0xffffffff, dot, offset);

    if (threadIdx.x == 0)
        sim[i * batch + j] = dot * temperature;
}

// Kernel 2: Row-wise softmax
__global__ void softmaxRowKernel(
    const float* __restrict__ sim,
    float* __restrict__ prob,
    const LongType batch) {

    const int i = blockIdx.x;
    if (i >= batch) return;

    extern __shared__ char sharedMem[];
    float* warpBuf = reinterpret_cast<float*>(sharedMem);

    const int lane = threadIdx.x % CL_BP_WARP_SIZE;
    const int wid = threadIdx.x / CL_BP_WARP_SIZE;
    const int numWarps = (blockDim.x + CL_BP_WARP_SIZE - 1) / CL_BP_WARP_SIZE;

    float threadMax = -FLT_MAX;
    for (LongType j = threadIdx.x; j < batch; j += blockDim.x)
        threadMax = sd::math::sd_max<float>(threadMax, sim[i * batch + j]);

    for (int offset = CL_BP_WARP_SIZE / 2; offset > 0; offset /= 2)
        threadMax = sd::math::sd_max<float>(threadMax, __shfl_down_sync(0xffffffff, threadMax, offset));
    if (lane == 0) warpBuf[wid] = threadMax;
    __syncthreads();

    float rowMax = -FLT_MAX;
    if (threadIdx.x < numWarps) rowMax = warpBuf[threadIdx.x];
    for (int offset = CL_BP_WARP_SIZE / 2; offset > 0; offset /= 2)
        rowMax = sd::math::sd_max<float>(rowMax, __shfl_down_sync(0xffffffff, rowMax, offset));
    __shared__ float sharedMax;
    if (threadIdx.x == 0) sharedMax = rowMax;
    __syncthreads();
    rowMax = sharedMax;

    float threadSum = 0.0f;
    for (LongType j = threadIdx.x; j < batch; j += blockDim.x) {
        float val = sd::math::sd_exp<float, float>(sim[i * batch + j] - rowMax);
        prob[i * batch + j] = val;
        threadSum += val;
    }

    for (int offset = CL_BP_WARP_SIZE / 2; offset > 0; offset /= 2)
        threadSum += __shfl_down_sync(0xffffffff, threadSum, offset);
    if (lane == 0) warpBuf[wid] = threadSum;
    __syncthreads();

    float rowSum = 0.0f;
    if (threadIdx.x < numWarps) rowSum = warpBuf[threadIdx.x];
    for (int offset = CL_BP_WARP_SIZE / 2; offset > 0; offset /= 2)
        rowSum += __shfl_down_sync(0xffffffff, rowSum, offset);
    __shared__ float sharedInvSum;
    if (threadIdx.x == 0) sharedInvSum = 1.0f / rowSum;
    __syncthreads();

    for (LongType j = threadIdx.x; j < batch; j += blockDim.x)
        prob[i * batch + j] *= sharedInvSum;
}

// Kernel 3: Column-wise softmax
__global__ void softmaxColKernel(
    const float* __restrict__ sim,
    float* __restrict__ prob,
    const LongType batch) {

    const int j = blockIdx.x;
    if (j >= batch) return;

    extern __shared__ char sharedMem[];
    float* warpBuf = reinterpret_cast<float*>(sharedMem);

    const int lane = threadIdx.x % CL_BP_WARP_SIZE;
    const int wid = threadIdx.x / CL_BP_WARP_SIZE;
    const int numWarps = (blockDim.x + CL_BP_WARP_SIZE - 1) / CL_BP_WARP_SIZE;

    float threadMax = -FLT_MAX;
    for (LongType i = threadIdx.x; i < batch; i += blockDim.x)
        threadMax = sd::math::sd_max<float>(threadMax, sim[i * batch + j]);

    for (int offset = CL_BP_WARP_SIZE / 2; offset > 0; offset /= 2)
        threadMax = sd::math::sd_max<float>(threadMax, __shfl_down_sync(0xffffffff, threadMax, offset));
    if (lane == 0) warpBuf[wid] = threadMax;
    __syncthreads();

    float colMax = -FLT_MAX;
    if (threadIdx.x < numWarps) colMax = warpBuf[threadIdx.x];
    for (int offset = CL_BP_WARP_SIZE / 2; offset > 0; offset /= 2)
        colMax = sd::math::sd_max<float>(colMax, __shfl_down_sync(0xffffffff, colMax, offset));
    __shared__ float sharedMax;
    if (threadIdx.x == 0) sharedMax = colMax;
    __syncthreads();
    colMax = sharedMax;

    float threadSum = 0.0f;
    for (LongType i = threadIdx.x; i < batch; i += blockDim.x) {
        float val = sd::math::sd_exp<float, float>(sim[i * batch + j] - colMax);
        prob[j * batch + i] = val;
        threadSum += val;
    }

    for (int offset = CL_BP_WARP_SIZE / 2; offset > 0; offset /= 2)
        threadSum += __shfl_down_sync(0xffffffff, threadSum, offset);
    if (lane == 0) warpBuf[wid] = threadSum;
    __syncthreads();

    float colSum = 0.0f;
    if (threadIdx.x < numWarps) colSum = warpBuf[threadIdx.x];
    for (int offset = CL_BP_WARP_SIZE / 2; offset > 0; offset /= 2)
        colSum += __shfl_down_sync(0xffffffff, colSum, offset);
    __shared__ float sharedInvSum;
    if (threadIdx.x == 0) sharedInvSum = 1.0f / colSum;
    __syncthreads();

    for (LongType i = threadIdx.x; i < batch; i += blockDim.x)
        prob[j * batch + i] *= sharedInvSum;
}

// Kernel 4: Compute embedding gradients
template <typename T>
__global__ void contrastiveGradKernel(
    const float* __restrict__ probRow,
    const float* __restrict__ probCol,
    const T* __restrict__ otherEmb,
    T* __restrict__ dLdEmb,
    const LongType batch,
    const LongType embedDim,
    const float temperature,
    const float invTwoBatch,
    const bool isImageGrad) {

    const LongType total = batch * embedDim;
    for (LongType idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += blockDim.x * gridDim.x) {

        const int i = idx / embedDim;
        const int k = idx % embedDim;

        float grad = 0.0f;
        for (LongType j = 0; j < batch; j++) {
            float rowG, colG;
            if (isImageGrad) {
                rowG = probRow[i * batch + j] - (i == j ? 1.0f : 0.0f);
                colG = probCol[j * batch + i] - (i == j ? 1.0f : 0.0f);
            } else {
                rowG = probRow[j * batch + i] - (i == j ? 1.0f : 0.0f);
                colG = probCol[i * batch + j] - (i == j ? 1.0f : 0.0f);
            }
            float gradSim = (rowG + colG) * invTwoBatch;
            grad += gradSim * static_cast<float>(otherEmb[j * embedDim + k]);
        }
        dLdEmb[idx] = static_cast<T>(grad * temperature);
    }
}

template <typename T>
void contrastiveLossBpCudaLauncher(const cudaStream_t* stream,
                                    const void* vImageEmb, const void* vTextEmb,
                                    void* vDLdImage, void* vDLdText,
                                    void* vSim, void* vProbRow, void* vProbCol,
                                    LongType batch, LongType embedDim,
                                    float temperature) {
    auto imageEmb = reinterpret_cast<const T*>(vImageEmb);
    auto textEmb = reinterpret_cast<const T*>(vTextEmb);
    auto dLdImage = reinterpret_cast<T*>(vDLdImage);
    auto dLdText = reinterpret_cast<T*>(vDLdText);
    auto sim = reinterpret_cast<float*>(vSim);
    auto probRow = reinterpret_cast<float*>(vProbRow);
    auto probCol = reinterpret_cast<float*>(vProbCol);

    float invTwoBatch = 1.0f / (2.0f * batch);

    // Similarity matrix
    {
        int simThreads = 256;
        if (embedDim < 256) {
            simThreads = ((embedDim + CL_BP_WARP_SIZE - 1) / CL_BP_WARP_SIZE) * CL_BP_WARP_SIZE;
            if (simThreads < CL_BP_WARP_SIZE) simThreads = CL_BP_WARP_SIZE;
        }
        int numWarps = (simThreads + CL_BP_WARP_SIZE - 1) / CL_BP_WARP_SIZE;
        dim3 grid(batch, batch);
        contrastiveBpSimKernel<T><<<grid, simThreads, numWarps * sizeof(float), *stream>>>(
            imageEmb, textEmb, sim, batch, embedDim, temperature);
        DebugHelper::checkGlobalErrorCode("contrastiveBpSim failed");
    }

    // Softmax row and column
    {
        int smThreads = 256;
        if (batch < 256) {
            smThreads = ((batch + CL_BP_WARP_SIZE - 1) / CL_BP_WARP_SIZE) * CL_BP_WARP_SIZE;
            if (smThreads < CL_BP_WARP_SIZE) smThreads = CL_BP_WARP_SIZE;
        }
        int numWarps = (smThreads + CL_BP_WARP_SIZE - 1) / CL_BP_WARP_SIZE;
        size_t sharedSize = numWarps * sizeof(float);

        softmaxRowKernel<<<batch, smThreads, sharedSize, *stream>>>(sim, probRow, batch);
        DebugHelper::checkGlobalErrorCode("softmaxRow failed");

        softmaxColKernel<<<batch, smThreads, sharedSize, *stream>>>(sim, probCol, batch);
        DebugHelper::checkGlobalErrorCode("softmaxCol failed");
    }

    // Gradients
    {
        LongType totalElems = batch * embedDim;
        int threads = 256;
        int blocks = (totalElems + threads - 1) / threads;

        contrastiveGradKernel<T><<<blocks, threads, 0, *stream>>>(
            probRow, probCol, textEmb, dLdImage,
            batch, embedDim, temperature, invTwoBatch, true);
        DebugHelper::checkGlobalErrorCode("contrastiveGradImage failed");

        contrastiveGradKernel<T><<<blocks, threads, 0, *stream>>>(
            probRow, probCol, imageEmb, dLdText,
            batch, embedDim, temperature, invTwoBatch, false);
        DebugHelper::checkGlobalErrorCode("contrastiveGradText failed");
    }
}

BUILD_SINGLE_TEMPLATE(void contrastiveLossBpCudaLauncher,
                     (const cudaStream_t* stream,
                      const void* vImageEmb, const void* vTextEmb,
                      void* vDLdImage, void* vDLdText,
                      void* vSim, void* vProbRow, void* vProbCol,
                      LongType batch, LongType embedDim, float temperature),
                     SD_FLOAT_TYPES);

void contrastiveLossBp(NDArray* imageEmbeddings, NDArray* textEmbeddings,
                        NDArray* dLdImage, NDArray* dLdText,
                        double temperature, LaunchContext* context) {
    auto batch = imageEmbeddings->sizeAt(0);
    auto embedDim = imageEmbeddings->sizeAt(1);
    auto stream = context->getCudaStream();

    auto sim = NDArrayFactory::create('c', {batch, batch}, DataType::FLOAT32, context);
    auto probRow = NDArrayFactory::create('c', {batch, batch}, DataType::FLOAT32, context);
    auto probCol = NDArrayFactory::create('c', {batch, batch}, DataType::FLOAT32, context);

    NDArray::prepareSpecialUse({dLdImage, dLdText}, {imageEmbeddings, textEmbeddings});

    BUILD_SINGLE_SELECTOR(imageEmbeddings->dataType(), contrastiveLossBpCudaLauncher,
                         (stream,
                          imageEmbeddings->specialBuffer(), textEmbeddings->specialBuffer(),
                          dLdImage->specialBuffer(), dLdText->specialBuffer(),
                          sim->specialBuffer(), probRow->specialBuffer(), probCol->specialBuffer(),
                          batch, embedDim, static_cast<float>(temperature)),
                         SD_FLOAT_TYPES);

    NDArray::registerSpecialUse({dLdImage, dLdText}, {imageEmbeddings, textEmbeddings});

    delete sim;
    delete probRow;
    delete probCol;
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
