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

static constexpr int CL_WARP_SIZE = 32;

// Kernel 1: Compute similarity matrix: sim[i][j] = dot(image[i], text[j]) * temperature
template <typename T>
SD_KERNEL void contrastiveSimKernel(
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

    const int lane = threadIdx.x % CL_WARP_SIZE;
    const int wid = threadIdx.x / CL_WARP_SIZE;
    const int numWarps = (blockDim.x + CL_WARP_SIZE - 1) / CL_WARP_SIZE;

    float threadDot = 0.0f;
    for (LongType k = threadIdx.x; k < embedDim; k += blockDim.x)
        threadDot += static_cast<float>(imageEmb[i * embedDim + k]) *
                     static_cast<float>(textEmb[j * embedDim + k]);

    for (int offset = CL_WARP_SIZE / 2; offset > 0; offset /= 2)
        threadDot += __shfl_down_sync(0xffffffff, threadDot, offset);
    if (lane == 0) warpBuf[wid] = threadDot;
    __syncthreads();

    float dot = 0.0f;
    if (threadIdx.x < numWarps) dot = warpBuf[threadIdx.x];
    for (int offset = CL_WARP_SIZE / 2; offset > 0; offset /= 2)
        dot += __shfl_down_sync(0xffffffff, dot, offset);

    if (threadIdx.x == 0)
        sim[i * batch + j] = dot * temperature;
}

// Kernel 2: Row-wise softmax + cross-entropy loss for one row
// Returns -log(softmax[target]) where target = row index (diagonal)
SD_KERNEL void contrastiveRowCEKernel(
    const float* __restrict__ sim,
    float* __restrict__ rowLosses,
    const LongType batch) {

    const int i = blockIdx.x;
    if (i >= batch) return;

    extern __shared__ char sharedMem[];
    float* warpBuf = reinterpret_cast<float*>(sharedMem);

    const int lane = threadIdx.x % CL_WARP_SIZE;
    const int wid = threadIdx.x / CL_WARP_SIZE;
    const int numWarps = (blockDim.x + CL_WARP_SIZE - 1) / CL_WARP_SIZE;

    // Find max
    float threadMax = -FLT_MAX;
    for (LongType j = threadIdx.x; j < batch; j += blockDim.x)
        threadMax = sd::math::sd_max<float>(threadMax, sim[i * batch + j]);

    for (int offset = CL_WARP_SIZE / 2; offset > 0; offset /= 2)
        threadMax = sd::math::sd_max<float>(threadMax, __shfl_down_sync(0xffffffff, threadMax, offset));
    if (lane == 0) warpBuf[wid] = threadMax;
    __syncthreads();

    float rowMax = -FLT_MAX;
    if (threadIdx.x < numWarps) rowMax = warpBuf[threadIdx.x];
    for (int offset = CL_WARP_SIZE / 2; offset > 0; offset /= 2)
        rowMax = sd::math::sd_max<float>(rowMax, __shfl_down_sync(0xffffffff, rowMax, offset));
    __shared__ float sharedMax;
    if (threadIdx.x == 0) sharedMax = rowMax;
    __syncthreads();
    rowMax = sharedMax;

    // Compute sum of exp
    float threadSum = 0.0f;
    for (LongType j = threadIdx.x; j < batch; j += blockDim.x)
        threadSum += sd::math::sd_exp<float, float>(sim[i * batch + j] - rowMax);

    for (int offset = CL_WARP_SIZE / 2; offset > 0; offset /= 2)
        threadSum += __shfl_down_sync(0xffffffff, threadSum, offset);
    if (lane == 0) warpBuf[wid] = threadSum;
    __syncthreads();

    float rowSum = 0.0f;
    if (threadIdx.x < numWarps) rowSum = warpBuf[threadIdx.x];
    for (int offset = CL_WARP_SIZE / 2; offset > 0; offset /= 2)
        rowSum += __shfl_down_sync(0xffffffff, rowSum, offset);

    // CE = -log(exp(sim[i,i] - max) / sum) = -(sim[i,i] - max) + log(sum)
    if (threadIdx.x == 0) {
        float logSum = logf(rowSum);
        rowLosses[i] = -(sim[i * batch + i] - rowMax) + logSum;
    }
}

// Kernel 3: Column-wise CE loss
SD_KERNEL void contrastiveColCEKernel(
    const float* __restrict__ sim,
    float* __restrict__ colLosses,
    const LongType batch) {

    const int j = blockIdx.x;
    if (j >= batch) return;

    extern __shared__ char sharedMem[];
    float* warpBuf = reinterpret_cast<float*>(sharedMem);

    const int lane = threadIdx.x % CL_WARP_SIZE;
    const int wid = threadIdx.x / CL_WARP_SIZE;
    const int numWarps = (blockDim.x + CL_WARP_SIZE - 1) / CL_WARP_SIZE;

    float threadMax = -FLT_MAX;
    for (LongType i = threadIdx.x; i < batch; i += blockDim.x)
        threadMax = sd::math::sd_max<float>(threadMax, sim[i * batch + j]);

    for (int offset = CL_WARP_SIZE / 2; offset > 0; offset /= 2)
        threadMax = sd::math::sd_max<float>(threadMax, __shfl_down_sync(0xffffffff, threadMax, offset));
    if (lane == 0) warpBuf[wid] = threadMax;
    __syncthreads();

    float colMax = -FLT_MAX;
    if (threadIdx.x < numWarps) colMax = warpBuf[threadIdx.x];
    for (int offset = CL_WARP_SIZE / 2; offset > 0; offset /= 2)
        colMax = sd::math::sd_max<float>(colMax, __shfl_down_sync(0xffffffff, colMax, offset));
    __shared__ float sharedMax;
    if (threadIdx.x == 0) sharedMax = colMax;
    __syncthreads();
    colMax = sharedMax;

    float threadSum = 0.0f;
    for (LongType i = threadIdx.x; i < batch; i += blockDim.x)
        threadSum += sd::math::sd_exp<float, float>(sim[i * batch + j] - colMax);

    for (int offset = CL_WARP_SIZE / 2; offset > 0; offset /= 2)
        threadSum += __shfl_down_sync(0xffffffff, threadSum, offset);
    if (lane == 0) warpBuf[wid] = threadSum;
    __syncthreads();

    float colSum = 0.0f;
    if (threadIdx.x < numWarps) colSum = warpBuf[threadIdx.x];
    for (int offset = CL_WARP_SIZE / 2; offset > 0; offset /= 2)
        colSum += __shfl_down_sync(0xffffffff, colSum, offset);

    if (threadIdx.x == 0) {
        float logSum = logf(colSum);
        colLosses[j] = -(sim[j * batch + j] - colMax) + logSum;
    }
}

// Kernel 4: Sum losses and average
template <typename T>
SD_KERNEL void contrastiveSumLossKernel(
    const float* __restrict__ rowLosses,
    const float* __restrict__ colLosses,
    T* __restrict__ output,
    const LongType batch) {

    float total = 0.0f;
    for (LongType i = threadIdx.x; i < batch; i += blockDim.x) {
        total += rowLosses[i] + colLosses[i];
    }

    // Simple warp reduction
    for (int offset = CL_WARP_SIZE / 2; offset > 0; offset /= 2)
        total += __shfl_down_sync(0xffffffff, total, offset);

    if (threadIdx.x == 0)
        output[0] = static_cast<T>(total / (2.0f * batch));
}

template <typename T>
void contrastiveLossCudaLauncher(const cudaStream_t* stream,
                                  const void* vImageEmb, const void* vTextEmb,
                                  void* vOutput, void* vSim,
                                  void* vRowLosses, void* vColLosses,
                                  LongType batch, LongType embedDim,
                                  float temperature) {
    auto imageEmb = reinterpret_cast<const T*>(vImageEmb);
    auto textEmb = reinterpret_cast<const T*>(vTextEmb);
    auto sim = reinterpret_cast<float*>(vSim);
    auto rowLosses = reinterpret_cast<float*>(vRowLosses);
    auto colLosses = reinterpret_cast<float*>(vColLosses);
    auto output = reinterpret_cast<T*>(vOutput);

    // Similarity matrix
    {
        int simThreads = 256;
        if (embedDim < 256) {
            simThreads = ((embedDim + CL_WARP_SIZE - 1) / CL_WARP_SIZE) * CL_WARP_SIZE;
            if (simThreads < CL_WARP_SIZE) simThreads = CL_WARP_SIZE;
        }
        int numWarps = (simThreads + CL_WARP_SIZE - 1) / CL_WARP_SIZE;
        dim3 grid(batch, batch);
        contrastiveSimKernel<T><<<grid, simThreads, numWarps * sizeof(float), *stream>>>(
            imageEmb, textEmb, sim, batch, embedDim, temperature);
        DebugHelper::checkGlobalErrorCode("contrastiveSim failed");
    }

    // Row-wise and column-wise CE
    {
        int smThreads = 256;
        if (batch < 256) {
            smThreads = ((batch + CL_WARP_SIZE - 1) / CL_WARP_SIZE) * CL_WARP_SIZE;
            if (smThreads < CL_WARP_SIZE) smThreads = CL_WARP_SIZE;
        }
        int numWarps = (smThreads + CL_WARP_SIZE - 1) / CL_WARP_SIZE;
        size_t sharedSize = numWarps * sizeof(float);

        contrastiveRowCEKernel<<<batch, smThreads, sharedSize, *stream>>>(
            sim, rowLosses, batch);
        DebugHelper::checkGlobalErrorCode("contrastiveRowCE failed");

        contrastiveColCEKernel<<<batch, smThreads, sharedSize, *stream>>>(
            sim, colLosses, batch);
        DebugHelper::checkGlobalErrorCode("contrastiveColCE failed");
    }

    // Sum and average
    contrastiveSumLossKernel<T><<<1, CL_WARP_SIZE, 0, *stream>>>(
        rowLosses, colLosses, output, batch);
    DebugHelper::checkGlobalErrorCode("contrastiveSumLoss failed");
}

BUILD_SINGLE_TEMPLATE(void contrastiveLossCudaLauncher,
                      (const cudaStream_t* stream,
                       const void* vImageEmb, const void* vTextEmb,
                       void* vOutput, void* vSim,
                       void* vRowLosses, void* vColLosses,
                       LongType batch, LongType embedDim, float temperature),
                      SD_FLOAT_TYPES);

void contrastiveLoss(NDArray* imageEmbeddings, NDArray* textEmbeddings,
                      NDArray* output, double temperature,
                      LaunchContext* context) {
    auto batch = imageEmbeddings->sizeAt(0);
    auto embedDim = imageEmbeddings->sizeAt(1);
    auto stream = context->getCudaStream();

    auto sim = NDArrayFactory::create('c', {batch, batch}, DataType::FLOAT32, context);
    auto rowLosses = NDArrayFactory::create('c', {batch}, DataType::FLOAT32, context);
    auto colLosses = NDArrayFactory::create('c', {batch}, DataType::FLOAT32, context);

    NDArray::prepareSpecialUse({output}, {imageEmbeddings, textEmbeddings});

    BUILD_SINGLE_SELECTOR(imageEmbeddings->dataType(), contrastiveLossCudaLauncher,
                          (stream,
                           imageEmbeddings->specialBuffer(), textEmbeddings->specialBuffer(),
                           output->specialBuffer(), sim->specialBuffer(),
                           rowLosses->specialBuffer(), colLosses->specialBuffer(),
                           batch, embedDim, static_cast<float>(temperature)),
                          SD_FLOAT_TYPES);

    NDArray::registerSpecialUse({output}, {imageEmbeddings, textEmbeddings});

    delete sim;
    delete rowLosses;
    delete colLosses;
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
