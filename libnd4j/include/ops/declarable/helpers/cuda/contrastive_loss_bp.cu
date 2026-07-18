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
#include <ops/declarable/helpers/cuda/device_primitives.cuh>

namespace sd {
namespace ops {
namespace helpers {

static constexpr int CL_BP_WARP_SIZE = 32;

// Accumulator/scratch type: double when T=double for precision, float otherwise.
template <typename T>
struct AccType { using type = float; };
template <>
struct AccType<double> { using type = double; };

// Kernel 1: Similarity matrix
template <typename T>
SD_KERNEL void contrastiveBpSimKernel(
    const T* __restrict__ imageEmb,
    const T* __restrict__ textEmb,
    typename AccType<T>::type* __restrict__ sim,
    const LongType batch,
    const LongType embedDim,
    const float temperature) {

    using AccT = typename AccType<T>::type;

    const int i = blockIdx.y;
    const int j = blockIdx.x;
    if (i >= batch || j >= batch) return;

    extern __shared__ char sharedMem[];
    AccT* warpBuf = reinterpret_cast<AccT*>(sharedMem);

    const int lane = threadIdx.x % CL_BP_WARP_SIZE;
    const int wid = threadIdx.x / CL_BP_WARP_SIZE;
    const int numWarps = (blockDim.x + CL_BP_WARP_SIZE - 1) / CL_BP_WARP_SIZE;

    AccT threadDot = static_cast<AccT>(0);
    for (LongType k = threadIdx.x; k < embedDim; k += blockDim.x)
        threadDot += static_cast<AccT>(imageEmb[i * embedDim + k]) *
                     static_cast<AccT>(textEmb[j * embedDim + k]);

    AccT dot = sd::device::blockReduceSum(threadDot, warpBuf);

    if (threadIdx.x == 0)
        sim[i * batch + j] = dot * static_cast<AccT>(temperature);
}

// Kernel 2: Row-wise softmax
template <typename AccT>
SD_KERNEL void softmaxRowKernel(
    const AccT* __restrict__ sim,
    AccT* __restrict__ prob,
    const LongType batch) {

    const int i = blockIdx.x;
    if (i >= batch) return;

    extern __shared__ char sharedMem[];
    AccT* warpBuf = reinterpret_cast<AccT*>(sharedMem);

    const int lane = threadIdx.x % CL_BP_WARP_SIZE;
    const int wid = threadIdx.x / CL_BP_WARP_SIZE;
    const int numWarps = (blockDim.x + CL_BP_WARP_SIZE - 1) / CL_BP_WARP_SIZE;

    AccT threadMax = -sd::DataTypeUtils::max<AccT>();
    for (LongType j = threadIdx.x; j < batch; j += blockDim.x)
        threadMax = sd::math::sd_max<AccT>(threadMax, sim[i * batch + j]);

    AccT rowMax = sd::device::blockAllReduceMax(threadMax, warpBuf);

    AccT threadSum = static_cast<AccT>(0);
    for (LongType j = threadIdx.x; j < batch; j += blockDim.x) {
        AccT val = sd::math::sd_exp<AccT, AccT>(sim[i * batch + j] -rowMax);
        prob[i * batch + j] = val;
        threadSum += val;
    }

    AccT rowSum = sd::device::blockAllReduceSum(threadSum, warpBuf);
    AccT sharedInvSum = static_cast<AccT>(1) / rowSum;

    for (LongType j = threadIdx.x; j < batch; j += blockDim.x)
        prob[i * batch + j] *= sharedInvSum;
}

// Kernel 3: Column-wise softmax
template <typename AccT>
SD_KERNEL void softmaxColKernel(
    const AccT* __restrict__ sim,
    AccT* __restrict__ prob,
    const LongType batch) {

    const int j = blockIdx.x;
    if (j >= batch) return;

    extern __shared__ char sharedMem[];
    AccT* warpBuf = reinterpret_cast<AccT*>(sharedMem);

    const int lane = threadIdx.x % CL_BP_WARP_SIZE;
    const int wid = threadIdx.x / CL_BP_WARP_SIZE;
    const int numWarps = (blockDim.x + CL_BP_WARP_SIZE - 1) / CL_BP_WARP_SIZE;

    AccT threadMax = -sd::DataTypeUtils::max<AccT>();
    for (LongType i = threadIdx.x; i < batch; i += blockDim.x)
        threadMax = sd::math::sd_max<AccT>(threadMax, sim[i * batch + j]);

    AccT colMax = sd::device::blockAllReduceMax(threadMax, warpBuf);

    AccT threadSum = static_cast<AccT>(0);
    for (LongType i = threadIdx.x; i < batch; i += blockDim.x) {
        AccT val = sd::math::sd_exp<AccT, AccT>(sim[i * batch + j] -colMax);
        prob[j * batch + i] = val;
        threadSum += val;
    }

    AccT colSum = sd::device::blockAllReduceSum(threadSum, warpBuf);
    AccT sharedInvSum = static_cast<AccT>(1) / colSum;

    for (LongType i = threadIdx.x; i < batch; i += blockDim.x)
        prob[j * batch + i] *= sharedInvSum;
}

// Kernel 4: Compute embedding gradients
template <typename T>
SD_KERNEL void contrastiveGradKernel(
    const typename AccType<T>::type* __restrict__ probRow,
    const typename AccType<T>::type* __restrict__ probCol,
    const T* __restrict__ otherEmb,
    T* __restrict__ dLdEmb,
    const LongType batch,
    const LongType embedDim,
    const float temperature,
    const float invTwoBatch,
    const bool isImageGrad) {

    using AccT = typename AccType<T>::type;

    const LongType total = batch * embedDim;
    for (LongType idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += blockDim.x * gridDim.x) {

        const int i = idx / embedDim;
        const int k = idx % embedDim;

        AccT grad = static_cast<AccT>(0);
        for (LongType j = 0; j < batch; j++) {
            AccT rowG, colG;
            if (isImageGrad) {
                rowG = probRow[i * batch + j] - (i == j ? static_cast<AccT>(1) : static_cast<AccT>(0));
                colG = probCol[j * batch + i] - (i == j ? static_cast<AccT>(1) : static_cast<AccT>(0));
            } else {
                rowG = probRow[j * batch + i] - (i == j ? static_cast<AccT>(1) : static_cast<AccT>(0));
                colG = probCol[i * batch + j] - (i == j ? static_cast<AccT>(1) : static_cast<AccT>(0));
            }
            AccT gradSim = (rowG + colG) * static_cast<AccT>(invTwoBatch);
            grad += gradSim * static_cast<AccT>(otherEmb[j * embedDim + k]);
        }
        dLdEmb[idx] = static_cast<T>(grad * static_cast<AccT>(temperature));
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
    using AccT = typename AccType<T>::type;
    auto sim = reinterpret_cast<AccT*>(vSim);
    auto probRow = reinterpret_cast<AccT*>(vProbRow);
    auto probCol = reinterpret_cast<AccT*>(vProbCol);

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
        contrastiveBpSimKernel<T><<<grid, simThreads, numWarps * sizeof(AccT), *stream>>>(
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
        size_t sharedSize = numWarps * sizeof(AccT);

        softmaxRowKernel<AccT><<<batch, smThreads, sharedSize, *stream>>>(sim, probRow, batch);
        DebugHelper::checkGlobalErrorCode("softmaxRow failed");

        softmaxColKernel<AccT><<<batch, smThreads, sharedSize, *stream>>>(sim, probCol, batch);
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

    auto accDtype = imageEmbeddings->dataType() == DataType::DOUBLE ? DataType::DOUBLE : DataType::FLOAT32;
    auto sim = NDArrayFactory::create('c', {batch, batch}, accDtype, context);
    auto probRow = NDArrayFactory::create('c', {batch, batch}, accDtype, context);
    auto probCol = NDArrayFactory::create('c', {batch, batch}, accDtype, context);

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
