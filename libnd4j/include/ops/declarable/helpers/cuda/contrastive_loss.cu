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

static constexpr int CL_WARP_SIZE = 32;

// Accumulator/scratch type: double when T=double for precision, float otherwise.
// The similarity/probability scratch buffers follow this type so double inputs
// are not silently narrowed to float.
template <typename T>
struct AccType { using type = float; };
template <>
struct AccType<double> { using type = double; };

// Kernel 1: Compute similarity matrix: sim[i][j] = dot(image[i], text[j]) * temperature
template <typename T>
SD_KERNEL void contrastiveSimKernel(
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

    const int lane = threadIdx.x % CL_WARP_SIZE;
    const int wid = threadIdx.x / CL_WARP_SIZE;
    const int numWarps = (blockDim.x + CL_WARP_SIZE - 1) / CL_WARP_SIZE;

    AccT threadDot = static_cast<AccT>(0);
    for (LongType k = threadIdx.x; k < embedDim; k += blockDim.x)
        threadDot += static_cast<AccT>(imageEmb[i * embedDim + k]) *
                     static_cast<AccT>(textEmb[j * embedDim + k]);

    AccT dot = sd::device::blockReduceSum(threadDot, warpBuf);

    if (threadIdx.x == 0)
        sim[i * batch + j] = dot * static_cast<AccT>(temperature);
}

// Kernel 2: Row-wise softmax + cross-entropy loss for one row
// Returns -log(softmax[target]) where target = row index (diagonal)
template <typename AccT>
SD_KERNEL void contrastiveRowCEKernel(
    const AccT* __restrict__ sim,
    AccT* __restrict__ rowLosses,
    const LongType batch) {

    const int i = blockIdx.x;
    if (i >= batch) return;

    extern __shared__ char sharedMem[];
    AccT* warpBuf = reinterpret_cast<AccT*>(sharedMem);

    // Find max
    AccT threadMax = -sd::DataTypeUtils::max<AccT>();
    for (LongType j = threadIdx.x; j < batch; j += blockDim.x)
        threadMax = sd::math::sd_max<AccT>(threadMax, sim[i * batch + j]);

    AccT rowMax = sd::device::blockAllReduceMax(threadMax, warpBuf);

    // Compute sum of exp
    AccT threadSum = static_cast<AccT>(0);
    for (LongType j = threadIdx.x; j < batch; j += blockDim.x)
        threadSum += sd::math::sd_exp<AccT, AccT>(sim[i * batch + j] - rowMax);

    AccT rowSum = sd::device::blockReduceSum(threadSum, warpBuf);

    // CE = -log(exp(sim[i,i] - max) / sum) = -(sim[i,i] - max) + log(sum)
    if (threadIdx.x == 0) {
        AccT logSum = sd::math::sd_log<AccT, AccT>(rowSum);
        rowLosses[i] = -(sim[i * batch + i] - rowMax) + logSum;
    }
}

// Kernel 3: Column-wise CE loss
template <typename AccT>
SD_KERNEL void contrastiveColCEKernel(
    const AccT* __restrict__ sim,
    AccT* __restrict__ colLosses,
    const LongType batch) {

    const int j = blockIdx.x;
    if (j >= batch) return;

    extern __shared__ char sharedMem[];
    AccT* warpBuf = reinterpret_cast<AccT*>(sharedMem);

    AccT threadMax = -sd::DataTypeUtils::max<AccT>();
    for (LongType i = threadIdx.x; i < batch; i += blockDim.x)
        threadMax = sd::math::sd_max<AccT>(threadMax, sim[i * batch + j]);

    AccT colMax = sd::device::blockAllReduceMax(threadMax, warpBuf);

    AccT threadSum = static_cast<AccT>(0);
    for (LongType i = threadIdx.x; i < batch; i += blockDim.x)
        threadSum += sd::math::sd_exp<AccT, AccT>(sim[i * batch + j] - colMax);

    AccT colSum = sd::device::blockReduceSum(threadSum, warpBuf);

    if (threadIdx.x == 0) {
        AccT logSum = sd::math::sd_log<AccT, AccT>(colSum);
        colLosses[j] = -(sim[j * batch + j] - colMax) + logSum;
    }
}

// Kernel 4: Sum losses and average
template <typename T>
SD_KERNEL void contrastiveSumLossKernel(
    const typename AccType<T>::type* __restrict__ rowLosses,
    const typename AccType<T>::type* __restrict__ colLosses,
    T* __restrict__ output,
    const LongType batch) {

    using AccT = typename AccType<T>::type;
    AccT total = static_cast<AccT>(0);
    for (LongType i = threadIdx.x; i < batch; i += blockDim.x) {
        total += rowLosses[i] + colLosses[i];
    }

    // Simple warp reduction
    total = sd::device::warpReduceSum(total);

    if (threadIdx.x == 0)
        output[0] = static_cast<T>(total / static_cast<AccT>(2 * batch));
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
    using AccT = typename AccType<T>::type;
    auto sim = reinterpret_cast<AccT*>(vSim);
    auto rowLosses = reinterpret_cast<AccT*>(vRowLosses);
    auto colLosses = reinterpret_cast<AccT*>(vColLosses);
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
        contrastiveSimKernel<T><<<grid, simThreads, numWarps * sizeof(AccT), *stream>>>(
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
        size_t sharedSize = numWarps * sizeof(AccT);

        contrastiveRowCEKernel<AccT><<<batch, smThreads, sharedSize, *stream>>>(
            sim, rowLosses, batch);
        DebugHelper::checkGlobalErrorCode("contrastiveRowCE failed");

        contrastiveColCEKernel<AccT><<<batch, smThreads, sharedSize, *stream>>>(
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

    auto accDtype = imageEmbeddings->dataType() == DataType::DOUBLE ? DataType::DOUBLE : DataType::FLOAT32;
    auto sim = NDArrayFactory::create('c', {batch, batch}, accDtype, context);
    auto rowLosses = NDArrayFactory::create('c', {batch}, accDtype, context);
    auto colLosses = NDArrayFactory::create('c', {batch}, accDtype, context);

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
