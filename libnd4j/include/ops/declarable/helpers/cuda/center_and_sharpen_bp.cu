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

#include <ops/declarable/helpers/center_and_sharpen.h>
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

static constexpr int CS_BP_WARP_SIZE = 32;

// Accumulator type: double when T=double for precision, float otherwise.
template <typename T>
struct AccType { using type = float; };
template <>
struct AccType<double> { using type = double; };

// Phase 1: Compute softmax output (same as forward)
template <typename T>
SD_KERNEL void csBpSoftmaxKernel(const T* __restrict__ input,
                                   const T* __restrict__ center,
                                   T* __restrict__ softmaxOut,
                                   const LongType batch,
                                   const LongType dim,
                                   const T invTemp) {
    using AccT = typename AccType<T>::type;

    extern __shared__ char sharedMem[];
    AccT* warpBuf = reinterpret_cast<AccT*>(sharedMem);

    const int b = blockIdx.x;
    if (b >= batch) return;

    const T* inRow = input + b * dim;
    T* outRow = softmaxOut + b * dim;

    // Center, scale, find max
    AccT threadMax = -sd::DataTypeUtils::max<AccT>();
    for (LongType f = threadIdx.x; f < dim; f += blockDim.x) {
        T val = (inRow[f] - center[f]) * invTemp;
        outRow[f] = val;
        threadMax = sd::math::sd_max<AccT>(threadMax, static_cast<AccT>(val));
    }

    AccT rowMax = sd::device::blockAllReduceMax(threadMax, warpBuf);

    // exp and sum
    AccT threadSum = static_cast<AccT>(0);
    for (LongType f = threadIdx.x; f < dim; f += blockDim.x) {
        T val = sd::math::sd_exp<T, T>(outRow[f] - static_cast<T>(rowMax));
        outRow[f] = val;
        threadSum += static_cast<AccT>(val);
    }

    AccT rowSum = sd::device::blockAllReduceSum(threadSum, warpBuf);
    AccT sharedInvSum = static_cast<AccT>(1) / rowSum;

    for (LongType f = threadIdx.x; f < dim; f += blockDim.x) {
        outRow[f] = outRow[f] * static_cast<T>(sharedInvSum);
    }
}

// Phase 2: Compute gradients
// dL/dz = s * (gradOutput - dot(gradOutput, s))  per row
// dL/dinput = dL/dz / temperature
// dL/dcenter = -sum_batch(dL/dinput)
template <typename T>
SD_KERNEL void csBpGradKernel(const T* __restrict__ softmaxOut,
                                const T* __restrict__ gradOutput,
                                T* __restrict__ dLdInput,
                                typename AccType<T>::type* __restrict__ dLdCenterAcc,
                                const LongType batch,
                                const LongType dim,
                                const T invTemp) {
    using AccT = typename AccType<T>::type;

    extern __shared__ char sharedMem[];
    AccT* warpBuf = reinterpret_cast<AccT*>(sharedMem);

    const int b = blockIdx.x;
    if (b >= batch) return;

    const T* sRow = softmaxOut + b * dim;
    const T* gRow = gradOutput + b * dim;
    T* dRow = dLdInput + b * dim;

    // Compute dot(gradOutput, softmax) for this row
    AccT threadDot = static_cast<AccT>(0);
    for (LongType f = threadIdx.x; f < dim; f += blockDim.x) {
        threadDot += static_cast<AccT>(gRow[f]) * static_cast<AccT>(sRow[f]);
    }

    AccT dotYG = sd::device::blockAllReduceSum(threadDot, warpBuf);
    T dotYG_T = static_cast<T>(dotYG);

    // Compute gradient: s * (grad - dot) / temp
    for (LongType f = threadIdx.x; f < dim; f += blockDim.x) {
        T y = sRow[f];
        T dLdz = y * (gRow[f] - dotYG_T);
        T dLdx = dLdz * invTemp;
        dRow[f] = dLdx;
        atomicAdd(&dLdCenterAcc[f], -static_cast<AccT>(dLdx));
    }
}

// Phase 3: Convert accumulated float center gradient to output type
template <typename T>
SD_KERNEL void convertCenterGradKernel(const typename AccType<T>::type* __restrict__ dLdCenterAcc,
                                         T* __restrict__ dLdCenter,
                                         const LongType dim) {
    for (LongType f = blockIdx.x * blockDim.x + threadIdx.x;
         f < dim; f += blockDim.x * gridDim.x) {
        dLdCenter[f] = static_cast<T>(dLdCenterAcc[f]);
    }
}

template <typename T>
void centerAndSharpenBpCudaLauncher(const cudaStream_t* stream,
                                     const void* vInput, const void* vCenter,
                                     const void* vGradOutput,
                                     void* vDLdInput, void* vDLdCenter,
                                     void* vSoftmaxOut, void* vDLdCenterAcc,
                                     LongType batch, LongType dim,
                                     double temperature) {
    auto input = reinterpret_cast<const T*>(vInput);
    auto center = reinterpret_cast<const T*>(vCenter);
    auto gradOutput = reinterpret_cast<const T*>(vGradOutput);
    auto dLdInput = reinterpret_cast<T*>(vDLdInput);
    auto dLdCenter = reinterpret_cast<T*>(vDLdCenter);
    auto softmaxOut = reinterpret_cast<T*>(vSoftmaxOut);
    auto dLdCenterAcc = reinterpret_cast<typename AccType<T>::type*>(vDLdCenterAcc);

    using AccT = typename AccType<T>::type;
    T invTemp = static_cast<T>(1.0 / temperature);

    int threads = 256;
    if (dim < 256) {
        threads = ((dim + CS_BP_WARP_SIZE - 1) / CS_BP_WARP_SIZE) * CS_BP_WARP_SIZE;
        if (threads < CS_BP_WARP_SIZE) threads = CS_BP_WARP_SIZE;
    }
    int numWarps = (threads + CS_BP_WARP_SIZE - 1) / CS_BP_WARP_SIZE;
    size_t sharedSize = numWarps * sizeof(AccT);

    // Zero accumulator
    cudaMemsetAsync(dLdCenterAcc, 0, dim * sizeof(typename AccType<T>::type), *stream);

    // Recompute softmax
    csBpSoftmaxKernel<T><<<batch, threads, sharedSize, *stream>>>(
        input, center, softmaxOut, batch, dim, invTemp);
    DebugHelper::checkGlobalErrorCode("csBpSoftmax failed");

    // Compute gradients
    csBpGradKernel<T><<<batch, threads, sharedSize, *stream>>>(
        softmaxOut, gradOutput, dLdInput, dLdCenterAcc, batch, dim, invTemp);
    DebugHelper::checkGlobalErrorCode("csBpGrad failed");

    // Convert center gradient
    int convThreads = 256;
    int convBlocks = (dim + convThreads - 1) / convThreads;
    convertCenterGradKernel<T><<<convBlocks, convThreads, 0, *stream>>>(
        dLdCenterAcc, dLdCenter, dim);
    DebugHelper::checkGlobalErrorCode("convertCenterGrad failed");
}

BUILD_SINGLE_TEMPLATE(void centerAndSharpenBpCudaLauncher,
                      (const cudaStream_t* stream,
                       const void* vInput, const void* vCenter,
                       const void* vGradOutput,
                       void* vDLdInput, void* vDLdCenter,
                       void* vSoftmaxOut, void* vDLdCenterAcc,
                       LongType batch, LongType dim, double temperature),
                      SD_FLOAT_TYPES);

void centerAndSharpenBp(NDArray* input, NDArray* center, NDArray* gradOutput,
                         NDArray* dLdInput, NDArray* dLdCenter,
                         double temperature, LaunchContext* context) {
    auto batch = input->sizeAt(0);
    auto dim = input->sizeAt(1);
    auto stream = context->getCudaStream();

    // Temp arrays. dLdCenterAcc holds the AccT accumulator (double when T=double,
    // float otherwise) — its dtype MUST match AccType<T>::type or the launcher's
    // reinterpret_cast + cudaMemsetAsync(sizeof(AccT)) overruns the buffer.
    auto accDtype = input->dataType() == DataType::DOUBLE ? DataType::DOUBLE : DataType::FLOAT32;
    auto softmaxOut = NDArrayFactory::create('c', {batch, dim}, input->dataType(), context);
    auto dLdCenterAcc = NDArrayFactory::create('c', {dim}, accDtype, context);

    NDArray::prepareSpecialUse({dLdInput, dLdCenter}, {input, center, gradOutput});

    BUILD_SINGLE_SELECTOR(input->dataType(), centerAndSharpenBpCudaLauncher,
                          (stream, input->specialBuffer(), center->specialBuffer(),
                           gradOutput->specialBuffer(),
                           dLdInput->specialBuffer(), dLdCenter->specialBuffer(),
                           softmaxOut->specialBuffer(), dLdCenterAcc->specialBuffer(),
                           batch, dim, temperature),
                          SD_FLOAT_TYPES);

    NDArray::registerSpecialUse({dLdInput, dLdCenter}, {input, center, gradOutput});

    delete softmaxOut;
    delete dLdCenterAcc;
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
