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

namespace sd {
namespace ops {
namespace helpers {

static constexpr int CS_BP_WARP_SIZE = 32;

// Phase 1: Compute softmax output (same as forward)
template <typename T>
SD_KERNEL void csBpSoftmaxKernel(const T* __restrict__ input,
                                   const T* __restrict__ center,
                                   T* __restrict__ softmaxOut,
                                   const LongType batch,
                                   const LongType dim,
                                   const T invTemp) {
    extern __shared__ char sharedMem[];
    float* warpBuf = reinterpret_cast<float*>(sharedMem);

    const int b = blockIdx.x;
    if (b >= batch) return;

    const T* inRow = input + b * dim;
    T* outRow = softmaxOut + b * dim;

    const int lane = threadIdx.x % CS_BP_WARP_SIZE;
    const int wid = threadIdx.x / CS_BP_WARP_SIZE;
    const int numWarps = (blockDim.x + CS_BP_WARP_SIZE - 1) / CS_BP_WARP_SIZE;

    // Center, scale, find max
    float threadMax = -FLT_MAX;
    for (LongType f = threadIdx.x; f < dim; f += blockDim.x) {
        T val = (inRow[f] - center[f]) * invTemp;
        outRow[f] = val;
        threadMax = sd::math::sd_max<float>(threadMax, static_cast<float>(val));
    }

    for (int offset = CS_BP_WARP_SIZE / 2; offset > 0; offset /= 2)
        threadMax = sd::math::sd_max<float>(threadMax, __shfl_down_sync(0xffffffff, threadMax, offset));
    if (lane == 0) warpBuf[wid] = threadMax;
    __syncthreads();

    float rowMax = -FLT_MAX;
    if (threadIdx.x < numWarps) rowMax = warpBuf[threadIdx.x];
    for (int offset = CS_BP_WARP_SIZE / 2; offset > 0; offset /= 2)
        rowMax = sd::math::sd_max<float>(rowMax, __shfl_down_sync(0xffffffff, rowMax, offset));
    __shared__ float sharedMax;
    if (threadIdx.x == 0) sharedMax = rowMax;
    __syncthreads();
    rowMax = sharedMax;

    // exp and sum
    float threadSum = 0.0f;
    for (LongType f = threadIdx.x; f < dim; f += blockDim.x) {
        T val = sd::math::sd_exp<T, T>(outRow[f] - static_cast<T>(rowMax));
        outRow[f] = val;
        threadSum += static_cast<float>(val);
    }

    for (int offset = CS_BP_WARP_SIZE / 2; offset > 0; offset /= 2)
        threadSum += __shfl_down_sync(0xffffffff, threadSum, offset);
    if (lane == 0) warpBuf[wid] = threadSum;
    __syncthreads();

    float rowSum = 0.0f;
    if (threadIdx.x < numWarps) rowSum = warpBuf[threadIdx.x];
    for (int offset = CS_BP_WARP_SIZE / 2; offset > 0; offset /= 2)
        rowSum += __shfl_down_sync(0xffffffff, rowSum, offset);
    __shared__ float sharedInvSum;
    if (threadIdx.x == 0) sharedInvSum = 1.0f / rowSum;
    __syncthreads();

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
                                float* __restrict__ dLdCenterAcc,
                                const LongType batch,
                                const LongType dim,
                                const T invTemp) {
    extern __shared__ char sharedMem[];
    float* warpBuf = reinterpret_cast<float*>(sharedMem);

    const int b = blockIdx.x;
    if (b >= batch) return;

    const T* sRow = softmaxOut + b * dim;
    const T* gRow = gradOutput + b * dim;
    T* dRow = dLdInput + b * dim;

    const int lane = threadIdx.x % CS_BP_WARP_SIZE;
    const int wid = threadIdx.x / CS_BP_WARP_SIZE;
    const int numWarps = (blockDim.x + CS_BP_WARP_SIZE - 1) / CS_BP_WARP_SIZE;

    // Compute dot(gradOutput, softmax) for this row
    float threadDot = 0.0f;
    for (LongType f = threadIdx.x; f < dim; f += blockDim.x) {
        threadDot += static_cast<float>(gRow[f]) * static_cast<float>(sRow[f]);
    }

    for (int offset = CS_BP_WARP_SIZE / 2; offset > 0; offset /= 2)
        threadDot += __shfl_down_sync(0xffffffff, threadDot, offset);
    if (lane == 0) warpBuf[wid] = threadDot;
    __syncthreads();

    float dotYG = 0.0f;
    if (threadIdx.x < numWarps) dotYG = warpBuf[threadIdx.x];
    for (int offset = CS_BP_WARP_SIZE / 2; offset > 0; offset /= 2)
        dotYG += __shfl_down_sync(0xffffffff, dotYG, offset);
    __shared__ float sharedDot;
    if (threadIdx.x == 0) sharedDot = dotYG;
    __syncthreads();
    T dotYG_T = static_cast<T>(sharedDot);

    // Compute gradient: s * (grad - dot) / temp
    for (LongType f = threadIdx.x; f < dim; f += blockDim.x) {
        T y = sRow[f];
        T dLdz = y * (gRow[f] - dotYG_T);
        T dLdx = dLdz * invTemp;
        dRow[f] = dLdx;
        atomicAdd(&dLdCenterAcc[f], -static_cast<float>(dLdx));
    }
}

// Phase 3: Convert accumulated float center gradient to output type
template <typename T>
SD_KERNEL void convertCenterGradKernel(const float* __restrict__ dLdCenterAcc,
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
    auto dLdCenterAcc = reinterpret_cast<float*>(vDLdCenterAcc);

    T invTemp = static_cast<T>(1.0 / temperature);

    int threads = 256;
    if (dim < 256) {
        threads = ((dim + CS_BP_WARP_SIZE - 1) / CS_BP_WARP_SIZE) * CS_BP_WARP_SIZE;
        if (threads < CS_BP_WARP_SIZE) threads = CS_BP_WARP_SIZE;
    }
    int numWarps = (threads + CS_BP_WARP_SIZE - 1) / CS_BP_WARP_SIZE;
    size_t sharedSize = numWarps * sizeof(float);

    // Zero accumulator
    cudaMemsetAsync(dLdCenterAcc, 0, dim * sizeof(float), *stream);

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

    // Temp arrays
    auto softmaxOut = NDArrayFactory::create('c', {batch, dim}, input->dataType(), context);
    auto dLdCenterAcc = NDArrayFactory::create('c', {dim}, DataType::FLOAT32, context);

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
