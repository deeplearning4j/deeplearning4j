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
#include <helpers/DebugHelper.h>
#include <math/templatemath.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <ops/declarable/helpers/cuda/device_primitives.cuh>

namespace sd {
namespace ops {
namespace helpers {

static constexpr int CS_WARP_SIZE = 32;

// Accumulator type: double when T=double for precision, float otherwise.
template <typename T>
struct AccType { using type = float; };
template <>
struct AccType<double> { using type = double; };

// Phase 1: Center and scale: output[b][f] = (input[b][f] - center[f]) / temperature
// Phase 2: Row-wise softmax
template <typename T>
SD_KERNEL void centerAndSharpenKernel(const T* __restrict__ input,
                                        const T* __restrict__ center,
                                        T* __restrict__ output,
                                        const LongType batch,
                                        const LongType dim,
                                        const T invTemp) {
    using AccT = typename AccType<T>::type;

    extern __shared__ char sharedMem[];
    AccT* warpBuf = reinterpret_cast<AccT*>(sharedMem);

    const int b = blockIdx.x;
    if (b >= batch) return;

    const T* inRow = input + b * dim;
    T* outRow = output + b * dim;

    // Step 1: Center and scale, find max for numerical stability
    AccT threadMax = -sd::DataTypeUtils::max<AccT>();
    for (LongType f = threadIdx.x; f < dim; f += blockDim.x) {
        T val = (inRow[f] - center[f]) * invTemp;
        outRow[f] = val;
        threadMax = sd::math::sd_max<AccT>(threadMax, static_cast<AccT>(val));
    }

    // Block-reduce max (result broadcast to all threads) for numerical stability
    AccT rowMax = sd::device::blockAllReduceMax(threadMax, warpBuf);

    // Step 2: exp(val - max) and sum
    AccT threadSum = static_cast<AccT>(0);
    for (LongType f = threadIdx.x; f < dim; f += blockDim.x) {
        T val = sd::math::sd_exp<T, T>(outRow[f] - static_cast<T>(rowMax));
        outRow[f] = val;
        threadSum += static_cast<AccT>(val);
    }

    // Block-reduce sum (result broadcast to all threads)
    AccT rowSum = sd::device::blockAllReduceSum(threadSum, warpBuf);
    AccT sharedInvSum = static_cast<AccT>(1) / rowSum;

    // Step 3: Normalize
    for (LongType f = threadIdx.x; f < dim; f += blockDim.x) {
        outRow[f] = outRow[f] * static_cast<T>(sharedInvSum);
    }
}

template <typename T>
void centerAndSharpenCudaLauncher(const cudaStream_t* stream,
                                   const void* vInput, const void* vCenter,
                                   void* vOutput,
                                   LongType batch, LongType dim,
                                   double temperature) {
    auto input = reinterpret_cast<const T*>(vInput);
    auto center = reinterpret_cast<const T*>(vCenter);
    auto output = reinterpret_cast<T*>(vOutput);

    using AccT = typename AccType<T>::type;
    T invTemp = static_cast<T>(1.0 / temperature);

    int threads = 256;
    if (dim < 256) {
        threads = ((dim + CS_WARP_SIZE - 1) / CS_WARP_SIZE) * CS_WARP_SIZE;
        if (threads < CS_WARP_SIZE) threads = CS_WARP_SIZE;
    }
    int numWarps = (threads + CS_WARP_SIZE - 1) / CS_WARP_SIZE;
    size_t sharedSize = numWarps * sizeof(AccT);

    centerAndSharpenKernel<T><<<batch, threads, sharedSize, *stream>>>(
        input, center, output, batch, dim, invTemp);
    DebugHelper::checkGlobalErrorCode("centerAndSharpen kernel failed");
}

BUILD_SINGLE_TEMPLATE(void centerAndSharpenCudaLauncher,
                      (const cudaStream_t* stream,
                       const void* vInput, const void* vCenter,
                       void* vOutput,
                       LongType batch, LongType dim, double temperature),
                      SD_FLOAT_TYPES);

void centerAndSharpen(NDArray* input, NDArray* center, NDArray* output,
                       double temperature, LaunchContext* context) {
    auto batch = input->sizeAt(0);
    auto dim = input->sizeAt(1);
    auto stream = context->getCudaStream();

    NDArray::prepareSpecialUse({output}, {input, center});

    BUILD_SINGLE_SELECTOR(input->dataType(), centerAndSharpenCudaLauncher,
                          (stream, input->specialBuffer(), center->specialBuffer(),
                           output->specialBuffer(), batch, dim, temperature),
                          SD_FLOAT_TYPES);

    NDArray::registerSpecialUse({output}, {input, center});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
