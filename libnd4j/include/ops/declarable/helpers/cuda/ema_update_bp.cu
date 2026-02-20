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

#include <ops/declarable/helpers/ema_update.h>
#include <array/NDArray.h>
#include <helpers/DebugHelper.h>
#include <cuda_runtime.h>

namespace sd {
namespace ops {
namespace helpers {

template <typename T>
__global__ void emaUpdateBpKernel(const T* __restrict__ gradOutput,
                                   T* __restrict__ dLdModel,
                                   T* __restrict__ dLdShadow,
                                   const T oneMinusDecay,
                                   const T decay,
                                   const LongType length) {
    for (LongType idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < length; idx += blockDim.x * gridDim.x) {
        T g = gradOutput[idx];
        dLdModel[idx] = g * oneMinusDecay;
        dLdShadow[idx] = g * decay;
    }
}

template <typename T>
void emaUpdateBpCudaLauncher(const cudaStream_t* stream,
                              const void* vGradOutput,
                              void* vDLdModel, void* vDLdShadow,
                              double decay, LongType length) {
    auto gradOutput = reinterpret_cast<const T*>(vGradOutput);
    auto dLdModel = reinterpret_cast<T*>(vDLdModel);
    auto dLdShadow = reinterpret_cast<T*>(vDLdShadow);

    T decayT = static_cast<T>(decay);
    T oneMinusDecayT = static_cast<T>(1.0 - decay);

    int threads = 256;
    int blocks = (length + threads - 1) / threads;

    emaUpdateBpKernel<T><<<blocks, threads, 0, *stream>>>(
        gradOutput, dLdModel, dLdShadow, oneMinusDecayT, decayT, length);
    DebugHelper::checkGlobalErrorCode("emaUpdateBp kernel failed");
}

BUILD_SINGLE_TEMPLATE(void emaUpdateBpCudaLauncher,
                      (const cudaStream_t* stream,
                       const void* vGradOutput,
                       void* vDLdModel, void* vDLdShadow,
                       double decay, LongType length),
                      SD_FLOAT_TYPES);

void emaUpdateBp(NDArray* model, NDArray* shadow, NDArray* gradOutput,
                  NDArray* dLdModel, NDArray* dLdShadow,
                  double decay, LaunchContext* context) {
    auto stream = context->getCudaStream();
    auto length = gradOutput->lengthOf();

    NDArray::prepareSpecialUse({dLdModel, dLdShadow}, {gradOutput});

    BUILD_SINGLE_SELECTOR(gradOutput->dataType(), emaUpdateBpCudaLauncher,
                          (stream, gradOutput->specialBuffer(),
                           dLdModel->specialBuffer(), dLdShadow->specialBuffer(),
                           decay, length),
                          SD_FLOAT_TYPES);

    NDArray::registerSpecialUse({dLdModel, dLdShadow}, {gradOutput});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
