/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

#include <ops/declarable/helpers/activation_mul.h>
#include <array/NDArray.h>
#include <helpers/DebugHelper.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include "execution/cuda/LaunchDims.h"

namespace sd {
namespace ops {
namespace helpers {

// ─── SiLU and Mul (SwiGLU) CUDA kernel ──────────────────────────────────────

template <typename T>
static SD_KERNEL void siluAndMulKernel(const void* vGate, const void* vUp,
                                        void* vOutput, const LongType len) {
    auto gate = reinterpret_cast<const T*>(vGate);
    auto up = reinterpret_cast<const T*>(vUp);
    auto output = reinterpret_cast<T*>(vOutput);

    LongType idx = blockIdx.x * blockDim.x + threadIdx.x;
    LongType stride = gridDim.x * blockDim.x;

    for (LongType i = idx; i < len; i += stride) {
        float g = static_cast<float>(gate[i]);
        float u = static_cast<float>(up[i]);
        // silu(g) * u = g * sigmoid(g) * u
        float sigmoid_g = 1.0f / (1.0f + expf(-g));
        output[i] = static_cast<T>(g * sigmoid_g * u);
    }
}

template <typename T>
static void siluAndMulLauncher(NDArray* gate, NDArray* up, NDArray* output, LaunchContext* context) {
    auto stream = context->getCudaStream();
    auto len = gate->lengthOf();

    int blockSize = 256;
    int gridSize = (len + blockSize - 1) / blockSize;
    if (gridSize > 65535) gridSize = 65535;

    siluAndMulKernel<T><<<gridSize, blockSize, 0, *stream>>>(
        gate->specialBuffer(), up->specialBuffer(), output->specialBuffer(), len);
    DebugHelper::checkGlobalErrorCode("siluAndMulKernel failed");
}

void siluAndMul(LaunchContext* context, NDArray* gate, NDArray* up, NDArray* output) {
    NDArray::prepareSpecialUse({output}, {gate, up});
    BUILD_SINGLE_SELECTOR(gate->dataType(), siluAndMulLauncher,
                          (gate, up, output, context), SD_FLOAT_TYPES);
    NDArray::registerSpecialUse({output}, {gate, up});
}

// ─── GELU and Mul (GEGLU) CUDA kernel ───────────────────────────────────────

template <typename T>
static SD_KERNEL void geluAndMulKernel(const void* vGate, const void* vUp,
                                        void* vOutput, const LongType len) {
    auto gate = reinterpret_cast<const T*>(vGate);
    auto up = reinterpret_cast<const T*>(vUp);
    auto output = reinterpret_cast<T*>(vOutput);

    LongType idx = blockIdx.x * blockDim.x + threadIdx.x;
    LongType stride = gridDim.x * blockDim.x;

    for (LongType i = idx; i < len; i += stride) {
        float g = static_cast<float>(gate[i]);
        float u = static_cast<float>(up[i]);
        // gelu(g) = 0.5 * g * (1 + erf(g / sqrt(2)))
        float gelu_g = 0.5f * g * (1.0f + erff(g * 0.7071067811865476f));
        output[i] = static_cast<T>(gelu_g * u);
    }
}

template <typename T>
static void geluAndMulLauncher(NDArray* gate, NDArray* up, NDArray* output, LaunchContext* context) {
    auto stream = context->getCudaStream();
    auto len = gate->lengthOf();

    int blockSize = 256;
    int gridSize = (len + blockSize - 1) / blockSize;
    if (gridSize > 65535) gridSize = 65535;

    geluAndMulKernel<T><<<gridSize, blockSize, 0, *stream>>>(
        gate->specialBuffer(), up->specialBuffer(), output->specialBuffer(), len);
    DebugHelper::checkGlobalErrorCode("geluAndMulKernel failed");
}

void geluAndMul(LaunchContext* context, NDArray* gate, NDArray* up, NDArray* output) {
    NDArray::prepareSpecialUse({output}, {gate, up});
    BUILD_SINGLE_SELECTOR(gate->dataType(), geluAndMulLauncher,
                          (gate, up, output, context), SD_FLOAT_TYPES);
    NDArray::registerSpecialUse({output}, {gate, up});
}

// ─── GELU (tanh approx) and Mul CUDA kernel ─────────────────────────────────

template <typename T>
static SD_KERNEL void geluTanhAndMulKernel(const void* vGate, const void* vUp,
                                            void* vOutput, const LongType len) {
    auto gate = reinterpret_cast<const T*>(vGate);
    auto up = reinterpret_cast<const T*>(vUp);
    auto output = reinterpret_cast<T*>(vOutput);

    LongType idx = blockIdx.x * blockDim.x + threadIdx.x;
    LongType stride = gridDim.x * blockDim.x;

    constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;
    constexpr float GELU_COEFF = 0.044715f;

    for (LongType i = idx; i < len; i += stride) {
        float g = static_cast<float>(gate[i]);
        float u = static_cast<float>(up[i]);
        // gelu_tanh(g) = 0.5 * g * (1 + tanh(sqrt(2/pi) * (g + 0.044715 * g^3)))
        float inner = SQRT_2_OVER_PI * (g + GELU_COEFF * g * g * g);
        float gelu_g = 0.5f * g * (1.0f + tanhf(inner));
        output[i] = static_cast<T>(gelu_g * u);
    }
}

template <typename T>
static void geluTanhAndMulLauncher(NDArray* gate, NDArray* up, NDArray* output, LaunchContext* context) {
    auto stream = context->getCudaStream();
    auto len = gate->lengthOf();

    int blockSize = 256;
    int gridSize = (len + blockSize - 1) / blockSize;
    if (gridSize > 65535) gridSize = 65535;

    geluTanhAndMulKernel<T><<<gridSize, blockSize, 0, *stream>>>(
        gate->specialBuffer(), up->specialBuffer(), output->specialBuffer(), len);
    DebugHelper::checkGlobalErrorCode("geluTanhAndMulKernel failed");
}

void geluTanhAndMul(LaunchContext* context, NDArray* gate, NDArray* up, NDArray* output) {
    NDArray::prepareSpecialUse({output}, {gate, up});
    BUILD_SINGLE_SELECTOR(gate->dataType(), geluTanhAndMulLauncher,
                          (gate, up, output, context), SD_FLOAT_TYPES);
    NDArray::registerSpecialUse({output}, {gate, up});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
