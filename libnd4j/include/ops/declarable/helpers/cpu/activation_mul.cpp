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
#include <execution/Threads.h>
#include <math/templatemath.h>
#include <cmath>

namespace sd {
namespace ops {
namespace helpers {

static constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;  // sqrt(2/pi)
static constexpr float GELU_COEFF = 0.044715f;

// ─── SiLU and Mul (SwiGLU) ──────────────────────────────────────────────────

template <typename T>
static void siluAndMulImpl(NDArray* gate, NDArray* up, NDArray* output) {
    auto len = gate->lengthOf();
    auto gatePtr = gate->bufferAsT<T>();
    auto upPtr = up->bufferAsT<T>();
    auto outPtr = output->bufferAsT<T>();

    auto func = PRAGMA_THREADS_FOR {
        for (auto i = start; i < stop; ++i) {
            float g = static_cast<float>(gatePtr[i]);
            float u = static_cast<float>(upPtr[i]);
            // silu(g) = g * sigmoid(g) = g / (1 + exp(-g))
            float sigmoid_g = 1.0f / (1.0f + std::exp(-g));
            outPtr[i] = static_cast<T>(g * sigmoid_g * u);
        }
    };
    samediff::Threads::parallel_for(func, 0, len);
}

void siluAndMul(LaunchContext* context, NDArray* gate, NDArray* up, NDArray* output) {
    gate->syncToHost();
    up->syncToHost();
    BUILD_SINGLE_SELECTOR(gate->dataType(), siluAndMulImpl, (gate, up, output), SD_FLOAT_TYPES);
    output->tickWriteHost();
}

// ─── GELU and Mul (GEGLU) ───────────────────────────────────────────────────

template <typename T>
static void geluAndMulImpl(NDArray* gate, NDArray* up, NDArray* output) {
    auto len = gate->lengthOf();
    auto gatePtr = gate->bufferAsT<T>();
    auto upPtr = up->bufferAsT<T>();
    auto outPtr = output->bufferAsT<T>();

    auto func = PRAGMA_THREADS_FOR {
        for (auto i = start; i < stop; ++i) {
            float g = static_cast<float>(gatePtr[i]);
            float u = static_cast<float>(upPtr[i]);
            // gelu(g) = 0.5 * g * (1 + erf(g / sqrt(2)))
            float gelu_g = 0.5f * g * (1.0f + std::erf(g * 0.7071067811865476f));
            outPtr[i] = static_cast<T>(gelu_g * u);
        }
    };
    samediff::Threads::parallel_for(func, 0, len);
}

void geluAndMul(LaunchContext* context, NDArray* gate, NDArray* up, NDArray* output) {
    gate->syncToHost();
    up->syncToHost();
    BUILD_SINGLE_SELECTOR(gate->dataType(), geluAndMulImpl, (gate, up, output), SD_FLOAT_TYPES);
    output->tickWriteHost();
}

// ─── GELU (tanh approx) and Mul ─────────────────────────────────────────────

template <typename T>
static void geluTanhAndMulImpl(NDArray* gate, NDArray* up, NDArray* output) {
    auto len = gate->lengthOf();
    auto gatePtr = gate->bufferAsT<T>();
    auto upPtr = up->bufferAsT<T>();
    auto outPtr = output->bufferAsT<T>();

    auto func = PRAGMA_THREADS_FOR {
        for (auto i = start; i < stop; ++i) {
            float g = static_cast<float>(gatePtr[i]);
            float u = static_cast<float>(upPtr[i]);
            // gelu_tanh(g) = 0.5 * g * (1 + tanh(sqrt(2/pi) * (g + 0.044715 * g^3)))
            float inner = SQRT_2_OVER_PI * (g + GELU_COEFF * g * g * g);
            float gelu_g = 0.5f * g * (1.0f + std::tanh(inner));
            outPtr[i] = static_cast<T>(gelu_g * u);
        }
    };
    samediff::Threads::parallel_for(func, 0, len);
}

void geluTanhAndMul(LaunchContext* context, NDArray* gate, NDArray* up, NDArray* output) {
    gate->syncToHost();
    up->syncToHost();
    BUILD_SINGLE_SELECTOR(gate->dataType(), geluTanhAndMulImpl, (gate, up, output), SD_FLOAT_TYPES);
    output->tickWriteHost();
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
