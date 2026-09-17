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
#include <ops/ops.h>
#include <cmath>

namespace sd {
namespace ops {
namespace helpers {

static constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;  // sqrt(2/pi)
static constexpr float GELU_COEFF = 0.044715f;

// ─── SiLU and Mul (SwiGLU) ──────────────────────────────────────────────────

#if NOT_EXCLUDED(OP_swish_mul) || NOT_EXCLUDED(OP_silu_and_mul)
template <typename X, typename Y, typename Z>
static void siluAndMulImpl(NDArray* gate, NDArray* up, NDArray* output) {
    using ComputeT = typename sd::math::promote_type3<X, Y, Z>::type;
    using AccT = typename simdOps::AggregateType<ComputeT>::type;
    const auto len = output->lengthOf();
    const auto* gatePtr = gate->bufferAsT<X>();
    const auto* upPtr = up->bufferAsT<Y>();
    auto* outPtr = output->bufferAsT<Z>();
    const auto* gateShape = gate->shapeInfo();
    const auto* upShape = up->shapeInfo();
    const auto* outShape = output->shapeInfo();
    const bool contiguous = gate->isSameShape(output) && up->isSameShape(output) &&
        gate->ordering() == 'c' && up->ordering() == 'c' && output->ordering() == 'c' &&
        shape::strideDescendingCAscendingF(gate->shapeInfo()) &&
        shape::strideDescendingCAscendingF(up->shapeInfo()) &&
        shape::strideDescendingCAscendingF(output->shapeInfo());

    auto func = PRAGMA_THREADS_FOR {
        for (auto i = start; i < stop; i += increment) {
            const auto gOffset = contiguous ? i : shape::subArrayIndex(i, outShape, gateShape);
            const auto uOffset = contiguous ? i : shape::subArrayIndex(i, outShape, upShape);
            const auto zOffset = contiguous ? i : shape::subArrayIndex(i, outShape, outShape);
            // Read both operands before storing, including gate == up == output.
            const X silu = simdOps::Swish<X>::op(gatePtr[gOffset], nullptr);
            const AccT u = static_cast<AccT>(upPtr[uOffset]);
            outPtr[zOffset] = static_cast<Z>(static_cast<AccT>(silu) * u);
        }
    };
    samediff::Threads::parallel_for(func, 0, len);
}

void siluAndMul(LaunchContext* context, NDArray* gate, NDArray* up, NDArray* output) {
    if (output->isEmpty()) return;
    NDArray::preparePrimaryUse({output}, {gate, up});
    BUILD_TRIPLE_SELECTOR(gate->dataType(), up->dataType(), output->dataType(),
                          siluAndMulImpl, (gate, up, output),
                          SD_FLOAT_TYPES, SD_FLOAT_TYPES, SD_FLOAT_TYPES);
    NDArray::registerPrimaryUse({output}, {gate, up});
}
#endif

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
            float gelu_g = 0.5f * g * (1.0f + sd::math::sd_tanh<float, float>(inner));
            outPtr[i] = static_cast<T>(gelu_g * u);
        }
    };
    samediff::Threads::parallel_for(func, 0, len);
}

void geluTanhAndMul(LaunchContext* context, NDArray* gate, NDArray* up, NDArray* output) {
    BUILD_SINGLE_SELECTOR(gate->dataType(), geluTanhAndMulImpl, (gate, up, output), SD_FLOAT_TYPES);
    output->tickWriteHost();
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
