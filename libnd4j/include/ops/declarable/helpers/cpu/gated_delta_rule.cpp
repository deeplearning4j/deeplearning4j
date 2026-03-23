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

#include <execution/Threads.h>
#include <math/templatemath.h>
#include <ops/declarable/helpers/gated_delta_rule.h>

namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static void gatedDeltaRule_(LaunchContext* context, NDArray* Q, NDArray* K, NDArray* V,
                             NDArray* beta, NDArray* gate, NDArray* stateIn,
                             NDArray* output, NDArray* stateOut) {
    const auto B = Q->sizeAt(0);
    const auto L = Q->sizeAt(1);
    const auto H = Q->sizeAt(2);
    const auto D_k = Q->sizeAt(3);
    const auto D_v = V->sizeAt(3);

    const T* qBuf = Q->bufferAsT<T>();
    const T* kBuf = K->bufferAsT<T>();
    const T* vBuf = V->bufferAsT<T>();
    const T* betaBuf = beta->bufferAsT<T>();
    const T* gateBuf = gate->bufferAsT<T>();
    T* outBuf = output->bufferAsT<T>();
    T* stateOutBuf = stateOut->bufferAsT<T>();

    const auto qS0 = Q->strideAt(0), qS1 = Q->strideAt(1), qS2 = Q->strideAt(2), qS3 = Q->strideAt(3);
    const auto kS0 = K->strideAt(0), kS1 = K->strideAt(1), kS2 = K->strideAt(2), kS3 = K->strideAt(3);
    const auto vS0 = V->strideAt(0), vS1 = V->strideAt(1), vS2 = V->strideAt(2), vS3 = V->strideAt(3);
    const auto bS0 = beta->strideAt(0), bS1 = beta->strideAt(1), bS2 = beta->strideAt(2);
    const auto gS0 = gate->strideAt(0), gS1 = gate->strideAt(1), gS2 = gate->strideAt(2);
    const auto oS0 = output->strideAt(0), oS1 = output->strideAt(1), oS2 = output->strideAt(2), oS3 = output->strideAt(3);
    const auto sS0 = stateOut->strideAt(0), sS1 = stateOut->strideAt(1), sS2 = stateOut->strideAt(2), sS3 = stateOut->strideAt(3);

    // Working copy of recurrent state [B, H, D_k, D_v] stored contiguously
    const LongType stateSize = B * H * D_k * D_v;
    std::vector<T> stateBuf(stateSize, static_cast<T>(0));

    if (stateIn != nullptr) {
        const T* sInBuf = stateIn->bufferAsT<T>();
        const auto siS0 = stateIn->strideAt(0), siS1 = stateIn->strideAt(1);
        const auto siS2 = stateIn->strideAt(2), siS3 = stateIn->strideAt(3);
        for (LongType b = 0; b < B; ++b)
            for (LongType h = 0; h < H; ++h)
                for (LongType dk = 0; dk < D_k; ++dk)
                    for (LongType dv = 0; dv < D_v; ++dv)
                        stateBuf[((b * H + h) * D_k + dk) * D_v + dv] =
                            sInBuf[b * siS0 + h * siS1 + dk * siS2 + dv * siS3];
    }

    // Sequential over timesteps, parallel over batch*heads
    for (LongType t = 0; t < L; ++t) {
        auto func = PRAGMA_THREADS_FOR {
            for (auto bh = start; bh < stop; ++bh) {
                const LongType b = bh / H;
                const LongType h = bh % H;
                const T exp_g = sd::math::sd_exp<T, T>(gateBuf[b * gS0 + t * gS1 + h * gS2]);
                const T beta_val = betaBuf[b * bS0 + t * bS1 + h * bS2];
                T* sPtr = stateBuf.data() + ((b * H + h) * D_k) * D_v;

                for (LongType dv = 0; dv < D_v; ++dv) {
                    // prediction = S^T * k
                    T prediction = static_cast<T>(0);
                    for (LongType dk = 0; dk < D_k; ++dk)
                        prediction += sPtr[dk * D_v + dv] * kBuf[b * kS0 + t * kS1 + h * kS2 + dk * kS3];

                    // delta = v - exp(g) * prediction
                    const T delta = vBuf[b * vS0 + t * vS1 + h * vS2 + dv * vS3] - exp_g * prediction;

                    // S = exp(g) * S + beta * k * delta
                    for (LongType dk = 0; dk < D_k; ++dk) {
                        const T k_val = kBuf[b * kS0 + t * kS1 + h * kS2 + dk * kS3];
                        sPtr[dk * D_v + dv] = exp_g * sPtr[dk * D_v + dv] + beta_val * k_val * delta;
                    }
                }

                // output = S^T * q
                for (LongType dv = 0; dv < D_v; ++dv) {
                    T out_val = static_cast<T>(0);
                    for (LongType dk = 0; dk < D_k; ++dk)
                        out_val += sPtr[dk * D_v + dv] * qBuf[b * qS0 + t * qS1 + h * qS2 + dk * qS3];
                    outBuf[b * oS0 + t * oS1 + h * oS2 + dv * oS3] = out_val;
                }
            }
        };
        samediff::Threads::parallel_tad(func, 0, B * H);
    }

    // Copy final state out
    for (LongType b = 0; b < B; ++b)
        for (LongType h = 0; h < H; ++h)
            for (LongType dk = 0; dk < D_k; ++dk)
                for (LongType dv = 0; dv < D_v; ++dv)
                    stateOutBuf[b * sS0 + h * sS1 + dk * sS2 + dv * sS3] =
                        stateBuf[((b * H + h) * D_k + dk) * D_v + dv];
}

void gatedDeltaRule(LaunchContext* context, NDArray* Q, NDArray* K, NDArray* V,
                     NDArray* beta, NDArray* gate, NDArray* stateIn,
                     NDArray* output, NDArray* stateOut) {
    NDArray::preparePrimaryUse({output, stateOut}, {Q, K, V, beta, gate});
    if (stateIn != nullptr) NDArray::preparePrimaryUse({}, {stateIn});

    BUILD_SINGLE_SELECTOR(Q->dataType(), gatedDeltaRule_,
        (context, Q, K, V, beta, gate, stateIn, output, stateOut), SD_FLOAT_TYPES);

    NDArray::registerPrimaryUse({output, stateOut}, {Q, K, V, beta, gate});
    if (stateIn != nullptr) NDArray::registerPrimaryUse({}, {stateIn});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
