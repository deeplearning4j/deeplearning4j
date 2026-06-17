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

#include <cstring>
#include <vector>

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

    // Working state in TRANSPOSED layout: [B, H, D_v, D_k] instead of [B, H, D_k, D_v].
    // This makes the inner dk loops access contiguous memory (stride 1 instead of D_v),
    // enabling vectorization and eliminating cache thrashing.
    // Always float for accumulation — avoids FP16 overflow in D_k=128 sums.
    //
    // Thread-local buffer avoids per-call heap allocation (was 524KB+ × 18 layers/token).
    const LongType stateSize = B * H * D_v * D_k;
    static thread_local std::vector<float> tlStateBuf;
    if (static_cast<LongType>(tlStateBuf.size()) < stateSize) {
        tlStateBuf.resize(stateSize);
    }
    float* stateBuf = tlStateBuf.data();

    // Initialize from stateIn — transpose [B,H,D_k,D_v] → [B,H,D_v,D_k]
    if (stateIn != nullptr) {
        const T* sInBuf = stateIn->bufferAsT<T>();
        const auto siS0 = stateIn->strideAt(0), siS1 = stateIn->strideAt(1);
        const auto siS2 = stateIn->strideAt(2), siS3 = stateIn->strideAt(3);
        for (LongType b = 0; b < B; ++b)
            for (LongType h = 0; h < H; ++h)
                for (LongType dk = 0; dk < D_k; ++dk)
                    for (LongType dv = 0; dv < D_v; ++dv)
                        stateBuf[((b * H + h) * D_v + dv) * D_k + dk] =
                            static_cast<float>(sInBuf[b * siS0 + h * siS1 + dk * siS2 + dv * siS3]);
    } else {
        std::memset(stateBuf, 0, stateSize * sizeof(float));
    }

    // Sequential over timesteps, parallel over batch*heads
    for (LongType t = 0; t < L; ++t) {
        auto func = PRAGMA_THREADS_FOR {
            for (auto bh = start; bh < stop; ++bh) {
                const LongType b = bh / H;
                const LongType h = bh % H;
                const float exp_g_f = sd::math::sd_exp<float, float>(static_cast<float>(gateBuf[b * gS0 + t * gS1 + h * gS2]));
                const float beta_f = static_cast<float>(betaBuf[b * bS0 + t * bS1 + h * bS2]);
                float* sBase = stateBuf + (b * H + h) * D_v * D_k;

                // Pre-load k vector into contiguous float buffer on the stack.
                // Eliminates repeated strided kBuf access in the inner loops.
                // D_k is typically 64-128; 512 covers all known models.
                float kLocal[512];
                const LongType kBase = b * kS0 + t * kS1 + h * kS2;
                for (LongType dk = 0; dk < D_k; ++dk)
                    kLocal[dk] = static_cast<float>(kBuf[kBase + dk * kS3]);

                for (LongType dv = 0; dv < D_v; ++dv) {
                    float* sRow = sBase + dv * D_k;  // contiguous D_k row

                    // prediction = dot(state[dv,:], k[:]) — both arrays contiguous
                    float prediction = 0.0f;
                    for (LongType dk = 0; dk < D_k; ++dk)
                        prediction += sRow[dk] * kLocal[dk];

                    // delta = v - exp(g) * prediction
                    const float vVal = static_cast<float>(vBuf[b * vS0 + t * vS1 + h * vS2 + dv * vS3]);
                    const float delta = vVal - exp_g_f * prediction;

                    // S = exp(g) * S + beta * k * delta — contiguous update
                    const float beta_delta = beta_f * delta;
                    for (LongType dk = 0; dk < D_k; ++dk)
                        sRow[dk] = exp_g_f * sRow[dk] + beta_delta * kLocal[dk];
                }

                // Pre-load q vector for output dot products
                float qLocal[512];
                const LongType qBase = b * qS0 + t * qS1 + h * qS2;
                for (LongType dk = 0; dk < D_k; ++dk)
                    qLocal[dk] = static_cast<float>(qBuf[qBase + dk * qS3]);

                // output = dot(state[dv,:], q[:]) — both arrays contiguous
                for (LongType dv = 0; dv < D_v; ++dv) {
                    float* sRow = sBase + dv * D_k;
                    float out_val = 0.0f;
                    for (LongType dk = 0; dk < D_k; ++dk)
                        out_val += sRow[dk] * qLocal[dk];
                    outBuf[b * oS0 + t * oS1 + h * oS2 + dv * oS3] = static_cast<T>(out_val);
                }
            }
        };
        samediff::Threads::parallel_tad(func, 0, B * H);
    }

    // Write back: transpose [B,H,D_v,D_k] → [B,H,D_k,D_v] into stateOut
    for (LongType b = 0; b < B; ++b)
        for (LongType h = 0; h < H; ++h)
            for (LongType dv = 0; dv < D_v; ++dv)
                for (LongType dk = 0; dk < D_k; ++dk)
                    stateOutBuf[b * sS0 + h * sS1 + dk * sS2 + dv * sS3] =
                        static_cast<T>(stateBuf[((b * H + h) * D_v + dv) * D_k + dk]);
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
