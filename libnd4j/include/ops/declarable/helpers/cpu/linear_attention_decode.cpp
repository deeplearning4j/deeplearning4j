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

//
// linear_attention_decode.cpp — Optimized CPU decode for linear attention.
//
// Single-token recurrence:
//   state_new = exp(-decay[h]) * state_old + k (x) v
//   output    = q @ state_new
//
// State is always float32 regardless of input dtype.
//
// Input:  [batch, 1, numHeads, headDimK/V]
// State:  [batch, numHeads, headDimV, headDimK] (float32, in-place)
// Output: [batch, 1, numHeads, headDimV]
//

#include <ops/declarable/helpers/linear_attention_decode.h>
#include <execution/Threads.h>
#include <math/templatemath.h>
#include <cmath>

namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static void linearAttentionDecodeCpuImpl_(LaunchContext* context,
                                          NDArray* query,
                                          NDArray* key,
                                          NDArray* value,
                                          NDArray* decayRates,
                                          NDArray* state,
                                          NDArray* output) {
    const auto batch    = query->sizeAt(0);
    const auto numHeads = query->sizeAt(2);
    const auto headDimK = query->sizeAt(3);
    const auto headDimV = value->sizeAt(3);

    // Input strides: [batch, 1, numHeads, headDim]
    const auto qS0 = query->strideAt(0), qS2 = query->strideAt(2), qS3 = query->strideAt(3);
    const auto kS0 = key->strideAt(0),   kS2 = key->strideAt(2),   kS3 = key->strideAt(3);
    const auto vS0 = value->strideAt(0), vS2 = value->strideAt(2), vS3 = value->strideAt(3);
    const auto oS0 = output->strideAt(0), oS2 = output->strideAt(2), oS3 = output->strideAt(3);

    // State strides: [batch, numHeads, headDimV, headDimK]
    const auto sS0 = state->strideAt(0), sS1 = state->strideAt(1), sS2 = state->strideAt(2), sS3 = state->strideAt(3);

    // seqLen=1 strides for Q/K/V — offset by strideAt(1)
    const auto qSeqOff = query->strideAt(1) * 0;  // always index 0 in seqLen dim
    const auto kSeqOff = key->strideAt(1) * 0;
    const auto vSeqOff = value->strideAt(1) * 0;
    const auto oSeqOff = output->strideAt(1) * 0;

    const T* qBuf = query->bufferAsT<T>();
    const T* kBuf = key->bufferAsT<T>();
    const T* vBuf = value->bufferAsT<T>();
    T* oBuf = output->bufferAsT<T>();
    float* sBuf = state->bufferAsT<float>();
    const float* dBuf = decayRates->bufferAsT<float>();

    // Parallelize over batch * numHeads — each (b,h) pair is independent
    auto func = PRAGMA_THREADS_FOR {
        for (auto idx = start; idx < stop; idx++) {
            const auto b = idx / numHeads;
            const auto h = idx % numHeads;
            const float decayScale = std::exp(-dBuf[h]);

            // Pre-load k and v into contiguous float buffers
            float kLocal[512];
            float vLocal[512];
            for (LongType dk = 0; dk < headDimK; dk++) {
                kLocal[dk] = static_cast<float>(kBuf[b * kS0 + h * kS2 + dk * kS3]);
            }
            for (LongType dv = 0; dv < headDimV; dv++) {
                vLocal[dv] = static_cast<float>(vBuf[b * vS0 + h * vS2 + dv * vS3]);
            }

            // State update: S[dv, dk] = decay * S[dv, dk] + v[dv] * k[dk]
            for (LongType dv = 0; dv < headDimV; dv++) {
                const float vVal = vLocal[dv];
                PRAGMA_OMP_SIMD
                for (LongType dk = 0; dk < headDimK; dk++) {
                    const auto sIdx = b * sS0 + h * sS1 + dv * sS2 + dk * sS3;
                    sBuf[sIdx] = decayScale * sBuf[sIdx] + vVal * kLocal[dk];
                }
            }

            // Output: o[dv] = sum_dk q[dk] * S[dv, dk]
            float qLocal[512];
            for (LongType dk = 0; dk < headDimK; dk++) {
                qLocal[dk] = static_cast<float>(qBuf[b * qS0 + h * qS2 + dk * qS3]);
            }

            for (LongType dv = 0; dv < headDimV; dv++) {
                float sum = 0.0f;
                PRAGMA_OMP_SIMD
                for (LongType dk = 0; dk < headDimK; dk++) {
                    sum += qLocal[dk] * sBuf[b * sS0 + h * sS1 + dv * sS2 + dk * sS3];
                }
                oBuf[b * oS0 + h * oS2 + dv * oS3] = static_cast<T>(sum);
            }
        }
    };

    samediff::Threads::parallel_tad(func, 0, batch * numHeads);
}

void linearAttentionDecodeCpu(LaunchContext* context,
                              NDArray* query,
                              NDArray* key,
                              NDArray* value,
                              NDArray* decayRates,
                              NDArray* state,
                              NDArray* output) {
    NDArray::preparePrimaryUse({output, state}, {query, key, value, decayRates});

    BUILD_SINGLE_SELECTOR(query->dataType(), linearAttentionDecodeCpuImpl_,
        (context, query, key, value, decayRates, state, output), SD_FLOAT_TYPES);

    NDArray::registerPrimaryUse({output, state}, {query, key, value, decayRates});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
