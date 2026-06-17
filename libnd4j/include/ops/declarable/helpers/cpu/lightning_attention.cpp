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
// lightning_attention.cpp — Optimized CPU implementation of Lightning Attention.
//
// O(n) linear attention with per-head exponential decay via intra/inter-chunk
// decomposition. All NDArray access through template type T, working copies
// in float for accumulation stability. Thread-local buffers, SIMD inner loops.
//

#include <ops/declarable/helpers/lightning_attention.h>
#include <execution/Threads.h>
#include <math/templatemath.h>
#include <cmath>
#include <cstring>
#include <vector>

namespace sd {
namespace ops {
namespace helpers {

static constexpr LongType CHUNK_SIZE = 64;

template <typename T>
static void lightningAttentionCpuImpl_(LaunchContext* context,
                                       NDArray* query,
                                       NDArray* key,
                                       NDArray* value,
                                       NDArray* decayRates,
                                       NDArray* state,
                                       NDArray* output,
                                       bool isCausal) {
    const auto B = query->sizeAt(0);
    const auto L = query->sizeAt(1);
    const auto H = query->sizeAt(2);
    const auto D = query->sizeAt(3);

    const auto qS0 = query->strideAt(0), qS1 = query->strideAt(1), qS2 = query->strideAt(2), qS3 = query->strideAt(3);
    const auto kS0 = key->strideAt(0),   kS1 = key->strideAt(1),   kS2 = key->strideAt(2),   kS3 = key->strideAt(3);
    const auto vS0 = value->strideAt(0), vS1 = value->strideAt(1), vS2 = value->strideAt(2), vS3 = value->strideAt(3);
    const auto oS0 = output->strideAt(0), oS1 = output->strideAt(1), oS2 = output->strideAt(2), oS3 = output->strideAt(3);
    const auto sS0 = state->strideAt(0), sS1 = state->strideAt(1), sS2 = state->strideAt(2), sS3 = state->strideAt(3);
    const auto dS0 = decayRates->strideAt(0);

    const T* qBuf = query->bufferAsT<T>();
    const T* kBuf = key->bufferAsT<T>();
    const T* vBuf = value->bufferAsT<T>();
    T* oBuf = output->bufferAsT<T>();
    T* stBuf = state->bufferAsT<T>();
    const T* drBuf = decayRates->bufferAsT<T>();

    const auto numChunks = (L + CHUNK_SIZE - 1) / CHUNK_SIZE;

    // Thread-local working buffers in float for accumulation stability.
    // State working copy is [D, D] contiguous float — separate from the NDArray.
    // Chunk pre-load buffers are [CHUNK_SIZE, D] contiguous float.
    static thread_local std::vector<float> tl_state;
    static thread_local std::vector<float> tl_qc;
    static thread_local std::vector<float> tl_kc;
    static thread_local std::vector<float> tl_vc;
    static thread_local std::vector<float> tl_attn;

    auto func = PRAGMA_THREADS_FOR {
        if (static_cast<LongType>(tl_state.size()) < D * D)        tl_state.resize(D * D);
        if (static_cast<LongType>(tl_qc.size()) < CHUNK_SIZE * D)  tl_qc.resize(CHUNK_SIZE * D);
        if (static_cast<LongType>(tl_kc.size()) < CHUNK_SIZE * D)  tl_kc.resize(CHUNK_SIZE * D);
        if (static_cast<LongType>(tl_vc.size()) < CHUNK_SIZE * D)  tl_vc.resize(CHUNK_SIZE * D);
        if (static_cast<LongType>(tl_attn.size()) < CHUNK_SIZE * CHUNK_SIZE) tl_attn.resize(CHUNK_SIZE * CHUNK_SIZE);

        float* S  = tl_state.data();
        float* qc = tl_qc.data();
        float* kc = tl_kc.data();
        float* vc = tl_vc.data();
        float* A  = tl_attn.data();

        for (auto bh = start; bh < stop; ++bh) {
            const LongType b = bh / H;
            const LongType h = bh % H;
            const float decay = static_cast<float>(drBuf[h * dS0]);

            // Load initial state [D, D] from T* NDArray into float working copy
            for (LongType i = 0; i < D; ++i)
                for (LongType j = 0; j < D; ++j)
                    S[i * D + j] = static_cast<float>(stBuf[b * sS0 + h * sS1 + i * sS2 + j * sS3]);

            for (LongType chunk = 0; chunk < numChunks; ++chunk) {
                const auto cs = chunk * CHUNK_SIZE;
                const auto C = std::min(CHUNK_SIZE, L - cs);

                // Pre-load Q, K, V from T* into contiguous float working copies
                for (LongType s = 0; s < C; ++s) {
                    const LongType t = cs + s;
                    const LongType qBase = b * qS0 + t * qS1 + h * qS2;
                    const LongType kBase = b * kS0 + t * kS1 + h * kS2;
                    const LongType vBase = b * vS0 + t * vS1 + h * vS2;
                    for (LongType d = 0; d < D; ++d) {
                        qc[s * D + d] = static_cast<float>(qBuf[qBase + d * qS3]);
                        kc[s * D + d] = static_cast<float>(kBuf[kBase + d * kS3]);
                        vc[s * D + d] = static_cast<float>(vBuf[vBase + d * vS3]);
                    }
                }

                // === Per-position output: O = O_inter + O_intra ===
                for (LongType s = 0; s < C; ++s) {
                    const float* qRow = qc + s * D;
                    const LongType t = cs + s;
                    const LongType oBase = b * oS0 + t * oS1 + h * oS2;
                    const float interDecay = sd::math::sd_exp<float>(-decay * (s + 1));

                    // Intra-chunk scores: A[s2] = (Q[s,:] . K[s2,:]) * decay^(s-s2)
                    const LongType upper = isCausal ? (s + 1) : C;
                    for (LongType s2 = 0; s2 < upper; ++s2) {
                        const float* kRow = kc + s2 * D;
                        float dot = 0.0f;
                        PRAGMA_OMP_SIMD
                        for (LongType d = 0; d < D; ++d)
                            dot += qRow[d] * kRow[d];
                        A[s2] = dot * sd::math::sd_exp<float>(-decay * static_cast<float>(s - s2));
                    }

                    // Output[s, dv] = inter + intra
                    for (LongType dv = 0; dv < D; ++dv) {
                        // Inter: Q[s,:] @ S[:,dv] * decay^(s+1)
                        float inter = 0.0f;
                        PRAGMA_OMP_SIMD
                        for (LongType dk = 0; dk < D; ++dk)
                            inter += qRow[dk] * S[dk * D + dv];
                        inter *= interDecay;

                        // Intra: sum_s2 A[s2] * V[s2, dv]
                        float intra = 0.0f;
                        for (LongType s2 = 0; s2 < upper; ++s2)
                            intra += A[s2] * vc[s2 * D + dv];

                        oBuf[oBase + dv * oS3] = static_cast<T>(inter + intra);
                    }
                }

                // === State update: S = decay^C * S + sum_s decay^(C-1-s) * k_s (x) v_s ===
                const float chunkDecay = sd::math::sd_exp<float>(-decay * C);

                PRAGMA_OMP_SIMD
                for (LongType i = 0; i < D * D; ++i)
                    S[i] *= chunkDecay;

                for (LongType s = 0; s < C; ++s) {
                    const float posDecay = sd::math::sd_exp<float>(-decay * (C - 1 - s));
                    const float* kRow = kc + s * D;
                    const float* vRow = vc + s * D;
                    for (LongType dk = 0; dk < D; ++dk) {
                        const float kVal = kRow[dk] * posDecay;
                        float* sRow = S + dk * D;
                        PRAGMA_OMP_SIMD
                        for (LongType dv = 0; dv < D; ++dv)
                            sRow[dv] += kVal * vRow[dv];
                    }
                }
            }

            // Write float working state back to T* NDArray
            for (LongType i = 0; i < D; ++i)
                for (LongType j = 0; j < D; ++j)
                    stBuf[b * sS0 + h * sS1 + i * sS2 + j * sS3] = static_cast<T>(S[i * D + j]);
        }
    };

    samediff::Threads::parallel_tad(func, 0, B * H);
}

void lightningAttentionCpu(LaunchContext* context,
                           NDArray* query,
                           NDArray* key,
                           NDArray* value,
                           NDArray* decayRates,
                           NDArray* state,
                           NDArray* output,
                           bool isCausal) {
    NDArray::preparePrimaryUse({output, state}, {query, key, value, decayRates});

    BUILD_SINGLE_SELECTOR(query->dataType(), lightningAttentionCpuImpl_,
        (context, query, key, value, decayRates, state, output, isCausal), SD_FLOAT_TYPES);

    NDArray::registerPrimaryUse({output, state}, {query, key, value, decayRates});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
