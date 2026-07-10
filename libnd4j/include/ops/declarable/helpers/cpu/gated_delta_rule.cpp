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

#include <algorithm>
#include <cstring>
#include <vector>

namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static void gatedDeltaRule_(LaunchContext* context, NDArray* Q, NDArray* K, NDArray* V,
                             NDArray* beta, NDArray* gate, NDArray* stateIn,
                             NDArray* actualLen, NDArray* output, NDArray* stateOut) {
    const auto B = Q->sizeAt(0);
    const auto L = Q->sizeAt(1);
    const auto H = Q->sizeAt(2);
    const auto D_k = Q->sizeAt(3);
    const auto D_v = V->sizeAt(3);
    LongType effectiveLen = L;
    if (actualLen != nullptr) {
        effectiveLen = actualLen->e<LongType>(0);
        if (effectiveLen < 0) effectiveLen = 0;
        if (effectiveLen > L) effectiveLen = L;
    }

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
                const bool updateState = t < effectiveLen;
                float* sBase = stateBuf + (b * H + h) * D_v * D_k;

                if (updateState) {
                    const float exp_g_f = sd::math::sd_exp<float, float>(static_cast<float>(gateBuf[b * gS0 + t * gS1 + h * gS2]));
                    const float beta_f = static_cast<float>(betaBuf[b * bS0 + t * bS1 + h * bS2]);

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

// ============================================================================
// Chunked WY-representation prefill (CPU version)
// Matches the CUDA chunked kernel math exactly.
// Parallelizes over (b, h); processes chunks sequentially within each worker.
// Chunk size C=64 to match CUDA path and enable exact parity tests.
// ============================================================================

static constexpr int GDN_CHUNK_CPU = 64;

template <typename T>
static void gatedDeltaRuleChunked_(LaunchContext* context, NDArray* Q, NDArray* K, NDArray* V,
                                    NDArray* beta, NDArray* gate, NDArray* stateIn,
                                    NDArray* output, NDArray* stateOut) {
    const auto B   = Q->sizeAt(0);
    const auto L   = Q->sizeAt(1);
    const auto H   = Q->sizeAt(2);
    const auto D_k = Q->sizeAt(3);
    const auto D_v = V->sizeAt(3);

    const T* qBuf    = Q->bufferAsT<T>();
    const T* kBuf    = K->bufferAsT<T>();
    const T* vBuf    = V->bufferAsT<T>();
    const T* betaBuf = beta->bufferAsT<T>();
    const T* gateBuf = gate->bufferAsT<T>();
    T*       outBuf  = output->bufferAsT<T>();
    T*       soBuf   = stateOut->bufferAsT<T>();

    const auto qS0 = Q->strideAt(0), qS1 = Q->strideAt(1), qS2 = Q->strideAt(2), qS3 = Q->strideAt(3);
    const auto kS0 = K->strideAt(0), kS1 = K->strideAt(1), kS2 = K->strideAt(2), kS3 = K->strideAt(3);
    const auto vS0 = V->strideAt(0), vS1 = V->strideAt(1), vS2 = V->strideAt(2), vS3 = V->strideAt(3);
    const auto bS0 = beta->strideAt(0), bS1 = beta->strideAt(1), bS2 = beta->strideAt(2);
    const auto gS0 = gate->strideAt(0), gS1 = gate->strideAt(1), gS2 = gate->strideAt(2);
    const auto oS0 = output->strideAt(0), oS1 = output->strideAt(1), oS2 = output->strideAt(2), oS3 = output->strideAt(3);
    const auto ssS0 = stateOut->strideAt(0), ssS1 = stateOut->strideAt(1), ssS2 = stateOut->strideAt(2), ssS3 = stateOut->strideAt(3);

    const LongType C   = GDN_CHUNK_CPU;
    const LongType nC  = (L + C - 1) / C;

    // State: working as float32, transposed [B, H, D_v, D_k] for cache efficiency
    const LongType stateSize = B * H * D_v * D_k;
    static thread_local std::vector<float> tlStateBuf;
    if ((LongType)tlStateBuf.size() < stateSize)
        tlStateBuf.resize(stateSize);
    float* stateBuf = tlStateBuf.data();

    // Initialize state from stateIn (transpose [B,H,Dk,Dv] -> [B,H,Dv,Dk])
    if (stateIn != nullptr) {
        const T* siBuf = stateIn->bufferAsT<T>();
        const auto siS0 = stateIn->strideAt(0), siS1 = stateIn->strideAt(1);
        const auto siS2 = stateIn->strideAt(2), siS3 = stateIn->strideAt(3);
        for (LongType b = 0; b < B; ++b)
            for (LongType h = 0; h < H; ++h)
                for (LongType dk = 0; dk < D_k; ++dk)
                    for (LongType dv = 0; dv < D_v; ++dv)
                        stateBuf[((b * H + h) * D_v + dv) * D_k + dk] =
                            static_cast<float>(siBuf[b * siS0 + h * siS1 + dk * siS2 + dv * siS3]);
    } else {
        std::memset(stateBuf, 0, stateSize * sizeof(float));
    }

    // Per-bh scratch buffers — allocated once per parallel_tad worker, reused across chunks.
    // Each parallel thread gets its own scratch by allocating inside the lambda.
    // Sizes: A,M=[C*C], Kt,Qeff=[C*Dk], U0,MU0,U=[C*Dv], lcg,bet,eg,bg=[C]
    const LongType s_cc = C * C;
    const LongType s_dk = C * D_k;
    const LongType s_dv = C * D_v;

    auto func = PRAGMA_THREADS_FOR {
        // Per-worker heap scratch (allocated once per thread, reused across bh iterations
        // that this thread processes and across chunks within each bh).
        std::vector<float> wA(s_cc), wM(s_cc);
        std::vector<float> wKt(s_dk), wQeff(s_dk);
        std::vector<float> wU0(s_dv), wMU0(s_dv), wU(s_dv);
        std::vector<float> wLcg(C), wBet(C), wEg(C), wBg(C);

        float* A    = wA.data();
        float* M    = wM.data();
        float* Kt   = wKt.data();
        float* Qeff = wQeff.data();
        float* U0   = wU0.data();
        float* MU0  = wMU0.data();
        float* U    = wU.data();
        float* lcg  = wLcg.data();
        float* bet  = wBet.data();
        float* eg   = wEg.data();
        float* bg   = wBg.data();

        for (auto bh = start; bh < stop; ++bh) {
            const LongType b = bh / H;
            const LongType h = bh % H;
            float* sBase = stateBuf + bh * D_v * D_k;  // [D_v, D_k] transposed state

            for (LongType c = 0; c < nC; ++c) {
                const LongType t0 = c * C;
                const LongType tt = std::min(C, L - t0);  // valid tokens in this chunk

                // ---- Compute lcg, bet, eg, bg ----
                {
                    float acc = 0.f;
                    for (LongType i = 0; i < C; ++i) {
                        float gi = (i < tt) ? static_cast<float>(gateBuf[b * gS0 + (t0 + i) * gS1 + h * gS2]) : 0.f;
                        acc += gi;
                        lcg[i] = acc;
                        bet[i] = (i < tt) ? static_cast<float>(betaBuf[b * bS0 + (t0 + i) * bS1 + h * bS2]) : 0.f;
                        eg[i]  = sd::math::sd_exp<float,float>(lcg[i]);
                        bg[i]  = bet[i] * eg[i];
                    }
                }

                // ---- Compute A[i,j] = k[i].k[j] (raw dot, accumulate) then scale ----
                // Also compute M[i,j] = q[i].k[j] (lower inclusive)
                for (LongType i = 0; i < C; ++i) {
                    for (LongType j = 0; j < C; ++j) {
                        A[i * C + j] = 0.f;
                        M[i * C + j] = 0.f;
                    }
                }
                for (LongType i = 0; i < tt; ++i) {
                    const LongType kBase_i = b * kS0 + (t0 + i) * kS1 + h * kS2;
                    const LongType qBase_i = b * qS0 + (t0 + i) * qS1 + h * qS2;
                    for (LongType j = 0; j <= i; ++j) {
                        const LongType kBase_j = b * kS0 + (t0 + j) * kS1 + h * kS2;
                        float dotKK = 0.f, dotQK = 0.f;
                        for (LongType dk = 0; dk < D_k; ++dk) {
                            const float ki = static_cast<float>(kBuf[kBase_i + dk * kS3]);
                            const float kj = static_cast<float>(kBuf[kBase_j + dk * kS3]);
                            const float qi = static_cast<float>(qBuf[qBase_i + dk * qS3]);
                            dotKK += ki * kj;
                            dotQK += qi * kj;
                        }
                        // A[i,j] (strict lower): scale by beta[i] * exp(lcg[i]-lcg[j])
                        if (j < i)
                            A[i * C + j] = bet[i] * sd::math::sd_exp<float,float>(lcg[i] - lcg[j]) * dotKK;
                        // M[i,j] (lower inclusive): scale by exp(lcg[i]-lcg[j])
                        M[i * C + j] = sd::math::sd_exp<float,float>(lcg[i] - lcg[j]) * dotQK;
                    }
                }

                // ---- X = (I + A)^{-1} via forward substitution ----
                // Overwrites A with X in-place.
                // Row i: X[i,j] = delta_{ij} - sum_{k<i} A[i,k]*X[k,j]
                for (LongType i = 0; i < C; ++i) {
                    for (LongType j = 0; j < C; ++j) {
                        float x = (i == j) ? 1.f : 0.f;
                        for (LongType kk = 0; kk < i; ++kk)
                            x -= A[i * C + kk] * A[kk * C + j];  // A[kk,:] already == X[kk,:] at this point
                        A[i * C + j] = x;  // now A[i,:] = X[i,:]
                    }
                }
                // A is now X.

                // ---- Kt[i,dk] = sum_j X[i,j] * bg[j] * k[j,dk] ----
                // ---- U0[i,dv] = sum_j X[i,j] * bet[j] * v[j,dv] ----
                for (LongType i = 0; i < C; ++i) {
                    for (LongType dk = 0; dk < D_k; ++dk) Kt[i * D_k + dk] = 0.f;
                    for (LongType dv = 0; dv < D_v; ++dv) U0[i * D_v + dv] = 0.f;
                    for (LongType j = 0; j < tt; ++j) {
                        const float x_ij  = A[i * C + j];
                        const float bg_j  = bg[j];
                        const float bet_j = bet[j];
                        const LongType kBase_j = b * kS0 + (t0 + j) * kS1 + h * kS2;
                        const LongType vBase_j = b * vS0 + (t0 + j) * vS1 + h * vS2;
                        for (LongType dk = 0; dk < D_k; ++dk)
                            Kt[i * D_k + dk] += x_ij * bg_j * static_cast<float>(kBuf[kBase_j + dk * kS3]);
                        for (LongType dv = 0; dv < D_v; ++dv)
                            U0[i * D_v + dv] += x_ij * bet_j * static_cast<float>(vBuf[vBase_j + dv * vS3]);
                    }
                }

                // ---- MU0[i,dv] = sum_j M[i,j] * U0[j,dv] ----
                // ---- Qeff[i,dk] = eg[i]*q[i,dk] - sum_j M[i,j]*Kt[j,dk] ----
                for (LongType i = 0; i < C; ++i) {
                    for (LongType dv = 0; dv < D_v; ++dv) MU0[i * D_v + dv] = 0.f;
                    for (LongType dk = 0; dk < D_k; ++dk) Qeff[i * D_k + dk] = 0.f;
                    for (LongType j = 0; j <= i && j < C; ++j) {
                        const float m_ij = M[i * C + j];
                        for (LongType dv = 0; dv < D_v; ++dv)
                            MU0[i * D_v + dv] += m_ij * U0[j * D_v + dv];
                        for (LongType dk = 0; dk < D_k; ++dk)
                            Qeff[i * D_k + dk] -= m_ij * Kt[j * D_k + dk];
                    }
                    if (i < tt) {
                        const LongType qBase_i = b * qS0 + (t0 + i) * qS1 + h * qS2;
                        for (LongType dk = 0; dk < D_k; ++dk)
                            Qeff[i * D_k + dk] += eg[i] * static_cast<float>(qBuf[qBase_i + dk * qS3]);
                    }
                }

                // ---- Inter-chunk state update: sequential over i within chunk ----
                // U[i,dv]  = U0[i,dv] - dot(Kt[i,:], S[dv,:])
                // y[i,dv]  = MU0[i,dv] + dot(Qeff[i,:], S[dv,:])
                // S[dv,dk] = exp(lcg_last)*S[dv,dk] + sum_i r[i]*U[i,dv]*k[i,dk]
                const float lcg_last = lcg[C - 1];
                const float gl = sd::math::sd_exp<float,float>(lcg_last);

                // Compute U[i,dv] and write y[i,dv] before state update
                for (LongType i = 0; i < tt; ++i) {
                    for (LongType dv = 0; dv < D_v; ++dv) {
                        float* sRow = sBase + dv * D_k;
                        float accU = 0.f, accY = 0.f;
                        for (LongType dk = 0; dk < D_k; ++dk) {
                            accU += Kt[i * D_k + dk] * sRow[dk];
                            accY += Qeff[i * D_k + dk] * sRow[dk];
                        }
                        U[i * D_v + dv] = U0[i * D_v + dv] - accU;
                        outBuf[b * oS0 + (t0 + i) * oS1 + h * oS2 + dv * oS3] =
                            static_cast<T>(MU0[i * D_v + dv] + accY);
                    }
                }

                // State update: S[dv,dk] = gl*S[dv,dk] + sum_i r[i]*U[i,dv]*k[i,dk]
                for (LongType dv = 0; dv < D_v; ++dv) {
                    float* sRow = sBase + dv * D_k;
                    for (LongType dk = 0; dk < D_k; ++dk)
                        sRow[dk] *= gl;
                    for (LongType i = 0; i < tt; ++i) {
                        const float r_i = sd::math::sd_exp<float,float>(lcg_last - lcg[i]);
                        const float u_iv = U[i * D_v + dv];
                        const LongType kBase_i = b * kS0 + (t0 + i) * kS1 + h * kS2;
                        for (LongType dk = 0; dk < D_k; ++dk)
                            sRow[dk] += r_i * u_iv * static_cast<float>(kBuf[kBase_i + dk * kS3]);
                    }
                }
            }  // chunk loop

            // Write back state: transpose [B,H,Dv,Dk] -> [B,H,Dk,Dv]
            for (LongType dv = 0; dv < D_v; ++dv)
                for (LongType dk = 0; dk < D_k; ++dk)
                    soBuf[b * ssS0 + h * ssS1 + dk * ssS2 + dv * ssS3] =
                        static_cast<T>(sBase[dv * D_k + dk]);
        }  // bh loop
    };
    samediff::Threads::parallel_tad(func, 0, B * H);
}

void gatedDeltaRule(LaunchContext* context, NDArray* Q, NDArray* K, NDArray* V,
                     NDArray* beta, NDArray* gate, NDArray* stateIn,
                     NDArray* actualLen, NDArray* output, NDArray* stateOut) {
    NDArray::preparePrimaryUse({output, stateOut}, {Q, K, V, beta, gate, actualLen});
    if (stateIn != nullptr) NDArray::preparePrimaryUse({}, {stateIn});

    const auto L   = Q->sizeAt(1);
    // Chunked path: L >= C=64, no actualLen masking (chunked doesn't support partial masking)
    const bool useChunked = (L >= GDN_CHUNK_CPU) && (actualLen == nullptr);

    if (useChunked) {
        BUILD_SINGLE_SELECTOR(Q->dataType(), gatedDeltaRuleChunked_,
            (context, Q, K, V, beta, gate, stateIn, output, stateOut), SD_FLOAT_TYPES);
    } else {
        BUILD_SINGLE_SELECTOR(Q->dataType(), gatedDeltaRule_,
            (context, Q, K, V, beta, gate, stateIn, actualLen, output, stateOut), SD_FLOAT_TYPES);
    }

    NDArray::registerPrimaryUse({output, stateOut}, {Q, K, V, beta, gate, actualLen});
    if (stateIn != nullptr) NDArray::registerPrimaryUse({}, {stateIn});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
