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
#include <ops/op_types.h>
#include <ops/declarable/helpers/gated_delta_rule.h>
#include <ops/declarable/helpers/reproducible_math.h>

#include <algorithm>
#include <cstring>
#include <vector>

namespace sd {
namespace ops {
namespace helpers {

static constexpr int GDR_MAX_HEAD_DIM = 512;

/**
 * Materialize products, then reduce them through one fixed pairwise tree. This keeps
 * recurrent GDR rounding independent of host SIMD width and FMA contraction while
 * retaining the framework's promoted aggregate type and portable math operations.
 */
template <typename AccT>
static SD_INLINE AccT gatedDeltaReproducibleDot(
    const AccT* left, const AccT* right, int length) {
    AccT partials[GDR_MAX_HEAD_DIM];
    for (int index = 0; index < length; ++index) {
        partials[index] = reproducible::multiply<AccT>(left[index], right[index]);
    }

    int active = length;
    while (active > 1) {
        const int pairs = active / 2;
        for (int index = 0; index < pairs; ++index) {
            partials[index] = reproducible::add<AccT>(
                partials[index * 2], partials[index * 2 + 1]);
        }
        if ((active & 1) != 0) {
            partials[pairs] = partials[active - 1];
        }
        active = pairs + (active & 1);
    }
    return active == 0 ? static_cast<AccT>(0) : partials[0];
}

template <typename T>
static void gatedDeltaRule_(LaunchContext* context, NDArray* Q, NDArray* K, NDArray* V,
                             NDArray* beta, NDArray* gate, NDArray* stateIn,
                             NDArray* actualLen, NDArray* output, NDArray* stateOut) {
    using AccT = typename simdOps::AggregateType<T>::type;

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
    // AggregateType promotes HALF/BFLOAT16 to FLOAT and preserves wider input types.
    // Thread-local storage is reused across layers/tokens without fixing its dtype.
    const LongType stateSize = B * H * D_v * D_k;
    static thread_local std::vector<AccT> tlStateBuf;
    if (static_cast<LongType>(tlStateBuf.size()) < stateSize) {
        tlStateBuf.resize(stateSize);
    }
    AccT* stateBuf = tlStateBuf.data();

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
                            static_cast<AccT>(sInBuf[b * siS0 + h * siS1 + dk * siS2 + dv * siS3]);
    } else {
        std::memset(stateBuf, 0, stateSize * sizeof(AccT));
    }

    // Sequential over timesteps, parallel over batch*heads
    for (LongType t = 0; t < L; ++t) {
        auto func = PRAGMA_THREADS_FOR {
            for (auto bh = start; bh < stop; ++bh) {
                const LongType b = bh / H;
                const LongType h = bh % H;
                const bool updateState = t < effectiveLen;
                AccT* sBase = stateBuf + (b * H + h) * D_v * D_k;

                if (updateState) {
                    const AccT expGate = reproducible::exp<AccT>(
                        static_cast<AccT>(gateBuf[b * gS0 + t * gS1 + h * gS2]));
                    const AccT betaValue = static_cast<AccT>(
                        betaBuf[b * bS0 + t * bS1 + h * bS2]);

                    // Pre-load k vector in the promoted aggregate type.
                    // D_k is typically 64-128; GDR_MAX_HEAD_DIM covers all known models.
                    AccT kLocal[GDR_MAX_HEAD_DIM];
                    const LongType kBase = b * kS0 + t * kS1 + h * kS2;
                    for (LongType dk = 0; dk < D_k; ++dk)
                        kLocal[dk] = static_cast<AccT>(kBuf[kBase + dk * kS3]);

                    for (LongType dv = 0; dv < D_v; ++dv) {
                        AccT* sRow = sBase + dv * D_k;
                        const AccT prediction = gatedDeltaReproducibleDot<AccT>(
                            sRow, kLocal, static_cast<int>(D_k));
                        const AccT vValue = static_cast<AccT>(
                            vBuf[b * vS0 + t * vS1 + h * vS2 + dv * vS3]);
                        const AccT decayedPrediction = reproducible::multiply<AccT>(
                            expGate, prediction);
                        const AccT delta = reproducible::subtract<AccT>(
                            vValue, decayedPrediction);
                        const AccT betaDelta = reproducible::multiply<AccT>(
                            betaValue, delta);
                        for (LongType dk = 0; dk < D_k; ++dk) {
                            const AccT decayedState = reproducible::multiply<AccT>(
                                expGate, sRow[dk]);
                            const AccT stateUpdate = reproducible::multiply<AccT>(
                                betaDelta, kLocal[dk]);
                            sRow[dk] = reproducible::add<AccT>(decayedState, stateUpdate);
                        }
                    }
                }

                AccT qLocal[GDR_MAX_HEAD_DIM];
                const LongType qBase = b * qS0 + t * qS1 + h * qS2;
                for (LongType dk = 0; dk < D_k; ++dk)
                    qLocal[dk] = static_cast<AccT>(qBuf[qBase + dk * qS3]);

                for (LongType dv = 0; dv < D_v; ++dv) {
                    AccT* sRow = sBase + dv * D_k;
                    const AccT outputValue = gatedDeltaReproducibleDot<AccT>(
                        sRow, qLocal, static_cast<int>(D_k));
                    outBuf[b * oS0 + t * oS1 + h * oS2 + dv * oS3] =
                        static_cast<T>(outputValue);
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
    using AccT = typename simdOps::AggregateType<T>::type;

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

    // State uses the framework aggregate type and a transposed [B, H, D_v, D_k] layout.
    const LongType stateSize = B * H * D_v * D_k;
    static thread_local std::vector<AccT> tlStateBuf;
    if ((LongType)tlStateBuf.size() < stateSize)
        tlStateBuf.resize(stateSize);
    AccT* stateBuf = tlStateBuf.data();

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
                            static_cast<AccT>(siBuf[b * siS0 + h * siS1 + dk * siS2 + dv * siS3]);
    } else {
        std::memset(stateBuf, 0, stateSize * sizeof(AccT));
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
        std::vector<AccT> wA(s_cc), wM(s_cc);
        std::vector<AccT> wKt(s_dk), wQeff(s_dk);
        std::vector<AccT> wU0(s_dv), wMU0(s_dv), wU(s_dv);
        std::vector<AccT> wLcg(C), wBet(C), wEg(C), wBg(C);

        AccT* A    = wA.data();
        AccT* M    = wM.data();
        AccT* Kt   = wKt.data();
        AccT* Qeff = wQeff.data();
        AccT* U0   = wU0.data();
        AccT* MU0  = wMU0.data();
        AccT* U    = wU.data();
        AccT* lcg  = wLcg.data();
        AccT* bet  = wBet.data();
        AccT* eg   = wEg.data();
        AccT* bg   = wBg.data();

        for (auto bh = start; bh < stop; ++bh) {
            const LongType b = bh / H;
            const LongType h = bh % H;
            AccT* sBase = stateBuf + bh * D_v * D_k;  // [D_v, D_k] transposed state

            for (LongType c = 0; c < nC; ++c) {
                const LongType t0 = c * C;
                const LongType tt = std::min(C, L - t0);  // valid tokens in this chunk

                // ---- Compute lcg, bet, eg, bg ----
                {
                    AccT acc = static_cast<AccT>(0);
                    for (LongType i = 0; i < C; ++i) {
                        AccT gi = (i < tt)
                            ? static_cast<AccT>(gateBuf[b * gS0 + (t0 + i) * gS1 + h * gS2])
                            : static_cast<AccT>(0);
                        acc = reproducible::add<AccT>(acc, gi);
                        lcg[i] = acc;
                        bet[i] = (i < tt)
                            ? static_cast<AccT>(betaBuf[b * bS0 + (t0 + i) * bS1 + h * bS2])
                            : static_cast<AccT>(0);
                        eg[i]  = reproducible::exp<AccT>(lcg[i]);
                        bg[i]  = reproducible::multiply<AccT>(bet[i], eg[i]);
                    }
                }

                // ---- Compute A[i,j] = k[i].k[j] (raw dot, accumulate) then scale ----
                // Also compute M[i,j] = q[i].k[j] (lower inclusive)
                for (LongType i = 0; i < C; ++i) {
                    for (LongType j = 0; j < C; ++j) {
                        A[i * C + j] = static_cast<AccT>(0);
                        M[i * C + j] = static_cast<AccT>(0);
                    }
                }
                for (LongType i = 0; i < tt; ++i) {
                    const LongType kBase_i = b * kS0 + (t0 + i) * kS1 + h * kS2;
                    const LongType qBase_i = b * qS0 + (t0 + i) * qS1 + h * qS2;
                    for (LongType j = 0; j <= i; ++j) {
                        const LongType kBase_j = b * kS0 + (t0 + j) * kS1 + h * kS2;
                        AccT dotKK = static_cast<AccT>(0);
                        AccT dotQK = static_cast<AccT>(0);
                        // Match CUDA's staged 32-value tile topology exactly:
                        // left-fold within each tile, then merge tile totals.
                        for (LongType tile = 0; tile < D_k; tile += 32) {
                            const LongType tileEnd = std::min(tile + 32, D_k);
                            AccT tileKK = static_cast<AccT>(0);
                            AccT tileQK = static_cast<AccT>(0);
                            for (LongType dk = tile; dk < tileEnd; ++dk) {
                                const AccT ki = static_cast<AccT>(kBuf[kBase_i + dk * kS3]);
                                const AccT kj = static_cast<AccT>(kBuf[kBase_j + dk * kS3]);
                                const AccT qi = static_cast<AccT>(qBuf[qBase_i + dk * qS3]);
                                tileKK = reproducible::add<AccT>(
                                    tileKK, reproducible::multiply<AccT>(ki, kj));
                                tileQK = reproducible::add<AccT>(
                                    tileQK, reproducible::multiply<AccT>(qi, kj));
                            }
                            dotKK = reproducible::add<AccT>(dotKK, tileKK);
                            dotQK = reproducible::add<AccT>(dotQK, tileQK);
                        }
                        const AccT decay = reproducible::exp<AccT>(
                            reproducible::subtract<AccT>(lcg[i], lcg[j]));
                        // A[i,j] (strict lower): scale by beta[i] * exp(lcg[i]-lcg[j])
                        if (j < i)
                            A[i * C + j] = reproducible::multiply<AccT>(
                                bet[i], reproducible::multiply<AccT>(
                                    decay, dotKK));
                        // M[i,j] (lower inclusive): scale by exp(lcg[i]-lcg[j])
                        M[i * C + j] = reproducible::multiply<AccT>(decay, dotQK);
                    }
                }

                // ---- X = (I + A)^{-1} via forward substitution ----
                // Overwrites A with X in-place.
                // Row i: X[i,j] = delta_{ij} - sum_{k<i} A[i,k]*X[k,j]
                for (LongType i = 0; i < C; ++i) {
                    for (LongType j = 0; j < C; ++j) {
                        AccT x = (i == j) ? static_cast<AccT>(1) : static_cast<AccT>(0);
                        for (LongType kk = 0; kk < i; ++kk) {
                            x = reproducible::subtract<AccT>(
                                x, reproducible::multiply<AccT>(
                                    A[i * C + kk], A[kk * C + j]));
                        }
                        A[i * C + j] = x;  // now A[i,:] = X[i,:]
                    }
                }
                // A is now X.

                // ---- Kt[i,dk] = sum_j X[i,j] * bg[j] * k[j,dk] ----
                // ---- U0[i,dv] = sum_j X[i,j] * bet[j] * v[j,dv] ----
                for (LongType i = 0; i < C; ++i) {
                    for (LongType dk = 0; dk < D_k; ++dk)
                        Kt[i * D_k + dk] = static_cast<AccT>(0);
                    for (LongType dv = 0; dv < D_v; ++dv)
                        U0[i * D_v + dv] = static_cast<AccT>(0);
                    for (LongType j = 0; j < tt; ++j) {
                        const AccT x_ij  = A[i * C + j];
                        const AccT bg_j  = bg[j];
                        const AccT bet_j = bet[j];
                        const LongType kBase_j = b * kS0 + (t0 + j) * kS1 + h * kS2;
                        const LongType vBase_j = b * vS0 + (t0 + j) * vS1 + h * vS2;
                        for (LongType dk = 0; dk < D_k; ++dk) {
                            Kt[i * D_k + dk] = reproducible::add<AccT>(
                                Kt[i * D_k + dk], reproducible::multiply<AccT>(
                                    x_ij, reproducible::multiply<AccT>(
                                        bg_j, static_cast<AccT>(kBuf[kBase_j + dk * kS3]))));
                        }
                        for (LongType dv = 0; dv < D_v; ++dv) {
                            U0[i * D_v + dv] = reproducible::add<AccT>(
                                U0[i * D_v + dv], reproducible::multiply<AccT>(
                                    x_ij, reproducible::multiply<AccT>(
                                        bet_j, static_cast<AccT>(vBuf[vBase_j + dv * vS3]))));
                        }
                    }
                }

                // ---- MU0[i,dv] = sum_j M[i,j] * U0[j,dv] ----
                // ---- Qeff[i,dk] = eg[i]*q[i,dk] - sum_j M[i,j]*Kt[j,dk] ----
                for (LongType i = 0; i < C; ++i) {
                    for (LongType dv = 0; dv < D_v; ++dv)
                        MU0[i * D_v + dv] = static_cast<AccT>(0);
                    for (LongType dk = 0; dk < D_k; ++dk)
                        Qeff[i * D_k + dk] = static_cast<AccT>(0);
                    for (LongType j = 0; j <= i && j < C; ++j) {
                        const AccT m_ij = M[i * C + j];
                        for (LongType dv = 0; dv < D_v; ++dv) {
                            MU0[i * D_v + dv] = reproducible::add<AccT>(
                                MU0[i * D_v + dv], reproducible::multiply<AccT>(
                                    m_ij, U0[j * D_v + dv]));
                        }
                        for (LongType dk = 0; dk < D_k; ++dk) {
                            Qeff[i * D_k + dk] = reproducible::subtract<AccT>(
                                Qeff[i * D_k + dk], reproducible::multiply<AccT>(
                                    m_ij, Kt[j * D_k + dk]));
                        }
                    }
                    if (i < tt) {
                        const LongType qBase_i = b * qS0 + (t0 + i) * qS1 + h * qS2;
                        for (LongType dk = 0; dk < D_k; ++dk) {
                            Qeff[i * D_k + dk] = reproducible::add<AccT>(
                                Qeff[i * D_k + dk], reproducible::multiply<AccT>(
                                    eg[i], static_cast<AccT>(qBuf[qBase_i + dk * qS3])));
                        }
                    }
                }

                // ---- Inter-chunk state update: sequential over i within chunk ----
                // U[i,dv]  = U0[i,dv] - dot(Kt[i,:], S[dv,:])
                // y[i,dv]  = MU0[i,dv] + dot(Qeff[i,:], S[dv,:])
                // S[dv,dk] = exp(lcg_last)*S[dv,dk] + sum_i r[i]*U[i,dv]*k[i,dk]
                const AccT lcg_last = lcg[C - 1];
                const AccT gl = reproducible::exp<AccT>(lcg_last);

                // Compute U[i,dv] and write y[i,dv] before state update
                for (LongType i = 0; i < tt; ++i) {
                    for (LongType dv = 0; dv < D_v; ++dv) {
                        AccT* sRow = sBase + dv * D_k;
                        AccT accU = static_cast<AccT>(0);
                        AccT accY = static_cast<AccT>(0);
                        for (LongType dk = 0; dk < D_k; ++dk) {
                            accU = reproducible::add<AccT>(
                                accU, reproducible::multiply<AccT>(
                                    Kt[i * D_k + dk], sRow[dk]));
                            accY = reproducible::add<AccT>(
                                accY, reproducible::multiply<AccT>(
                                    Qeff[i * D_k + dk], sRow[dk]));
                        }
                        U[i * D_v + dv] = reproducible::subtract<AccT>(
                            U0[i * D_v + dv], accU);
                        outBuf[b * oS0 + (t0 + i) * oS1 + h * oS2 + dv * oS3] =
                            static_cast<T>(reproducible::add<AccT>(
                                MU0[i * D_v + dv], accY));
                    }
                }

                // State update: S[dv,dk] = gl*S[dv,dk] + sum_i r[i]*U[i,dv]*k[i,dk]
                for (LongType dv = 0; dv < D_v; ++dv) {
                    AccT* sRow = sBase + dv * D_k;
                    for (LongType dk = 0; dk < D_k; ++dk) {
                        sRow[dk] = reproducible::multiply<AccT>(sRow[dk], gl);
                    }
                    for (LongType i = 0; i < tt; ++i) {
                        const AccT r_i = reproducible::exp<AccT>(
                            reproducible::subtract<AccT>(lcg_last, lcg[i]));
                        const AccT u_iv = U[i * D_v + dv];
                        const LongType kBase_i = b * kS0 + (t0 + i) * kS1 + h * kS2;
                        for (LongType dk = 0; dk < D_k; ++dk) {
                            sRow[dk] = reproducible::add<AccT>(
                                sRow[dk], reproducible::multiply<AccT>(
                                    r_i, reproducible::multiply<AccT>(
                                        u_iv, static_cast<AccT>(kBuf[kBase_i + dk * kS3]))));
                        }
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
    if (Q->sizeAt(3) > GDR_MAX_HEAD_DIM) {
        THROW_EXCEPTION("gatedDeltaRule: key head dimension exceeds supported maximum");
    }
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
