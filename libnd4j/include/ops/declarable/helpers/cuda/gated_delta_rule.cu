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

#include <cuda_runtime.h>
#include <helpers/DebugHelper.h>
#include <array/NDArray.h>
#include <math/templatemath.h>
#include <memory/cuda/CudaMemoryPool.h>
#include <ops/op_types.h>
#include <types/float16.h>
#include <ops/declarable/helpers/gated_delta_rule.h>
#include <ops/declarable/helpers/reproducible_math.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <string>
#include <type_traits>

namespace sd {
namespace ops {
namespace helpers {

static constexpr int GDR_CUDA_MAX_HEAD_DIM = 512;
static constexpr int GDR_CUDA_PAIRWISE_LEVELS = 10;

template <typename AccT, typename T>
static SD_DEVICE SD_INLINE AccT gatedDeltaCudaReproducibleDot(
    const AccT* state, LongType stateStride,
    const T* vector, LongType vectorStride, LongType length) {
    AccT levels[GDR_CUDA_PAIRWISE_LEVELS];
    for (LongType index = 0; index < length; ++index) {
        AccT value = reproducible::multiply<AccT>(
            state[index * stateStride], static_cast<AccT>(vector[index * vectorStride]));
        LongType completed = index + 1;
        int level = 0;
        while ((completed & 1) == 0) {
            value = reproducible::add<AccT>(levels[level], value);
            completed >>= 1;
            ++level;
        }
        levels[level] = value;
    }

    AccT result = static_cast<AccT>(0);
    bool initialized = false;
    for (int level = 0; level < GDR_CUDA_PAIRWISE_LEVELS; ++level) {
        if ((length & (static_cast<LongType>(1) << level)) == 0) continue;
        result = initialized
            ? reproducible::add<AccT>(levels[level], result)
            : levels[level];
        initialized = true;
    }
    return result;
}

// One block per (batch, head). Threads cover D_v dimension.
// Sequential over timesteps (recurrent dependency). AggregateType promotes
// HALF/BFLOAT16 accumulation while preserving wider input types.
template <typename T>
SD_KERNEL void gatedDeltaRuleKernel(
    const T* __restrict__ q,
    const T* __restrict__ k,
    const T* __restrict__ v,
    const T* __restrict__ betaArr,
    const T* __restrict__ gateArr,
    const LongType* __restrict__ actualLen,
    typename simdOps::AggregateType<T>::type* __restrict__ state,
    T* __restrict__ out,
    const LongType B, const LongType L, const LongType H,
    const LongType D_k, const LongType D_v,
    const LongType t,
    const LongType qS0, const LongType qS1, const LongType qS2, const LongType qS3,
    const LongType kS0, const LongType kS1, const LongType kS2, const LongType kS3,
    const LongType vS0, const LongType vS1, const LongType vS2, const LongType vS3,
    const LongType bS0, const LongType bS1, const LongType bS2,
    const LongType gS0, const LongType gS1, const LongType gS2,
    const LongType oS0, const LongType oS1, const LongType oS2, const LongType oS3) {

    using AccT = typename simdOps::AggregateType<T>::type;

    const LongType bh = blockIdx.x;
    if (bh >= B * H) return;

    const LongType b = bh / H;
    const LongType h = bh % H;

    LongType effectiveLen = L;
    if (actualLen != nullptr) {
        effectiveLen = actualLen[0];
        if (effectiveLen < 0) effectiveLen = 0;
        if (effectiveLen > L) effectiveLen = L;
    }
    const bool updateState = t < effectiveLen;
    __shared__ AccT expGateShared;
    if (threadIdx.x == 0) {
        expGateShared = updateState
            ? reproducible::exp<AccT>(
                static_cast<AccT>(gateArr[b * gS0 + t * gS1 + h * gS2]))
            : static_cast<AccT>(1);
    }
    __syncthreads();
    const AccT expGate = expGateShared;
    const AccT betaValue = updateState
        ? static_cast<AccT>(betaArr[b * bS0 + t * bS1 + h * bS2])
        : static_cast<AccT>(0);
    AccT* sPtr = state + (b * H + h) * D_k * D_v;

    for (LongType dv = threadIdx.x; dv < D_v; dv += blockDim.x) {
        if (updateState) {
            const LongType kBase = b * kS0 + t * kS1 + h * kS2;
            const AccT prediction = gatedDeltaCudaReproducibleDot<AccT, T>(
                sPtr + dv, D_v, k + kBase, kS3, D_k);

            const AccT delta = reproducible::subtract<AccT>(
                static_cast<AccT>(v[b * vS0 + t * vS1 + h * vS2 + dv * vS3]),
                reproducible::multiply<AccT>(expGate, prediction));
            const AccT betaDelta = reproducible::multiply<AccT>(betaValue, delta);

            for (LongType dk = 0; dk < D_k; ++dk) {
                const AccT kValue = static_cast<AccT>(k[b * kS0 + t * kS1 + h * kS2 + dk * kS3]);
                sPtr[dk * D_v + dv] = reproducible::add<AccT>(
                    reproducible::multiply<AccT>(expGate, sPtr[dk * D_v + dv]),
                    reproducible::multiply<AccT>(betaDelta, kValue));
            }
        }

        const LongType qBase = b * qS0 + t * qS1 + h * qS2;
        const AccT outputValue = gatedDeltaCudaReproducibleDot<AccT, T>(
            sPtr + dv, D_v, q + qBase, qS3, D_k);
        out[b * oS0 + t * oS1 + h * oS2 + dv * oS3] = static_cast<T>(outputValue);
    }
}

template <typename Source, typename Target>
SD_KERNEL void convertStateKernel(
    const Source* __restrict__ src,
    Target* __restrict__ dst,
    const LongType total) {
    const LongType idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        dst[idx] = static_cast<Target>(src[idx]);
    }
}

template <typename T>
static void launchGatedDeltaRule(
    const T* q, const T* k, const T* v,
    const T* betaArr, const T* gateArr, const LongType* actualLen,
    typename simdOps::AggregateType<T>::type* workingState, T* out,
    LongType B, LongType L, LongType H, LongType D_k, LongType D_v,
    LongType qS0, LongType qS1, LongType qS2, LongType qS3,
    LongType kS0, LongType kS1, LongType kS2, LongType kS3,
    LongType vS0, LongType vS1, LongType vS2, LongType vS3,
    LongType bS0, LongType bS1, LongType bS2,
    LongType gS0, LongType gS1, LongType gS2,
    LongType oS0, LongType oS1, LongType oS2, LongType oS3,
    cudaStream_t stream) {

    int numBlocks = B * H;
    int threadsPerBlock = 256;
    if (D_v < threadsPerBlock) {
        threadsPerBlock = ((D_v + 31) / 32) * 32;
        if (threadsPerBlock < 32) threadsPerBlock = 32;
    }

    for (LongType t = 0; t < L; ++t) {
        gatedDeltaRuleKernel<T><<<numBlocks, threadsPerBlock, 0, stream>>>(
            q, k, v, betaArr, gateArr, actualLen, workingState, out,
            B, L, H, D_k, D_v, t,
            qS0, qS1, qS2, qS3, kS0, kS1, kS2, kS3,
            vS0, vS1, vS2, vS3, bS0, bS1, bS2,
            gS0, gS1, gS2, oS0, oS1, oS2, oS3);
    }
    DebugHelper::checkGlobalErrorCode("gatedDeltaRuleKernel failed");
}

// ============================================================================
// Chunked WY-representation prefill (arXiv:2412.06464 §4, faster for T >= 64)
// ============================================================================
//
// Two-kernel decomposition:
//   Kernel A: (b, h, chunk) -> compute intra-chunk quantities in shared memory
//             lcg = cumsum of log(gate) within chunk
//             A = beta*exp(lcg[i]-lcg[j])*k[i].k[j]  (strict lower tri)
//             X = (I+A)^-1 via forward substitution
//             Kt[i]  = X[i,:] . (beta*exp(lcg)*k)   -- effective key
//             U0[i]  = X[i,:] . (beta*v)             -- intra-chunk value contrib
//             M[i,j] = q[i].k[j]*exp(lcg[i]-lcg[j]) (lower incl.)
//             MU0[i] = M[i,:] . U0                   -- intra-chunk output
//             Qeff[i]= exp(lcg[i])*q[i] - M[i,:].Kt -- inter-chunk readout query
//
//   Kernel B: (b, h, dv_block), sequential over chunks
//             U[i,dv]  = U0[i,dv] - dot(Kt[i,:],  S[dv,:])
//             y[i,dv]  = MU0[i,dv] + dot(Qeff[i,:], S[dv,:])
//             S[dv,dk] = exp(lcg_last)*S[dv,dk] + sum_i r[i]*U[i,dv]*k[i,dk]
//               where r[i] = exp(lcg_last - lcg[i])
//
// Arithmetic uses AggregateType<T>. Gate input is already log-domain (g), so
// exp(g) == decay factor, matching the sequential kernel exactly.
//
// Constraint: C=64 (chunk size), DV_BLK=32 (Dv block in kernel B).
// Kernel A requires D_k <= 512 for staged tile loops (Dk=64/128 in practice).
// ============================================================================

static constexpr int GDN_CHUNK = 64;
static constexpr int GDN_DV_BLK = 32;

// ---------------------------------------------------------------------------
// Kernel A — intra-chunk (b, h, chunk) one-block-per-chunk
// Grid: (nC, B*H, 1), Block: (256, 1, 1)
// ---------------------------------------------------------------------------
template<typename T>
SD_KERNEL void gdnChunkIntraKernel(
    const T*     __restrict__ q,        // [B,L,H,Dk]  strides: qS0,qS1,qS2,qS3
    const T*     __restrict__ k,        // [B,L,H,Dk]
    const T*     __restrict__ v,        // [B,L,H,Dv]  strides: vS0,vS1,vS2,vS3
    const T*     __restrict__ betaArr,  // [B,L,H]
    const T*     __restrict__ gateArr,  // [B,L,H]
    typename simdOps::AggregateType<T>::type* __restrict__ Kt,
    typename simdOps::AggregateType<T>::type* __restrict__ U0,
    typename simdOps::AggregateType<T>::type* __restrict__ MU0,
    typename simdOps::AggregateType<T>::type* __restrict__ Qeff,
    typename simdOps::AggregateType<T>::type* __restrict__ lcgOut,
    LongType B, LongType L, LongType H, LongType Dk, LongType Dv,
    LongType nC,
    LongType qS0, LongType qS1, LongType qS2, LongType qS3,
    LongType kS0, LongType kS1, LongType kS2, LongType kS3,
    LongType vS0, LongType vS1, LongType vS2, LongType vS3,
    LongType bS0, LongType bS1, LongType bS2,
    LongType gS0, LongType gS1, LongType gS2)
{
    using AccT = typename simdOps::AggregateType<T>::type;
    // Each block handles one (chunk, bh) pair
    const int c   = blockIdx.x;   // chunk index
    const int bh  = blockIdx.y;   // b*H + h
    const int b   = bh / (int)H;
    const int h   = bh % (int)H;
    const int t0  = c * GDN_CHUNK;
    const int tt  = (int)min((LongType)GDN_CHUNK, L - (LongType)t0);  // valid tokens in this chunk
    const int tid = threadIdx.x;  // 0..255

    extern __shared__ char smem_raw[];
    AccT* lcg_s = reinterpret_cast<AccT*>(smem_raw);                         // [C]
    AccT* bet_s = lcg_s + GDN_CHUNK;                                        // [C]
    AccT* eg_s  = bet_s + GDN_CHUNK;                                        // [C]
    AccT* bg_s  = eg_s  + GDN_CHUNK;                                        // [C]
    // As[i][j]: row-major [C][C+1]
    AccT* As    = bg_s  + GDN_CHUNK;                                        // [C*(C+1)]
    // kst/qst: [C][Dk_tile] — we stage 32-wide d-tiles
    AccT* kst   = As    + GDN_CHUNK * (GDN_CHUNK + 1);                     // [C*36] (32+4 pad)
    AccT* qst   = kst   + GDN_CHUNK * 36;                                  // [C*36]

    // ---- Compute lcg (cumulative log-gate) and beta — sequential in thread 0 ----
    if (tid == 0) {
        AccT acc = static_cast<AccT>(0);
        for (int i = 0; i < GDN_CHUNK; ++i) {
            AccT gi = (i < tt)
                ? static_cast<AccT>(gateArr[b * gS0 + (LongType)(t0 + i) * gS1 + h * gS2])
                : static_cast<AccT>(0);
            acc = reproducible::add<AccT>(acc, gi);
            lcg_s[i] = acc;
        }
    }
    if (tid == 32) {
        for (int i = 0; i < GDN_CHUNK; ++i) {
            bet_s[i] = (i < tt)
                ? static_cast<AccT>(betaArr[b * bS0 + (LongType)(t0 + i) * bS1 + h * bS2])
                : static_cast<AccT>(0);
        }
    }
    __syncthreads();

    if (tid < GDN_CHUNK) {
        eg_s[tid] = reproducible::exp<AccT>(lcg_s[tid]);
        bg_s[tid] = reproducible::multiply<AccT>(bet_s[tid], eg_s[tid]);
    }
    __syncthreads();

    // ---- Compute A = KK^T and M = qK^T via staged 32-wide d-tiles ----
    // Thread tid owns 16 (i,j) pairs with flat index p = tid*16+r
    // Each pair (i,j): contributes to A if j<i, to M if j<=i
    AccT accA[16] = {};
    AccT accM[16] = {};

    // k/q base pointers for this (b, h)
    const T* k_bh = k + (LongType)b * kS0 + (LongType)h * kS2;
    const T* q_bh = q + (LongType)b * qS0 + (LongType)h * qS2;

    for (LongType dt = 0; dt < Dk; dt += 32) {
        const LongType tile_width = min((LongType)32, Dk - dt);
        // Cooperative load of k/q tiles: C rows x tile_width cols
        for (int p = tid; p < GDN_CHUNK * 32; p += 256) {
            const int r  = p / 32;
            const int cc = p % 32;
            AccT kv = static_cast<AccT>(0);
            AccT qv = static_cast<AccT>(0);
            if (r < tt && cc < (int)tile_width) {
                const LongType tidx = (LongType)(t0 + r) * kS1 + (dt + cc) * kS3;
                const LongType qidx = (LongType)(t0 + r) * qS1 + (dt + cc) * qS3;
                kv = static_cast<AccT>(k_bh[tidx]);
                qv = static_cast<AccT>(q_bh[qidx]);
            }
            kst[r * 36 + cc] = kv;
            qst[r * 36 + cc] = qv;
        }
        __syncthreads();

        for (int r = 0; r < 16; ++r) {
            const int p = tid * 16 + r;
            const int i = p / GDN_CHUNK;
            const int j = p % GDN_CHUNK;
            if (j <= i && i < tt) {
                AccT da = static_cast<AccT>(0);
                AccT dm = static_cast<AccT>(0);
                for (int dd = 0; dd < 32; ++dd) {
                    const AccT kjd = kst[j * 36 + dd];
                    da = reproducible::add<AccT>(
                        da, reproducible::multiply<AccT>(kst[i * 36 + dd], kjd));
                    dm = reproducible::add<AccT>(
                        dm, reproducible::multiply<AccT>(qst[i * 36 + dd], kjd));
                }
                accA[r] = reproducible::add<AccT>(accA[r], da);
                accM[r] = reproducible::add<AccT>(accM[r], dm);
            }
        }
        __syncthreads();
    }

    // ---- Write A (strict lower triangular, scaled by beta*decay) into As ----
    for (int r = 0; r < 16; ++r) {
        const int p = tid * 16 + r;
        const int i = p / GDN_CHUNK;
        const int j = p % GDN_CHUNK;
        AccT a = static_cast<AccT>(0);
        if (j < i && i < tt)
            a = reproducible::multiply<AccT>(
                bet_s[i], reproducible::multiply<AccT>(
                    reproducible::exp<AccT>(
                        reproducible::subtract<AccT>(lcg_s[i], lcg_s[j])),
                    accA[r]));
        As[i * (GDN_CHUNK + 1) + j] = a;
    }
    __syncthreads();

    // ---- X = (I + A)^{-1} via forward substitution (in-place, overwrites As) ----
    // Row i: X[i,j] = delta_{ij} - sum_{k<i} A[i,k]*X[k,j]
    // After this, As[i][0..C-1] = X[i,:]
    for (int i = 0; i < GDN_CHUNK; ++i) {
        AccT x = static_cast<AccT>(0);
        if (tid < GDN_CHUNK) {
            x = (tid == i) ? static_cast<AccT>(1) : static_cast<AccT>(0);
            for (int jj = 0; jj < i; ++jj) {
                x = reproducible::subtract<AccT>(
                    x, reproducible::multiply<AccT>(
                        As[i * (GDN_CHUNK + 1) + jj], As[jj * (GDN_CHUNK + 1) + tid]));
            }
        }
        __syncthreads();
        if (tid < GDN_CHUNK)
            As[i * (GDN_CHUNK + 1) + tid] = x;
        __syncthreads();
    }

    // ---- Write lcg to output ----
    {
        const LongType chunk_off = ((LongType)b * H + h) * nC * GDN_CHUNK + (LongType)c * GDN_CHUNK;
        if (tid < GDN_CHUNK)
            lcgOut[chunk_off + tid] = lcg_s[tid];
    }

    // ---- Kt[i] = X[i,:] . (bg*k) ;  U0[i] = X[i,:] . (beta*v) ----
    // Thread owns column d (mod Dk or Dv) and strides over rows.
    // Output layout: [B,H,nC,C,Dk] for Kt, [B,H,nC,C,Dv] for U0
    {
        const LongType base_off = ((LongType)b * H + h) * nC + c;
        const LongType kt_base  = base_off * (LongType)GDN_CHUNK * Dk;
        const LongType u0_base  = base_off * (LongType)GDN_CHUNK * Dv;

        // Kt: thread owns column d_col = tid % Dk, strides rows by (256/Dk)
        {
            const LongType d_col = tid % Dk;
            const LongType row_stride = 256 / Dk;  // e.g. Dk=64->stride=4, Dk=128->stride=2
            for (LongType i = tid / Dk; i < GDN_CHUNK; i += row_stride) {
                AccT aK = static_cast<AccT>(0);
                for (LongType j = 0; j < (LongType)tt; ++j) {
                    const AccT x_ij = As[(LongType)i * (GDN_CHUNK + 1) + j];
                    const LongType kidx = (LongType)b * kS0 + (LongType)(t0 + (int)j) * kS1
                                          + (LongType)h * kS2 + d_col * kS3;
                    aK = reproducible::add<AccT>(
                        aK, reproducible::multiply<AccT>(
                            x_ij, reproducible::multiply<AccT>(
                                bg_s[j], static_cast<AccT>(k[kidx]))));
                }
                Kt[kt_base + i * Dk + d_col] = aK;
            }
        }

        // U0: thread owns column d_col_v = tid % Dv
        {
            const LongType d_col_v = tid % Dv;
            const LongType row_stride_v = 256 / Dv;
            for (LongType i = tid / Dv; i < GDN_CHUNK; i += row_stride_v) {
                AccT aV = static_cast<AccT>(0);
                for (LongType j = 0; j < (LongType)tt; ++j) {
                    const AccT x_ij = As[(LongType)i * (GDN_CHUNK + 1) + j];
                    const LongType vidx = (LongType)b * vS0 + (LongType)(t0 + (int)j) * vS1
                                          + (LongType)h * vS2 + d_col_v * vS3;
                    aV = reproducible::add<AccT>(
                        aV, reproducible::multiply<AccT>(
                            x_ij, reproducible::multiply<AccT>(
                                bet_s[j], static_cast<AccT>(v[vidx]))));
                }
                U0[u0_base + i * Dv + d_col_v] = aV;
            }
        }
    }
    __syncthreads();

    // ---- Overwrite As with M (reuse As buffer; X no longer needed) ----
    for (int r = 0; r < 16; ++r) {
        const int p = tid * 16 + r;
        const int i = p / GDN_CHUNK;
        const int j = p % GDN_CHUNK;
        AccT m = static_cast<AccT>(0);
        if (j <= i && i < tt)
            m = reproducible::multiply<AccT>(
                accM[r], reproducible::exp<AccT>(
                    reproducible::subtract<AccT>(lcg_s[i], lcg_s[j])));
        As[i * (GDN_CHUNK + 1) + j] = m;
    }
    __syncthreads();

    // ---- MU0[i] = M[i,:] . U0 ;  Qeff[i] = exp(lcg[i])*q[i] - M[i,:].Kt ----
    {
        const LongType base_off2 = ((LongType)b * H + h) * nC + c;
        const LongType mu0_base  = base_off2 * (LongType)GDN_CHUNK * Dv;
        const LongType qe_base   = base_off2 * (LongType)GDN_CHUNK * Dk;
        const LongType kt_base2  = base_off2 * (LongType)GDN_CHUNK * Dk;
        const LongType u0_base2  = base_off2 * (LongType)GDN_CHUNK * Dv;

        // MU0: thread owns d_col_v = tid % Dv
        {
            const LongType d_col_v = tid % Dv;
            const LongType row_stride_v = 256 / Dv;
            for (LongType i = tid / Dv; i < GDN_CHUNK; i += row_stride_v) {
                AccT aU = static_cast<AccT>(0);
                for (LongType j = 0; j < (LongType)GDN_CHUNK; ++j) {
                    aU = reproducible::add<AccT>(
                        aU, reproducible::multiply<AccT>(
                            As[(LongType)i * (GDN_CHUNK + 1) + j], U0[u0_base2 + j * Dv + d_col_v]));
                }
                MU0[mu0_base + i * Dv + d_col_v] = aU;
            }
        }

        // Qeff: thread owns d_col_k = tid % Dk
        {
            const LongType d_col_k = tid % Dk;
            const LongType row_stride_k = 256 / Dk;
            for (LongType i = tid / Dk; i < GDN_CHUNK; i += row_stride_k) {
                AccT aQ = static_cast<AccT>(0);
                for (LongType j = 0; j < (LongType)GDN_CHUNK; ++j) {
                    aQ = reproducible::add<AccT>(
                        aQ, reproducible::multiply<AccT>(
                            As[(LongType)i * (GDN_CHUNK + 1) + j], Kt[kt_base2 + j * Dk + d_col_k]));
                }
                AccT qg = static_cast<AccT>(0);
                if (i < (LongType)tt) {
                    const LongType qidx = (LongType)b * qS0 + (LongType)(t0 + (int)i) * qS1
                                           + (LongType)h * qS2 + d_col_k * qS3;
                    qg = reproducible::multiply<AccT>(eg_s[i], static_cast<AccT>(q[qidx]));
                }
                Qeff[qe_base + i * Dk + d_col_k] =
                    reproducible::subtract<AccT>(qg, aQ);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel B — inter-chunk state scan (b, h, dv_block), sequential over chunks
// Grid: (Dv/DV_BLK, B*H, 1), Block: (256, 1, 1)
// ---------------------------------------------------------------------------
template<typename T>
SD_KERNEL void gdnChunkScanKernel(
    const T*     __restrict__ k,        // [B,L,H,Dk]
    const typename simdOps::AggregateType<T>::type* __restrict__ Kt,
    const typename simdOps::AggregateType<T>::type* __restrict__ U0,
    const typename simdOps::AggregateType<T>::type* __restrict__ MU0,
    const typename simdOps::AggregateType<T>::type* __restrict__ Qeff,
    const typename simdOps::AggregateType<T>::type* __restrict__ lcgIn,
    const typename simdOps::AggregateType<T>::type* __restrict__ stateIn,
    typename simdOps::AggregateType<T>::type* __restrict__ stateOut,
    T*           __restrict__ y,        // [B,L,H,Dv]
    LongType B, LongType L, LongType H, LongType Dk, LongType Dv,
    LongType nC,
    LongType kS0, LongType kS1, LongType kS2, LongType kS3,
    LongType yS0, LongType yS1, LongType yS2, LongType yS3)
{
    using AccT = typename simdOps::AggregateType<T>::type;
    const int blk = blockIdx.x;  // Dv/DV_BLK block
    const int bh  = blockIdx.y;
    const int b   = bh / (int)H;
    const int h   = bh % (int)H;
    const int dv0 = blk * GDN_DV_BLK;
    const int tid = threadIdx.x;  // 0..255

    // State slice in shared memory: [DV_BLK][Dk+4] for this (b,h,dv_block)
    // +4 pad to avoid bank conflicts in Dk=64/128 accesses
    extern __shared__ char smem_raw2[];
    AccT* S_s   = reinterpret_cast<AccT*>(smem_raw2);       // [DV_BLK * (Dk+4)]
    AccT* U_s   = S_s + GDN_DV_BLK * (128 + 4);            // [C * (DV_BLK+4)]
    AccT* lcg_s = U_s + GDN_CHUNK * (GDN_DV_BLK + 4);      // [C]
    AccT* r_s   = lcg_s + GDN_CHUNK;                        // [C]
    AccT* kst_s = r_s + GDN_CHUNK;                          // [C * (Dk+4)]

    // Load initial state slice: [DV_BLK, Dk]
    // stateIn layout: [B,H,Dk,Dv] with canonical strides (C order)
    for (int p = tid; p < GDN_DV_BLK * (int)Dk; p += 256) {
        const int dv = p / (int)Dk;
        const int dk = p % (int)Dk;
        // stateIn[b, h, dk, dv0+dv]
        const LongType sidx = ((LongType)b * H + h) * Dk * Dv + (LongType)dk * Dv + (dv0 + dv);
        S_s[dv * (128 + 4) + dk] = (sidx < (LongType)B * H * Dk * Dv)
            ? stateIn[sidx] : static_cast<AccT>(0);
    }
    __syncthreads();

    for (int c = 0; c < (int)nC; ++c) {
        const int t0 = c * GDN_CHUNK;
        const int tt = (int)min((LongType)GDN_CHUNK, L - (LongType)t0);
        const LongType chunk_off = ((LongType)b * H + h) * nC * GDN_CHUNK + (LongType)c * GDN_CHUNK;
        const LongType io_off   = (((LongType)b * H + h) * nC + c) * (LongType)GDN_CHUNK;

        // Load lcg and r for this chunk
        if (tid < GDN_CHUNK)
            lcg_s[tid] = lcgIn[chunk_off + tid];
        __syncthreads();

        const AccT lcgLast = lcg_s[GDN_CHUNK - 1];
        if (tid < GDN_CHUNK)
            r_s[tid] = reproducible::exp<AccT>(
                reproducible::subtract<AccT>(lcgLast, lcg_s[tid]));
        __syncthreads();

        // ---- U[i,dv] = U0[i,dv] - Kt[i,:].S[dv,:] ;  y[i,dv] = MU0[i,dv] + Qeff[i,:].S[dv,:] ----
        // Thread coverage: C * DV_BLK outputs = 64*32 = 2048; 256 threads -> 8 each
        for (int p = tid; p < GDN_CHUNK * GDN_DV_BLK; p += 256) {
            const int i  = p / GDN_DV_BLK;
            const int dv = p % GDN_DV_BLK;

            // dot products with state row S_s[dv][:]
            AccT accU = static_cast<AccT>(0);
            AccT accY = static_cast<AccT>(0);
            for (LongType dk = 0; dk < Dk; ++dk) {
                const AccT stateValue = S_s[dv * (128 + 4) + dk];
                accU = reproducible::add<AccT>(
                    accU, reproducible::multiply<AccT>(
                        Kt[io_off * Dk + (LongType)i * Dk + dk], stateValue));
                accY = reproducible::add<AccT>(
                    accY, reproducible::multiply<AccT>(
                        Qeff[io_off * Dk + (LongType)i * Dk + dk], stateValue));
            }
            const AccT u0v = U0[io_off * Dv + (LongType)i * Dv + (dv0 + dv)];
            const AccT mu0v = MU0[io_off * Dv + (LongType)i * Dv + (dv0 + dv)];

            U_s[i * (GDN_DV_BLK + 4) + dv] =
                reproducible::subtract<AccT>(u0v, accU);

            if (i < tt) {
                const AccT outputValue = reproducible::add<AccT>(mu0v, accY);
                const LongType yidx = (LongType)b * yS0 + (LongType)(t0 + i) * yS1
                                     + (LongType)h * yS2 + (LongType)(dv0 + dv) * yS3;
                y[yidx] = static_cast<T>(outputValue);
            }
        }
        __syncthreads();

        // ---- S[dv,dk] = exp(lcg_last)*S[dv,dk] + sum_i r[i]*U[i,dv]*k[i,dk] ----
        // Stage k tokens into kst_s [C][Dk+4] in 64-wide d-tiles
        const AccT gateScale = reproducible::exp<AccT>(lcgLast);

        for (LongType dt = 0; dt < Dk; dt += 64) {
            const LongType tile_w = min((LongType)64, Dk - dt);
            for (int p = tid; p < GDN_CHUNK * 64; p += 256) {
                const int i  = p / 64;
                const int dd = p % 64;
                AccT keyValue = static_cast<AccT>(0);
                if (i < tt && (LongType)dd < tile_w) {
                    const LongType kidx = (LongType)b * kS0 + (LongType)(t0 + i) * kS1
                                         + (LongType)h * kS2 + (dt + dd) * kS3;
                    keyValue = static_cast<AccT>(k[kidx]);
                }
                kst_s[i * (128 + 4) + dd] = keyValue;
            }
            __syncthreads();

            for (int p = tid; p < GDN_DV_BLK * 64; p += 256) {
                const int dv = p / 64;
                const int dd = p % 64;
                AccT acc = reproducible::multiply<AccT>(
                    gateScale, S_s[dv * (128 + 4) + (dt + dd)]);
                for (int i = 0; i < tt; ++i) {
                    acc = reproducible::add<AccT>(
                        acc, reproducible::multiply<AccT>(
                            r_s[i], reproducible::multiply<AccT>(
                                U_s[i * (GDN_DV_BLK + 4) + dv], kst_s[i * (128 + 4) + dd])));
                }
                S_s[dv * (128 + 4) + (dt + dd)] = acc;
            }
            __syncthreads();
        }
    }

    // Write final state slice back to stateOut
    for (int p = tid; p < GDN_DV_BLK * (int)Dk; p += 256) {
        const int dv = p / (int)Dk;
        const int dk = p % (int)Dk;
        const LongType sidx = ((LongType)b * H + h) * Dk * Dv + (LongType)dk * Dv + (dv0 + dv);
        stateOut[sidx] = S_s[dv * (128 + 4) + dk];
    }
}

// ---------------------------------------------------------------------------
// Host launcher for chunked path
// ---------------------------------------------------------------------------
template<typename T>
static void launchGatedDeltaRuleChunked(
    const T* q, const T* k, const T* v,
    const T* betaArr, const T* gateArr,
    typename simdOps::AggregateType<T>::type* workingStateIn,
    typename simdOps::AggregateType<T>::type* workingStateOut,
    T* out,
    LongType B, LongType L, LongType H, LongType Dk, LongType Dv,
    LongType qS0, LongType qS1, LongType qS2, LongType qS3,
    LongType kS0, LongType kS1, LongType kS2, LongType kS3,
    LongType vS0, LongType vS1, LongType vS2, LongType vS3,
    LongType bS0, LongType bS1, LongType bS2,
    LongType gS0, LongType gS1, LongType gS2,
    LongType oS0, LongType oS1, LongType oS2, LongType oS3,
    cudaStream_t stream)
{
    using AccT = typename simdOps::AggregateType<T>::type;
    const LongType nC = (L + GDN_CHUNK - 1) / GDN_CHUNK;

    // Allocate intermediate device buffers via cudaMallocAsync on stream
    // (no CudaMemoryPool API that takes a stream — use cudaMallocAsync which is capture-safe
    //  in eager mode; prefill runs pre-freeze so capture safety is not required here)
    int deviceId = sd::AffinityManager::currentDeviceId();
    const LongType scratch_elems_k = B * H * nC * GDN_CHUNK * Dk;
    const LongType scratch_elems_v = B * H * nC * GDN_CHUNK * Dv;
    const LongType scratch_elems_c = B * H * nC * GDN_CHUNK;  // lcg

    AccT* d_Kt   = reinterpret_cast<AccT*>(
        sd::memory::CudaMemoryPool::getInstance().allocate(scratch_elems_k * sizeof(AccT), deviceId, stream));
    AccT* d_U0   = reinterpret_cast<AccT*>(
        sd::memory::CudaMemoryPool::getInstance().allocate(scratch_elems_v * sizeof(AccT), deviceId, stream));
    AccT* d_MU0  = reinterpret_cast<AccT*>(
        sd::memory::CudaMemoryPool::getInstance().allocate(scratch_elems_v * sizeof(AccT), deviceId, stream));
    AccT* d_Qeff = reinterpret_cast<AccT*>(
        sd::memory::CudaMemoryPool::getInstance().allocate(scratch_elems_k * sizeof(AccT), deviceId, stream));
    AccT* d_lcg  = reinterpret_cast<AccT*>(
        sd::memory::CudaMemoryPool::getInstance().allocate(scratch_elems_c * sizeof(AccT), deviceId, stream));

    // Kernel A: (nC, B*H, 1) blocks, 256 threads each
    // Shared memory: 4*C + C*(C+1) + 2*C*36  (lcg,bet,eg,bg = 4C; As = C*(C+1); kst,qst = 2*C*36)
    const size_t smemA =
        (4 * GDN_CHUNK + GDN_CHUNK * (GDN_CHUNK + 1) + 2 * GDN_CHUNK * 36) * sizeof(AccT);
    dim3 gridA(nC, B * H, 1);
    if (cudaFuncSetAttribute(gdnChunkIntraKernel<T>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smemA)) != cudaSuccess) {
        THROW_EXCEPTION("gdnChunkIntraKernel shared-memory opt-in failed");
    }
    gdnChunkIntraKernel<T><<<gridA, 256, smemA, stream>>>(
        q, k, v, betaArr, gateArr,
        d_Kt, d_U0, d_MU0, d_Qeff, d_lcg,
        B, L, H, Dk, Dv, nC,
        qS0, qS1, qS2, qS3,
        kS0, kS1, kS2, kS3,
        vS0, vS1, vS2, vS3,
        bS0, bS1, bS2,
        gS0, gS1, gS2);
    DebugHelper::checkGlobalErrorCode("gdnChunkIntraKernel failed");

    // Kernel B: (Dv/DV_BLK, B*H, 1) blocks, 256 threads each
    // Shared memory: S_s[DV_BLK*(Dk+4)] + U_s[C*(DV_BLK+4)] + lcg_s[C] + r_s[C] + kst_s[C*(Dk+4)]
    // For Dk=128, DV_BLK=32, C=64:
    const size_t smemB = ((LongType)GDN_DV_BLK * (128 + 4) +
                          (LongType)GDN_CHUNK   * (GDN_DV_BLK + 4) +
                          2 * GDN_CHUNK +
                          (LongType)GDN_CHUNK   * (128 + 4)) * sizeof(AccT);
    const LongType numDvBlocks = (Dv + GDN_DV_BLK - 1) / GDN_DV_BLK;
    dim3 gridB(numDvBlocks, B * H, 1);
    // smemB (~60KB) exceeds the 48KB default dynamic shared-memory limit; launching
    // without this opt-in fails with cudaErrorInvalidValue on every architecture.
    if (cudaFuncSetAttribute(gdnChunkScanKernel<T>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smemB)) != cudaSuccess) {
        THROW_EXCEPTION("gdnChunkScanKernel shared-memory opt-in failed");
    }
    gdnChunkScanKernel<T><<<gridB, 256, smemB, stream>>>(
        k, d_Kt, d_U0, d_MU0, d_Qeff, d_lcg,
        workingStateIn, workingStateOut, out,
        B, L, H, Dk, Dv, nC,
        kS0, kS1, kS2, kS3,
        oS0, oS1, oS2, oS3);
    DebugHelper::checkGlobalErrorCode("gdnChunkScanKernel failed");

    sd::memory::CudaMemoryPool::getInstance().free(d_Kt,   deviceId, stream);
    sd::memory::CudaMemoryPool::getInstance().free(d_U0,   deviceId, stream);
    sd::memory::CudaMemoryPool::getInstance().free(d_MU0,  deviceId, stream);
    sd::memory::CudaMemoryPool::getInstance().free(d_Qeff, deviceId, stream);
    sd::memory::CudaMemoryPool::getInstance().free(d_lcg,  deviceId, stream);
}

template <typename T>
static void gatedDeltaRuleFromArrays(
                     LaunchContext* context, NDArray* Q, NDArray* K, NDArray* V,
                     NDArray* beta, NDArray* gate, NDArray* stateIn,
                     NDArray* actualLen, NDArray* output, NDArray* stateOut) {
    using AccT = typename simdOps::AggregateType<T>::type;

    const auto B = Q->sizeAt(0);
    const auto L = Q->sizeAt(1);
    const auto H = Q->sizeAt(2);
    const auto D_k = Q->sizeAt(3);
    const auto D_v = V->sizeAt(3);

    auto stream = context->getCudaStream();
    static unsigned long long gdrTraceCallCount = 0;
    const unsigned long long gdrCallId = gdrTraceCallCount++;
    const LongType stateElems = B * H * D_k * D_v;
    const size_t stateBytes = stateElems * sizeof(AccT);
    int deviceId = sd::AffinityManager::currentDeviceId();
    bool directState = false;
    if constexpr (std::is_same<T, AccT>::value) {
        const bool denseStateOut = stateOut->ordering() == 'c'
            && shape::strideDescendingCAscendingF(stateOut->shapeInfo());
        bool compatibleStateIn = stateIn == nullptr;
        if (stateIn != nullptr && stateIn->ordering() == 'c'
                && shape::strideDescendingCAscendingF(stateIn->shapeInfo())) {
            const auto inStart = reinterpret_cast<std::uintptr_t>(stateIn->specialBuffer());
            const auto outStart = reinterpret_cast<std::uintptr_t>(stateOut->specialBuffer());
            const bool overlaps = inStart < outStart + stateBytes
                && outStart < inStart + stateBytes;
            compatibleStateIn = !overlaps || inStart == outStart;
        }
        // actualLen forces the sequential path. Keep chunked and low-precision
        // execution on their existing promoted scratch contracts.
        directState = actualLen != nullptr && denseStateOut && compatibleStateIn;
    }
    AccT* workingState = directState
        ? reinterpret_cast<AccT*>(stateOut->specialBuffer())
        : reinterpret_cast<AccT*>(
            sd::memory::CudaMemoryPool::getInstance().allocate(
                stateBytes, deviceId, *stream));
    if (workingState == nullptr) {
        THROW_EXCEPTION("gatedDeltaRule: recurrent state allocation failed");
    }

    // GDR_PRECHECK: env-gated (ND4J_GDR_POSTCHECK=1 shares the switch). Syncs the
    // stream THEN scans every input for non-finite values AT LAUNCH TIME. This is
    // the at-launch counterpart to GDR_POSTCHECK: the fixture capture happens at
    // lineage-dump time (after failure), so clean fixture inputs do NOT prove the
    // kernel received clean inputs. If this fires, an input was corrupted between
    // its producing op and this call; if clean while the output postcheck fails,
    // the corruption is generated INSIDE the call (scratch/state race).
    static const bool gdrPreCheck = [] {
        const char* e = std::getenv("ND4J_GDR_POSTCHECK");
        return e != nullptr && e[0] == '1';
    }();
    if (gdrPreCheck && L <= 16) {
        // NEVER sync a stream that is capturing a CUDA graph — the sync invalidates
        // the capture and fails the whole plan execution (status 50 KERNEL_FAILURE).
        cudaStreamCaptureStatus captureStatus = cudaStreamCaptureStatusNone;
        cudaStreamIsCapturing(*stream, &captureStatus);
        if (captureStatus != cudaStreamCaptureStatusNone) {
            fprintf(stderr, "[GDR-PRECHECK] call=%llu L=%lld SKIP: stream capturing\n",
                    gdrCallId, (long long)L);
        } else {
        cudaStreamSynchronize(*stream);
        auto scanArr = [](const char* name, NDArray* a) -> std::string {
            if (a == nullptr) return name + std::string("=null");
            const LongType n = a->lengthOf();
            unsigned long long nanC = 0, infC = 0;
            double mn = 0.0, mx = 0.0;
            bool first = true;
            for (LongType i = 0; i < n; ++i) {
                const double v = static_cast<double>(a->e<AccT>(i));
                if (std::isnan(v)) { nanC++; continue; }
                if (std::isinf(v)) { infC++; continue; }
                if (first) { mn = mx = v; first = false; }
                else { mn = std::min(mn, v); mx = std::max(mx, v); }
            }
            char buf[192];
            snprintf(buf, sizeof(buf), "%s[n=%lld nan=%llu inf=%llu min=%.6g max=%.6g]",
                     name, (long long)n, nanC, infC, mn, mx);
            return std::string(buf);
        };
        fprintf(stderr, "[GDR-PRECHECK] call=%llu L=%lld ws=%p %s %s %s %s %s %s\n",
                gdrCallId, (long long)L, (void*)workingState,
                scanArr("Q", Q).c_str(), scanArr("K", K).c_str(), scanArr("V", V).c_str(),
                scanArr("beta", beta).c_str(), scanArr("gate", gate).c_str(),
                scanArr("sIn", stateIn).c_str());
        fflush(stderr);
        }
    }

    if (stateIn != nullptr) {
        if (stateIn->specialBuffer() != workingState) {
            int initBlocks = (stateElems + 255) / 256;
            convertStateKernel<T, AccT><<<initBlocks, 256, 0, *stream>>>(
                reinterpret_cast<const T*>(stateIn->specialBuffer()), workingState, stateElems);
            DebugHelper::checkGlobalErrorCode("gatedDeltaRule state initialization failed");
        }
    } else {
        cudaMemsetAsync(workingState, 0, stateElems * sizeof(AccT), *stream);
    }

    const size_t chunkIntraSharedMemory =
        (4 * GDN_CHUNK + GDN_CHUNK * (GDN_CHUNK + 1) + 2 * GDN_CHUNK * 36) * sizeof(AccT);
    const size_t chunkScanSharedMemory =
        ((LongType)GDN_DV_BLK * (128 + 4) +
         (LongType)GDN_CHUNK * (GDN_DV_BLK + 4) +
         2 * GDN_CHUNK +
         (LongType)GDN_CHUNK * (128 + 4)) * sizeof(AccT);
    int maxSharedMemory = 0;
    cudaDeviceGetAttribute(
        &maxSharedMemory, cudaDevAttrMaxSharedMemoryPerBlockOptin, deviceId);
    const bool useChunked = (L >= GDN_CHUNK)
        && (actualLen == nullptr)
        && (D_v % GDN_DV_BLK == 0)
        && (D_k <= 128)
        && (chunkIntraSharedMemory <= static_cast<size_t>(maxSharedMemory))
        && (chunkScanSharedMemory <= static_cast<size_t>(maxSharedMemory));

    // GDR_LAUNCH_TRACE: host-side only (no sync, no device reads, no value dumps).
    // Enabled via ND4J_GDR_LAUNCH_TRACE=1. Records exactly what device memory the
    // kernels were handed at launch time so a post-mortem lineage dump can be
    // diffed against it (catches stale input pointers, recycled pool scratch,
    // stream switches, and stale actualLen host mirror without perturbing timing).
    static const bool gdrLaunchTrace = [] {
        const char* e = std::getenv("ND4J_GDR_LAUNCH_TRACE");
        return e != nullptr && e[0] == '1';
    }();
    if (gdrLaunchTrace) {
        static void* gdrTraceLastWorkingState = nullptr;
        static void* gdrTraceLastStream = nullptr;
        const LongType lenHost =
            (actualLen != nullptr && actualLen->buffer() != nullptr)
                ? *reinterpret_cast<const LongType*>(actualLen->buffer())
                : -1;
        fprintf(stderr,
                "[GDR-TRACE] call=%llu seq=%d L=%lld B=%lld H=%lld dk=%lld dv=%lld "
                "Q=%p K=%p V=%p beta=%p gate=%p sIn=%p len=%p lenHost=%lld "
                "out=%p sOut=%p ws=%p wsPrev=%p stream=%p streamPrev=%p useChunked=%d directState=%d\n",
                gdrCallId,
                actualLen != nullptr ? 1 : 0,
                (long long)L, (long long)B, (long long)H,
                (long long)D_k, (long long)D_v,
                (void*)Q->specialBuffer(), (void*)K->specialBuffer(),
                (void*)V->specialBuffer(), (void*)beta->specialBuffer(),
                (void*)gate->specialBuffer(),
                stateIn != nullptr ? (void*)stateIn->specialBuffer() : nullptr,
                actualLen != nullptr ? (void*)actualLen->specialBuffer() : nullptr,
                (long long)lenHost,
                (void*)output->specialBuffer(), (void*)stateOut->specialBuffer(),
                (void*)workingState, gdrTraceLastWorkingState,
                (void*)*stream, gdrTraceLastStream,
                useChunked ? 1 : 0, directState ? 1 : 0);
        fflush(stderr);
        gdrTraceLastWorkingState = (void*)workingState;
        gdrTraceLastStream = (void*)*stream;
    }

    if (useChunked) {
        AccT* workingStateOut = reinterpret_cast<AccT*>(
            sd::memory::CudaMemoryPool::getInstance().allocate(
                stateElems * sizeof(AccT), deviceId, *stream));

        launchGatedDeltaRuleChunked<T>(
            reinterpret_cast<const T*>(Q->specialBuffer()),
            reinterpret_cast<const T*>(K->specialBuffer()),
            reinterpret_cast<const T*>(V->specialBuffer()),
            reinterpret_cast<const T*>(beta->specialBuffer()),
            reinterpret_cast<const T*>(gate->specialBuffer()),
            workingState, workingStateOut,
            reinterpret_cast<T*>(output->specialBuffer()),
            B, L, H, D_k, D_v,
            Q->strideAt(0), Q->strideAt(1), Q->strideAt(2), Q->strideAt(3),
            K->strideAt(0), K->strideAt(1), K->strideAt(2), K->strideAt(3),
            V->strideAt(0), V->strideAt(1), V->strideAt(2), V->strideAt(3),
            beta->strideAt(0), beta->strideAt(1), beta->strideAt(2),
            gate->strideAt(0), gate->strideAt(1), gate->strideAt(2),
            output->strideAt(0), output->strideAt(1), output->strideAt(2), output->strideAt(3),
            *stream);

        int copyBlocks = (stateElems + 255) / 256;
        convertStateKernel<AccT, T><<<copyBlocks, 256, 0, *stream>>>(
            workingStateOut, reinterpret_cast<T*>(stateOut->specialBuffer()), stateElems);

        sd::memory::CudaMemoryPool::getInstance().free(workingStateOut, deviceId, *stream);
    } else {
        launchGatedDeltaRule<T>(
            reinterpret_cast<const T*>(Q->specialBuffer()),
            reinterpret_cast<const T*>(K->specialBuffer()),
            reinterpret_cast<const T*>(V->specialBuffer()),
            reinterpret_cast<const T*>(beta->specialBuffer()),
            reinterpret_cast<const T*>(gate->specialBuffer()),
            actualLen ? reinterpret_cast<const LongType*>(actualLen->specialBuffer()) : nullptr,
            workingState,
            reinterpret_cast<T*>(output->specialBuffer()),
            B, L, H, D_k, D_v,
            Q->strideAt(0), Q->strideAt(1), Q->strideAt(2), Q->strideAt(3),
            K->strideAt(0), K->strideAt(1), K->strideAt(2), K->strideAt(3),
            V->strideAt(0), V->strideAt(1), V->strideAt(2), V->strideAt(3),
            beta->strideAt(0), beta->strideAt(1), beta->strideAt(2),
            gate->strideAt(0), gate->strideAt(1), gate->strideAt(2),
            output->strideAt(0), output->strideAt(1), output->strideAt(2), output->strideAt(3),
            *stream);

        if (!directState) {
            int copyBlocks = (stateElems + 255) / 256;
            convertStateKernel<AccT, T><<<copyBlocks, 256, 0, *stream>>>(
                workingState, reinterpret_cast<T*>(stateOut->specialBuffer()), stateElems);
        }
    }

    if (!directState) {
        sd::memory::CudaMemoryPool::getInstance().free(workingState, deviceId, *stream);
    }
    DebugHelper::checkGlobalErrorCode("gatedDeltaRule state write-back failed");

    // GDR_POSTCHECK: env-gated (ND4J_GDR_POSTCHECK=1) immediate post-call verification.
    // ONE cudaStreamSynchronize then a finite scan of output + stateOut. This pins down
    // WHEN slot 460 becomes garbage: if the scan right here reports non-finite values,
    // the GDR execution itself produced them (kernel/harness bug despite good inputs);
    // if it reports finite, something AFTER the call overwrote the buffers (post-call
    // clobber by another op / stale baked-address reader). Expensive — diagnostics only.
    static const bool gdrPostCheck = [] {
        const char* e = std::getenv("ND4J_GDR_POSTCHECK");
        return e != nullptr && e[0] == '1';
    }();
    if (gdrPostCheck && L > 1) {
        cudaStreamCaptureStatus captureStatus = cudaStreamCaptureStatusNone;
        cudaStreamIsCapturing(*stream, &captureStatus);
        if (captureStatus != cudaStreamCaptureStatusNone) {
            fprintf(stderr, "[GDR-POSTCHECK] call=%llu L=%lld SKIP: stream capturing\n",
                    gdrCallId, (long long)L);
        } else {
        cudaError_t syncErr = cudaStreamSynchronize(*stream);
        const auto L_ = static_cast<LongType>(L);
        (void)syncErr;
        const auto outElems = output->lengthOf();
        unsigned long long outNan = 0, outInf = 0, stNan = 0, stInf = 0;
        double outMin = 0.0, outMax = 0.0;
        bool first = true;
        for (LongType i = 0; i < outElems; ++i) {
            const double v = static_cast<double>(output->e<AccT>(i));
            if (std::isnan(v)) { outNan++; continue; }
            if (std::isinf(v)) { outInf++; continue; }
            if (first) { outMin = outMax = v; first = false; }
            else { outMin = std::min(outMin, v); outMax = std::max(outMax, v); }
        }
        const LongType stElems = stateOut->lengthOf();
        for (LongType i = 0; i < stElems; ++i) {
            const double v = static_cast<double>(stateOut->e<AccT>(i));
            if (std::isnan(v)) stNan++;
            else if (std::isinf(v)) stInf++;
        }
        fprintf(stderr,
                "[GDR-POSTCHECK] call=%llu L=%lld out=%p(%lld elems) outNan=%llu outInf=%llu "
                "min=%.6g max=%.6g | sOut=%p stNan=%llu stInf=%llu syncErr=%d\n",
                gdrCallId, (long long)L_, (void*)output->specialBuffer(), (long long)outElems,
                outNan, outInf, outMin, outMax,
                (void*)stateOut->specialBuffer(), stNan, stInf, (int)syncErr);
        if (outNan || outInf || stNan || stInf) {
            fprintf(stderr,
                    "[GDR-POSTCHECK] *** NON-FINITE DETECTED AT GDR OUTPUT — GDR produced the garbage itself ***\n");
        }
        }
    }
}

void gatedDeltaRule(LaunchContext* context, NDArray* Q, NDArray* K, NDArray* V,
                     NDArray* beta, NDArray* gate, NDArray* stateIn,
                     NDArray* actualLen, NDArray* output, NDArray* stateOut) {
    if (Q->sizeAt(3) > GDR_CUDA_MAX_HEAD_DIM) {
        THROW_EXCEPTION("gatedDeltaRule: key head dimension exceeds supported CUDA maximum");
    }
    NDArray::prepareSpecialUse({output, stateOut}, {Q, K, V, beta, gate, actualLen});
    if (stateIn != nullptr) NDArray::prepareSpecialUse({}, {stateIn});

    BUILD_SINGLE_SELECTOR(
        Q->dataType(), gatedDeltaRuleFromArrays,
        (context, Q, K, V, beta, gate, stateIn, actualLen, output, stateOut),
        SD_FLOAT_TYPES);

    NDArray::registerSpecialUse({output, stateOut}, {Q, K, V, beta, gate, actualLen});
    if (stateIn != nullptr) NDArray::registerSpecialUse({}, {stateIn});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
