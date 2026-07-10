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
#include <types/float16.h>
#include <ops/declarable/helpers/gated_delta_rule.h>

namespace sd {
namespace ops {
namespace helpers {

// One block per (batch, head). Threads cover D_v dimension.
// Sequential over timesteps (recurrent dependency).
// Working state is ALWAYS float32 regardless of T to prevent FP16 quantization
// error from compounding across timesteps (matches CPU behavior).
template <typename T>
SD_KERNEL void gatedDeltaRuleKernel(
    const T* __restrict__ q,
    const T* __restrict__ k,
    const T* __restrict__ v,
    const T* __restrict__ betaArr,
    const T* __restrict__ gateArr,
    const LongType* __restrict__ actualLen,
    float* __restrict__ state,
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
    const float exp_g_f = updateState ? sd::math::sd_exp<float, float>(static_cast<float>(gateArr[b * gS0 + t * gS1 + h * gS2])) : 1.0f;
    const float beta_f = updateState ? static_cast<float>(betaArr[b * bS0 + t * bS1 + h * bS2]) : 0.0f;
    float* sPtr = state + (b * H + h) * D_k * D_v;

    for (LongType dv = threadIdx.x; dv < D_v; dv += blockDim.x) {
        if (updateState) {
            // prediction = S^T * k  (state already float32, no cast needed)
            float prediction = 0.0f;
            for (LongType dk = 0; dk < D_k; ++dk)
                prediction += sPtr[dk * D_v + dv] * static_cast<float>(k[b * kS0 + t * kS1 + h * kS2 + dk * kS3]);

            // delta = v - exp(g) * prediction
            const float delta = static_cast<float>(v[b * vS0 + t * vS1 + h * vS2 + dv * vS3]) - exp_g_f * prediction;

            // S = exp(g) * S + beta * k * delta  (stays in float32)
            for (LongType dk = 0; dk < D_k; ++dk) {
                const float k_val = static_cast<float>(k[b * kS0 + t * kS1 + h * kS2 + dk * kS3]);
                sPtr[dk * D_v + dv] = exp_g_f * sPtr[dk * D_v + dv] + beta_f * k_val * delta;
            }
        }

        // output = S^T * q  (accumulated in float32)
        float out_val = 0.0f;
        for (LongType dk = 0; dk < D_k; ++dk)
            out_val += sPtr[dk * D_v + dv] * static_cast<float>(q[b * qS0 + t * qS1 + h * qS2 + dk * qS3]);
        out[b * oS0 + t * oS1 + h * oS2 + dv * oS3] = static_cast<T>(out_val);
    }
}

// Kernel to initialize float32 working state from T-typed stateIn
template <typename T>
SD_KERNEL void stateInToFloat32Kernel(
    const T* __restrict__ src,
    float* __restrict__ dst,
    const LongType total) {
    const LongType idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        dst[idx] = static_cast<float>(src[idx]);
    }
}

// Kernel to write back float32 working state to T-typed stateOut
template <typename T>
SD_KERNEL void float32ToStateOutKernel(
    const float* __restrict__ src,
    T* __restrict__ dst,
    const LongType total) {
    const LongType idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        dst[idx] = static_cast<T>(src[idx]);
    }
}

template <typename T>
static void launchGatedDeltaRule(
    const T* q, const T* k, const T* v,
    const T* betaArr, const T* gateArr, const LongType* actualLen,
    float* workingState, T* out,
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
// All arithmetic in float32. Gate input is already log-domain (g), so
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
    float*       __restrict__ Kt,       // [B,H,nC,C,Dk]  (device scratch)
    float*       __restrict__ U0,       // [B,H,nC,C,Dv]
    float*       __restrict__ MU0,      // [B,H,nC,C,Dv]
    float*       __restrict__ Qeff,     // [B,H,nC,C,Dk]
    float*       __restrict__ lcgOut,   // [B,H,nC,C]
    LongType B, LongType L, LongType H, LongType Dk, LongType Dv,
    LongType nC,
    LongType qS0, LongType qS1, LongType qS2, LongType qS3,
    LongType kS0, LongType kS1, LongType kS2, LongType kS3,
    LongType vS0, LongType vS1, LongType vS2, LongType vS3,
    LongType bS0, LongType bS1, LongType bS2,
    LongType gS0, LongType gS1, LongType gS2)
{
    // Each block handles one (chunk, bh) pair
    const int c   = blockIdx.x;   // chunk index
    const int bh  = blockIdx.y;   // b*H + h
    const int b   = bh / (int)H;
    const int h   = bh % (int)H;
    const int t0  = c * GDN_CHUNK;
    const int tt  = (int)min((LongType)GDN_CHUNK, L - (LongType)t0);  // valid tokens in this chunk
    const int tid = threadIdx.x;  // 0..255

    // Shared memory layout:
    //   lcg_s[C]       float  — cumulative log-gate
    //   bet_s[C]       float  — beta per token
    //   eg_s[C]        float  — exp(lcg)
    //   bg_s[C]        float  — beta*exp(lcg)
    //   As[C][C+1]     float  — A/X matrix (+1 pad avoids bank conflicts)
    //   kst[C][128+4]  float  — staged k tile (up to Dk=128, +4 pad)
    //   qst[C][128+4]  float  — staged q tile

    extern __shared__ char smem_raw[];
    float* lcg_s = reinterpret_cast<float*>(smem_raw);                     // [C]
    float* bet_s = lcg_s + GDN_CHUNK;                                       // [C]
    float* eg_s  = bet_s + GDN_CHUNK;                                       // [C]
    float* bg_s  = eg_s  + GDN_CHUNK;                                       // [C]
    // As[i][j]: row-major [C][C+1]
    float* As    = bg_s  + GDN_CHUNK;                                       // [C*(C+1)]
    // kst/qst: [C][Dk_tile] — we stage 32-wide d-tiles
    float* kst   = As    + GDN_CHUNK * (GDN_CHUNK + 1);                    // [C*36] (32+4 pad)
    float* qst   = kst   + GDN_CHUNK * 36;                                 // [C*36]

    // ---- Compute lcg (cumulative log-gate) and beta — sequential in thread 0 ----
    if (tid == 0) {
        float acc = 0.0f;
        for (int i = 0; i < GDN_CHUNK; ++i) {
            float gi = (i < tt)
                ? static_cast<float>(gateArr[b * gS0 + (LongType)(t0 + i) * gS1 + h * gS2])
                : 0.0f;  // gate=0 -> exp(0)=1 -> no decay in padding
            acc += gi;   // gate already in log domain
            lcg_s[i] = acc;
        }
    }
    if (tid == 32) {
        for (int i = 0; i < GDN_CHUNK; ++i) {
            bet_s[i] = (i < tt)
                ? static_cast<float>(betaArr[b * bS0 + (LongType)(t0 + i) * bS1 + h * bS2])
                : 0.0f;
        }
    }
    __syncthreads();

    if (tid < GDN_CHUNK) {
        eg_s[tid] = sd::math::sd_exp<float, float>(lcg_s[tid]);
        bg_s[tid] = bet_s[tid] * eg_s[tid];
    }
    __syncthreads();

    // ---- Compute A = KK^T and M = qK^T via staged 32-wide d-tiles ----
    // Thread tid owns 16 (i,j) pairs with flat index p = tid*16+r
    // Each pair (i,j): contributes to A if j<i, to M if j<=i
    float accA[16] = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    float accM[16] = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};

    // k/q base pointers for this (b, h)
    const T* k_bh = k + (LongType)b * kS0 + (LongType)h * kS2;
    const T* q_bh = q + (LongType)b * qS0 + (LongType)h * qS2;

    for (LongType dt = 0; dt < Dk; dt += 32) {
        const LongType tile_width = min((LongType)32, Dk - dt);
        // Cooperative load of k/q tiles: C rows x tile_width cols
        for (int p = tid; p < GDN_CHUNK * 32; p += 256) {
            const int r  = p / 32;
            const int cc = p % 32;
            float kv = 0.f, qv = 0.f;
            if (r < tt && cc < (int)tile_width) {
                const LongType tidx = (LongType)(t0 + r) * kS1 + (dt + cc) * kS3;
                const LongType qidx = (LongType)(t0 + r) * qS1 + (dt + cc) * qS3;
                kv = static_cast<float>(k_bh[tidx]);
                qv = static_cast<float>(q_bh[qidx]);
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
                float da = 0.f, dm = 0.f;
                for (int dd = 0; dd < 32; ++dd) {
                    const float kjd = kst[j * 36 + dd];
                    da += kst[i * 36 + dd] * kjd;
                    dm += qst[i * 36 + dd] * kjd;
                }
                accA[r] += da;
                accM[r] += dm;
            }
        }
        __syncthreads();
    }

    // ---- Write A (strict lower triangular, scaled by beta*decay) into As ----
    for (int r = 0; r < 16; ++r) {
        const int p = tid * 16 + r;
        const int i = p / GDN_CHUNK;
        const int j = p % GDN_CHUNK;
        float a = 0.f;
        if (j < i && i < tt)
            a = bet_s[i] * sd::math::sd_exp<float,float>(lcg_s[i] - lcg_s[j]) * accA[r];
        As[i * (GDN_CHUNK + 1) + j] = a;
    }
    __syncthreads();

    // ---- X = (I + A)^{-1} via forward substitution (in-place, overwrites As) ----
    // Row i: X[i,j] = delta_{ij} - sum_{k<i} A[i,k]*X[k,j]
    // After this, As[i][0..C-1] = X[i,:]
    for (int i = 0; i < GDN_CHUNK; ++i) {
        float x = 0.f;
        if (tid < GDN_CHUNK) {
            x = (tid == i) ? 1.0f : 0.0f;
            for (int jj = 0; jj < i; ++jj)
                x -= As[i * (GDN_CHUNK + 1) + jj] * As[jj * (GDN_CHUNK + 1) + tid];
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
                float aK = 0.f;
                for (LongType j = 0; j < (LongType)tt; ++j) {
                    const float x_ij = As[(LongType)i * (GDN_CHUNK + 1) + j];
                    const LongType kidx = (LongType)b * kS0 + (LongType)(t0 + (int)j) * kS1
                                          + (LongType)h * kS2 + d_col * kS3;
                    aK += x_ij * bg_s[j] * static_cast<float>(k[kidx]);
                }
                Kt[kt_base + i * Dk + d_col] = aK;
            }
        }

        // U0: thread owns column d_col_v = tid % Dv
        {
            const LongType d_col_v = tid % Dv;
            const LongType row_stride_v = 256 / Dv;
            for (LongType i = tid / Dv; i < GDN_CHUNK; i += row_stride_v) {
                float aV = 0.f;
                for (LongType j = 0; j < (LongType)tt; ++j) {
                    const float x_ij = As[(LongType)i * (GDN_CHUNK + 1) + j];
                    const LongType vidx = (LongType)b * vS0 + (LongType)(t0 + (int)j) * vS1
                                          + (LongType)h * vS2 + d_col_v * vS3;
                    aV += x_ij * bet_s[j] * static_cast<float>(v[vidx]);
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
        float m = 0.f;
        if (j <= i && i < tt)
            m = accM[r] * sd::math::sd_exp<float,float>(lcg_s[i] - lcg_s[j]);
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
                float aU = 0.f;
                for (LongType j = 0; j < (LongType)GDN_CHUNK; ++j)
                    aU += As[(LongType)i * (GDN_CHUNK + 1) + j] * U0[u0_base2 + j * Dv + d_col_v];
                MU0[mu0_base + i * Dv + d_col_v] = aU;
            }
        }

        // Qeff: thread owns d_col_k = tid % Dk
        {
            const LongType d_col_k = tid % Dk;
            const LongType row_stride_k = 256 / Dk;
            for (LongType i = tid / Dk; i < GDN_CHUNK; i += row_stride_k) {
                float aQ = 0.f;
                for (LongType j = 0; j < (LongType)GDN_CHUNK; ++j)
                    aQ += As[(LongType)i * (GDN_CHUNK + 1) + j] * Kt[kt_base2 + j * Dk + d_col_k];
                float qg = 0.f;
                if (i < (LongType)tt) {
                    const LongType qidx = (LongType)b * qS0 + (LongType)(t0 + (int)i) * qS1
                                           + (LongType)h * qS2 + d_col_k * qS3;
                    qg = eg_s[i] * static_cast<float>(q[qidx]);
                }
                Qeff[qe_base + i * Dk + d_col_k] = qg - aQ;
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
    const float* __restrict__ Kt,       // [B,H,nC,C,Dk]
    const float* __restrict__ U0,       // [B,H,nC,C,Dv]
    const float* __restrict__ MU0,      // [B,H,nC,C,Dv]
    const float* __restrict__ Qeff,     // [B,H,nC,C,Dk]
    const float* __restrict__ lcgIn,    // [B,H,nC,C]
    const float* __restrict__ stateIn,  // [B,H,Dk,Dv] float32
    float*       __restrict__ stateOut, // [B,H,Dk,Dv] float32
    T*           __restrict__ y,        // [B,L,H,Dv]
    LongType B, LongType L, LongType H, LongType Dk, LongType Dv,
    LongType nC,
    LongType kS0, LongType kS1, LongType kS2, LongType kS3,
    LongType yS0, LongType yS1, LongType yS2, LongType yS3)
{
    const int blk = blockIdx.x;  // Dv/DV_BLK block
    const int bh  = blockIdx.y;
    const int b   = bh / (int)H;
    const int h   = bh % (int)H;
    const int dv0 = blk * GDN_DV_BLK;
    const int tid = threadIdx.x;  // 0..255

    // State slice in shared memory: [DV_BLK][Dk+4] for this (b,h,dv_block)
    // +4 pad to avoid bank conflicts in Dk=64/128 accesses
    extern __shared__ char smem_raw2[];
    float* S_s   = reinterpret_cast<float*>(smem_raw2);    // [DV_BLK * (Dk+4)]
    float* U_s   = S_s + GDN_DV_BLK * (128 + 4);          // [C * (DV_BLK+4)]  (C=64, DV_BLK=32)
    float* lcg_s = U_s + GDN_CHUNK * (GDN_DV_BLK + 4);    // [C]
    float* r_s   = lcg_s + GDN_CHUNK;                      // [C]
    float* kst_s = r_s + GDN_CHUNK;                        // [C * (Dk+4)]

    // Load initial state slice: [DV_BLK, Dk]
    // stateIn layout: [B,H,Dk,Dv] with canonical strides (C order)
    for (int p = tid; p < GDN_DV_BLK * (int)Dk; p += 256) {
        const int dv = p / (int)Dk;
        const int dk = p % (int)Dk;
        // stateIn[b, h, dk, dv0+dv]
        const LongType sidx = ((LongType)b * H + h) * Dk * Dv + (LongType)dk * Dv + (dv0 + dv);
        S_s[dv * (128 + 4) + dk] = (sidx < (LongType)B * H * Dk * Dv)
            ? stateIn[sidx] : 0.f;
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

        const float lcg_last = lcg_s[GDN_CHUNK - 1];
        if (tid < GDN_CHUNK)
            r_s[tid] = sd::math::sd_exp<float,float>(lcg_last - lcg_s[tid]);
        __syncthreads();

        // ---- U[i,dv] = U0[i,dv] - Kt[i,:].S[dv,:] ;  y[i,dv] = MU0[i,dv] + Qeff[i,:].S[dv,:] ----
        // Thread coverage: C * DV_BLK outputs = 64*32 = 2048; 256 threads -> 8 each
        for (int p = tid; p < GDN_CHUNK * GDN_DV_BLK; p += 256) {
            const int i  = p / GDN_DV_BLK;
            const int dv = p % GDN_DV_BLK;

            // dot products with state row S_s[dv][:]
            float accU = 0.f, accY = 0.f;
            for (LongType dk = 0; dk < Dk; ++dk) {
                const float s = S_s[dv * (128 + 4) + dk];
                accU += Kt[  io_off * Dk + (LongType)i * Dk + dk] * s;
                accY += Qeff[io_off * Dk + (LongType)i * Dk + dk] * s;
            }
            const float u0v = U0[  io_off * Dv + (LongType)i * Dv + (dv0 + dv)];
            const float mu0v = MU0[io_off * Dv + (LongType)i * Dv + (dv0 + dv)];

            U_s[i * (GDN_DV_BLK + 4) + dv] = u0v - accU;

            if (i < tt) {
                const float yv = mu0v + accY;
                const LongType yidx = (LongType)b * yS0 + (LongType)(t0 + i) * yS1
                                     + (LongType)h * yS2 + (LongType)(dv0 + dv) * yS3;
                y[yidx] = static_cast<T>(yv);
            }
        }
        __syncthreads();

        // ---- S[dv,dk] = exp(lcg_last)*S[dv,dk] + sum_i r[i]*U[i,dv]*k[i,dk] ----
        // Stage k tokens into kst_s [C][Dk+4] in 64-wide d-tiles
        const float gl = sd::math::sd_exp<float,float>(lcg_last);

        for (LongType dt = 0; dt < Dk; dt += 64) {
            const LongType tile_w = min((LongType)64, Dk - dt);
            for (int p = tid; p < GDN_CHUNK * 64; p += 256) {
                const int i  = p / 64;
                const int dd = p % 64;
                float kv = 0.f;
                if (i < tt && (LongType)dd < tile_w) {
                    const LongType kidx = (LongType)b * kS0 + (LongType)(t0 + i) * kS1
                                         + (LongType)h * kS2 + (dt + dd) * kS3;
                    kv = static_cast<float>(k[kidx]);
                }
                kst_s[i * (128 + 4) + dd] = kv;  // max tile_w=64, fits in [C][Dk+4]
            }
            __syncthreads();

            for (int p = tid; p < GDN_DV_BLK * 64; p += 256) {
                const int dv = p / 64;
                const int dd = p % 64;
                float acc = gl * S_s[dv * (128 + 4) + (dt + dd)];
                for (int i = 0; i < tt; ++i)
                    acc += r_s[i] * U_s[i * (GDN_DV_BLK + 4) + dv] * kst_s[i * (128 + 4) + dd];
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
    float* workingStateIn, float* workingStateOut, T* out,
    LongType B, LongType L, LongType H, LongType Dk, LongType Dv,
    LongType qS0, LongType qS1, LongType qS2, LongType qS3,
    LongType kS0, LongType kS1, LongType kS2, LongType kS3,
    LongType vS0, LongType vS1, LongType vS2, LongType vS3,
    LongType bS0, LongType bS1, LongType bS2,
    LongType gS0, LongType gS1, LongType gS2,
    LongType oS0, LongType oS1, LongType oS2, LongType oS3,
    cudaStream_t stream)
{
    const LongType nC = (L + GDN_CHUNK - 1) / GDN_CHUNK;

    // Allocate intermediate device buffers via cudaMallocAsync on stream
    // (no CudaMemoryPool API that takes a stream — use cudaMallocAsync which is capture-safe
    //  in eager mode; prefill runs pre-freeze so capture safety is not required here)
    int deviceId = sd::AffinityManager::currentDeviceId();
    const LongType scratch_elems_k = B * H * nC * GDN_CHUNK * Dk;
    const LongType scratch_elems_v = B * H * nC * GDN_CHUNK * Dv;
    const LongType scratch_elems_c = B * H * nC * GDN_CHUNK;  // lcg

    float* d_Kt   = reinterpret_cast<float*>(
        sd::memory::CudaMemoryPool::getInstance().allocate(scratch_elems_k * sizeof(float), deviceId, stream));
    float* d_U0   = reinterpret_cast<float*>(
        sd::memory::CudaMemoryPool::getInstance().allocate(scratch_elems_v * sizeof(float), deviceId, stream));
    float* d_MU0  = reinterpret_cast<float*>(
        sd::memory::CudaMemoryPool::getInstance().allocate(scratch_elems_v * sizeof(float), deviceId, stream));
    float* d_Qeff = reinterpret_cast<float*>(
        sd::memory::CudaMemoryPool::getInstance().allocate(scratch_elems_k * sizeof(float), deviceId, stream));
    float* d_lcg  = reinterpret_cast<float*>(
        sd::memory::CudaMemoryPool::getInstance().allocate(scratch_elems_c * sizeof(float), deviceId, stream));

    // Kernel A: (nC, B*H, 1) blocks, 256 threads each
    // Shared memory: 4*C + C*(C+1) + 2*C*36  (lcg,bet,eg,bg = 4C; As = C*(C+1); kst,qst = 2*C*36)
    const size_t smemA = (4 * GDN_CHUNK + GDN_CHUNK * (GDN_CHUNK + 1) + 2 * GDN_CHUNK * 36) * sizeof(float);
    dim3 gridA(nC, B * H, 1);
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
    //   S_s = 32*132 = 4224 floats, U_s = 64*36 = 2304, lcg+r = 128, kst = 64*132 = 8448
    //   Total = (4224+2304+128+8448)*4 = 60416 bytes (within 64KB limit)
    const size_t smemB = ((LongType)GDN_DV_BLK * (128 + 4) +
                          (LongType)GDN_CHUNK   * (GDN_DV_BLK + 4) +
                          2 * GDN_CHUNK +
                          (LongType)GDN_CHUNK   * (128 + 4)) * sizeof(float);
    const LongType numDvBlocks = (Dv + GDN_DV_BLK - 1) / GDN_DV_BLK;
    dim3 gridB(numDvBlocks, B * H, 1);
    // smemB (~60KB) exceeds the 48KB default dynamic shared-memory limit; launching
    // without this opt-in fails with cudaErrorInvalidValue on every architecture.
    cudaFuncSetAttribute(gdnChunkScanKernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(smemB));
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

void gatedDeltaRule(LaunchContext* context, NDArray* Q, NDArray* K, NDArray* V,
                     NDArray* beta, NDArray* gate, NDArray* stateIn,
                     NDArray* actualLen, NDArray* output, NDArray* stateOut) {
    const auto B = Q->sizeAt(0);
    const auto L = Q->sizeAt(1);
    const auto H = Q->sizeAt(2);
    const auto D_k = Q->sizeAt(3);
    const auto D_v = V->sizeAt(3);

    NDArray::prepareSpecialUse({output, stateOut}, {Q, K, V, beta, gate, actualLen});
    if (stateIn != nullptr) NDArray::prepareSpecialUse({}, {stateIn});

    auto stream = context->getCudaStream();
    const LongType stateElems = B * H * D_k * D_v;

    // Float32 working state buffer on device.
    // This prevents FP16 quantization error from compounding across timesteps.
    int deviceId = sd::AffinityManager::currentDeviceId();
    float* workingState = reinterpret_cast<float*>(
        sd::memory::CudaMemoryPool::getInstance().allocate(stateElems * sizeof(float), deviceId, *stream));

    auto dtype = Q->dataType();

    if (stateIn != nullptr) {
        // Convert stateIn (type T) to float32 working buffer
        int initBlocks = (stateElems + 255) / 256;
        if (dtype == DataType::FLOAT32) {
            // stateIn is already float32, just memcpy
            cudaMemcpyAsync(workingState, stateIn->specialBuffer(),
                           stateElems * sizeof(float), cudaMemcpyDeviceToDevice, *stream);
        } else if (dtype == DataType::HALF) {
            stateInToFloat32Kernel<float16><<<initBlocks, 256, 0, *stream>>>(
                reinterpret_cast<const float16*>(stateIn->specialBuffer()),
                workingState, stateElems);
        } else if (dtype == DataType::DOUBLE) {
            stateInToFloat32Kernel<double><<<initBlocks, 256, 0, *stream>>>(
                reinterpret_cast<const double*>(stateIn->specialBuffer()),
                workingState, stateElems);
        }
    } else {
        cudaMemsetAsync(workingState, 0, stateElems * sizeof(float), *stream);
    }

    // Chunked WY path: faster prefill for T >= C=64, when:
    //   - no actualLen masking (chunked kernel doesn't support partial-sequence masking)
    //   - Dv divisible by DV_BLK=32 (threadblock tiling requirement)
    //   - Dk <= 128 (staged-tile shared-memory constraint)
    //   - dtype is FLOAT32 or HALF (chunked kernels only support these; DOUBLE uses sequential)
    const bool useChunked = (L >= GDN_CHUNK)
        && (actualLen == nullptr)
        && (D_v % GDN_DV_BLK == 0)
        && (D_k <= 128)
        && (dtype == DataType::FLOAT32 || dtype == DataType::HALF);

    if (useChunked) {
        // The chunked scan kernel writes stateOut directly as float32 (workingStateOut).
        // We need a separate float32 output-state buffer.
        float* workingStateOut = reinterpret_cast<float*>(
            sd::memory::CudaMemoryPool::getInstance().allocate(stateElems * sizeof(float), deviceId, *stream));

        if (dtype == DataType::FLOAT32) {
            launchGatedDeltaRuleChunked<float>(
                reinterpret_cast<const float*>(Q->specialBuffer()),
                reinterpret_cast<const float*>(K->specialBuffer()),
                reinterpret_cast<const float*>(V->specialBuffer()),
                reinterpret_cast<const float*>(beta->specialBuffer()),
                reinterpret_cast<const float*>(gate->specialBuffer()),
                workingState, workingStateOut,
                reinterpret_cast<float*>(output->specialBuffer()),
                B, L, H, D_k, D_v,
                Q->strideAt(0), Q->strideAt(1), Q->strideAt(2), Q->strideAt(3),
                K->strideAt(0), K->strideAt(1), K->strideAt(2), K->strideAt(3),
                V->strideAt(0), V->strideAt(1), V->strideAt(2), V->strideAt(3),
                beta->strideAt(0), beta->strideAt(1), beta->strideAt(2),
                gate->strideAt(0), gate->strideAt(1), gate->strideAt(2),
                output->strideAt(0), output->strideAt(1), output->strideAt(2), output->strideAt(3),
                *stream);
        } else {  // HALF
            launchGatedDeltaRuleChunked<float16>(
                reinterpret_cast<const float16*>(Q->specialBuffer()),
                reinterpret_cast<const float16*>(K->specialBuffer()),
                reinterpret_cast<const float16*>(V->specialBuffer()),
                reinterpret_cast<const float16*>(beta->specialBuffer()),
                reinterpret_cast<const float16*>(gate->specialBuffer()),
                workingState, workingStateOut,
                reinterpret_cast<float16*>(output->specialBuffer()),
                B, L, H, D_k, D_v,
                Q->strideAt(0), Q->strideAt(1), Q->strideAt(2), Q->strideAt(3),
                K->strideAt(0), K->strideAt(1), K->strideAt(2), K->strideAt(3),
                V->strideAt(0), V->strideAt(1), V->strideAt(2), V->strideAt(3),
                beta->strideAt(0), beta->strideAt(1), beta->strideAt(2),
                gate->strideAt(0), gate->strideAt(1), gate->strideAt(2),
                output->strideAt(0), output->strideAt(1), output->strideAt(2), output->strideAt(3),
                *stream);
        }

        // workingStateOut is float32 in [B,H,Dk,Dv] layout — convert to stateOut dtype
        int copyBlocks = (stateElems + 255) / 256;
        if (dtype == DataType::FLOAT32) {
            cudaMemcpyAsync(stateOut->specialBuffer(), workingStateOut,
                           stateElems * sizeof(float), cudaMemcpyDeviceToDevice, *stream);
        } else {  // HALF
            float32ToStateOutKernel<float16><<<copyBlocks, 256, 0, *stream>>>(
                workingStateOut, reinterpret_cast<float16*>(stateOut->specialBuffer()), stateElems);
        }

        sd::memory::CudaMemoryPool::getInstance().free(workingStateOut, deviceId, *stream);
    } else {
        // Sequential path — oracle for parity tests and fallback for unsupported configs
        if (dtype == DataType::FLOAT32) {
            launchGatedDeltaRule<float>(
                reinterpret_cast<const float*>(Q->specialBuffer()),
                reinterpret_cast<const float*>(K->specialBuffer()),
                reinterpret_cast<const float*>(V->specialBuffer()),
                reinterpret_cast<const float*>(beta->specialBuffer()),
                reinterpret_cast<const float*>(gate->specialBuffer()),
                actualLen ? reinterpret_cast<const LongType*>(actualLen->specialBuffer()) : nullptr,
                workingState,
                reinterpret_cast<float*>(output->specialBuffer()),
                B, L, H, D_k, D_v,
                Q->strideAt(0), Q->strideAt(1), Q->strideAt(2), Q->strideAt(3),
                K->strideAt(0), K->strideAt(1), K->strideAt(2), K->strideAt(3),
                V->strideAt(0), V->strideAt(1), V->strideAt(2), V->strideAt(3),
                beta->strideAt(0), beta->strideAt(1), beta->strideAt(2),
                gate->strideAt(0), gate->strideAt(1), gate->strideAt(2),
                output->strideAt(0), output->strideAt(1), output->strideAt(2), output->strideAt(3),
                *stream);
        } else if (dtype == DataType::DOUBLE) {
            launchGatedDeltaRule<double>(
                reinterpret_cast<const double*>(Q->specialBuffer()),
                reinterpret_cast<const double*>(K->specialBuffer()),
                reinterpret_cast<const double*>(V->specialBuffer()),
                reinterpret_cast<const double*>(beta->specialBuffer()),
                reinterpret_cast<const double*>(gate->specialBuffer()),
                actualLen ? reinterpret_cast<const LongType*>(actualLen->specialBuffer()) : nullptr,
                workingState,
                reinterpret_cast<double*>(output->specialBuffer()),
                B, L, H, D_k, D_v,
                Q->strideAt(0), Q->strideAt(1), Q->strideAt(2), Q->strideAt(3),
                K->strideAt(0), K->strideAt(1), K->strideAt(2), K->strideAt(3),
                V->strideAt(0), V->strideAt(1), V->strideAt(2), V->strideAt(3),
                beta->strideAt(0), beta->strideAt(1), beta->strideAt(2),
                gate->strideAt(0), gate->strideAt(1), gate->strideAt(2),
                output->strideAt(0), output->strideAt(1), output->strideAt(2), output->strideAt(3),
                *stream);
        } else if (dtype == DataType::HALF) {
            launchGatedDeltaRule<float16>(
                reinterpret_cast<const float16*>(Q->specialBuffer()),
                reinterpret_cast<const float16*>(K->specialBuffer()),
                reinterpret_cast<const float16*>(V->specialBuffer()),
                reinterpret_cast<const float16*>(beta->specialBuffer()),
                reinterpret_cast<const float16*>(gate->specialBuffer()),
                actualLen ? reinterpret_cast<const LongType*>(actualLen->specialBuffer()) : nullptr,
                workingState,
                reinterpret_cast<float16*>(output->specialBuffer()),
                B, L, H, D_k, D_v,
                Q->strideAt(0), Q->strideAt(1), Q->strideAt(2), Q->strideAt(3),
                K->strideAt(0), K->strideAt(1), K->strideAt(2), K->strideAt(3),
                V->strideAt(0), V->strideAt(1), V->strideAt(2), V->strideAt(3),
                beta->strideAt(0), beta->strideAt(1), beta->strideAt(2),
                gate->strideAt(0), gate->strideAt(1), gate->strideAt(2),
                output->strideAt(0), output->strideAt(1), output->strideAt(2), output->strideAt(3),
                *stream);
        } else {
            THROW_EXCEPTION("gatedDeltaRule: Unsupported data type");
        }

        // Write back float32 working state to stateOut (type T)
        int copyBlocks = (stateElems + 255) / 256;
        if (dtype == DataType::FLOAT32) {
            cudaMemcpyAsync(stateOut->specialBuffer(), workingState,
                           stateElems * sizeof(float), cudaMemcpyDeviceToDevice, *stream);
        } else if (dtype == DataType::HALF) {
            float32ToStateOutKernel<float16><<<copyBlocks, 256, 0, *stream>>>(
                workingState, reinterpret_cast<float16*>(stateOut->specialBuffer()), stateElems);
        } else if (dtype == DataType::DOUBLE) {
            float32ToStateOutKernel<double><<<copyBlocks, 256, 0, *stream>>>(
                workingState, reinterpret_cast<double*>(stateOut->specialBuffer()), stateElems);
        }
    }

    // Free working state (input side) via pool
    sd::memory::CudaMemoryPool::getInstance().free(workingState, deviceId, *stream);

    NDArray::registerSpecialUse({output, stateOut}, {Q, K, V, beta, gate, actualLen});
    if (stateIn != nullptr) NDArray::registerSpecialUse({}, {stateIn});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
