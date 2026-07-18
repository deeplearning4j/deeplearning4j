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

//
// CUDA implementation of ggml_qmatmul — fused dequant-dot kernels.
//
// Grid: (N_blocks, M_blocks)  where N_blocks = ceil(N / WARP_PER_BLOCK) and
//       M_blocks  = M  (one block per output row m).
//
// GEMV / decode path (small M):
//   Each warp handles one output element (m, n).
//   Threads 0..31 each load their slice of the K-dimension packed block,
//   dequantize on-register, multiply with the activation, and warp-reduce.
//
// For larger M the kernel loops over M rows, reusing the loaded weight block.
// This is a correctness-first implementation; the GEMV (M=1) path is the
// performance target.
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_ggml_qmatmul)

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <ops/declarable/helpers/ggml_qmatmul.h>
#include <ops/declarable/helpers/ggml_dequantize.h>
#include <array/NDArray.h>
#include <helpers/DebugHelper.h>
#include <types/float16.h>
#include <ops/declarable/helpers/cuda/device_primitives.cuh>

namespace sd {
namespace ops {
namespace helpers {

// ─── device fp16 → float ─────────────────────────────────────────────────────
static SD_DEVICE SD_INLINE float devFp16ToFloat(uint16_t h) {
    __half hv;
    memcpy(&hv, &h, sizeof(__half));
    return __half2float(hv);
}

// ─── K-quant scale helper ─────────────────────────────────────────────────────
static SD_DEVICE SD_INLINE void devGetScaleMinK4(int j, const uint8_t* q, int& sc, int& m) {
    if (j < 4) {
        sc = q[j] & 63;
        m  = q[j + 4] & 63;
    } else {
        sc = (q[j + 4] & 0x0F) | (((q[j - 4] >> 6) & 3) << 4);
        m  = ((q[j + 4] >> 4) & 0x0F) | (((q[j] >> 6) & 3) << 4);
    }
}

// ─── Q8_0 fused dequant+dot kernel ───────────────────────────────────────────
// Grid: (ceil(N/WARPS_PER_BLOCK), M)   Block: (32 * WARPS_PER_BLOCK)
// Each warp computes one (m,n) output element.
// numBlocks = K / 32

#define WARPS_PER_BLOCK_QMM 4

template <typename ActT, typename OutT>
SD_KERNEL static void ggmlQMatMulQ8_0Kernel(
    const ActT*    __restrict__ act,        // [M, K]
    const uint8_t* __restrict__ wBytes,     // [N * numBlocks * 34]
    OutT*          __restrict__ out,        // [M, N]
    long long M,
    long long N,
    long long K,
    long long numBlocks)                    // K / 32
{
    // warpId within block → which n we handle
    int warpId   = threadIdx.x / 32;
    int laneId   = threadIdx.x % 32;
    long long n  = (long long)blockIdx.x * WARPS_PER_BLOCK_QMM + warpId;
    if (n >= N) return;

    long long m  = (long long)blockIdx.y;
    if (m >= M) return;

    const uint8_t* wRow = wBytes + n * numBlocks * 34;
    const ActT*    aRow = act    + m * K;

    float acc = 0.0f;

    for (long long b = 0; b < numBlocks; ++b) {
        const uint8_t* block = wRow + b * 34;
        uint16_t dRaw;
        // All lanes read the scale (2 bytes at start)
        memcpy(&dRaw, block, sizeof(uint16_t));
        float d = devFp16ToFloat(dRaw);

        // Each lane handles one int8 element (lane 0..31 → elements 0..31)
        int8_t q;
        memcpy(&q, block + 2 + laneId, 1);

        long long kIdx = b * 32 + laneId;
        float av = (kIdx < K) ? static_cast<float>(aRow[kIdx]) : 0.0f;
        acc += av * (d * static_cast<float>(q));
    }

    // Warp reduction (one warp per output element)
    acc = sd::device::warpReduceSum<float>(acc);

    if (laneId == 0) {
        long long outIdx = m * N + n;
        out[outIdx] = static_cast<OutT>(acc);
    }
}

// ─── Q4_K fused dequant+dot kernel ───────────────────────────────────────────
// numBlocks = K / 256.  Each block has 256 elements packed in 144 bytes.
// We assign 8 warps (256 threads) per (m,n) output element — each warp of
// 32 threads handles 32 elements of one half-chunk (64/2 each side per block).
//
// Simpler approach: one block per (m,n), one thread handles multiple elements.
// BLOCK_SIZE_THREADS = 128 (4 warps), each thread processes 2 elements per Q4_K block.

template <typename ActT, typename OutT>
SD_KERNEL static void ggmlQMatMulQ4KKernel(
    const ActT*    __restrict__ act,
    const uint8_t* __restrict__ wBytes,
    OutT*          __restrict__ out,
    long long M,
    long long N,
    long long K,
    long long numBlocks)   // K / 256
{
    long long n = (long long)blockIdx.x;
    long long m = (long long)blockIdx.y;
    if (n >= N || m >= M) return;

    const uint8_t* wRow = wBytes + n * numBlocks * 144;
    const ActT*    aRow = act    + m * K;

    // Each thread accumulates its share
    float acc = 0.0f;
    int tid   = threadIdx.x;                // 0..blockDim.x-1
    int nThreads = blockDim.x;

    for (long long b = 0; b < numBlocks; ++b) {
        const uint8_t* block   = wRow + b * 144;
        uint16_t dRaw, dminRaw;
        memcpy(&dRaw,    block,     2);
        memcpy(&dminRaw, block + 2, 2);
        float d    = devFp16ToFloat(dRaw);
        float dmin = devFp16ToFloat(dminRaw);
        const uint8_t* scaleBytes = block + 4;
        const uint8_t* qs         = block + 16;  // 128 nibble bytes

        long long kBase = b * 256;

        // Iterate over 4 chunks of 64 elements (2 sub-blocks per chunk: low/high nibbles)
        // Total: 4 chunks × 32 nibble-bytes × 2 nibbles = 256 elements
        // Distribute 256 elements across threads in strided fashion
        for (int e = tid; e < 256; e += nThreads) {
            int chunk  = e / 64;  // 0..3 → maps to is = chunk*2
            int pos    = e % 64;  // position within chunk
            bool hiNib = (pos >= 32);
            int  l     = pos % 32;

            int is = chunk * 2;
            int sc1, m1, sc2, m2;
            devGetScaleMinK4(is,     scaleBytes, sc1, m1);
            devGetScaleMinK4(is + 1, scaleBytes, sc2, m2);

            int qIdx = chunk * 32 + l;
            uint8_t qByte = qs[qIdx];
            int nibble = hiNib ? ((qByte >> 4) & 0x0F) : (qByte & 0x0F);
            float sc = hiNib ? (float)sc2 : (float)sc1;
            float mf = hiNib ? (float)m2  : (float)m1;

            float wVal = d * sc * (float)nibble - dmin * mf;
            long long kIdx = kBase + e;
            if (kIdx < K) {
                acc += static_cast<float>(aRow[kIdx]) * wVal;
            }
        }
    }

    // Block reduction (result on thread 0)
    __shared__ float warpBuf[32];
    float total = sd::device::blockReduceSum<float>(acc, warpBuf);
    if (tid == 0) {
        out[m * N + n] = static_cast<OutT>(total);
    }
}

// ─── Q6_K fused dequant+dot kernel ───────────────────────────────────────────
// numBlocks = K / 256.  Block: 128 threads, each handles 2 elements per Q6_K block.

template <typename ActT, typename OutT>
SD_KERNEL static void ggmlQMatMulQ6KKernel(
    const ActT*    __restrict__ act,
    const uint8_t* __restrict__ wBytes,
    OutT*          __restrict__ out,
    long long M,
    long long N,
    long long K,
    long long numBlocks)
{
    long long n = (long long)blockIdx.x;
    long long m = (long long)blockIdx.y;
    if (n >= N || m >= M) return;

    const uint8_t* wRow = wBytes + n * numBlocks * 210;
    const ActT*    aRow = act    + m * K;

    float acc = 0.0f;
    int tid     = threadIdx.x;
    int nThreads = blockDim.x;

    for (long long b = 0; b < numBlocks; ++b) {
        const uint8_t* block   = wRow + b * 210;
        const uint8_t*  ql     = block;
        const uint8_t*  qh     = block + 128;
        const int8_t*   scales = reinterpret_cast<const int8_t*>(block + 192);
        uint16_t dRaw;
        memcpy(&dRaw, block + 208, 2);
        float d = devFp16ToFloat(dRaw);

        long long kBase = b * 256;

        // 256 elements distributed across threads
        for (int e = tid; e < 256; e += nThreads) {
            int outer = e / 128;    // 0 or 1
            int inner = e % 128;    // position within half

            int l       = inner % 32;
            int quadrant = inner / 32;  // 0..3

            // Compute 6-bit value
            int qlOff = outer * 64;
            int qhOff = outer * 32;

            float q6;
            uint8_t qlByte, qhByte;
            if (quadrant == 0) {
                qlByte = ql[qlOff + l];
                qhByte = qh[qhOff + l];
                q6 = (float)(((qlByte & 0x0F) | (((qhByte >> 0) & 3) << 4)) - 32);
            } else if (quadrant == 1) {
                qlByte = ql[qlOff + l + 32];
                qhByte = qh[qhOff + l];
                q6 = (float)(((qlByte & 0x0F) | (((qhByte >> 2) & 3) << 4)) - 32);
            } else if (quadrant == 2) {
                qlByte = ql[qlOff + l];
                qhByte = qh[qhOff + l];
                q6 = (float)((((qlByte >> 4) & 0x0F) | (((qhByte >> 4) & 3) << 4)) - 32);
            } else {
                qlByte = ql[qlOff + l + 32];
                qhByte = qh[qhOff + l];
                q6 = (float)((((qlByte >> 4) & 0x0F) | (((qhByte >> 6) & 3) << 4)) - 32);
            }

            // Canonical Q6_K scale addressing (must match dequantize_q6_K in
            // helpers/cpu/ggml_dequantize.cpp): scales[outer*8 + l/16 + quadrant*2], range 0..15.
            int scOff = outer * 8;
            int scIdx = scOff + (l / 16) + quadrant * 2;

            float wVal = d * (float)scales[scIdx] * q6;
            long long kIdx = kBase + e;
            if (kIdx < K) {
                acc += static_cast<float>(aRow[kIdx]) * wVal;
            }
        }
    }

    // Block reduction (result on thread 0)
    __shared__ float warpBuf[32];
    float total = sd::device::blockReduceSum<float>(acc, warpBuf);
    if (tid == 0) {
        out[m * N + n] = static_cast<OutT>(total);
    }
}

// ─── template dispatch launcher ──────────────────────────────────────────────
template <typename ActT, typename OutT>
static void launchQMatMul(const ActT* act, const uint8_t* wBytes, OutT* out,
                           long long M, long long N, long long K,
                           int quantType, cudaStream_t stream) {
    if (quantType == 4) {  // Q8_0
        long long numBlocks = K / 32;
        // Grid: (ceil(N/WARPS_PER_BLOCK), M)   Block: (32 * WARPS_PER_BLOCK)
        dim3 grid((int)((N + WARPS_PER_BLOCK_QMM - 1) / WARPS_PER_BLOCK_QMM), (int)M);
        dim3 block(32 * WARPS_PER_BLOCK_QMM);
        ggmlQMatMulQ8_0Kernel<ActT, OutT><<<grid, block, 0, stream>>>(
            act, wBytes, out, M, N, K, numBlocks);
    } else if (quantType == 8) {  // Q4_K
        long long numBlocks = K / 256;
        // One block per (n, m), 128 threads
        dim3 grid((int)N, (int)M);
        dim3 block(128);
        ggmlQMatMulQ4KKernel<ActT, OutT><<<grid, block, 0, stream>>>(
            act, wBytes, out, M, N, K, numBlocks);
    } else if (quantType == 10) {  // Q6_K
        long long numBlocks = K / 256;
        dim3 grid((int)N, (int)M);
        dim3 block(128);
        ggmlQMatMulQ6KKernel<ActT, OutT><<<grid, block, 0, stream>>>(
            act, wBytes, out, M, N, K, numBlocks);
    } else {
        THROW_EXCEPTION("ggml_qmatmul CUDA: unsupported quantType");
    }
}

// ─── public entry point ───────────────────────────────────────────────────────
void ggmlQMatMul(sd::LaunchContext* context,
                 sd::NDArray* activations,
                 sd::NDArray* packedWeights,
                 sd::NDArray* output,
                 int quantType,
                 sd::LongType N,
                 sd::LongType K,
                 int outputDtype) {

    if (ggmlQMatMulBytesPerBlock(quantType) < 0) {
        THROW_EXCEPTION("ggml_qmatmul CUDA: unsupported quantType");
    }

    NDArray::prepareSpecialUse({output}, {activations, packedWeights});
    auto stream = context->getCudaStream();

    // Collapse leading dims into M
    sd::LongType M = 1;
    for (int i = 0; i < activations->rankOf() - 1; ++i) M *= activations->sizeAt(i);

    const uint8_t* wDev = reinterpret_cast<const uint8_t*>(packedWeights->specialBuffer());

    bool actFp16 = (activations->dataType() == sd::DataType::HALF);
    bool outFp16 = (outputDtype == 1);

    if (!actFp16 && !outFp16) {
        launchQMatMul<float, float>(
            reinterpret_cast<const float*>(activations->specialBuffer()),
            wDev,
            reinterpret_cast<float*>(output->specialBuffer()),
            M, N, K, quantType, *stream);
    } else if (actFp16 && !outFp16) {
        launchQMatMul<float16, float>(
            reinterpret_cast<const float16*>(activations->specialBuffer()),
            wDev,
            reinterpret_cast<float*>(output->specialBuffer()),
            M, N, K, quantType, *stream);
    } else if (!actFp16 && outFp16) {
        launchQMatMul<float, float16>(
            reinterpret_cast<const float*>(activations->specialBuffer()),
            wDev,
            reinterpret_cast<float16*>(output->specialBuffer()),
            M, N, K, quantType, *stream);
    } else {
        launchQMatMul<float16, float16>(
            reinterpret_cast<const float16*>(activations->specialBuffer()),
            wDev,
            reinterpret_cast<float16*>(output->specialBuffer()),
            M, N, K, quantType, *stream);
    }

    NDArray::registerSpecialUse({output}, {activations, packedWeights});
}

// ggmlQMatMulBytesPerBlock and ggmlQMatMulElemsPerBlock are declared as
// static SD_INLINE in ggml_qmatmul.h — each TU gets its own inline copy.
// No redefinition needed here.

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_ggml_qmatmul)
