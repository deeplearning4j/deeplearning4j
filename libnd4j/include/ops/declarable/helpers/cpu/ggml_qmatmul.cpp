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
// CPU implementation of ggml_qmatmul.
//
// Strategy: correctness-first.
//   For each output row n, decode one block of K/blockSize packed bytes into a
//   small stack-allocated float buffer, then dot it with the matching K elements
//   of the activation row.  PRAGMA_OMP_PARALLEL_FOR over N.
//
// Block math is kept byte-for-byte identical to ggml_dequantize.cpp so tests can
// use the dequantize path as a reference.
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_ggml_qmatmul)

#include <ops/declarable/helpers/ggml_qmatmul.h>
#include <ops/declarable/helpers/ggml_dequantize.h>
#include <system/openmp_pragmas.h>
#include <types/float16.h>
#include <helpers/ShapeUtils.h>
#include <array/NDArray.h>

#include <cstring>
#include <cmath>
#include <stdexcept>

namespace sd {
namespace ops {
namespace helpers {

// ─── portable fp16 → float (no __half) ──────────────────────────────────────
static inline float fp16ToFloatCpu(uint16_t h) {
    uint32_t sign = (h >> 15) & 1u;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    if (exp == 0) {
        if (mant == 0) { uint32_t r = sign << 31; float f; memcpy(&f, &r, 4); return f; }
        // subnormal
        while (!(mant & 0x400)) { mant <<= 1; --exp; }
        ++exp; mant &= ~0x400u;
    } else if (exp == 31) {
        uint32_t r = (sign << 31) | 0x7F800000u | (mant << 13);
        float f; memcpy(&f, &r, 4); return f;
    }
    uint32_t r = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    float f; memcpy(&f, &r, 4); return f;
}

// ─── scale/min extractor for K-quant blocks (shared with dequantize) ─────────
static inline void getScaleMinK4(int j, const uint8_t* q, int& sc, int& m) {
    if (j < 4) {
        sc = q[j] & 63;
        m  = q[j + 4] & 63;
    } else {
        sc = (q[j + 4] & 0x0F) | (((q[j - 4] >> 6) & 3) << 4);
        m  = ((q[j + 4] >> 4) & 0x0F) | (((q[j] >> 6) & 3) << 4);
    }
}

// ─── per-format block dequantizers into a caller-supplied buffer ──────────────

// Q8_0 : 34 bytes / 32 elements
static void dequantBlockQ8_0(const uint8_t* block, float* dst) {
    uint16_t dRaw; memcpy(&dRaw, block, 2);
    float d = fp16ToFloatCpu(dRaw);
    const int8_t* qs = reinterpret_cast<const int8_t*>(block + 2);
    for (int i = 0; i < 32; ++i) dst[i] = d * qs[i];
}

// Q4_K : 144 bytes / 256 elements
static void dequantBlockQ4_K(const uint8_t* block, float* dst) {
    uint16_t dRaw, dminRaw;
    memcpy(&dRaw,    block,     2);
    memcpy(&dminRaw, block + 2, 2);
    float d    = fp16ToFloatCpu(dRaw);
    float dmin = fp16ToFloatCpu(dminRaw);
    const uint8_t* scaleBytes = block + 4;
    const uint8_t* qs         = block + 16;   // 128 nibble bytes

    int is = 0;
    for (int j = 0; j < 256; j += 64, is += 2) {
        int sc1, m1, sc2, m2;
        getScaleMinK4(is,     scaleBytes, sc1, m1);
        getScaleMinK4(is + 1, scaleBytes, sc2, m2);
        float d1 = d * sc1,  m1f = dmin * m1;
        float d2 = d * sc2,  m2f = dmin * m2;
        int qIdx = j / 2;  // 32 bytes cover 64 elements
        for (int l = 0; l < 32; ++l) {
            dst[j + l]      = d1 * (qs[qIdx + l] & 0x0F) - m1f;
            dst[j + l + 32] = d2 * ((qs[qIdx + l] >> 4) & 0x0F) - m2f;
        }
    }
}

// Q6_K : 210 bytes / 256 elements
static void dequantBlockQ6_K(const uint8_t* block, float* dst) {
    const uint8_t*  ql     = block;            // 128 bytes
    const uint8_t*  qh     = block + 128;      // 64 bytes
    const int8_t*   scales = reinterpret_cast<const int8_t*>(block + 192);  // 16 bytes
    uint16_t dRaw; memcpy(&dRaw, block + 208, 2);
    float d = fp16ToFloatCpu(dRaw);

    int qlOff = 0, qhOff = 0, scOff = 0;
    for (int outer = 0; outer < 2; ++outer) {
        for (int l = 0; l < 32; ++l) {
            float q1 = (float)(((ql[qlOff + l]        & 0x0F) | (((qh[qhOff + l] >> 0) & 3) << 4)) - 32);
            float q2 = (float)(((ql[qlOff + l + 32]   & 0x0F) | (((qh[qhOff + l] >> 2) & 3) << 4)) - 32);
            float q3 = (float)((((ql[qlOff + l]        >> 4) & 0x0F) | (((qh[qhOff + l] >> 4) & 3) << 4)) - 32);
            float q4 = (float)((((ql[qlOff + l + 32]   >> 4) & 0x0F) | (((qh[qhOff + l] >> 6) & 3) << 4)) - 32);

            // Canonical Q6_K scale addressing (must match dequantize_q6_K in ggml_dequantize.cpp)
            int is = l / 16;   // 0 or 1 within this outer half
            dst[outer * 128 + l]       = d * scales[scOff + is + 0] * q1;
            dst[outer * 128 + l + 32]  = d * scales[scOff + is + 2] * q2;
            dst[outer * 128 + l + 64]  = d * scales[scOff + is + 4] * q3;
            dst[outer * 128 + l + 96]  = d * scales[scOff + is + 6] * q4;
        }
        qlOff += 64;
        qhOff += 32;
        scOff += 8;
    }
}

// ─── main CPU entry point ─────────────────────────────────────────────────────
// ggmlQMatMulBytesPerBlock and ggmlQMatMulElemsPerBlock are defined inline
// in ggml_qmatmul.h to be visible in both CPU and CUDA translation units.
void ggmlQMatMul(sd::LaunchContext* /*context*/,
                 sd::NDArray* activations,
                 sd::NDArray* packedWeights,
                 sd::NDArray* output,
                 int quantType,
                 sd::LongType N,
                 sd::LongType K,
                 int outputDtype) {

    int bpb = ggmlQMatMulBytesPerBlock(quantType);
    int epb = ggmlQMatMulElemsPerBlock(quantType);
    if (bpb < 0 || epb < 0) {
        THROW_EXCEPTION("ggml_qmatmul CPU: unsupported quantType");
    }

    // Collapse leading dims into M
    sd::LongType totalElems = 1;
    int rank = activations->rankOf();
    for (int i = 0; i < rank - 1; ++i) totalElems *= activations->sizeAt(i);
    sd::LongType M = totalElems;

    sd::LongType numBlocks = K / epb;  // blocks per weight row

    const uint8_t* wBytes = reinterpret_cast<const uint8_t*>(packedWeights->buffer());

    // Activations may be FP32 or FP16
    bool actFp16 = (activations->dataType() == sd::DataType::HALF);
    const float*   actF32 = actFp16 ? nullptr : reinterpret_cast<const float*>(activations->buffer());
    const float16* actF16 = actFp16 ? reinterpret_cast<const float16*>(activations->buffer()) : nullptr;

    bool outFp16 = (outputDtype == 1);

    // Temporary buffer for one dequantized row of K floats
    // Allocated per thread via stack (max K supported ~65536 elements)
    // For very large K, we process block-by-block accumulating into acc.

    PRAGMA_OMP_PARALLEL_FOR
    for (sd::LongType n = 0; n < N; ++n) {
        const uint8_t* wRow = wBytes + n * numBlocks * bpb;

        // Dequantize the full weight row n into a temp buffer
        // Stack-allocate up to 256 floats per block, process block by block
        float wBlock[256];  // max elems per block = 256 (Q4_K, Q6_K)

        for (sd::LongType m = 0; m < M; ++m) {
            double acc = 0.0;
            sd::LongType aRowBase = m * K;

            for (sd::LongType b = 0; b < numBlocks; ++b) {
                const uint8_t* block = wRow + b * bpb;

                // Dequantize one block
                if (quantType == 4) {         // Q8_0
                    dequantBlockQ8_0(block, wBlock);
                } else if (quantType == 8) {  // Q4_K
                    dequantBlockQ4_K(block, wBlock);
                } else {                       // Q6_K
                    dequantBlockQ6_K(block, wBlock);
                }

                // Dot product with activation slice [aRowBase + b*epb .. b*epb+epb)
                sd::LongType kBase = b * epb;
                for (int e = 0; e < epb; ++e) {
                    float av;
                    if (actFp16) {
                        av = static_cast<float>(actF16[aRowBase + kBase + e]);
                    } else {
                        av = actF32[aRowBase + kBase + e];
                    }
                    acc += (double)(av * wBlock[e]);
                }
            }

            // Write output
            sd::LongType outIdx = m * N + n;
            if (outFp16) {
                reinterpret_cast<float16*>(output->buffer())[outIdx] = static_cast<float>(acc);
            } else {
                reinterpret_cast<float*>(output->buffer())[outIdx]   = static_cast<float>(acc);
            }
        }
    }
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_ggml_qmatmul)
