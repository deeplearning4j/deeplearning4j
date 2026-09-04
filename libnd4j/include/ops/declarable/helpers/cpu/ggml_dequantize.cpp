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
// Standalone CPU dequantization kernels for GGML quantization formats.
// No external dependencies — this is our own baseline implementation.
//

#include <execution/Threads.h>
#include <math/templatemath.h>
#include <ops/declarable/helpers/ggml_dequantize.h>
#include <types/float16.h>
#include <cmath>
#include <cstring>
#include <vector>

#if NOT_EXCLUDED(OP_ggml_dequantize)

namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
// FP16 conversion utility
//////////////////////////////////////////////////////////////////////////
static SD_INLINE float fp16ToFloat(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;

    if (exponent == 0) {
        if (mantissa == 0) return sign ? -0.0f : 0.0f;
        // Subnormal
        float val = mantissa / 1024.0f * std::pow(2.0f, -14.0f);
        return sign ? -val : val;
    } else if (exponent == 31) {
        return mantissa == 0 ? (sign ? -INFINITY : INFINITY) : NAN;
    }

    float val = (1.0f + mantissa / 1024.0f) * std::pow(2.0f, (int)exponent - 15);
    return sign ? -val : val;
}

//////////////////////////////////////////////////////////////////////////
// Block constants
//////////////////////////////////////////////////////////////////////////
static constexpr int QK_K = 256;   // K-quant super-block size
static constexpr int QK4_0 = 32;   // Q4_0 block size
static constexpr int QK4_1 = 32;   // Q4_1 block size
static constexpr int QK5_0 = 32;   // Q5_0 block size
static constexpr int QK5_1 = 32;   // Q5_1 block size
static constexpr int QK8_0 = 32;   // Q8_0 block size
static constexpr int QK8_1 = 32;   // Q8_1 block size

//////////////////////////////////////////////////////////////////////////
// Q4_0: 4-bit quantization, type 0
// Block: 2 bytes (FP16 d) + 16 bytes (4-bit quants) = 18 bytes per 32 elements
//////////////////////////////////////////////////////////////////////////
static void dequantize_q4_0(const uint8_t* data, float* output, LongType numElements) {
    constexpr int BLOCK_SIZE = 18;
    LongType numBlocks = (numElements + QK4_0 - 1) / QK4_0;

    PRAGMA_OMP_PARALLEL_FOR
    for (LongType b = 0; b < numBlocks; b++) {
        const uint8_t* block = data + b * BLOCK_SIZE;
        uint16_t dRaw;
        memcpy(&dRaw, block, 2);
        float d = fp16ToFloat(dRaw);
        const uint8_t* qs = block + 2;
        LongType outBase = b * QK4_0;

        for (int j = 0; j < QK4_0 / 2; j++) {
            LongType idx0 = outBase + j * 2;
            LongType idx1 = outBase + j * 2 + 1;
            int v0 = (qs[j] & 0x0F) - 8;
            int v1 = ((qs[j] >> 4) & 0x0F) - 8;
            if (idx0 < numElements) output[idx0] = d * v0;
            if (idx1 < numElements) output[idx1] = d * v1;
        }
    }
}

//////////////////////////////////////////////////////////////////////////
// Q4_1: 4-bit quantization, type 1
// Block: 2 bytes (FP16 d) + 2 bytes (FP16 m) + 16 bytes (4-bit quants) = 20 bytes per 32 elements
//////////////////////////////////////////////////////////////////////////
static void dequantize_q4_1(const uint8_t* data, float* output, LongType numElements) {
    constexpr int BLOCK_SIZE = 20;
    LongType numBlocks = (numElements + QK4_1 - 1) / QK4_1;

    PRAGMA_OMP_PARALLEL_FOR
    for (LongType b = 0; b < numBlocks; b++) {
        const uint8_t* block = data + b * BLOCK_SIZE;
        uint16_t dRaw, mRaw;
        memcpy(&dRaw, block, 2);
        memcpy(&mRaw, block + 2, 2);
        float d = fp16ToFloat(dRaw);
        float m = fp16ToFloat(mRaw);
        const uint8_t* qs = block + 4;
        LongType outBase = b * QK4_1;

        for (int j = 0; j < QK4_1 / 2; j++) {
            LongType idx0 = outBase + j * 2;
            LongType idx1 = outBase + j * 2 + 1;
            int v0 = qs[j] & 0x0F;
            int v1 = (qs[j] >> 4) & 0x0F;
            if (idx0 < numElements) output[idx0] = d * v0 + m;
            if (idx1 < numElements) output[idx1] = d * v1 + m;
        }
    }
}

//////////////////////////////////////////////////////////////////////////
// Q5_0: 5-bit quantization, type 0
// Block: 2 bytes (FP16 d) + 4 bytes (high bits) + 16 bytes (low 4 bits) = 22 bytes per 32 elements
//////////////////////////////////////////////////////////////////////////
static void dequantize_q5_0(const uint8_t* data, float* output, LongType numElements) {
    constexpr int BLOCK_SIZE = 22;
    LongType numBlocks = (numElements + QK5_0 - 1) / QK5_0;

    PRAGMA_OMP_PARALLEL_FOR
    for (LongType b = 0; b < numBlocks; b++) {
        const uint8_t* block = data + b * BLOCK_SIZE;
        uint16_t dRaw;
        memcpy(&dRaw, block, 2);
        float d = fp16ToFloat(dRaw);

        uint32_t qh;
        memcpy(&qh, block + 2, 4);
        const uint8_t* qs = block + 6;
        LongType outBase = b * QK5_0;

        // Q5_0 GGML element layout (from dequantize_row_q5_0 reference):
        //   qs[j] low nibble  → element j      (j = 0..15)
        //   qs[j] high nibble → element j + 16 (j = 0..15)
        //   qh bit j          → 5th bit of element j
        //   qh bit (j + 16)   → 5th bit of element j + 16
        for (int j = 0; j < QK5_0 / 2; j++) {
            LongType idx0 = outBase + j;          // element j (first half)
            LongType idx1 = outBase + j + 16;     // element j+16 (second half)
            int xh0 = ((qh >> j) & 1) << 4;
            int xh1 = ((qh >> (j + 16)) & 1) << 4;
            int v0 = (qs[j] & 0x0F) | xh0;
            int v1 = ((qs[j] >> 4) & 0x0F) | xh1;
            if (idx0 < numElements) output[idx0] = d * (v0 - 16);
            if (idx1 < numElements) output[idx1] = d * (v1 - 16);
        }
    }
}

//////////////////////////////////////////////////////////////////////////
// Q5_1: 5-bit quantization, type 1
// Block: 2 bytes (FP16 d) + 2 bytes (FP16 m) + 4 bytes (high bits) + 16 bytes (low 4 bits) = 24 bytes per 32 elements
//////////////////////////////////////////////////////////////////////////
static void dequantize_q5_1(const uint8_t* data, float* output, LongType numElements) {
    constexpr int BLOCK_SIZE = 24;
    LongType numBlocks = (numElements + QK5_1 - 1) / QK5_1;

    PRAGMA_OMP_PARALLEL_FOR
    for (LongType b = 0; b < numBlocks; b++) {
        const uint8_t* block = data + b * BLOCK_SIZE;
        uint16_t dRaw, mRaw;
        memcpy(&dRaw, block, 2);
        memcpy(&mRaw, block + 2, 2);
        float d = fp16ToFloat(dRaw);
        float m = fp16ToFloat(mRaw);

        uint32_t qh;
        memcpy(&qh, block + 4, 4);
        const uint8_t* qs = block + 8;
        LongType outBase = b * QK5_1;

        // Q5_1 GGML element layout (same as Q5_0 but unsigned with min offset):
        //   qs[j] low nibble  → element j      (j = 0..15)
        //   qs[j] high nibble → element j + 16 (j = 0..15)
        //   qh bit j          → 5th bit of element j
        //   qh bit (j + 16)   → 5th bit of element j + 16
        for (int j = 0; j < QK5_1 / 2; j++) {
            LongType idx0 = outBase + j;          // element j (first half)
            LongType idx1 = outBase + j + 16;     // element j+16 (second half)
            int xh0 = ((qh >> j) & 1) << 4;
            int xh1 = ((qh >> (j + 16)) & 1) << 4;
            int v0 = (qs[j] & 0x0F) | xh0;
            int v1 = ((qs[j] >> 4) & 0x0F) | xh1;
            if (idx0 < numElements) output[idx0] = d * v0 + m;
            if (idx1 < numElements) output[idx1] = d * v1 + m;
        }
    }
}

//////////////////////////////////////////////////////////////////////////
// Q8_0: 8-bit quantization, type 0
// Block: 2 bytes (FP16 d) + 32 bytes (int8 quants) = 34 bytes per 32 elements
//////////////////////////////////////////////////////////////////////////
static void dequantize_q8_0(const uint8_t* data, float* output, LongType numElements) {
    constexpr int BLOCK_SIZE = 34;
    LongType numBlocks = (numElements + QK8_0 - 1) / QK8_0;

    PRAGMA_OMP_PARALLEL_FOR
    for (LongType b = 0; b < numBlocks; b++) {
        const uint8_t* block = data + b * BLOCK_SIZE;
        uint16_t dRaw;
        memcpy(&dRaw, block, 2);
        float d = fp16ToFloat(dRaw);
        const int8_t* qs = reinterpret_cast<const int8_t*>(block + 2);
        LongType outBase = b * QK8_0;

        for (int j = 0; j < QK8_0; j++) {
            LongType idx = outBase + j;
            if (idx < numElements) output[idx] = d * qs[j];
        }
    }
}

//////////////////////////////////////////////////////////////////////////
// Q8_1: 8-bit quantization, type 1
// Block: 4 bytes (FP32 d) + 4 bytes (FP32 s) + 32 bytes (int8 quants) = 40 bytes per 32 elements
//////////////////////////////////////////////////////////////////////////
static void dequantize_q8_1(const uint8_t* data, float* output, LongType numElements) {
    constexpr int BLOCK_SIZE = 40;
    LongType numBlocks = (numElements + QK8_1 - 1) / QK8_1;

    PRAGMA_OMP_PARALLEL_FOR
    for (LongType b = 0; b < numBlocks; b++) {
        const uint8_t* block = data + b * BLOCK_SIZE;
        float d, s;
        memcpy(&d, block, 4);
        memcpy(&s, block + 4, 4);
        const int8_t* qs = reinterpret_cast<const int8_t*>(block + 8);
        LongType outBase = b * QK8_1;

        for (int j = 0; j < QK8_1; j++) {
            LongType idx = outBase + j;
            if (idx < numElements) output[idx] = d * qs[j] + s;
        }
    }
}

//////////////////////////////////////////////////////////////////////////
// K-quant helper: get_scale_min_k4
//////////////////////////////////////////////////////////////////////////
static SD_INLINE void get_scale_min_k4(int j, const uint8_t* q, int& sc, int& m) {
    if (j < 4) {
        sc = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        sc = ((q[j + 4] & 0xF)) | (((q[j - 4] >> 6) & 3) << 4);
        m = ((q[j + 4] >> 4) & 0xF) | (((q[j] >> 6) & 3) << 4);
    }
}

//////////////////////////////////////////////////////////////////////////
// Q2_K: 2-bit K-quant
// Block: 256/16 * 2 bytes scales + 256/4 bytes qs + 2 bytes d + 2 bytes dmin = 84 bytes per 256 elements
//////////////////////////////////////////////////////////////////////////
static void dequantize_q2_K(const uint8_t* data, float* output, LongType numElements) {
    constexpr int BLOCK_SIZE = 84;  // 16 + 64 + 2 + 2
    LongType numBlocks = (numElements + QK_K - 1) / QK_K;

    PRAGMA_OMP_PARALLEL_FOR
    for (LongType b = 0; b < numBlocks; b++) {
        const uint8_t* block = data + b * BLOCK_SIZE;
        const uint8_t* scales = block;
        const uint8_t* qs = block + 16;
        uint16_t dRaw, dminRaw;
        memcpy(&dRaw, block + 80, 2);
        memcpy(&dminRaw, block + 82, 2);
        float d = fp16ToFloat(dRaw);
        float dmin = fp16ToFloat(dminRaw);
        LongType outBase = b * QK_K;

        int qIdx = 0;
        int localOff = 0;
        for (int i = 0; i < QK_K; i += 128) {
            for (int l = 0; l < 32; l++) {
                int is = i / 16 + l / 16;
                int sc = scales[is];
                float dl = d * (sc & 0xF);
                float ml = dmin * (sc >> 4);
                LongType idx = outBase + localOff++;
                if (idx < numElements) output[idx] = dl * (qs[qIdx + l] & 3) - ml;
            }
            for (int l = 0; l < 32; l++) {
                int is = i / 16 + l / 16 + 2;
                int sc = scales[is];
                float dl = d * (sc & 0xF);
                float ml = dmin * (sc >> 4);
                LongType idx = outBase + localOff++;
                if (idx < numElements) output[idx] = dl * ((qs[qIdx + l] >> 2) & 3) - ml;
            }
            for (int l = 0; l < 32; l++) {
                int is = i / 16 + l / 16 + 4;
                int sc = scales[is];
                float dl = d * (sc & 0xF);
                float ml = dmin * (sc >> 4);
                LongType idx = outBase + localOff++;
                if (idx < numElements) output[idx] = dl * ((qs[qIdx + l] >> 4) & 3) - ml;
            }
            for (int l = 0; l < 32; l++) {
                int is = i / 16 + l / 16 + 6;
                int sc = scales[is];
                float dl = d * (sc & 0xF);
                float ml = dmin * (sc >> 4);
                LongType idx = outBase + localOff++;
                if (idx < numElements) output[idx] = dl * ((qs[qIdx + l] >> 6) & 3) - ml;
            }
            qIdx += 32;
        }
    }
}

//////////////////////////////////////////////////////////////////////////
// Q3_K: 3-bit K-quant
// Block: 256/8 bytes hmask + 256/4*3 bytes qs + 12 bytes scales + 2 bytes d = 110 bytes per 256 elements
//////////////////////////////////////////////////////////////////////////
static void dequantize_q3_K(const uint8_t* data, float* output, LongType numElements) {
    constexpr int BLOCK_SIZE = 110;  // 32 + 64 + 12 + 2
    LongType numBlocks = (numElements + QK_K - 1) / QK_K;

    PRAGMA_OMP_PARALLEL_FOR
    for (LongType b = 0; b < numBlocks; b++) {
        const uint8_t* block = data + b * BLOCK_SIZE;
        const uint8_t* hmask = block;
        const uint8_t* qs = block + 32;
        const uint8_t* scaleBytes = block + 96;
        uint16_t dRaw;
        memcpy(&dRaw, block + 108, 2);
        float d = fp16ToFloat(dRaw);
        LongType outBase = b * QK_K;

        // Q3_K stores eight 4-bit low scale nibbles and sixteen 2-bit high
        // scale bits in the final four bytes. Reconstruct the canonical
        // signed 6-bit scales before decoding the four 2-bit planes.
        constexpr uint32_t kmask1 = 0x03030303;
        constexpr uint32_t kmask2 = 0x0f0f0f0f;
        uint32_t aux[4];
        const int8_t* scales = reinterpret_cast<const int8_t*>(aux);
        memcpy(aux, scaleBytes, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        uint8_t m = 1;
        int is = 0;
        int localOff = 0;
        const uint8_t* q = qs;
        const uint8_t* hm = hmask;
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                float dl = d * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    int value = static_cast<int>((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4);
                    LongType idx = outBase + localOff++;
                    if (idx < numElements) output[idx] = dl * value;
                }

                dl = d * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    int value = static_cast<int>((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4);
                    LongType idx = outBase + localOff++;
                    if (idx < numElements) output[idx] = dl * value;
                }

                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
    }
}

//////////////////////////////////////////////////////////////////////////
// Q4_K: 4-bit K-quant
// Block: 2 bytes d + 2 bytes dmin + 12 bytes scales + 128 bytes qs = 144 bytes per 256 elements
//////////////////////////////////////////////////////////////////////////
static void dequantize_q4_K(const uint8_t* data, float* output, LongType numElements) {
    constexpr int BLOCK_SIZE = 144;
    LongType numBlocks = (numElements + QK_K - 1) / QK_K;

    PRAGMA_OMP_PARALLEL_FOR
    for (LongType b = 0; b < numBlocks; b++) {
        const uint8_t* block = data + b * BLOCK_SIZE;
        uint16_t dRaw, dminRaw;
        memcpy(&dRaw, block, 2);
        memcpy(&dminRaw, block + 2, 2);
        float d = fp16ToFloat(dRaw);
        float dmin = fp16ToFloat(dminRaw);
        const uint8_t* scaleBytes = block + 4;
        const uint8_t* qs = block + 16;
        LongType outBase = b * QK_K;

        int is = 0;
        int qIdx = 0;
        int localOff = 0;
        for (int j = 0; j < QK_K; j += 64) {
            int sc1, m1, sc2, m2;
            get_scale_min_k4(is, scaleBytes, sc1, m1);
            float d1 = d * sc1;
            float m1f = dmin * m1;
            get_scale_min_k4(is + 1, scaleBytes, sc2, m2);
            float d2 = d * sc2;
            float m2f = dmin * m2;

            for (int l = 0; l < 32; l++) {
                int val = qs[qIdx + l] & 0x0F;
                LongType idx = outBase + localOff++;
                if (idx < numElements) output[idx] = d1 * val - m1f;
            }
            for (int l = 0; l < 32; l++) {
                int val = (qs[qIdx + l] >> 4) & 0x0F;
                LongType idx = outBase + localOff++;
                if (idx < numElements) output[idx] = d2 * val - m2f;
            }
            qIdx += 32;
            is += 2;
        }
    }
}

//////////////////////////////////////////////////////////////////////////
// Q5_K: 5-bit K-quant
// Block: 2 bytes d + 2 bytes dmin + 12 bytes scales + 32 bytes qh + 128 bytes qs = 176 bytes per 256 elements
//////////////////////////////////////////////////////////////////////////
static void dequantize_q5_K(const uint8_t* data, float* output, LongType numElements) {
    constexpr int BLOCK_SIZE = 176;
    LongType numBlocks = (numElements + QK_K - 1) / QK_K;

    PRAGMA_OMP_PARALLEL_FOR
    for (LongType b = 0; b < numBlocks; b++) {
        const uint8_t* block = data + b * BLOCK_SIZE;
        uint16_t dRaw, dminRaw;
        memcpy(&dRaw, block, 2);
        memcpy(&dminRaw, block + 2, 2);
        float d = fp16ToFloat(dRaw);
        float dmin = fp16ToFloat(dminRaw);
        const uint8_t* scaleBytes = block + 4;
        const uint8_t* qh = block + 16;
        const uint8_t* qs = block + 48;
        LongType outBase = b * QK_K;

        int is = 0;
        int qlOff = 0;
        uint8_t u1 = 1, u2 = 2;
        int localOff = 0;

        for (int j = 0; j < QK_K; j += 64) {
            int sc1, m1, sc2, m2;
            get_scale_min_k4(is, scaleBytes, sc1, m1);
            float d1 = d * sc1;
            float m1f = dmin * m1;
            get_scale_min_k4(is + 1, scaleBytes, sc2, m2);
            float d2 = d * sc2;
            float m2f = dmin * m2;

            for (int l = 0; l < 32; l++) {
                int lowVal = qs[qlOff + l] & 0x0F;
                int highBit = (qh[l] & u1) ? 16 : 0;
                LongType idx = outBase + localOff++;
                if (idx < numElements) output[idx] = d1 * (lowVal + highBit) - m1f;
            }
            for (int l = 0; l < 32; l++) {
                int highNibble = (qs[qlOff + l] >> 4) & 0x0F;
                int highBit = (qh[l] & u2) ? 16 : 0;
                LongType idx = outBase + localOff++;
                if (idx < numElements) output[idx] = d2 * (highNibble + highBit) - m2f;
            }
            qlOff += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

//////////////////////////////////////////////////////////////////////////
// Q6_K: 6-bit K-quant
// Block: 128 bytes ql + 64 bytes qh + 16 bytes scales + 2 bytes d = 210 bytes per 256 elements
//////////////////////////////////////////////////////////////////////////
static void dequantize_q6_K(const uint8_t* data, float* output, LongType numElements) {
    constexpr int BLOCK_SIZE = 210;
    LongType numBlocks = (numElements + QK_K - 1) / QK_K;

    PRAGMA_OMP_PARALLEL_FOR
    for (LongType b = 0; b < numBlocks; b++) {
        const uint8_t* block = data + b * BLOCK_SIZE;
        const uint8_t* ql = block;
        const uint8_t* qh = block + 128;
        const int8_t* scales = reinterpret_cast<const int8_t*>(block + 192);
        uint16_t dRaw;
        memcpy(&dRaw, block + 208, 2);
        float d = fp16ToFloat(dRaw);
        LongType outBase = b * QK_K;

        int qlOff = 0;
        int qhOff = 0;
        int scOff = 0;

        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; l++) {
                int is = l / 16;

                int q1 = ((ql[qlOff + l] & 0xF) | (((qh[qhOff + l] >> 0) & 3) << 4)) - 32;
                int q2 = ((ql[qlOff + l + 32] & 0xF) | (((qh[qhOff + l] >> 2) & 3) << 4)) - 32;
                int q3 = (((ql[qlOff + l] >> 4) & 0xF) | (((qh[qhOff + l] >> 4) & 3) << 4)) - 32;
                int q4 = (((ql[qlOff + l + 32] >> 4) & 0xF) | (((qh[qhOff + l] >> 6) & 3) << 4)) - 32;

                LongType idx = outBase + n + l;
                if (idx < numElements) output[idx] = d * scales[scOff + is] * q1;
                idx = outBase + n + l + 32;
                if (idx < numElements) output[idx] = d * scales[scOff + is + 2] * q2;
                idx = outBase + n + l + 64;
                if (idx < numElements) output[idx] = d * scales[scOff + is + 4] * q3;
                idx = outBase + n + l + 96;
                if (idx < numElements) output[idx] = d * scales[scOff + is + 6] * q4;
            }
            qlOff += 64;
            qhOff += 32;
            scOff += 8;
        }
    }
}

//////////////////////////////////////////////////////////////////////////
// Q8_K: 8-bit K-quant
// Block: 4 bytes d (float32) + 256 bytes qs + 32 bytes bsums = 292 bytes per 256 elements
//////////////////////////////////////////////////////////////////////////
static void dequantize_q8_K(const uint8_t* data, float* output, LongType numElements) {
    constexpr int BLOCK_SIZE = 292;
    LongType numBlocks = (numElements + QK_K - 1) / QK_K;

    PRAGMA_OMP_PARALLEL_FOR
    for (LongType b = 0; b < numBlocks; b++) {
        const uint8_t* block = data + b * BLOCK_SIZE;
        float d;
        memcpy(&d, block, 4);
        const int8_t* qs = reinterpret_cast<const int8_t*>(block + 4);
        LongType outBase = b * QK_K;

        for (int j = 0; j < QK_K; j++) {
            LongType idx = outBase + j;
            if (idx < numElements) output[idx] = d * qs[j];
        }
    }
}

//////////////////////////////////////////////////////////////////////////
// IQ4_NL: Non-linear 4-bit importance quantization
// Block: 2 bytes d (FP16) + 16 bytes qs = 18 bytes per 32 elements
// Uses a 16-entry codebook (kvalues_iq4nl) instead of linear scale.
//////////////////////////////////////////////////////////////////////////
static const int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
    1, 13, 25, 38, 53, 69, 89, 113
};

static void dequantize_iq4_nl(const uint8_t* data, float* output, LongType numElements) {
    constexpr int QK4_NL = 32;
    constexpr int BLOCK_SIZE = 18;
    LongType numBlocks = (numElements + QK4_NL - 1) / QK4_NL;

    PRAGMA_OMP_PARALLEL_FOR
    for (LongType b = 0; b < numBlocks; b++) {
        const uint8_t* block = data + b * BLOCK_SIZE;
        uint16_t dRaw;
        memcpy(&dRaw, block, 2);
        float d = fp16ToFloat(dRaw);
        const uint8_t* qs = block + 2;
        LongType outBase = b * QK4_NL;

        // GGML stores low nibbles in values [0, 15] and high nibbles in
        // values [16, 31] for each IQ4_NL block; they are not interleaved.
        for (int j = 0; j < QK4_NL / 2; j++) {
            LongType idx0 = outBase + j;
            LongType idx1 = outBase + j + QK4_NL / 2;
            int lo = qs[j] & 0x0F;
            int hi = (qs[j] >> 4) & 0x0F;
            if (idx0 < numElements) output[idx0] = d * kvalues_iq4nl[lo];
            if (idx1 < numElements) output[idx1] = d * kvalues_iq4nl[hi];
        }
    }
}

//////////////////////////////////////////////////////////////////////////
// IQ4_XS: 4-bit importance quantization with extra scales
// Block: 2 bytes d + 2 bytes scales_h + 4 bytes scales_l + 128 bytes qs
//        = 136 bytes per 256 elements
//////////////////////////////////////////////////////////////////////////
static void dequantize_iq4_xs(const uint8_t* data, float* output, LongType numElements) {
    constexpr int BLOCK_SIZE = 136;
    LongType numBlocks = (numElements + QK_K - 1) / QK_K;

    PRAGMA_OMP_PARALLEL_FOR
    for (LongType b = 0; b < numBlocks; b++) {
        const uint8_t* block = data + b * BLOCK_SIZE;
        uint16_t dRaw;
        memcpy(&dRaw, block, 2);
        float d = fp16ToFloat(dRaw);

        uint16_t scales_h;
        memcpy(&scales_h, block + 2, 2);
        const uint8_t* scales_l = block + 4;
        const uint8_t* qs = block + 8;
        LongType outBase = b * QK_K;

        // Decode 8 sub-block scales (6 bits each)
        int scales[8];
        for (int i = 0; i < 4; i++) {
            scales[2 * i] = (scales_l[i] & 0x0F) | (((scales_h >> (2 * i)) & 3) << 4);
            scales[2 * i + 1] = ((scales_l[i] >> 4) & 0x0F) | (((scales_h >> (2 * i + 1)) & 3) << 4);
        }

        for (int ib = 0; ib < 8; ib++) {
            float dl = d * (scales[ib] - 32);
            const uint8_t* qBlock = qs + ib * 16;
            for (int j = 0; j < 16; j++) {
                int lo = qBlock[j] & 0x0F;
                int hi = (qBlock[j] >> 4) & 0x0F;
                LongType idx1 = outBase + ib * 32 + j;
                LongType idx2 = outBase + ib * 32 + j + 16;
                if (idx1 < numElements) output[idx1] = dl * kvalues_iq4nl[lo];
                if (idx2 < numElements) output[idx2] = dl * kvalues_iq4nl[hi];
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////
// IQ3_XXS: 3-bit importance quantization (extra extra small)
// Reference fallback: linear 3-bit dequantization.
// Block: 98 bytes per 256 elements
//////////////////////////////////////////////////////////////////////////
static void dequantize_iq3_xxs(const uint8_t* data, float* output, LongType numElements) {
    constexpr int BLOCK_SIZE = 98;
    LongType numBlocks = (numElements + QK_K - 1) / QK_K;

    PRAGMA_OMP_PARALLEL_FOR
    for (LongType b = 0; b < numBlocks; b++) {
        const uint8_t* block = data + b * BLOCK_SIZE;
        uint16_t dRaw;
        memcpy(&dRaw, block, 2);
        float d = fp16ToFloat(dRaw);
        const uint8_t* qBytes = block + 2;
        LongType outBase = b * QK_K;

        int qOff = 0;
        int bitAccum = 0;
        int bitsInAccum = 0;

        for (int j = 0; j < QK_K; j++) {
            while (bitsInAccum < 3 && qOff < (BLOCK_SIZE - 2)) {
                bitAccum |= (static_cast<int>(qBytes[qOff++]) << bitsInAccum);
                bitsInAccum += 8;
            }
            int val = bitAccum & 0x7;
            bitAccum >>= 3;
            bitsInAccum -= 3;
            LongType idx = outBase + j;
            if (idx < numElements) output[idx] = d * (val - 4);
        }
    }
}

//////////////////////////////////////////////////////////////////////////
// IQ3_S: 3-bit importance quantization (standard)
// Reference fallback with sign bits.
// Block: 110 bytes per 256 elements
//////////////////////////////////////////////////////////////////////////
static void dequantize_iq3_s(const uint8_t* data, float* output, LongType numElements) {
    constexpr int BLOCK_SIZE = 110;
    LongType numBlocks = (numElements + QK_K - 1) / QK_K;

    PRAGMA_OMP_PARALLEL_FOR
    for (LongType b = 0; b < numBlocks; b++) {
        const uint8_t* block = data + b * BLOCK_SIZE;
        uint16_t dRaw;
        memcpy(&dRaw, block, 2);
        float d = fp16ToFloat(dRaw);

        const uint8_t* qs = block + 2;       // 96 bytes
        const uint8_t* signs = block + 98;   // 32 bytes (after qs+qh)
        const uint8_t* scales = block + 102; // 8 bytes
        LongType outBase = b * QK_K;

        int qOff = 0;
        int bitAccum = 0;
        int bitsInAccum = 0;
        int localOff = 0;

        for (int ib = 0; ib < 8; ib++) {
            float dl = d * (1 + 2 * (scales[ib] & 0x0F));
            for (int j = 0; j < 32; j++) {
                while (bitsInAccum < 3 && qOff < 96) {
                    bitAccum |= (static_cast<int>(qs[qOff++]) << bitsInAccum);
                    bitsInAccum += 8;
                }
                int val = bitAccum & 0x7;
                bitAccum >>= 3;
                bitsInAccum -= 3;

                int signIdx = ib * 4 + j / 8;
                int signBit = (signIdx < 32) ? ((signs[signIdx] >> (j % 8)) & 1) : 0;
                float fval = dl * (val - 4);
                LongType idx = outBase + localOff++;
                if (idx < numElements) output[idx] = signBit ? -fval : fval;
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////
// IQ2_XXS: 2-bit importance quantization (extra extra small)
// Reference fallback: linear 2-bit dequantization.
// Block: 66 bytes per 256 elements
//////////////////////////////////////////////////////////////////////////
static void dequantize_iq2_xxs(const uint8_t* data, float* output, LongType numElements) {
    constexpr int BLOCK_SIZE = 66;
    LongType numBlocks = (numElements + QK_K - 1) / QK_K;

    PRAGMA_OMP_PARALLEL_FOR
    for (LongType b = 0; b < numBlocks; b++) {
        const uint8_t* block = data + b * BLOCK_SIZE;
        uint16_t dRaw;
        memcpy(&dRaw, block, 2);
        float d = fp16ToFloat(dRaw);
        const uint8_t* qBytes = block + 2;
        LongType outBase = b * QK_K;

        for (int j = 0; j < QK_K; j++) {
            int byteIdx = j / 4;
            int bitOff = (j % 4) * 2;
            LongType idx = outBase + j;
            if (byteIdx < 64) {
                int val = (qBytes[byteIdx] >> bitOff) & 0x3;
                if (idx < numElements) output[idx] = d * (val - 1.5f);
            } else {
                if (idx < numElements) output[idx] = 0.0f;
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////
// IQ2_XS: 2-bit importance quantization (extra small)
// Reference fallback with sub-block scales.
// Block: 74 bytes per 256 elements
//////////////////////////////////////////////////////////////////////////
static void dequantize_iq2_xs(const uint8_t* data, float* output, LongType numElements) {
    constexpr int BLOCK_SIZE = 74;
    LongType numBlocks = (numElements + QK_K - 1) / QK_K;

    PRAGMA_OMP_PARALLEL_FOR
    for (LongType b = 0; b < numBlocks; b++) {
        const uint8_t* block = data + b * BLOCK_SIZE;
        uint16_t dRaw;
        memcpy(&dRaw, block, 2);
        float d = fp16ToFloat(dRaw);
        const uint8_t* qs = block + 2;      // 64 bytes
        const uint8_t* scales = block + 66; // 8 bytes
        LongType outBase = b * QK_K;

        int localOff = 0;
        for (int ib = 0; ib < 8; ib++) {
            float dl = d * (1 + 2 * (scales[ib] & 0x0F));
            for (int j = 0; j < 32; j++) {
                int byteIdx = ib * 8 + j / 4;
                int bitOff = (j % 4) * 2;
                LongType idx = outBase + localOff++;
                if (byteIdx < 64) {
                    int val = (qs[byteIdx] >> bitOff) & 0x3;
                    if (idx < numElements) output[idx] = dl * (val - 1.5f);
                } else {
                    if (idx < numElements) output[idx] = 0.0f;
                }
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////
// IQ2_S: 2-bit importance quantization (standard)
// Reference fallback with sign bits from qh.
// Block: 82 bytes per 256 elements
//////////////////////////////////////////////////////////////////////////
static void dequantize_iq2_s(const uint8_t* data, float* output, LongType numElements) {
    constexpr int BLOCK_SIZE = 82;
    LongType numBlocks = (numElements + QK_K - 1) / QK_K;

    PRAGMA_OMP_PARALLEL_FOR
    for (LongType b = 0; b < numBlocks; b++) {
        const uint8_t* block = data + b * BLOCK_SIZE;
        uint16_t dRaw;
        memcpy(&dRaw, block, 2);
        float d = fp16ToFloat(dRaw);
        const uint8_t* qs = block + 2;   // 64 bytes
        const uint8_t* qh = block + 66;  // 16 bytes
        LongType outBase = b * QK_K;

        for (int j = 0; j < QK_K; j++) {
            int byteIdx = j / 4;
            int bitOff = (j % 4) * 2;
            LongType idx = outBase + j;
            if (byteIdx < 64) {
                int val = (qs[byteIdx] >> bitOff) & 0x3;
                int qhIdx = j / 16;
                int qhBit = j % 8;
                int signBit = (qhIdx < 16) ? ((qh[qhIdx] >> qhBit) & 1) : 0;
                float fval = d * (val - 1.5f);
                if (idx < numElements) output[idx] = signBit ? -fval : fval;
            } else {
                if (idx < numElements) output[idx] = 0.0f;
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////
// IQ1_S: ~1.5-bit ternary importance quantization
// Reference fallback: approximate ternary {-1, 0, +1} * d.
// Block: 50 bytes per 256 elements
//////////////////////////////////////////////////////////////////////////
static void dequantize_iq1_s(const uint8_t* data, float* output, LongType numElements) {
    constexpr int BLOCK_SIZE = 50;
    LongType numBlocks = (numElements + QK_K - 1) / QK_K;

    PRAGMA_OMP_PARALLEL_FOR
    for (LongType b = 0; b < numBlocks; b++) {
        const uint8_t* block = data + b * BLOCK_SIZE;
        uint16_t dRaw;
        memcpy(&dRaw, block, 2);
        float d = fp16ToFloat(dRaw);
        const uint8_t* qs = block + 2;     // 32 bytes
        const uint8_t* qh = block + 34;    // 16 bytes (sign bits)
        LongType outBase = b * QK_K;

        for (int j = 0; j < QK_K; j++) {
            int byteIdx = j / 8;
            int bitOff = j % 8;
            LongType idx = outBase + j;
            if (byteIdx < 32) {
                int bit = (qs[byteIdx] >> bitOff) & 1;
                int signBit = (byteIdx < 16) ? ((qh[byteIdx] >> bitOff) & 1) : 0;
                float val = bit ? d : 0.0f;
                if (idx < numElements) output[idx] = signBit ? -val : val;
            } else {
                if (idx < numElements) output[idx] = 0.0f;
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////
// IQ1_M: ~1.75-bit ternary importance quantization
// Reference fallback with sub-block scales.
// Block: 56 bytes per 256 elements
//////////////////////////////////////////////////////////////////////////
static void dequantize_iq1_m(const uint8_t* data, float* output, LongType numElements) {
    constexpr int BLOCK_SIZE = 56;
    LongType numBlocks = (numElements + QK_K - 1) / QK_K;

    PRAGMA_OMP_PARALLEL_FOR
    for (LongType b = 0; b < numBlocks; b++) {
        const uint8_t* block = data + b * BLOCK_SIZE;
        const uint8_t* qs = block;          // 32 bytes
        const uint8_t* qh = block + 32;     // 16 bytes
        const uint8_t* scalesRaw = block + 48; // 8 bytes
        LongType outBase = b * QK_K;

        // d is encoded in the last 2 bytes of scales
        uint16_t dRaw;
        memcpy(&dRaw, scalesRaw + 6, 2);
        float d = fp16ToFloat(dRaw);

        int localOff = 0;
        for (int ib = 0; ib < 8; ib++) {
            float dl = d * (1 + 2 * (scalesRaw[ib < 6 ? ib : 0] & 0x0F));
            for (int j = 0; j < 32; j++) {
                int idx2 = ib * 4 + j / 8;
                int bitOff = j % 8;
                LongType idx = outBase + localOff++;
                if (idx2 < 32) {
                    int bit = (qs[idx2] >> bitOff) & 1;
                    int signIdx = ib * 2 + j / 8;
                    int sign = (signIdx < 16) ? ((qh[signIdx] >> (j % 8)) & 1) : 0;
                    float val = bit ? dl : 0.0f;
                    if (idx < numElements) output[idx] = sign ? -val : val;
                } else {
                    if (idx < numElements) output[idx] = 0.0f;
                }
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////
// TQ1_0: Ternary 1-bit quantization
// Reference fallback: simple ternary {-1, 0, +1} * d
// Block: ~54 bytes per 256 elements
//////////////////////////////////////////////////////////////////////////
static void dequantize_tq1_0(const uint8_t* data, float* output, LongType numElements) {
    constexpr int BLOCK_SIZE = 54;
    LongType numBlocks = (numElements + QK_K - 1) / QK_K;

    PRAGMA_OMP_PARALLEL_FOR
    for (LongType b = 0; b < numBlocks; b++) {
        const uint8_t* block = data + b * BLOCK_SIZE;
        uint16_t dRaw;
        memcpy(&dRaw, block, 2);
        float d = fp16ToFloat(dRaw);
        const uint8_t* qs = block + 2;
        LongType outBase = b * QK_K;

        // Each byte encodes ~5 ternary values via base-3
        for (int j = 0; j < QK_K; j++) {
            int byteIdx = j / 5;
            int posInByte = j % 5;
            LongType idx = outBase + j;
            if (byteIdx < 52) {
                int packed = qs[byteIdx] & 0xFF;
                // Extract base-3 digit
                for (int p = 0; p < posInByte; p++) packed /= 3;
                int trit = packed % 3; // 0, 1, 2 -> -1, 0, +1
                if (idx < numElements) output[idx] = d * (trit - 1);
            } else {
                if (idx < numElements) output[idx] = 0.0f;
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////
// TQ2_0: Ternary 2-bit quantization
// Reference fallback: 2-bit ternary
// Block: ~66 bytes per 256 elements
//////////////////////////////////////////////////////////////////////////
static void dequantize_tq2_0(const uint8_t* data, float* output, LongType numElements) {
    constexpr int BLOCK_SIZE = 66;
    LongType numBlocks = (numElements + QK_K - 1) / QK_K;

    PRAGMA_OMP_PARALLEL_FOR
    for (LongType b = 0; b < numBlocks; b++) {
        const uint8_t* block = data + b * BLOCK_SIZE;
        uint16_t dRaw;
        memcpy(&dRaw, block, 2);
        float d = fp16ToFloat(dRaw);
        const uint8_t* qs = block + 2;
        LongType outBase = b * QK_K;

        for (int j = 0; j < QK_K; j++) {
            int byteIdx = j / 4;
            int bitOff = (j % 4) * 2;
            LongType idx = outBase + j;
            if (byteIdx < 64) {
                int val = (qs[byteIdx] >> bitOff) & 0x3;
                // 2-bit ternary: 0->-1, 1->0, 2->+1, 3->0
                float tval;
                switch (val) {
                    case 0: tval = -1.0f; break;
                    case 2: tval = 1.0f; break;
                    default: tval = 0.0f; break;
                }
                if (idx < numElements) output[idx] = d * tval;
            } else {
                if (idx < numElements) output[idx] = 0.0f;
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////
// Dispatch to the right dequantizer, then convert F32 to target type
//////////////////////////////////////////////////////////////////////////
static void dequantizeToFloat32(const uint8_t* rawBytes, float* output, int quantType, LongType numElements) {
    switch (quantType) {
        case GGML_QUANT_Q4_0: dequantize_q4_0(rawBytes, output, numElements); break;
        case GGML_QUANT_Q4_1: dequantize_q4_1(rawBytes, output, numElements); break;
        case GGML_QUANT_Q5_0: dequantize_q5_0(rawBytes, output, numElements); break;
        case GGML_QUANT_Q5_1: dequantize_q5_1(rawBytes, output, numElements); break;
        case GGML_QUANT_Q8_0: dequantize_q8_0(rawBytes, output, numElements); break;
        case GGML_QUANT_Q8_1: dequantize_q8_1(rawBytes, output, numElements); break;
        case GGML_QUANT_Q2_K: dequantize_q2_K(rawBytes, output, numElements); break;
        case GGML_QUANT_Q3_K: dequantize_q3_K(rawBytes, output, numElements); break;
        case GGML_QUANT_Q4_K: dequantize_q4_K(rawBytes, output, numElements); break;
        case GGML_QUANT_Q5_K: dequantize_q5_K(rawBytes, output, numElements); break;
        case GGML_QUANT_Q6_K: dequantize_q6_K(rawBytes, output, numElements); break;
        case GGML_QUANT_Q8_K: dequantize_q8_K(rawBytes, output, numElements); break;
        case GGML_QUANT_IQ4_NL: dequantize_iq4_nl(rawBytes, output, numElements); break;
        case GGML_QUANT_IQ4_XS: dequantize_iq4_xs(rawBytes, output, numElements); break;
        case GGML_QUANT_IQ3_XXS: dequantize_iq3_xxs(rawBytes, output, numElements); break;
        case GGML_QUANT_IQ3_S: dequantize_iq3_s(rawBytes, output, numElements); break;
        case GGML_QUANT_IQ2_XXS: dequantize_iq2_xxs(rawBytes, output, numElements); break;
        case GGML_QUANT_IQ2_XS: dequantize_iq2_xs(rawBytes, output, numElements); break;
        case GGML_QUANT_IQ2_S: dequantize_iq2_s(rawBytes, output, numElements); break;
        case GGML_QUANT_IQ1_S: dequantize_iq1_s(rawBytes, output, numElements); break;
        case GGML_QUANT_IQ1_M: dequantize_iq1_m(rawBytes, output, numElements); break;
        case GGML_QUANT_TQ1_0: dequantize_tq1_0(rawBytes, output, numElements); break;
        case GGML_QUANT_TQ2_0: dequantize_tq2_0(rawBytes, output, numElements); break;
        default:
            THROW_EXCEPTION("ggmlDequantize: unsupported quant type");
    }
}

//////////////////////////////////////////////////////////////////////////
// Public interface
//////////////////////////////////////////////////////////////////////////
void ggmlDequantize(
    LaunchContext* context,
    NDArray* input,
    NDArray* output,
    int quantType) {

    NDArray::preparePrimaryUse({output}, {input});

    const auto* rawBytes = reinterpret_cast<const uint8_t*>(input->buffer());
    LongType numElements = output->lengthOf();
    auto outputDtype = output->dataType();

    if (outputDtype == DataType::FLOAT32) {
        // Direct dequantize to F32
        dequantizeToFloat32(rawBytes, output->bufferAsT<float>(), quantType, numElements);
    } else {
        // HALF/BFLOAT16 use FLOAT32 accumulation, but repeatedly allocating this scratch
        // buffer makes the Android allocator retain one model-sized arena per streamed
        // chunk. Reuse one scratch allocation per calling thread instead.
        thread_local std::vector<float> tmpBuf;
        tmpBuf.resize(static_cast<size_t>(numElements));
        float* tmpData = tmpBuf.data();
        dequantizeToFloat32(rawBytes, tmpData, quantType, numElements);

        if (outputDtype == DataType::HALF) {
            auto* dst = reinterpret_cast<float16*>(output->buffer());
            PRAGMA_OMP_PARALLEL_FOR
            for (LongType i = 0; i < numElements; i++) {
                dst[i] = static_cast<float16>(tmpData[i]);
            }
        } else if (outputDtype == DataType::BFLOAT16) {
            auto* dst = reinterpret_cast<bfloat16*>(output->buffer());
            PRAGMA_OMP_PARALLEL_FOR
            for (LongType i = 0; i < numElements; i++) {
                dst[i] = static_cast<bfloat16>(tmpData[i]);
            }
        } else {
            THROW_EXCEPTION("ggmlDequantize: unsupported output type");
        }
    }

    NDArray::registerPrimaryUse({output}, {input});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_ggml_dequantize)
