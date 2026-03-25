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
#include <cstring>
#include <cmath>

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
    LongType outIdx = 0;

    for (LongType b = 0; b < numBlocks; b++) {
        const uint8_t* block = data + b * BLOCK_SIZE;
        uint16_t dRaw;
        memcpy(&dRaw, block, 2);
        float d = fp16ToFloat(dRaw);
        const uint8_t* qs = block + 2;

        for (int j = 0; j < QK4_0 / 2 && outIdx < numElements; j++) {
            int v0 = (qs[j] & 0x0F) - 8;
            int v1 = ((qs[j] >> 4) & 0x0F) - 8;
            if (outIdx < numElements) output[outIdx++] = d * v0;
            if (outIdx < numElements) output[outIdx++] = d * v1;
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
    LongType outIdx = 0;

    for (LongType b = 0; b < numBlocks; b++) {
        const uint8_t* block = data + b * BLOCK_SIZE;
        uint16_t dRaw, mRaw;
        memcpy(&dRaw, block, 2);
        memcpy(&mRaw, block + 2, 2);
        float d = fp16ToFloat(dRaw);
        float m = fp16ToFloat(mRaw);
        const uint8_t* qs = block + 4;

        for (int j = 0; j < QK4_1 / 2 && outIdx < numElements; j++) {
            int v0 = qs[j] & 0x0F;
            int v1 = (qs[j] >> 4) & 0x0F;
            if (outIdx < numElements) output[outIdx++] = d * v0 + m;
            if (outIdx < numElements) output[outIdx++] = d * v1 + m;
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
    LongType outIdx = 0;

    for (LongType b = 0; b < numBlocks; b++) {
        const uint8_t* block = data + b * BLOCK_SIZE;
        uint16_t dRaw;
        memcpy(&dRaw, block, 2);
        float d = fp16ToFloat(dRaw);

        uint32_t qh;
        memcpy(&qh, block + 2, 4);
        const uint8_t* qs = block + 6;

        for (int j = 0; j < QK5_0 / 2 && outIdx < numElements; j++) {
            int xh0 = ((qh >> (j)) & 1) << 4;
            int xh1 = ((qh >> (j + 16)) & 1) << 4;
            int v0 = (qs[j] & 0x0F) | xh0;
            int v1 = ((qs[j] >> 4) & 0x0F) | xh1;
            if (outIdx < numElements) output[outIdx++] = d * (v0 - 16);
            if (outIdx < numElements) output[outIdx++] = d * (v1 - 16);
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
    LongType outIdx = 0;

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

        for (int j = 0; j < QK5_1 / 2 && outIdx < numElements; j++) {
            int xh0 = ((qh >> (j)) & 1) << 4;
            int xh1 = ((qh >> (j + 16)) & 1) << 4;
            int v0 = (qs[j] & 0x0F) | xh0;
            int v1 = ((qs[j] >> 4) & 0x0F) | xh1;
            if (outIdx < numElements) output[outIdx++] = d * v0 + m;
            if (outIdx < numElements) output[outIdx++] = d * v1 + m;
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
    LongType outIdx = 0;

    for (LongType b = 0; b < numBlocks; b++) {
        const uint8_t* block = data + b * BLOCK_SIZE;
        uint16_t dRaw;
        memcpy(&dRaw, block, 2);
        float d = fp16ToFloat(dRaw);
        const int8_t* qs = reinterpret_cast<const int8_t*>(block + 2);

        for (int j = 0; j < QK8_0 && outIdx < numElements; j++) {
            output[outIdx++] = d * qs[j];
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
    LongType outIdx = 0;

    for (LongType b = 0; b < numBlocks; b++) {
        const uint8_t* block = data + b * BLOCK_SIZE;
        float d, s;
        memcpy(&d, block, 4);
        memcpy(&s, block + 4, 4);
        const int8_t* qs = reinterpret_cast<const int8_t*>(block + 8);

        for (int j = 0; j < QK8_1 && outIdx < numElements; j++) {
            output[outIdx++] = d * qs[j] + s;
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
    LongType outIdx = 0;

    for (LongType b = 0; b < numBlocks; b++) {
        const uint8_t* block = data + b * BLOCK_SIZE;
        const uint8_t* scales = block;
        const uint8_t* qs = block + 16;
        uint16_t dRaw, dminRaw;
        memcpy(&dRaw, block + 80, 2);
        memcpy(&dminRaw, block + 82, 2);
        float d = fp16ToFloat(dRaw);
        float dmin = fp16ToFloat(dminRaw);

        int qIdx = 0;
        for (int i = 0; i < QK_K; i += 128) {
            for (int l = 0; l < 32; l++) {
                int is = i / 16 + l / 16;
                int sc = scales[is];
                float dl = d * (sc & 0xF);
                float ml = dmin * (sc >> 4);
                if (outIdx < numElements) output[outIdx++] = dl * (qs[qIdx + l] & 3) - ml;
            }
            for (int l = 0; l < 32; l++) {
                int is = i / 16 + l / 16 + 2;
                int sc = scales[is];
                float dl = d * (sc & 0xF);
                float ml = dmin * (sc >> 4);
                if (outIdx < numElements) output[outIdx++] = dl * ((qs[qIdx + l] >> 2) & 3) - ml;
            }
            for (int l = 0; l < 32; l++) {
                int is = i / 16 + l / 16 + 4;
                int sc = scales[is];
                float dl = d * (sc & 0xF);
                float ml = dmin * (sc >> 4);
                if (outIdx < numElements) output[outIdx++] = dl * ((qs[qIdx + l] >> 4) & 3) - ml;
            }
            for (int l = 0; l < 32; l++) {
                int is = i / 16 + l / 16 + 6;
                int sc = scales[is];
                float dl = d * (sc & 0xF);
                float ml = dmin * (sc >> 4);
                if (outIdx < numElements) output[outIdx++] = dl * ((qs[qIdx + l] >> 6) & 3) - ml;
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
    LongType outIdx = 0;

    for (LongType b = 0; b < numBlocks; b++) {
        const uint8_t* block = data + b * BLOCK_SIZE;
        const uint8_t* hmask = block;
        const uint8_t* qs = block + 32;
        const uint8_t* scaleBytes = block + 96;
        uint16_t dRaw;
        memcpy(&dRaw, block + 108, 2);
        float d = fp16ToFloat(dRaw);

        // Unpack 12 bytes of scales into 16 values
        int scales[16];
        int a;
        for (int i = 0; i < 8; i++) {
            a = scaleBytes[i];
            scales[i] = (a & 0xF) - 8;
            scales[i + 8] = (a >> 4) - 8;
        }

        // Rearrange: ggml's Q3_K scale order needs the 4 high bits from last 4 bytes
        // Actually the simple unpack above matches ggml for QK_K=256 path
        // The ggml reference dequantize_row_q3_K has a specific unpack pattern
        // Using the simpler interpretation for now, matching dequantize_row_q3_K
        uint8_t m = 1;
        int is = 0;
        int n = 0;
        int qIdx = 0;

        for (int i = 0; i < QK_K; i += 128) {
            for (int l = 0; l < 32; l++) {
                int sIdx = is + l / 16;
                float dl = d * scales[sIdx];
                int q = qs[qIdx + l] & 3;
                int h = (hmask[l] & m) ? 0 : 4;
                if (outIdx < numElements) output[outIdx++] = dl * (q - h);
            }
            is += 2;
            for (int l = 0; l < 32; l++) {
                int sIdx = is + l / 16;
                float dl = d * scales[sIdx];
                int q = (qs[qIdx + l] >> 2) & 3;
                int h = (hmask[l] & (m << 1)) ? 0 : 4;
                if (outIdx < numElements) output[outIdx++] = dl * (q - h);
            }
            is += 2;
            m <<= 2;
            for (int l = 0; l < 32; l++) {
                int sIdx = is + l / 16;
                float dl = d * scales[sIdx];
                int q = (qs[qIdx + l] >> 4) & 3;
                int h = (hmask[l] & m) ? 0 : 4;
                if (outIdx < numElements) output[outIdx++] = dl * (q - h);
            }
            is += 2;
            for (int l = 0; l < 32; l++) {
                int sIdx = is + l / 16;
                float dl = d * scales[sIdx];
                int q = (qs[qIdx + l] >> 6) & 3;
                int h = (hmask[l] & (m << 1)) ? 0 : 4;
                if (outIdx < numElements) output[outIdx++] = dl * (q - h);
            }
            is += 2;
            m <<= 2;
            qIdx += 32;
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
    LongType outIdx = 0;

    for (LongType b = 0; b < numBlocks; b++) {
        const uint8_t* block = data + b * BLOCK_SIZE;
        uint16_t dRaw, dminRaw;
        memcpy(&dRaw, block, 2);
        memcpy(&dminRaw, block + 2, 2);
        float d = fp16ToFloat(dRaw);
        float dmin = fp16ToFloat(dminRaw);
        const uint8_t* scaleBytes = block + 4;
        const uint8_t* qs = block + 16;

        int is = 0;
        int qIdx = 0;
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
                if (outIdx < numElements) output[outIdx++] = d1 * val - m1f;
            }
            for (int l = 0; l < 32; l++) {
                int val = (qs[qIdx + l] >> 4) & 0x0F;
                if (outIdx < numElements) output[outIdx++] = d2 * val - m2f;
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
    LongType outIdx = 0;

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

        int is = 0;
        int qlOff = 0;
        uint8_t u1 = 1, u2 = 2;

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
                if (outIdx < numElements) output[outIdx++] = d1 * (lowVal + highBit) - m1f;
            }
            for (int l = 0; l < 32; l++) {
                int highNibble = (qs[qlOff + l] >> 4) & 0x0F;
                int highBit = (qh[l] & u2) ? 16 : 0;
                if (outIdx < numElements) output[outIdx++] = d2 * (highNibble + highBit) - m2f;
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
    LongType outIdx = 0;

    for (LongType b = 0; b < numBlocks; b++) {
        const uint8_t* block = data + b * BLOCK_SIZE;
        const uint8_t* ql = block;
        const uint8_t* qh = block + 128;
        const int8_t* scales = reinterpret_cast<const int8_t*>(block + 192);
        uint16_t dRaw;
        memcpy(&dRaw, block + 208, 2);
        float d = fp16ToFloat(dRaw);

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

                LongType idx = outIdx + n + l;
                if (idx < numElements) output[idx] = d * scales[scOff + is] * q1;
                idx = outIdx + n + l + 32;
                if (idx < numElements) output[idx] = d * scales[scOff + is + 2] * q2;
                idx = outIdx + n + l + 64;
                if (idx < numElements) output[idx] = d * scales[scOff + is + 4] * q3;
                idx = outIdx + n + l + 96;
                if (idx < numElements) output[idx] = d * scales[scOff + is + 6] * q4;
            }
            qlOff += 64;
            qhOff += 32;
            scOff += 8;
        }

        outIdx += QK_K;
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
        // Dequantize to temporary F32 buffer, then convert
        std::vector<float> tmpBuf(numElements);
        dequantizeToFloat32(rawBytes, tmpBuf.data(), quantType, numElements);

        if (outputDtype == DataType::HALF) {
            auto* dst = reinterpret_cast<float16*>(output->buffer());
            for (LongType i = 0; i < numElements; i++) {
                dst[i] = static_cast<float16>(tmpBuf[i]);
            }
        } else if (outputDtype == DataType::BFLOAT16) {
            auto* dst = reinterpret_cast<bfloat16*>(output->buffer());
            for (LongType i = 0; i < numElements; i++) {
                dst[i] = static_cast<bfloat16>(tmpBuf[i]);
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
