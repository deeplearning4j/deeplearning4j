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

package org.nd4j.ggml.quantization;

import org.nd4j.ggml.format.GGMLDataType;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Quantizer for Q6_K format (k-quant).
 * This is the reverse operation of {@link Q6_KDequantizer}.
 *
 * Q6_K block layout (from ggml-common.h block_q6_K):
 * <pre>
 *   uint8_t ql[128]    — lower 4 bits of each 6-bit quant (128 bytes)
 *   uint8_t qh[64]     — upper 2 bits of each 6-bit quant (64 bytes)
 *   int8_t  scales[16] — per-sub-block signed scales (16 bytes)
 *   ggml_half d        — super-block FP16 scale (2 bytes)
 * </pre>
 * Total: 210 bytes per super-block of 256 elements.
 *
 * Each 6-bit quantized value is stored biased: q_stored = q_signed + 32 (so 0..63).
 * Dequantization: val = d * scales[subBlock] * (q_stored - 32)
 *
 * The ql/qh layout matches ggml's dequantize_row_q6_K exactly.
 */
public class Q6_KQuantizer implements Quantizer {

    private static final int BLOCK_SIZE = 256;
    private static final int BYTES_PER_BLOCK = 210; // 128 + 64 + 16 + 2
    private static final int SUB_BLOCK_SIZE = 16;
    private static final int NUM_SUB_BLOCKS = 16;

    @Override
    public GGMLDataType getQuantType() {
        return GGMLDataType.GGML_TYPE_Q6_K;
    }

    @Override
    public int getBlockSize() {
        return BLOCK_SIZE;
    }

    @Override
    public int getBytesPerBlock() {
        return BYTES_PER_BLOCK;
    }

    @Override
    public byte[] quantize(float[] floatData) {
        int numBlocks = (floatData.length + BLOCK_SIZE - 1) / BLOCK_SIZE;
        byte[] output = new byte[numBlocks * BYTES_PER_BLOCK];
        ByteBuffer buffer = ByteBuffer.wrap(output).order(ByteOrder.LITTLE_ENDIAN);

        for (int block = 0; block < numBlocks; block++) {
            int blockStart = block * BLOCK_SIZE;

            // Step 1: compute per-sub-block max absolute value and scale
            float[] subBlockScales = new float[NUM_SUB_BLOCKS];

            for (int sb = 0; sb < NUM_SUB_BLOCKS; sb++) {
                int start = blockStart + sb * SUB_BLOCK_SIZE;
                int end = Math.min(start + SUB_BLOCK_SIZE, floatData.length);

                float maxAbs = 0;
                for (int i = start; i < end; i++) {
                    float a = Math.abs(floatData[i]);
                    if (a > maxAbs) maxAbs = a;
                }
                // 6-bit signed: -32..31; use 31 as the positive limit for symmetric quantization
                subBlockScales[sb] = maxAbs / 31.0f;
            }

            // Step 2: compute super-block d
            float maxScale = 0;
            for (int sb = 0; sb < NUM_SUB_BLOCKS; sb++) {
                if (subBlockScales[sb] > maxScale) maxScale = subBlockScales[sb];
            }

            float d = maxScale / 127.0f; // sub-block scales stored as int8 (0..127 unsigned)
            if (d == 0) d = 1.0f;

            // Step 3: quantize sub-block scales to signed int8
            // dequantizer: val = d * scales[sb] * q_signed
            // where q_signed = q_stored - 32 and scales[sb] is signed int8
            // We want: subBlockScales[sb] ≈ d * quantizedScales[sb]
            // => quantizedScales[sb] = round(subBlockScales[sb] / d), clamped to -128..127
            byte[] scales = new byte[NUM_SUB_BLOCKS];
            for (int sb = 0; sb < NUM_SUB_BLOCKS; sb++) {
                int qs = (int) Math.round(subBlockScales[sb] / d);
                scales[sb] = (byte) Math.max(-128, Math.min(127, qs));
            }

            // Step 4: quantize data values to 6-bit signed, stored as q_stored = q_signed + 32 (0..63)
            // val ≈ d * scales[sb] * q_signed  =>  q_signed = round(val / (d * scales[sb]))
            // Then q_stored = q_signed + 32
            int[] qStored = new int[BLOCK_SIZE];
            for (int sb = 0; sb < NUM_SUB_BLOCKS; sb++) {
                // scales[sb] is signed int8; cast to int gives signed value
                float effScale = d * (float)((int)(scales[sb]));
                float invScale = (effScale != 0) ? 1.0f / effScale : 0.0f;

                for (int i = 0; i < SUB_BLOCK_SIZE; i++) {
                    int idx = blockStart + sb * SUB_BLOCK_SIZE + i;
                    float val = (idx < floatData.length) ? floatData[idx] : 0;
                    int qSigned = (invScale != 0) ? (int) Math.round(val * invScale) : 0;
                    qSigned = Math.max(-32, Math.min(31, qSigned));
                    qStored[sb * SUB_BLOCK_SIZE + i] = qSigned + 32; // 0..63
                }
            }

            // Step 5: pack qStored into ql[128] and qh[64] matching dequantizer layout.
            //
            // The dequantizer processes elements in 2 groups of 128 elements:
            // Group 0 (n=0, qlOff=0, qhOff=0, scOff=0):
            //   For l=0..31:
            //     element[l]    = (ql[l]    & 0xF) | ((qh[l] & 0x03) << 4)  - 32  (scale: scales[0+l/16])
            //     element[l+32] = (ql[l+32] & 0xF) | ((qh[l] & 0x0C) << 2)  - 32  (scale: scales[2+l/16])
            //     element[l+64] = (ql[l]    >> 4)  | ((qh[l] & 0x30)     )  - 32  (scale: scales[4+l/16])
            //     element[l+96] = (ql[l+32] >> 4)  | ((qh[l] & 0xC0) >> 2)  - 32  (scale: scales[6+l/16])
            //
            // Group 1 (n=128, qlOff=64, qhOff=32, scOff=8):
            //   For l=0..31:
            //     element[128+l]    = (ql[64+l]    & 0xF) | ((qh[32+l] & 0x03) << 4) - 32
            //     element[128+l+32] = (ql[64+l+32] & 0xF) | ((qh[32+l] & 0x0C) << 2) - 32
            //     element[128+l+64] = (ql[64+l]    >> 4)  | ((qh[32+l] & 0x30)     ) - 32
            //     element[128+l+96] = (ql[64+l+32] >> 4)  | ((qh[32+l] & 0xC0) >> 2) - 32
            //
            // scOff sub-block index mapping:
            //   is = l / 16; element[l] uses scales[scOff + is]
            //   is = l / 16; element[l+32] uses scales[scOff + is + 2]
            //   is = l / 16; element[l+64] uses scales[scOff + is + 4]
            //   is = l / 16; element[l+96] uses scales[scOff + is + 6]
            //
            // Reverse-engineer: given qStored[idx], we need to set ql and qh bits.
            byte[] ql = new byte[128];
            byte[] qh = new byte[64];

            // Group 0: elements 0..127, qlOff=0, qhOff=0
            for (int l = 0; l < 32; l++) {
                int v0  = qStored[l];       // element[l]    -> ql[l]    lower nibble, qh[l] bits[1:0]
                int v1  = qStored[l + 32];  // element[l+32] -> ql[l+32] lower nibble, qh[l] bits[3:2]
                int v2  = qStored[l + 64];  // element[l+64] -> ql[l]    upper nibble, qh[l] bits[5:4]
                int v3  = qStored[l + 96];  // element[l+96] -> ql[l+32] upper nibble, qh[l] bits[7:6]

                ql[l]      |= (byte) (v0 & 0x0F);          // lower nibble of ql[l]
                ql[l]      |= (byte) ((v2 & 0x0F) << 4);   // upper nibble of ql[l]
                ql[l + 32] |= (byte) (v1 & 0x0F);          // lower nibble of ql[l+32]
                ql[l + 32] |= (byte) ((v3 & 0x0F) << 4);   // upper nibble of ql[l+32]

                qh[l] |= (byte) ((v0 >> 4) & 0x03);        // bits[1:0] from upper 2 bits of v0
                qh[l] |= (byte) (((v1 >> 4) & 0x03) << 2); // bits[3:2] from upper 2 bits of v1
                qh[l] |= (byte) (((v2 >> 4) & 0x03) << 4); // bits[5:4] from upper 2 bits of v2
                qh[l] |= (byte) (((v3 >> 4) & 0x03) << 6); // bits[7:6] from upper 2 bits of v3
            }

            // Group 1: elements 128..255, qlOff=64, qhOff=32
            for (int l = 0; l < 32; l++) {
                int v0  = qStored[128 + l];       // -> ql[64+l]    lower nibble, qh[32+l] bits[1:0]
                int v1  = qStored[128 + l + 32];  // -> ql[64+l+32] lower nibble, qh[32+l] bits[3:2]
                int v2  = qStored[128 + l + 64];  // -> ql[64+l]    upper nibble, qh[32+l] bits[5:4]
                int v3  = qStored[128 + l + 96];  // -> ql[64+l+32] upper nibble, qh[32+l] bits[7:6]

                ql[64 + l]      |= (byte) (v0 & 0x0F);
                ql[64 + l]      |= (byte) ((v2 & 0x0F) << 4);
                ql[64 + l + 32] |= (byte) (v1 & 0x0F);
                ql[64 + l + 32] |= (byte) ((v3 & 0x0F) << 4);

                qh[32 + l] |= (byte) ((v0 >> 4) & 0x03);
                qh[32 + l] |= (byte) (((v1 >> 4) & 0x03) << 2);
                qh[32 + l] |= (byte) (((v2 >> 4) & 0x03) << 4);
                qh[32 + l] |= (byte) (((v3 >> 4) & 0x03) << 6);
            }

            // Step 6: write in dequantizer read order: ql, qh, scales, d
            buffer.put(ql);
            buffer.put(qh);
            buffer.put(scales);
            buffer.putShort(floatToFp16(d));
        }

        return output;
    }

    @Override
    public byte[] quantize(INDArray array) {
        INDArray flat = array.isVector() ? array : array.reshape(array.length());
        return quantize(flat.toFloatVector());
    }

    private static short floatToFp16(float value) {
        int bits = Float.floatToIntBits(value);
        int sign = (bits >> 16) & 0x8000;
        int exponent = ((bits >> 23) & 0xFF) - 127 + 15;
        int mantissa = bits & 0x7FFFFF;

        if (exponent <= 0) {
            if (exponent < -10) return (short) sign;
            mantissa |= 0x800000;
            mantissa = mantissa >> (1 - exponent);
            return (short) (sign | (mantissa >> 13));
        } else if (exponent >= 31) {
            if ((bits & 0x7FFFFFFF) > 0x7F800000) return (short) (sign | 0x7E00);
            return (short) (sign | 0x7C00);
        }
        return (short) (sign | (exponent << 10) | (mantissa >> 13));
    }
}
