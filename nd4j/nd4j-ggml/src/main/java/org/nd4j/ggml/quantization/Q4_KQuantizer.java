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
 * Quantizer for Q4_K format (k-quant).
 * This is the reverse operation of {@link Q4_KDequantizer}.
 *
 * Q4_K block layout (from ggml-common.h block_q4_K):
 * <pre>
 *   ggml_half d         — super-block scale for quantized scales (2 bytes)
 *   ggml_half dmin      — super-block scale for quantized mins (2 bytes)
 *   uint8_t scales[12]  — scales and mins packed with ggml's get_scale_min_k4 packing (12 bytes)
 *   uint8_t qs[128]     — 4-bit quants (256 / 2) (128 bytes)
 * </pre>
 * Total: 144 bytes per super-block of 256 elements.
 *
 * The 12-byte scales[] array encodes 8 sub-block scales (6-bit each) and 8 sub-block
 * mins (6-bit each) using ggml's compact packing (same as in dequantize_row_q4_K).
 */
public class Q4_KQuantizer implements Quantizer {

    private static final int BLOCK_SIZE = 256;
    private static final int BYTES_PER_BLOCK = 144; // 2 + 2 + 12 + 128
    private static final int SUB_BLOCK_SIZE = 32;
    private static final int NUM_SUB_BLOCKS = 8;

    @Override
    public GGMLDataType getQuantType() {
        return GGMLDataType.GGML_TYPE_Q4_K;
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

            // Step 1: compute per-sub-block min and scale (range / 15.0)
            float[] subBlockScales = new float[NUM_SUB_BLOCKS]; // raw ranges
            float[] subBlockMins = new float[NUM_SUB_BLOCKS];

            for (int sb = 0; sb < NUM_SUB_BLOCKS; sb++) {
                int start = blockStart + sb * SUB_BLOCK_SIZE;
                int end = Math.min(start + SUB_BLOCK_SIZE, floatData.length);

                if (start >= floatData.length) {
                    subBlockMins[sb] = 0;
                    subBlockScales[sb] = 0;
                    continue;
                }

                float minVal = floatData[start];
                float maxVal = floatData[start];
                for (int i = start + 1; i < end; i++) {
                    if (floatData[i] < minVal) minVal = floatData[i];
                    if (floatData[i] > maxVal) maxVal = floatData[i];
                }

                subBlockMins[sb] = -minVal;  // stored as positive value (negated)
                subBlockScales[sb] = (maxVal - minVal) / 15.0f; // 4-bit range 0..15
            }

            // Step 2: compute super-block d (scale for scales) and dmin (scale for mins)
            float maxScale = 0;
            float maxMin = 0;
            for (int sb = 0; sb < NUM_SUB_BLOCKS; sb++) {
                if (subBlockScales[sb] > maxScale) maxScale = subBlockScales[sb];
                if (subBlockMins[sb] > maxMin) maxMin = subBlockMins[sb];
            }

            float invMaxScale = (maxScale > 0) ? 63.0f / maxScale : 0.0f;
            float invMaxMin   = (maxMin > 0)   ? 63.0f / maxMin   : 0.0f;

            float d    = maxScale / 63.0f;
            float dmin = maxMin / 63.0f;

            // Step 3: quantize sub-block scales and mins to 6 bits each
            int[] L = new int[NUM_SUB_BLOCKS]; // quantized scales (0..63)
            int[] M = new int[NUM_SUB_BLOCKS]; // quantized mins   (0..63)
            for (int sb = 0; sb < NUM_SUB_BLOCKS; sb++) {
                L[sb] = (int) Math.min(63, Math.round(subBlockScales[sb] * invMaxScale));
                M[sb] = (int) Math.min(63, Math.round(subBlockMins[sb] * invMaxMin));
            }

            // Step 4: pack L[] and M[] into 12 bytes using ggml's get_scale_min_k4 packing.
            // The packing layout is (from ggml-common.h make_qkx2_quants / set_scale_min_k4):
            //   bytes[0..3]:  L[0]..L[3] in bits 0..5, upper 2 bits of M[0]..M[3] in bits 6..7
            //   bytes[4..7]:  M[0]..M[3] in bits 0..5  (lower 6 bits; upper 2 stored in bytes[0..3])
            //   bytes[8..11]: bottom nibble = lower 4 bits of L[4..7] or M[4..7],
            //                 top nibble    = upper 2 bits packed
            // This is the exact inverse of get_scale_min_k4 used in Q4_KDequantizer.
            //
            // get_scale_min_k4(j, q):
            //   j < 4:  sc = q[j] & 63,       m = q[j+4] & 63
            //   j >= 4: sc = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4),
            //           m  = (q[j+4] >> 4)   | ((q[j]   >> 6) << 4)
            //
            // Reverse-engineering the packing for L and M:
            // For j < 4:
            //   q[j]   = L[j] | ((M[j] & 0x30) << 2)  — lower 6 bits of L[j] + upper 2 bits of M[j] in bits 6-7
            //   q[j+4] = M[j] & 0x3F                   — lower 6 bits of M[j]
            // For j in [4..7]:
            //   scale sc[j] = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4)
            //   min   m[j]  = (q[j+4] >> 4)  | ((q[j]   >> 6) << 4)
            //   => q[j+4] lower nibble = L[j] & 0xF, upper nibble = M[j] & 0xF
            //   => q[j-4] bits 6-7 = (L[j] >> 4) & 3  -- these are the 2 high bits of the scale
            //   => q[j]   bits 6-7 = (M[j] >> 4) & 3  -- these are the 2 high bits of the min
            byte[] scaleBytes = new byte[12];

            // Fill bytes[0..3] and bytes[4..7] for j in [0..3]
            for (int j = 0; j < 4; j++) {
                // lower 6 bits of L[j]; upper 2 bits of M[j] go into bits 6-7
                scaleBytes[j]     = (byte) ((L[j] & 0x3F) | ((M[j] & 0x30) << 2));
                // lower 6 bits of M[j]
                scaleBytes[j + 4] = (byte) (M[j] & 0x3F);
            }
            // Fill bytes[8..11] for j in [4..7], and update high bits in bytes[j-4] and bytes[j]
            for (int j = 4; j < 8; j++) {
                // bytes[j+4] = bytes[8..11]: lower nibble = lower 4 bits of L[j], upper nibble = lower 4 bits of M[j]
                scaleBytes[j + 4] = (byte) ((L[j] & 0x0F) | ((M[j] & 0x0F) << 4));
                // Put upper 2 bits of L[j] into bits 6-7 of bytes[j-4]
                scaleBytes[j - 4] |= (byte) (((L[j] >> 4) & 3) << 6);
                // Put upper 2 bits of M[j] into bits 6-7 of bytes[j]
                // but bytes[j] for j in [4..7] means scaleBytes[4..7]
                scaleBytes[j]     |= (byte) (((M[j] >> 4) & 3) << 6);
            }

            // Step 5: write d and dmin as FP16
            buffer.putShort(floatToFp16(d));
            buffer.putShort(floatToFp16(dmin));

            // Step 6: write the 12 scale bytes
            buffer.put(scaleBytes);

            // Step 7: quantize all 256 data values and write 128 bytes matching
            // ggml's dequantize_row_q4_K layout exactly.
            //
            // The dequantizer outer loop processes j in {0, 64, 128, 192}, each group of 64.
            // For group at j (is = j/32 in sub-block index), with qIdx = j/2:
            //   First 32 elements  (j..j+31):  low  nibbles of qs[qIdx..qIdx+31], scale d1 = d*sc[is]
            //   Second 32 elements (j+32..j+63): high nibbles of qs[qIdx..qIdx+31], scale d2 = d*sc[is+1]
            //
            // So the qs layout is:
            //   qs[0..31]:   low nibble = element 0..31   (sub-block 0, scale L[0])
            //                high nibble = element 32..63  (sub-block 1, scale L[1])
            //   qs[32..63]:  low nibble = element 64..95  (sub-block 2, scale L[2])
            //                high nibble = element 96..127 (sub-block 3, scale L[3])
            //   qs[64..95]:  low nibble = element 128..159 (sub-block 4, scale L[4])
            //                high nibble = element 160..191 (sub-block 5, scale L[5])
            //   qs[96..127]: low nibble = element 192..223 (sub-block 6, scale L[6])
            //                high nibble = element 224..255 (sub-block 7, scale L[7])

            // First, quantize all 256 elements into a temp array using sub-block scale/min
            int[] q = new int[BLOCK_SIZE];
            for (int sb = 0; sb < NUM_SUB_BLOCKS; sb++) {
                float effScale = d * L[sb];
                float effMin   = dmin * M[sb];
                float invScale = (effScale > 0) ? 1.0f / effScale : 0.0f;

                for (int i = 0; i < SUB_BLOCK_SIZE; i++) {
                    int idx = blockStart + sb * SUB_BLOCK_SIZE + i;
                    float val = (idx < floatData.length) ? floatData[idx] : 0;
                    int qi = (int) Math.round((val + effMin) * invScale);
                    q[sb * SUB_BLOCK_SIZE + i] = Math.max(0, Math.min(15, qi));
                }
            }

            // Now pack into qs[128] in dequantizer-expected layout
            byte[] qs = new byte[128];
            for (int group = 0; group < 4; group++) {
                int baseLow  = group * 64;       // start of "low nibble" sub-block in q[]
                int baseHigh = group * 64 + 32;  // start of "high nibble" sub-block in q[]
                int qsBase   = group * 32;       // start byte in qs[]
                for (int l = 0; l < 32; l++) {
                    qs[qsBase + l] = (byte) ((q[baseLow + l] & 0x0F) | ((q[baseHigh + l] & 0x0F) << 4));
                }
            }
            buffer.put(qs);
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
