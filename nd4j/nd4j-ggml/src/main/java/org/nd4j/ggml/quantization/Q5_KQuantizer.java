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
 * Quantizer for Q5_K format (k-quant).
 * This is the reverse operation of {@link Q5_KDequantizer}.
 *
 * Q5_K block layout (from ggml-common.h block_q5_K):
 * <pre>
 *   ggml_half d         — super-block scale for quantized scales (2 bytes)
 *   ggml_half dmin      — super-block scale for quantized mins (2 bytes)
 *   uint8_t scales[12]  — scales and mins, same packing as Q4_K (12 bytes)
 *   uint8_t qh[32]      — high bit of each 5-bit value (256 / 8 = 32 bytes)
 *   uint8_t qs[128]     — low 4 bits of each 5-bit value (256 / 2 = 128 bytes)
 * </pre>
 * Total: 176 bytes per super-block of 256 elements.
 *
 * The qh/qs layout matches ggml's dequantize_row_q5_K exactly.
 */
public class Q5_KQuantizer implements Quantizer {

    private static final int BLOCK_SIZE = 256;
    private static final int BYTES_PER_BLOCK = 176; // 2 + 2 + 12 + 32 + 128
    private static final int SUB_BLOCK_SIZE = 32;
    private static final int NUM_SUB_BLOCKS = 8;

    @Override
    public GGMLDataType getQuantType() {
        return GGMLDataType.GGML_TYPE_Q5_K;
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

            // Step 1: compute per-sub-block min and scale (range / 31.0 for 5-bit)
            float[] subBlockScales = new float[NUM_SUB_BLOCKS];
            float[] subBlockMins   = new float[NUM_SUB_BLOCKS];

            for (int sb = 0; sb < NUM_SUB_BLOCKS; sb++) {
                int start = blockStart + sb * SUB_BLOCK_SIZE;
                int end = Math.min(start + SUB_BLOCK_SIZE, floatData.length);

                if (start >= floatData.length) {
                    subBlockMins[sb]   = 0;
                    subBlockScales[sb] = 0;
                    continue;
                }

                float minVal = floatData[start];
                float maxVal = floatData[start];
                for (int i = start + 1; i < end; i++) {
                    if (floatData[i] < minVal) minVal = floatData[i];
                    if (floatData[i] > maxVal) maxVal = floatData[i];
                }

                subBlockMins[sb]   = -minVal;  // stored as positive (negated)
                subBlockScales[sb] = (maxVal - minVal) / 31.0f; // 5-bit range 0..31
            }

            // Step 2: compute super-block d and dmin
            float maxScale = 0;
            float maxMin   = 0;
            for (int sb = 0; sb < NUM_SUB_BLOCKS; sb++) {
                if (subBlockScales[sb] > maxScale) maxScale = subBlockScales[sb];
                if (subBlockMins[sb]   > maxMin)   maxMin   = subBlockMins[sb];
            }

            float invMaxScale = (maxScale > 0) ? 63.0f / maxScale : 0.0f;
            float invMaxMin   = (maxMin   > 0) ? 63.0f / maxMin   : 0.0f;

            float d    = maxScale / 63.0f;
            float dmin = maxMin   / 63.0f;

            // Step 3: quantize sub-block scales and mins to 6 bits each
            int[] L = new int[NUM_SUB_BLOCKS]; // quantized scales (0..63)
            int[] M = new int[NUM_SUB_BLOCKS]; // quantized mins   (0..63)
            for (int sb = 0; sb < NUM_SUB_BLOCKS; sb++) {
                L[sb] = (int) Math.min(63, Math.round(subBlockScales[sb] * invMaxScale));
                M[sb] = (int) Math.min(63, Math.round(subBlockMins[sb]   * invMaxMin));
            }

            // Step 4: pack L[] and M[] into 12 bytes using get_scale_min_k4 packing
            // (identical layout to Q4_K)
            byte[] scaleBytes = new byte[12];
            for (int j = 0; j < 4; j++) {
                scaleBytes[j]     = (byte) ((L[j] & 0x3F) | ((M[j] & 0x30) << 2));
                scaleBytes[j + 4] = (byte) (M[j] & 0x3F);
            }
            for (int j = 4; j < 8; j++) {
                scaleBytes[j + 4] = (byte) ((L[j] & 0x0F) | ((M[j] & 0x0F) << 4));
                scaleBytes[j - 4] |= (byte) (((L[j] >> 4) & 3) << 6);
                scaleBytes[j]     |= (byte) (((M[j] >> 4) & 3) << 6);
            }

            // Step 5: quantize all 256 data values to 5-bit integers (0..31)
            // Using: quant = round((val + dmin * M[sb]) / (d * L[sb]))
            int[] quantizedData = new int[BLOCK_SIZE];
            for (int sb = 0; sb < NUM_SUB_BLOCKS; sb++) {
                float effScale = d * L[sb];
                float effMin   = dmin * M[sb];
                float invScale = (effScale > 0) ? 1.0f / effScale : 0.0f;

                for (int i = 0; i < SUB_BLOCK_SIZE; i++) {
                    int idx = blockStart + sb * SUB_BLOCK_SIZE + i;
                    float val = (idx < floatData.length) ? floatData[idx] : 0;
                    int q = (int) Math.round((val + effMin) * invScale);
                    quantizedData[sb * SUB_BLOCK_SIZE + i] = Math.max(0, Math.min(31, q));
                }
            }

            // Step 6: write d and dmin as FP16
            buffer.putShort(floatToFp16(d));
            buffer.putShort(floatToFp16(dmin));

            // Step 7: write the 12 scale bytes
            buffer.put(scaleBytes);

            // Step 8: build qh[32] and qs[128] matching ggml's dequantize_row_q5_K layout.
            //
            // The dequantizer processes elements in groups of 64:
            //   j=0:   elements  0..31  use qs[0..31] low  nibble + bit0 of qh[0..31]
            //          elements 32..63  use qs[0..31] high nibble + bit1 of qh[0..31]
            //   j=64:  elements 64..95  use qs[32..63] low  nibble + bit2 of qh[0..31]
            //          elements 96..127 use qs[32..63] high nibble + bit3 of qh[0..31]
            //   j=128: elements128..159 use qs[64..95] low  nibble + bit4 of qh[0..31]
            //          elements160..191 use qs[64..95] high nibble + bit5 of qh[0..31]
            //   j=192: elements192..223 use qs[96..127] low  nibble + bit6 of qh[0..31]
            //          elements224..255 use qs[96..127] high nibble + bit7 of qh[0..31]
            //
            // So qs is laid out as 4 groups of 32 bytes where each group covers a span
            // of 32 positions in the element space. The high bit accumulates in qh[0..31].
            byte[] qh = new byte[32];
            byte[] qs = new byte[128];

            // Process 4 groups of 64 elements each
            for (int group = 0; group < 4; group++) {
                int groupBase = group * 64; // base index into quantizedData[]
                int qsBase    = group * 32; // base index into qs[]
                int u1 = 1 << (group * 2);     // bit mask for low-element high bit
                int u2 = 1 << (group * 2 + 1); // bit mask for high-element high bit

                for (int l = 0; l < 32; l++) {
                    int v0 = quantizedData[groupBase + l];       // element for low nibble
                    int v1 = quantizedData[groupBase + l + 32];  // element for high nibble

                    // Pack low 4 bits of both values into one qs byte
                    qs[qsBase + l] = (byte) ((v0 & 0x0F) | ((v1 & 0x0F) << 4));

                    // Accumulate high bit (bit 4) into qh
                    if ((v0 & 0x10) != 0) qh[l] |= (byte) u1;
                    if ((v1 & 0x10) != 0) qh[l] |= (byte) u2;
                }
            }

            // Step 9: write qh and qs
            buffer.put(qh);
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
