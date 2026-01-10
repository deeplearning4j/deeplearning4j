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
 * Q4_K is a k-quant format with super-blocks:
 * - Super-block size: 256 elements
 * - Contains 8 sub-blocks of 32 elements each
 * - Per super-block: 2 FP16 scales (d, dmin) + 12 bytes scales + 4 bytes mins + 128 bytes data
 * - Total: 144 bytes per super-block
 */
public class Q4_KQuantizer implements Quantizer {

    private static final int BLOCK_SIZE = 256;
    private static final int BYTES_PER_BLOCK = 144;
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

            // Calculate per-sub-block scales and mins
            float[] subBlockScales = new float[NUM_SUB_BLOCKS];
            float[] subBlockMins = new float[NUM_SUB_BLOCKS];

            for (int sb = 0; sb < NUM_SUB_BLOCKS; sb++) {
                int start = blockStart + sb * SUB_BLOCK_SIZE;
                int end = Math.min(start + SUB_BLOCK_SIZE, floatData.length);

                float minVal = Float.MAX_VALUE;
                float maxVal = Float.MIN_VALUE;

                for (int i = start; i < end; i++) {
                    minVal = Math.min(minVal, floatData[i]);
                    maxVal = Math.max(maxVal, floatData[i]);
                }

                if (start >= floatData.length) {
                    minVal = 0;
                    maxVal = 0;
                }

                subBlockMins[sb] = minVal;
                subBlockScales[sb] = (maxVal - minVal) / 15.0f;
                if (subBlockScales[sb] == 0) subBlockScales[sb] = 1.0f;
            }

            // Find super-block d and dmin (max of sub-block scales/mins)
            float maxScale = 0;
            float maxMin = 0;
            for (int sb = 0; sb < NUM_SUB_BLOCKS; sb++) {
                maxScale = Math.max(maxScale, subBlockScales[sb]);
                maxMin = Math.max(maxMin, Math.abs(subBlockMins[sb]));
            }

            float d = maxScale / 63.0f;  // 6-bit scale range
            float dmin = maxMin / 15.0f; // 4-bit min range

            if (d == 0) d = 1.0f;
            if (dmin == 0) dmin = 1.0f;

            // Write d and dmin as FP16
            buffer.putShort(floatToFp16(d));
            buffer.putShort(floatToFp16(dmin));

            // Quantize and pack scales (6-bit each, 12 bytes total for 8 values)
            int[] quantizedScales = new int[NUM_SUB_BLOCKS];
            for (int sb = 0; sb < NUM_SUB_BLOCKS; sb++) {
                quantizedScales[sb] = Math.round(subBlockScales[sb] / d);
                quantizedScales[sb] = Math.max(0, Math.min(63, quantizedScales[sb]));
            }

            // Pack 8 x 6-bit scales into 12 bytes (48 bits total, aligned to 12 bytes)
            byte[] scaleBytes = new byte[12];
            packScales6Bit(quantizedScales, scaleBytes);
            buffer.put(scaleBytes);

            // Quantize and pack mins (4-bit each, 4 bytes for 8 values)
            byte[] minBytes = new byte[4];
            for (int sb = 0; sb < NUM_SUB_BLOCKS; sb++) {
                int quantizedMin = Math.round(Math.abs(subBlockMins[sb]) / dmin);
                quantizedMin = Math.max(0, Math.min(15, quantizedMin));
                int byteIdx = sb / 2;
                int shift = (sb % 2) * 4;
                minBytes[byteIdx] |= (quantizedMin << shift);
            }
            buffer.put(minBytes);

            // Quantize data (4-bit per value, 128 bytes for 256 values)
            for (int sb = 0; sb < NUM_SUB_BLOCKS; sb++) {
                float scale = quantizedScales[sb] * d;
                float min = subBlockMins[sb];
                float invScale = scale > 0 ? 1.0f / scale : 1.0f;

                for (int i = 0; i < SUB_BLOCK_SIZE / 2; i++) {
                    int idx0 = blockStart + sb * SUB_BLOCK_SIZE + i * 2;
                    int idx1 = idx0 + 1;

                    float val0 = idx0 < floatData.length ? floatData[idx0] : min;
                    float val1 = idx1 < floatData.length ? floatData[idx1] : min;

                    int q0 = Math.round((val0 - min) * invScale);
                    int q1 = Math.round((val1 - min) * invScale);
                    q0 = Math.max(0, Math.min(15, q0));
                    q1 = Math.max(0, Math.min(15, q1));

                    buffer.put((byte) ((q0 & 0x0F) | ((q1 & 0x0F) << 4)));
                }
            }
        }

        return output;
    }

    private void packScales6Bit(int[] scales, byte[] output) {
        // Pack 8 x 6-bit values into 6 bytes (48 bits)
        // Simplified packing - fits 8 x 6-bit = 48 bits into first 6 bytes
        long packed = 0;
        for (int i = 0; i < 8; i++) {
            packed |= ((long) (scales[i] & 0x3F)) << (i * 6);
        }
        for (int i = 0; i < 6; i++) {
            output[i] = (byte) ((packed >> (i * 8)) & 0xFF);
        }
        // Remaining 6 bytes are padding/unused in simplified implementation
    }

    @Override
    public byte[] quantize(INDArray array) {
        return quantize(array.toFloatVector());
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
