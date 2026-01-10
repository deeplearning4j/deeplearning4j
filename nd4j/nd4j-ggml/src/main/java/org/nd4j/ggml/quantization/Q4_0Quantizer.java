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
 * Quantizer for Q4_0 format.
 * This is the reverse operation of {@link Q4_0Dequantizer}.
 *
 * Q4_0 format:
 * - Block size: 32 elements
 * - Block structure: 2 bytes (FP16 scale) + 16 bytes (32 x 4-bit values)
 * - Total: 18 bytes per block
 * - Values are stored as unsigned 4-bit integers 0-15, representing -8 to 7
 */
public class Q4_0Quantizer implements Quantizer {

    private static final int BLOCK_SIZE = 32;
    private static final int BYTES_PER_BLOCK = 18; // 2 (scale) + 16 (data)

    @Override
    public GGMLDataType getQuantType() {
        return GGMLDataType.GGML_TYPE_Q4_0;
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
            int start = block * BLOCK_SIZE;
            int end = Math.min(start + BLOCK_SIZE, floatData.length);

            // Find max absolute value for scaling
            float maxAbs = 0;
            for (int i = start; i < end; i++) {
                maxAbs = Math.max(maxAbs, Math.abs(floatData[i]));
            }

            // Calculate scale (4-bit signed range is -8 to 7)
            float scale = maxAbs / 7.0f;
            if (scale == 0) scale = 1.0f; // Avoid division by zero

            float invScale = 1.0f / scale;

            // Write scale as FP16
            buffer.putShort(floatToFp16(scale));

            // Quantize and pack 32 values into 16 bytes (4 bits each)
            for (int i = 0; i < 16; i++) {
                int idx0 = start + i * 2;
                int idx1 = start + i * 2 + 1;

                float val0 = idx0 < floatData.length ? floatData[idx0] : 0;
                float val1 = idx1 < floatData.length ? floatData[idx1] : 0;

                // Quantize to -8..7 range, then add 8 for storage as 0..15
                int q0 = quantizeValue(val0, invScale);
                int q1 = quantizeValue(val1, invScale);

                // Pack two 4-bit values into one byte
                // Low nibble = q0, High nibble = q1
                buffer.put((byte) ((q0 & 0x0F) | ((q1 & 0x0F) << 4)));
            }
        }

        return output;
    }

    private int quantizeValue(float value, float invScale) {
        // Quantize to -8..7 range
        int q = Math.round(value * invScale);
        q = Math.max(-8, Math.min(7, q));
        // Store as 0..15 (add 8)
        return q + 8;
    }

    @Override
    public byte[] quantize(INDArray array) {
        return quantize(array.toFloatVector());
    }

    /**
     * Convert float to FP16 (IEEE 754 half-precision)
     */
    private static short floatToFp16(float value) {
        int bits = Float.floatToIntBits(value);
        int sign = (bits >> 16) & 0x8000;
        int exponent = ((bits >> 23) & 0xFF) - 127 + 15;
        int mantissa = bits & 0x7FFFFF;

        if (exponent <= 0) {
            if (exponent < -10) {
                return (short) sign;
            }
            mantissa |= 0x800000;
            int shift = 1 - exponent;
            mantissa = mantissa >> shift;
            return (short) (sign | (mantissa >> 13));
        } else if (exponent >= 31) {
            if ((bits & 0x7FFFFFFF) > 0x7F800000) {
                return (short) (sign | 0x7E00);
            }
            return (short) (sign | 0x7C00);
        }

        return (short) (sign | (exponent << 10) | (mantissa >> 13));
    }
}
