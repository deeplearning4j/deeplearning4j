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
 * Quantizer for Q5_0 format.
 * This is the reverse operation of {@link Q5_0Dequantizer}.
 *
 * Q5_0 format:
 * - Block size: 32 elements
 * - Block structure: 2 bytes (FP16 scale) + 4 bytes (high bits) + 16 bytes (low 4 bits)
 * - Total: 22 bytes per block
 * - Values: 5-bit signed integers (combining 4 low bits + 1 high bit)
 */
public class Q5_0Quantizer implements Quantizer {

    private static final int BLOCK_SIZE = 32;
    private static final int BYTES_PER_BLOCK = 22; // 2 (scale) + 4 (high bits) + 16 (low bits)

    @Override
    public GGMLDataType getQuantType() {
        return GGMLDataType.GGML_TYPE_Q5_0;
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

            // Calculate scale (5-bit signed range is -16 to 15)
            float scale = maxAbs / 15.0f;
            if (scale == 0) scale = 1.0f;

            float invScale = 1.0f / scale;

            // Write scale as FP16
            buffer.putShort(floatToFp16(scale));

            // Quantize values and separate into high bits and low bits
            int[] quantized = new int[BLOCK_SIZE];
            for (int i = 0; i < BLOCK_SIZE; i++) {
                int idx = start + i;
                float value = idx < floatData.length ? floatData[idx] : 0;
                int q = Math.round(value * invScale);
                q = Math.max(-16, Math.min(15, q));
                // Store as 0..31 (add 16)
                quantized[i] = q + 16;
            }

            // Write high bits (bit 4 of each value) - 4 bytes for 32 bits
            int highBits = 0;
            for (int i = 0; i < 32; i++) {
                if ((quantized[i] & 0x10) != 0) {
                    highBits |= (1 << i);
                }
            }
            buffer.putInt(highBits);

            // Write low 4 bits - pack pairs into bytes
            for (int i = 0; i < 16; i++) {
                int low0 = quantized[i * 2] & 0x0F;
                int low1 = quantized[i * 2 + 1] & 0x0F;
                buffer.put((byte) (low0 | (low1 << 4)));
            }
        }

        return output;
    }

    @Override
    public byte[] quantize(INDArray array) {
        // Flatten to 1D if needed for toFloatVector()
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
