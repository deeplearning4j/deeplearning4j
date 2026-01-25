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
 * Quantizer for Q8_0 format.
 * This is the reverse operation of {@link Q8_0Dequantizer}.
 *
 * Q8_0 format:
 * - Block size: 32 elements
 * - Block structure: 2 bytes (FP16 scale) + 32 bytes (32 x 8-bit values)
 * - Total: 34 bytes per block
 * - Values are stored as signed 8-bit integers (-128 to 127)
 */
public class Q8_0Quantizer implements Quantizer {

    private static final int BLOCK_SIZE = 32;
    private static final int BYTES_PER_BLOCK = 34; // 2 (scale) + 32 (data)

    @Override
    public GGMLDataType getQuantType() {
        return GGMLDataType.GGML_TYPE_Q8_0;
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

            // Calculate scale (8-bit signed range is -128 to 127)
            float scale = maxAbs / 127.0f;
            if (scale == 0) scale = 1.0f; // Avoid division by zero

            float invScale = 1.0f / scale;

            // Write scale as FP16
            buffer.putShort(floatToFp16(scale));

            // Quantize and write 32 values as 8-bit signed integers
            for (int i = 0; i < BLOCK_SIZE; i++) {
                int idx = start + i;
                float value = idx < floatData.length ? floatData[idx] : 0;

                // Quantize to -128..127 range
                int quantized = Math.round(value * invScale);
                quantized = Math.max(-128, Math.min(127, quantized));

                buffer.put((byte) quantized);
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

    /**
     * Convert float to FP16 (IEEE 754 half-precision)
     */
    private static short floatToFp16(float value) {
        int bits = Float.floatToIntBits(value);
        int sign = (bits >> 16) & 0x8000;
        int exponent = ((bits >> 23) & 0xFF) - 127 + 15;
        int mantissa = bits & 0x7FFFFF;

        if (exponent <= 0) {
            // Subnormal or zero
            if (exponent < -10) {
                return (short) sign; // Too small, become zero
            }
            mantissa |= 0x800000; // Add implicit 1
            int shift = 1 - exponent;
            mantissa = mantissa >> shift;
            return (short) (sign | (mantissa >> 13));
        } else if (exponent >= 31) {
            // Infinity or NaN
            if ((bits & 0x7FFFFFFF) > 0x7F800000) {
                // NaN
                return (short) (sign | 0x7E00);
            }
            return (short) (sign | 0x7C00); // Infinity
        }

        return (short) (sign | (exponent << 10) | (mantissa >> 13));
    }
}
