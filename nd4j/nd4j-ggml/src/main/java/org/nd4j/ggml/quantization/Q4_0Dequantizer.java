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

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Dequantizer for Q4_0 quantization format.
 *
 * Q4_0 format:
 * - Block size: 32 elements
 * - Block structure: 2 bytes (FP16 scale) + 16 bytes (32 x 4-bit values)
 * - Total: 18 bytes per block
 * - Values are stored as signed 4-bit integers (-8 to 7)
 */
public class Q4_0Dequantizer implements Dequantizer {

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
    public float[] dequantize(byte[] quantizedData, long numElements) {
        if (numElements > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Too many elements: " + numElements);
        }

        int totalElements = (int) numElements;
        float[] result = new float[totalElements];
        ByteBuffer buffer = ByteBuffer.wrap(quantizedData).order(ByteOrder.LITTLE_ENDIAN);

        int numBlocks = (totalElements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int outputIdx = 0;

        for (int block = 0; block < numBlocks && buffer.hasRemaining(); block++) {
            // Read scale as FP16
            short scaleRaw = buffer.getShort();
            float scale = fp16ToFloat(scaleRaw);

            // Read 16 bytes = 32 x 4-bit values
            for (int i = 0; i < 16 && outputIdx < totalElements; i++) {
                byte packed = buffer.get();

                // Low nibble (bits 0-3)
                int val0 = (packed & 0x0F) - 8;
                if (outputIdx < totalElements) {
                    result[outputIdx++] = val0 * scale;
                }

                // High nibble (bits 4-7)
                int val1 = ((packed >> 4) & 0x0F) - 8;
                if (outputIdx < totalElements) {
                    result[outputIdx++] = val1 * scale;
                }
            }
        }

        return result;
    }

    @Override
    public QuantizationInfo extractQuantizationInfo(byte[] quantizedData, long[] shape) {
        long numElements = calculateNumElements(shape);
        int numBlocks = (int) ((numElements + BLOCK_SIZE - 1) / BLOCK_SIZE);

        float[] scales = new float[numBlocks];
        ByteBuffer buffer = ByteBuffer.wrap(quantizedData).order(ByteOrder.LITTLE_ENDIAN);

        for (int block = 0; block < numBlocks && buffer.remaining() >= BYTES_PER_BLOCK; block++) {
            short scaleRaw = buffer.getShort();
            scales[block] = fp16ToFloat(scaleRaw);
            buffer.position(buffer.position() + 16); // Skip data bytes
        }

        return QuantizationInfo.builder()
                .quantType(GGMLDataType.GGML_TYPE_Q4_0)
                .blockSize(BLOCK_SIZE)
                .numBlocks(numBlocks)
                .scales(scales)
                .originalShape(shape)
                .build();
    }

    private static long calculateNumElements(long[] shape) {
        long total = 1;
        for (long dim : shape) {
            total *= dim;
        }
        return total;
    }

    /**
     * Convert FP16 (IEEE 754 half-precision) to float
     */
    private static float fp16ToFloat(short fp16) {
        int sign = (fp16 >> 15) & 0x1;
        int exponent = (fp16 >> 10) & 0x1F;
        int mantissa = fp16 & 0x3FF;

        if (exponent == 0) {
            if (mantissa == 0) {
                return sign == 0 ? 0.0f : -0.0f;
            }
            // Subnormal
            float value = mantissa / 1024.0f * (float) Math.pow(2, -14);
            return sign == 0 ? value : -value;
        } else if (exponent == 31) {
            if (mantissa == 0) {
                return sign == 0 ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY;
            }
            return Float.NaN;
        }

        float value = (1.0f + mantissa / 1024.0f) * (float) Math.pow(2, exponent - 15);
        return sign == 0 ? value : -value;
    }
}
