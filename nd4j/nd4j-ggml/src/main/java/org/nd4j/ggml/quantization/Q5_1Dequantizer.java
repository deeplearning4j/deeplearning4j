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
 * Dequantizer for Q5_1 quantization format.
 *
 * Q5_1 format:
 * - Block size: 32 elements
 * - Block structure: 2 bytes (FP16 scale) + 2 bytes (FP16 min) + 4 bytes (high bits) + 16 bytes (low 4 bits)
 * - Total: 24 bytes per block
 * - Values are 5-bit unsigned integers with min offset
 */
public class Q5_1Dequantizer implements Dequantizer {

    private static final int BLOCK_SIZE = 32;
    private static final int BYTES_PER_BLOCK = 24; // 2 (scale) + 2 (min) + 4 (high bits) + 16 (low nibbles)

    @Override
    public GGMLDataType getQuantType() {
        return GGMLDataType.GGML_TYPE_Q5_1;
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
            // Read scale and min as FP16
            short scaleRaw = buffer.getShort();
            short minRaw = buffer.getShort();
            float scale = fp16ToFloat(scaleRaw);
            float min = fp16ToFloat(minRaw);

            // Read high bits (4 bytes = 32 bits)
            int highBits = buffer.getInt();

            // Read 16 bytes = 32 x 4-bit low values
            byte[] lowBytes = new byte[16];
            buffer.get(lowBytes);

            // Q5_1 GGML element layout within a 32-element block:
            //   qs[j] low nibble  = element j      (j = 0..15)
            //   qs[j] high nibble = element j + 16 (j = 0..15)
            //   qh bit j          = 5th bit of element j
            //   qh bit (j + 16)   = 5th bit of element j + 16
            for (int j = 0; j < 16; j++) {
                byte packed = lowBytes[j];

                // Element j (first half of block: positions 0..15)
                int xh0 = (highBits >> j) & 1;
                int low0 = packed & 0x0F;
                int val0 = low0 | (xh0 << 4);
                int idx0 = outputIdx + j;
                if (idx0 < totalElements) {
                    result[idx0] = val0 * scale + min;
                }

                // Element j+16 (second half of block: positions 16..31)
                int xh1 = (highBits >> (j + 16)) & 1;
                int low1 = (packed >> 4) & 0x0F;
                int val1 = low1 | (xh1 << 4);
                int idx1 = outputIdx + j + 16;
                if (idx1 < totalElements) {
                    result[idx1] = val1 * scale + min;
                }
            }
            outputIdx += 32;
        }

        return result;
    }

    @Override
    public QuantizationInfo extractQuantizationInfo(byte[] quantizedData, long[] shape) {
        long numElements = calculateNumElements(shape);
        int numBlocks = (int) ((numElements + BLOCK_SIZE - 1) / BLOCK_SIZE);

        float[] scales = new float[numBlocks];
        float[] mins = new float[numBlocks];
        ByteBuffer buffer = ByteBuffer.wrap(quantizedData).order(ByteOrder.LITTLE_ENDIAN);

        for (int block = 0; block < numBlocks && buffer.remaining() >= BYTES_PER_BLOCK; block++) {
            short scaleRaw = buffer.getShort();
            short minRaw = buffer.getShort();
            scales[block] = fp16ToFloat(scaleRaw);
            mins[block] = fp16ToFloat(minRaw);
            buffer.position(buffer.position() + 20); // Skip high bits + data
        }

        return QuantizationInfo.builder()
                .quantType(GGMLDataType.GGML_TYPE_Q5_1)
                .blockSize(BLOCK_SIZE)
                .numBlocks(numBlocks)
                .scales(scales)
                .mins(mins)
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

    private static float fp16ToFloat(short fp16) {
        int sign = (fp16 >> 15) & 0x1;
        int exponent = (fp16 >> 10) & 0x1F;
        int mantissa = fp16 & 0x3FF;

        if (exponent == 0) {
            if (mantissa == 0) return sign == 0 ? 0.0f : -0.0f;
            float value = mantissa / 1024.0f * (float) Math.pow(2, -14);
            return sign == 0 ? value : -value;
        } else if (exponent == 31) {
            return mantissa == 0 ? (sign == 0 ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY) : Float.NaN;
        }

        float value = (1.0f + mantissa / 1024.0f) * (float) Math.pow(2, exponent - 15);
        return sign == 0 ? value : -value;
    }
}
