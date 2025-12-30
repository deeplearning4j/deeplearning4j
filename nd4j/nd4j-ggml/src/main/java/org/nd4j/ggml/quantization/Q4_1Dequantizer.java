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
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Dequantizer for Q4_1 quantization format.
 *
 * Q4_1 format:
 * - Block size: 32 elements
 * - Block structure: 2 bytes (FP16 scale) + 2 bytes (FP16 min) + 16 bytes (32 x 4-bit values)
 * - Total: 20 bytes per block
 * - Values are stored as unsigned 4-bit integers (0 to 15)
 * - Dequantized: value = quantized * scale + min
 */
public class Q4_1Dequantizer implements Dequantizer {

    private static final int BLOCK_SIZE = 32;
    private static final int BYTES_PER_BLOCK = 20; // 2 (scale) + 2 (min) + 16 (data)

    @Override
    public GGMLDataType getQuantType() {
        return GGMLDataType.GGML_TYPE_Q4_1;
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

            // Read 16 bytes = 32 x 4-bit values
            for (int i = 0; i < 16 && outputIdx < totalElements; i++) {
                byte packed = buffer.get();

                // Low nibble (bits 0-3) - unsigned
                int val0 = packed & 0x0F;
                if (outputIdx < totalElements) {
                    result[outputIdx++] = val0 * scale + min;
                }

                // High nibble (bits 4-7) - unsigned
                int val1 = (packed >> 4) & 0x0F;
                if (outputIdx < totalElements) {
                    result[outputIdx++] = val1 * scale + min;
                }
            }
        }

        return result;
    }

    @Override
    public INDArray dequantizeToArray(byte[] quantizedData, long[] shape, DataType targetType) {
        long numElements = calculateNumElements(shape);
        float[] floatData = dequantize(quantizedData, numElements);

        INDArray array = Nd4j.create(floatData, shape);
        if (targetType != DataType.FLOAT) {
            array = array.castTo(targetType);
        }
        return array;
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
            buffer.position(buffer.position() + 16); // Skip data bytes
        }

        return QuantizationInfo.builder()
                .quantType(GGMLDataType.GGML_TYPE_Q4_1)
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
            if (mantissa == 0) {
                return sign == 0 ? 0.0f : -0.0f;
            }
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
