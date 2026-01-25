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
 * Dequantizer for Q4_K quantization format.
 *
 * Q4_K is a k-quant format with super-blocks:
 * - Super-block size: 256 elements
 * - Contains 8 sub-blocks of 32 elements each
 * - Per super-block: 2 FP16 scales (d, dmin) = 4 bytes + 12 bytes scales + 4 bytes mins + 128 bytes data
 * - Total: 148 bytes per super-block (256 elements)
 */
public class Q4_KDequantizer implements Dequantizer {

    private static final int BLOCK_SIZE = 256;
    private static final int BYTES_PER_BLOCK = 148;
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
    public float[] dequantize(byte[] quantizedData, long numElements) {
        if (numElements > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Too many elements: " + numElements);
        }

        int totalElements = (int) numElements;
        float[] result = new float[totalElements];
        ByteBuffer buffer = ByteBuffer.wrap(quantizedData).order(ByteOrder.LITTLE_ENDIAN);

        int numBlocks = (totalElements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int outputIdx = 0;

        for (int block = 0; block < numBlocks && buffer.remaining() >= BYTES_PER_BLOCK; block++) {
            // Read super-block scales
            short dRaw = buffer.getShort();
            short dminRaw = buffer.getShort();
            float d = fp16ToFloat(dRaw);
            float dmin = fp16ToFloat(dminRaw);

            // Read 12 bytes of 6-bit scales (for 8 sub-blocks, packed)
            byte[] scaleBytes = new byte[12];
            buffer.get(scaleBytes);

            // Read 4 bytes of 6-bit mins (packed, lower 4 bits only used)
            byte[] minBytes = new byte[4];
            buffer.get(minBytes);

            // Unpack scales and mins for each sub-block
            float[] scales = new float[NUM_SUB_BLOCKS];
            float[] mins = new float[NUM_SUB_BLOCKS];

            // Unpack 8 x 6-bit scales from 6 bytes (48 bits)
            long packedScales = 0;
            for (int i = 0; i < 6; i++) {
                packedScales |= ((long) (scaleBytes[i] & 0xFF)) << (i * 8);
            }
            for (int i = 0; i < NUM_SUB_BLOCKS; i++) {
                int scaleVal = (int) ((packedScales >> (i * 6)) & 0x3F);
                scales[i] = d * scaleVal;

                // Extract 4-bit min for each sub-block (1 sign bit + 3 magnitude bits)
                int byteIdx = i / 2;
                int shift = (i % 2) * 4;
                int packedMin = (minBytes[byteIdx] >> shift) & 0x0F;
                int sign = (packedMin >> 3) & 0x01;
                int magnitude = packedMin & 0x07;
                mins[i] = dmin * magnitude * (sign == 1 ? -1 : 1);
            }

            // Read 128 bytes of quantized data (256 x 4 bits)
            byte[] dataBytes = new byte[128];
            buffer.get(dataBytes);

            // Dequantize each sub-block
            // Formula: original = quantized * scale + min
            for (int sb = 0; sb < NUM_SUB_BLOCKS && outputIdx < totalElements; sb++) {
                float scale = scales[sb];
                float min = mins[sb];

                for (int i = 0; i < SUB_BLOCK_SIZE / 2 && outputIdx < totalElements; i++) {
                    int byteIdx = sb * 16 + i;
                    byte packed = dataBytes[byteIdx];

                    // Low nibble
                    int val0 = packed & 0x0F;
                    if (outputIdx < totalElements) {
                        result[outputIdx++] = val0 * scale + min;
                    }

                    // High nibble
                    int val1 = (packed >> 4) & 0x0F;
                    if (outputIdx < totalElements) {
                        result[outputIdx++] = val1 * scale + min;
                    }
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

        return QuantizationInfo.builder()
                .quantType(GGMLDataType.GGML_TYPE_Q4_K)
                .blockSize(BLOCK_SIZE)
                .numBlocks(numBlocks)
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
