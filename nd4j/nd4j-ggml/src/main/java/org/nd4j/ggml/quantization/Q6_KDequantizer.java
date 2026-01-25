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
 * Dequantizer for Q6_K quantization format.
 *
 * Q6_K is a 6-bit k-quant format with super-blocks of 256 elements.
 */
public class Q6_KDequantizer implements Dequantizer {

    private static final int BLOCK_SIZE = 256;
    private static final int BYTES_PER_BLOCK = 210;

    @Override
    public GGMLDataType getQuantType() {
        return GGMLDataType.GGML_TYPE_Q6_K;
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
            // Read d as FP16 (2 bytes) - must match quantizer order
            short dRaw = buffer.getShort();
            float d = fp16ToFloat(dRaw);

            // Read scales (16 bytes for 16 sub-blocks, 8-bit each)
            byte[] scales = new byte[16];
            buffer.get(scales);

            // Read packed data (192 bytes: 4 x 6-bit values per 3 bytes)
            byte[] packedData = new byte[192];
            buffer.get(packedData);

            // Unpack 6-bit quantized values
            int[] quantizedValues = new int[256];
            for (int i = 0; i < 64; i++) {
                int byteIdx = i * 3;
                int b0 = packedData[byteIdx] & 0xFF;
                int b1 = packedData[byteIdx + 1] & 0xFF;
                int b2 = packedData[byteIdx + 2] & 0xFF;

                // Unpack 4 x 6-bit values from 3 bytes (matches quantizer packing)
                int idx = i * 4;
                quantizedValues[idx] = b0 & 0x3F;
                quantizedValues[idx + 1] = ((b0 >> 6) | (b1 << 2)) & 0x3F;
                quantizedValues[idx + 2] = ((b1 >> 4) | (b2 << 4)) & 0x3F;
                quantizedValues[idx + 3] = (b2 >> 2) & 0x3F;
            }

            // Dequantize: original = (quantized - 32) * scale
            // Quantizer stored values as 0-63 (added 32 to signed -32..31)
            for (int j = 0; j < 256 && outputIdx < totalElements; j++) {
                int subBlock = j / 16;
                float scale = d * (scales[subBlock] & 0x7F);  // 7-bit unsigned scale

                int val = quantizedValues[j] - 32;  // Convert back to signed
                result[outputIdx++] = val * scale;
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
                .quantType(GGMLDataType.GGML_TYPE_Q6_K)
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
