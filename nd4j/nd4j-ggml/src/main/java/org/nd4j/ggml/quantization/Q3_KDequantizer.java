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
 * Dequantizer for Q3_K quantization format.
 *
 * Q3_K is a 3-bit k-quant format with super-blocks of 256 elements.
 */
public class Q3_KDequantizer implements Dequantizer {

    private static final int BLOCK_SIZE = 256;
    private static final int BYTES_PER_BLOCK = 110;

    @Override
    public GGMLDataType getQuantType() {
        return GGMLDataType.GGML_TYPE_Q3_K;
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
            // Read high bits (32 bytes)
            byte[] hmask = new byte[32];
            buffer.get(hmask);

            // Read low 2 bits (64 bytes for 256 elements)
            byte[] qs = new byte[64];
            buffer.get(qs);

            // Read scales (12 bytes)
            byte[] scales = new byte[12];
            buffer.get(scales);

            // Read d
            short dRaw = buffer.getShort();
            float d = fp16ToFloat(dRaw);

            // Dequantize (simplified)
            for (int i = 0; i < 256 && outputIdx < totalElements; i++) {
                int qsIdx = i / 4;
                int qsShift = (i % 4) * 2;
                int lowBits = (qs[qsIdx] >> qsShift) & 0x03;

                int hmaskIdx = i / 8;
                int hmaskShift = i % 8;
                int highBit = (hmask[hmaskIdx] >> hmaskShift) & 0x01;

                int val = lowBits | (highBit << 2);
                val -= 4; // Signed offset

                int scaleIdx = i / 16;
                float scale = d * ((scales[scaleIdx] & 0x3F) - 32);

                result[outputIdx++] = val * scale;
            }
        }

        return result;
    }

    @Override
    public QuantizationInfo extractQuantizationInfo(byte[] quantizedData, long[] shape) {
        long numElements = calculateNumElements(shape);
        int numBlocks = (int) ((numElements + BLOCK_SIZE - 1) / BLOCK_SIZE);

        return QuantizationInfo.builder()
                .quantType(GGMLDataType.GGML_TYPE_Q3_K)
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
