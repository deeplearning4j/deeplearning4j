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
 * Dequantizer for TQ2_0 (2.0625-bit ternary quantization).
 *
 * Uses 2-bit encoding for ternary values {-1, 0, +1}.
 * Each byte stores 4 ternary values in 2-bit pairs.
 * Mapping: 0b00 -> +1, 0b01 -> 0, 0b10 -> -1, 0b11 -> 0 (unused/padding)
 *
 * Block layout (QK_K = 256):
 *   uint8_t qs[64]       -- 64 bytes, 2-bit encoded ternary (4 values/byte)
 *   ggml_half d          -- 2 bytes, super-block scale
 * Total: 66 bytes per block of 256 elements.
 *
 * Note: Reference fallback. Native llamacpp path preferred.
 */
public class TQ2_0Dequantizer implements Dequantizer {

    private static final int QK_K = 256;
    // 64 bytes qs + 2 bytes d
    private static final int BYTES_PER_BLOCK = 66;

    @Override
    public GGMLDataType getQuantType() {
        return GGMLDataType.GGML_TYPE_TQ2_0;
    }

    @Override
    public int getBlockSize() {
        return QK_K;
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

        int numBlocks = (totalElements + QK_K - 1) / QK_K;
        int outputIdx = 0;

        for (int block = 0; block < numBlocks && buffer.remaining() >= BYTES_PER_BLOCK; block++) {
            byte[] qs = new byte[64];
            buffer.get(qs);
            float d = fp16ToFloat(buffer.getShort());

            // Decode 2-bit ternary: 4 values per byte
            for (int i = 0; i < 64 && outputIdx < totalElements; i++) {
                int b = qs[i] & 0xFF;
                for (int t = 0; t < 4 && outputIdx < totalElements; t++) {
                    int val = (b >> (t * 2)) & 0x3;
                    // 0b00=+1, 0b01=0, 0b10=-1, 0b11=0
                    float fval;
                    switch (val) {
                        case 0: fval = d; break;
                        case 2: fval = -d; break;
                        default: fval = 0.0f; break;
                    }
                    result[outputIdx++] = fval;
                }
            }
        }

        return result;
    }

    @Override
    public QuantizationInfo extractQuantizationInfo(byte[] quantizedData, long[] shape) {
        long numElements = calculateNumElements(shape);
        int numBlocks = (int) ((numElements + QK_K - 1) / QK_K);
        return QuantizationInfo.builder()
                .quantType(GGMLDataType.GGML_TYPE_TQ2_0)
                .blockSize(QK_K)
                .numBlocks(numBlocks)
                .originalShape(shape)
                .build();
    }

    private static long calculateNumElements(long[] shape) {
        long total = 1;
        for (long dim : shape) total *= dim;
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
