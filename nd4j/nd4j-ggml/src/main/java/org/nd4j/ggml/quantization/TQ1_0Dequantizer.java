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
 * Dequantizer for TQ1_0 (1.6875-bit ternary quantization).
 *
 * Uses base-3 encoding: 5 trits per byte, each trit is {-1, 0, +1}.
 * Packs 256 elements into 54 ternary bytes + 2 bytes for scale.
 *
 * Block layout (QK_K = 256):
 *   uint8_t qs[54]       -- 54 bytes, base-3 encoded trits (5 trits/byte, ceil(256/5)=52 used bytes)
 *   ggml_half d          -- 2 bytes, super-block scale
 * Total: 56 bytes per block of 256 elements.
 *
 * Decoding: for each byte b, extract 5 trits via repeated div/mod 3:
 *   trit = (b / 3^i) % 3  for i in 0..4
 *   value = (trit - 1) * d   maps {0,1,2} -> {-1,0,+1}
 *
 * Note: Reference fallback. Native llamacpp path preferred.
 */
public class TQ1_0Dequantizer implements Dequantizer {

    private static final int QK_K = 256;
    // 54 bytes qs + 2 bytes d
    private static final int BYTES_PER_BLOCK = 56;

    @Override
    public GGMLDataType getQuantType() {
        return GGMLDataType.GGML_TYPE_TQ1_0;
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
            byte[] qs = new byte[54];
            buffer.get(qs);
            float d = fp16ToFloat(buffer.getShort());

            // Decode base-3 encoded trits: 5 trits per byte
            for (int i = 0; i < 52 && outputIdx < totalElements; i++) {
                int b = qs[i] & 0xFF;
                for (int t = 0; t < 5 && outputIdx < totalElements; t++) {
                    int trit = b % 3;
                    b /= 3;
                    // trit: 0 -> -1, 1 -> 0, 2 -> +1
                    result[outputIdx++] = (trit - 1) * d;
                }
            }
            // Remaining elements (256 - 52*5 = 256 - 260; actually we only use first 256)
            // The 52 bytes encode 260 trits; we only consume 256
            // Handle any leftover in block if outputIdx < block end
            int blockEnd = Math.min((block + 1) * QK_K, totalElements);
            while (outputIdx < blockEnd) {
                result[outputIdx++] = 0.0f;
            }
        }

        return result;
    }

    @Override
    public QuantizationInfo extractQuantizationInfo(byte[] quantizedData, long[] shape) {
        long numElements = calculateNumElements(shape);
        int numBlocks = (int) ((numElements + QK_K - 1) / QK_K);
        return QuantizationInfo.builder()
                .quantType(GGMLDataType.GGML_TYPE_TQ1_0)
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
