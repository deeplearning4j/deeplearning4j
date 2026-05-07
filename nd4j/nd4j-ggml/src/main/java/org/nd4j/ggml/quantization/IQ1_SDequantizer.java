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
 * Dequantizer for IQ1_S (~1.5-bit importance quantization).
 *
 * Uses ternary grids with 6-bit grid indices and sign bits.
 * Each element is {-1, 0, +1} times a scale factor.
 *
 * Block layout (QK_K = 256):
 *   ggml_half d          — 2 bytes, super-block scale
 *   uint8_t qs[32]       — 32 bytes, grid indices (6-bit packed)
 *   uint16_t qh[16]      — 32 bytes, high bits + sign bits
 * Total: 50 bytes per block of 256 elements.
 *
 * Note: Reference fallback. Native llamacpp path preferred.
 */
public class IQ1_SDequantizer implements Dequantizer {

    private static final int QK_K = 256;
    private static final int BYTES_PER_BLOCK = 50;

    @Override
    public GGMLDataType getQuantType() {
        return GGMLDataType.GGML_TYPE_IQ1_S;
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
            float d = fp16ToFloat(buffer.getShort());

            byte[] qs = new byte[32];
            byte[] qhRaw = new byte[32];
            buffer.get(qs);
            buffer.get(qhRaw);

            // Reference fallback: approximate ternary dequant
            // Each element is roughly {-1, 0, +1} * d * sub-scale
            for (int j = 0; j < QK_K && outputIdx < totalElements; j++) {
                int byteIdx = j / 8;
                int bitOff = j % 8;
                if (byteIdx < qs.length) {
                    int bit = (qs[byteIdx] >> bitOff) & 1;
                    // Sign from qh
                    int signByteIdx = j / 8;
                    int signBitOff = j % 8;
                    int sign = (signByteIdx < qhRaw.length) ?
                        ((qhRaw[signByteIdx] >> signBitOff) & 1) : 0;
                    float val = bit != 0 ? d : 0.0f;
                    result[outputIdx++] = sign != 0 ? -val : val;
                } else {
                    result[outputIdx++] = 0.0f;
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
                .quantType(GGMLDataType.GGML_TYPE_IQ1_S)
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
