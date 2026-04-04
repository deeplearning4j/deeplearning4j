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
 * Dequantizer for Q4_K quantization format.
 *
 * Q4_K is a k-quant format with super-blocks of 256 elements.
 *
 * Block layout (from ggml-common.h):
 * <pre>
 *   ggml_half d         — super-block scale for quantized scales (2 bytes)
 *   ggml_half dmin      — super-block scale for quantized mins (2 bytes)
 *   uint8_t scales[12]  — scales and mins, quantized with 6 bits
 *   uint8_t qs[128]     — 4-bit quants (256 / 2)
 * </pre>
 * Total: 144 bytes per block of 256 elements.
 *
 * Dequantization follows ggml's dequantize_row_q4_K exactly.
 */
public class Q4_KDequantizer implements Dequantizer {

    private static final int QK_K = 256;
    private static final int BYTES_PER_BLOCK = 144; // 2 + 2 + 12 + 128

    @Override
    public GGMLDataType getQuantType() {
        return GGMLDataType.GGML_TYPE_Q4_K;
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

        byte[] scaleBytes = new byte[12];
        byte[] qs = new byte[128];

        for (int block = 0; block < numBlocks && buffer.remaining() >= BYTES_PER_BLOCK; block++) {
            // Read d and dmin (FP16 super-block scales)
            short dRaw = buffer.getShort();
            short dminRaw = buffer.getShort();
            float d = fp16ToFloat(dRaw);
            float min = fp16ToFloat(dminRaw);

            // Read 12 bytes of packed scales
            buffer.get(scaleBytes);

            // Read 128 bytes of 4-bit quantized data
            buffer.get(qs);

            // Dequantize following ggml's dequantize_row_q4_K exactly
            int is = 0;
            int qIdx = 0;
            for (int j = 0; j < QK_K; j += 64) {
                // get_scale_min_k4 for is+0
                int sc1 = getScale(is, scaleBytes);
                int m1 = getMin(is, scaleBytes);
                float d1 = d * sc1;
                float m1f = min * m1;

                // get_scale_min_k4 for is+1
                int sc2 = getScale(is + 1, scaleBytes);
                int m2 = getMin(is + 1, scaleBytes);
                float d2 = d * sc2;
                float m2f = min * m2;

                // First 32: low nibbles with scale d1
                for (int l = 0; l < 32; l++) {
                    int val = qs[qIdx + l] & 0x0F;
                    if (outputIdx < totalElements) {
                        result[outputIdx++] = d1 * val - m1f;
                    }
                }
                // Next 32: high nibbles with scale d2
                for (int l = 0; l < 32; l++) {
                    int val = (qs[qIdx + l] >> 4) & 0x0F;
                    if (outputIdx < totalElements) {
                        result[outputIdx++] = d2 * val - m2f;
                    }
                }
                qIdx += 32;
                is += 2;
            }
        }

        return result;
    }

    /**
     * Extract scale value for sub-block j from the packed 12-byte scales array.
     * Matches ggml's get_scale_min_k4 for the 'd' (scale) output.
     */
    private static int getScale(int j, byte[] q) {
        if (j < 4) {
            return q[j] & 63;
        } else {
            return ((q[j + 4] & 0xFF) & 0xF) | ((((q[j - 4] & 0xFF) >> 6) & 3) << 4);
        }
    }

    /**
     * Extract min value for sub-block j from the packed 12-byte scales array.
     * Matches ggml's get_scale_min_k4 for the 'm' (min) output.
     */
    private static int getMin(int j, byte[] q) {
        if (j < 4) {
            return q[j + 4] & 63;
        } else {
            return (((q[j + 4] & 0xFF) >> 4) & 0xF) | ((((q[j] & 0xFF) >> 6) & 3) << 4);
        }
    }

    @Override
    public QuantizationInfo extractQuantizationInfo(byte[] quantizedData, long[] shape) {
        long numElements = calculateNumElements(shape);
        int numBlocks = (int) ((numElements + QK_K - 1) / QK_K);

        return QuantizationInfo.builder()
                .quantType(GGMLDataType.GGML_TYPE_Q4_K)
                .blockSize(QK_K)
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
