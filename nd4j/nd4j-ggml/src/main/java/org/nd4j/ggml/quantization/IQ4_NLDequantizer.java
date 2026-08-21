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
 * Dequantizer for IQ4_NL (non-linear 4-bit importance quantization).
 *
 * Uses a hardcoded 16-entry lookup table (kvalues_iq4nl) instead of
 * linear scale+offset. Each 4-bit index maps to a specific float value.
 *
 * Block layout (QK4_NL = 32):
 *   ggml_half d        — 2 bytes, block scale
 *   uint8_t qs[16]     — 16 bytes, packed 4-bit indices (2 per byte)
 * Total: 18 bytes per block of 32 elements.  The low nibbles encode the
 * first 16 values and the high nibbles encode the second 16 values.
 */
public class IQ4_NLDequantizer implements Dequantizer {

    private static final int QK4_NL = 32;
    private static final int BYTES_PER_BLOCK = 18; // 2 + 16

    // From ggml-common.h: kvalues_iq4nl
    private static final int[] KVALUES_IQ4NL = {
        -127, -104, -83, -65, -49, -35, -22, -10,
        1, 13, 25, 38, 53, 69, 89, 113
    };

    @Override
    public GGMLDataType getQuantType() {
        return GGMLDataType.GGML_TYPE_IQ4_NL;
    }

    @Override
    public int getBlockSize() {
        return QK4_NL;
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

        int numBlocks = (totalElements + QK4_NL - 1) / QK4_NL;

        for (int block = 0; block < numBlocks && buffer.remaining() >= BYTES_PER_BLOCK; block++) {
            float d = fp16ToFloat(buffer.getShort());
            byte[] qs = new byte[16];
            buffer.get(qs);

            // GGML stores each block's low nibbles in values [0, 15] and
            // high nibbles in values [16, 31]; they are not interleaved.
            int blockOutput = block * QK4_NL;
            for (int j = 0; j < 16; j++) {
                int lo = qs[j] & 0x0F;
                int hi = (qs[j] >> 4) & 0x0F;
                int lowIndex = blockOutput + j;
                int highIndex = blockOutput + j + 16;
                if (lowIndex < totalElements) result[lowIndex] = d * KVALUES_IQ4NL[lo];
                if (highIndex < totalElements) result[highIndex] = d * KVALUES_IQ4NL[hi];
            }
        }

        return result;
    }

    @Override
    public QuantizationInfo extractQuantizationInfo(byte[] quantizedData, long[] shape) {
        long numElements = calculateNumElements(shape);
        int numBlocks = (int) ((numElements + QK4_NL - 1) / QK4_NL);
        return QuantizationInfo.builder()
                .quantType(GGMLDataType.GGML_TYPE_IQ4_NL)
                .blockSize(QK4_NL)
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
