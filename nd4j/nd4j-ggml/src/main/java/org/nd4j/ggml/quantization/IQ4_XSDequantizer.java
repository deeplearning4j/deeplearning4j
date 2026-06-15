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
 * Dequantizer for IQ4_XS (4-bit importance quantization with extra scales).
 *
 * Super-block of 256 elements with per-sub-block 6-bit scales packed into
 * a uint16 bitmask, plus the IQ4_NL codebook for value lookup.
 *
 * Block layout (QK_K = 256):
 *   ggml_half d          — 2 bytes, super-block scale
 *   uint16_t scales_h    — 2 bytes, high bits of sub-block scales
 *   uint8_t scales_l[4]  — 4 bytes, low nibbles of sub-block scales
 *   uint8_t qs[128]      — 128 bytes, packed 4-bit codebook indices
 * Total: 136 bytes per block of 256 elements.
 */
public class IQ4_XSDequantizer implements Dequantizer {

    private static final int QK_K = 256;
    private static final int BYTES_PER_BLOCK = 136; // 2 + 2 + 4 + 128

    // From ggml-common.h: kvalues_iq4nl (same as IQ4_NL)
    private static final int[] KVALUES_IQ4NL = {
        -127, -104, -83, -65, -49, -35, -22, -10,
        1, 13, 25, 38, 53, 69, 89, 113
    };

    @Override
    public GGMLDataType getQuantType() {
        return GGMLDataType.GGML_TYPE_IQ4_XS;
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
            int scales_h = buffer.getShort() & 0xFFFF;
            byte[] scales_l = new byte[4];
            buffer.get(scales_l);
            byte[] qs = new byte[128];
            buffer.get(qs);

            // Decode 8 sub-block scales (6 bits each)
            int[] scales = new int[8];
            for (int i = 0; i < 4; i++) {
                scales[2 * i] = (scales_l[i] & 0x0F) | (((scales_h >> (2 * i)) & 3) << 4);
                scales[2 * i + 1] = ((scales_l[i] >> 4) & 0x0F) | (((scales_h >> (2 * i + 1)) & 3) << 4);
            }

            // Dequantize each sub-block of 32 elements
            for (int ib = 0; ib < 8; ib++) {
                float dl = d * (scales[ib] - 32);
                int qOff = ib * 16;
                for (int j = 0; j < 16; j++) {
                    int lo = qs[qOff + j] & 0x0F;
                    int hi = (qs[qOff + j] >> 4) & 0x0F;
                    int idx1 = outputIdx + ib * 32 + j;
                    int idx2 = outputIdx + ib * 32 + j + 16;
                    if (idx1 < totalElements) result[idx1] = dl * KVALUES_IQ4NL[lo];
                    if (idx2 < totalElements) result[idx2] = dl * KVALUES_IQ4NL[hi];
                }
            }

            outputIdx += QK_K;
        }

        return result;
    }

    @Override
    public QuantizationInfo extractQuantizationInfo(byte[] quantizedData, long[] shape) {
        long numElements = calculateNumElements(shape);
        int numBlocks = (int) ((numElements + QK_K - 1) / QK_K);
        return QuantizationInfo.builder()
                .quantType(GGMLDataType.GGML_TYPE_IQ4_XS)
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
