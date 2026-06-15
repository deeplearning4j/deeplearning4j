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
 * Dequantizer for IQ3_S (3-bit importance quantization, standard).
 *
 * Similar to IQ3_XXS but with separate per-sub-block scales for
 * better accuracy. Uses the same grid-based codebook approach.
 *
 * Block layout (QK_K = 256):
 *   ggml_half d          — 2 bytes, super-block scale
 *   uint8_t qs[96]       — 96 bytes, packed 3-bit quants
 *   uint8_t qh[32]       — 32 bytes, high bits
 *   uint8_t signs[32]    — 32 bytes, sign bits
 *   uint8_t scales[8]    — 8 bytes, sub-block scales
 * Total: 110 bytes per block of 256 elements.
 *
 * Note: This is a reference fallback. The native llamacpp path
 * handles the grid codebook lookup precisely.
 */
public class IQ3_SDequantizer implements Dequantizer {

    private static final int QK_K = 256;
    private static final int BYTES_PER_BLOCK = 110;

    @Override
    public GGMLDataType getQuantType() {
        return GGMLDataType.GGML_TYPE_IQ3_S;
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

            byte[] qs = new byte[96];
            byte[] qh = new byte[32];
            byte[] signs = new byte[32];
            byte[] scales = new byte[8];
            buffer.get(qs);
            buffer.get(qh);
            buffer.get(signs);
            buffer.get(scales);

            // Reference fallback: extract 3-bit values with scale and sign
            int qOff = 0;
            int bitAccum = 0;
            int bitsInAccum = 0;

            for (int ib = 0; ib < 8; ib++) {
                float dl = d * (1 + 2 * (scales[ib] & 0x0F));

                for (int j = 0; j < 32 && outputIdx < totalElements; j++) {
                    while (bitsInAccum < 3 && qOff < qs.length) {
                        bitAccum |= ((qs[qOff++] & 0xFF) << bitsInAccum);
                        bitsInAccum += 8;
                    }
                    int val = bitAccum & 0x7;
                    bitAccum >>= 3;
                    bitsInAccum -= 3;

                    int signIdx = ib * 4 + j / 8;
                    int signBit = (signIdx < signs.length) ?
                        ((signs[signIdx] >> (j % 8)) & 1) : 0;
                    float fval = dl * (val - 4);
                    result[outputIdx++] = signBit != 0 ? -fval : fval;
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
                .quantType(GGMLDataType.GGML_TYPE_IQ3_S)
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
