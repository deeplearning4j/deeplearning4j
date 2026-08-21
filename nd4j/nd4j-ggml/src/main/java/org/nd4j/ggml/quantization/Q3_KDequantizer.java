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
            // Q3_K stores 32 high-bit bytes, 64 low-bit bytes, twelve packed
            // scale bytes, and one FP16 super-block scale.  The twelve scale
            // bytes encode sixteen signed (0..63, then -32) sub-block scales.
            byte[] hmask = new byte[32];
            buffer.get(hmask);
            byte[] qs = new byte[64];
            buffer.get(qs);
            byte[] packedScales = new byte[12];
            buffer.get(packedScales);

            short dRaw = buffer.getShort();
            float d = fp16ToFloat(dRaw);

            // This is the same unpacking used by ggml's dequantize_row_q3_K:
            // the low four bits of the first two words and the high two bits
            // packed in the third word form sixteen scale bytes.
            int aux0 = littleEndianInt(packedScales, 0);
            int aux1 = littleEndianInt(packedScales, 4);
            int aux2 = littleEndianInt(packedScales, 8);
            int tmp = aux2;
            int decoded0 = (aux0 & 0x0F0F0F0F) | (((tmp)       & 0x03030303) << 4);
            int decoded1 = (aux1 & 0x0F0F0F0F) | (((tmp >>> 2) & 0x03030303) << 4);
            int decoded2 = ((aux0 >>> 4) & 0x0F0F0F0F) | (((tmp >>> 4) & 0x03030303) << 4);
            int decoded3 = ((aux1 >>> 4) & 0x0F0F0F0F) | (((tmp >>> 6) & 0x03030303) << 4);
            int[] scales = {decoded0, decoded1, decoded2, decoded3};

            int scaleIndex = 0;
            int qOffset = 0;
            int highMask = 1;
            for (int half = 0; half < 2 && outputIdx < totalElements; half++) {
                int shift = 0;
                for (int group = 0; group < 4 && outputIdx < totalElements; group++) {
                    float scale = d * (scaleByte(scales, scaleIndex++) - 32);
                    for (int lane = 0; lane < 16 && outputIdx < totalElements; lane++) {
                        int lowBits = (qs[qOffset + lane] >>> shift) & 0x03;
                        int highBit = (hmask[lane] & highMask) != 0 ? 0 : 4;
                        result[outputIdx++] = scale * (lowBits - highBit);
                    }

                    scale = d * (scaleByte(scales, scaleIndex++) - 32);
                    for (int lane = 0; lane < 16 && outputIdx < totalElements; lane++) {
                        int lowBits = (qs[qOffset + 16 + lane] >>> shift) & 0x03;
                        int highBit = (hmask[16 + lane] & highMask) != 0 ? 0 : 4;
                        result[outputIdx++] = scale * (lowBits - highBit);
                    }

                    shift += 2;
                    highMask <<= 1;
                }
                qOffset += 32;
            }
        }

        return result;
    }

    private static int littleEndianInt(byte[] bytes, int offset) {
        return (bytes[offset] & 0xFF)
                | ((bytes[offset + 1] & 0xFF) << 8)
                | ((bytes[offset + 2] & 0xFF) << 16)
                | ((bytes[offset + 3] & 0xFF) << 24);
    }

    private static int scaleByte(int[] packedScales, int index) {
        return (packedScales[index >>> 2] >>> ((index & 3) * 8)) & 0xFF;
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
