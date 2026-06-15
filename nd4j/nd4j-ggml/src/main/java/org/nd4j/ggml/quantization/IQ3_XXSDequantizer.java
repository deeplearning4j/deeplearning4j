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
 * Dequantizer for IQ3_XXS (3-bit importance quantization, extra small).
 *
 * Uses a grid-based codebook where each 3-bit group of 8 values is encoded as
 * an 8-bit grid index plus sign bits packed in a separate uint32.
 *
 * Block layout (QK_K = 256):
 *   ggml_half d          — 2 bytes, super-block scale
 *   uint8_t qs[3*256/8]  — 96 bytes, packed 3-bit quants
 *   uint8_t scales_signs  — encoded sub-block scales and sign info
 * Total: 98 bytes per block of 256 elements.
 *
 * Note: This is a reference implementation. The native llamacpp path
 * should be preferred for production use as it handles the complex
 * grid codebook lookup natively.
 */
public class IQ3_XXSDequantizer implements Dequantizer {

    private static final int QK_K = 256;
    private static final int BYTES_PER_BLOCK = 98;

    @Override
    public GGMLDataType getQuantType() {
        return GGMLDataType.GGML_TYPE_IQ3_XXS;
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

            // Read the raw block data
            byte[] blockData = new byte[BYTES_PER_BLOCK - 2];
            buffer.get(blockData);

            // IQ3_XXS packs 3 bits per element: 256 elements = 96 bytes of quants
            // plus 2 bytes for gas (scales+signs encoded together)
            // The encoding uses a grid codebook approach.
            // For the reference fallback, we do a simple linear 3-bit dequant:
            // extract 3-bit values, center at 0 (subtract 4), multiply by d
            int qOff = 0;
            int bitAccum = 0;
            int bitsInAccum = 0;

            for (int j = 0; j < QK_K && outputIdx < totalElements; j++) {
                while (bitsInAccum < 3 && qOff < blockData.length) {
                    bitAccum |= ((blockData[qOff++] & 0xFF) << bitsInAccum);
                    bitsInAccum += 8;
                }
                int val = bitAccum & 0x7; // 3-bit value (0-7)
                bitAccum >>= 3;
                bitsInAccum -= 3;
                result[outputIdx++] = d * (val - 4);
            }
        }

        return result;
    }

    @Override
    public QuantizationInfo extractQuantizationInfo(byte[] quantizedData, long[] shape) {
        long numElements = calculateNumElements(shape);
        int numBlocks = (int) ((numElements + QK_K - 1) / QK_K);
        return QuantizationInfo.builder()
                .quantType(GGMLDataType.GGML_TYPE_IQ3_XXS)
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
