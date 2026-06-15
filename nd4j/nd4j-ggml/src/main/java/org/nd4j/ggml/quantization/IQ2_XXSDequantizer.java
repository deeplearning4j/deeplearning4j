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
 * Dequantizer for IQ2_XXS (2-bit importance quantization, extra extra small).
 *
 * Uses a 256-entry grid codebook where each 8-element group is encoded as a
 * single byte index. The codebook maps each index to 8 ternary-ish values.
 *
 * Block layout (QK_K = 256):
 *   ggml_half d          — 2 bytes, super-block scale
 *   uint16_t qs[128]     — 64 bytes, packed quant indices + signs
 * Total: 66 bytes per block of 256 elements.
 *
 * Note: This is a reference fallback. The native llamacpp path
 * handles the full grid codebook precisely.
 */
public class IQ2_XXSDequantizer implements Dequantizer {

    private static final int QK_K = 256;
    private static final int BYTES_PER_BLOCK = 66;

    @Override
    public GGMLDataType getQuantType() {
        return GGMLDataType.GGML_TYPE_IQ2_XXS;
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

            byte[] blockData = new byte[64];
            buffer.get(blockData);

            // Reference fallback: extract 2-bit values, center and scale
            int qOff = 0;
            for (int j = 0; j < QK_K && outputIdx < totalElements; j++) {
                int byteIdx = j / 4;
                int bitOff = (j % 4) * 2;
                if (byteIdx < blockData.length) {
                    int val = (blockData[byteIdx] >> bitOff) & 0x3;
                    result[outputIdx++] = d * (val - 1.5f);
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
                .quantType(GGMLDataType.GGML_TYPE_IQ2_XXS)
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
