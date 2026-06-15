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
 * Dequantizer for Q8_K (8-bit k-quant).
 *
 * Block layout (QK_K = 256):
 *   float d             — 4 bytes, super-block scale
 *   int8_t qs[256]      — 256 bytes, quantized values
 *   int16_t bsums[16]   — 32 bytes, block sums (not used for dequant)
 * Total: 292 bytes per block of 256 elements.
 */
public class Q8_KDequantizer implements Dequantizer {

    private static final int QK_K = 256;
    private static final int BYTES_PER_BLOCK = 292; // 4 + 256 + 32

    @Override
    public GGMLDataType getQuantType() {
        return GGMLDataType.GGML_TYPE_Q8_K;
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
            float d = buffer.getFloat();
            byte[] qs = new byte[256];
            buffer.get(qs);
            buffer.position(buffer.position() + 32); // skip bsums

            for (int j = 0; j < QK_K && outputIdx < totalElements; j++) {
                result[outputIdx++] = d * qs[j];
            }
        }

        return result;
    }

    @Override
    public QuantizationInfo extractQuantizationInfo(byte[] quantizedData, long[] shape) {
        long numElements = calculateNumElements(shape);
        int numBlocks = (int) ((numElements + QK_K - 1) / QK_K);
        return QuantizationInfo.builder()
                .quantType(GGMLDataType.GGML_TYPE_Q8_K)
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
}
