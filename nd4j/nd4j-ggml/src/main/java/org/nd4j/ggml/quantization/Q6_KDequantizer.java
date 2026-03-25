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
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Dequantizer for Q6_K quantization format.
 *
 * Q6_K is a 6-bit k-quant format with super-blocks of 256 elements.
 *
 * Block layout (from ggml-common.h):
 * <pre>
 *   uint8_t ql[128]   — lower 4 bits of each quant value
 *   uint8_t qh[64]    — upper 2 bits of each quant value
 *   int8_t  scales[16] — per-sub-block scales (signed 8-bit)
 *   ggml_half d        — super-block scale (FP16)
 * </pre>
 * Total: 210 bytes per block of 256 elements.
 *
 * Dequantization follows ggml's dequantize_row_q6_K exactly.
 */
public class Q6_KDequantizer implements Dequantizer {

    private static final int QK_K = 256;
    private static final int BYTES_PER_BLOCK = 210; // 128 + 64 + 16 + 2

    @Override
    public GGMLDataType getQuantType() {
        return GGMLDataType.GGML_TYPE_Q6_K;
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

        byte[] ql = new byte[128];
        byte[] qh = new byte[64];
        byte[] scales = new byte[16];

        for (int block = 0; block < numBlocks && buffer.remaining() >= BYTES_PER_BLOCK; block++) {
            // Read fields in the correct GGUF order: ql, qh, scales, d
            buffer.get(ql);      // 128 bytes: lower 4 bits of quants
            buffer.get(qh);      //  64 bytes: upper 2 bits of quants
            buffer.get(scales);  //  16 bytes: signed 8-bit scales
            short dRaw = buffer.getShort(); // 2 bytes: FP16 super-block scale
            float d = fp16ToFloat(dRaw);

            // Dequantize following ggml's dequantize_row_q6_K exactly:
            // Process 2 groups of 128 elements each
            int qlOff = 0;
            int qhOff = 0;
            int scOff = 0;

            for (int n = 0; n < QK_K; n += 128) {
                for (int l = 0; l < 32; l++) {
                    int is = l / 16;

                    // Reconstruct 6-bit values from split ql (lower 4) and qh (upper 2)
                    int q1 = ((ql[qlOff + l] & 0xF) | (((qh[qhOff + l] >> 0) & 3) << 4)) - 32;
                    int q2 = ((ql[qlOff + l + 32] & 0xF) | (((qh[qhOff + l] >> 2) & 3) << 4)) - 32;
                    int q3 = (((ql[qlOff + l] >> 4) & 0xF) | (((qh[qhOff + l] >> 4) & 3) << 4)) - 32;
                    int q4 = (((ql[qlOff + l + 32] >> 4) & 0xF) | (((qh[qhOff + l] >> 6) & 3) << 4)) - 32;

                    int idx = outputIdx + n + l;
                    if (idx < totalElements) result[idx] = d * scales[scOff + is] * q1;
                    idx = outputIdx + n + l + 32;
                    if (idx < totalElements) result[idx] = d * scales[scOff + is + 2] * q2;
                    idx = outputIdx + n + l + 64;
                    if (idx < totalElements) result[idx] = d * scales[scOff + is + 4] * q3;
                    idx = outputIdx + n + l + 96;
                    if (idx < totalElements) result[idx] = d * scales[scOff + is + 6] * q4;
                }
                qlOff += 64;
                qhOff += 32;
                scOff += 8;
            }

            outputIdx += QK_K;
        }

        return result;
    }

    @Override
    public INDArray dequantizeToArray(byte[] quantizedData, long[] shape, DataType targetType) {
        long numElements = calculateNumElements(shape);
        float[] floatData = dequantize(quantizedData, numElements);

        INDArray array = Nd4j.create(floatData, shape);
        if (targetType != DataType.FLOAT) {
            array = array.castTo(targetType);
        }
        return array;
    }

    @Override
    public QuantizationInfo extractQuantizationInfo(byte[] quantizedData, long[] shape) {
        long numElements = calculateNumElements(shape);
        int numBlocks = (int) ((numElements + QK_K - 1) / QK_K);

        return QuantizationInfo.builder()
                .quantType(GGMLDataType.GGML_TYPE_Q6_K)
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
