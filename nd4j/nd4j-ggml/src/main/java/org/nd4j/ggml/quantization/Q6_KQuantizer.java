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
import org.nd4j.linalg.api.ndarray.INDArray;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Quantizer for Q6_K format (k-quant).
 * This is the reverse operation of {@link Q6_KDequantizer}.
 *
 * Q6_K is a k-quant format with super-blocks:
 * - Super-block size: 256 elements
 * - Contains 16 sub-blocks of 16 elements each
 * - Per super-block: 2 FP16 scales + 16 bytes scales + 192 bytes data (6-bit per element)
 * - Total: 210 bytes per super-block
 */
public class Q6_KQuantizer implements Quantizer {

    private static final int BLOCK_SIZE = 256;
    private static final int BYTES_PER_BLOCK = 210;
    private static final int SUB_BLOCK_SIZE = 16;
    private static final int NUM_SUB_BLOCKS = 16;

    @Override
    public GGMLDataType getQuantType() {
        return GGMLDataType.GGML_TYPE_Q6_K;
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
    public byte[] quantize(float[] floatData) {
        int numBlocks = (floatData.length + BLOCK_SIZE - 1) / BLOCK_SIZE;
        byte[] output = new byte[numBlocks * BYTES_PER_BLOCK];
        ByteBuffer buffer = ByteBuffer.wrap(output).order(ByteOrder.LITTLE_ENDIAN);

        for (int block = 0; block < numBlocks; block++) {
            int blockStart = block * BLOCK_SIZE;

            // Calculate per-sub-block scales
            float[] subBlockScales = new float[NUM_SUB_BLOCKS];

            for (int sb = 0; sb < NUM_SUB_BLOCKS; sb++) {
                int start = blockStart + sb * SUB_BLOCK_SIZE;
                int end = Math.min(start + SUB_BLOCK_SIZE, floatData.length);

                float maxAbs = 0;
                for (int i = start; i < end; i++) {
                    maxAbs = Math.max(maxAbs, Math.abs(floatData[i]));
                }

                subBlockScales[sb] = maxAbs / 31.0f; // 6-bit signed range (-32 to 31)
                if (subBlockScales[sb] == 0) subBlockScales[sb] = 1.0f;
            }

            // Find super-block scale
            float maxScale = 0;
            for (int sb = 0; sb < NUM_SUB_BLOCKS; sb++) {
                maxScale = Math.max(maxScale, subBlockScales[sb]);
            }

            float d = maxScale / 127.0f; // 8-bit scale quantization
            if (d == 0) d = 1.0f;

            // Write d as FP16
            buffer.putShort(floatToFp16(d));

            // Quantize and write sub-block scales (8-bit each, 16 bytes)
            int[] quantizedScales = new int[NUM_SUB_BLOCKS];
            for (int sb = 0; sb < NUM_SUB_BLOCKS; sb++) {
                quantizedScales[sb] = Math.round(subBlockScales[sb] / d);
                quantizedScales[sb] = Math.max(0, Math.min(127, quantizedScales[sb]));
                buffer.put((byte) quantizedScales[sb]);
            }

            // Quantize data to 6-bit signed values
            int[] quantizedData = new int[BLOCK_SIZE];
            for (int sb = 0; sb < NUM_SUB_BLOCKS; sb++) {
                float scale = quantizedScales[sb] * d;
                float invScale = scale > 0 ? 1.0f / scale : 1.0f;

                for (int i = 0; i < SUB_BLOCK_SIZE; i++) {
                    int idx = blockStart + sb * SUB_BLOCK_SIZE + i;
                    float val = idx < floatData.length ? floatData[idx] : 0;
                    int q = Math.round(val * invScale);
                    // Store as 0-63 (add 32 to signed -32..31)
                    quantizedData[sb * SUB_BLOCK_SIZE + i] = Math.max(0, Math.min(63, q + 32));
                }
            }

            // Write quantized data (6-bit per value, 192 bytes for 256 values)
            // Pack 4 values (24 bits) into 3 bytes
            for (int i = 0; i < 64; i++) {
                int idx = i * 4;
                int v0 = quantizedData[idx] & 0x3F;
                int v1 = quantizedData[idx + 1] & 0x3F;
                int v2 = quantizedData[idx + 2] & 0x3F;
                int v3 = quantizedData[idx + 3] & 0x3F;

                // Pack 4 x 6-bit values into 3 bytes
                buffer.put((byte) (v0 | ((v1 & 0x03) << 6)));
                buffer.put((byte) ((v1 >> 2) | ((v2 & 0x0F) << 4)));
                buffer.put((byte) ((v2 >> 4) | (v3 << 2)));
            }
        }

        return output;
    }

    @Override
    public byte[] quantize(INDArray array) {
        return quantize(array.toFloatVector());
    }

    private static short floatToFp16(float value) {
        int bits = Float.floatToIntBits(value);
        int sign = (bits >> 16) & 0x8000;
        int exponent = ((bits >> 23) & 0xFF) - 127 + 15;
        int mantissa = bits & 0x7FFFFF;

        if (exponent <= 0) {
            if (exponent < -10) return (short) sign;
            mantissa |= 0x800000;
            mantissa = mantissa >> (1 - exponent);
            return (short) (sign | (mantissa >> 13));
        } else if (exponent >= 31) {
            if ((bits & 0x7FFFFFFF) > 0x7F800000) return (short) (sign | 0x7E00);
            return (short) (sign | 0x7C00);
        }
        return (short) (sign | (exponent << 10) | (mantissa >> 13));
    }
}
