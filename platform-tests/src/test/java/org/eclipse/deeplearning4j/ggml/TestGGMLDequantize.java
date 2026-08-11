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

package org.eclipse.deeplearning4j.ggml;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.GGMLDequantize;
import org.nd4j.linalg.factory.Nd4j;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the native ggml_dequantize op.
 * Validates that our standalone dequantization kernels correctly decode
 * GGML quantized data formats.
 */
@DisplayName("GGML Native Dequantize Op Test")
class TestGGMLDequantize {

    /**
     * Test Q4_0 dequantization with known data.
     * Q4_0 block: 2 bytes (FP16 d) + 16 bytes (4-bit quants) = 18 bytes per 32 elements.
     * Values are (nibble - 8) * d.
     */
    @Test
    @DisplayName("Q4_0 dequantize produces correct values")
    void testQ4_0Dequantize() {
        // Create a single Q4_0 block: d=1.0 (FP16), all nibbles = 8 (zero-centered)
        ByteBuffer buf = ByteBuffer.allocate(18).order(ByteOrder.LITTLE_ENDIAN);
        buf.putShort(fp32ToFp16(1.0f));  // d = 1.0
        // 16 bytes of quants: each byte has low nibble and high nibble
        // nibble value 8 -> (8-8)*1.0 = 0.0
        // nibble value 9 -> (9-8)*1.0 = 1.0
        // nibble value 7 -> (7-8)*1.0 = -1.0
        byte[] qs = new byte[16];
        qs[0] = (byte) ((9 << 4) | 8);   // elem 0 = 0.0, elem 1 = 1.0
        qs[1] = (byte) ((8 << 4) | 7);   // elem 2 = -1.0, elem 3 = 0.0
        for (int i = 2; i < 16; i++) {
            qs[i] = (byte) 0x88;          // nibble 8 -> all zeros
        }
        buf.put(qs);

        byte[] data = buf.array();
        INDArray rawBytes = Nd4j.createFromArray(data).castTo(DataType.INT8);

        GGMLDequantize op = new GGMLDequantize(rawBytes, GGMLDequantize.QUANT_Q4_0, new long[]{32});
        INDArray result = Nd4j.exec(op)[0];

        assertEquals(DataType.FLOAT, result.dataType());
        assertEquals(32, result.length());

        float[] values = result.toFloatVector();
        assertEquals(0.0f, values[0], 1e-5f, "elem 0 should be 0.0");
        assertEquals(1.0f, values[1], 1e-5f, "elem 1 should be 1.0");
        assertEquals(-1.0f, values[2], 1e-5f, "elem 2 should be -1.0");
        assertEquals(0.0f, values[3], 1e-5f, "elem 3 should be 0.0");

        // Rest should be zero
        for (int i = 4; i < 32; i++) {
            assertEquals(0.0f, values[i], 1e-5f, "elem " + i + " should be 0.0");
        }

        rawBytes.close();
        result.close();
    }

    /**
     * Test Q8_0 dequantization with known data.
     * Q8_0 block: 2 bytes (FP16 d) + 32 bytes (int8 quants) = 34 bytes per 32 elements.
     * Values are qs[i] * d.
     */
    @Test
    @DisplayName("Q8_0 dequantize produces correct values")
    void testQ8_0Dequantize() {
        ByteBuffer buf = ByteBuffer.allocate(34).order(ByteOrder.LITTLE_ENDIAN);
        buf.putShort(fp32ToFp16(0.5f));  // d = 0.5

        // int8 quants: [-128..127], value = qs * d
        byte[] qs = new byte[32];
        qs[0] = 2;    // 2 * 0.5 = 1.0
        qs[1] = -2;   // -2 * 0.5 = -1.0
        qs[2] = 0;    // 0 * 0.5 = 0.0
        qs[3] = 10;   // 10 * 0.5 = 5.0
        buf.put(qs);

        byte[] data = buf.array();
        INDArray rawBytes = Nd4j.createFromArray(data).castTo(DataType.INT8);

        GGMLDequantize op = new GGMLDequantize(rawBytes, GGMLDequantize.QUANT_Q8_0, new long[]{32});
        INDArray result = Nd4j.exec(op)[0];

        float[] values = result.toFloatVector();
        assertEquals(1.0f, values[0], 1e-3f);
        assertEquals(-1.0f, values[1], 1e-3f);
        assertEquals(0.0f, values[2], 1e-3f);
        assertEquals(5.0f, values[3], 1e-3f);

        rawBytes.close();
        result.close();
    }

    /**
     * Test that F16 output works — dequantize directly to HALF without intermediate F32.
     */
    @Test
    @DisplayName("Q8_0 dequantize to FLOAT16")
    void testQ8_0DequantizeToF16() {
        ByteBuffer buf = ByteBuffer.allocate(34).order(ByteOrder.LITTLE_ENDIAN);
        buf.putShort(fp32ToFp16(1.0f));
        byte[] qs = new byte[32];
        qs[0] = 3;   // 3.0
        qs[1] = -3;  // -3.0
        buf.put(qs);

        byte[] data = buf.array();
        INDArray rawBytes = Nd4j.createFromArray(data).castTo(DataType.INT8);

        GGMLDequantize op = new GGMLDequantize(rawBytes, GGMLDequantize.QUANT_Q8_0, DataType.HALF, new long[]{32});
        INDArray result = Nd4j.exec(op)[0];

        assertEquals(DataType.HALF, result.dataType());
        // Convert to float for comparison
        INDArray f32 = result.castTo(DataType.FLOAT);
        float[] values = f32.toFloatVector();
        assertEquals(3.0f, values[0], 0.1f);
        assertEquals(-3.0f, values[1], 0.1f);

        rawBytes.close();
        result.close();
        f32.close();
    }

    @Test
    @DisplayName("Direct unsupported FP8 output is rejected instead of silently using FLOAT32")
    void testDirectFp8OutputRejected() {
        INDArray rawBytes = Nd4j.zeros(DataType.INT8, 34);
        try {
            IllegalArgumentException exception = assertThrows(IllegalArgumentException.class,
                    () -> new GGMLDequantize(rawBytes, GGMLDequantize.QUANT_Q8_0,
                            DataType.FLOAT8, new long[]{32}));
            assertTrue(exception.getMessage().contains("only supports direct"));
        } finally {
            rawBytes.close();
        }
    }

    /**
     * Test Q4_K dequantization produces non-zero reasonable values.
     * Uses synthetic block data with known scale.
     */
    @Test
    @DisplayName("Q4_K dequantize produces non-zero values")
    void testQ4_KDequantize() {
        // Q4_K: 144 bytes per 256 elements
        // Layout: 2 bytes d + 2 bytes dmin + 12 bytes scales + 128 bytes qs
        ByteBuffer buf = ByteBuffer.allocate(144).order(ByteOrder.LITTLE_ENDIAN);
        buf.putShort(fp32ToFp16(0.1f));    // d = 0.1
        buf.putShort(fp32ToFp16(0.01f));   // dmin = 0.01

        // 12 bytes scales: simple pattern
        byte[] scales = new byte[12];
        scales[0] = 10;   // scale for sub-block 0
        scales[1] = 10;   // scale for sub-block 1
        scales[4] = 5;    // min for sub-block 0
        scales[5] = 5;    // min for sub-block 1
        buf.put(scales);

        // 128 bytes of 4-bit quants: all set to 0x55 (nibbles 5 and 5)
        byte[] qs = new byte[128];
        for (int i = 0; i < 128; i++) {
            qs[i] = 0x55;
        }
        buf.put(qs);

        byte[] data = buf.array();
        INDArray rawBytes = Nd4j.createFromArray(data).castTo(DataType.INT8);

        GGMLDequantize op = new GGMLDequantize(rawBytes, GGMLDequantize.QUANT_Q4_K, new long[]{256});
        INDArray result = Nd4j.exec(op)[0];

        assertEquals(256, result.length());
        float[] values = result.toFloatVector();

        // Verify non-zero output
        boolean hasNonZero = false;
        for (float v : values) {
            if (Math.abs(v) > 1e-10f) {
                hasNonZero = true;
                break;
            }
        }
        assertTrue(hasNonZero, "Q4_K dequantized values should be non-zero");

        // Values should be in reasonable range given d=0.1, scale=10 -> effective d=1.0
        // nibble 5 * 1.0 - dmin*min = 5.0 - 0.01*5 = 4.95
        assertEquals(4.95f, values[0], 0.1f, "First element should be ~4.95");

        rawBytes.close();
        result.close();
    }

    /**
     * Test multi-block dequantization with 2D output shape.
     */
    @Test
    @DisplayName("Q4_0 multi-block with 2D shape")
    void testQ4_0MultiBlock() {
        int numElements = 64;  // 2 blocks of 32
        int numBlocks = 2;
        ByteBuffer buf = ByteBuffer.allocate(numBlocks * 18).order(ByteOrder.LITTLE_ENDIAN);

        for (int b = 0; b < numBlocks; b++) {
            buf.putShort(fp32ToFp16(1.0f));
            byte[] qs = new byte[16];
            for (int i = 0; i < 16; i++) {
                qs[i] = (byte) 0x98;  // low=8->0, high=9->1
            }
            buf.put(qs);
        }

        byte[] data = buf.array();
        INDArray rawBytes = Nd4j.createFromArray(data).castTo(DataType.INT8);

        // 2D output shape
        GGMLDequantize op = new GGMLDequantize(rawBytes, GGMLDequantize.QUANT_Q4_0, new long[]{2, 32});
        INDArray result = Nd4j.exec(op)[0];

        assertEquals(2, result.rank());
        assertEquals(2, result.size(0));
        assertEquals(32, result.size(1));

        float[] values = result.reshape(64).toFloatVector();
        // Pattern: every even index = 0.0, every odd index = 1.0
        for (int i = 0; i < 64; i += 2) {
            assertEquals(0.0f, values[i], 1e-5f, "even elem " + i);
            assertEquals(1.0f, values[i + 1], 1e-5f, "odd elem " + (i + 1));
        }

        rawBytes.close();
        result.close();
    }

    /**
     * Convert float to FP16 (IEEE 754 half-precision).
     */
    private static short fp32ToFp16(float value) {
        int bits = Float.floatToIntBits(value);
        int sign = (bits >>> 16) & 0x8000;
        int exponent = ((bits >>> 23) & 0xFF) - 127 + 15;
        int mantissa = bits & 0x7FFFFF;

        if (exponent <= 0) {
            return (short) sign;
        } else if (exponent >= 31) {
            return (short) (sign | 0x7C00);
        }

        return (short) (sign | (exponent << 10) | (mantissa >> 13));
    }
}
