/*
 *  ******************************************************************************
 *
 *  This program and the accompanying materials are made available under the
 *  terms of the Apache License, Version 2.0 which is available at
 *  https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.eclipse.deeplearning4j.ggml;

import org.junit.jupiter.api.*;
import org.nd4j.ggml.format.GGMLDataType;
import org.nd4j.ggml.format.GGMLTensorInfo;
import org.nd4j.ggml.format.GGUFReader;
import org.nd4j.ggml.quantization.Q5_0Dequantizer;

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests Q5_0 dequantization specifically — Q5_0 is the dominant type (133/291 tensors)
 * in the q4_k_m.gguf model, making it the prime candidate for the garbage-output bug.
 *
 * Tests both synthetic (known-answer) and real-model-data correctness.
 *
 * Key issue under test:
 *   The high-bit extraction in Q5_0Dequantizer uses (highBits >> (i*2)) for element i*2+0
 *   and (highBits >> (i*2+1)) for element i*2+1.
 *   The GGML reference uses bit i for element i*2+0 and bit (i+16) for element i*2+1.
 *   These diverge as soon as i > 0.
 */
@DisplayName("Q5_0 Dequantizer Bug Reproduction Test")
class Q5_0DequantBugTest {

    private static final String Q4K_PATH = System.getProperty("user.home")
            + "/.kompile/models/chat/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    private static final String FP16_PATH = System.getProperty("user.home")
            + "/.kompile/models/chat/qwen2.5-0.5b-instruct-fp16.gguf";

    // ── Synthetic test: known-answer verification ─────────────────────────

    /**
     * Verify Q5_0 dequant with a hand-crafted block where we can compute expected values.
     *
     * Q5_0 block layout (22 bytes / 32 elements):
     *   2 bytes: fp16 scale d
     *   4 bytes: high bits (qh), one bit per element — first 16 elements in low 16 bits,
     *            next 16 elements in bits [16..31]
     *   16 bytes: low nibbles, 2 nibbles per byte (low nibble = element i*2, high nibble = element i*2+1)
     *
     * GGML formula for element j*2+0: val = (qs[j]&0xF | ((qh>>j)&1)<<4) - 16
     * GGML formula for element j*2+1: val = ((qs[j]>>4) | ((qh>>(j+16))&1)<<4) - 16
     */
    @Test
    @DisplayName("Q5_0 synthetic: high bit at position j=1 must use qh bit 1 (not bit 2)")
    void testQ5_0HighBitExtraction() {
        // Build a single Q5_0 block with d=1.0 and controlled high bits
        // We set qh so that bit 0 = 0, bit 1 = 1, bit 16 = 0, bit 17 = 1
        // So for j=0: element 0 high bit = 0, element 1 high bit = 0
        //    for j=1: element 2 high bit = 1, element 3 high bit = 1
        //    (element 3 gets bit j+16 = bit 17 = 1)

        ByteBuffer buf = ByteBuffer.allocate(22).order(ByteOrder.LITTLE_ENDIAN);
        buf.putShort(fp32ToFp16(1.0f));   // d = 1.0

        // qh = bit 1 set AND bit 17 set
        // bit 1 = high bit for elements [2,3] first part
        // bit 17 = high bit for element 3 (j=1, second element)
        int qh = (1 << 1) | (1 << 17);
        buf.putInt(qh);

        // 16 bytes of nibbles: set all to 0 (nibble value 0)
        // So low4 = 0 for all elements
        byte[] qs = new byte[16];
        buf.put(qs);

        byte[] data = buf.array();
        Q5_0Dequantizer dequant = new Q5_0Dequantizer();
        float[] result = dequant.dequantize(data, 32);

        // Expected values per GGML formula (d=1.0, low4=0):
        // j=0: elem[0] = (0 | ((qh>>0)&1)<<4) - 16 = (0|0) - 16 = -16  * 1.0 = -16
        //      elem[1] = (0 | ((qh>>16)&1)<<4) - 16 = (0|0) - 16 = -16  * 1.0 = -16
        // j=1: elem[2] = (0 | ((qh>>1)&1)<<4) - 16 = (0|16) - 16 = 0    * 1.0 = 0
        //      elem[3] = (0 | ((qh>>17)&1)<<4) - 16 = (0|16) - 16 = 0   * 1.0 = 0
        // j=2..15: elem[*] = (0 | 0<<4) - 16 = -16  * 1.0 = -16

        // With CORRECT element ordering (j and j+16):
        //   elem[j]    uses low nibble of qs[j] and qh bit j
        //   elem[j+16] uses high nibble of qs[j] and qh bit j+16
        // qh bit1=1 → elem[1]=0; qh bit17=1 → elem[17]=0; all others -16
        System.out.println("Q5_0 synthetic high-bit test results (d=1.0, low=0, qh bit1=1, bit17=1):");
        for (int i = 0; i < 20; i++) {
            System.out.printf("  [%2d] = %.1f%n", i, result[i]);
        }

        System.out.printf("Expected: elem[0]=-16, elem[1]=0, elem[2..15]=-16, elem[16]=-16, elem[17]=0, elem[18..31]=-16%n");

        assertEquals(-16.0f, result[0],  1e-4f, "elem[0]: j=0, bit 0 of qh is 0 → -16");
        assertEquals(  0.0f, result[1],  1e-4f, "elem[1]: j=1, bit 1 of qh is 1 → 0");
        for (int i = 2; i < 16; i++) {
            assertEquals(-16.0f, result[i], 1e-4f, "elem[" + i + "] should be -16 (qh bit unset)");
        }
        assertEquals(-16.0f, result[16], 1e-4f, "elem[16]: j=0, bit 16 of qh is 0 → -16");
        assertEquals(  0.0f, result[17], 1e-4f, "elem[17]: j=1, bit 17 of qh is 1 → 0");
        for (int i = 18; i < 32; i++) {
            assertEquals(-16.0f, result[i], 1e-4f, "elem[" + i + "] should be -16 (qh bit unset)");
        }
        System.out.println("PASS: Q5_0 high-bit extraction is correct.");
    }

    /**
     * Direct test of high-bit extraction at position j=15 (the boundary case).
     * qh bit 15 → first element; qh bit 31 → second element.
     */
    @Test
    @DisplayName("Q5_0 synthetic: high bit at j=15 uses qh bit 15 and bit 31")
    void testQ5_0HighBitBoundary() {
        ByteBuffer buf = ByteBuffer.allocate(22).order(ByteOrder.LITTLE_ENDIAN);
        buf.putShort(fp32ToFp16(1.0f));

        // Set bit 15 and bit 31 of qh
        int qh = (1 << 15) | (1 << 31);
        buf.putInt(qh);

        byte[] qs = new byte[16]; // all nibbles = 0
        buf.put(qs);

        byte[] data = buf.array();
        Q5_0Dequantizer dequant = new Q5_0Dequantizer();
        float[] result = dequant.dequantize(data, 32);

        // With CORRECT element ordering (j and j+16):
        // qh bit15=1 → elem[15]=0; qh bit31=1 → elem[31]=0
        System.out.println("Q5_0 boundary high-bit test (bit 15 and 31 set):");
        for (int i = 12; i < 32; i++) {
            System.out.printf("  [%2d] = %.1f%n", i, result[i]);
        }

        assertEquals(0.0f, result[15], 1e-4f, "elem[15]: j=15, bit 15 of qh → 1 → val=0");
        assertEquals(0.0f, result[31], 1e-4f, "elem[31]: j=15, bit 31 of qh → 1 → val=0");
        // elements 0-14 and 16-30 should be -16 (all other qh bits = 0)
        for (int i = 0; i < 15; i++) {
            assertEquals(-16.0f, result[i], 1e-4f, "elem[" + i + "] should be -16 (qh bit unset)");
        }
        for (int i = 16; i < 31; i++) {
            assertEquals(-16.0f, result[i], 1e-4f, "elem[" + i + "] should be -16 (qh bit unset)");
        }
        System.out.println("PASS: Q5_0 boundary case is correct.");
    }

    /**
     * Compare Java Q5_0 dequantizer against fp16 ground truth from real model.
     * This is the decisive end-to-end check.
     */
    @Test
    @DisplayName("Q5_0 real model: dequant matches fp16 ground truth (cosine > 0.99)")
    void testQ5_0RealModelMatchesFp16() throws Exception {
        File q4kFile = new File(Q4K_PATH);
        File fp16File = new File(FP16_PATH);

        if (!q4kFile.exists() || !fp16File.exists()) {
            System.out.println("SKIP: model files not found.");
            return;
        }

        GGMLTensorInfo q5_0Info = null;
        GGMLTensorInfo fp16Info = null;
        byte[] q5_0Bytes = null;
        float[] fp16Values = null;

        try (GGUFReader q4kReader = new GGUFReader(q4kFile);
             GGUFReader fp16Reader = new GGUFReader(fp16File)) {

            List<GGMLTensorInfo> q4kTensors = q4kReader.getMetadata().getTensors();
            List<GGMLTensorInfo> fp16Tensors = fp16Reader.getMetadata().getTensors();

            // Find a Q5_0 tensor in q4k file
            for (GGMLTensorInfo t : q4kTensors) {
                if (t.getDataType() == GGMLDataType.GGML_TYPE_Q5_0) {
                    for (GGMLTensorInfo tf : fp16Tensors) {
                        if (tf.getName().equals(t.getName()) &&
                                tf.getDataType() == GGMLDataType.GGML_TYPE_F16) {
                            q5_0Info = t;
                            fp16Info = tf;
                            break;
                        }
                    }
                    if (q5_0Info != null) break;
                }
            }

            if (q5_0Info == null) {
                System.out.println("SKIP: no matching Q5_0+F16 tensor pair found.");
                return;
            }

            System.out.println("Selected Q5_0 tensor: " + q5_0Info.getName()
                    + " shape=" + q5_0Info.getShapeString()
                    + " elements=" + q5_0Info.getNumElements());

            q5_0Bytes = q4kReader.readTensorData(q5_0Info);
            byte[] fp16RawBytes = fp16Reader.readTensorData(fp16Info);
            fp16Values = fp16BytesToFloat(fp16RawBytes, (int) fp16Info.getNumElements());
        }

        // Dequantize Q5_0 using Java dequantizer
        Q5_0Dequantizer dequant = new Q5_0Dequantizer();
        float[] q5_0Values = dequant.dequantize(q5_0Bytes, q5_0Info.getNumElements());

        // Compare first 64 elements
        System.out.println("First 32 elements (Q5_0 dequant vs fp16):");
        for (int i = 0; i < Math.min(32, q5_0Values.length); i++) {
            System.out.printf("  [%3d] q5_0=%.6f  fp16=%.6f  diff=%.6f%n",
                    i, q5_0Values[i], fp16Values[i], Math.abs(q5_0Values[i] - fp16Values[i]));
        }

        // Cosine similarity
        int limit = Math.min(q5_0Values.length, 4096);
        double dot = 0, n1 = 0, n2 = 0;
        for (int i = 0; i < limit; i++) {
            dot += (double) q5_0Values[i] * fp16Values[i];
            n1 += (double) q5_0Values[i] * q5_0Values[i];
            n2 += (double) fp16Values[i] * fp16Values[i];
        }
        double cosine = dot / (Math.sqrt(n1) * Math.sqrt(n2) + 1e-10);

        System.out.printf("%nCosine similarity over %d elements: %.6f%n", limit, cosine);

        assertTrue(cosine >= 0.99,
                String.format("Q5_0 cosine similarity %.6f < 0.99 — dequantization is WRONG. " +
                        "This would cause garbage output since 133/291 tensors use Q5_0.", cosine));

        System.out.println("PASS: Q5_0 dequant matches fp16 ground truth.");
    }

    // ── Utilities ─────────────────────────────────────────────────────────

    private static float[] fp16BytesToFloat(byte[] bytes, int numElements) {
        float[] result = new float[numElements];
        ByteBuffer bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < numElements && bb.remaining() >= 2; i++) {
            result[i] = fp16ToFloat(bb.getShort());
        }
        return result;
    }

    private static float fp16ToFloat(short fp16) {
        int h = fp16 & 0xFFFF;
        int sign = (h >> 15) & 0x1;
        int exp = (h >> 10) & 0x1F;
        int mant = h & 0x3FF;
        if (exp == 0) {
            if (mant == 0) return sign == 0 ? 0.0f : -0.0f;
            float v = mant / 1024.0f * (float) Math.pow(2, -14);
            return sign == 0 ? v : -v;
        } else if (exp == 31) {
            return mant == 0 ? (sign == 0 ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY) : Float.NaN;
        }
        float v = (1.0f + mant / 1024.0f) * (float) Math.pow(2, exp - 15);
        return sign == 0 ? v : -v;
    }

    private static short fp32ToFp16(float value) {
        int bits = Float.floatToIntBits(value);
        int sign = (bits >>> 16) & 0x8000;
        int exp = ((bits >>> 23) & 0xFF) - 127 + 15;
        int mant = bits & 0x7FFFFF;
        if (exp <= 0) return (short) sign;
        if (exp >= 31) return (short) (sign | 0x7C00);
        return (short) (sign | (exp << 10) | (mant >> 13));
    }
}
