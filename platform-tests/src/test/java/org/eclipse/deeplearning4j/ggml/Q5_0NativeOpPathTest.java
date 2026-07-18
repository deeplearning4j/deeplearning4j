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
import org.nd4j.ggml.quantization.DequantizerFactory;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests the PRODUCTION code path: DequantizerFactory.dequantizeToArray for Q5_0.
 * This is what GGMLToSameDiffConverter actually calls during model loading.
 *
 * The production path tries the native ggml_dequantize op first.
 * If the native op fails, it falls back to the Java Q5_0Dequantizer (which has a bug).
 *
 * This test determines which path is active and whether the result is correct.
 */
@DisplayName("Q5_0 Production Path (DequantizerFactory) Test")
class Q5_0NativeOpPathTest {

    private static final String Q4K_PATH = System.getProperty("user.home")
            + "/.kompile/models/chat/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    private static final String FP16_PATH = System.getProperty("user.home")
            + "/.kompile/models/chat/qwen2.5-0.5b-instruct-fp16.gguf";

    /**
     * Test the exact production path: DequantizerFactory.dequantizeToArray.
     * This calls the native GGMLDequantize op first, falls back to Java on failure.
     */
    @Test
    @DisplayName("Production path (DequantizerFactory.dequantizeToArray) for Q5_0")
    void testProductionPathQ5_0() throws Exception {
        File q4kFile = new File(Q4K_PATH);
        File fp16File = new File(FP16_PATH);

        if (!q4kFile.exists() || !fp16File.exists()) {
            System.out.println("SKIP: model files not found.");
            return;
        }

        GGMLTensorInfo q5_0Info = null;
        byte[] q5_0Bytes = null;
        float[] fp16Values = null;
        long[] ggufShape = null;

        try (GGUFReader q4kReader = new GGUFReader(q4kFile);
             GGUFReader fp16Reader = new GGUFReader(fp16File)) {

            List<GGMLTensorInfo> q4kTensors = q4kReader.getMetadata().getTensors();
            List<GGMLTensorInfo> fp16Tensors = fp16Reader.getMetadata().getTensors();

            for (GGMLTensorInfo t : q4kTensors) {
                if (t.getDataType() == GGMLDataType.GGML_TYPE_Q5_0) {
                    for (GGMLTensorInfo tf : fp16Tensors) {
                        if (tf.getName().equals(t.getName()) &&
                                tf.getDataType() == GGMLDataType.GGML_TYPE_F16) {
                            q5_0Info = t;
                            q5_0Bytes = q4kReader.readTensorData(t);
                            byte[] fp16Raw = fp16Reader.readTensorData(tf);
                            fp16Values = fp16BytesToFloat(fp16Raw, (int) tf.getNumElements());
                            ggufShape = t.getShape();
                            break;
                        }
                    }
                    if (q5_0Info != null) break;
                }
            }
        }

        if (q5_0Info == null) {
            System.out.println("SKIP: no Q5_0 tensor found.");
            return;
        }

        System.out.println("Testing tensor: " + q5_0Info.getName() + " shape=" + q5_0Info.getShapeString());

        // ── Compute ND4J shape exactly as GGMLToSameDiffConverter does ─────
        // reverseShape() reverses GGUF column-major shape to ND4J row-major
        long[] nd4jShape = reverseShape(ggufShape);
        System.out.println("GGUF shape (col-major): " + java.util.Arrays.toString(ggufShape));
        System.out.println("ND4J shape (row-major): " + java.util.Arrays.toString(nd4jShape));

        // ── Call production path: DequantizerFactory.dequantizeToArray ──────
        System.out.println("Calling DequantizerFactory.dequantizeToArray (production path)...");
        INDArray result = DequantizerFactory.dequantizeToArray(
                q5_0Bytes, GGMLDataType.GGML_TYPE_Q5_0, nd4jShape, DataType.FLOAT);

        assertNotNull(result, "Result should not be null");
        System.out.println("Result shape: " + java.util.Arrays.toString(result.shape()));
        System.out.println("Result dtype: " + result.dataType());
        System.out.println("Result length: " + result.length());

        float[] resultVals = result.toFloatVector();

        // Print first 32 values
        System.out.println("First 32 values (production path vs fp16):");
        for (int i = 0; i < Math.min(32, resultVals.length); i++) {
            System.out.printf("  [%3d] prod=%.6f  fp16=%.6f  diff=%.6f%n",
                    i, resultVals[i], fp16Values[i], Math.abs(resultVals[i] - fp16Values[i]));
        }

        // Cosine similarity
        int limit = Math.min(resultVals.length, 4096);
        double dot = 0, n1 = 0, n2 = 0;
        for (int i = 0; i < limit; i++) {
            dot += (double) resultVals[i] * fp16Values[i];
            n1 += (double) resultVals[i] * resultVals[i];
            n2 += (double) fp16Values[i] * fp16Values[i];
        }
        double cosine = dot / (Math.sqrt(n1) * Math.sqrt(n2) + 1e-10);
        System.out.printf("Production path cosine similarity: %.6f%n", cosine);

        if (cosine >= 0.99) {
            System.out.println("PASS: production path (native op) is CORRECT.");
        } else {
            System.out.println("FAIL: production path cosine=" + cosine + " < 0.99 — WRONG dequant in production.");
            System.out.println("  This means the native ggml_dequantize op ALSO fails/falls back to broken Java.");
        }

        assertTrue(cosine >= 0.99,
                String.format("Production path Q5_0 cosine %.6f < 0.99. " +
                        "Native op likely failing and falling back to broken Java dequantizer.", cosine));

        result.close();
    }

    /**
     * Synthetic test: DequantizerFactory with a known-good Q5_0 block.
     * Verifies the native op handles the high-bit layout correctly.
     */
    @Test
    @DisplayName("Production path synthetic: Q5_0 high-bit at j=1 (native op vs Java fallback)")
    void testProductionPathSynthetic() {
        // Same synthetic block as Q5_0DequantBugTest.testQ5_0HighBitExtraction
        // qh bit1=1, bit17=1; all nibbles=0; d=1.0
        ByteBuffer buf = ByteBuffer.allocate(22).order(ByteOrder.LITTLE_ENDIAN);
        buf.putShort(fp32ToFp16(1.0f));
        int qh = (1 << 1) | (1 << 17);
        buf.putInt(qh);
        byte[] qs = new byte[16];
        buf.put(qs);
        byte[] data = buf.array();

        // With correct element ordering (j and j+16, NOT j*2 and j*2+1):
        //   elem[j]    uses low nibble of qs[j] and qh bit j
        //   elem[j+16] uses high nibble of qs[j] and qh bit j+16
        // With qh=(1<<1)|(1<<17), all qs=0, d=1.0:
        //   elem[0]:  j=0, bit 0=0 → (0|0)-16=-16
        //   elem[1]:  j=1, bit 1=1 → (0|16)-16=0
        //   elem[16]: j=0, bit 16=0 → (0|0)-16=-16
        //   elem[17]: j=1, bit 17=1 → (0|16)-16=0
        //   all others: -16
        System.out.println("Synthetic Q5_0 production path test (d=1.0, qh bit1=1, bit17=1):");

        INDArray result = DequantizerFactory.dequantizeToArray(
                data, GGMLDataType.GGML_TYPE_Q5_0, new long[]{32}, DataType.FLOAT);

        float[] vals = result.toFloatVector();
        System.out.println("First 8 values:");
        for (int i = 0; i < 8; i++) {
            System.out.printf("  [%2d] = %.1f%n", i, vals[i]);
        }
        System.out.println("Expected: [0]=-16, [1]=0, [16]=-16, [17]=0, rest=-16");

        // The CORRECT values per GGML element ordering:
        assertEquals(-16.0f, vals[0], 1e-4f, "elem[0]: j=0, qh bit 0=0 → -16");
        assertEquals(  0.0f, vals[1], 1e-4f, "elem[1]: j=1, qh bit 1=1 → 0");
        assertEquals(-16.0f, vals[2], 1e-4f, "elem[2]: j=2, qh bit 2=0 → -16");
        assertEquals(-16.0f, vals[3], 1e-4f, "elem[3]: j=3, qh bit 3=0 → -16");

        result.close();
        System.out.println("PASS: production path gives correct Q5_0 dequant.");
    }

    // ── Utilities ──────────────────────────────────────────────────────────

    private static long[] reverseShape(long[] shape) {
        if (shape == null || shape.length <= 1) return shape;
        long[] reversed = new long[shape.length];
        for (int i = 0; i < shape.length; i++) {
            reversed[i] = shape[shape.length - 1 - i];
        }
        return reversed;
    }

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
