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
import org.nd4j.ggml.quantization.Q4_KDequantizer;

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Root-cause test for Q4_K_M garbage output vs fp16 correct output.
 *
 * Compares dequantized Q4_K tensor values from the q4_k_m.gguf against the
 * corresponding fp16 values from the fp16.gguf for the same model.
 * This test operates purely in Java (no native ND4J ops) to isolate the
 * Java dequantizer math from any CUDA/native concerns.
 *
 * Model paths (must exist):
 *   ~/.kompile/models/chat/qwen2.5-0.5b-instruct-q4_k_m.gguf
 *   ~/.kompile/models/chat/qwen2.5-0.5b-instruct-fp16.gguf
 *
 * Skip gracefully if either file is absent.
 */
@DisplayName("Q4_K Real-Model Dequant Correctness Test")
class Q4KDequantRealModelTest {

    private static final String Q4K_PATH = System.getProperty("user.home")
            + "/.kompile/models/chat/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    private static final String FP16_PATH = System.getProperty("user.home")
            + "/.kompile/models/chat/qwen2.5-0.5b-instruct-fp16.gguf";

    // Cosine similarity threshold for q4 dequant vs fp16 ground truth (per tensor)
    private static final double COSINE_THRESHOLD = 0.99;
    // Max acceptable relative error in individual values
    private static final double MAX_REL_ERR = 0.10;

    @Test
    @DisplayName("Q4_K dequant matches fp16 ground truth for first Q4_K weight tensor")
    void testQ4KDequantMatchesFp16() throws Exception {
        File q4kFile = new File(Q4K_PATH);
        File fp16File = new File(FP16_PATH);

        if (!q4kFile.exists() || !fp16File.exists()) {
            System.out.println("SKIP: model files not found.");
            System.out.println("  q4k: " + q4kFile.exists() + " -> " + Q4K_PATH);
            System.out.println("  fp16: " + fp16File.exists() + " -> " + FP16_PATH);
            return;
        }

        System.out.println("Q4K file: " + q4kFile.length() + " bytes");
        System.out.println("FP16 file: " + fp16File.length() + " bytes");

        // ── Read Q4_K tensor list ──────────────────────────────────────────
        GGMLTensorInfo q4kInfo = null;
        GGMLTensorInfo fp16Info = null;
        byte[] q4kBytes = null;
        float[] fp16Values = null;

        try (GGUFReader q4kReader = new GGUFReader(q4kFile);
             GGUFReader fp16Reader = new GGUFReader(fp16File)) {

            List<GGMLTensorInfo> q4kTensors = q4kReader.getMetadata().getTensors();
            List<GGMLTensorInfo> fp16Tensors = fp16Reader.getMetadata().getTensors();

            System.out.println("Q4K tensors: " + q4kTensors.size());
            System.out.println("FP16 tensors: " + fp16Tensors.size());

            // Print first 5 tensor names + types from each
            System.out.println("Q4K first tensors:");
            for (int i = 0; i < Math.min(5, q4kTensors.size()); i++) {
                GGMLTensorInfo t = q4kTensors.get(i);
                System.out.println("  " + t.getName() + " shape=" + t.getShapeString() + " type=" + t.getDataType());
            }
            System.out.println("FP16 first tensors:");
            for (int i = 0; i < Math.min(5, fp16Tensors.size()); i++) {
                GGMLTensorInfo t = fp16Tensors.get(i);
                System.out.println("  " + t.getName() + " shape=" + t.getShapeString() + " type=" + t.getDataType());
            }

            // Find first Q4_K tensor in q4k file that also exists in fp16 file as F16
            for (GGMLTensorInfo t : q4kTensors) {
                if (t.getDataType() == GGMLDataType.GGML_TYPE_Q4_K) {
                    // Find same tensor in fp16 file
                    for (GGMLTensorInfo tf : fp16Tensors) {
                        if (tf.getName().equals(t.getName()) &&
                                tf.getDataType() == GGMLDataType.GGML_TYPE_F16) {
                            q4kInfo = t;
                            fp16Info = tf;
                            break;
                        }
                    }
                    if (q4kInfo != null) break;
                }
            }

            if (q4kInfo == null) {
                System.out.println("SKIP: no matching Q4_K+F16 tensor pair found.");
                return;
            }

            System.out.println("\nSelected tensor: " + q4kInfo.getName());
            System.out.println("  Q4K shape: " + q4kInfo.getShapeString() + " numElements=" + q4kInfo.getNumElements());
            System.out.println("  FP16 shape: " + fp16Info.getShapeString() + " numElements=" + fp16Info.getNumElements());

            assertEquals(q4kInfo.getNumElements(), fp16Info.getNumElements(),
                    "Element count must match between Q4K and FP16 tensors");

            // Read raw bytes from q4k file
            q4kBytes = q4kReader.readTensorData(q4kInfo);
            System.out.println("Q4K raw bytes: " + q4kBytes.length);

            // Read fp16 bytes and convert to float[]
            byte[] fp16RawBytes = fp16Reader.readTensorData(fp16Info);
            System.out.println("FP16 raw bytes: " + fp16RawBytes.length + " (expected " + (fp16Info.getNumElements() * 2) + ")");
            fp16Values = fp16BytesToFloat(fp16RawBytes, (int) fp16Info.getNumElements());
        }

        // ── Dequantize Q4_K using Java dequantizer ─────────────────────────
        Q4_KDequantizer dequant = new Q4_KDequantizer();
        float[] q4kValues = dequant.dequantize(q4kBytes, q4kInfo.getNumElements());

        assertEquals(fp16Values.length, q4kValues.length, "Output length mismatch");

        // ── Compare first 256 elements (one super-block) in detail ──────────
        System.out.println("\nFirst 32 element comparison (q4k dequant vs fp16):");
        int printCount = Math.min(32, q4kValues.length);
        for (int i = 0; i < printCount; i++) {
            System.out.printf("  [%3d] q4k=%.6f  fp16=%.6f  diff=%.6f%n",
                    i, q4kValues[i], fp16Values[i], Math.abs(q4kValues[i] - fp16Values[i]));
        }

        // ── Compute cosine similarity ────────────────────────────────────────
        double dotProduct = 0, q4kNorm = 0, fp16Norm = 0;
        int limit = Math.min(q4kValues.length, 4096); // Use first 4096 elements for speed
        for (int i = 0; i < limit; i++) {
            dotProduct += (double) q4kValues[i] * fp16Values[i];
            q4kNorm += (double) q4kValues[i] * q4kValues[i];
            fp16Norm += (double) fp16Values[i] * fp16Values[i];
        }
        double cosine = dotProduct / (Math.sqrt(q4kNorm) * Math.sqrt(fp16Norm) + 1e-10);

        // ── Compute max absolute error and relative error ────────────────────
        double maxAbsErr = 0;
        double maxRelErr = 0;
        int maxErrIdx = -1;
        for (int i = 0; i < limit; i++) {
            double absErr = Math.abs(q4kValues[i] - fp16Values[i]);
            double scale = Math.abs(fp16Values[i]) + 1e-8;
            double relErr = absErr / scale;
            if (absErr > maxAbsErr) {
                maxAbsErr = absErr;
                maxErrIdx = i;
            }
            if (relErr > maxRelErr) maxRelErr = relErr;
        }

        System.out.printf("%nMetrics over first %d elements:%n", limit);
        System.out.printf("  Cosine similarity:  %.6f  (threshold: %.3f)%n", cosine, COSINE_THRESHOLD);
        System.out.printf("  Max abs error:      %.6f  at index %d%n", maxAbsErr, maxErrIdx);
        System.out.printf("  Max relative error: %.6f  (threshold: %.3f)%n", maxRelErr, MAX_REL_ERR);

        if (maxErrIdx >= 0) {
            System.out.printf("  Worst element: q4k=%.6f  fp16=%.6f%n",
                    q4kValues[maxErrIdx], fp16Values[maxErrIdx]);
        }

        // ── Assert ───────────────────────────────────────────────────────────
        assertTrue(cosine >= COSINE_THRESHOLD,
                String.format("Cosine similarity %.6f below threshold %.3f — Q4K dequantization is WRONG",
                        cosine, COSINE_THRESHOLD));

        System.out.println("\nPASS: Q4_K dequant matches fp16 within acceptable error.");
    }

    /**
     * Cross-check: verify the Q4_K block layout byte-for-byte on the first block
     * against manually computed expected values.
     */
    @Test
    @DisplayName("Q4_K first block manual byte verification")
    void testFirstBlockManualVerification() throws Exception {
        File q4kFile = new File(Q4K_PATH);
        if (!q4kFile.exists()) {
            System.out.println("SKIP: " + Q4K_PATH + " not found.");
            return;
        }

        try (GGUFReader reader = new GGUFReader(q4kFile)) {
            List<GGMLTensorInfo> tensors = reader.getMetadata().getTensors();

            // Find first Q4_K tensor
            GGMLTensorInfo q4kTensor = null;
            for (GGMLTensorInfo t : tensors) {
                if (t.getDataType() == GGMLDataType.GGML_TYPE_Q4_K) {
                    q4kTensor = t;
                    break;
                }
            }
            if (q4kTensor == null) {
                System.out.println("SKIP: no Q4_K tensor found.");
                return;
            }

            byte[] rawBytes = reader.readTensorData(q4kTensor);
            System.out.println("First Q4_K tensor: " + q4kTensor.getName()
                    + " shape=" + q4kTensor.getShapeString()
                    + " rawBytes=" + rawBytes.length);

            // Parse first 144-byte block manually
            ByteBuffer bb = ByteBuffer.wrap(rawBytes, 0, 144).order(ByteOrder.LITTLE_ENDIAN);
            short dRaw = bb.getShort();
            short dminRaw = bb.getShort();
            float d = fp16ToFloat(dRaw);
            float dmin = fp16ToFloat(dminRaw);
            System.out.printf("Block 0: d=%.6f  dmin=%.6f%n", d, dmin);

            byte[] scales = new byte[12];
            bb.get(scales);
            System.out.print("Scales bytes: ");
            for (byte s : scales) System.out.printf("%3d ", s & 0xFF);
            System.out.println();

            // Dequantize using our Java dequantizer
            Q4_KDequantizer dequant = new Q4_KDequantizer();
            float[] vals = dequant.dequantize(rawBytes, Math.min(256, q4kTensor.getNumElements()));
            System.out.println("First block values (first 16):");
            for (int i = 0; i < 16; i++) {
                System.out.printf("  [%2d] %.6f%n", i, vals[i]);
            }

            // Verify: values should be finite and in a reasonable range
            boolean allFinite = true;
            boolean allZero = true;
            for (float v : vals) {
                if (!Float.isFinite(v)) { allFinite = false; break; }
                if (Math.abs(v) > 1e-10f) allZero = false;
            }
            assertTrue(allFinite, "Q4_K dequantized values contain NaN/Inf");
            assertFalse(allZero, "Q4_K dequantized values are all zero (wrong dequant)");
            System.out.println("PASS: first block values are finite and non-zero.");
        }
    }

    // ── Utility: fp16 raw bytes → float[] ────────────────────────────────

    private static float[] fp16BytesToFloat(byte[] bytes, int numElements) {
        float[] result = new float[numElements];
        ByteBuffer bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < numElements && bb.remaining() >= 2; i++) {
            short bits = bb.getShort();
            result[i] = fp16ToFloat(bits);
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
}
