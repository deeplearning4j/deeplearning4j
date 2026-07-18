/*
 *  SPDX-License-Identifier: Apache-2.0
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
 * Deep diagnostic: compare Q5_0 block bytes manually against fp16 ground truth
 * to find exactly where the Java dequantizer diverges from the correct answer.
 *
 * Tests a SMALL tensor (not token_embd.weight which is 136M elements) to avoid
 * size-related issues, then verifies block-by-block.
 */
@DisplayName("Q5_0 Layout Diagnostic Test")
class Q5_0LayoutDiagTest {

    private static final String Q4K_PATH = System.getProperty("user.home")
            + "/.kompile/models/chat/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    private static final String FP16_PATH = System.getProperty("user.home")
            + "/.kompile/models/chat/qwen2.5-0.5b-instruct-fp16.gguf";

    /**
     * Find the SMALLEST Q5_0 tensor (to make it tractable), print block 0 raw bytes,
     * manually compute expected dequant vs actual Java output, and compare with fp16.
     */
    @Test
    @DisplayName("Q5_0 small tensor block-0 raw-byte diagnostic")
    void testSmallQ5_0BlockDiagnostic() throws Exception {
        File q4kFile = new File(Q4K_PATH);
        File fp16File = new File(FP16_PATH);

        if (!q4kFile.exists() || !fp16File.exists()) {
            System.out.println("SKIP: model files not found.");
            return;
        }

        // Find smallest Q5_0 tensor
        GGMLTensorInfo smallQ5 = null;
        byte[] q5Bytes = null;
        float[] fp16Values = null;

        try (GGUFReader q4kReader = new GGUFReader(q4kFile);
             GGUFReader fp16Reader = new GGUFReader(fp16File)) {

            List<GGMLTensorInfo> q4kTensors = q4kReader.getMetadata().getTensors();
            List<GGMLTensorInfo> fp16Tensors = fp16Reader.getMetadata().getTensors();

            for (GGMLTensorInfo t : q4kTensors) {
                if (t.getDataType() == GGMLDataType.GGML_TYPE_Q5_0) {
                    if (smallQ5 == null || t.getNumElements() < smallQ5.getNumElements()) {
                        // Check fp16 counterpart exists
                        for (GGMLTensorInfo tf : fp16Tensors) {
                            if (tf.getName().equals(t.getName()) && tf.getDataType() == GGMLDataType.GGML_TYPE_F16) {
                                smallQ5 = t;
                                break;
                            }
                        }
                    }
                }
            }

            if (smallQ5 == null) {
                System.out.println("SKIP: no Q5_0 tensor found.");
                return;
            }

            q5Bytes = q4kReader.readTensorData(smallQ5);

            for (GGMLTensorInfo tf : fp16Tensors) {
                if (tf.getName().equals(smallQ5.getName())) {
                    byte[] fp16Raw = fp16Reader.readTensorData(tf);
                    fp16Values = fp16BytesToFloat(fp16Raw, (int) tf.getNumElements());
                    break;
                }
            }
        }

        System.out.println("Smallest Q5_0 tensor: " + smallQ5.getName()
                + "  shape=" + smallQ5.getShapeString()
                + "  elements=" + smallQ5.getNumElements()
                + "  rawBytes=" + q5Bytes.length);

        // Print block 0 raw bytes
        ByteBuffer bb = ByteBuffer.wrap(q5Bytes).order(ByteOrder.LITTLE_ENDIAN);
        short scaleRaw = bb.getShort();
        float d = fp16ToFloat(scaleRaw);
        int qh = bb.getInt();
        byte[] qs = new byte[16];
        bb.get(qs);

        System.out.printf("Block 0: d=%.6f (raw=0x%04X)%n", d, scaleRaw & 0xFFFF);
        System.out.printf("         qh=0x%08X%n", qh);
        System.out.print("         qs=");
        for (byte b : qs) System.out.printf("%02X ", b & 0xFF);
        System.out.println();

        // Manually compute first 8 elements from block 0
        System.out.println("Manual block-0 decode (GGML formula):");
        for (int j = 0; j < 8; j++) {
            int xh0 = ((qh >> j) & 1);
            int xh1 = ((qh >> (j + 16)) & 1);
            int low0 = qs[j] & 0x0F;
            int low1 = (qs[j] >> 4) & 0x0F;
            int v0 = (low0 | (xh0 << 4)) - 16;
            int v1 = (low1 | (xh1 << 4)) - 16;
            float f0 = d * v0;
            float f1 = d * v1;
            System.out.printf("  j=%2d: elem[%2d]=%.6f (xh0=%d, low0=%d, v0=%d)  "
                            + "elem[%2d]=%.6f (xh1=%d, low1=%d, v1=%d)"
                            + "  fp16[%d]=%.6f  fp16[%d]=%.6f%n",
                    j, j*2, f0, xh0, low0, v0,
                    j*2+1, f1, xh1, low1, v1,
                    j*2, fp16Values[j*2], j*2+1, fp16Values[j*2+1]);
        }

        // Run Java dequantizer on ALL elements
        Q5_0Dequantizer dequant = new Q5_0Dequantizer();
        float[] result = dequant.dequantize(q5Bytes, smallQ5.getNumElements());

        System.out.println("\nJava dequantizer first 16 elements vs fp16:");
        for (int i = 0; i < 16; i++) {
            System.out.printf("  [%2d] java=%.6f  fp16=%.6f  diff=%.6f%n",
                    i, result[i], fp16Values[i], Math.abs(result[i] - fp16Values[i]));
        }

        // Cosine over ALL elements
        int limit = result.length;
        double dot = 0, n1 = 0, n2 = 0;
        for (int i = 0; i < limit; i++) {
            dot += (double) result[i] * fp16Values[i];
            n1 += (double) result[i] * result[i];
            n2 += (double) fp16Values[i] * fp16Values[i];
        }
        double cosine = dot / (Math.sqrt(n1) * Math.sqrt(n2) + 1e-10);
        System.out.printf("%nCosine (all %d elements): %.6f%n", limit, cosine);

        assertTrue(cosine >= 0.99,
                String.format("Small Q5_0 tensor cosine %.6f < 0.99 — Java dequantizer wrong.", cosine));
        System.out.println("PASS");
    }

    // ── Utilities ─────────────────────────────────────────────────────────

    private static float[] fp16BytesToFloat(byte[] bytes, int numElements) {
        float[] result = new float[numElements];
        ByteBuffer b = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < numElements && b.remaining() >= 2; i++) {
            result[i] = fp16ToFloat(b.getShort());
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
