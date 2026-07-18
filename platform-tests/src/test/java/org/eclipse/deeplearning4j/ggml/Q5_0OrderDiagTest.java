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
 * Diagnostic: find whether the Q5_0 dequant values appear in fp16 at a different position.
 * Also checks if tensor was transposed between q4k and fp16 files.
 * Also tests a simple 2-block-tensor to check scale sign and high bit independence.
 */
@DisplayName("Q5_0 Element Ordering Diagnostic")
class Q5_0OrderDiagTest {

    private static final String Q4K_PATH = System.getProperty("user.home")
            + "/.kompile/models/chat/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    private static final String FP16_PATH = System.getProperty("user.home")
            + "/.kompile/models/chat/qwen2.5-0.5b-instruct-fp16.gguf";

    /**
     * 1. Compare tensor SHAPES between files.
     * 2. Print both fp16 and Q5_0 first 64 values to spot a transposition pattern.
     * 3. Check if block 0 fp16 values == block 0 q5_0 values rearranged.
     */
    @Test
    @DisplayName("Q5_0 element-order comparison: is fp16 transposed?")
    void testElementOrderComparison() throws Exception {
        File q4kFile = new File(Q4K_PATH);
        File fp16File = new File(FP16_PATH);

        if (!q4kFile.exists() || !fp16File.exists()) {
            System.out.println("SKIP: model files not found.");
            return;
        }

        GGMLTensorInfo smallQ5 = null;
        GGMLTensorInfo fp16TensorInfo = null;
        byte[] q5Bytes = null;
        float[] fp16Values = null;

        try (GGUFReader q4kReader = new GGUFReader(q4kFile);
             GGUFReader fp16Reader = new GGUFReader(fp16File)) {

            List<GGMLTensorInfo> q4kTensors = q4kReader.getMetadata().getTensors();
            List<GGMLTensorInfo> fp16Tensors = fp16Reader.getMetadata().getTensors();

            // Find smallest Q5_0 tensor with fp16 counterpart
            for (GGMLTensorInfo t : q4kTensors) {
                if (t.getDataType() == GGMLDataType.GGML_TYPE_Q5_0) {
                    for (GGMLTensorInfo tf : fp16Tensors) {
                        if (tf.getName().equals(t.getName())) {
                            if (smallQ5 == null || t.getNumElements() < smallQ5.getNumElements()) {
                                smallQ5 = t;
                                fp16TensorInfo = tf;
                            }
                            break;
                        }
                    }
                }
            }

            if (smallQ5 == null) {
                System.out.println("SKIP: no Q5_0 tensor found.");
                return;
            }

            q5Bytes = q4kReader.readTensorData(smallQ5);
            byte[] fp16Raw = fp16Reader.readTensorData(fp16TensorInfo);
            fp16Values = fp16BytesToFloat(fp16Raw, (int) fp16TensorInfo.getNumElements());
        }

        System.out.println("Q5_0 tensor: " + smallQ5.getName()
                + " shape=" + smallQ5.getShapeString()
                + " elements=" + smallQ5.getNumElements());
        System.out.println("FP16 tensor: " + fp16TensorInfo.getName()
                + " shape=" + fp16TensorInfo.getShapeString()
                + " elements=" + fp16TensorInfo.getNumElements());

        // Dequantize using Java
        Q5_0Dequantizer dequant = new Q5_0Dequantizer();
        float[] q5vals = dequant.dequantize(q5Bytes, smallQ5.getNumElements());

        System.out.println("\nFirst 64 fp16 values:");
        for (int i = 0; i < 64; i++) {
            System.out.printf("  fp16[%3d] = % .6f%n", i, fp16Values[i]);
        }

        System.out.println("\nFirst 64 Q5_0 dequant values:");
        for (int i = 0; i < 64; i++) {
            System.out.printf("  q5[%3d] = % .6f%n", i, q5vals[i]);
        }

        // Check if Q5_0 block-0 values = fp16 values at ROW-TRANSPOSED positions
        // attn_k.weight shape [896, 128] in GGUF (col-major) = 896 rows × 128 cols
        // If transposed, element [row, col] = fp16[col * rows + row] vs q5[row * cols + col]
        long[] shape = smallQ5.getShape();
        System.out.println("\nShape: " + java.util.Arrays.toString(shape));

        if (shape.length == 2) {
            long dim0 = shape[0];  // "fast" GGUF dim = "rows" in GGUF col-major
            long dim1 = shape[1];  // "slow" GGUF dim = "cols" in GGUF col-major

            System.out.printf("Checking if q5[row][col] == fp16[col][row] (transpose): dim0=%d, dim1=%d%n",
                    dim0, dim1);

            // Compare first 8x8 corner
            int mismatches = 0;
            for (int col = 0; col < Math.min(8, dim1); col++) {
                for (int row = 0; row < Math.min(8, dim0); row++) {
                    float q5val = q5vals[(int)(row + col * dim0)];  // GGUF col-major: row varies fast
                    float fp16val = fp16Values[(int)(row + col * dim0)];  // same layout
                    float fp16transposed = fp16Values[(int)(col + row * dim1)];  // transposed
                    System.out.printf("  [%3d,%3d] q5=%.4f fp16_same=%.4f fp16_transposed=%.4f%n",
                            row, col, q5val, fp16val, fp16transposed);
                    if (Math.abs(q5val - fp16val) > 0.01f && Math.abs(q5val - fp16transposed) < 0.01f) {
                        mismatches++;
                    }
                }
            }

            if (mismatches > 10) {
                System.out.println("TRANSPOSITION detected: q5 values match fp16 at transposed positions");
            }
        }

        // Try comparing first 32 q5_0 values against first 32 fp16 in ROW-MAJOR order
        // GGUF shape [896, 128] in col-major: elements go [0,0], [1,0], [2,0], ... [895,0], [0,1], ...
        // Row-major: elements go [0,0], [0,1], ..., [0,127], [1,0], ...
        System.out.println("\nAttempting col-major → row-major reorder comparison:");
        if (shape.length == 2) {
            long rows = shape[1];  // In GGUF col-major, shape[1] is the "major" dim
            long cols = shape[0];  // shape[0] is the "minor" (fast) dim

            System.out.printf("Treating as rows=%d, cols=%d (row-major reorder)%n", rows, cols);

            int nCheck = (int) Math.min(64, rows * cols);
            int matches = 0;
            for (int i = 0; i < nCheck; i++) {
                // Row-major index i → col-major index
                int r = i / (int)cols;  // row
                int c = i % (int)cols;  // col
                int colMajorIdx = (int)(c + r * cols); // This is the SAME as row-major! Wrong.
                // Actually col-major: element[row][col] stored at row + col*rows
                // Row-major: element[row][col] stored at row*cols + col
                // So col-major[r][c] = q5[r + c * rows]
                //    fp16_rowmajor[r][c] = fp16[r * cols + c]
                int colMajorQ5Idx = (int)(r + (long)c * rows);   // col-major q5
                float q5v = q5vals[colMajorQ5Idx];
                float fp16v = fp16Values[i];  // fp16 might be row-major
                if (Math.abs(q5v - fp16v) < 0.01f) matches++;
            }
            System.out.printf("Row-vs-colMajor matches in first %d: %d / %d%n", nCheck, matches, nCheck);
        }

        // Final: just compute cosine of raw linear layout
        int limit = (int) Math.min(q5vals.length, fp16Values.length);
        double dot = 0, n1 = 0, n2 = 0;
        for (int i = 0; i < limit; i++) {
            dot += (double) q5vals[i] * fp16Values[i];
            n1 += (double) q5vals[i] * q5vals[i];
            n2 += (double) fp16Values[i] * fp16Values[i];
        }
        double cosine = dot / (Math.sqrt(n1) * Math.sqrt(n2) + 1e-10);
        System.out.printf("%nCosine (linear, all %d elements): %.6f%n", limit, cosine);

        // GGUF is ALWAYS stored in the same order for both files (no reordering)
        // So cosine > 0.99 should hold if the Java dequantizer is correct
        assertTrue(cosine >= 0.99,
                String.format("Q5_0 cosine %.6f < 0.99 after fix — dequantizer bug remains or fp16 has different element order", cosine));
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
