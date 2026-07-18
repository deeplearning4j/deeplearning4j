/*
 *  SPDX-License-Identifier: Apache-2.0
 */

package org.eclipse.deeplearning4j.ggml;

import org.junit.jupiter.api.*;
import org.nd4j.ggml.format.GGMLDataType;
import org.nd4j.ggml.format.GGMLTensorInfo;
import org.nd4j.ggml.format.GGUFReader;

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;

/**
 * Print RAW bytes of Q5_0 block 0 and also fp16 block 0,
 * then manually decode to find the correct scale D and element values.
 *
 * Also verifies that the fp16 and Q5_0 tensor data offsets are correct
 * (not misaligned by GGUF alignment padding).
 */
@DisplayName("Q5_0 Raw Byte Diagnostic")
class Q5_0RawByteDiagTest {

    private static final String Q4K_PATH = System.getProperty("user.home")
            + "/.kompile/models/chat/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    private static final String FP16_PATH = System.getProperty("user.home")
            + "/.kompile/models/chat/qwen2.5-0.5b-instruct-fp16.gguf";

    @Test
    @DisplayName("Raw bytes of Q5_0 block 0 vs fp16 ground truth first 4 values")
    void testRawBlockBytes() throws Exception {
        File q4kFile = new File(Q4K_PATH);
        File fp16File = new File(FP16_PATH);

        if (!q4kFile.exists() || !fp16File.exists()) {
            System.out.println("SKIP: model files not found.");
            return;
        }

        GGMLTensorInfo q5Info = null;
        GGMLTensorInfo fp16Info = null;
        byte[] q5Bytes = null;
        byte[] fp16Bytes = null;

        try (GGUFReader q4kReader = new GGUFReader(q4kFile);
             GGUFReader fp16Reader = new GGUFReader(fp16File)) {

            List<GGMLTensorInfo> q4kTensors = q4kReader.getMetadata().getTensors();
            List<GGMLTensorInfo> fp16Tensors = fp16Reader.getMetadata().getTensors();

            for (GGMLTensorInfo t : q4kTensors) {
                if (t.getDataType() == GGMLDataType.GGML_TYPE_Q5_0) {
                    for (GGMLTensorInfo tf : fp16Tensors) {
                        if (tf.getName().equals(t.getName())) {
                            if (q5Info == null || t.getNumElements() < q5Info.getNumElements()) {
                                q5Info = t;
                                fp16Info = tf;
                            }
                            break;
                        }
                    }
                }
            }

            q5Bytes = q4kReader.readTensorData(q5Info);
            fp16Bytes = fp16Reader.readTensorData(fp16Info);
        }

        System.out.println("Tensor: " + q5Info.getName()
                + " Q5_0 dataOffset=" + q5Info.getDataOffset()
                + " FP16 dataOffset=" + fp16Info.getDataOffset());

        // Print first 44 bytes (2 blocks) of Q5_0 raw data
        System.out.println("\nQ5_0 raw bytes (hex), first 44 bytes:");
        for (int i = 0; i < 44 && i < q5Bytes.length; i++) {
            System.out.printf("%02X ", q5Bytes[i] & 0xFF);
            if ((i + 1) % 22 == 0) System.out.println(" ← block " + (i/22));
        }
        System.out.println();

        // Print first 64 bytes of FP16 raw data (= 32 fp16 values)
        System.out.println("\nFP16 raw bytes (hex), first 64 bytes:");
        for (int i = 0; i < 64 && i < fp16Bytes.length; i++) {
            System.out.printf("%02X ", fp16Bytes[i] & 0xFF);
            if ((i + 1) % 16 == 0) System.out.println();
        }
        System.out.println();

        // Decode Q5_0 block 0 manually
        ByteBuffer bbQ5 = ByteBuffer.wrap(q5Bytes).order(ByteOrder.LITTLE_ENDIAN);
        short dRaw = bbQ5.getShort();
        int qh = bbQ5.getInt();
        byte[] qs = new byte[16];
        bbQ5.get(qs);
        float d = fp16ToFloat(dRaw);

        System.out.printf("Q5_0 block 0: d=%.8f (raw=0x%04X), qh=0x%08X%n", d, dRaw & 0xFFFF, qh);
        System.out.printf("qs bytes: ");
        for (byte b : qs) System.out.printf("%02X ", b & 0xFF);
        System.out.println();

        // Decode fp16 first 32 values
        ByteBuffer bbFp16 = ByteBuffer.wrap(fp16Bytes).order(ByteOrder.LITTLE_ENDIAN);
        float[] fp16vals = new float[32];
        for (int i = 0; i < 32; i++) {
            fp16vals[i] = fp16ToFloat(bbFp16.getShort());
        }
        System.out.println("FP16 first 32 values:");
        for (int i = 0; i < 32; i++) {
            System.out.printf("  fp16[%2d] = % .6f  (raw=0x%04X)%n",
                    i, fp16vals[i], fp16Bytes[i*2] & 0xFF | ((fp16Bytes[i*2+1] & 0xFF) << 8));
        }

        // Now decode Q5_0 elements 0-31 from block 0
        System.out.println("\nQ5_0 block 0 decoded elements (GGML formula):");
        float[] q5vals = new float[32];
        for (int j = 0; j < 16; j++) {
            int xh0 = ((qh >> j) & 1);
            int xh1 = ((qh >> (j + 16)) & 1);
            int low0 = qs[j] & 0x0F;
            int low1 = (qs[j] >> 4) & 0x0F;
            int v0 = (low0 | (xh0 << 4)) - 16;
            int v1 = (low1 | (xh1 << 4)) - 16;
            q5vals[j*2]   = d * v0;
            q5vals[j*2+1] = d * v1;
            System.out.printf("  j=%2d elem[%2d]=%.6f v0=%d  elem[%2d]=%.6f v1=%d%n",
                    j, j*2, q5vals[j*2], v0, j*2+1, q5vals[j*2+1], v1);
        }

        // Cosine between q5 and fp16 for first 32 elements
        double dot = 0, n1 = 0, n2 = 0;
        for (int i = 0; i < 32; i++) {
            dot += q5vals[i] * fp16vals[i];
            n1 += q5vals[i] * q5vals[i];
            n2 += fp16vals[i] * fp16vals[i];
        }
        double cosine = dot / (Math.sqrt(n1) * Math.sqrt(n2) + 1e-10);
        System.out.printf("%nBlock 0 cosine (Q5_0 vs FP16): %.6f%n", cosine);

        if (cosine >= 0.95) {
            System.out.println("Block 0 decoded correctly.");
        } else {
            System.out.printf("Block 0 WRONG (cosine=%.4f). Something is wrong with the block layout or the fp16 ground truth.%n", cosine);
        }
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
