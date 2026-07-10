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

package org.eclipse.deeplearning4j.nd4j.linalg.ops;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.GgmlQMatMul;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Tests for the ggml_qmatmul op — runtime quantized matmul against GGML-packed weights.
 *
 * <h2>Reference strategy</h2>
 * Tests generate structurally valid random packed GGML blocks (random quant bytes + sane
 * fp16 scales), then dequantize them using the existing ggml_dequantize path to produce a
 * reference fp32 weight matrix. The ggml_qmatmul output is then compared against
 * matmul(A, W^T) where W is the dequantized reference.  Because both paths use the
 * IDENTICAL dequant math, the expected tolerance is tight (< 1e-3 relative error for
 * fp32 outputs; a small additional FP16 rounding budget for fp16 outputs).
 *
 * <h2>Running a single test</h2>
 * <pre>
 * cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && \
 *   /home/agibsonccc/dev-apps/mvn/bin/mvn test -Dbackend.artifactId=nd4j-cuda-12.9 \
 *   -Dtest=TestGgmlQMatMul 2>&1 | tee /tmp/test-ggml-qmatmul.log
 * </pre>
 *
 * For parameterized methods use a wildcard suffix:
 * <pre>
 * -Dtest=TestGgmlQMatMul#testQ8_0*
 * -Dtest=TestGgmlQMatMul#testQ4K*
 * -Dtest=TestGgmlQMatMul#testQ6K*
 * </pre>
 */
@Slf4j
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
public class TestGgmlQMatMul {

    // ─── Block layout constants (must match C++ ggml_qmatmul.h) ─────────────
    private static final int BLOCK_Q8_0  = 34;
    private static final int ELEMS_Q8_0  = 32;
    private static final int BLOCK_Q4_K  = 144;
    private static final int ELEMS_Q4_K  = 256;
    private static final int BLOCK_Q6_K  = 210;
    private static final int ELEMS_Q6_K  = 256;

    // GgmlQuantType enum values (positional, matching C++ GgmlQuantType enum)
    private static final int QUANT_Q8_0 = 4;
    private static final int QUANT_Q4_K = 8;
    private static final int QUANT_Q6_K = 10;

    // ─── Block builders (identical math to C++ kernel) ────────────────────

    /**
     * Build a single Q8_0 block: 2B fp16 scale d + 32B int8 quants.
     * Dequant: x[i] = d * qs[i]
     */
    private static void buildQ8_0Block(byte[] buf, int offset, float d, int[] qs) {
        // encode d as fp16
        short dFp16 = floatToFp16(d);
        buf[offset]   = (byte)(dFp16 & 0xFF);
        buf[offset+1] = (byte)((dFp16 >> 8) & 0xFF);
        for (int i = 0; i < 32; i++) {
            buf[offset + 2 + i] = (byte)(qs[i] & 0xFF);
        }
    }

    /**
     * Dequantize one Q8_0 block into dst[0..31].
     */
    private static void dequantQ8_0Block(byte[] buf, int offset, float[] dst) {
        float d = fp16ToFloat(buf[offset] & 0xFF | ((buf[offset+1] & 0xFF) << 8));
        for (int i = 0; i < 32; i++) {
            dst[i] = d * buf[offset + 2 + i];  // signed byte
        }
    }

    /**
     * Build a single Q4_K block: 2B d + 2B dmin + 12B scales + 128B nibbles.
     * Dequant: x[j+0..31] = d*sc1*nibble - dmin*m1, x[j+32..63] = d*sc2*nibble - dmin*m2
     * (for j in 0,64,128,192)
     */
    private static void buildQ4_KBlock(byte[] buf, int offset, float d, float dmin,
                                        int[] scalesRaw, int[] nibblesFlat) {
        // d, dmin as fp16
        short dFp16    = floatToFp16(d);
        short dminFp16 = floatToFp16(dmin);
        buf[offset]   = (byte)(dFp16 & 0xFF);
        buf[offset+1] = (byte)((dFp16 >> 8) & 0xFF);
        buf[offset+2] = (byte)(dminFp16 & 0xFF);
        buf[offset+3] = (byte)((dminFp16 >> 8) & 0xFF);
        // 12 scale bytes (6-bit packed — for simplicity, just store in low 6 bits)
        // scalesRaw: 8 values (alternating sc/m for 4 sub-blocks), each 0..63
        // Packing: first 4 values at bits [5:0] of bytes 0-3
        //          next 4 values at bits [5:0] of bytes 4-7
        //          upper 2 bits spill into bytes 8-11 (we keep it simple: zeros here)
        for (int i = 0; i < 8; i++) {
            buf[offset + 4 + i] = (byte)(scalesRaw[i] & 0x3F);  // 6-bit
        }
        // remaining 4 scale bytes = 0 (no upper-bit spill for these small values)
        buf[offset + 12] = 0; buf[offset + 13] = 0;
        buf[offset + 14] = 0; buf[offset + 15] = 0;
        // 128 nibble bytes (256 nibbles)
        for (int i = 0; i < 128; i++) {
            int lo = nibblesFlat[2 * i]     & 0x0F;
            int hi = nibblesFlat[2 * i + 1] & 0x0F;
            buf[offset + 16 + i] = (byte)(lo | (hi << 4));
        }
    }

    /**
     * Dequantize one Q4_K block into dst[0..255].
     * Must match the C++ dequantBlockQ4_K logic exactly.
     */
    private static void dequantQ4_KBlock(byte[] buf, int offset, float[] dst) {
        float d    = fp16ToFloat(buf[offset] & 0xFF | ((buf[offset+1] & 0xFF) << 8));
        float dmin = fp16ToFloat(buf[offset+2] & 0xFF | ((buf[offset+3] & 0xFF) << 8));
        byte[] scaleBytes = Arrays.copyOfRange(buf, offset + 4, offset + 16);
        byte[] qs         = Arrays.copyOfRange(buf, offset + 16, offset + 144);

        int is = 0;
        for (int j = 0; j < 256; j += 64, is += 2) {
            int[] sc1m1 = getScaleMinK4(is,     scaleBytes);
            int[] sc2m2 = getScaleMinK4(is + 1, scaleBytes);
            float d1 = d * sc1m1[0], m1f = dmin * sc1m1[1];
            float d2 = d * sc2m2[0], m2f = dmin * sc2m2[1];
            int qIdx = (j / 64) * 32;
            for (int l = 0; l < 32; l++) {
                int qByte = qs[qIdx + l] & 0xFF;
                dst[j + l]      = d1 * (qByte & 0x0F) - m1f;
                dst[j + l + 32] = d2 * ((qByte >> 4) & 0x0F) - m2f;
            }
        }
    }

    /**
     * Build a single Q6_K block: 128B ql + 64B qh + 16B int8 scales + 2B fp16 d.
     */
    private static void buildQ6_KBlock(byte[] buf, int offset, float d, int[] scales6bit,
                                        int[] values6bit) {
        // ql[128], qh[64], scales[16], d (FP16)
        // Encode 6-bit values into ql (low 4 bits) and qh (high 2 bits)
        // values6bit[i] is in [-32..31] before the +32 stored offset
        // stored = value + 32, range [0..63]
        for (int i = 0; i < 256; i++) {
            int val = values6bit[i] & 0x3F;  // 6-bit: 0..63 (stored = original+32)
            int lo4 = val & 0x0F;
            int hi2 = (val >> 4) & 0x03;

            // Map index i to ql/qh positions (matching C++ Q6_K layout)
            // 256 values arranged in 2 outer halves of 128 each
            // Within each half: ql[0..63] and qh[0..31]
            int outer = i / 128;
            int inner = i % 128;
            int quadrant = inner / 32;  // 0..3
            int l = inner % 32;

            int qlOffset = outer * 64;
            int qhOffset = outer * 32;

            if (quadrant == 0) {
                // lo4 in low nibble of ql[qlOffset+l], hi2 in bits [1:0] of qh[qhOffset+l]
                buf[offset + qlOffset + l] = (byte)((buf[offset + qlOffset + l] & 0xF0) | lo4);
                buf[offset + 128 + qhOffset + l] = (byte)((buf[offset + 128 + qhOffset + l] & 0xFC) | hi2);
            } else if (quadrant == 1) {
                // lo4 in low nibble of ql[qlOffset+l+32], hi2 in bits [3:2] of qh[qhOffset+l]
                buf[offset + qlOffset + l + 32] = (byte)((buf[offset + qlOffset + l + 32] & 0xF0) | lo4);
                buf[offset + 128 + qhOffset + l] = (byte)((buf[offset + 128 + qhOffset + l] & 0xF3) | (hi2 << 2));
            } else if (quadrant == 2) {
                // lo4 in high nibble of ql[qlOffset+l], hi2 in bits [5:4] of qh[qhOffset+l]
                buf[offset + qlOffset + l] = (byte)((buf[offset + qlOffset + l] & 0x0F) | (lo4 << 4));
                buf[offset + 128 + qhOffset + l] = (byte)((buf[offset + 128 + qhOffset + l] & 0xCF) | (hi2 << 4));
            } else {
                // lo4 in high nibble of ql[qlOffset+l+32], hi2 in bits [7:6] of qh[qhOffset+l]
                buf[offset + qlOffset + l + 32] = (byte)((buf[offset + qlOffset + l + 32] & 0x0F) | (lo4 << 4));
                buf[offset + 128 + qhOffset + l] = (byte)((buf[offset + 128 + qhOffset + l] & 0x3F) | (hi2 << 6));
            }
        }
        // Scales: 16 int8 values at offset 192
        for (int i = 0; i < 16; i++) {
            buf[offset + 192 + i] = (byte)(scales6bit[i]);
        }
        // d as fp16 at offset 208
        short dFp16 = floatToFp16(d);
        buf[offset + 208] = (byte)(dFp16 & 0xFF);
        buf[offset + 209] = (byte)((dFp16 >> 8) & 0xFF);
    }

    /**
     * Dequantize one Q6_K block into dst[0..255].
     * Must match C++ dequantBlockQ6_K exactly.
     */
    private static void dequantQ6_KBlock(byte[] buf, int offset, float[] dst) {
        float d = fp16ToFloat(buf[offset + 208] & 0xFF | ((buf[offset + 209] & 0xFF) << 8));
        byte[] ql     = Arrays.copyOfRange(buf, offset,       offset + 128);
        byte[] qh     = Arrays.copyOfRange(buf, offset + 128, offset + 192);
        byte[] scales = Arrays.copyOfRange(buf, offset + 192, offset + 208);

        int qlOff = 0, qhOff = 0, scOff = 0;
        for (int outer = 0; outer < 2; outer++) {
            for (int l = 0; l < 32; l++) {
                int q1 = ((ql[qlOff + l]      & 0x0F) | (((qh[qhOff + l] >> 0) & 3) << 4)) - 32;
                int q2 = ((ql[qlOff + l + 32] & 0x0F) | (((qh[qhOff + l] >> 2) & 3) << 4)) - 32;
                int q3 = (((ql[qlOff + l]      >> 4) & 0x0F) | (((qh[qhOff + l] >> 4) & 3) << 4)) - 32;
                int q4 = (((ql[qlOff + l + 32] >> 4) & 0x0F) | (((qh[qhOff + l] >> 6) & 3) << 4)) - 32;

                int is = l / 16;
                dst[outer * 128 + l]      = d * scales[scOff + is + 0] * q1;
                dst[outer * 128 + l + 32] = d * scales[scOff + is + 2] * q2;
                dst[outer * 128 + l + 64] = d * scales[scOff + is + 4] * q3;
                dst[outer * 128 + l + 96] = d * scales[scOff + is + 6] * q4;
            }
            qlOff += 64;
            qhOff += 32;
            scOff += 8;
        }
    }

    // ─── K4 scale/min unpacker (matching C++ getScaleMinK4) ─────────────────
    private static int[] getScaleMinK4(int j, byte[] q) {
        int sc, m;
        if (j < 4) {
            sc = q[j] & 63;
            m  = q[j + 4] & 63;
        } else {
            sc = (q[j + 4] & 0x0F) | (((q[j - 4] >> 6) & 3) << 4);
            m  = ((q[j + 4] >> 4) & 0x0F) | (((q[j] >> 6) & 3) << 4);
        }
        return new int[]{sc, m};
    }

    // ─── FP16 helpers ────────────────────────────────────────────────────────
    private static short floatToFp16(float f) {
        int bits = Float.floatToIntBits(f);
        int sign = (bits >> 31) & 1;
        int exp  = (bits >> 23) & 0xFF;
        int mant = bits & 0x7FFFFF;
        int fp16Exp  = exp - 127 + 15;
        int fp16Mant = mant >> 13;
        if (fp16Exp <= 0) return (short)(sign << 15);
        if (fp16Exp >= 31) return (short)((sign << 15) | 0x7C00);
        return (short)((sign << 15) | (fp16Exp << 10) | fp16Mant);
    }

    private static float fp16ToFloat(int h) {
        int sign = (h >> 15) & 1;
        int exp  = (h >> 10) & 0x1F;
        int mant = h & 0x3FF;
        if (exp == 0) {
            if (mant == 0) return sign == 0 ? 0.0f : -0.0f;
            while ((mant & 0x400) == 0) { mant <<= 1; exp--; }
            exp++; mant &= ~0x400;
        } else if (exp == 31) {
            return (sign == 0) ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY;
        }
        int bits = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
        return Float.intBitsToFloat(bits);
    }

    // ─── Reference matmul for comparison ─────────────────────────────────────
    /**
     * Pure Java reference: out[m,n] = sum_k A[m,k] * W[n,k]  (A @ W^T)
     */
    private static float[][] referenceMatMul(float[][] A, float[][] W) {
        int M = A.length, K = A[0].length, N = W.length;
        float[][] out = new float[M][N];
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++) {
                double acc = 0;
                for (int k = 0; k < K; k++) acc += A[m][k] * W[n][k];
                out[m][n] = (float) acc;
            }
        return out;
    }

    // ─── Q8_0 tests ──────────────────────────────────────────────────────────

    @Test
    void testQ8_0_M1_K256_N7_FP32() {
        runQ8_0Test(1, 256, 7, false, DataType.FLOAT);
    }

    @Test
    void testQ8_0_M4_K256_N64_FP32() {
        runQ8_0Test(4, 256, 64, false, DataType.FLOAT);
    }

    @Test
    void testQ8_0_M33_K256_N64_FP32() {
        runQ8_0Test(33, 256, 64, false, DataType.FLOAT);
    }

    @Test
    void testQ8_0_M1_K512_N64_FP32() {
        runQ8_0Test(1, 512, 64, false, DataType.FLOAT);
    }

    @Test
    void testQ8_0_rank3_FP32() {
        runQ8_0Test(4, 256, 64, true, DataType.FLOAT);  // rank-3: [1,4,256]
    }

    @Test
    void testQ8_0_FP16Output() {
        runQ8_0Test(4, 256, 64, false, DataType.HALF);
    }

    // ─── Q4_K tests ──────────────────────────────────────────────────────────

    @Test
    void testQ4K_M1_K256_N7_FP32() {
        runQ4KTest(1, 256, 7, false, DataType.FLOAT);
    }

    @Test
    void testQ4K_M4_K256_N64_FP32() {
        runQ4KTest(4, 256, 64, false, DataType.FLOAT);
    }

    @Test
    void testQ4K_M33_K256_N64_FP32() {
        runQ4KTest(33, 256, 64, false, DataType.FLOAT);
    }

    @Test
    void testQ4K_M1_K512_N64_FP32() {
        runQ4KTest(1, 512, 64, false, DataType.FLOAT);
    }

    @Test
    void testQ4K_rank3_FP32() {
        runQ4KTest(4, 256, 64, true, DataType.FLOAT);
    }

    @Test
    void testQ4K_FP16Output() {
        runQ4KTest(4, 256, 64, false, DataType.HALF);
    }

    // ─── Q6_K tests ──────────────────────────────────────────────────────────

    @Test
    void testQ6K_M1_K256_N7_FP32() {
        runQ6KTest(1, 256, 7, false, DataType.FLOAT);
    }

    @Test
    void testQ6K_M4_K256_N64_FP32() {
        runQ6KTest(4, 256, 64, false, DataType.FLOAT);
    }

    @Test
    void testQ6K_M33_K256_N64_FP32() {
        runQ6KTest(33, 256, 64, false, DataType.FLOAT);
    }

    @Test
    void testQ6K_M1_K512_N64_FP32() {
        runQ6KTest(1, 512, 64, false, DataType.FLOAT);
    }

    @Test
    void testQ6K_rank3_FP32() {
        runQ6KTest(4, 256, 64, true, DataType.FLOAT);
    }

    @Test
    void testQ6K_FP16Output() {
        runQ6KTest(4, 256, 64, false, DataType.HALF);
    }

    // ─── Validation error cases ───────────────────────────────────────────────

    @Test
    void testValidationBadK() {
        // K not divisible by block size for Q8_0 (blockSize=32)
        INDArray act = Nd4j.zeros(DataType.FLOAT, 1, 48);  // K=48, not divisible by 32
        INDArray packed = Nd4j.zeros(DataType.BYTE, 34);    // placeholder
        Assertions.assertThrows(Exception.class, () -> {
            GgmlQMatMul.exec(act, packed, QUANT_Q8_0, 1L, 48L, GgmlQMatMul.OUTPUT_FLOAT32);
        }, "Should reject K=48 for Q8_0 (blockSize=32)");
    }

    @Test
    void testValidationBadByteLength() {
        // packed weight has wrong number of bytes
        int N = 4, K = 256;
        long expectedBytes = GgmlQMatMul.expectedPackedBytes(QUANT_Q8_0, N, K);
        INDArray act    = Nd4j.zeros(DataType.FLOAT, 1, K);
        INDArray packed = Nd4j.zeros(DataType.BYTE, expectedBytes + 1);  // deliberately wrong
        Assertions.assertThrows(Exception.class, () -> {
            GgmlQMatMul.exec(act, packed, QUANT_Q8_0, N, K, GgmlQMatMul.OUTPUT_FLOAT32);
        }, "Should reject wrong byte buffer length");
    }

    @Test
    void testValidationUnsupportedType() {
        // ggmlType=1 (Q4_1) is not supported by v1
        int N = 4, K = 32;
        INDArray act    = Nd4j.zeros(DataType.FLOAT, 1, K);
        INDArray packed = Nd4j.zeros(DataType.BYTE, N * (K / 32) * 20);  // Q4_1 size
        Assertions.assertThrows(Exception.class, () -> {
            GgmlQMatMul.exec(act, packed, 1 /*Q4_1*/, N, K, GgmlQMatMul.OUTPUT_FLOAT32);
        }, "Should reject unsupported ggmlType=1");
    }

    @Test
    void testValidationBadRank() {
        // rank 1 activations
        INDArray act    = Nd4j.zeros(DataType.FLOAT, 256);
        INDArray packed = Nd4j.zeros(DataType.BYTE, 1 * (256 / 32) * 34);
        Assertions.assertThrows(Exception.class, () -> {
            GgmlQMatMul.exec(act, packed, QUANT_Q8_0, 1L, 256L, GgmlQMatMul.OUTPUT_FLOAT32);
        }, "Should reject rank-1 activations");
    }

    // ─── GgmlQMatMul Java util tests ─────────────────────────────────────────

    @Test
    void testBytesPerBlock() {
        Assertions.assertEquals(34,  GgmlQMatMul.bytesPerBlock(QUANT_Q8_0));
        Assertions.assertEquals(144, GgmlQMatMul.bytesPerBlock(QUANT_Q4_K));
        Assertions.assertEquals(210, GgmlQMatMul.bytesPerBlock(QUANT_Q6_K));
        Assertions.assertEquals(-1,  GgmlQMatMul.bytesPerBlock(99));
    }

    @Test
    void testExpectedPackedBytes() {
        Assertions.assertEquals(34L * (256L / 32L),   GgmlQMatMul.expectedPackedBytes(QUANT_Q8_0, 1, 256));
        Assertions.assertEquals(144L * (512L / 256L), GgmlQMatMul.expectedPackedBytes(QUANT_Q4_K, 1, 512));
        Assertions.assertEquals(210L * (256L / 256L), GgmlQMatMul.expectedPackedBytes(QUANT_Q6_K, 1, 256));
    }

    // ─── Internal test runners ────────────────────────────────────────────────

    private void runQ8_0Test(int M, int K, int N, boolean rank3Act, DataType outDtype) {
        assert K % 32 == 0 : "K must be divisible by 32 for Q8_0";
        Random rng = new Random(42L + M * 100L + K + N);

        int numBlocksPerRow = K / ELEMS_Q8_0;
        long packedLen = (long) N * numBlocksPerRow * BLOCK_Q8_0;
        byte[] packed = new byte[(int) packedLen];

        // Reference W [N][K]
        float[][] W = new float[N][K];

        for (int n = 0; n < N; n++) {
            for (int b = 0; b < numBlocksPerRow; b++) {
                float d   = (rng.nextFloat() * 0.5f + 0.1f) * (rng.nextBoolean() ? 1f : -1f);
                int[] qs  = new int[32];
                for (int i = 0; i < 32; i++) qs[i] = (byte)(rng.nextInt(256) - 128);
                int blockOffset = (n * numBlocksPerRow + b) * BLOCK_Q8_0;
                buildQ8_0Block(packed, blockOffset, d, qs);
                // Reference dequant
                float[] wDst = new float[32];
                dequantQ8_0Block(packed, blockOffset, wDst);
                System.arraycopy(wDst, 0, W[n], b * 32, 32);
            }
        }

        float[][] A = makeRandomActivations(M, K, rng);
        float[][] refOut = referenceMatMul(A, W);

        runAndCompare(packed, A, W, refOut, M, K, N, QUANT_Q8_0, rank3Act, outDtype, 1e-3f);
    }

    private void runQ4KTest(int M, int K, int N, boolean rank3Act, DataType outDtype) {
        assert K % 256 == 0 : "K must be divisible by 256 for Q4_K";
        Random rng = new Random(77L + M * 100L + K + N);

        int numBlocksPerRow = K / ELEMS_Q4_K;
        long packedLen = (long) N * numBlocksPerRow * BLOCK_Q4_K;
        byte[] packed = new byte[(int) packedLen];

        float[][] W = new float[N][K];

        for (int n = 0; n < N; n++) {
            for (int b = 0; b < numBlocksPerRow; b++) {
                float d    = rng.nextFloat() * 0.3f + 0.05f;
                float dmin = rng.nextFloat() * 0.1f + 0.01f;
                // 8 scale values, each 0..15 (keep small to avoid overflow in 6-bit packing)
                int[] sc = new int[8];
                for (int i = 0; i < 8; i++) sc[i] = rng.nextInt(16);
                // 256 nibbles, each 0..15
                int[] nibbles = new int[256];
                for (int i = 0; i < 256; i++) nibbles[i] = rng.nextInt(16);

                int blockOffset = (n * numBlocksPerRow + b) * BLOCK_Q4_K;
                buildQ4_KBlock(packed, blockOffset, d, dmin, sc, nibbles);
                float[] wDst = new float[256];
                dequantQ4_KBlock(packed, blockOffset, wDst);
                System.arraycopy(wDst, 0, W[n], b * 256, 256);
            }
        }

        float[][] A = makeRandomActivations(M, K, rng);
        float[][] refOut = referenceMatMul(A, W);

        // Q4_K is only 4-bit precision; allow slightly larger tolerance
        runAndCompare(packed, A, W, refOut, M, K, N, QUANT_Q4_K, rank3Act, outDtype, 2e-3f);
    }

    private void runQ6KTest(int M, int K, int N, boolean rank3Act, DataType outDtype) {
        assert K % 256 == 0 : "K must be divisible by 256 for Q6_K";
        Random rng = new Random(13L + M * 100L + K + N);

        int numBlocksPerRow = K / ELEMS_Q6_K;
        long packedLen = (long) N * numBlocksPerRow * BLOCK_Q6_K;
        byte[] packed = new byte[(int) packedLen];

        float[][] W = new float[N][K];

        for (int n = 0; n < N; n++) {
            for (int b = 0; b < numBlocksPerRow; b++) {
                float d = rng.nextFloat() * 0.2f + 0.05f;
                // 16 int8 scales in [-4, 4] (small to keep numerical stability)
                int[] scales = new int[16];
                for (int i = 0; i < 16; i++) scales[i] = rng.nextInt(9) - 4;
                // 256 6-bit values in [-32, 31] → stored as value+32 → [0..63]
                int[] vals = new int[256];
                for (int i = 0; i < 256; i++) vals[i] = rng.nextInt(64);  // stored form (0..63)

                int blockOffset = (n * numBlocksPerRow + b) * BLOCK_Q6_K;
                // Clear the block first (buildQ6_KBlock ORs into existing bytes)
                Arrays.fill(packed, blockOffset, blockOffset + BLOCK_Q6_K, (byte) 0);
                buildQ6_KBlock(packed, blockOffset, d, scales, vals);
                float[] wDst = new float[256];
                dequantQ6_KBlock(packed, blockOffset, wDst);
                System.arraycopy(wDst, 0, W[n], b * 256, 256);
            }
        }

        float[][] A = makeRandomActivations(M, K, rng);
        float[][] refOut = referenceMatMul(A, W);

        runAndCompare(packed, A, W, refOut, M, K, N, QUANT_Q6_K, rank3Act, outDtype, 1e-3f);
    }

    private float[][] makeRandomActivations(int M, int K, Random rng) {
        float[][] A = new float[M][K];
        for (int m = 0; m < M; m++)
            for (int k = 0; k < K; k++)
                A[m][k] = (rng.nextFloat() - 0.5f) * 2.0f;
        return A;
    }

    private void runAndCompare(byte[] packed, float[][] A, float[][] W, float[][] refOut,
                               int M, int K, int N, int quantType,
                               boolean rank3Act, DataType outDtype, float tolRel) {

        // Build packed INDArray (INT8)
        INDArray packedArr = Nd4j.create(DataType.BYTE, packed.length);
        for (int i = 0; i < packed.length; i++) packedArr.putScalar(i, packed[i]);

        // Build activation INDArray
        INDArray actArr;
        if (rank3Act) {
            // Shape [1, M, K]
            float[] flat = new float[M * K];
            for (int m = 0; m < M; m++) System.arraycopy(A[m], 0, flat, m * K, K);
            actArr = Nd4j.create(flat, new long[]{1, M, K}, DataType.FLOAT);
        } else {
            float[] flat = new float[M * K];
            for (int m = 0; m < M; m++) System.arraycopy(A[m], 0, flat, m * K, K);
            actArr = Nd4j.create(flat, new long[]{M, K}, DataType.FLOAT);
        }

        int outputDtype = (outDtype == DataType.HALF) ? GgmlQMatMul.OUTPUT_FLOAT16 : GgmlQMatMul.OUTPUT_FLOAT32;

        // Execute ggml_qmatmul
        INDArray result = GgmlQMatMul.exec(actArr, packedArr, quantType, N, K, outputDtype);

        // Check output shape
        long[] expectedShape;
        if (rank3Act) {
            expectedShape = new long[]{1, M, N};
        } else {
            expectedShape = new long[]{M, N};
        }
        Assertions.assertArrayEquals(expectedShape, result.shape(),
            "Output shape mismatch");
        Assertions.assertEquals(outDtype, result.dataType(), "Output dtype mismatch");

        // Compare against reference
        INDArray resultF32 = result.castTo(DataType.FLOAT);
        float maxAbsErr = 0f, maxRelErr = 0f;
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                float expected = refOut[m][n];
                float actual;
                if (rank3Act) {
                    actual = resultF32.getFloat(0, m, n);
                } else {
                    actual = resultF32.getFloat(m, n);
                }
                float absErr = Math.abs(actual - expected);
                float scale  = Math.max(Math.abs(expected), 1e-6f);
                float relErr = absErr / scale;
                maxAbsErr = Math.max(maxAbsErr, absErr);
                maxRelErr = Math.max(maxRelErr, relErr);
            }
        }

        float effectiveTol = (outDtype == DataType.HALF) ? tolRel * 100f : tolRel;
        log.debug("quantType={} M={} K={} N={} rank3={} outDtype={}: maxRelErr={} maxAbsErr={}",
            quantType, M, K, N, rank3Act, outDtype, maxRelErr, maxAbsErr);

        Assertions.assertTrue(maxRelErr <= effectiveTol,
            String.format("Max relative error %.6f exceeds tolerance %.6f (quantType=%d M=%d K=%d N=%d outDtype=%s maxAbsErr=%.6f)",
                maxRelErr, effectiveTol, quantType, M, K, N, outDtype, maxAbsErr));
    }
}
