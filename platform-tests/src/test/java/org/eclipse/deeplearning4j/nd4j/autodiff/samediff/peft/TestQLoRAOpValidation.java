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

package org.eclipse.deeplearning4j.nd4j.autodiff.samediff.peft;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.validation.OpValidation;
import org.nd4j.autodiff.validation.TestCase;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.GgmlQMatMul;
import org.nd4j.linalg.api.ops.impl.transforms.custom.GgmlQMatMulBp;
import org.nd4j.linalg.api.ops.impl.transforms.custom.GgmlQMatMulLora;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Map;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Op-level validation tests for QLoRA (quantized-base LoRA) ops:
 * <ul>
 *   <li>{@link org.nd4j.linalg.api.ops.impl.transforms.custom.GgmlQMatMulBp} —
 *       backward pass through ggml_qmatmul</li>
 *   <li>{@link GgmlQMatMulLora} — fused quantized-base + LoRA forward; doDiff wiring</li>
 * </ul>
 *
 * <p>Tests use Q8_0 packing (simplest format) so reference dequantization is trivial.
 * Forward parity: GgmlQMatMulLora forward == GgmlQMatMul forward + graph-level delta.
 *
 * <h2>Run</h2>
 * <pre>
 * cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests &amp;&amp; \
 *   /home/agibsonccc/dev-apps/mvn/bin/mvn test -Dbackend.artifactId=nd4j-cuda-12.9 \
 *   -Dtest=TestQLoRAOpValidation 2>&amp;1 | tee /tmp/test-qlora-ops.log
 * </pre>
 */
@Slf4j
@NativeTag
@Tag(TagNames.SAMEDIFF)
@DisplayName("QLoRA Op Validation Tests")
public class TestQLoRAOpValidation
        extends org.eclipse.deeplearning4j.nd4j.autodiff.opvalidation.BaseOpValidation {

    // Q8_0 block constants (must match C++ ggml_qmatmul.h)
    private static final int BLOCK_Q8_0 = 34;   // bytes per block
    private static final int ELEMS_Q8_0 = 32;   // elements per block
    private static final int QUANT_Q8_0 = 4;    // GgmlQuantType enum

    @Override
    public long getTimeoutMilliseconds() { return 120_000L; }

    // ─── Forward parity: GgmlQMatMulLora == GgmlQMatMul + graph-level residual ────

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("GgmlQMatMulLora forward == base + delta (zero-init loraB)")
    public void testGgmlQMatMulLoraForwardEqualsBase(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(42);
        int M = 4, K = 64, N = 8, rank = 2;
        double scaling = 1.0;

        // Build Q8_0 packed weights [N][K]
        byte[] packed = buildQ8_0Packed(N, K, 7777L);
        INDArray packedArr = bytesToINDArray(packed);

        // Activations [M, K]
        INDArray actArr = Nd4j.rand(DataType.FLOAT, M, K).muli(0.1);

        // loraA [rank, K], loraB [N, rank] — zero-init loraB → delta = 0
        INDArray loraAArr = Nd4j.rand(DataType.FLOAT, rank, K).muli(0.01);
        INDArray loraBArr = Nd4j.zeros(DataType.FLOAT, N, rank);

        // Reference: base forward only
        INDArray baseOut = GgmlQMatMul.exec(actArr.dup(), packedArr.dup(),
            QUANT_Q8_0, N, K, GgmlQMatMul.OUTPUT_FLOAT32);

        // Fused op
        INDArray fusedOut = GgmlQMatMulLora.exec(
            actArr.dup(), packedArr.dup(), loraAArr, loraBArr,
            scaling, QUANT_Q8_0, N, K, GgmlQMatMul.OUTPUT_FLOAT32);

        assertArrayEquals(baseOut.shape(), fusedOut.shape(),
            "Shape mismatch between base and fused output");

        // With zero loraB the fused output must equal the base output exactly
        double maxDiff = baseOut.sub(fusedOut).amaxNumber().doubleValue();
        assertTrue(maxDiff < 1e-5,
            "Fused QLoRA with zero loraB should equal base output; maxDiff=" + maxDiff);

        log.info("GgmlQMatMulLora forward parity check passed: maxDiff={}", maxDiff);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("GgmlQMatMulLora forward with non-zero loraB adds correct delta")
    public void testGgmlQMatMulLoraForwardDelta(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(99);
        int M = 3, K = 64, N = 8, rank = 2;
        double scaling = 2.0;

        byte[] packed = buildQ8_0Packed(N, K, 1234L);
        INDArray packedArr = bytesToINDArray(packed);
        INDArray actArr    = Nd4j.rand(DataType.FLOAT, M, K).muli(0.1);
        INDArray loraAArr  = Nd4j.rand(DataType.FLOAT, rank, K).muli(0.01);
        INDArray loraBArr  = Nd4j.rand(DataType.FLOAT, N, rank).muli(0.01);

        // Reference: base + explicit graph-level delta
        INDArray baseOut = GgmlQMatMul.exec(actArr.dup(), packedArr.dup(),
            QUANT_Q8_0, N, K, GgmlQMatMul.OUTPUT_FLOAT32);
        // delta = scaling * (act @ loraA^T) @ loraB^T   [M,N]
        INDArray afterA = actArr.mmul(loraAArr.transpose());  // [M, rank]
        INDArray afterB = afterA.mmul(loraBArr.transpose());  // [M, N]
        INDArray refOut = baseOut.add(afterB.muli(scaling));

        // Fused op
        INDArray fusedOut = GgmlQMatMulLora.exec(
            actArr.dup(), packedArr.dup(), loraAArr, loraBArr,
            scaling, QUANT_Q8_0, N, K, GgmlQMatMul.OUTPUT_FLOAT32);

        assertArrayEquals(refOut.shape(), fusedOut.shape());
        double maxDiff = refOut.sub(fusedOut).amaxNumber().doubleValue();
        assertTrue(maxDiff < 1e-4,
            "GgmlQMatMulLora delta mismatch; maxDiff=" + maxDiff);
        log.info("GgmlQMatMulLora delta forward check passed: maxDiff={}", maxDiff);
    }

    // ─── doDiff graph wiring: loraA and loraB receive non-zero gradients ─────────

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("GgmlQMatMulLora doDiff: loraA and loraB receive gradients in graph")
    public void testGgmlQMatMulLoraGradientWiring(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(55);
        int M = 4, K = 64, N = 8, rank = 2;
        double scaling = 1.0;

        byte[] packed = buildQ8_0Packed(N, K, 999L);

        SameDiff sd = SameDiff.create();
        INDArray actInit   = Nd4j.rand(DataType.FLOAT, M, K).muli(0.1);
        INDArray loraAInit = Nd4j.rand(DataType.FLOAT, rank, K).muli(0.01);
        INDArray loraBInit = Nd4j.zeros(DataType.FLOAT, N, rank);  // zero → initial delta=0
        INDArray packInit  = Nd4j.create(DataType.BYTE, packed.length);
        for (int i = 0; i < packed.length; i++) packInit.putScalar(i, packed[i]);

        SDVariable activations = sd.var("act", actInit);
        SDVariable packedW     = sd.constant("packedW", packInit);
        SDVariable loraA       = sd.var("loraA", loraAInit);
        SDVariable loraB       = sd.var("loraB", loraBInit);

        // Build the ggml_qmatmul_lora op inside the graph
        SDVariable out = new GgmlQMatMulLora(sd, activations, packedW, loraA, loraB,
            scaling, QUANT_Q8_0, N, K, GgmlQMatMul.OUTPUT_FLOAT32).outputVariables()[0];

        // Loss = sum(out)
        SDVariable loss = out.sum();
        sd.setLossVariables(loss.name());

        // Execute forward + backward in one call
        // calculateGradients(placeholders, gradientVarNames...) runs forward then backward
        Map<String, INDArray> grads = sd.calculateGradients(
            null, loraA.name(), loraB.name());

        // loraB starts at zero but must still receive a non-zero gradient because
        // dLoraB = scaling * gradOut^T @ (act @ loraA^T) which is generally non-zero
        INDArray gradLoraB = grads.get(loraB.name());
        assertNotNull(gradLoraB, "gradient of loraB must not be null");
        assertArrayEquals(new long[]{N, rank}, gradLoraB.shape(),
            "gradLoraB shape should be [N, rank]");

        INDArray gradLoraA = grads.get(loraA.name());
        assertNotNull(gradLoraA, "gradient of loraA must not be null");
        assertArrayEquals(new long[]{rank, K}, gradLoraA.shape(),
            "gradLoraA shape should be [rank, K]");

        log.info("GgmlQMatMulLora doDiff wiring: gradLoraB={}, gradLoraA={}",
            gradLoraB.shapeInfoToString(), gradLoraA.shapeInfoToString());
    }

    // ─── Numerical gradient validation ───────────────────────────────────────────

    /**
     * Dequantize a packed [N,K] weight back to a dense FLOAT [N,K] matrix by running the
     * forward ggml_qmatmul against a K×K identity: out[k,n] = dequant(W)[n,k], so
     * dequant(W) = out^T. This reuses the (independently verified) forward kernel as the
     * source of truth for the dequantized values — no duplicated dequant logic in the test.
     */
    private static INDArray dequantizeWeight(INDArray packedArr, int N, int K) {
        INDArray identity = Nd4j.eye(K).castTo(DataType.FLOAT);
        INDArray out = GgmlQMatMul.exec(identity, packedArr.dup(), QUANT_Q8_0, N, K,
            GgmlQMatMul.OUTPUT_FLOAT32);   // [K, N]
        return out.transpose().dup();       // [N, K]
    }

    /**
     * ggml_qmatmul_bp must compute dActivations = gradOut @ dequant(W) exactly.
     * This is the op that makes multi-layer QLoRA backprop possible (the activation
     * gradient flows through the frozen quantized base to the layers below).
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("ggml_qmatmul_bp: dActivations == gradOut @ dequant(W)")
    public void testGgmlQMatMulBpEqualsGradOutTimesDequant(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(31);
        int M = 4, K = 64, N = 8;

        byte[] packed = buildQ8_0Packed(N, K, 4242L);
        INDArray packedArr = bytesToINDArray(packed);
        INDArray act     = Nd4j.rand(DataType.FLOAT, M, K).muli(0.1);
        INDArray gradOut = Nd4j.rand(DataType.FLOAT, M, N).muli(0.5);

        // Reference: gradOut [M,N] @ dequant(W) [N,K] -> [M,K]
        INDArray Wf = dequantizeWeight(packedArr, N, K);
        INDArray dActRef = gradOut.mmul(Wf);

        // Actual native backward op
        INDArray dAct = GgmlQMatMulBp.exec(act, packedArr.dup(), gradOut, QUANT_Q8_0, N, K);

        assertArrayEquals(new long[]{M, K}, dAct.shape(), "dActivations shape must be [M,K]");
        double maxDiff = dActRef.sub(dAct).amaxNumber().doubleValue();
        assertTrue(maxDiff < 1e-3,
            "ggml_qmatmul_bp dActivations must equal gradOut @ dequant(W); maxDiff=" + maxDiff);
        assertEquals(DataType.FLOAT, dAct.dataType(), "ggml_qmatmul_bp should return FP32 activation gradients");
        log.info("ggml_qmatmul_bp numerical check passed: maxDiff={}", maxDiff);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("ggml_qmatmul_bp: HALF activations still produce finite FP32 gradients")
    public void testGgmlQMatMulBpHalfActivationsReturnFiniteFloatGradients(Nd4jBackend backend) {
        int M = 1, K = ELEMS_Q8_0, N = 1;
        byte[] packed = new byte[BLOCK_Q8_0];
        short scale = floatToFp16(1024.0f);
        packed[0] = (byte) (scale & 0xFF);
        packed[1] = (byte) ((scale >> 8) & 0xFF);
        for (int i = 0; i < ELEMS_Q8_0; i++) {
            packed[2 + i] = 127;
        }

        INDArray packedArr = bytesToINDArray(packed);
        INDArray act = Nd4j.zeros(DataType.FLOAT16, M, K);
        INDArray gradOut = Nd4j.ones(DataType.FLOAT16, M, N);

        INDArray dAct = GgmlQMatMulBp.exec(act, packedArr, gradOut, QUANT_Q8_0, N, K);

        assertArrayEquals(new long[]{M, K}, dAct.shape(), "dActivations shape must be [M,K]");
        assertEquals(DataType.FLOAT, dAct.dataType(), "HALF activations must not force HALF BP output");
        assertFalse(dAct.isNaN().any(), "ggml_qmatmul_bp dActivations has NaN");
        assertFalse(dAct.isInfinite().any(), "ggml_qmatmul_bp dActivations has Inf");
        assertTrue(dAct.norm1Number().doubleValue() > 65504.0,
                "regression should exercise values that would overflow a HALF gradient");
    }

    /**
     * The fused ggml_qmatmul_lora backward must produce the same gradients (dAct, dLoraA,
     * dLoraB) as a DENSE reference graph built from the dequantized weight. The dense
     * reference uses only standard, gradcheck-validated matmul/matmul_bp ops, so a bug in
     * either quantized backward (ggml_qmatmul_bp or ggml_qmatmul_lora_bp) cannot hide.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("ggml_qmatmul_lora backward == dense reference gradients")
    public void testGgmlQMatMulLoraGradientsMatchDenseReference(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(77);
        int M = 4, K = 64, N = 8, rank = 3;
        double scaling = 1.5;

        byte[] packed = buildQ8_0Packed(N, K, 5150L);
        INDArray packedArr = bytesToINDArray(packed);
        INDArray actInit   = Nd4j.rand(DataType.FLOAT, M, K).muli(0.1);
        INDArray loraAInit = Nd4j.rand(DataType.FLOAT, rank, K).muli(0.05);
        INDArray loraBInit = Nd4j.rand(DataType.FLOAT, N, rank).muli(0.05);  // non-zero → all grads nontrivial
        INDArray Wf = dequantizeWeight(packedArr, N, K);

        // Dense reference graph (activation trainable so dAct flows through matmul_bp)
        SameDiff ref = SameDiff.create();
        SDVariable rAct   = ref.var("act", actInit.dup());
        SDVariable rWf    = ref.constant("Wf", Wf.dup());
        SDVariable rLoraA = ref.var("loraA", loraAInit.dup());
        SDVariable rLoraB = ref.var("loraB", loraBInit.dup());
        SDVariable rBase   = ref.mmul(rAct, ref.transpose(rWf));             // [M,N]
        SDVariable rAfterA = ref.mmul(rAct, ref.transpose(rLoraA));          // [M,rank]
        SDVariable rAfterB = ref.mmul(rAfterA, ref.transpose(rLoraB));       // [M,N]
        SDVariable rOut    = rBase.add(rAfterB.mul(scaling));
        ref.setLossVariables(rOut.sum().name());
        Map<String, INDArray> refGrads = ref.calculateGradients(
            null, rAct.name(), rLoraA.name(), rLoraB.name());

        // Quantized fused graph
        SameDiff q = SameDiff.create();
        SDVariable qAct    = q.var("act", actInit.dup());
        SDVariable qPacked = q.constant("packedW", packedArr.dup());
        SDVariable qLoraA  = q.var("loraA", loraAInit.dup());
        SDVariable qLoraB  = q.var("loraB", loraBInit.dup());
        SDVariable qOut = new GgmlQMatMulLora(q, qAct, qPacked, qLoraA, qLoraB,
            scaling, QUANT_Q8_0, N, K, GgmlQMatMul.OUTPUT_FLOAT32).outputVariables()[0];
        q.setLossVariables(qOut.sum().name());
        Map<String, INDArray> qGrads = q.calculateGradients(
            null, qAct.name(), qLoraA.name(), qLoraB.name());

        double dAct  = refGrads.get(rAct.name()).sub(qGrads.get(qAct.name())).amaxNumber().doubleValue();
        double dLA   = refGrads.get(rLoraA.name()).sub(qGrads.get(qLoraA.name())).amaxNumber().doubleValue();
        double dLB   = refGrads.get(rLoraB.name()).sub(qGrads.get(qLoraB.name())).amaxNumber().doubleValue();
        assertTrue(dAct < 1e-3, "dActivations mismatch fused vs dense; maxDiff=" + dAct);
        assertTrue(dLA  < 1e-3, "dLoraA mismatch fused vs dense; maxDiff=" + dLA);
        assertTrue(dLB  < 1e-3, "dLoraB mismatch fused vs dense; maxDiff=" + dLB);
        log.info("ggml_qmatmul_lora backward matches dense reference: dAct={}, dLoraA={}, dLoraB={}",
            dAct, dLA, dLB);
    }

    // ─── Utility helpers ─────────────────────────────────────────────────────────

    /**
     * Build a valid Q8_0 packed byte array for a logical [N, K] weight matrix.
     */
    static byte[] buildQ8_0Packed(int N, int K, long seed) {
        assert K % ELEMS_Q8_0 == 0 : "K must be divisible by 32 for Q8_0";
        int numBlocksPerRow = K / ELEMS_Q8_0;
        byte[] packed = new byte[N * numBlocksPerRow * BLOCK_Q8_0];
        Random rng = new Random(seed);
        for (int n = 0; n < N; n++) {
            for (int b = 0; b < numBlocksPerRow; b++) {
                int off = (n * numBlocksPerRow + b) * BLOCK_Q8_0;
                // fp16 scale
                float d = (rng.nextFloat() * 0.4f + 0.1f) * (rng.nextBoolean() ? 1 : -1);
                short dFp16 = floatToFp16(d);
                packed[off]   = (byte)(dFp16 & 0xFF);
                packed[off+1] = (byte)((dFp16 >> 8) & 0xFF);
                // 32 int8 quantized values
                for (int i = 0; i < 32; i++) {
                    packed[off + 2 + i] = (byte)(rng.nextInt(256) - 128);
                }
            }
        }
        return packed;
    }

    static INDArray bytesToINDArray(byte[] bytes) {
        INDArray arr = Nd4j.create(DataType.BYTE, bytes.length);
        for (int i = 0; i < bytes.length; i++) arr.putScalar(i, bytes[i]);
        return arr;
    }

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
}
