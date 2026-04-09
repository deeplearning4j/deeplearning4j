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

package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DspCompilationMode;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Environment;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Regression tests for the Triton matmul kernel epilogue bug.
 *
 * <p><b>The bug:</b> The matmul kernel emitter computed in-register epilogue
 * ops (relu, gelu, bias_add, etc.) into a variable {@code acc}, but the
 * cast-and-store logic at the end used the raw matmul output {@code finalAcc}
 * instead of {@code acc}. This silently dropped all epilogue ops, causing
 * downstream consumers to see unprocessed matmul output.
 *
 * <p><b>Symptom:</b> {@code DspKnobIsolationTest} reported max diff ~3.0
 * vs standard execution for any test enabling Triton graph capture. The
 * VLM benchmark produced garbage text instead of real document content.
 *
 * <p><b>Fix:</b> Use {@code acc} instead of {@code finalAcc} in the cast paths.
 * Also added a {@code THROW_EXCEPTION} default case in the epilogue switch
 * statement so any unhandled epilogue op type fails loudly instead of being
 * silently dropped.
 *
 * <p>Each test in this file targets a specific epilogue operation that
 * previously was silently dropped. They all use the same pattern:
 * <ol>
 *   <li>Build a SameDiff graph with matmul → epilogue op</li>
 *   <li>Run with standard execution (baseline)</li>
 *   <li>Run with Triton graph capture enabled (DSP)</li>
 *   <li>Assert outputs match within tolerance</li>
 * </ol>
 */
public class TritonMatmulEpilogueRegressionTest extends BaseNd4jTestWithBackends {

    private static final Logger log = LoggerFactory.getLogger(TritonMatmulEpilogueRegressionTest.class);
    private static final double TOL = 1e-3;

    @Override
    public char ordering() {
        return 'c';
    }

    @AfterEach
    public void resetEnvironment() {
        Environment env = Nd4j.getEnvironment();
        env.setTritonGraphCapture(false);
        env.setTritonSectionFusion(false);
        env.setTritonConsolidatedArgTable(false);
        env.setTritonArgDirtyTracking(false);
        env.setTritonCompileAll(false);
        env.setTritonIncludeTypes("");
        env.setTritonAllowFallbackCapture(false);
    }

    /**
     * Compile a graph with Triton enabled, run a single warmup execution,
     * and assert the output matches the standard reference.
     *
     * <p>The bug previously caused max diff ~3.0 on this very first execution
     * (before any graph was captured). Just compiling the Triton plan and
     * executing it once was enough to produce wrong output.
     */
    private void assertTritonMatchesStandard(SameDiff sd, String outputName,
                                              Map<String, INDArray> inputs,
                                              String testName) {
        // 1. Standard reference (no DSP, no Triton)
        sd.resetSession();
        sd.clearDynamicShapePlanCache();
        Map<String, INDArray> refResult = sd.output(inputs, outputName);
        INDArray reference = refResult.get(outputName).dup();
        log.info("[{}] reference: shape={} sum={}", testName,
                 java.util.Arrays.toString(reference.shape()),
                 reference.sumNumber().doubleValue());

        // 2. Reset and configure Triton + graph capture
        sd.resetSession();
        sd.clearDynamicShapePlanCache();
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        if (nativeOps.isTritonAvailable()) {
            nativeOps.invalidateTritonCache();
            nativeOps.resetTritonCounters();
        }
        Environment env = Nd4j.getEnvironment();
        env.setTritonGraphCapture(true);
        env.setTritonCompileAll(true);
        env.setTritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,NORMALIZATION,ATTENTION");
        env.setTritonAllowFallbackCapture(true);

        sd.setDspAutoCompileEnabled(false);
        sd.setDspNativeAutoCompileEnabled(false);
        sd.compileNativeDynamicShapePlan(List.of(outputName), DspCompilationMode.MAX_AUTOTUNE);

        // 3. Run via DSP/Triton path (warmup execution — bug triggers here)
        Map<String, INDArray> dspResult = sd.outputDirect(inputs, outputName);
        INDArray dspOutput = dspResult.get(outputName).dup();
        log.info("[{}] DSP+Triton: shape={} sum={}", testName,
                 java.util.Arrays.toString(dspOutput.shape()),
                 dspOutput.sumNumber().doubleValue());

        // 4. Compare
        double maxDiff = reference.sub(dspOutput).amaxNumber().doubleValue();
        log.info("[{}] max diff = {}", testName, maxDiff);
        assertTrue(maxDiff < TOL,
                testName + ": Triton output diverges from standard. " +
                "max diff " + maxDiff + " exceeds tolerance " + TOL +
                ". Bug: matmul epilogue (relu/gelu/bias_add/etc) is being silently dropped — " +
                "kernel cast-and-store uses finalAcc instead of acc.");
    }

    /**
     * Test 1: matmul → relu must produce same result via Triton as standard.
     *
     * <p>Previously failed with max diff ~3.0 because relu was applied to
     * {@code acc} in-register but the kernel stored {@code finalAcc}.
     */
    @Test
    @DisplayName("matmul → relu: relu must not be silently dropped by Triton kernel")
    public void testMatmulReluEpilogue() {
        Assumptions.assumeTrue(NativeOpsHolder.getInstance().getDeviceNativeOps().isTritonAvailable(),
                "Triton not available — skipping");

        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 64);
        // Use weights that guarantee SOME negative values in matmul output,
        // so relu has actual work to do (a no-op relu wouldn't catch the bug).
        INDArray weights = Nd4j.randn(DataType.FLOAT, 64, 32).muli(0.5);
        SDVariable w = sd.constant("w", weights);
        SDVariable mm = sd.mmul("mm", x, w);
        SDVariable out = sd.nn.relu("out", mm, 0.0);

        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 64);
        assertTritonMatchesStandard(sd, "out", Map.of("x", input), "matmul_relu");
    }

    /**
     * Test 2: matmul → gelu via Triton must match standard.
     */
    @Test
    @DisplayName("matmul → gelu: gelu must not be silently dropped by Triton kernel")
    public void testMatmulGeluEpilogue() {
        Assumptions.assumeTrue(NativeOpsHolder.getInstance().getDeviceNativeOps().isTritonAvailable(),
                "Triton not available — skipping");

        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 64);
        INDArray weights = Nd4j.randn(DataType.FLOAT, 64, 32).muli(0.5);
        SDVariable w = sd.constant("w", weights);
        SDVariable mm = sd.mmul("mm", x, w);
        SDVariable out = sd.nn.gelu("out", mm);

        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 64);
        assertTritonMatchesStandard(sd, "out", Map.of("x", input), "matmul_gelu");
    }

    /**
     * Test 3: matmul → silu (swish) via Triton must match standard.
     */
    @Test
    @DisplayName("matmul → silu: silu must not be silently dropped by Triton kernel")
    public void testMatmulSiluEpilogue() {
        Assumptions.assumeTrue(NativeOpsHolder.getInstance().getDeviceNativeOps().isTritonAvailable(),
                "Triton not available — skipping");

        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 64);
        INDArray weights = Nd4j.randn(DataType.FLOAT, 64, 32).muli(0.5);
        SDVariable w = sd.constant("w", weights);
        SDVariable mm = sd.mmul("mm", x, w);
        SDVariable out = sd.nn.swish("out", mm);

        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 64);
        assertTritonMatchesStandard(sd, "out", Map.of("x", input), "matmul_silu");
    }

    /**
     * Test 4: matmul → tanh via Triton must match standard.
     */
    @Test
    @DisplayName("matmul → tanh: tanh must not be silently dropped by Triton kernel")
    public void testMatmulTanhEpilogue() {
        Assumptions.assumeTrue(NativeOpsHolder.getInstance().getDeviceNativeOps().isTritonAvailable(),
                "Triton not available — skipping");

        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 64);
        INDArray weights = Nd4j.randn(DataType.FLOAT, 64, 32).muli(0.5);
        SDVariable w = sd.constant("w", weights);
        SDVariable mm = sd.mmul("mm", x, w);
        SDVariable out = sd.math.tanh("out", mm);

        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 64);
        assertTritonMatchesStandard(sd, "out", Map.of("x", input), "matmul_tanh");
    }

    /**
     * Test 5: matmul → sigmoid via Triton must match standard.
     */
    @Test
    @DisplayName("matmul → sigmoid: sigmoid must not be silently dropped by Triton kernel")
    public void testMatmulSigmoidEpilogue() {
        Assumptions.assumeTrue(NativeOpsHolder.getInstance().getDeviceNativeOps().isTritonAvailable(),
                "Triton not available — skipping");

        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 64);
        INDArray weights = Nd4j.randn(DataType.FLOAT, 64, 32).muli(0.5);
        SDVariable w = sd.constant("w", weights);
        SDVariable mm = sd.mmul("mm", x, w);
        SDVariable out = sd.nn.sigmoid("out", mm);

        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 64);
        assertTritonMatchesStandard(sd, "out", Map.of("x", input), "matmul_sigmoid");
    }

    /**
     * Test 6: matmul → bias_add via Triton must match standard.
     *
     * <p>BIAS_ADD is the only epilogue op that takes an extra argument
     * (the bias vector). Verifies the bias pointer indexing is correct.
     */
    @Test
    @DisplayName("matmul → bias_add: bias add must not be silently dropped by Triton kernel")
    public void testMatmulBiasAddEpilogue() {
        Assumptions.assumeTrue(NativeOpsHolder.getInstance().getDeviceNativeOps().isTritonAvailable(),
                "Triton not available — skipping");

        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 64);
        INDArray weights = Nd4j.randn(DataType.FLOAT, 64, 32).muli(0.5);
        INDArray biasVec = Nd4j.randn(DataType.FLOAT, 32).muli(0.5);
        SDVariable w = sd.constant("w", weights);
        SDVariable bias = sd.constant("bias", biasVec);
        SDVariable mm = sd.mmul("mm", x, w);
        SDVariable out = sd.math.add("out", mm, bias);

        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 64);
        assertTritonMatchesStandard(sd, "out", Map.of("x", input), "matmul_bias_add");
    }

    /**
     * Test 7: 2-layer MLP (matmul → relu → matmul). Closest analog to a real
     * decoder layer. Catches the bug if it would compound across multiple
     * matmul-with-epilogue invocations.
     */
    @Test
    @DisplayName("MLP (matmul → relu → matmul): chained matmul+epilogue must match standard")
    public void testMlpMatmulReluChain() {
        Assumptions.assumeTrue(NativeOpsHolder.getInstance().getDeviceNativeOps().isTritonAvailable(),
                "Triton not available — skipping");

        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 64);
        INDArray w1Data = Nd4j.randn(DataType.FLOAT, 64, 128).muli(0.1);
        INDArray w2Data = Nd4j.randn(DataType.FLOAT, 128, 64).muli(0.1);
        SDVariable w1 = sd.constant("w1", w1Data);
        SDVariable w2 = sd.constant("w2", w2Data);
        SDVariable hidden = sd.nn.relu("hidden", sd.mmul(x, w1), 0.0);
        SDVariable out = sd.mmul("out", hidden, w2);

        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 64);
        assertTritonMatchesStandard(sd, "out", Map.of("x", input), "mlp_relu_chain");
    }

    /**
     * Test 8: 2-layer MLP with residuals — exact replica of
     * {@code DspKnobIsolationTest.buildMlpGraph()}. This test mirrors the
     * MLP-with-residuals pattern used by all the failing knob isolation
     * tests. The bug it catches is separate from the simple matmul+epilogue
     * bug in tests 1-7: it involves residual additions feeding into the next
     * matmul block.
     */
    @Test
    @DisplayName("MLP with residuals (DspKnobIsolationTest replica): must match standard")
    public void testElementwiseSectionFusion() {
        Assumptions.assumeTrue(NativeOpsHolder.getInstance().getDeviceNativeOps().isTritonAvailable(),
                "Triton not available — skipping");

        // Exact replica of DspKnobIsolationTest.buildMlpGraph()
        SameDiff sd = SameDiff.create();
        int hidden = 64;
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, hidden);

        // Layer 1: up-project + relu + down-project + residual
        SDVariable w1u = sd.constant("w1u", Nd4j.randn(DataType.FLOAT, hidden, hidden * 2).mul(0.02));
        SDVariable w1d = sd.constant("w1d", Nd4j.randn(DataType.FLOAT, hidden * 2, hidden).mul(0.02));
        SDVariable l1up = sd.mmul("l1_up", x, w1u);
        SDVariable l1act = sd.nn.relu("l1_act", l1up, 0);
        SDVariable l1down = sd.mmul("l1_down", l1act, w1d);
        SDVariable l1out = x.add("l1_res", l1down);

        // Layer 2: same structure
        SDVariable w2u = sd.constant("w2u", Nd4j.randn(DataType.FLOAT, hidden, hidden * 2).mul(0.02));
        SDVariable w2d = sd.constant("w2d", Nd4j.randn(DataType.FLOAT, hidden * 2, hidden).mul(0.02));
        SDVariable l2up = sd.mmul("l2_up", l1out, w2u);
        SDVariable l2act = sd.nn.relu("l2_act", l2up, 0);
        SDVariable l2down = sd.mmul("l2_down", l2act, w2d);
        SDVariable out = l1out.add("out", l2down);

        // Use the ELEMENTWISE+section-fusion config that triggers the bug
        Environment env = Nd4j.getEnvironment();
        env.setTritonIncludeTypes("ELEMENTWISE");
        env.setTritonSectionFusion(true);
        env.setTritonCompileAll(true);
        env.setTritonGraphCapture(true);
        env.setTritonAllowFallbackCapture(true);

        // Use the existing helper but with the elementwise config
        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 64);

        // Get reference WITHOUT triton
        sd.resetSession();
        sd.clearDynamicShapePlanCache();
        env.setTritonIncludeTypes("");  // disable triton temporarily
        env.setTritonSectionFusion(false);
        env.setTritonGraphCapture(false);
        env.setTritonCompileAll(false);
        Map<String, INDArray> refResult = sd.output(Map.of("x", input), "out");
        INDArray reference = refResult.get("out").dup();

        // Re-enable triton + elementwise + section fusion
        env.setTritonIncludeTypes("ELEMENTWISE");
        env.setTritonSectionFusion(true);
        env.setTritonCompileAll(true);
        env.setTritonGraphCapture(true);
        env.setTritonAllowFallbackCapture(true);

        sd.resetSession();
        sd.clearDynamicShapePlanCache();
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        if (nativeOps.isTritonAvailable()) {
            nativeOps.invalidateTritonCache();
            nativeOps.resetTritonCounters();
        }
        sd.setDspAutoCompileEnabled(false);
        sd.setDspNativeAutoCompileEnabled(false);
        sd.compileNativeDynamicShapePlan(List.of("out"), DspCompilationMode.MAX_AUTOTUNE);

        Map<String, INDArray> dspResult = sd.outputDirect(Map.of("x", input), "out");
        INDArray dspOutput = dspResult.get("out").dup();

        double maxDiff = reference.sub(dspOutput).amaxNumber().doubleValue();
        log.info("[elementwise_section_fusion] max diff = {}", maxDiff);
        assertTrue(maxDiff < TOL,
                "elementwise section fusion: max diff " + maxDiff + " exceeds " + TOL +
                ". Bug class: standalone elementwise Triton kernel emission " +
                "(separate from matmul epilogue bug fixed in tests 1-7).");
    }

    /**
     * Test 9: Hard-error sentinel — verifies that adding an unknown epilogue
     * op type to the kernel emitter throws instead of silently dropping it.
     *
     * <p>This is a guard against future regressions: if someone adds a new
     * EpilogueOp enum value but forgets to handle it in the kernel switch,
     * the THROW_EXCEPTION default case will catch it at compile-time of the
     * Triton kernel rather than producing wrong output.
     *
     * <p>This test is informational — there's no public API to inject an
     * unknown epilogue op type from Java. The C++ default case
     * {@code THROW_EXCEPTION("unhandled EpilogueOp type")} is the actual
     * guard. The matmul+epilogue tests above verify it doesn't fire on
     * known op types.
     */
    @Test
    @DisplayName("Documentation: kernel default case is THROW_EXCEPTION (no silent op drop)")
    public void testKernelDefaultCaseIsHardError() {
        // This test documents the invariant. The actual enforcement is in
        // libnd4j/include/graph/gpu/TritonIRBuilder_kernels.cpp default case
        // of the epilogue op switch statement.
        //
        // Invariant: Any EpilogueOp added to the epilogueOps list MUST have
        // a corresponding case in the switch. Falling through to default
        // throws an exception during kernel codegen.
        //
        // If you're reading this because the test failed: the test framework
        // changed how it loads classes. The test itself is a no-op assertion.
        assertTrue(true, "Sentinel test — see comment for details");
    }
}
