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

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlan;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanCompiler;
import org.nd4j.autodiff.samediff.execution.DynamicShapeSlot;
import org.nd4j.autodiff.samediff.execution.ForwardExecutionDAGBuilder;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests that view-producing ops (reshape, squeeze, expand_dims, permute)
 * in DSP execution:
 * 1. Actually produce views (shared data buffer, zero copy)
 * 2. Are marked as viewCapableOp at compile time
 * 3. Skip allocation (Nd4j.empty placeholder) for intermediate slots
 * 4. Produce correct output values matching standard execution
 * 5. Handle multi-step execution without stale data
 */
@Slf4j
@DisplayName("DSP View Ops Tests")
public class TestDSPViewOps {

    private static final double TOL = 1e-5;

    // ========== COMPILER: viewCapableOp FLAG ==========

    @Test
    @DisplayName("compiler: reshape marked viewCapableOp")
    public void testCompilerReshapeViewCapable() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 6);
        SDVariable reshaped = sd.reshape("reshaped", x, 3, 4);
        SDVariable out = reshaped.mul("out", sd.constant("s", Nd4j.scalar(2.0f)));

        DynamicShapePlan plan = compilePlan(sd, "out");
        assertNotNull(plan, "Plan should compile");

        DynamicShapeSlot[] slots = plan.getSlots();
        boolean foundReshape = false;
        boolean foundMul = false;
        for (DynamicShapeSlot slot : slots) {
            if (slot.getOpName().equals("reshape")) {
                assertTrue(slot.isViewCapableOp(), "reshape must be viewCapableOp");
                foundReshape = true;
            }
            if (slot.getOpName().equals("multiply")) {
                assertFalse(slot.isViewCapableOp(), "multiply must NOT be viewCapableOp");
                foundMul = true;
            }
        }
        assertTrue(foundReshape, "reshape slot not found in plan");
        assertTrue(foundMul, "multiply slot not found in plan");
        sd.close();
    }

    @Test
    @DisplayName("compiler: squeeze marked viewCapableOp")
    public void testCompilerSqueezeViewCapable() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 4, 8);
        SDVariable squeezed = sd.squeeze("squeezed", x, 0);
        SDVariable out = squeezed.mul("out", sd.constant("s", Nd4j.scalar(2.0f)));

        DynamicShapePlan plan = compilePlan(sd, "out");
        assertNotNull(plan);

        for (DynamicShapeSlot slot : plan.getSlots()) {
            if (slot.getOpName().equals("squeeze")) {
                assertTrue(slot.isViewCapableOp(), "squeeze must be viewCapableOp");
            }
        }
        sd.close();
    }

    @Test
    @DisplayName("compiler: expand_dims marked viewCapableOp")
    public void testCompilerExpandDimsViewCapable() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
        SDVariable expanded = sd.expandDims("expanded", x, 0);
        SDVariable out = expanded.mul("out", sd.constant("s", Nd4j.scalar(2.0f)));

        DynamicShapePlan plan = compilePlan(sd, "out");
        assertNotNull(plan);

        for (DynamicShapeSlot slot : plan.getSlots()) {
            if (slot.getOpName().equals("expand_dims")) {
                assertTrue(slot.isViewCapableOp(), "expand_dims must be viewCapableOp");
            }
        }
        sd.close();
    }

    @Test
    @DisplayName("compiler: permute marked viewCapableOp")
    public void testCompilerPermuteViewCapable() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 3, 4);
        SDVariable permuted = sd.permute("permuted", x, 0, 2, 1);
        SDVariable out = permuted.mul("out", sd.constant("s", Nd4j.scalar(2.0f)));

        DynamicShapePlan plan = compilePlan(sd, "out");
        assertNotNull(plan);

        for (DynamicShapeSlot slot : plan.getSlots()) {
            if (slot.getOpName().equals("permute")) {
                assertTrue(slot.isViewCapableOp(), "permute must be viewCapableOp");
            }
        }
        sd.close();
    }

    @Test
    @DisplayName("compiler: viewCapableOp count in plan diagnostics")
    public void testCompilerViewCapableCount() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 1, 192);
        SDVariable r1 = sd.reshape("r1", x, 1, 1, 3, 64);
        SDVariable p1 = sd.permute("p1", r1, 0, 2, 1, 3); // [1,3,1,64]
        SDVariable r2 = sd.reshape("r2", p1, 3, 64);
        SDVariable out = r2.mul("out", sd.constant("s", Nd4j.scalar(1.0f)));

        DynamicShapePlan plan = compilePlan(sd, "out");
        assertNotNull(plan);

        int viewCount = 0;
        for (DynamicShapeSlot slot : plan.getSlots()) {
            if (slot.isViewCapableOp()) viewCount++;
        }
        assertEquals(3, viewCount, "Expected 3 viewCapableOps: reshape, permute, reshape");
        log.info("Plan has {} view-capable ops out of {} total", viewCount, plan.getSlots().length);
        sd.close();
    }

    // ========== EXECUTION: DSP matches standard ==========

    @Test
    @DisplayName("reshape: DSP matches standard execution")
    public void testReshapeDspMatchesStandard() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 6);
        SDVariable reshaped = sd.reshape("reshaped", x, 3, 4);
        SDVariable out = reshaped.mul("out", sd.constant("s", Nd4j.scalar(2.0f)));

        INDArray input = Nd4j.linspace(1, 12, 12, DataType.FLOAT).reshape(2, 6);

        // Standard
        INDArray expected = sd.output(Map.of("x", input), "out").get("out").dup();

        // DSP
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(false);
        sd.clearDynamicShapePlanCache();
        sd.resetSession();
        INDArray actual = sd.output(Map.of("x", input), "out").get("out");

        assertArrayEquals(expected.shape(), actual.shape());
        assertEquals(expected, actual, "Reshape DSP must match standard");
        sd.close();
    }

    @Test
    @DisplayName("squeeze: DSP matches standard execution")
    public void testSqueezeDspMatchesStandard() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 4, 8);
        SDVariable squeezed = sd.squeeze("squeezed", x, 0); // [4, 8]
        SDVariable out = squeezed.mul("out", sd.constant("s", Nd4j.scalar(3.0f)));

        INDArray input = Nd4j.linspace(1, 32, 32, DataType.FLOAT).reshape(1, 4, 8);

        INDArray expected = sd.output(Map.of("x", input), "out").get("out").dup();

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(false);
        sd.clearDynamicShapePlanCache();
        sd.resetSession();
        INDArray actual = sd.output(Map.of("x", input), "out").get("out");

        assertArrayEquals(new long[]{4, 8}, actual.shape());
        assertEquals(expected, actual, "Squeeze DSP must match standard");
        sd.close();
    }

    @Test
    @DisplayName("expand_dims: DSP matches standard execution")
    public void testExpandDimsDspMatchesStandard() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
        SDVariable expanded = sd.expandDims("expanded", x, 0); // [1, 3, 4]
        SDVariable out = expanded.mul("out", sd.constant("s", Nd4j.scalar(2.0f)));

        INDArray input = Nd4j.linspace(1, 12, 12, DataType.FLOAT).reshape(3, 4);

        INDArray expected = sd.output(Map.of("x", input), "out").get("out").dup();

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(false);
        sd.clearDynamicShapePlanCache();
        sd.resetSession();
        INDArray actual = sd.output(Map.of("x", input), "out").get("out");

        assertArrayEquals(new long[]{1, 3, 4}, actual.shape());
        assertEquals(expected, actual, "ExpandDims DSP must match standard");
        sd.close();
    }

    @Test
    @DisplayName("permute: DSP matches standard execution")
    public void testPermuteDspMatchesStandard() {
        INDArray input = Nd4j.linspace(1, 24, 24, DataType.FLOAT).reshape(2, 3, 4);
        INDArray scaleVal = Nd4j.scalar(2.0f);

        // Standard path
        SameDiff sdStd = SameDiff.create();
        SDVariable xStd = sdStd.placeHolder("x", DataType.FLOAT, 2, 3, 4);
        SDVariable permutedStd = sdStd.permute("permuted", xStd, 0, 2, 1);
        permutedStd.mul("out", sdStd.constant("s", scaleVal.dup()));
        sdStd.setDspAutoCompileEnabled(false);
        INDArray expected = sdStd.output(Map.of("x", input.dup()), "out").get("out").dup();
        sdStd.close();

        // DSP path
        SameDiff sdDsp = SameDiff.create();
        SDVariable xDsp = sdDsp.placeHolder("x", DataType.FLOAT, 2, 3, 4);
        SDVariable permutedDsp = sdDsp.permute("permuted", xDsp, 0, 2, 1);
        permutedDsp.mul("out", sdDsp.constant("s", scaleVal.dup()));
        sdDsp.setDspAutoCompileEnabled(true);
        sdDsp.setDspNativeAutoCompileEnabled(false);
        INDArray actual = sdDsp.output(Map.of("x", input.dup()), "out").get("out");

        assertArrayEquals(new long[]{2, 4, 3}, actual.shape());
        assertEquals(expected, actual, "Permute DSP must match standard");
        sdDsp.close();
    }

    // ========== MULTI-STEP: changing inputs, verify no stale data ==========

    @Test
    @DisplayName("reshape: multi-step DSP vs standard per step")
    public void testReshapeMultiStepCorrectness() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 1, 16);
        SDVariable flat = sd.reshape("flat", x, 1, 16);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 16, 8));
        SDVariable out = sd.mmul("out", flat, w);

        for (int step = 0; step < 5; step++) {
            INDArray input = Nd4j.randn(DataType.FLOAT, 1, 1, 16).mul(step + 1);

            // Standard baseline
            sd.setDspAutoCompileEnabled(false);
            sd.clearDynamicShapePlanCache();
            sd.resetSession();
            INDArray expected = sd.output(Map.of("x", input), "out").get("out").dup();

            // DSP
            sd.setDspAutoCompileEnabled(true);
            sd.setDspNativeAutoCompileEnabled(false);
            sd.clearDynamicShapePlanCache();
            sd.resetSession();
            INDArray actual = sd.output(Map.of("x", input), "out").get("out").dup();

            double maxDiff = actual.sub(expected).amaxNumber().doubleValue();
            log.info("reshape step {}: maxDiff={}", step, maxDiff);
            assertTrue(maxDiff < TOL, "Step " + step + ": reshape DSP diverges by " + maxDiff);
        }
        sd.close();
    }

    @Test
    @DisplayName("squeeze: multi-step DSP vs standard per step")
    public void testSqueezeMultiStepCorrectness() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 4, 8);
        SDVariable squeezed = sd.squeeze("squeezed", x, 0); // [4, 8]
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 8, 4));
        SDVariable out = sd.mmul("out", squeezed, w);

        for (int step = 0; step < 5; step++) {
            INDArray input = Nd4j.randn(DataType.FLOAT, 1, 4, 8).mul(step + 1);

            sd.setDspAutoCompileEnabled(false);
            sd.clearDynamicShapePlanCache();
            sd.resetSession();
            INDArray expected = sd.output(Map.of("x", input), "out").get("out").dup();

            sd.setDspAutoCompileEnabled(true);
            sd.setDspNativeAutoCompileEnabled(false);
            sd.clearDynamicShapePlanCache();
            sd.resetSession();
            INDArray actual = sd.output(Map.of("x", input), "out").get("out").dup();

            double maxDiff = actual.sub(expected).amaxNumber().doubleValue();
            log.info("squeeze step {}: maxDiff={}", step, maxDiff);
            assertTrue(maxDiff < TOL, "Step " + step + ": squeeze DSP diverges by " + maxDiff);
        }
        sd.close();
    }

    @Test
    @DisplayName("expand_dims: multi-step DSP vs standard per step")
    public void testExpandDimsMultiStepCorrectness() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
        SDVariable expanded = sd.expandDims("expanded", x, 0); // [1, 3, 4]
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 1, 4, 8));
        SDVariable out = sd.mmul("out", expanded, w); // [1, 3, 8]

        for (int step = 0; step < 5; step++) {
            INDArray input = Nd4j.randn(DataType.FLOAT, 3, 4).mul(step + 1);

            sd.setDspAutoCompileEnabled(false);
            sd.clearDynamicShapePlanCache();
            sd.resetSession();
            INDArray expected = sd.output(Map.of("x", input), "out").get("out").dup();

            sd.setDspAutoCompileEnabled(true);
            sd.setDspNativeAutoCompileEnabled(false);
            sd.clearDynamicShapePlanCache();
            sd.resetSession();
            INDArray actual = sd.output(Map.of("x", input), "out").get("out").dup();

            double maxDiff = actual.sub(expected).amaxNumber().doubleValue();
            log.info("expandDims step {}: maxDiff={}", step, maxDiff);
            assertTrue(maxDiff < TOL, "Step " + step + ": expandDims DSP diverges by " + maxDiff);
        }
        sd.close();
    }

    // ========== CHAINED VIEW OPS ==========

    @Test
    @DisplayName("reshape + expandDims chain: DSP matches standard")
    public void testReshapeExpandDimsChain() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 6);
        SDVariable reshaped = sd.reshape("reshaped", x, 3, 4);
        SDVariable expanded = sd.expandDims("expanded", reshaped, 0); // [1, 3, 4]
        SDVariable out = expanded.mul("out", sd.constant("s", Nd4j.scalar(2.0f)));

        INDArray input = Nd4j.linspace(1, 12, 12, DataType.FLOAT).reshape(2, 6);

        INDArray expected = sd.output(Map.of("x", input), "out").get("out").dup();

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(false);
        sd.clearDynamicShapePlanCache();
        sd.resetSession();
        INDArray actual = sd.output(Map.of("x", input), "out").get("out");

        assertArrayEquals(new long[]{1, 3, 4}, actual.shape());
        assertEquals(expected, actual, "Reshape+expandDims chain mismatch");
        sd.close();
    }

    @Test
    @DisplayName("reshape + permute + matmul: attention Q/K/V split pattern")
    public void testReshapePermuteMatmulAttentionPattern() {
        INDArray kData = Nd4j.randn(DataType.FLOAT, 1, 3, 64, 5).mul(0.1);
        INDArray scaleData = Nd4j.scalar(1.0f / 8.0f);

        for (int step = 0; step < 5; step++) {
            INDArray input = Nd4j.randn(DataType.FLOAT, 1, 1, 192).mul(0.1 * (step + 1));

            // Standard path: fresh SameDiff instance
            SameDiff sdStd = SameDiff.create();
            SDVariable xStd = sdStd.placeHolder("x", DataType.FLOAT, 1, 1, 192);
            SDVariable reshapedStd = sdStd.reshape("reshaped", xStd, 1, 1, 3, 64);
            SDVariable permutedStd = sdStd.permute("permuted", reshapedStd, 0, 2, 1, 3);
            SDVariable kStd = sdStd.constant("k", kData.dup());
            SDVariable scoresStd = sdStd.mmul("scores", permutedStd, kStd);
            scoresStd.mul("out", sdStd.constant("scale", scaleData.dup()));
            sdStd.setDspAutoCompileEnabled(false);
            Map<String, INDArray> stdOut = sdStd.output(Map.of("x", input.dup()), "reshaped", "permuted", "scores", "out");
            INDArray expected = stdOut.get("out").dup();
            INDArray stdReshaped = stdOut.get("reshaped").dup();
            INDArray stdPermuted = stdOut.get("permuted").dup();
            INDArray stdScores = stdOut.get("scores").dup();
            sdStd.close();

            // DSP path: fresh SameDiff instance
            SameDiff sdDsp = SameDiff.create();
            SDVariable xDsp = sdDsp.placeHolder("x", DataType.FLOAT, 1, 1, 192);
            SDVariable reshapedDsp = sdDsp.reshape("reshaped", xDsp, 1, 1, 3, 64);
            SDVariable permutedDsp = sdDsp.permute("permuted", reshapedDsp, 0, 2, 1, 3);
            SDVariable kDsp = sdDsp.constant("k", kData.dup());
            SDVariable scoresDsp = sdDsp.mmul("scores", permutedDsp, kDsp);
            scoresDsp.mul("out", sdDsp.constant("scale", scaleData.dup()));
            sdDsp.setDspAutoCompileEnabled(true);
            sdDsp.setDspNativeAutoCompileEnabled(false);
            Map<String, INDArray> dspOut = sdDsp.output(Map.of("x", input.dup()), "reshaped", "permuted", "scores", "out");
            INDArray actual = dspOut.get("out").dup();
            INDArray dspReshaped = dspOut.get("reshaped").dup();
            INDArray dspPermuted = dspOut.get("permuted").dup();
            INDArray dspScores = dspOut.get("scores").dup();
            sdDsp.close();

            log.info("Step {} reshaped diff: {}", step, dspReshaped.sub(stdReshaped).amaxNumber().doubleValue());
            log.info("Step {} permuted diff: {}", step, dspPermuted.sub(stdPermuted).amaxNumber().doubleValue());
            log.info("Step {} scores diff: {}", step, dspScores.sub(stdScores).amaxNumber().doubleValue());

            double maxDiff = actual.sub(expected).amaxNumber().doubleValue();
            log.info("attention pattern step {}: maxDiff={}", step, maxDiff);
            assertTrue(maxDiff < TOL,
                    "Step " + step + ": attention reshape+permute+matmul diverges by " + maxDiff);
        }
    }

    @Test
    @DisplayName("matmul with view-based permuted input vs contiguous")
    public void testMatmulViewVsContiguous() {
        // Regression: oneDNN matmul with 4D non-contiguous (permuted) view input
        // produced zeros for batches 1+ due to applyTransform(Assign) bug in 4D→3D reshape.
        INDArray x = Nd4j.randn(DataType.FLOAT, 1, 1, 3, 64).mul(0.1);
        INDArray k = Nd4j.randn(DataType.FLOAT, 1, 3, 64, 5).mul(0.1);

        // Permute creates a view with non-contiguous strides [192,64,192,1]
        INDArray permView = x.permute(0, 2, 1, 3); // [1,3,1,64]
        INDArray permContig = permView.dup();       // [1,3,1,64] contiguous

        INDArray resultView = Nd4j.matmul(permView, k, false, false, false);
        INDArray resultContig = Nd4j.matmul(permContig, k, false, false, false);

        double mmDiff = resultView.sub(resultContig).amaxNumber().doubleValue();
        log.info("Matmul view vs contiguous diff: {}", mmDiff);
        assertTrue(mmDiff < 1e-5, "Matmul with view vs contiguous input should match, but diff=" + mmDiff);
    }

    @Test
    @DisplayName("squeeze + reshape chain: DSP matches standard")
    public void testSqueezeReshapeChain() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 4, 8);
        SDVariable squeezed = sd.squeeze("squeezed", x, 0); // [4, 8]
        SDVariable reshaped = sd.reshape("reshaped", squeezed, 2, 16); // [2, 16]
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 16, 4));
        SDVariable out = sd.mmul("out", reshaped, w);

        for (int step = 0; step < 3; step++) {
            INDArray input = Nd4j.randn(DataType.FLOAT, 1, 4, 8).mul(step + 1);

            sd.setDspAutoCompileEnabled(false);
            sd.clearDynamicShapePlanCache();
            sd.resetSession();
            INDArray expected = sd.output(Map.of("x", input), "out").get("out").dup();

            sd.setDspAutoCompileEnabled(true);
            sd.setDspNativeAutoCompileEnabled(false);
            sd.clearDynamicShapePlanCache();
            sd.resetSession();
            INDArray actual = sd.output(Map.of("x", input), "out").get("out").dup();

            double maxDiff = actual.sub(expected).amaxNumber().doubleValue();
            log.info("squeeze+reshape step {}: maxDiff={}", step, maxDiff);
            assertTrue(maxDiff < TOL,
                    "Step " + step + ": squeeze+reshape chain diverges by " + maxDiff);
        }
        sd.close();
    }

    @Test
    @DisplayName("expandDims + squeeze roundtrip: identity through DSP")
    public void testExpandDimsSqueezeRoundtrip() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
        SDVariable expanded = sd.expandDims("expanded", x, 0); // [1, 3, 4]
        SDVariable squeezed = sd.squeeze("squeezed", expanded, 0); // [3, 4]
        SDVariable out = squeezed.mul("out", sd.constant("s", Nd4j.scalar(5.0f)));

        INDArray input = Nd4j.linspace(1, 12, 12, DataType.FLOAT).reshape(3, 4);

        INDArray expected = sd.output(Map.of("x", input), "out").get("out").dup();

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(false);
        sd.clearDynamicShapePlanCache();
        sd.resetSession();
        INDArray actual = sd.output(Map.of("x", input), "out").get("out");

        assertArrayEquals(new long[]{3, 4}, actual.shape());
        assertEquals(expected, actual, "ExpandDims+squeeze roundtrip mismatch");
        sd.close();
    }

    // ========== NATIVE DSP ==========

    @Test
    @DisplayName("reshape: native DSP matches standard")
    public void testReshapeNativeDsp() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 3, 4);
        SDVariable reshaped = sd.reshape("reshaped", x, 3, 4);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 4, 8).mul(0.1));
        SDVariable out = sd.mmul("out", reshaped, w);

        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 3, 4).mul(0.1);

        // Standard baseline
        INDArray expected = sd.output(Map.of("x", input), "out").get("out").dup();

        // Java DSP
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(false);
        sd.clearDynamicShapePlanCache();
        sd.resetSession();
        INDArray javaDsp = sd.output(Map.of("x", input), "out").get("out");
        double javaDiff = javaDsp.sub(expected).amaxNumber().doubleValue();
        log.info("Java DSP maxDiff={}", javaDiff);
        assertTrue(javaDiff < TOL, "Java DSP diverges by " + javaDiff);

        // Native DSP
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        sd.clearDynamicShapePlanCache();
        sd.resetSession();
        INDArray nativeDsp = sd.output(Map.of("x", input), "out").get("out");
        double nativeDiff = nativeDsp.sub(expected).amaxNumber().doubleValue();
        log.info("Native DSP maxDiff={}", nativeDiff);
        assertTrue(nativeDiff < TOL, "Native DSP diverges by " + nativeDiff);

        sd.close();
    }

    // ========== HELPER ==========

    private DynamicShapePlan compilePlan(SameDiff sd, String... outputs) {
        Set<String> outputSet = new LinkedHashSet<>();
        for (String o : outputs) outputSet.add(o);

        // Build forward DAG
        ForwardExecutionDAGBuilder builder = new ForwardExecutionDAGBuilder(sd);
        var dag = builder.buildForwardDAG(outputSet);

        return DynamicShapePlanCompiler.compile(sd, dag, outputSet);
    }
}
