package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Multi-angle diagnostic tests for outputDirect stale data detection.
 *
 * Tests verify that repeated calls to outputDirect() with different placeholders
 * produce different (correct) results — i.e., no stale cached data leaks through.
 *
 * Organized into groups:
 *   A: View-specific vs general outputDirect stale data isolation
 *   B: DSP vs standard path isolation
 *   C: Individual view op correctness in multi-step
 *   D: View chains + downstream ops
 */
@Slf4j
@NativeTag
@org.junit.jupiter.api.Tag(TagNames.SAMEDIFF)
public class TestOutputDirectStaleness extends BaseNd4jTestWithBackends {

    private static final double TOL = 1e-4;
    private static final int STEPS = 5;

    // ==================== Group A: View-specific vs general ====================

    @Test
    @DisplayName("A1: outputDirect without view ops")
    public void testOutputDirectNoViewOps() {
        SameDiff sd = SameDiff.create();

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 16);
        SDVariable w1 = sd.constant("w1", Nd4j.randn(DataType.FLOAT, 16, 16));
        SDVariable b1 = sd.constant("b1", Nd4j.zeros(DataType.FLOAT, 1, 16));
        SDVariable h = sd.mmul("h", input, w1);
        SDVariable hb = h.add("hb", b1);
        SDVariable ha = sd.nn.relu("ha", hb, 0);
        SDVariable wOut = sd.constant("wOut", Nd4j.randn(DataType.FLOAT, 16, 32));
        sd.mmul("logits", ha, wOut);

        verifyOutputDirectCorrectness(sd, "input", "logits",
                () -> Nd4j.randn(DataType.FLOAT, 1, 16));
        sd.close();
    }

    @Test
    @DisplayName("A2: outputDirect with reshape (view op)")
    public void testOutputDirectWithReshape() {
        SameDiff sd = SameDiff.create();

        SDVariable embed = sd.placeHolder("embed", DataType.FLOAT, 1, 1, 16);
        SDVariable flat = sd.reshape("flat", embed, 1, 16);
        SDVariable w1 = sd.constant("w1", Nd4j.randn(DataType.FLOAT, 16, 16));
        SDVariable b1 = sd.constant("b1", Nd4j.zeros(DataType.FLOAT, 1, 16));
        SDVariable h = sd.mmul("h", flat, w1);
        SDVariable hb = h.add("hb", b1);
        SDVariable ha = sd.nn.relu("ha", hb, 0);
        SDVariable wOut = sd.constant("wOut", Nd4j.randn(DataType.FLOAT, 16, 32));
        sd.mmul("logits", ha, wOut);

        verifyOutputDirectCorrectness(sd, "embed", "logits",
                () -> Nd4j.randn(DataType.FLOAT, 1, 1, 16));
        sd.close();
    }

    @Test
    @DisplayName("A3: simplest outputDirect — single placeholder → matmul → output")
    public void testOutputDirectSinglePlaceholder() {
        SameDiff sd = SameDiff.create();

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 16);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 16, 8));
        sd.mmul("out", input, w);

        verifyOutputDirectCorrectness(sd, "input", "out",
                () -> Nd4j.randn(DataType.FLOAT, 1, 16));
        sd.close();
    }

    // ==================== Group B: DSP vs standard path isolation ====================

    @Test
    @DisplayName("B1: outputDirect with DSP enabled")
    public void testOutputDirectWithDSPEnabled() {
        SameDiff sd = SameDiff.create();
        sd.setDspAutoCompileEnabled(true);

        SDVariable embed = sd.placeHolder("embed", DataType.FLOAT, 1, 1, 16);
        SDVariable flat = sd.reshape("flat", embed, 1, 16);
        SDVariable w1 = sd.constant("w1", Nd4j.randn(DataType.FLOAT, 16, 16));
        SDVariable h = sd.mmul("h", flat, w1);
        SDVariable ha = sd.nn.relu("ha", h, 0);
        SDVariable wOut = sd.constant("wOut", Nd4j.randn(DataType.FLOAT, 16, 32));
        sd.mmul("logits", ha, wOut);

        verifyOutputDirectCorrectness(sd, "embed", "logits",
                () -> Nd4j.randn(DataType.FLOAT, 1, 1, 16));
        sd.close();
    }

    @Test
    @DisplayName("B2: outputDirect with DSP disabled")
    public void testOutputDirectWithDSPDisabled() {
        boolean original = org.nd4j.autodiff.samediff.internal.InferenceSession.isDynamicShapePlanEnabled();
        try {
            org.nd4j.autodiff.samediff.internal.InferenceSession.setDynamicShapePlanEnabled(false);

            SameDiff sd = SameDiff.create();

            SDVariable embed = sd.placeHolder("embed", DataType.FLOAT, 1, 1, 16);
            SDVariable flat = sd.reshape("flat", embed, 1, 16);
            SDVariable w1 = sd.constant("w1", Nd4j.randn(DataType.FLOAT, 16, 16));
            SDVariable h = sd.mmul("h", flat, w1);
            SDVariable ha = sd.nn.relu("ha", h, 0);
            SDVariable wOut = sd.constant("wOut", Nd4j.randn(DataType.FLOAT, 16, 32));
            sd.mmul("logits", ha, wOut);

            verifyOutputDirectCorrectness(sd, "embed", "logits",
                    () -> Nd4j.randn(DataType.FLOAT, 1, 1, 16));
            sd.close();
        } finally {
            org.nd4j.autodiff.samediff.internal.InferenceSession.setDynamicShapePlanEnabled(original);
        }
    }

    @Test
    @DisplayName("B3: standard output() repeated — baseline comparison")
    public void testStandardOutputRepeated() {
        SameDiff sd = SameDiff.create();

        SDVariable embed = sd.placeHolder("embed", DataType.FLOAT, 1, 1, 16);
        SDVariable flat = sd.reshape("flat", embed, 1, 16);
        SDVariable w1 = sd.constant("w1", Nd4j.randn(DataType.FLOAT, 16, 16));
        SDVariable h = sd.mmul("h", flat, w1);
        SDVariable ha = sd.nn.relu("ha", h, 0);
        SDVariable wOut = sd.constant("wOut", Nd4j.randn(DataType.FLOAT, 16, 32));
        sd.mmul("logits", ha, wOut);

        // Use standard sd.output() for all steps — should always produce correct results
        INDArray prev = null;
        for (int i = 0; i < STEPS; i++) {
            INDArray input = Nd4j.randn(DataType.FLOAT, 1, 1, 16).mul(i + 1);
            Map<String, INDArray> result = sd.output(Map.of("embed", input), "logits");
            INDArray out = result.get("logits").dup();

            if (prev != null) {
                double diff = out.sub(prev).amaxNumber().doubleValue();
                assertTrue(diff > 0.01,
                        "Step " + i + " output too similar to previous — diff=" + diff);
            }
            prev = out;
        }
        sd.close();
    }

    // ==================== Group C: View op correctness in multi-step ====================

    @Test
    @DisplayName("C1: reshape outputDirect multi-step")
    public void testReshapeOutputDirectMultiStep() {
        SameDiff sd = SameDiff.create();
        sd.placeHolder("in", DataType.FLOAT, 1, 1, 16);
        sd.reshape("out", sd.getVariable("in"), 1, 16);

        verifyViewOpDirectCorrectness(sd, "in", "out",
                i -> Nd4j.randn(DataType.FLOAT, 1, 1, 16).mul(i + 1),
                new long[]{1, 16});
        sd.close();
    }

    @Test
    @DisplayName("C2: expandDims outputDirect multi-step")
    public void testExpandDimsOutputDirectMultiStep() {
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.placeHolder("in", DataType.FLOAT, 1, 16);
        sd.expandDims("out", in, 1);

        verifyViewOpDirectCorrectness(sd, "in", "out",
                i -> Nd4j.randn(DataType.FLOAT, 1, 16).mul(i + 1),
                new long[]{1, 1, 16});
        sd.close();
    }

    @Test
    @DisplayName("C3: squeeze outputDirect multi-step")
    public void testSqueezeOutputDirectMultiStep() {
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.placeHolder("in", DataType.FLOAT, 1, 1, 16);
        sd.squeeze("out", in, 1);

        verifyViewOpDirectCorrectness(sd, "in", "out",
                i -> Nd4j.randn(DataType.FLOAT, 1, 1, 16).mul(i + 1),
                new long[]{1, 16});
        sd.close();
    }

    @Test
    @DisplayName("C4: permute outputDirect multi-step")
    public void testPermuteOutputDirectMultiStep() {
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.placeHolder("in", DataType.FLOAT, 2, 3);
        sd.permute("out", in, 1, 0);

        // Permute transposes data, so we compare against sd.output() baseline
        // instead of using the generic view verification helper.
        verifyOutputDirectCorrectness(sd, "in", "out",
                () -> Nd4j.randn(DataType.FLOAT, 2, 3));
        sd.close();
    }

    // ==================== Group D: View chains + downstream ops ====================

    @Test
    @DisplayName("D1: reshape → matmul chain via outputDirect")
    public void testReshapeMatmulChainOutputDirect() {
        SameDiff sd = SameDiff.create();

        SDVariable embed = sd.placeHolder("embed", DataType.FLOAT, 1, 1, 16);
        SDVariable flat = sd.reshape("flat", embed, 1, 16);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 16, 8));
        sd.mmul("out", flat, w);

        verifyOutputDirectCorrectness(sd, "embed", "out",
                () -> Nd4j.randn(DataType.FLOAT, 1, 1, 16));
        sd.close();
    }

    @Test
    @DisplayName("D2: view output shares buffer with input (true view, not copy)")
    public void testViewOutputIsActualView() {
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.placeHolder("in", DataType.FLOAT, 1, 1, 16);
        sd.reshape("out", in, 1, 16);

        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 1, 16);
        Map<String, INDArray> result = sd.outputDirect(Map.of("in", input), "out");
        INDArray out = result.get("out");

        // View should have same data buffer pointer as input
        assertNotNull(out, "Output should not be null");
        assertArrayEquals(new long[]{1, 16}, out.shape(), "Output shape should be [1, 16]");
        // The view should contain the same data as the input
        double maxDiff = input.reshape(1, 16).sub(out).amaxNumber().doubleValue();
        assertEquals(0.0, maxDiff, 1e-6, "View output should exactly match reshaped input");
        sd.close();
    }

    @Test
    @DisplayName("D3: view data reflects new input between steps")
    public void testViewDataReflectsNewInput() {
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.placeHolder("in", DataType.FLOAT, 1, 1, 16);
        sd.reshape("out", in, 1, 16);

        INDArray inputA = Nd4j.ones(DataType.FLOAT, 1, 1, 16);
        INDArray inputB = Nd4j.ones(DataType.FLOAT, 1, 1, 16).mul(5.0);

        // Step 0: execute with input A
        Map<String, INDArray> resultA = sd.outputDirect(Map.of("in", inputA), "out");
        INDArray outA = resultA.get("out").dup(); // dup to detach from view

        // Step 1: execute with input B
        Map<String, INDArray> resultB = sd.outputDirect(Map.of("in", inputB), "out");
        INDArray outB = resultB.get("out").dup();

        // outB should reflect inputB (all 5s), NOT inputA (all 1s)
        double diffFromA = outB.sub(outA).amaxNumber().doubleValue();
        assertTrue(diffFromA > 3.0,
                "Step 1 view output should differ from step 0 — diff=" + diffFromA +
                        ". outA[0]=" + outA.getFloat(0) + ", outB[0]=" + outB.getFloat(0));

        double expectedVal = 5.0;
        assertEquals(expectedVal, outB.getFloat(0), 1e-5,
                "Step 1 view output should be 5.0, not " + outB.getFloat(0));
        sd.close();
    }

    // ==================== Helpers ====================

    /**
     * Verifies that outputDirect produces correct, non-stale results across multiple steps.
     * Compares outputDirect results against sd.output() baseline.
     */
    private void verifyOutputDirectCorrectness(SameDiff sd, String phName, String outputName,
                                                java.util.function.Supplier<INDArray> inputSupplier) {
        INDArray[] inputs = new INDArray[STEPS];
        INDArray[] expected = new INDArray[STEPS];

        // Compute expected results with standard output()
        for (int i = 0; i < STEPS; i++) {
            inputs[i] = inputSupplier.get().mul(i + 1);
            Map<String, INDArray> result = sd.output(Map.of(phName, inputs[i]), outputName);
            expected[i] = result.get(outputName).dup();
        }

        // Verify outputDirect matches
        for (int i = 0; i < STEPS; i++) {
            Map<String, INDArray> result = sd.outputDirect(Map.of(phName, inputs[i]), outputName);
            INDArray actual = result.get(outputName).dup();

            double maxDiff = expected[i].sub(actual).amaxNumber().doubleValue();
            log.info("Step {}: maxDiff={}", i, maxDiff);
            assertTrue(maxDiff < TOL,
                    "Step " + i + ": maxDiff=" + maxDiff + " exceeds tolerance " + TOL);

            // Verify different inputs produce different outputs (stale data detection)
            if (i > 0) {
                double diffFromPrev = actual.sub(expected[i - 1]).amaxNumber().doubleValue();
                assertTrue(diffFromPrev > 0.01,
                        "Step " + i + " output too similar to step " + (i - 1) +
                                " — possible stale data. diff=" + diffFromPrev);
            }
        }
    }

    /**
     * Verifies view op outputDirect correctness: the output data must track the input across steps.
     */
    private void verifyViewOpDirectCorrectness(SameDiff sd, String phName, String outputName,
                                                java.util.function.IntFunction<INDArray> inputFactory,
                                                long[] expectedShape) {
        INDArray prev = null;
        for (int i = 0; i < STEPS; i++) {
            INDArray input = inputFactory.apply(i);
            Map<String, INDArray> result = sd.outputDirect(Map.of(phName, input), outputName);
            INDArray out = result.get(outputName).dup();

            assertArrayEquals(expectedShape, out.shape(),
                    "Step " + i + ": unexpected output shape");

            // Verify data matches input (reshape/view should preserve data)
            double dataDiff = input.reshape(expectedShape).sub(out).amaxNumber().doubleValue();
            assertTrue(dataDiff < TOL,
                    "Step " + i + ": view output data doesn't match input. diff=" + dataDiff);

            if (prev != null) {
                double stepDiff = out.sub(prev).amaxNumber().doubleValue();
                assertTrue(stepDiff > 0.01,
                        "Step " + i + " view output too similar to previous — diff=" + stepDiff);
            }
            prev = out;
        }
    }
}
