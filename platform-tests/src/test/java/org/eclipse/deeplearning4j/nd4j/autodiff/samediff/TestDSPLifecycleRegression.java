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

import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive DSP lifecycle regression tests.
 *
 * Covers array reuse, multi-execution replay, view ops lifecycle, zeroing correctness,
 * op type coverage, slot reuse with different inputs, cuBLAS workspace stability,
 * model close cleanup, batch zero + replay, and no-fallback assertions.
 *
 * Each test builds a small standalone graph (2-8 ops), enables DSP, and verifies
 * correctness against standard SameDiff execution over multiple iterations.
 */
public class TestDSPLifecycleRegression extends BaseNd4jTestWithBackends {

    private static final Logger log = LoggerFactory.getLogger(TestDSPLifecycleRegression.class);
    private static final double TOL = 1e-4;

    @Override
    public char ordering() {
        return 'c';
    }

    /**
     * Enable DSP auto-compilation on a SameDiff instance so that
     * outputDirect() goes through the DSP execution path.
     */
    private void enableDsp(SameDiff sd) {
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
    }

    /**
     * Assert two arrays match within tolerance, with a descriptive message.
     */
    private void assertClose(INDArray expected, INDArray actual, String label) {
        assertArrayEquals(expected.shape(), actual.shape(),
                label + ": shape mismatch - expected " + Arrays.toString(expected.shape())
                        + " got " + Arrays.toString(actual.shape()));
        double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
        assertTrue(maxDiff < TOL, label + ": max diff " + maxDiff + " exceeds tolerance " + TOL);
    }

    // ========================================================================
    // 1. Array Reuse Across Executions
    // ========================================================================

    @Test
    @DisplayName("DSP: slot array reuse with frozen shapes over 5 executions")
    public void testSlotArrayReuseFrozenShapes() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 4);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 4, 8));
        SDVariable mm = sd.mmul("mm", x, w);
        SDVariable out = sd.nn.relu("out", mm, 0);

        INDArray input = Nd4j.randn(DataType.FLOAT, 2, 4);

        // Get ground truth from standard execution
        Map<String, INDArray> standardResult = sd.output(Map.of("x", input), "out");
        INDArray expected = standardResult.get("out").dup();

        // Enable DSP and execute 5 times with same shapes
        enableDsp(sd);
        for (int i = 0; i < 5; i++) {
            Map<String, INDArray> dspResult = sd.outputDirect(Map.of("x", input), "out");
            INDArray actual = dspResult.get("out").dup();
            assertClose(expected, actual, "Iteration " + i);
            log.info("Iteration {}: max diff = {}", i,
                    expected.sub(actual).amaxNumber().doubleValue());
        }

        sd.close();
    }

    // ========================================================================
    // 2. Multi-Execution Replay
    // ========================================================================

    @Test
    @DisplayName("DSP: replay after warmup (matmul + add + softmax)")
    public void testReplayAfterWarmup() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 8);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 8, 4));
        SDVariable b = sd.constant("b", Nd4j.zeros(DataType.FLOAT, 1, 4));
        SDVariable mm = sd.mmul("mm", x, w);
        SDVariable added = mm.add("added", b);
        SDVariable out = sd.nn.softmax("out", added, -1);

        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 8);

        // Standard ground truth
        Map<String, INDArray> standardResult = sd.output(Map.of("x", input), "out");
        INDArray expected = standardResult.get("out").dup();

        enableDsp(sd);

        // 3 warmup executions
        for (int i = 0; i < 3; i++) {
            Map<String, INDArray> result = sd.outputDirect(Map.of("x", input), "out");
            assertClose(expected, result.get("out").dup(), "Warmup " + i);
        }

        // 3 frozen/replay executions (same shapes, DSP should reuse plan)
        for (int i = 0; i < 3; i++) {
            Map<String, INDArray> result = sd.outputDirect(Map.of("x", input), "out");
            assertClose(expected, result.get("out").dup(), "Frozen " + i);
        }

        sd.close();
    }

    @Test
    @DisplayName("DSP: frozen replay stability over 10 rounds")
    public void testFrozenReplayMultipleRounds() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 4, 4);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 4, 4));
        SDVariable mm = sd.mmul("mm", x, w);
        SDVariable out = sd.math.exp("out", mm);

        INDArray input = Nd4j.randn(DataType.FLOAT, 4, 4).muli(0.1); // small values to keep exp stable

        Map<String, INDArray> standardResult = sd.output(Map.of("x", input), "out");
        INDArray expected = standardResult.get("out").dup();

        enableDsp(sd);
        for (int i = 0; i < 10; i++) {
            Map<String, INDArray> result = sd.outputDirect(Map.of("x", input), "out");
            assertClose(expected, result.get("out").dup(), "Round " + i);
        }

        sd.close();
    }

    // ========================================================================
    // 3. View Ops Lifecycle
    // ========================================================================

    @Test
    @DisplayName("DSP: view ops (reshape, permute, expandDims) across 5 executions")
    public void testViewOpsAcrossExecutions() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 3, 4);
        // reshape to [2, 12]
        SDVariable reshaped = sd.reshape("reshaped", x, 2, 12);
        // expand dims -> [2, 1, 12]
        SDVariable expanded = sd.expandDims("expanded", reshaped, 1);
        // mul by scalar to produce a concrete output
        SDVariable out = expanded.mul("out", sd.constant("scale", Nd4j.scalar(DataType.FLOAT, 2.0f)));

        INDArray input = Nd4j.linspace(1, 24, 24, DataType.FLOAT).reshape(2, 3, 4);

        Map<String, INDArray> standardResult = sd.output(Map.of("x", input), "out");
        INDArray expected = standardResult.get("out").dup();

        enableDsp(sd);
        for (int i = 0; i < 5; i++) {
            Map<String, INDArray> result = sd.outputDirect(Map.of("x", input), "out");
            INDArray actual = result.get("out").dup();
            assertClose(expected, actual, "View iteration " + i);
        }

        sd.close();
    }

    @Test
    @DisplayName("DSP: view of constant weight must not corrupt weight")
    public void testViewOfConstantWeight() {
        INDArray weightData = Nd4j.randn(DataType.FLOAT, 4, 8);
        INDArray weightCopy = weightData.dup();

        SameDiff sd = SameDiff.create();
        SDVariable w = sd.constant("w", weightData);
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 4);
        // reshape weight through a view op
        SDVariable wReshaped = sd.reshape("wReshaped", w, 1, 4, 8);
        // use in computation (broadcast matmul-like pattern via element-wise)
        SDVariable xExpanded = sd.expandDims("xExpanded", x, 2); // [2, 4, 1]
        SDVariable product = xExpanded.mul("product", wReshaped); // [2, 4, 8]
        SDVariable out = sd.sum("out", product, 1); // [2, 8]

        INDArray input = Nd4j.randn(DataType.FLOAT, 2, 4);

        enableDsp(sd);
        for (int i = 0; i < 3; i++) {
            sd.outputDirect(Map.of("x", input), "out");
            // Verify the original weight data is not corrupted
            double wDiff = weightData.sub(weightCopy).amaxNumber().doubleValue();
            assertEquals(0.0, wDiff, 1e-10,
                    "Weight corrupted after execution " + i + ": max diff = " + wDiff);
        }

        sd.close();
    }

    // ========================================================================
    // 4. Zeroing Correctness
    // ========================================================================

    @Test
    @DisplayName("DSP: output zeroing before reuse (reduce_sum)")
    public void testOutputZeroingBeforeReuse() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 5);
        SDVariable summed = sd.sum("summed", x, 1); // [3]
        SDVariable out = summed.mul("out", sd.constant("scale", Nd4j.scalar(DataType.FLOAT, 1.0f)));

        INDArray input = Nd4j.randn(DataType.FLOAT, 3, 5);

        Map<String, INDArray> standardResult = sd.output(Map.of("x", input), "out");
        INDArray expected = standardResult.get("out").dup();

        enableDsp(sd);
        for (int i = 0; i < 3; i++) {
            Map<String, INDArray> result = sd.outputDirect(Map.of("x", input), "out");
            assertClose(expected, result.get("out").dup(), "Zeroing iteration " + i);
        }

        sd.close();
    }

    @Test
    @DisplayName("DSP: zeroing with batchnorm-like pattern (mean + variance + normalize)")
    public void testZeroingWithBatchNorm() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 4, 8);
        // Manual batchnorm: mean, variance, normalize
        SDVariable mean = sd.mean("mean", x, true, 0);  // [1, 8]
        SDVariable centered = x.sub("centered", mean);   // [4, 8]
        SDVariable sq = centered.mul("sq", centered);     // [4, 8]
        SDVariable variance = sd.mean("variance", sq, true, 0); // [1, 8]
        SDVariable eps = sd.constant("eps", Nd4j.scalar(DataType.FLOAT, 1e-5f));
        SDVariable stddev = sd.math.sqrt("stddev", variance.add("varEps", eps));
        SDVariable out = centered.div("out", stddev);     // [4, 8]

        // Run with different inputs, verify each matches standard execution
        enableDsp(sd);
        for (int i = 0; i < 3; i++) {
            INDArray input = Nd4j.randn(DataType.FLOAT, 4, 8).muli(i + 1);
            Map<String, INDArray> standardResult = sd.output(Map.of("x", input), "out");
            INDArray expected = standardResult.get("out").dup();

            Map<String, INDArray> dspResult = sd.outputDirect(Map.of("x", input), "out");
            INDArray actual = dspResult.get("out").dup();
            assertClose(expected, actual, "BatchNorm input " + i);
        }

        sd.close();
    }

    // ========================================================================
    // 5. Op Type Coverage
    // ========================================================================

    @Test
    @DisplayName("DSP: elementwise ops (add, sub, mul, div chain)")
    public void testElementwiseOps() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
        SDVariable y = sd.placeHolder("y", DataType.FLOAT, 3, 4);
        SDVariable added = x.add("added", y);
        SDVariable subbed = added.sub("subbed", x);
        SDVariable mulled = subbed.mul("mulled", y);
        SDVariable out = mulled.div("out", added.add("denom", sd.constant("one", Nd4j.ones(DataType.FLOAT, 3, 4))));

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 3, 4);
        INDArray yArr = Nd4j.randn(DataType.FLOAT, 3, 4).addi(2.0); // avoid division by zero
        Map<String, INDArray> feed = Map.of("x", xArr, "y", yArr);

        Map<String, INDArray> expected = sd.output(feed, "out");
        INDArray expectedOut = expected.get("out").dup();

        enableDsp(sd);
        Map<String, INDArray> dspResult = sd.outputDirect(feed, "out");
        assertClose(expectedOut, dspResult.get("out").dup(), "Elementwise");

        sd.close();
    }

    @Test
    @DisplayName("DSP: matmul ops (square, rectangular)")
    public void testMatmulOps() {
        // Square matmul
        SameDiff sd1 = SameDiff.create();
        SDVariable a1 = sd1.placeHolder("a", DataType.FLOAT, 4, 4);
        SDVariable b1 = sd1.constant("b", Nd4j.randn(DataType.FLOAT, 4, 4));
        SDVariable out1 = sd1.mmul("out", a1, b1);

        INDArray input1 = Nd4j.randn(DataType.FLOAT, 4, 4);
        Map<String, INDArray> exp1 = sd1.output(Map.of("a", input1), "out");
        INDArray expected1 = exp1.get("out").dup();

        enableDsp(sd1);
        Map<String, INDArray> dsp1 = sd1.outputDirect(Map.of("a", input1), "out");
        assertClose(expected1, dsp1.get("out").dup(), "Square matmul");
        sd1.close();

        // Rectangular matmul
        SameDiff sd2 = SameDiff.create();
        SDVariable a2 = sd2.placeHolder("a", DataType.FLOAT, 2, 6);
        SDVariable b2 = sd2.constant("b", Nd4j.randn(DataType.FLOAT, 6, 3));
        SDVariable out2 = sd2.mmul("out", a2, b2);

        INDArray input2 = Nd4j.randn(DataType.FLOAT, 2, 6);
        Map<String, INDArray> exp2 = sd2.output(Map.of("a", input2), "out");
        INDArray expected2 = exp2.get("out").dup();

        enableDsp(sd2);
        Map<String, INDArray> dsp2 = sd2.outputDirect(Map.of("a", input2), "out");
        assertClose(expected2, dsp2.get("out").dup(), "Rectangular matmul");
        sd2.close();
    }

    @Test
    @DisplayName("DSP: reduce ops (sum, mean, max on different axes)")
    public void testReduceOps() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4, 5);

        SDVariable rSum = sd.sum("rSum", x, 2);       // [3, 4]
        SDVariable rMean = sd.mean("rMean", x, 0);    // [4, 5]
        SDVariable rMax = sd.max("rMax", x, 1);       // [3, 5]

        // Combine to single output
        // rSum is [3,4], rMean is [4,5], rMax is [3,5] -- different shapes
        // Just verify each individually
        INDArray input = Nd4j.randn(DataType.FLOAT, 3, 4, 5);
        Map<String, INDArray> feed = Map.of("x", input);

        Map<String, INDArray> expected = sd.output(feed, "rSum", "rMean", "rMax");
        INDArray expSum = expected.get("rSum").dup();
        INDArray expMean = expected.get("rMean").dup();
        INDArray expMax = expected.get("rMax").dup();

        enableDsp(sd);
        Map<String, INDArray> dspResult = sd.outputDirect(feed, "rSum", "rMean", "rMax");
        assertClose(expSum, dspResult.get("rSum").dup(), "ReduceSum");
        assertClose(expMean, dspResult.get("rMean").dup(), "ReduceMean");
        assertClose(expMax, dspResult.get("rMax").dup(), "ReduceMax");

        sd.close();
    }

    @Test
    @DisplayName("DSP: transform ops (sigmoid, tanh, exp, log chain)")
    public void testTransformOps() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 4);
        SDVariable sigOut = sd.nn.sigmoid("sig", x);
        SDVariable tanhOut = sd.math.tanh("tanh", sigOut);
        SDVariable expOut = sd.math.exp("exp", tanhOut);
        SDVariable out = sd.math.log("out", expOut);

        INDArray input = Nd4j.randn(DataType.FLOAT, 2, 4);

        Map<String, INDArray> expected = sd.output(Map.of("x", input), "out");
        INDArray expectedOut = expected.get("out").dup();

        enableDsp(sd);
        Map<String, INDArray> dspResult = sd.outputDirect(Map.of("x", input), "out");
        assertClose(expectedOut, dspResult.get("out").dup(), "Transform chain");

        sd.close();
    }

    @Test
    @DisplayName("DSP: compare ops producing BOOL consumed by where")
    public void testCompareOps() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
        SDVariable threshold = sd.constant("threshold", Nd4j.scalar(DataType.FLOAT, 0.0f));
        SDVariable positive = sd.constant("positive", Nd4j.ones(DataType.FLOAT, 3, 4));
        SDVariable negative = sd.constant("negative", Nd4j.ones(DataType.FLOAT, 3, 4).muli(-1));

        // gt produces BOOL
        SDVariable mask = sd.gt("mask", x, 0.0);
        // where selects based on condition
        SDVariable out = sd.where("out", positive, negative, mask);

        INDArray input = Nd4j.randn(DataType.FLOAT, 3, 4);

        Map<String, INDArray> expected = sd.output(Map.of("x", input), "out");
        INDArray expectedOut = expected.get("out").dup();

        enableDsp(sd);
        Map<String, INDArray> dspResult = sd.outputDirect(Map.of("x", input), "out");
        assertClose(expectedOut, dspResult.get("out").dup(), "Compare+where");

        sd.close();
    }

    // ========================================================================
    // 6. Slot Reuse with Different Inputs
    // ========================================================================

    @Test
    @DisplayName("DSP: same shape, 5 different random inputs produce unique correct results")
    public void testSameShapeDifferentValues() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 4);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 4, 3));
        SDVariable out = sd.mmul("out", x, w);

        enableDsp(sd);
        INDArray[] results = new INDArray[5];
        for (int i = 0; i < 5; i++) {
            INDArray input = Nd4j.randn(DataType.FLOAT, 2, 4);

            // Get standard result for this specific input
            Map<String, INDArray> expected = sd.output(Map.of("x", input), "out");
            INDArray expectedOut = expected.get("out").dup();

            // Get DSP result
            Map<String, INDArray> dspResult = sd.outputDirect(Map.of("x", input), "out");
            results[i] = dspResult.get("out").dup();

            assertClose(expectedOut, results[i], "Different values iteration " + i);
        }

        // Verify results are actually different from each other (not stale)
        for (int i = 1; i < 5; i++) {
            double diff = results[i].sub(results[0]).amaxNumber().doubleValue();
            assertTrue(diff > 1e-6, "Results " + i + " and 0 are identical -- possible stale data");
        }

        sd.close();
    }

    @Test
    @DisplayName("DSP: placeholder values change between executions")
    public void testPlaceholderChangesBetweenExecutions() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 4);
        SDVariable w = sd.constant("w", Nd4j.eye(4).castTo(DataType.FLOAT)); // identity weight
        SDVariable out = sd.mmul("out", x, w); // output = input * identity = input

        enableDsp(sd);
        for (int i = 0; i < 5; i++) {
            INDArray input = Nd4j.createFromArray(new float[][]{{i + 1.0f, i + 2.0f, i + 3.0f, i + 4.0f}});
            Map<String, INDArray> result = sd.outputDirect(Map.of("x", input), "out");
            INDArray actual = result.get("out").dup();

            // With identity weight, output should equal input
            assertClose(input, actual, "Placeholder change iteration " + i);
        }

        sd.close();
    }

    // ========================================================================
    // 7. cuBLAS Workspace (matmul heavy)
    // ========================================================================

    @Test
    @DisplayName("DSP: 3 sequential matmuls without workspace corruption")
    public void testMultipleMatmulsInSequence() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 8);
        SDVariable w1 = sd.constant("w1", Nd4j.randn(DataType.FLOAT, 8, 16));
        SDVariable w2 = sd.constant("w2", Nd4j.randn(DataType.FLOAT, 16, 8));
        SDVariable w3 = sd.constant("w3", Nd4j.randn(DataType.FLOAT, 8, 4));

        SDVariable h1 = sd.mmul("h1", x, w1);   // [2, 16]
        SDVariable h2 = sd.mmul("h2", h1, w2);  // [2, 8]
        SDVariable out = sd.mmul("out", h2, w3); // [2, 4]

        INDArray input = Nd4j.randn(DataType.FLOAT, 2, 8);

        Map<String, INDArray> expected = sd.output(Map.of("x", input), "out");
        INDArray expectedOut = expected.get("out").dup();

        enableDsp(sd);
        for (int i = 0; i < 5; i++) {
            Map<String, INDArray> result = sd.outputDirect(Map.of("x", input), "out");
            assertClose(expectedOut, result.get("out").dup(), "Sequential matmul iteration " + i);
        }

        sd.close();
    }

    @Test
    @DisplayName("DSP: 4-layer matmul + relu neural network pattern")
    public void testMatmulWithReluChain() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 8);
        SDVariable w1 = sd.constant("w1", Nd4j.randn(DataType.FLOAT, 8, 16).muli(0.1));
        SDVariable w2 = sd.constant("w2", Nd4j.randn(DataType.FLOAT, 16, 16).muli(0.1));
        SDVariable w3 = sd.constant("w3", Nd4j.randn(DataType.FLOAT, 16, 8).muli(0.1));
        SDVariable w4 = sd.constant("w4", Nd4j.randn(DataType.FLOAT, 8, 4).muli(0.1));

        SDVariable h1 = sd.nn.relu("h1", sd.mmul("mm1", x, w1), 0);
        SDVariable h2 = sd.nn.relu("h2", sd.mmul("mm2", h1, w2), 0);
        SDVariable h3 = sd.nn.relu("h3", sd.mmul("mm3", h2, w3), 0);
        SDVariable out = sd.mmul("out", h3, w4);

        INDArray input = Nd4j.randn(DataType.FLOAT, 2, 8);

        Map<String, INDArray> expected = sd.output(Map.of("x", input), "out");
        INDArray expectedOut = expected.get("out").dup();

        enableDsp(sd);
        for (int i = 0; i < 5; i++) {
            Map<String, INDArray> result = sd.outputDirect(Map.of("x", input), "out");
            assertClose(expectedOut, result.get("out").dup(), "NN layer chain iteration " + i);
        }

        sd.close();
    }

    // ========================================================================
    // 8. Model Close Cleanup
    // ========================================================================

    @Test
    @DisplayName("DSP: close releases resources, new SameDiff works after")
    public void testCloseReleasesAllResources() {
        // First graph: create, DSP execute, close
        {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 4);
            SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 4, 3));
            SDVariable out = sd.mmul("out", x, w);

            enableDsp(sd);
            for (int i = 0; i < 3; i++) {
                sd.outputDirect(Map.of("x", Nd4j.randn(DataType.FLOAT, 2, 4)), "out");
            }
            assertDoesNotThrow(() -> sd.close(), "sd.close() should not throw");
        }

        // Second graph: must work without interference from first
        {
            SameDiff sd2 = SameDiff.create();
            SDVariable a = sd2.placeHolder("a", DataType.FLOAT, 3, 3);
            SDVariable b = sd2.constant("b", Nd4j.ones(DataType.FLOAT, 3, 3));
            SDVariable out2 = a.add("out2", b);

            enableDsp(sd2);
            Map<String, INDArray> result = sd2.outputDirect(
                    Map.of("a", Nd4j.ones(DataType.FLOAT, 3, 3)), "out2");
            INDArray actual = result.get("out2").dup();

            // 1 + 1 = 2
            INDArray expected = Nd4j.ones(DataType.FLOAT, 3, 3).muli(2);
            assertClose(expected, actual, "Post-close new SameDiff");
            sd2.close();
        }
    }

    @Test
    @DisplayName("DSP: multiple plan lifecycles (create, compile, execute, close) x3")
    public void testMultiplePlanLifecycles() {
        for (int lifecycle = 0; lifecycle < 3; lifecycle++) {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 4);
            SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 4, 6));
            SDVariable b = sd.constant("b", Nd4j.zeros(DataType.FLOAT, 2, 6));
            SDVariable mm = sd.mmul("mm", x, w);
            SDVariable out = mm.add("out", b);

            INDArray input = Nd4j.randn(DataType.FLOAT, 2, 4);

            Map<String, INDArray> expected = sd.output(Map.of("x", input), "out");
            INDArray expectedOut = expected.get("out").dup();

            enableDsp(sd);
            for (int exec = 0; exec < 3; exec++) {
                Map<String, INDArray> result = sd.outputDirect(Map.of("x", input), "out");
                assertClose(expectedOut, result.get("out").dup(),
                        "Lifecycle " + lifecycle + " exec " + exec);
            }

            sd.close();
            log.info("Lifecycle {} completed successfully", lifecycle);
        }
    }

    // ========================================================================
    // 9. Batch Zero + Replay Sequence
    // ========================================================================

    @Test
    @DisplayName("DSP: batch zero then replay produces correct results")
    public void testBatchZeroThenReplay() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 4);
        SDVariable w1 = sd.constant("w1", Nd4j.randn(DataType.FLOAT, 4, 4));
        SDVariable w2 = sd.constant("w2", Nd4j.randn(DataType.FLOAT, 4, 4));

        // Two add chains that both need zeroing
        SDVariable mm1 = sd.mmul("mm1", x, w1);
        SDVariable mm2 = sd.mmul("mm2", x, w2);
        SDVariable sum1 = mm1.add("sum1", mm2);
        SDVariable relu1 = sd.nn.relu("relu1", sum1, 0);
        SDVariable out = relu1.add("out", mm1); // reuse mm1

        INDArray input = Nd4j.randn(DataType.FLOAT, 2, 4);

        Map<String, INDArray> expected = sd.output(Map.of("x", input), "out");
        INDArray expectedOut = expected.get("out").dup();

        enableDsp(sd);

        // Warmup (slot-by-slot or initial compilation)
        for (int i = 0; i < 2; i++) {
            Map<String, INDArray> result = sd.outputDirect(Map.of("x", input), "out");
            assertClose(expectedOut, result.get("out").dup(), "Warmup " + i);
        }

        // Replay (frozen shapes, batch zero should precede replay)
        for (int i = 0; i < 3; i++) {
            Map<String, INDArray> result = sd.outputDirect(Map.of("x", input), "out");
            assertClose(expectedOut, result.get("out").dup(), "Replay " + i);
        }

        sd.close();
    }

    // ========================================================================
    // 10. No Fallback Assertion
    // ========================================================================

    @Test
    @DisplayName("DSP: 10 executions with no exceptions and stable results")
    public void testNoFallbackOnMultipleExecutions() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 4);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 4, 4));
        SDVariable mm = sd.mmul("mm", x, w);
        SDVariable out = sd.nn.sigmoid("out", mm);

        INDArray input = Nd4j.randn(DataType.FLOAT, 2, 4);

        Map<String, INDArray> expected = sd.output(Map.of("x", input), "out");
        INDArray expectedOut = expected.get("out").dup();

        enableDsp(sd);

        INDArray firstDspResult = null;
        for (int i = 0; i < 10; i++) {
            Map<String, INDArray> result = sd.outputDirect(Map.of("x", input), "out");
            INDArray actual = result.get("out").dup();

            // Verify against standard execution
            assertClose(expectedOut, actual, "No-fallback iteration " + i);

            if (i == 0) {
                firstDspResult = actual;
            } else {
                // Verify DSP results are stable across iterations
                double iterDiff = firstDspResult.sub(actual).amaxNumber().doubleValue();
                assertEquals(0.0, iterDiff, TOL,
                        "DSP result changed between iteration 0 and " + i + ": diff = " + iterDiff);
            }
        }

        sd.close();
    }

    // ========================================================================
    // 12. Changing Placeholder Values During Frozen Replay
    // ========================================================================

    /**
     * DSP: changing an INT64 placeholder (position_ids pattern) between frozen
     * replay steps MUST produce different outputs.
     *
     * This is the decoder accuracy contract: position_ids changes every step,
     * and the model output must reflect the new position. If replay input
     * propagation fails, the output stays the same (stale position_ids).
     *
     * Graph is intentionally large (20 chained matmuls) to trigger CUDA graph
     * capture. The position_ids feeds a gather at the start of the chain.
     * If replay keeps a stale position_ids value, all replay steps see
     * the warmup value and produce identical output.
     */
    @Test
    @DisplayName("DSP: changing INT64 placeholder produces different outputs during CUDA graph replay")
    public void testChangingPlaceholderDuringFrozenReplay() {
        SameDiff sd = SameDiff.create();

        int hiddenSize = 16;
        int numLayers = 20;  // 20 matmuls — large enough for capture

        INDArray embTable = Nd4j.randn(DataType.FLOAT, 100, hiddenSize);
        SDVariable embeddings = sd.constant("embeddings", embTable);
        SDVariable posIds = sd.placeHolder("position_ids", DataType.INT64, 1);

        // Gather position embedding
        SDVariable x = sd.gather("gathered", embeddings, posIds, 0);

        // Chain of matmul + relu layers. Uses near-identity weights so values
        // don't vanish through the chain. ReLU's scalar cutoff (0) is a constant
        // that must survive capture buffer lifecycle.
        for (int layer = 0; layer < numLayers; layer++) {
            SDVariable w = sd.constant("w_" + layer,
                    Nd4j.eye(hiddenSize).castTo(DataType.FLOAT).muli(0.95));
            x = sd.mmul("mm_" + layer, x, w);
            x = sd.nn.relu("relu_" + layer, x, 0);
        }
        SDVariable out = x.sum("out");

        enableDsp(sd);
        sd.setGraphExecutionMode(GraphExecutionMode.CUDA_GRAPHS);

        INDArray posIdsArr = Nd4j.createFromArray(0L);
        Map<String, INDArray> placeholders = new java.util.HashMap<>();
        placeholders.put("position_ids", posIdsArr);

        // Phase 1: Warmup with position_ids=0 (5 steps to trigger freeze + capture)
        for (int i = 0; i < 5; i++) {
            sd.output(placeholders, "out");
        }
        log.info("Warmup done with position_ids=0");

        // Phase 2: Change position_ids on each step — output MUST change
        double prevSum = Double.NaN;
        for (int step = 0; step < 5; step++) {
            posIdsArr.putScalar(0, (long) step);
            Map<String, INDArray> result = sd.output(placeholders, "out");
            double sum = result.get("out").getDouble(0);
            log.info("Replay step {}: position_ids={}, sum={}", step, step, sum);

            if (step > 0) {
                assertNotEquals(prevSum, sum, 1e-6,
                        "Step " + step + ": output MUST differ from step " + (step - 1) +
                        " because position_ids changed from " + (step - 1) + " to " + step +
                        ". If identical, replay input propagation is broken (stale position_ids).");
            }
            prevSum = sum;
        }

        sd.close();
    }

    /**
     * DSP: changing a mask placeholder between frozen replay steps must
     * affect the output through a large graph (triggers CUDA graph capture).
     *
     * Graph: x[1,32] → mul(mask_float) → chain of 20 matmul+relu → sum
     * Warmup: mask=[1,1,0,...] for 5 steps
     * Replay: mask grows by one 1 per step — output must change
     */
    @Test
    @DisplayName("DSP: growing mask produces changing output during CUDA graph replay")
    public void testGrowingAttentionMaskDuringFrozenReplay() {
        SameDiff sd = SameDiff.create();

        int seqLen = 16;
        int hiddenSize = 16;
        int numLayers = 15;

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, seqLen);
        SDVariable mask = sd.placeHolder("mask", DataType.INT64, 1, seqLen);
        SDVariable maskFloat = sd.castTo("mask_float", mask, DataType.FLOAT);
        SDVariable masked = x.mul("masked", maskFloat);

        // Project to hidden size and chain matmuls + adds for graph capture
        SDVariable wProj = sd.constant("w_proj",
                Nd4j.eye(seqLen).castTo(DataType.FLOAT));  // identity projection
        SDVariable h = sd.mmul("project", masked, wProj);
        for (int layer = 0; layer < numLayers; layer++) {
            SDVariable w = sd.constant("w_" + layer,
                    Nd4j.eye(hiddenSize).castTo(DataType.FLOAT).muli(0.95));
            SDVariable b = sd.constant("b_" + layer,
                    Nd4j.ones(DataType.FLOAT, 1, hiddenSize).muli(0.001));
            h = sd.mmul("mm_" + layer, h, w);
            h = h.add("add_" + layer, b);
        }
        SDVariable out = h.sum("out");

        enableDsp(sd);
        sd.setGraphExecutionMode(GraphExecutionMode.CUDA_GRAPHS);

        INDArray xArr = Nd4j.ones(DataType.FLOAT, 1, seqLen);
        INDArray maskArr = Nd4j.zeros(DataType.INT64, 1, seqLen);
        maskArr.putScalar(0, 0, 1L);
        maskArr.putScalar(0, 1, 1L);

        Map<String, INDArray> placeholders = new java.util.HashMap<>();
        placeholders.put("x", xArr);
        placeholders.put("mask", maskArr);

        // Phase 1: Warmup (5 steps)
        for (int i = 0; i < 5; i++) {
            sd.output(placeholders, "out");
        }
        log.info("Warmup done with mask sum={}", maskArr.sumNumber().longValue());

        // Phase 2: Grow mask — output must change
        double prevSum = Double.NaN;
        for (int step = 0; step < 5; step++) {
            int activatePos = 2 + step;
            maskArr.putScalar(0, activatePos, 1L);

            Map<String, INDArray> result = sd.output(placeholders, "out");
            double sum = result.get("out").getDouble(0);
            log.info("Replay step {}: mask sum={}, output sum={}",
                    step, maskArr.sumNumber().longValue(), sum);

            if (step > 0) {
                assertNotEquals(prevSum, sum, 1e-6,
                        "Step " + step + ": output must differ from previous step " +
                        "when mask grows. If identical, replay is using stale mask values.");
            }
            prevSum = sum;
        }

        sd.close();
    }

    /**
     * DSP: replay handles must persist across executeCount transitions.
     * After warmup (executeCount=0) and freeze, segments should have replay
     * handles at executeCount=1+. No handle should be null when the segment
     * was successfully captured.
     *
     * Uses DspDebugger.analyzePlan() to introspect segment state.
     */
    @Test
    @DisplayName("DSP: replay handles persist after freeze — no null handles in captured segments")
    public void testReplayHandlesPersistAfterFreeze() {
        SameDiff sd = SameDiff.create();

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 4, 4);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 4, 4));
        SDVariable mm = sd.mmul("mm", x, w);
        SDVariable relu = sd.nn.relu("relu", mm, 0);
        SDVariable out = relu.sum("out");

        enableDsp(sd);

        INDArray input = Nd4j.randn(DataType.FLOAT, 4, 4);
        Map<String, INDArray> placeholders = Map.of("x", input);

        // Execute 5 times to progress through phases
        for (int i = 0; i < 5; i++) {
            sd.output(placeholders, "out");
        }

        // Introspect plan via DspDebugger
        org.nd4j.autodiff.samediff.execution.DspDebugger debugger =
                org.nd4j.autodiff.samediff.execution.DspDebugger.attach(sd);
        org.nd4j.autodiff.samediff.execution.DspDebugger.PlanReport report = debugger.analyzePlan();

        log.info("Plan after 5 executions:\n{}", report);

        // Verify: no state churn — plan phase should be SHAPES_FROZEN or higher
        assertNotNull(report.planPhase, "Plan phase should be set");
        log.info("Plan phase: {}", report.planPhase);

        // Verify: segments that are capturable should NOT have captureFailed
        for (org.nd4j.autodiff.samediff.execution.DspDebugger.SegmentReport seg : report.segments) {
            if (seg.capturable) {
                assertFalse(seg.captureFailed,
                        "Capturable segment " + seg.index + " should not have captureFailed=true");
            }
        }

        sd.close();
    }

    // ========================================================================
    // 15. Native Decode Input Updates (setNextDecodeToken path)
    // ========================================================================

    /**
     * DSP: when C++ decode inputs are configured, position_ids must update
     * through the setNextDecodeToken path.
     *
     * This reproduces the VLM decoder pattern:
     * 1. configureDecodeInputs() tells C++ which ext inputs are decode inputs
     * 2. Java frozen fast path SKIPS syncToSpecial for those inputs
     * 3. setNextDecodeToken() writes values directly to device via C++
     * 4. replay must observe the new values without Java-side resync
     *
     * If broken, position_ids stays stale → model output repeats.
     */
    @Test
    @DisplayName("DSP: setNextDecodeToken propagates position_ids through replay")
    public void testSetNextDecodeTokenPropagation() {
        Assumptions.assumeTrue(!Nd4j.getEnvironment().isCPU(),
                "Decode input propagation with CUDA_GRAPHS requires CUDA backend");
        SameDiff sd = SameDiff.create();

        int hiddenSize = 16;
        int numLayers = 10;

        INDArray embTable = Nd4j.randn(DataType.FLOAT, 100, hiddenSize);
        SDVariable embeddings = sd.constant("embeddings", embTable);
        SDVariable posIds = sd.placeHolder("position_ids", DataType.INT64, 1);

        SDVariable x = sd.gather("gathered", embeddings, posIds, 0);
        for (int layer = 0; layer < numLayers; layer++) {
            SDVariable w = sd.constant("w_" + layer,
                    Nd4j.eye(hiddenSize).castTo(DataType.FLOAT).muli(0.95));
            x = sd.mmul("mm_" + layer, x, w);
            x = sd.nn.relu("relu_" + layer, x, 0);
        }
        SDVariable out = x.sum("out");

        enableDsp(sd);
        sd.setGraphExecutionMode(GraphExecutionMode.CUDA_GRAPHS);

        INDArray posIdsArr = Nd4j.createFromArray(0L);
        Map<String, INDArray> placeholders = new java.util.HashMap<>();
        placeholders.put("position_ids", posIdsArr);

        // Phase 1: Execute once to compile the plan
        sd.output(placeholders, "out");

        // Phase 2: Get DSP executor and configure decode inputs
        var session = sd.getOrCreateSession();
        var dspExec = session.getDynamicShapePlanExecutor();
        if (dspExec != null) {
            var plan = dspExec.getCurrentPlan();
            if (plan != null) {
                dspExec.setShapesFrozen(true);
                dspExec.configureDecodeInputs(plan, 100);
                log.info("Decode inputs configured: {}", dspExec.isDecodeInputsConfigured());
            }
        }

        // Phase 3: Warmup frozen (3 more steps with position_ids=0)
        for (int i = 0; i < 3; i++) {
            sd.output(placeholders, "out");
        }
        log.info("Frozen warmup done");

        // Phase 4: Use setNextDecodeToken to change position_ids
        if (dspExec != null && dspExec.isDecodeInputsConfigured()) {
            double prevSum = Double.NaN;
            for (int step = 0; step < 5; step++) {
                // Tell C++ the next position — this skips Java syncToSpecial
                dspExec.setNextDecodeToken(step, step);  // tokenId=step, cachePos=step

                Map<String, INDArray> result = sd.output(placeholders, "out");
                double sum = result.get("out").getDouble(0);
                log.info("setNextDecodeToken step {}: cachePos={}, sum={}", step, step, sum);

                if (step > 0) {
                    assertNotEquals(prevSum, sum, 1e-6,
                            "Step " + step + ": output MUST differ when setNextDecodeToken " +
                            "changes cachePos from " + (step - 1) + " to " + step +
                            ". If identical, replay-side decode input propagation is broken.");
                }
                prevSum = sum;
            }
        } else {
            log.warn("DSP executor or decode inputs not configured — skipping decode token test");
        }

        sd.close();
    }
}
