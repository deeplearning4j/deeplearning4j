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

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.CausalConv1d;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for causal_conv1d op with wFormat parameter.
 *
 * Validates cross-correlation correctness for both weight formats:
 *   wFormat=0: [D, K] (PyTorch/ONNX convention)
 *   wFormat=1: [K, D] (TensorFlow convention)
 */
public class TestCausalConv1d {

    /**
     * Test cross-correlation correctness with wFormat=0 [D, K].
     *
     * Manual computation:
     *   x = [[[1, 2], [3, 4], [5, 6]]]  shape [1, 3, 2]
     *   weight = [[0.1, 0.2], [0.3, 0.4]]  shape [2, 2] (D=2, K=2)
     *
     * Cross-correlation (causal, K=2):
     *   output[t] = weight[:, K-1] * x[t] + weight[:, K-2] * x[t-1]
     *   weight[:, 1] = [0.2, 0.4] (current timestep)
     *   weight[:, 0] = [0.1, 0.3] (previous timestep)
     *
     *   t=0: d=0: 0.2*1 + 0.1*0 = 0.2,  d=1: 0.4*2 + 0.3*0 = 0.8
     *   t=1: d=0: 0.2*3 + 0.1*1 = 0.7,  d=1: 0.4*4 + 0.3*2 = 2.2
     *   t=2: d=0: 0.2*5 + 0.1*3 = 1.3,  d=1: 0.4*6 + 0.3*4 = 3.6
     */
    @Test
    public void testWFormat0_DK() {
        INDArray x = Nd4j.createFromArray(new float[]{1, 2, 3, 4, 5, 6}).reshape(1, 3, 2);
        INDArray weight = Nd4j.createFromArray(new float[]{0.1f, 0.2f, 0.3f, 0.4f}).reshape(2, 2);

        INDArray[] results = Nd4j.exec(new CausalConv1d(x, weight, null, null, 0, 0));
        INDArray output = results[0];

        assertEquals(3, output.rank());
        assertEquals(1, output.size(0));
        assertEquals(3, output.size(1));
        assertEquals(2, output.size(2));

        assertEquals(0.2f, output.getFloat(0, 0, 0), 1e-5f);
        assertEquals(0.8f, output.getFloat(0, 0, 1), 1e-5f);
        assertEquals(0.7f, output.getFloat(0, 1, 0), 1e-5f);
        assertEquals(2.2f, output.getFloat(0, 1, 1), 1e-5f);
        assertEquals(1.3f, output.getFloat(0, 2, 0), 1e-5f);
        assertEquals(3.6f, output.getFloat(0, 2, 1), 1e-5f);
    }

    /**
     * Test that wFormat=1 [K, D] produces identical output to wFormat=0 [D, K]
     * when using the transposed weight matrix.
     */
    @Test
    public void testWFormat1_KD_matchesWFormat0() {
        INDArray x = Nd4j.createFromArray(new float[]{1, 2, 3, 4, 5, 6}).reshape(1, 3, 2);
        INDArray weightDK = Nd4j.createFromArray(new float[]{0.1f, 0.2f, 0.3f, 0.4f}).reshape(2, 2);
        INDArray weightKD = weightDK.transpose().dup();  // [K, D]

        INDArray[] resultsDK = Nd4j.exec(new CausalConv1d(x, weightDK, null, null, 0, 0));
        INDArray[] resultsKD = Nd4j.exec(new CausalConv1d(x, weightKD, null, null, 0, 1));

        INDArray outputDK = resultsDK[0];
        INDArray outputKD = resultsKD[0];

        assertEquals(outputDK, outputKD, "wFormat=0 and wFormat=1 with transposed weights should produce identical output");
    }

    /**
     * Test wFormat=1 [K, D] with K > 2 and D > 2 to verify stride mapping
     * for non-trivial dimensions.
     */
    @Test
    public void testWFormat1_LargerKernelAndChannels() {
        int B = 1, L = 4, D = 3, K = 3;
        INDArray x = Nd4j.linspace(1, B * L * D, B * L * D, DataType.FLOAT).reshape(B, L, D);

        // Create deterministic weights in [D, K] format
        INDArray weightDK = Nd4j.linspace(0.1, 0.9, D * K, DataType.FLOAT).reshape(D, K);
        // Transpose to [K, D] for wFormat=1
        INDArray weightKD = weightDK.transpose().dup();

        INDArray[] resultsDK = Nd4j.exec(new CausalConv1d(x, weightDK, null, null, 0, 0));
        INDArray[] resultsKD = Nd4j.exec(new CausalConv1d(x, weightKD, null, null, 0, 1));

        INDArray outputDK = resultsDK[0];
        INDArray outputKD = resultsKD[0];

        // Both state outputs should also match
        INDArray stateDK = resultsDK[1];
        INDArray stateKD = resultsKD[1];

        for (int t = 0; t < L; t++) {
            for (int d = 0; d < D; d++) {
                assertEquals(outputDK.getFloat(0, t, d), outputKD.getFloat(0, t, d), 1e-4f,
                        String.format("Output mismatch at t=%d, d=%d", t, d));
            }
        }

        for (int d = 0; d < D; d++) {
            for (int kk = 0; kk < K - 1; kk++) {
                assertEquals(stateDK.getFloat(0, d, kk), stateKD.getFloat(0, d, kk), 1e-4f,
                        String.format("State mismatch at d=%d, kk=%d", d, kk));
            }
        }
    }

    /**
     * Test single timestep (L=1): only weight[:, K-1] should contribute.
     */
    @Test
    public void testSingleTimestep() {
        INDArray x = Nd4j.createFromArray(new float[]{2.0f, 3.0f}).reshape(1, 1, 2);
        INDArray weight = Nd4j.createFromArray(new float[]{0.1f, 0.2f, 0.5f, 0.3f, 0.4f, 0.6f}).reshape(2, 3);  // D=2, K=3

        INDArray[] results = Nd4j.exec(new CausalConv1d(x, weight, null, null, 0, 0));
        INDArray output = results[0];

        // For L=1, t=0: only weight[:, K-1] = weight[:, 2] = [0.5, 0.6] contributes
        // d=0: 0.5 * 2.0 = 1.0
        // d=1: 0.6 * 3.0 = 1.8
        assertEquals(1.0f, output.getFloat(0, 0, 0), 1e-5f);
        assertEquals(1.8f, output.getFloat(0, 0, 1), 1e-5f);

        // Same test with wFormat=1 [K, D]
        INDArray weightKD = weight.transpose().dup();
        INDArray[] resultsKD = Nd4j.exec(new CausalConv1d(x, weightKD, null, null, 0, 1));
        assertEquals(output, resultsKD[0], "wFormat=1 single timestep should match wFormat=0");
    }

    /**
     * Test SiLU activation (activation=1).
     */
    @Test
    public void testSiLUActivation() {
        INDArray x = Nd4j.createFromArray(new float[]{1.0f, -1.0f}).reshape(1, 1, 2);
        INDArray weight = Nd4j.createFromArray(new float[]{2.0f, 3.0f}).reshape(2, 1);  // D=2, K=1

        // No activation
        INDArray[] resultsNone = Nd4j.exec(new CausalConv1d(x, weight, null, null, 0, 0));
        // With SiLU
        INDArray[] resultsSilu = Nd4j.exec(new CausalConv1d(x, weight, null, null, 1, 0));

        float val0 = resultsNone[0].getFloat(0, 0, 0);  // 2.0 * 1.0 = 2.0
        float val1 = resultsNone[0].getFloat(0, 0, 1);  // 3.0 * (-1.0) = -3.0

        // SiLU(x) = x * sigmoid(x)
        float expected0 = val0 * (1.0f / (1.0f + (float) Math.exp(-val0)));
        float expected1 = val1 * (1.0f / (1.0f + (float) Math.exp(-val1)));

        assertEquals(expected0, resultsSilu[0].getFloat(0, 0, 0), 1e-4f, "SiLU for positive value");
        assertEquals(expected1, resultsSilu[0].getFloat(0, 0, 1), 1e-4f, "SiLU for negative value");
    }

    /**
     * Test state output: stateOut should contain last K-1 timesteps of input.
     */
    @Test
    public void testStateOutput() {
        int K = 3;
        INDArray x = Nd4j.createFromArray(new float[]{1, 2, 3, 4, 5, 6, 7, 8}).reshape(1, 4, 2);
        INDArray weight = Nd4j.ones(DataType.FLOAT, 2, K);

        INDArray[] results = Nd4j.exec(new CausalConv1d(x, weight, null, null, 0, 0));
        INDArray stateOut = results[1];  // [B, D, K-1]

        assertEquals(1, stateOut.size(0));
        assertEquals(2, stateOut.size(1));
        assertEquals(K - 1, stateOut.size(2));

        // Last K-1=2 timesteps: t=2 ([5,6]) and t=3 ([7,8])
        // stateOut[0, d, 0] = x[0, 2, d], stateOut[0, d, 1] = x[0, 3, d]
        assertEquals(5.0f, stateOut.getFloat(0, 0, 0), 1e-5f);
        assertEquals(7.0f, stateOut.getFloat(0, 0, 1), 1e-5f);
        assertEquals(6.0f, stateOut.getFloat(0, 1, 0), 1e-5f);
        assertEquals(8.0f, stateOut.getFloat(0, 1, 1), 1e-5f);
    }

    /**
     * Test state input continuity: feeding stateOut from one call as stateIn
     * to the next should produce the same result as processing the full sequence.
     */
    @Test
    public void testStateInputContinuity() {
        int B = 1, D = 2, K = 3, L = 6;
        INDArray xFull = Nd4j.linspace(1, B * L * D, B * L * D, DataType.FLOAT).reshape(B, L, D);
        INDArray weight = Nd4j.linspace(0.1, 0.6, D * K, DataType.FLOAT).reshape(D, K);

        // Process full sequence in one shot
        INDArray[] fullResults = Nd4j.exec(new CausalConv1d(xFull, weight, null, null, 0, 0));
        INDArray fullOutput = fullResults[0];

        // Process in two chunks: first 3 timesteps, then next 3 with state
        INDArray x1 = xFull.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 3), NDArrayIndex.all()).dup();
        INDArray x2 = xFull.get(NDArrayIndex.all(), NDArrayIndex.interval(3, 6), NDArrayIndex.all()).dup();

        INDArray[] results1 = Nd4j.exec(new CausalConv1d(x1, weight, null, null, 0, 0));
        INDArray stateOut1 = results1[1];

        INDArray[] results2 = Nd4j.exec(new CausalConv1d(x2, weight, null, stateOut1, 0, 0));
        INDArray output2 = results2[0];

        // The second chunk's output should match the full output's last 3 timesteps
        for (int t = 0; t < 3; t++) {
            for (int d = 0; d < D; d++) {
                assertEquals(fullOutput.getFloat(0, t + 3, d), output2.getFloat(0, t, d), 1e-4f,
                        String.format("State continuity mismatch at t=%d, d=%d", t + 3, d));
            }
        }
    }

    @Test
    public void testTwoTokenWindowMatchesChainedSingleTokenSteps() {
        Nd4j.getRandom().setSeed(23456);
        int B = 1, L = 2, D = 16, K = 4;
        INDArray x = Nd4j.randn(DataType.FLOAT, B, L, D).muli(0.1);
        INDArray weight = Nd4j.randn(DataType.FLOAT, D, K).muli(0.1);
        INDArray bias = Nd4j.randn(DataType.FLOAT, D).muli(0.01);
        INDArray stateIn = Nd4j.randn(DataType.FLOAT, B, D, K - 1).muli(0.1);

        INDArray[] window = Nd4j.exec(new CausalConv1d(
                x, weight, bias, stateIn, Nd4j.scalar(DataType.INT64, 2L), 1, 0));

        INDArray x0 = x.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 1), NDArrayIndex.all()).dup();
        INDArray x1 = x.get(NDArrayIndex.all(), NDArrayIndex.interval(1, 2), NDArrayIndex.all()).dup();
        INDArray[] first = Nd4j.exec(new CausalConv1d(
                x0, weight, bias, stateIn.dup(), Nd4j.scalar(DataType.INT64, 1L), 1, 0));
        INDArray[] second = Nd4j.exec(new CausalConv1d(
                x1, weight, bias, first[1], Nd4j.scalar(DataType.INT64, 1L), 1, 0));

        INDArray row0 = window[0].get(
                NDArrayIndex.all(), NDArrayIndex.interval(0, 1), NDArrayIndex.all());
        INDArray row1 = window[0].get(
                NDArrayIndex.all(), NDArrayIndex.interval(1, 2), NDArrayIndex.all());
        assertEquals(0.0, row0.sub(first[0]).amaxNumber().doubleValue(), 1e-5,
                "W=2 row 0 must match the first W=1 call");
        assertEquals(0.0, row1.sub(second[0]).amaxNumber().doubleValue(), 1e-5,
                "W=2 row 1 must match the chained W=1 call");
        assertEquals(0.0, window[1].sub(second[1]).amaxNumber().doubleValue(), 1e-5,
                "W=2 final state must match two chained W=1 calls");
    }

    @Test
    public void testFiveTokenWindowExactlyMatchesChainedSingleTokenStepsAtQwenDimensions() {
        Nd4j.getRandom().setSeed(45678);
        int B = 1, L = 5, D = 2048, K = 10;
        INDArray x = Nd4j.randn(DataType.FLOAT, B, L, D).muli(0.1);
        INDArray weight = Nd4j.randn(DataType.FLOAT, D, K).muli(0.1);
        INDArray bias = Nd4j.randn(DataType.FLOAT, D).muli(0.01);
        INDArray stateIn = Nd4j.randn(DataType.FLOAT, B, D, K - 1).muli(0.1);

        INDArray[] window = Nd4j.exec(new CausalConv1d(
                x, weight, bias, stateIn.dup(), Nd4j.scalar(DataType.INT64, L), 1, 0));

        INDArray chainedState = stateIn.dup();
        for (int t = 0; t < L; t++) {
            INDArray xScalar = x.get(
                    NDArrayIndex.all(), NDArrayIndex.interval(t, t + 1),
                    NDArrayIndex.all()).dup();
            INDArray[] scalar = Nd4j.exec(new CausalConv1d(
                    xScalar, weight, bias, chainedState,
                    Nd4j.scalar(DataType.INT64, 1L), 1, 0));
            INDArray windowRow = window[0].get(
                    NDArrayIndex.all(), NDArrayIndex.interval(t, t + 1),
                    NDArrayIndex.all()).dup();
            double maxDiff = windowRow.sub(scalar[0]).amaxNumber().doubleValue();
            double l1Diff = windowRow.sub(scalar[0]).norm1Number().doubleValue();
            assertEquals(0.0, maxDiff, 0.0,
                    "W=5 causal conv row " + t + " differs from chained W=1: maxDiff="
                            + maxDiff + ", l1Diff=" + l1Diff);
            chainedState = scalar[1];
        }

        double stateMaxDiff = window[1].sub(chainedState).amaxNumber().doubleValue();
        double stateL1Diff = window[1].sub(chainedState).norm1Number().doubleValue();
        assertEquals(0.0, stateMaxDiff, 0.0,
                "W=5 causal conv final state differs from chained W=1: maxDiff="
                        + stateMaxDiff + ", l1Diff=" + stateL1Diff);
    }

    @Test
    public void testFrozenFiveTokenBufferActivePrefixExactlyMatchesChainedStepsAtQwenDimensions() {
        Nd4j.getRandom().setSeed(67890);
        int B = 1, L = 5, D = 2048, K = 10;
        INDArray x = Nd4j.randn(DataType.FLOAT, B, L, D).muli(0.1);
        INDArray weight = Nd4j.randn(DataType.FLOAT, D, K).muli(0.1);
        INDArray bias = Nd4j.randn(DataType.FLOAT, D).muli(0.01);
        INDArray stateIn = Nd4j.randn(DataType.FLOAT, B, D, K - 1).muli(0.1);

        for (int activeLength : new int[]{1, 2}) {
            INDArray[] window = Nd4j.exec(new CausalConv1d(
                    x, weight, bias, stateIn.dup(),
                    Nd4j.scalar(DataType.INT64, activeLength), 1, 0));

            INDArray chainedState = stateIn.dup();
            for (int t = 0; t < activeLength; t++) {
                INDArray xScalar = x.get(
                        NDArrayIndex.all(), NDArrayIndex.interval(t, t + 1),
                        NDArrayIndex.all()).dup();
                INDArray[] scalar = Nd4j.exec(new CausalConv1d(
                        xScalar, weight, bias, chainedState,
                        Nd4j.scalar(DataType.INT64, 1L), 1, 0));
                INDArray windowRow = window[0].get(
                        NDArrayIndex.all(), NDArrayIndex.interval(t, t + 1),
                        NDArrayIndex.all()).dup();
                double maxDiff = windowRow.sub(scalar[0]).amaxNumber().doubleValue();
                double l1Diff = windowRow.sub(scalar[0]).norm1Number().doubleValue();
                assertEquals(0.0, maxDiff, 0.0,
                        "Frozen W=5 causal conv activeLength=" + activeLength + " row=" + t
                                + " differs from chained W=1: maxDiff=" + maxDiff
                                + ", l1Diff=" + l1Diff);
                chainedState = scalar[1];
            }

            double stateMaxDiff = window[1].sub(chainedState).amaxNumber().doubleValue();
            double stateL1Diff = window[1].sub(chainedState).norm1Number().doubleValue();
            assertEquals(0.0, stateMaxDiff, 0.0,
                    "Frozen W=5 causal conv activeLength=" + activeLength
                            + " final state differs from chained W=1: maxDiff="
                            + stateMaxDiff + ", l1Diff=" + stateL1Diff);
        }
    }

    /**
     * Test multi-timestep L > K to verify full causal window.
     */
    @Test
    public void testMultiTimestepFullWindow() {
        // D=1, K=2, L=5: verify all positions use the full kernel window
        INDArray x = Nd4j.createFromArray(new float[]{1, 2, 3, 4, 5}).reshape(1, 5, 1);
        INDArray weight = Nd4j.createFromArray(new float[]{0.5f, 1.0f}).reshape(1, 2);  // D=1, K=2

        INDArray[] results = Nd4j.exec(new CausalConv1d(x, weight, null, null, 0, 0));
        INDArray output = results[0];

        // Cross-correlation: weight[1]*x[t] + weight[0]*x[t-1]
        // t=0: 1.0*1 + 0.5*0 = 1.0
        // t=1: 1.0*2 + 0.5*1 = 2.5
        // t=2: 1.0*3 + 0.5*2 = 4.0
        // t=3: 1.0*4 + 0.5*3 = 5.5
        // t=4: 1.0*5 + 0.5*4 = 7.0
        assertEquals(1.0f, output.getFloat(0, 0, 0), 1e-5f);
        assertEquals(2.5f, output.getFloat(0, 1, 0), 1e-5f);
        assertEquals(4.0f, output.getFloat(0, 2, 0), 1e-5f);
        assertEquals(5.5f, output.getFloat(0, 3, 0), 1e-5f);
        assertEquals(7.0f, output.getFloat(0, 4, 0), 1e-5f);
    }

    /**
     * Test with bias.
     */
    @Test
    public void testWithBias() {
        INDArray x = Nd4j.createFromArray(new float[]{1.0f, 2.0f}).reshape(1, 1, 2);
        INDArray weight = Nd4j.createFromArray(new float[]{0.5f, 0.3f}).reshape(2, 1);  // D=2, K=1
        INDArray bias = Nd4j.createFromArray(new float[]{0.1f, -0.2f});

        INDArray[] results = Nd4j.exec(new CausalConv1d(x, weight, bias, null, 0, 0));
        INDArray output = results[0];

        // d=0: 0.5*1.0 + 0.1 = 0.6
        // d=1: 0.3*2.0 + (-0.2) = 0.4
        assertEquals(0.6f, output.getFloat(0, 0, 0), 1e-5f);
        assertEquals(0.4f, output.getFloat(0, 0, 1), 1e-5f);
    }

    /**
     * Test backward compatibility: calling with only activation (no wFormat)
     * should default to wFormat=0.
     */
    @Test
    public void testBackwardCompatNoWFormat() {
        INDArray x = Nd4j.createFromArray(new float[]{1, 2, 3, 4}).reshape(1, 2, 2);
        INDArray weight = Nd4j.createFromArray(new float[]{0.5f, 1.0f, 0.3f, 0.7f}).reshape(2, 2);

        // Old-style constructor (no wFormat)
        INDArray[] resultsOld = Nd4j.exec(new CausalConv1d(x, weight, null, null, 0));
        // New-style with explicit wFormat=0
        INDArray[] resultsNew = Nd4j.exec(new CausalConv1d(x, weight, null, null, 0, 0));

        assertEquals(resultsOld[0], resultsNew[0], "Backward compat: old constructor should match wFormat=0");
        assertEquals(resultsOld[1], resultsNew[1], "Backward compat: state should also match");
    }

    /**
     * Test via Nd4j.nn convenience API to verify generated code works.
     */
    @Test
    public void testNdNNApi() {
        INDArray x = Nd4j.createFromArray(new float[]{1, 2, 3, 4, 5, 6}).reshape(1, 3, 2);
        INDArray weight = Nd4j.createFromArray(new float[]{0.1f, 0.2f, 0.3f, 0.4f}).reshape(2, 2);

        // Use the generated NDNN API with wFormat
        INDArray[] results = Nd4j.nn.causalConv1d(x, weight, null, null, 0, 0);
        assertNotNull(results);
        assertEquals(2, results.length);
        assertEquals(3, results[0].rank());
        assertEquals(1, results[0].size(0));
        assertEquals(3, results[0].size(1));
        assertEquals(2, results[0].size(2));

        // Verify matches direct op execution
        INDArray[] directResults = Nd4j.exec(new CausalConv1d(x, weight, null, null, 0, 0));
        assertEquals(directResults[0], results[0]);
    }

    /**
     * Test batch dimension: B > 1 with both wFormats.
     */
    @Test
    public void testBatchDimension() {
        int B = 3, L = 4, D = 2, K = 2;
        INDArray x = Nd4j.linspace(1, B * L * D, B * L * D, DataType.FLOAT).reshape(B, L, D);
        INDArray weightDK = Nd4j.linspace(0.1, 0.4, D * K, DataType.FLOAT).reshape(D, K);
        INDArray weightKD = weightDK.transpose().dup();

        INDArray[] resultsDK = Nd4j.exec(new CausalConv1d(x, weightDK, null, null, 0, 0));
        INDArray[] resultsKD = Nd4j.exec(new CausalConv1d(x, weightKD, null, null, 0, 1));

        assertEquals(B, resultsDK[0].size(0));
        assertEquals(L, resultsDK[0].size(1));
        assertEquals(D, resultsDK[0].size(2));

        for (int b = 0; b < B; b++) {
            for (int t = 0; t < L; t++) {
                for (int d = 0; d < D; d++) {
                    assertEquals(resultsDK[0].getFloat(b, t, d), resultsKD[0].getFloat(b, t, d), 1e-4f,
                            String.format("Batch mismatch at b=%d, t=%d, d=%d", b, t, d));
                }
            }
        }
    }

    @Test
    public void testFloatActivationWithHalfWeightsMatchesFloatWeights() {
        int B = 1, L = 3, D = 2, K = 2;
        INDArray xFloat = Nd4j.createFromArray(new float[]{
                1f, 2f,
                3f, 4f,
                5f, 6f
        }).reshape(B, L, D);
        INDArray weightFloat = Nd4j.createFromArray(new float[]{
                0.5f, 1f,
                -0.25f, 0.75f
        }).reshape(D, K);
        INDArray biasFloat = Nd4j.createFromArray(new float[]{0.125f, -0.5f});
        INDArray stateFloat = Nd4j.createFromArray(new float[]{0.25f, -0.75f})
                .reshape(B, D, K - 1);
        INDArray actualLength = Nd4j.scalar(DataType.INT64, L);

        INDArray[] expectedFloat = Nd4j.exec(new CausalConv1d(
                xFloat, weightFloat, biasFloat, stateFloat, actualLength, 0, 0));
        INDArray[] actualFloat = Nd4j.exec(new CausalConv1d(
                xFloat, weightFloat.castTo(DataType.HALF), biasFloat,
                stateFloat.castTo(DataType.HALF), actualLength, 0, 0));

        assertEquals(DataType.FLOAT, actualFloat[0].dataType());
        assertEquals(DataType.HALF, actualFloat[1].dataType());
        for (long i = 0; i < expectedFloat[0].length(); i++) {
            assertEquals(expectedFloat[0].getFloat(i), actualFloat[0].getFloat(i), 1e-5f,
                    "FLOAT activation with HALF weight/state output mismatch at flat index " + i);
        }
        for (long i = 0; i < expectedFloat[1].length(); i++) {
            assertEquals(expectedFloat[1].getFloat(i), actualFloat[1].getFloat(i), 1e-5f,
                    "FLOAT activation with HALF weight/state update mismatch at flat index " + i);
        }

        INDArray xHalf = xFloat.castTo(DataType.HALF);
        INDArray weightHalf = weightFloat.castTo(DataType.HALF);
        INDArray biasHalf = biasFloat.castTo(DataType.HALF);
        INDArray[] expectedHalf = Nd4j.exec(new CausalConv1d(
                xHalf, weightHalf, biasHalf, stateFloat.castTo(DataType.HALF),
                actualLength, 0, 0));
        INDArray[] actualHalf = Nd4j.exec(new CausalConv1d(
                xHalf, weightHalf, biasHalf, stateFloat, actualLength, 0, 0));

        assertEquals(DataType.HALF, actualHalf[0].dataType());
        assertEquals(DataType.FLOAT, actualHalf[1].dataType());
        for (long i = 0; i < expectedHalf[0].length(); i++) {
            assertEquals(expectedHalf[0].getFloat(i), actualHalf[0].getFloat(i), 1e-3f,
                    "HALF activation with FLOAT state output mismatch at flat index " + i);
        }
        for (long i = 0; i < expectedHalf[1].length(); i++) {
            assertEquals(expectedHalf[1].getFloat(i), actualHalf[1].getFloat(i), 1e-3f,
                    "HALF activation with FLOAT state update mismatch at flat index " + i);
        }
    }

    @Test
    public void testDoubleInputsPreserveDoubleAccumulator() {
        int B = 1, L = 1, D = 1, K = 3;
        double x0 = 1.0000000001;
        double oldestState = -0.3333333337;
        double newestState = 0.6666666669;
        double oldestWeight = 0.1250000003;
        double middleWeight = -0.2500000002;
        double currentWeight = 0.5000000004;
        double bias0 = -0.0625000001;

        INDArray x = Nd4j.createFromArray(new double[]{x0}).reshape(B, L, D);
        INDArray weight = Nd4j.createFromArray(new double[]{
                oldestWeight, middleWeight, currentWeight
        }).reshape(D, K);
        INDArray bias = Nd4j.createFromArray(new double[]{bias0});
        INDArray state = Nd4j.createFromArray(new double[]{oldestState, newestState})
                .reshape(B, D, K - 1);

        INDArray[] result = Nd4j.exec(new CausalConv1d(x, weight, bias, state, 0, 0));
        double expected = currentWeight * x0
                + middleWeight * newestState
                + oldestWeight * oldestState
                + bias0;

        assertEquals(DataType.DOUBLE, result[0].dataType());
        assertEquals(DataType.DOUBLE, result[1].dataType());
        assertEquals(expected, result[0].getDouble(0, 0, 0), 1e-14);
    }

    @Test
    public void testActualLengthStopsStateUpdate() {
        int B = 1, L = 5, D = 2, K = 3;
        INDArray x = Nd4j.createFromArray(new float[]{
                1f, 2f,
                3f, 4f,
                5f, 6f,
                100f, 200f,
                300f, 400f
        }).reshape(B, L, D);
        INDArray weight = Nd4j.ones(DataType.FLOAT, D, K);
        INDArray actualLen = Nd4j.scalar(DataType.INT64, 3L);

        INDArray[] results = Nd4j.exec(new CausalConv1d(x, weight, null, null, actualLen, 0, 0));
        INDArray stateOut = results[1];

        assertEquals(3.0f, stateOut.getFloat(0, 0, 0), 1e-5f);
        assertEquals(5.0f, stateOut.getFloat(0, 0, 1), 1e-5f);
        assertEquals(4.0f, stateOut.getFloat(0, 1, 0), 1e-5f);
        assertEquals(6.0f, stateOut.getFloat(0, 1, 1), 1e-5f);
    }
}
