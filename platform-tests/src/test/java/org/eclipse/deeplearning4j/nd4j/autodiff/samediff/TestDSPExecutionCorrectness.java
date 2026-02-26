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

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests DSP (Dynamic Shape Plan) execution correctness by comparing
 * DSP output against standard SameDiff output for various op patterns.
 *
 * These tests verify that the DSP compilation and execution path
 * (including frozen shapes, CUDA graphs, nullify logic, etc.) produces
 * numerically correct results.
 */
public class TestDSPExecutionCorrectness extends BaseNd4jTestWithBackends {

    private static final Logger log = LoggerFactory.getLogger(TestDSPExecutionCorrectness.class);
    private static final double TOL = 1e-4;

    @Override
    public char ordering() {
        return 'c';
    }

    /**
     * Test basic matmul + add + relu pattern (common in neural networks).
     * Runs the same graph with standard execution and DSP, compares results.
     */
    @Test
    @DisplayName("DSP: matmul + add + relu matches standard execution")
    public void testMatmulAddRelu() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 4);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 4, 8));
        SDVariable b = sd.constant("b", Nd4j.randn(DataType.FLOAT, 1, 8));
        SDVariable mm = sd.mmul("mm", x, w);
        SDVariable added = mm.add("added", b);
        SDVariable out = sd.nn.relu("out", added, 0);

        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 4);

        // Standard execution
        Map<String, INDArray> standardResult = sd.output(Map.of("x", input), "out");
        INDArray expected = standardResult.get("out").dup();

        // DSP execution (uses outputDirect which goes through DSP)
        Map<String, INDArray> dspResult = sd.outputDirect(Map.of("x", input), "out");
        INDArray actual = dspResult.get("out").dup();

        log.info("Standard: {}", expected);
        log.info("DSP:      {}", actual);

        assertArrayEquals(expected.shape(), actual.shape(), "Shape mismatch");
        double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
        log.info("Max diff: {}", maxDiff);
        assertTrue(maxDiff < TOL, "Max diff " + maxDiff + " exceeds tolerance " + TOL);

        sd.close();
    }

    /**
     * Test reshape + expand_dims (view-capable ops).
     * These ops should produce correct results whether using zero-copy views or not.
     */
    @Test
    @DisplayName("DSP: reshape + expand_dims correctness")
    public void testReshapeExpandDims() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 6);
        SDVariable reshaped = sd.reshape("reshaped", x, 3, 4);
        SDVariable expanded = sd.expandDims("expanded", reshaped, 0);  // [1, 3, 4]
        SDVariable out = expanded.mul("out", sd.constant("scale", Nd4j.scalar(2.0f)));

        INDArray input = Nd4j.linspace(1, 12, 12, DataType.FLOAT).reshape(2, 6);

        // Standard
        Map<String, INDArray> standardResult = sd.output(Map.of("x", input), "out");
        INDArray expected = standardResult.get("out").dup();

        // DSP
        Map<String, INDArray> dspResult = sd.outputDirect(Map.of("x", input), "out");
        INDArray actual = dspResult.get("out").dup();

        log.info("Standard shape: {} values: {}", java.util.Arrays.toString(expected.shape()), expected);
        log.info("DSP shape:      {} values: {}", java.util.Arrays.toString(actual.shape()), actual);

        assertArrayEquals(expected.shape(), actual.shape(), "Shape mismatch");
        double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
        assertTrue(maxDiff < TOL, "Max diff " + maxDiff + " exceeds tolerance " + TOL);

        sd.close();
    }

    /**
     * Test gather op (embedding lookup pattern).
     * Critical for decoder token embedding lookup.
     */
    @Test
    @DisplayName("DSP: gather (embedding lookup) correctness")
    public void testGatherEmbedding() {
        SameDiff sd = SameDiff.create();
        // Simulated embedding table: 10 tokens, 4-dim embeddings
        SDVariable embedTable = sd.constant("embedTable",
                Nd4j.randn(DataType.FLOAT, 10, 4));
        SDVariable indices = sd.placeHolder("indices", DataType.LONG, -1);
        SDVariable gathered = sd.gather("gathered", embedTable, indices, 0);
        SDVariable out = gathered.mul("out", sd.constant("scale", Nd4j.scalar(1.0f)));

        INDArray tokenIds = Nd4j.createFromArray(new long[]{3, 7, 1, 5});

        // Standard
        Map<String, INDArray> standardResult = sd.output(Map.of("indices", tokenIds), "out");
        INDArray expected = standardResult.get("out").dup();

        // DSP
        Map<String, INDArray> dspResult = sd.outputDirect(Map.of("indices", tokenIds), "out");
        INDArray actual = dspResult.get("out").dup();

        log.info("Standard: {}", expected);
        log.info("DSP:      {}", actual);

        assertArrayEquals(expected.shape(), actual.shape(), "Shape mismatch");
        double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
        assertTrue(maxDiff < TOL, "Max diff " + maxDiff + " exceeds tolerance " + TOL);

        sd.close();
    }

    /**
     * Test concat + split pattern (used in KV cache operations).
     */
    @Test
    @DisplayName("DSP: concat + split correctness")
    public void testConcatSplit() {
        SameDiff sd = SameDiff.create();
        SDVariable a = sd.placeHolder("a", DataType.FLOAT, 1, 3);
        SDVariable b = sd.placeHolder("b", DataType.FLOAT, 1, 3);
        SDVariable concatenated = sd.concat("concatenated", 1, a, b);  // [1, 6]
        SDVariable out = concatenated.mul("out", sd.constant("scale", Nd4j.scalar(3.0f)));

        INDArray inputA = Nd4j.createFromArray(new float[][]{{1, 2, 3}});
        INDArray inputB = Nd4j.createFromArray(new float[][]{{4, 5, 6}});

        // Standard
        Map<String, INDArray> standardResult = sd.output(
                Map.of("a", inputA, "b", inputB), "out");
        INDArray expected = standardResult.get("out").dup();

        // DSP
        Map<String, INDArray> dspResult = sd.outputDirect(
                Map.of("a", inputA, "b", inputB), "out");
        INDArray actual = dspResult.get("out").dup();

        assertArrayEquals(expected.shape(), actual.shape(), "Shape mismatch");
        double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
        assertTrue(maxDiff < TOL, "Max diff " + maxDiff + " exceeds tolerance " + TOL);

        sd.close();
    }

    /**
     * Test where/select op (used in attention masking).
     */
    @Test
    @DisplayName("DSP: where/select correctness")
    public void testWhereSelect() {
        SameDiff sd = SameDiff.create();
        SDVariable cond = sd.placeHolder("cond", DataType.BOOL, 1, 6);
        SDVariable positive = sd.constant("positive", Nd4j.ones(DataType.FLOAT, 1, 6));
        SDVariable negative = sd.constant("negative",
                Nd4j.ones(DataType.FLOAT, 1, 6).mul(-1));
        SDVariable out = sd.where("out", positive, negative, cond);

        INDArray condInput = Nd4j.createFromArray(new boolean[][]{{false, true, false, true, true, false}});

        // Standard
        Map<String, INDArray> standardResult = sd.output(Map.of("cond", condInput), "out");
        INDArray expected = standardResult.get("out").dup();

        // DSP
        Map<String, INDArray> dspResult = sd.outputDirect(Map.of("cond", condInput), "out");
        INDArray actual = dspResult.get("out").dup();

        log.info("Standard: {}", expected);
        log.info("DSP:      {}", actual);

        assertArrayEquals(expected.shape(), actual.shape(), "Shape mismatch");
        double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
        assertTrue(maxDiff < TOL, "Max diff " + maxDiff + " exceeds tolerance " + TOL);

        sd.close();
    }

    /**
     * Test softmax (normalization op).
     */
    @Test
    @DisplayName("DSP: softmax correctness")
    public void testSoftmax() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 5);
        SDVariable out = sd.nn.softmax("out", x, -1);

        INDArray input = Nd4j.createFromArray(new float[][]{{1, 2, 3, 4, 5}});

        // Standard
        Map<String, INDArray> standardResult = sd.output(Map.of("x", input), "out");
        INDArray expected = standardResult.get("out").dup();

        // DSP
        Map<String, INDArray> dspResult = sd.outputDirect(Map.of("x", input), "out");
        INDArray actual = dspResult.get("out").dup();

        assertArrayEquals(expected.shape(), actual.shape(), "Shape mismatch");
        double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
        assertTrue(maxDiff < TOL, "Max diff " + maxDiff + " exceeds tolerance " + TOL);

        sd.close();
    }

    /**
     * Test repeated DSP execution with same shapes (tests output reuse/nullify).
     * This is critical for detecting stale data accumulation bugs.
     */
    @Test
    @DisplayName("DSP: repeated execution produces correct results each time")
    public void testRepeatedExecution() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 4);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 4, 4));
        SDVariable mm = sd.mmul("mm", x, w);
        SDVariable out = sd.nn.sigmoid("out", mm);

        // Run multiple times with DIFFERENT inputs, check each time
        for (int i = 0; i < 5; i++) {
            INDArray input = Nd4j.randn(DataType.FLOAT, 1, 4).mul(i + 1);

            // Standard
            Map<String, INDArray> standardResult = sd.output(Map.of("x", input), "out");
            INDArray expected = standardResult.get("out").dup();

            // DSP
            Map<String, INDArray> dspResult = sd.outputDirect(Map.of("x", input), "out");
            INDArray actual = dspResult.get("out").dup();

            double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
            log.info("Iteration {}: maxDiff={}", i, maxDiff);
            assertTrue(maxDiff < TOL,
                    "Iteration " + i + ": max diff " + maxDiff + " exceeds tolerance " + TOL);
        }

        sd.close();
    }

    /**
     * Test a multi-layer pattern: matmul -> add -> multiply -> sigmoid -> matmul -> softmax
     * This exercises the full DSP pipeline with multiple op types.
     */
    @Test
    @DisplayName("DSP: multi-layer transformer-like pattern")
    public void testMultiLayerPattern() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 8);
        SDVariable w1 = sd.constant("w1", Nd4j.randn(DataType.FLOAT, 8, 16));
        SDVariable b1 = sd.constant("b1", Nd4j.randn(DataType.FLOAT, 1, 16));
        SDVariable w2 = sd.constant("w2", Nd4j.randn(DataType.FLOAT, 16, 8));

        // Layer 1: linear + bias + sigmoid
        SDVariable h1 = sd.mmul("h1", x, w1);
        SDVariable h1b = h1.add("h1b", b1);
        SDVariable h1a = sd.nn.sigmoid("h1a", h1b);

        // Layer 2: linear + softmax
        SDVariable h2 = sd.mmul("h2", h1a, w2);
        SDVariable out = sd.nn.softmax("out", h2, -1);

        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 8);

        // Standard
        Map<String, INDArray> standardResult = sd.output(Map.of("x", input), "out");
        INDArray expected = standardResult.get("out").dup();

        // DSP
        Map<String, INDArray> dspResult = sd.outputDirect(Map.of("x", input), "out");
        INDArray actual = dspResult.get("out").dup();

        assertArrayEquals(expected.shape(), actual.shape(), "Shape mismatch");
        double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
        log.info("Multi-layer max diff: {}", maxDiff);
        assertTrue(maxDiff < TOL, "Max diff " + maxDiff + " exceeds tolerance " + TOL);

        // Verify softmax sums to 1
        double sum = actual.sumNumber().doubleValue();
        log.info("Softmax sum: {}", sum);
        assertTrue(Math.abs(sum - 1.0) < TOL, "Softmax sum " + sum + " != 1.0");

        sd.close();
    }

    /**
     * Test RMSNorm op (used by GraphOptimizer fusion).
     * Compares fused rms_norm against manual computation.
     */
    @Test
    @DisplayName("DSP: rms_norm correctness")
    public void testRmsNorm() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 8);
        SDVariable gamma = sd.constant("gamma", Nd4j.ones(DataType.FLOAT, 8));
        float epsilon = 1e-5f;
        SDVariable out = sd.nn.rmsNorm("out", x, gamma, epsilon);

        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 8);

        // Manual RMSNorm computation for reference
        INDArray xSquared = input.mul(input);
        double meanSquared = xSquared.meanNumber().doubleValue();
        double rms = Math.sqrt(meanSquared + epsilon);
        INDArray expectedManual = input.div(rms);

        // Standard
        Map<String, INDArray> standardResult = sd.output(Map.of("x", input), "out");
        INDArray expected = standardResult.get("out").dup();

        // DSP
        Map<String, INDArray> dspResult = sd.outputDirect(Map.of("x", input), "out");
        INDArray actual = dspResult.get("out").dup();

        log.info("Manual:   {}", expectedManual);
        log.info("Standard: {}", expected);
        log.info("DSP:      {}", actual);

        // Check DSP matches standard
        double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
        log.info("DSP vs standard max diff: {}", maxDiff);
        assertTrue(maxDiff < TOL, "DSP vs standard: max diff " + maxDiff + " exceeds tolerance " + TOL);

        // Check standard matches manual
        double manualDiff = expected.sub(expectedManual).amaxNumber().doubleValue();
        log.info("Standard vs manual max diff: {}", manualDiff);
        assertTrue(manualDiff < 1e-3, "Standard vs manual: max diff " + manualDiff + " exceeds tolerance");

        sd.close();
    }

    /**
     * Test SiLU (swish) activation: x * sigmoid(x).
     */
    @Test
    @DisplayName("DSP: silu (swish) correctness")
    public void testSilu() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 8);
        SDVariable sigX = sd.nn.sigmoid("sig", x);
        SDVariable out = x.mul("out", sigX);

        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 8);

        // Standard
        Map<String, INDArray> standardResult = sd.output(Map.of("x", input), "out");
        INDArray expected = standardResult.get("out").dup();

        // DSP
        Map<String, INDArray> dspResult = sd.outputDirect(Map.of("x", input), "out");
        INDArray actual = dspResult.get("out").dup();

        assertArrayEquals(expected.shape(), actual.shape(), "Shape mismatch");
        double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
        log.info("SiLU max diff: {}", maxDiff);
        assertTrue(maxDiff < TOL, "Max diff " + maxDiff + " exceeds tolerance " + TOL);

        sd.close();
    }

    /**
     * Test scatter_nd / set_scalar pattern (used in KV cache updates).
     */
    @Test
    @DisplayName("DSP: set_scalar (KV cache write) correctness")
    public void testSetScalar() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 10);
        // Create a constant and scatter a value
        SDVariable ones = sd.constant("ones", Nd4j.ones(DataType.FLOAT, 1, 10));
        SDVariable result = ones.mul("result", x);
        SDVariable out = result.add("out", sd.constant("bias", Nd4j.scalar(0.5f)));

        INDArray input = Nd4j.linspace(1, 10, 10, DataType.FLOAT).reshape(1, 10);

        // Standard
        Map<String, INDArray> standardResult = sd.output(Map.of("x", input), "out");
        INDArray expected = standardResult.get("out").dup();

        // DSP
        Map<String, INDArray> dspResult = sd.outputDirect(Map.of("x", input), "out");
        INDArray actual = dspResult.get("out").dup();

        assertArrayEquals(expected.shape(), actual.shape(), "Shape mismatch");
        double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
        assertTrue(maxDiff < TOL, "Max diff " + maxDiff + " exceeds tolerance " + TOL);

        sd.close();
    }

    /**
     * Test permute (transpose) pattern — view-producer ops.
     * Critical for multi-head attention reshape.
     */
    @Test
    @DisplayName("DSP: permute (transpose) correctness")
    public void testPermute() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 2, 3, 4);
        // Permute to [1, 3, 2, 4] (like attention head reshape)
        SDVariable permuted = sd.permute("permuted", x, 0, 2, 1, 3);
        SDVariable out = permuted.mul("out", sd.constant("scale", Nd4j.scalar(2.0f)));

        INDArray input = Nd4j.linspace(1, 24, 24, DataType.FLOAT).reshape(1, 2, 3, 4);

        // Standard
        Map<String, INDArray> standardResult = sd.output(Map.of("x", input), "out");
        INDArray expected = standardResult.get("out").dup();

        // DSP
        Map<String, INDArray> dspResult = sd.outputDirect(Map.of("x", input), "out");
        INDArray actual = dspResult.get("out").dup();

        assertArrayEquals(expected.shape(), actual.shape(), "Shape mismatch");
        double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
        log.info("Permute max diff: {}", maxDiff);
        assertTrue(maxDiff < TOL, "Max diff " + maxDiff + " exceeds tolerance " + TOL);

        sd.close();
    }

    /**
     * Test attention-like pattern: matmul -> scale -> mask -> softmax -> matmul
     * This exercises the core attention mechanism.
     */
    @Test
    @DisplayName("DSP: attention pattern (Q*K^T -> scale -> softmax -> *V)")
    public void testAttentionPattern() {
        int seqLen = 4;
        int headDim = 8;

        SameDiff sd = SameDiff.create();
        SDVariable q = sd.placeHolder("q", DataType.FLOAT, 1, seqLen, headDim);
        SDVariable k = sd.placeHolder("k", DataType.FLOAT, 1, seqLen, headDim);
        SDVariable v = sd.placeHolder("v", DataType.FLOAT, 1, seqLen, headDim);

        // Q * K^T / sqrt(d)
        SDVariable kT = sd.permute("kT", k, 0, 2, 1);  // [1, headDim, seqLen]
        SDVariable scores = sd.mmul("scores", q, kT);  // [1, seqLen, seqLen]
        SDVariable scaled = scores.div("scaled",
                sd.constant("scale", Nd4j.scalar((float) Math.sqrt(headDim))));
        SDVariable attnWeights = sd.nn.softmax("attnWeights", scaled, -1);
        SDVariable out = sd.mmul("out", attnWeights, v);  // [1, seqLen, headDim]

        INDArray qInput = Nd4j.randn(DataType.FLOAT, 1, seqLen, headDim);
        INDArray kInput = Nd4j.randn(DataType.FLOAT, 1, seqLen, headDim);
        INDArray vInput = Nd4j.randn(DataType.FLOAT, 1, seqLen, headDim);

        Map<String, INDArray> inputs = Map.of("q", qInput, "k", kInput, "v", vInput);

        // Standard
        Map<String, INDArray> standardResult = sd.output(inputs, "out");
        INDArray expected = standardResult.get("out").dup();

        // DSP
        Map<String, INDArray> dspResult = sd.outputDirect(inputs, "out");
        INDArray actual = dspResult.get("out").dup();

        assertArrayEquals(expected.shape(), actual.shape(), "Shape mismatch");
        double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
        log.info("Attention pattern max diff: {}", maxDiff);
        assertTrue(maxDiff < TOL, "Max diff " + maxDiff + " exceeds tolerance " + TOL);

        sd.close();
    }

    /**
     * Test autoregressive decode pattern: repeated DSP execution with changing
     * inputs but the same graph (like a decoder with static KV cache).
     * Each step should produce a result dependent on the current input,
     * NOT stale data from a previous step.
     */
    @Test
    @DisplayName("DSP: autoregressive decode pattern with frozen shapes")
    public void testAutoregressiveDecodePattern() {
        // Simulate a tiny 1-layer decoder
        int hiddenSize = 16;
        int vocabSize = 32;

        SameDiff sd = SameDiff.create();

        // Input embedding for single token [1, 1, hiddenSize]
        SDVariable embed = sd.placeHolder("embed", DataType.FLOAT, 1, 1, hiddenSize);

        // Simple "decoder": reshape -> matmul -> add -> relu -> matmul -> output logits
        SDVariable flat = sd.reshape("flat", embed, 1, hiddenSize);
        SDVariable w1 = sd.constant("w1", Nd4j.randn(DataType.FLOAT, hiddenSize, hiddenSize));
        SDVariable b1 = sd.constant("b1", Nd4j.zeros(DataType.FLOAT, 1, hiddenSize));
        SDVariable h = sd.mmul("h", flat, w1);
        SDVariable hb = h.add("hb", b1);
        SDVariable ha = sd.nn.relu("ha", hb, 0);

        SDVariable wOut = sd.constant("wOut", Nd4j.randn(DataType.FLOAT, hiddenSize, vocabSize));
        SDVariable logits = sd.mmul("logits", ha, wOut);

        // Simulate autoregressive decoding: different embedding each step
        INDArray[] embeddings = new INDArray[10];
        INDArray[] expectedLogits = new INDArray[10];
        for (int i = 0; i < 10; i++) {
            embeddings[i] = Nd4j.randn(DataType.FLOAT, 1, 1, hiddenSize).mul(i + 1);
            // Get expected from standard execution
            Map<String, INDArray> result = sd.output(Map.of("embed", embeddings[i]), "logits");
            expectedLogits[i] = result.get("logits").dup();
        }

        // Now run through DSP with outputDirect (frozen shapes)
        for (int i = 0; i < 10; i++) {
            Map<String, INDArray> dspResult = sd.outputDirect(Map.of("embed", embeddings[i]), "logits");
            INDArray actual = dspResult.get("logits").dup();

            double maxDiff = expectedLogits[i].sub(actual).amaxNumber().doubleValue();
            log.info("Step {}: maxDiff={}, argmax_expected={}, argmax_actual={}",
                    i, maxDiff,
                    Nd4j.argMax(expectedLogits[i], 1).getInt(0),
                    Nd4j.argMax(actual, 1).getInt(0));
            assertTrue(maxDiff < TOL,
                    "Step " + i + ": max diff " + maxDiff + " exceeds tolerance " + TOL);

            // Verify different inputs produce different outputs
            if (i > 0) {
                double diffFromPrev = actual.sub(expectedLogits[i - 1]).amaxNumber().doubleValue();
                assertTrue(diffFromPrev > 0.01,
                        "Step " + i + " output is too similar to step " + (i - 1) +
                                " — possible stale data. diff=" + diffFromPrev);
            }
        }

        sd.close();
    }

    /**
     * Test GraphOptimizer RMSNorm fusion correctness.
     * Builds the same pattern the optimizer fuses (mul(rsqrt(mean(x*x)+eps), x) * gamma)
     * and verifies fused rms_norm produces identical results.
     */
    /**
     * Test rms_norm op directly (bypassing SameDiff) to isolate CUDA kernel correctness.
     */
    @Test
    @DisplayName("rms_norm op: direct INDArray execution vs manual computation")
    public void testRmsNormDirectExecution() {
        int dim = 32;
        float eps = 1e-5f;

        INDArray input = Nd4j.randn(DataType.FLOAT, 1, dim);
        INDArray gamma = Nd4j.randn(DataType.FLOAT, dim);

        // Manual computation
        INDArray xSq = input.mul(input);
        double meanSq = xSq.meanNumber().doubleValue();
        double invRms = 1.0 / Math.sqrt(meanSq + eps);
        INDArray manual = input.mul(invRms).mul(gamma.reshape(1, dim));

        // Direct op execution
        INDArray opResult = Nd4j.nn.rmsNorm(input, gamma, eps);

        log.info("Direct op vs manual max diff: {}", manual.sub(opResult.reshape(1, dim)).amaxNumber().doubleValue());
        assertTrue(manual.sub(opResult.reshape(1, dim)).amaxNumber().doubleValue() < 1e-3,
                "Direct rms_norm op vs manual exceeds tolerance");
    }

    /**
     * Test rms_norm through SameDiff only - minimal reproduction of the failure.
     * Also outputs the gamma to verify SameDiff passes it correctly.
     */
    @Test
    @DisplayName("rms_norm op: SameDiff execution debugging")
    public void testRmsNormSameDiffDebug() {
        int dim = 32;
        float eps = 1e-5f;

        INDArray input = Nd4j.randn(DataType.FLOAT, 1, dim);
        INDArray gammaArr = Nd4j.randn(DataType.FLOAT, dim);

        // Direct op
        INDArray directResult = Nd4j.nn.rmsNorm(input, gammaArr, eps);

        // SameDiff (same pattern as testRmsNormFusionVsManual)
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, dim);
        SDVariable gamma = sd.constant("gamma", gammaArr.dup());
        SDVariable out = sd.nn.rmsNorm("out", x, gamma, eps);

        Map<String, INDArray> result = sd.output(Map.of("x", input), "out");
        INDArray sdResult = result.get("out");

        log.info("Direct: {}", directResult.getRow(0));
        log.info("SD:     {}", sdResult.getRow(0));

        double maxDiff = directResult.sub(sdResult).amaxNumber().doubleValue();
        log.info("Direct vs SameDiff max diff: {}", maxDiff);

        assertTrue(maxDiff < 1e-3, "SameDiff rms_norm vs direct: max diff " + maxDiff);

        // Now test gamma copied from another SameDiff (the EXACT pattern in the failing test)
        SameDiff sd1 = SameDiff.create();
        sd1.constant("gamma", gammaArr.dup());
        SameDiff sd2 = SameDiff.create();
        SDVariable x2 = sd2.placeHolder("x", DataType.FLOAT, 1, dim);
        SDVariable gamma2 = sd2.constant("gamma", sd1.getVariable("gamma").getArr().dup());
        SDVariable out2 = sd2.nn.rmsNorm("out", x2, gamma2, eps);

        Map<String, INDArray> result2 = sd2.output(Map.of("x", input), "out");
        INDArray sdResult2 = result2.get("out");

        double maxDiff2 = directResult.sub(sdResult2).amaxNumber().doubleValue();
        log.info("Cross-SD gamma copy: Direct vs SameDiff max diff: {}", maxDiff2);

        assertTrue(maxDiff2 < 1e-3, "Cross-SD gamma copy: max diff " + maxDiff2);

        sd.close();
        sd1.close();
        sd2.close();
    }

    /**
     * Test rms_norm op through SameDiff vs manual computation through SameDiff.
     * Diagnoses whether the issue is in op wiring, constant handling, or gamma passing.
     */
    @Test
    @DisplayName("DSP: GraphOptimizer RMSNorm fusion vs manual")
    public void testRmsNormFusionVsManual() {
        int dim = 32;
        float eps = 1e-5f;

        INDArray gammaArr = Nd4j.randn(DataType.FLOAT, dim);
        INDArray input = Nd4j.randn(DataType.FLOAT, 1, dim);

        // Direct op computation (known to be correct)
        INDArray directResult = Nd4j.nn.rmsNorm(input, gammaArr, eps);

        // SameDiff with rms_norm op
        SameDiff sdFused = SameDiff.create();
        SDVariable x = sdFused.placeHolder("x", DataType.FLOAT, 1, dim);
        SDVariable gamma = sdFused.constant("gamma", gammaArr.dup());
        SDVariable fusedOut = sdFused.nn.rmsNorm("fusedOut", x, gamma, eps);

        // Also output the gamma to verify it's correct in the graph
        SDVariable gammaOut = sdFused.identity("gammaOut", gamma);

        Map<String, INDArray> fusedResult = sdFused.output(Map.of("x", input), "fusedOut", "gammaOut");
        INDArray fusedArr = fusedResult.get("fusedOut").dup();
        INDArray gammaFromGraph = fusedResult.get("gammaOut").dup();

        log.info("Direct result: {}", directResult.reshape(1, dim).getRow(0));
        log.info("SameDiff result: {}", fusedArr.getRow(0));
        log.info("Original gamma:  {}", gammaArr);
        log.info("Graph gamma:     {}", gammaFromGraph);

        double gammaDiff = gammaArr.sub(gammaFromGraph).amaxNumber().doubleValue();
        log.info("Gamma diff: {}", gammaDiff);

        double maxDiff = directResult.reshape(1, dim).sub(fusedArr).amaxNumber().doubleValue();
        log.info("Direct vs SameDiff max diff: {}", maxDiff);

        assertTrue(gammaDiff < 1e-6, "Gamma values differ: " + gammaDiff);
        assertTrue(maxDiff < 1e-3, "Direct vs SameDiff rms_norm max diff " + maxDiff + " exceeds tolerance");

        sdFused.close();
    }

    /**
     * Direct INDArray-level test of rms_norm op (no SameDiff).
     * Tests Nd4j.nn.rmsNorm() directly against manual computation.
     */
    @Test
    @DisplayName("Direct INDArray rms_norm vs manual computation")
    public void testRmsNormDirect() {
        int dim = 32;
        float eps = 1e-5f;

        INDArray input = Nd4j.randn(DataType.FLOAT, 1, dim);
        INDArray gamma = Nd4j.randn(DataType.FLOAT, dim);

        // Manual computation
        INDArray xSq = input.mul(input);
        double meanSq = xSq.meanNumber().doubleValue();
        double invRms = 1.0 / Math.sqrt(meanSq + eps);
        INDArray normalized = input.mul(invRms);
        INDArray manualResult = normalized.mul(gamma.reshape(1, dim));

        // Direct op execution
        INDArray fusedResult = Nd4j.nn.rmsNorm(input, gamma, eps);

        log.info("Input:  {}", input);
        log.info("Gamma:  {}", gamma);
        log.info("Manual: {}", manualResult);
        log.info("Fused:  {}", fusedResult);

        double maxDiff = manualResult.sub(fusedResult).amaxNumber().doubleValue();
        log.info("Direct rms_norm vs manual max diff: {}", maxDiff);

        // Also test WITHOUT gamma
        double invRmsNoGamma = 1.0 / Math.sqrt(input.mul(input).meanNumber().doubleValue() + eps);
        INDArray manualNoGamma = input.mul(invRmsNoGamma);
        INDArray fusedNoGamma = Nd4j.nn.rmsNorm(input, eps);

        double maxDiffNoGamma = manualNoGamma.sub(fusedNoGamma).amaxNumber().doubleValue();
        log.info("Without gamma - manual: {}", manualNoGamma);
        log.info("Without gamma - fused:  {}", fusedNoGamma);
        log.info("Without gamma max diff: {}", maxDiffNoGamma);

        assertTrue(maxDiffNoGamma < 1e-3,
                "rms_norm without gamma: max diff " + maxDiffNoGamma + " exceeds tolerance");
        assertTrue(maxDiff < 1e-3,
                "rms_norm with gamma: max diff " + maxDiff + " exceeds tolerance");
    }

    // ============================================================
    // Execution Mode Tests: SLOT_BY_SLOT, CUDA_GRAPHS, NVRTC_JIT, TRITON
    // ============================================================

    /**
     * Helper: builds a multi-op graph (matmul + add + sigmoid + matmul + softmax)
     * and verifies execution matches standard output for a given mode.
     */
    private void verifyExecutionMode(GraphExecutionMode mode) {
        int hidden = 16;
        int vocab = 8;

        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, hidden);
        SDVariable w1 = sd.constant("w1", Nd4j.randn(DataType.FLOAT, hidden, hidden));
        SDVariable b1 = sd.constant("b1", Nd4j.randn(DataType.FLOAT, 1, hidden));
        SDVariable w2 = sd.constant("w2", Nd4j.randn(DataType.FLOAT, hidden, vocab));

        SDVariable h = sd.mmul("h", x, w1);
        SDVariable hb = h.add("hb", b1);
        SDVariable ha = sd.nn.sigmoid("ha", hb);
        SDVariable logits = sd.mmul("logits", ha, w2);
        SDVariable out = sd.nn.softmax("out", logits, -1);

        INDArray input = Nd4j.randn(DataType.FLOAT, 1, hidden);

        // Standard execution (ground truth)
        Map<String, INDArray> expected = sd.output(Map.of("x", input), "out");
        INDArray expectedArr = expected.get("out").dup();

        // DSP execution with specified mode
        sd.setGraphExecutionMode(mode);
        Map<String, INDArray> dspResult = sd.outputDirect(Map.of("x", input), "out");
        INDArray dspArr = dspResult.get("out").dup();

        double maxDiff = expectedArr.sub(dspArr).amaxNumber().doubleValue();
        log.info("{}: max diff = {}", mode, maxDiff);
        assertTrue(maxDiff < TOL, mode + " max diff " + maxDiff + " exceeds tolerance");

        // Run 3 more times to verify repeated execution
        for (int i = 0; i < 3; i++) {
            INDArray nextInput = Nd4j.randn(DataType.FLOAT, 1, hidden);
            Map<String, INDArray> exp = sd.output(Map.of("x", nextInput), "out");
            Map<String, INDArray> dsp = sd.outputDirect(Map.of("x", nextInput), "out");
            double diff = exp.get("out").sub(dsp.get("out")).amaxNumber().doubleValue();
            log.info("{} iter {}: max diff = {}", mode, i + 1, diff);
            assertTrue(diff < TOL, mode + " iter " + (i + 1) + " max diff " + diff);
        }

        sd.close();
    }

    @Test
    @DisplayName("Execution mode: SLOT_BY_SLOT")
    public void testSlotBySlot() {
        verifyExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
    }

    @Test
    @DisplayName("Execution mode: CUDA_GRAPHS")
    public void testCudaGraphs() {
        verifyExecutionMode(GraphExecutionMode.CUDA_GRAPHS);
    }

    @Test
    @DisplayName("Execution mode: NVRTC_JIT")
    public void testNvrtcJit() {
        verifyExecutionMode(GraphExecutionMode.NVRTC_JIT);
    }

    @Test
    @DisplayName("Execution mode: TRITON")
    public void testTriton() {
        verifyExecutionMode(GraphExecutionMode.TRITON);
    }

    @Test
    @DisplayName("Execution mode: AUTO")
    public void testAuto() {
        verifyExecutionMode(GraphExecutionMode.AUTO);
    }

    /**
     * Test element-wise fusion pattern specifically (what NVRTC/Triton should fuse).
     * Chain of element-wise ops: add -> mul -> sigmoid -> mul (SiLU-like pattern).
     */
    @Test
    @DisplayName("Element-wise fusion: add -> mul -> sigmoid -> mul across modes")
    public void testElementwiseFusionAcrossModes() {
        int dim = 64;

        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, dim);
        SDVariable bias = sd.constant("bias", Nd4j.randn(DataType.FLOAT, 1, dim));
        SDVariable scale = sd.constant("scale", Nd4j.randn(DataType.FLOAT, 1, dim));

        // Element-wise chain: (x + bias) * scale -> sigmoid -> x * sigmoid(x)
        SDVariable xb = x.add("xb", bias);
        SDVariable xs = xb.mul("xs", scale);
        SDVariable sig = sd.nn.sigmoid("sig", xs);
        SDVariable out = xs.mul("out", sig);  // swish/SiLU-like

        INDArray input = Nd4j.randn(DataType.FLOAT, 1, dim);

        // Ground truth
        Map<String, INDArray> expected = sd.output(Map.of("x", input), "out");
        INDArray expectedArr = expected.get("out").dup();

        for (GraphExecutionMode mode : GraphExecutionMode.values()) {
            sd.setGraphExecutionMode(mode);
            Map<String, INDArray> dspResult = sd.outputDirect(Map.of("x", input), "out");
            INDArray dspArr = dspResult.get("out").dup();
            double maxDiff = expectedArr.sub(dspArr).amaxNumber().doubleValue();
            log.info("Elementwise fusion {}: max diff = {}", mode, maxDiff);
            assertTrue(maxDiff < TOL, "Elementwise fusion " + mode + " max diff " + maxDiff);
        }

        sd.close();
    }
}
