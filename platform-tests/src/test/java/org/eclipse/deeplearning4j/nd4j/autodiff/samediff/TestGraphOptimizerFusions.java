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
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.optimize.GraphOptimizer;
import org.nd4j.autodiff.samediff.optimize.OptimizerSet;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.transforms.custom.RmsNorm;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for GraphOptimizer fusion patterns on synthetic graphs.
 *
 * These tests construct small SameDiff graphs that mimic the decomposed
 * patterns found in ONNX-exported models and verify that the optimizer
 * correctly fuses them into single ops.
 */
@Slf4j
public class TestGraphOptimizerFusions {

    /**
     * Optimize a graph without setting requiredOutputs (avoids dup variable name issues).
     * We pass an empty list so GraphOptimizer.optimize() doesn't call setOutputs.
     */
    private SameDiff optimizeGraph(SameDiff sd) {
        return GraphOptimizer.optimize(sd, Collections.emptyList());
    }

    /**
     * Test RmsNorm op directly with SameDiff (forward pass).
     * Constructs rms_norm(x, gamma, eps) and verifies output shape.
     */
    @Test
    @DisplayName("RmsNorm op: forward pass shape and value")
    public void testRmsNormOpForward() {
        SameDiff sd = SameDiff.create();
        int features = 8;
        int batch = 2;
        int seq = 4;

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, -1, features);
        SDVariable gamma = sd.constant("gamma", Nd4j.ones(DataType.FLOAT, features));
        SDVariable out = new RmsNorm(sd, x, gamma, 1e-5).outputVariable();
        sd.setOutputs(Collections.singletonList(out.name()));

        // Execute with concrete values
        INDArray xArr = Nd4j.randn(DataType.FLOAT, batch, seq, features);
        Map<String, INDArray> placeholders = new HashMap<>();
        placeholders.put("x", xArr);

        Map<String, INDArray> results = sd.output(placeholders, out.name());
        INDArray result = results.get(out.name());

        assertNotNull(result, "RmsNorm output should not be null");
        assertArrayEquals(new long[]{batch, seq, features}, result.shape(),
            "RmsNorm output shape should match input");

        // Verify numerical correctness: rms_norm(x) = x / sqrt(mean(x^2) + eps) * gamma
        // With gamma=1, this simplifies to x / sqrt(mean(x^2) + eps)
        INDArray xSquared = xArr.mul(xArr);
        INDArray meanSquared = xSquared.mean(true, -1);
        INDArray rms = Nd4j.math().sqrt(meanSquared.add(1e-5));
        INDArray expected = xArr.div(rms);

        double maxDiff = expected.sub(result).amaxNumber().doubleValue();
        log.info("RmsNorm max absolute error: {}", maxDiff);
        assertTrue(maxDiff < 1e-4, "RmsNorm output should match manual computation, maxDiff=" + maxDiff);
    }

    /**
     * Test RmsNorm op with non-unit gamma weights.
     */
    @Test
    @DisplayName("RmsNorm op: scaled gamma weights")
    public void testRmsNormOpWithGamma() {
        SameDiff sd = SameDiff.create();
        int features = 4;

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, features);
        INDArray gammaArr = Nd4j.create(new float[]{2.0f, 0.5f, 1.0f, 3.0f});
        SDVariable gamma = sd.constant("gamma", gammaArr);
        SDVariable out = new RmsNorm(sd, x, gamma, 1e-6).outputVariable();
        sd.setOutputs(Collections.singletonList(out.name()));

        INDArray xArr = Nd4j.create(new float[][]{{1.0f, 2.0f, 3.0f, 4.0f}});
        Map<String, INDArray> results = sd.output(Map.of("x", xArr), out.name());
        INDArray result = results.get(out.name());

        assertNotNull(result);
        // Verify gamma scaling: output should be scaled by gamma
        // rms = sqrt(mean([1,4,9,16]) + 1e-6) = sqrt(7.5 + 1e-6)
        double rms = Math.sqrt(7.5 + 1e-6);
        double[] expectedVals = new double[]{
            1.0 / rms * 2.0,
            2.0 / rms * 0.5,
            3.0 / rms * 1.0,
            4.0 / rms * 3.0
        };

        for (int i = 0; i < features; i++) {
            double actual = result.getDouble(0, i);
            assertEquals(expectedVals[i], actual, 1e-4,
                "RmsNorm output[" + i + "] should match expected value");
        }
    }

    /**
     * Test that the decomposed RMSNorm pattern (ONNX-style) is fused into a single rms_norm op.
     *
     * Pattern: pow(x,2) -> reduce_mean -> add_scalar(eps) -> sqrt -> div(x, _) -> mul(_, weight)
     */
    @Test
    @DisplayName("RMSNorm fusion: ONNX decomposed pattern B")
    public void testRMSNormDecomposedFusion() {
        SameDiff sd = SameDiff.create();
        int features = 8;

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, features);
        // Use non-unit weights to avoid MultiplyOne algebraic optimization removing the anchor mul
        SDVariable weight = sd.constant("weight", Nd4j.rand(DataType.FLOAT, features).addi(0.5));

        // Build decomposed RMSNorm: pow(x,2) -> mean -> add(eps) -> sqrt -> div(x,_) -> mul(_,weight)
        SDVariable xSquared = sd.math().pow(x, 2.0);
        SDVariable meanSquared = xSquared.mean(true, -1);
        SDVariable meanPlusEps = meanSquared.add(1e-5);
        SDVariable sqrtVal = sd.math().sqrt(meanPlusEps);
        SDVariable normalized = x.div(sqrtVal);
        SDVariable output = normalized.mul(weight);

        int opsBefore = sd.getOps().size();
        log.info("Decomposed RMSNorm graph: {} ops before optimization", opsBefore);
        logOpTypes(sd, "Before");

        SameDiff optimized = optimizeGraph(sd);

        int opsAfter = optimized.getOps().size();
        log.info("After optimization: {} ops (removed {})", opsAfter, opsBefore - opsAfter);
        logOpTypes(optimized, "After");

        assertTrue(hasOpType(optimized, "rms_norm"), "Optimized graph should contain rms_norm op");
        assertFalse(hasOpTypeIgnoreCase(optimized, "pow"), "Optimized graph should not contain pow op");
        assertTrue(opsAfter < opsBefore, "Optimized graph should have fewer ops");
    }

    /**
     * Test that the decomposed RMSNorm pattern A (rsqrt-based) is fused.
     *
     * Pattern: mul(x,x) -> mean -> add(eps) -> rsqrt -> mul(x, _) -> mul(_, weight)
     */
    @Test
    @DisplayName("RMSNorm fusion: rsqrt-based pattern A")
    public void testRMSNormRsqrtFusion() {
        SameDiff sd = SameDiff.create();
        int features = 8;

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, features);
        // Use non-unit weights to avoid MultiplyOne algebraic optimization removing the anchor mul
        SDVariable weight = sd.constant("weight", Nd4j.rand(DataType.FLOAT, features).addi(0.5));

        // Build rsqrt-based RMSNorm: mul(x,x) -> mean -> add_scalar(eps) -> rsqrt -> mul(x,_) -> mul(_,weight)
        SDVariable xSquared = x.mul(x);
        SDVariable meanSquared = xSquared.mean(true, -1);
        SDVariable meanPlusEps = meanSquared.add(1e-5);
        SDVariable rsqrtVal = sd.math().rsqrt(meanPlusEps);
        SDVariable normalized = x.mul(rsqrtVal);
        SDVariable output = normalized.mul(weight);

        int opsBefore = sd.getOps().size();
        log.info("Rsqrt RMSNorm graph: {} ops before optimization", opsBefore);

        SameDiff optimized = optimizeGraph(sd);

        int opsAfter = optimized.getOps().size();
        log.info("After optimization: {} ops (removed {})", opsAfter, opsBefore - opsAfter);
        logOpTypes(optimized, "After");

        assertTrue(hasOpType(optimized, "rms_norm"), "Optimized graph should contain rms_norm op");
        assertTrue(opsAfter < opsBefore, "Optimized graph should have fewer ops");
    }

    /**
     * Test that the decomposed softmax pattern is fused into a single softmax op.
     *
     * Pattern: reduce_max(x) -> sub(x, max) -> exp -> reduce_sum(exp) -> div(exp, sum)
     */
    @Test
    @DisplayName("Softmax fusion: decomposed pattern")
    public void testDecomposedSoftmaxFusion() {
        SameDiff sd = SameDiff.create();

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 10);

        // Build decomposed softmax: reduce_max -> sub -> exp -> reduce_sum -> div
        SDVariable maxVal = x.max(true, -1);
        SDVariable shifted = x.sub(maxVal);
        SDVariable expVal = sd.math().exp(shifted);
        SDVariable sumExp = expVal.sum(true, -1);
        SDVariable output = expVal.div(sumExp);

        int opsBefore = sd.getOps().size();
        log.info("Decomposed softmax graph: {} ops before optimization", opsBefore);
        logOpTypes(sd, "Before");

        SameDiff optimized = optimizeGraph(sd);

        int opsAfter = optimized.getOps().size();
        log.info("After optimization: {} ops (removed {})", opsAfter, opsBefore - opsAfter);
        logOpTypes(optimized, "After");

        assertTrue(hasOpType(optimized, "softmax"), "Optimized graph should contain softmax op");
        assertFalse(hasOpType(optimized, "reduce_max"), "Optimized graph should not contain reduce_max op");
        assertTrue(opsAfter < opsBefore, "Optimized graph should have fewer ops");
    }

    /**
     * Test that the decomposed softmax with explicit axis is correctly fused
     * and the axis is preserved.
     */
    @Test
    @DisplayName("Softmax fusion: axis preservation")
    public void testDecomposedSoftmaxAxisPreservation() {
        SameDiff sd = SameDiff.create();

        // 3D input: [batch, seq, vocab]
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, -1, 32);

        // Decomposed softmax along axis=2 (last dim for 3D)
        SDVariable maxVal = x.max(true, 2);
        SDVariable shifted = x.sub(maxVal);
        SDVariable expVal = sd.math().exp(shifted);
        SDVariable sumExp = expVal.sum(true, 2);
        SDVariable output = expVal.div(sumExp);

        SameDiff optimized = optimizeGraph(sd);

        assertTrue(hasOpType(optimized, "softmax"), "Optimized graph should contain softmax op");

        // Find the softmax op and verify its dimension
        for (SameDiffOp op : optimized.getOps().values()) {
            if ("softmax".equals(op.getOp().opName()) && op.getOp() instanceof DynamicCustomOp) {
                long[] iArgs = ((DynamicCustomOp) op.getOp()).iArgs();
                log.info("Softmax op iArgs: {}", Arrays.toString(iArgs));
                assertTrue(iArgs != null && iArgs.length > 0 && iArgs[0] == 2,
                    "Softmax dimension should be 2, got: " + Arrays.toString(iArgs));
                break;
            }
        }
    }

    /**
     * Test that a non-softmax div pattern is NOT incorrectly fused.
     */
    @Test
    @DisplayName("Softmax fusion: negative test (non-matching div)")
    public void testNonSoftmaxDivNotFused() {
        SameDiff sd = SameDiff.create();

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 10);
        SDVariable y = sd.placeHolder("y", DataType.FLOAT, -1, 10);
        SDVariable output = x.div(y);

        SameDiff optimized = optimizeGraph(sd);

        assertFalse(hasOpType(optimized, "softmax"), "Non-softmax div should not produce a softmax op");
    }

    /**
     * Test that a non-RMSNorm mul pattern is NOT incorrectly fused.
     */
    @Test
    @DisplayName("RMSNorm fusion: negative test (non-matching mul)")
    public void testNonRMSNormMulNotFused() {
        SameDiff sd = SameDiff.create();

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 10);
        SDVariable y = sd.placeHolder("y", DataType.FLOAT, -1, 10);
        SDVariable output = x.mul(y);

        SameDiff optimized = optimizeGraph(sd);

        assertFalse(hasOpType(optimized, "rms_norm"), "Non-RMSNorm mul should not produce rms_norm op");
    }

    /**
     * Test multiple RMSNorm + softmax patterns in the same graph (mimics a transformer layer).
     */
    @Test
    @DisplayName("Multi-pattern fusion: RMSNorm + Softmax in one graph")
    public void testMultiPatternFusion() {
        SameDiff sd = SameDiff.create();
        int features = 8;

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, features);
        // Use non-unit weights to avoid MultiplyOne algebraic optimization removing the anchor mul
        SDVariable normWeight = sd.constant("norm_weight", Nd4j.rand(DataType.FLOAT, features).addi(0.5));

        // RMSNorm pattern (pattern B)
        SDVariable xSquared = sd.math().pow(x, 2.0);
        SDVariable meanSquared = xSquared.mean(true, -1);
        SDVariable meanPlusEps = meanSquared.add(1e-5);
        SDVariable sqrtVal = sd.math().sqrt(meanPlusEps);
        SDVariable normalized = x.div(sqrtVal);
        SDVariable normOut = normalized.mul(normWeight);

        // Softmax pattern
        SDVariable maxVal = normOut.max(true, -1);
        SDVariable shifted = normOut.sub(maxVal);
        SDVariable expVal = sd.math().exp(shifted);
        SDVariable sumExp = expVal.sum(true, -1);
        SDVariable output = expVal.div(sumExp);

        int opsBefore = sd.getOps().size();
        log.info("Multi-pattern graph: {} ops before optimization", opsBefore);

        SameDiff optimized = optimizeGraph(sd);

        int opsAfter = optimized.getOps().size();
        log.info("After optimization: {} ops (removed {})", opsAfter, opsBefore - opsAfter);
        logOpTypes(optimized, "After");

        assertTrue(hasOpType(optimized, "rms_norm"), "Optimized graph should contain rms_norm op");
        assertTrue(hasOpType(optimized, "softmax"), "Optimized graph should contain softmax op");
        assertTrue(opsAfter < opsBefore, "Optimized graph should have fewer ops");

        log.info("Multi-pattern: {} -> {} ops ({}% reduction)", opsBefore, opsAfter,
            String.format("%.1f", 100.0 * (opsBefore - opsAfter) / opsBefore));
    }

    /**
     * Test sigmoid(x) * x -> swish(x) fusion.
     */
    @Test
    @DisplayName("Activation fusion: sigmoid * x -> swish")
    public void testSigmoidMulToSwishFusion() {
        SameDiff sd = SameDiff.create();

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        SDVariable sigmoidX = sd.nn().sigmoid(x);
        SDVariable output = sigmoidX.mul(x);

        int opsBefore = sd.getOps().size();
        SameDiff optimized = optimizeGraph(sd);
        int opsAfter = optimized.getOps().size();

        log.info("Sigmoid*x -> swish: {} -> {} ops", opsBefore, opsAfter);

        assertTrue(hasOpType(optimized, "swish"), "Optimized graph should contain swish op");
        assertFalse(hasOpType(optimized, "sigmoid"), "Optimized graph should not contain sigmoid op");
    }

    /**
     * Confirm fused RmsNorm kernel matches decomposed computation numerically.
     * Builds both a decomposed graph and a direct RmsNorm op, runs with same input,
     * and verifies outputs match within tolerance.
     */
    @Test
    @DisplayName("RmsNorm: fused kernel vs decomposed numerical equivalence")
    public void testRmsNormFusedVsDecomposed() {
        int[] hiddenSizes = {4, 64, 576};  // 576 = SmolDocling hidden dim
        int batch = 2;
        int seq = 8;

        for (int features : hiddenSizes) {
            INDArray xArr = Nd4j.randn(DataType.FLOAT, batch, seq, features);
            INDArray gammaArr = Nd4j.randn(DataType.FLOAT, features).addi(1.0);  // gamma ~N(1,1)
            double eps = 1e-5;

            // Decomposed reference: x / sqrt(mean(x^2) + eps) * gamma
            INDArray xSquared = xArr.mul(xArr);
            INDArray meanSq = xSquared.mean(true, -1);
            INDArray rms = Nd4j.math().sqrt(meanSq.add(eps));
            INDArray normalized = xArr.div(rms);
            INDArray expected = normalized.mul(gammaArr);

            // Fused kernel via SameDiff RmsNorm op
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, -1, features);
            SDVariable gamma = sd.constant("gamma", gammaArr);
            SDVariable out = new RmsNorm(sd, x, gamma, eps).outputVariable();
            sd.setOutputs(Collections.singletonList(out.name()));

            INDArray result = sd.output(Map.of("x", xArr), out.name()).get(out.name());

            double maxDiff = expected.sub(result).amaxNumber().doubleValue();
            log.info("RmsNorm fused vs decomposed (features={}): maxDiff={}", features, maxDiff);
            assertTrue(maxDiff < 1e-3,
                "Fused RmsNorm should match decomposed for features=" + features + ", maxDiff=" + maxDiff);
        }
    }

    /**
     * Confirm that optimizing a decomposed RMSNorm graph produces numerically equivalent
     * output when executed. This tests the full pipeline: build decomposed graph ->
     * optimize (fuse to rms_norm) -> execute -> compare to reference.
     */
    @Test
    @DisplayName("RmsNorm fusion: end-to-end numerical correctness")
    public void testRmsNormFusionEndToEndCorrectness() {
        int features = 64;
        int batch = 2;
        int seq = 4;
        double eps = 1e-5;

        INDArray xArr = Nd4j.randn(DataType.FLOAT, batch, seq, features);
        INDArray gammaArr = Nd4j.randn(DataType.FLOAT, features).addi(1.0);

        // Compute reference output using decomposed ops
        INDArray xSquared = xArr.mul(xArr);
        INDArray meanSq = xSquared.mean(true, -1);
        INDArray rms = Nd4j.math().sqrt(meanSq.add(eps));
        INDArray expected = xArr.div(rms).mul(gammaArr);

        // Build decomposed graph (Pattern B: pow -> mean -> add -> sqrt -> div -> mul)
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, -1, features);
        SDVariable weight = sd.constant("weight", gammaArr);

        SDVariable xSq = sd.math().pow(x, 2.0);
        SDVariable meanSquared = xSq.mean(true, -1);
        SDVariable meanPlusEps = meanSquared.add(eps);
        SDVariable sqrtVal = sd.math().sqrt(meanPlusEps);
        SDVariable normalized = x.div(sqrtVal);
        SDVariable output = normalized.mul(weight);
        sd.setOutputs(Collections.singletonList(output.name()));

        // Execute BEFORE optimization
        INDArray beforeOpt = sd.output(Map.of("x", xArr), output.name()).get(output.name());
        double beforeDiff = expected.sub(beforeOpt).amaxNumber().doubleValue();
        log.info("Before optimization maxDiff: {}", beforeDiff);
        assertTrue(beforeDiff < 1e-4, "Pre-optimization output should match reference");

        // Optimize (should fuse to rms_norm)
        String outputName = output.name();
        SameDiff optimized = optimizeGraph(sd);
        assertTrue(hasOpType(optimized, "rms_norm"), "Should fuse to rms_norm");

        // Find the rms_norm output variable for execution
        String rmsNormOutput = findOpOutput(optimized, "rms_norm");
        assertNotNull(rmsNormOutput, "Should find rms_norm output variable");
        optimized.setOutputs(Collections.singletonList(rmsNormOutput));

        // Execute AFTER optimization — the fused kernel must produce same output
        INDArray afterOpt = optimized.output(Map.of("x", xArr), rmsNormOutput).get(rmsNormOutput);
        double afterDiff = expected.sub(afterOpt).amaxNumber().doubleValue();
        log.info("After optimization maxDiff: {}", afterDiff);
        assertTrue(afterDiff < 1e-3,
            "Fused rms_norm output should match decomposed reference, maxDiff=" + afterDiff);

        // Also verify fused matches pre-optimization
        double fusedVsDecomposed = beforeOpt.sub(afterOpt).amaxNumber().doubleValue();
        log.info("Fused vs decomposed graph maxDiff: {}", fusedVsDecomposed);
        assertTrue(fusedVsDecomposed < 1e-3,
            "Fused graph should produce same output as decomposed graph, maxDiff=" + fusedVsDecomposed);
    }

    /**
     * Test RmsNorm with transformer-sized hidden dimensions and multiple stacked layers.
     * Mimics a real decoder: rms_norm -> linear -> rms_norm -> linear.
     */
    @Test
    @DisplayName("RmsNorm fusion: stacked layers like a transformer decoder")
    public void testRmsNormStackedLayers() {
        int features = 128;
        int batch = 1;
        int seq = 4;
        double eps = 1e-5;

        INDArray xArr = Nd4j.randn(DataType.FLOAT, batch, seq, features);
        INDArray gamma1 = Nd4j.randn(DataType.FLOAT, features).addi(1.0);
        INDArray gamma2 = Nd4j.randn(DataType.FLOAT, features).addi(1.0);
        INDArray wProj = Nd4j.randn(DataType.FLOAT, features, features).divi(Math.sqrt(features));

        // Reference: rms_norm -> matmul -> rms_norm -> matmul
        INDArray ref1 = rmsNormRef(xArr, gamma1, eps);
        INDArray projected = ref1.reshape(batch * seq, features).mmul(wProj).reshape(batch, seq, features);
        INDArray ref2 = rmsNormRef(projected, gamma2, eps);

        // Build decomposed graph with 2 stacked rms_norm + linear layers
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, -1, features);
        SDVariable g1 = sd.constant("gamma1", gamma1);
        SDVariable g2 = sd.constant("gamma2", gamma2);
        SDVariable w = sd.constant("w_proj", wProj);

        // Layer 1: decomposed rms_norm (pattern B)
        SDVariable xSq1 = sd.math().pow(x, 2.0);
        SDVariable mean1 = xSq1.mean(true, -1);
        SDVariable eps1 = mean1.add(eps);
        SDVariable sqrt1 = sd.math().sqrt(eps1);
        SDVariable norm1 = x.div(sqrt1);
        SDVariable out1 = norm1.mul(g1);

        // Linear projection
        SDVariable proj = sd.mmul(out1, w);

        // Layer 2: decomposed rms_norm (pattern B)
        SDVariable xSq2 = sd.math().pow(proj, 2.0);
        SDVariable mean2 = xSq2.mean(true, -1);
        SDVariable eps2 = mean2.add(eps);
        SDVariable sqrt2 = sd.math().sqrt(eps2);
        SDVariable norm2 = proj.div(sqrt2);
        SDVariable out2 = norm2.mul(g2);
        sd.setOutputs(Collections.singletonList(out2.name()));

        int opsBefore = sd.getOps().size();

        // Optimize
        SameDiff optimized = optimizeGraph(sd);
        int opsAfter = optimized.getOps().size();
        log.info("Stacked RmsNorm: {} -> {} ops", opsBefore, opsAfter);

        // Should have 2 rms_norm ops
        long rmsNormCount = optimized.getOps().values().stream()
            .filter(op -> "rms_norm".equals(op.getOp().opName())).count();
        assertEquals(2, rmsNormCount, "Should have 2 fused rms_norm ops");

        // Find the last rms_norm output for execution
        String lastRmsNormOutput = null;
        for (SameDiffOp op : optimized.getOps().values()) {
            if ("rms_norm".equals(op.getOp().opName())) {
                List<String> outputs = op.getOutputsOfOp();
                if (outputs != null && !outputs.isEmpty()) {
                    lastRmsNormOutput = outputs.get(0);
                }
            }
        }
        assertNotNull(lastRmsNormOutput, "Should find rms_norm output");
        optimized.setOutputs(Collections.singletonList(lastRmsNormOutput));

        // Execute and compare
        INDArray result = optimized.output(Map.of("x", xArr), lastRmsNormOutput).get(lastRmsNormOutput);
        double maxDiff = ref2.sub(result).amaxNumber().doubleValue();
        log.info("Stacked layers maxDiff: {}", maxDiff);
        assertTrue(maxDiff < 1e-2,
            "Stacked fused rms_norm should match reference, maxDiff=" + maxDiff);
    }

    /**
     * Reference RMS norm computation for test verification.
     */
    private INDArray rmsNormRef(INDArray x, INDArray gamma, double eps) {
        INDArray xSq = x.mul(x);
        INDArray meanSq = xSq.mean(true, -1);
        INDArray rms = Nd4j.math().sqrt(meanSq.add(eps));
        return x.div(rms).mul(gamma);
    }

    /**
     * Test RMSNorm fusion with VARIABLE gamma (not CONSTANT) and an additional
     * reciprocal consumer on the sqrt output — matches the SmolDocling
     * SimplifiedLayerNormalization ONNX import pattern.
     */
    @Test
    @DisplayName("RmsNorm fusion: VARIABLE gamma + reciprocal consumer (SmolDocling pattern)")
    public void testRmsNormVariableGammaWithReciprocal() {
        SameDiff sd = SameDiff.create();
        int features = 576;  // SmolDocling hidden size

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, -1, features);
        // VARIABLE type gamma (like ONNX import produces for layernorm weights)
        SDVariable weight = sd.var("model.layers.0.input_layernorm.weight",
                Nd4j.rand(DataType.FLOAT, features).addi(0.5));

        // Build decomposed RMSNorm matching SimplifiedLayerNormalization.kt output:
        // pow(x,2) -> mean(keepDims=true) -> add_scalar(eps) -> sqrt -> div(x,_) -> mul(_,weight)
        SDVariable xSquared = sd.math().pow(x, 2.0);
        SDVariable meanSquared = xSquared.mean(true, -1);
        SDVariable meanPlusEps = meanSquared.add(1e-5);
        SDVariable sqrtVal = sd.math().sqrt(meanPlusEps);
        SDVariable normalized = x.div(sqrtVal);
        SDVariable output = normalized.mul(weight);

        // SmolDocling also creates a reciprocal of sqrt for the inv_std_var output
        // This gives sqrt TWO consumers, which previously blocked fusion
        SDVariable invRms = sd.math().reciprocal(sqrtVal);

        // Don't set outputs — GraphOptimizer removes the output variable and replaces it

        int opsBefore = sd.getOps().size();
        log.info("SmolDocling-pattern RMSNorm: {} ops before optimization", opsBefore);
        logOpTypes(sd, "Before");

        SameDiff optimized = optimizeGraph(sd);
        int opsAfter = optimized.getOps().size();
        log.info("After optimization: {} ops (removed {})", opsAfter, opsBefore - opsAfter);
        logOpTypes(optimized, "After");

        assertTrue(hasOpType(optimized, "rms_norm"),
            "Should fuse to rms_norm even with reciprocal consumer on sqrt");

        // Verify correctness
        INDArray xArr = Nd4j.randn(DataType.FLOAT, 1, 10, features);
        INDArray gammaArr = sd.getVariablesArrays().getArray("model.layers.0.input_layernorm.weight");
        INDArray ref = rmsNormRef(xArr, gammaArr, 1e-5);

        // Execute optimized graph
        String rmsNormOutput = null;
        for (SameDiffOp op : optimized.getOps().values()) {
            if ("rms_norm".equals(op.getOp().opName())) {
                rmsNormOutput = op.getOutputsOfOp().get(0);
                break;
            }
        }
        assertNotNull(rmsNormOutput, "Should find rms_norm output");
        optimized.setOutputs(Collections.singletonList(rmsNormOutput));

        INDArray result = optimized.output(Map.of("x", xArr), rmsNormOutput).get(rmsNormOutput);
        double maxDiff = ref.sub(result).amaxNumber().doubleValue();
        log.info("SmolDocling pattern maxDiff: {}", maxDiff);
        assertTrue(maxDiff < 1e-2,
            "Fused rms_norm should match reference, maxDiff=" + maxDiff);
    }

    // ─── MatMul + Add -> XwPlusB fusion tests ──────────────────────────────

    /**
     * Test that FuseMatMulWithAdd produces correct results for rank-2 inputs.
     * Pattern: matmul(x, w) + bias -> xw_plus_b(x, w, bias)
     */
    @Test
    @DisplayName("FuseMatMulWithAdd: rank-2 correctness")
    public void testFuseMatMulWithAddRank2() {
        int inFeatures = 64;
        int outFeatures = 32;
        int batch = 4;

        INDArray xArr = Nd4j.randn(DataType.FLOAT, batch, inFeatures);
        INDArray wArr = Nd4j.randn(DataType.FLOAT, inFeatures, outFeatures).divi(Math.sqrt(inFeatures));
        INDArray bArr = Nd4j.randn(DataType.FLOAT, outFeatures).muli(0.01);

        // Reference: matmul + add
        INDArray expected = xArr.mmul(wArr).addiRowVector(bArr);

        // Build decomposed graph
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, inFeatures);
        SDVariable w = sd.constant("w", wArr);
        SDVariable b = sd.constant("b", bArr);
        SDVariable mm = sd.mmul(x, w);
        SDVariable output = mm.add(b);
        sd.setOutputs(Collections.singletonList(output.name()));

        // Execute before optimization
        INDArray beforeOpt = sd.output(Map.of("x", xArr), output.name()).get(output.name());
        double beforeDiff = expected.sub(beforeOpt).amaxNumber().doubleValue();
        log.info("MatMul+Add rank-2 before opt maxDiff: {}", beforeDiff);
        assertTrue(beforeDiff < 1e-4, "Pre-opt should match reference");

        // Optimize
        SameDiff optimized = GraphOptimizer.optimize(sd, Collections.emptyList(),
            Collections.singletonList(new org.nd4j.autodiff.samediff.optimize.optimizations.LinearFusionOptimizations()));

        assertTrue(hasOpType(optimized, "xw_plus_b"), "Should fuse to xw_plus_b");
        assertFalse(hasOpType(optimized, "mmul"), "Should remove mmul");

        // Debug: dump xw_plus_b op inputs
        for (SameDiffOp sdOp : optimized.getOps().values()) {
            if ("xw_plus_b".equals(sdOp.getOp().opName())) {
                log.info("xw_plus_b inputs: {}", sdOp.getInputsToOp());
                log.info("xw_plus_b outputs: {}", sdOp.getOutputsOfOp());
                for (String input : sdOp.getInputsToOp()) {
                    SDVariable v = optimized.getVariable(input);
                    log.info("  input '{}': type={}, shape={}", input,
                        v != null ? v.getVariableType() : "null",
                        v != null ? java.util.Arrays.toString(v.getShape()) : "null");
                }
            }
        }

        String xwpbOutput = findOpOutput(optimized, "xw_plus_b");
        assertNotNull(xwpbOutput);
        optimized.setOutputs(Collections.singletonList(xwpbOutput));

        INDArray afterOpt = optimized.output(Map.of("x", xArr), xwpbOutput).get(xwpbOutput);
        double afterDiff = expected.sub(afterOpt).amaxNumber().doubleValue();
        log.info("MatMul+Add rank-2 after opt maxDiff: {}", afterDiff);
        log.info("Expected first row: {}", expected.getRow(0));
        log.info("Got first row: {}", afterOpt.getRow(0));
        assertTrue(afterDiff < 1e-3,
            "Fused xw_plus_b should match reference, maxDiff=" + afterDiff);
    }

    /**
     * Test that FuseMatMulWithAdd produces correct results for rank-3 inputs.
     * This is the critical case for transformers: [batch, seq, hidden] @ [hidden, out] + bias
     */
    @Test
    @DisplayName("FuseMatMulWithAdd: rank-3 correctness (transformer pattern)")
    public void testFuseMatMulWithAddRank3() {
        int hidden = 64;
        int outDim = 32;
        int batch = 1;
        int seqLen = 8;

        INDArray xArr = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden);
        INDArray wArr = Nd4j.randn(DataType.FLOAT, hidden, outDim).divi(Math.sqrt(hidden));
        INDArray bArr = Nd4j.randn(DataType.FLOAT, outDim).muli(0.01);

        // Reference: batch matmul + bias
        // For [1, 8, 64] @ [64, 32] = [1, 8, 32], then add [32] bias
        INDArray xFlat = xArr.reshape(batch * seqLen, hidden);
        INDArray expected = xFlat.mmul(wArr).addiRowVector(bArr).reshape(batch, seqLen, outDim);

        // Build decomposed graph
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, -1, hidden);
        SDVariable w = sd.constant("w", wArr);
        SDVariable b = sd.constant("b", bArr);
        SDVariable mm = sd.mmul(x, w);
        SDVariable output = mm.add(b);
        sd.setOutputs(Collections.singletonList(output.name()));

        // Execute before optimization
        INDArray beforeOpt = sd.output(Map.of("x", xArr), output.name()).get(output.name());
        double beforeDiff = expected.sub(beforeOpt).amaxNumber().doubleValue();
        log.info("MatMul+Add rank-3 before opt maxDiff: {}", beforeDiff);
        assertTrue(beforeDiff < 1e-4, "Pre-opt should match reference");

        // Optimize with only LinearFusion
        SameDiff optimized = GraphOptimizer.optimize(sd, Collections.emptyList(),
            Collections.singletonList(new org.nd4j.autodiff.samediff.optimize.optimizations.LinearFusionOptimizations()));

        assertTrue(hasOpType(optimized, "xw_plus_b"), "Should fuse to xw_plus_b");

        String xwpbOutput = findOpOutput(optimized, "xw_plus_b");
        assertNotNull(xwpbOutput);
        optimized.setOutputs(Collections.singletonList(xwpbOutput));

        INDArray afterOpt = optimized.output(Map.of("x", xArr), xwpbOutput).get(xwpbOutput);
        double afterDiff = expected.sub(afterOpt).amaxNumber().doubleValue();
        log.info("MatMul+Add rank-3 after opt maxDiff: {}", afterDiff);
        assertTrue(afterDiff < 1e-3,
            "Fused xw_plus_b rank-3 should match reference, maxDiff=" + afterDiff);

        // Verify output shape
        assertArrayEquals(new long[]{batch, seqLen, outDim}, afterOpt.shape(),
            "Output shape should be [batch, seq, outDim]");
    }

    /**
     * Test stacked matmul+add -> matmul+add (like transformer's attention projections).
     * Pattern: rms_norm -> matmul+add -> ... -> matmul+add
     */
    @Test
    @DisplayName("FuseMatMulWithAdd: stacked linear layers")
    public void testFuseMatMulWithAddStacked() {
        int hidden = 64;
        int batch = 1;
        int seqLen = 4;

        INDArray xArr = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden);
        INDArray w1 = Nd4j.randn(DataType.FLOAT, hidden, hidden).divi(Math.sqrt(hidden));
        INDArray b1 = Nd4j.randn(DataType.FLOAT, hidden).muli(0.01);
        INDArray w2 = Nd4j.randn(DataType.FLOAT, hidden, hidden).divi(Math.sqrt(hidden));
        INDArray b2 = Nd4j.randn(DataType.FLOAT, hidden).muli(0.01);

        // Reference: two stacked linear layers
        INDArray xFlat = xArr.reshape(batch * seqLen, hidden);
        INDArray layer1 = xFlat.mmul(w1).addiRowVector(b1);
        INDArray layer2 = layer1.mmul(w2).addiRowVector(b2).reshape(batch, seqLen, hidden);

        // Build decomposed graph
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, -1, hidden);
        SDVariable weight1 = sd.constant("w1", w1);
        SDVariable bias1 = sd.constant("b1", b1);
        SDVariable weight2 = sd.constant("w2", w2);
        SDVariable bias2 = sd.constant("b2", b2);

        // Layer 1: matmul + add
        SDVariable mm1 = sd.mmul(x, weight1);
        SDVariable out1 = mm1.add(bias1);

        // Layer 2: matmul + add
        SDVariable mm2 = sd.mmul(out1, weight2);
        SDVariable out2 = mm2.add(bias2);
        sd.setOutputs(Collections.singletonList(out2.name()));

        int opsBefore = sd.getOps().size();

        // Optimize with only LinearFusion
        SameDiff optimized = GraphOptimizer.optimize(sd, Collections.emptyList(),
            Collections.singletonList(new org.nd4j.autodiff.samediff.optimize.optimizations.LinearFusionOptimizations()));

        int opsAfter = optimized.getOps().size();
        log.info("Stacked linear: {} -> {} ops", opsBefore, opsAfter);

        long xwpbCount = optimized.getOps().values().stream()
            .filter(op -> "xw_plus_b".equals(op.getOp().opName())).count();
        assertEquals(2, xwpbCount, "Should have 2 xw_plus_b ops");

        // Find the last xw_plus_b output
        String lastOutput = null;
        for (SameDiffOp op : optimized.getOps().values()) {
            if ("xw_plus_b".equals(op.getOp().opName())) {
                List<String> outputs = op.getOutputsOfOp();
                if (outputs != null && !outputs.isEmpty()) {
                    lastOutput = outputs.get(0);
                }
            }
        }
        assertNotNull(lastOutput);
        optimized.setOutputs(Collections.singletonList(lastOutput));

        INDArray result = optimized.output(Map.of("x", xArr), lastOutput).get(lastOutput);
        double maxDiff = layer2.sub(result).amaxNumber().doubleValue();
        log.info("Stacked linear maxDiff: {}", maxDiff);
        assertTrue(maxDiff < 1e-2,
            "Stacked xw_plus_b should match reference, maxDiff=" + maxDiff);
    }

    /**
     * Test FuseMatMulWithAdd with SwiGLU activation (swish(gate) * up pattern).
     * This is the decoder MLP pattern: linear -> swish_mul -> linear
     */
    @Test
    @DisplayName("FuseMatMulWithAdd + SwiGLU: decoder MLP pattern")
    public void testFuseMatMulWithAddAndActivation() {
        int hidden = 64;
        int ffnDim = hidden * 4;  // 256
        int batch = 1;
        int seqLen = 4;

        INDArray xArr = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden);
        INDArray wGate = Nd4j.randn(DataType.FLOAT, hidden, ffnDim).divi(Math.sqrt(hidden));
        INDArray bGate = Nd4j.randn(DataType.FLOAT, ffnDim).muli(0.01);
        INDArray wUp = Nd4j.randn(DataType.FLOAT, hidden, ffnDim).divi(Math.sqrt(hidden));
        INDArray bUp = Nd4j.randn(DataType.FLOAT, ffnDim).muli(0.01);
        INDArray wDown = Nd4j.randn(DataType.FLOAT, ffnDim, hidden).divi(Math.sqrt(ffnDim));
        INDArray bDown = Nd4j.randn(DataType.FLOAT, hidden).muli(0.01);

        // Build decomposed graph
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, -1, hidden);

        // Gate projection: matmul + add
        SDVariable mmGate = sd.mmul(x, sd.constant("wGate", wGate));
        SDVariable gateOut = mmGate.add(sd.constant("bGate", bGate));

        // Swish activation: sigmoid(gate) * gate
        SDVariable sigmoidGate = sd.nn().sigmoid(gateOut);
        SDVariable swishOut = gateOut.mul(sigmoidGate);

        // Up projection: matmul + add
        SDVariable mmUp = sd.mmul(x, sd.constant("wUp", wUp));
        SDVariable upOut = mmUp.add(sd.constant("bUp", bUp));

        // SwiGLU: swish * up
        SDVariable ffn = swishOut.mul(upOut);

        // Down projection: matmul + add
        SDVariable mmDown = sd.mmul(ffn, sd.constant("wDown", wDown));
        SDVariable output = mmDown.add(sd.constant("bDown", bDown));
        sd.setOutputs(Collections.singletonList(output.name()));

        // Execute before optimization — use decomposed graph output as reference
        INDArray expected = sd.output(Map.of("x", xArr), output.name()).get(output.name());
        log.info("SwiGLU MLP decomposed output shape: {}, ordering: {}",
            java.util.Arrays.toString(expected.shape()), expected.ordering());

        // Optimize with ALL fusions (this is the key test)
        SameDiff optimized = optimizeGraph(sd);

        // Find a suitable output variable
        // Try the original output first, then look for the last xw_plus_b
        String optOutput = null;
        if (optimized.hasVariable(output.name())) {
            optOutput = output.name();
        } else {
            // Find the xw_plus_b that feeds into nothing (the last one)
            for (SameDiffOp op : optimized.getOps().values()) {
                if ("xw_plus_b".equals(op.getOp().opName())) {
                    List<String> outs = op.getOutputsOfOp();
                    if (outs != null && !outs.isEmpty()) {
                        optOutput = outs.get(0);
                    }
                }
            }
        }
        assertNotNull(optOutput, "Should find output variable after optimization");
        optimized.setOutputs(Collections.singletonList(optOutput));

        INDArray afterOpt = optimized.output(Map.of("x", xArr), optOutput).get(optOutput);
        double afterDiff = expected.sub(afterOpt).amaxNumber().doubleValue();
        log.info("SwiGLU MLP after opt maxDiff: {}", afterDiff);
        assertTrue(afterDiff < 1e-2,
            "Full fusion (matmul+add, swish, swiglu) should match reference, maxDiff=" + afterDiff);
    }

    /**
     * Direct test of xw_plus_b op vs mmul + add, bypassing the graph optimizer.
     * This isolates whether the op itself is broken.
     */
    @Test
    @DisplayName("xw_plus_b op: direct execution vs mmul+add")
    public void testXwPlusBDirect() {
        int inFeatures = 8;
        int outFeatures = 4;
        int batch = 2;

        INDArray x = Nd4j.randn(DataType.FLOAT, batch, inFeatures);
        INDArray w = Nd4j.randn(DataType.FLOAT, inFeatures, outFeatures);
        INDArray b = Nd4j.randn(DataType.FLOAT, outFeatures);

        // Reference: mmul + add
        INDArray expected = x.mmul(w).addiRowVector(b);

        // Direct xw_plus_b op execution
        INDArray xwpbResult = Nd4j.create(DataType.FLOAT, batch, outFeatures);
        DynamicCustomOp op = DynamicCustomOp.builder("xw_plus_b")
            .addInputs(x, w, b)
            .addOutputs(xwpbResult)
            .build();
        Nd4j.exec(op);

        double maxDiff = expected.sub(xwpbResult).amaxNumber().doubleValue();
        log.info("Direct xw_plus_b vs mmul+add maxDiff: {}", maxDiff);
        log.info("Expected:\n{}", expected);
        log.info("xw_plus_b result:\n{}", xwpbResult);
        assertTrue(maxDiff < 1e-4,
            "Direct xw_plus_b should match mmul+add, maxDiff=" + maxDiff);
    }

    /**
     * Direct test of xw_plus_b op with rank-3 input (bypassing graph optimizer).
     * Isolates whether the op handles higher-rank inputs correctly.
     */
    @Test
    @DisplayName("xw_plus_b op: direct rank-3 execution vs mmul+add")
    public void testXwPlusBDirectRank3() {
        int inFeatures = 8;
        int outFeatures = 4;
        int batch = 2;
        int seqLen = 3;

        INDArray x = Nd4j.randn(DataType.FLOAT, batch, seqLen, inFeatures);
        INDArray w = Nd4j.randn(DataType.FLOAT, inFeatures, outFeatures);
        INDArray b = Nd4j.randn(DataType.FLOAT, outFeatures);

        // Reference: flatten to 2D, mmul + add, reshape back
        INDArray xFlat = x.reshape(batch * seqLen, inFeatures);
        INDArray expected = xFlat.mmul(w).addiRowVector(b).reshape(batch, seqLen, outFeatures);

        // Direct xw_plus_b op execution with rank-3 input
        // Note: xw_plus_b's shape function should produce [batch, seqLen, outFeatures]
        DynamicCustomOp op = DynamicCustomOp.builder("xw_plus_b")
            .addInputs(x, w, b)
            .build();
        // Let the op calculate its own output shape
        INDArray[] results = Nd4j.exec(op);
        INDArray xwpbResult = results[0];

        log.info("Direct rank-3 xw_plus_b shape: {}, ordering: {}",
            java.util.Arrays.toString(xwpbResult.shape()), xwpbResult.ordering());
        log.info("Expected shape: {}, ordering: {}",
            java.util.Arrays.toString(expected.shape()), expected.ordering());

        double maxDiff = expected.sub(xwpbResult).amaxNumber().doubleValue();
        log.info("Direct rank-3 xw_plus_b vs mmul+add maxDiff: {}", maxDiff);

        if (maxDiff > 1e-4) {
            // Debug: print first slice
            log.info("Expected first slice:\n{}", expected.get(NDArrayIndex.point(0)));
            log.info("Got first slice:\n{}", xwpbResult.get(NDArrayIndex.point(0)));
        }

        assertTrue(maxDiff < 1e-4,
            "Direct rank-3 xw_plus_b should match mmul+add, maxDiff=" + maxDiff);
    }

    /**
     * Direct test of xw_plus_b op through SameDiff (bypassing optimizer but using DSP).
     * This tests the SameDiff -> DSP -> xw_plus_b path.
     */
    @Test
    @DisplayName("xw_plus_b op: SameDiff direct (no optimizer) rank-3")
    public void testXwPlusBSameDiffDirectRank3() {
        int inFeatures = 8;
        int outFeatures = 4;
        int batch = 2;
        int seqLen = 3;

        INDArray xArr = Nd4j.randn(DataType.FLOAT, batch, seqLen, inFeatures);
        INDArray wArr = Nd4j.randn(DataType.FLOAT, inFeatures, outFeatures);
        INDArray bArr = Nd4j.randn(DataType.FLOAT, outFeatures);

        // Reference: flatten to 2D, mmul + add, reshape back
        INDArray xFlat = xArr.reshape(batch * seqLen, inFeatures);
        INDArray expected = xFlat.mmul(wArr).addiRowVector(bArr).reshape(batch, seqLen, outFeatures);

        // Build SameDiff graph with xw_plus_b directly (no optimizer)
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, -1, inFeatures);
        SDVariable w = sd.constant("w", wArr);
        SDVariable b = sd.constant("b", bArr);

        // Manually wire xw_plus_b
        SDVariable result = new org.nd4j.linalg.api.ops.impl.transforms.custom.XwPlusB(
            sd, x, w, b, false, false, false).outputVariable();
        sd.setOutputs(Collections.singletonList(result.name()));

        INDArray actual = sd.output(Map.of("x", xArr), result.name()).get(result.name());
        log.info("SameDiff direct xw_plus_b shape: {}, ordering: {}",
            java.util.Arrays.toString(actual.shape()), actual.ordering());

        double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
        log.info("SameDiff direct rank-3 xw_plus_b maxDiff: {}", maxDiff);

        if (maxDiff > 1e-4) {
            log.info("Expected first slice:\n{}", expected.get(NDArrayIndex.point(0)));
            log.info("Got first slice:\n{}", actual.get(NDArrayIndex.point(0)));
        }

        assertTrue(maxDiff < 1e-3,
            "SameDiff direct rank-3 xw_plus_b should match, maxDiff=" + maxDiff);
    }

    // ─── Helpers ────────────────────────────────────────────────────────────

    /**
     * Find the output variable name of the first op with the given type.
     */
    private String findOpOutput(SameDiff sd, String opType) {
        for (SameDiffOp op : sd.getOps().values()) {
            if (opType.equals(op.getOp().opName())) {
                List<String> outputs = op.getOutputsOfOp();
                if (outputs != null && !outputs.isEmpty()) {
                    return outputs.get(0);
                }
            }
        }
        return null;
    }

    private boolean hasOpType(SameDiff sd, String opName) {
        return sd.getOps().values().stream()
            .anyMatch(op -> opName.equals(op.getOp().opName()));
    }

    private boolean hasOpTypeIgnoreCase(SameDiff sd, String opName) {
        return sd.getOps().values().stream()
            .anyMatch(op -> opName.equalsIgnoreCase(op.getOp().opName()));
    }

    @Test
    @DisplayName("SameDiff.dup() round-trip preserves LONG array values (byte-order test)")
    public void testDupPreservesLongValues() {
        // This tests the byte-order handling in BaseNDArray.toFlatArray() / Nd4j.createFromFlatArray().
        // asBytes() always returns big-endian bytes (DataOutputStream), so toFlatArray must label
        // the byteOrder as BE so createFromFlatArray byte-swaps correctly on LE platforms.
        SameDiff sd = SameDiff.create();
        long[] shapeValues = new long[]{-1, 3, 512, 512};
        INDArray shapeConst = Nd4j.createFromArray(shapeValues);
        sd.constant("target_shape", shapeConst);

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, -1, 3, 512, 512);
        SDVariable shapeVar = sd.getVariable("target_shape");
        SDVariable reshaped = sd.reshape(x, shapeVar);
        sd.setOutputs(Collections.singletonList(reshaped.name()));

        // Round-trip through FlatBuffers (same as GraphOptimizer.optimize)
        SameDiff duped = sd.dup();

        INDArray dupedShape = duped.getVariable("target_shape").getArr();
        assertNotNull(dupedShape, "Duped shape array should not be null");
        long[] dupedValues = dupedShape.toLongVector();
        log.info("Original: {}, Duped: {}", Arrays.toString(shapeValues), Arrays.toString(dupedValues));
        assertArrayEquals(shapeValues, dupedValues,
                "LONG array values corrupted by dup() round-trip. Expected " +
                Arrays.toString(shapeValues) + " but got " + Arrays.toString(dupedValues));
    }

    @Test
    @DisplayName("Constant folding through GraphOptimizer preserves LONG concat values")
    public void testConstantFoldingPreservesLongConcat() {
        // Tests that ConstantFunctionOptimizations.FoldConstantFunctions correctly
        // preserves LONG values through the optimize→dup cycle.
        SameDiff sd = SameDiff.create();
        // Create the pattern: Concat of scalar constants → Reshape target
        // This mimics the ONNX vision encoder pattern that was corrupted.
        INDArray c1 = Nd4j.createFromArray(new long[]{-1});
        INDArray c2 = Nd4j.createFromArray(new long[]{3});
        INDArray c3 = Nd4j.createFromArray(new long[]{512});
        INDArray c4 = Nd4j.createFromArray(new long[]{512});
        SDVariable s1 = sd.constant("s1", c1);
        SDVariable s2 = sd.constant("s2", c2);
        SDVariable s3 = sd.constant("s3", c3);
        SDVariable s4 = sd.constant("s4", c4);

        // Concat the [1]-shaped constants into a [4] shape array
        SDVariable shape = sd.concat(0, s1, s2, s3, s4);
        shape.rename("target_shape");

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, -1, 3, 512, 512);
        SDVariable reshaped = sd.reshape(x, shape);
        sd.setOutputs(Collections.singletonList(reshaped.name()));

        log.info("Before optimize: {} ops", sd.getOps().size());

        // Run full optimizer (includes ConstantFunctionOptimizations + dup)
        SameDiff optimized = GraphOptimizer.optimize(sd, sd.outputs());

        log.info("After optimize: {} ops", optimized.getOps().size());

        // The Concat should have been constant-folded. Check the folded value.
        INDArray foldedShape = optimized.getVariable("target_shape").getArr();
        assertNotNull(foldedShape, "Folded shape should not be null");
        long[] foldedValues = foldedShape.toLongVector();
        long[] expected = new long[]{-1, 3, 512, 512};
        log.info("Expected: {}, Folded: {}", Arrays.toString(expected), Arrays.toString(foldedValues));
        assertArrayEquals(expected, foldedValues,
                "Constant-folded LONG concat values corrupted. Expected " +
                Arrays.toString(expected) + " but got " + Arrays.toString(foldedValues) +
                ". This is the byte-order bug: asBytes() returns BE but toFlatArray labels as LE.");
    }

    private void logOpTypes(SameDiff sd, String label) {
        Map<String, Integer> counts = new TreeMap<>();
        for (SameDiffOp op : sd.getOps().values()) {
            String name = op.getOp() != null ? op.getOp().opName() : "null";
            counts.merge(name, 1, Integer::sum);
        }
        log.info("{} ops: {}", label, counts);
    }
}
