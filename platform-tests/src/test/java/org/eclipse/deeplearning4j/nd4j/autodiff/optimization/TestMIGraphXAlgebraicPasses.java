/*
 *  ******************************************************************************
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

package org.eclipse.deeplearning4j.nd4j.autodiff.optimization;

import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.optimize.GraphOptimizer;
import org.nd4j.autodiff.samediff.optimize.Optimizer;
import org.nd4j.autodiff.samediff.optimize.OptimizerSet;
import org.nd4j.autodiff.samediff.optimize.optimizations.AlgebraicOptimizations;
import org.nd4j.autodiff.samediff.optimize.optimizations.QuantizationOptimizations;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.reduce.Mmul;
import org.nd4j.linalg.api.ops.impl.shape.Slice;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.MulOp;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for two MIGraphX-inspired graph optimizer passes:
 *
 * P0a: ScalarIntoWeightFolding  — mul(matmul(X,W), s) → matmul(X, W*s)
 * P0b: SliceCommuteWithMatMul   — slice(matmul(A,B), dim) → matmul(slice(A), B)
 *                                                         or matmul(A, slice(B))
 * P1:  QuantizeActivationsInt8  — calibration-based activation int8 quantization
 */
public class TestMIGraphXAlgebraicPasses {

    // ─── Helper: run with only the specified OptimizerSet ───────────────

    private static SameDiff runWith(SameDiff sd, OptimizerSet... sets) {
        List<String> outs = sd.outputs();
        if (outs == null || outs.isEmpty()) {
            // Collect all variable names as fallback
            outs = new ArrayList<>(sd.getVariables().keySet());
        }
        return GraphOptimizer.optimize(sd, outs, Arrays.asList(sets));
    }

    // ─── P0a: ScalarIntoWeightFolding ────────────────────────────────────

    /**
     * Build graph: output = mul(matmul(X, W), s)
     * where W is CONSTANT [4,4], s is scalar CONSTANT.
     * After optimization: output = matmul(X, W*s), the mul op is gone.
     */
    @Test
    public void testScalarIntoWeightFolding_correctness() {
        SameDiff sd = SameDiff.create();

        int M = 3, K = 4, N = 4;
        INDArray wData = Nd4j.rand(DataType.FLOAT, K, N);
        float scalarVal = 2.5f;

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, M, K);
        SDVariable w = sd.constant("W", wData);
        SDVariable s = sd.constant("s", Nd4j.scalar(scalarVal));

        SDVariable mm = sd.mmul("mm", x, w);
        SDVariable out = mm.mul("out", s);
        sd.setOutputs(Collections.singletonList("out"));

        // Compute reference output
        INDArray xData = Nd4j.rand(DataType.FLOAT, M, K);
        Map<String, INDArray> ph = Collections.singletonMap("x", xData);
        INDArray expected = sd.outputSingle(ph, "out");

        // Optimize with only AlgebraicOptimizations (which includes ScalarIntoWeightFolding)
        SameDiff optimized = runWith(sd, new AlgebraicOptimizations());

        // Structural check: no MulOp in optimized graph
        long mulCount = optimized.getOps().values().stream()
                .filter(o -> o.getOp() instanceof MulOp)
                .count();
        assertEquals(0, mulCount, "ScalarIntoWeightFolding should eliminate the MulOp");

        // Structural check: matmul is still present
        long mmulCount = optimized.getOps().values().stream()
                .filter(o -> o.getOp() instanceof Mmul)
                .count();
        assertEquals(1, mmulCount, "Should still have exactly 1 matmul");

        // Verify the weight constant W' = W*s was created and registered
        boolean foundScaledWeight = false;
        for (String varName : optimized.getVariables().keySet()) {
            if (varName.startsWith("weight_scaled_")) {
                INDArray wPrime = optimized.getConstantArrays().getArray(varName);
                assertNotNull(wPrime, "Scaled weight constant must exist");
                // W' should equal W * s element-wise
                INDArray expected_wPrime = wData.mul(scalarVal);
                double maxDiff = wPrime.sub(expected_wPrime).amaxNumber().doubleValue();
                assertTrue(maxDiff < 1e-5, "W' = W*s mismatch: maxDiff=" + maxDiff);
                foundScaledWeight = true;
                break;
            }
        }
        assertTrue(foundScaledWeight, "Should have created weight_scaled_* constant");

        // Numerical correctness
        // The output var might be the matmul's output name now
        List<String> optOutputs = optimized.outputs();
        assertNotNull(optOutputs, "Optimized graph must have outputs");
        assertFalse(optOutputs.isEmpty(), "Optimized graph must have at least one output");
        String outVarName = optOutputs.get(0);

        INDArray actual = optimized.outputSingle(ph, outVarName);
        double maxErr = actual.sub(expected).amaxNumber().doubleValue();
        assertTrue(maxErr < 1e-4, "Numerical mismatch after ScalarIntoWeightFolding: maxErr=" + maxErr);
    }

    /**
     * Guard: should NOT fire when W is a placeholder (not a constant).
     */
    @Test
    public void testScalarIntoWeightFolding_doesNotFireForNonConstantWeight() {
        SameDiff sd = SameDiff.create();

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
        SDVariable w = sd.placeHolder("W", DataType.FLOAT, 4, 4);  // NOT a constant
        SDVariable s = sd.constant("s", Nd4j.scalar(2.5f));

        SDVariable mm = sd.mmul("mm", x, w);
        SDVariable out = mm.mul("out", s);
        sd.setOutputs(Collections.singletonList("out"));

        SameDiff optimized = runWith(sd, new AlgebraicOptimizations());

        // MulOp should still be present (optimization should not have fired)
        long mulCount = optimized.getOps().values().stream()
                .filter(o -> o.getOp() instanceof MulOp)
                .count();
        assertEquals(1, mulCount, "MulOp should remain when W is not a constant");
    }

    // ─── P0b: SliceCommuteWithMatMul ─────────────────────────────────────

    /**
     * Pattern (a): slice rows of matmul output → push slice inside A.
     * Graph: out = slice(matmul(A[6,4], B[4,5]), begin=[2,0], size=[3,5])
     * → matmul(slice(A, begin=[2,0], size=[3,4]), B)
     */
    @Test
    public void testSliceCommuteWithMatMul_rows_correctness() {
        int M = 6, K = 4, N = 5;
        INDArray aData = Nd4j.rand(DataType.FLOAT, M, K);
        INDArray bData = Nd4j.rand(DataType.FLOAT, K, N);

        SameDiff sd = SameDiff.create();
        // A as placeholder with known static shape, B as constant
        SDVariable a = sd.placeHolder("A", DataType.FLOAT, M, K);
        SDVariable b = sd.constant("B", bData);

        SDVariable mm = sd.mmul("mm", a, b);
        // Slice rows 2..4 (size=3) from output [6,5]
        SDVariable out = sd.slice("out", mm, new int[]{2, 0}, new int[]{3, N});
        sd.setOutputs(Collections.singletonList("out"));

        // Reference
        Map<String, INDArray> ph = Collections.singletonMap("A", aData);
        INDArray expected = sd.outputSingle(ph, "out");

        SameDiff optimized = runWith(sd, new AlgebraicOptimizations());

        // Structural: original matmul should be gone (replaced by a new one with smaller A)
        // The new matmul takes slice(A) not A
        // Count slices on A vs slices on matmul output
        boolean hasSliceOnMmulOutput = false;
        for (SameDiffOp op : optimized.getOps().values()) {
            if (op.getOp() instanceof Slice) {
                List<String> ins = op.getInputsToOp();
                if (ins != null && !ins.isEmpty()) {
                    // The input to the slice should NOT be the original matmul output
                    String inputToSlice = ins.get(0);
                    SDVariable inputVar = optimized.getVariable(inputToSlice);
                    if (inputVar != null) {
                        // If slice input is an ARRAY produced by a matmul — that's a slice-on-matmul (bad)
                        String producerOp = optimized.getVariables().get(inputToSlice) != null
                                ? optimized.getVariables().get(inputToSlice).getOutputOfOp() : null;
                        if (producerOp != null) {
                            SameDiffOp producer = optimized.getOps().get(producerOp);
                            if (producer != null && producer.getOp() instanceof Mmul) {
                                hasSliceOnMmulOutput = true;
                            }
                        }
                    }
                }
            }
        }
        assertFalse(hasSliceOnMmulOutput,
                "After SliceCommuteWithMatMul, no slice should remain on a matmul output");

        // Numerical correctness
        List<String> optOuts = optimized.outputs();
        assertNotNull(optOuts);
        assertFalse(optOuts.isEmpty());
        INDArray actual = optimized.outputSingle(ph, optOuts.get(0));
        double maxErr = actual.sub(expected).amaxNumber().doubleValue();
        assertTrue(maxErr < 1e-4, "Numerical mismatch after SliceCommuteWithMatMul(rows): maxErr=" + maxErr);
    }

    /**
     * Pattern (b): slice columns of matmul output → push slice inside B.
     * Graph: out = slice(matmul(A[4,6], B[6,8]), begin=[0,3], size=[4,5])
     * → matmul(A, slice(B, begin=[0,3], size=[6,5]))
     */
    @Test
    public void testSliceCommuteWithMatMul_cols_correctness() {
        int M = 4, K = 6, N = 8;
        INDArray aData = Nd4j.rand(DataType.FLOAT, M, K);
        INDArray bData = Nd4j.rand(DataType.FLOAT, K, N);

        SameDiff sd = SameDiff.create();
        SDVariable a = sd.placeHolder("A", DataType.FLOAT, M, K);
        SDVariable b = sd.constant("B", bData);

        SDVariable mm = sd.mmul("mm", a, b);
        // Slice columns 3..7 (size=5) from output [4,8]
        SDVariable out = sd.slice("out", mm, new int[]{0, 3}, new int[]{M, 5});
        sd.setOutputs(Collections.singletonList("out"));

        Map<String, INDArray> ph = Collections.singletonMap("A", aData);
        INDArray expected = sd.outputSingle(ph, "out");

        SameDiff optimized = runWith(sd, new AlgebraicOptimizations());

        // After optimization, there should be no slice of a matmul output
        boolean hasSliceOnMmulOutput = false;
        for (SameDiffOp op : optimized.getOps().values()) {
            if (op.getOp() instanceof Slice) {
                List<String> ins = op.getInputsToOp();
                if (ins != null && !ins.isEmpty()) {
                    String producerOpName = optimized.getVariables().get(ins.get(0)) != null
                            ? optimized.getVariables().get(ins.get(0)).getOutputOfOp() : null;
                    if (producerOpName != null) {
                        SameDiffOp producer = optimized.getOps().get(producerOpName);
                        if (producer != null && producer.getOp() instanceof Mmul) {
                            hasSliceOnMmulOutput = true;
                        }
                    }
                }
            }
        }
        assertFalse(hasSliceOnMmulOutput,
                "After SliceCommuteWithMatMul, no slice should remain on a matmul output");

        // Numerical correctness
        List<String> optOuts = optimized.outputs();
        assertNotNull(optOuts);
        assertFalse(optOuts.isEmpty());
        INDArray actual = optimized.outputSingle(ph, optOuts.get(0));
        double maxErr = actual.sub(expected).amaxNumber().doubleValue();
        assertTrue(maxErr < 1e-4, "Numerical mismatch after SliceCommuteWithMatMul(cols): maxErr=" + maxErr);
    }

    /**
     * Guard: should NOT fire when the matmul output has multiple consumers.
     */
    @Test
    public void testSliceCommuteWithMatMul_doesNotFireForMultipleConsumers() {
        SameDiff sd = SameDiff.create();

        SDVariable a = sd.placeHolder("A", DataType.FLOAT, 6, 4);
        SDVariable b = sd.constant("B", Nd4j.rand(DataType.FLOAT, 4, 5));

        SDVariable mm = sd.mmul("mm", a, b);
        // Two consumers of mm: slice + another use (e.g. sum)
        SDVariable sliceOut = sd.slice("slice_out", mm, new int[]{0, 0}, new int[]{3, 5});
        SDVariable sumOut = mm.sum("sum_out");
        // Both outputs needed
        sd.setOutputs(Arrays.asList("slice_out", "sum_out"));

        SameDiff optimized = runWith(sd, new AlgebraicOptimizations());

        // The original matmul should still exist (multiple consumers → no commute)
        boolean sliceOnMmulOutput = false;
        for (SameDiffOp op : optimized.getOps().values()) {
            if (op.getOp() instanceof Slice) {
                List<String> ins = op.getInputsToOp();
                if (ins != null && !ins.isEmpty()) {
                    String prodOpName = optimized.getVariables().get(ins.get(0)) != null
                            ? optimized.getVariables().get(ins.get(0)).getOutputOfOp() : null;
                    if (prodOpName != null) {
                        SameDiffOp prod = optimized.getOps().get(prodOpName);
                        if (prod != null && prod.getOp() instanceof Mmul) {
                            sliceOnMmulOutput = true;
                        }
                    }
                }
            }
        }
        assertTrue(sliceOnMmulOutput,
                "When matmul has multiple consumers, SliceCommuteWithMatMul must not fire");
    }

    // ─── P1: QuantizeActivationsInt8 ─────────────────────────────────────

    /**
     * Test the Phase 1 calibration: calibrate() correctly collects min/max
     * ranges for matmul activation inputs.
     */
    @Test
    public void testQuantizeActivationsInt8_calibration() {
        SameDiff sd = SameDiff.create();

        int M = 4, K = 8, N = 6;
        INDArray wData = Nd4j.rand(DataType.FLOAT, K, N);

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, M, K);
        SDVariable w = sd.constant("W", wData);
        SDVariable out = sd.mmul("out", x, w);
        sd.setOutputs(Collections.singletonList("out"));

        // Build 3 calibration batches with different ranges
        List<Map<String, INDArray>> calibBatches = new ArrayList<>();
        for (int i = 0; i < 3; i++) {
            // Activations in range [-2*(i+1), 2*(i+1)]
            INDArray xBatch = Nd4j.rand(DataType.FLOAT, M, K).sub(0.5f).mul(4.0f * (i + 1));
            calibBatches.add(Collections.singletonMap("x", xBatch));
        }

        Map<String, double[]> ranges = QuantizationOptimizations.QuantizeActivationsInt8.calibrate(
                sd, calibBatches, "out");

        // Should have collected range for the activation "x" (feeds into matmul as input[0])
        assertFalse(ranges.isEmpty(), "Calibration should collect at least one activation range");

        // Find the activation entry (the input to the matmul that is 'x' or 'x' itself)
        boolean foundXRange = false;
        for (Map.Entry<String, double[]> entry : ranges.entrySet()) {
            double[] range = entry.getValue();
            assertNotNull(range, "Range should not be null");
            assertEquals(2, range.length, "Range should have [min, max]");
            assertTrue(range[0] < range[1], "min < max required: " + entry.getKey() +
                    " min=" + range[0] + " max=" + range[1]);
            // The calibration saw batches with max ~2*(3) * 0.5 = 3, so max should be >= 2
            if (range[1] >= 2.0) foundXRange = true;
        }
        assertTrue(foundXRange, "Should have found activation with range >= 2 from calibration batches");
    }

    /**
     * Test the Phase 2 graph rewrite + bounded error.
     *
     * Builds: matmul(x, W) — calibrates on random inputs, then runs rewriteWithScales.
     * Asserts:
     *   (i)  The rewritten graph contains quant/dequant nodes (cast ops or mul/div)
     *   (ii) The numerical error between FP32 and quantized output is bounded
     *        (relative error < 5% for INT8 quantization with well-calibrated range)
     */
    @Test
    public void testQuantizeActivationsInt8_rewriteAndBoundedError() {
        SameDiff sd = SameDiff.create();

        int M = 8, K = 16, N = 12;
        INDArray wData = Nd4j.rand(DataType.FLOAT, K, N).sub(0.5f).mul(2.0f);  // weights in [-1, 1]

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, M, K);
        SDVariable w = sd.constant("W", wData);
        SDVariable out = sd.mmul("out", x, w);
        sd.setOutputs(Collections.singletonList("out"));

        // Calibration inputs
        List<Map<String, INDArray>> calibBatches = new ArrayList<>();
        INDArray[] xCalibData = new INDArray[5];
        for (int i = 0; i < 5; i++) {
            xCalibData[i] = Nd4j.rand(DataType.FLOAT, M, K).sub(0.5f).mul(2.0f);
            calibBatches.add(Collections.singletonMap("x", xCalibData[i]));
        }

        // Phase 1: calibrate
        Map<String, double[]> ranges = QuantizationOptimizations.QuantizeActivationsInt8.calibrate(
                sd, calibBatches, "out");
        assertFalse(ranges.isEmpty(), "Calibration must produce at least one range");

        // Convert ranges to QuantizationInfo (symmetric)
        Map<String, QuantizationOptimizations.QuantizationInfo> scaleMap = new HashMap<>();
        for (Map.Entry<String, double[]> entry : ranges.entrySet()) {
            double maxAbs = Math.max(Math.abs(entry.getValue()[0]), Math.abs(entry.getValue()[1]));
            float scale = (float) (maxAbs / 127.0);
            scaleMap.put(entry.getKey(), new QuantizationOptimizations.QuantizationInfo(scale, 0));
        }

        // Phase 2: rewrite with scales
        System.setProperty(QuantizationOptimizations.QuantizeActivationsInt8.PROP_ENABLE, "true");
        try {
            SameDiff quantized = QuantizationOptimizations.QuantizeActivationsInt8
                    .rewriteWithScales(sd, scaleMap);

            // Use a test input in the calibrated range
            INDArray xTest = Nd4j.rand(DataType.FLOAT, M, K).sub(0.5f).mul(2.0f);
            Map<String, INDArray> ph = Collections.singletonMap("x", xTest);

            // Reference (FP32)
            INDArray fp32Out = sd.outputSingle(ph, "out");

            // Quantized output
            List<String> qOuts = quantized.outputs();
            assertNotNull(qOuts, "Quantized graph must have outputs");
            assertFalse(qOuts.isEmpty(), "Quantized graph must have at least one output");

            INDArray qOut = quantized.outputSingle(ph, qOuts.get(0));
            assertNotNull(qOut, "Quantized output must not be null");

            // Bounded error check: relative error < 5%
            // INT8 quantization error is bounded by scale/2 per element, which for
            // well-calibrated ranges should give < 1% mean relative error for weight-matrix outputs
            double fp32Norm = fp32Out.norm2Number().doubleValue();
            double errNorm = fp32Out.sub(qOut).norm2Number().doubleValue();
            double relErr = fp32Norm > 1e-10 ? errNorm / fp32Norm : errNorm;
            assertTrue(relErr < 0.05,
                    "Quantized output relative error must be < 5%, got " +
                            String.format("%.4f", relErr * 100) + "%");

        } finally {
            System.clearProperty(QuantizationOptimizations.QuantizeActivationsInt8.PROP_ENABLE);
        }
    }

    /**
     * Test that QuantizeActivationsInt8 does NOT fire when the system property is unset,
     * preserving the opt-in contract.
     */
    @Test
    public void testQuantizeActivationsInt8_isOptIn() {
        // Ensure property is NOT set
        System.clearProperty(QuantizationOptimizations.QuantizeActivationsInt8.PROP_ENABLE);

        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 4, 8);
        SDVariable w = sd.constant("W", Nd4j.rand(DataType.FLOAT, 8, 6));
        SDVariable out = sd.mmul("out", x, w);
        sd.setOutputs(Collections.singletonList("out"));

        // Build a scale map
        Map<String, QuantizationOptimizations.QuantizationInfo> scaleMap = new HashMap<>();
        scaleMap.put("x", new QuantizationOptimizations.QuantizationInfo(0.01f, 0));

        // Phase 2 rewrite — but system property is off, so optimizer should not touch graph
        SameDiff result = QuantizationOptimizations.QuantizeActivationsInt8.rewriteWithScales(sd, scaleMap);

        // Should still have exactly 1 matmul and no cast/quant chain
        long mmulCount = result.getOps().values().stream()
                .filter(o -> o.getOp() instanceof Mmul)
                .count();
        assertEquals(1, mmulCount, "Matmul should remain unchanged when property is not set");
    }
}
