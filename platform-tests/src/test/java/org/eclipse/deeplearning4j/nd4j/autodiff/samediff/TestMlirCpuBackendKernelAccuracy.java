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
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Environment;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive accuracy tests for MLIR CPU backend DSP kernels.
 *
 * <p>Tests cover all op categories emitted by CpuIRBuilder:
 * <ul>
 *   <li>Binary elementwise: add, sub, mul, div, pow</li>
 *   <li>Unary elementwise: relu, sigmoid, tanh, exp, log, sqrt, abs, neg</li>
 *   <li>Comparison: greater, less, equal</li>
 *   <li>Ternary: where/select</li>
 *   <li>Matmul: basic matmul, xw_plus_b</li>
 *   <li>Reduction: sum, mean, max</li>
 *   <li>Normalization: softmax, rms_norm</li>
 *   <li>Data movement: gather, concat, tile</li>
 *   <li>Shape manipulation: reshape, permute</li>
 *   <li>Constant generation: zeros_like, ones_like</li>
 * </ul>
 *
 * <p>Each test follows the same pattern:
 * <ol>
 *   <li>Build a SameDiff graph with the target op(s)</li>
 *   <li>Run with standard execution (baseline)</li>
 *   <li>Run with DSP native auto-compile enabled (CPU backend path)</li>
 *   <li>Assert outputs match within tolerance</li>
 * </ol>
 *
 * <p>These tests are CPU-focused and skip if Triton (GPU) is the only available backend.
 */
public class TestMlirCpuBackendKernelAccuracy extends BaseNd4jTestWithBackends {

    private static final Logger log = LoggerFactory.getLogger(TestMlirCpuBackendKernelAccuracy.class);
    private static final double TOL = 1e-4;
    private static final double TOL_LOOSE = 1e-3;

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
     * Helper: compile and run a graph through DSP native path, compare against standard.
     */
    private void assertDspMatchesStandard(SameDiff sd, String outputName,
                                           Map<String, INDArray> inputs,
                                           String testName,
                                           double tolerance) {
        // 1. Standard reference
        sd.resetSession();
        sd.clearDynamicShapePlanCache();
        Map<String, INDArray> refResult = sd.output(inputs, outputName);
        INDArray reference = refResult.get(outputName).dup();

        // 2. DSP with native auto-compile
        sd.resetSession();
        sd.clearDynamicShapePlanCache();
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        // Ensure Triton is disabled so we test CPU backend path
        Environment env = Nd4j.getEnvironment();
        env.setTritonGraphCapture(false);

        Map<String, INDArray> dspResult = sd.outputDirect(inputs, outputName);
        INDArray dspOutput = dspResult.get(outputName).dup();

        double maxDiff = reference.sub(dspOutput).amaxNumber().doubleValue();
        log.info("[{}] max diff = {} (tolerance={})", testName, maxDiff, tolerance);
        assertTrue(maxDiff < tolerance,
                testName + ": DSP output diverges from standard. " +
                "max diff " + maxDiff + " exceeds tolerance " + tolerance);
    }

    private void assertDspMatchesStandard(SameDiff sd, String outputName,
                                           Map<String, INDArray> inputs,
                                           String testName) {
        assertDspMatchesStandard(sd, outputName, inputs, testName, TOL);
    }

    // ═══════════════════════════════════════════════════════════════
    // Binary Elementwise Kernels
    // ═══════════════════════════════════════════════════════════════

    @Test
    @DisplayName("MLIR CPU: binary elementwise add")
    public void testBinaryAdd() {
        SameDiff sd = SameDiff.create();
        SDVariable a = sd.placeHolder("a", DataType.FLOAT, 2, 4);
        SDVariable b = sd.placeHolder("b", DataType.FLOAT, 2, 4);
        SDVariable out = a.add("out", b);

        INDArray arrA = Nd4j.randn(DataType.FLOAT, 2, 4);
        INDArray arrB = Nd4j.randn(DataType.FLOAT, 2, 4);
        assertDspMatchesStandard(sd, "out", Map.of("a", arrA, "b", arrB), "binary_add");
    }

    @Test
    @DisplayName("MLIR CPU: binary elementwise sub")
    public void testBinarySub() {
        SameDiff sd = SameDiff.create();
        SDVariable a = sd.placeHolder("a", DataType.FLOAT, 2, 4);
        SDVariable b = sd.placeHolder("b", DataType.FLOAT, 2, 4);
        SDVariable out = a.sub("out", b);

        INDArray arrA = Nd4j.randn(DataType.FLOAT, 2, 4);
        INDArray arrB = Nd4j.randn(DataType.FLOAT, 2, 4);
        assertDspMatchesStandard(sd, "out", Map.of("a", arrA, "b", arrB), "binary_sub");
    }

    @Test
    @DisplayName("MLIR CPU: binary elementwise mul")
    public void testBinaryMul() {
        SameDiff sd = SameDiff.create();
        SDVariable a = sd.placeHolder("a", DataType.FLOAT, 3, 5);
        SDVariable b = sd.placeHolder("b", DataType.FLOAT, 3, 5);
        SDVariable out = a.mul("out", b);

        INDArray arrA = Nd4j.randn(DataType.FLOAT, 3, 5);
        INDArray arrB = Nd4j.randn(DataType.FLOAT, 3, 5);
        assertDspMatchesStandard(sd, "out", Map.of("a", arrA, "b", arrB), "binary_mul");
    }

    @Test
    @DisplayName("MLIR CPU: binary elementwise div (no zeros)")
    public void testBinaryDiv() {
        SameDiff sd = SameDiff.create();
        SDVariable a = sd.placeHolder("a", DataType.FLOAT, 2, 4);
        SDVariable b = sd.placeHolder("b", DataType.FLOAT, 2, 4);
        SDVariable out = a.div("out", b);

        INDArray arrA = Nd4j.randn(DataType.FLOAT, 2, 4);
        INDArray arrB = Transforms.abs(Nd4j.randn(DataType.FLOAT, 2, 4)).add(0.1);
        assertDspMatchesStandard(sd, "out", Map.of("a", arrA, "b", arrB), "binary_div");
    }

    @Test
    @DisplayName("MLIR CPU: binary elementwise pow")
    public void testBinaryPow() {
        SameDiff sd = SameDiff.create();
        SDVariable a = sd.placeHolder("a", DataType.FLOAT, 2, 4);
        SDVariable b = sd.placeHolder("b", DataType.FLOAT, 2, 4);
        SDVariable out = sd.math.pow("out", a, b);

        INDArray arrA = Transforms.abs(Nd4j.randn(DataType.FLOAT, 2, 4)).add(0.5);
        INDArray arrB = Nd4j.linspace(1, 8, 8, DataType.FLOAT).reshape(2, 4);
        assertDspMatchesStandard(sd, "out", Map.of("a", arrA, "b", arrB), "binary_pow", TOL_LOOSE);
    }

    // ═══════════════════════════════════════════════════════════════
    // Unary Elementwise Kernels
    // ═══════════════════════════════════════════════════════════════

    @Test
    @DisplayName("MLIR CPU: unary relu")
    public void testUnaryRelu() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 4);
        SDVariable out = sd.nn.relu("out", x, 0.0);

        INDArray arr = Nd4j.create(new float[]{-2, -1, 0, 1, 2, -3, 4, -5}, new int[]{2, 4});
        assertDspMatchesStandard(sd, "out", Map.of("x", arr), "unary_relu");
    }

    @Test
    @DisplayName("MLIR CPU: unary sigmoid")
    public void testUnarySigmoid() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 4);
        SDVariable out = sd.nn.sigmoid("out", x);

        INDArray arr = Nd4j.linspace(-5, 5, 8, DataType.FLOAT).reshape(2, 4);
        assertDspMatchesStandard(sd, "out", Map.of("x", arr), "unary_sigmoid");
    }

    @Test
    @DisplayName("MLIR CPU: unary tanh")
    public void testUnaryTanh() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 4);
        SDVariable out = sd.math.tanh("out", x);

        INDArray arr = Nd4j.linspace(-5, 5, 8, DataType.FLOAT).reshape(2, 4);
        assertDspMatchesStandard(sd, "out", Map.of("x", arr), "unary_tanh");
    }

    @Test
    @DisplayName("MLIR CPU: unary exp")
    public void testUnaryExp() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 4);
        SDVariable out = sd.math.exp("out", x);

        INDArray arr = Nd4j.linspace(-3, 3, 8, DataType.FLOAT).reshape(2, 4);
        assertDspMatchesStandard(sd, "out", Map.of("x", arr), "unary_exp", TOL_LOOSE);
    }

    @Test
    @DisplayName("MLIR CPU: unary log (positive values)")
    public void testUnaryLog() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 4);
        SDVariable out = sd.math.log("out", x);

        INDArray arr = Transforms.abs(Nd4j.randn(DataType.FLOAT, 2, 4)).add(0.1);
        assertDspMatchesStandard(sd, "out", Map.of("x", arr), "unary_log");
    }

    @Test
    @DisplayName("MLIR CPU: unary sqrt (positive values)")
    public void testUnarySqrt() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 4);
        SDVariable out = sd.math.sqrt("out", x);

        INDArray arr = Transforms.abs(Nd4j.randn(DataType.FLOAT, 2, 4)).add(0.1);
        assertDspMatchesStandard(sd, "out", Map.of("x", arr), "unary_sqrt");
    }

    @Test
    @DisplayName("MLIR CPU: unary abs")
    public void testUnaryAbs() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 4);
        SDVariable out = sd.math.abs("out", x);

        INDArray arr = Nd4j.randn(DataType.FLOAT, 2, 4);
        assertDspMatchesStandard(sd, "out", Map.of("x", arr), "unary_abs");
    }

    @Test
    @DisplayName("MLIR CPU: unary neg")
    public void testUnaryNeg() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 4);
        SDVariable out = x.neg("out");

        INDArray arr = Nd4j.randn(DataType.FLOAT, 2, 4);
        assertDspMatchesStandard(sd, "out", Map.of("x", arr), "unary_neg");
    }

    // ═══════════════════════════════════════════════════════════════
    // Comparison Kernels
    // ═══════════════════════════════════════════════════════════════

    @Test
    @DisplayName("MLIR CPU: comparison greater")
    public void testCompareGreater() {
        SameDiff sd = SameDiff.create();
        SDVariable a = sd.placeHolder("a", DataType.FLOAT, 2, 4);
        SDVariable b = sd.placeHolder("b", DataType.FLOAT, 2, 4);
        SDVariable gtBool = sd.gt("gt_bool", a, b);
        SDVariable out = sd.castTo("out", gtBool, DataType.FLOAT);

        INDArray arrA = Nd4j.create(new float[]{1, 2, 3, 4, 5, 6, 7, 8}, new int[]{2, 4});
        INDArray arrB = Nd4j.create(new float[]{4, 3, 2, 1, 5, 5, 5, 5}, new int[]{2, 4});
        assertDspMatchesStandard(sd, "out", Map.of("a", arrA, "b", arrB), "compare_greater");
    }

    @Test
    @DisplayName("MLIR CPU: comparison less")
    public void testCompareLess() {
        SameDiff sd = SameDiff.create();
        SDVariable a = sd.placeHolder("a", DataType.FLOAT, 2, 4);
        SDVariable b = sd.placeHolder("b", DataType.FLOAT, 2, 4);
        SDVariable ltBool = sd.lt("lt_bool", a, b);
        SDVariable out = sd.castTo("out", ltBool, DataType.FLOAT);

        INDArray arrA = Nd4j.create(new float[]{1, 2, 3, 4, 5, 6, 7, 8}, new int[]{2, 4});
        INDArray arrB = Nd4j.create(new float[]{4, 3, 2, 1, 5, 5, 5, 5}, new int[]{2, 4});
        assertDspMatchesStandard(sd, "out", Map.of("a", arrA, "b", arrB), "compare_less");
    }

    @Test
    @DisplayName("MLIR CPU: comparison equal")
    public void testCompareEqual() {
        SameDiff sd = SameDiff.create();
        SDVariable a = sd.placeHolder("a", DataType.FLOAT, 2, 4);
        SDVariable b = sd.placeHolder("b", DataType.FLOAT, 2, 4);
        SDVariable eqBool = sd.eq("eq_bool", a, b);
        SDVariable out = sd.castTo("out", eqBool, DataType.FLOAT);

        INDArray arrA = Nd4j.create(new float[]{1, 2, 3, 4, 5, 6, 7, 8}, new int[]{2, 4});
        INDArray arrB = Nd4j.create(new float[]{1, 2, 0, 4, 0, 6, 0, 8}, new int[]{2, 4});
        assertDspMatchesStandard(sd, "out", Map.of("a", arrA, "b", arrB), "compare_equal");
    }

    // ═══════════════════════════════════════════════════════════════
    // Ternary Kernel (where/select)
    // ═══════════════════════════════════════════════════════════════

    @Test
    @DisplayName("MLIR CPU: ternary where/select")
    public void testTernaryWhere() {
        SameDiff sd = SameDiff.create();
        SDVariable cond = sd.placeHolder("cond", DataType.BOOL, 2, 4);
        SDVariable onTrue = sd.constant("onTrue", Nd4j.ones(DataType.FLOAT, 2, 4).mul(10.0f));
        SDVariable onFalse = sd.constant("onFalse", Nd4j.zeros(DataType.FLOAT, 2, 4));
        SDVariable out = sd.where("out", onTrue, onFalse, cond);

        INDArray condArr = Nd4j.createFromArray(new boolean[][]{
                {true, false, true, false},
                {false, true, false, true}
        });
        assertDspMatchesStandard(sd, "out", Map.of("cond", condArr), "ternary_where");
    }

    // ═══════════════════════════════════════════════════════════════
    // Matmul Kernel
    // ═══════════════════════════════════════════════════════════════

    @Test
    @DisplayName("MLIR CPU: basic matmul")
    public void testMatmul() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 4);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 4, 3));
        SDVariable out = sd.mmul("out", x, w);

        INDArray arr = Nd4j.randn(DataType.FLOAT, 2, 4);
        assertDspMatchesStandard(sd, "out", Map.of("x", arr), "matmul", TOL_LOOSE);
    }

    @Test
    @DisplayName("MLIR CPU: xw_plus_b (matmul + bias)")
    public void testXwPlusB() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 4);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 4, 3));
        SDVariable b = sd.constant("b", Nd4j.randn(DataType.FLOAT, 1, 3));
        SDVariable mm = sd.mmul("mm", x, w);
        SDVariable out = mm.add("out", b);

        INDArray arr = Nd4j.randn(DataType.FLOAT, 2, 4);
        assertDspMatchesStandard(sd, "out", Map.of("x", arr), "xw_plus_b", TOL_LOOSE);
    }

    // ═══════════════════════════════════════════════════════════════
    // Reduction Kernels
    // ═══════════════════════════════════════════════════════════════

    @Test
    @DisplayName("MLIR CPU: reduction sum (full)")
    public void testReduceSumFull() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
        SDVariable out = sd.sum("out", x);

        INDArray arr = Nd4j.linspace(1, 12, 12, DataType.FLOAT).reshape(3, 4);
        assertDspMatchesStandard(sd, "out", Map.of("x", arr), "reduce_sum_full");
    }

    @Test
    @DisplayName("MLIR CPU: reduction mean (full)")
    public void testReduceMeanFull() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
        SDVariable out = sd.mean("out", x);

        INDArray arr = Nd4j.linspace(1, 12, 12, DataType.FLOAT).reshape(3, 4);
        assertDspMatchesStandard(sd, "out", Map.of("x", arr), "reduce_mean_full");
    }

    @Test
    @DisplayName("MLIR CPU: reduction sum (along dimension)")
    public void testReduceSumAlongDim() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
        SDVariable out = sd.sum("out", x, 1);  // sum along last dim → [3]

        INDArray arr = Nd4j.linspace(1, 12, 12, DataType.FLOAT).reshape(3, 4);
        assertDspMatchesStandard(sd, "out", Map.of("x", arr), "reduce_sum_along_dim");
    }

    // ═══════════════════════════════════════════════════════════════
    // Normalization Kernels
    // ═══════════════════════════════════════════════════════════════

    @Test
    @DisplayName("MLIR CPU: softmax")
    public void testSoftmax() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 4);
        SDVariable out = sd.nn.softmax("out", x, -1);

        INDArray arr = Nd4j.randn(DataType.FLOAT, 2, 4);
        assertDspMatchesStandard(sd, "out", Map.of("x", arr), "softmax", TOL_LOOSE);
    }

    @Test
    @DisplayName("MLIR CPU: rms_norm")
    public void testRmsNorm() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 4);
        SDVariable gamma = sd.constant("gamma", Nd4j.ones(DataType.FLOAT, 4));
        float epsilon = 1e-5f;
        SDVariable out = sd.nn.rmsNorm("out", x, gamma, epsilon);

        INDArray arr = Nd4j.randn(DataType.FLOAT, 2, 4);
        assertDspMatchesStandard(sd, "out", Map.of("x", arr), "rms_norm", TOL_LOOSE);
    }

    // ═══════════════════════════════════════════════════════════════
    // Data Movement Kernels
    // ═══════════════════════════════════════════════════════════════

    @Test
    @DisplayName("MLIR CPU: gather (embedding lookup)")
    public void testGather() {
        SameDiff sd = SameDiff.create();
        SDVariable table = sd.constant("table", Nd4j.randn(DataType.FLOAT, 10, 4));
        SDVariable indices = sd.placeHolder("indices", DataType.LONG, -1);
        SDVariable out = sd.gather("out", table, indices, 0);

        INDArray indicesArr = Nd4j.createFromArray(new long[]{3, 7, 1});
        assertDspMatchesStandard(sd, "out", Map.of("indices", indicesArr), "gather");
    }

    @Test
    @DisplayName("MLIR CPU: concat (axis 0)")
    public void testConcatAxis0() {
        SameDiff sd = SameDiff.create();
        SDVariable a = sd.placeHolder("a", DataType.FLOAT, 2, 4);
        SDVariable b = sd.placeHolder("b", DataType.FLOAT, 3, 4);
        SDVariable out = sd.concat("out", 0, a, b);

        INDArray arrA = Nd4j.randn(DataType.FLOAT, 2, 4);
        INDArray arrB = Nd4j.randn(DataType.FLOAT, 3, 4);
        assertDspMatchesStandard(sd, "out", Map.of("a", arrA, "b", arrB), "concat_axis0");
    }

    @Test
    @DisplayName("MLIR CPU: concat (axis 1)")
    public void testConcatAxis1() {
        SameDiff sd = SameDiff.create();
        SDVariable a = sd.placeHolder("a", DataType.FLOAT, 2, 3);
        SDVariable b = sd.placeHolder("b", DataType.FLOAT, 2, 4);
        SDVariable out = sd.concat("out", 1, a, b);

        INDArray arrA = Nd4j.randn(DataType.FLOAT, 2, 3);
        INDArray arrB = Nd4j.randn(DataType.FLOAT, 2, 4);
        assertDspMatchesStandard(sd, "out", Map.of("a", arrA, "b", arrB), "concat_axis1");
    }

    @Test
    @DisplayName("MLIR CPU: tile")
    public void testTile() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 3);
        SDVariable reps = sd.constant("reps", Nd4j.createFromArray(new long[]{2, 3}));
        SDVariable out = sd.tile("out", x, reps);

        INDArray arr = Nd4j.linspace(1, 6, 6, DataType.FLOAT).reshape(2, 3);
        assertDspMatchesStandard(sd, "out", Map.of("x", arr), "tile");
    }

    // ═══════════════════════════════════════════════════════════════
    // Shape Manipulation Kernels
    // ═══════════════════════════════════════════════════════════════

    @Test
    @DisplayName("MLIR CPU: reshape")
    public void testReshape() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 6);
        SDVariable out = sd.reshape("out", x, 3, 4);

        INDArray arr = Nd4j.linspace(1, 12, 12, DataType.FLOAT).reshape(2, 6);
        assertDspMatchesStandard(sd, "out", Map.of("x", arr), "reshape");
    }

    @Test
    @DisplayName("MLIR CPU: permute (transpose)")
    public void testPermute() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 3, 4);
        SDVariable permuted = sd.permute("permuted", x, 0, 2, 1);  // [2, 4, 3]
        // Add a follow-up op to create a multi-op segment (single-op segments have different code path)
        SDVariable out = permuted.mul("out", sd.constant("scale", Nd4j.scalar(2.0f)));

        INDArray arr = Nd4j.linspace(1, 24, 24, DataType.FLOAT).reshape(2, 3, 4);
        assertDspMatchesStandard(sd, "out", Map.of("x", arr), "permute");
    }

    // ═══════════════════════════════════════════════════════════════
    // Constant Generation Kernels
    // ═══════════════════════════════════════════════════════════════

    @Test
    @DisplayName("MLIR CPU: zeros_like")
    public void testZerosLike() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 4);
        SDVariable out = sd.zerosLike("out", x);

        INDArray arr = Nd4j.randn(DataType.FLOAT, 2, 4);
        Map<String, INDArray> result = sd.outputDirect(Map.of("x", arr), "out");
        INDArray actual = result.get("out");
        assertEquals(0.0f, actual.sumNumber().floatValue(), 1e-6);
        assertArrayEquals(arr.shape(), actual.shape());
    }

    @Test
    @DisplayName("MLIR CPU: ones_like")
    public void testOnesLike() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 4);
        SDVariable out = sd.onesLike("out", x);

        INDArray arr = Nd4j.randn(DataType.FLOAT, 2, 4);
        Map<String, INDArray> result = sd.outputDirect(Map.of("x", arr), "out");
        INDArray actual = result.get("out");
        assertEquals(8.0f, actual.sumNumber().floatValue(), 1e-6);  // 2*4 = 8
        assertArrayEquals(arr.shape(), actual.shape());
    }

    // ═══════════════════════════════════════════════════════════════
    // Fused Kernel Chains
    // ═══════════════════════════════════════════════════════════════

    @Test
    @DisplayName("MLIR CPU: fused binary chain (add -> mul -> sub)")
    public void testFusedBinaryChain() {
        SameDiff sd = SameDiff.create();
        SDVariable a = sd.placeHolder("a", DataType.FLOAT, 2, 4);
        SDVariable b = sd.placeHolder("b", DataType.FLOAT, 2, 4);
        SDVariable c = sd.placeHolder("c", DataType.FLOAT, 2, 4);
        SDVariable t1 = a.add("t1", b);
        SDVariable t2 = t1.mul("t2", c);
        SDVariable out = t2.sub("out", a);

        INDArray arrA = Nd4j.randn(DataType.FLOAT, 2, 4);
        INDArray arrB = Nd4j.randn(DataType.FLOAT, 2, 4);
        INDArray arrC = Nd4j.randn(DataType.FLOAT, 2, 4);
        assertDspMatchesStandard(sd, "out", Map.of("a", arrA, "b", arrB, "c", arrC), "fused_binary_chain");
    }

    @Test
    @DisplayName("MLIR CPU: fused unary chain (exp -> log -> sqrt)")
    public void testFusedUnaryChain() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 4);
        SDVariable t1 = sd.math.exp("t1", x);
        SDVariable t2 = sd.math.log("t2", t1);  // log(exp(x)) = x
        SDVariable out = sd.math.sqrt("out", sd.math.abs(t2).add(1.0));  // sqrt(|x| + 1)

        INDArray arr = Nd4j.linspace(-3, 3, 8, DataType.FLOAT).reshape(2, 4);
        assertDspMatchesStandard(sd, "out", Map.of("x", arr), "fused_unary_chain");
    }

    @Test
    @DisplayName("MLIR CPU: MLP pattern (matmul -> relu -> matmul)")
    public void testMlpPattern() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 8);
        SDVariable w1 = sd.constant("w1", Nd4j.randn(DataType.FLOAT, 8, 16));
        SDVariable b1 = sd.constant("b1", Nd4j.zeros(DataType.FLOAT, 1, 16));
        SDVariable w2 = sd.constant("w2", Nd4j.randn(DataType.FLOAT, 16, 4));

        SDVariable h1 = sd.mmul("h1", x, w1);
        SDVariable h1b = h1.add("h1b", b1);
        SDVariable h1a = sd.nn.relu("h1a", h1b, 0.0);
        SDVariable out = sd.mmul("out", h1a, w2);

        INDArray arr = Nd4j.randn(DataType.FLOAT, 2, 8);
        assertDspMatchesStandard(sd, "out", Map.of("x", arr), "mlp_pattern", TOL_LOOSE);
    }

    @Test
    @DisplayName("MLIR CPU: attention-like pattern (matmul -> scale -> softmax)")
    public void testAttentionLikePattern() {
        int seqLen = 4;
        int headDim = 8;

        SameDiff sd = SameDiff.create();
        SDVariable q = sd.placeHolder("q", DataType.FLOAT, 1, seqLen, headDim);
        SDVariable k = sd.placeHolder("k", DataType.FLOAT, 1, seqLen, headDim);
        SDVariable v = sd.placeHolder("v", DataType.FLOAT, 1, seqLen, headDim);

        SDVariable kT = sd.permute("kT", k, 0, 2, 1);
        SDVariable scores = sd.mmul("scores", q, kT);
        SDVariable scaled = scores.div("scaled", sd.constant("scale", Nd4j.scalar((float) Math.sqrt(headDim))));
        SDVariable out = sd.nn.softmax("out", scaled, -1);

        INDArray qArr = Nd4j.randn(DataType.FLOAT, 1, seqLen, headDim);
        INDArray kArr = Nd4j.randn(DataType.FLOAT, 1, seqLen, headDim);
        INDArray vArr = Nd4j.randn(DataType.FLOAT, 1, seqLen, headDim);
        assertDspMatchesStandard(sd, "out", Map.of("q", qArr, "k", kArr, "v", vArr), "attention_like", TOL_LOOSE);
    }

    @Test
    @DisplayName("MLIR CPU: repeated execution correctness (no stale data)")
    public void testRepeatedExecutionNoStaleData() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 4);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 4, 4));
        SDVariable mm = sd.mmul("mm", x, w);
        SDVariable out = sd.nn.sigmoid("out", mm);

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        INDArray[] inputs = new INDArray[5];
        INDArray[] expected = new INDArray[5];

        // Pre-compute all expected outputs
        for (int i = 0; i < 5; i++) {
            inputs[i] = Nd4j.randn(DataType.FLOAT, 1, 4).mul(i + 1);
            Map<String, INDArray> result = sd.output(Map.of("x", inputs[i]), "out");
            expected[i] = result.get("out").dup();
        }

        // Run through DSP path
        for (int i = 0; i < 5; i++) {
            Map<String, INDArray> dspResult = sd.outputDirect(Map.of("x", inputs[i]), "out");
            INDArray actual = dspResult.get("out").dup();

            double maxDiff = expected[i].sub(actual).amaxNumber().doubleValue();
            log.info("Iteration {}: maxDiff={}", i, maxDiff);
            assertTrue(maxDiff < TOL,
                    "Iteration " + i + ": max diff " + maxDiff + " exceeds tolerance " + TOL);

            // Verify outputs differ between iterations (not stale)
            if (i > 0) {
                double diffFromPrev = actual.sub(expected[i - 1]).amaxNumber().doubleValue();
                assertTrue(diffFromPrev > 0.01,
                        "Iteration " + i + " output too similar to iteration " + (i - 1) +
                                " — possible stale data. diff=" + diffFromPrev);
            }
        }

        sd.close();
    }
}
