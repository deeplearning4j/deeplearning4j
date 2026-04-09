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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive accuracy tests for OpenVINO DSP backend kernels.
 *
 * <p>OpenVINO provides broader op coverage than oneDNN Graph (~200 vs ~80 ops),
 * including Gather, GatherND, ScatterNDUpdate, Where/Select, Split, Slice,
 * and full comparison/logical op families. It also offers Snippets JIT for
 * element-wise fusion and oneDNN BRGEMM for matmul/conv.
 *
 * <p>Tests cover OpenVINO-specific op patterns:
 * <ul>
 *   <li>Elementwise: binary/unary ops with broadcasting</li>
 *   <li>Matmul: basic and batched matmul</li>
 *   <li>Gather/GatherND: multi-dimensional indexing</li>
 *   <li>Concat/Split: multi-way concatenations and splits</li>
 *   <li>Slice: strided slice operations</li>
 *   <li>Where/Select: conditional selection</li>
 *   <li>Comparison/Logical: full boolean op families</li>
 *   <li>Normalization: softmax</li>
 * </ul>
 *
 * <p>Each test follows the pattern:
 * <ol>
 *   <li>Build a SameDiff graph with the target op(s)</li>
 *   <li>Run with standard execution (baseline)</li>
 *   <li>Run with DSP native auto-compile (OpenVINO backend path)</li>
 *   <li>Assert outputs match within tolerance</li>
 * </ol>
 */
public class TestOpenVinoBackendKernelAccuracy extends BaseNd4jTestWithBackends {

    private static final Logger log = LoggerFactory.getLogger(TestOpenVinoBackendKernelAccuracy.class);
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
    // Elementwise with Broadcasting
    // ═══════════════════════════════════════════════════════════════

    @Test
    @DisplayName("OpenVINO: binary add with broadcasting (scalar)")
    public void testAddBroadcastScalar() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 4);
        SDVariable bias = sd.constant("bias", Nd4j.scalar(1.5f));
        SDVariable out = x.add("out", bias);

        INDArray arr = Nd4j.randn(DataType.FLOAT, 2, 4);
        assertDspMatchesStandard(sd, "out", Map.of("x", arr), "add_broadcast_scalar");
    }

    @Test
    @DisplayName("OpenVINO: binary add with broadcasting (vector)")
    public void testAddBroadcastVector() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
        SDVariable bias = sd.constant("bias", Nd4j.randn(DataType.FLOAT, 4));
        SDVariable out = x.add("out", bias);

        INDArray arr = Nd4j.randn(DataType.FLOAT, 3, 4);
        assertDspMatchesStandard(sd, "out", Map.of("x", arr), "add_broadcast_vector");
    }

    @Test
    @DisplayName("OpenVINO: binary mul with broadcasting (column)")
    public void testMulBroadcastColumn() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
        SDVariable scale = sd.constant("scale", Nd4j.randn(DataType.FLOAT, 3, 1));
        SDVariable out = x.mul("out", scale);

        INDArray arr = Nd4j.randn(DataType.FLOAT, 3, 4);
        assertDspMatchesStandard(sd, "out", Map.of("x", arr), "mul_broadcast_column");
    }

    // ═══════════════════════════════════════════════════════════════
    // Matmul
    // ═══════════════════════════════════════════════════════════════

    @Test
    @DisplayName("OpenVINO: basic matmul")
    public void testMatmul() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 4);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 4, 3));
        SDVariable out = sd.mmul("out", x, w);

        INDArray arr = Nd4j.randn(DataType.FLOAT, 2, 4);
        assertDspMatchesStandard(sd, "out", Map.of("x", arr), "matmul", TOL_LOOSE);
    }

    @Test
    @DisplayName("OpenVINO: batched matmul")
    public void testBatchedMatmul() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 2, 4);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 3, 4, 5));
        SDVariable out = sd.mmul("out", x, w);

        INDArray arr = Nd4j.randn(DataType.FLOAT, 3, 2, 4);
        assertDspMatchesStandard(sd, "out", Map.of("x", arr), "batched_matmul", TOL_LOOSE);
    }

    // ═══════════════════════════════════════════════════════════════
    // Gather
    // ═══════════════════════════════════════════════════════════════

    @Test
    @DisplayName("OpenVINO: gather (1D indices)")
    public void testGather1DIndices() {
        SameDiff sd = SameDiff.create();
        SDVariable table = sd.constant("table", Nd4j.randn(DataType.FLOAT, 10, 4));
        SDVariable indices = sd.placeHolder("indices", DataType.LONG, -1);
        SDVariable out = sd.gather("out", table, indices, 0);

        INDArray indicesArr = Nd4j.createFromArray(new long[]{3, 7, 1, 5});
        assertDspMatchesStandard(sd, "out", Map.of("indices", indicesArr), "gather_1d_indices");
    }

    @Test
    @DisplayName("OpenVINO: gather along axis 1")
    public void testGatherAlongAxis1() {
        SameDiff sd = SameDiff.create();
        SDVariable data = sd.constant("data", Nd4j.randn(DataType.FLOAT, 4, 6, 8));
        SDVariable indices = sd.placeHolder("indices", DataType.LONG, -1);
        SDVariable out = sd.gather("out", data, indices, 1);

        INDArray indicesArr = Nd4j.createFromArray(new long[]{2, 0, 4});
        assertDspMatchesStandard(sd, "out", Map.of("indices", indicesArr), "gather_axis1");
    }

    // ═══════════════════════════════════════════════════════════════
    // Concat / Split
    // ═══════════════════════════════════════════════════════════════

    @Test
    @DisplayName("OpenVINO: concat (axis 0)")
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
    @DisplayName("OpenVINO: concat (axis 1)")
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
    @DisplayName("OpenVINO: concat 3 inputs")
    public void testConcatThreeInputs() {
        SameDiff sd = SameDiff.create();
        SDVariable a = sd.placeHolder("a", DataType.FLOAT, 1, 4);
        SDVariable b = sd.placeHolder("b", DataType.FLOAT, 1, 4);
        SDVariable c = sd.placeHolder("c", DataType.FLOAT, 1, 4);
        SDVariable out = sd.concat("out", 0, a, b, c);

        INDArray arrA = Nd4j.ones(DataType.FLOAT, 1, 4).mul(1.0f);
        INDArray arrB = Nd4j.ones(DataType.FLOAT, 1, 4).mul(2.0f);
        INDArray arrC = Nd4j.ones(DataType.FLOAT, 1, 4).mul(3.0f);
        assertDspMatchesStandard(sd, "out", Map.of("a", arrA, "b", arrB, "c", arrC), "concat_three_inputs");
    }

    // ═══════════════════════════════════════════════════════════════
    // Slice / Strided Slice
    // ═══════════════════════════════════════════════════════════════

    @Test
    @DisplayName("OpenVINO: strided slice with step > 1")
    public void testSliceWithStep() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 16);
        SDVariable out = sd.stridedSlice("out", x,
                new long[]{0, 0}, new long[]{1, 16},
                new long[]{1, 2});  // every other element

        INDArray arr = Nd4j.linspace(1, 16, 16, DataType.FLOAT).reshape(1, 16);
        assertDspMatchesStandard(sd, "out", Map.of("x", arr), "slice_with_step");
    }

    // ═══════════════════════════════════════════════════════════════
    // Where / Select
    // ═══════════════════════════════════════════════════════════════

    @Test
    @DisplayName("OpenVINO: where/select with boolean condition")
    public void testWhereSelect() {
        SameDiff sd = SameDiff.create();
        SDVariable cond = sd.placeHolder("cond", DataType.BOOL, 2, 4);
        SDVariable onTrue = sd.constant("onTrue", Nd4j.ones(DataType.FLOAT, 2, 4).mul(10.0f));
        SDVariable onFalse = sd.constant("onFalse", Nd4j.zeros(DataType.FLOAT, 2, 4));
        SDVariable out = sd.where("out", onTrue, onFalse, cond);

        INDArray condArr = Nd4j.createFromArray(new boolean[][]{
                {true, false, true, false},
                {false, true, false, true}
        });
        assertDspMatchesStandard(sd, "out", Map.of("cond", condArr), "where_select");
    }

    @Test
    @DisplayName("OpenVINO: where with computed condition")
    public void testWhereComputedCondition() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 4);
        SDVariable cond = sd.gt("cond", x, sd.constant("zero", Nd4j.scalar(0.0f)));
        SDVariable out = sd.where("out", x, x.neg(), cond);

        INDArray arr = Nd4j.randn(DataType.FLOAT, 2, 4);
        assertDspMatchesStandard(sd, "out", Map.of("x", arr), "where_computed_condition");
    }

    // ═══════════════════════════════════════════════════════════════
    // Comparison and Logical Ops
    // ═══════════════════════════════════════════════════════════════

    @Test
    @DisplayName("OpenVINO: comparison chain (gt -> lt -> logical_and)")
    public void testComparisonLogicalChain() {
        SameDiff sd = SameDiff.create();
        SDVariable a = sd.placeHolder("a", DataType.FLOAT, 2, 4);
        SDVariable b = sd.placeHolder("b", DataType.FLOAT, 2, 4);
        SDVariable c = sd.placeHolder("c", DataType.FLOAT, 2, 4);

        SDVariable gt = sd.gt("gt", a, b);
        SDVariable lt = sd.lt("lt", b, c);
        SDVariable outBool = sd.booleanAnd("out_bool", gt, lt);
        SDVariable out = sd.castTo("out", outBool, DataType.FLOAT);

        INDArray arrA = Nd4j.create(new float[]{1, 2, 3, 4, 5, 6, 7, 8}, new int[]{2, 4});
        INDArray arrB = Nd4j.create(new float[]{2, 2, 2, 2, 6, 6, 6, 6}, new int[]{2, 4});
        INDArray arrC = Nd4j.create(new float[]{3, 1, 4, 1, 7, 5, 8, 5}, new int[]{2, 4});
        assertDspMatchesStandard(sd, "out", Map.of("a", arrA, "b", arrB, "c", arrC), "comparison_logical_chain");
    }

    @Test
    @DisplayName("OpenVINO: boolean or")
    public void testBooleanOr() {
        SameDiff sd = SameDiff.create();
        SDVariable a = sd.placeHolder("a", DataType.FLOAT, 2, 4);
        SDVariable b = sd.placeHolder("b", DataType.FLOAT, 2, 4);

        SDVariable condA = sd.gt("condA", a, sd.constant("z", Nd4j.scalar(0.0f)));
        SDVariable condB = sd.gt("condB", b, sd.constant("z2", Nd4j.scalar(0.0f)));
        SDVariable outBool = sd.booleanOr("out_bool", condA, condB);
        SDVariable out = sd.castTo("out", outBool, DataType.FLOAT);

        INDArray arrA = Nd4j.randn(DataType.FLOAT, 2, 4);
        INDArray arrB = Nd4j.randn(DataType.FLOAT, 2, 4);
        assertDspMatchesStandard(sd, "out", Map.of("a", arrA, "b", arrB), "boolean_or");
    }

    @Test
    @DisplayName("OpenVINO: boolean not")
    public void testBooleanNot() {
        SameDiff sd = SameDiff.create();
        SDVariable a = sd.placeHolder("a", DataType.FLOAT, 2, 4);
        SDVariable cond = sd.eq("cond", a, sd.constant("z", Nd4j.scalar(0.0f)));
        SDVariable outBool = sd.booleanNot("out_bool", cond);
        SDVariable out = sd.castTo("out", outBool, DataType.FLOAT);

        INDArray arrA = Nd4j.create(new float[]{0, 1, 0, 2, 0, 0, 3, 0}, new int[]{2, 4});
        assertDspMatchesStandard(sd, "out", Map.of("a", arrA), "logical_not");
    }

    // ═══════════════════════════════════════════════════════════════
    // Normalization
    // ═══════════════════════════════════════════════════════════════

    @Test
    @DisplayName("OpenVINO: softmax")
    public void testSoftmax() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 5);
        SDVariable out = sd.nn.softmax("out", x, -1);

        INDArray arr = Nd4j.randn(DataType.FLOAT, 2, 5);
        assertDspMatchesStandard(sd, "out", Map.of("x", arr), "softmax", TOL_LOOSE);
    }

    // ═══════════════════════════════════════════════════════════════
    // Fused Kernel Chains
    // ═══════════════════════════════════════════════════════════════

    @Test
    @DisplayName("OpenVINO: MLP pattern (matmul -> add -> relu -> matmul)")
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
    @DisplayName("OpenVINO: attention pattern (Q*K^T -> scale -> softmax -> *V)")
    public void testAttentionPattern() {
        int seqLen = 4;
        int headDim = 8;

        SameDiff sd = SameDiff.create();
        SDVariable q = sd.placeHolder("q", DataType.FLOAT, 1, seqLen, headDim);
        SDVariable k = sd.placeHolder("k", DataType.FLOAT, 1, seqLen, headDim);
        SDVariable v = sd.placeHolder("v", DataType.FLOAT, 1, seqLen, headDim);

        SDVariable kT = sd.permute("kT", k, 0, 2, 1);
        SDVariable scores = sd.mmul("scores", q, kT);
        SDVariable scaled = scores.div("scaled", sd.constant("scale",
                Nd4j.scalar((float) Math.sqrt(headDim))));
        SDVariable weights = sd.nn.softmax("weights", scaled, -1);
        SDVariable out = sd.mmul("out", weights, v);

        INDArray qArr = Nd4j.randn(DataType.FLOAT, 1, seqLen, headDim);
        INDArray kArr = Nd4j.randn(DataType.FLOAT, 1, seqLen, headDim);
        INDArray vArr = Nd4j.randn(DataType.FLOAT, 1, seqLen, headDim);
        assertDspMatchesStandard(sd, "out", Map.of("q", qArr, "k", kArr, "v", vArr),
                "attention_pattern", TOL_LOOSE);
    }

    @Test
    @DisplayName("OpenVINO: embedding lookup + matmul pattern")
    public void testEmbeddingMatmulPattern() {
        SameDiff sd = SameDiff.create();
        SDVariable embedTable = sd.constant("embed", Nd4j.randn(DataType.FLOAT, 100, 16));
        SDVariable indices = sd.placeHolder("indices", DataType.LONG, 1, 8);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 16, 4));

        SDVariable embedded = sd.gather("embedded", embedTable, indices, 0);
        SDVariable out = sd.mmul("out", embedded, w);

        INDArray indicesArr = Nd4j.createFromArray(new long[][]{{3, 7, 15, 42, 99, 1, 50, 25}});
        assertDspMatchesStandard(sd, "out", Map.of("indices", indicesArr), "embedding_matmul", TOL_LOOSE);
    }

    @Test
    @DisplayName("OpenVINO: concat with dynamic shapes (KV cache pattern)")
    public void testConcatDynamicKvCache() {
        SameDiff sd = SameDiff.create();

        // Simulate KV cache concat: past_kv [1,3,seq,64] concat new_kv [1,3,1,64]
        SDVariable pastKv = sd.placeHolder("past_kv", DataType.FLOAT, -1, 3, -1, 64);
        SDVariable newKv = sd.placeHolder("new_kv", DataType.FLOAT, 1, 3, 1, 64);
        SDVariable fullKv = sd.concat("full_kv", 2, pastKv, newKv);
        SDVariable out = sd.sum("out", fullKv, 2, 3);  // sum to scalar per batch/head

        // Step 0: empty KV cache
        INDArray emptyPast = Nd4j.zeros(DataType.FLOAT, 1, 3, 0, 64);
        INDArray newToken = Nd4j.ones(DataType.FLOAT, 1, 3, 1, 64);

        Map<String, INDArray> inputs0 = Map.of("past_kv", emptyPast, "new_kv", newToken);
        assertDspMatchesStandard(sd, "out", inputs0, "kv_concat_step0");

        // Step 1: filled KV cache
        INDArray filledPast = Nd4j.randn(DataType.FLOAT, 1, 3, 5, 64);
        Map<String, INDArray> inputs1 = Map.of("past_kv", filledPast, "new_kv", newToken);
        assertDspMatchesStandard(sd, "out", inputs1, "kv_concat_step1");
    }

    @Test
    @DisplayName("OpenVINO: repeated execution correctness (no stale data)")
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

    @Test
    @DisplayName("OpenVINO: shape change across steps (dynamic reshape)")
    public void testShapeChangeAcrossSteps() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, -1);
        SDVariable reshaped = sd.reshape("reshaped", x, -1);  // flatten
        SDVariable out = sd.sum("out", reshaped);

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        // Step 1: [2, 3]
        INDArray arr1 = Nd4j.ones(DataType.FLOAT, 2, 3);
        Map<String, INDArray> result1 = sd.outputDirect(Map.of("x", arr1), "out");
        assertEquals(6.0f, result1.get("out").sumNumber().floatValue(), 1e-5);

        // Step 2: [3, 4]
        INDArray arr2 = Nd4j.ones(DataType.FLOAT, 3, 4);
        Map<String, INDArray> result2 = sd.outputDirect(Map.of("x", arr2), "out");
        assertEquals(12.0f, result2.get("out").sumNumber().floatValue(), 1e-5);

        // Step 3: [1, 8]
        INDArray arr3 = Nd4j.ones(DataType.FLOAT, 1, 8);
        Map<String, INDArray> result3 = sd.outputDirect(Map.of("x", arr3), "out");
        assertEquals(8.0f, result3.get("out").sumNumber().floatValue(), 1e-5);

        sd.close();
    }
}
