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
package org.eclipse.deeplearning4j.nd4j.autodiff.optimization;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.internal.Variable;
import org.nd4j.autodiff.samediff.optimize.GraphOptimizer;
import org.nd4j.autodiff.samediff.optimize.OptimizerSet;
import org.nd4j.autodiff.samediff.optimize.optimizations.IdentityFunctionOptimizations;
import org.nd4j.autodiff.samediff.optimize.optimizations.UnusedFunctionOptimizations;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the graph optimizer input merge fix.
 *
 * Validates that:
 * 1. replaceOpInputsWith MERGES inputsForOp instead of REPLACING
 * 2. Identity removal doesn't corrupt reshape ops that share inputs
 * 3. Graph validation catches missing variables after optimization
 * 4. Full optimization pipeline preserves execution correctness
 */
@Tag(TagNames.DL4J_OLD_API)
public class TestGraphOptimizerInputMerge extends BaseNd4jTestWithBackends {

    @Override
    public char ordering() {
        return 'c';
    }

    @Override
    public long getTimeoutMilliseconds() {
        return 1_000_000_000L;
    }

    /**
     * Helper: find the op that produces a given output variable name.
     */
    private static SameDiffOp findOpProducingVariable(SameDiff sd, String varName) {
        Variable v = sd.getVariables().get(varName);
        if (v == null || v.getOutputOfOp() == null) return null;
        return sd.getOps().get(v.getOutputOfOp());
    }

    /**
     * Helper: validate that all ops in the graph have valid input references.
     */
    private static void assertAllOpsHaveValidInputs(SameDiff sd) {
        for (Map.Entry<String, SameDiffOp> entry : sd.getOps().entrySet()) {
            SameDiffOp op = entry.getValue();
            List<String> inputs = op.getInputsToOp();
            if (inputs != null) {
                for (String input : inputs) {
                    assertTrue(sd.getVariables().containsKey(input),
                            "Op '" + entry.getKey() + "' references missing variable '" + input + "'");
                }
            }
        }
    }

    /**
     * Core regression test for the VLM reshape corruption bug.
     *
     * Pattern: a variable X feeds through identity(X) -> Y, and Y is used as
     * the data input to a reshape op. Another constant is the shape input.
     * When identity is removed, Y is replaced with X. The old code REPLACED
     * X's inputsForOp with Y's, losing X's existing consumers.
     *
     * After the fix, X's consumers are MERGED with Y's consumers.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testIdentityRemovalPreservesReshapeInputs(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();

        // data: a placeholder that feeds into both an identity and another op directly
        SDVariable data = sd.placeHolder("data", DataType.FLOAT, 2, 6);

        // shapeConst: constant shape for reshape [2, 3, 2]
        SDVariable shapeConst = sd.constant("shapeConst", Nd4j.createFromArray(new long[]{2, 3, 2}));

        // identity wrapping data — optimizer should remove this
        SDVariable identData = sd.identity("identData", data);

        // reshape uses identData (which will be replaced by data) + shapeConst
        SDVariable reshaped = sd.reshape("reshaped", identData, shapeConst);

        // data is also used directly in another op (this is the key scenario —
        // data already has consumers before the identity removal merges more)
        SDVariable sum = sd.sum("sum", data);

        // final output combines both paths
        SDVariable reshapeSum = sd.sum("reshapeSum", reshaped);
        SDVariable out = reshapeSum.add("out", sum);

        // Run with identity removal + unused function removal
        List<OptimizerSet> opts = Arrays.asList(
                new IdentityFunctionOptimizations(),
                new UnusedFunctionOptimizations()
        );
        SameDiff optimized = GraphOptimizer.optimize(sd, Collections.singletonList("out"), opts);

        // Identity should be removed
        assertFalse(optimized.hasVariable("identData"),
                "Identity variable should be removed after optimization");

        // All ops should still have valid inputs
        assertAllOpsHaveValidInputs(optimized);

        // The reshape variable should still exist and its producing op should have 2 inputs
        SameDiffOp reshapeOp = findOpProducingVariable(optimized, "reshaped");
        assertNotNull(reshapeOp, "Op producing 'reshaped' should still exist");
        assertEquals(2, reshapeOp.getInputsToOp().size(),
                "Reshape op should have exactly 2 inputs (data + shape), got: " + reshapeOp.getInputsToOp());
        assertTrue(reshapeOp.getInputsToOp().contains("data"),
                "Reshape op should reference 'data' after identity removal");
        assertTrue(reshapeOp.getInputsToOp().contains("shapeConst"),
                "Reshape op should still reference 'shapeConst'");

        // data's inputsForOp should include BOTH the sum op and the reshape op
        Variable dataVar = optimized.getVariables().get("data");
        assertNotNull(dataVar, "data variable should exist");
        assertNotNull(dataVar.getInputsForOp(), "data should have consumers");
        assertTrue(dataVar.getInputsForOp().size() >= 2,
                "data should be consumed by at least 2 ops, got: " + dataVar.getInputsForOp());

        // Execute and verify correctness
        INDArray input = Nd4j.linspace(1, 12, 12, DataType.FLOAT).reshape(2, 6);
        INDArray originalResult = sd.outputSingle(Collections.singletonMap("data", input), "out");
        INDArray optimizedResult = optimized.outputSingle(Collections.singletonMap("data", input), "out");
        assertEquals(originalResult, optimizedResult,
                "Optimized graph should produce the same result as original");
    }

    /**
     * Test that multiple identity removals through the same variable correctly
     * accumulate all consumers. If X feeds identity->A and identity->B, and
     * both A and B are consumed by different ops, removing both identities
     * must merge all consumers onto X.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMultipleIdentityRemovalsMergeConsumers(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);

        // Two identity ops on the same input
        SDVariable id1 = sd.identity("id1", x);
        SDVariable id2 = sd.identity("id2", x);

        // Each identity output feeds a different downstream op
        SDVariable sum1 = sd.sum("sum1", id1);
        SDVariable sum2 = sd.sum("sum2", id2);

        SDVariable out = sum1.add("out", sum2);

        List<OptimizerSet> opts = Arrays.asList(
                new IdentityFunctionOptimizations(),
                new UnusedFunctionOptimizations()
        );
        SameDiff optimized = GraphOptimizer.optimize(sd, Collections.singletonList("out"), opts);

        // Both identities should be removed
        assertFalse(optimized.hasVariable("id1"));
        assertFalse(optimized.hasVariable("id2"));

        // x should now be consumed by the ops that produce sum1 and sum2
        Variable xVar = optimized.getVariables().get("x");
        assertNotNull(xVar);
        assertNotNull(xVar.getInputsForOp());
        // x should have at least 2 consumers (the reduce ops for sum1 and sum2)
        assertTrue(xVar.getInputsForOp().size() >= 2,
                "x should have at least 2 consumers after identity removal, got: " + xVar.getInputsForOp());

        // All ops should have valid inputs
        assertAllOpsHaveValidInputs(optimized);

        // Verify correctness
        INDArray input = Nd4j.rand(DataType.FLOAT, 3, 4);
        INDArray originalResult = sd.outputSingle(Collections.singletonMap("x", input), "out");
        INDArray optimizedResult = optimized.outputSingle(Collections.singletonMap("x", input), "out");
        assertEquals(originalResult, optimizedResult,
                "Optimized result should match original");
    }

    /**
     * Simulates the exact VLM decoder pattern: reshape(identity(data), constantShape).
     * The identity on the data input is removed, and the reshape must retain both
     * its data and shape inputs.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReshapeWithIdentityOnDataInput(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();

        // Mimic ONNX-imported reshape pattern:
        // data_input -> identity -> reshape(identity_out, shape_constant) -> downstream
        SDVariable data = sd.placeHolder("data_input", DataType.FLOAT, 1, 12);
        SDVariable shape = sd.constant("shape_const", Nd4j.createFromArray(new long[]{1, 3, 4}));
        SDVariable ident = sd.identity("ident_data", data);
        SDVariable reshaped = sd.reshape("reshape_out", ident, shape);
        SDVariable out = sd.sum("output", reshaped);

        // Run only identity removal
        List<OptimizerSet> opts = Arrays.asList(
                new IdentityFunctionOptimizations(),
                new UnusedFunctionOptimizations()
        );
        SameDiff optimized = GraphOptimizer.optimize(sd, Collections.singletonList("output"), opts);

        // Validate: find the op that produces reshape_out
        SameDiffOp reshapeOp = findOpProducingVariable(optimized, "reshape_out");
        assertNotNull(reshapeOp, "Op producing 'reshape_out' must still exist");
        List<String> reshapeInputs = reshapeOp.getInputsToOp();
        assertEquals(2, reshapeInputs.size(),
                "reshape must have 2 inputs, got " + reshapeInputs.size() + ": " + reshapeInputs);

        // Both inputs must resolve to existing variables
        for (String inp : reshapeInputs) {
            assertTrue(optimized.getVariables().containsKey(inp),
                    "reshape input '" + inp + "' not found in variables");
        }

        // All ops valid
        assertAllOpsHaveValidInputs(optimized);

        // Execute and compare
        INDArray input = Nd4j.linspace(1, 12, 12, DataType.FLOAT).reshape(1, 12);
        INDArray originalResult = sd.outputSingle(Collections.singletonMap("data_input", input), "output");
        INDArray optimizedResult = optimized.outputSingle(Collections.singletonMap("data_input", input), "output");
        assertEquals(originalResult, optimizedResult,
                "Optimized reshape graph should produce the same result");
    }

    /**
     * Test that graph validation works on a valid graph (no errors).
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGraphValidationOnValidGraph(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        SDVariable a = sd.constant("a", Nd4j.scalar(3.0));
        SDVariable b = sd.constant("b", Nd4j.scalar(4.0));
        SDVariable out = a.add("out", b);

        // Verify original works
        INDArray result = sd.outputSingle(Collections.emptyMap(), "out");
        assertEquals(7.0, result.getDouble(0), 1e-5);

        // Full optimization should work fine on a valid graph
        SameDiff optimized = GraphOptimizer.optimize(sd, Collections.singletonList("out"));
        INDArray optimizedResult = optimized.outputSingle(Collections.emptyMap(), "out");
        assertEquals(7.0, optimizedResult.getDouble(0), 1e-5);

        // All ops valid
        assertAllOpsHaveValidInputs(optimized);
    }

    /**
     * Test that the full default optimization pipeline doesn't corrupt a
     * graph with the identity-before-reshape pattern. This is an end-to-end
     * test using all default optimizations.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testFullOptimizationPipelineWithReshape(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 8);
        SDVariable weights = sd.var("weights", Nd4j.rand(DataType.FLOAT, 8, 4));
        SDVariable matmul = sd.mmul("matmul", input, weights);

        // Identity before reshape (common in ONNX imports)
        SDVariable identMatmul = sd.identity("ident_matmul", matmul);
        SDVariable shape = sd.constant("target_shape", Nd4j.createFromArray(new long[]{-1, 2, 2}));
        SDVariable reshaped = sd.reshape("reshaped", identMatmul, shape);

        // Another identity
        SDVariable identReshaped = sd.identity("ident_reshaped", reshaped);
        SDVariable out = sd.sum("out", identReshaped);

        // Run full optimization pipeline
        SameDiff optimized = GraphOptimizer.optimize(sd, "out");

        // Identities should be removed
        assertFalse(optimized.hasVariable("ident_matmul"));
        assertFalse(optimized.hasVariable("ident_reshaped"));

        // Validate all ops have valid inputs
        assertAllOpsHaveValidInputs(optimized);

        // Execute and compare
        INDArray inputArr = Nd4j.rand(DataType.FLOAT, 3, 8);
        INDArray originalResult = sd.outputSingle(Collections.singletonMap("input", inputArr), "out");
        INDArray optimizedResult = optimized.outputSingle(Collections.singletonMap("input", inputArr), "out");
        assertEquals(originalResult.getDouble(0), optimizedResult.getDouble(0), 1e-4,
                "Full optimization pipeline should preserve result correctness");
    }

    /**
     * Test the scenario where a variable is consumed by many ops (fan-out),
     * and an identity on that variable is removed. All consumers must be preserved.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testHighFanOutIdentityRemoval(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 4);

        // x is directly used by op0
        SDVariable directUse = sd.math.square("op0_square", x);

        // identity(x) is used by op1, op2, op3
        SDVariable idX = sd.identity("id_x", x);
        SDVariable op1 = sd.sum("op1_sum", idX);
        SDVariable op2 = sd.mean("op2_mean", idX);
        SDVariable op3 = sd.math.abs("op3_abs", idX);

        SDVariable partial = directUse.sum("directSum");
        SDVariable op3Sum = op3.sum("op3Sum");
        SDVariable combined = partial.add("combined", op1).add("combined2", op2).add("out", op3Sum);

        List<OptimizerSet> opts = Arrays.asList(
                new IdentityFunctionOptimizations(),
                new UnusedFunctionOptimizations()
        );
        SameDiff optimized = GraphOptimizer.optimize(sd, Collections.singletonList("out"), opts);

        // identity should be gone
        assertFalse(optimized.hasVariable("id_x"));

        // x should now be consumed by at least 4 ops (square, sum, mean, abs)
        Variable xVar = optimized.getVariables().get("x");
        assertNotNull(xVar);
        List<String> consumers = xVar.getInputsForOp();
        assertNotNull(consumers);
        assertTrue(consumers.size() >= 4,
                "x should have at least 4 consumers after identity removal, got " + consumers.size() + ": " + consumers);

        // All ops valid
        assertAllOpsHaveValidInputs(optimized);

        // Verify correctness
        INDArray input = Nd4j.createFromArray(new float[]{1, 2, 3, 4});
        INDArray originalResult = sd.outputSingle(Collections.singletonMap("x", input), "out");
        INDArray optimizedResult = optimized.outputSingle(Collections.singletonMap("x", input), "out");
        assertEquals(originalResult.getDouble(0), optimizedResult.getDouble(0), 1e-4,
                "High fan-out identity removal should preserve correctness");
    }

    /**
     * Test that chained identities (identity(identity(x))) are correctly removed
     * and all consumers are properly merged back to x.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testChainedIdentityRemoval(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 3);
        SDVariable id1 = sd.identity("id1", x);
        SDVariable id2 = sd.identity("id2", id1);
        SDVariable id3 = sd.identity("id3", id2);
        SDVariable out = sd.sum("out", id3);

        List<OptimizerSet> opts = Arrays.asList(
                new IdentityFunctionOptimizations(),
                new UnusedFunctionOptimizations()
        );
        SameDiff optimized = GraphOptimizer.optimize(sd, Collections.singletonList("out"), opts);

        // All identities removed
        assertFalse(optimized.hasVariable("id1"));
        assertFalse(optimized.hasVariable("id2"));
        assertFalse(optimized.hasVariable("id3"));

        // All ops valid
        assertAllOpsHaveValidInputs(optimized);

        // Verify execution
        INDArray input = Nd4j.rand(DataType.FLOAT, 2, 3);
        INDArray originalResult = sd.outputSingle(Collections.singletonMap("x", input), "out");
        INDArray optimizedResult = optimized.outputSingle(Collections.singletonMap("x", input), "out");
        assertEquals(originalResult.getDouble(0), optimizedResult.getDouble(0), 1e-5,
                "Chained identity removal should preserve correctness");
    }
}
