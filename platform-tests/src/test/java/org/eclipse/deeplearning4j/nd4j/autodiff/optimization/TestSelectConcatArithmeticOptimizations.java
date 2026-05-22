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
import org.nd4j.autodiff.samediff.optimize.GraphOptimizer;
import org.nd4j.autodiff.samediff.optimize.OptimizerSet;
import org.nd4j.autodiff.samediff.optimize.optimizations.*;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for SelectWhereOptimizations, ConcatSplitOptimizations, and ArithmeticChainOptimizations.
 */
public class TestSelectConcatArithmeticOptimizations {

    // ─── SelectWhereOptimizations: constant all-true condition ────────

    @Test
    public void testWhereConstantTrue() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable y = sd.placeHolder("y", DataType.FLOAT, -1, 4);
        // All-true condition
        SDVariable cond = sd.constant("cond", Nd4j.ones(DataType.BOOL, 2, 4));
        // where(x, y, cond) -> should simplify to x
        SDVariable selected = sd.where("output", x, y, cond);
        sd.setOutputs(Collections.singletonList("output"));

        SameDiff optimized = GraphOptimizer.optimize(sd, sd.outputs(),
                Collections.<OptimizerSet>singletonList(new SelectWhereOptimizations()));

        // The Where op should be eliminated
        boolean hasWhere = false;
        for (var entry : optimized.getOps().entrySet()) {
            if (entry.getValue().getOp().opName().equals("Where")) hasWhere = true;
        }
        assertFalse(hasWhere, "Where should be eliminated with all-true condition");

        // After optimization, output was replaced with "x" in graph outputs
        // Verify the graph output is now "x" (the true branch)
        List<String> optOutputs = optimized.outputs();
        assertTrue(optOutputs.contains("x"), "Graph output should be 'x' after optimization, got: " + optOutputs);
    }

    @Test
    public void testWhereConstantFalse() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable y = sd.placeHolder("y", DataType.FLOAT, -1, 4);
        // All-false condition
        SDVariable cond = sd.constant("cond", Nd4j.zeros(DataType.BOOL, 2, 4));
        SDVariable selected = sd.where("output", x, y, cond);
        sd.setOutputs(Collections.singletonList("output"));

        SameDiff optimized = GraphOptimizer.optimize(sd, sd.outputs(),
                Collections.<OptimizerSet>singletonList(new SelectWhereOptimizations()));

        boolean hasWhere = false;
        for (var entry : optimized.getOps().entrySet()) {
            if (entry.getValue().getOp().opName().equals("Where")) hasWhere = true;
        }
        assertFalse(hasWhere, "Where should be eliminated with all-false condition");

        // After optimization, output was replaced with "y" (the false branch)
        List<String> optOutputs = optimized.outputs();
        assertTrue(optOutputs.contains("y"), "Graph output should be 'y' after optimization, got: " + optOutputs);
    }

    @Test
    public void testWhereMixedCondition_NoOptimization() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 4);
        SDVariable y = sd.placeHolder("y", DataType.FLOAT, 4);
        // Mixed condition — cannot simplify
        INDArray condArr = Nd4j.createFromArray(true, false, true, false);
        SDVariable cond = sd.constant("cond", condArr);
        SDVariable selected = sd.where("output", x, y, cond);
        sd.setOutputs(Collections.singletonList("output"));

        SameDiff optimized = GraphOptimizer.optimize(sd, sd.outputs(),
                Collections.<OptimizerSet>singletonList(new SelectWhereOptimizations()));

        boolean hasWhere = false;
        for (var entry : optimized.getOps().entrySet()) {
            if (entry.getValue().getOp().opName().equals("Where")) hasWhere = true;
        }
        assertTrue(hasWhere, "Where with mixed condition should be preserved");
    }

    // ─── ConcatSplitOptimizations: flatten nested concat ─────────────

    @Test
    public void testFlattenNestedConcat() {
        SameDiff sd = SameDiff.create();
        SDVariable a = sd.placeHolder("a", DataType.FLOAT, -1, 4);
        SDVariable b = sd.placeHolder("b", DataType.FLOAT, -1, 4);
        SDVariable c = sd.placeHolder("c", DataType.FLOAT, -1, 4);

        // concat(concat(a, b, axis=0), c, axis=0) → concat(a, b, c, axis=0)
        SDVariable innerConcat = sd.concat("inner_concat", 0, a, b);
        SDVariable output = sd.concat("output", 0, innerConcat, c);
        sd.setOutputs(Collections.singletonList("output"));

        SameDiff optimized = GraphOptimizer.optimize(sd, sd.outputs(),
                Collections.<OptimizerSet>singletonList(new ConcatSplitOptimizations()));

        // Should have only 1 concat now
        int concatCount = 0;
        for (var entry : optimized.getOps().entrySet()) {
            if (entry.getValue().getOp().opName().equals("concat")) concatCount++;
        }
        assertEquals(1, concatCount, "Should have exactly 1 concat after flattening");

        // Verify correctness
        INDArray aArr = Nd4j.rand(DataType.FLOAT, 2, 4);
        INDArray bArr = Nd4j.rand(DataType.FLOAT, 3, 4);
        INDArray cArr = Nd4j.rand(DataType.FLOAT, 1, 4);
        INDArray expected = Nd4j.concat(0, aArr, bArr, cArr);

        Map<String, INDArray> result = optimized.output(
                Map.of("a", aArr, "b", bArr, "c", cArr), "output");
        double diff = expected.sub(result.get("output")).amaxNumber().doubleValue();
        assertTrue(diff < 1e-5, "Flattened concat diff = " + diff);
    }

    @Test
    public void testNestedConcatDifferentAxis_NoFlatten() {
        SameDiff sd = SameDiff.create();
        SDVariable a = sd.placeHolder("a", DataType.FLOAT, 2, 4);
        SDVariable b = sd.placeHolder("b", DataType.FLOAT, 2, 4);
        SDVariable c = sd.placeHolder("c", DataType.FLOAT, 2, 8);

        // concat(concat(a, b, axis=1), c, axis=1) but inner uses axis=1, outer axis=1
        // Actually let's test different axes: inner axis=1, outer axis=0 — should NOT flatten
        SDVariable innerConcat = sd.concat("inner_concat", 1, a, b);
        // inner produces [2, 8], c is [2, 8], concat on axis 0 => [4, 8]
        SDVariable output = sd.concat("output", 0, innerConcat, c);
        sd.setOutputs(Collections.singletonList("output"));

        SameDiff optimized = GraphOptimizer.optimize(sd, sd.outputs(),
                Collections.<OptimizerSet>singletonList(new ConcatSplitOptimizations()));

        // Should still have 2 concats (axes differ)
        int concatCount = 0;
        for (var entry : optimized.getOps().entrySet()) {
            if (entry.getValue().getOp().opName().equals("concat")) concatCount++;
        }
        assertEquals(2, concatCount, "Should preserve 2 concats with different axes");
    }

    // ─── ConcatSplitOptimizations: eliminate concat-split pairs ──────

    @Test
    public void testEliminateConcatSplit() {
        SameDiff sd = SameDiff.create();
        SDVariable a = sd.placeHolder("a", DataType.FLOAT, 2, 4);
        SDVariable b = sd.placeHolder("b", DataType.FLOAT, 2, 4);

        // concat then split along same axis with same number of pieces
        SDVariable concatenated = sd.concat("concatenated", 0, a, b);
        SDVariable[] splits = sd.split(new String[]{"split_0", "split_1"}, concatenated, 2, 0);
        // Use both splits as outputs so we can verify they map back to a, b
        sd.setOutputs(Arrays.asList("split_0", "split_1"));

        SameDiff optimized = GraphOptimizer.optimize(sd, sd.outputs(),
                Collections.<OptimizerSet>singletonList(new ConcatSplitOptimizations()));

        // The concat and split should both be eliminated
        boolean hasConcat = false;
        boolean hasSplit = false;
        for (var entry : optimized.getOps().entrySet()) {
            if (entry.getValue().getOp().opName().equals("concat")) hasConcat = true;
            if (entry.getValue().getOp().opName().equals("split")) hasSplit = true;
        }
        assertFalse(hasConcat, "concat should be eliminated");
        assertFalse(hasSplit, "split should be eliminated");

        // Graph outputs should now point to "a" and "b" directly
        List<String> optOutputs = optimized.outputs();
        assertTrue(optOutputs.contains("a"), "split_0 should resolve to a, got: " + optOutputs);
        assertTrue(optOutputs.contains("b"), "split_1 should resolve to b, got: " + optOutputs);
    }

    // ─── ArithmeticChainOptimizations: fold add chain ────────────────

    @Test
    public void testFoldAddChain() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable c1 = sd.constant("c1", Nd4j.scalar(3.0f));
        SDVariable c2 = sd.constant("c2", Nd4j.scalar(5.0f));
        SDVariable inner = x.add("inner_add", c1);
        SDVariable output = inner.add("output", c2);
        sd.setOutputs(Collections.singletonList("output"));

        SameDiff optimized = GraphOptimizer.optimize(sd, sd.outputs(),
                Collections.<OptimizerSet>singletonList(new ArithmeticChainOptimizations()));

        // Should have exactly 1 add (folded c1+c2=8)
        int addCount = 0;
        for (var entry : optimized.getOps().entrySet()) {
            if (entry.getValue().getOp().opName().equals("add")) addCount++;
        }
        assertEquals(1, addCount, "Should have exactly 1 add after folding");

        // Verify correctness: x + 3 + 5 = x + 8
        INDArray xArr = Nd4j.rand(DataType.FLOAT, 2, 4);
        INDArray expected = xArr.add(8.0f);
        Map<String, INDArray> result = optimized.output(Map.of("x", xArr.dup()), "output");
        double diff = expected.sub(result.get("output")).amaxNumber().doubleValue();
        assertTrue(diff < 1e-5, "add(add(x, 3), 5) -> add(x, 8) diff = " + diff);
    }

    @Test
    public void testFoldMulChain() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable c1 = sd.constant("c1", Nd4j.scalar(2.0f));
        SDVariable c2 = sd.constant("c2", Nd4j.scalar(3.0f));
        SDVariable inner = x.mul("inner_mul", c1);
        SDVariable output = inner.mul("output", c2);
        sd.setOutputs(Collections.singletonList("output"));

        SameDiff optimized = GraphOptimizer.optimize(sd, sd.outputs(),
                Collections.<OptimizerSet>singletonList(new ArithmeticChainOptimizations()));

        // Should have exactly 1 mul (folded c1*c2=6)
        int mulCount = 0;
        for (var entry : optimized.getOps().entrySet()) {
            if (entry.getValue().getOp().opName().equals("multiply")) mulCount++;
        }
        assertEquals(1, mulCount, "Should have exactly 1 mul after folding");

        // Verify correctness: x * 2 * 3 = x * 6
        INDArray xArr = Nd4j.rand(DataType.FLOAT, 2, 4);
        INDArray expected = xArr.mul(6.0f);
        Map<String, INDArray> result = optimized.output(Map.of("x", xArr.dup()), "output");
        double diff = expected.sub(result.get("output")).amaxNumber().doubleValue();
        assertTrue(diff < 1e-5, "mul(mul(x, 2), 3) -> mul(x, 6) diff = " + diff);
    }

    @Test
    public void testFoldAddChain_NonConstant_NoFold() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable y = sd.placeHolder("y", DataType.FLOAT, -1, 4);
        SDVariable c = sd.constant("c", Nd4j.scalar(5.0f));
        // add(add(x, y), c) — inner has no constant, should NOT fold
        SDVariable inner = x.add("inner_add", y);
        SDVariable output = inner.add("output", c);
        sd.setOutputs(Collections.singletonList("output"));

        SameDiff optimized = GraphOptimizer.optimize(sd, sd.outputs(),
                Collections.<OptimizerSet>singletonList(new ArithmeticChainOptimizations()));

        // Should still have 2 adds
        int addCount = 0;
        for (var entry : optimized.getOps().entrySet()) {
            if (entry.getValue().getOp().opName().equals("add")) addCount++;
        }
        assertEquals(2, addCount, "Should preserve 2 adds when inner has no constant");
    }

    @Test
    public void testFoldMulChain_SharedInner_NoFold() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable c1 = sd.constant("c1", Nd4j.scalar(2.0f));
        SDVariable c2 = sd.constant("c2", Nd4j.scalar(3.0f));
        SDVariable inner = x.mul("inner_mul", c1);
        SDVariable output1 = inner.mul("output1", c2);
        // inner is also used by another op — should NOT fold
        SDVariable output2 = inner.add("output2", c2);
        sd.setOutputs(Arrays.asList("output1", "output2"));

        SameDiff optimized = GraphOptimizer.optimize(sd, sd.outputs(),
                Collections.<OptimizerSet>singletonList(new ArithmeticChainOptimizations()));

        // Should still have 2 muls (inner has multiple consumers)
        int mulCount = 0;
        for (var entry : optimized.getOps().entrySet()) {
            if (entry.getValue().getOp().opName().equals("multiply")) mulCount++;
        }
        assertEquals(2, mulCount, "Should preserve 2 muls when inner result is shared");
    }
}
