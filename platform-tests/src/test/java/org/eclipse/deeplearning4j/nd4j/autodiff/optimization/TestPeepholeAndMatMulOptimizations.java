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
 * Tests for additional peephole optimizations and matmul chain optimizations.
 */
public class TestPeepholeAndMatMulOptimizations {

    // ─── PeepholeOptimizations: add(x, neg(y)) → sub(x, y) ────────────

    @Test
    public void testAddNegToSub() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable y = sd.placeHolder("y", DataType.FLOAT, -1, 4);
        SDVariable negY = y.neg("neg_y");
        SDVariable out = x.add("output", negY);
        sd.setOutputs(Collections.singletonList("output"));

        SameDiff optimized = GraphOptimizer.optimize(sd, sd.outputs(),
                Collections.<OptimizerSet>singletonList(new PeepholeOptimizations()));

        // The add and neg should be replaced by a sub
        boolean hasAdd = false;
        boolean hasNeg = false;
        for (var entry : optimized.getOps().entrySet()) {
            if (entry.getValue().getOp().opName().equals("add")) hasAdd = true;
            if (entry.getValue().getOp().opName().equals("neg")) hasNeg = true;
        }
        assertFalse(hasAdd, "add should be eliminated");

        // Verify execution correctness
        INDArray xArr = Nd4j.rand(DataType.FLOAT, 2, 4);
        INDArray yArr = Nd4j.rand(DataType.FLOAT, 2, 4);
        INDArray expected = xArr.sub(yArr);

        Map<String, INDArray> result = optimized.output(
                Map.of("x", xArr, "y", yArr), "output");
        double diff = expected.sub(result.get("output")).amaxNumber().doubleValue();
        assertTrue(diff < 1e-5, "add(x, neg(y)) -> sub(x, y) diff = " + diff);
    }

    // ─── PeepholeOptimizations: mul(x, -1) → neg(x) ──────────────────

    @Test
    public void testMulNegOneToNeg() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable negOne = sd.constant("neg_one", Nd4j.createFromArray(-1.0f));
        SDVariable out = x.mul("output", negOne);
        sd.setOutputs(Collections.singletonList("output"));

        SameDiff optimized = GraphOptimizer.optimize(sd, sd.outputs(),
                Collections.<OptimizerSet>singletonList(new PeepholeOptimizations()));

        // The mul should be replaced by neg
        boolean hasMul = false;
        boolean hasNeg = false;
        for (var entry : optimized.getOps().entrySet()) {
            if (entry.getValue().getOp().opName().equals("multiply")) hasMul = true;
            if (entry.getValue().getOp().opName().equals("neg")) hasNeg = true;
        }
        assertFalse(hasMul, "mul should be eliminated");
        assertTrue(hasNeg, "neg should be present");

        // Verify execution correctness
        INDArray xArr = Nd4j.rand(DataType.FLOAT, 2, 4);
        INDArray expected = xArr.neg();

        Map<String, INDArray> result = optimized.output(
                Map.of("x", xArr.dup()), "output");
        double diff = expected.sub(result.get("output")).amaxNumber().doubleValue();
        assertTrue(diff < 1e-5, "mul(x, -1) -> neg(x) diff = " + diff);
    }

    // ─── PeepholeOptimizations: sub(0, x) → neg(x) ───────────────────

    @Test
    public void testSubZeroToNeg() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable zero = sd.constant("zero", Nd4j.createFromArray(0.0f));
        SDVariable out = zero.sub("output", x);
        sd.setOutputs(Collections.singletonList("output"));

        SameDiff optimized = GraphOptimizer.optimize(sd, sd.outputs(),
                Collections.<OptimizerSet>singletonList(new PeepholeOptimizations()));

        // The sub should be replaced by neg
        boolean hasSub = false;
        boolean hasNeg = false;
        for (var entry : optimized.getOps().entrySet()) {
            if (entry.getValue().getOp().opName().equals("subtract")) hasSub = true;
            if (entry.getValue().getOp().opName().equals("neg")) hasNeg = true;
        }
        assertFalse(hasSub, "sub should be eliminated");
        assertTrue(hasNeg, "neg should be present");

        // Verify execution correctness
        INDArray xArr = Nd4j.rand(DataType.FLOAT, 2, 4);
        INDArray expected = xArr.neg();

        Map<String, INDArray> result = optimized.output(
                Map.of("x", xArr.dup()), "output");
        double diff = expected.sub(result.get("output")).amaxNumber().doubleValue();
        assertTrue(diff < 1e-5, "sub(0, x) -> neg(x) diff = " + diff);
    }

    // ─── PeepholeOptimizations: negative cases ────────────────────────

    @Test
    public void testMulByTwo_NotConverted() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable two = sd.constant("two", Nd4j.createFromArray(2.0f));
        SDVariable out = x.mul("output", two);
        sd.setOutputs(Collections.singletonList("output"));

        SameDiff optimized = GraphOptimizer.optimize(sd, sd.outputs(),
                Collections.<OptimizerSet>singletonList(new PeepholeOptimizations()));

        // mul by 2 should NOT be converted to neg
        boolean hasMul = false;
        for (var entry : optimized.getOps().entrySet()) {
            if (entry.getValue().getOp().opName().equals("multiply")) hasMul = true;
        }
        assertTrue(hasMul, "mul by 2 should be preserved");
    }

    // ─── MatMulChainOptimizations: fold constant chain ────────────────

    @Test
    public void testFoldConstantMatMulChain() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable w1 = sd.constant("w1", Nd4j.rand(DataType.FLOAT, 4, 8));
        SDVariable w2 = sd.constant("w2", Nd4j.rand(DataType.FLOAT, 8, 3));
        SDVariable inner = sd.mmul("inner", x, w1);
        SDVariable output = sd.mmul("output", inner, w2);
        sd.setOutputs(Collections.singletonList("output"));

        // Get reference output before optimization
        INDArray xArr = Nd4j.rand(DataType.FLOAT, 2, 4);
        Map<String, INDArray> refResult = sd.output(Map.of("x", xArr.dup()), "output");

        SameDiff optimized = GraphOptimizer.optimize(sd, sd.outputs(),
                Collections.<OptimizerSet>singletonList(new MatMulChainOptimizations()));

        // Should have only 1 matmul now (the chain folded)
        int mmulCount = 0;
        for (var entry : optimized.getOps().entrySet()) {
            if (entry.getValue().getOp().opName().equals("matmul")) mmulCount++;
        }
        assertEquals(1, mmulCount, "Should have exactly 1 matmul after folding");

        // Verify correctness
        Map<String, INDArray> optResult = optimized.output(Map.of("x", xArr.dup()), "output");
        double diff = refResult.get("output").sub(optResult.get("output")).amaxNumber().doubleValue();
        assertTrue(diff < 1e-4, "Folded matmul chain diff = " + diff);
    }

    @Test
    public void testFoldConstantMatMulChain_NonConstantWeight_NoFold() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable w1 = sd.placeHolder("w1", DataType.FLOAT, 4, 8);  // NOT constant
        SDVariable w2 = sd.constant("w2", Nd4j.rand(DataType.FLOAT, 8, 3));
        SDVariable inner = sd.mmul("inner", x, w1);
        SDVariable output = sd.mmul("output", inner, w2);
        sd.setOutputs(Collections.singletonList("output"));

        SameDiff optimized = GraphOptimizer.optimize(sd, sd.outputs(),
                Collections.<OptimizerSet>singletonList(new MatMulChainOptimizations()));

        // Should still have 2 matmuls (w1 is not constant)
        int mmulCount = 0;
        for (var entry : optimized.getOps().entrySet()) {
            if (entry.getValue().getOp().opName().equals("matmul")) mmulCount++;
        }
        assertEquals(2, mmulCount, "Should still have 2 matmuls");
    }

    // ─── MatMulChainOptimizations: transpose-matmul fusion ────────────

    @Test
    public void testTransposeMatMulFusion_TransposeA() {
        SameDiff sd = SameDiff.create();
        SDVariable a = sd.placeHolder("a", DataType.FLOAT, 4, -1);  // Will be transposed
        SDVariable b = sd.placeHolder("b", DataType.FLOAT, 4, 3);
        SDVariable aT = sd.transpose("a_transposed", a);
        SDVariable output = sd.mmul("output", aT, b);
        sd.setOutputs(Collections.singletonList("output"));

        // Reference
        INDArray aArr = Nd4j.rand(DataType.FLOAT, 4, 2);
        INDArray bArr = Nd4j.rand(DataType.FLOAT, 4, 3);
        Map<String, INDArray> refResult = sd.output(Map.of("a", aArr.dup(), "b", bArr.dup()), "output");

        SameDiff optimized = GraphOptimizer.optimize(sd, sd.outputs(),
                Collections.<OptimizerSet>singletonList(new MatMulChainOptimizations()));

        // Transpose should be absorbed
        boolean hasTranspose = false;
        for (var entry : optimized.getOps().entrySet()) {
            if (entry.getValue().getOp().opName().equals("transpose")) hasTranspose = true;
        }
        assertFalse(hasTranspose, "Transpose should be absorbed into matmul");

        // Verify correctness
        Map<String, INDArray> optResult = optimized.output(
                Map.of("a", aArr.dup(), "b", bArr.dup()), "output");
        double diff = refResult.get("output").sub(optResult.get("output")).amaxNumber().doubleValue();
        assertTrue(diff < 1e-5, "Transpose-matmul fusion diff = " + diff);
    }

    @Test
    public void testTransposeMatMulFusion_TransposeB() {
        SameDiff sd = SameDiff.create();
        SDVariable a = sd.placeHolder("a", DataType.FLOAT, -1, 4);
        SDVariable b = sd.placeHolder("b", DataType.FLOAT, 3, 4);  // Will be transposed
        SDVariable bT = sd.transpose("b_transposed", b);
        SDVariable output = sd.mmul("output", a, bT);
        sd.setOutputs(Collections.singletonList("output"));

        // Reference
        INDArray aArr = Nd4j.rand(DataType.FLOAT, 2, 4);
        INDArray bArr = Nd4j.rand(DataType.FLOAT, 3, 4);
        Map<String, INDArray> refResult = sd.output(Map.of("a", aArr.dup(), "b", bArr.dup()), "output");

        SameDiff optimized = GraphOptimizer.optimize(sd, sd.outputs(),
                Collections.<OptimizerSet>singletonList(new MatMulChainOptimizations()));

        // Transpose should be absorbed
        boolean hasTranspose = false;
        for (var entry : optimized.getOps().entrySet()) {
            if (entry.getValue().getOp().opName().equals("transpose")) hasTranspose = true;
        }
        assertFalse(hasTranspose, "Transpose should be absorbed into matmul");

        // Verify correctness
        Map<String, INDArray> optResult = optimized.output(
                Map.of("a", aArr.dup(), "b", bArr.dup()), "output");
        double diff = refResult.get("output").sub(optResult.get("output")).amaxNumber().doubleValue();
        assertTrue(diff < 1e-5, "Transpose-matmul fusion B diff = " + diff);
    }

    @Test
    public void testTransposeSharedNotAbsorbed() {
        SameDiff sd = SameDiff.create();
        SDVariable a = sd.placeHolder("a", DataType.FLOAT, 4, -1);
        SDVariable b = sd.placeHolder("b", DataType.FLOAT, 4, 3);
        SDVariable aT = sd.transpose("a_transposed", a);
        SDVariable output1 = sd.mmul("output", aT, b);
        // aT is also used by another op — should NOT be absorbed
        SDVariable output2 = aT.add("output2", b);
        sd.setOutputs(Arrays.asList("output", "output2"));

        SameDiff optimized = GraphOptimizer.optimize(sd, sd.outputs(),
                Collections.<OptimizerSet>singletonList(new MatMulChainOptimizations()));

        // Transpose should be preserved since it has multiple consumers
        boolean hasTranspose = false;
        for (var entry : optimized.getOps().entrySet()) {
            if (entry.getValue().getOp().opName().equals("transpose")) hasTranspose = true;
        }
        assertTrue(hasTranspose, "Transpose should be preserved (shared)");
    }
}
