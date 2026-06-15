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
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.optimize.GraphOptimizer;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.shape.Transpose;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.AddOp;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.MulOp;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for operator reordering optimizations:
 * - Associative reassociation: (a + c1) + c2 -> a + (c1+c2)
 * - Double transpose elimination: transpose(transpose(x)) -> x
 */
@Tag(TagNames.SAMEDIFF)
public class TestReorderingOptimizations {

    // ─── Associative reassociation (add) ───────────────────────────────────

    /**
     * (x + 3) + 5 should be reassociated to x + 8.
     * After reassociation + constant folding, only one add op should remain.
     */
    @Test
    void testReassociateAdd_TwoConstants() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable c1 = sd.constant("c1", Nd4j.ones(DataType.FLOAT, 4).mul(3.0));
        SDVariable c2 = sd.constant("c2", Nd4j.ones(DataType.FLOAT, 4).mul(5.0));
        // Build: (x + c1) + c2
        SDVariable inner = x.add("inner", c1);
        SDVariable outer = inner.add("output", c2);
        sd.setOutputs("output");

        INDArray input = Nd4j.randn(DataType.FLOAT, 2, 4);
        INDArray refOut = sd.output(Map.of("x", input), "output").get("output").dup();

        SameDiff optimized = GraphOptimizer.optimize(sd, "output");

        // Should have at most 1 add op (the reassociated one; constant folding
        // may further reduce if the folded constant is merged)
        int addCount = countOpsOfType(optimized, AddOp.class);
        assertTrue(addCount <= 1,
                "Reassociation should reduce to at most 1 add op, got " + addCount);

        // Numerical correctness
        INDArray optOut = optimized.output(Map.of("x", input), "output").get("output");
        double diff = refOut.sub(optOut).amaxNumber().doubleValue();
        assertTrue(diff < 1e-5, "Reassociate add diff = " + diff);
    }

    /**
     * (x * 2) * 3 should be reassociated to x * 6.
     */
    @Test
    void testReassociateMul_TwoConstants() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable c1 = sd.constant("c1", Nd4j.ones(DataType.FLOAT, 4).mul(2.0));
        SDVariable c2 = sd.constant("c2", Nd4j.ones(DataType.FLOAT, 4).mul(3.0));
        SDVariable inner = x.mul("inner", c1);
        SDVariable outer = inner.mul("output", c2);
        sd.setOutputs("output");

        INDArray input = Nd4j.randn(DataType.FLOAT, 2, 4);
        INDArray refOut = sd.output(Map.of("x", input), "output").get("output").dup();

        SameDiff optimized = GraphOptimizer.optimize(sd, "output");

        int mulCount = countOpsOfType(optimized, MulOp.class);
        assertTrue(mulCount <= 1,
                "Reassociation should reduce to at most 1 mul op, got " + mulCount);

        INDArray optOut = optimized.output(Map.of("x", input), "output").get("output");
        double diff = refOut.sub(optOut).amaxNumber().doubleValue();
        assertTrue(diff < 1e-5, "Reassociate mul diff = " + diff);
    }

    /**
     * (x + c1) + y should NOT be reassociated (y is not a constant).
     */
    @Test
    void testReassociate_NonConstant_NoChange() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable y = sd.placeHolder("y", DataType.FLOAT, -1, 4);
        SDVariable c1 = sd.constant("c1", Nd4j.ones(DataType.FLOAT, 4).mul(3.0));
        SDVariable inner = x.add("inner", c1);
        SDVariable outer = inner.add("output", y);
        sd.setOutputs("output");

        INDArray inputX = Nd4j.randn(DataType.FLOAT, 2, 4);
        INDArray inputY = Nd4j.randn(DataType.FLOAT, 2, 4);
        INDArray refOut = sd.output(Map.of("x", inputX, "y", inputY), "output").get("output").dup();

        SameDiff optimized = GraphOptimizer.optimize(sd, "output");

        // Should still have 2 add ops (no reassociation possible)
        int addCount = countOpsOfType(optimized, AddOp.class);
        assertTrue(addCount >= 1,
                "Non-constant outer input should preserve add ops, got " + addCount);

        INDArray optOut = optimized.output(Map.of("x", inputX, "y", inputY), "output").get("output");
        double diff = refOut.sub(optOut).amaxNumber().doubleValue();
        assertTrue(diff < 1e-5, "Non-constant reassociate diff = " + diff);
    }

    /**
     * Mixed ops: (x + c1) * c2 should NOT be reassociated (add != mul).
     */
    @Test
    void testReassociate_MixedOps_NoChange() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable c1 = sd.constant("c1", Nd4j.ones(DataType.FLOAT, 4).mul(3.0));
        SDVariable c2 = sd.constant("c2", Nd4j.ones(DataType.FLOAT, 4).mul(5.0));
        SDVariable inner = x.add("inner", c1);
        SDVariable outer = inner.mul("output", c2);
        sd.setOutputs("output");

        INDArray input = Nd4j.randn(DataType.FLOAT, 2, 4);
        INDArray refOut = sd.output(Map.of("x", input), "output").get("output").dup();

        SameDiff optimized = GraphOptimizer.optimize(sd, "output");

        // Numerical correctness (no reassociation but other opts may fire)
        INDArray optOut = optimized.output(Map.of("x", input), "output").get("output");
        double diff = refOut.sub(optOut).amaxNumber().doubleValue();
        assertTrue(diff < 1e-5, "Mixed ops reassociate diff = " + diff);
    }

    // ─── Double transpose elimination ──────────────────────────────────────

    /**
     * transpose(transpose(x)) should be eliminated to just x.
     */
    @Test
    void testDoubleTranspose_Eliminated() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
        SDVariable t1 = sd.transpose("t1", x);
        SDVariable t2 = sd.transpose("output", t1);
        sd.setOutputs("output");

        INDArray input = Nd4j.randn(DataType.FLOAT, 3, 4);
        INDArray refOut = sd.output(Map.of("x", input), "output").get("output").dup();

        SameDiff optimized = GraphOptimizer.optimize(sd, "output");

        // Both transposes should be eliminated
        int transposeCount = countOpsOfType(optimized, Transpose.class);
        assertEquals(0, transposeCount,
                "Double transpose should be fully eliminated, got " + transposeCount);

        // Output should be the input placeholder
        String actualOutput = optimized.outputs().get(0);
        assertEquals("x", actualOutput,
                "Output should be input placeholder after double transpose elimination");

        // Numerical correctness: transpose(transpose(x)) == x
        INDArray optOut = optimized.output(Map.of("x", input), actualOutput).get(actualOutput);
        double diff = refOut.sub(optOut).amaxNumber().doubleValue();
        assertTrue(diff < 1e-5, "Double transpose elimination diff = " + diff);
    }

    /**
     * A single transpose should NOT be eliminated.
     */
    @Test
    void testSingleTranspose_Preserved() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
        SDVariable t = sd.transpose("output", x);
        sd.setOutputs("output");

        SameDiff optimized = GraphOptimizer.optimize(sd, "output");

        assertTrue(hasOpOfType(optimized, Transpose.class),
                "Single transpose should be preserved");
    }

    /**
     * Triple transpose: transpose(transpose(transpose(x))) should reduce to
     * a single transpose.
     */
    @Test
    void testTripleTranspose_ReducesToSingle() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
        SDVariable t1 = sd.transpose("t1", x);
        SDVariable t2 = sd.transpose("t2", t1);
        SDVariable t3 = sd.transpose("output", t2);
        sd.setOutputs("output");

        INDArray input = Nd4j.randn(DataType.FLOAT, 3, 4);
        INDArray refOut = sd.output(Map.of("x", input), "output").get("output").dup();

        SameDiff optimized = GraphOptimizer.optimize(sd, "output");

        int transposeCount = countOpsOfType(optimized, Transpose.class);
        assertEquals(1, transposeCount,
                "Triple transpose should reduce to 1. Got " + transposeCount);

        String actualOutput = optimized.outputs().get(0);
        INDArray optOut = optimized.output(Map.of("x", input), actualOutput).get(actualOutput);
        double diff = refOut.sub(optOut).amaxNumber().doubleValue();
        assertTrue(diff < 1e-5, "Triple transpose reduction diff = " + diff);
    }

    /**
     * Quadruple transpose should fully eliminate (two pairs cancel out).
     */
    @Test
    void testQuadTranspose_FullyEliminated() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
        SDVariable t1 = sd.transpose("t1", x);
        SDVariable t2 = sd.transpose("t2", t1);
        SDVariable t3 = sd.transpose("t3", t2);
        SDVariable t4 = sd.transpose("output", t3);
        sd.setOutputs("output");

        INDArray input = Nd4j.randn(DataType.FLOAT, 3, 4);
        INDArray refOut = sd.output(Map.of("x", input), "output").get("output").dup();

        SameDiff optimized = GraphOptimizer.optimize(sd, "output");

        int transposeCount = countOpsOfType(optimized, Transpose.class);
        assertEquals(0, transposeCount,
                "Quad transpose should fully eliminate. Got " + transposeCount);

        String actualOutput = optimized.outputs().get(0);
        INDArray optOut = optimized.output(Map.of("x", input), actualOutput).get(actualOutput);
        double diff = refOut.sub(optOut).amaxNumber().doubleValue();
        assertTrue(diff < 1e-5, "Quad transpose elimination diff = " + diff);
    }

    // ─── Helpers ────────────────────────────────────────────────────────────

    private boolean hasOpOfType(SameDiff sd, Class<?> opType) {
        for (SameDiffOp op : sd.getOps().values()) {
            if (opType.isInstance(op.getOp())) return true;
        }
        return false;
    }

    private int countOpsOfType(SameDiff sd, Class<?> opType) {
        int count = 0;
        for (SameDiffOp op : sd.getOps().values()) {
            if (opType.isInstance(op.getOp())) count++;
        }
        return count;
    }
}
