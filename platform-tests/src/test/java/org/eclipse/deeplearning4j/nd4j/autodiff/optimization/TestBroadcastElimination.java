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
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastTo;
import org.nd4j.linalg.api.ops.impl.transforms.same.Negative;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for broadcast elimination, double negation elimination, and
 * commutative op canonicalization optimizations.
 */
@Tag(TagNames.SAMEDIFF)
public class TestBroadcastElimination {

    // ─── Double negation elimination ────────────────────────────────────────

    /**
     * neg(neg(x)) should be eliminated to just x.
     */
    @Test
    void testDoubleNegation_Eliminated() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        SDVariable neg1 = sd.math().neg("neg1", x);
        SDVariable neg2 = sd.math().neg("output", neg1);
        sd.setOutputs("output");

        INDArray input = Nd4j.randn(DataType.FLOAT, 2, 8);
        INDArray refOut = sd.output(Map.of("x", input), "output").get("output").dup();

        SameDiff optimized = GraphOptimizer.optimize(sd, "output");

        // Both neg ops should be eliminated
        assertFalse(hasOpOfType(optimized, Negative.class),
                "Double negation should be fully eliminated");

        // Verify the output is directly the input placeholder
        String actualOutput = optimized.outputs().get(0);
        assertEquals("x", actualOutput,
                "Output should be the input placeholder after double negation elimination");

        // Verify numerical correctness
        INDArray optOut = optimized.output(Map.of("x", input), actualOutput).get(actualOutput);
        double diff = refOut.sub(optOut).amaxNumber().doubleValue();
        assertTrue(diff < 1e-5, "Double negation elimination diff = " + diff);
    }

    /**
     * A single negation should NOT be eliminated.
     */
    @Test
    void testSingleNegation_Preserved() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        SDVariable neg = sd.math().neg("output", x);
        sd.setOutputs("output");

        SameDiff optimized = GraphOptimizer.optimize(sd, "output");

        // Single neg should remain
        assertTrue(hasOpOfType(optimized, Negative.class),
                "Single negation should be preserved");
    }

    /**
     * Triple negation neg(neg(neg(x))) should reduce to neg(x).
     */
    @Test
    void testTripleNegation_ReducesToSingle() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable neg1 = sd.math().neg("neg1", x);
        SDVariable neg2 = sd.math().neg("neg2", neg1);
        SDVariable neg3 = sd.math().neg("output", neg2);
        sd.setOutputs("output");

        INDArray input = Nd4j.randn(DataType.FLOAT, 2, 4);
        INDArray refOut = sd.output(Map.of("x", input), "output").get("output").dup();

        SameDiff optimized = GraphOptimizer.optimize(sd, "output");

        // Count remaining neg ops -- should be exactly 1
        int negCount = countOpsOfType(optimized, Negative.class);
        assertEquals(1, negCount,
                "Triple negation should reduce to single neg. Got " + negCount);

        // Verify numerical correctness
        String actualOutput = optimized.outputs().get(0);
        INDArray optOut = optimized.output(Map.of("x", input), actualOutput).get(actualOutput);
        double diff = refOut.sub(optOut).amaxNumber().doubleValue();
        assertTrue(diff < 1e-5, "Triple negation reduction diff = " + diff);
    }

    // ─── Commutative op canonicalization ────────────────────────────────────

    /**
     * add(const, x) should be rewritten to add(x, const).
     * Verify by checking that the optimized graph still computes correctly.
     */
    @Test
    void testCanonicalizeAdd_ConstantToRight() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable c = sd.constant("bias", Nd4j.ones(DataType.FLOAT, 4).mul(3.0));
        // Build add(const, x) -- constant on the left
        SDVariable out = c.add("output", x);
        sd.setOutputs("output");

        INDArray input = Nd4j.randn(DataType.FLOAT, 2, 4);
        INDArray refOut = sd.output(Map.of("x", input), "output").get("output").dup();

        SameDiff optimized = GraphOptimizer.optimize(sd, "output");

        // Verify numerical correctness (canonicalization must not change semantics)
        INDArray optOut = optimized.output(Map.of("x", input), "output").get("output");
        double diff = refOut.sub(optOut).amaxNumber().doubleValue();
        assertTrue(diff < 1e-5, "Commutative canonicalization diff = " + diff);
    }

    /**
     * mul(const, x) should be rewritten to mul(x, const).
     */
    @Test
    void testCanonicalizeMul_ConstantToRight() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable c = sd.constant("scale", Nd4j.ones(DataType.FLOAT, 4).mul(2.5));
        // Build mul(const, x) -- constant on the left
        SDVariable out = c.mul("output", x);
        sd.setOutputs("output");

        INDArray input = Nd4j.randn(DataType.FLOAT, 2, 4);
        INDArray refOut = sd.output(Map.of("x", input), "output").get("output").dup();

        SameDiff optimized = GraphOptimizer.optimize(sd, "output");

        INDArray optOut = optimized.output(Map.of("x", input), "output").get("output");
        double diff = refOut.sub(optOut).amaxNumber().doubleValue();
        assertTrue(diff < 1e-5, "Commutative mul canonicalization diff = " + diff);
    }

    // ─── Broadcast elimination ──────────────────────────────────────────────

    /**
     * broadcast_to(x, shape) → add(y, ...) should eliminate the broadcast_to
     * since add handles broadcasting natively.
     */
    @Test
    void testBroadcastBeforeAdd_Eliminated() {
        SameDiff sd = SameDiff.create();
        // x is [1, 4], y is [3, 4]
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 4);
        SDVariable y = sd.placeHolder("y", DataType.FLOAT, 3, 4);
        // Explicitly broadcast x to [3, 4] before adding
        SDVariable shape = sd.constant("shape", Nd4j.createFromArray(new long[]{3, 4}));
        SDVariable broadcasted = new BroadcastTo(sd, x, shape).outputVariable();
        String bcastName = broadcasted.name();
        SDVariable out = broadcasted.add("output", y);
        sd.setOutputs("output");

        INDArray inputX = Nd4j.randn(DataType.FLOAT, 1, 4);
        INDArray inputY = Nd4j.randn(DataType.FLOAT, 3, 4);
        INDArray refOut = sd.output(Map.of("x", inputX, "y", inputY), "output").get("output").dup();

        SameDiff optimized = GraphOptimizer.optimize(sd, "output");

        // broadcast_to should be eliminated
        assertFalse(hasOpOfType(optimized, BroadcastTo.class),
                "broadcast_to should be eliminated before add");

        // Verify numerical correctness
        INDArray optOut = optimized.output(Map.of("x", inputX, "y", inputY), "output").get("output");
        double diff = refOut.sub(optOut).amaxNumber().doubleValue();
        assertTrue(diff < 1e-5, "Broadcast elimination before add diff = " + diff);
    }

    /**
     * broadcast_to(x, shape) → mul(y, ...) should eliminate the broadcast_to
     * since mul handles broadcasting natively.
     */
    @Test
    void testBroadcastBeforeMul_Eliminated() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 4);
        SDVariable y = sd.placeHolder("y", DataType.FLOAT, 3, 4);
        SDVariable shape = sd.constant("shape", Nd4j.createFromArray(new long[]{3, 4}));
        SDVariable broadcasted = new BroadcastTo(sd, x, shape).outputVariable();
        SDVariable out = broadcasted.mul("output", y);
        sd.setOutputs("output");

        INDArray inputX = Nd4j.randn(DataType.FLOAT, 1, 4);
        INDArray inputY = Nd4j.randn(DataType.FLOAT, 3, 4);
        INDArray refOut = sd.output(Map.of("x", inputX, "y", inputY), "output").get("output").dup();

        SameDiff optimized = GraphOptimizer.optimize(sd, "output");

        assertFalse(hasOpOfType(optimized, BroadcastTo.class),
                "broadcast_to should be eliminated before mul");

        INDArray optOut = optimized.output(Map.of("x", inputX, "y", inputY), "output").get("output");
        double diff = refOut.sub(optOut).amaxNumber().doubleValue();
        assertTrue(diff < 1e-5, "Broadcast elimination before mul diff = " + diff);
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
