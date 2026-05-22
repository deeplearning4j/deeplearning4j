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
import org.nd4j.linalg.api.ops.impl.shape.Permute;
import org.nd4j.linalg.api.ops.impl.shape.Reshape;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.DivOp;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for identity permute removal, no-op reshape elimination,
 * and x/x -> 1 algebraic simplification.
 */
@Tag(TagNames.SAMEDIFF)
public class TestIdentityAndShapeElimination {

    // ─── Identity Permute ──────────────────────────────────────────────────

    /**
     * permute(x, [0, 1]) on a 2D tensor should be eliminated.
     */
    @Test
    void testIdentityPermute_Eliminated() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
        SDVariable p = sd.permute("output", x, 0, 1);
        sd.setOutputs("output");

        INDArray input = Nd4j.randn(DataType.FLOAT, 3, 4);
        INDArray refOut = sd.output(Map.of("x", input), "output").get("output").dup();

        SameDiff optimized = GraphOptimizer.optimize(sd, "output");

        assertFalse(hasOpOfType(optimized, Permute.class),
                "Identity permute [0,1] should be eliminated");

        String actualOutput = optimized.outputs().get(0);
        INDArray optOut = optimized.output(Map.of("x", input), actualOutput).get(actualOutput);
        double diff = refOut.sub(optOut).amaxNumber().doubleValue();
        assertTrue(diff < 1e-5, "Identity permute elimination diff = " + diff);
    }

    /**
     * permute(x, [0, 1, 2]) on a 3D tensor should be eliminated.
     */
    @Test
    void testIdentityPermute3D_Eliminated() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 3, 4);
        SDVariable p = sd.permute("output", x, 0, 1, 2);
        sd.setOutputs("output");

        INDArray input = Nd4j.randn(DataType.FLOAT, 2, 3, 4);
        INDArray refOut = sd.output(Map.of("x", input), "output").get("output").dup();

        SameDiff optimized = GraphOptimizer.optimize(sd, "output");

        assertFalse(hasOpOfType(optimized, Permute.class),
                "Identity permute [0,1,2] should be eliminated");

        String actualOutput = optimized.outputs().get(0);
        INDArray optOut = optimized.output(Map.of("x", input), actualOutput).get(actualOutput);
        double diff = refOut.sub(optOut).amaxNumber().doubleValue();
        assertTrue(diff < 1e-5, "Identity permute 3D elimination diff = " + diff);
    }

    /**
     * Non-identity permute should be preserved.
     */
    @Test
    void testNonIdentityPermute_Preserved() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
        SDVariable p = sd.permute("output", x, 1, 0);
        sd.setOutputs("output");

        SameDiff optimized = GraphOptimizer.optimize(sd, "output");

        // [1, 0] is NOT identity -- should be preserved
        // Note: This might be a Transpose which is different from Permute
        // Just check numerical correctness
        INDArray input = Nd4j.randn(DataType.FLOAT, 3, 4);
        INDArray refOut = sd.output(Map.of("x", input), "output").get("output").dup();
        INDArray optOut = optimized.output(Map.of("x", input), "output").get("output");
        double diff = refOut.sub(optOut).amaxNumber().doubleValue();
        assertTrue(diff < 1e-5, "Non-identity permute should preserve correctness, diff = " + diff);
    }

    // ─── No-op Reshape ─────────────────────────────────────────────────────

    /**
     * reshape(x, same_shape) should be eliminated.
     */
    @Test
    void testNoopReshape_Eliminated() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
        // Reshape to the same shape [3, 4]
        SDVariable r = sd.reshape("output", x, 3, 4);
        sd.setOutputs("output");

        INDArray input = Nd4j.randn(DataType.FLOAT, 3, 4);
        INDArray refOut = sd.output(Map.of("x", input), "output").get("output").dup();

        SameDiff optimized = GraphOptimizer.optimize(sd, "output");

        assertFalse(hasOpOfType(optimized, Reshape.class),
                "No-op reshape to same shape should be eliminated");

        String actualOutput = optimized.outputs().get(0);
        INDArray optOut = optimized.output(Map.of("x", input), actualOutput).get(actualOutput);
        double diff = refOut.sub(optOut).amaxNumber().doubleValue();
        assertTrue(diff < 1e-5, "No-op reshape elimination diff = " + diff);
    }

    /**
     * Reshape to a different shape should be preserved.
     */
    @Test
    void testReshapeDifferentShape_Preserved() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
        SDVariable r = sd.reshape("output", x, 12);
        sd.setOutputs("output");

        INDArray input = Nd4j.randn(DataType.FLOAT, 3, 4);
        INDArray refOut = sd.output(Map.of("x", input), "output").get("output").dup();

        SameDiff optimized = GraphOptimizer.optimize(sd, "output");

        INDArray optOut = optimized.output(Map.of("x", input), "output").get("output");
        double diff = refOut.sub(optOut).amaxNumber().doubleValue();
        assertTrue(diff < 1e-5, "Reshape to different shape should preserve correctness, diff = " + diff);
    }

    // ─── x / x -> 1 ───────────────────────────────────────────────────────

    /**
     * x / x should simplify to a constant 1.
     */
    @Test
    void testDivideSelf_SimplifiesToOne() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable out = x.div("output", x);
        sd.setOutputs("output");

        INDArray input = Nd4j.randn(DataType.FLOAT, 2, 4).add(1.0); // avoid div by 0
        INDArray refOut = sd.output(Map.of("x", input), "output").get("output").dup();

        SameDiff optimized = GraphOptimizer.optimize(sd, "output");

        // Div op should be eliminated
        assertFalse(hasOpOfType(optimized, DivOp.class),
                "x / x should eliminate the div op");

        // Numerical correctness: result should be all 1s
        String actualOutput = optimized.outputs().get(0);
        INDArray optOut = optimized.output(Map.of("x", input), actualOutput).get(actualOutput);
        double diff = refOut.sub(optOut).amaxNumber().doubleValue();
        assertTrue(diff < 1e-5, "x / x -> 1 diff = " + diff);
    }

    /**
     * x / y (different variables) should NOT be simplified.
     */
    @Test
    void testDivideDifferent_Preserved() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable y = sd.placeHolder("y", DataType.FLOAT, -1, 4);
        SDVariable out = x.div("output", y);
        sd.setOutputs("output");

        SameDiff optimized = GraphOptimizer.optimize(sd, "output");

        assertTrue(hasOpOfType(optimized, DivOp.class),
                "x / y (different vars) should preserve div");
    }

    // ─── Helpers ────────────────────────────────────────────────────────────

    private boolean hasOpOfType(SameDiff sd, Class<?> opType) {
        for (SameDiffOp op : sd.getOps().values()) {
            if (opType.isInstance(op.getOp())) return true;
        }
        return false;
    }
}
