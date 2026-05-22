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
import org.nd4j.linalg.api.ops.impl.scalar.Pow;
import org.nd4j.linalg.api.ops.impl.transforms.floating.Sqrt;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.DivOp;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.MulOp;
import org.nd4j.linalg.api.ops.impl.transforms.same.Reciprocal;
import org.nd4j.linalg.api.ops.impl.transforms.same.Square;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for strength reduction optimizations:
 * - pow(x,2) → square(x)
 * - pow(x,0.5) → sqrt(x)
 * - pow(x,-1) → reciprocal(x)
 * - pow(x,1) → x (identity)
 * - div(x,const) → mul(x,1/const)
 */
@Tag(TagNames.SAMEDIFF)
public class TestStrengthReduction {

    @Test
    void testPowSquare() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        // pow(x, 2) via scalar pow
        SDVariable p = new Pow(sd, x, 2.0).outputVariable();
        String pName = p.name();
        sd.setOutputs(pName);

        INDArray input = Nd4j.rand(DataType.FLOAT, 2, 4).addi(0.1);
        INDArray refOut = sd.output(Map.of("x", input), pName).get(pName).dup();

        SameDiff optimized = GraphOptimizer.optimize(sd, pName);

        // Should have replaced Pow with Square
        assertFalse(hasOpOfType(optimized, Pow.class), "Pow should be eliminated");
        assertTrue(hasOpOfType(optimized, Square.class), "Square should be present");

        INDArray optOut = optimized.output(Map.of("x", input), pName).get(pName);
        double diff = refOut.sub(optOut).amaxNumber().doubleValue();
        assertTrue(diff < 1e-5, "pow(x,2) vs square(x) diff = " + diff);
    }

    @Test
    void testPowSqrt() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable p = new Pow(sd, x, 0.5).outputVariable();
        String pName = p.name();
        sd.setOutputs(pName);

        INDArray input = Nd4j.rand(DataType.FLOAT, 2, 4).addi(0.1);
        INDArray refOut = sd.output(Map.of("x", input), pName).get(pName).dup();

        SameDiff optimized = GraphOptimizer.optimize(sd, pName);

        assertFalse(hasOpOfType(optimized, Pow.class), "Pow should be eliminated");
        assertTrue(hasOpOfType(optimized, Sqrt.class), "Sqrt should be present");

        INDArray optOut = optimized.output(Map.of("x", input), pName).get(pName);
        double diff = refOut.sub(optOut).amaxNumber().doubleValue();
        assertTrue(diff < 1e-5, "pow(x,0.5) vs sqrt(x) diff = " + diff);
    }

    @Test
    void testPowReciprocal() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable p = new Pow(sd, x, -1.0).outputVariable();
        String pName = p.name();
        sd.setOutputs(pName);

        INDArray input = Nd4j.rand(DataType.FLOAT, 2, 4).addi(0.5);
        INDArray refOut = sd.output(Map.of("x", input), pName).get(pName).dup();

        SameDiff optimized = GraphOptimizer.optimize(sd, pName);

        assertFalse(hasOpOfType(optimized, Pow.class), "Pow should be eliminated");
        assertTrue(hasOpOfType(optimized, Reciprocal.class), "Reciprocal should be present");

        INDArray optOut = optimized.output(Map.of("x", input), pName).get(pName);
        double diff = refOut.sub(optOut).amaxNumber().doubleValue();
        assertTrue(diff < 1e-4, "pow(x,-1) vs reciprocal(x) diff = " + diff);
    }

    @Test
    void testPowIdentity() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable p = new Pow(sd, x, 1.0).outputVariable();
        String pName = p.name();
        sd.setOutputs(pName);

        INDArray input = Nd4j.rand(DataType.FLOAT, 2, 4);
        INDArray refOut = sd.output(Map.of("x", input), pName).get(pName).dup();

        SameDiff optimized = GraphOptimizer.optimize(sd, pName);

        // pow(x,1) should be eliminated entirely (identity)
        assertFalse(hasOpOfType(optimized, Pow.class), "Pow should be eliminated");

        // The graph output becomes "x" directly — verify structurally and numerically
        String actualOutput = optimized.outputs().get(0);
        assertEquals("x", actualOutput, "Output should be the input placeholder after identity elimination");

        // Execute the zero-op graph — DSP handles passthrough plans
        INDArray optOut = optimized.output(Map.of("x", input), actualOutput).get(actualOutput);
        double diff = refOut.sub(optOut).amaxNumber().doubleValue();
        assertTrue(diff < 1e-5, "pow(x,1) identity execution diff = " + diff);
    }

    @Test
    void testDivByConstant() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable c = sd.constant("divisor", Nd4j.scalar(DataType.FLOAT, 3.0));
        SDVariable d = x.div("output", c);
        sd.setOutputs("output");

        INDArray input = Nd4j.rand(DataType.FLOAT, 2, 4);
        INDArray refOut = sd.output(Map.of("x", input), "output").get("output").dup();

        SameDiff optimized = GraphOptimizer.optimize(sd, "output");

        // div should be replaced with mul
        assertFalse(hasOpOfType(optimized, DivOp.class), "DivOp should be eliminated");
        assertTrue(hasOpOfType(optimized, MulOp.class), "MulOp should be present");

        INDArray optOut = optimized.output(Map.of("x", input), "output").get("output");
        double diff = refOut.sub(optOut).amaxNumber().doubleValue();
        assertTrue(diff < 1e-5, "div(x,3) vs mul(x,1/3) diff = " + diff);
    }

    @Test
    void testDivByOne_NotConverted() {
        // div(x, 1) should be handled by DivideOne in AlgebraicOptimizations, not strength reduction
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable c = sd.constant("divisor", Nd4j.scalar(DataType.FLOAT, 1.0));
        SDVariable d = x.div("output", c);
        sd.setOutputs("output");

        INDArray input = Nd4j.rand(DataType.FLOAT, 2, 4);
        INDArray refOut = sd.output(Map.of("x", input), "output").get("output").dup();

        SameDiff optimized = GraphOptimizer.optimize(sd, "output");

        // Should be fully eliminated (DivideOne handles it)
        assertFalse(hasOpOfType(optimized, DivOp.class), "DivOp should be eliminated by DivideOne");

        // The graph output becomes "x" directly — verify structurally and numerically
        String actualOutput = optimized.outputs().get(0);
        assertEquals("x", actualOutput, "Output should be the input placeholder after identity elimination");

        // Execute the zero-op graph — DSP handles passthrough plans
        INDArray optOut = optimized.output(Map.of("x", input), actualOutput).get(actualOutput);
        double diff = refOut.sub(optOut).amaxNumber().doubleValue();
        assertTrue(diff < 1e-5, "div(x,1) identity execution diff = " + diff);
    }

    @Test
    void testPowNonSpecialExponent_NoReduction() {
        // pow(x, 3) should NOT be reduced (no special case for cube)
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable p = new Pow(sd, x, 3.0).outputVariable();
        String pName = p.name();
        sd.setOutputs(pName);

        SameDiff optimized = GraphOptimizer.optimize(sd, pName);

        // Pow should remain
        assertTrue(hasOpOfType(optimized, Pow.class), "Pow(x,3) should remain (no special case)");
    }

    private boolean hasOpOfType(SameDiff sd, Class<?> opType) {
        for (SameDiffOp op : sd.getOps().values()) {
            if (opType.isInstance(op.getOp())) return true;
        }
        return false;
    }
}
