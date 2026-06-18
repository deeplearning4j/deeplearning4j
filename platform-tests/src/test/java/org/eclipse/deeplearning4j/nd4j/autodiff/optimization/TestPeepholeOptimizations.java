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
import org.nd4j.linalg.api.ops.impl.scalar.RectifiedLinear;
import org.nd4j.linalg.api.ops.impl.transforms.custom.Max;
import org.nd4j.linalg.api.ops.impl.transforms.custom.Min;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.AddOp;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.SubOp;
import org.nd4j.linalg.api.ops.impl.transforms.same.Abs;
import org.nd4j.linalg.api.ops.impl.transforms.same.Negative;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Exp;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Log;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for peephole optimizations:
 * - Idempotent ops: relu(relu(x)), abs(abs(x))
 * - Inverse pairs: exp(log(x)), log(exp(x))
 * - Redundant binary: min(x,x), max(x,x)
 * - Negation propagation: sub(x, neg(y)) -> add(x, y)
 */
@Tag(TagNames.SAMEDIFF)
public class TestPeepholeOptimizations {

    // ─── Idempotent: relu(relu(x)) → relu(x) ────────────────────────────

    @Test
    void testIdempotentRelu_Eliminated() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
        SDVariable relu1 = sd.nn().relu("relu1", x, 0);
        SDVariable relu2 = sd.nn().relu("output", relu1, 0);
        sd.setOutputs("output");

        INDArray input = Nd4j.randn(DataType.FLOAT, 3, 4);
        INDArray refOut = sd.output(Map.of("x", input), "output").get("output").dup();

        SameDiff optimized = GraphOptimizer.optimize(sd, "output");

        // Should have exactly one relu, not two
        int reluCount = countOpsOfType(optimized, RectifiedLinear.class);
        assertTrue(reluCount <= 1, "Expected at most 1 relu, got " + reluCount);

        String actualOutput = optimized.outputs().get(0);
        INDArray optOut = optimized.output(Map.of("x", input), actualOutput).get(actualOutput);
        double diff = refOut.sub(optOut).amaxNumber().doubleValue();
        assertTrue(diff < 1e-5, "relu(relu(x)) diff = " + diff);
    }

    @Test
    void testSingleRelu_Preserved() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
        SDVariable relu = sd.nn().relu("output", x, 0);
        sd.setOutputs("output");

        SameDiff optimized = GraphOptimizer.optimize(sd, "output");

        assertEquals(1, countOpsOfType(optimized, RectifiedLinear.class),
                "Single relu should be preserved");
    }

    // ─── Idempotent: abs(abs(x)) → abs(x) ───────────────────────────────

    @Test
    void testIdempotentAbs_Eliminated() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
        SDVariable abs1 = sd.math().abs("abs1", x);
        SDVariable abs2 = sd.math().abs("output", abs1);
        sd.setOutputs("output");

        INDArray input = Nd4j.randn(DataType.FLOAT, 3, 4);
        INDArray refOut = sd.output(Map.of("x", input), "output").get("output").dup();

        SameDiff optimized = GraphOptimizer.optimize(sd, "output");

        int absCount = countOpsOfType(optimized, Abs.class);
        assertTrue(absCount <= 1, "Expected at most 1 abs, got " + absCount);

        String actualOutput = optimized.outputs().get(0);
        INDArray optOut = optimized.output(Map.of("x", input), actualOutput).get(actualOutput);
        double diff = refOut.sub(optOut).amaxNumber().doubleValue();
        assertTrue(diff < 1e-5, "abs(abs(x)) diff = " + diff);
    }

    // ─── Inverse pairs: exp(log(x)) → x ─────────────────────────────────

    @Test
    void testInverseExpLog_Eliminated() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
        SDVariable logX = sd.math().log("logx", x);
        SDVariable expLogX = sd.math().exp("output", logX);
        sd.setOutputs("output");

        // Use positive values only (log requires positive input)
        INDArray input = Nd4j.rand(DataType.FLOAT, 3, 4).addi(0.1);
        INDArray refOut = sd.output(Map.of("x", input), "output").get("output").dup();

        SameDiff optimized = GraphOptimizer.optimize(sd, "output");

        // Both exp and log should be eliminated
        assertFalse(hasOpOfType(optimized, Exp.class),
                "exp should be eliminated in exp(log(x))");

        String actualOutput = optimized.outputs().get(0);
        INDArray optOut = optimized.output(Map.of("x", input), actualOutput).get(actualOutput);
        double diff = refOut.sub(optOut).amaxNumber().doubleValue();
        assertTrue(diff < 1e-4, "exp(log(x)) diff = " + diff);
    }

    // ─── Inverse pairs: log(exp(x)) → x ─────────────────────────────────

    @Test
    void testInverseLogExp_Eliminated() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
        SDVariable expX = sd.math().exp("expx", x);
        SDVariable logExpX = sd.math().log("output", expX);
        sd.setOutputs("output");

        // Use small values to avoid exp overflow
        INDArray input = Nd4j.rand(DataType.FLOAT, 3, 4).muli(2).subi(1);
        INDArray refOut = sd.output(Map.of("x", input), "output").get("output").dup();

        SameDiff optimized = GraphOptimizer.optimize(sd, "output");

        assertFalse(hasOpOfType(optimized, Log.class),
                "log should be eliminated in log(exp(x))");

        String actualOutput = optimized.outputs().get(0);
        INDArray optOut = optimized.output(Map.of("x", input), actualOutput).get(actualOutput);
        double diff = refOut.sub(optOut).amaxNumber().doubleValue();
        assertTrue(diff < 1e-4, "log(exp(x)) diff = " + diff);
    }

    // ─── Redundant binary: min(x, x) → x ────────────────────────────────

    @Test
    void testRedundantMin_Eliminated() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
        SDVariable minXX = sd.math().min("output", x, x);
        sd.setOutputs("output");

        INDArray input = Nd4j.randn(DataType.FLOAT, 3, 4);
        INDArray refOut = sd.output(Map.of("x", input), "output").get("output").dup();

        SameDiff optimized = GraphOptimizer.optimize(sd, "output");

        assertFalse(hasOpOfType(optimized, Min.class),
                "min(x, x) should be eliminated");

        String actualOutput = optimized.outputs().get(0);
        INDArray optOut = optimized.output(Map.of("x", input), actualOutput).get(actualOutput);
        double diff = refOut.sub(optOut).amaxNumber().doubleValue();
        assertTrue(diff < 1e-5, "min(x,x) diff = " + diff);
    }

    // ─── Redundant binary: max(x, x) → x ────────────────────────────────

    @Test
    void testRedundantMax_Eliminated() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
        SDVariable maxXX = sd.math().max("output", x, x);
        sd.setOutputs("output");

        INDArray input = Nd4j.randn(DataType.FLOAT, 3, 4);
        INDArray refOut = sd.output(Map.of("x", input), "output").get("output").dup();

        SameDiff optimized = GraphOptimizer.optimize(sd, "output");

        assertFalse(hasOpOfType(optimized, Max.class),
                "max(x, x) should be eliminated");

        String actualOutput = optimized.outputs().get(0);
        INDArray optOut = optimized.output(Map.of("x", input), actualOutput).get(actualOutput);
        double diff = refOut.sub(optOut).amaxNumber().doubleValue();
        assertTrue(diff < 1e-5, "max(x,x) diff = " + diff);
    }

    @Test
    void testMinDifferentInputs_Preserved() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
        SDVariable y = sd.placeHolder("y", DataType.FLOAT, 3, 4);
        SDVariable minXY = sd.math().min("output", x, y);
        sd.setOutputs("output");

        SameDiff optimized = GraphOptimizer.optimize(sd, "output");

        assertTrue(hasOpOfType(optimized, Min.class),
                "min(x, y) with different inputs should be preserved");
    }

    // ─── Negation propagation: sub(x, neg(y)) → add(x, y) ──────────────

    @Test
    void testSubNegToAdd() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
        SDVariable y = sd.placeHolder("y", DataType.FLOAT, 3, 4);
        SDVariable negY = sd.math().neg("negy", y);
        SDVariable out = x.sub("output", negY);
        sd.setOutputs("output");

        INDArray inputX = Nd4j.randn(DataType.FLOAT, 3, 4);
        INDArray inputY = Nd4j.randn(DataType.FLOAT, 3, 4);
        INDArray refOut = sd.output(Map.of("x", inputX, "y", inputY), "output").get("output").dup();

        SameDiff optimized = GraphOptimizer.optimize(sd, "output");

        // Sub should be replaced by add, neg should be eliminated
        assertFalse(hasOpOfType(optimized, SubOp.class),
                "sub should be replaced by add");
        assertFalse(hasOpOfType(optimized, Negative.class),
                "neg should be eliminated");
        assertTrue(hasOpOfType(optimized, AddOp.class),
                "add should be present");

        String actualOutput = optimized.outputs().get(0);
        INDArray optOut = optimized.output(Map.of("x", inputX, "y", inputY), actualOutput).get(actualOutput);
        double diff = refOut.sub(optOut).amaxNumber().doubleValue();
        assertTrue(diff < 1e-5, "sub(x, neg(y)) diff = " + diff);
    }

    @Test
    void testSubWithoutNeg_Preserved() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
        SDVariable y = sd.placeHolder("y", DataType.FLOAT, 3, 4);
        SDVariable out = x.sub("output", y);
        sd.setOutputs("output");

        SameDiff optimized = GraphOptimizer.optimize(sd, "output");

        assertTrue(hasOpOfType(optimized, SubOp.class),
                "sub(x, y) without neg should be preserved");
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
