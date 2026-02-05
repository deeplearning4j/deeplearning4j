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
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.optimize.GraphOptimizer;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.AddOp;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.MulOp;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.SubOp;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.DivOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.nio.file.Path;
import java.util.Collections;
import java.util.Map;

import static org.junit.Assert.*;

/**
 * Tests for AlgebraicOptimizations - algebraic simplifications like x+0->x, x*1->x, etc.
 */
@Tag(TagNames.DL4J_OLD_API)
public class TestAlgebraicOptimizations extends BaseNd4jTestWithBackends {

    @TempDir
    Path tempDir;

    @Override
    public char ordering() {
        return 'c';
    }

    @Override
    public long getTimeoutMilliseconds() {
        return 1_000_000_000L;
    }

    /**
     * Test x + 0 -> x optimization
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAddZero(Nd4jBackend nd4jBackend) {
        SameDiff sd = SameDiff.create();

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable zero = sd.constant("zero", Nd4j.zeros(DataType.FLOAT, 1));

        // x + 0 should be simplified to x
        SDVariable added = x.add(zero);
        SDVariable out = sd.nn.softmax("out", added);

        Map<String, INDArray> ph = Collections.singletonMap("x", Nd4j.rand(DataType.FLOAT, 2, 4));
        INDArray expected = sd.outputSingle(ph, "out");

        SameDiff optimized = GraphOptimizer.optimize(sd, "out");

        // The add(x, 0) should be removed
        int addCount = 0;
        for (String opName : optimized.getOps().keySet()) {
            if (optimized.getOps().get(opName).getOp() instanceof AddOp) {
                addCount++;
            }
        }
        assertEquals("Add with zero should be removed", 0, addCount);

        INDArray actual = optimized.outputSingle(ph, "out");
        assertEquals("Output should match after optimization", expected, actual);
    }

    /**
     * Test 0 + x -> x optimization (commutativity)
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testZeroAddX(Nd4jBackend nd4jBackend) {
        SameDiff sd = SameDiff.create();

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable zero = sd.constant("zero", Nd4j.zeros(DataType.FLOAT, 1));

        // 0 + x should be simplified to x
        SDVariable added = zero.add(x);
        SDVariable out = sd.nn.softmax("out", added);

        Map<String, INDArray> ph = Collections.singletonMap("x", Nd4j.rand(DataType.FLOAT, 2, 4));
        INDArray expected = sd.outputSingle(ph, "out");

        SameDiff optimized = GraphOptimizer.optimize(sd, "out");

        INDArray actual = optimized.outputSingle(ph, "out");
        assertEquals("Output should match after optimization", expected, actual);
    }

    /**
     * Test x - 0 -> x optimization
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSubtractZero(Nd4jBackend nd4jBackend) {
        SameDiff sd = SameDiff.create();

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable zero = sd.constant("zero", Nd4j.zeros(DataType.FLOAT, 1));

        // x - 0 should be simplified to x
        SDVariable subtracted = x.sub(zero);
        SDVariable out = sd.nn.softmax("out", subtracted);

        Map<String, INDArray> ph = Collections.singletonMap("x", Nd4j.rand(DataType.FLOAT, 2, 4));
        INDArray expected = sd.outputSingle(ph, "out");

        SameDiff optimized = GraphOptimizer.optimize(sd, "out");

        // The sub(x, 0) should be removed
        int subCount = 0;
        for (String opName : optimized.getOps().keySet()) {
            if (optimized.getOps().get(opName).getOp() instanceof SubOp) {
                subCount++;
            }
        }
        assertEquals("Subtract zero should be removed", 0, subCount);

        INDArray actual = optimized.outputSingle(ph, "out");
        assertEquals("Output should match after optimization", expected, actual);
    }

    /**
     * Test x * 1 -> x optimization
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMultiplyOne(Nd4jBackend nd4jBackend) {
        SameDiff sd = SameDiff.create();

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable one = sd.constant("one", Nd4j.ones(DataType.FLOAT, 1));

        // x * 1 should be simplified to x
        SDVariable multiplied = x.mul(one);
        SDVariable out = sd.nn.softmax("out", multiplied);

        Map<String, INDArray> ph = Collections.singletonMap("x", Nd4j.rand(DataType.FLOAT, 2, 4));
        INDArray expected = sd.outputSingle(ph, "out");

        SameDiff optimized = GraphOptimizer.optimize(sd, "out");

        // The mul(x, 1) should be removed
        int mulCount = 0;
        for (String opName : optimized.getOps().keySet()) {
            if (optimized.getOps().get(opName).getOp() instanceof MulOp) {
                mulCount++;
            }
        }
        assertEquals("Multiply by one should be removed", 0, mulCount);

        INDArray actual = optimized.outputSingle(ph, "out");
        assertEquals("Output should match after optimization", expected, actual);
    }

    /**
     * Test x * 0 -> 0 optimization
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMultiplyZero(Nd4jBackend nd4jBackend) {
        SameDiff sd = SameDiff.create();

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable zero = sd.constant("zero", Nd4j.zeros(DataType.FLOAT, 1));

        // x * 0 should be simplified to 0
        SDVariable multiplied = x.mul(zero);
        // Add another operation to avoid the whole graph being constant
        SDVariable other = sd.placeHolder("other", DataType.FLOAT, -1, 4);
        SDVariable out = sd.nn.softmax("out", multiplied.add(other));

        Map<String, INDArray> ph = Map.of(
            "x", Nd4j.rand(DataType.FLOAT, 2, 4),
            "other", Nd4j.rand(DataType.FLOAT, 2, 4)
        );
        INDArray expected = sd.outputSingle(ph, "out");

        SameDiff optimized = GraphOptimizer.optimize(sd, "out");

        INDArray actual = optimized.outputSingle(ph, "out");
        assertEquals("Output should match after optimization", expected, actual);
    }

    /**
     * Test x / 1 -> x optimization
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDivideOne(Nd4jBackend nd4jBackend) {
        SameDiff sd = SameDiff.create();

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable one = sd.constant("one", Nd4j.ones(DataType.FLOAT, 1));

        // x / 1 should be simplified to x
        SDVariable divided = x.div(one);
        SDVariable out = sd.nn.softmax("out", divided);

        Map<String, INDArray> ph = Collections.singletonMap("x", Nd4j.rand(DataType.FLOAT, 2, 4));
        INDArray expected = sd.outputSingle(ph, "out");

        SameDiff optimized = GraphOptimizer.optimize(sd, "out");

        // The div(x, 1) should be removed
        int divCount = 0;
        for (String opName : optimized.getOps().keySet()) {
            if (optimized.getOps().get(opName).getOp() instanceof DivOp) {
                divCount++;
            }
        }
        assertEquals("Divide by one should be removed", 0, divCount);

        INDArray actual = optimized.outputSingle(ph, "out");
        assertEquals("Output should match after optimization", expected, actual);
    }

    /**
     * Test that non-identity operations are not incorrectly simplified
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNoIncorrectSimplification(Nd4jBackend nd4jBackend) {
        SameDiff sd = SameDiff.create();

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable two = sd.constant("two", Nd4j.scalar(DataType.FLOAT, 2.0f));

        // x + 2, x * 2 should NOT be simplified
        SDVariable added = x.add(two);
        SDVariable multiplied = x.mul(two);
        SDVariable out = sd.nn.softmax("out", added.add(multiplied));

        Map<String, INDArray> ph = Collections.singletonMap("x", Nd4j.rand(DataType.FLOAT, 2, 4));
        INDArray expected = sd.outputSingle(ph, "out");

        SameDiff optimized = GraphOptimizer.optimize(sd, "out");

        INDArray actual = optimized.outputSingle(ph, "out");
        assertEquals("Output should match after optimization", expected, actual);
    }

    /**
     * Test chained algebraic simplifications
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testChainedSimplifications(Nd4jBackend nd4jBackend) {
        SameDiff sd = SameDiff.create();

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable zero = sd.constant("zero", Nd4j.zeros(DataType.FLOAT, 1));
        SDVariable one = sd.constant("one", Nd4j.ones(DataType.FLOAT, 1));

        // ((x + 0) * 1) / 1 should simplify to x
        SDVariable step1 = x.add(zero);
        SDVariable step2 = step1.mul(one);
        SDVariable step3 = step2.div(one);
        SDVariable out = sd.nn.softmax("out", step3);

        Map<String, INDArray> ph = Collections.singletonMap("x", Nd4j.rand(DataType.FLOAT, 2, 4));
        INDArray expected = sd.outputSingle(ph, "out");

        SameDiff optimized = GraphOptimizer.optimize(sd, "out");

        // All identity operations should be removed
        int identityOpCount = 0;
        for (String opName : optimized.getOps().keySet()) {
            var op = optimized.getOps().get(opName).getOp();
            if (op instanceof AddOp || op instanceof MulOp || op instanceof DivOp) {
                identityOpCount++;
            }
        }
        assertEquals("All identity ops should be removed", 0, identityOpCount);

        INDArray actual = optimized.outputSingle(ph, "out");
        assertEquals("Output should match after optimization", expected, actual);
    }
}
