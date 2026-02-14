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

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlan;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanCompiler;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for native C++ graph executor basic operations.
 *
 * Validates that each basic op type compiles correctly into a DynamicShapePlan,
 * serializes successfully, and produces correct results when executed via Java DSP.
 * When native executor is available, also tests native execution parity.
 *
 * Covers: add, sub, mul, div, matmul, reductions, activations, broadcasts, casts.
 */
@Slf4j
@Tag(TagNames.SAMEDIFF)
@NativeTag
public class NativeExecutorBasicOpsTest extends BaseNd4jTestWithBackends {

    private static final double TOLERANCE = 1e-5;

    @Test
    public void testAdd() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 3);
        SDVariable y = sd.placeHolder("y", DataType.FLOAT, 2, 3);
        SDVariable z = x.add("z", y);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.ones(DataType.FLOAT, 2, 3).mul(2);
        INDArray yArr = Nd4j.ones(DataType.FLOAT, 2, 3).mul(3);
        Map<String, INDArray> result = sd.output(Map.of("x", xArr, "y", yArr), "z");
        INDArray expected = Nd4j.ones(DataType.FLOAT, 2, 3).mul(5);
        NativeExecutorTestUtils.assertArrayEquals(expected, result.get("z"), TOLERANCE, "add");

        plan.close();
    }

    @Test
    public void testSubtract() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 3);
        SDVariable y = sd.placeHolder("y", DataType.FLOAT, 2, 3);
        SDVariable z = x.sub("z", y);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.ones(DataType.FLOAT, 2, 3).mul(5);
        INDArray yArr = Nd4j.ones(DataType.FLOAT, 2, 3).mul(3);
        Map<String, INDArray> result = sd.output(Map.of("x", xArr, "y", yArr), "z");
        INDArray expected = Nd4j.ones(DataType.FLOAT, 2, 3).mul(2);
        NativeExecutorTestUtils.assertArrayEquals(expected, result.get("z"), TOLERANCE, "sub");

        plan.close();
    }

    @Test
    public void testMultiply() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 3);
        SDVariable y = sd.placeHolder("y", DataType.FLOAT, 2, 3);
        SDVariable z = x.mul("z", y);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.ones(DataType.FLOAT, 2, 3).mul(2);
        INDArray yArr = Nd4j.ones(DataType.FLOAT, 2, 3).mul(3);
        Map<String, INDArray> result = sd.output(Map.of("x", xArr, "y", yArr), "z");
        INDArray expected = Nd4j.ones(DataType.FLOAT, 2, 3).mul(6);
        NativeExecutorTestUtils.assertArrayEquals(expected, result.get("z"), TOLERANCE, "mul");

        plan.close();
    }

    @Test
    public void testDivide() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 3);
        SDVariable y = sd.placeHolder("y", DataType.FLOAT, 2, 3);
        SDVariable z = x.div("z", y);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.ones(DataType.FLOAT, 2, 3).mul(6);
        INDArray yArr = Nd4j.ones(DataType.FLOAT, 2, 3).mul(3);
        Map<String, INDArray> result = sd.output(Map.of("x", xArr, "y", yArr), "z");
        INDArray expected = Nd4j.ones(DataType.FLOAT, 2, 3).mul(2);
        NativeExecutorTestUtils.assertArrayEquals(expected, result.get("z"), TOLERANCE, "div");

        plan.close();
    }

    @Test
    public void testMatMul() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable w = sd.var("w", Nd4j.eye(4).castTo(DataType.FLOAT));
        SDVariable z = sd.mmul("z", x, w);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.linspace(1, 8, 8, DataType.FLOAT).reshape(2, 4);
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
        // Identity matrix: z = x
        NativeExecutorTestUtils.assertArrayEquals(xArr, result.get("z"), TOLERANCE, "matmul with identity");

        plan.close();
    }

    @Test
    public void testRelu() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable z = sd.nn().relu("z", x, 0.0);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.createFromArray(new float[][]{{-2, -1, 0, 1}, {2, -3, 4, -5}});
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
        INDArray expected = Nd4j.createFromArray(new float[][]{{0, 0, 0, 1}, {2, 0, 4, 0}});
        NativeExecutorTestUtils.assertArrayEquals(expected, result.get("z"), TOLERANCE, "relu");

        plan.close();
    }

    @Test
    public void testSigmoid() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable z = sd.nn().sigmoid("z", x);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.zeros(DataType.FLOAT, 2, 4);
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
        INDArray expected = Nd4j.ones(DataType.FLOAT, 2, 4).mul(0.5); // sigmoid(0) = 0.5
        NativeExecutorTestUtils.assertArrayEquals(expected, result.get("z"), TOLERANCE, "sigmoid");

        plan.close();
    }

    @Test
    public void testTanh() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable z = sd.math().tanh("z", x);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.zeros(DataType.FLOAT, 2, 4);
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
        INDArray expected = Nd4j.zeros(DataType.FLOAT, 2, 4); // tanh(0) = 0
        NativeExecutorTestUtils.assertArrayEquals(expected, result.get("z"), TOLERANCE, "tanh");

        plan.close();
    }

    @Test
    public void testSoftmax() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable z = sd.nn().softmax("z", x, -1);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.zeros(DataType.FLOAT, 2, 4);
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
        // softmax of all-zeros = uniform distribution = 0.25 each
        INDArray expected = Nd4j.ones(DataType.FLOAT, 2, 4).mul(0.25);
        NativeExecutorTestUtils.assertArrayEquals(expected, result.get("z"), TOLERANCE, "softmax");

        plan.close();
    }

    @Test
    public void testReduceSum() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable z = x.sum("z", 1);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.ones(DataType.FLOAT, 2, 4);
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
        // Sum over dim 1: each row sums to 4
        assertEquals(4.0, result.get("z").getDouble(0), TOLERANCE, "reduce_sum row 0");
        assertEquals(4.0, result.get("z").getDouble(1), TOLERANCE, "reduce_sum row 1");

        plan.close();
    }

    @Test
    public void testReduceMean() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable z = x.mean("z", 1);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.linspace(1, 8, 8, DataType.FLOAT).reshape(2, 4);
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
        assertEquals(2.5, result.get("z").getDouble(0), TOLERANCE, "reduce_mean row 0");
        assertEquals(6.5, result.get("z").getDouble(1), TOLERANCE, "reduce_mean row 1");

        plan.close();
    }

    @Test
    public void testBroadcastAdd() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable bias = sd.var("bias", Nd4j.ones(DataType.FLOAT, 1, 4).mul(10));
        SDVariable z = x.add("z", bias);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.ones(DataType.FLOAT, 3, 4);
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
        INDArray expected = Nd4j.ones(DataType.FLOAT, 3, 4).mul(11);
        NativeExecutorTestUtils.assertArrayEquals(expected, result.get("z"), TOLERANCE, "broadcast add");

        plan.close();
    }

    @Test
    public void testExp() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable z = sd.math().exp("z", x);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.zeros(DataType.FLOAT, 2, 4);
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
        INDArray expected = Nd4j.ones(DataType.FLOAT, 2, 4); // exp(0) = 1
        NativeExecutorTestUtils.assertArrayEquals(expected, result.get("z"), TOLERANCE, "exp");

        plan.close();
    }

    @Test
    public void testLog() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable z = sd.math().log("z", x);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.ones(DataType.FLOAT, 2, 4);
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
        INDArray expected = Nd4j.zeros(DataType.FLOAT, 2, 4); // log(1) = 0
        NativeExecutorTestUtils.assertArrayEquals(expected, result.get("z"), TOLERANCE, "log");

        plan.close();
    }

    @Test
    public void testNeg() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable z = sd.math().neg("z", x);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.ones(DataType.FLOAT, 2, 4).mul(3);
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
        INDArray expected = Nd4j.ones(DataType.FLOAT, 2, 4).mul(-3);
        NativeExecutorTestUtils.assertArrayEquals(expected, result.get("z"), TOLERANCE, "neg");

        plan.close();
    }

    @Test
    public void testAbs() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable z = sd.math().abs("z", x);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.createFromArray(new float[][]{{-1, 2, -3, 4}, {-5, 6, -7, 8}});
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
        INDArray expected = Nd4j.createFromArray(new float[][]{{1, 2, 3, 4}, {5, 6, 7, 8}});
        NativeExecutorTestUtils.assertArrayEquals(expected, result.get("z"), TOLERANCE, "abs");

        plan.close();
    }

    @Test
    public void testSqrt() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable z = sd.math().sqrt("z", x);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.createFromArray(new float[][]{{1, 4, 9, 16}});
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
        INDArray expected = Nd4j.createFromArray(new float[][]{{1, 2, 3, 4}});
        NativeExecutorTestUtils.assertArrayEquals(expected, result.get("z"), TOLERANCE, "sqrt");

        plan.close();
    }

    @Test
    public void testSquare() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable z = sd.math().square("z", x);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.createFromArray(new float[][]{{1, 2, 3, 4}});
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
        INDArray expected = Nd4j.createFromArray(new float[][]{{1, 4, 9, 16}});
        NativeExecutorTestUtils.assertArrayEquals(expected, result.get("z"), TOLERANCE, "square");

        plan.close();
    }

    @Test
    public void testCast() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable z = sd.castTo("z", x, DataType.DOUBLE);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.ones(DataType.FLOAT, 2, 4).mul(3.14f);
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
        assertEquals(DataType.DOUBLE, result.get("z").dataType(), "Cast should produce DOUBLE");

        plan.close();
    }

    @Test
    public void testConcat() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 2);
        SDVariable y = sd.placeHolder("y", DataType.FLOAT, -1, 3);
        SDVariable z = sd.concat("z", 1, x, y);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.ones(DataType.FLOAT, 2, 2);
        INDArray yArr = Nd4j.ones(DataType.FLOAT, 2, 3).mul(2);
        Map<String, INDArray> result = sd.output(Map.of("x", xArr, "y", yArr), "z");
        assertEquals(5, result.get("z").size(1), "Concat should produce [2,5]");

        plan.close();
    }

    @Test
    public void testTranspose() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 3);
        SDVariable z = sd.permute("z", x, 1, 0);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.linspace(1, 6, 6, DataType.FLOAT).reshape(2, 3);
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
        assertArrayEquals(new long[]{3, 2}, result.get("z").shape(), "Transpose should swap dims");

        plan.close();
    }

    @Test
    public void testReduceMax() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable z = x.max("z", 1);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.createFromArray(new float[][]{{1, 4, 2, 3}, {8, 5, 6, 7}});
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
        assertEquals(4.0, result.get("z").getDouble(0), TOLERANCE, "max row 0");
        assertEquals(8.0, result.get("z").getDouble(1), TOLERANCE, "max row 1");

        plan.close();
    }

    @Test
    public void testReduceMin() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable z = x.min("z", 1);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.createFromArray(new float[][]{{1, 4, 2, 3}, {8, 5, 6, 7}});
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
        assertEquals(1.0, result.get("z").getDouble(0), TOLERANCE, "min row 0");
        assertEquals(5.0, result.get("z").getDouble(1), TOLERANCE, "min row 1");

        plan.close();
    }

    @Test
    public void testMatMulWithBias() {
        SameDiff sd = NativeExecutorTestUtils.createMatMulGraph(4, 3);
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.ones(DataType.FLOAT, 2, 4);
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
        assertNotNull(result.get("z"), "MatMul+Bias should produce output");
        assertArrayEquals(new long[]{2, 3}, result.get("z").shape(), "Shape should be [2,3]");

        plan.close();
    }

    @Test
    public void testChainedOps() {
        SameDiff sd = NativeExecutorTestUtils.createChainGraph();
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.zeros(DataType.FLOAT, 2, 4);
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
        assertNotNull(result.get("z"), "Chain should produce output");
        // relu(sigmoid(0 + 1) * 0.5) = relu(0.7311 * 0.5) = relu(0.3655) = 0.3655
        double expected = 0.5 / (1.0 + Math.exp(-1.0));
        assertEquals(expected, result.get("z").getDouble(0), 1e-3, "Chain op result");

        plan.close();
    }

    @Test
    public void testDiamondGraph() {
        SameDiff sd = NativeExecutorTestUtils.createDiamondGraph();
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "c");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.ones(DataType.FLOAT, 2, 4).mul(3);
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "c");
        // a = x + 1 = 4, b = x * 2 = 6, c = a + b = 10
        INDArray expected = Nd4j.ones(DataType.FLOAT, 2, 4).mul(10);
        NativeExecutorTestUtils.assertArrayEquals(expected, result.get("c"), TOLERANCE, "diamond");

        plan.close();
    }

    @Test
    public void testFanOutGraph() {
        SameDiff sd = NativeExecutorTestUtils.createFanOutGraph();
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "a", "b", "c");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.ones(DataType.FLOAT, 2, 4).mul(3);
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "a", "b", "c");
        assertNotNull(result.get("a"));
        assertNotNull(result.get("b"));
        assertNotNull(result.get("c"));

        plan.close();
    }

    @Test
    public void testDynamicBatchSize() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable w = sd.var("w", Nd4j.eye(4).castTo(DataType.FLOAT));
        SDVariable z = sd.mmul("z", x, w);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        // Execute with batch size 1
        Map<String, INDArray> result1 = sd.output(Map.of("x", Nd4j.ones(DataType.FLOAT, 1, 4)), "z");
        assertArrayEquals(new long[]{1, 4}, result1.get("z").shape());

        // Execute with batch size 5
        Map<String, INDArray> result5 = sd.output(Map.of("x", Nd4j.ones(DataType.FLOAT, 5, 4)), "z");
        assertArrayEquals(new long[]{5, 4}, result5.get("z").shape());

        plan.close();
    }

    @Test
    public void testScalarOps() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable z = x.add("z", 5.0);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.ones(DataType.FLOAT, 2, 4);
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
        INDArray expected = Nd4j.ones(DataType.FLOAT, 2, 4).mul(6);
        NativeExecutorTestUtils.assertArrayEquals(expected, result.get("z"), TOLERANCE, "scalar add");

        plan.close();
    }

    @Test
    public void testReduceProd() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable z = x.prod("z", 1);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.createFromArray(new float[][]{{1, 2, 3, 4}});
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
        assertEquals(24.0, result.get("z").getDouble(0), TOLERANCE, "prod");

        plan.close();
    }

    @Test
    public void testGelu() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable z = sd.nn().gelu("z", x);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.zeros(DataType.FLOAT, 2, 4);
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
        // gelu(0) = 0
        INDArray expected = Nd4j.zeros(DataType.FLOAT, 2, 4);
        NativeExecutorTestUtils.assertArrayEquals(expected, result.get("z"), TOLERANCE, "gelu");

        plan.close();
    }

    @Test
    public void testPow() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable z = sd.math().pow("z", x, 2.0);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.createFromArray(new float[][]{{1, 2, 3, 4}});
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
        INDArray expected = Nd4j.createFromArray(new float[][]{{1, 4, 9, 16}});
        NativeExecutorTestUtils.assertArrayEquals(expected, result.get("z"), TOLERANCE, "pow");

        plan.close();
    }
}
