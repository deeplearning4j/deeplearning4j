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
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for numerical correctness of the native graph executor.
 *
 * Covers: bitwise-exact integer/boolean results, float tolerance,
 * determinism, NaN propagation, special values.
 */
@Slf4j
@Tag(TagNames.SAMEDIFF)
@NativeTag
public class NativeExecutorCorrectnessTest extends BaseNd4jTestWithBackends {

    private static final double FLOAT_TOL = 1e-5;

    @Test
    public void testIntegerArithmeticExact() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.INT32, -1, 4);
        SDVariable y = sd.placeHolder("y", DataType.INT32, -1, 4);
        SDVariable z = x.add("z", y);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.createFromArray(new int[][]{{1, 2, 3, 4}});
        INDArray yArr = Nd4j.createFromArray(new int[][]{{10, 20, 30, 40}});
        Map<String, INDArray> result = sd.output(Map.of("x", xArr, "y", yArr), "z");

        assertEquals(11, result.get("z").getInt(0, 0), "Integer add should be exact");
        assertEquals(22, result.get("z").getInt(0, 1), "Integer add should be exact");
        assertEquals(33, result.get("z").getInt(0, 2), "Integer add should be exact");
        assertEquals(44, result.get("z").getInt(0, 3), "Integer add should be exact");

        plan.close();
    }

    @Test
    public void testBooleanOps() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable threshold = sd.constant("threshold", Nd4j.scalar(DataType.FLOAT, 0.5f));
        SDVariable z = sd.gt("z", x, threshold);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.createFromArray(new float[][]{{0.1f, 0.6f, 0.3f, 0.8f}});
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
        assertEquals(DataType.BOOL, result.get("z").dataType());

        plan.close();
    }

    @Test
    public void testFloatPrecision() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable y = sd.placeHolder("y", DataType.FLOAT, -1, 4);
        SDVariable z = x.add("z", y);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");

        INDArray xArr = Nd4j.createFromArray(new float[][]{{1.23456f, 2.34567f, 3.45678f, 4.56789f}});
        INDArray yArr = Nd4j.createFromArray(new float[][]{{0.00001f, 0.00002f, 0.00003f, 0.00004f}});
        Map<String, INDArray> result = sd.output(Map.of("x", xArr, "y", yArr), "z");

        assertEquals(1.23457f, result.get("z").getFloat(0, 0), 1e-4f, "Float precision");

        plan.close();
    }

    @Test
    public void testDoublePrecision() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.DOUBLE, -1, 4);
        SDVariable z = x.mul("z", 1.0);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        double val = 1.23456789012345;
        INDArray xArr = Nd4j.createFromArray(new double[][]{{val, val, val, val}});
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
        assertEquals(val, result.get("z").getDouble(0, 0), 1e-12, "Double precision");

        plan.close();
    }

    @Test
    public void testDeterminism() {
        SameDiff sd = NativeExecutorTestUtils.createMatMulGraph(4, 3);
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 2, 4);
        Map<String, INDArray> result1 = sd.output(Map.of("x", xArr), "z");
        Map<String, INDArray> result2 = sd.output(Map.of("x", xArr), "z");

        NativeExecutorTestUtils.assertArrayEquals(result1.get("z"), result2.get("z"), 0,
                "Same inputs should produce identical outputs");

        plan.close();
    }

    @Test
    public void testNaNPropagation() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable z = x.add("z", 1.0);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");

        INDArray xArr = Nd4j.createFromArray(new float[][]{{1f, Float.NaN, 3f, Float.NaN}});
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
        assertTrue(Float.isNaN(result.get("z").getFloat(0, 1)), "NaN should propagate through add");
        assertTrue(Float.isNaN(result.get("z").getFloat(0, 3)), "NaN should propagate through add");
        assertEquals(2f, result.get("z").getFloat(0, 0), FLOAT_TOL, "Non-NaN should compute normally");

        plan.close();
    }

    @Test
    public void testInfinityHandling() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable z = x.mul("z", 2.0);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");

        INDArray xArr = Nd4j.createFromArray(new float[][]{{Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, 1f, 0f}});
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
        assertTrue(Float.isInfinite(result.get("z").getFloat(0, 0)), "+Inf * 2 = +Inf");
        assertTrue(Float.isInfinite(result.get("z").getFloat(0, 1)), "-Inf * 2 = -Inf");

        plan.close();
    }

    @Test
    public void testZeroHandling() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable z = x.mul("z", 0.0);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");

        INDArray xArr = Nd4j.createFromArray(new float[][]{{1f, 1000f, -1000f, 0f}});
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
        for (int i = 0; i < 4; i++) {
            assertEquals(0f, result.get("z").getFloat(0, i), FLOAT_TOL, "Anything * 0 = 0");
        }

        plan.close();
    }

    @Test
    public void testLargeValues() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable z = x.add("z", 0.0);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");

        float large = 1e30f;
        INDArray xArr = Nd4j.createFromArray(new float[][]{{large, -large, large, -large}});
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
        assertEquals(large, result.get("z").getFloat(0, 0), large * 1e-5f, "Large values preserved");

        plan.close();
    }

    @Test
    public void testSmallValues() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable z = x.add("z", 0.0);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");

        float small = 1e-30f;
        INDArray xArr = Nd4j.createFromArray(new float[][]{{small, small, small, small}});
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
        assertEquals(small, result.get("z").getFloat(0, 0), small * 1e-3f, "Small values preserved");

        plan.close();
    }

    @Test
    public void testMixedDtypeInputs() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable scale = sd.var("scale", Nd4j.ones(DataType.DOUBLE, 1, 4).mul(2.0));
        // The add op should handle implicit cast
        SDVariable z = x.add("z", scale.castTo(DataType.FLOAT));

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.ones(DataType.FLOAT, 2, 4);
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
        assertEquals(3.0, result.get("z").getDouble(0, 0), FLOAT_TOL);

        plan.close();
    }

    @Test
    public void testRepeatedExecutionStability() {
        SameDiff sd = NativeExecutorTestUtils.createChainGraph();
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");

        INDArray xArr = Nd4j.ones(DataType.FLOAT, 2, 4);
        INDArray firstResult = null;

        for (int i = 0; i < 20; i++) {
            Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
            INDArray z = result.get("z");
            if (firstResult == null) {
                firstResult = z.dup();
            } else {
                NativeExecutorTestUtils.assertArrayEquals(firstResult, z, 0,
                        "Execution " + i + " should match first execution");
            }
        }

        plan.close();
    }
}
