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

import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.ControlFlow;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for SameDiff ControlFlow operations (ifCond, whileLoop, etc.)
 * These tests verify the behavior of conditional execution in SameDiff graphs.
 */
public class ControlFlowTest {

    @Test
    public void testBasicSameDiffExec() throws Exception {
        // First verify basic SameDiff execution works without control flow
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 3);
        SDVariable result = sd.math.mul("result", x, sd.constant(2.0f));

        INDArray input = Nd4j.ones(2, 3);
        Map<String, INDArray> out = sd.output(Map.of("x", input), "result");

        INDArray resultArr = out.get("result");
        assertNotNull(resultArr, "Result should not be null");
        assertFalse(resultArr.wasClosed(), "Result should not be closed");

        INDArray expected = Nd4j.ones(2, 3).mul(2.0f);
        assertEquals(expected, resultArr);
    }

    @Test
    public void testIfCondTrueBranch() {
        SameDiff sd = SameDiff.create();

        // Create a boolean condition that is true
        SDVariable cond = sd.constant("cond", Nd4j.scalar(true));

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 3);

        // If true, multiply by 2; if false, multiply by 10
        SDVariable result = ControlFlow.ifCond(
            sd,
            "result",
            "test_if",
            s -> cond,
            s -> s.math.mul("true_branch", x, sd.constant(2.0f)),
            s -> s.math.mul("false_branch", x, sd.constant(10.0f))
        );

        INDArray input = Nd4j.ones(2, 3);
        Map<String, INDArray> out = sd.output(Map.of("x", input), "result");

        INDArray expected = Nd4j.ones(2, 3).mul(2.0f);
        INDArray actual = out.get("result").dup();
        assertEquals(expected, actual);
    }

    @Test
    public void testIfCondFalseBranch() {
        SameDiff sd = SameDiff.create();

        // Create a boolean condition that is false
        SDVariable cond = sd.constant("cond", Nd4j.scalar(false));

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 3);

        // If true, multiply by 2; if false, multiply by 10
        SDVariable result = ControlFlow.ifCond(
            sd,
            "result",
            "test_if",
            s -> cond,
            s -> s.math.mul("true_branch", x, sd.constant(2.0f)),
            s -> s.math.mul("false_branch", x, sd.constant(10.0f))
        );

        INDArray input = Nd4j.ones(2, 3);
        Map<String, INDArray> out = sd.output(Map.of("x", input), "result");

        INDArray expected = Nd4j.ones(2, 3).mul(10.0f);
        INDArray actual = out.get("result").dup(); // dup to prevent closed array issues
        assertEquals(expected, actual);
    }

    @Test
    public void testIfCondDynamicCondition() {
        SameDiff sd = SameDiff.create();

        // Create a placeholder for the condition
        SDVariable condInput = sd.placeHolder("condInput", DataType.FLOAT);
        SDVariable threshold = sd.constant("threshold", Nd4j.scalar(5.0f));

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 3);

        // Condition: condInput > threshold
        SDVariable cond = condInput.gt(threshold);

        // If condInput > 5, multiply by 2; otherwise multiply by 10
        SDVariable result = ControlFlow.ifCond(
            sd,
            "result",
            "test_if",
            s -> cond,
            s -> s.math.mul("true_branch", x, sd.constant(2.0f)),
            s -> s.math.mul("false_branch", x, sd.constant(10.0f))
        );

        INDArray input = Nd4j.ones(2, 3);

        // Test with condInput = 10 (> 5, so true branch)
        Map<String, INDArray> out1 = sd.output(
            Map.of("x", input, "condInput", Nd4j.scalar(10.0f)),
            "result"
        );
        INDArray expected1 = Nd4j.ones(2, 3).mul(2.0f);
        INDArray actual1 = out1.get("result").dup(); // dup to prevent closed array issues
        assertEquals(expected1, actual1, "Should take true branch when condInput > 5");

        // Test with condInput = 3 (< 5, so false branch)
        Map<String, INDArray> out2 = sd.output(
            Map.of("x", input, "condInput", Nd4j.scalar(3.0f)),
            "result"
        );
        INDArray expected2 = Nd4j.ones(2, 3).mul(10.0f);
        INDArray actual2 = out2.get("result").dup(); // dup to prevent closed array issues
        assertEquals(expected2, actual2, "Should take false branch when condInput < 5");
    }

    @Test
    public void testIfCondWithRankCheck() {
        // This test simulates the ReduceMean use case where we check rank at runtime
        SameDiff sd = SameDiff.create();

        // Test with 4D input (should reduce spatial dims)
        SDVariable input4d = sd.placeHolder("input", DataType.FLOAT, -1, -1, -1, -1);

        SDVariable rankVar = sd.rank(input4d).castTo(DataType.INT32); // Cast to INT32
        SDVariable four = sd.constant("four", Nd4j.scalar(DataType.INT32, 4));
        SDVariable isRank4 = rankVar.eq(four);

        // If rank is 4, reduce dims [2,3] (spatial); otherwise reduce all
        SDVariable result = ControlFlow.ifCond(
            sd,
            "result",
            "rank_check",
            s -> isRank4,
            s -> s.mean("spatial_mean", input4d, true, 2, 3),
            s -> s.mean("all_mean", input4d, true)
        );

        // Test with 4D input [1, 640, 7, 7] - should reduce spatial dims to [1, 640, 1, 1]
        INDArray input = Nd4j.ones(1, 640, 7, 7);
        Map<String, INDArray> out = sd.output(Map.of("input", input), "result");

        INDArray output = out.get("result").dup(); // dup to prevent closed array issues
        System.out.println("Input shape: " + java.util.Arrays.toString(input.shape()));
        System.out.println("Output shape: " + java.util.Arrays.toString(output.shape()));
        System.out.println("Output value: " + output);

        // For 4D input, should keep dims and produce [1, 640, 1, 1]
        assertArrayEquals(new long[]{1, 640, 1, 1}, output.shape(),
            "4D input should reduce to [1, 640, 1, 1] with keepDims=true");
    }

    @Test
    public void testIfCondDifferentOutputShapes() {
        // Test that ifCond can handle branches with different output shapes
        // This is important for the ReduceMean case
        SameDiff sd = SameDiff.create();

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, -1, -1, -1);
        SDVariable condInput = sd.placeHolder("cond", DataType.BOOL);

        // True branch: mean over axes [2,3] with keepDims - shape [N, C, 1, 1]
        // False branch: mean over all axes with keepDims - shape [1, 1, 1, 1]
        SDVariable result = ControlFlow.ifCond(
            sd,
            "result",
            "shape_test",
            s -> condInput,
            s -> s.mean("spatial", input, true, 2, 3),
            s -> s.mean("all", input, true)
        );

        INDArray inputData = Nd4j.ones(1, 640, 7, 7);

        // Test true branch
        Map<String, INDArray> out1 = sd.output(
            Map.of("input", inputData, "cond", Nd4j.scalar(true)),
            "result"
        );
        System.out.println("True branch output shape: " + java.util.Arrays.toString(out1.get("result").shape()));

        // Test false branch
        Map<String, INDArray> out2 = sd.output(
            Map.of("input", inputData, "cond", Nd4j.scalar(false)),
            "result"
        );
        System.out.println("False branch output shape: " + java.util.Arrays.toString(out2.get("result").shape()));
    }

    @Test
    public void testMeanWithDynamicAxesSimulation() {
        // Simpler test: just verify that mean works with different axis configurations
        SameDiff sd = SameDiff.create();

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 640, 7, 7);

        // Spatial mean (axes 2, 3)
        SDVariable spatialMean = sd.mean("spatial_mean", input, true, 2, 3);

        // Global mean (all axes)
        SDVariable globalMean = sd.mean("global_mean", input, true);

        INDArray inputData = Nd4j.ones(1, 640, 7, 7);
        Map<String, INDArray> out = sd.output(Map.of("input", inputData), "spatial_mean", "global_mean");

        System.out.println("Spatial mean shape: " + java.util.Arrays.toString(out.get("spatial_mean").shape()));
        System.out.println("Global mean shape: " + java.util.Arrays.toString(out.get("global_mean").shape()));

        assertArrayEquals(new long[]{1, 640, 1, 1}, out.get("spatial_mean").shape());
        assertArrayEquals(new long[]{1, 1, 1, 1}, out.get("global_mean").shape());
    }
}
