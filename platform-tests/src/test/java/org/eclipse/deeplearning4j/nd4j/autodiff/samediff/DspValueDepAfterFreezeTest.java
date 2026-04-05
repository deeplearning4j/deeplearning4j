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

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests that value-dependent ops (reshape, gather, broadcast_to, etc.) produce
 * correct results after DSP shapes are frozen.
 *
 * The critical scenario: a reshape op whose target shape comes from a placeholder
 * input. After DSP freezes shapes (because shapes have been stable for N steps),
 * the shape key must still be recomputed for segments containing value-dep ops.
 * If the cached shape key is incorrectly reused, the CUDA graph replays with
 * stale output shapes → wrong results or SIGSEGV.
 *
 * This test exercises the fix in NativeDynamicShapePlan_gpubackend.cpp where
 * segments with hasValueDepOps=true always recompute their shape key.
 */
@DisplayName("DSP Value-Dependent Ops After Freeze")
public class DspValueDepAfterFreezeTest extends BaseNd4jTestWithBackends {

    private static final Logger log = LoggerFactory.getLogger(DspValueDepAfterFreezeTest.class);
    private static final double TOL = 1e-4;

    @Override
    public char ordering() {
        return 'c';
    }

    @AfterEach
    public void resetEnvironment() {
        // No global state to reset — DSP enable/disable is per-SameDiff instance
    }

    /**
     * Test: reshape with a variable shape input that stays constant during warmup
     * but changes after freeze.
     *
     * Graph: x[1,12] → reshape(shape_var) → relu → sum → out
     * Warmup: shape_var = [2,6] for 10 steps (triggers freeze)
     * After freeze: shape_var = [3,4] — must produce correct output
     */
    @Test
    @DisplayName("Reshape with changing shape input after freeze")
    public void testReshapeValueDepAfterFreeze() {
        SameDiff sd = SameDiff.create();

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 12);
        SDVariable shapeVar = sd.placeHolder("shape_var", DataType.INT64, 2);
        SDVariable reshaped = sd.reshape("reshaped", x, shapeVar);
        SDVariable activated = sd.nn.relu("activated", reshaped, 0);
        SDVariable out = activated.sum("out");

        // Enable DSP
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        sd.setGraphExecutionMode(GraphExecutionMode.CUDA_GRAPHS);

        INDArray input = Nd4j.linspace(1, 12, 12, DataType.FLOAT).reshape(1, 12);

        // Phase 1: warmup with shape [2,6] — run enough steps to trigger freeze
        INDArray shape2x6 = Nd4j.createFromArray(2L, 6L);
        INDArray lastWarmupOutput = null;
        for (int i = 0; i < 10; i++) {
            Map<String, INDArray> result = sd.output(
                    Map.of("x", input, "shape_var", shape2x6), "out");
            lastWarmupOutput = result.get("out").dup();
        }
        assertNotNull(lastWarmupOutput);

        // Compute expected warmup result: reshape [1,12]->[2,6], relu, sum
        double expectedWarmupSum = 0;
        for (int i = 1; i <= 12; i++) expectedWarmupSum += Math.max(0, i);
        assertEquals(expectedWarmupSum, lastWarmupOutput.getDouble(0), TOL,
                "Warmup output should match manual computation");

        // Phase 2: change shape_var to [3,4] — this tests value-dep detection after freeze
        INDArray shape3x4 = Nd4j.createFromArray(3L, 4L);
        Map<String, INDArray> afterFreezeResult = sd.output(
                Map.of("x", input, "shape_var", shape3x4), "out");
        INDArray afterFreezeOutput = afterFreezeResult.get("out").dup();

        // The sum should be the same (same elements, just different reshape)
        assertEquals(expectedWarmupSum, afterFreezeOutput.getDouble(0), TOL,
                "After-freeze output with shape [3,4] should produce same sum as [2,6]");

        // Phase 3: verify shape [4,3] also works
        INDArray shape4x3 = Nd4j.createFromArray(4L, 3L);
        Map<String, INDArray> shape4x3Result = sd.output(
                Map.of("x", input, "shape_var", shape4x3), "out");
        INDArray shape4x3Output = shape4x3Result.get("out").dup();

        assertEquals(expectedWarmupSum, shape4x3Output.getDouble(0), TOL,
                "Shape [4,3] should also produce correct sum");

        log.info("Value-dep after freeze test passed: warmup={}, afterFreeze={}, shape4x3={}",
                lastWarmupOutput.getDouble(0), afterFreezeOutput.getDouble(0),
                shape4x3Output.getDouble(0));
    }

    /**
     * Test: gather op with changing indices after freeze.
     *
     * Graph: data[4,8] → gather(indices, axis=0) → sum → out
     * Warmup: indices = [0,1] for 10 steps
     * After freeze: indices = [2,3] — must gather different rows
     */
    @Test
    @DisplayName("Gather with changing indices after freeze")
    public void testGatherValueDepAfterFreeze() {
        SameDiff sd = SameDiff.create();

        SDVariable data = sd.placeHolder("data", DataType.FLOAT, 4, 8);
        SDVariable indices = sd.placeHolder("indices", DataType.INT64, 2);
        SDVariable gathered = sd.gather("gathered", data, indices, 0);
        SDVariable out = gathered.sum("out");

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        sd.setGraphExecutionMode(GraphExecutionMode.CUDA_GRAPHS);

        // Data: row i has all values = (i+1)*10
        INDArray dataArr = Nd4j.zeros(DataType.FLOAT, 4, 8);
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 8; c++) {
                dataArr.putScalar(r, c, (r + 1) * 10.0f);
            }
        }

        // Phase 1: warmup with indices [0,1]
        INDArray indices01 = Nd4j.createFromArray(0L, 1L);
        INDArray lastWarmupOutput = null;
        for (int i = 0; i < 10; i++) {
            Map<String, INDArray> result = sd.output(
                    Map.of("data", dataArr, "indices", indices01), "out");
            lastWarmupOutput = result.get("out").dup();
        }
        // Rows 0,1: values 10 and 20, 8 elements each → sum = 8*10 + 8*20 = 240
        double expectedWarmup = 8 * 10.0 + 8 * 20.0;
        assertEquals(expectedWarmup, lastWarmupOutput.getDouble(0), TOL,
                "Warmup gather [0,1] sum");

        // Phase 2: change indices to [2,3] after freeze
        INDArray indices23 = Nd4j.createFromArray(2L, 3L);
        Map<String, INDArray> afterFreezeResult = sd.output(
                Map.of("data", dataArr, "indices", indices23), "out");
        INDArray afterFreezeOutput = afterFreezeResult.get("out").dup();

        // Rows 2,3: values 30 and 40 → sum = 8*30 + 8*40 = 560
        double expectedAfterFreeze = 8 * 30.0 + 8 * 40.0;
        assertEquals(expectedAfterFreeze, afterFreezeOutput.getDouble(0), TOL,
                "After-freeze gather [2,3] must return different sum than [0,1]");

        log.info("Gather value-dep after freeze test passed: warmup={} (expected {}), " +
                        "afterFreeze={} (expected {})",
                lastWarmupOutput.getDouble(0), expectedWarmup,
                afterFreezeOutput.getDouble(0), expectedAfterFreeze);
    }

    /**
     * Test: multiple freeze-thaw cycles with alternating shapes.
     *
     * Graph: x[1,24] → reshape(shape_var) → mul(2.0) → sum → out
     * Cycle 1: shape [2,12] for 10 steps
     * Cycle 2: shape [3,8] for 10 steps
     * Cycle 3: back to shape [2,12] for 5 steps
     * Each cycle must produce correct results.
     */
    @Test
    @DisplayName("Multiple freeze-thaw cycles with alternating shapes")
    public void testMultipleFreezeThawCycles() {
        SameDiff sd = SameDiff.create();

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 24);
        SDVariable shapeVar = sd.placeHolder("shape_var", DataType.INT64, 2);
        SDVariable reshaped = sd.reshape("reshaped", x, shapeVar);
        SDVariable scaled = reshaped.mul("scaled", 2.0);
        SDVariable out = scaled.sum("out");

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        sd.setGraphExecutionMode(GraphExecutionMode.CUDA_GRAPHS);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 24).mul(3.0f);
        // Expected sum: 24 elements * 3.0 * 2.0 = 144.0 (regardless of reshape)
        double expectedSum = 24 * 3.0 * 2.0;

        // Cycle 1: shape [2,12]
        INDArray shape2x12 = Nd4j.createFromArray(2L, 12L);
        for (int i = 0; i < 10; i++) {
            Map<String, INDArray> result = sd.output(
                    Map.of("x", input, "shape_var", shape2x12), "out");
            assertEquals(expectedSum, result.get("out").getDouble(0), TOL,
                    "Cycle 1 step " + i);
        }

        // Cycle 2: shape [3,8]
        INDArray shape3x8 = Nd4j.createFromArray(3L, 8L);
        for (int i = 0; i < 10; i++) {
            Map<String, INDArray> result = sd.output(
                    Map.of("x", input, "shape_var", shape3x8), "out");
            assertEquals(expectedSum, result.get("out").getDouble(0), TOL,
                    "Cycle 2 step " + i);
        }

        // Cycle 3: back to [2,12]
        for (int i = 0; i < 5; i++) {
            Map<String, INDArray> result = sd.output(
                    Map.of("x", input, "shape_var", shape2x12), "out");
            assertEquals(expectedSum, result.get("out").getDouble(0), TOL,
                    "Cycle 3 step " + i);
        }

        log.info("Multi-cycle freeze-thaw test passed: all 25 steps produced correct sum={}", expectedSum);
    }
}
