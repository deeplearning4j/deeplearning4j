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

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests that the DSP shape key computation does not poison device buffers.
 *
 * Bug: computeSegmentShapeKey() calls syncToHost() on small INT/LONG/BOOL
 * inputs to hash their values. This marks the host buffer as primary.
 * When the GPU kernel subsequently executes, it reads the now-stale device
 * buffer, producing wrong results.
 *
 * The fix: read values from device buffer directly (cudaMemcpy to stack)
 * without modifying the NDArray's sync state.
 *
 * These tests verify correctness by running graphs through DSP where
 * INT/LONG inputs feed both shape-determining ops (reshape, gather) and
 * compute ops that need correct device data. If shape key computation
 * poisons the device buffer, the compute ops produce wrong results.
 */
@DisplayName("DSP Shape Key Device Sync Safety")
public class DspShapeKeyDeviceSyncTest extends BaseNd4jTestWithBackends {

    private static final Logger log = LoggerFactory.getLogger(DspShapeKeyDeviceSyncTest.class);
    private static final double TOL = 1e-4;

    @Override
    public char ordering() {
        return 'c';
    }

    /**
     * Test: INT32 indices used for both gather (value-dependent shape) and
     * arithmetic. If shape key computation poisons the indices' device buffer,
     * the arithmetic reads stale data.
     *
     * Graph: indices[2] → gather(data, indices) → sum → out1
     *        indices[2] → cast(FLOAT) → sum → out2
     *
     * out2 depends on indices having correct device data AFTER the shape key
     * was computed for the gather segment. If syncToHost poisoned the device
     * buffer, out2 will be wrong (stale values from a previous step).
     */
    @Test
    @DisplayName("INT32 indices: device data survives shape key computation")
    public void testInt32IndicesDeviceDataSurvivesShapeKey() {
        SameDiff sd = SameDiff.create();

        SDVariable data = sd.placeHolder("data", DataType.FLOAT, 4, 3);
        SDVariable indices = sd.placeHolder("indices", DataType.INT32, 2);

        // Path 1: gather uses indices for value-dependent shape
        SDVariable gathered = sd.gather("gathered", data, indices, 0);
        SDVariable gatherSum = gathered.sum("gather_sum");

        // Path 2: cast indices to float and sum — needs correct device data
        SDVariable indicesFloat = sd.castTo("indices_float", indices, DataType.FLOAT);
        SDVariable indexSum = indicesFloat.sum("index_sum");

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        // Data: row i has all values = (i+1)*100
        INDArray dataArr = Nd4j.zeros(DataType.FLOAT, 4, 3);
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 3; c++) {
                dataArr.putScalar(r, c, (r + 1) * 100.0f);
            }
        }

        // Step 1: indices = [0, 1]
        INDArray idx01 = Nd4j.createFromArray(0, 1);
        Map<String, INDArray> result1 = sd.output(
                Map.of("data", dataArr, "indices", idx01), "gather_sum", "index_sum");

        double expectedGatherSum1 = 3 * 100.0 + 3 * 200.0;  // rows 0,1
        double expectedIndexSum1 = 0.0 + 1.0;
        assertEquals(expectedGatherSum1, result1.get("gather_sum").getDouble(0), TOL,
                "Step 1: gather sum should be rows 0+1");
        assertEquals(expectedIndexSum1, result1.get("index_sum").getDouble(0), TOL,
                "Step 1: index sum should be 0+1=1");

        // Step 2: indices = [2, 3] — different values
        INDArray idx23 = Nd4j.createFromArray(2, 3);
        Map<String, INDArray> result2 = sd.output(
                Map.of("data", dataArr, "indices", idx23), "gather_sum", "index_sum");

        double expectedGatherSum2 = 3 * 300.0 + 3 * 400.0;  // rows 2,3
        double expectedIndexSum2 = 2.0 + 3.0;
        assertEquals(expectedGatherSum2, result2.get("gather_sum").getDouble(0), TOL,
                "Step 2: gather sum should be rows 2+3");
        assertEquals(expectedIndexSum2, result2.get("index_sum").getDouble(0), TOL,
                "Step 2: index sum should be 2+3=5 (NOT stale 0+1=1)");

        // Step 3: run again with same indices to verify replay consistency
        Map<String, INDArray> result3 = sd.output(
                Map.of("data", dataArr, "indices", idx23), "gather_sum", "index_sum");
        assertEquals(expectedGatherSum2, result3.get("gather_sum").getDouble(0), TOL,
                "Step 3: gather sum should be stable on replay");
        assertEquals(expectedIndexSum2, result3.get("index_sum").getDouble(0), TOL,
                "Step 3: index sum should be stable on replay");

        log.info("INT32 test passed: step1 gatherSum={} indexSum={}, step2 gatherSum={} indexSum={}",
                result1.get("gather_sum").getDouble(0), result1.get("index_sum").getDouble(0),
                result2.get("gather_sum").getDouble(0), result2.get("index_sum").getDouble(0));
    }

    /**
     * Test: INT64 shape tensor used for reshape AND downstream arithmetic.
     * The shape tensor feeds a value-dependent reshape op, which triggers
     * shape key value hashing (syncToHost). The same tensor is also cast
     * to float and summed. If device data is poisoned, the sum is wrong.
     *
     * Graph: x[1,12] → reshape(shape) → relu → sum → out1
     *        shape[2] → cast(FLOAT) → sum → out2
     */
    @Test
    @DisplayName("INT64 shape tensor: device data survives shape key computation")
    public void testInt64ShapeTensorDeviceDataSurvivesShapeKey() {
        SameDiff sd = SameDiff.create();

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 12);
        SDVariable shape = sd.placeHolder("shape", DataType.INT64, 2);

        // Path 1: reshape uses shape values (triggers shape key value hashing)
        SDVariable reshaped = sd.reshape("reshaped", x, shape);
        SDVariable activated = sd.nn.relu("activated", reshaped, 0);
        SDVariable reshapeSum = activated.sum("reshape_sum");

        // Path 2: cast shape to float and sum — needs correct device data
        SDVariable shapeFloat = sd.castTo("shape_float", shape, DataType.FLOAT);
        SDVariable shapeSum = shapeFloat.sum("shape_sum");

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        INDArray input = Nd4j.linspace(1, 12, 12, DataType.FLOAT).reshape(1, 12);

        // Step 1: shape = [2, 6]
        INDArray shape2x6 = Nd4j.createFromArray(2L, 6L);
        Map<String, INDArray> result1 = sd.output(
                Map.of("x", input, "shape", shape2x6), "reshape_sum", "shape_sum");

        double expectedDataSum = 0;
        for (int i = 1; i <= 12; i++) expectedDataSum += Math.max(0, i);
        double expectedShapeSum1 = 2.0 + 6.0;

        assertEquals(expectedDataSum, result1.get("reshape_sum").getDouble(0), TOL,
                "Step 1: reshape+relu+sum should be correct");
        assertEquals(expectedShapeSum1, result1.get("shape_sum").getDouble(0), TOL,
                "Step 1: shape sum should be 2+6=8");

        // Step 2: shape = [3, 4] — shape values change
        INDArray shape3x4 = Nd4j.createFromArray(3L, 4L);
        Map<String, INDArray> result2 = sd.output(
                Map.of("x", input, "shape", shape3x4), "reshape_sum", "shape_sum");

        double expectedShapeSum2 = 3.0 + 4.0;
        assertEquals(expectedDataSum, result2.get("reshape_sum").getDouble(0), TOL,
                "Step 2: reshape+relu+sum should still be correct");
        assertEquals(expectedShapeSum2, result2.get("shape_sum").getDouble(0), TOL,
                "Step 2: shape sum should be 3+4=7 (NOT stale 2+6=8)");

        // Step 3: shape = [4, 3]
        INDArray shape4x3 = Nd4j.createFromArray(4L, 3L);
        Map<String, INDArray> result3 = sd.output(
                Map.of("x", input, "shape", shape4x3), "reshape_sum", "shape_sum");

        double expectedShapeSum3 = 4.0 + 3.0;
        assertEquals(expectedDataSum, result3.get("reshape_sum").getDouble(0), TOL,
                "Step 3: reshape+relu+sum should still be correct");
        assertEquals(expectedShapeSum3, result3.get("shape_sum").getDouble(0), TOL,
                "Step 3: shape sum should be 4+3=7 (NOT stale)");

        log.info("INT64 test passed: shape sums = {}, {}, {}",
                result1.get("shape_sum").getDouble(0),
                result2.get("shape_sum").getDouble(0),
                result3.get("shape_sum").getDouble(0));
    }

    /**
     * Test: BOOL condition tensor used for masking AND downstream cast+sum.
     * BOOL arrays trigger shape key value hashing (syncToHost). If device
     * data is poisoned, the cast+sum reads stale values.
     *
     * Graph: condition[4] → cast(FLOAT) → multiply(x) → sum → out1  (masked sum)
     *        condition[4] → cast(FLOAT) → sum → out2  (count of true values)
     */
    @Test
    @DisplayName("BOOL condition: device data survives shape key computation")
    public void testBoolConditionDeviceDataSurvivesShapeKey() {
        SameDiff sd = SameDiff.create();

        SDVariable condition = sd.placeHolder("condition", DataType.BOOL, 4);
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 4);

        // Cast condition to float — triggers shape key hashing on BOOL input
        SDVariable condFloat = sd.castTo("cond_float", condition, DataType.FLOAT);

        // Path 1: masked sum = x * condition_as_float → sum
        SDVariable masked = condFloat.mul("masked", x);
        SDVariable maskedSum = masked.sum("masked_sum");

        // Path 2: count of true values
        SDVariable condSum = condFloat.sum("cond_sum");

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        INDArray xVals = Nd4j.createFromArray(10.0f, 20.0f, 30.0f, 40.0f);

        // Step 1: condition = [true, false, true, false]
        INDArray cond1 = Nd4j.createFromArray(true, false, true, false);
        Map<String, INDArray> result1 = sd.output(
                Map.of("condition", cond1, "x", xVals), "masked_sum", "cond_sum");

        double expectedMaskedSum1 = 10.0 + 0.0 + 30.0 + 0.0;  // T,F,T,F
        double expectedCondSum1 = 2.0;
        assertEquals(expectedMaskedSum1, result1.get("masked_sum").getDouble(0), TOL,
                "Step 1: masked sum should select T,F,T,F");
        assertEquals(expectedCondSum1, result1.get("cond_sum").getDouble(0), TOL,
                "Step 1: cond sum should be 2");

        // Step 2: condition = [false, true, false, true] — flipped
        INDArray cond2 = Nd4j.createFromArray(false, true, false, true);
        Map<String, INDArray> result2 = sd.output(
                Map.of("condition", cond2, "x", xVals), "masked_sum", "cond_sum");

        double expectedMaskedSum2 = 0.0 + 20.0 + 0.0 + 40.0;  // F,T,F,T
        double expectedCondSum2 = 2.0;
        assertEquals(expectedMaskedSum2, result2.get("masked_sum").getDouble(0), TOL,
                "Step 2: masked sum should be F,T,F,T (NOT stale T,F,T,F)");
        assertEquals(expectedCondSum2, result2.get("cond_sum").getDouble(0), TOL,
                "Step 2: cond sum should be 2");

        // Step 3: condition = [true, true, true, true] — all true
        INDArray cond3 = Nd4j.createFromArray(true, true, true, true);
        Map<String, INDArray> result3 = sd.output(
                Map.of("condition", cond3, "x", xVals), "masked_sum", "cond_sum");

        double expectedMaskedSum3 = 10.0 + 20.0 + 30.0 + 40.0;
        double expectedCondSum3 = 4.0;
        assertEquals(expectedMaskedSum3, result3.get("masked_sum").getDouble(0), TOL,
                "Step 3: masked sum should include all values");
        assertEquals(expectedCondSum3, result3.get("cond_sum").getDouble(0), TOL,
                "Step 3: cond sum should be 4 (NOT stale 2)");

        log.info("BOOL test passed: masked sums = {}, {}, {}, cond sums = {}, {}, {}",
                result1.get("masked_sum").getDouble(0),
                result2.get("masked_sum").getDouble(0),
                result3.get("masked_sum").getDouble(0),
                result1.get("cond_sum").getDouble(0),
                result2.get("cond_sum").getDouble(0),
                result3.get("cond_sum").getDouble(0));
    }

    /**
     * Test: Multiple frozen replay steps with stable INT64 inputs.
     * Verifies that shape key computation during frozen replay doesn't
     * corrupt device data over many iterations (regression over time).
     *
     * Graph: x[1,6] → reshape(shape=[2,3]) → matmul(w[3,4]) → sum → out
     *        shape[2] → cast(FLOAT) → sum → shape_check
     *
     * Run 20 steps with same shape. If syncToHost poisoning accumulates,
     * later steps will diverge.
     */
    @Test
    @DisplayName("Frozen replay: INT64 device data stable over 20 steps")
    public void testFrozenReplayDeviceDataStability() {
        SameDiff sd = SameDiff.create();

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 6);
        SDVariable shape = sd.placeHolder("shape", DataType.INT64, 2);
        SDVariable w = sd.var("w", Nd4j.ones(DataType.FLOAT, 3, 4));

        SDVariable reshaped = sd.reshape("reshaped", x, shape);
        SDVariable matmul = sd.mmul("matmul", reshaped, w);
        SDVariable out = matmul.sum("out");

        // Shape check: cast shape to float and sum
        SDVariable shapeFloat = sd.castTo("shape_float", shape, DataType.FLOAT);
        SDVariable shapeCheck = shapeFloat.sum("shape_check");

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 6);
        INDArray shapeArr = Nd4j.createFromArray(2L, 3L);

        Double firstOut = null;
        Double firstShapeCheck = null;

        for (int step = 0; step < 20; step++) {
            Map<String, INDArray> result = sd.output(
                    Map.of("x", input, "shape", shapeArr), "out", "shape_check");

            double outVal = result.get("out").getDouble(0);
            double shapeCheckVal = result.get("shape_check").getDouble(0);

            if (firstOut == null) {
                firstOut = outVal;
                firstShapeCheck = shapeCheckVal;
                // Matmul: [2,3] x [3,4] = [2,4], all ones → each element = 3, sum = 2*4*3 = 24
                assertEquals(24.0, outVal, TOL, "Step 0: matmul sum");
                assertEquals(5.0, shapeCheckVal, TOL, "Step 0: shape sum (2+3)");
            } else {
                assertEquals(firstOut, outVal, TOL,
                        "Step " + step + ": output diverged from step 0 (device data poisoned?)");
                assertEquals(firstShapeCheck, shapeCheckVal, TOL,
                        "Step " + step + ": shape check diverged (device data poisoned?)");
            }
        }

        log.info("Frozen replay stability test passed: 20 steps, out={}, shapeCheck={}",
                firstOut, firstShapeCheck);
    }
}
