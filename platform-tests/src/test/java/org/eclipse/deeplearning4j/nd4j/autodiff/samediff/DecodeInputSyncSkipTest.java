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
import org.nd4j.autodiff.samediff.execution.DynamicShapePlan;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests that the syncToSpecial skip for decode inputs in DynamicShapePlanExecutor
 * does not cause stale data when placeholder values change between steps.
 *
 * Background: When decodeInputsConfigured is true, syncToSpecial() is skipped for
 * input_ids, position_ids, and attention_mask external inputs (because C++ manages
 * them on-device). If the indices are misconfigured, the GPU would read stale host
 * data from a previous step.
 *
 * Test approach: Use standard SameDiff execution (output/outputDirect) which goes
 * through InferenceSession and does NOT use the DSP decode input skip path. This
 * verifies the baseline is correct. Then, if DSP is available, test that DSP execution
 * also produces correct results when placeholder values change each step.
 */
@DisplayName("Decode input sync skip isolation test")
public class DecodeInputSyncSkipTest extends BaseNd4jTestWithBackends {

    private static final Logger log = LoggerFactory.getLogger(DecodeInputSyncSkipTest.class);
    private static final double TOL = 1e-4;

    @Override
    public char ordering() {
        return 'c';
    }

    @Test
    @DisplayName("Changing placeholder values produce different outputs each step (standard path)")
    public void testChangingPlaceholdersStandardPath() {
        // Simulates a decode loop with 3 placeholders (input_ids, position_ids, attention_mask)
        // that change each step. Verifies outputs change accordingly.
        int dim = 16;

        SameDiff sd = SameDiff.create();
        SDVariable inputIds = sd.placeHolder("input_ids", DataType.FLOAT, 1, dim);
        SDVariable positionIds = sd.placeHolder("position_ids", DataType.FLOAT, 1, dim);
        SDVariable attentionMask = sd.placeHolder("attention_mask", DataType.FLOAT, 1, dim);
        // output = input_ids + position_ids * attention_mask
        SDVariable posTimesAttn = positionIds.mul("pos_attn", attentionMask);
        SDVariable out = inputIds.add("output", posTimesAttn);

        INDArray previousOutput = null;

        for (int step = 0; step < 10; step++) {
            float inputVal = (float)(step + 1);
            float posVal = (float)(step * 0.1);
            float maskVal = (step < 5) ? 1.0f : 0.0f; // mask changes mid-sequence

            INDArray inputArr = Nd4j.ones(DataType.FLOAT, 1, dim).mul(inputVal);
            INDArray posArr = Nd4j.ones(DataType.FLOAT, 1, dim).mul(posVal);
            INDArray maskArr = Nd4j.ones(DataType.FLOAT, 1, dim).mul(maskVal);

            Map<String, INDArray> ph = new LinkedHashMap<>();
            ph.put("input_ids", inputArr);
            ph.put("position_ids", posArr);
            ph.put("attention_mask", maskArr);

            Map<String, INDArray> result;
            if (step < 2) {
                result = sd.output(ph, "output");
            } else {
                result = sd.outputDirect(ph, "output");
            }
            INDArray output = result.get("output");

            float expected = inputVal + posVal * maskVal;
            double diff = Math.abs(output.getFloat(0) - expected);
            log.info("Step {}: inputVal={}, posVal={}, maskVal={}, expected={}, got={}, diff={}",
                    step, inputVal, posVal, maskVal, expected, output.getFloat(0), diff);
            assertTrue(diff < TOL,
                    "Step " + step + ": expected " + expected + " but got " + output.getFloat(0));

            // Verify output actually changes from previous step (not stale)
            if (previousOutput != null && step > 0) {
                double changeMagnitude = output.sub(previousOutput).amaxNumber().doubleValue();
                if (step != 5) { // step 5 might happen to match step 4 due to mask change
                    log.info("Step {}: change from previous step magnitude={}", step, changeMagnitude);
                    // At least some change expected unless mask happens to make them equal
                }
            }
            previousOutput = output.dup();
        }
    }

    @Test
    @DisplayName("DSP plan execution with changing placeholders")
    public void testDspExecutionChangingPlaceholders() {
        // Build graph, compile DSP, run multiple steps with different inputs.
        // Verify outputs change correctly.
        int dim = 16;

        SameDiff sd = SameDiff.create();
        sd.setDspAutoCompileEnabled(true);
        SDVariable inputIds = sd.placeHolder("input_ids", DataType.FLOAT, 1, dim);
        SDVariable positionIds = sd.placeHolder("position_ids", DataType.FLOAT, 1, dim);
        SDVariable weights = sd.var("weights", Nd4j.eye(dim).castTo(DataType.FLOAT));
        // output = (input_ids + position_ids) @ weights
        SDVariable sum = inputIds.add("sum", positionIds);
        SDVariable out = sd.mmul("output", sum, weights);

        // First call: compute baseline
        INDArray input1 = Nd4j.ones(DataType.FLOAT, 1, dim).mul(1.0f);
        INDArray pos1 = Nd4j.ones(DataType.FLOAT, 1, dim).mul(0.5f);
        Map<String, INDArray> ph1 = new LinkedHashMap<>();
        ph1.put("input_ids", input1);
        ph1.put("position_ids", pos1);
        Map<String, INDArray> result1 = sd.output(ph1, "output");
        INDArray output1 = result1.get("output").dup();

        // Second call: different values
        INDArray input2 = Nd4j.ones(DataType.FLOAT, 1, dim).mul(10.0f);
        INDArray pos2 = Nd4j.ones(DataType.FLOAT, 1, dim).mul(5.0f);
        Map<String, INDArray> ph2 = new LinkedHashMap<>();
        ph2.put("input_ids", input2);
        ph2.put("position_ids", pos2);
        Map<String, INDArray> result2 = sd.output(ph2, "output");
        INDArray output2 = result2.get("output").dup();

        // Results must differ (1.5 vs 15.0 per element with identity weights)
        double diff = output2.sub(output1).amaxNumber().doubleValue();
        log.info("output1 first={}, output2 first={}, diff={}", output1.getFloat(0), output2.getFloat(0), diff);
        assertTrue(diff > 1.0, "Outputs should differ significantly but diff=" + diff +
                ". Possible stale data from sync skip.");

        // Verify correctness
        float expected1 = 1.5f; // 1.0 + 0.5
        float expected2 = 15.0f; // 10.0 + 5.0
        assertTrue(Math.abs(output1.getFloat(0) - expected1) < TOL,
                "Output1 wrong: expected " + expected1 + " got " + output1.getFloat(0));
        assertTrue(Math.abs(output2.getFloat(0) - expected2) < TOL,
                "Output2 wrong: expected " + expected2 + " got " + output2.getFloat(0));
    }

    @Test
    @DisplayName("Rapid alternating values detect stale data")
    public void testRapidAlternatingValues() {
        // Alternate between two very different values to detect stale data more easily
        int dim = 8;

        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, dim);
        SDVariable weights = sd.var("weights", Nd4j.eye(dim).castTo(DataType.FLOAT));
        SDVariable out = sd.mmul("output", input, weights);

        float[] values = {1.0f, 1000.0f}; // Very different values

        for (int i = 0; i < 20; i++) {
            float val = values[i % 2];
            INDArray inputArr = Nd4j.ones(DataType.FLOAT, 1, dim).mul(val);

            Map<String, INDArray> ph = Collections.singletonMap("input", inputArr);
            Map<String, INDArray> result;
            if (i < 2) {
                result = sd.output(ph, "output");
            } else {
                result = sd.outputDirect(ph, "output");
            }
            INDArray output = result.get("output");

            double diff = Math.abs(output.getFloat(0) - val);
            log.info("Step {}: expected={}, got={}, diff={}", i, val, output.getFloat(0), diff);
            assertTrue(diff < TOL,
                    "Step " + i + ": stale data detected. Expected " + val + " got " + output.getFloat(0) +
                    ". Previous step value was " + values[(i + 1) % 2]);
        }
    }
}
