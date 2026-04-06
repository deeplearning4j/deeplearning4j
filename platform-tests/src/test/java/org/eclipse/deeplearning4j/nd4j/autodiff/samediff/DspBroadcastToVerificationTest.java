/*
 *  ******************************************************************************
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
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DspDebugger;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastTo;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Verifies broadcast_to behavior using DspDebugger framework.
 *
 * The broadcast_to op was rewritten after baseline b28e4eed8c with
 * resolveTargetShape() that applies "numpy broadcast rules". These tests
 * verify the shape behavior and detect if the rules cause wrong shapes.
 */
@Slf4j
@DisplayName("DSP broadcast_to Verification")
public class DspBroadcastToVerificationTest extends BaseNd4jTestWithBackends {

    private static final double TOL = 1e-4;

    @Override
    public char ordering() {
        return 'c';
    }

    private SDVariable addBroadcastTo(SameDiff sd, SDVariable input, SDVariable shape) {
        BroadcastTo op = new BroadcastTo(sd, input, shape);
        return op.outputVariables()[0];
    }

    /**
     * Standard broadcast: input[1,4] → broadcast_to(shape=[3,4]).
     * Both baseline and rewrite should produce [3,4].
     */
    @Test
    @DisplayName("Standard broadcast: [1,4] → [3,4]")
    public void testStandardBroadcast() {
        SameDiff sd = SameDiff.create();

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 4);
        SDVariable shape = sd.constant("shape", Nd4j.createFromArray(3L, 4L));
        SDVariable result = addBroadcastTo(sd, input, shape);
        SDVariable out = result.sum("out");

        DspDebugger debugger = DspDebugger.attach(sd);

        INDArray inputArr = Nd4j.createFromArray(new float[][]{{1, 2, 3, 4}});
        Map<String, INDArray> results = sd.output(Map.of("input", inputArr), "out");

        assertEquals(30.0, results.get("out").getDouble(0), TOL,
                "Sum should be 3*(1+2+3+4) = 30");

        DspDebugger.StepReport step = debugger.validateStep(Map.of("input", inputArr), "out");
        assertFalse(step.hasErrors(), "Should have no errors: " + step);
        log.info("Standard broadcast test passed. Step report: {}", step);
    }

    /**
     * The "no shrinking" case: input[8,4] → broadcast_to(shape=[1,4]).
     * Documents current behavior vs baseline.
     *
     * ONNX Expand: max(8,1)=8 → output [8,4]. But if ONNX graph provides
     * the already-resolved shape, the op should use it verbatim.
     */
    @Test
    @DisplayName("No-shrinking case: [8,4] → broadcast_to(shape=[1,4])")
    public void testNoShrinkingCase() {
        SameDiff sd = SameDiff.create();

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 8, 4);
        SDVariable shape = sd.constant("shape", Nd4j.createFromArray(1L, 4L));
        SDVariable result = addBroadcastTo(sd, input, shape);
        SDVariable out = result.sum("out");

        INDArray inputArr = Nd4j.ones(DataType.FLOAT, 8, 4);

        try {
            Map<String, INDArray> results = sd.output(Map.of("input", inputArr), "out");
            double sum = results.get("out").getDouble(0);

            // If resolveTargetShape applied "no shrinking": output is [8,4], sum=32
            // If baseline verbatim: should error (validation rejects 8 != 1 && 8 != 1)
            log.info("No-shrinking case produced sum={} (resolveTargetShape likely changed [1,4]→[8,4])", sum);
            if (Math.abs(sum - 32.0) < TOL) {
                log.warn("CONFIRMED: resolveTargetShape modified shape [1,4]→[8,4]. " +
                        "This changes ONNX model behavior vs baseline.");
            }
        } catch (Exception e) {
            log.info("BASELINE BEHAVIOR: validation rejected input[8,4] → shape[1,4]. " +
                    "Error: {}", e.getMessage());
        }
    }

    /**
     * Multi-step broadcast with DspDebugger validation.
     * Simulates the VLM vision encoder pattern.
     */
    @Test
    @DisplayName("Multi-step broadcast with DspDebugger validation")
    public void testMultiStepBroadcastWithDebugger() {
        SameDiff sd = SameDiff.create();

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 4);
        SDVariable shape = sd.placeHolder("shape", DataType.INT64, 2);
        SDVariable result = addBroadcastTo(sd, input, shape);
        SDVariable out = result.sum("out");

        DspDebugger debugger = DspDebugger.attach(sd);

        INDArray inputArr = Nd4j.createFromArray(new float[][]{{1, 2, 3, 4}});

        // Step 1: broadcast to [2,4]
        INDArray shape2x4 = Nd4j.createFromArray(2L, 4L);
        DspDebugger.StepReport step1 = debugger.validateStep(
                Map.of("input", inputArr, "shape", shape2x4), "out");
        assertFalse(step1.hasErrors(), "Step 1 should have no errors: " + step1);

        Map<String, INDArray> result1 = sd.output(
                Map.of("input", inputArr, "shape", shape2x4), "out");
        assertEquals(20.0, result1.get("out").getDouble(0), TOL,
                "Step 1: sum = 2*(1+2+3+4) = 20");

        // Step 2: broadcast to [5,4]
        INDArray shape5x4 = Nd4j.createFromArray(5L, 4L);
        DspDebugger.StepReport step2 = debugger.validateStep(
                Map.of("input", inputArr, "shape", shape5x4), "out");
        assertFalse(step2.hasErrors(), "Step 2 should have no errors: " + step2);

        Map<String, INDArray> result2 = sd.output(
                Map.of("input", inputArr, "shape", shape5x4), "out");
        assertEquals(50.0, result2.get("out").getDouble(0), TOL,
                "Step 2: sum = 5*(1+2+3+4) = 50");

        // Analyze plan
        DspDebugger.PlanReport report = debugger.analyzePlan();
        log.info("Plan report:\n{}", report);

        // Check broadcast_to slot flags
        for (DspDebugger.SlotInfo slot : report.slots) {
            if (slot.opName.contains("broadcast_to")) {
                log.info("broadcast_to slot: {}", slot);
                assertTrue(slot.isShapeDependsOnValues(),
                        "broadcast_to should have FLAG_SHAPE_DEPENDS_ON_VALUES");
            }
        }
    }
}
