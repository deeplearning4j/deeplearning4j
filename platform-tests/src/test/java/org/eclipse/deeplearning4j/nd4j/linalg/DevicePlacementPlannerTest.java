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

package org.eclipse.deeplearning4j.nd4j.linalg;

import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DevicePlacementPlanner;
import org.nd4j.autodiff.samediff.execution.DevicePlacementPlanner.PlacementPlan;
import org.nd4j.autodiff.samediff.execution.DevicePlacementPlanner.PlacementStrategy;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlan;
import org.nd4j.autodiff.samediff.execution.DynamicShapeSlot;
import org.nd4j.autodiff.samediff.execution.ModelMemoryEstimator;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link DevicePlacementPlanner} and {@link ModelMemoryEstimator}.
 * These tests use mock/minimal DSP plans and don't require multi-GPU hardware.
 */
public class DevicePlacementPlannerTest {

    @Test
    public void testDisabledByDefault() {
        // When placement is not enabled, compiling a DSP plan should not set targetDeviceId
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 4);
        SDVariable weights = sd.var("weights", Nd4j.randn(DataType.FLOAT, 4, 8));
        SDVariable output = sd.mmul("output", input, weights);

        // Verify placementStrategy is null by default
        assertNull(sd.getPlacementStrategy());
        assertNull(sd.getCustomDevicePlacement());
    }

    @Test
    public void testSingleDevicePlacement() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 4);
        SDVariable weights = sd.var("weights", Nd4j.randn(DataType.FLOAT, 4, 8));
        SDVariable output = sd.mmul("output", input, weights);

        // Create a minimal mock plan
        DynamicShapeSlot[] slots = createMockSlots(5);
        DynamicShapePlan plan = createMockPlan(slots);

        PlacementPlan placement = DevicePlacementPlanner.plan(plan, sd, PlacementStrategy.SINGLE_DEVICE);

        assertNotNull(placement);
        assertTrue(placement.isValid());
        assertEquals(PlacementStrategy.SINGLE_DEVICE, placement.getStrategy());

        // All slots should be on device 0
        int[] slotDevices = placement.getSlotDeviceIds();
        assertEquals(5, slotDevices.length);
        for (int device : slotDevices) {
            assertEquals(0, device);
        }
    }

    @Test
    public void testCustomPlacement() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 4);
        SDVariable w1 = sd.var("w1", Nd4j.randn(DataType.FLOAT, 4, 8));
        SDVariable w2 = sd.var("w2", Nd4j.randn(DataType.FLOAT, 8, 4));
        SDVariable h = sd.mmul("hidden", input, w1);
        SDVariable output = sd.mmul("output", h, w2);

        // Create slots with output var names matching the model
        DynamicShapeSlot[] slots = new DynamicShapeSlot[]{
                DynamicShapeSlot.builder()
                        .opName("mmul")
                        .outputVarNames(new String[]{"hidden"})
                        .inputVarNames(new String[]{"input", "w1"})
                        .outputSlotIndices(new int[]{0})
                        .inputSourceIndices(new int[]{-1, -2})
                        .inputSourceTypes(new byte[]{0, 0})
                        .stepIndex(0)
                        .build(),
                DynamicShapeSlot.builder()
                        .opName("mmul")
                        .outputVarNames(new String[]{"output"})
                        .inputVarNames(new String[]{"hidden", "w2"})
                        .outputSlotIndices(new int[]{1})
                        .inputSourceIndices(new int[]{0, -3})
                        .inputSourceTypes(new byte[]{1, 0})
                        .stepIndex(1)
                        .build()
        };

        DynamicShapePlan plan = createMockPlan(slots);

        // Custom mapping: "hidden" on device 0, "output" on device 1
        Map<String, Integer> customMap = new HashMap<>();
        customMap.put("hidden", 0);
        customMap.put("output", 1);

        PlacementPlan placement = DevicePlacementPlanner.planCustom(plan, sd, customMap);

        assertNotNull(placement);
        assertTrue(placement.isValid());
        assertEquals(PlacementStrategy.CUSTOM, placement.getStrategy());

        int[] slotDevices = placement.getSlotDeviceIds();
        assertEquals(2, slotDevices.length);
        assertEquals(0, slotDevices[0]); // "hidden" slot on device 0
        assertEquals(1, slotDevices[1]); // "output" slot on device 1
    }

    @Test
    public void testApplyPlan() {
        DynamicShapeSlot[] slots = createMockSlots(3);
        DynamicShapePlan plan = createMockPlan(slots);

        int[] slotDevices = {0, 1, 0};
        Map<Integer, Long> estimates = new HashMap<>();
        estimates.put(0, 1024L);
        estimates.put(1, 512L);

        PlacementPlan placement = new PlacementPlan(
                slotDevices, new HashMap<>(), PlacementStrategy.MEMORY_FIT, estimates, true);

        DevicePlacementPlanner.applyPlan(plan, placement);

        // Verify targetDeviceId was set on each slot
        assertEquals(0, slots[0].getTargetDeviceId());
        assertEquals(1, slots[1].getTargetDeviceId());
        assertEquals(0, slots[2].getTargetDeviceId());
    }

    @Test
    public void testApplyPlanInvalid() {
        DynamicShapeSlot[] slots = createMockSlots(3);
        DynamicShapePlan plan = createMockPlan(slots);

        PlacementPlan invalidPlan = new PlacementPlan(
                new int[0], new HashMap<>(), PlacementStrategy.SINGLE_DEVICE, new HashMap<>(), false);

        DevicePlacementPlanner.applyPlan(plan, invalidPlan);

        // Slots should remain at default (-1) since plan is invalid
        assertEquals(-1, slots[0].getTargetDeviceId());
        assertEquals(-1, slots[1].getTargetDeviceId());
        assertEquals(-1, slots[2].getTargetDeviceId());
    }

    @Test
    public void testModelMemoryEstimator() {
        SameDiff sd = SameDiff.create();
        // Create constants with known sizes
        INDArray w1 = Nd4j.randn(DataType.FLOAT, 100, 200); // 100*200*4 = 80,000 bytes
        INDArray w2 = Nd4j.randn(DataType.FLOAT, 200, 50);  // 200*50*4 = 40,000 bytes
        sd.constant("w1", w1);
        sd.constant("w2", w2);
        sd.placeHolder("input", DataType.FLOAT, -1, 100);

        long weightMem = ModelMemoryEstimator.estimateWeightMemory(sd);
        // Should be exactly 80000 + 40000 = 120000 bytes
        assertEquals(120000L, weightMem);
    }

    @Test
    public void testModelMemoryEstimatorAll() {
        SameDiff sd = SameDiff.create();
        INDArray w1 = Nd4j.randn(DataType.FLOAT, 10, 20);
        sd.constant("const1", w1);
        sd.placeHolder("ph1", DataType.FLOAT, -1, 10);

        List<ModelMemoryEstimator.MemoryEstimate> estimates = ModelMemoryEstimator.estimateAll(sd);
        assertFalse(estimates.isEmpty());

        // Find the constant estimate
        ModelMemoryEstimator.MemoryEstimate constEstimate = estimates.stream()
                .filter(e -> e.getVariableName().equals("const1"))
                .findFirst().orElse(null);
        assertNotNull(constEstimate);
        assertTrue(constEstimate.isConstant());
        assertFalse(constEstimate.isActivation());
        assertEquals(10 * 20 * 4, constEstimate.getEstimatedBytes());
    }

    @Test
    public void testPeakActivationEstimate() {
        // Create a plan with known release schedule
        DynamicShapeSlot[] slots = createMockSlots(4);
        // releaseAtStep[i] = output slot indices released after step i
        int[][] releaseAtStep = {
                {},        // step 0: nothing released
                {0},       // step 1: release output from step 0
                {},        // step 2: nothing released
                {1, 2, 3}  // step 3: release outputs from steps 1, 2, 3
        };

        DynamicShapePlan plan = createMockPlanWithRelease(slots, releaseAtStep, 4);

        long peak = ModelMemoryEstimator.estimatePeakMemory(plan, 0, 4);
        // Peak should be positive (at least some memory is live)
        assertTrue(peak > 0);
    }

    @Test
    public void testSameDiffPlacementFields() {
        SameDiff sd = SameDiff.create();

        // Test setter/getter for placement strategy
        assertNull(sd.getPlacementStrategy());
        sd.setPlacementStrategy(PlacementStrategy.MEMORY_FIT);
        assertEquals(PlacementStrategy.MEMORY_FIT, sd.getPlacementStrategy());

        // Test setter/getter for custom placement
        assertNull(sd.getCustomDevicePlacement());
        Map<String, Integer> custom = new HashMap<>();
        custom.put("var1", 0);
        custom.put("var2", 1);
        sd.setCustomDevicePlacement(custom);
        assertEquals(custom, sd.getCustomDevicePlacement());
    }

    // ---- Helper methods ----

    private DynamicShapeSlot[] createMockSlots(int count) {
        DynamicShapeSlot[] slots = new DynamicShapeSlot[count];
        for (int i = 0; i < count; i++) {
            slots[i] = DynamicShapeSlot.builder()
                    .opName("mock_op_" + i)
                    .outputSlotIndices(new int[]{i})
                    .outputVarNames(new String[]{"out_" + i})
                    .inputVarNames(new String[]{})
                    .inputSourceIndices(new int[]{})
                    .inputSourceTypes(new byte[]{})
                    .stepIndex(i)
                    .build();
        }
        return slots;
    }

    private DynamicShapePlan createMockPlan(DynamicShapeSlot[] slots) {
        int[][] releaseAtStep = new int[slots.length][];
        for (int i = 0; i < slots.length; i++) {
            releaseAtStep[i] = new int[0];
        }
        return new DynamicShapePlan(
                slots,
                slots.length,   // totalOutputSlots
                releaseAtStep,
                null,           // opContextPool
                new String[0],  // externalInputKeys
                new byte[0],    // extSourceTypes
                new java.util.LinkedHashSet<>(), // requestedOutputs
                new HashMap<>(), // outputNameToSlotIndex
                false,          // hasControlFlowOps
                null,           // loopRegions
                null,           // predecessorCounts
                null,           // predecessors
                null,           // successors
                null,           // consumerCounts
                null            // rootSlots
        );
    }

    private DynamicShapePlan createMockPlanWithRelease(DynamicShapeSlot[] slots,
                                                       int[][] releaseAtStep,
                                                       int totalOutputSlots) {
        return new DynamicShapePlan(
                slots,
                totalOutputSlots,
                releaseAtStep,
                null,           // opContextPool
                new String[0],  // externalInputKeys
                new byte[0],    // extSourceTypes
                new java.util.LinkedHashSet<>(), // requestedOutputs
                new HashMap<>(), // outputNameToSlotIndex
                false,          // hasControlFlowOps
                null,           // loopRegions
                null,           // predecessorCounts
                null,           // predecessors
                null,           // successors
                null,           // consumerCounts
                null            // rootSlots
        );
    }
}
