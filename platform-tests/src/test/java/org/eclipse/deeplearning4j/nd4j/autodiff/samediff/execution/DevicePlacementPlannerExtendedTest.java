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
package org.eclipse.deeplearning4j.nd4j.autodiff.samediff.execution;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DevicePlacementPlanner;
import org.nd4j.autodiff.samediff.execution.DevicePlacementPlanner.PlacementPlan;
import org.nd4j.autodiff.samediff.execution.DevicePlacementPlanner.PlacementStrategy;
import org.nd4j.autodiff.samediff.execution.DeviceKey;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlan;
import org.nd4j.autodiff.samediff.execution.DynamicShapeSlot;
import org.nd4j.autodiff.samediff.execution.ModelMemoryEstimator;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Extended tests for {@link DevicePlacementPlanner} and its output data model.
 * Tests use mock plans built from {@link DynamicShapeSlot} builders.
 *
 * Complements the existing {@link org.eclipse.deeplearning4j.nd4j.linalg.DevicePlacementPlannerTest}
 * with additional edge cases and validation.
 */
@DisplayName("DevicePlacementPlanner extended tests")
public class DevicePlacementPlannerExtendedTest {

    // ─── Helpers ──────────────────────────────────────────────────────────────

    private static DynamicShapeSlot slot(String opName, int stepIndex, int targetDeviceId,
                                          int[] inputSourceIndices, byte[] inputSourceTypes,
                                          String[] inputVarNames, int[] outputSlotIndices,
                                          String[] outputVarNames) {
        return DynamicShapeSlot.builder()
                .opName(opName)
                .stepIndex(stepIndex)
                .targetDeviceId(targetDeviceId)
                .inputSourceIndices(inputSourceIndices)
                .inputSourceTypes(inputSourceTypes)
                .inputVarNames(inputVarNames)
                .outputSlotIndices(outputSlotIndices)
                .outputVarNames(outputVarNames)
                .build();
    }

    private static DynamicShapePlan makePlan(DynamicShapeSlot[] slots) {
        int[][] releaseAtStep = new int[slots.length][];
        for (int i = 0; i < slots.length; i++) releaseAtStep[i] = new int[0];
        return new DynamicShapePlan(
                slots, slots.length, releaseAtStep,
                null, new String[0], new byte[0],
                new LinkedHashSet<>(), Map.of(),
                false, null, null, null, null, null, null
        );
    }

    // ─── PlacementPlan data model ────────────────────────────────────────────

    @Test
    @DisplayName("PlacementPlan: valid plan returns true")
    void testPlacementPlanValid() {
        PlacementPlan plan = new PlacementPlan(
                new int[]{0, 0, 1},
                Map.of("w1", 0, "w2", 1),
                PlacementStrategy.MEMORY_FIT,
                Map.of(0, 1000L, 1, 500L),
                true
        );
        assertTrue(plan.isValid());
        assertEquals(PlacementStrategy.MEMORY_FIT, plan.getStrategy());
        assertEquals(3, plan.getSlotDeviceIds().length);
        assertEquals(2, plan.getConstantDeviceIds().size());
        assertEquals(2, plan.getDeviceMemoryEstimates().size());
    }

    @Test
    @DisplayName("PlacementPlan: invalid plan returns false")
    void testPlacementPlanInvalid() {
        PlacementPlan plan = new PlacementPlan(
                new int[0],
                Map.of(),
                PlacementStrategy.SINGLE_DEVICE,
                Map.of(),
                false
        );
        assertFalse(plan.isValid());
    }

    @Test
    @DisplayName("PlacementPlan: toString includes strategy")
    void testPlacementPlanToString() {
        PlacementPlan plan = new PlacementPlan(
                new int[]{0, 1}, Map.of(),
                PlacementStrategy.PIPELINE_PARALLEL, Map.of(0, 100L, 1, 200L),
                true
        );
        String s = plan.toString();
        assertTrue(s.contains("PIPELINE_PARALLEL"));
    }

    // ─── DeviceKey ────────────────────────────────────────────────────────────

    @Test
    @DisplayName("DeviceKey: isCompatibleWith same type and arch")
    void testDeviceKeyCompatible() {
        DeviceKey k1 = new DeviceKey(
                DeviceKey.Type.CUDA_GPU, 0, "sm86");
        DeviceKey k2 = new DeviceKey(
                DeviceKey.Type.CUDA_GPU, 1, "sm86");
        assertTrue(k1.isCompatibleWith(k2));
    }

    @Test
    @DisplayName("DeviceKey: not compatible with different type")
    void testDeviceKeyNotCompatible() {
        DeviceKey k1 = new DeviceKey(
                DeviceKey.Type.CUDA_GPU, 0, "sm86");
        DeviceKey k2 = new DeviceKey(
                DeviceKey.Type.CPU, 0, "");
        assertFalse(k1.isCompatibleWith(k2));
    }

    @Test
    @DisplayName("DeviceKey: not compatible with different arch")
    void testDeviceKeyNotCompatibleDiffArch() {
        DeviceKey k1 = new DeviceKey(
                DeviceKey.Type.CUDA_GPU, 0, "sm86");
        DeviceKey k2 = new DeviceKey(
                DeviceKey.Type.CUDA_GPU, 0, "sm90");
        assertFalse(k1.isCompatibleWith(k2));
    }

    @Test
    @DisplayName("DeviceKey: toString format")
    void testDeviceKeyToString() {
        DeviceKey key = new DeviceKey(
                DeviceKey.Type.CUDA_GPU, 2, "sm86");
        assertEquals("cuda_gpu_2_sm86", key.toString());
    }

    @Test
    @DisplayName("DeviceKey: ordinal mapping")
    void testDeviceKeyOrdinal() {
        assertEquals(0, DeviceKey.Type.CPU.getValue());
        assertEquals(1, DeviceKey.Type.CUDA_GPU.getValue());
        DeviceKey.Type t = DeviceKey.Type.fromOrdinal(3);
        assertEquals(DeviceKey.Type.VULKAN_GPU, t);
    }

    // ─── SINGLE_DEVICE strategy ───────────────────────────────────────────────

    @Test
    @DisplayName("SINGLE_DEVICE: all slots on device 0")
    void testSingleDevicePlacement() {
        DynamicShapeSlot[] slots = new DynamicShapeSlot[5];
        for (int i = 0; i < 5; i++) {
            slots[i] = slot("op_" + i, i, -1,
                    new int[]{-1}, new byte[]{0}, new String[]{"in_" + i},
                    new int[]{i}, new String[]{"out_" + i});
        }
        DynamicShapePlan plan = makePlan(slots);

        SameDiff sd = SameDiff.create();
        PlacementPlan placement = DevicePlacementPlanner.plan(plan, sd, PlacementStrategy.SINGLE_DEVICE);

        assertNotNull(placement);
        assertTrue(placement.isValid());
        assertEquals(PlacementStrategy.SINGLE_DEVICE, placement.getStrategy());
        assertArrayEquals(new int[]{0, 0, 0, 0, 0}, placement.getSlotDeviceIds());
    }

    @Test
    @DisplayName("SINGLE_DEVICE: empty plan returns valid plan")
    void testSingleDeviceEmptyPlan() {
        DynamicShapePlan plan = makePlan(new DynamicShapeSlot[0]);
        SameDiff sd = SameDiff.create();
        PlacementPlan placement = DevicePlacementPlanner.plan(plan, sd, PlacementStrategy.SINGLE_DEVICE);

        assertNotNull(placement);
        assertEquals(0, placement.getSlotDeviceIds().length);
    }

    // ─── MEMORY_FIT strategy ──────────────────────────────────────────────────

    @Test
    @DisplayName("MEMORY_FIT: deterministic slot assignment")
    void testMemoryFitPlacement() {
        DynamicShapeSlot[] slots = new DynamicShapeSlot[8];
        for (int i = 0; i < 8; i++) {
            slots[i] = slot("op_" + i, i, -1,
                    new int[]{-1}, new byte[]{0}, new String[]{"in_" + i},
                    new int[]{i}, new String[]{"out_" + i});
        }
        DynamicShapePlan plan = makePlan(slots);

        SameDiff sd = SameDiff.create();
        PlacementPlan placement = DevicePlacementPlanner.plan(plan, sd, PlacementStrategy.MEMORY_FIT);

        assertNotNull(placement);
        assertTrue(placement.isValid());
        assertEquals(8, placement.getSlotDeviceIds().length);
        // All slots should be placed (no -1 values)
        for (int dev : placement.getSlotDeviceIds()) {
            assertTrue(dev >= 0, "All slots should have non-negative device ID");
        }
    }

    // ─── PIPELINE_PARALLEL strategy ──────────────────────────────────────────

    @Test
    @DisplayName("PIPELINE_PARALLEL: produces valid plan with multiple slots")
    void testPipelineParallelPlacement() {
        DynamicShapeSlot[] slots = new DynamicShapeSlot[16];
        for (int i = 0; i < 16; i++) {
            slots[i] = slot("op_" + i, i, -1,
                    new int[]{i - 1}, new byte[]{3},
                    new String[]{i > 0 ? "out_" + (i - 1) : "input"},
                    new int[]{i}, new String[]{"out_" + i});
        }
        DynamicShapePlan plan = makePlan(slots);

        SameDiff sd = SameDiff.create();
        PlacementPlan placement = DevicePlacementPlanner.plan(plan, sd, PlacementStrategy.PIPELINE_PARALLEL);

        assertNotNull(placement);
        assertTrue(placement.isValid());
        assertEquals(16, placement.getSlotDeviceIds().length);
    }

    // ─── applyPlan ────────────────────────────────────────────────────────────

    @Test
    @DisplayName("applyPlan: sets targetDeviceId on all slots")
    void testApplyPlanSetsDeviceIds() {
        DynamicShapeSlot[] slots = new DynamicShapeSlot[4];
        for (int i = 0; i < 4; i++) {
            slots[i] = slot("op_" + i, i, -1,
                    new int[0], new byte[0], new String[0],
                    new int[]{i}, new String[]{"out_" + i});
        }
        DynamicShapePlan plan = makePlan(slots);

        int[] assignments = {0, 1, 0, 1};
        Map<Integer, Long> estimates = new HashMap<>();
        estimates.put(0, 1000L);
        estimates.put(1, 500L);
        PlacementPlan placement = new PlacementPlan(
                assignments, Map.of(), PlacementStrategy.MEMORY_FIT, estimates, true);

        DevicePlacementPlanner.applyPlan(plan, placement);

        for (int i = 0; i < 4; i++) {
            assertEquals(assignments[i], slots[i].getTargetDeviceId(),
                    "Slot " + i + " should have device " + assignments[i]);
        }
    }

    @Test
    @DisplayName("applyPlan: invalid plan does not change slots")
    void testApplyPlanInvalidNoChange() {
        DynamicShapeSlot[] slots = new DynamicShapeSlot[3];
        for (int i = 0; i < 3; i++) {
            slots[i] = slot("op_" + i, i, -1,
                    new int[0], new byte[0], new String[0],
                    new int[]{i}, new String[]{"out_" + i});
        }
        DynamicShapePlan plan = makePlan(slots);

        PlacementPlan invalid = new PlacementPlan(
                new int[0], Map.of(), PlacementStrategy.SINGLE_DEVICE, Map.of(), false);
        DevicePlacementPlanner.applyPlan(plan, invalid);

        for (DynamicShapeSlot slot : slots) {
            assertEquals(-1, slot.getTargetDeviceId(), "Slot should remain at default (-1)");
        }
    }

    @Test
    @DisplayName("applyPlan: null placement is safe")
    void testApplyPlanNullIsSafe() {
        DynamicShapeSlot[] slots = new DynamicShapeSlot[2];
        for (int i = 0; i < 2; i++) {
            slots[i] = slot("op_" + i, i, -1,
                    new int[0], new byte[0], new String[0],
                    new int[]{i}, new String[]{"out_" + i});
        }
        DynamicShapePlan plan = makePlan(slots);

        // Should not throw NPE
        DevicePlacementPlanner.applyPlan(plan, null);
    }

    // ─── Custom placement ────────────────────────────────────────────────────

    @Test
    @DisplayName("planCustom: maps variables to specified devices")
    void testPlanCustomMapping() {
        SameDiff sd = SameDiff.create();
        SDVariable w1 = sd.var("w1", Nd4j.randn(DataType.FLOAT, 4, 8));
        SDVariable w2 = sd.var("w2", Nd4j.randn(DataType.FLOAT, 8, 4));
        SDVariable in = sd.placeHolder("in", DataType.FLOAT, -1, 4);
        SDVariable h = sd.mmul("hidden", in, w1);
        SDVariable out = sd.mmul("output", h, w2);

        DynamicShapeSlot[] slots = new DynamicShapeSlot[]{
                DynamicShapeSlot.builder()
                        .opName("mmul")
                        .outputVarNames(new String[]{"hidden"})
                        .inputVarNames(new String[]{"in", "w1"})
                        .outputSlotIndices(new int[]{0})
                        .inputSourceIndices(new int[]{-1, -2})
                        .inputSourceTypes(new byte[]{2, 0})
                        .stepIndex(0)
                        .build(),
                DynamicShapeSlot.builder()
                        .opName("mmul")
                        .outputVarNames(new String[]{"output"})
                        .inputVarNames(new String[]{"hidden", "w2"})
                        .outputSlotIndices(new int[]{1})
                        .inputSourceIndices(new int[]{0, -3})
                        .inputSourceTypes(new byte[]{3, 0})
                        .stepIndex(1)
                        .build()
        };
        DynamicShapePlan plan = makePlan(slots);

        Map<String, Integer> customMap = new HashMap<>();
        customMap.put("hidden", 0);
        customMap.put("output", 1);

        PlacementPlan placement = DevicePlacementPlanner.planCustom(plan, sd, customMap);

        assertNotNull(placement);
        assertTrue(placement.isValid());
        assertEquals(PlacementStrategy.CUSTOM, placement.getStrategy());
        assertEquals(2, placement.getSlotDeviceIds().length);
        // "hidden" is slot 0's output → slot 0 should be on device 0
        assertEquals(0, placement.getSlotDeviceIds()[0]);
        // "output" is slot 1's output → slot 1 should be on device 1
        assertEquals(1, placement.getSlotDeviceIds()[1]);
    }

    @Test
    @DisplayName("planCustom: unmapped variables use default device")
    void testPlanCustomUnmappedUsesDefault() {
        SameDiff sd = SameDiff.create();
        DynamicShapeSlot[] slots = new DynamicShapeSlot[]{
                DynamicShapeSlot.builder()
                        .opName("mmul")
                        .outputVarNames(new String[]{"out"})
                        .inputVarNames(new String[]{})
                        .outputSlotIndices(new int[]{0})
                        .inputSourceIndices(new int[0])
                        .inputSourceTypes(new byte[0])
                        .stepIndex(0)
                        .build()
        };
        DynamicShapePlan plan = makePlan(slots);

        Map<String, Integer> customMap = new HashMap<>();
        customMap.put("unrelated_var", 1);

        PlacementPlan placement = DevicePlacementPlanner.planCustom(plan, sd, customMap);

        assertNotNull(placement);
        assertTrue(placement.isValid());
        // No matching variables → slot falls back to default device (0)
        assertEquals(0, placement.getSlotDeviceIds()[0]);
    }

    @Test
    @DisplayName("planCustom: empty mapping falls back to single device")
    void testPlanCustomEmptyMapFallback() {
        SameDiff sd = SameDiff.create();
        DynamicShapeSlot[] slots = new DynamicShapeSlot[]{
                DynamicShapeSlot.builder()
                        .opName("mmul")
                        .outputVarNames(new String[]{"out"})
                        .inputVarNames(new String[]{})
                        .outputSlotIndices(new int[]{0})
                        .inputSourceIndices(new int[0])
                        .inputSourceTypes(new byte[0])
                        .stepIndex(0)
                        .build()
        };
        DynamicShapePlan plan = makePlan(slots);

        PlacementPlan placement = DevicePlacementPlanner.planCustom(plan, sd, Map.of());

        assertNotNull(placement);
        assertTrue(placement.isValid());
        assertEquals(0, placement.getSlotDeviceIds()[0]);
    }

    // ─── ModelMemoryEstimator ────────────────────────────────────────────────

    @Test
    @DisplayName("ModelMemoryEstimator: constant weight size")
    void testModelMemoryEstimatorConstants() {
        SameDiff sd = SameDiff.create();
        INDArray w1 = Nd4j.randn(DataType.FLOAT, 100, 200);
        INDArray w2 = Nd4j.randn(DataType.FLOAT, 200, 50);
        sd.constant("w1", w1);
        sd.constant("w2", w2);
        sd.placeHolder("input", DataType.FLOAT, -1, 100);

        long weightMem = ModelMemoryEstimator.estimateWeightMemory(sd);
        // 100*200*4 + 200*50*4 = 80000 + 40000 = 120000
        assertEquals(120000L, weightMem);
    }

    @Test
    @DisplayName("ModelMemoryEstimator: all variable types")
    void testModelMemoryEstimatorAllTypes() {
        SameDiff sd = SameDiff.create();
        INDArray w1 = Nd4j.randn(DataType.FLOAT, 10, 20);
        sd.constant("const1", w1);
        sd.placeHolder("ph1", DataType.FLOAT, -1, 10);

        var estimates = ModelMemoryEstimator.estimateAll(sd);
        assertFalse(estimates.isEmpty());

        var constEstimate = estimates.stream()
                .filter(e -> e.getVariableName().equals("const1"))
                .findFirst().orElse(null);
        assertNotNull(constEstimate);
        assertTrue(constEstimate.isConstant());
        assertFalse(constEstimate.isActivation());
        assertEquals(10 * 20 * 4, constEstimate.getEstimatedBytes());

        var phEstimate = estimates.stream()
                .filter(e -> e.getVariableName().equals("ph1"))
                .findFirst().orElse(null);
        assertNotNull(phEstimate);
        assertFalse(phEstimate.isConstant());
        assertTrue(phEstimate.isActivation());
    }

    @Test
    @DisplayName("ModelMemoryEstimator: peak activation memory")
    void testPeakActivationEstimate() {
        DynamicShapeSlot[] slots = new DynamicShapeSlot[4];
        for (int i = 0; i < 4; i++) {
            slots[i] = slot("op_" + i, i, -1,
                    new int[0], new byte[0], new String[0],
                    new int[]{i}, new String[]{"out_" + i});
        }
        // Release slot 0 after step 1, slots 1-3 after step 3
        int[][] releaseAtStep = {{}, {0}, {}, {1, 2, 3}};
        int numOutputSlots = 4;
        DynamicShapePlan plan = new DynamicShapePlan(
                slots, numOutputSlots, releaseAtStep,
                null, new String[0], new byte[0],
                new LinkedHashSet<>(), Map.of(),
                false, null, null, null, null, null, null
        );

        long peak = ModelMemoryEstimator.estimatePeakMemory(plan, 0, 4);
        assertTrue(peak > 0, "Peak memory should be positive");
    }

    // ─── SameDiff placement fields ────────────────────────────────────────────

    @Test
    @DisplayName("SameDiff: placement strategy setter/getter")
    void testSameDiffPlacementFields() {
        SameDiff sd = SameDiff.create();

        assertNull(sd.getPlacementStrategy());
        sd.setPlacementStrategy(PlacementStrategy.MEMORY_FIT);
        assertEquals(PlacementStrategy.MEMORY_FIT, sd.getPlacementStrategy());

        assertNull(sd.getCustomDevicePlacement());
        Map<String, Integer> custom = Map.of("var1", 0, "var2", 1);
        sd.setCustomDevicePlacement(custom);
        assertEquals(custom, sd.getCustomDevicePlacement());
    }

    @Test
    @DisplayName("SameDiff: DSP auto-compile flags")
    void testSameDiffDspAutoCompileFlags() {
        SameDiff sd = SameDiff.create();

        // Default values — both default to true (see SameDiff field initializers)
        assertTrue(sd.isDspAutoCompileEnabled());
        assertTrue(sd.isDspNativeAutoCompileEnabled());

        sd.setDspNativeAutoCompileEnabled(false);
        assertFalse(sd.isDspNativeAutoCompileEnabled());
    }
}