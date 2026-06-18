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
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlan;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the placeholder override API in SameDiff.
 *
 * The placeholder override mechanism allows intermediate ARRAY variables to be treated
 * as PLACEHOLDERs for DSP compilation, pruning the producing subgraph. This is used
 * for bias override in attention models (attn_mask_reformat bypass).
 */
@Slf4j
@DisplayName("PlaceholderOverrideTest")
public class PlaceholderOverrideTest {

    /**
     * Basic test: build graph x -> mul(2) -> intermediate -> add(1) -> output.
     * Without override: output = 2*x + 1.
     * With override: output = customValue + 1 (mul op bypassed).
     */
    @Test
    @DisplayName("testPlaceholderOverrideBasic")
    public void testPlaceholderOverrideBasic() {
        SameDiff sd = SameDiff.create();

        // Build graph: x (PH) -> mul(2) -> intermediate -> add(1) -> output
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable two = sd.constant("two", Nd4j.scalar(DataType.FLOAT, 2.0f));
        SDVariable intermediate = x.mul("intermediate", two);
        SDVariable one = sd.constant("one", Nd4j.scalar(DataType.FLOAT, 1.0f));
        SDVariable output = intermediate.add("output", one);

        // Run without override: output = 2*x + 1
        INDArray xVal = Nd4j.ones(DataType.FLOAT, 1, 4).mul(3.0f); // x = [3,3,3,3]
        Map<String, INDArray> ph = new HashMap<>();
        ph.put("x", xVal);
        Map<String, INDArray> result = sd.output(ph, "output");
        INDArray expected = xVal.mul(2).add(1); // [7,7,7,7]
        assertEquals(expected, result.get("output"), "Without override: output should be 2*x + 1");

        // Add placeholder override for "intermediate"
        sd.addPlaceholderOverride("intermediate");
        assertEquals(VariableType.PLACEHOLDER, sd.getVariable("intermediate").getVariableType(),
                "intermediate should now be PLACEHOLDER");
        assertTrue(sd.getPlaceholderOverrides().contains("intermediate"));

        // Run with override: output = customValue + 1
        INDArray customValue = Nd4j.ones(DataType.FLOAT, 1, 4).mul(10.0f); // [10,10,10,10]
        ph.put("intermediate", customValue);
        result = sd.output(ph, "output");
        INDArray expectedOverride = customValue.add(1); // [11,11,11,11]
        assertEquals(expectedOverride, result.get("output"),
                "With override: output should be customValue + 1");

        // Clean up
        sd.clearPlaceholderOverrides();
    }

    /**
     * Test that adding a placeholder override compiles a DSP plan where the overridden
     * variable appears in externalInputKeys (as a placeholder), and the producing
     * subgraph is pruned from the plan.
     */
    @Test
    @DisplayName("testPlaceholderOverrideWithDSP")
    public void testPlaceholderOverrideWithDSP() {
        SameDiff sd = SameDiff.create();

        // Build graph: x (PH) -> mul(2) -> intermediate -> add(1) -> output
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable two = sd.constant("two", Nd4j.scalar(DataType.FLOAT, 2.0f));
        SDVariable intermediate = x.mul("intermediate", two);
        SDVariable one = sd.constant("one", Nd4j.scalar(DataType.FLOAT, 1.0f));
        SDVariable output = intermediate.add("output", one);

        // Compile DSP plan WITHOUT override — should have mul op in the plan
        DynamicShapePlan planBefore = sd.compileDynamicShapePlan("output");
        assertNotNull(planBefore, "Plan should compile");
        int slotsBefore = planBefore.getSlots().length;
        log.info("Plan before override: {} slots", slotsBefore);

        // Add override and recompile
        sd.addPlaceholderOverride("intermediate");
        DynamicShapePlan planAfter = sd.compileDynamicShapePlan("output");
        assertNotNull(planAfter, "Plan with override should compile");
        int slotsAfter = planAfter.getSlots().length;
        log.info("Plan after override: {} slots", slotsAfter);

        // The plan with override should have fewer slots (mul op is pruned)
        assertTrue(slotsAfter < slotsBefore,
                "Plan with override should have fewer slots (pruned subgraph): before=" + slotsBefore + ", after=" + slotsAfter);

        // Verify the overridden variable is in externalInputKeys
        String[] extKeys = planAfter.getExternalInputKeys();
        boolean foundOverride = Arrays.asList(extKeys).contains("intermediate");
        assertTrue(foundOverride, "externalInputKeys should contain 'intermediate'");

        // Execute with override — should produce correct results
        INDArray customValue = Nd4j.ones(DataType.FLOAT, 1, 4).mul(5.0f);
        Map<String, INDArray> ph = new HashMap<>();
        ph.put("x", Nd4j.ones(DataType.FLOAT, 1, 4)); // not used due to override
        ph.put("intermediate", customValue);
        Map<String, INDArray> result = sd.output(ph, "output");
        INDArray expected = customValue.add(1); // [6,6,6,6]
        assertEquals(expected, result.get("output"));

        // Remove override — should restore original behavior
        sd.removePlaceholderOverride("intermediate");
        assertEquals(VariableType.ARRAY, sd.getVariable("intermediate").getVariableType(),
                "intermediate should be restored to ARRAY");
        assertTrue(sd.getPlaceholderOverrides().isEmpty());

        // Recompile and verify original plan size
        DynamicShapePlan planRestored = sd.compileDynamicShapePlan("output");
        assertEquals(slotsBefore, planRestored.getSlots().length,
                "Restored plan should have same slot count as original");

        sd.clearPlaceholderOverrides();
    }

    /**
     * Test that addPlaceholderOverride invalidates the DSP plan cache,
     * and clearPlaceholderOverrides also invalidates it.
     */
    @Test
    @DisplayName("testPlaceholderOverrideInvalidatesCache")
    public void testPlaceholderOverrideInvalidatesCache() {
        SameDiff sd = SameDiff.create();

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable two = sd.constant("two", Nd4j.scalar(DataType.FLOAT, 2.0f));
        SDVariable intermediate = x.mul("intermediate", two);
        SDVariable one = sd.constant("one", Nd4j.scalar(DataType.FLOAT, 1.0f));
        SDVariable output = intermediate.add("output", one);

        // Compile initial plan — should cache it
        DynamicShapePlan plan1 = sd.compileDynamicShapePlan("output");
        assertNotNull(plan1);

        // Add override — cache should be cleared
        sd.addPlaceholderOverride("intermediate");

        // Compile again — should get a different plan (fewer slots)
        DynamicShapePlan plan2 = sd.compileDynamicShapePlan("output");
        assertNotNull(plan2);
        assertNotSame(plan1, plan2, "Plan should be recompiled after override");
        assertTrue(plan2.getSlots().length < plan1.getSlots().length,
                "New plan should have fewer slots");

        // Clear overrides — cache should be cleared again
        sd.clearPlaceholderOverrides();

        // Compile again — should match original
        DynamicShapePlan plan3 = sd.compileDynamicShapePlan("output");
        assertNotNull(plan3);
        assertEquals(plan1.getSlots().length, plan3.getSlots().length,
                "Plan after clearing overrides should match original");
    }

    /**
     * Test that after adding and clearing overrides, the model produces
     * correct results in both modes.
     */
    @Test
    @DisplayName("testPlaceholderOverridePreservesModelReuse")
    public void testPlaceholderOverridePreservesModelReuse() {
        SameDiff sd = SameDiff.create();

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable two = sd.constant("two", Nd4j.scalar(DataType.FLOAT, 2.0f));
        SDVariable intermediate = x.mul("intermediate", two);
        SDVariable one = sd.constant("one", Nd4j.scalar(DataType.FLOAT, 1.0f));
        SDVariable output = intermediate.add("output", one);

        INDArray xVal = Nd4j.ones(DataType.FLOAT, 1, 4).mul(3.0f);

        // Run 1: normal mode
        Map<String, INDArray> ph = new HashMap<>();
        ph.put("x", xVal);
        Map<String, INDArray> result1 = sd.output(ph, "output");
        INDArray expected = xVal.mul(2).add(1); // [7,7,7,7]
        assertEquals(expected, result1.get("output"));

        // Run 2: with override
        sd.addPlaceholderOverride("intermediate");
        INDArray customValue = Nd4j.ones(DataType.FLOAT, 1, 4).mul(10.0f);
        ph.put("intermediate", customValue);
        Map<String, INDArray> result2 = sd.output(ph, "output");
        assertEquals(customValue.add(1), result2.get("output"));

        // Run 3: clear override, back to normal
        sd.clearPlaceholderOverrides();
        ph.remove("intermediate");
        Map<String, INDArray> result3 = sd.output(ph, "output");
        assertEquals(expected, result3.get("output"),
                "After clearing overrides, original computation should work correctly");

        // Verify variable type is fully restored
        assertEquals(VariableType.ARRAY, sd.getVariable("intermediate").getVariableType());
    }

    /**
     * Test that adding override for a non-existent variable throws.
     */
    @Test
    @DisplayName("testPlaceholderOverrideNonExistentVariable")
    public void testPlaceholderOverrideNonExistentVariable() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1);
        SDVariable output = x.add("output", 1.0);

        assertThrows(IllegalArgumentException.class, () -> {
            sd.addPlaceholderOverride("nonexistent");
        });
    }

    /**
     * Test that removing a non-overridden variable is a no-op.
     */
    @Test
    @DisplayName("testRemoveNonOverriddenIsNoOp")
    public void testRemoveNonOverriddenIsNoOp() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1);
        SDVariable output = x.add("output", 1.0);

        // Should not throw
        sd.removePlaceholderOverride("x");
        sd.clearPlaceholderOverrides();
    }

    /**
     * Test that double-adding an override is idempotent.
     */
    @Test
    @DisplayName("testDoubleAddIsIdempotent")
    public void testDoubleAddIsIdempotent() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable intermediate = x.mul("intermediate", 2.0);
        SDVariable output = intermediate.add("output", 1.0);

        sd.addPlaceholderOverride("intermediate");
        assertEquals(1, sd.getPlaceholderOverrides().size());

        // Double-add should not change anything
        sd.addPlaceholderOverride("intermediate");
        assertEquals(1, sd.getPlaceholderOverrides().size());
        assertEquals(VariableType.PLACEHOLDER, sd.getVariable("intermediate").getVariableType());

        // Remove should still restore original type
        sd.removePlaceholderOverride("intermediate");
        assertEquals(VariableType.ARRAY, sd.getVariable("intermediate").getVariableType());
    }

    /**
     * Test that host-only arrays work correctly with DSP when used as overrides.
     * This verifies the ensureLocation(DEVICE) fix in resolveExternalInputs.
     */
    @Test
    @DisplayName("testOverrideWithHostOnlyArray")
    public void testOverrideWithHostOnlyArray() {
        SameDiff sd = SameDiff.create();

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable intermediate = x.mul("intermediate", 2.0);
        SDVariable output = intermediate.add("output", 1.0);

        sd.addPlaceholderOverride("intermediate");

        // Create array modified on host only (putScalar uses host buffer)
        INDArray hostArray = Nd4j.zeros(DataType.FLOAT, 1, 4);
        hostArray.putScalar(0, 0, 42.0f);
        hostArray.putScalar(0, 1, 43.0f);
        hostArray.putScalar(0, 2, 44.0f);
        hostArray.putScalar(0, 3, 45.0f);

        Map<String, INDArray> ph = new HashMap<>();
        ph.put("x", Nd4j.ones(DataType.FLOAT, 1, 4)); // unused
        ph.put("intermediate", hostArray);
        Map<String, INDArray> result = sd.output(ph, "output");

        INDArray expected = hostArray.add(1);
        assertEquals(expected, result.get("output"),
                "Host-only override array should produce correct output (not zeros)");

        sd.clearPlaceholderOverrides();
    }

    /**
     * Test that overrides work correctly across 10 consecutive executions.
     */
    @Test
    @DisplayName("testOverridePreservesAcrossMultipleExecutions")
    public void testOverridePreservesAcrossMultipleExecutions() {
        SameDiff sd = SameDiff.create();

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable intermediate = x.mul("intermediate", 2.0);
        SDVariable output = intermediate.add("output", 1.0);

        sd.addPlaceholderOverride("intermediate");

        INDArray overrideVal = Nd4j.ones(DataType.FLOAT, 1, 4).mul(50.0f);
        for (int i = 0; i < 10; i++) {
            Map<String, INDArray> ph = new HashMap<>();
            ph.put("x", Nd4j.ones(DataType.FLOAT, 1, 4));
            ph.put("intermediate", overrideVal);
            Map<String, INDArray> result = sd.output(ph, "output");
            INDArray expected = overrideVal.add(1);
            assertEquals(expected, result.get("output"),
                    "Execution " + i + ": override should produce consistent results");
        }

        sd.clearPlaceholderOverrides();
    }

    /**
     * Test that different values work correctly on each execution.
     */
    @Test
    @DisplayName("testOverrideWithDifferentValuesEachExecution")
    public void testOverrideWithDifferentValuesEachExecution() {
        SameDiff sd = SameDiff.create();

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable intermediate = x.mul("intermediate", 2.0);
        SDVariable output = intermediate.add("output", 1.0);

        sd.addPlaceholderOverride("intermediate");

        for (int i = 0; i < 10; i++) {
            float val = (i + 1) * 7.0f;
            INDArray overrideVal = Nd4j.ones(DataType.FLOAT, 1, 4).mul(val);
            Map<String, INDArray> ph = new HashMap<>();
            ph.put("x", Nd4j.ones(DataType.FLOAT, 1, 4));
            ph.put("intermediate", overrideVal);
            Map<String, INDArray> result = sd.output(ph, "output");
            INDArray expected = overrideVal.add(1);
            assertEquals(expected, result.get("output"),
                    "Execution " + i + ": should use value " + val);
        }

        sd.clearPlaceholderOverrides();
    }
}
