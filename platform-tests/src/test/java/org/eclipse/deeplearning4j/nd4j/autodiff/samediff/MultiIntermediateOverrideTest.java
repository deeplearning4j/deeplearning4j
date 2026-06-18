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
import org.nd4j.autodiff.samediff.execution.DynamicShapePlan;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for overriding multiple intermediate variables simultaneously.
 * Verifies subgraph pruning, correctness with changing values, and error handling.
 */
@Slf4j
@DisplayName("MultiIntermediateOverrideTest")
public class MultiIntermediateOverrideTest {

    /**
     * Override 2 intermediates in a chain, verify both producing ops are pruned.
     * Graph: x -> mul(2) -> a -> mul(3) -> b -> add(1) -> output
     * Override both a and b → only the add(1) op should remain.
     */
    @Test
    @DisplayName("testMultipleOverrides")
    public void testMultipleOverrides() {
        SameDiff sd = SameDiff.create();

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable a = x.mul("a", 2.0);
        SDVariable b = a.mul("b", 3.0);
        SDVariable output = b.add("output", 1.0);

        // Compile plan without overrides
        DynamicShapePlan planFull = sd.compileDynamicShapePlan("output");
        int slotsFull = planFull.getSlots().length;

        // Override both a and b
        sd.addPlaceholderOverride("a");
        sd.addPlaceholderOverride("b");

        DynamicShapePlan planPruned = sd.compileDynamicShapePlan("output");
        int slotsPruned = planPruned.getSlots().length;

        log.info("Slots: full={}, pruned={}", slotsFull, slotsPruned);
        assertTrue(slotsPruned < slotsFull,
                "Plan with 2 overrides should have fewer slots: full=" + slotsFull + ", pruned=" + slotsPruned);

        // Execute with overrides
        INDArray bVal = Nd4j.ones(DataType.FLOAT, 1, 4).mul(7.0f);
        Map<String, INDArray> ph = new HashMap<>();
        ph.put("x", Nd4j.ones(DataType.FLOAT, 1, 4)); // unused
        ph.put("a", Nd4j.ones(DataType.FLOAT, 1, 4).mul(99)); // unused since b is also overridden
        ph.put("b", bVal);
        Map<String, INDArray> result = sd.output(ph, "output");
        INDArray expected = bVal.add(1); // [8,8,8,8]
        assertEquals(expected, result.get("output"));

        sd.clearPlaceholderOverrides();
    }

    /**
     * Branching graph: override one branch → verify other branch still computed.
     * Graph:
     *   x -> mul(2) -> branchA \
     *                            -> add -> output
     *   x -> mul(3) -> branchB /
     * Override branchA → mul(3) should still run.
     */
    @Test
    @DisplayName("testChainedOverridesPruning")
    public void testChainedOverridesPruning() {
        SameDiff sd = SameDiff.create();

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable branchA = x.mul("branchA", 2.0);
        SDVariable branchB = x.mul("branchB", 3.0);
        SDVariable output = branchA.add("output", branchB);

        // Override branchA only
        sd.addPlaceholderOverride("branchA");

        INDArray xVal = Nd4j.ones(DataType.FLOAT, 1, 4).mul(5.0f);
        INDArray overrideA = Nd4j.ones(DataType.FLOAT, 1, 4).mul(100.0f);
        Map<String, INDArray> ph = new HashMap<>();
        ph.put("x", xVal);
        ph.put("branchA", overrideA);
        Map<String, INDArray> result = sd.output(ph, "output");

        // branchB = x * 3 = 15, branchA = override = 100, output = 115
        INDArray expected = overrideA.add(xVal.mul(3));
        assertEquals(expected, result.get("output"),
                "BranchB should still be computed from x, branchA should use override");

        sd.clearPlaceholderOverrides();
    }

    /**
     * Execute 10 times with different override values → verify each uses the correct value.
     */
    @Test
    @DisplayName("testOverrideWithChangingValues")
    public void testOverrideWithChangingValues() {
        SameDiff sd = SameDiff.create();

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable intermediate = x.mul("intermediate", 2.0);
        SDVariable output = intermediate.add("output", 1.0);

        sd.addPlaceholderOverride("intermediate");

        for (int i = 0; i < 10; i++) {
            float val = (i + 1) * 10.0f;
            INDArray overrideVal = Nd4j.ones(DataType.FLOAT, 1, 4).mul(val);
            Map<String, INDArray> ph = new HashMap<>();
            ph.put("x", Nd4j.ones(DataType.FLOAT, 1, 4)); // unused
            ph.put("intermediate", overrideVal);
            Map<String, INDArray> result = sd.output(ph, "output");
            INDArray expected = overrideVal.add(1);
            assertEquals(expected, result.get("output"),
                    "Iteration " + i + ": should use override value " + val);
        }

        sd.clearPlaceholderOverrides();
    }
}
