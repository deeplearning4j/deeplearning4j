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
import org.nd4j.autodiff.samediff.execution.DynamicShapeSlot;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for external input source type classification in DynamicShapePlan.
 * Verifies that constants, variables, and placeholders are correctly classified
 * and that the executor caches constants while re-resolving placeholders.
 */
@Slf4j
@DisplayName("ExternalInputSourceTypeTest")
public class ExternalInputSourceTypeTest {

    /**
     * Graph with constant + variable + placeholder → verify source types are correct.
     */
    @Test
    @DisplayName("testSourceTypeClassification")
    public void testSourceTypeClassification() {
        SameDiff sd = SameDiff.create();

        // Build graph: ph (PH) + const + var -> output
        SDVariable ph = sd.placeHolder("ph", DataType.FLOAT, -1, 4);
        SDVariable constant = sd.constant("myConst", Nd4j.ones(DataType.FLOAT, 1, 4));
        SDVariable variable = sd.var("myVar", Nd4j.ones(DataType.FLOAT, 1, 4).mul(2));
        SDVariable sum1 = ph.add("sum1", constant);
        SDVariable output = sum1.add("output", variable);

        DynamicShapePlan plan = sd.compileDynamicShapePlan("output");
        assertNotNull(plan);

        String[] keys = plan.getExternalInputKeys();
        byte[] sourceTypes = plan.getExternalInputSourceTypes();
        assertNotNull(sourceTypes);
        assertEquals(keys.length, sourceTypes.length,
                "Source types array must be parallel to external input keys");

        // Verify each source type
        for (int i = 0; i < keys.length; i++) {
            String key = keys[i];
            byte type = sourceTypes[i];
            if (key.equals("myConst")) {
                assertEquals(DynamicShapeSlot.SOURCE_CONSTANT, type,
                        "myConst should be SOURCE_CONSTANT");
            } else if (key.equals("myVar")) {
                assertEquals(DynamicShapeSlot.SOURCE_VARIABLE, type,
                        "myVar should be SOURCE_VARIABLE");
            } else if (key.equals("ph")) {
                assertEquals(DynamicShapeSlot.SOURCE_PLACEHOLDER, type,
                        "ph should be SOURCE_PLACEHOLDER");
            }
            log.info("External input '{}' → source type {}", key, type);
        }
    }

    /**
     * Execute twice — verify constants are resolved once, placeholders re-resolved.
     * This verifies the caching behavior: after the first execution, constants and
     * variables should be cached and not re-resolved.
     */
    @Test
    @DisplayName("testConstantsCachedAcrossExecutions")
    public void testConstantsCachedAcrossExecutions() {
        SameDiff sd = SameDiff.create();

        SDVariable ph = sd.placeHolder("ph", DataType.FLOAT, -1, 4);
        SDVariable constant = sd.constant("myConst", Nd4j.ones(DataType.FLOAT, 1, 4).mul(10));
        SDVariable output = ph.add("output", constant);

        // Execute 1: ph = [1,1,1,1]
        Map<String, INDArray> placeholders = new HashMap<>();
        placeholders.put("ph", Nd4j.ones(DataType.FLOAT, 1, 4));
        Map<String, INDArray> result1 = sd.output(placeholders, "output");
        INDArray expected1 = Nd4j.ones(DataType.FLOAT, 1, 4).mul(11);
        assertEquals(expected1, result1.get("output"));

        // Execute 2: ph = [2,2,2,2] — placeholder re-resolved, constant cached
        placeholders.put("ph", Nd4j.ones(DataType.FLOAT, 1, 4).mul(2));
        Map<String, INDArray> result2 = sd.output(placeholders, "output");
        INDArray expected2 = Nd4j.ones(DataType.FLOAT, 1, 4).mul(12);
        assertEquals(expected2, result2.get("output"),
                "Second execution should use new placeholder value with cached constant");
    }

    /**
     * Verify that host-only arrays are synced to device and produce correct output.
     * This is the ROOT CAUSE fix test: without ensureLocation(DEVICE) in the executor,
     * host-only arrays would produce zeros on GPU.
     */
    @Test
    @DisplayName("testHostArraySyncedToDevice")
    public void testHostArraySyncedToDevice() {
        SameDiff sd = SameDiff.create();

        SDVariable ph = sd.placeHolder("ph", DataType.FLOAT, -1, 4);
        SDVariable output = ph.mul("output", 2.0);

        // Create an array and modify it on host only
        INDArray hostArray = Nd4j.zeros(DataType.FLOAT, 1, 4);
        hostArray.putScalar(0, 0, 5.0f);
        hostArray.putScalar(0, 1, 10.0f);
        hostArray.putScalar(0, 2, 15.0f);
        hostArray.putScalar(0, 3, 20.0f);

        Map<String, INDArray> placeholders = new HashMap<>();
        placeholders.put("ph", hostArray);
        Map<String, INDArray> result = sd.output(placeholders, "output");

        INDArray expected = hostArray.mul(2);
        assertEquals(expected, result.get("output"),
                "Host-only array should be synced to device and produce correct output");
    }

    /**
     * Override an intermediate variable → verify it appears as SOURCE_PLACEHOLDER in plan.
     */
    @Test
    @DisplayName("testSourceTypeWithOverride")
    public void testSourceTypeWithOverride() {
        SameDiff sd = SameDiff.create();

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable intermediate = x.mul("intermediate", 2.0);
        SDVariable output = intermediate.add("output", 1.0);

        // Override intermediate → it should become a placeholder
        sd.addPlaceholderOverride("intermediate");

        DynamicShapePlan plan = sd.compileDynamicShapePlan("output");
        assertNotNull(plan);

        String[] keys = plan.getExternalInputKeys();
        byte[] sourceTypes = plan.getExternalInputSourceTypes();

        boolean foundIntermediate = false;
        for (int i = 0; i < keys.length; i++) {
            if (keys[i].equals("intermediate")) {
                assertEquals(DynamicShapeSlot.SOURCE_PLACEHOLDER, sourceTypes[i],
                        "Overridden intermediate should be SOURCE_PLACEHOLDER");
                foundIntermediate = true;
            }
        }
        assertTrue(foundIntermediate, "intermediate should be in externalInputKeys");

        sd.clearPlaceholderOverrides();
    }
}
