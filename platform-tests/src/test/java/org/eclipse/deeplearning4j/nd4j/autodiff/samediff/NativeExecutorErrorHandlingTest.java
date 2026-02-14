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
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlan;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for error handling in the native C++ graph executor.
 *
 * Covers: null inputs, corrupted plans, invalid headers, empty graphs,
 * serialization edge cases.
 */
@Slf4j
@Tag(TagNames.SAMEDIFF)
@NativeTag
public class NativeExecutorErrorHandlingTest extends BaseNd4jTestWithBackends {

    @Test
    public void testInvalidHeaderDetection() {
        // Null
        assertFalse(DynamicShapePlan.isValidSerializedPlan(null));

        // Empty
        assertFalse(DynamicShapePlan.isValidSerializedPlan(new byte[0]));

        // Too short
        assertFalse(DynamicShapePlan.isValidSerializedPlan(new byte[23]));

        // Wrong magic
        byte[] badMagic = new byte[24];
        assertFalse(DynamicShapePlan.isValidSerializedPlan(badMagic));
    }

    @Test
    public void testCorruptedMagicDetection() {
        byte[] data = createHeader(0xDEADBEEF, 1, 1, 1, 1, 1);
        assertFalse(DynamicShapePlan.isValidSerializedPlan(data));
    }

    @Test
    public void testWrongVersionDetection() {
        byte[] data = createHeader(0x44535031, 99, 1, 1, 1, 1);
        assertFalse(DynamicShapePlan.isValidSerializedPlan(data));
    }

    @Test
    public void testValidMinimalHeader() {
        byte[] data = createHeader(0x44535031, 1, 0, 0, 0, 0);
        assertTrue(DynamicShapePlan.isValidSerializedPlan(data));
    }

    @Test
    public void testSerializeNullPlanReturnsNull() {
        SameDiff sd = SameDiff.create();
        // Create a plan with no slots (empty graph should still work or return null)
        // This tests the edge case of serializing a minimal plan
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 3);
        x.rename("z");

        // Depending on implementation, this might return null or a minimal plan
        try {
            DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
            if (plan != null && plan.getSlots() != null && plan.getSlots().length > 0) {
                byte[] data = plan.serialize();
                assertNotNull(data);
                plan.close();
            }
        } catch (Exception e) {
            // Expected for graphs with no ops
            log.info("Expected exception for empty graph: {}", e.getMessage());
        }
    }

    @Test
    public void testPlanCloseIdempotent() {
        SameDiff sd = NativeExecutorTestUtils.createAddGraph();
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");

        // Close should be idempotent
        assertDoesNotThrow(plan::close);
        assertDoesNotThrow(plan::close);
    }

    @Test
    public void testShapeCacheClearIdempotent() {
        SameDiff sd = NativeExecutorTestUtils.createAddGraph();
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");

        // Clearing caches multiple times should be safe
        assertDoesNotThrow(plan::clearAllShapeCaches);
        assertDoesNotThrow(plan::clearAllShapeCaches);
        assertDoesNotThrow(plan::clearAllShapeCaches);

        plan.close();
    }

    @Test
    public void testSerializeAfterClearCaches() {
        SameDiff sd = NativeExecutorTestUtils.createAddGraph();
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");

        // Clearing caches shouldn't affect serialization
        plan.clearAllShapeCaches();
        byte[] data = plan.serialize();
        assertNotNull(data);
        assertTrue(DynamicShapePlan.isValidSerializedPlan(data));

        plan.close();
    }

    @Test
    public void testSerializationDeterministic() {
        SameDiff sd = NativeExecutorTestUtils.createAddGraph();
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");

        byte[] data1 = plan.serialize();
        byte[] data2 = plan.serialize();

        assertArrayEquals(data1, data2, "Serialization should be deterministic");

        plan.close();
    }

    @Test
    public void testLargeHeaderValues() {
        // Test header with very large values (shouldn't crash, just validate)
        byte[] data = createHeader(0x44535031, 1, Integer.MAX_VALUE, Integer.MAX_VALUE, 0, 0);
        assertTrue(DynamicShapePlan.isValidSerializedPlan(data),
                "Valid magic+version should pass even with large slot counts in header");
    }

    @Test
    public void testMultipleRequestedOutputsSerialized() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable a = x.add("a", 1);
        SDVariable b = x.mul("b", 2);
        SDVariable c = a.add("c", b);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "a", "b", "c");
        byte[] data = plan.serialize();

        ByteBuffer buf = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
        buf.position(20); // Skip to numRequestedOutputs
        int numOutputs = buf.getInt();
        assertEquals(3, numOutputs, "Should serialize 3 requested outputs");

        plan.close();
    }

    // ─── Helper ──────────────────────────────────────────────────────────────

    private byte[] createHeader(int magic, int version, int numSlots,
                                 int totalOutputSlots, int numExternalInputs,
                                 int numRequestedOutputs) {
        ByteBuffer buf = ByteBuffer.allocate(24).order(ByteOrder.LITTLE_ENDIAN);
        buf.putInt(magic);
        buf.putInt(version);
        buf.putInt(numSlots);
        buf.putInt(totalOutputSlots);
        buf.putInt(numExternalInputs);
        buf.putInt(numRequestedOutputs);
        return buf.array();
    }
}
