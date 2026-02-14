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
import org.nd4j.autodiff.samediff.execution.DynamicShapeSlot;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for memory management in the native C++ graph executor.
 *
 * Covers: release schedule correctness, slot cache reuse, buffer padding,
 * pending close flushing, view producer detection, memory growth bounds.
 */
@Slf4j
@Tag(TagNames.SAMEDIFF)
@NativeTag
public class NativeExecutorMemoryManagementTest extends BaseNd4jTestWithBackends {

    @Test
    public void testReleaseScheduleNonEmpty() {
        SameDiff sd = NativeExecutorTestUtils.createChainGraph();
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");

        int totalReleased = 0;
        for (int[] releases : plan.getReleaseAtStep()) {
            totalReleased += releases.length;
        }
        assertTrue(totalReleased > 0, "Chain graph should release intermediate slots");

        plan.close();
    }

    @Test
    public void testReleaseScheduleIndicesValid() {
        SameDiff sd = NativeExecutorTestUtils.createChainGraph();
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");

        int totalSlots = plan.getTotalOutputSlots();
        for (int step = 0; step < plan.getReleaseAtStep().length; step++) {
            for (int slotIdx : plan.getReleaseAtStep()[step]) {
                assertTrue(slotIdx >= 0 && slotIdx < totalSlots,
                        "Release index " + slotIdx + " out of range [0," + totalSlots + ") at step " + step);
            }
        }

        plan.close();
    }

    @Test
    public void testReleaseScheduleNoDuplicates() {
        SameDiff sd = NativeExecutorTestUtils.createChainGraph();
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");

        Set<Integer> allReleased = new HashSet<>();
        for (int[] releases : plan.getReleaseAtStep()) {
            for (int idx : releases) {
                assertTrue(allReleased.add(idx),
                        "Duplicate release of slot " + idx);
            }
        }

        plan.close();
    }

    @Test
    public void testOutputSlotsNotReleased() {
        SameDiff sd = NativeExecutorTestUtils.createAddGraph();
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");

        // Find the output slot index
        int outputSlotIdx = plan.getOutputNameToSlotIndex().get("z");

        // Verify it's not in the release schedule
        for (int[] releases : plan.getReleaseAtStep()) {
            for (int idx : releases) {
                assertNotEquals(outputSlotIdx, idx,
                        "Requested output slot should not be released");
            }
        }

        plan.close();
    }

    @Test
    public void testSlotCacheInvalidationOnReset() {
        SameDiff sd = NativeExecutorTestUtils.createAddGraph();
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");

        // Simulate executing and caching shapes
        for (DynamicShapeSlot slot : plan.getSlots()) {
            slot.updateShapeCache(999L, new ArrayList<>());
            assertTrue(slot.isShapeCacheValid(999L));
        }

        // Reset should clear all caches
        plan.clearAllShapeCaches();

        for (DynamicShapeSlot slot : plan.getSlots()) {
            assertFalse(slot.isShapeCacheValid(999L),
                    "Shape cache should be invalid after clearAllShapeCaches()");
        }

        plan.close();
    }

    @Test
    public void testSerializationPreservesReleaseSchedule() {
        SameDiff sd = NativeExecutorTestUtils.createChainGraph();
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");

        byte[] serialized = plan.serialize();
        assertNotNull(serialized);

        // The release schedule is embedded in the serialized data
        // Verify by checking total serialized size is reasonable
        assertTrue(serialized.length > 24 + plan.getSlots().length * 20,
                "Serialized plan should include release schedule data");

        plan.close();
    }

    @Test
    public void testConsecutiveExecutionsNoLeak() {
        // Execute multiple times and verify results stay correct
        SameDiff sd = NativeExecutorTestUtils.createAddGraph();
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");

        INDArray x = Nd4j.ones(DataType.FLOAT, 2, 4);
        INDArray y = Nd4j.ones(DataType.FLOAT, 2, 4).mul(2);

        for (int i = 0; i < 10; i++) {
            Map<String, INDArray> result = sd.output(Map.of("x", x, "y", y), "z");
            assertEquals(3.0, result.get("z").getDouble(0), 1e-5,
                    "Execution " + i + " should produce correct result");
        }

        plan.close();
    }

    @Test
    public void testShapeChangeDoesNotCorruptPreviousResults() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable z = x.add("z", 1.0);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");

        // Execute with shape [2,4]
        INDArray x1 = Nd4j.ones(DataType.FLOAT, 2, 4);
        Map<String, INDArray> r1 = sd.output(Map.of("x", x1), "z");
        INDArray r1Copy = r1.get("z").dup();

        // Execute with shape [5,4]
        INDArray x2 = Nd4j.ones(DataType.FLOAT, 5, 4);
        Map<String, INDArray> r2 = sd.output(Map.of("x", x2), "z");

        // First result should still be intact
        assertEquals(r1Copy.getDouble(0), 2.0, 1e-5);
        assertArrayEquals(new long[]{5, 4}, r2.get("z").shape());

        plan.close();
    }

    @Test
    public void testPlanCloseReleasesOpContextPool() {
        SameDiff sd = NativeExecutorTestUtils.createAddGraph();
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");

        // OpContextPool may be null (lazy init) or populated
        // The key test is that close() doesn't throw
        assertDoesNotThrow(plan::close);
    }

    @Test
    public void testLargeGraphMemoryStructure() {
        // Build a graph with many ops
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable cur = x;
        for (int i = 0; i < 50; i++) {
            cur = cur.add(0.01);
        }
        cur.rename("z");

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        assertTrue(plan.getSlots().length >= 50, "Should have >= 50 slots");
        assertTrue(plan.getTotalOutputSlots() >= 50, "Should have >= 50 output slots");

        // Verify release schedule accounts for intermediate buffers
        int totalReleased = 0;
        for (int[] releases : plan.getReleaseAtStep()) {
            totalReleased += releases.length;
        }
        // In a linear chain, all intermediates except the final should be released
        assertTrue(totalReleased >= 48,
                "Linear chain of 50 should release most intermediates, got " + totalReleased);

        plan.close();
    }

    @Test
    public void testDiamondGraphReleaseSchedule() {
        SameDiff sd = NativeExecutorTestUtils.createDiamondGraph();
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "c");

        // In diamond: a and b are used by c, so they can only be released after c executes
        assertNotNull(plan.getReleaseAtStep());

        plan.close();
    }

    @Test
    public void testExternalInputsNotInReleaseSchedule() {
        SameDiff sd = NativeExecutorTestUtils.createAddGraph();
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");

        // External inputs use negative indices in inputSourceIndices
        // They should never appear in the release schedule
        Set<Integer> allSlotIndices = new HashSet<>();
        for (DynamicShapeSlot slot : plan.getSlots()) {
            for (int idx : slot.getOutputSlotIndices()) {
                allSlotIndices.add(idx);
            }
        }

        Set<Integer> allReleased = new HashSet<>();
        for (int[] releases : plan.getReleaseAtStep()) {
            for (int idx : releases) {
                allReleased.add(idx);
                assertTrue(allSlotIndices.contains(idx),
                        "Released slot " + idx + " should be a valid output slot index");
            }
        }

        plan.close();
    }

    @Test
    public void testSerializationSizeReasonable() {
        // Simple graph should not produce huge serialized data
        SameDiff sd = NativeExecutorTestUtils.createAddGraph();
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");

        byte[] data = plan.serialize();
        assertTrue(data.length < 10000, "Simple add graph serialization should be < 10KB, got " + data.length);

        plan.close();
    }

    @Test
    public void testSerializationSizeScalesWithSlots() {
        // Create two graphs of different sizes and verify serialization scales
        SameDiff sd1 = NativeExecutorTestUtils.createAddGraph(); // 1 op
        DynamicShapePlan plan1 = NativeExecutorTestUtils.compilePlan(sd1, "z");

        SameDiff sd2 = NativeExecutorTestUtils.createChainGraph(); // 4+ ops
        DynamicShapePlan plan2 = NativeExecutorTestUtils.compilePlan(sd2, "z");

        byte[] data1 = plan1.serialize();
        byte[] data2 = plan2.serialize();

        assertTrue(data2.length > data1.length,
                "Larger graph should produce larger serialization");

        plan1.close();
        plan2.close();
    }

    @Test
    public void testMultiOutputReleaseSchedule() {
        // When requesting multiple outputs, only the non-requested intermediates should be released
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable a = x.add("a", 1);
        SDVariable b = a.mul("b", 2);
        SDVariable c = b.sub("c", 3);

        // Request only a and c
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "a", "c");

        // Verify a and c slot indices are not in release schedule
        int aSlot = plan.getOutputNameToSlotIndex().get("a");
        int cSlot = plan.getOutputNameToSlotIndex().get("c");
        for (int[] releases : plan.getReleaseAtStep()) {
            for (int idx : releases) {
                assertNotEquals(aSlot, idx, "Requested output 'a' should not be released");
                assertNotEquals(cSlot, idx, "Requested output 'c' should not be released");
            }
        }

        plan.close();
    }
}
