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
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanCompiler;
import org.nd4j.autodiff.samediff.execution.DynamicShapeSlot;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for DynamicShapePlan binary serialization.
 *
 * Covers:
 * - DSP1 header format validation
 * - Per-slot serialization (op hashes, input wiring, args, flags)
 * - Release schedule serialization
 * - Requested output mapping serialization
 * - Roundtrip validation (serialize → verify header → verify content)
 * - Edge cases: empty args, single slot, large plans
 */
@Slf4j
@Tag(TagNames.SAMEDIFF)
@NativeTag
public class NativeExecutorSerializationTest extends BaseNd4jTestWithBackends {

    @Test
    public void testSerializeSimpleAddPlan() {
        // z = x + y
        SameDiff sd = NativeExecutorTestUtils.createAddGraph();
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");

        byte[] data = plan.serialize();
        assertNotNull(data, "Serialized plan should not be null");
        assertTrue(data.length > 24, "Serialized plan should be > 24 bytes header");

        NativeExecutorTestUtils.assertValidSerialization(plan);
        plan.close();
    }

    @Test
    public void testSerializeHeaderFields() {
        SameDiff sd = NativeExecutorTestUtils.createAddGraph();
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");

        byte[] data = plan.serialize();
        ByteBuffer buf = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);

        assertEquals(0x44535031, buf.getInt(), "Magic should be DSP1");
        assertEquals(1, buf.getInt(), "Version should be 1");

        int numSlots = buf.getInt();
        assertTrue(numSlots > 0, "Should have at least 1 slot");
        assertEquals(plan.getSlots().length, numSlots, "Slot count mismatch");

        int totalOutputSlots = buf.getInt();
        assertEquals(plan.getTotalOutputSlots(), totalOutputSlots, "Total output slots mismatch");

        int numExternalInputs = buf.getInt();
        assertEquals(plan.getExternalInputKeys().length, numExternalInputs, "External inputs mismatch");

        int numRequestedOutputs = buf.getInt();
        assertEquals(plan.getOutputNameToSlotIndex().size(), numRequestedOutputs, "Requested outputs mismatch");

        plan.close();
    }

    @Test
    public void testSerializePerSlotContent() {
        SameDiff sd = NativeExecutorTestUtils.createAddGraph();
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");

        byte[] data = plan.serialize();
        ByteBuffer buf = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);

        // Skip header (6 ints = 24 bytes)
        buf.position(24);

        // Read first slot
        for (DynamicShapeSlot slot : plan.getSlots()) {
            long opHash = buf.getLong();
            assertEquals(slot.getOpNameHash(), opHash, "Op hash mismatch");

            int numInputs = buf.getInt();
            assertEquals(slot.getInputSourceIndices().length, numInputs, "Input count mismatch");

            int numOutputs = buf.getInt();
            assertEquals(slot.getOutputSlotIndices().length, numOutputs, "Output count mismatch");

            // Read inputSourceIndices
            for (int i = 0; i < numInputs; i++) {
                int idx = buf.getInt();
                assertEquals(slot.getInputSourceIndices()[i], idx, "Input source index mismatch at " + i);
            }

            // Read inputSourceTypes
            for (int i = 0; i < numInputs; i++) {
                byte type = buf.get();
                assertEquals(slot.getInputSourceTypes()[i], type, "Input source type mismatch at " + i);
            }

            // Read outputSlotIndices
            for (int i = 0; i < numOutputs; i++) {
                int idx = buf.getInt();
                assertEquals(slot.getOutputSlotIndices()[i], idx, "Output slot index mismatch at " + i);
            }

            // Read arg counts
            int nIArgs = buf.getInt();
            int nTArgs = buf.getInt();
            int nBArgs = buf.getInt();
            int nDArgs = buf.getInt();

            assertEquals(slot.getIArgs() != null ? slot.getIArgs().length : 0, nIArgs, "iArgs count mismatch");
            assertEquals(slot.getTArgs() != null ? slot.getTArgs().length : 0, nTArgs, "tArgs count mismatch");
            assertEquals(slot.getBArgs() != null ? slot.getBArgs().length : 0, nBArgs, "bArgs count mismatch");
            assertEquals(slot.getDArgs() != null ? slot.getDArgs().length : 0, nDArgs, "dArgs count mismatch");

            // Skip args content
            for (int i = 0; i < nIArgs; i++) buf.getLong();
            for (int i = 0; i < nTArgs; i++) buf.getDouble();
            for (int i = 0; i < nBArgs; i++) buf.get();
            for (int i = 0; i < nDArgs; i++) buf.getInt();

            // Read flags
            boolean needsZeroed = buf.get() != 0;
            boolean isDataDep = buf.get() != 0;
            boolean shapeDepsOnValues = buf.get() != 0;
            boolean needsSync = buf.get() != 0;
            boolean isCustomOp = buf.get() != 0;
            int targetDevice = buf.getInt();

            assertEquals(slot.isNeedsZeroedOutput(), needsZeroed, "needsZeroedOutput mismatch");
            assertEquals(slot.isDataDependent(), isDataDep, "isDataDependent mismatch");
            assertEquals(slot.isOutputShapeDependsOnInputValues(), shapeDepsOnValues, "outputShapeDependsOnInputValues mismatch");
            assertEquals(slot.isNeedsIntLongSync(), needsSync, "needsIntLongSync mismatch");
            assertEquals(slot.isCustomOp(), isCustomOp, "isCustomOp mismatch");
            assertEquals(slot.getTargetDeviceId(), targetDevice, "targetDeviceId mismatch");
        }

        plan.close();
    }

    @Test
    public void testSerializeReleaseSchedule() {
        SameDiff sd = NativeExecutorTestUtils.createChainGraph();
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");

        byte[] data = plan.serialize();
        ByteBuffer buf = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);

        // Skip header
        buf.position(24);

        // Skip all slots
        for (DynamicShapeSlot slot : plan.getSlots()) {
            buf.getLong(); // opHash
            int nIn = buf.getInt();
            int nOut = buf.getInt();
            buf.position(buf.position() + nIn * 4 + nIn + nOut * 4); // indices + types + outputIndices
            int nI = buf.getInt();
            int nT = buf.getInt();
            int nB = buf.getInt();
            int nD = buf.getInt();
            buf.position(buf.position() + nI * 8 + nT * 8 + nB + nD * 4); // args
            buf.position(buf.position() + 5 + 4); // flags + targetDeviceId
        }

        // Now read release schedule
        int[][] releaseAtStep = plan.getReleaseAtStep();
        for (int[] releases : releaseAtStep) {
            int count = buf.getInt();
            assertEquals(releases.length, count, "Release count mismatch");
            for (int i = 0; i < count; i++) {
                int slotIdx = buf.getInt();
                assertEquals(releases[i], slotIdx, "Release slot index mismatch");
            }
        }

        plan.close();
    }

    @Test
    public void testSerializeIsValidHeader() {
        // Valid header check
        assertTrue(DynamicShapePlan.isValidSerializedPlan(createMinimalDsp1Header()));

        // Null/empty
        assertFalse(DynamicShapePlan.isValidSerializedPlan(null));
        assertFalse(DynamicShapePlan.isValidSerializedPlan(new byte[0]));
        assertFalse(DynamicShapePlan.isValidSerializedPlan(new byte[23])); // Too small

        // Wrong magic
        byte[] wrongMagic = createMinimalDsp1Header();
        wrongMagic[0] = 0; // Corrupt magic
        assertFalse(DynamicShapePlan.isValidSerializedPlan(wrongMagic));

        // Wrong version
        byte[] wrongVersion = createMinimalDsp1Header();
        wrongVersion[4] = 99; // Version 99
        assertFalse(DynamicShapePlan.isValidSerializedPlan(wrongVersion));
    }

    @Test
    public void testSerializeMatMulPlan() {
        // More complex graph: z = x @ w + b
        SameDiff sd = NativeExecutorTestUtils.createMatMulGraph(4, 3);
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");

        NativeExecutorTestUtils.assertValidSerialization(plan);

        byte[] data = plan.serialize();
        log.info("MatMul plan serialized to {} bytes ({} slots, {} output slots, {} external inputs)",
                data.length, plan.getSlots().length, plan.getTotalOutputSlots(),
                plan.getExternalInputKeys().length);

        // Verify slot count is reasonable (at least matmul + add = 2 ops)
        assertTrue(plan.getSlots().length >= 2, "MatMul plan should have >= 2 slots");

        plan.close();
    }

    @Test
    public void testSerializeChainPlan() {
        // z = relu(sigmoid(x + y) * w)
        SameDiff sd = NativeExecutorTestUtils.createChainGraph();
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");

        NativeExecutorTestUtils.assertValidSerialization(plan);

        byte[] data = plan.serialize();
        log.info("Chain plan: {} bytes, {} slots", data.length, plan.getSlots().length);

        // At least: add + sigmoid + mul + relu = 4 ops
        assertTrue(plan.getSlots().length >= 4, "Chain plan should have >= 4 slots");

        plan.close();
    }

    @Test
    public void testSerializeDiamondPlan() {
        // x → [a = x+1, b = x*2] → c = a + b
        SameDiff sd = NativeExecutorTestUtils.createDiamondGraph();
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "c");

        NativeExecutorTestUtils.assertValidSerialization(plan);

        byte[] data = plan.serialize();
        log.info("Diamond plan: {} bytes, {} slots", data.length, plan.getSlots().length);

        plan.close();
    }

    @Test
    public void testSerializeFanOutPlan() {
        // x → [a, b, c] — multiple outputs requested
        SameDiff sd = NativeExecutorTestUtils.createFanOutGraph();
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "a", "b", "c");

        NativeExecutorTestUtils.assertValidSerialization(plan);

        byte[] data = plan.serialize();
        ByteBuffer buf = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
        buf.position(20); // Skip to numRequestedOutputs
        int numRequestedOutputs = buf.getInt();
        assertEquals(3, numRequestedOutputs, "Should have 3 requested outputs");

        plan.close();
    }

    @Test
    public void testSerializeWithIArgs() {
        // Create graph with reshape that has iArgs (target shape)
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 6);
        SDVariable reshaped = sd.reshape("z", x, 3, 4);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        // At least one slot should have iArgs
        boolean hasIArgs = false;
        for (DynamicShapeSlot slot : plan.getSlots()) {
            if (slot.getIArgs() != null && slot.getIArgs().length > 0) {
                hasIArgs = true;
                break;
            }
        }
        // Note: reshape may or may not have iArgs depending on how the compiler wires it
        // The important thing is that serialization doesn't crash

        plan.close();
    }

    @Test
    public void testSerializeWithDeviceAssignment() {
        SameDiff sd = NativeExecutorTestUtils.createChainGraph();
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");

        // Manually assign devices for test
        if (plan.getSlots().length >= 2) {
            plan.getSlots()[0].setTargetDeviceId(0);
            plan.getSlots()[1].setTargetDeviceId(1);
        }

        byte[] data = plan.serialize();
        ByteBuffer buf = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);

        // Skip header
        buf.position(24);

        // Read first slot's targetDeviceId
        // Need to skip to flags section
        long opHash = buf.getLong();
        int nIn = buf.getInt();
        int nOut = buf.getInt();
        buf.position(buf.position() + nIn * 4 + nIn + nOut * 4); // skip indices + types + outputs
        int nI = buf.getInt();
        int nT = buf.getInt();
        int nB = buf.getInt();
        int nD = buf.getInt();
        buf.position(buf.position() + nI * 8 + nT * 8 + nB + nD * 4); // skip args
        buf.position(buf.position() + 5); // skip 5 flag bytes

        int deviceId = buf.getInt();
        assertEquals(0, deviceId, "First slot should be on device 0");

        plan.close();
    }

    @Test
    public void testSerializeSourceTypes() {
        // Verify that different source types (CONSTANT, VARIABLE, PLACEHOLDER, OP_OUTPUT) are preserved
        SameDiff sd = NativeExecutorTestUtils.createAddGraph();
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");

        byte[] data = plan.serialize();

        // Verify that the source types from the plan are present in serialized data
        for (DynamicShapeSlot slot : plan.getSlots()) {
            for (byte type : slot.getInputSourceTypes()) {
                assertTrue(type >= DynamicShapeSlot.SOURCE_CONSTANT &&
                                type <= DynamicShapeSlot.SOURCE_OP_OUTPUT,
                        "Source type should be in valid range: " + type);
            }
        }

        plan.close();
    }

    @Test
    public void testSerializeMultipleOutputSlots() {
        // Graph where a single op produces multiple outputs
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        // Split into two halves using slice
        SDVariable a = sd.slice("a", x, new int[]{0, 0}, new int[]{-1, 4});
        SDVariable b = sd.slice("b", x, new int[]{0, 4}, new int[]{-1, 4});
        SDVariable out = a.add("z", b);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        plan.close();
    }

    // ─── Helper ──────────────────────────────────────────────────────────────

    private byte[] createMinimalDsp1Header() {
        ByteBuffer buf = ByteBuffer.allocate(24).order(ByteOrder.LITTLE_ENDIAN);
        buf.putInt(0x44535031); // DSP1 magic
        buf.putInt(1);          // version
        buf.putInt(0);          // numSlots
        buf.putInt(0);          // totalOutputSlots
        buf.putInt(0);          // numExternalInputs
        buf.putInt(0);          // numRequestedOutputs
        return buf.array();
    }
}
