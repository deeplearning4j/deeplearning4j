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
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlan;
import org.nd4j.autodiff.samediff.execution.DynamicShapeSlot;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.device.DeviceMemoryManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Smoke tests for DynamicShapePlan device placement and multi-device allocation.
 *
 * Tests:
 * 1. DynamicShapeSlot targetDeviceId field get/set
 * 2. DynamicShapePlan.assignDevices() proportional distribution
 * 3. DynamicShapePlan.assignDeviceToRange() manual assignment
 * 4. Multi-device allocation via NativeOps (CUDA-only, skipped on CPU)
 * 5. Full DSP execution with device placement (simple graph, verifies correctness)
 */
@Slf4j
@NativeTag
@Tag(TagNames.SAMEDIFF)
public class TestDSPDevicePlacement extends BaseNd4jTestWithBackends {

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSlotTargetDeviceIdDefault(Nd4jBackend backend) {
        // Verify default targetDeviceId is -1 (use current thread's device)
        DynamicShapeSlot slot = DynamicShapeSlot.builder()
                .opName("test_op")
                .customOp(true)
                .inputSourceIndices(new int[0])
                .inputSourceTypes(new byte[0])
                .inputVarNames(new String[0])
                .outputSlotIndices(new int[]{0})
                .outputVarNames(new String[]{"out"})
                .stepIndex(0)
                .opNameHash(12345L)
                .inputArraysBuffer(new INDArray[0])
                .build();

        assertEquals(-1, slot.getTargetDeviceId(),
                "Default targetDeviceId should be -1 (no override)");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSlotTargetDeviceIdSetGet(Nd4jBackend backend) {
        DynamicShapeSlot slot = DynamicShapeSlot.builder()
                .opName("test_op")
                .customOp(true)
                .inputSourceIndices(new int[0])
                .inputSourceTypes(new byte[0])
                .inputVarNames(new String[0])
                .outputSlotIndices(new int[]{0})
                .outputVarNames(new String[]{"out"})
                .stepIndex(0)
                .opNameHash(12345L)
                .inputArraysBuffer(new INDArray[0])
                .build();

        slot.setTargetDeviceId(1);
        assertEquals(1, slot.getTargetDeviceId());

        slot.setTargetDeviceId(0);
        assertEquals(0, slot.getTargetDeviceId());

        slot.setTargetDeviceId(-1);
        assertEquals(-1, slot.getTargetDeviceId());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPlanAssignDevicesProportional(Nd4jBackend backend) {
        // Create a plan with 100 slots
        int numSlots = 100;
        DynamicShapeSlot[] slots = new DynamicShapeSlot[numSlots];
        for (int i = 0; i < numSlots; i++) {
            slots[i] = DynamicShapeSlot.builder()
                    .opName("op_" + i)
                    .customOp(true)
                    .inputSourceIndices(new int[0])
                    .inputSourceTypes(new byte[0])
                    .inputVarNames(new String[0])
                    .outputSlotIndices(new int[]{i})
                    .outputVarNames(new String[]{"out_" + i})
                    .stepIndex(i)
                    .opNameHash(i * 31L)
                    .inputArraysBuffer(new INDArray[0])
                    .build();
        }

        DynamicShapePlan plan = new DynamicShapePlan(
                slots, numSlots, new int[numSlots][0], null,
                new String[0], Set.of("out_99"),
                Map.of("out_99", 99), false);

        // Simulate 2 devices: device 0 has 18GB, device 1 has 6GB (3:1 ratio)
        Map<Integer, Long> budgets = new LinkedHashMap<>();
        budgets.put(0, 18L * 1024 * 1024 * 1024);
        budgets.put(1, 6L * 1024 * 1024 * 1024);

        plan.assignDevices(budgets);

        // Count per-device assignments
        int dev0Count = 0, dev1Count = 0;
        for (DynamicShapeSlot slot : plan.getSlots()) {
            if (slot.getTargetDeviceId() == 0) dev0Count++;
            else if (slot.getTargetDeviceId() == 1) dev1Count++;
        }

        log.info("Proportional assignment: device0={} slots, device1={} slots", dev0Count, dev1Count);

        // With 3:1 ratio, expect ~75 on device 0 and ~25 on device 1
        assertTrue(dev0Count >= 70 && dev0Count <= 80,
                "Device 0 should get ~75 slots (3:1 ratio), got " + dev0Count);
        assertTrue(dev1Count >= 20 && dev1Count <= 30,
                "Device 1 should get ~25 slots (3:1 ratio), got " + dev1Count);
        assertEquals(numSlots, dev0Count + dev1Count,
                "All slots must be assigned");

        // Verify device 0 gets the first slots (largest-first ordering)
        for (int i = 0; i < dev0Count; i++) {
            assertEquals(0, plan.getSlots()[i].getTargetDeviceId(),
                    "First " + dev0Count + " slots should be on device 0");
        }

        // Verify summary string
        String summary = plan.getDeviceAssignmentSummary();
        assertTrue(summary.contains("device0"), "Summary should mention device0: " + summary);
        assertTrue(summary.contains("device1"), "Summary should mention device1: " + summary);
        log.info("Assignment summary: {}", summary);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPlanAssignDeviceToRange(Nd4jBackend backend) {
        int numSlots = 30;
        DynamicShapeSlot[] slots = new DynamicShapeSlot[numSlots];
        for (int i = 0; i < numSlots; i++) {
            slots[i] = DynamicShapeSlot.builder()
                    .opName("layer_op_" + i)
                    .customOp(true)
                    .inputSourceIndices(new int[0])
                    .inputSourceTypes(new byte[0])
                    .inputVarNames(new String[0])
                    .outputSlotIndices(new int[]{i})
                    .outputVarNames(new String[]{"out_" + i})
                    .stepIndex(i)
                    .opNameHash(i * 31L)
                    .inputArraysBuffer(new INDArray[0])
                    .build();
        }

        DynamicShapePlan plan = new DynamicShapePlan(
                slots, numSlots, new int[numSlots][0], null,
                new String[0], Set.of("out_29"),
                Map.of("out_29", 29), false);

        // Manual pipeline parallelism: first 20 on GPU 0, last 10 on GPU 1
        plan.assignDeviceToRange(0, 20, 0);
        plan.assignDeviceToRange(20, 30, 1);

        for (int i = 0; i < 20; i++) {
            assertEquals(0, plan.getSlots()[i].getTargetDeviceId(),
                    "Slot " + i + " should be on device 0");
        }
        for (int i = 20; i < 30; i++) {
            assertEquals(1, plan.getSlots()[i].getTargetDeviceId(),
                    "Slot " + i + " should be on device 1");
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPlanAssignDevicesSingleDevice(Nd4jBackend backend) {
        // Single-device budget should be a no-op (slots stay at -1)
        int numSlots = 10;
        DynamicShapeSlot[] slots = new DynamicShapeSlot[numSlots];
        for (int i = 0; i < numSlots; i++) {
            slots[i] = DynamicShapeSlot.builder()
                    .opName("op_" + i)
                    .customOp(true)
                    .inputSourceIndices(new int[0])
                    .inputSourceTypes(new byte[0])
                    .inputVarNames(new String[0])
                    .outputSlotIndices(new int[]{i})
                    .outputVarNames(new String[]{"out_" + i})
                    .stepIndex(i)
                    .opNameHash(i * 31L)
                    .inputArraysBuffer(new INDArray[0])
                    .build();
        }

        DynamicShapePlan plan = new DynamicShapePlan(
                slots, numSlots, new int[numSlots][0], null,
                new String[0], Set.of("out_9"),
                Map.of("out_9", 9), false);

        // Single device - should be no-op
        Map<Integer, Long> singleDevice = new LinkedHashMap<>();
        singleDevice.put(0, 24L * 1024 * 1024 * 1024);
        plan.assignDevices(singleDevice);

        // All slots should remain at default (-1)
        for (DynamicShapeSlot slot : plan.getSlots()) {
            assertEquals(-1, slot.getTargetDeviceId(),
                    "Single-device assignment should leave slots at -1 (default)");
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMultiDeviceMemoryQuery(Nd4jBackend backend) {
        // Test NativeOps device memory query APIs
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        int numDevices = nativeOps.getAvailableDevices();
        log.info("Available devices: {}", numDevices);
        assertTrue(numDevices >= 1, "Should have at least 1 device");

        for (int d = 0; d < numDevices; d++) {
            long totalMem = nativeOps.getDeviceTotalMemory(d);
            long freeMem = nativeOps.getDeviceFreeMemory(d);
            log.info("Device {}: total={}MB, free={}MB",
                    d, totalMem / (1024 * 1024), freeMem / (1024 * 1024));

            assertTrue(totalMem > 0, "Device " + d + " total memory should be > 0");
            assertTrue(freeMem >= 0, "Device " + d + " free memory should be >= 0");
            assertTrue(freeMem <= totalMem, "Free memory should not exceed total for device " + d);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMultiDeviceAllocation(Nd4jBackend backend) {
        // Test allocating arrays and checking affinity
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        int numDevices = nativeOps.getAvailableDevices();
        if (numDevices < 2) {
            log.info("Only {} device(s) available, skipping multi-device allocation test", numDevices);
            return;
        }

        int currentDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        log.info("Current thread device: {}", currentDevice);

        // Allocate on current device
        INDArray arr0 = Nd4j.zeros(DataType.FLOAT, 100, 100);
        int arr0Device = Nd4j.getAffinityManager().getDeviceForArray(arr0);
        log.info("Array allocated on device: {}", arr0Device);
        assertEquals(currentDevice, arr0Device,
                "Array should be on current thread's device");

        // Switch to alternate device and allocate
        int altDevice = (currentDevice == 0) ? 1 : 0;
        try {
            DeviceMemoryManager.getInstance().switchDevice(altDevice, "TestDSPDevicePlacement", "test-device-switch");
            INDArray arr1 = Nd4j.zeros(DataType.FLOAT, 100, 100);
            int arr1Device = Nd4j.getAffinityManager().getDeviceForArray(arr1);
            log.info("Array on alt device: {}", arr1Device);
            assertEquals(altDevice, arr1Device,
                    "Array should be on alternate device after switchDevice");

            // Both arrays should be valid
            assertFalse(arr0.wasClosed(), "Original array should still be valid");
            assertFalse(arr1.wasClosed(), "Alt device array should still be valid");

            // Clean up
            arr1.close();
        } finally {
            // Restore original device
            DeviceMemoryManager.getInstance().switchDevice(currentDevice, "TestDSPDevicePlacement", "test-device-switch");
        }
        arr0.close();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDSPExecutionWithDevicePlacement(Nd4jBackend backend) {
        // Build a simple graph: z = (x * W) + b
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable w = sd.constant("W", Nd4j.randn(DataType.FLOAT, 4, 3));
        SDVariable b = sd.constant("b", Nd4j.randn(DataType.FLOAT, 1, 3));

        SDVariable xw = sd.mmul("xw", x, w);
        SDVariable z = xw.add("z", b);

        // Execute normally first to get reference output
        INDArray xInput = Nd4j.randn(DataType.FLOAT, 2, 4);
        Map<String, INDArray> refOutput = sd.output(Map.of("x", xInput), "z");
        INDArray expectedZ = refOutput.get("z");
        assertNotNull(expectedZ, "Reference output should not be null");
        log.info("Reference output shape: {}", java.util.Arrays.toString(expectedZ.shape()));

        // Execute again to verify determinism
        Map<String, INDArray> refOutput2 = sd.output(Map.of("x", xInput), "z");
        INDArray expectedZ2 = refOutput2.get("z");
        assertEquals(expectedZ, expectedZ2, "Repeated execution should give same result");

        // If multi-GPU available, test with device placement
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        int numDevices = nativeOps.getAvailableDevices();
        log.info("Devices available for DSP device placement test: {}", numDevices);

        if (numDevices >= 2) {
            log.info("Multi-GPU: testing DSP execution with device placement");
            // Reset session to force new plan compilation
            sd.clearPlaceholders(false);
            sd.clearOpInputs();
            sd.resetSession();

            // Execute with device placement - should produce same result
            Map<String, INDArray> gpuOutput = sd.output(Map.of("x", xInput), "z");
            INDArray gpuZ = gpuOutput.get("z");
            assertNotNull(gpuZ, "GPU output should not be null");

            // Verify numerical correctness (allow small floating point differences)
            double maxDiff = expectedZ.sub(gpuZ).amaxNumber().doubleValue();
            log.info("Max difference between reference and multi-GPU output: {}", maxDiff);
            assertTrue(maxDiff < 1e-4,
                    "Multi-GPU output should match reference within tolerance, maxDiff=" + maxDiff);
        }

        // Clean up
        expectedZ.close();
        expectedZ2.close();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPlanAssignDevicesThreeGpus(Nd4jBackend backend) {
        // Verify proportional distribution works with 3 devices
        int numSlots = 120;
        DynamicShapeSlot[] slots = new DynamicShapeSlot[numSlots];
        for (int i = 0; i < numSlots; i++) {
            slots[i] = DynamicShapeSlot.builder()
                    .opName("op_" + i)
                    .customOp(true)
                    .inputSourceIndices(new int[0])
                    .inputSourceTypes(new byte[0])
                    .inputVarNames(new String[0])
                    .outputSlotIndices(new int[]{i})
                    .outputVarNames(new String[]{"out_" + i})
                    .stepIndex(i)
                    .opNameHash(i * 31L)
                    .inputArraysBuffer(new INDArray[0])
                    .build();
        }

        DynamicShapePlan plan = new DynamicShapePlan(
                slots, numSlots, new int[numSlots][0], null,
                new String[0], Set.of("out_119"),
                Map.of("out_119", 119), false);

        // 3 devices: 24GB, 8GB, 8GB (3:1:1 ratio)
        Map<Integer, Long> budgets = new LinkedHashMap<>();
        budgets.put(0, 24L * 1024 * 1024 * 1024);
        budgets.put(1, 8L * 1024 * 1024 * 1024);
        budgets.put(2, 8L * 1024 * 1024 * 1024);

        plan.assignDevices(budgets);

        // Count
        Map<Integer, Integer> counts = new LinkedHashMap<>();
        for (DynamicShapeSlot slot : plan.getSlots()) {
            counts.merge(slot.getTargetDeviceId(), 1, Integer::sum);
        }

        log.info("3-GPU assignment: {}", counts);

        // 24/(24+8+8) = 60%, 8/40 = 20% each
        int dev0 = counts.getOrDefault(0, 0);
        int dev1 = counts.getOrDefault(1, 0);
        int dev2 = counts.getOrDefault(2, 0);
        assertTrue(dev0 >= 65 && dev0 <= 80, "Device 0 (24GB) should get ~72 slots, got " + dev0);
        assertTrue(dev1 >= 15 && dev1 <= 30, "Device 1 (8GB) should get ~24 slots, got " + dev1);
        assertTrue(dev2 >= 15 && dev2 <= 30, "Device 2 (8GB) should get ~24 slots, got " + dev2);
        assertEquals(numSlots, dev0 + dev1 + dev2, "All 120 slots must be assigned");
    }

    /**
     * Reproducer for cross-device migrate failure (H2D "invalid argument").
     * When CudaMemoryPool::allocate() doesn't switch to the target device before
     * cudaMallocAsync, the allocation ends up on the wrong device. Subsequent
     * cudaMemcpy(H2D) with a pointer from the wrong device → "invalid argument".
     *
     * Test: create arrays on device 0, replicate to device 1, verify data.
     * Requires 2+ CUDA GPUs; skipped on CPU backend or single-GPU.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCrossDeviceMigrateRoundTrip(Nd4jBackend backend) {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        int numDevices = nativeOps.getAvailableDevices();
        if (numDevices < 2) {
            log.info("Skipping testCrossDeviceMigrateRoundTrip: only {} device(s)", numDevices);
            return;
        }

        // Test with various sizes to exercise different allocation paths.
        long[] sizes = {128, 1024, 4096, 128 * 128};

        for (long size : sizes) {
            // Create array on device 0 with known values
            INDArray src = Nd4j.linspace(1, size, size, DataType.FLOAT);
            log.info("Replicating array of size {} from device 0 to device 1", size);

            // Replicate to device 1
            INDArray dst = Nd4j.getAffinityManager().replicateToDevice(1, src);

            // Verify data integrity
            assertNotNull(dst, "Replicated array should not be null for size " + size);
            assertFalse(dst.isEmpty(), "Replicated array should not be empty for size " + size);
            assertEquals(size, dst.length(), "Length mismatch for size " + size);
            assertEquals(src.getFloat(0), dst.getFloat(0), 1e-5f,
                    "First element mismatch for size " + size);
            assertEquals(src.getFloat(src.length() - 1), dst.getFloat(dst.length() - 1), 1e-5f,
                    "Last element mismatch for size " + size);

            // Clean up
            src.close();
            dst.close();
        }

        log.info("Cross-device migrate round-trip: all {} sizes passed", sizes.length);
    }

    /**
     * Tests that SameDiff execution with DSP device placement correctly migrates
     * inputs and produces correct results across non-P2P devices.
     * This is a higher-level reproducer for the VLM migrate failure.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDSPCrossDeviceExecution(Nd4jBackend backend) {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        int numDevices = nativeOps.getAvailableDevices();
        if (numDevices < 2) {
            log.info("Skipping testDSPCrossDeviceExecution: only {} device(s)", numDevices);
            return;
        }

        // Build a graph that will have ops assigned to both devices
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 128);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 128, 64));
        SDVariable b = sd.constant("b", Nd4j.randn(DataType.FLOAT, 1, 64));
        SDVariable mm = sd.mmul("mm", x, w);
        SDVariable y = mm.add("y", b);

        // Run 5 iterations to verify stability
        for (int i = 0; i < 5; i++) {
            INDArray input = Nd4j.randn(DataType.FLOAT, 4, 128);
            Map<String, INDArray> placeholders = new LinkedHashMap<>();
            placeholders.put("x", input);

            Map<String, INDArray> result = sd.output(placeholders, "y");

            INDArray output = result.get("y");
            assertNotNull(output, "Output should not be null at iteration " + i);
            assertFalse(output.isEmpty(), "Output should not be empty at iteration " + i);
            assertArrayEquals(new long[]{4, 64}, output.shape(),
                    "Output shape should be [4,64] at iteration " + i);

            output.close();
            input.close();
        }

        sd.resetSession();
        log.info("DSP cross-device execution: 5 iterations passed");
    }
}
