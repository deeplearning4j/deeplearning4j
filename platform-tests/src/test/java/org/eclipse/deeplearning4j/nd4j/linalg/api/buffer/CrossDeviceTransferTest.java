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

package org.eclipse.deeplearning4j.nd4j.linalg.api.buffer;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.*;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.device.DeviceMemoryManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Isolation tests for cross-device buffer transfers.
 *
 * These tests verify that ensureAvailableOn() and cross-device op execution
 * work correctly without destroying source buffers — the root cause of
 * CUDA error 700 (illegal memory access) in multi-GPU speculative decode.
 *
 * The scenario: target model on device 0, draft model on device 1.
 * DeviceAwareOpExecutioner transfers inputs between devices for op execution.
 * The transfer must be NON-DESTRUCTIVE — the source buffer must remain valid
 * on its original device after the transfer.
 */
@Slf4j
public class CrossDeviceTransferTest extends BaseNd4jTestWithBackends {

    private DeviceMemoryManager memoryManager;
    private NativeOps nativeOps;

    private boolean isCudaBackend() {
        String backendName = Nd4j.getBackend().getClass().getSimpleName().toLowerCase();
        return backendName.contains("cuda") || backendName.contains("jcublas");
    }

    private int getNumDevices() {
        try {
            return Nd4j.getAffinityManager().getNumberOfDevices();
        } catch (Exception e) {
            return 0;
        }
    }

    private int getBufferDeviceId(INDArray arr) {
        if (arr == null || arr.data() == null || arr.data().wasClosed()) return -1;
        return nativeOps.dbDeviceId(arr.data().opaqueBuffer());
    }

    @BeforeEach
    void setUp() {
        memoryManager = DeviceMemoryManager.getInstance();
        memoryManager.clearAllMemorySimulation();
        nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
    }

    @AfterEach
    void tearDown() {
        if (memoryManager != null) {
            memoryManager.clearAllMemorySimulation();
        }
    }

    // ===================================================================
    // Test 1: ensureAvailableOn must NOT destroy source buffer
    // ===================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("ensureAvailableOn does not destroy source buffer")
    void testEnsureAvailableOnNonDestructive(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Test requires CUDA backend");
        assumeTrue(getNumDevices() >= 2, "Test requires at least 2 CUDA devices");

        // Create array on current device with known values
        INDArray arr = Nd4j.createFromArray(1.0f, 2.0f, 3.0f, 4.0f, 5.0f);
        Nd4j.getExecutioner().commit();

        int originalDevice = getBufferDeviceId(arr);
        log.info("Array created on device {}", originalDevice);

        // Read values BEFORE transfer
        float[] valuesBefore = arr.dup().data().asFloat();

        // Transfer to the other device
        int targetDeviceIdx = (originalDevice == 0) ? 1 : 0;
        var targetDevice = org.nd4j.linalg.api.device.DeviceDescriptor.cuda(targetDeviceIdx);

        // This is the call that was destructive (migrate) — after fix it should be non-destructive
        arr.data().asHybrid().ensureAvailableOn(targetDevice);

        // CRITICAL: array must still be usable on the ORIGINAL device
        // Restore device context to original
        DeviceMemoryManager.getInstance().switchDevice(originalDevice,
                "CrossDeviceTransferTest", "restore-after-ensure");

        // Values must still be correct
        float[] valuesAfter = arr.dup().data().asFloat();
        assertArrayEquals(valuesBefore, valuesAfter, 1e-5f,
                "Data must be preserved after ensureAvailableOn");

        arr.close();
    }

    // ===================================================================
    // Test 2: Op on array after cross-device transfer must not error 700
    // ===================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Op execution after cross-device ensureAvailableOn succeeds")
    void testOpAfterCrossDeviceEnsureAvailableOn(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Test requires CUDA backend");
        assumeTrue(getNumDevices() >= 2, "Test requires at least 2 CUDA devices");

        // Create arrays on device 0
        INDArray a = Nd4j.createFromArray(1.0f, 2.0f, 3.0f);
        INDArray b = Nd4j.createFromArray(10.0f, 20.0f, 30.0f);
        Nd4j.getExecutioner().commit();

        int device0 = getBufferDeviceId(a);
        int device1 = (device0 == 0) ? 1 : 0;

        // Simulate what DeviceAwareOpExecutioner does:
        // Transfer 'a' to device 1
        var targetDevice = org.nd4j.linalg.api.device.DeviceDescriptor.cuda(device1);
        a.data().asHybrid().ensureAvailableOn(targetDevice);

        // Switch back to original device
        DeviceMemoryManager.getInstance().switchDevice(device0,
                "CrossDeviceTransferTest", "restore");

        // Now run an op using 'a' on the ORIGINAL device — this must NOT error 700
        INDArray result = a.add(b);
        assertNotNull(result);

        float[] expected = new float[]{11.0f, 22.0f, 33.0f};
        assertArrayEquals(expected, result.toFloatVector(), 1e-5f,
                "Op on original device must produce correct results after cross-device transfer");

        result.close();
        a.close();
        b.close();
    }

    // ===================================================================
    // Test 3: Repeated cross-device transfers don't accumulate errors
    // ===================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Repeated cross-device transfers remain stable")
    void testRepeatedCrossDeviceTransfers(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Test requires CUDA backend");
        assumeTrue(getNumDevices() >= 2, "Test requires at least 2 CUDA devices");

        INDArray arr = Nd4j.createFromArray(42.0f, 43.0f, 44.0f);
        Nd4j.getExecutioner().commit();

        int device0 = getBufferDeviceId(arr);
        int device1 = (device0 == 0) ? 1 : 0;

        var desc0 = org.nd4j.linalg.api.device.DeviceDescriptor.cuda(device0);
        var desc1 = org.nd4j.linalg.api.device.DeviceDescriptor.cuda(device1);

        // Bounce the array back and forth 10 times
        for (int i = 0; i < 10; i++) {
            arr.data().asHybrid().ensureAvailableOn(desc1);
            DeviceMemoryManager.getInstance().switchDevice(device0,
                    "CrossDeviceTransferTest", "bounce-back-" + i);

            arr.data().asHybrid().ensureAvailableOn(desc0);

            // Verify data integrity each round
            float[] values = arr.dup().data().asFloat();
            assertArrayEquals(new float[]{42.0f, 43.0f, 44.0f}, values, 1e-5f,
                    "Data corrupted after bounce " + i);
        }

        arr.close();
    }

    // ===================================================================
    // Test 4: Simulate speculative decode scenario — model on device 0,
    // draft model arrays on device 1, cross-device assign
    // ===================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Cross-device assign simulating speculative decode")
    void testCrossDeviceAssignSpeculativeDecode(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Test requires CUDA backend");
        assumeTrue(getNumDevices() >= 2, "Test requires at least 2 CUDA devices");

        int currentDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        int otherDevice = (currentDevice == 0) ? 1 : 0;

        // Simulate: logits array allocated on device 0 (target model output)
        // Shape matches SmolDocling logits: [seqLen, vocabSize]
        INDArray logits = Nd4j.rand(DataType.FLOAT, 6, 1024); // Smaller vocab for test
        Nd4j.getExecutioner().commit();
        int logitsDevice = getBufferDeviceId(logits);
        log.info("Logits on device {}", logitsDevice);

        // Simulate: draft model weight on other device
        // Force allocation on other device via memory simulation
        // Weight must be >1MB to trigger routing (128MB headroom for >1MB allocs)
        memoryManager.setSimulatedFreeMemory(currentDevice, 1L * 1024 * 1024);    // 1MB
        memoryManager.setSimulatedFreeMemory(otherDevice, 4L * 1024 * 1024 * 1024); // 4GB
        memoryManager.setMemorySimulationEnabled(true);

        // Directly switch to other device and allocate there (no simulation needed — both GPUs have real memory)
        memoryManager.clearAllMemorySimulation();
        DeviceMemoryManager.getInstance().switchDevice(otherDevice,
                "CrossDeviceTransferTest", "force-other-device");

        // 2MB weight allocated on other device
        INDArray draftWeight = Nd4j.rand(DataType.FLOAT, 512, 1024);
        Nd4j.getExecutioner().commit();

        // Switch back to original device
        DeviceMemoryManager.getInstance().switchDevice(currentDevice,
                "CrossDeviceTransferTest", "restore-after-draft-alloc");

        int weightDevice = getBufferDeviceId(draftWeight);
        log.info("Draft weight on device {} (native query)", weightDevice);

        // Now simulate what happens during speculative decode:
        // 1. Draft model runs on device where its weights are
        // 2. Output logits need to be compared with target model logits
        // 3. This involves cross-device data movement

        // Create a result array and assign from logits (simulates the assign that errors)
        INDArray slice = logits.getRow(0).dup();
        Nd4j.getExecutioner().commit();

        // This assign should work even if arrays are on different devices
        INDArray target = Nd4j.create(DataType.FLOAT, 1024);
        target.assign(slice);
        Nd4j.getExecutioner().commit();

        // Verify data integrity
        float[] sliceData = slice.dup().data().asFloat();
        float[] targetData = target.dup().data().asFloat();
        assertArrayEquals(sliceData, targetData, 1e-5f,
                "Cross-device assign must preserve data");

        // CRITICAL: original logits array must still be usable
        INDArray logitsSum = logits.sum();
        assertNotNull(logitsSum);
        assertTrue(Float.isFinite(logitsSum.getFloat(0)),
                "Original logits must still be valid after cross-device operations");

        logits.close();
        draftWeight.close();
        slice.close();
        target.close();
        logitsSum.close();
    }

    // ===================================================================
    // Test 5: Device context preserved after ensureAvailableOn
    // This was the root cause of error 700 — ensureAvailableOn called
    // switchDevice + migrate but never restored the caller's device
    // ===================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Device context preserved after ensureAvailableOn")
    void testDeviceContextPreservedAfterEnsureAvailableOn(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Test requires CUDA backend");
        assumeTrue(getNumDevices() >= 2, "Test requires at least 2 CUDA devices");

        int deviceBefore = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        int otherDevice = (deviceBefore == 0) ? 1 : 0;

        // Allocate on other device so ensureAvailableOn actually triggers migration
        DeviceMemoryManager.getInstance().switchDevice(otherDevice,
                "CrossDeviceTransferTest", "alloc-on-other");
        INDArray arr = Nd4j.createFromArray(1.0f, 2.0f, 3.0f);
        Nd4j.getExecutioner().commit();
        DeviceMemoryManager.getInstance().switchDevice(deviceBefore,
                "CrossDeviceTransferTest", "restore-before-test");

        assertEquals(otherDevice, getBufferDeviceId(arr), "Array must be on other device");
        assertEquals(deviceBefore, Nd4j.getAffinityManager().getDeviceForCurrentThread(),
                "Must be on original device before ensureAvailableOn");

        // ensureAvailableOn triggers cross-device migrate
        var targetDesc = org.nd4j.linalg.api.device.DeviceDescriptor.cuda(deviceBefore);
        arr.data().asHybrid().ensureAvailableOn(targetDesc);

        // CRITICAL: device context must be preserved after ensureAvailableOn
        int deviceAfter = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        assertEquals(deviceBefore, deviceAfter,
                "Device context must be preserved after ensureAvailableOn. " +
                "Was " + deviceBefore + " before, is " + deviceAfter + " after.");

        arr.close();
    }

    // ===================================================================
    // Test 6: Multiple arrays, mixed devices, all ops correct
    // Allocations >1MB to trigger routing (smaller arrays use min headroom)
    // ===================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Mixed-device arrays all produce correct op results")
    void testMixedDeviceOpsCorrectness(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Test requires CUDA backend");
        assumeTrue(getNumDevices() >= 2, "Test requires at least 2 CUDA devices");

        int currentDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        int otherDevice = (currentDevice == 0) ? 1 : 0;

        // Array on current device — 2MB
        int elemCount = 512 * 1024; // 2MB at FLOAT
        INDArray onCurrent = Nd4j.rand(DataType.FLOAT, 1, elemCount);
        Nd4j.getExecutioner().commit();
        float currentSum = onCurrent.sumNumber().floatValue();

        // Switch to other device and allocate there directly
        DeviceMemoryManager.getInstance().switchDevice(otherDevice,
                "CrossDeviceTransferTest", "alloc-on-other");

        INDArray onOther = Nd4j.rand(DataType.FLOAT, 1, elemCount); // 2MB
        Nd4j.getExecutioner().commit();
        float otherSum = onOther.sumNumber().floatValue();

        // Switch back to current device
        DeviceMemoryManager.getInstance().switchDevice(currentDevice,
                "CrossDeviceTransferTest", "restore-after-other");

        int deviceA = getBufferDeviceId(onCurrent);
        int deviceB = getBufferDeviceId(onOther);
        log.info("onCurrent device={}, onOther device={}", deviceA, deviceB);

        // Verify they're on different devices
        assertNotEquals(deviceA, deviceB,
                "Arrays must be on different devices for this test to be valid. " +
                "onCurrent=" + deviceA + " onOther=" + deviceB);

        // Scalar ops on each — should work regardless of device
        INDArray r1 = onCurrent.add(100.0f);
        float r1Sum = r1.sumNumber().floatValue();
        assertEquals(currentSum + 100.0f * elemCount, r1Sum, Math.abs(r1Sum) * 1e-4f,
                "Op on current device array must be correct");

        INDArray r2 = onOther.add(100.0f);
        float r2Sum = r2.sumNumber().floatValue();
        assertEquals(otherSum + 100.0f * elemCount, r2Sum, Math.abs(r2Sum) * 1e-4f,
                "Op on other device array must be correct");

        r1.close();
        r2.close();
        onCurrent.close();
        onOther.close();
    }

    // ===================================================================
    // Test 7: Binary op with inputs on different devices (the actual error 700 scenario)
    // DeviceAwareOpExecutioner must transparently handle this
    // ===================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Binary op with cross-device inputs works correctly")
    void testBinaryOpCrossDeviceInputs(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Test requires CUDA backend");
        assumeTrue(getNumDevices() >= 2, "Test requires at least 2 CUDA devices");

        int device0 = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        int device1 = (device0 == 0) ? 1 : 0;

        // Create array A on device 0
        INDArray a = Nd4j.createFromArray(1.0f, 2.0f, 3.0f, 4.0f);
        Nd4j.getExecutioner().commit();
        assertEquals(device0, getBufferDeviceId(a), "A must be on device 0");

        // Create array B on device 1
        DeviceMemoryManager.getInstance().switchDevice(device1,
                "CrossDeviceTransferTest", "alloc-b-on-device1");
        INDArray b = Nd4j.createFromArray(10.0f, 20.0f, 30.0f, 40.0f);
        Nd4j.getExecutioner().commit();

        // Switch back to device 0
        DeviceMemoryManager.getInstance().switchDevice(device0,
                "CrossDeviceTransferTest", "restore-after-b");

        assertEquals(device1, getBufferDeviceId(b), "B must be on device 1");

        // Now: a.add(b) — inputs on DIFFERENT devices!
        // DeviceAwareOpExecutioner must handle this transparently.
        // Previous bug: ensureAvailableOn() migrate destroyed source buffer,
        // causing error 700 when the op tried to read the migrated-away data.
        INDArray result = a.add(b);
        Nd4j.getExecutioner().commit();

        assertNotNull(result);
        float[] expected = new float[]{11.0f, 22.0f, 33.0f, 44.0f};
        assertArrayEquals(expected, result.toFloatVector(), 1e-5f,
                "Binary op with cross-device inputs must produce correct results");

        // CRITICAL: Both source arrays must still be valid after the op
        float[] aValues = a.dup().data().asFloat();
        assertArrayEquals(new float[]{1.0f, 2.0f, 3.0f, 4.0f}, aValues, 1e-5f,
                "Array A must not be corrupted after cross-device op");

        float[] bValues = b.dup().data().asFloat();
        assertArrayEquals(new float[]{10.0f, 20.0f, 30.0f, 40.0f}, bValues, 1e-5f,
                "Array B must not be corrupted after cross-device op");

        result.close();
        a.close();
        b.close();
    }

    // ===================================================================
    // Test 8: Sequential ops with cross-device data (simulates decode loop)
    // ===================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Sequential ops with cross-device data remain stable")
    void testSequentialCrossDeviceOps(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Test requires CUDA backend");
        assumeTrue(getNumDevices() >= 2, "Test requires at least 2 CUDA devices");

        int device0 = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        int device1 = (device0 == 0) ? 1 : 0;

        // "Model weights" on device 0
        INDArray weights = Nd4j.rand(DataType.FLOAT, 64, 64);
        Nd4j.getExecutioner().commit();

        // "Input" on device 1
        DeviceMemoryManager.getInstance().switchDevice(device1,
                "CrossDeviceTransferTest", "input-on-device1");
        INDArray input = Nd4j.rand(DataType.FLOAT, 1, 64);
        Nd4j.getExecutioner().commit();
        DeviceMemoryManager.getInstance().switchDevice(device0,
                "CrossDeviceTransferTest", "restore");

        assertEquals(device0, getBufferDeviceId(weights));
        assertEquals(device1, getBufferDeviceId(input));

        // Simulate 5 decode steps: each step does matmul + add
        // This exercises repeated cross-device transfers
        INDArray current = input;
        for (int step = 0; step < 5; step++) {
            INDArray output = current.mmul(weights);
            Nd4j.getExecutioner().commit();

            assertNotNull(output, "Step " + step + " output must not be null");
            assertEquals(1, output.rows());
            assertEquals(64, output.columns());

            // Verify no NaN/Inf
            float sum = output.sumNumber().floatValue();
            assertTrue(Float.isFinite(sum),
                    "Step " + step + " must produce finite output, got " + sum);

            if (current != input) current.close();
            current = output;
        }

        // Verify weights weren't corrupted by the cross-device ops
        float weightsSum = weights.sumNumber().floatValue();
        assertTrue(Float.isFinite(weightsSum),
                "Weights must not be corrupted after cross-device ops");

        current.close();
        input.close();
        weights.close();
    }

    // ===================================================================
    // Test 9: Large buffer cross-device transfer (simulates logits)
    // ===================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Large buffer cross-device transfer preserves data")
    void testLargeBufferCrossDeviceTransfer(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Test requires CUDA backend");
        assumeTrue(getNumDevices() >= 2, "Test requires at least 2 CUDA devices");

        // ~1.1 MB — matches the logits transfer size that triggered error 700
        // [6, 49280] * 4 bytes = 1.18 MB, we use a smaller vocab for test speed
        INDArray large = Nd4j.rand(DataType.FLOAT, 6, 49280);
        Nd4j.getExecutioner().commit();

        int originalDevice = getBufferDeviceId(large);
        int targetDeviceIdx = (originalDevice == 0) ? 1 : 0;

        // Capture checksum before transfer
        float sumBefore = large.sumNumber().floatValue();

        var targetDesc = org.nd4j.linalg.api.device.DeviceDescriptor.cuda(targetDeviceIdx);
        large.data().asHybrid().ensureAvailableOn(targetDesc);

        // Restore context
        DeviceMemoryManager.getInstance().switchDevice(originalDevice,
                "CrossDeviceTransferTest", "restore-large");

        // Verify data integrity
        float sumAfter = large.sumNumber().floatValue();
        assertEquals(sumBefore, sumAfter, Math.abs(sumBefore) * 1e-5f,
                "Large buffer sum must match after cross-device transfer");

        large.close();
    }
}
