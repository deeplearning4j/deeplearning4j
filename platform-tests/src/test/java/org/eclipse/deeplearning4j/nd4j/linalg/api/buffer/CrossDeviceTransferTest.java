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

    // ===================================================================
    // Test: Isolate replicateToDevice round-trip to find zero-data bug
    // Each step is validated independently.
    // ===================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("replicateToDevice round-trip data integrity")
    void testReplicateToDeviceRoundTrip(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Test requires CUDA backend");
        assumeTrue(getNumDevices() >= 2, "Test requires at least 2 CUDA devices");

        DeviceMemoryManager mgr = DeviceMemoryManager.getInstance();
        int dev0 = mgr.getCurrentDeviceId();
        int dev1 = (dev0 == 0) ? 1 : 0;

        float[] srcData = {1.0f, 2.0f, 3.0f, 4.0f};

        // Step 1: Create array on dev0 with known values
        INDArray orig = Nd4j.createFromArray(srcData).reshape(2, 2);
        Nd4j.getExecutioner().commit();
        int origDev = getBufferDeviceId(orig);
        log.info("STEP1: orig created on dev={}, values={}", origDev, java.util.Arrays.toString(orig.dup().data().asFloat()));
        assertArrayEquals(srcData, orig.dup().data().asFloat(), 1e-6f, "Step1: orig data wrong");

        // Step 2: Replicate orig -> dev1
        INDArray onDev1 = Nd4j.getAffinityManager().replicateToDevice(dev1, orig);
        Nd4j.getExecutioner().commit();
        int dev1Actual = getBufferDeviceId(onDev1);
        log.info("STEP2: onDev1 deviceId={}", dev1Actual);

        // Switch to dev1 to read the data there
        mgr.switchDevice(dev1, "CrossDeviceTransferTest", "read-dev1-data");
        Nd4j.getExecutioner().commit();
        float[] dev1Data = onDev1.dup().data().asFloat();
        log.info("STEP2: onDev1 values={}", java.util.Arrays.toString(dev1Data));
        assertArrayEquals(srcData, dev1Data, 1e-6f, "Step2: dev0->dev1 replication lost data");

        // Step 3: Switch back to dev0, replicate onDev1 -> dev0
        mgr.switchDevice(dev0, "CrossDeviceTransferTest", "back-to-dev0");
        Nd4j.getExecutioner().commit();

        INDArray backOnDev0 = Nd4j.getAffinityManager().replicateToDevice(dev0, onDev1);
        Nd4j.getExecutioner().commit();
        int dev0Actual = getBufferDeviceId(backOnDev0);
        log.info("STEP3: backOnDev0 deviceId={}", dev0Actual);

        float[] dev0Data = backOnDev0.dup().data().asFloat();
        log.info("STEP3: backOnDev0 values={}", java.util.Arrays.toString(dev0Data));
        assertArrayEquals(srcData, dev0Data, 1e-6f, "Step3: dev1->dev0 round-trip lost data");

        // Step 4: Verify original is still intact
        float[] origCheck = orig.dup().data().asFloat();
        log.info("STEP4: orig still = {}", java.util.Arrays.toString(origCheck));
        assertArrayEquals(srcData, origCheck, 1e-6f, "Step4: original array corrupted");

        orig.close();
        onDev1.close();
        backOnDev0.close();
    }

    // ===================================================================
    // Test: Isolate syncToPrimary on a cross-device buffer
    // Verifies the C++ D2H path returns correct data for a buffer
    // that was populated via memcpySync H2D (no C++ counter update).
    // ===================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("syncToPrimary returns correct data after cross-device memcpy")
    void testSyncToPrimaryAfterCrossDeviceMemcpy(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Test requires CUDA backend");
        assumeTrue(getNumDevices() >= 2, "Test requires at least 2 CUDA devices");

        DeviceMemoryManager mgr = DeviceMemoryManager.getInstance();
        int dev0 = mgr.getCurrentDeviceId();
        int dev1 = (dev0 == 0) ? 1 : 0;

        // Create on dev0 with known data
        INDArray src = Nd4j.createFromArray(10.0f, 20.0f, 30.0f, 40.0f);
        Nd4j.getExecutioner().commit();
        log.info("src dev={}, data={}", getBufferDeviceId(src),
                java.util.Arrays.toString(src.dup().data().asFloat()));

        // Replicate to dev1 — this goes through the memcpySync H2D path in CudaZeroHandler
        INDArray onDev1 = Nd4j.getAffinityManager().replicateToDevice(dev1, src);
        Nd4j.getExecutioner().commit();

        // Now explicitly sync the dev1 buffer's device data to host.
        // This is the exact path that was producing zeros.
        onDev1.data().opaqueBuffer().syncToPrimary();

        // Read the host data via primaryBuffer
        org.bytedeco.javacpp.Pointer hostPtr = onDev1.data().opaqueBuffer().primaryBuffer();
        assertNotNull(hostPtr, "Primary buffer must exist after syncToPrimary");
        assertFalse(hostPtr.isNull(), "Primary buffer must not be null");

        org.bytedeco.javacpp.FloatPointer fp = new org.bytedeco.javacpp.FloatPointer(hostPtr);
        float[] hostVals = new float[4];
        for (int i = 0; i < 4; i++) {
            hostVals[i] = fp.get(i);
        }
        log.info("After syncToPrimary on dev1 buffer: hostVals={}", java.util.Arrays.toString(hostVals));

        assertArrayEquals(new float[]{10.0f, 20.0f, 30.0f, 40.0f}, hostVals, 1e-6f,
                "syncToPrimary must return correct data for cross-device buffer");

        // Also verify via the standard dup path
        mgr.switchDevice(dev1, "CrossDeviceTransferTest", "verify-dup-dev1");
        float[] dupVals = onDev1.dup().data().asFloat();
        log.info("dup on dev1: {}", java.util.Arrays.toString(dupVals));
        assertArrayEquals(new float[]{10.0f, 20.0f, 30.0f, 40.0f}, dupVals, 1e-6f,
                "dup on dev1 must return correct data");

        // Restore to dev0
        mgr.switchDevice(dev0, "CrossDeviceTransferTest", "restore-final");

        src.close();
        onDev1.close();
    }

    // ===================================================================
    // Cross-device STREAM tests
    // ===================================================================

    /**
     * Test: Replication after in-flight async kernel.
     *
     * Creates data via an async kernel (mmul), then immediately replicates
     * to another device WITHOUT explicit commit first.  replicateToDevice
     * must internally commit/sync the source device before the D2H copy
     * so the host-staged data reflects the kernel output, not stale zeros.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("replicateToDevice after in-flight async kernel gets correct data")
    void testReplicateAfterAsyncKernel(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Test requires CUDA backend");
        assumeTrue(getNumDevices() >= 2, "Test requires at least 2 CUDA devices");

        DeviceMemoryManager mgr = DeviceMemoryManager.getInstance();
        int dev0 = mgr.getCurrentDeviceId();
        int dev1 = (dev0 == 0) ? 1 : 0;

        // Create known inputs on dev0
        INDArray a = Nd4j.createFromArray(new float[]{1, 2, 3, 4}).reshape(2, 2);
        INDArray b = Nd4j.createFromArray(new float[]{5, 6, 7, 8}).reshape(2, 2);
        Nd4j.getExecutioner().commit();

        // Launch async kernel (mmul) — result is in GPU memory, not yet committed
        INDArray c = a.mmul(b);
        // Intentionally NO commit() here — replicateToDevice must handle it

        // Replicate to dev1 — must sync dev0's stream first
        INDArray cOnDev1 = Nd4j.getAffinityManager().replicateToDevice(dev1, c);

        // Read back on dev1
        mgr.switchDevice(dev1, "CrossDeviceTransferTest", "read-dev1");
        float[] dev1Vals = cOnDev1.dup().data().asFloat();
        log.info("mmul result on dev1: {}", java.util.Arrays.toString(dev1Vals));

        // Expected: [[1,2],[3,4]] x [[5,6],[7,8]] = [[19,22],[43,50]]
        float[] expected = {19.0f, 22.0f, 43.0f, 50.0f};
        assertArrayEquals(expected, dev1Vals, 1e-3f,
                "Replicated mmul result must match expected (stream must be synced before D2H)");

        mgr.switchDevice(dev0, "CrossDeviceTransferTest", "restore");
        a.close(); b.close(); c.close(); cOnDev1.close();
    }

    /**
     * Test: Cross-device replication preserves data when source has
     * pending writes from multiple kernels chained without intermediate commits.
     *
     * This catches the case where syncToPrimary skips the D2H copy because
     * isPrimaryActual() returns true (stale counter state), or because
     * the stream sync doesn't wait for all pending work.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("replicateToDevice after chained async ops preserves data")
    void testReplicateAfterChainedOps(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Test requires CUDA backend");
        assumeTrue(getNumDevices() >= 2, "Test requires at least 2 CUDA devices");

        DeviceMemoryManager mgr = DeviceMemoryManager.getInstance();
        int dev0 = mgr.getCurrentDeviceId();
        int dev1 = (dev0 == 0) ? 1 : 0;

        // Chain: create -> add -> mul -> add (all async on dev0, no commits)
        INDArray x = Nd4j.createFromArray(1.0f, 2.0f, 3.0f, 4.0f);
        Nd4j.getExecutioner().commit(); // commit creation only
        INDArray y = x.add(10.0f);      // [11, 12, 13, 14]
        INDArray z = y.mul(2.0f);       // [22, 24, 26, 28]
        INDArray w = z.add(100.0f);     // [122, 124, 126, 128]
        // NO commit — pending on dev0's exec stream

        INDArray wOnDev1 = Nd4j.getAffinityManager().replicateToDevice(dev1, w);

        mgr.switchDevice(dev1, "CrossDeviceTransferTest", "read-chained");
        float[] vals = wOnDev1.dup().data().asFloat();
        log.info("Chained ops result on dev1: {}", java.util.Arrays.toString(vals));

        assertArrayEquals(new float[]{122, 124, 126, 128}, vals, 1e-3f,
                "Chained async ops must be fully synced before cross-device D2H");

        mgr.switchDevice(dev0, "CrossDeviceTransferTest", "restore-chained");
        x.close(); y.close(); z.close(); w.close(); wOnDev1.close();
    }

    /**
     * Test: Cross-device matmul with large operands exercises the actual
     * cuBLAS stream-to-copy synchronization path.
     *
     * Large enough to have meaningful async overlap (256x256 matmul),
     * small enough to run fast.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Large matmul cross-device transfer has correct stream sync")
    void testLargeMatmulCrossDeviceStreamSync(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Test requires CUDA backend");
        assumeTrue(getNumDevices() >= 2, "Test requires at least 2 CUDA devices");

        DeviceMemoryManager mgr = DeviceMemoryManager.getInstance();
        int dev0 = mgr.getCurrentDeviceId();
        int dev1 = (dev0 == 0) ? 1 : 0;

        INDArray a = Nd4j.rand(DataType.FLOAT, 256, 256);
        INDArray b = Nd4j.rand(DataType.FLOAT, 256, 256);
        Nd4j.getExecutioner().commit();

        // Compute reference on dev0 with explicit sync
        INDArray ref = a.mmul(b);
        Nd4j.getExecutioner().commit();
        float refSum = ref.sumNumber().floatValue();

        // New matmul — NO commit before replication
        INDArray c = a.mmul(b);
        INDArray cOnDev1 = Nd4j.getAffinityManager().replicateToDevice(dev1, c);

        mgr.switchDevice(dev1, "CrossDeviceTransferTest", "read-large-matmul");
        float dev1Sum = cOnDev1.sumNumber().floatValue();

        assertEquals(refSum, dev1Sum, Math.abs(refSum) * 1e-4f,
                "Large matmul cross-device transfer must match reference (stream sync issue)");

        mgr.switchDevice(dev0, "CrossDeviceTransferTest", "restore-large-matmul");
        a.close(); b.close(); ref.close(); c.close(); cOnDev1.close();
    }

    /**
     * Test: Concurrent operations on both devices after cross-device transfer.
     *
     * After replicating A from dev0 to dev1, run independent ops on BOTH
     * devices simultaneously. This exercises stream isolation — dev0's
     * exec stream must not be entangled with dev1's after the transfer.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Post-transfer ops on both devices are stream-isolated")
    void testPostTransferStreamIsolation(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Test requires CUDA backend");
        assumeTrue(getNumDevices() >= 2, "Test requires at least 2 CUDA devices");

        DeviceMemoryManager mgr = DeviceMemoryManager.getInstance();
        int dev0 = mgr.getCurrentDeviceId();
        int dev1 = (dev0 == 0) ? 1 : 0;

        // Create and compute on dev0
        INDArray orig = Nd4j.rand(DataType.FLOAT, 128, 128);
        Nd4j.getExecutioner().commit();

        // Replicate to dev1
        INDArray onDev1 = Nd4j.getAffinityManager().replicateToDevice(dev1, orig);

        // Op on dev0 with orig — should use dev0 stream
        INDArray dev0Result = orig.mul(2.0f);
        Nd4j.getExecutioner().commit();
        float dev0Sum = dev0Result.sumNumber().floatValue();

        // Op on dev1 with replica — should use dev1 stream
        mgr.switchDevice(dev1, "CrossDeviceTransferTest", "dev1-op");
        INDArray dev1Result = onDev1.mul(3.0f);
        Nd4j.getExecutioner().commit();
        float dev1Sum = dev1Result.sumNumber().floatValue();

        mgr.switchDevice(dev0, "CrossDeviceTransferTest", "restore-isolation");

        // Verify: dev0Sum = 2 * origSum, dev1Sum = 3 * origSum
        float origSum = orig.sumNumber().floatValue();
        assertEquals(2.0f * origSum, dev0Sum, Math.abs(origSum) * 1e-4f,
                "Dev0 op result incorrect — stream contamination?");
        assertEquals(3.0f * origSum, dev1Sum, Math.abs(origSum) * 1e-4f,
                "Dev1 op result incorrect — stream contamination?");

        orig.close(); onDev1.close(); dev0Result.close(); dev1Result.close();
    }

    /**
     * Test: syncToPrimary on a buffer written by a kernel on a DIFFERENT device
     * than the current thread's device.
     *
     * This is the exact scenario that produces zeros: buffer lives on dev1,
     * thread is on dev0, syncToPrimary must switch to dev1 for the D2H copy.
     * The stream used for D2H (stream 0 on dev1) must implicitly sync with
     * the kernel that wrote the data.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("syncToPrimary from wrong-device thread returns correct data")
    void testSyncToPrimaryFromWrongDeviceThread(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Test requires CUDA backend");
        assumeTrue(getNumDevices() >= 2, "Test requires at least 2 CUDA devices");

        DeviceMemoryManager mgr = DeviceMemoryManager.getInstance();
        int dev0 = mgr.getCurrentDeviceId();
        int dev1 = (dev0 == 0) ? 1 : 0;

        // Create and compute on dev1
        mgr.switchDevice(dev1, "CrossDeviceTransferTest", "create-on-dev1");
        INDArray x = Nd4j.createFromArray(5.0f, 10.0f, 15.0f, 20.0f);
        INDArray y = x.mul(3.0f); // [15, 30, 45, 60] — kernel on dev1
        Nd4j.getExecutioner().commit();

        // Switch to dev0 — thread is now on dev0, but y's data is on dev1
        mgr.switchDevice(dev0, "CrossDeviceTransferTest", "switch-to-dev0");

        // syncToPrimary must handle the device mismatch internally
        y.data().opaqueBuffer().syncToPrimary();

        org.bytedeco.javacpp.Pointer hostPtr = y.data().opaqueBuffer().primaryBuffer();
        assertNotNull(hostPtr, "Primary buffer must exist");
        assertFalse(hostPtr.isNull(), "Primary buffer must not be null");

        org.bytedeco.javacpp.FloatPointer fp = new org.bytedeco.javacpp.FloatPointer(hostPtr);
        float[] vals = new float[4];
        for (int i = 0; i < 4; i++) vals[i] = fp.get(i);
        log.info("syncToPrimary from wrong device: {}", java.util.Arrays.toString(vals));

        assertArrayEquals(new float[]{15.0f, 30.0f, 45.0f, 60.0f}, vals, 1e-3f,
                "syncToPrimary from wrong-device thread must return correct data");

        mgr.switchDevice(dev1, "CrossDeviceTransferTest", "cleanup-dev1");
        x.close(); y.close();
        mgr.switchDevice(dev0, "CrossDeviceTransferTest", "restore-final-wrongdev");
    }

    // ===================================================================
    // DSP Stream tests — exercises the DSP execution path with
    // cross-device inputs, verifying stream lifecycle through SameDiff
    // ===================================================================

    /**
     * Test: SameDiff DSP execution with input on wrong device.
     *
     * Creates a simple SameDiff graph, runs it with input on dev0 (normal),
     * then with input on dev1 (cross-device). The DSP executor must
     * migrate the input from dev1 to dev0 and produce correct output.
     * This exercises tl_dspExecutionStream handling during migration.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("DSP handles cross-device input migration with correct stream sync")
    void testDspCrossDeviceInputMigration(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Test requires CUDA backend");
        assumeTrue(getNumDevices() >= 2, "Test requires at least 2 CUDA devices");

        DeviceMemoryManager mgr = DeviceMemoryManager.getInstance();
        int dev0 = mgr.getCurrentDeviceId();
        int dev1 = (dev0 == 0) ? 1 : 0;

        // Build a simple matmul graph: out = x @ w + bias
        org.nd4j.autodiff.samediff.SameDiff sd = org.nd4j.autodiff.samediff.SameDiff.create();
        org.nd4j.autodiff.samediff.SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        org.nd4j.autodiff.samediff.SDVariable w = sd.var("w", Nd4j.rand(DataType.FLOAT, 4, 8));
        org.nd4j.autodiff.samediff.SDVariable bias = sd.var("bias", Nd4j.ones(DataType.FLOAT, 8));
        org.nd4j.autodiff.samediff.SDVariable mm = sd.mmul("mm", x, w);
        org.nd4j.autodiff.samediff.SDVariable out = mm.add("out", bias);

        try {
            // Reference: input on dev0, SLOT_BY_SLOT
            INDArray inputDev0 = Nd4j.rand(DataType.FLOAT, 2, 4);
            Nd4j.getExecutioner().commit();

            sd.setGraphExecutionMode(org.nd4j.autodiff.samediff.execution.GraphExecutionMode.SLOT_BY_SLOT);
            java.util.Map<String, INDArray> refOut = sd.output(java.util.Map.of("x", inputDev0), "out");
            INDArray ref = refOut.get("out").dup();
            refOut.values().forEach(v -> { if (v.closeable() && !v.wasClosed()) v.close(); });

            // Create same input data on dev1
            mgr.switchDevice(dev1, "CrossDeviceTransferTest", "create-input-dev1");
            INDArray inputDev1 = Nd4j.createFromArray(inputDev0.dup().data().asFloat()).reshape(2, 4);
            Nd4j.getExecutioner().commit();
            mgr.switchDevice(dev0, "CrossDeviceTransferTest", "back-to-dev0-for-dsp");

            // Now run DSP with input from dev1 — DSP must migrate it
            sd.setGraphExecutionMode(org.nd4j.autodiff.samediff.execution.GraphExecutionMode.AUTO);
            sd.setDspAutoCompileEnabled(true);
            sd.setDspNativeAutoCompileEnabled(true);
            java.util.Map<String, INDArray> dspOut = sd.output(java.util.Map.of("x", inputDev1), "out");
            INDArray dspResult = dspOut.get("out").dup();
            dspOut.values().forEach(v -> { if (v.closeable() && !v.wasClosed()) v.close(); });

            // Must match reference
            float[] refVals = ref.dup().data().asFloat();
            float[] dspVals = dspResult.dup().data().asFloat();
            log.info("Ref: {}, DSP with cross-dev input: {}",
                    java.util.Arrays.toString(java.util.Arrays.copyOf(refVals, Math.min(4, refVals.length))),
                    java.util.Arrays.toString(java.util.Arrays.copyOf(dspVals, Math.min(4, dspVals.length))));

            assertArrayEquals(refVals, dspVals, 1e-4f,
                    "DSP output with cross-device input must match SLOT_BY_SLOT reference");

            ref.close(); dspResult.close();
            inputDev0.close(); inputDev1.close();
        } finally {
            sd.close();
        }
    }

    /**
     * Test: DSP execution on dev0, then dev1, then dev0 again.
     *
     * Exercises stream state isolation between DSP executions on different
     * devices. The tl_dspExecutionStream is thread-local but CUDA streams
     * are device-specific — if DSP doesn't properly manage the stream per
     * device, the second execution may use a stream from the wrong device.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("DSP stream state isolated across device switches")
    void testDspStreamStateIsolation(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Test requires CUDA backend");
        assumeTrue(getNumDevices() >= 2, "Test requires at least 2 CUDA devices");

        DeviceMemoryManager mgr = DeviceMemoryManager.getInstance();
        int dev0 = mgr.getCurrentDeviceId();
        int dev1 = (dev0 == 0) ? 1 : 0;

        // Build simple graph: out = relu(x * 2 + 1)
        org.nd4j.autodiff.samediff.SameDiff sd = org.nd4j.autodiff.samediff.SameDiff.create();
        org.nd4j.autodiff.samediff.SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        org.nd4j.autodiff.samediff.SDVariable scaled = x.mul("scaled", 2.0);
        org.nd4j.autodiff.samediff.SDVariable shifted = scaled.add("shifted", 1.0);
        org.nd4j.autodiff.samediff.SDVariable out = sd.nn.relu("out", shifted, 0);

        sd.setGraphExecutionMode(org.nd4j.autodiff.samediff.execution.GraphExecutionMode.SLOT_BY_SLOT);

        try {
            float[] inputData = {-2, -1, 0, 1, 2, 3, 4, 5};
            // Expected: relu(x*2+1) = relu([-3,-1,1,3,5,7,9,11]) = [0,0,1,3,5,7,9,11]
            float[] expected = {0, 0, 1, 3, 5, 7, 9, 11};

            // Run on dev0
            INDArray inputDev0 = Nd4j.createFromArray(inputData).reshape(1, 8);
            Nd4j.getExecutioner().commit();
            java.util.Map<String, INDArray> out0 = sd.output(java.util.Map.of("x", inputDev0), "out");
            float[] vals0 = out0.get("out").dup().data().asFloat();
            out0.values().forEach(v -> { if (v.closeable() && !v.wasClosed()) v.close(); });
            assertArrayEquals(expected, vals0, 1e-4f, "Dev0 execution incorrect");

            // Switch to dev1 and run
            mgr.switchDevice(dev1, "CrossDeviceTransferTest", "dsp-dev1");
            INDArray inputDev1 = Nd4j.createFromArray(inputData).reshape(1, 8);
            Nd4j.getExecutioner().commit();
            java.util.Map<String, INDArray> out1 = sd.output(java.util.Map.of("x", inputDev1), "out");
            float[] vals1 = out1.get("out").dup().data().asFloat();
            out1.values().forEach(v -> { if (v.closeable() && !v.wasClosed()) v.close(); });
            assertArrayEquals(expected, vals1, 1e-4f,
                    "Dev1 execution incorrect — stream state contamination from dev0?");

            // Switch back to dev0 and run again
            mgr.switchDevice(dev0, "CrossDeviceTransferTest", "dsp-dev0-again");
            INDArray inputDev0Again = Nd4j.createFromArray(inputData).reshape(1, 8);
            Nd4j.getExecutioner().commit();
            java.util.Map<String, INDArray> out2 = sd.output(java.util.Map.of("x", inputDev0Again), "out");
            float[] vals2 = out2.get("out").dup().data().asFloat();
            out2.values().forEach(v -> { if (v.closeable() && !v.wasClosed()) v.close(); });
            assertArrayEquals(expected, vals2, 1e-4f,
                    "Dev0 re-execution incorrect after dev1 — stream state not restored?");

            inputDev0.close(); inputDev1.close(); inputDev0Again.close();
        } finally {
            sd.close();
        }
    }

    /**
     * Test: DSP AUTO mode with cross-device input, repeated iterations.
     *
     * Simulates a decode loop where the input comes from a different device
     * each iteration. This exercises the DSP plan cache, stream management,
     * and migration path under repeated device switching.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("DSP AUTO with alternating device inputs stays correct")
    void testDspAutoAlternatingDeviceInputs(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Test requires CUDA backend");
        assumeTrue(getNumDevices() >= 2, "Test requires at least 2 CUDA devices");

        DeviceMemoryManager mgr = DeviceMemoryManager.getInstance();
        int dev0 = mgr.getCurrentDeviceId();
        int dev1 = (dev0 == 0) ? 1 : 0;

        // Build graph: out = x @ w
        org.nd4j.autodiff.samediff.SameDiff sd = org.nd4j.autodiff.samediff.SameDiff.create();
        org.nd4j.autodiff.samediff.SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        org.nd4j.autodiff.samediff.SDVariable w = sd.var("w", Nd4j.eye(4).castTo(DataType.FLOAT));
        sd.mmul("out", x, w);

        sd.setGraphExecutionMode(org.nd4j.autodiff.samediff.execution.GraphExecutionMode.AUTO);
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        try {
            float[] data = {1, 2, 3, 4};

            for (int iter = 0; iter < 6; iter++) {
                int srcDev = (iter % 2 == 0) ? dev0 : dev1;
                mgr.switchDevice(srcDev, "CrossDeviceTransferTest", "input-dev-" + srcDev);

                INDArray input = Nd4j.createFromArray(data).reshape(1, 4);
                Nd4j.getExecutioner().commit();

                mgr.switchDevice(dev0, "CrossDeviceTransferTest", "exec-dev0-iter-" + iter);
                java.util.Map<String, INDArray> result = sd.output(java.util.Map.of("x", input), "out");
                float[] vals = result.get("out").dup().data().asFloat();
                result.values().forEach(v -> { if (v.closeable() && !v.wasClosed()) v.close(); });

                // With identity weight, output should equal input
                assertArrayEquals(data, vals, 1e-4f,
                        "Iter " + iter + " (input from dev" + srcDev + ") incorrect — " +
                        "DSP stream/migration bug");

                input.close();
            }
        } finally {
            sd.close();
        }
    }
}
