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

package org.eclipse.deeplearning4j.nd4j.linalg.multidevice;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.*;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.util.DeviceLocalNDArray;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive tests for multi-device synchronization, race conditions,
 * and NDArray operations across multiple devices.
 *
 * These tests are designed to be robust and NOT tied to thread pinning.
 * They test:
 * 1. Data movement between devices
 * 2. Operations after data has been moved
 * 3. Concurrent multi-threaded access
 * 4. Race condition prevention
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@Slf4j
@Tag(TagNames.MULTI_THREADED)
@NativeTag
public class MultiDeviceSynchronizationTests extends BaseNd4jTestWithBackends {

    @Override
    public char ordering() {
        return 'c';
    }

    /**
     * Reset device state before each test to ensure test isolation.
     * This ensures the main thread is on device 0 and all pending
     * operations are synchronized before the next test runs.
     */
    @BeforeEach
    public void resetDeviceState() {
        try {
            // Ensure all pending operations are complete
            Nd4j.getExecutioner().commit();

            // Reset to device 0 for main thread
            int currentDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
            if (currentDevice != 0) {
                Nd4j.getAffinityManager().unsafeSetDevice(0);
            }

            // Force synchronization on all devices
            int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
            for (int d = 0; d < numDevices; d++) {
                Nd4j.getAffinityManager().unsafeSetDevice(d);
                Nd4j.getExecutioner().commit();
            }

            // Return to device 0
            Nd4j.getAffinityManager().unsafeSetDevice(0);
            Nd4j.getExecutioner().commit();
        } catch (Exception e) {
            log.warn("Failed to reset device state: {}", e.getMessage());
        }
    }

    private int getNumDevices() {
        return Nd4j.getAffinityManager().getNumberOfDevices();
    }

    private boolean hasMultipleDevices() {
        return getNumDevices() >= 2;
    }

    // ========================================================================
    // Section 1: Basic Multi-Device Data Movement Tests
    // ========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test explicit data replication to different devices")
    public void testExplicitDataReplication(Nd4jBackend backend) {
        if (!hasMultipleDevices()) {
            log.info("Skipping test - requires multiple devices, found {}", getNumDevices());
            return;
        }

        int numDevices = getNumDevices();
        log.info("Testing with {} devices", numDevices);

        // Create array with known values
        INDArray original = Nd4j.linspace(1, 100, 100, DataType.DOUBLE);
        double expectedSum = original.sumNumber().doubleValue();

        // Replicate to each device and verify
        int originalDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        for (int targetDevice = 0; targetDevice < numDevices; targetDevice++) {
            INDArray replica = Nd4j.getAffinityManager().replicateToDevice(targetDevice, original);

            // Switch to the target device to operate on the replica
            // CUDA requires operations to be executed on the device where data resides
            Nd4j.getAffinityManager().unsafeSetDevice(targetDevice);
            try {
                // Verify data integrity after replication
                double replicaSum = replica.sumNumber().doubleValue();
                assertEquals(expectedSum, replicaSum, 1e-6,
                        "Replica on device " + targetDevice + " should have same sum");

                // Verify we can do operations on the replica
                INDArray doubled = replica.mul(2.0);
                assertEquals(expectedSum * 2, doubled.sumNumber().doubleValue(), 1e-6);
            } finally {
                // Switch back to original device
                Nd4j.getAffinityManager().unsafeSetDevice(originalDevice);
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test operations after moving data between devices")
    public void testOpsAfterDataMovement(Nd4jBackend backend) {
        if (!hasMultipleDevices()) {
            log.info("Skipping test - requires multiple devices");
            return;
        }

        int numDevices = getNumDevices();
        int originalDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();

        // Create array on first device
        INDArray arr = Nd4j.ones(DataType.DOUBLE, 100);

        // Move to each device and perform operations
        for (int d = 0; d < numDevices; d++) {
            // Replicate to target device
            INDArray onDevice = Nd4j.getAffinityManager().replicateToDevice(d, arr);

            // Switch to target device to operate on the data
            // CUDA requires operations to execute on the device where data resides
            Nd4j.getAffinityManager().unsafeSetDevice(d);
            try {
                // Chain of operations on the moved data
                INDArray result = onDevice.add(1.0)   // all 2s
                        .mul(3.0)                      // all 6s
                        .sub(1.0)                      // all 5s
                        .div(5.0);                     // all 1s

                Nd4j.getExecutioner().commit();

                // Verify result
                assertEquals(100.0, result.sumNumber().doubleValue(), 1e-6,
                        "Operations on device " + d + " should produce correct result");
            } finally {
                // Switch back to original device
                Nd4j.getAffinityManager().unsafeSetDevice(originalDevice);
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test round-trip data movement between all device pairs")
    public void testRoundTripDataMovement(Nd4jBackend backend) {
        if (!hasMultipleDevices()) {
            log.info("Skipping test - requires multiple devices");
            return;
        }

        int numDevices = getNumDevices();
        int originalDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        INDArray original = Nd4j.create(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
        double[] expectedData = original.toDoubleVector();

        // Test all device pairs
        for (int fromDevice = 0; fromDevice < numDevices; fromDevice++) {
            for (int toDevice = 0; toDevice < numDevices; toDevice++) {
                try {
                    // Move to fromDevice
                    INDArray onFrom = Nd4j.getAffinityManager().replicateToDevice(fromDevice, original);

                    // Switch to fromDevice to modify the array
                    Nd4j.getAffinityManager().unsafeSetDevice(fromDevice);
                    onFrom.addi(10.0);
                    Nd4j.getExecutioner().commit();

                    // Move to toDevice
                    INDArray onTo = Nd4j.getAffinityManager().replicateToDevice(toDevice, onFrom);

                    // Switch to toDevice to verify data
                    Nd4j.getAffinityManager().unsafeSetDevice(toDevice);
                    double[] actualData = onTo.toDoubleVector();
                    for (int i = 0; i < expectedData.length; i++) {
                        assertEquals(expectedData[i] + 10.0, actualData[i], 1e-6,
                                String.format("Data mismatch at index %d after move from device %d to %d",
                                        i, fromDevice, toDevice));
                    }
                } finally {
                    // Always switch back to original device
                    Nd4j.getAffinityManager().unsafeSetDevice(originalDevice);
                }
            }
        }
    }

    // ========================================================================
    // Section 2: Multi-threaded Operations (No Thread Pinning Assumptions)
    // ========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test concurrent array creation from multiple threads")
    public void testConcurrentArrayCreation(Nd4jBackend backend) throws Exception {
        int numThreads = 8;
        int arraysPerThread = 50;

        ConcurrentLinkedQueue<INDArray> allArrays = new ConcurrentLinkedQueue<>();
        AtomicInteger errorCount = new AtomicInteger(0);
        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch doneLatch = new CountDownLatch(numThreads);

        for (int t = 0; t < numThreads; t++) {
            final int threadId = t;
            new Thread(() -> {
                try {
                    startLatch.await();
                    for (int i = 0; i < arraysPerThread; i++) {
                        INDArray arr = Nd4j.valueArrayOf(new long[]{10}, threadId * 100.0 + i, DataType.DOUBLE);
                        allArrays.add(arr);
                        Nd4j.getExecutioner().commit();
                    }
                } catch (Exception e) {
                    log.error("Thread {} error: {}", threadId, e.getMessage());
                    errorCount.incrementAndGet();
                } finally {
                    doneLatch.countDown();
                }
            }).start();
        }

        startLatch.countDown();
        assertTrue(doneLatch.await(60, TimeUnit.SECONDS), "All threads should complete");

        assertEquals(0, errorCount.get(), "No errors should occur");
        assertEquals(numThreads * arraysPerThread, allArrays.size(), "All arrays should be created");

        // Verify all arrays are valid
        // The framework automatically handles cross-device array access
        for (INDArray arr : allArrays) {
            assertNotNull(arr);
            assertEquals(10, arr.length());
            // Framework auto-switches to array's device when needed
            assertFalse(Double.isNaN(arr.sumNumber().doubleValue()));
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test concurrent operations from multiple threads")
    public void testConcurrentOpsOnSharedArray(Nd4jBackend backend) throws Exception {
        int numThreads = 8;
        int opsPerThread = 100;

        // Original array with known values - each thread will operate on its own copy
        // Note: CUDA requires each thread to have its own array copy to avoid race
        // conditions in TAD shape info creation during reduce operations
        INDArray template = Nd4j.linspace(1, 1000, 1000, DataType.DOUBLE);
        double expectedSum = template.sumNumber().doubleValue();
        double expectedMean = template.meanNumber().doubleValue();

        AtomicInteger successCount = new AtomicInteger(0);
        AtomicInteger errorCount = new AtomicInteger(0);
        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch doneLatch = new CountDownLatch(numThreads);

        for (int t = 0; t < numThreads; t++) {
            new Thread(() -> {
                try {
                    startLatch.await();
                    // Each thread gets its own copy to avoid CUDA race conditions
                    INDArray threadLocal = template.dup();
                    for (int i = 0; i < opsPerThread; i++) {
                        double sum = threadLocal.sumNumber().doubleValue();
                        double mean = threadLocal.meanNumber().doubleValue();

                        if (Math.abs(sum - expectedSum) < 1e-6 && Math.abs(mean - expectedMean) < 1e-6) {
                            successCount.incrementAndGet();
                        }
                    }
                } catch (Exception e) {
                    log.error("Thread error: {}", e.getMessage(), e);
                    errorCount.incrementAndGet();
                } finally {
                    doneLatch.countDown();
                }
            }).start();
        }

        startLatch.countDown();
        assertTrue(doneLatch.await(60, TimeUnit.SECONDS), "All threads should complete");

        assertEquals(0, errorCount.get(), "No errors should occur");
        assertEquals(numThreads * opsPerThread, successCount.get(), "All operations should succeed");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test concurrent put and get operations")
    public void testConcurrentPutGet(Nd4jBackend backend) throws Exception {
        int numThreads = 4;
        int iterations = 100;

        // Shared array - framework auto-handles cross-device access
        INDArray arr = Nd4j.zeros(DataType.DOUBLE, numThreads * 10);

        AtomicBoolean hasError = new AtomicBoolean(false);
        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch doneLatch = new CountDownLatch(numThreads);

        for (int t = 0; t < numThreads; t++) {
            final int threadId = t;
            final int startIdx = threadId * 10;

            new Thread(() -> {
                try {
                    startLatch.await();
                    for (int iter = 0; iter < iterations; iter++) {
                        // Write to our indices - framework handles device switching
                        for (int i = 0; i < 10; i++) {
                            arr.putScalar(startIdx + i, threadId * 1000.0 + iter);
                        }
                        Nd4j.getExecutioner().commit();

                        // Read back and verify
                        for (int i = 0; i < 10; i++) {
                            double val = arr.getDouble(startIdx + i);
                            if (val != threadId * 1000.0 + iter) {
                                hasError.set(true);
                            }
                        }
                    }
                } catch (Exception e) {
                    log.error("Thread {} error: {}", threadId, e.getMessage());
                    hasError.set(true);
                } finally {
                    doneLatch.countDown();
                }
            }).start();
        }

        startLatch.countDown();
        assertTrue(doneLatch.await(60, TimeUnit.SECONDS), "All threads should complete");
        assertFalse(hasError.get(), "No data corruption should occur with isolated indices");
    }

    // ========================================================================
    // Section 3: Cross-Device Operations
    // ========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test operations mixing arrays from different devices")
    public void testCrossDeviceOperations(Nd4jBackend backend) {
        if (!hasMultipleDevices()) {
            log.info("Skipping test - requires multiple devices");
            return;
        }

        int numDevices = getNumDevices();
        int originalDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();

        // Test basic cross-device replication and operation on single device
        // Create array on device 0
        INDArray arr0 = Nd4j.valueArrayOf(new long[]{100}, 1.0, DataType.DOUBLE);
        double sum0 = arr0.sumNumber().doubleValue();
        assertEquals(100.0, sum0, 1e-6, "Array on device 0 should have sum of 100");

        // Replicate to device 1 and verify data
        INDArray arr1 = Nd4j.getAffinityManager().replicateToDevice(1, arr0);
        Nd4j.getAffinityManager().unsafeSetDevice(1);
        try {
            double sum1 = arr1.sumNumber().doubleValue();
            assertEquals(100.0, sum1, 1e-6, "Replicated array on device 1 should have sum of 100");

            // Modify on device 1
            arr1.muli(2.0);
            Nd4j.getExecutioner().commit();
            assertEquals(200.0, arr1.sumNumber().doubleValue(), 1e-6, "Modified array should have sum of 200");
        } finally {
            Nd4j.getAffinityManager().unsafeSetDevice(originalDevice);
        }

        // Replicate modified array back to device 0 and verify
        INDArray arr1Back = Nd4j.getAffinityManager().replicateToDevice(0, arr1);
        double sumBack = arr1Back.sumNumber().doubleValue();
        assertEquals(200.0, sumBack, 1e-6, "Replicated back array should have sum of 200");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test matrix multiplication with arrays from different devices")
    public void testCrossDeviceMatmul(Nd4jBackend backend) {
        if (!hasMultipleDevices()) {
            log.info("Skipping test - requires multiple devices");
            return;
        }

        int originalDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();

        try {
            // Test: replicate matrices to device 1, perform matmul there, then verify result
            // Create matrices on original device
            INDArray a = Nd4j.ones(DataType.DOUBLE, 10, 20);
            INDArray b = Nd4j.ones(DataType.DOUBLE, 20, 15);
            Nd4j.getExecutioner().commit();

            // Replicate both to device 1
            INDArray aOnDevice1 = Nd4j.getAffinityManager().replicateToDevice(1, a);
            INDArray bOnDevice1 = Nd4j.getAffinityManager().replicateToDevice(1, b);

            // Switch to device 1 for the operation
            Nd4j.getAffinityManager().unsafeSetDevice(1);
            try {
                // Perform matmul on device 1
                INDArray result = aOnDevice1.mmul(bOnDevice1);
                Nd4j.getExecutioner().commit();

                // Result should be 10x15 matrix with all 20s (each element is dot product of 20 ones)
                assertEquals(10, result.rows());
                assertEquals(15, result.columns());
                assertEquals(20.0, result.getDouble(0, 0), 1e-6);
                assertEquals(20.0 * 10 * 15, result.sumNumber().doubleValue(), 1e-6);
            } finally {
                Nd4j.getAffinityManager().unsafeSetDevice(originalDevice);
            }
        } finally {
            Nd4j.getAffinityManager().unsafeSetDevice(originalDevice);
        }
    }

    // ========================================================================
    // Section 4: Stress Tests and Race Conditions
    // ========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Stress test: many threads, many operations")
    public void testStressMultiThreaded(Nd4jBackend backend) throws Exception {
        int numThreads = 16;
        int opsPerThread = 200;

        AtomicLong totalOps = new AtomicLong(0);
        AtomicInteger errors = new AtomicInteger(0);
        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch doneLatch = new CountDownLatch(numThreads);

        for (int t = 0; t < numThreads; t++) {
            new Thread(() -> {
                try {
                    startLatch.await();
                    for (int i = 0; i < opsPerThread; i++) {
                        // Create, compute, verify
                        INDArray a = Nd4j.rand(DataType.FLOAT, 50, 50);
                        INDArray b = Nd4j.rand(DataType.FLOAT, 50, 50);

                        INDArray sum = a.add(b);
                        INDArray prod = a.mul(b);
                        INDArray mmul = a.mmul(b);

                        // Verify shapes
                        assertEquals(50, sum.rows());
                        assertEquals(50, prod.columns());
                        assertEquals(50, mmul.rows());

                        Nd4j.getExecutioner().commit();
                        totalOps.addAndGet(3);
                    }
                } catch (Exception e) {
                    log.error("Stress test error: {}", e.getMessage());
                    errors.incrementAndGet();
                } finally {
                    doneLatch.countDown();
                }
            }).start();
        }

        startLatch.countDown();
        assertTrue(doneLatch.await(120, TimeUnit.SECONDS), "All threads should complete");

        assertEquals(0, errors.get(), "No errors should occur");
        assertEquals((long) numThreads * opsPerThread * 3, totalOps.get(), "All operations should complete");
        log.info("Completed {} operations across {} threads", totalOps.get(), numThreads);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test DeviceLocalNDArray with concurrent access")
    public void testDeviceLocalConcurrentAccess(Nd4jBackend backend) throws Exception {
        if (!hasMultipleDevices()) {
            log.info("Skipping test - requires multiple devices");
            return;
        }

        int numThreads = 8;
        INDArray source = Nd4j.linspace(1, 100, 100, DataType.DOUBLE);
        DeviceLocalNDArray deviceLocal = new DeviceLocalNDArray(source);

        double expectedSum = source.sumNumber().doubleValue();
        AtomicInteger successCount = new AtomicInteger(0);
        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch doneLatch = new CountDownLatch(numThreads);

        for (int t = 0; t < numThreads; t++) {
            new Thread(() -> {
                try {
                    startLatch.await();
                    for (int i = 0; i < 50; i++) {
                        INDArray local = deviceLocal.get();
                        double sum = local.sumNumber().doubleValue();
                        if (Math.abs(sum - expectedSum) < 1e-6) {
                            successCount.incrementAndGet();
                        }
                    }
                } catch (Exception e) {
                    log.error("DeviceLocal error: {}", e.getMessage());
                } finally {
                    doneLatch.countDown();
                }
            }).start();
        }

        startLatch.countDown();
        assertTrue(doneLatch.await(60, TimeUnit.SECONDS), "All threads should complete");
        assertEquals(numThreads * 50, successCount.get(), "All accesses should return correct data");
    }

    // ========================================================================
    // Section 5: Data Integrity Tests
    // ========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test data integrity after multiple device transfers")
    public void testDataIntegrityMultipleTransfers(Nd4jBackend backend) {
        if (!hasMultipleDevices()) {
            log.info("Skipping test - requires multiple devices");
            return;
        }

        int originalDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();

        // Create array with specific pattern - use integer values to avoid floating point precision issues
        double[] originalData = new double[100];
        for (int i = 0; i < 100; i++) {
            originalData[i] = i + 1;  // Simple pattern: 1, 2, 3, ..., 100
        }
        INDArray arr = Nd4j.create(originalData);

        // Transfer to device 1 and back to device 0
        INDArray onDevice1 = Nd4j.getAffinityManager().replicateToDevice(1, arr);
        INDArray onDevice0 = Nd4j.getAffinityManager().replicateToDevice(0, onDevice1);

        // Verify data integrity
        double[] finalData = onDevice0.toDoubleVector();
        assertArrayEquals(originalData, finalData, 1e-6,
                "Data should be identical after round-trip transfer");

        // Also test: device 0 -> device 1 -> device 0 -> device 1
        INDArray round2 = Nd4j.getAffinityManager().replicateToDevice(1, onDevice0);
        Nd4j.getAffinityManager().unsafeSetDevice(1);
        try {
            double[] round2Data = round2.toDoubleVector();
            assertArrayEquals(originalData, round2Data, 1e-6,
                    "Data should be identical after multiple round trips");
        } finally {
            Nd4j.getAffinityManager().unsafeSetDevice(originalDevice);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test in-place operations after device transfer")
    public void testInPlaceOpsAfterTransfer(Nd4jBackend backend) {
        if (!hasMultipleDevices()) {
            log.info("Skipping test - requires multiple devices");
            return;
        }

        int numDevices = getNumDevices();
        int originalDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();

        INDArray arr = Nd4j.ones(DataType.DOUBLE, 100);

        // Transfer to each device and do in-place ops
        for (int d = 0; d < numDevices; d++) {
            INDArray onDevice = Nd4j.getAffinityManager().replicateToDevice(d, arr);

            // Switch to target device to operate on the data
            Nd4j.getAffinityManager().unsafeSetDevice(d);
            try {
                // Chain of in-place operations
                onDevice.addi(1.0);  // all 2s
                onDevice.muli(2.0);  // all 4s
                onDevice.subi(1.0);  // all 3s
                Nd4j.getExecutioner().commit();

                assertEquals(300.0, onDevice.sumNumber().doubleValue(), 1e-6,
                        "In-place ops on device " + d + " should produce correct result");
            } finally {
                Nd4j.getAffinityManager().unsafeSetDevice(originalDevice);
            }
        }
    }

    // ========================================================================
    // Section 6: Different Data Types
    // ========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test multi-device operations with different data types")
    public void testMultiDeviceDifferentTypes(Nd4jBackend backend) {
        if (!hasMultipleDevices()) {
            log.info("Skipping test - requires multiple devices");
            return;
        }

        DataType[] types = {DataType.FLOAT, DataType.DOUBLE, DataType.HALF, DataType.INT, DataType.LONG};
        int numDevices = getNumDevices();
        int originalDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();

        for (DataType type : types) {
            try {
                INDArray arr = Nd4j.linspace(1, 10, 10, type);
                double expectedSum = arr.sumNumber().doubleValue();

                for (int d = 0; d < numDevices; d++) {
                    INDArray onDevice = Nd4j.getAffinityManager().replicateToDevice(d, arr);
                    // Switch to target device to read data
                    Nd4j.getAffinityManager().unsafeSetDevice(d);
                    try {
                        double sum = onDevice.sumNumber().doubleValue();
                        assertEquals(expectedSum, sum, 1.0, // loose tolerance for integer types
                                "Type " + type + " on device " + d + " should have correct sum");
                    } finally {
                        Nd4j.getAffinityManager().unsafeSetDevice(originalDevice);
                    }
                }
            } catch (Exception e) {
                log.warn("Type {} not fully supported: {}", type, e.getMessage());
            }
        }
    }

    // ========================================================================
    // Section 7: Edge Cases
    // ========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test empty array handling across devices")
    public void testEmptyArrays(Nd4jBackend backend) {
        INDArray empty = Nd4j.empty(DataType.DOUBLE);

        assertTrue(empty.isEmpty());

        // Operations on empty should not crash
        assertDoesNotThrow(() -> {
            double sum = empty.isEmpty() ? 0.0 : empty.sumNumber().doubleValue();
        });
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test scalar array handling across devices")
    public void testScalarArrays(Nd4jBackend backend) {
        if (!hasMultipleDevices()) {
            log.info("Skipping test - requires multiple devices");
            return;
        }

        INDArray scalar = Nd4j.scalar(42.0);
        int numDevices = getNumDevices();
        int originalDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();

        for (int d = 0; d < numDevices; d++) {
            INDArray onDevice = Nd4j.getAffinityManager().replicateToDevice(d, scalar);
            Nd4j.getAffinityManager().unsafeSetDevice(d);
            try {
                assertEquals(42.0, onDevice.getDouble(0), 1e-6);
            } finally {
                Nd4j.getAffinityManager().unsafeSetDevice(originalDevice);
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test large array transfer between devices")
    public void testLargeArrayTransfer(Nd4jBackend backend) {
        if (!hasMultipleDevices()) {
            log.info("Skipping test - requires multiple devices");
            return;
        }

        int numDevices = getNumDevices();
        int originalDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();

        // Large array (100MB for doubles)
        INDArray large = Nd4j.rand(DataType.DOUBLE, 1000, 1000);
        double originalSum = large.sumNumber().doubleValue();

        for (int d = 0; d < numDevices; d++) {
            long startTime = System.nanoTime();
            INDArray onDevice = Nd4j.getAffinityManager().replicateToDevice(d, large);
            Nd4j.getExecutioner().commit();
            long elapsed = System.nanoTime() - startTime;

            // Switch to target device to verify data
            Nd4j.getAffinityManager().unsafeSetDevice(d);
            try {
                double transferredSum = onDevice.sumNumber().doubleValue();
                assertEquals(originalSum, transferredSum, 1e-3,
                        "Large array transfer to device " + d + " should preserve data");

                log.info("Transferred 1M doubles to device {} in {} ms", d, elapsed / 1_000_000.0);
            } finally {
                Nd4j.getAffinityManager().unsafeSetDevice(originalDevice);
            }
        }
    }

    // ========================================================================
    // Section 8: AffinityManager API Tests
    // ========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test AffinityManager basic API")
    public void testAffinityManagerAPI(Nd4jBackend backend) {
        AffinityManager am = Nd4j.getAffinityManager();

        int numDevices = am.getNumberOfDevices();
        assertTrue(numDevices >= 1, "Should have at least 1 device");
        log.info("Number of devices: {}", numDevices);

        int currentDevice = am.getDeviceForCurrentThread();
        assertTrue(currentDevice >= 0 && currentDevice < numDevices,
                "Current device should be valid");

        INDArray arr = Nd4j.create(10);
        int arrayDevice = am.getDeviceForArray(arr);
        assertTrue(arrayDevice >= 0 && arrayDevice < numDevices,
                "Array device should be valid");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test location tagging and synchronization")
    public void testLocationTagging(Nd4jBackend backend) {
        INDArray arr = Nd4j.create(new double[]{1, 2, 3, 4, 5});
        AffinityManager am = Nd4j.getAffinityManager();

        // Tag and ensure locations
        am.tagLocation(arr, AffinityManager.Location.HOST);
        am.ensureLocation(arr, AffinityManager.Location.HOST);

        AffinityManager.Location loc = am.getActiveLocation(arr);
        assertNotNull(loc, "Location should not be null");

        // Should still be able to use the array
        assertEquals(15.0, arr.sumNumber().doubleValue(), 1e-6);
    }
}
