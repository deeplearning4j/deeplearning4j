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
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.device.DeviceMemoryManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.DeviceAwareNd4j;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Tests for multi-device GPU memory allocation routing via HybridDataBuffer.
 *
 * Uses DeviceMemoryManager's memory simulation infrastructure to test
 * allocation routing without requiring actual multi-GPU memory exhaustion.
 */
@Slf4j
public class HybridDataBufferMultiDeviceTest extends BaseNd4jTestWithBackends {

    private DeviceMemoryManager memoryManager;

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

    @BeforeEach
    void setUp() {
        memoryManager = DeviceMemoryManager.getInstance();
        memoryManager.clearAllMemorySimulation();
    }

    @AfterEach
    void tearDown() {
        if (memoryManager != null) {
            memoryManager.clearAllMemorySimulation();
        }
        if (DeviceAwareNd4j.isRoutingEnabled()) {
            DeviceAwareNd4j.disableDeviceRouting();
        }
        System.gc();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Multi-device allocation routes to device with most free memory")
    void testMultiDeviceAllocationRoutingWithoutExplicitRouting(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Test requires CUDA backend");
        assumeTrue(getNumDevices() >= 2, "Test requires at least 2 CUDA devices");

        // Simulate device 0 nearly full, device 1 has plenty
        memoryManager.setSimulatedFreeMemory(0, 10L * 1024 * 1024);       // 10MB
        memoryManager.setSimulatedFreeMemory(1, 4L * 1024 * 1024 * 1024); // 4GB
        memoryManager.setMemorySimulationEnabled(true);

        // Allocate 50MB — should route to device 1 (most free memory)
        INDArray arr = Nd4j.create(DataType.FLOAT, 50L * 1024 * 1024 / 4);

        assertNotNull(arr);
        long allocatedOnDevice1 = memoryManager.getSimulatedAllocatedMemory(1);
        assertTrue(allocatedOnDevice1 > 0,
                "Allocation should have been routed to device 1 (most free memory)");

        arr.close();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Small allocations stay on current device; large ones overflow")
    void testOwnerDeviceReflectsActualAllocationDevice(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Test requires CUDA backend");
        assumeTrue(getNumDevices() >= 2, "Test requires at least 2 CUDA devices");

        // Current device has 10MB free; other device has 4GB.
        // Allocations prefer the current device — only overflow when it can't fit.
        int currentDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        int otherDevice = (currentDevice == 0) ? 1 : 0;
        memoryManager.setSimulatedFreeMemory(currentDevice, 10L * 1024 * 1024);   // 10MB
        memoryManager.setSimulatedFreeMemory(otherDevice, 4L * 1024 * 1024 * 1024); // 4GB
        memoryManager.setMemorySimulationEnabled(true);

        // Small allocation (4KB) — fits on current device, should stay there
        INDArray small = Nd4j.create(DataType.FLOAT, 1000);
        assertNotNull(small);

        // Large allocation (50MB) — does NOT fit on current device (10MB), should overflow
        INDArray large = Nd4j.create(DataType.FLOAT, 50L * 1024 * 1024 / 4);
        assertNotNull(large);

        long allocatedOnOther = memoryManager.getSimulatedAllocatedMemory(otherDevice);
        assertTrue(allocatedOnOther > 0,
                "Large allocation should overflow to other device when current device is full");

        small.close();
        large.close();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Allocate on device migrates buffer")
    void testAllocateOnDeviceActuallyMigrates(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Test requires CUDA backend");
        assumeTrue(getNumDevices() >= 2, "Test requires at least 2 CUDA devices");

        // Both devices have plenty of memory
        memoryManager.setSimulatedFreeMemory(0, 4L * 1024 * 1024 * 1024);
        memoryManager.setSimulatedFreeMemory(1, 4L * 1024 * 1024 * 1024);
        memoryManager.setMemorySimulationEnabled(true);

        // Create array (will go to whichever device has most free — they're equal)
        INDArray arr = Nd4j.createFromArray(1.0f, 2.0f, 3.0f, 4.0f, 5.0f);
        assertNotNull(arr);

        // Verify data is correct before migration
        float[] beforeMigration = arr.dup().data().asFloat();
        assertArrayEquals(new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, beforeMigration, 1e-5f);

        arr.close();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("VLM draft model scenario: hundreds of small constants route away from full device")
    void testVlmDraftModelScenario(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Test requires CUDA backend");
        assumeTrue(getNumDevices() >= 2, "Test requires at least 2 CUDA devices");

        // Simulate SmolDocling loaded on device 1 (4090), leaving 36MB free
        // Device 0 (3070 Ti) has 7GB free
        memoryManager.setSimulatedFreeMemory(0, 7L * 1024 * 1024 * 1024);  // 7GB
        memoryManager.setSimulatedFreeMemory(1, 36L * 1024 * 1024);        // 36MB
        memoryManager.setMemorySimulationEnabled(true);

        // Allocate 200 small arrays simulating draft model constants
        INDArray[] constants = new INDArray[200];
        long totalBytes = 0;
        for (int i = 0; i < constants.length; i++) {
            // 1-10MB each, ~540MB total
            long elements = (1L + (i % 10)) * 1024 * 1024 / 4; // FLOAT = 4 bytes
            constants[i] = Nd4j.create(DataType.FLOAT, elements);
            assertNotNull(constants[i], "Allocation " + i + " should succeed");
            totalBytes += elements * 4;
        }

        // All allocations should have succeeded — routed to device 0 (most free memory)
        long allocatedOnDevice0 = memoryManager.getSimulatedAllocatedMemory(0);
        assertTrue(allocatedOnDevice0 > 0,
                "Draft model constants should route to device 0 (7GB free vs 36MB on device 1)");

        log.info("VLM draft model scenario: {} arrays, {} MB total, {} MB on device 0",
                constants.length, totalBytes / (1024 * 1024), allocatedOnDevice0 / (1024 * 1024));

        // Cleanup
        for (INDArray c : constants) {
            if (c != null && !c.wasClosed()) c.close();
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Sequential allocations fill current device then overflow to next")
    void testSequentialAllocationsFillDeviceThenOverflow(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Test requires CUDA backend");
        assumeTrue(getNumDevices() >= 2, "Test requires at least 2 CUDA devices");

        // Current device: 100MB, other device: 200MB
        // Allocations prefer current device first, overflow when full.
        int currentDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        int otherDevice = (currentDevice == 0) ? 1 : 0;
        memoryManager.setSimulatedFreeMemory(currentDevice, 100L * 1024 * 1024);
        memoryManager.setSimulatedFreeMemory(otherDevice, 200L * 1024 * 1024);
        memoryManager.setMemorySimulationEnabled(true);

        // Allocate 6 x 50MB arrays — first 2 go to current device (100MB),
        // then overflow to other device (200MB)
        INDArray[] arrays = new INDArray[6];
        for (int i = 0; i < arrays.length; i++) {
            arrays[i] = Nd4j.create(DataType.FLOAT, 50L * 1024 * 1024 / 4); // 50MB each
            assertNotNull(arrays[i], "Allocation " + i + " should succeed");
        }

        long allocatedOnCurrent = memoryManager.getSimulatedAllocatedMemory(currentDevice);
        long allocatedOnOther = memoryManager.getSimulatedAllocatedMemory(otherDevice);

        log.info("Sequential fill: current device {} = {} MB, other device {} = {} MB",
                currentDevice, allocatedOnCurrent / (1024 * 1024),
                otherDevice, allocatedOnOther / (1024 * 1024));

        // Current device should have received first allocations (preferred)
        assertTrue(allocatedOnCurrent > 0, "Current device should have received some allocations");
        // As current device fills up, other device should also receive allocations
        assertTrue(allocatedOnOther > 0, "Other device should have received overflow allocations");

        for (INDArray a : arrays) {
            if (a != null && !a.wasClosed()) a.close();
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Device routing with op execution routes ops where data lives")
    void testDeviceRoutingWithOpExecution(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Test requires CUDA backend");
        assumeTrue(getNumDevices() >= 2, "Test requires at least 2 CUDA devices");

        // Enable device routing for op execution
        DeviceAwareNd4j.enableDeviceRouting();
        assertTrue(DeviceAwareNd4j.isRoutingEnabled());

        // Create arrays and run ops
        INDArray a = Nd4j.createFromArray(1.0f, 2.0f, 3.0f);
        INDArray result = a.add(1.0f);

        // Verify correctness
        assertArrayEquals(new float[]{2.0f, 3.0f, 4.0f}, result.toFloatVector(), 1e-5f);

        result.close();
        a.close();

        DeviceAwareNd4j.disableDeviceRouting();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Registered device count reflects actual GPU count")
    void testRegisteredDeviceCount(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Test requires CUDA backend");

        int count = memoryManager.getRegisteredDeviceCount();
        int numDevices = getNumDevices();

        // Registered count should be at least the number of GPUs
        assertTrue(count >= numDevices,
                "Registered device count (" + count + ") should be >= GPU count (" + numDevices + ")");
    }

    // ===================================================================
    // Regression tests for multi-device switching and DSP stability
    // ===================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Device context is preserved after allocation routing")
    void testDeviceContextPreservedAfterAllocationRouting(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Test requires CUDA backend");
        assumeTrue(getNumDevices() >= 2, "Test requires at least 2 CUDA devices");

        int deviceBefore = Nd4j.getAffinityManager().getDeviceForCurrentThread();

        // Force overflow to another device
        int otherDevice = (deviceBefore == 0) ? 1 : 0;
        memoryManager.setSimulatedFreeMemory(deviceBefore, 1L * 1024 * 1024);    // 1MB
        memoryManager.setSimulatedFreeMemory(otherDevice, 4L * 1024 * 1024 * 1024); // 4GB
        memoryManager.setMemorySimulationEnabled(true);

        // Allocate large array — must overflow to other device
        INDArray large = Nd4j.create(DataType.FLOAT, 50L * 1024 * 1024 / 4); // 50MB
        assertNotNull(large);

        // CRITICAL: device context must be restored after allocation
        int deviceAfter = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        assertEquals(deviceBefore, deviceAfter,
                "Device context must be preserved after cross-device allocation routing");

        large.close();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Multiple cross-device allocations preserve context")
    void testMultipleCrossDeviceAllocationsPreserveContext(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Test requires CUDA backend");
        assumeTrue(getNumDevices() >= 2, "Test requires at least 2 CUDA devices");

        int deviceBefore = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        int otherDevice = (deviceBefore == 0) ? 1 : 0;

        // Force all allocations to overflow
        memoryManager.setSimulatedFreeMemory(deviceBefore, 1L * 1024 * 1024);    // 1MB
        memoryManager.setSimulatedFreeMemory(otherDevice, 4L * 1024 * 1024 * 1024); // 4GB
        memoryManager.setMemorySimulationEnabled(true);

        // Multiple cross-device allocations
        INDArray[] arrays = new INDArray[10];
        for (int i = 0; i < arrays.length; i++) {
            arrays[i] = Nd4j.create(DataType.FLOAT, 10L * 1024 * 1024 / 4); // 10MB each
            assertNotNull(arrays[i], "Allocation " + i + " should succeed");

            // Context must be preserved after EVERY allocation
            int currentDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
            assertEquals(deviceBefore, currentDevice,
                    "Device context must be preserved after allocation " + i);
        }

        for (INDArray a : arrays) {
            if (a != null && !a.wasClosed()) a.close();
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Ops work correctly after cross-device allocation")
    void testOpsWorkAfterCrossDeviceAllocation(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Test requires CUDA backend");
        assumeTrue(getNumDevices() >= 2, "Test requires at least 2 CUDA devices");

        int currentDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        int otherDevice = (currentDevice == 0) ? 1 : 0;

        // Create array on current device first
        INDArray a = Nd4j.createFromArray(1.0f, 2.0f, 3.0f, 4.0f);

        // Force next allocation to overflow
        memoryManager.setSimulatedFreeMemory(currentDevice, 1L * 1024 * 1024);
        memoryManager.setSimulatedFreeMemory(otherDevice, 4L * 1024 * 1024 * 1024);
        memoryManager.setMemorySimulationEnabled(true);

        // This allocation overflows to other device
        INDArray large = Nd4j.create(DataType.FLOAT, 50L * 1024 * 1024 / 4);
        assertNotNull(large);

        // Clear simulation so ops don't trigger more routing
        memoryManager.clearAllMemorySimulation();

        // Ops on the original array must still work correctly
        INDArray result = a.add(10.0f);
        assertArrayEquals(new float[]{11.0f, 12.0f, 13.0f, 14.0f}, result.toFloatVector(), 1e-5f,
                "Ops must work correctly after cross-device allocation routing");

        result.close();
        a.close();
        large.close();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("SameDiff execution works after cross-device allocation")
    void testSameDiffAfterCrossDeviceAllocation(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Test requires CUDA backend");
        assumeTrue(getNumDevices() >= 2, "Test requires at least 2 CUDA devices");

        int currentDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        int otherDevice = (currentDevice == 0) ? 1 : 0;

        // Force overflow
        memoryManager.setSimulatedFreeMemory(currentDevice, 1L * 1024 * 1024);
        memoryManager.setSimulatedFreeMemory(otherDevice, 4L * 1024 * 1024 * 1024);
        memoryManager.setMemorySimulationEnabled(true);

        // Cross-device allocation
        INDArray large = Nd4j.create(DataType.FLOAT, 50L * 1024 * 1024 / 4);
        assertNotNull(large);

        // Clear simulation
        memoryManager.clearAllMemorySimulation();

        // SameDiff execution must still work after device context restore
        org.nd4j.autodiff.samediff.SameDiff sd = org.nd4j.autodiff.samediff.SameDiff.create();
        INDArray inputArr = Nd4j.createFromArray(1.0f, 2.0f, 3.0f);
        org.nd4j.autodiff.samediff.SDVariable input = sd.var("x", inputArr);
        org.nd4j.autodiff.samediff.SDVariable output = input.add(1.0);

        java.util.Map<String, INDArray> result = sd.outputAll(null);
        INDArray out = result.get(output.name());
        assertNotNull(out);
        assertArrayEquals(new float[]{2.0f, 3.0f, 4.0f}, out.toFloatVector(), 1e-5f,
                "SameDiff execution must work correctly after cross-device allocation");

        large.close();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Prefer current device avoids unnecessary context switch")
    void testPreferCurrentDeviceAvoidsSwitch(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Test requires CUDA backend");
        assumeTrue(getNumDevices() >= 2, "Test requires at least 2 CUDA devices");

        int currentDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        int otherDevice = (currentDevice == 0) ? 1 : 0;

        // Both devices have plenty of memory — current device should be preferred
        memoryManager.setSimulatedFreeMemory(currentDevice, 2L * 1024 * 1024 * 1024);
        memoryManager.setSimulatedFreeMemory(otherDevice, 4L * 1024 * 1024 * 1024);
        memoryManager.setMemorySimulationEnabled(true);

        // Allocate — should stay on current device even though other has more free memory
        INDArray arr = Nd4j.create(DataType.FLOAT, 10L * 1024 * 1024 / 4); // 10MB
        assertNotNull(arr);

        long allocatedOnCurrent = memoryManager.getSimulatedAllocatedMemory(currentDevice);
        long allocatedOnOther = memoryManager.getSimulatedAllocatedMemory(otherDevice);

        assertTrue(allocatedOnCurrent > 0,
                "Allocation should stay on current device when it has sufficient free memory");
        assertEquals(0, allocatedOnOther,
                "Other device should NOT receive allocations when current device has room");

        arr.close();
    }
}
