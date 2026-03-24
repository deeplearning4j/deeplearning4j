package org.eclipse.deeplearning4j.nd4j.linalg.api.buffer;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.*;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.device.DeviceMemoryManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.time.Duration;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Regression test for the O(n^2) data transfer bug in NDArray::linspace().
 *
 * Root cause: NDArray::linspace() called p<T>() per element, and each p<T>()
 * triggered syncToDevice() which copied the ENTIRE buffer from host to device.
 * For N elements with a buffer of size B bytes:
 *   - Total data transfer = N * B = N * (N * elemSize) = O(N^2) bytes
 *   - 4M elements (16MB buffer): 4M * 16MB = 64TB of memcpy → hung for hours
 *
 * Fix: Write all elements directly to the host buffer, then sync once.
 * Total transfer becomes N * elemSize = O(N) bytes.
 *
 * The bug was exposed when memory simulation routed allocations to a non-peer
 * device, but the O(n^2) pattern existed for all backends.
 */
@Slf4j
@Tag("multi-gpu")
public class NonPeerDeadlockDiagnosticTest {

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
    }

    /**
     * Verify linspace completes in reasonable time for all sizes up to 4M elements.
     * Before the fix, 1M+ elements would hang due to O(n^2) sync.
     */
    @Test
    @Timeout(60)
    public void testLinspaceSizeProgression() {
        assumeTrue(isCudaBackend(), "Requires CUDA backend");
        assumeTrue(getNumDevices() >= 2, "Requires 2+ GPUs");

        long[] sizes = {1024, 4096, 16384, 65536, 262144, 1048576, 4194304};

        for (long size : sizes) {
            memoryManager.clearAllMemorySimulation();
            memoryManager.switchDevice(0, "DiagnosticTest", "reset");
            memoryManager.setSimulatedFreeMemory(0, 1L * 1024 * 1024);
            memoryManager.setSimulatedFreeMemory(1, 4L * 1024 * 1024 * 1024);
            memoryManager.setMemorySimulationEnabled(true);

            INDArray arr = Nd4j.linspace(1, size, size, DataType.FLOAT);
            assertNotNull(arr, "linspace(" + size + ") should succeed");
            assertEquals(size, arr.length(), "linspace(" + size + ") should have correct length");

            // Verify first and last values
            memoryManager.clearAllMemorySimulation();
            assertEquals(1.0f, arr.getFloat(0), 0.1f,
                    "First element of linspace(" + size + ") should be ~1.0");
            assertEquals((float) size, arr.getFloat(size - 1), 1.0f,
                    "Last element of linspace(" + size + ") should be ~" + size);

            arr.close();
        }
    }

    /**
     * Verify linspace with pre-allocated output on device 1 completes.
     * Before fix: even with simulation cleared before execution, linspace hung
     * because the C++ linspace loop called syncToDevice per element.
     */
    @Test
    @Timeout(30)
    public void testLinspaceWithPreallocatedOutputOnDevice1() {
        assumeTrue(isCudaBackend(), "Requires CUDA backend");
        assumeTrue(getNumDevices() >= 2, "Requires 2+ GPUs");

        long size = 4 * 1024 * 1024;

        // Allocate output on device 1 via simulation
        memoryManager.switchDevice(0, "DiagnosticTest", "setup");
        memoryManager.setSimulatedFreeMemory(0, 1L * 1024 * 1024);
        memoryManager.setSimulatedFreeMemory(1, 4L * 1024 * 1024 * 1024);
        memoryManager.setMemorySimulationEnabled(true);

        INDArray output = Nd4j.createUninitialized(DataType.FLOAT, size);
        memoryManager.clearAllMemorySimulation();

        // Execute Linspace with pre-allocated output on device 1
        var linspace = new org.nd4j.linalg.api.ops.impl.shape.Linspace(1.0, (double) size, size, DataType.FLOAT, true);
        linspace.addOutputArgument(output);
        Nd4j.getExecutioner().exec(linspace);

        assertEquals(1.0f, output.getFloat(0), 0.1f);
        assertEquals((float) size, output.getFloat(size - 1), 1.0f);
        output.close();
    }

    /**
     * Verify linspace + simulation end-to-end with data integrity.
     * The full path: allocate with simulation → linspace fills → read back.
     */
    @Test
    @Timeout(30)
    public void testLinspaceEndToEndWithSimulation() {
        assumeTrue(isCudaBackend(), "Requires CUDA backend");
        assumeTrue(getNumDevices() >= 2, "Requires 2+ GPUs");

        long size = 4 * 1024 * 1024;

        memoryManager.switchDevice(0, "DiagnosticTest", "setup");
        memoryManager.setSimulatedFreeMemory(0, 1L * 1024 * 1024);
        memoryManager.setSimulatedFreeMemory(1, 4L * 1024 * 1024 * 1024);
        memoryManager.setMemorySimulationEnabled(true);

        INDArray arr = Nd4j.linspace(1, size, size, DataType.FLOAT);

        memoryManager.clearAllMemorySimulation();

        // Verify data integrity via reduction
        double sum = arr.sumNumber().doubleValue();
        double expected = (double) size * ((double) size + 1.0) / 2.0;
        assertEquals(expected, sum, expected * 1e-4,
                "Sum of linspace(1, " + size + ") should be n*(n+1)/2");

        arr.close();
    }

    /**
     * Verify other ops still work on device 1 buffers (sanity check).
     */
    @Test
    @Timeout(30)
    public void testVariousOpsOnDevice1Buffer() {
        assumeTrue(isCudaBackend(), "Requires CUDA backend");
        assumeTrue(getNumDevices() >= 2, "Requires 2+ GPUs");

        long size = 4 * 1024 * 1024;

        memoryManager.switchDevice(0, "DiagnosticTest", "setup");
        memoryManager.setSimulatedFreeMemory(0, 1L * 1024 * 1024);
        memoryManager.setSimulatedFreeMemory(1, 4L * 1024 * 1024 * 1024);
        memoryManager.setMemorySimulationEnabled(true);

        INDArray a = Nd4j.createUninitialized(DataType.FLOAT, size);
        INDArray b = Nd4j.createUninitialized(DataType.FLOAT, size);

        memoryManager.clearAllMemorySimulation();

        a.assign(1.0f);
        b.assign(2.0f);

        INDArray result = a.add(b);
        assertEquals(3.0f, result.getFloat(0), 1e-5);

        double sum = a.sumNumber().doubleValue();
        assertEquals(size, sum, 1.0);

        a.close();
        b.close();
        result.close();
    }
}
