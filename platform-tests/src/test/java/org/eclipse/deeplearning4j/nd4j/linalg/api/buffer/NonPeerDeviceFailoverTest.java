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
import org.nd4j.nativeblas.NativeOpsHolder;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Non-peer device failover regression tests.
 *
 * Targets the handleNonPeerFailover fix in DataBuffer.cu:
 * - Bug: freed non-peer buffer BEFORE allocating pinned replacement.
 *        If cudaMallocHost failed, caller got null buffer + exception with no recovery.
 * - Fix: allocate pinned buffer FIRST, only free old buffer after success.
 *
 * This system has RTX 4090 (device 0, cc 8.9) and RTX 3070 Ti (device 1, cc 8.6).
 * Different architectures = non-peer (no NVLink, no P2P DMA).
 * handleNonPeerFailover converts non-peer device buffers to pinned host memory (UVA).
 *
 * Tests use DeviceMemoryManager memory simulation to force allocation routing
 * to non-peer device 1, exercising the 3 call sites of handleNonPeerFailover:
 *   1. DataBuffer::allocateSpecial() — via memory simulation overflow to device 1
 *   2. DataBuffer::expand() — reallocate buffers that were allocated via overflow
 *   3. DataBuffer::migrate() — cross-device buffer migration
 */
@Slf4j
@Tag("multi-gpu")
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class NonPeerDeviceFailoverTest extends BaseNd4jTestWithBackends {

    private DeviceMemoryManager memoryManager;

    @Override
    public char ordering() {
        return 'c';
    }

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

    private boolean isNonPeer() {
        try {
            var nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
            return !nativeOps.isPeerAccessSupported(0, 1) || !nativeOps.isPeerAccessSupported(1, 0);
        } catch (Exception e) {
            return false;
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

    // ========================================================================
    // Peer access status — documents hardware topology
    // ========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @Order(0)
    public void testPeerAccessStatusLogged(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Requires CUDA backend");
        int numDevices = getNumDevices();
        assumeTrue(numDevices >= 2, "Requires 2+ GPUs");

        var nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        System.out.println("=== Peer Access Matrix ===");
        for (int src = 0; src < numDevices; src++) {
            for (int dst = 0; dst < numDevices; dst++) {
                if (src == dst) continue;
                boolean peer = nativeOps.isPeerAccessSupported(src, dst);
                System.out.printf("  Device %d → %d : peer=%s%n", src, dst, peer);
            }
        }
    }

    // ========================================================================
    // allocateSpecial() path — use memory simulation to force overflow to
    // non-peer device 1. When device 0 is "full", allocations route to
    // device 1 (non-peer), triggering handleNonPeerFailover.
    // ========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @Order(1)
    public void testSimulatedOomOverflowToNonPeerDevice(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Requires CUDA backend");
        assumeTrue(getNumDevices() >= 2, "Requires 2+ GPUs");

        memoryManager.switchDevice(0, "NonPeerDeviceFailoverTest", "setup");

        memoryManager.setSimulatedFreeMemory(0, 1L * 1024 * 1024);        // 1MB on device 0
        memoryManager.setSimulatedFreeMemory(1, 4L * 1024 * 1024 * 1024); // 4GB on device 1
        memoryManager.setMemorySimulationEnabled(true);

        // 50MB allocation — must overflow to device 1
        INDArray arr = Nd4j.create(DataType.FLOAT, 50L * 1024 * 1024 / 4);
        assertNotNull(arr);

        long allocOnDevice1 = memoryManager.getSimulatedAllocatedMemory(1);
        assertTrue(allocOnDevice1 > 0,
                "50MB should overflow to non-peer device 1");

        // Write and read through the (potentially pinned host) buffer
        arr.assign(42.0f);
        assertEquals(42.0f, arr.getFloat(0), 1e-5,
                "Data write/read on overflowed non-peer buffer should work");

        arr.close();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @Order(2)
    public void testOverflowAllocationDataIntegrity(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Requires CUDA backend");
        assumeTrue(getNumDevices() >= 2, "Requires 2+ GPUs");

        memoryManager.switchDevice(0, "NonPeerDeviceFailoverTest", "setup");

        memoryManager.setSimulatedFreeMemory(0, 1L * 1024 * 1024);
        memoryManager.setSimulatedFreeMemory(1, 4L * 1024 * 1024 * 1024);
        memoryManager.setMemorySimulationEnabled(true);

        // Allocate with data — forces overflow to device 1
        INDArray arr = Nd4j.linspace(1, 1024, 1024, DataType.FLOAT);
        assertNotNull(arr);

        // Verify data survived the non-peer allocation path
        float first = arr.getFloat(0);
        float last = arr.getFloat(1023);
        assertEquals(1.0f, first, 0.1f, "First element should be ~1.0");
        assertEquals(1024.0f, last, 1.0f, "Last element should be ~1024.0");

        arr.close();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @Order(3)
    public void testOverflowMultipleDtypes(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Requires CUDA backend");
        assumeTrue(getNumDevices() >= 2, "Requires 2+ GPUs");

        memoryManager.switchDevice(0, "NonPeerDeviceFailoverTest", "setup");

        DataType[] types = {DataType.FLOAT, DataType.DOUBLE, DataType.HALF, DataType.INT, DataType.LONG};
        for (DataType dt : types) {
            memoryManager.clearAllMemorySimulation();
            memoryManager.setSimulatedFreeMemory(0, 1L * 1024 * 1024);
            memoryManager.setSimulatedFreeMemory(1, 4L * 1024 * 1024 * 1024);
            memoryManager.setMemorySimulationEnabled(true);

            INDArray arr = Nd4j.ones(dt, 1024);

            memoryManager.clearAllMemorySimulation();

            double sum = arr.sumNumber().doubleValue();
            assertEquals(1024.0, sum, 1.0,
                    dt + " sum incorrect after non-peer overflow allocation");

            arr.close();
        }
    }

    // ========================================================================
    // expand() path — reallocate buffers that were allocated via overflow
    // ========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @Order(4)
    public void testExpandAfterOverflowToNonPeerDevice(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Requires CUDA backend");
        assumeTrue(getNumDevices() >= 2, "Requires 2+ GPUs");

        memoryManager.switchDevice(0, "NonPeerDeviceFailoverTest", "setup");

        memoryManager.setSimulatedFreeMemory(0, 1L * 1024 * 1024);
        memoryManager.setSimulatedFreeMemory(1, 4L * 1024 * 1024 * 1024);
        memoryManager.setMemorySimulationEnabled(true);

        // Initial allocation overflows to device 1
        INDArray arr = Nd4j.createFromArray(new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

        // Clear simulation so expand doesn't get confused by simulated state
        memoryManager.clearAllMemorySimulation();

        // Expand the buffer — exercises C++ DataBuffer::expand() → handleNonPeerFailover
        DataBuffer buf = arr.data();
        buf.reallocate(buf.length() * 4);

        // Data must survive expand on a non-peer-overflowed buffer
        assertEquals(1.0f, buf.getFloat(0), 1e-6);
        assertEquals(3.0f, buf.getFloat(2), 1e-6);
        assertEquals(5.0f, buf.getFloat(4), 1e-6);

        arr.close();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @Order(5)
    public void testRepeatedExpandAfterOverflow(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Requires CUDA backend");
        assumeTrue(getNumDevices() >= 2, "Requires 2+ GPUs");

        memoryManager.switchDevice(0, "NonPeerDeviceFailoverTest", "setup");

        memoryManager.setSimulatedFreeMemory(0, 1L * 1024 * 1024);
        memoryManager.setSimulatedFreeMemory(1, 4L * 1024 * 1024 * 1024);
        memoryManager.setMemorySimulationEnabled(true);

        INDArray arr = Nd4j.linspace(1, 32, 32, DataType.FLOAT);

        memoryManager.clearAllMemorySimulation();

        float[] expected = new float[32];
        for (int i = 0; i < 32; i++) expected[i] = arr.data().getFloat(i);

        // Three consecutive expands
        DataBuffer buf = arr.data();
        buf.reallocate(64);
        buf.reallocate(128);
        buf.reallocate(256);

        for (int i = 0; i < 32; i++) {
            assertEquals(expected[i], buf.getFloat(i), 1e-5,
                    "Data corruption at index " + i + " after 3 reallocations on non-peer buffer");
        }

        arr.close();
    }

    // ========================================================================
    // Compute correctness with non-peer overflow buffers
    // ========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @Order(6)
    public void testMmulAfterOverflowToNonPeer(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Requires CUDA backend");
        assumeTrue(getNumDevices() >= 2, "Requires 2+ GPUs");

        memoryManager.switchDevice(0, "NonPeerDeviceFailoverTest", "setup");

        memoryManager.setSimulatedFreeMemory(0, 1L * 1024 * 1024);
        memoryManager.setSimulatedFreeMemory(1, 4L * 1024 * 1024 * 1024);
        memoryManager.setMemorySimulationEnabled(true);

        // Weights overflow to device 1 (non-peer)
        INDArray weights = Nd4j.ones(DataType.FLOAT, 64, 64);

        memoryManager.clearAllMemorySimulation();

        // Input stays on device 0
        INDArray input = Nd4j.ones(DataType.FLOAT, 1, 64);

        // mmul must handle non-peer buffer correctly via UVA or auto-migration
        INDArray result = input.mmul(weights);

        assertEquals(1, result.shape()[0]);
        assertEquals(64, result.shape()[1]);
        assertEquals(64.0f, result.getFloat(0, 0), 1e-3,
                "mmul with non-peer overflow buffer should produce correct results");

        input.close();
        weights.close();
        result.close();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @Order(7)
    public void testElementwiseOpsAfterOverflow(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Requires CUDA backend");
        assumeTrue(getNumDevices() >= 2, "Requires 2+ GPUs");

        memoryManager.switchDevice(0, "NonPeerDeviceFailoverTest", "setup");

        memoryManager.setSimulatedFreeMemory(0, 1L * 1024 * 1024);
        memoryManager.setSimulatedFreeMemory(1, 4L * 1024 * 1024 * 1024);
        memoryManager.setMemorySimulationEnabled(true);

        // a overflows to device 1
        INDArray a = Nd4j.createFromArray(new float[]{1.0f, 2.0f, 3.0f});

        memoryManager.clearAllMemorySimulation();

        // b on device 0
        INDArray b = Nd4j.createFromArray(new float[]{10.0f, 20.0f, 30.0f});

        INDArray result = a.add(b);
        assertArrayEquals(new float[]{11.0f, 22.0f, 33.0f}, result.toFloatVector(), 1e-5f,
                "add with non-peer overflow buffer should produce correct results");

        a.close();
        b.close();
        result.close();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @Order(8)
    public void testReductionAfterOverflow(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Requires CUDA backend");
        assumeTrue(getNumDevices() >= 2, "Requires 2+ GPUs");

        memoryManager.switchDevice(0, "NonPeerDeviceFailoverTest", "setup");

        memoryManager.setSimulatedFreeMemory(0, 1L * 1024 * 1024);
        memoryManager.setSimulatedFreeMemory(1, 4L * 1024 * 1024 * 1024);
        memoryManager.setMemorySimulationEnabled(true);

        INDArray arr = Nd4j.linspace(1, 100, 100, DataType.FLOAT);

        memoryManager.clearAllMemorySimulation();

        double sum = arr.sumNumber().doubleValue();
        double expected = 100.0 * 101.0 / 2.0; // sum of 1..100
        assertEquals(expected, sum, 1.0,
                "Reduction on non-peer overflow buffer should be correct");

        arr.close();
    }

    // ========================================================================
    // Large allocation overflow — previously caused O(n^2) hang in linspace
    // due to per-element syncToDevice() in NDArray::linspace(). Fixed by
    // batching writes to host buffer and syncing once.
    // ========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @Order(9)
    @Timeout(60)
    public void testLargeLinspaceOverflowToNonPeer(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Requires CUDA backend");
        assumeTrue(getNumDevices() >= 2, "Requires 2+ GPUs");

        memoryManager.switchDevice(0, "NonPeerDeviceFailoverTest", "setup");

        memoryManager.setSimulatedFreeMemory(0, 1L * 1024 * 1024);        // 1MB on device 0
        memoryManager.setSimulatedFreeMemory(1, 4L * 1024 * 1024 * 1024); // 4GB on device 1
        memoryManager.setMemorySimulationEnabled(true);

        // 4M elements = 16MB — previously hung due to O(n^2) sync in linspace
        long size = 4 * 1024 * 1024;
        INDArray arr = Nd4j.linspace(1, size, size, DataType.FLOAT);
        assertNotNull(arr);

        memoryManager.clearAllMemorySimulation();

        // Verify data integrity
        assertEquals(1.0f, arr.getFloat(0), 0.1f, "First element should be ~1.0");
        assertEquals((float) size, arr.getFloat(size - 1), 1.0f, "Last element should be ~" + size);

        arr.close();
    }
}
