package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.device.DeviceMemoryManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Environment;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.DeviceLocalNDArray;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Regression tests for multi-GPU buffer management and fusion scoring environment flags.
 *
 * Issue 1: handleNonPeerFailover freed buffer before replacement allocation —
 *          if pinned allocation failed, caller got null buffer with no recovery.
 *          Fix: allocate new pinned buffer BEFORE freeing old non-peer buffer.
 *
 * Issue 2: FusionScoring had dead variables (nearAttnA/nearAttnB) and an overly
 *          aggressive attention bonus (50.0f dominating typical 30-40 scores).
 *          Fix: removed dead vars, reduced bonus to 15.0f.
 *
 * Coverage targets:
 * - DataBuffer::expand() via reallocate() — primary path for the handleNonPeerFailover fix
 * - DataBuffer::allocateSpecial() via normal array creation
 * - DataBuffer::migrate() via replicateToDevice
 * - Multiple data types through allocation paths
 * - View safety after buffer operations
 * - Environment flag round-trips for fusion scoring configuration
 * - Environment flag defaults
 */
@Tag("multi-gpu")
public class MultiGpuBufferRegressionTest {

    private static final class InspectableDeviceLocalNDArray extends DeviceLocalNDArray {
        private InspectableDeviceLocalNDArray(INDArray array) {
            super(array, true);
        }

        private INDArray delayedSnapshot() {
            return delayedArray;
        }

        private int pendingSourceFor(int deviceId) {
            return updatesMap.get(deviceId).get();
        }
    }

    // ========================================================================
    // DataBuffer::expand() path — exercises handleNonPeerFailover in expand()
    // reallocate() calls ptrDataBuffer.expand() which calls C++ DataBuffer::expand()
    // ========================================================================

    /**
     * Verify DataBuffer.reallocate() preserves existing data after buffer expansion.
     * This exercises the C++ DataBuffer::expand() → handleNonPeerFailover path.
     * The bug: expand() freed the old buffer before allocating the new pinned buffer,
     * so if pinned allocation failed the caller got a null buffer.
     */
    @Test
    public void testReallocatePreservesData() {
        INDArray arr = Nd4j.createFromArray(new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
        DataBuffer buf = arr.data();
        long originalLength = buf.length();

        // Reallocate to double the length — triggers C++ expand()
        buf.reallocate(originalLength * 2);

        // Original data must survive the reallocation
        assertEquals(1.0f, buf.getFloat(0), 1e-6, "Element 0 lost after reallocate");
        assertEquals(2.0f, buf.getFloat(1), 1e-6, "Element 1 lost after reallocate");
        assertEquals(3.0f, buf.getFloat(2), 1e-6, "Element 2 lost after reallocate");
        assertEquals(4.0f, buf.getFloat(3), 1e-6, "Element 3 lost after reallocate");
        assertEquals(5.0f, buf.getFloat(4), 1e-6, "Element 4 lost after reallocate");

        arr.close();
    }

    /**
     * Verify repeated reallocations (grow → grow → grow) don't corrupt data.
     * Each reallocate triggers a fresh expand() → handleNonPeerFailover cycle.
     */
    @Test
    public void testRepeatedReallocatePreservesData() {
        INDArray arr = Nd4j.linspace(1, 32, 32, DataType.FLOAT);
        DataBuffer buf = arr.data();

        float[] expected = new float[32];
        for (int i = 0; i < 32; i++) expected[i] = buf.getFloat(i);

        // Grow three times
        buf.reallocate(64);
        buf.reallocate(128);
        buf.reallocate(256);

        // Original 32 elements must survive all three reallocations
        for (int i = 0; i < 32; i++) {
            assertEquals(expected[i], buf.getFloat(i), 1e-5,
                    "Data corruption at index " + i + " after 3 reallocations");
        }

        arr.close();
    }

    /**
     * Verify reallocate works with a large buffer (1M elements = 4MB FLOAT).
     * Larger allocations are more likely to trigger failover on multi-GPU systems.
     */
    @Test
    public void testLargeBufferReallocate() {
        int size = 1024 * 1024; // 1M elements = 4MB
        INDArray arr = Nd4j.linspace(0, size - 1, size, DataType.FLOAT);
        DataBuffer buf = arr.data();

        // Spot-check before reallocate
        assertEquals(0.0f, buf.getFloat(0), 1e-1);
        assertEquals((float)(size / 2), buf.getFloat(size / 2), 1.0f);
        assertEquals((float)(size - 1), buf.getFloat(size - 1), 1.0f);

        // Double the buffer
        buf.reallocate(size * 2L);

        // Spot-check after reallocate
        assertEquals(0.0f, buf.getFloat(0), 1e-1, "First element corrupted after large reallocate");
        assertEquals((float)(size / 2), buf.getFloat(size / 2), 1.0f,
                "Middle element corrupted after large reallocate");
        assertEquals((float)(size - 1), buf.getFloat(size - 1), 1.0f,
                "Last element corrupted after large reallocate");

        arr.close();
    }

    // ========================================================================
    // DataBuffer::allocateSpecial() path — exercises handleNonPeerFailover
    // in initial device allocation (every Nd4j.create/zeros/ones goes here)
    // ========================================================================

    /**
     * Verify allocation and data correctness across multiple data types.
     * Each dtype has different element sizes, exercising different allocation sizes.
     */
    @ParameterizedTest
    @EnumSource(value = DataType.class, names = {"FLOAT", "DOUBLE", "HALF", "INT", "LONG", "SHORT"})
    public void testAllocationAcrossDataTypes(DataType dtype) {
        int len = 1024;
        INDArray arr = Nd4j.ones(dtype, len);

        // Verify all elements are 1
        for (int i = 0; i < Math.min(len, 10); i++) {
            assertEquals(1.0, arr.getDouble(i), 1e-5,
                    dtype + " element " + i + " should be 1.0");
        }

        // Verify via reduction (sums all elements)
        double sum = arr.sumNumber().doubleValue();
        assertEquals(len, sum, 1e-2, dtype + " sum should be " + len);

        arr.close();
    }

    /**
     * Verify cross-device buffer replication produces correct data.
     * dup() allocates a new buffer (allocateSpecial) and copies data.
     * On multi-GPU, this may trigger failover + handleNonPeerFailover.
     */
    @Test
    public void testCrossDeviceReplicationCorrectness() {
        INDArray original = Nd4j.linspace(1, 100, 100, DataType.FLOAT);
        float[] expectedData = original.toFloatVector();

        INDArray replica = original.dup();

        float[] replicaData = replica.toFloatVector();
        assertEquals(expectedData.length, replicaData.length, "Replica length should match");
        for (int i = 0; i < expectedData.length; i++) {
            assertEquals(expectedData[i], replicaData[i], 1e-5,
                    "Replica data mismatch at index " + i);
        }

        original.close();
        replica.close();
    }

    // ========================================================================
    // DataBuffer::migrate() path — exercises handleNonPeerFailover in migrate()
    // replicateToDevice triggers buffer migration to a specific GPU
    // ========================================================================

    /**
     * Verify replicateToDevice(0) produces correct data and device assignment.
     * On single-GPU this is a same-device copy; on multi-GPU it exercises migrate().
     */
    @Test
    public void testReplicateToDeviceCorrectness() {
        INDArray original = Nd4j.linspace(1, 256, 256, DataType.FLOAT);
        float[] expected = original.toFloatVector();

        // Replicate to device 0 (always exists)
        INDArray replica = Nd4j.getAffinityManager().replicateToDevice(0, original);

        int replicaDevice = Nd4j.getAffinityManager().getDeviceForArray(replica);
        assertEquals(0, replicaDevice, "Replica should be on device 0");

        float[] actual = replica.dup().toFloatVector();
        assertEquals(expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], actual[i], 1e-5, "Replicate mismatch at " + i);
        }

        original.close();
        replica.close();
    }

    /**
     * Verify the production affinity primitive independently of DeviceLocalNDArray.
     * This distinguishes a cross-device transfer regression from lazy-replica bookkeeping.
     */
    @Test
    public void testReplicateToDevicePreservesConstantDataAcrossGpus() {
        int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
        org.junit.jupiter.api.Assumptions.assumeTrue(numDevices > 1,
                "Requires at least two automatically discovered GPU devices");

        int sourceDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        int targetDevice = (sourceDevice + 1) % numDevices;
        long[] expected = {12L, 64L, -1L, 768L};

        INDArray source = Nd4j.createFromArray(expected);
        source.data().setConstant(true);
        source.shapeInfoDataBuffer().setConstant(true);
        source.setCloseable(false);
        DeviceMemoryManager.getInstance().ensureHostAccess(source);

        INDArray replica;
        try {
            DeviceMemoryManager.getInstance().switchDevice(targetDevice,
                    "MultiGpuBufferRegressionTest", "direct-constant-replica-target");
            try (org.nd4j.linalg.api.memory.MemoryWorkspace ws =
                         Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                replica = Nd4j.getAffinityManager().replicateToDevice(targetDevice, source);
            }
            assertEquals(targetDevice, Nd4j.getAffinityManager().getDeviceForCurrentThread(),
                    "Replication must restore the caller's target-device context");
            assertEquals(targetDevice, Nd4j.getAffinityManager().getDeviceForArray(replica),
                    "Direct replica must be allocated on the selected target device");
            assertArrayEquals(expected, replica.toLongVector(),
                    "Production affinity replication must preserve constant values before sealing");

            Nd4j.getDeallocatorService().registerPendingConstant(replica);
            try {
                assertTrue(replica.data().isConstant(),
                        "Registering a pending constant must seal the replica");
                assertArrayEquals(expected, replica.toLongVector(),
                        "Sealing a device-authoritative replica must preserve its values");
            } finally {
                Nd4j.getDeallocatorService().releasePendingConstant(replica);
            }
            assertArrayEquals(expected, replica.toLongVector(),
                    "Releasing the pending-constant guard must preserve replica values");
        } finally {
            DeviceMemoryManager.getInstance().switchDevice(sourceDevice,
                    "MultiGpuBufferRegressionTest", "restore-direct-constant-source");
        }

        assertArrayEquals(expected, source.toLongVector(),
                "Direct cross-device replication must not mutate the source");
    }

    /**
     * Verify that a device-authoritative replica can be sealed as constant before
     * any host read synchronizes its primary allocation.
     */
    @Test
    public void testReplicatedConstantCanBeSealedBeforeHostRead() {
        int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
        org.junit.jupiter.api.Assumptions.assumeTrue(numDevices > 1,
                "Requires at least two automatically discovered GPU devices");

        int sourceDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        int targetDevice = (sourceDevice + 1) % numDevices;
        long[] expected = {12L, 64L, -1L, 768L};

        INDArray source = Nd4j.createFromArray(expected);
        source.data().setConstant(true);
        source.shapeInfoDataBuffer().setConstant(true);
        source.setCloseable(false);
        DeviceMemoryManager.getInstance().ensureHostAccess(source);

        INDArray replica;
        try {
            DeviceMemoryManager.getInstance().switchDevice(targetDevice,
                    "MultiGpuBufferRegressionTest", "seal-before-host-read-target");
            replica = Nd4j.getAffinityManager().replicateToDevice(targetDevice, source);
            assertEquals(targetDevice, Nd4j.getAffinityManager().getDeviceForArray(replica),
                    "Replica must be allocated on the selected target device");

            replica = Nd4j.getDeallocatorService().registerPendingConstant(replica);
            try {
                assertTrue(replica.data().isConstant(),
                        "Target replica must be sealed as constant");
            } finally {
                Nd4j.getDeallocatorService().releasePendingConstant(replica);
            }

            assertArrayEquals(expected, replica.toLongVector(),
                    "Sealing a device-authoritative replica before its first host read must preserve values");
        } finally {
            DeviceMemoryManager.getInstance().switchDevice(sourceDevice,
                    "MultiGpuBufferRegressionTest", "restore-seal-before-host-read-source");
        }
    }

    /**
     * Verify that lazy device-local replication preserves small integral control values.
     * SameDiff uses this path for VARIABLE and CONSTANT arrays. A DSP plan may select a
     * different GPU after model import, so the first lookup on that GPU must copy both
     * the shape and the contents rather than materializing a zero-filled replica.
     */
    @Test
    public void testLazyDeviceLocalReplicationPreservesControlsAcrossGpus() {
        int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
        org.junit.jupiter.api.Assumptions.assumeTrue(numDevices > 1,
                "Requires at least two automatically discovered GPU devices");

        int sourceDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        int targetDevice = (sourceDevice + 1) % numDevices;
        long[] expected = {12L, 64L, -1L, 768L};

        INDArray source = Nd4j.createFromArray(expected);
        source.data().setConstant(true);
        source.shapeInfoDataBuffer().setConstant(true);
        source.setCloseable(false);
        assertFalse(source.isAttached(), "Focused source must not require workspace detachment");
        InspectableDeviceLocalNDArray deviceLocal = new InspectableDeviceLocalNDArray(source);
        assertSame(source, deviceLocal.delayedSnapshot(),
                "Unattached source must remain the delayed snapshot rather than an implicit copy");
        assertEquals(sourceDevice, deviceLocal.pendingSourceFor(targetDevice),
                "Target device must resolve its pending update from the source device");
        assertArrayEquals(expected, deviceLocal.delayedSnapshot().toLongVector(),
                "Delayed snapshot values must be intact before lazy replication");
        assertArrayEquals(expected, deviceLocal.get().toLongVector(),
                "Source-device values must be intact before lazy replication");

        try {
            DeviceMemoryManager.getInstance().switchDevice(targetDevice,
                    "MultiGpuBufferRegressionTest", "lazy-device-local-target");
            INDArray target = deviceLocal.get();
            assertEquals(targetDevice, Nd4j.getAffinityManager().getDeviceForArray(target),
                    "Lazy replica must be allocated on the selected target device");
            assertArrayEquals(expected, target.toLongVector(),
                    "Lazy cross-device replica must preserve integral control values");
        } finally {
            DeviceMemoryManager.getInstance().switchDevice(sourceDevice,
                    "MultiGpuBufferRegressionTest", "restore-source-device");
        }

        assertArrayEquals(expected, deviceLocal.get().toLongVector(),
                "Creating a target replica must not mutate the source-device copy");
    }

    /**
     * Verify that constant transition does not overwrite device-authoritative data.
     * CUDA arithmetic makes the special buffer current while the host allocation is
     * stale. Marking the array constant must preserve that device result, and lazy
     * replication must then relay the same values to another discovered GPU.
     */
    @Test
    public void testLazyDeviceLocalReplicationPreservesDeviceAuthoritativeConstant() {
        int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
        org.junit.jupiter.api.Assumptions.assumeTrue(numDevices > 1,
                "Requires at least two automatically discovered GPU devices");

        int sourceDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        int targetDevice = (sourceDevice + 1) % numDevices;
        long[] expected = {7L, 7L, 7L, 7L};

        INDArray source = Nd4j.zeros(DataType.INT64, expected.length);
        source.addi(7L);
        source.data().setConstant(true);
        source.shapeInfoDataBuffer().setConstant(true);
        source.setCloseable(false);
        DeviceLocalNDArray deviceLocal = new DeviceLocalNDArray(source, true);

        try {
            DeviceMemoryManager.getInstance().switchDevice(targetDevice,
                    "MultiGpuBufferRegressionTest", "device-authoritative-constant-target");
            INDArray target = deviceLocal.get();
            assertEquals(targetDevice, Nd4j.getAffinityManager().getDeviceForArray(target),
                    "Lazy replica must be allocated on the selected target device");
            assertArrayEquals(expected, target.toLongVector(),
                    "Constant transition must not replace device-authoritative values with stale host data");
        } finally {
            DeviceMemoryManager.getInstance().switchDevice(sourceDevice,
                    "MultiGpuBufferRegressionTest", "restore-device-authoritative-source");
        }

        assertArrayEquals(expected, deviceLocal.get().toLongVector(),
                "Cross-device replication must preserve the device-authoritative source");
    }

    /**
     * Verify device affinity is preserved through compute operations.
     * mmul allocates a result buffer (allocateSpecial path) on the current device.
     */
    @Test
    public void testDeviceAffinityPreservedThroughOps() {
        int deviceId = Nd4j.getAffinityManager().getDeviceForCurrentThread();

        INDArray a = Nd4j.ones(DataType.FLOAT, 256, 256);
        INDArray b = Nd4j.ones(DataType.FLOAT, 256, 256);
        INDArray result = a.mmul(b);

        int resultDevice = Nd4j.getAffinityManager().getDeviceForArray(result);
        assertEquals(deviceId, resultDevice,
                "Result device should match current thread's device affinity");

        assertEquals(256.0f, result.getFloat(0, 0), 1e-3);
        assertEquals(256.0f, result.getFloat(127, 127), 1e-3);

        a.close();
        b.close();
        result.close();
    }

    // ========================================================================
    // View safety — views share data buffers with parents.
    // Operations on the parent must not break views, and vice versa.
    // ========================================================================

    /**
     * Verify that a view remains valid and correct after the parent is duplicated.
     * Views share the parent's DataBuffer; if failover replaces a buffer incorrectly,
     * the view would read garbage.
     */
    @Test
    public void testViewSafetyAfterParentDup() {
        INDArray parent = Nd4j.linspace(1, 100, 100, DataType.FLOAT).reshape(10, 10);
        INDArray view = parent.getRow(5); // view into row 5

        // Verify view data before dup
        float firstVal = view.getFloat(0);

        // dup parent — allocates new buffer, parent's buffer should remain valid for view
        INDArray parentCopy = parent.dup();

        // View should still read the same data
        assertEquals(firstVal, view.getFloat(0), 1e-5,
                "View data changed after parent dup");

        // parentCopy should have independent data
        assertEquals(firstVal, parentCopy.getFloat(5, 0), 1e-5,
                "Parent copy should have same data");

        parent.close();
        parentCopy.close();
    }

    /**
     * Verify concat (which may trigger reallocation internally) produces correct results.
     * concat creates a new output buffer — exercises allocateSpecial.
     */
    @Test
    public void testConcatBufferAllocation() {
        INDArray a = Nd4j.ones(DataType.FLOAT, 512, 64);
        INDArray b = Nd4j.zeros(DataType.FLOAT, 512, 64);
        INDArray result = Nd4j.concat(0, a, b);

        assertEquals(1024, result.shape()[0]);
        assertEquals(64, result.shape()[1]);

        // First half should be 1s, second half 0s
        assertEquals(1.0f, result.getFloat(0, 0), 1e-6, "First half should be ones");
        assertEquals(1.0f, result.getFloat(511, 63), 1e-6, "Last of first half should be ones");
        assertEquals(0.0f, result.getFloat(512, 0), 1e-6, "Second half should be zeros");
        assertEquals(0.0f, result.getFloat(1023, 63), 1e-6, "Last element should be zero");

        a.close();
        b.close();
        result.close();
    }

    // ========================================================================
    // Environment flag tests — verify fusion scoring configuration round-trips
    // ========================================================================

    /**
     * Verify default values of fusion scoring flags.
     * These defaults affect DSP compilation behavior.
     */
    @Test
    public void testFusionScoringFlagDefaults() {
        Environment env = Nd4j.getEnvironment();

        // tritonFusionScoring defaults to true (fusion scoring enabled)
        assertTrue(env.tritonFusionScoring(), "tritonFusionScoring should default to true");

        // tritonFusionMinScore defaults to 5.0f (C++ Environment default, overrides Java interface default of 1.0f)
        assertEquals(5.0f, env.tritonFusionMinScore(), 1e-6,
                "tritonFusionMinScore should default to 5.0");

        // tritonFuseAttentionNeighborhoods defaults to true
        assertTrue(env.tritonFuseAttentionNeighborhoods(),
                "tritonFuseAttentionNeighborhoods should default to true");
    }

    @Test
    public void testTritonFusionScoringFlagRoundTrip() {
        Environment env = Nd4j.getEnvironment();
        boolean original = env.tritonFusionScoring();

        try {
            env.setTritonFusionScoring(false);
            assertFalse(env.tritonFusionScoring(), "Should be false after setting false");

            env.setTritonFusionScoring(true);
            assertTrue(env.tritonFusionScoring(), "Should be true after setting true");
        } finally {
            env.setTritonFusionScoring(original);
        }
    }

    @Test
    public void testTritonFusionMinScoreFlagRoundTrip() {
        Environment env = Nd4j.getEnvironment();
        float original = env.tritonFusionMinScore();

        try {
            env.setTritonFusionMinScore(42.5f);
            assertEquals(42.5f, env.tritonFusionMinScore(), 1e-6,
                    "Should read back the value that was set");

            env.setTritonFusionMinScore(0.0f);
            assertEquals(0.0f, env.tritonFusionMinScore(), 1e-6,
                    "Should support zero");

            env.setTritonFusionMinScore(100.0f);
            assertEquals(100.0f, env.tritonFusionMinScore(), 1e-6,
                    "Should support large values");

            // Negative values — the scoring system should accept them
            env.setTritonFusionMinScore(-5.0f);
            assertEquals(-5.0f, env.tritonFusionMinScore(), 1e-6,
                    "Should support negative values");
        } finally {
            env.setTritonFusionMinScore(original);
        }
    }

    @Test
    public void testTritonFuseAttentionNeighborhoodsFlagRoundTrip() {
        Environment env = Nd4j.getEnvironment();
        boolean original = env.tritonFuseAttentionNeighborhoods();

        try {
            env.setTritonFuseAttentionNeighborhoods(false);
            assertFalse(env.tritonFuseAttentionNeighborhoods(),
                    "Should be false after setting false");

            env.setTritonFuseAttentionNeighborhoods(true);
            assertTrue(env.tritonFuseAttentionNeighborhoods(),
                    "Should be true after setting true");
        } finally {
            env.setTritonFuseAttentionNeighborhoods(original);
        }
    }

    /**
     * Verify that toggling tritonFusionScoring off and back on doesn't corrupt
     * the min score value (independent settings shouldn't interfere).
     */
    @Test
    public void testFusionFlagsIndependence() {
        Environment env = Nd4j.getEnvironment();
        boolean origScoring = env.tritonFusionScoring();
        float origMinScore = env.tritonFusionMinScore();
        boolean origAttn = env.tritonFuseAttentionNeighborhoods();

        try {
            // Set all to non-default values
            env.setTritonFusionScoring(false);
            env.setTritonFusionMinScore(99.0f);
            env.setTritonFuseAttentionNeighborhoods(false);

            // Verify all are independent
            assertFalse(env.tritonFusionScoring());
            assertEquals(99.0f, env.tritonFusionMinScore(), 1e-6);
            assertFalse(env.tritonFuseAttentionNeighborhoods());

            // Change one, verify others unaffected
            env.setTritonFusionScoring(true);
            assertTrue(env.tritonFusionScoring());
            assertEquals(99.0f, env.tritonFusionMinScore(), 1e-6,
                    "minScore should not change when scoring toggled");
            assertFalse(env.tritonFuseAttentionNeighborhoods(),
                    "attn flag should not change when scoring toggled");
        } finally {
            env.setTritonFusionScoring(origScoring);
            env.setTritonFusionMinScore(origMinScore);
            env.setTritonFuseAttentionNeighborhoods(origAttn);
        }
    }
}
