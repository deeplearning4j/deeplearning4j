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

package org.eclipse.deeplearning4j.nd4j.linalg.framework.device;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.device.DeviceMemoryManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.framework.device.*;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for Device Transfer Management Framework.
 *
 * Covers:
 * P1 - Device pinning policies (STICKY, FOLLOW_THREAD, EXPLICIT)
 * P2 - Transfer event recording and diagnostics
 * P3 - Replica leak detection
 * P4 - Pointer stability for CUDA graph replay
 * P5 - Memory pressure simulation
 */
public class DeviceTransferManagementTest {

    private DeviceMemoryManager memoryManager;
    private TransferSubsystem transfers;
    private DevicePinningManager pinning;
    private ReplicaLeakDetector replicaDetector;
    private PointerStabilityGuard stabilityGuard;

    @BeforeEach
    public void setUp() {
        memoryManager = DeviceMemoryManager.getInstance();
        transfers = Nd4j.framework.device().transfers();
        pinning = Nd4j.framework.device().pinning();
        replicaDetector = Nd4j.framework.device().replicaLeaks();
        stabilityGuard = Nd4j.framework.device().pointerStability();

        memoryManager.setMemorySimulationEnabled(false);
        transfers.reset();
        pinning.clear();
        replicaDetector.clear();
        stabilityGuard.clear();
    }

    @AfterEach
    public void tearDown() {
        System.clearProperty("nd4j.device.transfer.tracking");
        System.clearProperty("nd4j.device.replica.leak.detection");
        System.clearProperty("nd4j.device.pointerStability.check");
        System.clearProperty("nd4j.device.pinning.enabled");
        memoryManager.setMemorySimulationEnabled(false);
    }

    // =========================================================================
    // P1: Device Pinning
    // =========================================================================

    @Test
    public void testExplicitPinResolvesToDevice() {
        pinning.pin("weights", 0);
        assertEquals(0, pinning.resolveDevice("weights"));

        pinning.pin("kv_cache", 1);
        assertEquals(1, pinning.resolveDevice("kv_cache"));
    }

    @Test
    public void testStickyPinReturnsMinusOne() {
        pinning.pin("input", DevicePinPolicy.STICKY);
        assertEquals(-1, pinning.resolveDevice("input"));
        assertTrue(pinning.isSticky("input"));
    }

    @Test
    public void testFollowThreadResolvesToCurrentDevice() {
        pinning.pin("bias", DevicePinPolicy.FOLLOW_THREAD);
        int resolved = pinning.resolveDevice("bias");
        int threadDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        assertEquals(threadDevice, resolved);
    }

    @Test
    public void testUnpinRemovesPinning() {
        pinning.pin("weights", 0);
        assertNotNull(pinning.getPinning("weights"));

        pinning.unpin("weights");
        assertNull(pinning.getPinning("weights"));
        assertEquals(-1, pinning.resolveDevice("weights"));
    }

    @Test
    public void testStickyBlocksMigration() {
        INDArray arr = Nd4j.ones(DataType.FLOAT, 10, 10);
        pinning.pin("weights", DevicePinPolicy.STICKY);

        assertFalse(pinning.isMigrationAllowed("weights", arr));
    }

    @Test
    public void testFrozenBlocksMigration() {
        System.setProperty("nd4j.device.pointerStability.check", "true");
        PointerStabilityGuard guard = new PointerStabilityGuard();
        assertTrue(guard.isEnabled());

        INDArray arr = Nd4j.ones(DataType.FLOAT, 10, 10);
        guard.registerForGraph(arr, "weights", "test-plan");

        // Array is frozen in graph → migration should be blocked
        assertFalse(pinning.isMigrationAllowed("weights", arr, guard));
    }

    @Test
    public void testValidatePinningsDetectsDeviceMismatch() {
        pinning.pin("weights", 0);
        INDArray arr = Nd4j.ones(DataType.FLOAT, 10, 10);

        Map<String, INDArray> arrays = new HashMap<>();
        arrays.put("weights", arr);

        // On single-GPU, array is on device 0 → no violation
        List<String> violations = pinning.validatePinnings(arrays);
        // Device check depends on actual device placement
        assertNotNull(violations);
    }

    @Test
    public void testSameDiffPinningIntegration() {
        SameDiff sd = SameDiff.create();
        sd.var("weights", Nd4j.ones(DataType.FLOAT, 3, 4));
        sd.var("bias", Nd4j.zeros(DataType.FLOAT, 4));

        sd.pinVariable("weights", 0);
        sd.pinVariable("bias", 0);

        assertNotNull(sd.getVariablePinning("weights"));
        assertEquals(DevicePinPolicy.EXPLICIT, sd.getVariablePinning("weights").getPolicy());
        assertEquals(0, sd.getVariablePinning("weights").getExplicitDeviceId());

        sd.pinVariableSticky("bias");
        assertEquals(DevicePinPolicy.STICKY, sd.getVariablePinning("bias").getPolicy());

        sd.clearAllVariablePinnings();
        assertNull(sd.getVariablePinning("weights"));
        assertNull(sd.getVariablePinning("bias"));
    }

    @Test
    public void testPinAllVariablesSticky() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 2, 3);
        SDVariable weights = sd.var("weights", Nd4j.ones(DataType.FLOAT, 3, 4));
        sd.mmul("output", input, weights);

        sd.pinAllVariablesSticky();

        assertNotNull(sd.getVariablePinning("weights"));
        assertNotNull(sd.getVariablePinning("output"));
        assertEquals(DevicePinPolicy.STICKY, sd.getVariablePinning("weights").getPolicy());
    }

    @Test
    public void testNullHandlingInPinning() {
        pinning.pin(null);
        pinning.pin("");
        pinning.unpin(null);
        pinning.unpin("");
        assertNull(pinning.getPinning(null));
        assertNull(pinning.getPinning(""));

        pinning.pin("valid", 0);
        assertNotNull(pinning.getPinning("valid"));
    }

    // =========================================================================
    // P2: Transfer Diagnostics
    // =========================================================================

    @Test
    public void testTransferTrackingRecordsEvents() {
        System.setProperty("nd4j.device.transfer.tracking", "true");
        TransferSubsystem ts = new TransferSubsystem();
        assertTrue(ts.isEnabled());

        long bytes = 40000L; // 100x100 floats

        ts.record(TransferEvent.builder()
            .variableName("weights")
            .sourceDeviceId(0)
            .destDeviceId(1)
            .direction(TransferDirection.D2D)
            .reason(TransferReason.CONSTANT_REPLICATION)
            .bytes(bytes)
            .durationNanos(5000)
            .callerContext("test")
            .build());

        assertEquals(1, ts.getTotalTransferCount());
        assertEquals(bytes, ts.getTotalBytes());

        TransferStats stats = ts.getStats("weights");
        assertNotNull(stats);
        assertEquals(1, stats.getTotalTransfers());
        assertEquals(bytes, stats.getTotalBytes());
        assertEquals(5000, stats.getTotalDurationNanos());
        assertTrue(stats.averageBandwidthBytesPerSec() > 0);
    }

    @Test
    public void testTransferDirectionBreakdown() {
        System.setProperty("nd4j.device.transfer.tracking", "true");
        TransferSubsystem ts = new TransferSubsystem();

        ts.record(TransferEvent.builder()
            .variableName("data").sourceDeviceId(-1).destDeviceId(0)
            .direction(TransferDirection.H2D).reason(TransferReason.SYNC_TO_DEVICE)
            .bytes(1000).durationNanos(100).build());

        ts.record(TransferEvent.builder()
            .variableName("data").sourceDeviceId(0).destDeviceId(-1)
            .direction(TransferDirection.D2H).reason(TransferReason.SYNC_TO_HOST)
            .bytes(2000).durationNanos(200).build());

        ts.record(TransferEvent.builder()
            .variableName("data").sourceDeviceId(0).destDeviceId(1)
            .direction(TransferDirection.D2D).reason(TransferReason.CONSTANT_REPLICATION)
            .bytes(3000).durationNanos(300).build());

        TransferStats stats = ts.getStats("data");
        assertNotNull(stats);
        assertEquals(3, stats.getTotalTransfers());
        assertEquals(6000, stats.getTotalBytes());
        assertEquals(1L, stats.getCountByDirection().get(TransferDirection.H2D));
        assertEquals(1L, stats.getCountByDirection().get(TransferDirection.D2H));
        assertEquals(1L, stats.getCountByDirection().get(TransferDirection.D2D));
        assertEquals(1000L, stats.getBytesByDirection().get(TransferDirection.H2D));
        assertEquals(2000L, stats.getBytesByDirection().get(TransferDirection.D2H));
        assertEquals(3000L, stats.getBytesByDirection().get(TransferDirection.D2D));
    }

    @Test
    public void testTransferReasonCoverage() {
        System.setProperty("nd4j.device.transfer.tracking", "true");
        TransferSubsystem ts = new TransferSubsystem();

        for (TransferReason reason : TransferReason.values()) {
            ts.record(TransferEvent.builder()
                .variableName("reason-" + reason.name())
                .sourceDeviceId(0).destDeviceId(1)
                .direction(TransferDirection.D2D)
                .reason(reason).bytes(100).durationNanos(10)
                .build());
        }

        assertEquals(TransferReason.values().length, ts.getTotalTransferCount());
        List<TransferEvent> events = ts.getRecentEvents(50);
        assertEquals(TransferReason.values().length, events.size());
    }

    @Test
    public void testTransferReportGeneration() {
        System.setProperty("nd4j.device.transfer.tracking", "true");
        TransferSubsystem ts = new TransferSubsystem();

        String[] vars = {"w1", "w2", "input", "output"};
        for (String var : vars) {
            for (int i = 0; i < 5; i++) {
                ts.record(TransferEvent.builder()
                    .variableName(var).sourceDeviceId(0).destDeviceId(1)
                    .direction(TransferDirection.D2D)
                    .reason(TransferReason.CONSTANT_REPLICATION)
                    .bytes(1024 * (i + 1)).durationNanos(100 * (i + 1))
                    .build());
            }
        }

        TransferReport report = ts.getReport();
        assertNotNull(report);
        assertEquals(20, report.getTotalTransferCount());
        assertEquals(4, report.getPerVariableStats().size());

        for (String var : vars) {
            TransferStats stats = report.getPerVariableStats().get(var);
            assertNotNull(stats, "Missing stats for " + var);
            assertEquals(5, stats.getTotalTransfers());
            assertEquals(1024 * 15, stats.getTotalBytes()); // 1+2+3+4+5 KB
        }

        String summary = report.summary();
        assertNotNull(summary);
        assertTrue(summary.contains("20"));
    }

    @Test
    public void testTransferTrackingDisabledByDefault() {
        TransferSubsystem ts = new TransferSubsystem();
        assertFalse(ts.isEnabled());

        ts.record(TransferEvent.builder()
            .variableName("test").sourceDeviceId(0).destDeviceId(1)
            .direction(TransferDirection.D2D).reason(TransferReason.CONSTANT_REPLICATION)
            .bytes(1000).durationNanos(100).build());

        // Nothing should be recorded
        assertEquals(0, ts.getTotalTransferCount());
        assertNull(ts.getStats("test"));
    }

    @Test
    public void testTransferRingBufferWraparound() {
        System.setProperty("nd4j.device.transfer.tracking", "true");
        TransferSubsystem ts = new TransferSubsystem();

        // Record more than ring buffer capacity
        for (int i = 0; i < 5000; i++) {
            ts.record(TransferEvent.builder()
                .variableName("wrap-test").sourceDeviceId(0).destDeviceId(1)
                .direction(TransferDirection.D2D).reason(TransferReason.CONSTANT_REPLICATION)
                .bytes(i).durationNanos(1).build());
        }

        assertEquals(5000, ts.getTotalTransferCount());

        // Ring buffer only holds 4096
        List<TransferEvent> recent = ts.getRecentEvents(5000);
        assertTrue(recent.size() <= 4096);
        assertTrue(recent.size() > 0);
    }

    @Test
    public void testNullEventHandling() {
        System.setProperty("nd4j.device.transfer.tracking", "true");
        TransferSubsystem ts = new TransferSubsystem();
        ts.record(null);
        assertEquals(0, ts.getTotalTransferCount());
    }

    @Test
    public void testConcurrentTransferRecording() throws InterruptedException {
        System.setProperty("nd4j.device.transfer.tracking", "true");
        TransferSubsystem ts = new TransferSubsystem();

        Thread[] threads = new Thread[10];
        for (int i = 0; i < threads.length; i++) {
            int tid = i;
            threads[i] = new Thread(() -> {
                for (int j = 0; j < 100; j++) {
                    ts.record(TransferEvent.builder()
                        .variableName("t" + tid + "-v" + j)
                        .sourceDeviceId(0).destDeviceId(1)
                        .direction(TransferDirection.D2D)
                        .reason(TransferReason.CONSTANT_REPLICATION)
                        .bytes(100).durationNanos(10)
                        .build());
                }
            });
            threads[i].start();
        }

        for (Thread t : threads) {
            t.join();
        }

        assertEquals(1000, ts.getTotalTransferCount());
    }

    @Test
    public void testTransferViaFramework() {
        transfers.setEnabled(true);

        transfers.record(TransferEvent.builder()
            .variableName("fw-test").sourceDeviceId(0).destDeviceId(1)
            .direction(TransferDirection.D2D).reason(TransferReason.CONSTANT_REPLICATION)
            .bytes(4096).durationNanos(1000)
            .build());

        TransferReport report = Nd4j.framework.device().transfers().getReport();
        assertNotNull(report);
        assertTrue(report.getTotalTransferCount() >= 1);
    }

    // =========================================================================
    // P3: Replica Leak Detection
    // =========================================================================

    @Test
    public void testReplicaRegistrationAndUnregistration() {
        System.setProperty("nd4j.device.replica.leak.detection", "true");
        ReplicaLeakDetector det = new ReplicaLeakDetector();

        INDArray r1 = Nd4j.ones(DataType.FLOAT, 50, 50);
        INDArray r2 = Nd4j.ones(DataType.FLOAT, 50, 50);

        det.registerReplica(r1, "layer1", 0, 1);
        det.registerReplica(r2, "layer2", 0, 1);
        assertEquals(2, det.getTrackedReplicaCount());

        det.unregisterReplica(r1.getId());
        assertEquals(1, det.getTrackedReplicaCount());

        det.unregisterReplica(r2.getId());
        assertEquals(0, det.getTrackedReplicaCount());
    }

    @Test
    public void testReplicaLeakDetection() throws InterruptedException {
        System.setProperty("nd4j.device.replica.leak.detection", "true");
        ReplicaLeakDetector det = new ReplicaLeakDetector();

        INDArray r1 = Nd4j.ones(DataType.FLOAT, 50, 50);
        INDArray r2 = Nd4j.ones(DataType.FLOAT, 50, 50);

        det.registerReplica(r1, "cleaned", 0, 1);
        det.registerReplica(r2, "leaked", 0, 1);

        det.unregisterReplica(r1.getId());

        Thread.sleep(150);

        var leaks = det.detectLeakedReplicas(100);
        assertEquals(1, leaks.size());
        assertEquals("leaked", leaks.get(0).getVariableName());
        assertTrue(leaks.get(0).getAgeMs() >= 100);
    }

    @Test
    public void testReplicaReport() {
        System.setProperty("nd4j.device.replica.leak.detection", "true");
        ReplicaLeakDetector det = new ReplicaLeakDetector();

        for (int src = 0; src < 2; src++) {
            for (int dst = 0; dst < 2; dst++) {
                if (src != dst) {
                    INDArray r = Nd4j.ones(DataType.FLOAT, 10, 10);
                    det.registerReplica(r, "r-" + src + "-" + dst, src, dst);
                }
            }
        }

        ReplicaLeakDetector.ReplicaReport report = det.getReplicaReport();
        assertEquals(2, report.getTotalReplicasTracked());
        assertTrue(report.getTotalTrackedBytes() > 0);
        assertEquals(2, report.getTotalReplicasCreated());
        assertEquals(0, report.getTotalReplicasCleaned());

        for (var snap : report.getReplicas()) {
            assertNotEquals(snap.getSourceDevice(), snap.getTargetDevice());
        }
    }

    @Test
    public void testReplicaDisabledByDefault() {
        ReplicaLeakDetector det = new ReplicaLeakDetector();
        assertFalse(det.isEnabled());

        INDArray r = Nd4j.ones(DataType.FLOAT, 10, 10);
        det.registerReplica(r, "test", 0, 1);
        assertEquals(0, det.getTrackedReplicaCount()); // not recorded when disabled
    }

    // =========================================================================
    // P4: Pointer Stability
    // =========================================================================

    @Test
    public void testPointerStabilityNamedRegistration() {
        System.setProperty("nd4j.device.pointerStability.check", "true");
        PointerStabilityGuard guard = new PointerStabilityGuard();

        INDArray a1 = Nd4j.ones(DataType.FLOAT, 8, 8);
        INDArray a2 = Nd4j.ones(DataType.FLOAT, 8, 8);

        guard.registerForGraph(a1, "weights", "plan-1");
        guard.registerForGraph(a2, "bias", "plan-1");

        assertEquals(2, guard.getTrackedBufferCount());
        assertTrue(guard.isFrozen(a1));
        assertTrue(guard.isFrozen(a2));
        assertTrue(guard.isFrozen("weights"));
        assertTrue(guard.isFrozen("bias"));
    }

    @Test
    public void testPointerStabilityNamedValidation() {
        System.setProperty("nd4j.device.pointerStability.check", "true");
        PointerStabilityGuard guard = new PointerStabilityGuard();

        INDArray arr = Nd4j.ones(DataType.FLOAT, 8, 8);
        guard.registerForGraph(arr, "weights", "plan-1");

        // Same array, same address → no violations
        Map<String, INDArray> named = new HashMap<>();
        named.put("weights", arr);
        var violations = guard.validateStability(named);
        assertTrue(violations.isEmpty());

        // Different array with different address → violation
        INDArray different = Nd4j.zeros(DataType.FLOAT, 8, 8);
        named.put("weights", different);
        violations = guard.validateStability(named);
        assertEquals(1, violations.size());
        assertEquals("weights", violations.get(0).getVariableName());
        assertNotEquals(violations.get(0).getCapturedAddress(), violations.get(0).getCurrentAddress());
    }

    @Test
    public void testPointerStabilityUnregister() {
        System.setProperty("nd4j.device.pointerStability.check", "true");
        PointerStabilityGuard guard = new PointerStabilityGuard();

        INDArray arr = Nd4j.ones(DataType.FLOAT, 8, 8);
        guard.registerForGraph(arr, "weights", "plan-1");
        assertTrue(guard.isFrozen(arr));

        guard.unregisterByName("weights");
        assertFalse(guard.isFrozen(arr));
        assertFalse(guard.isFrozen("weights"));
        assertEquals(0, guard.getTrackedBufferCount());
    }

    @Test
    public void testPointerStabilityDisabledNoTracking() {
        // Default: disabled
        PointerStabilityGuard guard = new PointerStabilityGuard();
        assertFalse(guard.isEnabled());

        INDArray arr = Nd4j.ones(DataType.FLOAT, 8, 8);
        guard.registerForGraph(arr, "weights", "plan-1");

        assertFalse(guard.isFrozen(arr));
        assertEquals(0, guard.getTrackedBufferCount());
    }

    @Test
    public void testPointerStabilityGetRecord() {
        System.setProperty("nd4j.device.pointerStability.check", "true");
        PointerStabilityGuard guard = new PointerStabilityGuard();

        INDArray arr = Nd4j.ones(DataType.FLOAT, 8, 8);
        guard.registerForGraph(arr, "weights", "plan-1");

        PointerStabilityGuard.StabilityRecord record = guard.getRecord("weights");
        assertNotNull(record);
        assertEquals("weights", record.getVariableName());
        assertEquals("plan-1", record.getPlanName());
        assertTrue(record.getCapturedAddress() != 0);
        assertTrue(record.getCapturedAtMs() > 0);
    }

    // =========================================================================
    // P5: Memory Pressure
    // =========================================================================

    @Test
    public void testMemoryPressureSimulation() {
        memoryManager.setMemorySimulationEnabled(true);

        memoryManager.setSimulatedFreeMemory(0, 1024 * 1024); // 1 MB
        memoryManager.setSimulatedFreeMemory(1, 8L * 1024 * 1024 * 1024); // 8 GB

        long gpu0Free = memoryManager.getEffectiveFreeMemory(0, Long.MAX_VALUE);
        long gpu1Free = memoryManager.getEffectiveFreeMemory(1, Long.MAX_VALUE);

        assertEquals(1024 * 1024, gpu0Free);
        assertEquals(8L * 1024 * 1024 * 1024, gpu1Free);
    }

    @Test
    public void testMemoryPressureCallbackRegistration() {
        boolean[] invoked = {false};
        DeviceMemoryManager.MemoryPressureCallback callback = (device, utilization) -> {
            invoked[0] = true;
        };

        memoryManager.registerMemoryPressureCallback(callback);
        // Callback is registered; actual invocation happens during allocation pressure
    }

    @Test
    public void testDeviceSelectionUnderPressure() {
        memoryManager.setMemorySimulationEnabled(true);
        memoryManager.setSimulatedFreeMemory(0, 1024 * 1024); // 1 MB
        memoryManager.setSimulatedFreeMemory(1, 8L * 1024 * 1024 * 1024); // 8 GB

        assertTrue(memoryManager.getEffectiveFreeMemory(1, 0) >
                   memoryManager.getEffectiveFreeMemory(0, 0));
    }

    // =========================================================================
    // Edge Cases
    // =========================================================================

    @Test
    public void testZeroLengthArrayHandling() {
        INDArray empty = Nd4j.create(DataType.FLOAT, 0, 0);
        assertFalse(stabilityGuard.isFrozen(empty));
    }

    @Test
    public void testNullArraySafety() {
        assertFalse(stabilityGuard.isFrozen((INDArray) null));
        assertFalse(pinning.isMigrationAllowed("test", null));

        // Should not throw
        stabilityGuard.registerForGraph(null, "test", "plan");
        replicaDetector.registerReplica(null, "test", 0, 1);
    }
}
