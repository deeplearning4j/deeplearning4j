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
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.execution.DspReplayTransferAnalytics;
import org.nd4j.autodiff.samediff.execution.ReplayProfileManager;
import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.device.DeviceMemoryManager;
import org.nd4j.linalg.api.device.DeviceType;
import org.nd4j.linalg.api.device.StubDeviceContextProvider;
import org.nd4j.linalg.api.device.StubDeviceDescriptor;
import org.nd4j.linalg.framework.device.TransferDirection;
import org.nd4j.linalg.framework.device.TransferEvent;
import org.nd4j.linalg.framework.device.TransferReason;
import org.nd4j.linalg.framework.device.TransferSubsystem;

import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Simulates multi-device memory switching leaks during DSP replay using the
 * stub device simulation API. Models a maxed-out GPU scenario where:
 *
 * <ul>
 *   <li>GPU 0 is loaded with a large model (~22 GB of 24 GB used)</li>
 *   <li>DSP replay steps progressively allocate capture buffers</li>
 *   <li>Memory pressure triggers rerouting to GPU 1 or CPU fallback</li>
 *   <li>Device switches during replay create potential leak points</li>
 *   <li>Analytics track all transfers and routing decisions for verification</li>
 * </ul>
 *
 * These tests verify that the simulation API faithfully models what a maxed-out
 * GPU looks like and that DSP infrastructure hooks correctly respond to pressure.
 */
public class DspMultiDeviceMemoryLeakSimulationTest {

    private DeviceMemoryManager memMgr;

    @BeforeEach
    void setUp() {
        memMgr = DeviceMemoryManager.getInstance();
    }

    @AfterEach
    void tearDown() {
        memMgr.clearStubTopology();
    }

    // =========================================================================
    // Helper: create a realistic maxed-out GPU topology
    // =========================================================================

    private List<StubDeviceDescriptor> maxedOutGpuTopology() {
        // GPU 0: RTX 4090 with 24GB, only 500MB free (model loaded)
        StubDeviceDescriptor gpu0 = StubDeviceDescriptor.builder(DeviceType.CUDA_GPU, 0)
                .deviceName("Stub RTX 4090 (maxed)")
                .totalMemory(24L * 1024 * 1024 * 1024)
                .availableMemory(500L * 1024 * 1024) // 500 MB free
                .isDefault(true)
                .addPeerDevice(1)
                .peerBandwidth(1, 300L * 1024 * 1024 * 1024) // NVLink
                .build();

        // GPU 1: RTX 3090 with 24GB, 20GB free (available for overflow)
        StubDeviceDescriptor gpu1 = StubDeviceDescriptor.builder(DeviceType.CUDA_GPU, 1)
                .deviceName("Stub RTX 3090 (available)")
                .totalMemory(24L * 1024 * 1024 * 1024)
                .availableMemory(20L * 1024 * 1024 * 1024) // 20 GB free
                .addPeerDevice(0)
                .peerBandwidth(0, 300L * 1024 * 1024 * 1024)
                .build();

        return Arrays.asList(gpu0, gpu1);
    }

    private List<StubDeviceDescriptor> singleMaxedGpuWithCpuFallback() {
        StubDeviceDescriptor gpu0 = StubDeviceDescriptor.builder(DeviceType.CUDA_GPU, 0)
                .deviceName("Stub RTX 4090 (near-OOM)")
                .totalMemory(24L * 1024 * 1024 * 1024)
                .availableMemory(100L * 1024 * 1024) // Only 100MB free
                .isDefault(true)
                .build();

        StubDeviceDescriptor cpu = StubDeviceDescriptor.builder(DeviceType.CPU, 0)
                .deviceName("Stub CPU")
                .totalMemory(64L * 1024 * 1024 * 1024) // 64 GB
                .availableMemory(60L * 1024 * 1024 * 1024) // 60 GB free
                .build();

        return Arrays.asList(gpu0, cpu);
    }

    // =========================================================================
    // Test 1: Maxed-out GPU triggers progressive rerouting during DSP replay
    // =========================================================================

    @Test
    @DisplayName("DSP replay on maxed-out GPU: progressive allocation forces rerouting")
    void testMaxedGpuProgressiveReplayRerouting() {
        List<StubDeviceDescriptor> stubs = maxedOutGpuTopology();
        memMgr.configureStubTopology(stubs);
        StubDeviceDescriptor gpu0 = stubs.get(0);
        StubDeviceDescriptor gpu1 = stubs.get(1);

        // Verify initial state: GPU 0 is near-full
        assertTrue(memMgr.isMemorySimulationEnabled());
        long gpu0Free = memMgr.getSimulatedFreeMemory(0);
        assertEquals(500L * 1024 * 1024, gpu0Free, "GPU 0 should start with 500MB free");

        TransferSubsystem transferSub = new TransferSubsystem();
        transferSub.setEnabled(true);
        DspReplayTransferAnalytics analytics = new DspReplayTransferAnalytics(transferSub, memMgr);

        // Record GPU 0 as near-full via allocation tracking
        memMgr.setMemoryCap(gpu0, gpu0.getTotalMemory());
        memMgr.recordAllocation(gpu0, (long)(23.5 * 1024 * 1024 * 1024)); // 23.5 GB allocated

        // Simulate 10 DSP replay steps, each needing 256MB capture buffer
        int numSteps = 10;
        long captureBufferSize = 256L * 1024 * 1024;
        int reroutedCount = 0;

        for (int step = 0; step < numSteps; step++) {
            analytics.beginStep(0, 99999L);

            // Check memory pressure before allocating capture buffer
            DeviceDescriptor target = analytics.checkMemoryPressureForTransfer(
                    gpu0, captureBufferSize, TransferReason.CAPTURE_BUFFER_COPY);

            if (!target.getDeviceId().equals(gpu0.getDeviceId())) {
                reroutedCount++;
                // Record the rerouted transfer
                analytics.recordTransfer(TransferEvent.builder()
                        .variableName("capture_buffer_" + step)
                        .direction(TransferDirection.D2D)
                        .reason(TransferReason.CAPTURE_BUFFER_COPY)
                        .bytes(captureBufferSize)
                        .durationNanos(5000)
                        .sourceDeviceId(0)
                        .destDeviceId(target.getDeviceIndex())
                        .build());
            } else {
                // Stays on GPU 0 — consume memory
                gpu0.consumeMemory(captureBufferSize);
                analytics.recordTransfer(TransferEvent.builder()
                        .variableName("capture_buffer_" + step)
                        .direction(TransferDirection.D2D)
                        .reason(TransferReason.CAPTURE_BUFFER_COPY)
                        .bytes(captureBufferSize)
                        .durationNanos(2000)
                        .sourceDeviceId(0)
                        .destDeviceId(0)
                        .build());
            }

            analytics.endStep();
        }

        // Verify: most steps should have been rerouted away from GPU 0
        DspReplayTransferAnalytics.ReplayTransferReport report = analytics.getReport();
        assertTrue(reroutedCount > 0,
                "Expected at least some steps rerouted from maxed GPU 0, got " + reroutedCount);
        assertEquals(numSteps, report.getTotalCount(),
                "All steps should have recorded transfers");
        assertTrue(report.getRoutingDecisions().size() > 0,
                "Expected routing decisions recorded");

        // Verify routing decisions point to GPU 1
        for (DspReplayTransferAnalytics.RoutingDecision decision : report.getRoutingDecisions()) {
            assertEquals(gpu0.getDeviceId(), decision.getIntendedDeviceId(),
                    "Intended device should be GPU 0");
            assertEquals(gpu1.getDeviceId(), decision.getActualDeviceId(),
                    "Rerouted device should be GPU 1");
            assertEquals(TransferReason.CAPTURE_BUFFER_COPY, decision.getReason());
        }

        // Verify segment summary is consistent
        DspReplayTransferAnalytics.SegmentTransferSummary seg0 = analytics.getSegmentSummary(0);
        assertNotNull(seg0);
        assertEquals(numSteps, seg0.getStepCount());
        assertEquals(numSteps, seg0.getTotalTransferCount());
    }

    // =========================================================================
    // Test 2: Device switch history during multi-segment DSP replay
    // =========================================================================

    @Test
    @DisplayName("Device switch history tracks all migrations during DSP replay segments")
    void testDeviceSwitchHistoryDuringDspReplay() {
        List<StubDeviceDescriptor> stubs = maxedOutGpuTopology();
        memMgr.configureStubTopology(stubs);

        StubDeviceContextProvider provider = memMgr.getStubContextProvider();
        assertNotNull(provider);
        provider.clearSwitchHistory();

        // Simulate DSP replay pattern: switch between devices per segment
        // Segment 0: execute on GPU 0 (weights are there)
        provider.switchDevice(0, "DspReplay", "segment_0_execute");

        // Segment 1: memory overflow, switch to GPU 1
        provider.switchDevice(1, "DspReplay", "segment_1_overflow");

        // Segment 2: back to GPU 0 for weight-bound ops
        provider.switchDevice(0, "DspReplay", "segment_2_weights");

        // Segment 3: capture buffer allocation, overflow again
        provider.switchDevice(1, "DspReplay", "segment_3_capture_buffer");

        // Segment 4: finalize on GPU 0
        provider.switchDevice(0, "DspReplay", "segment_4_finalize");

        List<StubDeviceContextProvider.DeviceSwitchRecord> history = provider.getSwitchHistory();
        assertEquals(5, history.size(), "Should record 5 device switches");

        // Verify alternating pattern indicates potential leak from cross-device refs
        int switchesToGpu1 = 0;
        int switchesToGpu0 = 0;
        for (StubDeviceContextProvider.DeviceSwitchRecord record : history) {
            if (record.getNewDeviceId() == 0) switchesToGpu0++;
            if (record.getNewDeviceId() == 1) switchesToGpu1++;
        }
        assertEquals(3, switchesToGpu0, "Should switch to GPU 0 three times");
        assertEquals(2, switchesToGpu1, "Should switch to GPU 1 twice");

        // Verify timestamps are monotonically increasing
        for (int i = 1; i < history.size(); i++) {
            assertTrue(history.get(i).getTimestampNanos() >= history.get(i - 1).getTimestampNanos(),
                    "Timestamps must be monotonically increasing");
        }
    }

    // =========================================================================
    // Test 3: Progressive OOM simulation across DSP replay steps
    // =========================================================================

    @Test
    @DisplayName("Progressive memory exhaustion during replay: GPU 0 fills up step-by-step")
    void testProgressiveOomDuringReplay() {
        // Start GPU 0 with 2 GB free, each step consumes 256 MB
        StubDeviceDescriptor gpu0 = StubDeviceDescriptor.builder(DeviceType.CUDA_GPU, 0)
                .deviceName("Stub GPU (filling)")
                .totalMemory(24L * 1024 * 1024 * 1024)
                .availableMemory(2L * 1024 * 1024 * 1024) // 2 GB free
                .isDefault(true)
                .build();

        StubDeviceDescriptor gpu1 = StubDeviceDescriptor.builder(DeviceType.CUDA_GPU, 1)
                .deviceName("Stub GPU (overflow)")
                .totalMemory(24L * 1024 * 1024 * 1024)
                .availableMemory(20L * 1024 * 1024 * 1024)
                .build();

        memMgr.configureStubTopology(Arrays.asList(gpu0, gpu1));

        TransferSubsystem transferSub = new TransferSubsystem();
        transferSub.setEnabled(true);
        DspReplayTransferAnalytics analytics = new DspReplayTransferAnalytics(transferSub, memMgr);

        long stepAllocation = 256L * 1024 * 1024; // 256 MB per step
        List<Long> gpu0FreeHistory = new ArrayList<>();
        int stepWhereOomStarts = -1;

        for (int step = 0; step < 12; step++) {
            long freeBefore = gpu0.getAvailableMemory();
            gpu0FreeHistory.add(freeBefore);

            analytics.beginStep(0, 42L);

            if (freeBefore >= stepAllocation) {
                // Can still fit on GPU 0
                gpu0.consumeMemory(stepAllocation);
                analytics.recordTransfer(TransferEvent.builder()
                        .variableName("step_buffer_" + step)
                        .direction(TransferDirection.D2D)
                        .reason(TransferReason.CAPTURE_BUFFER_COPY)
                        .bytes(stepAllocation)
                        .durationNanos(1000)
                        .sourceDeviceId(0)
                        .destDeviceId(0)
                        .build());
            } else {
                // GPU 0 exhausted — this is where leaks would show up
                if (stepWhereOomStarts < 0) stepWhereOomStarts = step;
                // Record overflow transfer to GPU 1
                analytics.recordTransfer(TransferEvent.builder()
                        .variableName("step_buffer_" + step)
                        .direction(TransferDirection.D2D)
                        .reason(TransferReason.FALLBACK_MIGRATION)
                        .bytes(stepAllocation)
                        .durationNanos(3000) // slower cross-device
                        .sourceDeviceId(0)
                        .destDeviceId(1)
                        .build());
            }

            analytics.endStep();
        }

        // 2 GB / 256 MB = 8 steps before OOM
        assertEquals(8, stepWhereOomStarts,
                "OOM should start at step 8 (2GB / 256MB per step)");

        // Verify GPU 0 is fully exhausted
        assertEquals(0, gpu0.getAvailableMemory(),
                "GPU 0 should be fully exhausted");

        // Verify segment summary captures both phases
        DspReplayTransferAnalytics.SegmentTransferSummary seg = analytics.getSegmentSummary(0);
        assertNotNull(seg);
        assertEquals(12, seg.getStepCount());
        assertTrue(seg.getBreakdownByReason().containsKey(TransferReason.CAPTURE_BUFFER_COPY));
        assertTrue(seg.getBreakdownByReason().containsKey(TransferReason.FALLBACK_MIGRATION));

        // Verify fallback transfers happened
        DspReplayTransferAnalytics.TransferBreakdown fallback =
                seg.getBreakdownByReason().get(TransferReason.FALLBACK_MIGRATION);
        assertEquals(4, fallback.getCount(), "Should have 4 fallback transfers (steps 8-11)");
        assertEquals(4 * stepAllocation, fallback.getBytes());
    }

    // =========================================================================
    // Test 4: Memory pressure callback fires during simulated OOM
    // =========================================================================

    @Test
    @DisplayName("Memory pressure callback fires when simulated GPU hits threshold")
    void testMemoryPressureCallbackWithSimulation() {
        // 10 GB total, start with 10 GB free (simulate progressive allocation)
        StubDeviceDescriptor gpu0 = StubDeviceDescriptor.builder(DeviceType.CUDA_GPU, 0)
                .totalMemory(10L * 1024 * 1024 * 1024)
                .availableMemory(10L * 1024 * 1024 * 1024)
                .isDefault(true)
                .build();
        memMgr.configureStubTopology(Collections.singletonList(gpu0));
        memMgr.setMemoryPressureThreshold(0.9); // 90%

        AtomicBoolean pressureFired = new AtomicBoolean(false);
        AtomicInteger pressureCount = new AtomicInteger(0);
        memMgr.addMemoryPressureCallback((device, utilization) -> {
            pressureFired.set(true);
            pressureCount.incrementAndGet();
        });

        // Allocate 8 GB via simulation — effective free = 10-8 = 2 GB → 80% utilization
        memMgr.recordAllocation(gpu0, 8L * 1024 * 1024 * 1024);
        double util80 = memMgr.getMemoryUtilization(gpu0);
        assertFalse(pressureFired.get(),
                "Pressure should NOT fire at " + (util80 * 100) + "% utilization");

        // Allocate 1.5 GB more — effective free = 2 - 1.5 = 0.5 GB → 95% utilization
        memMgr.recordAllocation(gpu0, (long)(1.5 * 1024 * 1024 * 1024));
        double util95 = memMgr.getMemoryUtilization(gpu0);
        assertTrue(pressureFired.get(),
                "Pressure SHOULD fire at " + (util95 * 100) + "% utilization (threshold 90%)");
        assertTrue(pressureCount.get() >= 1, "Callback should have fired at least once");

        // Verify hasMemoryPressure agrees
        assertTrue(memMgr.hasMemoryPressure(gpu0));
    }

    // =========================================================================
    // Test 5: CPU fallback when single GPU is maxed (no peer GPU available)
    // =========================================================================

    @Test
    @DisplayName("Single maxed GPU falls back to CPU for large allocations")
    void testSingleMaxedGpuCpuFallback() {
        List<StubDeviceDescriptor> stubs = singleMaxedGpuWithCpuFallback();
        memMgr.configureStubTopology(stubs);
        StubDeviceDescriptor gpu0 = stubs.get(0);
        StubDeviceDescriptor cpu = stubs.get(1);

        // GPU 0 has only 100 MB free, needs 512 MB
        memMgr.setMemoryCap(gpu0, gpu0.getTotalMemory());
        memMgr.recordAllocation(gpu0, (long)(23.9 * 1024 * 1024 * 1024));

        DspReplayTransferAnalytics analytics = new DspReplayTransferAnalytics(null, memMgr);

        DeviceDescriptor target = analytics.checkMemoryPressureForTransfer(
                gpu0, 512L * 1024 * 1024, TransferReason.CAPTURE_BUFFER_COPY);

        // Should be rerouted to CPU (only device with space)
        DspReplayTransferAnalytics.ReplayTransferReport report = analytics.getReport();
        assertTrue(report.getRoutingDecisions().size() > 0,
                "Should have a routing decision");
        assertNotEquals(gpu0.getDeviceId(), target.getDeviceId(),
                "Should NOT stay on maxed GPU 0");
    }

    // =========================================================================
    // Test 6: Simulated weight migration leak pattern
    // =========================================================================

    @Test
    @DisplayName("Weight migration across devices simulates leak from unreleased cross-device refs")
    void testWeightMigrationLeakPattern() {
        List<StubDeviceDescriptor> stubs = maxedOutGpuTopology();
        memMgr.configureStubTopology(stubs);
        StubDeviceDescriptor gpu0 = stubs.get(0);
        StubDeviceDescriptor gpu1 = stubs.get(1);

        TransferSubsystem transferSub = new TransferSubsystem();
        transferSub.setEnabled(true);
        DspReplayTransferAnalytics analytics = new DspReplayTransferAnalytics(transferSub, memMgr);

        // Simulate the leak pattern: model weights on GPU 0, replicated to GPU 1
        // for overflow execution. Without proper cleanup, both copies persist.
        long weightSize = 500L * 1024 * 1024; // 500 MB per weight tensor
        int numWeights = 8;
        long totalMigratedBytes = 0;

        for (int w = 0; w < numWeights; w++) {
            analytics.beginStep(w, 77777L);

            // Record weight replication: GPU 0 → GPU 1
            analytics.recordTransfer(TransferEvent.builder()
                    .variableName("weight_layer_" + w)
                    .direction(TransferDirection.D2D)
                    .reason(TransferReason.CONSTANT_REPLICATION)
                    .bytes(weightSize)
                    .durationNanos(weightSize / (300L * 1024 * 1024)) // NVLink speed
                    .sourceDeviceId(0)
                    .destDeviceId(1)
                    .build());

            // GPU 1 memory consumed by the replica
            gpu1.consumeMemory(weightSize);
            totalMigratedBytes += weightSize;

            analytics.endStep();
        }

        // Verify: 4 GB of weight replicas now on GPU 1
        assertEquals(numWeights * weightSize, totalMigratedBytes);
        long gpu1Used = (20L * 1024 * 1024 * 1024) - gpu1.getAvailableMemory();
        assertEquals(totalMigratedBytes, gpu1Used,
                "GPU 1 memory consumed should match total migrated weight bytes");

        // Verify per-variable tracking in the subsystem
        assertEquals(numWeights, transferSub.getTotalTransferCount());
        assertEquals(totalMigratedBytes, transferSub.getTotalBytes());

        // Verify segment breakdown: each weight is its own segment
        for (int w = 0; w < numWeights; w++) {
            DspReplayTransferAnalytics.SegmentTransferSummary seg = analytics.getSegmentSummary(w);
            assertNotNull(seg, "Segment " + w + " should exist");
            assertEquals(1, seg.getStepCount());
            assertEquals(weightSize, seg.getTotalTransferBytes());
            assertTrue(seg.getBreakdownByReason().containsKey(TransferReason.CONSTANT_REPLICATION));
        }
    }

    // =========================================================================
    // Test 7: Simulated free memory vs actual available memory consistency
    // =========================================================================

    @Test
    @DisplayName("Simulated free memory correctly overrides actual device queries")
    void testSimulatedFreeMemoryOverride() {
        List<StubDeviceDescriptor> stubs = maxedOutGpuTopology();
        memMgr.configureStubTopology(stubs);

        // Verify simulation is active
        assertTrue(memMgr.isMemorySimulationEnabled());

        // Verify getEffectiveFreeMemory returns simulated values
        long gpu0Sim = memMgr.getEffectiveFreeMemory(0, 999999L);
        assertEquals(500L * 1024 * 1024, gpu0Sim,
                "Effective free memory should use simulated value, not actual");

        long gpu1Sim = memMgr.getEffectiveFreeMemory(1, 999999L);
        assertEquals(20L * 1024 * 1024 * 1024, gpu1Sim);

        // Record simulated allocations and verify they decrease free memory
        memMgr.recordSimulatedAllocation(0, 200L * 1024 * 1024);
        long gpu0After = memMgr.getEffectiveFreeMemory(0, 999999L);
        assertEquals(300L * 1024 * 1024, gpu0After,
                "Should be 300 MB after allocating 200 MB from 500 MB");

        // Verify getSimulatedAllocatedMemory tracking
        assertEquals(200L * 1024 * 1024, memMgr.getSimulatedAllocatedMemory(0));
    }

    // =========================================================================
    // Test 8: Full DSP replay lifecycle with maxed GPU simulation
    // =========================================================================

    @Test
    @DisplayName("Full DSP replay lifecycle: compile → capture → replay with maxed GPU")
    void testFullReplayLifecycleMaxedGpu() {
        List<StubDeviceDescriptor> stubs = maxedOutGpuTopology();
        memMgr.configureStubTopology(stubs);
        StubDeviceDescriptor gpu0 = stubs.get(0);
        StubDeviceDescriptor gpu1 = stubs.get(1);

        StubDeviceContextProvider provider = memMgr.getStubContextProvider();
        provider.clearSwitchHistory();

        TransferSubsystem transferSub = new TransferSubsystem();
        transferSub.setEnabled(true);
        DspReplayTransferAnalytics analytics = new DspReplayTransferAnalytics(transferSub, memMgr);

        // Phase 1: Compile (on GPU 0 where model lives)
        provider.switchDevice(0, "DspLifecycle", "compile_plan");
        analytics.beginStep(0, 12345L);
        // External inputs synced H2D
        analytics.recordTransfer(TransferEvent.builder()
                .variableName("input_ids")
                .direction(TransferDirection.H2D)
                .reason(TransferReason.SYNC_TO_DEVICE)
                .bytes(1024)
                .durationNanos(100)
                .sourceDeviceId(-1).destDeviceId(0)
                .build());
        analytics.endStep();

        // Phase 2: Capture — needs capture buffers (512 MB)
        // GPU 0 only has 500 MB free — capture buffer won't fit
        memMgr.setMemoryCap(gpu0, gpu0.getTotalMemory());
        memMgr.recordAllocation(gpu0, (long)(23.5 * 1024 * 1024 * 1024));

        analytics.beginStep(1, 12345L);
        DeviceDescriptor captureTarget = analytics.checkMemoryPressureForTransfer(
                gpu0, 512L * 1024 * 1024, TransferReason.CAPTURE_BUFFER_COPY);
        // Capture buffer goes to overflow device
        analytics.recordTransfer(TransferEvent.builder()
                .variableName("capture_buffer_seg1")
                .direction(TransferDirection.D2D)
                .reason(TransferReason.CAPTURE_BUFFER_COPY)
                .bytes(512L * 1024 * 1024)
                .durationNanos(2000)
                .sourceDeviceId(0)
                .destDeviceId(captureTarget.getDeviceIndex())
                .build());
        analytics.endStep();

        // Phase 3: Replay — multiple steps using captured graph
        for (int replay = 0; replay < 5; replay++) {
            analytics.beginStep(2, 12345L);
            // Refresh capture buffers (D2D copy)
            analytics.recordTransfer(TransferEvent.builder()
                    .variableName("capture_refresh_" + replay)
                    .direction(TransferDirection.D2D)
                    .reason(TransferReason.CAPTURE_BUFFER_COPY)
                    .bytes(4096)
                    .durationNanos(500)
                    .sourceDeviceId(0).destDeviceId(0)
                    .build());
            analytics.endStep();
        }

        // Verify full lifecycle analytics
        DspReplayTransferAnalytics.ReplayTransferReport report = analytics.getReport();
        assertEquals(7, report.getTotalCount()); // 1 compile + 1 capture + 5 replay
        assertTrue(report.getRoutingDecisions().size() >= 1,
                "Capture phase should have triggered at least 1 routing decision");

        // Verify 3 distinct segments
        assertEquals(3, report.getSegmentSummaries().size(),
                "Should have 3 segments: compile(0), capture(1), replay(2)");

        // Verify replay segment has 5 steps
        DspReplayTransferAnalytics.SegmentTransferSummary replaySeg = analytics.getSegmentSummary(2);
        assertNotNull(replaySeg);
        assertEquals(5, replaySeg.getStepCount());

        // Verify device switch history
        List<StubDeviceContextProvider.DeviceSwitchRecord> switchHistory = provider.getSwitchHistory();
        assertTrue(switchHistory.size() >= 1, "Should have at least 1 device switch");
    }

    // =========================================================================
    // Test 9: Replay profile enrichment with multi-device analytics
    // =========================================================================

    @Test
    @DisplayName("ReplayProfile captures multi-device analytics: device IDs and memory state")
    void testReplayProfileMultiDeviceEnrichment() {
        List<StubDeviceDescriptor> stubs = maxedOutGpuTopology();
        memMgr.configureStubTopology(stubs);

        // Build a profile with analytics fields via JSON (fields are package-private)
        String json = "{\"shapeHash\":54321,\"numSegments\":3,\"captureTimestamp\":12345," +
                "\"backendName\":\"TritonGraphBackend\",\"primaryDeviceId\":0," +
                "\"deviceMemoryAtCapture\":{}," +
                "\"shapes\":{}," +
                "\"segments\":[" +
                "{\"segIdx\":0,\"capturable\":true,\"replayState\":3,\"replayCount\":10," +
                "\"numCaptureBuffers\":5,\"executionDeviceId\":0,\"transferBytes\":4096," +
                "\"transferCount\":0,\"transferDurationNanos\":0}," +
                "{\"segIdx\":1,\"capturable\":true,\"replayState\":3,\"replayCount\":10," +
                "\"numCaptureBuffers\":5,\"executionDeviceId\":1,\"transferBytes\":" + (512L * 1024 * 1024) + "," +
                "\"transferCount\":1,\"transferDurationNanos\":5000}," +
                "{\"segIdx\":2,\"capturable\":true,\"replayState\":3,\"replayCount\":10," +
                "\"numCaptureBuffers\":5,\"executionDeviceId\":0,\"transferBytes\":4096," +
                "\"transferCount\":0,\"transferDurationNanos\":0}" +
                "]}";

        ReplayProfileManager.ReplayProfile parsed = ReplayProfileManager.ReplayProfile.fromJson(json);
        assertEquals(54321, parsed.getShapeHash());
        assertEquals(0, parsed.getPrimaryDeviceId());
        assertEquals(3, parsed.getSegments().size());

        // Segment 1 was the overflow segment — executed on GPU 1
        assertEquals(1, parsed.getSegments().get(1).getExecutionDeviceId());
        assertEquals(512L * 1024 * 1024, parsed.getSegments().get(1).getTransferBytes());
        assertEquals(1, parsed.getSegments().get(1).getTransferCount());

        // Segments 0 and 2 stayed on GPU 0
        assertEquals(0, parsed.getSegments().get(0).getExecutionDeviceId());
        assertEquals(0, parsed.getSegments().get(2).getExecutionDeviceId());

        // Re-serialize and verify round-trip
        String reserialized = parsed.toJson();
        assertNotNull(reserialized);
        assertTrue(reserialized.contains("\"primaryDeviceId\":0"));
        assertTrue(reserialized.contains("\"executionDeviceId\":1"));

        ReplayProfileManager.ReplayProfile roundTripped =
                ReplayProfileManager.ReplayProfile.fromJson(reserialized);
        assertEquals(parsed.getShapeHash(), roundTripped.getShapeHash());
        assertEquals(parsed.getSegments().size(), roundTripped.getSegments().size());
        assertEquals(parsed.getSegments().get(1).getTransferBytes(),
                roundTripped.getSegments().get(1).getTransferBytes());
    }

    // =========================================================================
    // Test 10: Concurrent replay steps on different devices (thread safety)
    // =========================================================================

    @Test
    @DisplayName("Concurrent DSP replay steps on different devices preserve isolation")
    void testConcurrentReplayStepsThreadSafety() throws InterruptedException {
        List<StubDeviceDescriptor> stubs = maxedOutGpuTopology();
        memMgr.configureStubTopology(stubs);

        TransferSubsystem transferSub = new TransferSubsystem();
        transferSub.setEnabled(true);
        DspReplayTransferAnalytics analytics = new DspReplayTransferAnalytics(transferSub, memMgr);

        int numThreads = 4;
        int stepsPerThread = 50;
        Thread[] threads = new Thread[numThreads];
        AtomicBoolean anyError = new AtomicBoolean(false);

        for (int t = 0; t < numThreads; t++) {
            int threadIdx = t;
            int segmentIdx = t; // Each thread gets its own segment
            threads[t] = new Thread(() -> {
                try {
                    for (int step = 0; step < stepsPerThread; step++) {
                        analytics.beginStep(segmentIdx, (long)(threadIdx * 1000 + step));
                        analytics.recordTransfer(TransferEvent.builder()
                                .variableName("thread_" + threadIdx + "_var_" + step)
                                .direction(TransferDirection.D2D)
                                .reason(TransferReason.CAPTURE_BUFFER_COPY)
                                .bytes(1024)
                                .durationNanos(100)
                                .sourceDeviceId(0)
                                .destDeviceId(threadIdx % 2)
                                .build());
                        analytics.endStep();
                    }
                } catch (Exception e) {
                    anyError.set(true);
                }
            });
        }

        for (Thread t : threads) t.start();
        for (Thread t : threads) t.join(10000);

        assertFalse(anyError.get(), "No errors should occur during concurrent replay");

        // Verify each segment got its steps
        for (int t = 0; t < numThreads; t++) {
            DspReplayTransferAnalytics.SegmentTransferSummary seg = analytics.getSegmentSummary(t);
            assertNotNull(seg, "Segment " + t + " should exist");
            assertEquals(stepsPerThread, seg.getStepCount(),
                    "Segment " + t + " should have " + stepsPerThread + " steps");
            assertEquals(stepsPerThread, seg.getTotalTransferCount());
        }

        // Verify global totals
        DspReplayTransferAnalytics.ReplayTransferReport report = analytics.getReport();
        assertEquals(numThreads * stepsPerThread, report.getTotalCount());
        assertEquals(numThreads * stepsPerThread * 1024L, report.getTotalBytes());
    }

    // =========================================================================
    // Test 11: Memory simulation correctly reports canAllocate / hasMemoryPressure
    // =========================================================================

    @Test
    @DisplayName("canAllocate and hasMemoryPressure respect simulated memory state")
    void testCanAllocateAndPressureWithSimulation() {
        // GPU 0: 24 GB total, start with 2 GB free (simulates model-loaded state)
        // GPU 1: 24 GB total, 20 GB free
        StubDeviceDescriptor gpu0 = StubDeviceDescriptor.builder(DeviceType.CUDA_GPU, 0)
                .totalMemory(24L * 1024 * 1024 * 1024)
                .availableMemory(2L * 1024 * 1024 * 1024) // 2 GB free
                .isDefault(true)
                .build();
        StubDeviceDescriptor gpu1 = StubDeviceDescriptor.builder(DeviceType.CUDA_GPU, 1)
                .totalMemory(24L * 1024 * 1024 * 1024)
                .availableMemory(20L * 1024 * 1024 * 1024)
                .build();
        memMgr.configureStubTopology(Arrays.asList(gpu0, gpu1));
        memMgr.setMemoryPressureThreshold(0.9);

        // GPU 0: 2 GB free out of 24 GB → ~91.7% used → pressure (above 90%)
        assertTrue(memMgr.hasMemoryPressure(gpu0), "GPU 0 at ~92% should have pressure");
        assertFalse(memMgr.canAllocate(gpu0, 5L * 1024 * 1024 * 1024),
                "GPU 0 should NOT be able to allocate 5 GB with only 2 GB free");
        assertTrue(memMgr.canAllocate(gpu0, 1L * 1024 * 1024 * 1024),
                "GPU 0 should be able to allocate 1 GB (fits in 2 GB free)");

        // Now consume 1 GB on GPU 0 via simulated allocation
        memMgr.recordSimulatedAllocation(0, 1L * 1024 * 1024 * 1024);
        // Effective free: 2 GB - 1 GB = 1 GB
        assertTrue(memMgr.canAllocate(gpu0, 500L * 1024 * 1024),
                "GPU 0 should be able to allocate 500 MB (1 GB free remaining)");
        assertFalse(memMgr.canAllocate(gpu0, 2L * 1024 * 1024 * 1024),
                "GPU 0 should NOT be able to allocate 2 GB (only 1 GB free)");

        // GPU 1: 20 GB free out of 24 GB → ~17% used → no pressure
        assertFalse(memMgr.hasMemoryPressure(gpu1), "GPU 1 at 17% should NOT have pressure");
        assertTrue(memMgr.canAllocate(gpu1, 10L * 1024 * 1024 * 1024),
                "GPU 1 should be able to allocate 10 GB");
    }

    // =========================================================================
    // Test 12: selectDeviceForAllocation prefers GPU with most space
    // =========================================================================

    @Test
    @DisplayName("selectDeviceForAllocation routes to GPU with most available memory")
    void testSelectDeviceRoutesToMostFreeGpu() {
        // GPU 0: only 200 MB free (model loaded)
        // GPU 1: 20 GB free (overflow target)
        StubDeviceDescriptor gpu0 = StubDeviceDescriptor.builder(DeviceType.CUDA_GPU, 0)
                .totalMemory(24L * 1024 * 1024 * 1024)
                .availableMemory(200L * 1024 * 1024) // 200 MB free
                .isDefault(true)
                .build();
        StubDeviceDescriptor gpu1 = StubDeviceDescriptor.builder(DeviceType.CUDA_GPU, 1)
                .totalMemory(24L * 1024 * 1024 * 1024)
                .availableMemory(20L * 1024 * 1024 * 1024)
                .build();
        memMgr.configureStubTopology(Arrays.asList(gpu0, gpu1));

        // Request 1 GB — GPU 0 can't fit it (only 200 MB), GPU 1 can
        DeviceDescriptor selected = memMgr.selectDeviceForAllocation(1L * 1024 * 1024 * 1024);
        assertNotNull(selected);
        assertEquals(gpu1.getDeviceId(), selected.getDeviceId(),
                "Should select GPU 1 when GPU 0 can't fit 1 GB");

        // Request 100 MB — GPU 0 CAN fit this (200 MB free)
        DeviceDescriptor small = memMgr.selectDeviceForAllocation(100L * 1024 * 1024);
        assertNotNull(small);
        assertEquals(gpu0.getDeviceId(), small.getDeviceId(),
                "Default device should handle small allocations it can fit");
    }
}
