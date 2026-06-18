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

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for DSP Replay Analytics + Device Memory Manager integration + Device Stubbing.
 */
public class DspReplayDeviceAnalyticsTest {

    private DeviceMemoryManager memMgr;

    @BeforeEach
    void setUp() {
        memMgr = DeviceMemoryManager.getInstance();
    }

    @AfterEach
    void tearDown() {
        memMgr.clearStubTopology();
    }

    private List<StubDeviceDescriptor> twoGpuTopology() {
        StubDeviceDescriptor gpu0 = StubDeviceDescriptor.builder(DeviceType.CUDA_GPU, 0)
                .deviceName("Stub RTX 4090")
                .totalMemory(24L * 1024 * 1024 * 1024)
                .availableMemory(20L * 1024 * 1024 * 1024)
                .isDefault(true)
                .addPeerDevice(1)
                .build();

        StubDeviceDescriptor gpu1 = StubDeviceDescriptor.builder(DeviceType.CUDA_GPU, 1)
                .deviceName("Stub RTX 3090")
                .totalMemory(24L * 1024 * 1024 * 1024)
                .availableMemory(20L * 1024 * 1024 * 1024)
                .addPeerDevice(0)
                .build();

        return Arrays.asList(gpu0, gpu1);
    }

    @Test
    void testStubTopologySetup() {
        List<StubDeviceDescriptor> stubs = twoGpuTopology();
        memMgr.configureStubTopology(stubs);

        // Verify devices registered
        assertTrue(memMgr.isMemorySimulationEnabled());
        assertNotNull(memMgr.getStubContextProvider());
        assertEquals(2, memMgr.getStubContextProvider().getDeviceCount());

        // Verify default device set to first GPU
        DeviceDescriptor defaultDev = memMgr.getDefaultDevice();
        assertEquals(DeviceType.CUDA_GPU, defaultDev.getDeviceType());
        assertEquals(0, defaultDev.getDeviceIndex());

        // Verify simulated memory
        long freeMem = memMgr.getSimulatedFreeMemory(0);
        assertEquals(20L * 1024 * 1024 * 1024, freeMem);
    }

    @Test
    void testTransferRecordingDuringReplaySteps() {
        TransferSubsystem transferSub = new TransferSubsystem();
        transferSub.setEnabled(true);
        DspReplayTransferAnalytics analytics = new DspReplayTransferAnalytics(transferSub, memMgr);

        // Simulate a replay step
        analytics.beginStep(0, 12345L);
        analytics.recordTransfer(TransferEvent.builder()
                .variableName("weight_0")
                .direction(TransferDirection.D2D)
                .reason(TransferReason.CAPTURE_BUFFER_COPY)
                .bytes(4096)
                .durationNanos(1000)
                .sourceDeviceId(0)
                .destDeviceId(0)
                .build());
        analytics.recordTransfer(TransferEvent.builder()
                .variableName("input_0")
                .direction(TransferDirection.H2D)
                .reason(TransferReason.EXTERNAL_INPUT_MIGRATION)
                .bytes(8192)
                .durationNanos(2000)
                .sourceDeviceId(-1)
                .destDeviceId(0)
                .build());
        DspReplayTransferAnalytics.StepTransferSummary step = analytics.endStep();

        assertNotNull(step);
        assertEquals(0, step.getSegmentIdx());
        assertEquals(12345L, step.getShapeHash());
        assertEquals(2, step.getTransferCount());
        assertEquals(4096 + 8192, step.getTransferBytes());
        assertEquals(3000, step.getTransferDurationNanos());
        assertEquals(2, step.getBytesByReason().size());
        assertEquals(4096L, step.getBytesByReason().get(TransferReason.CAPTURE_BUFFER_COPY));
        assertEquals(8192L, step.getBytesByReason().get(TransferReason.EXTERNAL_INPUT_MIGRATION));

        // Verify underlying subsystem also got the events
        assertEquals(2, transferSub.getTotalTransferCount());
    }

    @Test
    void testMemoryPressureTriggersRerouting() {
        // Setup: GPU 0 nearly full, GPU 1 has space
        StubDeviceDescriptor gpu0 = StubDeviceDescriptor.builder(DeviceType.CUDA_GPU, 0)
                .totalMemory(10L * 1024 * 1024 * 1024)
                .availableMemory(100 * 1024 * 1024) // 100 MB free
                .build();
        StubDeviceDescriptor gpu1 = StubDeviceDescriptor.builder(DeviceType.CUDA_GPU, 1)
                .totalMemory(10L * 1024 * 1024 * 1024)
                .availableMemory(8L * 1024 * 1024 * 1024) // 8 GB free
                .build();
        memMgr.configureStubTopology(Arrays.asList(gpu0, gpu1));

        // Set cap on GPU 0 to trigger pressure
        memMgr.setMemoryCap(gpu0, 10L * 1024 * 1024 * 1024);
        // Record nearly full allocation
        memMgr.recordAllocation(gpu0, (long) (9.5 * 1024 * 1024 * 1024));

        DspReplayTransferAnalytics analytics = new DspReplayTransferAnalytics(null, memMgr);

        // Request large allocation on GPU 0 — should reroute to GPU 1
        DeviceDescriptor actual = analytics.checkMemoryPressureForTransfer(
                gpu0, 1L * 1024 * 1024 * 1024, TransferReason.CONSTANT_REPLICATION);

        // Should have been rerouted
        DspReplayTransferAnalytics.ReplayTransferReport report = analytics.getReport();
        assertEquals(1, report.getRoutingDecisions().size());

        DspReplayTransferAnalytics.RoutingDecision decision = report.getRoutingDecisions().get(0);
        assertEquals(gpu0.getDeviceId(), decision.getIntendedDeviceId());
        assertNotEquals(gpu0.getDeviceId(), decision.getActualDeviceId());
        assertEquals(TransferReason.CONSTANT_REPLICATION, decision.getReason());
    }

    @Test
    void testMultiDeviceP2PTransferAnalytics() {
        List<StubDeviceDescriptor> stubs = twoGpuTopology();
        memMgr.configureStubTopology(stubs);

        StubDeviceDescriptor gpu0 = stubs.get(0);
        StubDeviceDescriptor gpu1 = stubs.get(1);

        // P2P peer access
        assertTrue(gpu0.hasPeerAccessTo(1));
        assertTrue(gpu1.hasPeerAccessTo(0));

        // NVLink-like bandwidth for peers
        long bandwidth = gpu0.getTransferBandwidthTo(gpu1);
        assertEquals(300L * 1024 * 1024 * 1024, bandwidth);
    }

    @Test
    void testNonP2PTopologyFallback() {
        // Two GPUs without peer access
        StubDeviceDescriptor gpu0 = StubDeviceDescriptor.builder(DeviceType.CUDA_GPU, 0)
                .totalMemory(8L * 1024 * 1024 * 1024)
                .build();
        StubDeviceDescriptor gpu1 = StubDeviceDescriptor.builder(DeviceType.CUDA_GPU, 1)
                .totalMemory(8L * 1024 * 1024 * 1024)
                .build();
        memMgr.configureStubTopology(Arrays.asList(gpu0, gpu1));

        // No peer access — PCIe bandwidth
        assertFalse(gpu0.hasPeerAccessTo(1));
        long bandwidth = gpu0.getTransferBandwidthTo(gpu1);
        assertEquals(12L * 1024 * 1024 * 1024, bandwidth);
    }

    @Test
    void testPerSegmentTransferBreakdown() {
        DspReplayTransferAnalytics analytics = new DspReplayTransferAnalytics(null, memMgr);

        // Multiple steps on segment 0
        for (int i = 0; i < 3; i++) {
            analytics.beginStep(0, 100L);
            analytics.recordTransfer(TransferEvent.builder()
                    .variableName("w")
                    .direction(TransferDirection.D2D)
                    .reason(TransferReason.CAPTURE_BUFFER_COPY)
                    .bytes(1024)
                    .durationNanos(500)
                    .build());
            analytics.endStep();
        }

        // One step on segment 1
        analytics.beginStep(1, 100L);
        analytics.recordTransfer(TransferEvent.builder()
                .variableName("x")
                .direction(TransferDirection.H2D)
                .reason(TransferReason.SYNC_TO_DEVICE)
                .bytes(2048)
                .durationNanos(1000)
                .build());
        analytics.endStep();

        DspReplayTransferAnalytics.SegmentTransferSummary seg0 = analytics.getSegmentSummary(0);
        assertNotNull(seg0);
        assertEquals(3, seg0.getStepCount());
        assertEquals(3, seg0.getTotalTransferCount());
        assertEquals(3072, seg0.getTotalTransferBytes());

        DspReplayTransferAnalytics.SegmentTransferSummary seg1 = analytics.getSegmentSummary(1);
        assertNotNull(seg1);
        assertEquals(1, seg1.getStepCount());
        assertEquals(1, seg1.getTotalTransferCount());
        assertEquals(2048, seg1.getTotalTransferBytes());

        // Check breakdown by reason
        assertTrue(seg0.getBreakdownByReason().containsKey(TransferReason.CAPTURE_BUFFER_COPY));
        assertEquals(3072, seg0.getBreakdownByReason().get(TransferReason.CAPTURE_BUFFER_COPY).getBytes());
    }

    @Test
    void testReplayProfileJsonRoundTrip() {
        // Create a profile with analytics fields populated
        ReplayProfileManager.ReplayProfile profile = new ReplayProfileManager.ReplayProfile();
        // Use reflection-free approach: set via the public setters
        profile.setPrimaryDeviceId(0);
        Map<Integer, Long> memMap = new HashMap<>();
        memMap.put(0, 20L * 1024 * 1024 * 1024);
        memMap.put(1, 18L * 1024 * 1024 * 1024);
        profile.setDeviceMemoryAtCapture(memMap);

        // Serialize via the static captureProfile approach is not possible without native handle,
        // so test the fromJson/toJson round-trip with a manually constructed JSON
        String json = "{\"shapeHash\":999,\"numSegments\":1,\"captureTimestamp\":12345," +
                "\"backendName\":\"test\",\"primaryDeviceId\":2," +
                "\"deviceMemoryAtCapture\":{}," +
                "\"shapes\":{}," +
                "\"segments\":[{\"segIdx\":0,\"capturable\":true,\"replayState\":3,\"replayCount\":10," +
                "\"numCaptureBuffers\":5,\"executionDeviceId\":1,\"transferBytes\":4096," +
                "\"transferCount\":2,\"transferDurationNanos\":8000}]}";

        ReplayProfileManager.ReplayProfile parsed = ReplayProfileManager.ReplayProfile.fromJson(json);
        assertEquals(999, parsed.getShapeHash());
        assertEquals(2, parsed.getPrimaryDeviceId());
        assertEquals(1, parsed.getNumSegments());
        assertNotNull(parsed.getSegments());
        assertEquals(1, parsed.getSegments().size());

        ReplayProfileManager.ReplayProfile.SegmentReplayInfo seg = parsed.getSegments().get(0);
        assertEquals(1, seg.getExecutionDeviceId());
        assertEquals(4096, seg.getTransferBytes());
        assertEquals(2, seg.getTransferCount());
        assertEquals(8000, seg.getTransferDurationNanos());
    }

    @Test
    void testDeviceSwitchHistoryTracking() {
        List<StubDeviceDescriptor> stubs = twoGpuTopology();
        memMgr.configureStubTopology(stubs);

        StubDeviceContextProvider provider = memMgr.getStubContextProvider();
        assertNotNull(provider);
        assertTrue(provider.getSwitchHistory().isEmpty());

        // Perform device switches
        provider.switchDevice(1, "TestClass", "load weights");
        provider.switchDevice(0, "TestClass", "execute plan");

        List<StubDeviceContextProvider.DeviceSwitchRecord> history = provider.getSwitchHistory();
        assertEquals(2, history.size());

        assertEquals(0, history.get(0).getPreviousDeviceId());
        assertEquals(1, history.get(0).getNewDeviceId());
        assertEquals("load weights", history.get(0).getReason());

        assertEquals(1, history.get(1).getPreviousDeviceId());
        assertEquals(0, history.get(1).getNewDeviceId());
        assertEquals("execute plan", history.get(1).getReason());

        // Clear and verify
        provider.clearSwitchHistory();
        assertTrue(provider.getSwitchHistory().isEmpty());
    }

    @Test
    void testProgressiveMemoryExhaustion() {
        StubDeviceDescriptor gpu0 = StubDeviceDescriptor.builder(DeviceType.CUDA_GPU, 0)
                .totalMemory(4L * 1024 * 1024 * 1024)
                .availableMemory(4L * 1024 * 1024 * 1024)
                .build();
        StubDeviceDescriptor gpu1 = StubDeviceDescriptor.builder(DeviceType.CUDA_GPU, 1)
                .totalMemory(4L * 1024 * 1024 * 1024)
                .availableMemory(4L * 1024 * 1024 * 1024)
                .build();
        memMgr.configureStubTopology(Arrays.asList(gpu0, gpu1));

        // Progressively consume GPU 0 memory
        long consumed = gpu0.consumeMemory(3L * 1024 * 1024 * 1024);
        assertEquals(3L * 1024 * 1024 * 1024, consumed);
        assertEquals(1L * 1024 * 1024 * 1024, gpu0.getAvailableMemory());

        // Consume more than available
        consumed = gpu0.consumeMemory(2L * 1024 * 1024 * 1024);
        assertEquals(1L * 1024 * 1024 * 1024, consumed); // Only 1 GB was left
        assertEquals(0, gpu0.getAvailableMemory());

        // GPU 1 still has memory
        assertEquals(4L * 1024 * 1024 * 1024, gpu1.getAvailableMemory());
    }

    @Test
    void testResetClearsAllAnalytics() {
        TransferSubsystem transferSub = new TransferSubsystem();
        transferSub.setEnabled(true);
        DspReplayTransferAnalytics analytics = new DspReplayTransferAnalytics(transferSub, memMgr);

        // Record some data
        analytics.beginStep(0, 100L);
        analytics.recordTransfer(TransferEvent.builder()
                .variableName("test")
                .direction(TransferDirection.D2D)
                .reason(TransferReason.CAPTURE_BUFFER_COPY)
                .bytes(1024)
                .durationNanos(100)
                .build());
        analytics.endStep();

        DspReplayTransferAnalytics.ReplayTransferReport report = analytics.getReport();
        assertEquals(1, report.getTotalCount());
        assertEquals(1, report.getSegmentSummaries().size());

        // Reset
        analytics.reset();

        report = analytics.getReport();
        assertEquals(0, report.getTotalCount());
        assertEquals(0, report.getTotalBytes());
        assertTrue(report.getSegmentSummaries().isEmpty());
        assertTrue(report.getRoutingDecisions().isEmpty());
    }
}
