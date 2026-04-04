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

package org.nd4j.autodiff.samediff.execution;

import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.device.DeviceMemoryManager;
import org.nd4j.linalg.framework.device.TransferEvent;
import org.nd4j.linalg.framework.device.TransferReason;
import org.nd4j.linalg.framework.device.TransferSubsystem;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Bridge between DSP replay execution and transfer tracking.
 * Wraps TransferSubsystem and DeviceMemoryManager to provide per-segment,
 * per-step transfer analytics during replay execution.
 *
 * <p>Thread-safe: uses ThreadLocal for step context, ConcurrentHashMap for segment summaries.</p>
 *
 * <p>Usage:
 * <pre>{@code
 * DspReplayTransferAnalytics analytics = new DspReplayTransferAnalytics(transferSub, memMgr);
 * analytics.beginStep(segmentIdx, shapeHash);
 * // ... execute segment, record transfers ...
 * analytics.recordTransfer(event);
 * StepTransferSummary step = analytics.endStep();
 * // After all steps:
 * ReplayTransferReport report = analytics.getReport();
 * }</pre>
 */
public class DspReplayTransferAnalytics {

    private final TransferSubsystem transferSubsystem;
    private final DeviceMemoryManager memoryManager;

    // Per-segment accumulated stats
    private final ConcurrentHashMap<Integer, SegmentTransferSummary> segmentSummaries = new ConcurrentHashMap<>();

    // Thread-local step context
    private final ThreadLocal<StepContext> currentStep = new ThreadLocal<>();

    // Routing decisions across all steps
    private final List<RoutingDecision> routingDecisions = Collections.synchronizedList(new ArrayList<>());

    // Global counters
    private final AtomicLong totalTransferBytes = new AtomicLong(0);
    private final AtomicLong totalTransferCount = new AtomicLong(0);
    private final AtomicLong totalTransferDurationNanos = new AtomicLong(0);

    public DspReplayTransferAnalytics(TransferSubsystem transferSubsystem,
                                       DeviceMemoryManager memoryManager) {
        this.transferSubsystem = transferSubsystem;
        this.memoryManager = memoryManager;
    }

    /**
     * Begin a replay step. Call before executing a segment.
     *
     * @param segmentIdx the segment index
     * @param shapeHash  the shape configuration hash
     */
    public void beginStep(int segmentIdx, long shapeHash) {
        StepContext ctx = new StepContext(segmentIdx, shapeHash, System.nanoTime());
        currentStep.set(ctx);
    }

    /**
     * End the current replay step and return its summary.
     *
     * @return the step's transfer summary, or null if no step was active
     */
    public StepTransferSummary endStep() {
        StepContext ctx = currentStep.get();
        if (ctx == null) return null;
        currentStep.remove();

        long durationNanos = System.nanoTime() - ctx.startNanos;

        StepTransferSummary stepSummary = new StepTransferSummary(
                ctx.segmentIdx, ctx.shapeHash, durationNanos,
                ctx.transferCount, ctx.transferBytes, ctx.transferDurationNanos,
                Collections.unmodifiableMap(new HashMap<>(ctx.bytesByReason)));

        // Accumulate into segment summary
        segmentSummaries.computeIfAbsent(ctx.segmentIdx, SegmentTransferSummary::new)
                .accumulate(stepSummary);

        return stepSummary;
    }

    /**
     * Record a transfer event during the current step.
     * Delegates to TransferSubsystem and accumulates per-step stats.
     *
     * @param event the transfer event
     */
    public void recordTransfer(TransferEvent event) {
        // Delegate to underlying subsystem
        if (transferSubsystem != null) {
            transferSubsystem.record(event);
        }

        // Global counters
        totalTransferCount.incrementAndGet();
        totalTransferBytes.addAndGet(event.getBytes());
        totalTransferDurationNanos.addAndGet(event.getDurationNanos());

        // Per-step accumulation
        StepContext ctx = currentStep.get();
        if (ctx != null) {
            ctx.transferCount++;
            ctx.transferBytes += event.getBytes();
            ctx.transferDurationNanos += event.getDurationNanos();
            if (event.getReason() != null) {
                ctx.bytesByReason.merge(event.getReason(), event.getBytes(), Long::sum);
            }
        }
    }

    /**
     * Check memory pressure on a target device for a pending transfer.
     * If pressure is detected, selects an alternative device and records a RoutingDecision.
     *
     * @param targetDevice the intended target device
     * @param bytes        bytes to transfer
     * @param reason       reason for the transfer
     * @return the device to actually use (may differ from targetDevice if pressure detected)
     */
    public DeviceDescriptor checkMemoryPressureForTransfer(DeviceDescriptor targetDevice,
                                                            long bytes,
                                                            TransferReason reason) {
        if (memoryManager == null || targetDevice == null) {
            return targetDevice;
        }

        boolean pressure = memoryManager.hasMemoryPressure(targetDevice);
        if (!pressure && memoryManager.canAllocate(targetDevice, bytes)) {
            return targetDevice;
        }

        // Pressure detected — find alternative
        DeviceDescriptor alternative = memoryManager.selectDeviceForAllocation(bytes);
        if (alternative == null || alternative.getDeviceId().equals(targetDevice.getDeviceId())) {
            // No better alternative found
            return targetDevice;
        }

        RoutingDecision decision = new RoutingDecision(
                targetDevice.getDeviceId(), alternative.getDeviceId(),
                bytes, reason, pressure, System.nanoTime());
        routingDecisions.add(decision);

        return alternative;
    }

    /**
     * Get the per-segment transfer summary for a specific segment.
     */
    public SegmentTransferSummary getSegmentSummary(int segmentIdx) {
        return segmentSummaries.get(segmentIdx);
    }

    /**
     * Get all segment summaries.
     */
    public Map<Integer, SegmentTransferSummary> getAllSegmentSummaries() {
        return Collections.unmodifiableMap(segmentSummaries);
    }

    /**
     * Get the full replay transfer report.
     */
    public ReplayTransferReport getReport() {
        return new ReplayTransferReport(
                Collections.unmodifiableMap(new HashMap<>(segmentSummaries)),
                totalTransferBytes.get(),
                totalTransferCount.get(),
                totalTransferDurationNanos.get(),
                Collections.unmodifiableList(new ArrayList<>(routingDecisions)));
    }

    /**
     * Reset all analytics state.
     */
    public void reset() {
        segmentSummaries.clear();
        routingDecisions.clear();
        totalTransferBytes.set(0);
        totalTransferCount.set(0);
        totalTransferDurationNanos.set(0);
        currentStep.remove();
    }

    // =========================================================================
    // Inner classes
    // =========================================================================

    /**
     * Thread-local step context for accumulating per-step transfer data.
     */
    private static class StepContext {
        final int segmentIdx;
        final long shapeHash;
        final long startNanos;
        long transferCount;
        long transferBytes;
        long transferDurationNanos;
        final Map<TransferReason, Long> bytesByReason = new HashMap<>();

        StepContext(int segmentIdx, long shapeHash, long startNanos) {
            this.segmentIdx = segmentIdx;
            this.shapeHash = shapeHash;
            this.startNanos = startNanos;
        }
    }

    /**
     * Per-segment transfer summary, accumulated across multiple steps.
     */
    public static class SegmentTransferSummary {
        private final int segmentIdx;
        private long totalTransferCount;
        private long totalTransferBytes;
        private long totalTransferDurationNanos;
        private int stepCount;
        private final Map<TransferReason, TransferBreakdown> breakdownByReason =
                new ConcurrentHashMap<>();

        public SegmentTransferSummary(int segmentIdx) {
            this.segmentIdx = segmentIdx;
        }

        synchronized void accumulate(StepTransferSummary step) {
            totalTransferCount += step.getTransferCount();
            totalTransferBytes += step.getTransferBytes();
            totalTransferDurationNanos += step.getTransferDurationNanos();
            stepCount++;

            for (Map.Entry<TransferReason, Long> entry : step.getBytesByReason().entrySet()) {
                breakdownByReason.computeIfAbsent(entry.getKey(), TransferBreakdown::new)
                        .add(1, entry.getValue());
            }
        }

        public int getSegmentIdx() { return segmentIdx; }
        public long getTotalTransferCount() { return totalTransferCount; }
        public long getTotalTransferBytes() { return totalTransferBytes; }
        public long getTotalTransferDurationNanos() { return totalTransferDurationNanos; }
        public int getStepCount() { return stepCount; }
        public Map<TransferReason, TransferBreakdown> getBreakdownByReason() {
            return Collections.unmodifiableMap(breakdownByReason);
        }
    }

    /**
     * Breakdown of transfer stats for a specific reason.
     */
    public static class TransferBreakdown {
        private final TransferReason reason;
        private long count;
        private long bytes;

        public TransferBreakdown(TransferReason reason) {
            this.reason = reason;
        }

        synchronized void add(long count, long bytes) {
            this.count += count;
            this.bytes += bytes;
        }

        public TransferReason getReason() { return reason; }
        public long getCount() { return count; }
        public long getBytes() { return bytes; }
    }

    /**
     * Summary of transfers for a single step execution.
     */
    public static class StepTransferSummary {
        private final int segmentIdx;
        private final long shapeHash;
        private final long stepDurationNanos;
        private final long transferCount;
        private final long transferBytes;
        private final long transferDurationNanos;
        private final Map<TransferReason, Long> bytesByReason;

        public StepTransferSummary(int segmentIdx, long shapeHash, long stepDurationNanos,
                                    long transferCount, long transferBytes, long transferDurationNanos,
                                    Map<TransferReason, Long> bytesByReason) {
            this.segmentIdx = segmentIdx;
            this.shapeHash = shapeHash;
            this.stepDurationNanos = stepDurationNanos;
            this.transferCount = transferCount;
            this.transferBytes = transferBytes;
            this.transferDurationNanos = transferDurationNanos;
            this.bytesByReason = bytesByReason;
        }

        public int getSegmentIdx() { return segmentIdx; }
        public long getShapeHash() { return shapeHash; }
        public long getStepDurationNanos() { return stepDurationNanos; }
        public long getTransferCount() { return transferCount; }
        public long getTransferBytes() { return transferBytes; }
        public long getTransferDurationNanos() { return transferDurationNanos; }
        public Map<TransferReason, Long> getBytesByReason() { return bytesByReason; }
    }

    /**
     * Records when memory pressure causes a transfer to be rerouted to a different device.
     */
    public static class RoutingDecision {
        private final String intendedDeviceId;
        private final String actualDeviceId;
        private final long bytes;
        private final TransferReason reason;
        private final boolean pressureDetected;
        private final long timestampNanos;

        public RoutingDecision(String intendedDeviceId, String actualDeviceId,
                               long bytes, TransferReason reason,
                               boolean pressureDetected, long timestampNanos) {
            this.intendedDeviceId = intendedDeviceId;
            this.actualDeviceId = actualDeviceId;
            this.bytes = bytes;
            this.reason = reason;
            this.pressureDetected = pressureDetected;
            this.timestampNanos = timestampNanos;
        }

        public String getIntendedDeviceId() { return intendedDeviceId; }
        public String getActualDeviceId() { return actualDeviceId; }
        public long getBytes() { return bytes; }
        public TransferReason getReason() { return reason; }
        public boolean isPressureDetected() { return pressureDetected; }
        public long getTimestampNanos() { return timestampNanos; }
    }

    /**
     * Full report aggregating all segments with totals and routing decisions.
     */
    public static class ReplayTransferReport {
        private final Map<Integer, SegmentTransferSummary> segmentSummaries;
        private final long totalBytes;
        private final long totalCount;
        private final long totalDurationNanos;
        private final List<RoutingDecision> routingDecisions;

        public ReplayTransferReport(Map<Integer, SegmentTransferSummary> segmentSummaries,
                                     long totalBytes, long totalCount, long totalDurationNanos,
                                     List<RoutingDecision> routingDecisions) {
            this.segmentSummaries = segmentSummaries;
            this.totalBytes = totalBytes;
            this.totalCount = totalCount;
            this.totalDurationNanos = totalDurationNanos;
            this.routingDecisions = routingDecisions;
        }

        public Map<Integer, SegmentTransferSummary> getSegmentSummaries() { return segmentSummaries; }
        public long getTotalBytes() { return totalBytes; }
        public long getTotalCount() { return totalCount; }
        public long getTotalDurationNanos() { return totalDurationNanos; }
        public List<RoutingDecision> getRoutingDecisions() { return routingDecisions; }

        public String summary() {
            StringBuilder sb = new StringBuilder();
            sb.append("ReplayTransferReport: ");
            sb.append(totalCount).append(" transfers, ");
            sb.append(totalBytes / (1024 * 1024)).append(" MB total");
            if (totalDurationNanos > 0) {
                double bandwidthMBps = (totalBytes * 1_000_000_000.0) / totalDurationNanos / (1024.0 * 1024.0);
                sb.append(String.format(", %.2f MB/s", bandwidthMBps));
            }
            sb.append(", ").append(routingDecisions.size()).append(" routing redirects");
            sb.append(", ").append(segmentSummaries.size()).append(" segments");
            for (Map.Entry<Integer, SegmentTransferSummary> e : segmentSummaries.entrySet()) {
                SegmentTransferSummary seg = e.getValue();
                sb.append("\n  Segment ").append(e.getKey()).append(": ");
                sb.append(seg.getTotalTransferCount()).append(" transfers, ");
                sb.append(seg.getTotalTransferBytes() / 1024).append(" KB, ");
                sb.append(seg.getStepCount()).append(" steps");
            }
            return sb.toString();
        }
    }
}
