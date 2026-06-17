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

package org.nd4j.linalg.api.ops.executioner;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.device.DeviceType;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.LongAdder;

/**
 * Tracks and logs metrics for data transfers between devices.
 * Provides detailed timing, bandwidth, and overhead analysis for multi-device operations.
 *
 * <p>Key metrics tracked:
 * <ul>
 *   <li>Per-transfer: source, destination, bytes, time, bandwidth</li>
 *   <li>Cumulative: total transfers, bytes, time by transfer type</li>
 *   <li>Overhead: percentage of execution time spent on transfers</li>
 * </ul>
 *
 * <p>Example usage:
 * <pre>{@code
 * TransferMetrics metrics = TransferMetrics.getInstance();
 *
 * // Record a transfer
 * long startNanos = System.nanoTime();
 * // ... perform transfer ...
 * long endNanos = System.nanoTime();
 * metrics.recordTransfer(sourceDevice, targetDevice, bytes, endNanos - startNanos);
 *
 * // Log summary
 * metrics.logSummary();
 * }</pre>
 *
 * Adam Gibson
 */
@Slf4j
public class TransferMetrics {

    private static final TransferMetrics INSTANCE = new TransferMetrics();

    // Configuration
    @Getter
    private volatile double warningThresholdPercent = 10.0;  // Warn if transfers > 10% of time
    @Getter
    private volatile boolean logIndividualTransfers = true;
    @Getter
    private volatile long minBytesForLogging = 1024 * 1024;  // 1MB minimum to log individual transfers

    // Cumulative statistics
    private final LongAdder totalTransfers = new LongAdder();
    private final LongAdder totalBytes = new LongAdder();
    private final LongAdder totalTimeNanos = new LongAdder();
    private final LongAdder totalExecutionTimeNanos = new LongAdder();

    // Per-route statistics (source->target)
    private final Map<String, RouteStats> routeStats = new ConcurrentHashMap<>();

    // Per-device statistics
    private final Map<String, DeviceStats> deviceStats = new ConcurrentHashMap<>();

    // Recent transfers for analysis (bounded)
    private final Deque<TransferRecord> recentTransfers = new LinkedList<>();
    private static final int MAX_RECENT_TRANSFERS = 1000;
    private final Object recentLock = new Object();

    // Warning state
    private final AtomicLong lastWarningTime = new AtomicLong(0);
    private static final long WARNING_COOLDOWN_MS = 5000;  // 5 seconds between warnings

    private TransferMetrics() {
        // Singleton
    }

    public static TransferMetrics getInstance() {
        return INSTANCE;
    }

    // =========================================================================
    // Configuration
    // =========================================================================

    /**
     * Set the warning threshold for transfer overhead percentage.
     * Warnings are logged when cumulative transfer time exceeds this percentage
     * of total execution time.
     *
     * @param percent threshold percentage (0-100)
     */
    public void setWarningThresholdPercent(double percent) {
        this.warningThresholdPercent = percent;
    }

    /**
     * Enable or disable logging of individual transfers.
     */
    public void setLogIndividualTransfers(boolean enabled) {
        this.logIndividualTransfers = enabled;
    }

    /**
     * Set minimum bytes for logging individual transfers.
     * Transfers smaller than this won't be logged individually.
     */
    public void setMinBytesForLogging(long bytes) {
        this.minBytesForLogging = bytes;
    }

    // =========================================================================
    // Recording
    // =========================================================================

    /**
     * Record a data transfer between devices.
     *
     * @param source source device
     * @param target target device
     * @param bytes number of bytes transferred
     * @param timeNanos time taken in nanoseconds
     */
    public void recordTransfer(DeviceDescriptor source, DeviceDescriptor target,
                               long bytes, long timeNanos) {
        if (source == null || target == null) {
            return;
        }

        String sourceId = source.getDeviceId();
        String targetId = target.getDeviceId();
        String routeKey = sourceId + " -> " + targetId;

        // Update cumulative stats
        totalTransfers.increment();
        totalBytes.add(bytes);
        totalTimeNanos.add(timeNanos);

        // Update route stats
        RouteStats route = routeStats.computeIfAbsent(routeKey,
            k -> new RouteStats(source, target));
        route.record(bytes, timeNanos);

        // Update device stats
        deviceStats.computeIfAbsent(sourceId, k -> new DeviceStats(source))
            .recordOutgoing(bytes, timeNanos);
        deviceStats.computeIfAbsent(targetId, k -> new DeviceStats(target))
            .recordIncoming(bytes, timeNanos);

        // Create record
        TransferRecord record = new TransferRecord(
            source, target, bytes, timeNanos, System.currentTimeMillis()
        );

        // Add to recent transfers (bounded)
        synchronized (recentLock) {
            recentTransfers.addLast(record);
            while (recentTransfers.size() > MAX_RECENT_TRANSFERS) {
                recentTransfers.removeFirst();
            }
        }

        // Log individual transfer if enabled and significant
        if (logIndividualTransfers && bytes >= minBytesForLogging) {
            logTransfer(record);
        }

        // Check warning threshold
        checkWarningThreshold();
    }

    /**
     * Record execution time (for calculating overhead percentage).
     *
     * @param timeNanos execution time in nanoseconds
     */
    public void recordExecutionTime(long timeNanos) {
        totalExecutionTimeNanos.add(timeNanos);
    }

    /**
     * Record an op execution with its execution time.
     * Convenience method that also tracks execution time for overhead calculation.
     *
     * @param device the device where execution happened
     * @param timeNanos execution time in nanoseconds
     */
    public void recordOpExecution(DeviceDescriptor device, long timeNanos) {
        totalExecutionTimeNanos.add(timeNanos);
        if (device != null) {
            deviceStats.computeIfAbsent(device.getDeviceId(), k -> new DeviceStats(device))
                .recordExecution(timeNanos);
        }
    }

    /**
     * Record an op execution with op name for detailed tracking.
     * Useful for debugging which ops are running on which backends.
     *
     * @param device the device where execution happened
     * @param opName the name of the op executed
     * @param timeNanos execution time in nanoseconds
     */
    public void recordOpExecution(DeviceDescriptor device, String opName, long timeNanos) {
        recordOpExecution(device, timeNanos);

        if (log.isTraceEnabled() && device != null) {
            log.trace("[OP_EXEC] {} on {} | {:.3f} ms",
                    opName, device.getDeviceId(), timeNanos / 1_000_000.0);
        }
    }

    /**
     * Record a CPU fallback execution (when an op that would normally run on GPU
     * is executed on CPU due to spillover).
     *
     * @param opName the name of the op
     * @param timeNanos execution time
     * @param reason the reason for CPU fallback (e.g., "memory_pressure", "data_locality")
     */
    public void recordCpuFallbackExecution(String opName, long timeNanos, String reason) {
        cpuFallbackCount.increment();
        cpuFallbackTimeNanos.add(timeNanos);

        DeviceDescriptor cpuDevice = DeviceDescriptor.cpu();
        recordOpExecution(cpuDevice, opName, timeNanos);

        if (log.isDebugEnabled()) {
            log.debug("[CPU_FALLBACK] {} | reason: {} | {:.3f} ms",
                    opName, reason, timeNanos / 1_000_000.0);
        }
    }

    // CPU fallback tracking
    private final LongAdder cpuFallbackCount = new LongAdder();
    private final LongAdder cpuFallbackTimeNanos = new LongAdder();

    /**
     * Get the number of operations that fell back to CPU execution.
     */
    public long getCpuFallbackCount() {
        return cpuFallbackCount.sum();
    }

    /**
     * Get the total time spent in CPU fallback execution.
     */
    public long getCpuFallbackTimeNanos() {
        return cpuFallbackTimeNanos.sum();
    }

    /**
     * Get execution statistics for a specific device type.
     *
     * @param deviceType the type of device (CPU, CUDA_GPU, etc.)
     * @return map with stats: opCount, totalTimeNanos, avgTimeNanos
     */
    public Map<String, Object> getBackendExecutionStats(DeviceType deviceType) {
        Map<String, Object> stats = new HashMap<>();
        long totalOps = 0;
        long totalTime = 0;

        for (DeviceStats ds : deviceStats.values()) {
            if (ds.device.getDeviceType() == deviceType) {
                totalOps += ds.opCount.sum();
                totalTime += ds.opTimeNanos.sum();
            }
        }

        stats.put("opCount", totalOps);
        stats.put("totalTimeNanos", totalTime);
        stats.put("avgTimeNanos", totalOps > 0 ? totalTime / totalOps : 0);
        stats.put("totalTimeMs", totalTime / 1_000_000.0);

        return stats;
    }

    /**
     * Get a summary of CPU vs GPU execution for multi-backend analysis.
     *
     * @return formatted string with backend execution breakdown
     */
    public String getMultiBackendSummary() {
        StringBuilder sb = new StringBuilder();
        sb.append("\n=== Multi-Backend Execution Summary ===\n");

        Map<String, Object> cpuStats = getBackendExecutionStats(DeviceType.CPU);
        Map<String, Object> gpuStats = getBackendExecutionStats(DeviceType.CUDA_GPU);

        long cpuOps = (Long) cpuStats.get("opCount");
        long gpuOps = (Long) gpuStats.get("opCount");
        long totalOps = cpuOps + gpuOps;

        sb.append(String.format("CPU Operations:  %d (%.1f%%)\n",
                cpuOps, totalOps > 0 ? 100.0 * cpuOps / totalOps : 0));
        sb.append(String.format("GPU Operations:  %d (%.1f%%)\n",
                gpuOps, totalOps > 0 ? 100.0 * gpuOps / totalOps : 0));
        sb.append(String.format("CPU Time:        %.2f ms\n", (Double) cpuStats.get("totalTimeMs")));
        sb.append(String.format("GPU Time:        %.2f ms\n", (Double) gpuStats.get("totalTimeMs")));
        sb.append(String.format("CPU Fallbacks:   %d (%.2f ms total)\n",
                getCpuFallbackCount(), getCpuFallbackTimeNanos() / 1_000_000.0));

        long totalTransferBytes = totalBytes.sum();
        double transferTimeMs = totalTimeNanos.sum() / 1_000_000.0;
        sb.append(String.format("Data Transfers:  %d (%s, %.2f ms)\n",
                totalTransfers.sum(), formatBytes(totalTransferBytes), transferTimeMs));

        double overheadPct = getOverheadPercent();
        sb.append(String.format("Transfer Overhead: %.1f%%\n", overheadPct));

        return sb.toString();
    }

    // =========================================================================
    // Logging
    // =========================================================================

    private void logTransfer(TransferRecord record) {
        double bandwidthGBps = record.getBandwidthGBps();
        double timeMs = record.timeNanos / 1_000_000.0;
        String sizeStr = formatBytes(record.bytes);

        String bandwidthNote = getBandwidthNote(record.source, record.target, bandwidthGBps);

        log.info("[TRANSFER] {} -> {} | {} | {:.2f} ms | {:.2f} GB/s{}",
            record.source.getDeviceId(),
            record.target.getDeviceId(),
            sizeStr,
            timeMs,
            bandwidthGBps,
            bandwidthNote);
    }

    private String getBandwidthNote(DeviceDescriptor source, DeviceDescriptor target, double bandwidthGBps) {
        // Provide context about expected bandwidth
        boolean gpuToGpu = source.getDeviceType() == DeviceType.CUDA_GPU &&
                          target.getDeviceType() == DeviceType.CUDA_GPU;
        boolean cpuToGpu = source.getDeviceType() == DeviceType.CPU &&
                          target.getDeviceType() == DeviceType.CUDA_GPU;
        boolean gpuToCpu = source.getDeviceType() == DeviceType.CUDA_GPU &&
                          target.getDeviceType() == DeviceType.CPU;

        if (gpuToGpu) {
            if (bandwidthGBps < 10) {
                return " (slow - check NVLink/P2P)";
            } else if (bandwidthGBps > 200) {
                return " (NVLink)";
            }
        } else if (cpuToGpu || gpuToCpu) {
            if (bandwidthGBps < 5) {
                return " (PCIe bottleneck - consider pinned memory)";
            } else if (bandwidthGBps < 12) {
                return " (PCIe Gen3)";
            } else if (bandwidthGBps < 25) {
                return " (PCIe Gen4)";
            } else {
                return " (PCIe Gen5)";
            }
        }

        return "";
    }

    private void checkWarningThreshold() {
        long execTime = totalExecutionTimeNanos.sum();
        long transferTime = totalTimeNanos.sum();

        if (execTime <= 0) {
            return;
        }

        double overheadPercent = 100.0 * transferTime / (execTime + transferTime);

        if (overheadPercent > warningThresholdPercent) {
            long now = System.currentTimeMillis();
            long lastWarn = lastWarningTime.get();
            if (now - lastWarn > WARNING_COOLDOWN_MS &&
                lastWarningTime.compareAndSet(lastWarn, now)) {
                log.warn("[TRANSFER OVERHEAD] {:.1f}% of execution time spent on transfers " +
                        "(threshold: {:.1f}%) | Total: {} in {} transfers",
                    overheadPercent,
                    warningThresholdPercent,
                    formatBytes(totalBytes.sum()),
                    totalTransfers.sum());
            }
        }
    }

    // =========================================================================
    // Summary and Statistics
    // =========================================================================

    /**
     * Log a comprehensive summary of transfer metrics.
     */
    public void logSummary() {
        long transfers = totalTransfers.sum();
        long bytes = totalBytes.sum();
        long timeNanos = totalTimeNanos.sum();
        long execTimeNanos = totalExecutionTimeNanos.sum();

        if (transfers == 0) {
            log.info("[TRANSFER SUMMARY] No transfers recorded");
            return;
        }

        double totalTimeMs = timeNanos / 1_000_000.0;
        double execTimeMs = execTimeNanos / 1_000_000.0;
        double overheadPercent = execTimeNanos > 0 ?
            100.0 * timeNanos / (execTimeNanos + timeNanos) : 0;
        double avgBandwidthGBps = timeNanos > 0 ?
            (bytes / 1e9) / (timeNanos / 1e9) : 0;

        log.info("========== TRANSFER METRICS SUMMARY ==========");
        log.info("Total transfers:     {}", transfers);
        log.info("Total data moved:    {}", formatBytes(bytes));
        log.info("Total transfer time: {:.2f} ms", totalTimeMs);
        log.info("Avg bandwidth:       {:.2f} GB/s", avgBandwidthGBps);
        log.info("Transfer overhead:   {:.1f}% of execution time", overheadPercent);
        log.info("");

        // Per-route breakdown
        if (!routeStats.isEmpty()) {
            log.info("--- Per-Route Statistics ---");
            routeStats.values().stream()
                .sorted((a, b) -> Long.compare(b.totalBytes.sum(), a.totalBytes.sum()))
                .forEach(route -> {
                    log.info("  {} -> {}: {} in {} transfers ({:.2f} GB/s avg)",
                        route.source.getDeviceId(),
                        route.target.getDeviceId(),
                        formatBytes(route.totalBytes.sum()),
                        route.transferCount.sum(),
                        route.getAverageBandwidthGBps());
                });
            log.info("");
        }

        // Per-device breakdown
        if (!deviceStats.isEmpty()) {
            log.info("--- Per-Device Statistics ---");
            deviceStats.values().forEach(device -> {
                log.info("  {}: {} sent, {} received, {} executed",
                    device.device.getDeviceId(),
                    formatBytes(device.bytesOut.sum()),
                    formatBytes(device.bytesIn.sum()),
                    device.opCount.sum() + " ops");
            });
        }

        log.info("==============================================");
    }

    /**
     * Get overhead percentage (transfer time / total time).
     */
    public double getOverheadPercent() {
        long execTime = totalExecutionTimeNanos.sum();
        long transferTime = totalTimeNanos.sum();
        if (execTime + transferTime <= 0) {
            return 0;
        }
        return 100.0 * transferTime / (execTime + transferTime);
    }

    /**
     * Get average bandwidth in GB/s.
     */
    public double getAverageBandwidthGBps() {
        long bytes = totalBytes.sum();
        long timeNanos = totalTimeNanos.sum();
        if (timeNanos <= 0) {
            return 0;
        }
        return (bytes / 1e9) / (timeNanos / 1e9);
    }

    /**
     * Get all statistics as a map.
     */
    public Map<String, Object> getStatistics() {
        Map<String, Object> stats = new LinkedHashMap<>();
        stats.put("totalTransfers", totalTransfers.sum());
        stats.put("totalBytes", totalBytes.sum());
        stats.put("totalTimeNanos", totalTimeNanos.sum());
        stats.put("totalTimeMs", totalTimeNanos.sum() / 1_000_000.0);
        stats.put("totalExecutionTimeNanos", totalExecutionTimeNanos.sum());
        stats.put("overheadPercent", getOverheadPercent());
        stats.put("averageBandwidthGBps", getAverageBandwidthGBps());

        // Route stats
        Map<String, Map<String, Object>> routes = new LinkedHashMap<>();
        routeStats.forEach((key, route) -> {
            Map<String, Object> routeMap = new LinkedHashMap<>();
            routeMap.put("transfers", route.transferCount.sum());
            routeMap.put("bytes", route.totalBytes.sum());
            routeMap.put("timeNanos", route.totalTimeNanos.sum());
            routeMap.put("avgBandwidthGBps", route.getAverageBandwidthGBps());
            routes.put(key, routeMap);
        });
        stats.put("routes", routes);

        return stats;
    }

    /**
     * Reset all metrics.
     */
    public void reset() {
        totalTransfers.reset();
        totalBytes.reset();
        totalTimeNanos.reset();
        totalExecutionTimeNanos.reset();
        cpuFallbackCount.reset();
        cpuFallbackTimeNanos.reset();
        routeStats.clear();
        deviceStats.clear();
        synchronized (recentLock) {
            recentTransfers.clear();
        }
    }

    // =========================================================================
    // Utility Methods
    // =========================================================================

    private static String formatBytes(long bytes) {
        return org.nd4j.common.util.ND4JFileUtils.formatBytes(bytes);
    }

    // =========================================================================
    // Inner Classes
    // =========================================================================

    /**
     * Record of a single transfer.
     */
    public static class TransferRecord {
        public final DeviceDescriptor source;
        public final DeviceDescriptor target;
        public final long bytes;
        public final long timeNanos;
        public final long timestampMs;

        TransferRecord(DeviceDescriptor source, DeviceDescriptor target,
                      long bytes, long timeNanos, long timestampMs) {
            this.source = source;
            this.target = target;
            this.bytes = bytes;
            this.timeNanos = timeNanos;
            this.timestampMs = timestampMs;
        }

        public double getBandwidthGBps() {
            if (timeNanos <= 0) return 0;
            return (bytes / 1e9) / (timeNanos / 1e9);
        }

        public double getTimeMs() {
            return timeNanos / 1_000_000.0;
        }
    }

    /**
     * Statistics for a specific transfer route (source -> target).
     */
    private static class RouteStats {
        final DeviceDescriptor source;
        final DeviceDescriptor target;
        final LongAdder transferCount = new LongAdder();
        final LongAdder totalBytes = new LongAdder();
        final LongAdder totalTimeNanos = new LongAdder();

        RouteStats(DeviceDescriptor source, DeviceDescriptor target) {
            this.source = source;
            this.target = target;
        }

        void record(long bytes, long timeNanos) {
            transferCount.increment();
            totalBytes.add(bytes);
            totalTimeNanos.add(timeNanos);
        }

        double getAverageBandwidthGBps() {
            long time = totalTimeNanos.sum();
            if (time <= 0) return 0;
            return (totalBytes.sum() / 1e9) / (time / 1e9);
        }
    }

    /**
     * Statistics for a specific device.
     */
    private static class DeviceStats {
        final DeviceDescriptor device;
        final LongAdder bytesIn = new LongAdder();
        final LongAdder bytesOut = new LongAdder();
        final LongAdder timeInNanos = new LongAdder();
        final LongAdder timeOutNanos = new LongAdder();
        final LongAdder opCount = new LongAdder();
        final LongAdder opTimeNanos = new LongAdder();

        DeviceStats(DeviceDescriptor device) {
            this.device = device;
        }

        void recordIncoming(long bytes, long timeNanos) {
            bytesIn.add(bytes);
            timeInNanos.add(timeNanos);
        }

        void recordOutgoing(long bytes, long timeNanos) {
            bytesOut.add(bytes);
            timeOutNanos.add(timeNanos);
        }

        void recordExecution(long timeNanos) {
            opCount.increment();
            opTimeNanos.add(timeNanos);
        }
    }
}
