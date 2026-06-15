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

package org.nd4j.autodiff.samediff.internal.memory;

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.IdentityHashMap;

/**
 * Reusable instrumentation for intermediate DataBuffer cleanup.
 * Snapshots buffer state and native pool state before cleanup, accumulates
 * per-buffer results during cleanup, and logs a comprehensive before/after
 * summary at INFO level.
 *
 * Usage:
 * <pre>
 *   CleanupDiagnostics diag = CleanupDiagnostics.snapshot(uniqueBuffers, protectedBuffers);
 *   CleanupDiagnostics.CleanupResult result = diag.newResult(execStream);
 *   // ... for each buffer in cleanup loop ...
 *   result.recordNativeFreed(bufBytes);  // or result.recordOdbNull(); etc.
 *   // ... after loop ...
 *   diag.logResults(result, "no-cache");
 * </pre>
 */
@Slf4j
public class CleanupDiagnostics {

    private final int totalBuffers;
    private final int closeableCount;
    private final int nonCloseableCount;
    private final int constantPoisonedCount;
    private final long totalBytes;
    private final long physicalBytesBefore;
    private final long startNanos;
    // Per-device native pool stats before cleanup (parallel arrays: device 0..nDevices-1)
    private final int nDevices;
    private final long[] poolUsedBefore;   // bytes used in native pool per device
    private final long[] poolReservedBefore; // bytes reserved in native pool per device

    private CleanupDiagnostics(int totalBuffers, int closeableCount, int nonCloseableCount,
                               int constantPoisonedCount, long totalBytes, long physicalBytesBefore,
                               int nDevices, long[] poolUsedBefore, long[] poolReservedBefore) {
        this.totalBuffers = totalBuffers;
        this.closeableCount = closeableCount;
        this.nonCloseableCount = nonCloseableCount;
        this.constantPoisonedCount = constantPoisonedCount;
        this.totalBytes = totalBytes;
        this.physicalBytesBefore = physicalBytesBefore;
        this.startNanos = System.nanoTime();
        this.nDevices = nDevices;
        this.poolUsedBefore = poolUsedBefore;
        this.poolReservedBefore = poolReservedBefore;
    }

    /**
     * Snapshot the state of intermediate DataBuffers and native memory pools before cleanup.
     *
     * @param intermediateBuffers unique intermediate DataBuffers to be cleaned up
     * @param protectedBuffers    buffers that will be skipped during cleanup (may be null)
     * @return snapshot for later use with {@link #logResults}
     */
    public static CleanupDiagnostics snapshot(IdentityHashMap<DataBuffer, Boolean> intermediateBuffers,
                                              IdentityHashMap<DataBuffer, Boolean> protectedBuffers) {
        int total = 0, closeable = 0, nonCloseable = 0, constantPoisoned = 0;
        long totalBytes = 0;

        for (DataBuffer buf : intermediateBuffers.keySet()) {
            if (protectedBuffers != null && protectedBuffers.containsKey(buf)) continue;
            if (buf.wasClosed()) continue;
            total++;
            long bytes = buf.length() * buf.getElementSize();
            totalBytes += bytes;
            if (buf.closeable()) {
                closeable++;
            } else if (buf.isConstant() && !buf.isAttached()) {
                constantPoisoned++;
            } else {
                nonCloseable++;
            }
        }

        long physicalBytes = Pointer.physicalBytes();

        // Snapshot native pool stats per device
        int nDevices = 0;
        long[] poolUsed = null;
        long[] poolReserved = null;
        try {
            nDevices = Nd4j.getAffinityManager().getNumberOfDevices();
            if (nDevices > 0) {
                poolUsed = new long[nDevices];
                poolReserved = new long[nDevices];
                NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
                for (int d = 0; d < nDevices; d++) {
                    try (LongPointer usedPtr = new LongPointer(1);
                         LongPointer reservedPtr = new LongPointer(1)) {
                        nativeOps.getMemoryPoolStats(d, usedPtr, reservedPtr);
                        poolUsed[d] = usedPtr.get(0);
                        poolReserved[d] = reservedPtr.get(0);
                    } catch (Exception e) {
                        // Pool stats not available (CPU backend, etc.) - leave as 0
                    }
                }
            }
        } catch (Exception e) {
            // nDevices query failed - leave pool stats empty
        }

        return new CleanupDiagnostics(total, closeable, nonCloseable, constantPoisoned,
                totalBytes, physicalBytes, nDevices,
                poolUsed != null ? poolUsed : new long[0],
                poolReserved != null ? poolReserved : new long[0]);
    }

    /**
     * Create a new CleanupResult to accumulate per-buffer outcomes during the cleanup loop.
     *
     * @param execStream the execution stream pointer (null for CPU-only)
     * @return a fresh CleanupResult
     */
    public CleanupResult newResult(Pointer execStream) {
        return new CleanupResult(execStream);
    }

    /**
     * Log before/after cleanup summary at INFO level.
     * Kept for backward compatibility - delegates to the full version with an empty result.
     *
     * @param closedCount      number of buffers closed normally
     * @param forceClosedCount number of constant-poisoned buffers force-closed
     * @param label            label for the cleanup path (e.g., "cache-enabled" or "no-cache")
     */
    public void logResults(int closedCount, int forceClosedCount, String label) {
        CleanupResult result = new CleanupResult(null);
        result.nativeFreedCount = closedCount;
        result.forceClosedConstant = forceClosedCount;
        logResults(result, label);
    }

    /**
     * Log comprehensive before/after cleanup summary at INFO level.
     * Produces a single dense log line with all diagnostic data.
     *
     * @param result accumulated cleanup results from the buffer loop
     * @param label  label for the cleanup path (e.g., "cache-enabled" or "no-cache")
     */
    public void logResults(CleanupResult result, String label) {
        long physicalBytesAfter = Pointer.physicalBytes();
        long elapsedMs = (System.nanoTime() - startNanos) / 1_000_000;
        long physDeltaMB = (physicalBytesBefore - physicalBytesAfter) / (1024 * 1024);

        // Snapshot native pool stats after cleanup
        StringBuilder poolDelta = new StringBuilder();
        if (nDevices > 0 && poolUsedBefore.length > 0) {
            try {
                NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
                for (int d = 0; d < nDevices; d++) {
                    try (LongPointer usedPtr = new LongPointer(1);
                         LongPointer reservedPtr = new LongPointer(1)) {
                        nativeOps.getMemoryPoolStats(d, usedPtr, reservedPtr);
                        long usedAfter = usedPtr.get(0);
                        long reservedAfter = reservedPtr.get(0);
                        long usedDelta = poolUsedBefore[d] - usedAfter;
                        long reservedDelta = poolReservedBefore[d] - reservedAfter;
                        if (poolDelta.length() > 0) poolDelta.append(", ");
                        poolDelta.append(String.format("dev%d: used %+dMB (now %dMB), rsv %+dMB (now %dMB)",
                                d, usedDelta / (1024 * 1024), usedAfter / (1024 * 1024),
                                reservedDelta / (1024 * 1024), reservedAfter / (1024 * 1024)));
                    } catch (Exception e) {
                        // skip device
                    }
                }
            } catch (Exception e) {
                poolDelta.append("unavailable");
            }
        }

        // DeviceMemoryManager tracked allocation delta
        // (This tracks Java-side allocation tracking, separate from native pool)
        String devMgrInfo = "";
        try {
            var devMgr = org.nd4j.linalg.api.device.DeviceMemoryManager.getInstance();
            var stats = devMgr.getMemoryStats();
            if (!stats.isEmpty()) {
                StringBuilder sb = new StringBuilder();
                for (var entry : stats.entrySet()) {
                    if (sb.length() > 0) sb.append(", ");
                    sb.append(String.format("%s: alloc=%dMB avail=%dMB",
                            entry.getKey(),
                            entry.getValue().allocated / (1024 * 1024),
                            entry.getValue().available / (1024 * 1024)));
                }
                devMgrInfo = sb.toString();
            }
        } catch (Exception e) {
            // DeviceMemoryManager not available
        }

        // GPU free memory per device (actual CUDA free, not pool stats)
        String gpuFreeInfo = "";
        if (nDevices > 0) {
            try {
                NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
                StringBuilder sb = new StringBuilder();
                for (int d = 0; d < nDevices; d++) {
                    try {
                        long freeMem = nativeOps.getDeviceFreeMemory(d);
                        long totalMem = nativeOps.getDeviceTotalMemory(d);
                        if (sb.length() > 0) sb.append(", ");
                        sb.append(String.format("dev%d: %dMB/%dMB free",
                                d, freeMem / (1024 * 1024), totalMem / (1024 * 1024)));
                    } catch (Exception e) {
                        // skip
                    }
                }
                gpuFreeInfo = sb.toString();
            } catch (Exception e) {
                // skip
            }
        }

        String streamAddr = result.execStream != null
                ? String.format("0x%x", result.execStream.address())
                : "null";

        int totalFreed = result.nativeFreedCount + result.odbNullCount + result.javaFallbackCount;

        log.info("[{}] Cleanup: PRE: {} bufs (closeable={}, non-close={}, const-poisoned={}), {}MB logical" +
                        "  FREED: native={} ({}MB), odbNull={}, javaFallback={}, constUnpoisoned={}, total={}" +
                        "  MEMORY: physDelta={}MB, elapsed={}ms, stream={}, devices={}" +
                        (poolDelta.length() > 0 ? "  POOL: [" + poolDelta + "]" : "") +
                        (!gpuFreeInfo.isEmpty() ? "  GPU: [" + gpuFreeInfo + "]" : "") +
                        (!devMgrInfo.isEmpty() ? "  DEVMGR: [" + devMgrInfo + "]" : ""),
                label, totalBuffers, closeableCount, nonCloseableCount, constantPoisonedCount,
                totalBytes / (1024 * 1024),
                result.nativeFreedCount, result.nativeFreedBytes / (1024 * 1024),
                result.odbNullCount, result.javaFallbackCount, result.forceClosedConstant, totalFreed,
                physDeltaMB, elapsedMs, streamAddr, nDevices);
    }

    public int getTotalBuffers() { return totalBuffers; }
    public int getCloseableCount() { return closeableCount; }
    public int getConstantPoisonedCount() { return constantPoisonedCount; }
    public long getTotalBytes() { return totalBytes; }
    public long getPhysicalBytesBefore() { return physicalBytesBefore; }

    /**
     * Accumulator for per-buffer cleanup outcomes.
     * Create via {@link CleanupDiagnostics#newResult(Pointer)} and
     * call the record methods inside the cleanup loop.
     */
    public static class CleanupResult {
        private final Pointer execStream;
        private int nativeFreedCount;
        private long nativeFreedBytes;
        private int odbNullCount;
        private int javaFallbackCount;
        private int forceClosedConstant;

        CleanupResult(Pointer execStream) {
            this.execStream = execStream;
        }

        /** A buffer was freed via dbFreeBuffersOnStream / dbFreeBuffersOnly. */
        public void recordNativeFreed(long bytes) {
            nativeFreedCount++;
            nativeFreedBytes += bytes;
        }

        /** The buffer's OpaqueDataBuffer was null/isNull - fell back to buf.close(). */
        public void recordOdbNull() {
            odbNullCount++;
        }

        /** Native free failed, fell back to buf.close(). */
        public void recordJavaFallback() {
            javaFallbackCount++;
        }

        /** A constant-poisoned buffer was un-poisoned before freeing. */
        public void recordConstantUnpoisoned() {
            forceClosedConstant++;
        }

        public int getNativeFreedCount() { return nativeFreedCount; }
        public long getNativeFreedBytes() { return nativeFreedBytes; }
        public int getOdbNullCount() { return odbNullCount; }
        public int getJavaFallbackCount() { return javaFallbackCount; }
        public int getForceClosedConstant() { return forceClosedConstant; }
    }
}
