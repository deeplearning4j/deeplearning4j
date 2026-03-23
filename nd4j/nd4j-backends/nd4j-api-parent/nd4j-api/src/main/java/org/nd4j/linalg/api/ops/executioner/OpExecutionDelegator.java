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
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.device.DeviceMemoryManager;
import org.nd4j.linalg.api.device.DeviceRoutingConfiguration;
import org.nd4j.linalg.api.device.DeviceType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.CustomOp;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Handles automatic delegation and data transfer for op execution across devices.
 *
 * This class provides transparent cross-device execution by:
 * - Automatically transferring input data to the target device
 * - Selecting optimal execution device based on input locations
 * - Caching transferred data for reuse
 * - Managing synchronization between devices
 * - Tracking detailed transfer metrics (timing, bandwidth, overhead)
 *
 * <p>The delegator integrates with {@link OpExecutioner} to intercept op execution
 * and ensure all inputs are on the appropriate device before execution.</p>
 *
 * <p>Transfer metrics are automatically tracked and logged. Use {@link TransferMetrics}
 * to access detailed statistics including per-transfer timing, bandwidth, and overhead
 * analysis.</p>
 *
 * <p>Example integration:</p>
 * <pre>{@code
 * // Before executing an op
 * OpExecutionDelegator delegator = OpExecutionDelegator.getInstance();
 * DeviceDescriptor targetDevice = delegator.prepareForExecution(op);
 *
 * // Execute on target device
 * executioner.exec(op, targetDevice);
 *
 * // After execution
 * delegator.postExecution(op, targetDevice);
 *
 * // View transfer metrics
 * TransferMetrics.getInstance().logSummary();
 * }</pre>
 *
 * Adam Gibson
 */
@Slf4j
public class OpExecutionDelegator {

    private static volatile OpExecutionDelegator INSTANCE;
    private static final Object LOCK = new Object();

    // Transfer cache: device -> (array hash -> cached copy)
    private final Map<String, Map<Integer, CachedArray>> transferCache = new ConcurrentHashMap<>();

    // Statistics
    @Getter
    private final AtomicLong totalTransfers = new AtomicLong(0);
    @Getter
    private final AtomicLong totalBytesTransferred = new AtomicLong(0);
    @Getter
    private final AtomicLong cacheHits = new AtomicLong(0);
    @Getter
    private final AtomicLong cacheMisses = new AtomicLong(0);

    // Async transfer executor
    private final ExecutorService transferExecutor;

    // Pending async transfers
    private final Map<Integer, CompletableFuture<Void>> pendingTransfers = new ConcurrentHashMap<>();

    private OpExecutionDelegator() {
        this.transferExecutor = Executors.newFixedThreadPool(2, r -> {
            Thread t = new Thread(r, "OpExecutionDelegator-Transfer");
            t.setDaemon(true);
            return t;
        });
    }

    /**
     * Get the singleton instance.
     */
    public static OpExecutionDelegator getInstance() {
        if (INSTANCE == null) {
            synchronized (LOCK) {
                if (INSTANCE == null) {
                    INSTANCE = new OpExecutionDelegator();
                }
            }
        }
        return INSTANCE;
    }

    // =========================================================================
    // Op Preparation
    // =========================================================================

    /**
     * Prepare an op for execution by ensuring all inputs are on the target device.
     *
     * @param op the op to prepare
     * @return the target device for execution
     */
    public DeviceDescriptor prepareForExecution(Op op) {
        DeviceRoutingConfiguration config = DeviceRoutingConfiguration.current();

        // Determine target device
        DeviceDescriptor targetDevice = selectTargetDevice(op, config);

        if (config.isLogRoutingDecisions()) {
            log.info("Op {} routed to device {}", op.opName(), targetDevice.getDeviceId());
        }

        // Transfer inputs to target device
        if (config.isAutoTransferEnabled()) {
            transferInputsToDevice(op, targetDevice, config);
        }

        return targetDevice;
    }

    /**
     * Prepare a custom op for execution.
     */
    public DeviceDescriptor prepareForExecution(CustomOp op) {
        DeviceRoutingConfiguration config = DeviceRoutingConfiguration.current();

        DeviceDescriptor targetDevice = selectTargetDevice(op, config);

        if (config.isLogRoutingDecisions()) {
            log.info("CustomOp {} routed to device {}", op.opName(), targetDevice.getDeviceId());
        }

        if (config.isAutoTransferEnabled()) {
            transferInputsToDevice(op, targetDevice, config);
        }

        return targetDevice;
    }

    /**
     * Select the target device for op execution.
     */
    private DeviceDescriptor selectTargetDevice(Op op, DeviceRoutingConfiguration config) {
        String opName = op.opName();

        // Get inputs
        INDArray x = op.x();
        INDArray y = op.y();
        INDArray z = op.z();
        List<INDArray> inputs = new ArrayList<>();
        if (x != null) inputs.add(x);
        if (y != null) inputs.add(y);
        if (z != null) inputs.add(z);

        DeviceDescriptor overrideDevice = resolveDeviceOverride(opName, config);
        DeviceDescriptor pinnedDevice = resolvePinnedDevice(inputs, opName);
        if (pinnedDevice != null) {
            if (config.getCpuOnlyOps().contains(opName) && pinnedDevice.getDeviceType() != DeviceType.CPU) {
                throw new IllegalStateException("Op " + opName + " is CPU-only but inputs are pinned to "
                        + pinnedDevice.getDeviceId());
            }
            if (overrideDevice != null && !overrideDevice.getDeviceId().equals(pinnedDevice.getDeviceId())) {
                throw new IllegalStateException("Op " + opName + " is pinned to " + pinnedDevice.getDeviceId()
                        + " but override targets " + overrideDevice.getDeviceId());
            }
            return pinnedDevice;
        }

        // Check for CPU-only ops
        if (config.getCpuOnlyOps().contains(opName)) {
            return DeviceDescriptor.cpu();
        }

        // Check for explicit device override
        if (overrideDevice != null) {
            return overrideDevice;
        }

        // Check if we should route by input location
        if (config.isRouteOpsByInputLocation()) {
            DeviceDescriptor inputDevice = determineInputDevice(x, y, z);
            if (inputDevice != null) {
                long inputSize = getInputSize(x, y, z);
                if (inputSize >= config.getMinSizeForDeviceRouting()) {
                    return inputDevice;
                }
            }
        }

        // Check for GPU-preferred ops
        if (config.getGpuPreferredOps().contains(opName)) {
            DeviceMemoryManager mgr = DeviceMemoryManager.getInstance();
            long size = getInputSize(x, y, z);
            return mgr.selectDevice(size, DeviceMemoryManager.DeviceRoutingPolicy.PREFER_GPU);
        }

        // Use default routing
        DeviceMemoryManager mgr = DeviceMemoryManager.getInstance();
        long size = getOutputSize(op);
        return mgr.selectDevice(size, config.getDefaultPolicy());
    }

    /**
     * Select target device for custom op.
     */
    private DeviceDescriptor selectTargetDevice(CustomOp op, DeviceRoutingConfiguration config) {
        String opName = op.opName();

        // Get inputs
        List<INDArray> inputs = op.inputArguments();
        DeviceDescriptor overrideDevice = resolveDeviceOverride(opName, config);
        DeviceDescriptor pinnedDevice = resolvePinnedDevice(inputs, opName);
        if (pinnedDevice != null) {
            if (config.getCpuOnlyOps().contains(opName) && pinnedDevice.getDeviceType() != DeviceType.CPU) {
                throw new IllegalStateException("CustomOp " + opName + " is CPU-only but inputs are pinned to "
                        + pinnedDevice.getDeviceId());
            }
            if (overrideDevice != null && !overrideDevice.getDeviceId().equals(pinnedDevice.getDeviceId())) {
                throw new IllegalStateException("CustomOp " + opName + " is pinned to " + pinnedDevice.getDeviceId()
                        + " but override targets " + overrideDevice.getDeviceId());
            }
            return pinnedDevice;
        }

        // Check for CPU-only ops
        if (config.getCpuOnlyOps().contains(opName)) {
            return DeviceDescriptor.cpu();
        }

        // Check for explicit device override
        if (overrideDevice != null) {
            return overrideDevice;
        }

        // Route by input location
        if (config.isRouteOpsByInputLocation() && !inputs.isEmpty()) {
            DeviceDescriptor inputDevice = determineInputDevice(inputs);
            if (inputDevice != null) {
                long inputSize = getInputSize(inputs);
                if (inputSize >= config.getMinSizeForDeviceRouting()) {
                    return inputDevice;
                }
            }
        }

        // Use default routing
        DeviceMemoryManager mgr = DeviceMemoryManager.getInstance();
        long size = getInputSize(inputs);
        return mgr.selectDevice(size, config.getDefaultPolicy());
    }

    /**
     * Select the target device for a list of arrays.
     * This is useful when preparing arrays outside of a full Op context.
     *
     * @param arrays the arrays to analyze
     * @return the device where most data resides, or the default device
     */
    public DeviceDescriptor selectTargetDeviceForArrays(List<INDArray> arrays) {
        if (arrays == null || arrays.isEmpty()) {
            return DeviceMemoryManager.getInstance().getDefaultDevice();
        }

        DeviceDescriptor pinnedDevice = resolvePinnedDevice(arrays, "array inputs");
        if (pinnedDevice != null) {
            return pinnedDevice;
        }

        DeviceDescriptor device = determineInputDevice(arrays);
        if (device != null) {
            return device;
        }

        return DeviceMemoryManager.getInstance().getDefaultDevice();
    }

    /**
     * Determine which device most inputs reside on.
     */
    private DeviceDescriptor determineInputDevice(INDArray... arrays) {
        Map<String, Long> deviceSizes = new HashMap<>();

        for (INDArray arr : arrays) {
            if (arr == null || arr.data() == null) continue;

            DeviceDescriptor device = getArrayDevice(arr);
            if (device != null) {
                deviceSizes.merge(device.getDeviceId(),
                        arr.length() * arr.data().getElementSize(),
                        Long::sum);
            }
        }

        if (deviceSizes.isEmpty()) {
            return null;
        }

        // Return device with most data
        return deviceSizes.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .map(e -> DeviceDescriptor.fromId(e.getKey()))
                .orElse(null);
    }

    private DeviceDescriptor determineInputDevice(List<INDArray> arrays) {
        return determineInputDevice(arrays.toArray(new INDArray[0]));
    }

    /**
     * Get the device an array currently resides on.
     */
    private DeviceDescriptor getArrayDevice(INDArray array) {
        if (array == null) return null;

        DataBuffer buffer = array.data();
        if (buffer != null && buffer.isHybrid()) {
            return buffer.asHybrid().getEffectiveDevice();
        }

        // Non-hybrid buffer - assume CPU
        return DeviceDescriptor.cpu();
    }

    // =========================================================================
    // Data Transfer
    // =========================================================================

    /**
     * Transfer op inputs to the target device.
     */
    private void transferInputsToDevice(Op op, DeviceDescriptor targetDevice,
                                        DeviceRoutingConfiguration config) {
        transferArrayToDevice(op.x(), targetDevice, config);
        transferArrayToDevice(op.y(), targetDevice, config);
        // Note: z (output) typically allocated fresh on target device
    }

    /**
     * Transfer custom op inputs to target device.
     */
    private void transferInputsToDevice(CustomOp op, DeviceDescriptor targetDevice,
                                        DeviceRoutingConfiguration config) {
        for (INDArray input : op.inputArguments()) {
            transferArrayToDevice(input, targetDevice, config);
        }
    }

    /**
     * Transfer a single array to the target device.
     * Records detailed transfer metrics including timing and bandwidth.
     */
    public void transferArrayToDevice(INDArray array, DeviceDescriptor targetDevice,
                                      DeviceRoutingConfiguration config) {
        if (array == null || targetDevice == null || array.data() == null) {
            return;
        }

        DeviceDescriptor pinnedDevice = getPinnedDevice(array);
        if (pinnedDevice != null && !pinnedDevice.getDeviceId().equals(targetDevice.getDeviceId())) {
            throw new IllegalStateException("Array is pinned to " + pinnedDevice.getDeviceId()
                    + " and cannot be transferred to " + targetDevice.getDeviceId());
        }

        // Check if already on target device
        DeviceDescriptor currentDevice = getArrayDevice(array);
        if (currentDevice != null && currentDevice.getDeviceId().equals(targetDevice.getDeviceId())) {
            return; // Already on target
        }

        // Check cache
        if (config.isCacheTransferredData()) {
            INDArray cached = getCachedCopy(array, targetDevice);
            if (cached != null) {
                cacheHits.incrementAndGet();
                // Array is already available on device - just ensure buffer is synced
                return;
            }
            cacheMisses.incrementAndGet();
        }

        long bytes = array.length() * array.data().getElementSize();

        // Perform timed transfer
        boolean useAsync = config.isAsyncTransferEnabled() &&
                           bytes >= config.getAsyncTransferThresholdBytes();

        long startNanos = System.nanoTime();
        if (useAsync) {
            transferAsync(array, targetDevice, config);
        } else {
            transferSync(array, targetDevice);
        }
        long endNanos = System.nanoTime();
        long transferTimeNanos = endNanos - startNanos;

        // Record metrics
        TransferMetrics metrics = TransferMetrics.getInstance();
        metrics.recordTransfer(currentDevice, targetDevice, bytes, transferTimeNanos);

        // Update local statistics (kept for backward compatibility)
        totalTransfers.incrementAndGet();
        totalBytesTransferred.addAndGet(bytes);
    }

    /**
     * Synchronous transfer to device.
     */
    private void transferSync(INDArray array, DeviceDescriptor targetDevice) {
        DataBuffer buffer = array.data();
        if (buffer != null && buffer.isHybrid()) {
            buffer.asHybrid().ensureAvailableOn(targetDevice);
        }
    }

    /**
     * Asynchronous transfer to device.
     */
    private void transferAsync(INDArray array, DeviceDescriptor targetDevice,
                               DeviceRoutingConfiguration config) {
        int arrayId = System.identityHashCode(array);

        // Check if transfer already pending
        CompletableFuture<Void> existing = pendingTransfers.get(arrayId);
        if (existing != null && !existing.isDone()) {
            // Wait for existing transfer
            try {
                existing.get(5, TimeUnit.SECONDS);
            } catch (Exception e) {
                log.warn("Async transfer wait failed, falling back to sync", e);
                transferSync(array, targetDevice);
            }
            return;
        }

        // Start async transfer
        CompletableFuture<Void> future = CompletableFuture.runAsync(() -> {
            DataBuffer buffer = array.data();
            if (buffer != null && buffer.isHybrid()) {
                buffer.asHybrid().prefetch(targetDevice);
            }
        }, transferExecutor);

        pendingTransfers.put(arrayId, future);

        // Wait for completion (with timeout)
        try {
            future.get(10, TimeUnit.SECONDS);
        } catch (TimeoutException e) {
            log.warn("Async transfer timed out for array, operation may be slow");
        } catch (Exception e) {
            log.warn("Async transfer failed", e);
            transferSync(array, targetDevice);
        } finally {
            pendingTransfers.remove(arrayId);
        }

        // Cache the transferred copy
        if (config.isCacheTransferredData()) {
            cacheArray(array, targetDevice, config);
        }
    }

    // =========================================================================
    // Transfer Cache
    // =========================================================================

    /**
     * Get a cached copy of an array on a device.
     */
    private INDArray getCachedCopy(INDArray array, DeviceDescriptor device) {
        Map<Integer, CachedArray> deviceCache = transferCache.get(device.getDeviceId());
        if (deviceCache == null) {
            return null;
        }

        int arrayId = System.identityHashCode(array);
        CachedArray cached = deviceCache.get(arrayId);

        if (cached == null) {
            return null;
        }

        // Check if source array has been modified
        if (cached.sourceVersion != getArrayVersion(array)) {
            deviceCache.remove(arrayId);
            return null;
        }

        cached.lastAccess = System.currentTimeMillis();
        return cached.copy;
    }

    /**
     * Cache an array copy on a device.
     */
    private void cacheArray(INDArray array, DeviceDescriptor device,
                            DeviceRoutingConfiguration config) {
        String deviceId = device.getDeviceId();
        Map<Integer, CachedArray> deviceCache =
                transferCache.computeIfAbsent(deviceId, k -> new ConcurrentHashMap<>());

        // Check cache size
        long currentSize = deviceCache.values().stream()
                .mapToLong(c -> c.bytes)
                .sum();

        if (currentSize >= config.getMaxCachePerDevice()) {
            evictOldestEntries(deviceCache, config.getMaxCachePerDevice() / 2);
        }

        int arrayId = System.identityHashCode(array);
        long bytes = array.length() * array.data().getElementSize();

        deviceCache.put(arrayId, new CachedArray(
                array,
                getArrayVersion(array),
                bytes,
                System.currentTimeMillis()
        ));
    }

    /**
     * Evict oldest cache entries.
     */
    private void evictOldestEntries(Map<Integer, CachedArray> cache, long targetSize) {
        List<Map.Entry<Integer, CachedArray>> entries = new ArrayList<>(cache.entrySet());
        entries.sort(Comparator.comparingLong(e -> e.getValue().lastAccess));

        long currentSize = entries.stream().mapToLong(e -> e.getValue().bytes).sum();

        for (Map.Entry<Integer, CachedArray> entry : entries) {
            if (currentSize <= targetSize) {
                break;
            }
            cache.remove(entry.getKey());
            currentSize -= entry.getValue().bytes;
        }
    }

    /**
     * Get a version identifier for an array (for cache invalidation).
     */
    private long getArrayVersion(INDArray array) {
        // Simple version based on data buffer state
        DataBuffer buffer = array.data();
        if (buffer != null) {
            return buffer.hashCode();
        }
        return 0;
    }

    /**
     * Clear the transfer cache for a device.
     */
    public void clearCache(DeviceDescriptor device) {
        transferCache.remove(device.getDeviceId());
    }

    /**
     * Clear all transfer caches.
     */
    public void clearAllCaches() {
        transferCache.clear();
    }

    // =========================================================================
    // Post-Execution
    // =========================================================================

    /**
     * Handle post-execution tasks like synchronization.
     */
    public void postExecution(Op op, DeviceDescriptor device) {
        DeviceRoutingConfiguration config = DeviceRoutingConfiguration.current();

        if (config.isSyncAfterOp()) {
            synchronizeDevice(device);
        }
    }

    /**
     * Handle post-execution for custom ops.
     */
    public void postExecution(CustomOp op, DeviceDescriptor device) {
        DeviceRoutingConfiguration config = DeviceRoutingConfiguration.current();

        if (config.isSyncAfterOp()) {
            synchronizeDevice(device);
        }
    }

    /**
     * Synchronize a device (wait for all pending operations).
     */
    public void synchronizeDevice(DeviceDescriptor device) {
        // Wait for any pending async transfers to this device
        for (CompletableFuture<Void> future : pendingTransfers.values()) {
            try {
                future.get(1, TimeUnit.SECONDS);
            } catch (Exception e) {
                // Ignore
            }
        }

        // Device-specific sync would go here (cudaDeviceSynchronize, etc.)
    }

    // =========================================================================
    // Utility Methods
    // =========================================================================

    private long getInputSize(INDArray... arrays) {
        long size = 0;
        for (INDArray arr : arrays) {
            if (arr != null) {
                size += arr.length() * arr.data().getElementSize();
            }
        }
        return size;
    }

    private long getInputSize(List<INDArray> arrays) {
        return getInputSize(arrays.toArray(new INDArray[0]));
    }

    private long getOutputSize(Op op) {
        INDArray z = op.z();
        if (z != null) {
            return z.length() * z.data().getElementSize();
        }
        // Estimate from input
        INDArray x = op.x();
        if (x != null) {
            return x.length() * x.data().getElementSize();
        }
        return 0;
    }

    private DeviceDescriptor resolveDeviceOverride(String opName, DeviceRoutingConfiguration config) {
        for (Map.Entry<String, DeviceDescriptor> entry : config.getOpDeviceOverrides().entrySet()) {
            if (opName.matches(entry.getKey()) || opName.equals(entry.getKey())) {
                return entry.getValue();
            }
        }
        return null;
    }

    private DeviceDescriptor resolvePinnedDevice(List<INDArray> arrays, String context) {
        if (arrays == null || arrays.isEmpty()) {
            return null;
        }

        DeviceDescriptor pinned = null;
        for (INDArray array : arrays) {
            if (array == null) {
                continue;
            }
            DeviceDescriptor arrayPinned = getPinnedDevice(array);
            if (arrayPinned == null) {
                continue;
            }
            if (pinned == null) {
                pinned = arrayPinned;
            } else if (!pinned.getDeviceId().equals(arrayPinned.getDeviceId())) {
                throw new IllegalStateException("Pinned device mismatch for " + context + ": "
                        + pinned.getDeviceId() + " vs " + arrayPinned.getDeviceId());
            }
        }
        return pinned;
    }

    private DeviceDescriptor getPinnedDevice(INDArray array) {
        if (array == null) {
            return null;
        }
        DataBuffer buffer = array.data();
        if (buffer != null && buffer.isHybrid()) {
            return buffer.asHybrid().getPinnedDevice();
        }
        return null;
    }

    // =========================================================================
    // Statistics
    // =========================================================================

    /**
     * Get transfer statistics.
     */
    public Map<String, Object> getStatistics() {
        Map<String, Object> stats = new LinkedHashMap<>();
        stats.put("totalTransfers", totalTransfers.get());
        stats.put("totalBytesTransferred", totalBytesTransferred.get());
        stats.put("cacheHits", cacheHits.get());
        stats.put("cacheMisses", cacheMisses.get());
        stats.put("cacheHitRate", getCacheHitRate());
        stats.put("pendingTransfers", pendingTransfers.size());

        long totalCacheSize = transferCache.values().stream()
                .flatMap(m -> m.values().stream())
                .mapToLong(c -> c.bytes)
                .sum();
        stats.put("totalCacheSize", totalCacheSize);

        return stats;
    }

    /**
     * Get cache hit rate.
     */
    public double getCacheHitRate() {
        long hits = cacheHits.get();
        long misses = cacheMisses.get();
        long total = hits + misses;
        return total > 0 ? (double) hits / total : 0;
    }

    /**
     * Reset statistics.
     */
    public void resetStatistics() {
        totalTransfers.set(0);
        totalBytesTransferred.set(0);
        cacheHits.set(0);
        cacheMisses.set(0);
    }

    /**
     * Log current statistics including detailed transfer metrics.
     */
    public void logStatistics() {
        log.info("=== Op Execution Delegator Statistics ===");
        log.info("Total transfers: {}", totalTransfers.get());
        log.info("Total bytes transferred: {} MB", totalBytesTransferred.get() / (1024 * 1024));
        log.info("Cache hit rate: {:.1f}%", getCacheHitRate() * 100);
        log.info("Pending transfers: {}", pendingTransfers.size());

        // Also log detailed transfer metrics
        TransferMetrics.getInstance().logSummary();
    }

    /**
     * Get the transfer metrics tracker.
     * Use this for detailed timing and bandwidth analysis.
     *
     * @return the TransferMetrics singleton
     */
    public TransferMetrics getTransferMetrics() {
        return TransferMetrics.getInstance();
    }

    // =========================================================================
    // Shutdown
    // =========================================================================

    /**
     * Shutdown the delegator and release resources.
     */
    public void shutdown() {
        transferExecutor.shutdown();
        try {
            if (!transferExecutor.awaitTermination(5, TimeUnit.SECONDS)) {
                transferExecutor.shutdownNow();
            }
        } catch (InterruptedException e) {
            transferExecutor.shutdownNow();
            Thread.currentThread().interrupt();
        }

        clearAllCaches();
        pendingTransfers.clear();
    }

    // =========================================================================
    // Inner Classes
    // =========================================================================

    /**
     * Cached array entry.
     */
    private static class CachedArray {
        final INDArray copy;
        final long sourceVersion;
        final long bytes;
        long lastAccess;

        CachedArray(INDArray copy, long sourceVersion, long bytes, long lastAccess) {
            this.copy = copy;
            this.sourceVersion = sourceVersion;
            this.bytes = bytes;
            this.lastAccess = lastAccess;
        }
    }
}
