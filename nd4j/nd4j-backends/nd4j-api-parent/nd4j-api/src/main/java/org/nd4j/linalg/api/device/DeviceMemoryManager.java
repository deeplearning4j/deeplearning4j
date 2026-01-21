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

package org.nd4j.linalg.api.device;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Manages device memory allocation, tracking, and capping across multiple devices.
 *
 * This singleton tracks memory usage per device and enforces configurable memory caps.
 * It also provides device selection based on available memory and priorities.
 *
 * <p>Features:</p>
 * <ul>
 *   <li>Per-device memory tracking with atomic updates</li>
 *   <li>Configurable memory caps per device</li>
 *   <li>Device prioritization for allocation routing</li>
 *   <li>Automatic fallback when devices are full</li>
 *   <li>Memory pressure callbacks for eviction</li>
 * </ul>
 *
 * <p>Example usage:</p>
 * <pre>{@code
 * DeviceMemoryManager mgr = DeviceMemoryManager.getInstance();
 *
 * // Set memory cap for GPU 0 to 8GB
 * mgr.setMemoryCap(DeviceDescriptor.cuda(0), 8L * 1024 * 1024 * 1024);
 *
 * // Set device priorities (higher = preferred)
 * mgr.setDevicePriority(DeviceDescriptor.cuda(0), 100);
 * mgr.setDevicePriority(DeviceDescriptor.cuda(1), 90);
 * mgr.setDevicePriority(DeviceDescriptor.cpu(), 10);
 *
 * // Get best device for allocation
 * DeviceDescriptor device = mgr.selectDeviceForAllocation(1024 * 1024);
 * }</pre>
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@Slf4j
public class DeviceMemoryManager {

    private static volatile DeviceMemoryManager INSTANCE;
    private static final Object LOCK = new Object();

    // Memory tracking per device (by device ID)
    private final Map<String, AtomicLong> allocatedMemory = new ConcurrentHashMap<>();
    private final Map<String, AtomicLong> peakMemory = new ConcurrentHashMap<>();

    // Memory caps per device (0 = unlimited)
    private final Map<String, Long> memoryCaps = new ConcurrentHashMap<>();

    // Device priorities (higher = preferred)
    private final Map<String, Integer> devicePriorities = new ConcurrentHashMap<>();

    // Registered devices
    private final Map<String, DeviceDescriptor> registeredDevices = new ConcurrentHashMap<>();

    // Default device for allocations
    @Getter
    private volatile DeviceDescriptor defaultDevice;

    // Fallback device when primary is full
    @Getter
    private volatile DeviceDescriptor fallbackDevice;

    // Configuration
    @Getter
    private volatile boolean autoFallbackEnabled = true;

    @Getter
    private volatile double memoryPressureThreshold = 0.9; // 90% triggers pressure

    // Callbacks for memory pressure
    private final List<MemoryPressureCallback> pressureCallbacks = new ArrayList<>();
    private final ReentrantReadWriteLock callbackLock = new ReentrantReadWriteLock();

    private DeviceMemoryManager() {
        // Initialize with CPU as default
        defaultDevice = DeviceDescriptor.cpu();
        fallbackDevice = DeviceDescriptor.cpu();
        registerDevice(defaultDevice);
    }

    /**
     * Get the singleton instance.
     */
    public static DeviceMemoryManager getInstance() {
        if (INSTANCE == null) {
            synchronized (LOCK) {
                if (INSTANCE == null) {
                    INSTANCE = new DeviceMemoryManager();
                }
            }
        }
        return INSTANCE;
    }

    // =========================================================================
    // Device Registration
    // =========================================================================

    /**
     * Register a device with the memory manager.
     *
     * @param device the device to register
     */
    public void registerDevice(DeviceDescriptor device) {
        String id = device.getDeviceId();
        registeredDevices.put(id, device);
        allocatedMemory.putIfAbsent(id, new AtomicLong(0));
        peakMemory.putIfAbsent(id, new AtomicLong(0));

        // Set default priority based on device type
        if (!devicePriorities.containsKey(id)) {
            int priority = getDefaultPriority(device);
            devicePriorities.put(id, priority);
        }

        log.debug("Registered device: {} with priority {}", id, devicePriorities.get(id));
    }

    /**
     * Get default priority for a device type.
     */
    private int getDefaultPriority(DeviceDescriptor device) {
        switch (device.getDeviceType()) {
            case CUDA_GPU:
                return 100 - device.getDeviceIndex(); // GPU 0 = 100, GPU 1 = 99, etc.
            case ROCM_GPU:
            case METAL_GPU:
                return 80 - device.getDeviceIndex();
            case TPU:
                return 110; // TPU highest priority
            case CPU:
            default:
                return 10; // CPU lowest priority
        }
    }

    /**
     * Get all registered devices.
     */
    public Collection<DeviceDescriptor> getRegisteredDevices() {
        return Collections.unmodifiableCollection(registeredDevices.values());
    }

    /**
     * Check if a device is registered.
     *
     * @param device the device to check
     * @return true if registered
     */
    public boolean isDeviceRegistered(DeviceDescriptor device) {
        return registeredDevices.containsKey(device.getDeviceId());
    }

    /**
     * Get the number of registered devices.
     *
     * @return count of registered devices
     */
    public int getRegisteredDeviceCount() {
        return registeredDevices.size();
    }

    /**
     * Clear all registered devices and reset state.
     * Primarily for testing purposes.
     */
    public void clearDevices() {
        registeredDevices.clear();
        allocatedMemory.clear();
        peakMemory.clear();
        memoryCaps.clear();
        devicePriorities.clear();

        // Re-register default CPU device
        defaultDevice = DeviceDescriptor.cpu();
        fallbackDevice = DeviceDescriptor.cpu();
        registerDevice(defaultDevice);
    }

    // =========================================================================
    // Memory Caps and Limits
    // =========================================================================

    /**
     * Set a memory cap for a device.
     *
     * @param device the device
     * @param maxBytes maximum bytes allowed (0 = unlimited)
     */
    public void setMemoryCap(DeviceDescriptor device, long maxBytes) {
        registerDevice(device);
        memoryCaps.put(device.getDeviceId(), maxBytes);
        log.info("Set memory cap for {}: {} bytes", device.getDeviceId(), maxBytes);
    }

    /**
     * Set a memory cap as a fraction of total device memory.
     *
     * @param device the device
     * @param fraction fraction of total memory (0.0 to 1.0)
     */
    public void setMemoryCapFraction(DeviceDescriptor device, double fraction) {
        if (fraction < 0 || fraction > 1) {
            throw new IllegalArgumentException("Fraction must be between 0 and 1");
        }
        long maxBytes = (long) (device.getTotalMemory() * fraction);
        setMemoryCap(device, maxBytes);
    }

    /**
     * Get the memory cap for a device.
     *
     * @param device the device
     * @return memory cap in bytes (0 = unlimited)
     */
    public long getMemoryCap(DeviceDescriptor device) {
        return memoryCaps.getOrDefault(device.getDeviceId(), 0L);
    }

    /**
     * Get effective available memory for a device (considering cap).
     *
     * @param device the device
     * @return available bytes
     */
    public long getAvailableMemory(DeviceDescriptor device) {
        String id = device.getDeviceId();
        long allocated = allocatedMemory.getOrDefault(id, new AtomicLong(0)).get();
        long cap = memoryCaps.getOrDefault(id, 0L);

        if (cap > 0) {
            // Capped: return remaining under cap
            return Math.max(0, cap - allocated);
        } else {
            // Unlimited: return device's available memory minus our allocations
            return Math.max(0, device.getAvailableMemory() - allocated);
        }
    }

    /**
     * Check if a device can accommodate an allocation.
     *
     * @param device the device
     * @param bytes bytes to allocate
     * @return true if allocation would fit
     */
    public boolean canAllocate(DeviceDescriptor device, long bytes) {
        return getAvailableMemory(device) >= bytes;
    }

    // =========================================================================
    // Device Priority
    // =========================================================================

    /**
     * Set priority for a device (higher = preferred for allocation).
     *
     * @param device the device
     * @param priority priority value (0-1000 typical range)
     */
    public void setDevicePriority(DeviceDescriptor device, int priority) {
        registerDevice(device);
        devicePriorities.put(device.getDeviceId(), priority);
    }

    /**
     * Get priority for a device.
     */
    public int getDevicePriority(DeviceDescriptor device) {
        return devicePriorities.getOrDefault(device.getDeviceId(), 0);
    }

    /**
     * Set the default device for allocations.
     *
     * @param device the default device
     */
    public void setDefaultDevice(DeviceDescriptor device) {
        registerDevice(device);
        this.defaultDevice = device;
        log.info("Default device set to: {}", device.getDeviceId());
    }

    /**
     * Set the fallback device when primary devices are full.
     *
     * @param device the fallback device
     */
    public void setFallbackDevice(DeviceDescriptor device) {
        registerDevice(device);
        this.fallbackDevice = device;
    }

    /**
     * Get the default device for allocations.
     *
     * @return the default device
     */
    public DeviceDescriptor getDefaultDevice() {
        return defaultDevice;
    }

    /**
     * Get the fallback device.
     *
     * @return the fallback device
     */
    public DeviceDescriptor getFallbackDevice() {
        return fallbackDevice;
    }

    /**
     * Enable or disable automatic fallback to other devices.
     */
    public void setAutoFallbackEnabled(boolean enabled) {
        this.autoFallbackEnabled = enabled;
    }

    // =========================================================================
    // Device Selection
    // =========================================================================

    /**
     * Select the best device for an allocation of the given size.
     *
     * Selection criteria (in order):
     * 1. Has enough available memory
     * 2. Highest priority
     * 3. Most available memory (tiebreaker)
     *
     * @param bytes allocation size
     * @return best device, or fallback if none suitable
     */
    public DeviceDescriptor selectDeviceForAllocation(long bytes) {
        // First try default device
        if (canAllocate(defaultDevice, bytes)) {
            return defaultDevice;
        }

        // Find best alternative
        DeviceDescriptor best = null;
        int bestPriority = Integer.MIN_VALUE;
        long bestAvailable = 0;

        for (DeviceDescriptor device : registeredDevices.values()) {
            if (!canAllocate(device, bytes)) {
                continue;
            }

            int priority = getDevicePriority(device);
            long available = getAvailableMemory(device);

            if (priority > bestPriority ||
                (priority == bestPriority && available > bestAvailable)) {
                best = device;
                bestPriority = priority;
                bestAvailable = available;
            }
        }

        if (best != null) {
            return best;
        }

        // No device has space - use fallback if enabled
        if (autoFallbackEnabled && fallbackDevice != null) {
            log.warn("No device has {} bytes available, using fallback: {}",
                    bytes, fallbackDevice.getDeviceId());
            return fallbackDevice;
        }

        // Last resort: return default and let allocation fail naturally
        log.error("No device can accommodate {} bytes allocation", bytes);
        return defaultDevice;
    }

    /**
     * Select a device based on a routing policy.
     *
     * @param bytes allocation size
     * @param policy the routing policy
     * @return selected device
     */
    public DeviceDescriptor selectDevice(long bytes, DeviceRoutingPolicy policy) {
        switch (policy) {
            case PREFER_GPU:
                return selectPreferringType(bytes, DeviceType.CUDA_GPU, DeviceType.METAL_GPU);
            case PREFER_CPU:
                return selectPreferringType(bytes, DeviceType.CPU);
            case ROUND_ROBIN:
                return selectRoundRobin(bytes);
            case LEAST_LOADED:
                return selectLeastLoaded(bytes);
            case MEMORY_PRIORITY:
            default:
                return selectDeviceForAllocation(bytes);
        }
    }

    private DeviceDescriptor selectPreferringType(long bytes, DeviceType... preferredTypes) {
        Set<DeviceType> preferred = new HashSet<>(Arrays.asList(preferredTypes));

        // First pass: preferred types only
        for (DeviceDescriptor device : registeredDevices.values()) {
            if (preferred.contains(device.getDeviceType()) && canAllocate(device, bytes)) {
                return device;
            }
        }

        // Fall back to any device
        return selectDeviceForAllocation(bytes);
    }

    private int roundRobinIndex = 0;

    private synchronized DeviceDescriptor selectRoundRobin(long bytes) {
        List<DeviceDescriptor> devices = new ArrayList<>(registeredDevices.values());
        if (devices.isEmpty()) return defaultDevice;

        int attempts = devices.size();
        while (attempts-- > 0) {
            roundRobinIndex = (roundRobinIndex + 1) % devices.size();
            DeviceDescriptor device = devices.get(roundRobinIndex);
            if (canAllocate(device, bytes)) {
                return device;
            }
        }
        return defaultDevice;
    }

    private DeviceDescriptor selectLeastLoaded(long bytes) {
        DeviceDescriptor best = null;
        double lowestLoad = Double.MAX_VALUE;

        for (DeviceDescriptor device : registeredDevices.values()) {
            if (!canAllocate(device, bytes)) continue;

            long total = device.getTotalMemory();
            long allocated = getAllocatedMemory(device);
            double load = total > 0 ? (double) allocated / total : 1.0;

            if (load < lowestLoad) {
                lowestLoad = load;
                best = device;
            }
        }

        return best != null ? best : defaultDevice;
    }

    // =========================================================================
    // Memory Tracking
    // =========================================================================

    /**
     * Record a memory allocation on a device.
     *
     * @param device the device
     * @param bytes bytes allocated
     */
    public void recordAllocation(DeviceDescriptor device, long bytes) {
        String id = device.getDeviceId();
        allocatedMemory.computeIfAbsent(id, k -> new AtomicLong(0));

        long newTotal = allocatedMemory.get(id).addAndGet(bytes);

        // Update peak
        peakMemory.computeIfAbsent(id, k -> new AtomicLong(0));
        peakMemory.get(id).updateAndGet(peak -> Math.max(peak, newTotal));

        // Check memory pressure
        checkMemoryPressure(device);
    }

    /**
     * Record a memory deallocation on a device.
     *
     * @param device the device
     * @param bytes bytes deallocated
     */
    public void recordDeallocation(DeviceDescriptor device, long bytes) {
        String id = device.getDeviceId();
        AtomicLong allocated = allocatedMemory.get(id);
        if (allocated != null) {
            allocated.addAndGet(-bytes);
        }
    }

    /**
     * Get currently allocated memory on a device.
     */
    public long getAllocatedMemory(DeviceDescriptor device) {
        AtomicLong allocated = allocatedMemory.get(device.getDeviceId());
        return allocated != null ? allocated.get() : 0;
    }

    /**
     * Get peak memory usage on a device.
     */
    public long getPeakMemory(DeviceDescriptor device) {
        AtomicLong peak = peakMemory.get(device.getDeviceId());
        return peak != null ? peak.get() : 0;
    }

    /**
     * Get memory utilization ratio for a device.
     *
     * @param device the device
     * @return utilization (0.0 to 1.0)
     */
    public double getMemoryUtilization(DeviceDescriptor device) {
        long cap = getMemoryCap(device);
        long total = cap > 0 ? cap : device.getTotalMemory();
        if (total <= 0) return 0;
        return (double) getAllocatedMemory(device) / total;
    }

    /**
     * Reset peak memory tracking.
     */
    public void resetPeakMemory() {
        peakMemory.values().forEach(v -> v.set(0));
    }

    // =========================================================================
    // Memory Pressure
    // =========================================================================

    /**
     * Set memory pressure threshold (0.0 to 1.0).
     * When utilization exceeds this, pressure callbacks are triggered.
     */
    public void setMemoryPressureThreshold(double threshold) {
        this.memoryPressureThreshold = Math.max(0, Math.min(1, threshold));
    }

    /**
     * Register a callback for memory pressure events.
     */
    public void addMemoryPressureCallback(MemoryPressureCallback callback) {
        callbackLock.writeLock().lock();
        try {
            pressureCallbacks.add(callback);
        } finally {
            callbackLock.writeLock().unlock();
        }
    }

    /**
     * Remove a memory pressure callback.
     */
    public void removeMemoryPressureCallback(MemoryPressureCallback callback) {
        callbackLock.writeLock().lock();
        try {
            pressureCallbacks.remove(callback);
        } finally {
            callbackLock.writeLock().unlock();
        }
    }

    private void checkMemoryPressure(DeviceDescriptor device) {
        double utilization = getMemoryUtilization(device);
        if (utilization >= memoryPressureThreshold) {
            notifyMemoryPressure(device, utilization);
        }
    }

    private void notifyMemoryPressure(DeviceDescriptor device, double utilization) {
        callbackLock.readLock().lock();
        try {
            for (MemoryPressureCallback callback : pressureCallbacks) {
                try {
                    callback.onMemoryPressure(device, utilization);
                } catch (Exception e) {
                    log.warn("Memory pressure callback failed", e);
                }
            }
        } finally {
            callbackLock.readLock().unlock();
        }
    }

    // =========================================================================
    // Statistics
    // =========================================================================

    /**
     * Get memory statistics for all devices.
     */
    public Map<String, DeviceMemoryStats> getMemoryStats() {
        Map<String, DeviceMemoryStats> stats = new LinkedHashMap<>();
        for (DeviceDescriptor device : registeredDevices.values()) {
            stats.put(device.getDeviceId(), getMemoryStats(device));
        }
        return stats;
    }

    /**
     * Get memory statistics for a specific device.
     */
    public DeviceMemoryStats getMemoryStats(DeviceDescriptor device) {
        return new DeviceMemoryStats(
                device.getDeviceId(),
                device.getTotalMemory(),
                getAllocatedMemory(device),
                getAvailableMemory(device),
                getPeakMemory(device),
                getMemoryCap(device),
                getMemoryUtilization(device)
        );
    }

    /**
     * Log memory statistics for all devices.
     */
    public void logMemoryStats() {
        log.info("=== Device Memory Statistics ===");
        for (DeviceDescriptor device : registeredDevices.values()) {
            DeviceMemoryStats stats = getMemoryStats(device);
            log.info("{}: allocated={} MB, available={} MB, peak={} MB, cap={} MB, util={:.1f}%",
                    stats.deviceId,
                    stats.allocated / (1024 * 1024),
                    stats.available / (1024 * 1024),
                    stats.peak / (1024 * 1024),
                    stats.cap / (1024 * 1024),
                    stats.utilization * 100);
        }
    }

    // =========================================================================
    // Inner Classes
    // =========================================================================

    /**
     * Memory statistics for a device.
     */
    public static class DeviceMemoryStats {
        public final String deviceId;
        public final long total;
        public final long allocated;
        public final long available;
        public final long peak;
        public final long cap;
        public final double utilization;

        public DeviceMemoryStats(String deviceId, long total, long allocated,
                                  long available, long peak, long cap, double utilization) {
            this.deviceId = deviceId;
            this.total = total;
            this.allocated = allocated;
            this.available = available;
            this.peak = peak;
            this.cap = cap;
            this.utilization = utilization;
        }
    }

    /**
     * Callback interface for memory pressure events.
     */
    @FunctionalInterface
    public interface MemoryPressureCallback {
        /**
         * Called when memory utilization exceeds the pressure threshold.
         *
         * @param device the device under pressure
         * @param utilization current utilization (0.0 to 1.0)
         */
        void onMemoryPressure(DeviceDescriptor device, double utilization);
    }

    /**
     * Device routing policies.
     */
    public enum DeviceRoutingPolicy {
        /** Select device with highest priority that has space */
        MEMORY_PRIORITY,
        /** Prefer GPU devices */
        PREFER_GPU,
        /** Prefer CPU */
        PREFER_CPU,
        /** Round-robin across devices */
        ROUND_ROBIN,
        /** Select device with lowest memory utilization */
        LEAST_LOADED
    }
}
