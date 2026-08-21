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
import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.BackendRegistry;
import org.nd4j.nativeblas.NativeOpsHolder;

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
 * Adam Gibson
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
    private volatile boolean defaultDeviceUserSet = false;

    // Configuration
    @Getter
    private volatile boolean autoFallbackEnabled = true;

    @Getter
    private volatile double memoryPressureThreshold = 0.9; // 90% triggers pressure

    // Callbacks for memory pressure
    private final List<MemoryPressureCallback> pressureCallbacks = new ArrayList<>();
    private final ReentrantReadWriteLock callbackLock = new ReentrantReadWriteLock();

    // Additional callbacks for memory pressure (for framework integration)
    private final List<MemoryPressureCallback> memoryPressureCallbacks = new ArrayList<>();

    // Device routing policy
    private volatile DeviceRoutingPolicy deviceRoutingPolicy = DeviceRoutingPolicy.MEMORY_PRIORITY;

    // Device context provider (SPI — set by backend at init time)
    private volatile DeviceContextProvider contextProvider = new CpuDeviceContextProvider();

    // =========================================================================
    // Device Context Provider (SPI)
    // =========================================================================

    /**
     * Set the device context provider. Called by backends (CUDA, CPU) at initialization.
     *
     * @param provider the backend-specific provider
     */
    public void setContextProvider(DeviceContextProvider provider) {
        this.contextProvider = provider;
    }

    /**
     * Get the current device context provider.
     */
    public DeviceContextProvider getContextProvider() {
        return contextProvider;
    }

    /**
     * Switch to the specified device and return a fresh DeviceContext with valid
     * stream pointers. This is the SINGLE entry point for all device switching.
     *
     * Replaces all usages of {@code Nd4j.getAffinityManager().unsafeSetDevice()}.
     *
     * @param deviceId target device ID
     * @param caller   class/method performing the switch (for tracing)
     * @param reason   why the switch is happening (for tracing)
     * @return fresh DeviceContext snapshot (do NOT cache across switches)
     */
    public DeviceContext switchDevice(int deviceId, String caller, String reason) {
        return contextProvider.switchDevice(deviceId, caller, reason);
    }

    /**
     * Get a fresh DeviceContext for the current thread's device.
     * Returns valid stream pointers from the thread-local C++ ContextBuffers.
     */
    public DeviceContext getCurrentDeviceContext() {
        return contextProvider.getCurrentContext();
    }

    /**
     * Get a fresh execution stream pointer for the current device.
     * Use this instead of caching stream pointers across device switches.
     */
    public Pointer getFreshExecutionStream() {
        return contextProvider.getFreshExecutionStream();
    }

    /**
     * Get the current thread's device ID.
     */
    public int getCurrentDeviceId() {
        return contextProvider.getCurrentDeviceId();
    }

    /**
     * Ensure an array's data is accessible on the host (CPU) side.
     * On CUDA, this synchronizes device data to the host buffer so that
     * host-side reads (e.g., serialization via {@code data().asBytes()})
     * return current values instead of stale zeros.
     * On CPU, this is a no-op.
     *
     * @param array the array whose data must be readable on host
     */
    public void ensureHostAccess(INDArray array) {
        contextProvider.ensureHostAccess(array);
    }

    // =========================================================================
    // Memory Simulation for Testing
    // =========================================================================

    /**
     * Device ID constant for CPU in integer-based device routing.
     * GPUs use indices 0, 1, 2, etc. CPU uses -1.
     */
    public static final int CPU_DEVICE_ID = -1;

    /**
     * Return value when NO device can accommodate allocation.
     */
    public static final int NO_DEVICE_AVAILABLE = -2;

    /**
     * Enable/disable memory simulation mode globally.
     * When enabled, simulated free memory values override actual device queries.
     */
    private volatile boolean memorySimulationEnabled = false;

    /**
     * Simulated free memory per device (deviceId -> bytes).
     * Uses integer device IDs: GPU indices (0, 1, 2...) or CPU_DEVICE_ID (-1).
     */
    private final Map<Integer, Long> simulatedFreeMemory = new ConcurrentHashMap<>();

    /**
     * Track simulated allocations to decrease simulated free memory.
     */
    private final Map<Integer, Long> simulatedAllocatedMemory = new ConcurrentHashMap<>();

    private DeviceMemoryManager() {
        // Initialize with CPU as default
        defaultDevice = DeviceDescriptor.cpu();
        fallbackDevice = DeviceDescriptor.cpu();
        registerDevice(defaultDevice);
    }

    private void ensureDevicesRegistered() {
        boolean hasGpu = registeredDevices.values().stream()
                .anyMatch(device -> device.getDeviceType().isGpu());

        if (!hasGpu) {
            try {
                BackendRegistry registry = BackendRegistry.getInstance();
                for (DeviceDescriptor device : registry.getAllDevices()) {
                    registerDevice(device);
                }
                hasGpu = registeredDevices.values().stream()
                        .anyMatch(device -> device.getDeviceType().isGpu());
            } catch (Exception e) {
                log.debug("DeviceMemoryManager: unable to auto-register devices: {}", e.getMessage());
            }
        }

        if (!defaultDeviceUserSet && hasGpu &&
                (defaultDevice == null || defaultDevice.getDeviceType() == DeviceType.CPU)) {
            DeviceDescriptor preferred = null;
            try {
                preferred = BackendRegistry.getInstance().getDefaultGpuDevice();
            } catch (Exception e) {
                log.debug("DeviceMemoryManager: unable to resolve default GPU device: {}", e.getMessage());
            }
            if (preferred == null) {
                preferred = registeredDevices.values().stream()
                        .filter(device -> device.getDeviceType().isGpu())
                        .max(Comparator.comparingLong(DeviceDescriptor::getTotalMemory))
                        .orElse(null);
            }
            if (preferred != null) {
                defaultDevice = preferred;
            }
        }

        if (fallbackDevice == null) {
            DeviceDescriptor cpu = null;
            try {
                cpu = BackendRegistry.getInstance().getDefaultCpuDevice();
            } catch (Exception e) {
                log.debug("DeviceMemoryManager: unable to resolve default CPU device: {}", e.getMessage());
            }
            fallbackDevice = cpu != null ? cpu : DeviceDescriptor.cpu();
            registerDevice(fallbackDevice);
        }
    }

    private Integer simulationDeviceId(DeviceDescriptor device) {
        if (device == null) {
            return null;
        }
        if (device.getDeviceType() == DeviceType.CPU) {
            return CPU_DEVICE_ID;
        }
        if (device.getDeviceType().isGpu()) {
            return device.getDeviceIndex();
        }
        return null;
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
                // Prefer GPUs with more total memory (1 priority point per GB)
                // so a 24GB GPU gets ~124 while an 8GB GPU gets ~108
                return 100 + (int) (device.getTotalMemory() / (1024L * 1024L * 1024L));
            case ROCM_GPU:
            case METAL_GPU:
                return 80 + (int) (device.getTotalMemory() / (1024L * 1024L * 1024L));
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
     * Get a registered GPU device by its device index.
     *
     * @param deviceIndex the GPU device index (e.g., 0, 1)
     * @return the registered device descriptor, or null if not found
     */
    public DeviceDescriptor getRegisteredDevice(int deviceIndex) {
        ensureDevicesRegistered();
        for (DeviceDescriptor device : registeredDevices.values()) {
            if (device.getDeviceIndex() == deviceIndex && device.getDeviceType().isGpu()) {
                return device;
            }
        }
        return null;
    }

    /**
     * Get the number of registered devices.
     *
     * @return count of registered devices
     */
    public int getRegisteredDeviceCount() {
        ensureDevicesRegistered();
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
        defaultDeviceUserSet = false;
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
        long cap = memoryCaps.getOrDefault(id, 0L);
        long allocated = allocatedMemory.getOrDefault(id, new AtomicLong(0)).get();

        Integer simId = simulationDeviceId(device);
        if (memorySimulationEnabled && simId != null && simulatedFreeMemory.containsKey(simId)) {
            long simulatedAvailable = getEffectiveFreeMemory(simId, device.getAvailableMemory());
            if (cap > 0) {
                return Math.max(0, Math.min(cap, simulatedAvailable));
            }
            return Math.max(0, simulatedAvailable);
        }

        if (cap > 0) {
            // Capped: return remaining under cap
            return Math.max(0, cap - allocated);
        }

        // Unlimited: return device's available memory minus our allocations
        return Math.max(0, device.getAvailableMemory() - allocated);
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
        this.defaultDeviceUserSet = true;
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
        ensureDevicesRegistered();

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
     * Select the best device for an allocation, preferring a specific device if possible.
     *
     * @param bytes allocation size
     * @param preferred preferred device (checked first if non-null)
     * @return selected device
     */
    public DeviceDescriptor selectDeviceForAllocation(long bytes, DeviceDescriptor preferred) {
        ensureDevicesRegistered();
        if (preferred != null) {
            registerDevice(preferred);
            if (canAllocate(preferred, bytes)) {
                return preferred;
            }
        }
        return selectDeviceForAllocation(bytes);
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
            case MOST_FREE:
                return selectMostFree(bytes);
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

    /**
     * Select the GPU device with the most actual free memory that can accommodate
     * the requested allocation. Uses real CUDA free memory queries, not internal tracking.
     */
    private DeviceDescriptor selectMostFree(long bytes) {
        ensureDevicesRegistered();
        DeviceDescriptor best = null;
        long bestFree = -1;

        for (DeviceDescriptor device : registeredDevices.values()) {
            if (!device.getDeviceType().isGpu()) continue;

            long freeMem = getActualFreeMemory(device);
            if (freeMem >= bytes && freeMem > bestFree) {
                bestFree = freeMem;
                best = device;
            }
        }

        return best != null ? best : defaultDevice;
    }

    /**
     * Get actual free memory for a device by querying the native runtime.
     * For GPU devices, this calls into CUDA/ROCm to get real free memory.
     * Falls back to the descriptor's available memory if native query fails.
     */
    public long getActualFreeMemory(DeviceDescriptor device) {
        // Check simulation first — if enabled, simulated values override real queries
        Integer simId = simulationDeviceId(device);
        if (memorySimulationEnabled && simId != null && simulatedFreeMemory.containsKey(simId)) {
            return getEffectiveFreeMemory(simId, device.getAvailableMemory());
        }

        if (device.getDeviceType().isGpu()) {
            try {
                return NativeOpsHolder.getInstance().getDeviceNativeOps()
                        .getDeviceFreeMemory(device.getDeviceIndex());
            } catch (Exception e) {
                log.debug("Failed to query actual free memory for {}: {}", device.getDeviceId(), e.getMessage());
            }
        }
        return device.getAvailableMemory();
    }

    /**
     * Select the best GPU device based on actual free memory.
     * This is the single mechanism for all GPU device selection in the system.
     * Returns the device index (int) of the GPU with the most free memory.
     *
     * @return GPU device index with the most free memory, or 0 if query fails
     */
    public int selectBestGpu() {
        return selectBestGpu(null);
    }

    /**
     * Select the best GPU among the given candidate device indices, by pool-aware free memory.
     * Callers that maintain a ban/exclusion list (e.g. CudaAffinityManager after banDevice() on a
     * device whose CUDA context could not be created) pass the available (non-banned) device set so
     * a banned device is never re-selected. When {@code candidateDevices} is null or empty, all
     * physical GPUs are considered (identical to the historical no-arg behaviour).
     *
     * @param candidateDevices device indices eligible for selection, or null/empty for all
     * @return the eligible GPU device index with the most free memory, or 0 if the query fails
     */
    public int selectBestGpu(java.util.Collection<Integer> candidateDevices) {
        return selectBestGpu(candidateDevices, 0, false);
    }

    /**
     * Select the pool-aware GPU with enough available memory for a complete planned allocation.
     * Unlike per-buffer failover, this is an admission decision: it returns
     * {@link #NO_DEVICE_AVAILABLE} rather than committing a model to an undersized GPU.
     */
    public int selectBestGpuForAllocation(
            long requiredBytes, java.util.Collection<Integer> candidateDevices) {
        if (requiredBytes < 0) {
            throw new IllegalArgumentException("requiredBytes must not be negative");
        }
        return selectBestGpu(candidateDevices, requiredBytes, true);
    }

    private int selectBestGpu(
            java.util.Collection<Integer> candidateDevices,
            long requiredBytes,
            boolean requireCapacity) {
        try {
            var nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
            int numDevices = nativeOps.getAvailableDevices();
            boolean restrict = candidateDevices != null && !candidateDevices.isEmpty();

            long bestAvailable = -1;
            int bestDevice = -1;
            for (int i = 0; i < numDevices; i++) {
                if (restrict && !candidateDevices.contains(i)) continue;
                long available = getPoolAwareFreeMemory(i);
                if ((!requireCapacity || available >= requiredBytes) && available > bestAvailable) {
                    bestAvailable = available;
                    bestDevice = i;
                }
            }

            if (bestDevice < 0) {
                if (requireCapacity) {
                    log.warn("No eligible GPU has {} MB pool-aware memory available among candidates {}",
                            requiredBytes / (1024 * 1024), candidateDevices);
                    return NO_DEVICE_AVAILABLE;
                }
                log.warn("selectBestGpu: no eligible device among candidates {}, defaulting to device 0",
                        candidateDevices);
                return 0;
            }

            log.debug("selectBestGpu: device [{}] selected with {} MB available (required={} MB) out of {} devices{}",
                    bestDevice, bestAvailable / (1024 * 1024), requiredBytes / (1024 * 1024),
                    numDevices, restrict ? " (restricted to " + candidateDevices + ")" : "");
            return bestDevice;
        } catch (Exception e) {
            if (requireCapacity) {
                log.warn("Failed to query GPU capacity for {} bytes: {}", requiredBytes, e.getMessage());
                return NO_DEVICE_AVAILABLE;
            }
            log.warn("Failed to query GPU free memory, defaulting to device 0: {}", e.getMessage());
            return 0;
        }
    }

    /**
     * Pool-aware free memory for a single GPU device index: actual CUDA free memory plus the
     * reclaimable portion of the cudaMallocAsync pool (reserved-but-unused), which the next
     * allocation can reuse. This is the same accounting {@link #selectBestGpu()} uses, exposed
     * per-device so the failover path can ask "can this specific GPU still fit the allocation?".
     *
     * <p>Honors memory-simulation mode for testing. Returns {@link Long#MAX_VALUE} when the
     * native query itself fails, so a transient query error never triggers a false failover.
     *
     * @param deviceIndex GPU device index
     * @return pool-aware free bytes on that GPU
     */
    public long getPoolAwareFreeMemory(int deviceIndex) {
        long available;
        long poolUsed = 0;
        if (memorySimulationEnabled && simulatedFreeMemory.containsKey(deviceIndex)) {
            available = getEffectiveFreeMemory(deviceIndex, 0);
        } else {
            try {
                var nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
                long cudaFree = nativeOps.getDeviceFreeMemory(deviceIndex);
                long poolReusable = 0;
                try {
                    org.bytedeco.javacpp.LongPointer usedPtr = new org.bytedeco.javacpp.LongPointer(1);
                    org.bytedeco.javacpp.LongPointer reservedPtr = new org.bytedeco.javacpp.LongPointer(1);
                    nativeOps.getMemoryPoolStats(deviceIndex, usedPtr, reservedPtr);
                    poolUsed = Math.max(0, usedPtr.get());
                    poolReusable = Math.max(0, reservedPtr.get() - poolUsed);
                } catch (Exception ignored) {}
                available = cudaFree + poolReusable;
            } catch (Exception e) {
                log.debug("getPoolAwareFreeMemory({}) failed: {}", deviceIndex, e.getMessage());
                return Long.MAX_VALUE;
            }
        }

        DeviceDescriptor device = getRegisteredDevice(deviceIndex);
        if (device != null) {
            long cap = getMemoryCap(device);
            if (cap > 0) {
                long tracked = allocatedMemory.getOrDefault(
                        device.getDeviceId(), new AtomicLong(0)).get();
                long capRemaining = Math.max(0, cap - Math.max(poolUsed, tracked));
                available = Math.min(available, capRemaining);
            }
        }
        return available;
    }

    /**
     * Return the bytes currently occupied by the native allocator pool for one GPU.
     * This is the process-owned CUDA-pool signal used by memory watchdogs; it is
     * deliberately distinct from driver-wide used memory and from pool-aware free memory.
     *
     * @param deviceIndex GPU device index
     * @return native pool used bytes, or {@code -1} when the backend cannot report it
     */
    public long getNativePoolUsedMemory(int deviceIndex) {
        try {
            var nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
            org.bytedeco.javacpp.LongPointer usedPtr = new org.bytedeco.javacpp.LongPointer(1);
            org.bytedeco.javacpp.LongPointer reservedPtr = new org.bytedeco.javacpp.LongPointer(1);
            nativeOps.getMemoryPoolStats(deviceIndex, usedPtr, reservedPtr);
            return Math.max(0L, usedPtr.get());
        } catch (Exception e) {
            log.debug("getNativePoolUsedMemory({}) failed: {}", deviceIndex, e.getMessage());
            return -1L;
        }
    }

    /**
     * Select the device to fail an allocation over to when its current GPU is full.
     *
     * <p>This is the single failover entry point. It consults live, pool-aware device memory
     * via {@link DeviceMemoryManager} — NOT sticky thread affinity — and picks the GPU with the
     * most reclaimable free memory that (a) is not the device that just failed and (b) can fit
     * {@code bytes}. When no other GPU has room it returns the CPU/fallback device (if auto
     * fallback is enabled and it can fit). Returns {@code null} when nothing can accommodate the
     * allocation, so the caller can surface a real OOM instead of looping forever.
     *
     * @param bytes              allocation size in bytes
     * @param excludeDeviceIndex GPU index that just OOMed (excluded from the GPU search)
     * @return target device to retry on, or {@code null} if none can fit
     */
    public DeviceDescriptor selectFailoverDevice(long bytes, int excludeDeviceIndex) {
        ensureDevicesRegistered();

        DeviceDescriptor best = null;
        long bestFree = -1;
        for (DeviceDescriptor device : registeredDevices.values()) {
            if (!device.getDeviceType().isGpu()) continue;
            int idx = device.getDeviceIndex();
            if (idx == excludeDeviceIndex) continue;
            long free = getPoolAwareFreeMemory(idx);
            if (free >= bytes && free > bestFree) {
                bestFree = free;
                best = device;
            }
        }

        if (best != null) {
            log.warn("Allocation of {} bytes failing over from device [{}] to GPU [{}] ({} MB free, pool-aware)",
                    bytes, excludeDeviceIndex, best.getDeviceIndex(), bestFree / (1024 * 1024));
            return best;
        }

        // No other GPU has room — fall back to the CPU/fallback device if it can fit.
        if (autoFallbackEnabled && fallbackDevice != null && canAllocate(fallbackDevice, bytes)) {
            log.warn("Allocation of {} bytes failing over from device [{}] to fallback {}",
                    bytes, excludeDeviceIndex, fallbackDevice.getDeviceId());
            return fallbackDevice;
        }
        return null;
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
        registerDevice(device);

        Integer simId = simulationDeviceId(device);
        if (memorySimulationEnabled && simId != null && simulatedFreeMemory.containsKey(simId)) {
            recordSimulatedAllocation(simId, bytes);
            // Still check memory pressure with the simulated utilization
            checkMemoryPressure(device);
            return;
        }

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
        Integer simId = simulationDeviceId(device);
        if (memorySimulationEnabled && simId != null && simulatedFreeMemory.containsKey(simId)) {
            simulatedAllocatedMemory.merge(simId, -bytes, Long::sum);
            return;
        }

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
     * When simulation mode is enabled, computes utilization from simulated free memory
     * rather than the tracked allocation counter (which is routed to the simulation map).
     *
     * @param device the device
     * @return utilization (0.0 to 1.0)
     */
    public double getMemoryUtilization(DeviceDescriptor device) {
        Integer simId = simulationDeviceId(device);
        if (memorySimulationEnabled && simId != null && simulatedFreeMemory.containsKey(simId)) {
            long totalMem = device.getTotalMemory();
            if (totalMem <= 0) return 0;
            long effectiveFree = getEffectiveFreeMemory(simId, device.getAvailableMemory());
            return 1.0 - ((double) effectiveFree / totalMem);
        }
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
    // Stub Topology for Testing
    // =========================================================================

    // Reference to stub provider if configured
    private volatile StubDeviceContextProvider stubContextProvider;

    /**
     * Configure a stub multi-device topology for testing.
     * Clears all existing devices, registers stubs, injects StubDeviceContextProvider,
     * and auto-configures simulated memory from stub values.
     *
     * @param stubs list of stub device descriptors to configure
     */
    public void configureStubTopology(List<StubDeviceDescriptor> stubs) {
        if (stubs == null || stubs.isEmpty()) {
            throw new IllegalArgumentException("Must provide at least one stub device");
        }

        clearDevices();

        for (StubDeviceDescriptor stub : stubs) {
            registerDevice(stub);
            // Configure simulated memory from stub
            Integer simId = simulationDeviceId(stub);
            if (simId != null) {
                simulatedFreeMemory.put(simId, stub.getAvailableMemory());
            }
        }

        // Set first GPU as default, first CPU as fallback
        StubDeviceDescriptor firstGpu = null;
        StubDeviceDescriptor firstCpu = null;
        for (StubDeviceDescriptor stub : stubs) {
            if (stub.getDeviceType().isGpu() && firstGpu == null) {
                firstGpu = stub;
            }
            if (stub.getDeviceType() == DeviceType.CPU && firstCpu == null) {
                firstCpu = stub;
            }
        }
        if (firstGpu != null) {
            setDefaultDevice(firstGpu);
        }
        if (firstCpu != null) {
            setFallbackDevice(firstCpu);
        }

        // Create and inject stub context provider
        stubContextProvider = new StubDeviceContextProvider(stubs);
        setContextProvider(stubContextProvider);

        // Enable simulation mode
        memorySimulationEnabled = true;

        log.info("Configured stub topology with {} devices", stubs.size());
    }

    /**
     * Get the StubDeviceContextProvider if a stub topology is configured.
     *
     * @return the stub context provider, or null if not configured
     */
    public StubDeviceContextProvider getStubContextProvider() {
        return stubContextProvider;
    }

    /**
     * Clear the stub topology and restore default CpuDeviceContextProvider.
     */
    public void clearStubTopology() {
        stubContextProvider = null;
        setContextProvider(new CpuDeviceContextProvider());
        clearAllMemorySimulation();
        clearDevices();
        log.info("Cleared stub topology, restored CPU context provider");
    }

    // =========================================================================
    // Memory Simulation API (for testing)
    // =========================================================================

    /**
     * Enable memory simulation mode for testing OOM scenarios.
     * When enabled, {@link #getSimulatedFreeMemory} values override actual device queries.
     *
     * @param enabled true to enable simulation mode
     */
    public void setMemorySimulationEnabled(boolean enabled) {
        this.memorySimulationEnabled = enabled;
        if (!enabled) {
            simulatedAllocatedMemory.clear();
        }
        log.info("Memory simulation mode: {}", enabled ? "ENABLED" : "DISABLED");
    }

    /**
     * Check if memory simulation mode is enabled.
     *
     * @return true if simulation mode is enabled
     */
    public boolean isMemorySimulationEnabled() {
        return memorySimulationEnabled;
    }

    /**
     * Set simulated free memory for a specific device.
     * This value will be used instead of actual memory queries when simulation is enabled.
     *
     * <p>Example usage in tests:
     * <pre>{@code
     * DeviceMemoryManager mgr = DeviceMemoryManager.getInstance();
     * // Simulate OOM on GPU 0 (only 1MB free)
     * mgr.setSimulatedFreeMemory(0, 1024 * 1024);
     * // GPU 1 has plenty of memory
     * mgr.setSimulatedFreeMemory(1, 8L * 1024 * 1024 * 1024);
     * // CPU has 32GB
     * mgr.setSimulatedFreeMemory(DeviceMemoryManager.CPU_DEVICE_ID, 32L * 1024 * 1024 * 1024);
     * mgr.setMemorySimulationEnabled(true);
     * }</pre>
     *
     * @param deviceId the device ID (GPU index 0+, or CPU_DEVICE_ID for CPU)
     * @param freeBytes simulated free memory in bytes
     */
    public void setSimulatedFreeMemory(int deviceId, long freeBytes) {
        simulatedFreeMemory.put(deviceId, freeBytes);
        String deviceName = (deviceId == CPU_DEVICE_ID) ? "CPU" : "GPU " + deviceId;
        log.info("Set simulated free memory for {}: {} MB", deviceName, freeBytes / (1024 * 1024));
    }

    /**
     * Get the simulated free memory for a device.
     *
     * @param deviceId the device ID
     * @return simulated free bytes, or -1 if not set
     */
    public long getSimulatedFreeMemory(int deviceId) {
        return simulatedFreeMemory.getOrDefault(deviceId, -1L);
    }

    /**
     * Get the effective free memory for a device, considering simulation mode.
     * If simulation is enabled and a value is set, returns simulated value.
     * Otherwise, returns the actual free memory from the device.
     *
     * @param deviceId the device ID
     * @param actualFreeMemory the actual free memory (used if simulation not active)
     * @return effective free memory in bytes
     */
    public long getEffectiveFreeMemory(int deviceId, long actualFreeMemory) {
        if (memorySimulationEnabled && simulatedFreeMemory.containsKey(deviceId)) {
            long baseFree = simulatedFreeMemory.get(deviceId);
            long allocated = simulatedAllocatedMemory.getOrDefault(deviceId, 0L);
            long effective = Math.max(0, baseFree - allocated);
            String deviceName = (deviceId == CPU_DEVICE_ID) ? "CPU" : "GPU " + deviceId;
            log.trace("Simulated free memory for {}: base={} MB, allocated={} MB, effective={} MB",
                    deviceName, baseFree / (1024 * 1024), allocated / (1024 * 1024), effective / (1024 * 1024));
            return effective;
        }
        return actualFreeMemory;
    }

    /**
     * Record a simulated allocation to decrease simulated free memory.
     *
     * @param deviceId the device ID
     * @param bytes bytes allocated
     */
    public void recordSimulatedAllocation(int deviceId, long bytes) {
        if (memorySimulationEnabled && simulatedFreeMemory.containsKey(deviceId)) {
            simulatedAllocatedMemory.merge(deviceId, bytes, Long::sum);
            log.trace("Recorded simulated allocation of {} MB on device {}", bytes / (1024 * 1024), deviceId);
        }
    }

    /**
     * Get the simulated allocated memory for a device (for testing verification).
     *
     * @param deviceId the device ID
     * @return simulated allocated bytes
     */
    public long getSimulatedAllocatedMemory(int deviceId) {
        return simulatedAllocatedMemory.getOrDefault(deviceId, 0L);
    }

    /**
     * Clear simulated free memory for a specific device.
     *
     * @param deviceId the device ID to clear
     */
    public void clearSimulatedFreeMemory(int deviceId) {
        simulatedFreeMemory.remove(deviceId);
        simulatedAllocatedMemory.remove(deviceId);
    }

    /**
     * Clear all simulated memory settings and disable simulation mode.
     * Call this in test cleanup to restore normal operation.
     */
    public void clearAllMemorySimulation() {
        memorySimulationEnabled = false;
        simulatedFreeMemory.clear();
        simulatedAllocatedMemory.clear();
        log.info("Cleared all memory simulation settings");
    }

    /**
     * Get the number of devices with simulated memory limits.
     *
     * @return count of devices with simulation configured
     */
    public int getSimulatedDeviceCount() {
        return simulatedFreeMemory.size();
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
        /** Select device with the most actual free memory */
        MOST_FREE
    }

    /**
     * Check if a device has memory pressure.
     * @param device the device to check
     * @return true if memory pressure detected
     */
    public boolean hasMemoryPressure(DeviceDescriptor device) {
        if (device == null) return false;
        double utilization = getMemoryUtilization(device);
        return utilization >= memoryPressureThreshold;
    }

    /**
     * Register a callback for memory pressure events.
     * @param callback the callback to register
     */
    public void registerMemoryPressureCallback(MemoryPressureCallback callback) {
        memoryPressureCallbacks.add(callback);
    }

    /**
     * Get the current device routing policy.
     * @return the current routing policy
     */
    public DeviceRoutingPolicy getDeviceRoutingPolicy() {
        return deviceRoutingPolicy != null ? deviceRoutingPolicy : DeviceRoutingPolicy.MEMORY_PRIORITY;
    }

    /**
     * Set the device routing policy.
     * @param policy the new routing policy
     */
    public void setDeviceRoutingPolicy(DeviceRoutingPolicy policy) {
        this.deviceRoutingPolicy = policy;
    }

    /**
     * Get a summary of memory usage across all devices.
     * @return memory summary string
     */
    public String getMemorySummary() {
        StringBuilder sb = new StringBuilder();
        sb.append("Device Memory Summary:\n");
        for (Map.Entry<String, AtomicLong> entry : allocatedMemory.entrySet()) {
            String deviceId = entry.getKey();
            long allocated = entry.getValue().get();
            long peak = peakMemory.getOrDefault(deviceId, new AtomicLong(0)).get();
            long cap = memoryCaps.getOrDefault(deviceId, 0L);
            sb.append(String.format("  %s: allocated=%d MB, peak=%d MB, cap=%d MB\n",
                deviceId, allocated / (1024 * 1024), peak / (1024 * 1024), cap / (1024 * 1024)));
        }
        return sb.toString();
    }
}
