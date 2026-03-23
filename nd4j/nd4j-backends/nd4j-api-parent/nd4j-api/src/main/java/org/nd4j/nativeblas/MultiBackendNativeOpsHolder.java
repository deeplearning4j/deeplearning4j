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

package org.nd4j.nativeblas;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.common.config.ND4JClassLoading;
import org.nd4j.common.io.ReflectionUtils;
import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.device.DeviceType;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Multi-backend NativeOps holder that can load and manage multiple backend implementations
 * simultaneously (e.g., CPU, CUDA, ZLUDA, TPU, ROCm, Metal).
 *
 * <p>This enables true multi-device operation where:
 * <ul>
 *   <li>CPU arrays can be processed by the CPU backend</li>
 *   <li>GPU arrays can be processed by the appropriate GPU backend</li>
 *   <li>TPU arrays can be processed by the TPU backend</li>
 *   <li>Cross-device operations automatically transfer data as needed</li>
 * </ul>
 *
 * <h3>Supported Backends:</h3>
 * <ul>
 *   <li><b>CPU (nd4j-native)</b> - Standard CPU execution with OneDNN/MKL helpers</li>
 *   <li><b>CUDA (nd4j-cuda)</b> - NVIDIA GPU execution with cuDNN helpers</li>
 *   <li><b>ZLUDA (nd4j-zluda)</b> - AMD/Intel GPU via CUDA API translation</li>
 *   <li><b>TPU (nd4j-tpu)</b> - Google Cloud TPU via PJRT runtime</li>
 *   <li><b>ROCm</b> - Native AMD GPU (future)</li>
 *   <li><b>Metal</b> - Apple GPU (future)</li>
 * </ul>
 *
 * <h3>Usage:</h3>
 * <pre>{@code
 * // Enable multi-backend mode (discovers and loads all available backends)
 * MultiBackendNativeOpsHolder.enableMultiBackend();
 *
 * // Check what's available
 * Set<DeviceType> available = MultiBackendNativeOpsHolder.getInstance().getAvailableBackendTypes();
 *
 * // Get ops for a specific device
 * NativeOps cpuOps = MultiBackendNativeOpsHolder.getInstance().getOpsForDevice(DeviceDescriptor.cpu());
 * NativeOps gpuOps = MultiBackendNativeOpsHolder.getInstance().getOpsForDevice(DeviceDescriptor.cuda(0));
 *
 * // Get backend info
 * System.out.println(MultiBackendNativeOpsHolder.getInstance().getBackendInfo());
 * }</pre>
 *
 * Adam Gibson
 */
@Slf4j
public class MultiBackendNativeOpsHolder {

    private static final MultiBackendNativeOpsHolder INSTANCE = new MultiBackendNativeOpsHolder();

    /**
     * Registry of known backend implementations.
     * Maps DeviceType to the fully qualified class name of its NativeOps implementation.
     */
    private static final Map<DeviceType, BackendInfo> BACKEND_REGISTRY = new LinkedHashMap<>();

    static {
        // Register all known backends with their class names and priorities
        // Higher priority = preferred when multiple backends available
        BACKEND_REGISTRY.put(DeviceType.CPU, new BackendInfo(
                "org.nd4j.linalg.cpu.nativecpu.bindings.Nd4jCpu",
                "CPU (nd4j-native)",
                10
        ));
        BACKEND_REGISTRY.put(DeviceType.CUDA_GPU, new BackendInfo(
                "org.nd4j.linalg.jcublas.bindings.Nd4jCuda",
                "CUDA (nd4j-cuda)",
                100
        ));
        BACKEND_REGISTRY.put(DeviceType.ROCM_GPU, new BackendInfo(
                "org.nd4j.linalg.jzluda.bindings.Nd4jZluda",
                "ZLUDA/ROCm (nd4j-zluda)",
                90
        ));
        BACKEND_REGISTRY.put(DeviceType.TPU, new BackendInfo(
                "org.nd4j.linalg.jtpu.bindings.Nd4jTpu",
                "TPU (nd4j-tpu)",
                110
        ));
        BACKEND_REGISTRY.put(DeviceType.METAL_GPU, new BackendInfo(
                "org.nd4j.linalg.jmetal.bindings.Nd4jMetal",
                "Metal (nd4j-metal)",
                85
        ));
    }

    // Loaded backend implementations by device type
    private final Map<DeviceType, NativeOps> loadedBackends = new ConcurrentHashMap<>();

    // Device to backend mapping (for specific device instances)
    private final Map<String, NativeOps> deviceOpsMap = new ConcurrentHashMap<>();

    // Primary backend (highest priority available)
    @Getter
    private volatile NativeOps primaryOps;

    @Getter
    private volatile DeviceType primaryBackendType;

    // State tracking
    private final AtomicBoolean initialized = new AtomicBoolean(false);
    private final AtomicBoolean multiBackendEnabled = new AtomicBoolean(false);

    // Statistics per backend type
    private final Map<DeviceType, AtomicLong> opCounts = new ConcurrentHashMap<>();

    // Legacy accessors for backward compatibility
    @Getter
    private volatile NativeOps cpuOps;
    @Getter
    private volatile NativeOps cudaOps;

    private MultiBackendNativeOpsHolder() {
        // Initialize op counters for all device types
        for (DeviceType type : DeviceType.values()) {
            opCounts.put(type, new AtomicLong(0));
        }
    }

    public static MultiBackendNativeOpsHolder getInstance() {
        return INSTANCE;
    }

    /**
     * Enable multi-backend mode, attempting to load all available backends.
     *
     * @return true if at least one backend was loaded
     */
    public static boolean enableMultiBackend() {
        return getInstance().initializeMultiBackend();
    }

    /**
     * Register a pre-loaded NativeOps instance.
     * This is used by NativeOpsHolder to register the primary backend so that
     * MultiBackendNativeOpsHolder doesn't try to load it again.
     *
     * @param deviceType the device type this NativeOps handles
     * @param ops the NativeOps instance
     */
    public void registerPreloadedBackend(DeviceType deviceType, NativeOps ops) {
        if (deviceType != null && ops != null) {
            loadedBackends.putIfAbsent(deviceType, ops);
            log.debug("Registered pre-loaded backend for {}", deviceType);

            // Set legacy accessors if applicable
            if (deviceType == DeviceType.CPU && cpuOps == null) {
                cpuOps = ops;
            } else if (deviceType == DeviceType.CUDA_GPU && cudaOps == null) {
                cudaOps = ops;
            }
        }
    }

    /**
     * Check if multi-backend mode is enabled (more than one backend loaded).
     */
    public static boolean isMultiBackendEnabled() {
        return getInstance().multiBackendEnabled.get();
    }

    /**
     * Initialize multi-backend support by discovering and loading all available backends.
     */
    private synchronized boolean initializeMultiBackend() {
        if (initialized.get()) {
            return multiBackendEnabled.get();
        }

        log.info("Initializing multi-backend NativeOps support...");
        log.info("Discovering available backends from registry ({} known backends)", BACKEND_REGISTRY.size());

        int loadedCount = 0;
        int highestPriority = Integer.MIN_VALUE;

        // Try to load each registered backend
        for (Map.Entry<DeviceType, BackendInfo> entry : BACKEND_REGISTRY.entrySet()) {
            DeviceType deviceType = entry.getKey();
            BackendInfo info = entry.getValue();

            // Skip if already pre-loaded (e.g., by NativeOpsHolder)
            if (loadedBackends.containsKey(deviceType)) {
                NativeOps ops = loadedBackends.get(deviceType);
                loadedCount++;
                log.debug("Using pre-loaded {} backend", info.displayName);

                // Still register devices for pre-loaded backends
                registerDevicesForBackend(deviceType, ops);

                // Track primary backend (highest priority)
                if (info.priority > highestPriority) {
                    highestPriority = info.priority;
                    primaryOps = ops;
                    primaryBackendType = deviceType;
                }
                continue;
            }

            try {
                NativeOps ops = loadBackend(info.className);
                if (ops != null) {
                    loadedBackends.put(deviceType, ops);
                    loadedCount++;
                    log.info("Successfully loaded {} backend: {}", info.displayName, info.className);

                    // Initialize the backend
                    try {
                        ops.initializeDevicesAndFunctions();
                    } catch (Exception e) {
                        log.warn("Backend {} loaded but initialization failed: {}",
                                info.displayName, e.getMessage());
                    }

                    // Register devices for this backend
                    registerDevicesForBackend(deviceType, ops);

                    // Track primary backend (highest priority)
                    if (info.priority > highestPriority) {
                        highestPriority = info.priority;
                        primaryOps = ops;
                        primaryBackendType = deviceType;
                    }

                    // Set legacy accessors
                    if (deviceType == DeviceType.CPU) {
                        cpuOps = ops;
                    } else if (deviceType == DeviceType.CUDA_GPU) {
                        cudaOps = ops;
                    }
                }
            } catch (Exception e) {
                log.debug("{} backend not available: {}", info.displayName, e.getMessage());
            }
        }

        multiBackendEnabled.set(loadedCount > 1);
        initialized.set(true);

        // Log summary
        if (loadedCount == 0) {
            log.error("No backends loaded! ND4J will not function correctly.");
        } else if (multiBackendEnabled.get()) {
            log.info("Multi-backend mode enabled: {} backends available", loadedCount);
            log.info("Primary backend: {} (priority {})",
                    BACKEND_REGISTRY.get(primaryBackendType).displayName, highestPriority);
        } else {
            log.info("Single backend mode: {} only",
                    BACKEND_REGISTRY.get(primaryBackendType).displayName);
        }

        return loadedCount > 0;
    }

    /**
     * Register all devices for a given backend type.
     */
    private void registerDevicesForBackend(DeviceType deviceType, NativeOps ops) {
        try {
            int deviceCount = ops.getAvailableDevices();
            if (deviceCount <= 0) {
                deviceCount = 1; // At least one device (e.g., CPU)
            }

            for (int i = 0; i < deviceCount; i++) {
                DeviceDescriptor device = createDeviceDescriptor(deviceType, i);
                if (device != null) {
                    deviceOpsMap.put(device.getDeviceId(), ops);
                    log.debug("Registered device: {}", device.getDeviceId());
                }
            }
            log.info("Registered {} devices for {} backend", deviceCount, deviceType);
        } catch (Exception e) {
            log.debug("Could not enumerate devices for {}: {}", deviceType, e.getMessage());
            // Register at least one device
            DeviceDescriptor device = createDeviceDescriptor(deviceType, 0);
            if (device != null) {
                deviceOpsMap.put(device.getDeviceId(), ops);
            }
        }
    }

    /**
     * Create a device descriptor for the given type and index.
     */
    private DeviceDescriptor createDeviceDescriptor(DeviceType deviceType, int index) {
        switch (deviceType) {
            case CPU:
                return DeviceDescriptor.cpu(index);
            case CUDA_GPU:
                return DeviceDescriptor.cuda(index);
            case ROCM_GPU:
            case METAL_GPU:
            case TPU:
                // For other types, use the fromId factory if available
                return DeviceDescriptor.fromId(deviceType.getIdentifier() + ":gpu:" + index);
            default:
                return null;
        }
    }

    /**
     * Load a backend by class name.
     */
    private NativeOps loadBackend(String className) {
        try {
            Class<?> clazz = ND4JClassLoading.loadClassByName(className);
            if (clazz != null && NativeOps.class.isAssignableFrom(clazz)) {
                return (NativeOps) ReflectionUtils.newInstance(clazz.asSubclass(NativeOps.class));
            }
        } catch (Exception e) {
            log.trace("Could not load backend {}: {}", className, e.getMessage());
        }
        return null;
    }

    /**
     * Get the NativeOps implementation for a specific device.
     *
     * @param device the device descriptor
     * @return the appropriate NativeOps, or primary if not found
     */
    public NativeOps getOpsForDevice(DeviceDescriptor device) {
        if (!initialized.get()) {
            initializeMultiBackend();
        }

        if (device == null) {
            return primaryOps;
        }

        // First try exact device match
        NativeOps ops = deviceOpsMap.get(device.getDeviceId());
        if (ops != null) {
            return ops;
        }

        // Fallback to device type
        return getOpsForDeviceType(device.getDeviceType());
    }

    /**
     * Get the appropriate NativeOps for executing on a target device type.
     *
     * @param deviceType the device type
     * @return the NativeOps for that device type, or primary if not available
     */
    public NativeOps getOpsForDeviceType(DeviceType deviceType) {
        if (!initialized.get()) {
            initializeMultiBackend();
        }

        if (deviceType == null) {
            return primaryOps;
        }

        NativeOps ops = loadedBackends.get(deviceType);
        if (ops != null) {
            return ops;
        }

        // Handle ZLUDA case - ROCm devices can use ZLUDA backend
        if (deviceType == DeviceType.ROCM_GPU) {
            ops = loadedBackends.get(DeviceType.CUDA_GPU);
            if (ops != null) {
                return ops;
            }
        }

        return primaryOps;
    }

    /**
     * Check if a specific backend is available.
     */
    public boolean isBackendAvailable(DeviceType deviceType) {
        if (!initialized.get()) {
            initializeMultiBackend();
        }

        if (deviceType == null) {
            return false;
        }

        return loadedBackends.containsKey(deviceType);
    }

    /**
     * Get all available backend types.
     *
     * @return set of available device types
     */
    public Set<DeviceType> getAvailableBackendTypes() {
        if (!initialized.get()) {
            initializeMultiBackend();
        }
        return Collections.unmodifiableSet(loadedBackends.keySet());
    }

    /**
     * Get the number of loaded backends.
     */
    public int getLoadedBackendCount() {
        if (!initialized.get()) {
            initializeMultiBackend();
        }
        return loadedBackends.size();
    }

    /**
     * Get all registered device IDs.
     *
     * @return set of device IDs
     */
    public Set<String> getRegisteredDeviceIds() {
        if (!initialized.get()) {
            initializeMultiBackend();
        }
        return Collections.unmodifiableSet(deviceOpsMap.keySet());
    }

    /**
     * Get the number of available devices for a backend type.
     */
    public int getDeviceCount(DeviceType deviceType) {
        if (!initialized.get()) {
            initializeMultiBackend();
        }

        NativeOps ops = loadedBackends.get(deviceType);
        if (ops == null) {
            return 0;
        }

        try {
            return ops.getAvailableDevices();
        } catch (Exception e) {
            return deviceType == DeviceType.CPU ? 1 : 0;
        }
    }

    /**
     * Get detailed information about loaded backends.
     */
    public String getBackendInfo() {
        if (!initialized.get()) {
            initializeMultiBackend();
        }

        StringBuilder sb = new StringBuilder();
        sb.append("Multi-Backend NativeOps Status:\n");
        sb.append("  Initialized: ").append(initialized.get()).append("\n");
        sb.append("  Multi-backend enabled: ").append(multiBackendEnabled.get()).append("\n");
        sb.append("  Loaded backends: ").append(loadedBackends.size()).append("\n");

        for (Map.Entry<DeviceType, BackendInfo> entry : BACKEND_REGISTRY.entrySet()) {
            DeviceType type = entry.getKey();
            BackendInfo info = entry.getValue();
            boolean loaded = loadedBackends.containsKey(type);
            sb.append("  - ").append(info.displayName).append(": ");
            if (loaded) {
                int deviceCount = getDeviceCount(type);
                sb.append("loaded (").append(deviceCount).append(" device(s))");
                if (type == primaryBackendType) {
                    sb.append(" [PRIMARY]");
                }
            } else {
                sb.append("not available");
            }
            sb.append("\n");
        }

        sb.append("  Registered devices: ").append(deviceOpsMap.size()).append("\n");
        for (String deviceId : deviceOpsMap.keySet()) {
            sb.append("    - ").append(deviceId).append("\n");
        }

        return sb.toString();
    }

    /**
     * Record op execution for statistics.
     */
    public void recordOpExecution(DeviceType deviceType) {
        if (deviceType != null) {
            opCounts.computeIfAbsent(deviceType, k -> new AtomicLong(0)).incrementAndGet();
        }
    }

    /**
     * Get op execution count for a device type.
     */
    public long getOpCount(DeviceType deviceType) {
        AtomicLong count = opCounts.get(deviceType);
        return count != null ? count.get() : 0;
    }

    /**
     * Get total op execution count across all backends.
     */
    public long getTotalOpCount() {
        return opCounts.values().stream().mapToLong(AtomicLong::get).sum();
    }

    /**
     * Reset execution statistics.
     */
    public void resetStatistics() {
        opCounts.values().forEach(c -> c.set(0));
    }

    /**
     * Log execution statistics.
     */
    public void logStatistics() {
        long total = getTotalOpCount();
        log.info("Multi-backend execution statistics:");
        log.info("  Total ops: {}", total);

        for (DeviceType type : DeviceType.values()) {
            long count = getOpCount(type);
            if (count > 0) {
                double percent = total > 0 ? 100.0 * count / total : 0;
                log.info("  {} ops: {} ({:.1f}%)", type, count, percent);
            }
        }
    }

    // Legacy accessor methods for backward compatibility
    public long getCpuOpCount() {
        return getOpCount(DeviceType.CPU);
    }

    public long getCudaOpCount() {
        return getOpCount(DeviceType.CUDA_GPU);
    }

    /**
     * Backend information holder.
     */
    private static class BackendInfo {
        final String className;
        final String displayName;
        final int priority;

        BackendInfo(String className, String displayName, int priority) {
            this.className = className;
            this.displayName = displayName;
            this.priority = priority;
        }
    }
}
