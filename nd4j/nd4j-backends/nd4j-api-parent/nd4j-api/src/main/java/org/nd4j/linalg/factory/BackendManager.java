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

package org.nd4j.linalg.factory;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.device.DeviceMemoryManager;
import org.nd4j.linalg.api.device.DeviceRoutingConfiguration;
import org.nd4j.linalg.api.device.DeviceType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.CpuBackendLoader;
import org.nd4j.linalg.api.ops.executioner.DeviceAwareOpExecutioner;
import org.nd4j.linalg.api.ops.executioner.OpExecutionDelegator;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.nativeblas.MultiBackendNativeOpsHolder;
import org.nd4j.common.config.ND4JSystemProperties;

import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Supplier;
import java.util.stream.Collectors;

/**
 * Unified entry point for multi-backend configuration and management.
 *
 * <p>BackendManager provides seamless multi-backend support with:</p>
 * <ul>
 *   <li>Auto-initialization on Nd4j startup</li>
 *   <li>Priority-based device allocation with memory fallback</li>
 *   <li>Cross-device operation execution with automatic data transfer</li>
 *   <li>Scoped device forcing for specific code blocks</li>
 *   <li>Unified configuration via fluent API</li>
 * </ul>
 *
 * <p>Basic usage - everything works automatically:</p>
 * <pre>{@code
 * // Arrays are allocated on the best available device (GPU if present)
 * INDArray a = Nd4j.create(100, 100);
 *
 * // Check what's available
 * Nd4j.backends().info();
 * }</pre>
 *
 * <p>Configuration via fluent API:</p>
 * <pre>{@code
 * Nd4j.backends()
 *     .priority(DeviceType.CUDA_GPU, DeviceType.CPU)  // GPU first, CPU fallback
 *     .gpuMemoryFraction(0.9)                         // Use 90% of GPU memory
 *     .memoryFallback(true)                           // Auto-fallback on OOM
 *     .autoTransfer(true)                             // Auto cross-device transfer
 *     .apply();
 * }</pre>
 *
 * <p>Scoped device forcing:</p>
 * <pre>{@code
 * try (DeviceScope scope = Nd4j.backends().forceDevice(DeviceType.CPU)) {
 *     // All allocations in this scope go to CPU
 *     INDArray cpuArray = Nd4j.create(100, 100);
 * }
 * }</pre>
 *
 * Adam Gibson
 */
@Slf4j
public class BackendManager {

    private static volatile BackendManager INSTANCE;
    private static final Object LOCK = new Object();

    // System property keys — centralized in ND4JSystemProperties; kept here as aliases for backward compatibility
    public static final String PROP_PRIORITY = ND4JSystemProperties.BACKEND_PRIORITY;
    public static final String PROP_MEMORY_FALLBACK = ND4JSystemProperties.BACKEND_MEMORY_FALLBACK;
    public static final String PROP_GPU_MEMORY_FRACTION = ND4JSystemProperties.BACKEND_GPU_MEMORY_FRACTION;
    public static final String PROP_CPU_MEMORY_FRACTION = ND4JSystemProperties.BACKEND_CPU_MEMORY_FRACTION;
    public static final String PROP_AUTO_TRANSFER = ND4JSystemProperties.BACKEND_AUTO_TRANSFER;
    public static final String PROP_AUTO_INIT = ND4JSystemProperties.BACKEND_AUTO_INIT;

    // Delegates to existing singletons
    private final BackendRegistry registry;
    private final DeviceMemoryManager memoryManager;
    private final DeviceRoutingConfiguration routingConfig;

    // Configuration state
    private volatile List<DeviceType> devicePriority;
    private volatile boolean memoryFallbackEnabled = true;
    private volatile double gpuMemoryFraction = 0.9;
    private volatile double cpuMemoryFraction = 0.8;
    private volatile boolean autoTransferEnabled = true;
    private volatile boolean asyncTransferEnabled = true;

    // Initialization state
    private final AtomicBoolean initialized = new AtomicBoolean(false);
    private final AtomicBoolean autoInitEnabled = new AtomicBoolean(true);

    private BackendManager() {
        this.registry = BackendRegistry.getInstance();
        this.memoryManager = DeviceMemoryManager.getInstance();
        this.routingConfig = DeviceRoutingConfiguration.current();

        // Default priority: CUDA > ROCm > Metal > CPU
        this.devicePriority = Arrays.asList(
                DeviceType.CUDA_GPU,
                DeviceType.ROCM_GPU,
                DeviceType.METAL_GPU,
                DeviceType.TPU,
                DeviceType.CPU
        );

        // Load configuration from system properties
        loadSystemProperties();
    }

    /**
     * Get the singleton instance.
     *
     * @return the BackendManager instance
     */
    public static BackendManager getInstance() {
        if (INSTANCE == null) {
            synchronized (LOCK) {
                if (INSTANCE == null) {
                    INSTANCE = new BackendManager();
                }
            }
        }
        return INSTANCE;
    }

    // =========================================================================
    // Auto-Initialization
    // =========================================================================

    /**
     * Auto-initialize multi-backend support.
     * Called automatically during Nd4j initialization if enabled.
     *
     * <p>Multi-backend mode is enabled by default when multiple backends are on the classpath.
     * This can be disabled by setting the system property {@code org.nd4j.backend.multi.auto=false}.</p>
     */
    public void autoInitialize() {
        if (!autoInitEnabled.get()) {
            log.debug("Auto-initialization disabled via system property");
            return;
        }

        if (initialized.compareAndSet(false, true)) {
            log.debug("Auto-initializing BackendManager");

            // Initialize the registry (discovers backends and devices)
            registry.initialize();

            // Enable multi-backend mode by default (can be disabled via system property)
            // This discovers and initializes all available backends (CPU, CUDA, ZLUDA, TPU, etc.)
            boolean multiBackendDisabled = "false".equalsIgnoreCase(
                    System.getProperty(ND4JSystemProperties.MULTI_BACKEND_AUTO_ENABLED, "true"));

            if (!multiBackendDisabled) {
                boolean multiBackendAvailable = MultiBackendNativeOpsHolder.enableMultiBackend();
                if (multiBackendAvailable && MultiBackendNativeOpsHolder.isMultiBackendEnabled()) {
                    log.info("Multi-backend mode enabled: {} backend(s) available",
                            MultiBackendNativeOpsHolder.getInstance().getLoadedBackendCount());
                }
            } else {
                log.debug("Multi-backend mode disabled via system property");
            }

            // Configure memory caps
            configureMemoryCaps();

            // Set up routing configuration
            configureRouting();

            // Auto-install DeviceAwareOpExecutioner with CPU fallback if:
            // 1. Primary backend is GPU-based (CUDA, ROCm, etc.)
            // 2. CPU backend is available on classpath
            // This enables transparent CPU fallback for spillover scenarios
            autoInstallDeviceAwareExecutioner();

            // Log what we found
            logInitializationSummary();
        }
    }

    /**
     * Automatically install DeviceAwareOpExecutioner if conditions are met.
     * This enables transparent CPU fallback when GPU is the primary backend.
     */
    private void autoInstallDeviceAwareExecutioner() {
        try {
            // Check if primary executioner is GPU-based
            OpExecutioner currentExecutioner = Nd4j.getExecutioner();
            if (currentExecutioner == null) {
                return;
            }

            String execClassName = currentExecutioner.getClass().getName().toLowerCase();
            boolean isPrimaryGpu = execClassName.contains("cuda") ||
                    execClassName.contains("gpu") ||
                    execClassName.contains("rocm") ||
                    execClassName.contains("zluda");

            if (!isPrimaryGpu) {
                log.debug("Primary backend is CPU, no need for DeviceAwareOpExecutioner");
                return;
            }

            // Install DeviceAwareOpExecutioner for cross-device routing.
            // Multi-GPU: ops must execute on the device where data lives.
            // GPU+CPU: ops can spill to CPU when GPU is full.
            int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
            if (numDevices > 1) {
                // Multi-GPU: always install for cross-device op routing
                DeviceAwareOpExecutioner.install();
                log.info("DeviceAwareOpExecutioner installed for multi-GPU routing ({} devices)", numDevices);
            } else if (CpuBackendLoader.isCpuBackendAvailable()) {
                // Single GPU + CPU backend: install with multi-backend support
                boolean success = DeviceAwareOpExecutioner.installWithMultiBackend();
                if (success) {
                    log.info("DeviceAwareOpExecutioner installed with CPU fallback support");
                }
            }

        } catch (Exception e) {
            // Don't fail initialization if this doesn't work
            log.debug("Could not auto-install DeviceAwareOpExecutioner: {}", e.getMessage());
        }
    }

    /**
     * Explicitly initialize (re-initialize) the backend manager.
     * Use this after changing configuration to apply changes.
     *
     * @return this for chaining
     */
    public BackendManager initialize() {
        initialized.set(false);
        autoInitialize();
        return this;
    }

    /**
     * Check if the manager is initialized.
     *
     * @return true if initialized
     */
    public boolean isInitialized() {
        return initialized.get();
    }

    private void ensureInitialized() {
        if (!initialized.get()) {
            autoInitialize();
        }
    }

    private void loadSystemProperties() {
        // Priority
        String priorityProp = System.getProperty(PROP_PRIORITY);
        if (priorityProp != null && !priorityProp.isEmpty()) {
            List<DeviceType> types = new ArrayList<>();
            for (String s : priorityProp.split(",")) {
                try {
                    types.add(DeviceType.valueOf(s.trim().toUpperCase()));
                } catch (IllegalArgumentException e) {
                    // Try common aliases
                    String lower = s.trim().toLowerCase();
                    if ("cuda".equals(lower) || "gpu".equals(lower)) {
                        types.add(DeviceType.CUDA_GPU);
                    } else if ("cpu".equals(lower)) {
                        types.add(DeviceType.CPU);
                    } else if ("rocm".equals(lower)) {
                        types.add(DeviceType.ROCM_GPU);
                    }
                }
            }
            if (!types.isEmpty()) {
                devicePriority = types;
            }
        }

        // Memory fallback
        String fallbackProp = System.getProperty(PROP_MEMORY_FALLBACK);
        if (fallbackProp != null) {
            memoryFallbackEnabled = Boolean.parseBoolean(fallbackProp);
        }

        // GPU memory fraction
        String gpuMemProp = System.getProperty(PROP_GPU_MEMORY_FRACTION);
        if (gpuMemProp != null) {
            try {
                gpuMemoryFraction = Double.parseDouble(gpuMemProp);
            } catch (NumberFormatException e) {
                // Ignore
            }
        }

        // CPU memory fraction
        String cpuMemProp = System.getProperty(PROP_CPU_MEMORY_FRACTION);
        if (cpuMemProp != null) {
            try {
                cpuMemoryFraction = Double.parseDouble(cpuMemProp);
            } catch (NumberFormatException e) {
                // Ignore
            }
        }

        // Auto transfer
        String autoTransferProp = System.getProperty(PROP_AUTO_TRANSFER);
        if (autoTransferProp != null) {
            autoTransferEnabled = Boolean.parseBoolean(autoTransferProp);
        }

        // Auto init
        String autoInitProp = System.getProperty(PROP_AUTO_INIT);
        if (autoInitProp != null) {
            autoInitEnabled.set(Boolean.parseBoolean(autoInitProp));
        }
    }

    private void configureMemoryCaps() {
        for (DeviceDescriptor device : registry.getAllDevices()) {
            double fraction = device.getDeviceType().isGpu() ? gpuMemoryFraction : cpuMemoryFraction;
            if (fraction > 0 && fraction < 1.0) {
                memoryManager.setMemoryCapFraction(device, fraction);
            }
        }
    }

    private void configureRouting() {
        routingConfig.setAutoTransferEnabled(autoTransferEnabled);
        routingConfig.setAsyncTransferEnabled(asyncTransferEnabled);
        routingConfig.setAutoFallbackEnabled(memoryFallbackEnabled);
    }

    private void logInitializationSummary() {
        List<DeviceDescriptor> devices = registry.getAllDevices();
        log.info("BackendManager initialized: {} device(s) available", devices.size());

        for (DeviceDescriptor device : devices) {
            long totalMB = device.getTotalMemory() / (1024 * 1024);
            long capMB = memoryManager.getMemoryCap(device) / (1024 * 1024);
            log.info("  {} - {} MB total, {} MB cap",
                    device.getDeviceId(),
                    totalMB,
                    capMB > 0 ? capMB : "unlimited");
        }

        log.info("Priority order: {}", devicePriority.stream()
                .map(Enum::name)
                .collect(Collectors.joining(" > ")));

        // Log multi-backend execution status
        if (DeviceAwareOpExecutioner.isInstalled()) {
            DeviceAwareOpExecutioner exec = DeviceAwareOpExecutioner.getInstance();
            log.info("Multi-backend execution: ENABLED");
            log.info("  CPU fallback: {}", CpuBackendLoader.isCpuExecutionerLoaded() ? "Available" : "Not loaded");
            if (exec != null) {
                log.debug(exec.getBackendInfo());
            }
        } else {
            log.info("Multi-backend execution: Single backend mode");
        }
    }

    // =========================================================================
    // Fluent Configuration API
    // =========================================================================

    /**
     * Set device priority order for allocation.
     * Devices of the first type are preferred, falling back to subsequent types.
     *
     * @param types device types in priority order (highest first)
     * @return this for chaining
     */
    public BackendManager priority(DeviceType... types) {
        this.devicePriority = Arrays.asList(types);
        return this;
    }

    /**
     * Enable or disable automatic memory fallback.
     * When enabled, if allocation fails on preferred device, tries other devices.
     *
     * @param enabled true to enable fallback
     * @return this for chaining
     */
    public BackendManager memoryFallback(boolean enabled) {
        this.memoryFallbackEnabled = enabled;
        return this;
    }

    /**
     * Set GPU memory usage fraction (0.0 to 1.0).
     * Arrays won't be allocated on GPU if it would exceed this fraction.
     *
     * @param fraction fraction of GPU memory to use (e.g., 0.9 for 90%)
     * @return this for chaining
     */
    public BackendManager gpuMemoryFraction(double fraction) {
        if (fraction < 0 || fraction > 1) {
            throw new IllegalArgumentException("Fraction must be between 0 and 1");
        }
        this.gpuMemoryFraction = fraction;
        return this;
    }

    /**
     * Set CPU memory usage fraction (0.0 to 1.0).
     *
     * @param fraction fraction of CPU memory to use
     * @return this for chaining
     */
    public BackendManager cpuMemoryFraction(double fraction) {
        if (fraction < 0 || fraction > 1) {
            throw new IllegalArgumentException("Fraction must be between 0 and 1");
        }
        this.cpuMemoryFraction = fraction;
        return this;
    }

    /**
     * Enable or disable automatic cross-device data transfer.
     * When enabled, operations with inputs on different devices will
     * automatically transfer data to execute on the optimal device.
     *
     * @param enabled true to enable auto-transfer
     * @return this for chaining
     */
    public BackendManager autoTransfer(boolean enabled) {
        this.autoTransferEnabled = enabled;
        return this;
    }

    /**
     * Enable or disable asynchronous data transfer.
     * When enabled, large transfers happen in the background.
     *
     * @param enabled true to enable async transfer
     * @return this for chaining
     */
    public BackendManager asyncTransfer(boolean enabled) {
        this.asyncTransferEnabled = enabled;
        return this;
    }

    /**
     * Apply the current configuration.
     * Call this after making configuration changes to apply them.
     *
     * @return this for chaining
     */
    public BackendManager apply() {
        configureMemoryCaps();
        configureRouting();
        log.debug("BackendManager configuration applied");
        return this;
    }

    // =========================================================================
    // Device Queries
    // =========================================================================

    /**
     * Get all available devices.
     *
     * @return list of all devices
     */
    public List<DeviceDescriptor> devices() {
        ensureInitialized();
        return registry.getAllDevices();
    }

    /**
     * Get all GPU devices.
     *
     * @return list of GPU devices
     */
    public List<DeviceDescriptor> gpus() {
        ensureInitialized();
        return registry.getAllDevices().stream()
                .filter(d -> d.getDeviceType().isGpu())
                .collect(Collectors.toList());
    }

    /**
     * Get the default device for allocation.
     *
     * @return the default device
     */
    public DeviceDescriptor defaultDevice() {
        ensureInitialized();
        return registry.getDefaultDevice();
    }

    /**
     * Check if any GPU is available.
     *
     * @return true if a GPU is available
     */
    public boolean isGpuAvailable() {
        ensureInitialized();
        return registry.hasGpu();
    }

    /**
     * Check if CUDA is available.
     *
     * @return true if CUDA GPU is available
     */
    public boolean isCudaAvailable() {
        ensureInitialized();
        return !registry.getDevices(DeviceType.CUDA_GPU).isEmpty();
    }

    /**
     * Get a specific device by type and index.
     *
     * @param type  the device type
     * @param index the device index (0-based)
     * @return the device, or null if not found
     */
    public DeviceDescriptor device(DeviceType type, int index) {
        ensureInitialized();
        List<DeviceDescriptor> devices = registry.getDevices(type);
        return (devices != null && index < devices.size()) ? devices.get(index) : null;
    }

    /**
     * Get the effective device for an array.
     * Returns the pinned device if set, otherwise the owner device.
     *
     * @param array the array
     * @return the device, or CPU if unknown
     */
    public DeviceDescriptor getArrayDevice(INDArray array) {
        if (array == null) {
            return DeviceDescriptor.cpu();
        }
        DataBuffer buffer = array.data();
        if (buffer != null && buffer.isHybrid()) {
            DeviceDescriptor effective = buffer.asHybrid().getEffectiveDevice();
            return effective != null ? effective : DeviceDescriptor.cpu();
        }
        return DeviceDescriptor.cpu();
    }

    /**
     * Pin an array to a specific device.
     *
     * @param array the array to pin
     * @param device the device to pin to
     */
    public void pin(INDArray array, DeviceDescriptor device) {
        ensureInitialized();
        if (array == null || device == null) {
            return;
        }
        DataBuffer buffer = array.data();
        if (buffer != null && buffer.isHybrid()) {
            if (device.getDeviceType().isGpu() && !DeviceAwareOpExecutioner.isInstalled()) {
                DeviceAwareOpExecutioner.install();
            }
            buffer.asHybrid().pinTo(device);
        } else if (device.getDeviceType().isGpu()) {
            throw new IllegalArgumentException("Cannot pin non-hybrid array to " + device.getDeviceId());
        }
    }

    /**
     * Pin multiple arrays to a specific device.
     *
     * @param device the device to pin to
     * @param arrays the arrays to pin
     */
    public void pinAll(DeviceDescriptor device, INDArray... arrays) {
        ensureInitialized();
        if (arrays == null) {
            return;
        }
        for (INDArray array : arrays) {
            pin(array, device);
        }
    }

    /**
     * Remove pinning from an array.
     *
     * @param array the array to unpin
     */
    public void unpin(INDArray array) {
        ensureInitialized();
        if (array == null) {
            return;
        }
        DataBuffer buffer = array.data();
        if (buffer != null && buffer.isHybrid()) {
            buffer.asHybrid().unpin();
        }
    }

    /**
     * Get the pinned device for an array, if any.
     *
     * @param array the array to query
     * @return pinned device or null if not pinned
     */
    public DeviceDescriptor getPinnedDevice(INDArray array) {
        if (array == null) {
            return null;
        }
        DataBuffer buffer = array.data();
        if (buffer != null && buffer.isHybrid()) {
            return buffer.asHybrid().getPinnedDevice();
        }
        return null;
    }

    /**
     * Check if an array is pinned to a device.
     *
     * @param array the array to query
     * @return true if pinned
     */
    public boolean isPinned(INDArray array) {
        return getPinnedDevice(array) != null;
    }

    // =========================================================================
    // Device Info
    // =========================================================================

    /**
     * Get detailed information about a specific device.
     *
     * @param device the device
     * @return device information wrapper
     */
    public DeviceInfo deviceInfo(DeviceDescriptor device) {
        ensureInitialized();
        return new DeviceInfo(device, memoryManager, disabledDevices.contains(device.getDeviceId()));
    }

    /**
     * Get detailed information about all devices.
     *
     * @return list of device information
     */
    public List<DeviceInfo> deviceInfos() {
        ensureInitialized();
        return registry.getAllDevices().stream()
                .map(d -> new DeviceInfo(d, memoryManager, disabledDevices.contains(d.getDeviceId())))
                .collect(Collectors.toList());
    }

    /**
     * Get detailed information about the default device.
     *
     * @return default device information
     */
    public DeviceInfo defaultDeviceInfo() {
        return deviceInfo(defaultDevice());
    }

    /**
     * Print detailed device information to the console.
     */
    public void printDeviceInfo() {
        System.out.println(getDeviceInfoString());
    }

    /**
     * Get detailed device information as a formatted string.
     *
     * @return formatted device info string
     */
    public String getDeviceInfoString() {
        ensureInitialized();
        StringBuilder sb = new StringBuilder();
        sb.append("╔══════════════════════════════════════════════════════════════════╗\n");
        sb.append("║                     ND4J Device Information                      ║\n");
        sb.append("╠══════════════════════════════════════════════════════════════════╣\n");

        List<DeviceDescriptor> allDevices = registry.getAllDevices();
        DeviceDescriptor defaultDev = registry.getDefaultDevice();

        for (int i = 0; i < allDevices.size(); i++) {
            DeviceDescriptor device = allDevices.get(i);
            DeviceInfo info = deviceInfo(device);
            boolean isDefault = device.getDeviceId().equals(defaultDev.getDeviceId());
            boolean isDisabled = disabledDevices.contains(device.getDeviceId());

            sb.append(String.format("║ Device %d: %-54s ║\n", i, device.getDeviceName()));
            sb.append(String.format("║   ID:          %-50s ║\n", device.getDeviceId()));
            sb.append(String.format("║   Type:        %-50s ║\n", device.getDeviceType()));
            sb.append(String.format("║   Status:      %-50s ║\n",
                    isDisabled ? "DISABLED" : (isDefault ? "DEFAULT" : "Available")));
            sb.append(String.format("║   Total Memory:    %,15d MB                          ║\n",
                    device.getTotalMemory() / (1024 * 1024)));
            sb.append(String.format("║   Available:       %,15d MB                          ║\n",
                    device.getAvailableMemory() / (1024 * 1024)));
            sb.append(String.format("║   Allocated:       %,15d MB                          ║\n",
                    info.getAllocatedMemory() / (1024 * 1024)));

            long cap = info.getMemoryCap();
            if (cap > 0) {
                sb.append(String.format("║   Memory Cap:      %,15d MB                          ║\n",
                        cap / (1024 * 1024)));
            } else {
                sb.append(String.format("║   Memory Cap:      %-35s ║\n", "Unlimited"));
            }

            sb.append(String.format("║   Utilization:     %14.1f %%                          ║\n",
                    info.getUtilization() * 100));

            if (device.getComputeUnits() > 0) {
                sb.append(String.format("║   Compute Units:   %,15d                             ║\n",
                        device.getComputeUnits()));
            }

            if (device.getMemoryBandwidth() > 0) {
                sb.append(String.format("║   Memory BW:       %,15d GB/s                        ║\n",
                        device.getMemoryBandwidth() / (1024L * 1024 * 1024)));
            }

            // Capabilities
            Set<String> caps = device.getCapabilities();
            if (caps != null && !caps.isEmpty()) {
                String capsStr = String.join(", ", caps);
                if (capsStr.length() > 50) {
                    capsStr = capsStr.substring(0, 47) + "...";
                }
                sb.append(String.format("║   Capabilities:    %-46s ║\n", capsStr));
            }

            sb.append(String.format("║   Priority:        %,15d                             ║\n",
                    memoryManager.getDevicePriority(device)));

            if (i < allDevices.size() - 1) {
                sb.append("╟──────────────────────────────────────────────────────────────────╢\n");
            }
        }

        sb.append("╚══════════════════════════════════════════════════════════════════╝\n");

        // Summary
        sb.append("\nSummary:\n");
        sb.append(String.format("  Total devices: %d\n", allDevices.size()));
        sb.append(String.format("  GPU available: %s\n", isGpuAvailable() ? "Yes" : "No"));
        sb.append(String.format("  CUDA available: %s\n", isCudaAvailable() ? "Yes" : "No"));
        sb.append(String.format("  Default device: %s\n", defaultDev.getDeviceId()));
        sb.append(String.format("  Priority order: %s\n", devicePriority.stream()
                .map(Enum::name)
                .collect(Collectors.joining(" > "))));
        sb.append(String.format("  Memory fallback: %s\n", memoryFallbackEnabled ? "Enabled" : "Disabled"));
        sb.append(String.format("  Auto transfer: %s\n", autoTransferEnabled ? "Enabled" : "Disabled"));

        return sb.toString();
    }

    // =========================================================================
    // Device Management
    // =========================================================================

    // Set of disabled device IDs
    private final Set<String> disabledDevices = Collections.synchronizedSet(new HashSet<>());

    /**
     * Disable a device for allocation.
     * Disabled devices will not be selected for new allocations.
     *
     * @param device the device to disable
     * @return this for chaining
     */
    public BackendManager disableDevice(DeviceDescriptor device) {
        ensureInitialized();
        disabledDevices.add(device.getDeviceId());
        log.info("Disabled device: {}", device.getDeviceId());
        return this;
    }

    /**
     * Disable a device by type and index.
     *
     * @param type  the device type
     * @param index the device index
     * @return this for chaining
     */
    public BackendManager disableDevice(DeviceType type, int index) {
        DeviceDescriptor device = device(type, index);
        if (device != null) {
            disableDevice(device);
        }
        return this;
    }

    /**
     * Enable a previously disabled device.
     *
     * @param device the device to enable
     * @return this for chaining
     */
    public BackendManager enableDevice(DeviceDescriptor device) {
        ensureInitialized();
        disabledDevices.remove(device.getDeviceId());
        log.info("Enabled device: {}", device.getDeviceId());
        return this;
    }

    /**
     * Enable a device by type and index.
     *
     * @param type  the device type
     * @param index the device index
     * @return this for chaining
     */
    public BackendManager enableDevice(DeviceType type, int index) {
        DeviceDescriptor device = device(type, index);
        if (device != null) {
            enableDevice(device);
        }
        return this;
    }

    /**
     * Check if a device is enabled.
     *
     * @param device the device
     * @return true if enabled
     */
    public boolean isDeviceEnabled(DeviceDescriptor device) {
        return !disabledDevices.contains(device.getDeviceId());
    }

    /**
     * Get all enabled devices.
     *
     * @return list of enabled devices
     */
    public List<DeviceDescriptor> enabledDevices() {
        ensureInitialized();
        return registry.getAllDevices().stream()
                .filter(d -> !disabledDevices.contains(d.getDeviceId()))
                .collect(Collectors.toList());
    }

    /**
     * Get all disabled devices.
     *
     * @return list of disabled devices
     */
    public List<DeviceDescriptor> disabledDevices() {
        ensureInitialized();
        return registry.getAllDevices().stream()
                .filter(d -> disabledDevices.contains(d.getDeviceId()))
                .collect(Collectors.toList());
    }

    /**
     * Set the memory cap for a specific device.
     *
     * @param device   the device
     * @param maxBytes maximum bytes (0 = unlimited)
     * @return this for chaining
     */
    public BackendManager setDeviceMemoryCap(DeviceDescriptor device, long maxBytes) {
        ensureInitialized();
        memoryManager.setMemoryCap(device, maxBytes);
        log.info("Set memory cap for {}: {} MB", device.getDeviceId(), maxBytes / (1024 * 1024));
        return this;
    }

    /**
     * Set the memory cap as a fraction of total memory.
     *
     * @param device   the device
     * @param fraction fraction of total memory (0.0 to 1.0)
     * @return this for chaining
     */
    public BackendManager setDeviceMemoryFraction(DeviceDescriptor device, double fraction) {
        ensureInitialized();
        memoryManager.setMemoryCapFraction(device, fraction);
        log.info("Set memory fraction for {}: {}%", device.getDeviceId(), fraction * 100);
        return this;
    }

    /**
     * Set the priority for a specific device.
     * Higher priority devices are preferred for allocation.
     *
     * @param device   the device
     * @param priority priority value (higher = more preferred)
     * @return this for chaining
     */
    public BackendManager setDevicePriority(DeviceDescriptor device, int priority) {
        ensureInitialized();
        memoryManager.setDevicePriority(device, priority);
        log.info("Set priority for {}: {}", device.getDeviceId(), priority);
        return this;
    }

    /**
     * Set the default device for allocation.
     *
     * @param device the device to set as default
     * @return this for chaining
     */
    public BackendManager setDefaultDevice(DeviceDescriptor device) {
        ensureInitialized();
        memoryManager.setDefaultDevice(device);
        log.info("Set default device: {}", device.getDeviceId());
        return this;
    }

    /**
     * Set the default device by type and index.
     *
     * @param type  the device type
     * @param index the device index
     * @return this for chaining
     */
    public BackendManager setDefaultDevice(DeviceType type, int index) {
        DeviceDescriptor device = device(type, index);
        if (device != null) {
            setDefaultDevice(device);
        }
        return this;
    }

    /**
     * Reset all device configurations to defaults.
     *
     * @return this for chaining
     */
    public BackendManager resetDeviceConfigurations() {
        ensureInitialized();
        disabledDevices.clear();
        configureMemoryCaps();
        log.info("Reset all device configurations to defaults");
        return this;
    }

    // =========================================================================
    // Memory Management
    // =========================================================================

    /**
     * Select the best device for an allocation of the given size.
     * Uses priority order and falls back through all devices if needed.
     *
     * @param bytes number of bytes to allocate
     * @return the device to use for allocation
     */
    public DeviceDescriptor selectForAllocation(long bytes) {
        ensureInitialized();

        // 1. Try devices in priority order
        for (DeviceType type : devicePriority) {
            List<DeviceDescriptor> devices = registry.getDevices(type);
            if (devices != null) {
                for (DeviceDescriptor device : devices) {
                    if (memoryManager.canAllocate(device, bytes)) {
                        return device;
                    }
                }
            }
        }

        // 2. Try ALL devices regardless of priority (complete fallback chain)
        if (memoryFallbackEnabled) {
            for (DeviceDescriptor device : registry.getAllDevices()) {
                if (memoryManager.canAllocate(device, bytes)) {
                    log.warn("Priority devices full, falling back to {} for {} bytes",
                            device.getDeviceId(), bytes);
                    return device;
                }
            }

            // 3. All devices full - attempt GC and retry once
            log.warn("All devices low on memory, attempting garbage collection");
            gc();

            for (DeviceDescriptor device : registry.getAllDevices()) {
                if (memoryManager.canAllocate(device, bytes)) {
                    log.info("After GC, allocating on {}", device.getDeviceId());
                    return device;
                }
            }
        }

        // 4. Truly out of memory everywhere - throw with detailed info
        throw new OutOfMemoryError(buildOomMessage(bytes));
    }

    /**
     * Get memory statistics for all devices.
     *
     * @return map of device ID to memory stats
     */
    public Map<String, DeviceMemoryManager.DeviceMemoryStats> memoryStats() {
        ensureInitialized();
        return memoryManager.getMemoryStats();
    }

    /**
     * Trigger garbage collection across all devices.
     * Clears transfer caches and runs Java GC.
     */
    public void gc() {
        // Evict transfer caches
        try {
            OpExecutionDelegator.getInstance().clearAllCaches();
        } catch (Exception e) {
            log.debug("Failed to clear transfer caches", e);
        }

        // Run Java GC
        System.gc();

        // Brief pause for async deallocations
        try {
            Thread.sleep(50);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }

        // Backend-specific cleanup
        for (DeviceDescriptor device : registry.getAllDevices()) {
            try {
                reclaimMemory(device);
            } catch (Exception e) {
                log.debug("Failed to reclaim memory on {}", device.getDeviceId(), e);
            }
        }
    }

    /**
     * Attempt to reclaim unused memory on a device.
     *
     * @param device the device
     */
    public void reclaimMemory(DeviceDescriptor device) {
        // This is a hook for backend-specific cleanup
        // CUDA: cudaDeviceSynchronize + memory pool cleanup
        // For now, just run GC
        System.gc();
    }

    private String buildOomMessage(long bytes) {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("Cannot allocate %d bytes - all devices exhausted:%n", bytes));
        for (DeviceMemoryManager.DeviceMemoryStats stats : memoryManager.getMemoryStats().values()) {
            sb.append(String.format("  %s: %d/%d MB used (%.1f%%)%n",
                    stats.deviceId,
                    stats.allocated / (1024 * 1024),
                    stats.total / (1024 * 1024),
                    stats.utilization * 100));
        }
        return sb.toString();
    }

    // =========================================================================
    // Scoped Execution
    // =========================================================================

    /**
     * Create a scope that forces all allocations to a specific device type.
     * Use with try-with-resources.
     *
     * <pre>{@code
     * try (DeviceScope scope = Nd4j.backends().forceDevice(DeviceType.CPU)) {
     *     INDArray a = Nd4j.create(100, 100);  // On CPU
     * }
     * }</pre>
     *
     * @param type the device type to force
     * @return a closeable scope
     */
    public DeviceScope forceDevice(DeviceType type) {
        ensureInitialized();
        List<DeviceDescriptor> devices = registry.getDevices(type);
        DeviceDescriptor device = (devices != null && !devices.isEmpty())
                ? devices.get(0)
                : registry.getDefaultDevice();
        return new DeviceScope(device);
    }

    /**
     * Create a scope that forces all allocations to a specific device.
     *
     * @param device the device to force
     * @return a closeable scope
     */
    public DeviceScope forceDevice(DeviceDescriptor device) {
        ensureInitialized();
        return new DeviceScope(device);
    }

    /**
     * Execute code with a specific device.
     *
     * @param device   the device to use
     * @param supplier the code to execute
     * @param <T>      the return type
     * @return the result
     */
    public <T> T withDevice(DeviceDescriptor device, Supplier<T> supplier) {
        return MultiBackendContext.withDevice(device, supplier);
    }

    /**
     * Execute code with a specific device.
     *
     * @param device   the device to use
     * @param runnable the code to execute
     */
    public void withDevice(DeviceDescriptor device, Runnable runnable) {
        MultiBackendContext.withDevice(device, runnable);
    }

    /**
     * Execute code with a specific device type.
     *
     * @param type     the device type
     * @param supplier the code to execute
     * @param <T>      the return type
     * @return the result
     */
    public <T> T withDevice(DeviceType type, Supplier<T> supplier) {
        List<DeviceDescriptor> devices = registry.getDevices(type);
        DeviceDescriptor device = (devices != null && !devices.isEmpty())
                ? devices.get(0)
                : registry.getDefaultDevice();
        return MultiBackendContext.withDevice(device, supplier);
    }

    // =========================================================================
    // Info / Debug
    // =========================================================================

    /**
     * Print backend and device information to the log.
     */
    public void info() {
        ensureInitialized();

        log.info("=== ND4J Backend Information ===");
        log.info("Initialized: {}", initialized.get());
        log.info("Auto-transfer: {}", autoTransferEnabled);
        log.info("Memory fallback: {}", memoryFallbackEnabled);
        log.info("Priority: {}", devicePriority.stream()
                .map(Enum::name)
                .collect(Collectors.joining(" > ")));

        log.info("");
        log.info("Backends:");
        for (Map.Entry<String, Nd4jBackend> entry : registry.getBackends().entrySet()) {
            log.info("  {}: {}", entry.getKey(), entry.getValue().getClass().getSimpleName());
        }

        log.info("");
        log.info("Devices:");
        for (DeviceDescriptor device : registry.getAllDevices()) {
            DeviceMemoryManager.DeviceMemoryStats stats = memoryManager.getMemoryStats(device);
            log.info("  {} [{}]", device.getDeviceId(), device.getDeviceType());
            log.info("    Total: {} MB", stats.total / (1024 * 1024));
            log.info("    Allocated: {} MB", stats.allocated / (1024 * 1024));
            log.info("    Available: {} MB", stats.available / (1024 * 1024));
            log.info("    Cap: {} MB", stats.cap > 0 ? stats.cap / (1024 * 1024) : "unlimited");
            log.info("    Utilization: {:.1f}%", stats.utilization * 100);
        }

        log.info("");
        log.info("Default device: {}", registry.getDefaultDevice().getDeviceId());
    }

    /**
     * Get a summary string of backend information.
     *
     * @return summary string
     */
    public String summary() {
        ensureInitialized();

        StringBuilder sb = new StringBuilder();
        sb.append("BackendManager[");
        sb.append("devices=").append(registry.getAllDevices().size());
        sb.append(", gpuAvailable=").append(isGpuAvailable());
        sb.append(", default=").append(registry.getDefaultDevice().getDeviceId());
        sb.append(", priority=").append(devicePriority.stream()
                .map(Enum::name)
                .collect(Collectors.joining(">")));
        sb.append("]");
        return sb.toString();
    }

    @Override
    public String toString() {
        return summary();
    }

    // =========================================================================
    // DeviceScope Inner Class
    // =========================================================================

    /**
     * A scope that forces device selection for allocations.
     * Use with try-with-resources.
     */
    public static class DeviceScope implements AutoCloseable {
        private final MultiBackendContext context;

        DeviceScope(DeviceDescriptor device) {
            this.context = MultiBackendContext.builder()
                    .device(device)
                    .build();
        }

        /**
         * Get the device for this scope.
         *
         * @return the device
         */
        public DeviceDescriptor getDevice() {
            return context.getPreferredDevice();
        }

        @Override
        public void close() {
            context.close();
        }
    }

    // =========================================================================
    // DeviceInfo Inner Class
    // =========================================================================

    /**
     * Wrapper class providing detailed information about a device including
     * current memory statistics and management status.
     */
    public static class DeviceInfo {
        private final DeviceDescriptor device;
        private final DeviceMemoryManager memoryManager;
        private final boolean disabled;

        DeviceInfo(DeviceDescriptor device, DeviceMemoryManager memoryManager, boolean disabled) {
            this.device = device;
            this.memoryManager = memoryManager;
            this.disabled = disabled;
        }

        /**
         * Get the underlying device descriptor.
         *
         * @return the device descriptor
         */
        public DeviceDescriptor getDevice() {
            return device;
        }

        /**
         * Get the device ID.
         *
         * @return device ID
         */
        public String getDeviceId() {
            return device.getDeviceId();
        }

        /**
         * Get the device name.
         *
         * @return device name
         */
        public String getDeviceName() {
            return device.getDeviceName();
        }

        /**
         * Get the device type.
         *
         * @return device type
         */
        public DeviceType getDeviceType() {
            return device.getDeviceType();
        }

        /**
         * Get the device index within its type.
         *
         * @return device index
         */
        public int getDeviceIndex() {
            return device.getDeviceIndex();
        }

        /**
         * Get total memory on this device.
         *
         * @return total memory in bytes
         */
        public long getTotalMemory() {
            return device.getTotalMemory();
        }

        /**
         * Get available (free) memory on this device.
         *
         * @return available memory in bytes
         */
        public long getAvailableMemory() {
            return device.getAvailableMemory();
        }

        /**
         * Get currently allocated memory on this device.
         *
         * @return allocated memory in bytes
         */
        public long getAllocatedMemory() {
            DeviceMemoryManager.DeviceMemoryStats stats = memoryManager.getMemoryStats(device);
            return stats != null ? stats.allocated : 0;
        }

        /**
         * Get the memory cap for this device.
         *
         * @return memory cap in bytes (0 = unlimited)
         */
        public long getMemoryCap() {
            return memoryManager.getMemoryCap(device);
        }

        /**
         * Get memory utilization as a fraction (0.0 to 1.0).
         *
         * @return utilization fraction
         */
        public double getUtilization() {
            DeviceMemoryManager.DeviceMemoryStats stats = memoryManager.getMemoryStats(device);
            return stats != null ? stats.utilization : 0.0;
        }

        /**
         * Check if this device is enabled for allocation.
         *
         * @return true if enabled
         */
        public boolean isEnabled() {
            return !disabled;
        }

        /**
         * Check if this device is disabled.
         *
         * @return true if disabled
         */
        public boolean isDisabled() {
            return disabled;
        }

        /**
         * Check if this device is currently available.
         *
         * @return true if available
         */
        public boolean isAvailable() {
            return device.isAvailable() && !disabled;
        }

        /**
         * Check if this is the default device.
         *
         * @return true if default
         */
        public boolean isDefault() {
            return device.isDefault();
        }

        /**
         * Get the number of compute units (cores, SMs, etc.).
         *
         * @return number of compute units
         */
        public int getComputeUnits() {
            return device.getComputeUnits();
        }

        /**
         * Get the memory bandwidth in bytes per second.
         *
         * @return memory bandwidth
         */
        public long getMemoryBandwidth() {
            return device.getMemoryBandwidth();
        }

        /**
         * Get estimated FLOPS for this device.
         *
         * @return estimated FLOPS
         */
        public long getEstimatedFlops() {
            return device.getEstimatedFlops();
        }

        /**
         * Get the compute capability score.
         *
         * @return compute capability
         */
        public double getComputeCapability() {
            return device.getComputeCapability();
        }

        /**
         * Get device capabilities.
         *
         * @return set of capability strings
         */
        public Set<String> getCapabilities() {
            return device.getCapabilities();
        }

        /**
         * Check if device has a specific capability.
         *
         * @param capability the capability to check
         * @return true if supported
         */
        public boolean hasCapability(String capability) {
            return device.hasCapability(capability);
        }

        /**
         * Get the device priority for allocation.
         *
         * @return priority value
         */
        public int getPriority() {
            return memoryManager.getDevicePriority(device);
        }

        /**
         * Get the backend ID that owns this device.
         *
         * @return backend ID
         */
        public String getBackendId() {
            return device.getBackendId();
        }

        /**
         * Get device properties.
         *
         * @return map of properties
         */
        public Map<String, Object> getProperties() {
            return device.getProperties();
        }

        /**
         * Get a specific property.
         *
         * @param key the property key
         * @return the property value, or null
         */
        public Object getProperty(String key) {
            return device.getProperty(key);
        }

        /**
         * Check if the device can allocate the specified bytes.
         *
         * @param bytes bytes to allocate
         * @return true if allocation is possible
         */
        public boolean canAllocate(long bytes) {
            return !disabled && memoryManager.canAllocate(device, bytes);
        }

        /**
         * Get a formatted summary of this device.
         *
         * @return formatted summary string
         */
        public String toSummary() {
            StringBuilder sb = new StringBuilder();
            sb.append(String.format("%s (%s)%n", getDeviceName(), getDeviceId()));
            sb.append(String.format("  Type: %s%n", getDeviceType()));
            sb.append(String.format("  Status: %s%n", isEnabled() ? (isDefault() ? "DEFAULT" : "Available") : "DISABLED"));
            sb.append(String.format("  Memory: %,d/%,d MB (%.1f%% used)%n",
                    getAllocatedMemory() / (1024 * 1024),
                    getTotalMemory() / (1024 * 1024),
                    getUtilization() * 100));

            long cap = getMemoryCap();
            if (cap > 0) {
                sb.append(String.format("  Memory Cap: %,d MB%n", cap / (1024 * 1024)));
            }

            if (getComputeUnits() > 0) {
                sb.append(String.format("  Compute Units: %,d%n", getComputeUnits()));
            }

            Set<String> caps = getCapabilities();
            if (caps != null && !caps.isEmpty()) {
                sb.append(String.format("  Capabilities: %s%n", String.join(", ", caps)));
            }

            return sb.toString();
        }

        @Override
        public String toString() {
            return String.format("DeviceInfo[%s, type=%s, memory=%dMB/%dMB, enabled=%s]",
                    getDeviceId(),
                    getDeviceType(),
                    getAllocatedMemory() / (1024 * 1024),
                    getTotalMemory() / (1024 * 1024),
                    isEnabled());
        }
    }
}
