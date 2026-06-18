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

import lombok.Builder;
import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.memory.MultiBackendWorkspaceManager;
import org.nd4j.linalg.api.ops.executioner.TransferMetrics;

import java.util.*;

/**
 * Configuration for device routing and automatic data transfer policies.
 *
 * This class configures how ND4J routes operations and data across devices,
 * including automatic transfer policies, memory thresholds, and device preferences.
 *
 * <p>Example usage:</p>
 * <pre>{@code
 * DeviceRoutingConfiguration config = DeviceRoutingConfiguration.builder()
 *     .defaultPolicy(DeviceRoutingPolicy.PREFER_GPU)
 *     .autoTransferEnabled(true)
 *     .transferThresholdBytes(1024 * 1024)  // 1MB minimum for async transfer
 *     .gpuMemoryCapFraction(0.8)            // Use 80% of GPU memory max
 *     .build();
 *
 * config.apply();  // Apply to global configuration
 * }</pre>
 *
 * Adam Gibson
 */
@Slf4j
@Getter
@Setter
@Builder
public class DeviceRoutingConfiguration {

    private static volatile DeviceRoutingConfiguration CURRENT;
    private static final Object LOCK = new Object();
    private static final DeviceMemoryManager.MemoryPressureCallback PRESSURE_CALLBACK =
            (device, utilization) -> MultiBackendWorkspaceManager.getInstance().handleMemoryPressure(device, utilization);

    // =========================================================================
    // Device Selection Policy
    // =========================================================================

    /**
     * Default routing policy for new allocations.
     */
    @Builder.Default
    private DeviceMemoryManager.DeviceRoutingPolicy defaultPolicy =
            DeviceMemoryManager.DeviceRoutingPolicy.MEMORY_PRIORITY;

    /**
     * Whether to automatically fall back to other devices when preferred is full.
     */
    @Builder.Default
    private boolean autoFallbackEnabled = true;

    /**
     * Preferred device types in order of preference.
     */
    @Builder.Default
    private List<DeviceType> preferredDeviceTypes = Arrays.asList(
            DeviceType.CUDA_GPU,
            DeviceType.METAL_GPU,
            DeviceType.CPU
    );

    // =========================================================================
    // Automatic Data Transfer
    // =========================================================================

    /**
     * Whether to automatically transfer data between devices as needed.
     * When enabled, passing a CPU array to a GPU op will automatically copy it.
     */
    @Builder.Default
    private boolean autoTransferEnabled = true;

    /**
     * Whether to use asynchronous transfers when possible.
     */
    @Builder.Default
    private boolean asyncTransferEnabled = true;

    /**
     * Minimum size in bytes for async transfer (smaller transfers use sync).
     */
    @Builder.Default
    private long asyncTransferThresholdBytes = 64 * 1024; // 64KB

    /**
     * Whether to cache transferred data on the target device.
     * If true, subsequent ops on the same device can reuse the cached copy.
     */
    @Builder.Default
    private boolean cacheTransferredData = true;

    /**
     * Maximum cached data per device in bytes.
     */
    @Builder.Default
    private long maxCachePerDevice = 256 * 1024 * 1024; // 256MB

    /**
     * Whether to prefetch data for known execution patterns.
     */
    @Builder.Default
    private boolean prefetchEnabled = true;

    /**
     * Prefetch lookahead depth (number of ops to look ahead).
     */
    @Builder.Default
    private int prefetchDepth = 2;

    // =========================================================================
    // Memory Management
    // =========================================================================

    /**
     * Fraction of GPU memory to use (0.0 to 1.0).
     */
    @Builder.Default
    private double gpuMemoryCapFraction = 0.9;

    /**
     * Fraction of CPU memory to use for ND4J (0.0 to 1.0).
     */
    @Builder.Default
    private double cpuMemoryCapFraction = 0.8;

    /**
     * Per-device memory caps in bytes (overrides fraction-based caps).
     */
    @Builder.Default
    private Map<String, Long> deviceMemoryCaps = new HashMap<>();

    /**
     * Memory pressure threshold (0.0 to 1.0).
     * When utilization exceeds this, eviction may be triggered.
     */
    @Builder.Default
    private double memoryPressureThreshold = 0.85;

    /**
     * Whether to evict cached data when under memory pressure.
     */
    @Builder.Default
    private boolean evictOnPressure = true;

    // =========================================================================
    // Op Execution
    // =========================================================================

    /**
     * Whether to route ops to the device where most inputs reside.
     */
    @Builder.Default
    private boolean routeOpsByInputLocation = true;

    /**
     * Minimum input size sum (bytes) to consider device routing.
     * Smaller ops just run on default device.
     */
    @Builder.Default
    private long minSizeForDeviceRouting = 1024; // 1KB

    /**
     * Per-op device overrides.
     * Maps op name patterns to preferred devices.
     */
    @Builder.Default
    private Map<String, DeviceDescriptor> opDeviceOverrides = new HashMap<>();

    /**
     * Ops that should always run on CPU (e.g., certain IO ops).
     */
    @Builder.Default
    private Set<String> cpuOnlyOps = new HashSet<>();

    /**
     * Ops that should always run on GPU if available.
     */
    @Builder.Default
    private Set<String> gpuPreferredOps = new HashSet<>();

    // =========================================================================
    // Synchronization
    // =========================================================================

    /**
     * Whether to synchronize devices after each op.
     * Disabling improves performance but may cause race conditions.
     */
    @Builder.Default
    private boolean syncAfterOp = false;

    /**
     * Whether to synchronize before reading results back to host.
     */
    @Builder.Default
    private boolean syncBeforeHostRead = true;

    /**
     * Whether to use events for fine-grained synchronization.
     */
    @Builder.Default
    private boolean useEvents = true;

    // =========================================================================
    // Logging and Debugging
    // =========================================================================

    /**
     * Log device routing decisions.
     */
    @Builder.Default
    private boolean logRoutingDecisions = false;

    /**
     * Log data transfers.
     */
    @Builder.Default
    private boolean logDataTransfers = false;

    /**
     * Collect transfer statistics.
     */
    @Builder.Default
    private boolean collectStatistics = true;

    /**
     * Warning threshold for transfer overhead (percentage of execution time).
     * When cumulative transfer time exceeds this percentage, warnings are logged.
     */
    @Builder.Default
    private double transferOverheadWarningThreshold = 10.0;

    /**
     * Minimum bytes for logging individual transfers.
     * Transfers smaller than this won't be logged individually.
     */
    @Builder.Default
    private long minBytesForTransferLogging = 1024 * 1024;  // 1MB

    /**
     * Log transfer metrics summary periodically (every N ops).
     * Set to 0 to disable periodic logging.
     */
    @Builder.Default
    private int logMetricsSummaryEveryNOps = 0;

    // =========================================================================
    // Application Methods
    // =========================================================================

    /**
     * Apply this configuration globally.
     */
    public void apply() {
        synchronized (LOCK) {
            CURRENT = this;
        }

        DeviceMemoryManager mgr = DeviceMemoryManager.getInstance();

        // Apply routing policy
        mgr.setAutoFallbackEnabled(autoFallbackEnabled);
        mgr.setMemoryPressureThreshold(memoryPressureThreshold);
        if (evictOnPressure) {
            mgr.removeMemoryPressureCallback(PRESSURE_CALLBACK);
            mgr.addMemoryPressureCallback(PRESSURE_CALLBACK);
        } else {
            mgr.removeMemoryPressureCallback(PRESSURE_CALLBACK);
        }

        // Apply memory caps
        for (DeviceDescriptor device : mgr.getRegisteredDevices()) {
            String id = device.getDeviceId();

            // Check for explicit cap
            if (deviceMemoryCaps.containsKey(id)) {
                mgr.setMemoryCap(device, deviceMemoryCaps.get(id));
            } else {
                // Apply fraction-based cap
                double fraction = device.getDeviceType() == DeviceType.CPU ?
                        cpuMemoryCapFraction : gpuMemoryCapFraction;
                mgr.setMemoryCapFraction(device, fraction);
            }
        }

        // Set up preferred device as default
        DeviceDescriptor preferredDefault = findPreferredDevice(mgr);
        if (preferredDefault != null) {
            mgr.setDefaultDevice(preferredDefault);
        }

        // Set CPU as fallback
        mgr.setFallbackDevice(DeviceDescriptor.cpu());

        // Configure transfer metrics
        TransferMetrics metrics = TransferMetrics.getInstance();
        metrics.setWarningThresholdPercent(transferOverheadWarningThreshold);
        metrics.setLogIndividualTransfers(logDataTransfers);
        metrics.setMinBytesForLogging(minBytesForTransferLogging);

        log.info("Applied device routing configuration: policy={}, autoTransfer={}, gpuCap={}%, transferWarnThreshold={}%",
                defaultPolicy, autoTransferEnabled, (int)(gpuMemoryCapFraction * 100), (int)transferOverheadWarningThreshold);
    }

    private DeviceDescriptor findPreferredDevice(DeviceMemoryManager mgr) {
        for (DeviceType type : preferredDeviceTypes) {
            for (DeviceDescriptor device : mgr.getRegisteredDevices()) {
                if (device.getDeviceType() == type && device.isAvailable()) {
                    return device;
                }
            }
        }
        return null;
    }

    /**
     * Get the current global configuration.
     */
    public static DeviceRoutingConfiguration current() {
        if (CURRENT == null) {
            synchronized (LOCK) {
                if (CURRENT == null) {
                    CURRENT = DeviceRoutingConfiguration.builder().build();
                }
            }
        }
        return CURRENT;
    }

    /**
     * Set the current global configuration.
     *
     * @param config the configuration to set as current
     */
    public static void setCurrent(DeviceRoutingConfiguration config) {
        synchronized (LOCK) {
            CURRENT = config;
        }
    }

    /**
     * Get the default configuration.
     *
     * @return a new default configuration instance
     */
    public static DeviceRoutingConfiguration defaultConfig() {
        return DeviceRoutingConfiguration.builder().build();
    }

    /**
     * Reset to default configuration.
     */
    public static void reset() {
        synchronized (LOCK) {
            CURRENT = DeviceRoutingConfiguration.builder().build();
            CURRENT.apply();
        }
    }

    // =========================================================================
    // Builder Helpers
    // =========================================================================

    /**
     * Set a memory cap for a specific device.
     */
    public DeviceRoutingConfiguration withDeviceMemoryCap(DeviceDescriptor device, long bytes) {
        deviceMemoryCaps.put(device.getDeviceId(), bytes);
        return this;
    }

    /**
     * Add an op that should always run on CPU.
     */
    public DeviceRoutingConfiguration withCpuOnlyOp(String opName) {
        cpuOnlyOps.add(opName);
        return this;
    }

    /**
     * Add an op that should prefer GPU.
     */
    public DeviceRoutingConfiguration withGpuPreferredOp(String opName) {
        gpuPreferredOps.add(opName);
        return this;
    }

    /**
     * Set device override for a specific op.
     */
    public DeviceRoutingConfiguration withOpDeviceOverride(String opPattern, DeviceDescriptor device) {
        opDeviceOverrides.put(opPattern, device);
        return this;
    }

    // =========================================================================
    // Preset Configurations
    // =========================================================================

    /**
     * Configuration optimized for GPU-heavy workloads.
     */
    public static DeviceRoutingConfiguration gpuOptimized() {
        return builder()
                .defaultPolicy(DeviceMemoryManager.DeviceRoutingPolicy.PREFER_GPU)
                .gpuMemoryCapFraction(0.95)
                .autoTransferEnabled(true)
                .asyncTransferEnabled(true)
                .prefetchEnabled(true)
                .prefetchDepth(3)
                .routeOpsByInputLocation(true)
                .build();
    }

    /**
     * Configuration optimized for CPU-only workloads.
     */
    public static DeviceRoutingConfiguration cpuOnly() {
        return builder()
                .defaultPolicy(DeviceMemoryManager.DeviceRoutingPolicy.PREFER_CPU)
                .preferredDeviceTypes(Collections.singletonList(DeviceType.CPU))
                .autoTransferEnabled(false)
                .gpuMemoryCapFraction(0)
                .build();
    }

    /**
     * Configuration for multi-GPU setups with load balancing.
     */
    public static DeviceRoutingConfiguration multiGpuBalanced() {
        return builder()
                .defaultPolicy(DeviceMemoryManager.DeviceRoutingPolicy.MOST_FREE)
                .gpuMemoryCapFraction(0.85)
                .autoTransferEnabled(true)
                .asyncTransferEnabled(true)
                .prefetchEnabled(true)
                .cacheTransferredData(true)
                .routeOpsByInputLocation(true)
                .build();
    }

    /**
     * Configuration for memory-constrained environments.
     */
    public static DeviceRoutingConfiguration memoryConstrained() {
        return builder()
                .gpuMemoryCapFraction(0.5)
                .cpuMemoryCapFraction(0.5)
                .memoryPressureThreshold(0.7)
                .evictOnPressure(true)
                .cacheTransferredData(false)
                .maxCachePerDevice(64 * 1024 * 1024) // 64MB
                .build();
    }

    /**
     * Debug configuration with verbose logging.
     */
    public static DeviceRoutingConfiguration debug() {
        return builder()
                .logRoutingDecisions(true)
                .logDataTransfers(true)
                .collectStatistics(true)
                .syncAfterOp(true)
                .transferOverheadWarningThreshold(5.0)  // Warn at 5%
                .minBytesForTransferLogging(0)  // Log all transfers
                .build();
    }

    /**
     * Configuration for multi-GPU with detailed transfer metrics.
     * Logs all significant transfers and warns when overhead is high.
     */
    public static DeviceRoutingConfiguration multiGpuWithMetrics() {
        return builder()
                .defaultPolicy(DeviceMemoryManager.DeviceRoutingPolicy.MOST_FREE)
                .gpuMemoryCapFraction(0.85)
                .autoTransferEnabled(true)
                .asyncTransferEnabled(true)
                .prefetchEnabled(true)
                .cacheTransferredData(true)
                .routeOpsByInputLocation(true)
                .logDataTransfers(true)
                .collectStatistics(true)
                .transferOverheadWarningThreshold(10.0)
                .minBytesForTransferLogging(10 * 1024 * 1024)  // Log transfers > 10MB
                .build();
    }
}
