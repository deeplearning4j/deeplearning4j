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

package org.nd4j.linalg.framework.memory;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.allocator.impl.MemoryTracker;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.api.device.DeviceMemoryManager;

/**
 * Memory environment access for ND4J.
 * Provides access to memory configuration and environment settings.
 */
@Slf4j
public class MemoryEnvironmentAccess {
    
    private static final MemoryEnvironmentAccess INSTANCE = new MemoryEnvironmentAccess();
    
    /** Default memory fraction */
    private float defaultMemoryFraction = 0.9f;
    
    /** Minimum memory fraction */
    private float minMemoryFraction = 0.1f;
    
    /** Maximum memory fraction */
    private float maxMemoryFraction = 0.95f;
    
    /** Whether to use workspace */
    private boolean useWorkspace = true;
    
    /** Whether to enable leak detection */
    private boolean leakDetectionEnabled = false;
    
    /**
     * Default constructor
     */
    public MemoryEnvironmentAccess() {
        // Singleton constructor
        // Read system properties for configuration
        String memFrac = System.getProperty(ND4JSystemProperties.MEMORY_FRACTION);
        if (memFrac != null) {
            try {
                defaultMemoryFraction = Float.parseFloat(memFrac);
            } catch (NumberFormatException e) {
                log.warn("Invalid memory fraction: {}", memFrac);
            }
        }
    }
    
    /**
     * Get the singleton instance.
     */
    public static MemoryEnvironmentAccess getInstance() {
        return INSTANCE;
    }
    
    /**
     * Get default memory fraction.
     */
    public float getDefaultMemoryFraction() {
        return defaultMemoryFraction;
    }
    
    /**
     * Set default memory fraction.
     */
    public void setDefaultMemoryFraction(float fraction) {
        if (fraction < minMemoryFraction || fraction > maxMemoryFraction) {
            throw new IllegalArgumentException(
                "Memory fraction must be between " + minMemoryFraction + " and " + maxMemoryFraction);
        }
        this.defaultMemoryFraction = fraction;
    }
    
    /**
     * Get minimum memory fraction.
     */
    public float getMinMemoryFraction() {
        return minMemoryFraction;
    }
    
    /**
     * Get maximum memory fraction.
     */
    public float getMaxMemoryFraction() {
        return maxMemoryFraction;
    }
    
    /**
     * Check if workspace is enabled.
     */
    public boolean isWorkspaceEnabled() {
        return useWorkspace;
    }
    
    /**
     * Enable or disable workspace.
     */
    public void setWorkspaceEnabled(boolean enabled) {
        this.useWorkspace = enabled;
    }
    
    /**
     * Check if leak detection is enabled.
     */
    public boolean isLeakDetectionEnabled() {
        return leakDetectionEnabled;
    }
    
    /**
     * Enable or disable leak detection.
     */
    public void setLeakDetectionEnabled(boolean enabled) {
        this.leakDetectionEnabled = enabled;
    }
    
    /**
     * Get current memory usage from MemoryTracker for default device.
     */
    public long getUsedMemory() {
        MemoryTracker tracker = MemoryTracker.getInstance();
        if (tracker == null) return 0;
        String deviceIdStr = DeviceMemoryManager.getInstance().getDefaultDevice().getDeviceId();
        int deviceId = parseDeviceIndex(deviceIdStr);
        return tracker.getAllocatedAmount(deviceId);
    }
    
    /**
     * Get total available memory from MemoryTracker for default device.
     */
    public long getTotalMemory() {
        MemoryTracker tracker = MemoryTracker.getInstance();
        if (tracker == null) return Runtime.getRuntime().maxMemory();
        String deviceIdStr = DeviceMemoryManager.getInstance().getDefaultDevice().getDeviceId();
        int deviceId = parseDeviceIndex(deviceIdStr);
        return tracker.getTotalMemory(deviceId);
    }
    
    /**
     * Get free memory from MemoryTracker for default device.
     */
    public long getFreeMemory() {
        MemoryTracker tracker = MemoryTracker.getInstance();
        if (tracker == null) return Runtime.getRuntime().freeMemory();
        String deviceIdStr = DeviceMemoryManager.getInstance().getDefaultDevice().getDeviceId();
        int deviceId = parseDeviceIndex(deviceIdStr);
        return tracker.getPreciseFreeMemory(deviceId);
    }
    
    /**
     * Parse device index from device ID string (e.g., "cuda:gpu:0" -> 0).
     */
    private int parseDeviceIndex(String deviceId) {
        if (deviceId == null || deviceId.isEmpty()) return 0;
        String[] parts = deviceId.split(":");
        if (parts.length >= 3) {
            try {
                return Integer.parseInt(parts[2]);
            } catch (NumberFormatException e) {
                log.warn("Failed to parse device index from: {}", deviceId);
            }
        }
        return 0;
    }
    
    /**
     * Get memory usage percentage.
     */
    public double getMemoryUsagePercent() {
        long total = getTotalMemory();
        return total > 0 ? (double) getUsedMemory() / total * 100 : 0;
    }
    
    /**
     * Get memory status string.
     */
    public String getMemoryStatus() {
        return String.format("Memory: used=%d MB, free=%d MB, total=%d MB, usage=%.1f%%",
            getUsedMemory() / (1024 * 1024),
            getFreeMemory() / (1024 * 1024),
            getTotalMemory() / (1024 * 1024),
            getMemoryUsagePercent());
    }
}
