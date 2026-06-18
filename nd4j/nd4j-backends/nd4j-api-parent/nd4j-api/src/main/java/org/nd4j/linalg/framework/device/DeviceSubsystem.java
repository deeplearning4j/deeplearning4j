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

package org.nd4j.linalg.framework.device;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.device.DeviceContext;
import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.device.DeviceMemoryManager;
import org.nd4j.linalg.api.device.DeviceType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;
import java.util.Map;

/**
 * Device subsystem - provides access to device management, affinity control, and memory management.
 * 
 * Hierarchy:
 *   Nd4j.framework.device()
 *       ├── affinity()      - AffinityManager access
 *       ├── memoryManager() - DeviceMemoryManager access
 *       └── info()          - Device information
 * 
 * @author ND4J Team
 */
@Slf4j
public final class DeviceSubsystem {
    
    private final AffinityAccess affinity;
    private final DeviceMemoryManagerAccess memoryManager;
    private final DeviceInfoAccess info;
    private final DeviceEnvironmentAccess environment;
    private final TransferSubsystem transfers;
    private final DevicePinningManager pinning;
    private final ReplicaLeakDetector replicaLeaks;
    private final PointerStabilityGuard pointerStability;

    public DeviceSubsystem() {
        this.affinity = new AffinityAccess();
        this.memoryManager = new DeviceMemoryManagerAccess();
        this.info = new DeviceInfoAccess();
        this.environment = new DeviceEnvironmentAccess();
        this.transfers = new TransferSubsystem();
        this.pinning = new DevicePinningManager();
        this.replicaLeaks = new ReplicaLeakDetector();
        this.pointerStability = new PointerStabilityGuard();
    }

    /**
     * AffinityManager access for device/thread affinity control
     */
    public AffinityAccess affinity() {
        return affinity;
    }

    /**
     * DeviceMemoryManager access for device memory and routing control
     */
    public DeviceMemoryManagerAccess memoryManager() {
        return memoryManager;
    }

    /**
     * Device information access
     */
    public DeviceInfoAccess info() {
        return info;
    }

    /**
     * Device environment configuration
     */
    public DeviceEnvironmentAccess environment() {
        return environment;
    }

    /**
     * Transfer tracking subsystem
     */
    public TransferSubsystem transfers() {
        return transfers;
    }

    /**
     * Device pinning manager
     */
    public DevicePinningManager pinning() {
        return pinning;
    }

    /**
     * Replica leak detector
     */
    public ReplicaLeakDetector replicaLeaks() {
        return replicaLeaks;
    }

    /**
     * Pointer stability guard
     */
    public PointerStabilityGuard pointerStability() {
        return pointerStability;
    }

    // =========================================================================
    // AffinityManager Access
    // =========================================================================
    
    /**
     * AffinityManager access
     */
    public static class AffinityAccess {
        
        /**
         * Get the AffinityManager instance
         */
        public AffinityManager get() {
            return Nd4j.getAffinityManager();
        }
        
        /**
         * Get device ID for current thread
         */
        public int getDeviceForCurrentThread() {
            return Nd4j.getAffinityManager().getDeviceForCurrentThread();
        }
        
        /**
         * Get device ID for a specific thread
         */
        public int getDeviceForThread(long threadId) {
            return Nd4j.getAffinityManager().getDeviceForThread(threadId);
        }
        
        /**
         * Get device ID for an array
         */
        public int getDeviceForArray(INDArray array) {
            return Nd4j.getAffinityManager().getDeviceForArray(array);
        }
        
        /**
         * Get number of available devices
         */
        public int getNumberOfDevices() {
            return Nd4j.getAffinityManager().getNumberOfDevices();
        }
        
        /**
         * Get list of available device IDs
         */
        public List<Integer> getAvailableDeviceIds() {
            return Nd4j.getAffinityManager().getAvailableDeviceIds();
        }
        
        /**
         * Touch an array (associate with current device)
         */
        public void touch(INDArray array) {
            Nd4j.getAffinityManager().touch(array);
        }
        
        /**
         * Replicate array to target device
         */
        public INDArray replicateToDevice(int deviceId, INDArray array) {
            return Nd4j.getAffinityManager().replicateToDevice(deviceId, array);
        }
        
        /**
         * Tag array location
         */
        public void tagLocation(INDArray array, AffinityManager.Location location) {
            Nd4j.getAffinityManager().tagLocation(array, location);
        }
        
        /**
         * Ensure array is at specified location
         */
        public void ensureLocation(INDArray array, AffinityManager.Location location) {
            Nd4j.getAffinityManager().ensureLocation(array, location);
        }
        
        /**
         * Get active location of array
         */
        public AffinityManager.Location getActiveLocation(INDArray array) {
            return Nd4j.getAffinityManager().getActiveLocation(array);
        }
        
        /**
         * Get device descriptor
         */
        public DeviceDescriptor getDeviceDescriptor(int deviceId) {
            return Nd4j.getAffinityManager().getDeviceDescriptor(deviceId);
        }
        
        /**
         * Get device type
         */
        public DeviceType getDeviceType(int deviceId) {
            return Nd4j.getAffinityManager().getDeviceType(deviceId);
        }
        
        /**
         * Check if device is CUDA/GPU
         */
        public boolean isCudaDevice(int deviceId) {
            return getDeviceType(deviceId) == DeviceType.GPU;
        }
        
        /**
         * Check if device is CPU
         */
        public boolean isCpuDevice(int deviceId) {
            return getDeviceType(deviceId) == DeviceType.CPU;
        }
        
        /**
         * Set device for current thread (unsafe - use with caution)
         */
        public void setDeviceForCurrentThread(int deviceId) {
            Nd4j.getAffinityManager().setDeviceForCurrentThread(deviceId);
        }
        
        /**
         * Get device memory info (total, free, used)
         */
        public DeviceMemoryInfo getDeviceMemoryInfo(int deviceId) {
            try {
                AffinityManager am = Nd4j.getAffinityManager();
                DeviceDescriptor desc = am.getDeviceDescriptor(deviceId);
                if (desc != null) {
                    return DeviceMemoryInfo.builder()
                        .deviceId(deviceId)
                        .totalMemory(desc.getTotalMemory())
                        .build();
                }
            } catch (Exception e) {
                log.debug("Failed to get device memory info", e);
            }
            return DeviceMemoryInfo.builder().deviceId(deviceId).build();
        }
        
        /**
         * Get affinity manager type name
         */
        public String getType() {
            AffinityManager am = Nd4j.getAffinityManager();
            return am != null ? am.getClass().getSimpleName() : "null";
        }
    }
    
    // =========================================================================
    // DeviceMemoryManager Access
    // =========================================================================
    
    /**
     * DeviceMemoryManager access
     */
    public static class DeviceMemoryManagerAccess {
        
        /**
         * Get the DeviceMemoryManager instance
         */
        public DeviceMemoryManager get() {
            return DeviceMemoryManager.getInstance();
        }
        
        /**
         * Switch to specified device
         * @param deviceId Target device ID
         * @param caller Calling class/method (for tracing)
         * @param reason Reason for switch (for tracing)
         * @return DeviceContext for the switched device
         */
        public DeviceContext switchDevice(int deviceId, String caller, String reason) {
            return DeviceMemoryManager.getInstance().switchDevice(deviceId, caller, reason);
        }
        
        /**
         * Get current device ID for current thread
         */
        public int getCurrentDeviceId() {
            return DeviceMemoryManager.getInstance().getCurrentDeviceId();
        }
        
        /**
         * Get current device context
         */
        public DeviceContext getCurrentDeviceContext() {
            return DeviceMemoryManager.getInstance().getCurrentDeviceContext();
        }
        
        /**
         * Get fresh execution stream for current device
         */
        public org.bytedeco.javacpp.Pointer getFreshExecutionStream() {
            return DeviceMemoryManager.getInstance().getFreshExecutionStream();
        }
        
        /**
         * Get default device for allocations
         */
        public DeviceDescriptor getDefaultDevice() {
            return DeviceMemoryManager.getInstance().getDefaultDevice();
        }
        
        /**
         * Set default device
         */
        public void setDefaultDevice(DeviceDescriptor device) {
            DeviceMemoryManager.getInstance().setDefaultDevice(device);
        }
        
        /**
         * Get fallback device
         */
        public DeviceDescriptor getFallbackDevice() {
            return DeviceMemoryManager.getInstance().getFallbackDevice();
        }
        
        /**
         * Set memory cap for device
         * @param device Device descriptor
         * @param capBytes Memory cap in bytes (0 = unlimited)
         */
        public void setMemoryCap(DeviceDescriptor device, long capBytes) {
            DeviceMemoryManager.getInstance().setMemoryCap(device, capBytes);
        }
        
        /**
         * Get memory cap for device
         */
        public long getMemoryCap(DeviceDescriptor device) {
            return DeviceMemoryManager.getInstance().getMemoryCap(device);
        }
        
        /**
         * Set device priority (higher = preferred)
         */
        public void setDevicePriority(DeviceDescriptor device, int priority) {
            DeviceMemoryManager.getInstance().setDevicePriority(device, priority);
        }
        
        /**
         * Get device priority
         */
        public int getDevicePriority(DeviceDescriptor device) {
            return DeviceMemoryManager.getInstance().getDevicePriority(device);
        }
        
        /**
         * Select best device for allocation size
         * @param sizeBytes Allocation size in bytes
         * @return Selected device descriptor, or null if none available
         */
        public DeviceDescriptor selectDeviceForAllocation(long sizeBytes) {
            return DeviceMemoryManager.getInstance().selectDeviceForAllocation(sizeBytes);
        }
        
        /**
         * Get allocated memory for device
         */
        public long getAllocatedMemory(DeviceDescriptor device) {
            return DeviceMemoryManager.getInstance().getAllocatedMemory(device);
        }
        
        /**
         * Get peak memory for device
         */
        public long getPeakMemory(DeviceDescriptor device) {
            return DeviceMemoryManager.getInstance().getPeakMemory(device);
        }
        
        /**
         * Get actual free memory for device (from device)
         */
        public long getActualFreeMemory(DeviceDescriptor device) {
            return DeviceMemoryManager.getInstance().getActualFreeMemory(device);
        }
        
        /**
         * Get available memory for device (free - allocated, respecting cap)
         */
        public long getAvailableMemory(DeviceDescriptor device) {
            return DeviceMemoryManager.getInstance().getAvailableMemory(device);
        }
        
        /**
         * Check if device has memory pressure (>90% used)
         */
        public boolean hasMemoryPressure(DeviceDescriptor device) {
            return DeviceMemoryManager.getInstance().hasMemoryPressure(device);
        }
        
        /**
         * Enable/disable auto fallback when device is full
         */
        public void setAutoFallbackEnabled(boolean enabled) {
            DeviceMemoryManager.getInstance().setAutoFallbackEnabled(enabled);
        }
        
        /**
         * Check if auto fallback is enabled
         */
        public boolean isAutoFallbackEnabled() {
            return DeviceMemoryManager.getInstance().isAutoFallbackEnabled();
        }
        
        /**
         * Set memory pressure threshold (0.0-1.0)
         */
        public void setMemoryPressureThreshold(double threshold) {
            DeviceMemoryManager.getInstance().setMemoryPressureThreshold(threshold);
        }
        
        /**
         * Enable memory simulation mode (for testing)
         */
        public void enableMemorySimulation(boolean enabled) {
            DeviceMemoryManager.getInstance().setMemorySimulationEnabled(enabled);
        }
        
        /**
         * Set simulated free memory for device (testing only)
         */
        public void setSimulatedFreeMemory(int deviceId, long bytes) {
            DeviceMemoryManager.getInstance().setSimulatedFreeMemory(deviceId, bytes);
        }
        
        /**
         * Register memory pressure callback
         */
        public void registerMemoryPressureCallback(DeviceMemoryManager.MemoryPressureCallback callback) {
            DeviceMemoryManager.getInstance().registerMemoryPressureCallback(callback);
        }
        
        /**
         * Get device routing policy
         */
        public DeviceMemoryManager.DeviceRoutingPolicy getDeviceRoutingPolicy() {
            return DeviceMemoryManager.getInstance().getDeviceRoutingPolicy();
        }
        
        /**
         * Set device routing policy
         */
        public void setDeviceRoutingPolicy(DeviceMemoryManager.DeviceRoutingPolicy policy) {
            DeviceMemoryManager.getInstance().setDeviceRoutingPolicy(policy);
        }
        
        /**
         * Get memory summary for all devices
         */
        public String getMemorySummary() {
            return DeviceMemoryManager.getInstance().getMemorySummary();
        }
    }
    
    // =========================================================================
    // Device Environment Access
    // =========================================================================
    
    /**
     * Device environment configuration
     */
    public static class DeviceEnvironmentAccess {
        
        /**
         * Get default device ID for current thread
         */
        public int getDefaultDeviceId() {
            return org.nd4j.linalg.factory.Nd4j.getAffinityManager().getDeviceForCurrentThread();
        }
        
        /**
         * Set default device ID for current thread
         */
        public void setDefaultDeviceId(int deviceId) {
            org.nd4j.linalg.factory.Nd4j.getAffinityManager().setDeviceForCurrentThread(deviceId);
        }
        
        /**
         * Get number of available devices
         */
        public int getNumberOfDevices() {
            return org.nd4j.linalg.factory.Nd4j.getAffinityManager().getNumberOfDevices();
        }
        
        /**
         * Get device routing policy
         */
        public String getDeviceRoutingPolicy() {
            return DeviceMemoryManager.getInstance().getDeviceRoutingPolicy().name();
        }
        
        /**
         * Set device routing policy
         */
        public void setDeviceRoutingPolicy(String policy) {
            DeviceMemoryManager mgr = DeviceMemoryManager.getInstance();
            DeviceMemoryManager.DeviceRoutingPolicy p = 
                DeviceMemoryManager.DeviceRoutingPolicy.valueOf(policy.toUpperCase());
            mgr.setDeviceRoutingPolicy(p);
        }
        
        /**
         * Check if auto fallback is enabled
         */
        public boolean isAutoFallbackEnabled() {
            return DeviceMemoryManager.getInstance().isAutoFallbackEnabled();
        }
        
        /**
         * Enable/disable auto fallback
         */
        public void setAutoFallbackEnabled(boolean enabled) {
            DeviceMemoryManager.getInstance().setAutoFallbackEnabled(enabled);
        }
        
        /**
         * Get memory pressure threshold (0.0-1.0)
         */
        public double getMemoryPressureThreshold() {
            return DeviceMemoryManager.getInstance().getMemoryPressureThreshold();
        }
        
        /**
         * Set memory pressure threshold (0.0-1.0)
         */
        public void setMemoryPressureThreshold(double threshold) {
            DeviceMemoryManager.getInstance().setMemoryPressureThreshold(threshold);
        }
        
        /**
         * Get CUDA memory pool size (MB)
         */
        public int getCudaMemoryPoolSize() {
            return org.nd4j.linalg.factory.Nd4j.getEnvironment().cudaMemoryPoolSize();
        }
        
        /**
         * Set CUDA memory pool size (MB)
         */
        public void setCudaMemoryPoolSize(int sizeMB) {
            org.nd4j.linalg.factory.Nd4j.getEnvironment().setCudaMemoryPoolSize(sizeMB);
        }
        
        /**
         * Get summary of device environment settings
         */
        public String getSummary() {
            return String.format(
                "Device Environment: Devices=%d, Routing=%s, Auto Fallback=%s, " +
                "Pressure Threshold=%.1f%%, CUDA Pool=%dMB",
                getNumberOfDevices(),
                getDeviceRoutingPolicy(),
                isAutoFallbackEnabled() ? "on" : "off",
                getMemoryPressureThreshold() * 100,
                getCudaMemoryPoolSize()
            );
        }
    }
    
    // =========================================================================
    // Device Info Access
    // =========================================================================
    
    /**
     * Device information access
     */
    public static class DeviceInfoAccess {
        
        /**
         * Get number of available devices
         */
        public int count() {
            return Nd4j.getAffinityManager().getNumberOfDevices();
        }
        
        /**
         * Get device type name
         */
        public String getType(int deviceId) {
            DeviceType type = Nd4j.getAffinityManager().getDeviceType(deviceId);
            return type != null ? type.name() : "UNKNOWN";
        }
        
        /**
         * Get device descriptor
         */
        public DeviceDescriptor getDescriptor(int deviceId) {
            return Nd4j.getAffinityManager().getDeviceDescriptor(deviceId);
        }
        
        /**
         * Get device summary string
         */
        public String getSummary(int deviceId) {
            DeviceDescriptor desc = Nd4j.getAffinityManager().getDeviceDescriptor(deviceId);
            if (desc == null) {
                return "Device " + deviceId + ": Unknown";
            }
            
            return String.format("Device %d: %s (%.1f MB total)",
                deviceId,
                desc.getName() != null ? desc.getName() : desc.getType().name(),
                desc.getTotalMemory() / (1024.0 * 1024.0));
        }
        
        /**
         * Get all device summaries
         */
        public String[] getAllSummaries() {
            int count = count();
            String[] summaries = new String[count];
            for (int i = 0; i < count; i++) {
                summaries[i] = getSummary(i);
            }
            return summaries;
        }
        
        /**
         * Check if multi-device setup
         */
        public boolean isMultiDevice() {
            return count() > 1;
        }
        
        /**
         * Check if GPU available
         */
        public boolean hasGpu() {
            int count = count();
            for (int i = 0; i < count; i++) {
                if (getType(i).equals("GPU")) {
                    return true;
                }
            }
            return false;
        }
    }
    
    /**
     * Device memory information
     */
    @lombok.Data
    @lombok.Builder
    public static class DeviceMemoryInfo {
        private int deviceId;
        private long totalMemory;
        private long freeMemory;
        private long usedMemory;
        
        public long getUsedMemory() {
            return totalMemory - freeMemory;
        }
        
        public double getUtilizationPercent() {
            if (totalMemory <= 0) return 0.0;
            return (getUsedMemory() * 100.0) / totalMemory;
        }
    }
}
