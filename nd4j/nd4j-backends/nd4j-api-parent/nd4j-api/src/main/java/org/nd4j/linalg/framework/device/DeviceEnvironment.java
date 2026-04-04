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

import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.api.device.DeviceMemoryManager;
import org.nd4j.linalg.factory.Environment;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Device subsystem environment configuration.
 * 
 * Provides unified access to device-related environment settings
 * including affinity, device routing, and memory management.
 * 
 * Usage:
 * <pre>
 *   Nd4j.framework.device().environment().getDefaultDeviceId()
 *   Nd4j.framework.device().environment().setDeviceRoutingPolicy("MEMORY_BASED")
 * </pre>
 * 
 * @author ND4J Team
 */
public final class DeviceEnvironment {
    
    private DeviceEnvironment() {}
    
    // ========================================================================
    // Affinity Configuration
    // ========================================================================
    
    /**
     * Get default device ID for current thread
     */
    public int getDefaultDeviceId() {
        return Nd4j.getAffinityManager().getDeviceForCurrentThread();
    }
    
    /**
     * Set default device ID for current thread
     */
    public void setDefaultDeviceId(int deviceId) {
        Nd4j.getAffinityManager().setDeviceForCurrentThread(deviceId);
    }
    
    /**
     * Get number of available devices
     */
    public int getNumberOfDevices() {
        return Nd4j.getAffinityManager().getNumberOfDevices();
    }
    
    /**
     * Check if multi-device mode is enabled
     */
    public boolean isMultiDeviceEnabled() {
        return getNumberOfDevices() > 1;
    }
    
    // ========================================================================
    // Device Memory Manager Configuration
    // ========================================================================
    
    /**
     * Get device routing policy
     */
    public String getDeviceRoutingPolicy() {
        DeviceMemoryManager mgr = DeviceMemoryManager.getInstance();
        return mgr.getDeviceRoutingPolicy().name();
    }
    
    /**
     * Set device routing policy (MEMORY_BASED, PRIORITY_BASED, ROUND_ROBIN)
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
     * Enable/disable auto fallback when device is full
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
     * Get default device
     */
    public String getDefaultDevice() {
        return DeviceMemoryManager.getInstance().getDefaultDevice().toString();
    }
    
    /**
     * Get fallback device
     */
    public String getFallbackDevice() {
        return DeviceMemoryManager.getInstance().getFallbackDevice().toString();
    }
    
    // ========================================================================
    // CUDA-Specific Configuration
    // ========================================================================
    
    /**
     * Get CUDA memory pool size (MB)
     */
    public int getCudaMemoryPoolSize() {
        return Nd4j.getEnvironment().cudaMemoryPoolSize();
    }
    
    /**
     * Set CUDA memory pool size (MB)
     */
    public void setCudaMemoryPoolSize(int sizeMB) {
        Nd4j.getEnvironment().setCudaMemoryPoolSize(sizeMB);
    }
    
    /**
     * Get CUDA limit type value
     */
    public long getCudaLimit(int limitType) {
        return Nd4j.getEnvironment().cudaGetLimit(limitType);
    }
    
    /**
     * Set CUDA limit type value
     */
    public void setCudaLimit(int limitType, long value) {
        Nd4j.getEnvironment().cudaSetLimit(limitType, value);
    }
    
    /**
     * Get CUDA stack size
     */
    public long getCudaStackSize() {
        return getCudaLimit(Environment.CUDA_LIMIT_STACK_SIZE);
    }
    
    /**
     * Set CUDA stack size
     */
    public void setCudaStackSize(long size) {
        setCudaLimit(Environment.CUDA_LIMIT_STACK_SIZE, size);
    }
    
    /**
     * Get CUDA malloc heap size
     */
    public long getCudaMallocHeapSize() {
        return getCudaLimit(Environment.CUDA_LIMIT_MALLOC_HEAP_SIZE);
    }
    
    /**
     * Set CUDA malloc heap size
     */
    public void setCudaMallocHeapSize(long size) {
        setCudaLimit(Environment.CUDA_LIMIT_MALLOC_HEAP_SIZE, size);
    }
    
    // ========================================================================
    // System Properties
    // ========================================================================
    
    /**
     * Check if CUDA context per thread is enabled
     */
    public boolean isCudaContextPerThread() {
        return Boolean.getBoolean(ND4JSystemProperties.CUDA_CONTEXT_PER_THREAD);
    }
    
    /**
     * Enable/disable CUDA context per thread
     */
    public void setCudaContextPerThread(boolean enabled) {
        System.setProperty(ND4JSystemProperties.CUDA_CONTEXT_PER_THREAD, String.valueOf(enabled));
    }
    
    /**
     * Get CUDA use stream memory mode
     */
    public boolean isCudaUseStreamMemory() {
        return Boolean.getBoolean(ND4JSystemProperties.CUDA_USE_STREAM_MEMORY);
    }
    
    /**
     * Enable/disable CUDA use stream memory
     */
    public void setCudaUseStreamMemory(boolean enabled) {
        System.setProperty(ND4JSystemProperties.CUDA_USE_STREAM_MEMORY, String.valueOf(enabled));
    }
    
    // ========================================================================
    // Summary
    // ========================================================================
    
    /**
     * Get all device environment settings as summary
     */
    public String getSummary() {
        return String.format(
            "Device Environment: Devices=%d, Default=%s, Routing=%s, " +
            "Auto Fallback=%s, Pressure Threshold=%.1f%%, " +
            "CUDA Pool=%dMB, CUDA Stack=%dKB",
            getNumberOfDevices(),
            getDefaultDevice(),
            getDeviceRoutingPolicy(),
            isAutoFallbackEnabled() ? "on" : "off",
            getMemoryPressureThreshold() * 100,
            getCudaMemoryPoolSize(),
            getCudaStackSize() / 1024
        );
    }
}
