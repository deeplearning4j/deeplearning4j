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
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.memory.BasicMemoryManager;
import org.nd4j.linalg.api.memory.MemoryManager;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.deallocation.DeallocatorService;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.framework.history.*;
import org.nd4j.linalg.framework.stats.MemoryStats;

import java.util.List;

/**
 * Memory subsystem - provides access to memory management, allocation tracking, and diagnostics.
 * 
 * Hierarchy:
 *   Nd4j.framework.memory()
 *       ├── manager() - Direct MemoryManager access
 *       ├── deallocator() - DeallocatorService access
 *       ├── tracker() - MemoryTracker access
 *       ├── samples() - Historical sampling
 *       └── stats() - Current statistics
 * 
 * @author ND4J Team
 */
@Slf4j
public final class MemorySubsystem {
    
    private final MemoryManagerAccess manager;
    private final DeallocatorAccess deallocator;
    private final MemoryTrackerAccess tracker;
    private final MemorySamplingAccess samples;
    private final MemoryEnvironmentAccess environment;
    
    public MemorySubsystem() {
        this.manager = new MemoryManagerAccess();
        this.deallocator = new DeallocatorAccess();
        this.tracker = new MemoryTrackerAccess();
        this.samples = new MemorySamplingAccess();
        this.environment = new MemoryEnvironmentAccess();
    }
    
    /**
     * Direct MemoryManager access
     */
    public MemoryManagerAccess manager() {
        return manager;
    }
    
    /**
     * DeallocatorService access
     */
    public DeallocatorAccess deallocator() {
        return deallocator;
    }
    
    /**
     * MemoryTracker access
     */
    public MemoryTrackerAccess tracker() {
        return tracker;
    }
    
    /**
     * Historical memory sampling
     */
    public MemorySamplingAccess samples() {
        return samples;
    }
    
    /**
     * Memory environment configuration
     */
    public MemoryEnvironmentAccess environment() {
        return environment;
    }
    
    /**
     * Get comprehensive memory statistics
     */
    public MemoryStats stats() {
        Runtime runtime = Runtime.getRuntime();
        long heapUsed = runtime.totalMemory() - runtime.freeMemory();
        long heapMax = runtime.maxMemory();
        
        MemoryTracker memTracker = MemoryTracker.getInstance();
        long offHeapUsed = memTracker.getAllocatedHostAmount() + memTracker.getCachedHostAmount();
        
        long deviceUsed = 0;
        try {
            int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
            for (int i = 0; i < numDevices; i++) {
                deviceUsed += memTracker.getActiveMemory(i);
            }
        } catch (Exception e) {
            log.debug("Failed to get device memory", e);
        }
        
        long workspaceUsed = 0;
        try {
            MemoryWorkspace ws = Nd4j.getMemoryManager().getCurrentWorkspace();
            if (ws != null) {
                workspaceUsed = ws.getCurrentSize();
            }
        } catch (Exception e) {
            log.debug("Failed to get workspace memory", e);
        }
        
        return MemoryStats.builder()
            .heapUsedBytes(heapUsed)
            .heapMaxBytes(heapMax)
            .offHeapUsedBytes(offHeapUsed)
            .deviceUsedBytes(deviceUsed)
            .workspaceUsedBytes(workspaceUsed)
            .poolReservedBytes(0) // Would come from native calls
            .build();
    }
    
    /**
     * Trigger GC if heap is pressured
     */
    public void gcIfPressured() {
        MemoryManager mm = Nd4j.getMemoryManager();
        if (mm != null) {
            mm.gcIfHeapPressured();
        }
    }
    
    /**
     * Get memory growth trend (bytes/second)
     */
    public long getGrowthTrend(int windowSeconds) {
        return MemorySampler.getInstance().getMemoryGrowthTrend(windowSeconds);
    }
    
    // =========================================================================
    // MemoryManager Access
    // =========================================================================
    
    /**
     * Direct MemoryManager access
     */
    public static class MemoryManagerAccess {
        
        /**
         * Get the current MemoryManager instance
         */
        public MemoryManager get() {
            return Nd4j.getMemoryManager();
        }
        
        /**
         * Get current workspace for thread
         */
        public MemoryWorkspace getCurrentWorkspace() {
            MemoryManager mm = Nd4j.getMemoryManager();
            return mm != null ? mm.getCurrentWorkspace() : null;
        }
        
        /**
         * Set current workspace
         */
        public void setCurrentWorkspace(MemoryWorkspace workspace) {
            MemoryManager mm = Nd4j.getMemoryManager();
            if (mm != null) {
                mm.setCurrentWorkspace(workspace);
            }
        }
        
        /**
         * Invoke GC
         */
        public void invokeGc() {
            MemoryManager mm = Nd4j.getMemoryManager();
            if (mm != null) {
                mm.invokeGc();
            }
        }
        
        /**
         * Invoke GC occasionally (based on frequency settings)
         */
        public void invokeGcOccasionally() {
            MemoryManager mm = Nd4j.getMemoryManager();
            if (mm != null) {
                mm.invokeGcOccasionally();
            }
        }
        
        /**
         * Get GC frequency
         */
        public int getGcFrequency() {
            MemoryManager mm = Nd4j.getMemoryManager();
            return mm != null ? mm.getFrequency() : 0;
        }
        
        /**
         * Set GC frequency
         */
        public void setGcFrequency(int frequency) {
            MemoryManager mm = Nd4j.getMemoryManager();
            if (mm != null) {
                mm.setFrequency(frequency);
            }
        }
        
        /**
         * Toggle periodic GC
         */
        public void togglePeriodicGc(boolean enabled) {
            MemoryManager mm = Nd4j.getMemoryManager();
            if (mm != null) {
                if (enabled) {
                    mm.startPeriodicGc();
                } else {
                    mm.stopPeriodicGc();
                }
            }
        }
        
        /**
         * Purge memory caches
         */
        public void purgeCaches() {
            MemoryManager mm = Nd4j.getMemoryManager();
            if (mm != null) {
                mm.purgeCaches();
            }
        }
        
        /**
         * Get memory manager type name
         */
        public String getType() {
            MemoryManager mm = Nd4j.getMemoryManager();
            return mm != null ? mm.getClass().getSimpleName() : "null";
        }
    }
    
    // =========================================================================
    // Deallocator Access
    // =========================================================================
    
    /**
     * DeallocatorService access
     */
    public static class DeallocatorAccess {
        
        /**
         * Get the DeallocatorService instance
         */
        public DeallocatorService get() {
            return DeallocatorService.getInstance();
        }
        
        /**
         * Get total allocations count
         */
        public long getTotalAllocations() {
            return (long) DeallocatorService.getInstance().getAllocated().get("INDArray");
        }

        /**
         * Get total deallocations count
         */
        public long getTotalDeallocations() {
            return (long) DeallocatorService.getInstance().getDeallocated().get("INDArray");
        }
        
        /**
         * Get live reference count
         */
        public long getLiveReferenceCount() {
            return DeallocatorService.getInstance().getReferenceMap().size();
        }
        
        /**
         * Check if shutdown is in progress
         */
        public boolean isShutdownInProgress() {
            return DeallocatorService.shutdownInProgress();
        }
        
        /**
         * Block deallocator (for debugging)
         */
        public void blockDeallocator(boolean block) {
            DeallocatorService.setBlockDeallocator(block);
        }
        
        /**
         * Check if deallocator is blocked
         */
        public boolean isDeallocatorBlocked() {
            return DeallocatorService.isBlockDeallocator();
        }
        
        /**
         * Get deallocator thread count
         */
        public int getThreadCount() {
            return DeallocatorService.getInstance().getNumThreads();
        }
        
        /**
         * Enable/disable deallocator service
         */
        public void setEnabled(boolean enabled) {
            // Note: This would need a proper API in DeallocatorService
            log.warn("DeallocatorService cannot be dynamically enabled/disabled");
        }
    }
    
    // =========================================================================
    // MemoryTracker Access
    // =========================================================================
    
    /**
     * MemoryTracker access
     */
    public static class MemoryTrackerAccess {
        
        /**
         * Get the MemoryTracker instance
         */
        public MemoryTracker get() {
            return MemoryTracker.getInstance();
        }
        
        /**
         * Get allocated memory for device
         */
        public long getAllocatedAmount(int deviceId) {
            return MemoryTracker.getInstance().getAllocatedAmount(deviceId);
        }
        
        /**
         * Get cached memory for device
         */
        public long getCachedAmount(int deviceId) {
            return MemoryTracker.getInstance().getCachedAmount(deviceId);
        }
        
        /**
         * Get workspace allocated memory for device
         */
        public long getWorkspaceAllocatedAmount(int deviceId) {
            return MemoryTracker.getInstance().getWorkspaceAllocatedAmount(deviceId);
        }
        
        /**
         * Get total memory for device
         */
        public long getTotalMemory(int deviceId) {
            return MemoryTracker.getInstance().getTotalMemory(deviceId);
        }
        
        /**
         * Get free memory for device
         */
        public long getFreeMemory(int deviceId) {
            return MemoryTracker.getInstance().getFreeMemory(deviceId);
        }
        
        /**
         * Get active memory for device
         */
        public long getActiveMemory(int deviceId) {
            return MemoryTracker.getInstance().getActiveMemory(deviceId);
        }
        
        /**
         * Get managed memory for device
         */
        public long getManagedMemory(int deviceId) {
            return MemoryTracker.getInstance().getManagedMemory(deviceId);
        }
        
        /**
         * Get approximate free memory for device
         */
        public long getApproximateFreeMemory(int deviceId) {
            return MemoryTracker.getInstance().getApproximateFreeMemory(deviceId);
        }
        
        /**
         * Get precise free memory for device
         */
        public long getPreciseFreeMemory(int deviceId) {
            return MemoryTracker.getInstance().getPreciseFreeMemory(deviceId);
        }
        
        /**
         * Get host allocated memory
         */
        public long getAllocatedHostAmount() {
            return MemoryTracker.getInstance().getAllocatedHostAmount();
        }
        
        /**
         * Get host cached memory
         */
        public long getCachedHostAmount() {
            return MemoryTracker.getInstance().getCachedHostAmount();
        }
        
        /**
         * Get memory per device summary
         */
        public String getMemoryPerDevice() {
            return MemoryTracker.getInstance().memoryPerDevice();
        }
    }
    
    // =========================================================================
    // Memory Sampling Access
    // =========================================================================
    
    /**
     * Historical memory sampling
     */
    public static class MemorySamplingAccess {
        
        /**
         * Start background sampling
         */
        public void start() {
            MemorySampler.getInstance().startSampling();
        }
        
        /**
         * Start background sampling with custom interval
         */
        public void start(long intervalMs) {
            MemorySampler.getInstance().startSampling(intervalMs);
        }
        
        /**
         * Stop background sampling
         */
        public void stop() {
            MemorySampler.getInstance().stopSampling();
        }
        
        /**
         * Check if sampling is active
         */
        public boolean isSamplingActive() {
            return MemorySampler.getInstance().isSamplingActive();
        }
        
        /**
         * Get recent samples
         */
        public List<MemorySample> getRecent(int numSamples) {
            return MemorySampler.getInstance().getRecentSamples(numSamples);
        }
        
        /**
         * Get all samples
         */
        public List<MemorySample> getAll() {
            return MemorySampler.getInstance().getAllSamples();
        }
        
        /**
         * Clear sample history
         */
        public void clear() {
            MemorySampler.getInstance().clearSamples();
        }
        
        /**
         * Get sample count
         */
        public int getSampleCount() {
            return MemorySampler.getInstance().getSampleCount();
        }
        
        /**
         * Set max samples to retain
         */
        public void setMaxSamples(int maxSamples) {
            MemorySampler.getInstance().setMaxSamples(maxSamples);
        }
        
        /**
         * Manually record a sample
         */
        public void recordSample() {
            MemorySampler.getInstance().recordSample();
        }
    }
}
