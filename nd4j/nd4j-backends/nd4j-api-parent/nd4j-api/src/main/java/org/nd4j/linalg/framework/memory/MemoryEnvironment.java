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

import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.factory.Environment;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Memory subsystem environment configuration.
 * 
 * Provides unified access to memory-related environment settings
 * and system properties.
 * 
 * Usage:
 * <pre>
 *   Nd4j.framework.memory().environment().getWorkspaceDebugMode()
 *   Nd4j.framework.memory().environment().setGcFrequency(100)
 * </pre>
 * 
 * @author ND4J Team
 */
public final class MemoryEnvironment {
    
    private MemoryEnvironment() {}
    
    /**
     * Get GC frequency (number of allocations between GCs)
     */
    public int getGcFrequency() {
        return Nd4j.getMemoryManager().getFrequency();
    }
    
    /**
     * Set GC frequency
     * @param frequency Number of allocations between GCs (0 = disabled)
     */
    public void setGcFrequency(int frequency) {
        Nd4j.getMemoryManager().setFrequency(frequency);
    }
    
    /**
     * Get auto GC window (ms to wait before GC)
     */
    public int getAutoGcWindow() {
        return Nd4j.getMemoryManager().getAutoGcWindow();
    }
    
    /**
     * Set auto GC window
     * @param windowMs Window in milliseconds
     */
    public void setAutoGcWindow(int windowMs) {
        Nd4j.getMemoryManager().setAutoGcWindow(windowMs);
    }
    
    /**
     * Check if periodic GC is enabled
     */
    public boolean isPeriodicGcEnabled() {
        return Nd4j.getMemoryManager().isPeriodicGcActive();
    }
    
    /**
     * Enable/disable periodic GC
     */
    public void setPeriodicGcEnabled(boolean enabled) {
        if (enabled) {
            Nd4j.getMemoryManager().startPeriodicGc();
        } else {
            Nd4j.getMemoryManager().stopPeriodicGc();
        }
    }
    
    /**
     * Check if array GC is disabled
     */
    public boolean isNoArrayGc() {
        return Boolean.getBoolean(ND4JSystemProperties.NO_ARRAY_GC);
    }
    
    /**
     * Enable/disable array GC
     */
    public void setNoArrayGc(boolean noGc) {
        System.setProperty(ND4JSystemProperties.NO_ARRAY_GC, String.valueOf(noGc));
    }
    
    /**
     * Get deallocator service GC thread count
     */
    public int getDeallocatorGcThreads() {
        return Integer.getInteger(ND4JSystemProperties.DEALLOCATOR_SERVICE_GC_THREADS, 1);
    }
    
    /**
     * Set deallocator service GC thread count
     */
    public void setDeallocatorGcThreads(int threads) {
        System.setProperty(ND4JSystemProperties.DEALLOCATOR_SERVICE_GC_THREADS, String.valueOf(threads));
    }
    
    /**
     * Check if workspace tracking is enabled
     */
    public boolean isWorkspaceTrackingEnabled() {
        return Nd4j.getEnvironment().isTrackWorkspaceOpenClose();
    }
    
    /**
     * Enable/disable workspace tracking (high overhead)
     */
    public void setWorkspaceTrackingEnabled(boolean enabled) {
        Nd4j.getEnvironment().setTrackWorkspaceOpenClose(enabled);
    }
    
    /**
     * Get number of workspace events to keep
     */
    public int getWorkspaceEventRetention() {
        return Nd4j.getEnvironment().numWorkspaceEventsToKeep();
    }
    
    /**
     * Set number of workspace events to keep (-1 = unlimited)
     */
    public void setWorkspaceEventRetention(int retention) {
        System.setProperty(ND4JSystemProperties.WORKSPACE_META_SIZE, String.valueOf(retention));
    }
    
    /**
     * Check if func trace print on allocate is enabled
     */
    public boolean isFuncTraceAllocateEnabled() {
        return Nd4j.getEnvironment().isFuncTracePrintAllocate();
    }
    
    /**
     * Enable/disable func trace print on allocate
     */
    public void setFuncTraceAllocateEnabled(boolean enabled) {
        System.setProperty(ND4JSystemProperties.FUNCTRACE_PRINT_ALLOCATE, String.valueOf(enabled));
    }
    
    /**
     * Check if func trace print on free is enabled
     */
    public boolean isFuncTraceFreeEnabled() {
        return Nd4j.getEnvironment().isFuncTracePrintFree();
    }
    
    /**
     * Enable/disable func trace print on free
     */
    public void setFuncTraceFreeEnabled(boolean enabled) {
        System.setProperty(ND4JSystemProperties.FUNCTRACE_PRINT_FREE, String.valueOf(enabled));
    }
    
    /**
     * Get memory pressure threshold (0.0-1.0)
     */
    public double getMemoryPressureThreshold() {
        return Nd4j.getEnvironment().heapPressureThreshold();
    }
    
    /**
     * Set memory pressure threshold (0.0-1.0)
     */
    public void setMemoryPressureThreshold(double threshold) {
        System.setProperty(ND4JSystemProperties.HEAP_PRESSURE_THRESHOLD, String.valueOf(threshold));
    }
    
    /**
     * Get all memory environment settings as summary
     */
    public String getSummary() {
        return String.format(
            "Memory Environment: GC freq=%d, AutoGC window=%dms, PeriodicGC=%s, " +
            "NoArrayGC=%s, Deallocator threads=%d, Workspace tracking=%s, " +
            "Pressure threshold=%.1f%%",
            getGcFrequency(),
            getAutoGcWindow(),
            isPeriodicGcEnabled() ? "on" : "off",
            isNoArrayGc() ? "yes" : "no",
            getDeallocatorGcThreads(),
            isWorkspaceTrackingEnabled() ? "on" : "off",
            getMemoryPressureThreshold() * 100
        );
    }
}
