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

package org.nd4j.lifecycle;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.memory.deallocation.DeallocatorService;
import org.nd4j.linalg.framework.history.ArrayLifecycleEvent;
import org.nd4j.linalg.framework.history.ArrayLifecycleTracker;
import org.nd4j.linalg.framework.history.LifecycleStats;

/**
 * Lifecycle subsystem - provides access to array lifecycle tracking.
 * 
 * @author ND4J Team
 */
@Slf4j
public final class LifecycleSubsystem {
    
    private final LifecycleEnvironmentAccess environment;
    
    public LifecycleSubsystem() {
        this.environment = new LifecycleEnvironmentAccess();
    }
    
    /**
     * Get total arrays created
     */
    public long totalCreated() {
        return (long) DeallocatorService.getInstance().getAllocated().get("INDArray");
    }

    /**
     * Get total arrays destroyed
     */
    public long totalDestroyed() {
        return (long) DeallocatorService.getInstance().getDeallocated().get("INDArray");
    }
    
    /**
     * Get currently live arrays
     */
    public long liveCount() {
        return DeallocatorService.getInstance().getReferenceMap().size();
    }
    
    /**
     * Get recent lifecycle events
     */
    public java.util.List<ArrayLifecycleEvent> getHistory(int limit) {
        return ArrayLifecycleTracker.getInstance().getRecentEvents(limit);
    }
    
    /**
     * Enable lifecycle tracking
     */
    public void enableTracking() {
        ArrayLifecycleTracker.getInstance().enable();
    }
    
    /**
     * Enable stack trace capture
     */
    public void enableStackTraceCapture() {
        ArrayLifecycleTracker.getInstance().enableStackTraceCapture();
    }
    
    /**
     * Get lifecycle statistics
     */
    public LifecycleStats stats() {
        return ArrayLifecycleTracker.getInstance().getStats();
    }
    
    /**
     * Disable lifecycle tracking
     */
    public void disableTracking() {
        ArrayLifecycleTracker.getInstance().disable();
    }
    
    /**
     * Clear lifecycle history
     */
    public void clearHistory() {
        ArrayLifecycleTracker.getInstance().clearHistory();
    }
    
    /**
     * Lifecycle environment configuration
     */
    public LifecycleEnvironmentAccess environment() {
        return environment;
    }
    
    // =========================================================================
    // Lifecycle Environment Access
    // =========================================================================
    
    /**
     * Lifecycle environment configuration
     */
    public static class LifecycleEnvironmentAccess {
        
        /**
         * Check if array GC is disabled
         */
        public boolean isNoArrayGc() {
            return Boolean.getBoolean(org.nd4j.common.config.ND4JSystemProperties.NO_ARRAY_GC);
        }
        
        /**
         * Enable/disable array GC
         */
        public void setNoArrayGc(boolean noGc) {
            System.setProperty(org.nd4j.common.config.ND4JSystemProperties.NO_ARRAY_GC, String.valueOf(noGc));
        }
        
        /**
         * Get leak detection age threshold (ms)
         */
        public long getLeakDetectionAgeThreshold() {
            return Long.getLong(org.nd4j.common.config.ND4JSystemProperties.LEAK_DETECTION_AGE_THRESHOLD, 5 * 60 * 1000);
        }
        
        /**
         * Set leak detection age threshold (ms)
         */
        public void setLeakDetectionAgeThreshold(long thresholdMs) {
            System.setProperty(org.nd4j.common.config.ND4JSystemProperties.LEAK_DETECTION_AGE_THRESHOLD, String.valueOf(thresholdMs));
        }
        
        /**
         * Get summary
         */
        public String getSummary() {
            return String.format(
                "Lifecycle: NoArrayGC=%s, Leak Age Threshold=%ds",
                isNoArrayGc() ? "yes" : "no",
                getLeakDetectionAgeThreshold() / 1000
            );
        }
    }
}
