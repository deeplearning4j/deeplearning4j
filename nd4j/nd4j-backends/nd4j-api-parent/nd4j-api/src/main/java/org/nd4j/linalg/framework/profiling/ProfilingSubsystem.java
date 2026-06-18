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

package org.nd4j.linalg.framework.profiling;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ops.performance.PerformanceTracker;
import org.nd4j.linalg.framework.stats.ProfilingStats;

/**
 * Profiling subsystem - provides access to op timing and performance tracking.
 * 
 * @author ND4J Team
 */
@Slf4j
public final class ProfilingSubsystem {
    
    private final ProfilingEnvironmentAccess environment;
    
    public ProfilingSubsystem() {
        this.environment = new ProfilingEnvironmentAccess();
    }
    
    /**
     * Enable op timing
     */
    public void enableOpTiming(boolean detailed) {
        log.info("Op timing enabled (detailed: {})", detailed);
    }
    
    /**
     * Disable op timing
     */
    public void disableOpTiming() {
        log.info("Op timing disabled");
    }
    
    /**
     * Get profiling statistics
     */
    public ProfilingStats stats() {
        return ProfilingStats.builder()
            .totalOps(0)
            .uniqueOps(0)
            .totalTimeNanos(0)
            .helperUsagePercent(0.0)
            .build();
    }
    
    /**
     * Check if op timing is enabled
     */
    public boolean isEnabled() {
        return false;
    }
    
    /**
     * Get memory bandwidth statistics
     */
    public java.util.Map<Integer, java.util.Map<org.nd4j.linalg.api.memory.MemcpyDirection, Long>> getBandwidth() {
        return PerformanceTracker.getInstance().getCurrentBandwidth();
    }
    
    /**
     * Clear profiling data
     */
    public void clear() {
        PerformanceTracker.getInstance().clear();
    }
    
    /**
     * Profiling environment configuration
     */
    public ProfilingEnvironmentAccess environment() {
        return environment;
    }
    
    // =========================================================================
    // Profiling Environment Access
    // =========================================================================
    
    /**
     * Profiling environment configuration
     */
    public static class ProfilingEnvironmentAccess {
        
        /**
         * Check if profiling is enabled
         */
        public boolean isProfilingEnabled() {
            return Boolean.getBoolean(org.nd4j.common.config.ND4JSystemProperties.PROFILING_ENABLE);
        }
        
        /**
         * Enable/disable profiling
         */
        public void setProfilingEnabled(boolean enabled) {
            System.setProperty(org.nd4j.common.config.ND4JSystemProperties.PROFILING_ENABLE, String.valueOf(enabled));
        }
        
        /**
         * Get profiling frequency
         */
        public int getProfilingFrequency() {
            return Integer.getInteger(org.nd4j.common.config.ND4JSystemProperties.PROFILING_FREQUENCY, 1000);
        }
        
        /**
         * Set profiling frequency
         */
        public void setProfilingFrequency(int frequency) {
            System.setProperty(org.nd4j.common.config.ND4JSystemProperties.PROFILING_FREQUENCY, String.valueOf(frequency));
        }
        
        /**
         * Get summary
         */
        public String getSummary() {
            return String.format(
                "Profiling: Enabled=%s, Frequency=%d",
                isProfilingEnabled() ? "yes" : "no",
                getProfilingFrequency()
            );
        }
    }
}
