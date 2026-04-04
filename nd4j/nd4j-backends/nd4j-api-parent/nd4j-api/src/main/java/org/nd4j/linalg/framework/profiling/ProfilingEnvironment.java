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

import org.nd4j.common.config.ND4JSystemProperties;

/**
 * Profiling subsystem environment configuration.
 * 
 * @author ND4J Team
 */
public final class ProfilingEnvironment {
    
    private ProfilingEnvironment() {}
    
    /**
     * Check if profiling is enabled
     */
    public boolean isProfilingEnabled() {
        return Boolean.getBoolean(ND4JSystemProperties.PROFILING_ENABLE);
    }
    
    /**
     * Enable/disable profiling
     */
    public void setProfilingEnabled(boolean enabled) {
        System.setProperty(ND4JSystemProperties.PROFILING_ENABLE, String.valueOf(enabled));
    }
    
    /**
     * Get profiling frequency (operations per snapshot)
     */
    public int getProfilingFrequency() {
        return Integer.getInteger(ND4JSystemProperties.PROFILING_FREQUENCY, 1000);
    }
    
    /**
     * Set profiling frequency
     */
    public void setProfilingFrequency(int frequency) {
        System.setProperty(ND4JSystemProperties.PROFILING_FREQUENCY, String.valueOf(frequency));
    }
    
    /**
     * Check if verbose profiling is enabled
     */
    public boolean isProfilingVerbose() {
        return Boolean.getBoolean(ND4JSystemProperties.PROFILING_VERBOSE);
    }
    
    /**
     * Enable/disable verbose profiling
     */
    public void setProfilingVerbose(boolean enabled) {
        System.setProperty(ND4JSystemProperties.PROFILING_VERBOSE, String.valueOf(enabled));
    }
    
    /**
     * Get profiling output file
     */
    public String getProfilingOutputFile() {
        return System.getProperty(ND4JSystemProperties.PROFILING_OUTPUT_FILE);
    }
    
    /**
     * Set profiling output file
     */
    public void setProfilingOutputFile(String path) {
        System.setProperty(ND4JSystemProperties.PROFILING_OUTPUT_FILE, path);
    }
    
    /**
     * Check if bandwidth profiling is enabled
     */
    public boolean isBandwidthProfilingEnabled() {
        return Boolean.getBoolean(ND4JSystemProperties.PROFILING_BANDWIDTH);
    }
    
    /**
     * Enable/disable bandwidth profiling
     */
    public void setBandwidthProfilingEnabled(boolean enabled) {
        System.setProperty(ND4JSystemProperties.PROFILING_BANDWIDTH, String.valueOf(enabled));
    }
    
    /**
     * Get summary of profiling environment settings
     */
    public String getSummary() {
        return String.format(
            "Profiling Environment: Enabled=%s, Frequency=%d, Verbose=%s, " +
            "Bandwidth=%s, Output=%s",
            isProfilingEnabled() ? "yes" : "no",
            getProfilingFrequency(),
            isProfilingVerbose() ? "yes" : "no",
            isBandwidthProfilingEnabled() ? "yes" : "no",
            getProfilingOutputFile() != null ? getProfilingOutputFile() : "stdout"
        );
    }
}
