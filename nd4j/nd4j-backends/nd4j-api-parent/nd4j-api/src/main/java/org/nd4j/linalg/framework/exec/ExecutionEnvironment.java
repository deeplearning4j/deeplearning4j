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

package org.nd4j.linalg.framework.exec;

import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Execution subsystem environment configuration.
 * 
 * Provides unified access to execution-related environment settings
 * including DSP, op execution, and kernel selection.
 * 
 * Usage:
 * <pre>
 *   Nd4j.framework.execution().environment().isDspEnabled()
 *   Nd4j.framework.execution().environment().setDspDiagnosticsLevel("full")
 * </pre>
 * 
 * @author ND4J Team
 */
public final class ExecutionEnvironment {
    
    ExecutionEnvironment() {}
    
    // ========================================================================
    // DSP Configuration
    // ========================================================================
    
    /**
     * Check if DSP (Dynamic Shape Plan) is enabled
     */
    public boolean isDspEnabled() {
        return Boolean.getBoolean(ND4JSystemProperties.DYNAMIC_SHAPE_PLAN_ENABLED);
    }
    
    /**
     * Enable/disable DSP
     */
    public void setDspEnabled(boolean enabled) {
        System.setProperty(ND4JSystemProperties.DYNAMIC_SHAPE_PLAN_ENABLED, String.valueOf(enabled));
    }
    
    /**
     * Check if DSP native executor is enabled
     */
    public boolean isDspNativeExecutorEnabled() {
        return Boolean.getBoolean(ND4JSystemProperties.DSP_NATIVE_EXECUTOR_ENABLED);
    }
    
    /**
     * Enable/disable DSP native executor
     */
    public void setDspNativeExecutorEnabled(boolean enabled) {
        System.setProperty(ND4JSystemProperties.DSP_NATIVE_EXECUTOR_ENABLED, String.valueOf(enabled));
    }
    
    /**
     * Get DSP diagnostics categories
     */
    public String getDspDiagnostics() {
        return System.getProperty(ND4JSystemProperties.DSP_DIAGNOSTICS, "none");
    }
    
    /**
     * Set DSP diagnostics categories (e.g., "MEMORY", "EXECUTE", "ALL")
     */
    public void setDspDiagnostics(String categories) {
        System.setProperty(ND4JSystemProperties.DSP_DIAGNOSTICS, categories);
    }
    
    /**
     * Get DSP diagnostics level
     */
    public String getDspDiagnosticsLevel() {
        return System.getProperty(ND4JSystemProperties.DSP_DIAGNOSTICS_LEVEL, "summary");
    }
    
    /**
     * Set DSP diagnostics level (summary, detailed, full)
     */
    public void setDspDiagnosticsLevel(String level) {
        System.setProperty(ND4JSystemProperties.DSP_DIAGNOSTICS_LEVEL, level);
    }
    
    /**
     * Get DSP diagnostics JSON output file
     */
    public String getDspDiagnosticsFile() {
        return System.getProperty(ND4JSystemProperties.DSP_DIAGNOSTICS_FILE);
    }
    
    /**
     * Set DSP diagnostics JSON output file
     */
    public void setDspDiagnosticsFile(String path) {
        System.setProperty(ND4JSystemProperties.DSP_DIAGNOSTICS_FILE, path);
    }
    
    /**
     * Check if DSP capture pool is enabled
     */
    public boolean isDspCapturePoolEnabled() {
        return Boolean.getBoolean(ND4JSystemProperties.DSP_CAPTURE_POOL_ENABLED);
    }
    
    /**
     * Enable/disable DSP capture pool
     */
    public void setDspCapturePoolEnabled(boolean enabled) {
        System.setProperty(ND4JSystemProperties.DSP_CAPTURE_POOL_ENABLED, String.valueOf(enabled));
    }
    
    // ========================================================================
    // Op Execution Configuration
    // ========================================================================
    
    /**
     * Check if op execution profiling is enabled
     */
    public boolean isOpProfilingEnabled() {
        return Boolean.getBoolean(ND4JSystemProperties.PROFILING_ENABLE);
    }
    
    /**
     * Enable/disable op execution profiling
     */
    public void setOpProfilingEnabled(boolean enabled) {
        System.setProperty(ND4JSystemProperties.PROFILING_ENABLE, String.valueOf(enabled));
    }
    
    /**
     * Get operations between profiling snapshots
     */
    public int getProfilingFrequency() {
        return Integer.getInteger(ND4JSystemProperties.PROFILING_FREQUENCY, 1000);
    }
    
    /**
     * Set operations between profiling snapshots
     */
    public void setProfilingFrequency(int frequency) {
        System.setProperty(ND4JSystemProperties.PROFILING_FREQUENCY, String.valueOf(frequency));
    }
    
    // ========================================================================
    // Kernel Selection Configuration
    // ========================================================================
    
    /**
     * Check if kernel selection debugging is enabled
     */
    public boolean isKernelSelectionDebug() {
        return Boolean.getBoolean(ND4JSystemProperties.KERNEL_SELECTION_DEBUG);
    }
    
    /**
     * Enable/disable kernel selection debugging
     */
    public void setKernelSelectionDebug(boolean enabled) {
        System.setProperty(ND4JSystemProperties.KERNEL_SELECTION_DEBUG, String.valueOf(enabled));
    }
    
    /**
     * Get preferred backend
     */
    public String getPreferredBackend() {
        return System.getProperty(ND4JSystemProperties.PREFERRED_BACKEND, "auto");
    }
    
    /**
     * Set preferred backend (auto, cpu, cuda, etc.)
     */
    public void setPreferredBackend(String backend) {
        System.setProperty(ND4JSystemProperties.PREFERRED_BACKEND, backend);
    }
    
    // ========================================================================
    // Summary
    // ========================================================================
    
    /**
     * Get all execution environment settings as summary
     */
    public String getSummary() {
        return String.format(
            "Execution Environment: DSP=%s, DSP Native=%s, DSP Diagnostics=%s/%s, " +
            "Capture Pool=%s, Profiling=%s (freq=%d), Preferred Backend=%s",
            isDspEnabled() ? "on" : "off",
            isDspNativeExecutorEnabled() ? "on" : "off",
            getDspDiagnostics(),
            getDspDiagnosticsLevel(),
            isDspCapturePoolEnabled() ? "on" : "off",
            isOpProfilingEnabled() ? "on" : "off",
            getProfilingFrequency(),
            getPreferredBackend()
        );
    }
}
