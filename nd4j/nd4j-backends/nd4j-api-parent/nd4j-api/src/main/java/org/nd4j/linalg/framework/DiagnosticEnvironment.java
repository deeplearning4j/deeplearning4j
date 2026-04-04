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

package org.nd4j.linalg.framework;

import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Diagnostic subsystem environment configuration.
 * 
 * @author ND4J Team
 */
public final class DiagnosticEnvironment {
    
    private DiagnosticEnvironment() {}
    
    /**
     * Check if diagnostics are enabled
     */
    public boolean isDiagnosticsEnabled() {
        return Boolean.getBoolean(ND4JSystemProperties.DIAGNOSTICS_ENABLE);
    }
    
    /**
     * Enable/disable diagnostics
     */
    public void setDiagnosticsEnabled(boolean enabled) {
        System.setProperty(ND4JSystemProperties.DIAGNOSTICS_ENABLE, String.valueOf(enabled));
    }
    
    /**
     * Get diagnostics level
     */
    public String getDiagnosticsLevel() {
        return System.getProperty(ND4JSystemProperties.DIAGNOSTICS_LEVEL, "WARNING");
    }
    
    /**
     * Set diagnostics level (INFO, WARNING, ERROR, CRITICAL)
     */
    public void setDiagnosticsLevel(String level) {
        System.setProperty(ND4JSystemProperties.DIAGNOSTICS_LEVEL, level);
    }
    
    /**
     * Check if verbose diagnostics is enabled
     */
    public boolean isDiagnosticsVerbose() {
        return Boolean.getBoolean(ND4JSystemProperties.DIAGNOSTICS_VERBOSE);
    }
    
    /**
     * Enable/disable verbose diagnostics
     */
    public void setDiagnosticsVerbose(boolean enabled) {
        System.setProperty(ND4JSystemProperties.DIAGNOSTICS_VERBOSE, String.valueOf(enabled));
    }
    
    /**
     * Get diagnostics output file
     */
    public String getDiagnosticsOutputFile() {
        return System.getProperty(ND4JSystemProperties.DIAGNOSTICS_OUTPUT_FILE);
    }
    
    /**
     * Set diagnostics output file
     */
    public void setDiagnosticsOutputFile(String path) {
        System.setProperty(ND4JSystemProperties.DIAGNOSTICS_OUTPUT_FILE, path);
    }
    
    /**
     * Check if health monitoring is enabled
     */
    public boolean isHealthMonitoringEnabled() {
        return Boolean.getBoolean(ND4JSystemProperties.HEALTH_MONITORING_ENABLE);
    }
    
    /**
     * Enable/disable health monitoring
     */
    public void setHealthMonitoringEnabled(boolean enabled) {
        System.setProperty(ND4JSystemProperties.HEALTH_MONITORING_ENABLE, String.valueOf(enabled));
    }
    
    /**
     * Get health check interval (ms)
     */
    public long getHealthCheckInterval() {
        return Long.getLong(ND4JSystemProperties.HEALTH_CHECK_INTERVAL, 60000);
    }
    
    /**
     * Set health check interval (ms)
     */
    public void setHealthCheckInterval(long intervalMs) {
        System.setProperty(ND4JSystemProperties.HEALTH_CHECK_INTERVAL, String.valueOf(intervalMs));
    }
    
    /**
     * Check if logging is enabled for native array creation
     */
    public boolean isLogNativeArrayCreation() {
        return Nd4j.getEnvironment().isLogNativeNDArrayCreation();
    }
    
    /**
     * Enable/disable logging for native array creation
     */
    public void setLogNativeArrayCreation(boolean enabled) {
        Nd4j.getEnvironment().setLogNativeNDArrayCreation(enabled);
    }
    
    /**
     * Check if func trace is enabled
     */
    public boolean isFuncTraceEnabled() {
        return Boolean.getBoolean(ND4JSystemProperties.FUNCTRACE_ENABLE);
    }
    
    /**
     * Enable/disable func trace
     */
    public void setFuncTraceEnabled(boolean enabled) {
        System.setProperty(ND4JSystemProperties.FUNCTRACE_ENABLE, String.valueOf(enabled));
    }
    
    /**
     * Get summary of diagnostic environment settings
     */
    public String getSummary() {
        return String.format(
            "Diagnostic Environment: Enabled=%s, Level=%s, Verbose=%s, " +
            "Health Monitoring=%s (interval=%ds), " +
            "Log Native Arrays=%s, Func Trace=%s",
            isDiagnosticsEnabled() ? "yes" : "no",
            getDiagnosticsLevel(),
            isDiagnosticsVerbose() ? "yes" : "no",
            isHealthMonitoringEnabled() ? "on" : "off",
            getHealthCheckInterval() / 1000,
            isLogNativeArrayCreation() ? "yes" : "no",
            isFuncTraceEnabled() ? "yes" : "no"
        );
    }
}
