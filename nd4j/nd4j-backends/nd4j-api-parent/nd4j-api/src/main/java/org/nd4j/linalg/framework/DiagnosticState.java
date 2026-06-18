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

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Diagnostic subsystem state snapshot.
 * 
 * @author ND4J Team
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class DiagnosticState {
    
    /**
     * Whether diagnostics are enabled
     */
    private boolean enabled;
    
    /**
     * Diagnostics level
     */
    private String level;
    
    /**
     * Number of active issues
     */
    private int issueCount;
    
    /**
     * Number of critical issues
     */
    private int criticalIssues;
    
    /**
     * Number of error issues
     */
    private int errorIssues;
    
    /**
     * Number of warning issues
     */
    private int warningIssues;
    
    /**
     * Whether health monitoring is enabled
     */
    private boolean healthMonitoringEnabled;
    
    /**
     * Health check interval (ms)
     */
    private long healthCheckIntervalMs;
    
    /**
     * Whether framework is healthy
     */
    private boolean healthy;
    
    /**
     * Get health check interval in seconds
     */
    public double getHealthCheckIntervalSeconds() {
        return healthCheckIntervalMs / 1000.0;
    }
    
    /**
     * Get health status string
     */
    public String getHealthStatus() {
        if (healthy) return "HEALTHY";
        if (criticalIssues > 0) return "CRITICAL";
        if (errorIssues > 0) return "ERROR";
        if (warningIssues > 0) return "WARNING";
        return "UNKNOWN";
    }
    
    @Override
    public String toString() {
        return String.format(
            "Enabled: %s, Level: %s, Health: %s (%d issues: %d crit, %d err, %d warn), " +
            "Monitoring: %s (%.0f s)",
            enabled ? "YES" : "NO",
            level,
            getHealthStatus(),
            issueCount,
            criticalIssues,
            errorIssues,
            warningIssues,
            healthMonitoringEnabled ? "ON" : "OFF",
            getHealthCheckIntervalSeconds()
        );
    }
}
