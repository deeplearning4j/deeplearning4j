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

package org.nd4j.linalg.framework.leak;

import lombok.Builder;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;

import java.util.List;

/**
 * Comprehensive leak detection report.
 */
@Data
@Builder
@Slf4j
public class LeakReport {
    
    /** Timestamp of the analysis */
    private final long timestampMs;
    
    /** Whether leaks were detected */
    private final boolean leaksDetected;
    
    /** Number of potential leaks */
    private final int potentialLeakCount;
    
    /** Total bytes of potential leaks */
    private final long potentialLeakBytes;
    
    /** List of potential leak details */
    private final List<PotentialLeak> potentialLeaks;
    
    /** Memory growth rate in bytes per second */
    private final long memoryGrowthRateBytesPerSec;
    
    /** Duration of the analysis in milliseconds */
    private final long analysisDurationMs;
    
    /** Severity of the leak report */
    private final Severity severity;
    
    /** Severity enum for leak reports */
    public enum Severity {
        INFO,
        WARNING,
        ERROR,
        CRITICAL
    }
    
    /**
     * Get the severity based on leak detection results.
     */
    public Severity getSeverity() {
        if (severity != null) {
            return severity;
        }
        if (!leaksDetected) {
            return Severity.INFO;
        }
        if (potentialLeakCount > 100 || potentialLeakBytes > 1024 * 1024 * 1024) {
            return Severity.CRITICAL;
        }
        if (potentialLeakCount > 10 || potentialLeakBytes > 100 * 1024 * 1024) {
            return Severity.ERROR;
        }
        return Severity.WARNING;
    }
    
    /**
     * Get a human-readable summary of the report.
     */
    public String getSummary() {
        if (!leaksDetected) {
            return "No leaks detected";
        }
        return String.format("Potential leaks: %d objects, %d bytes, growth rate: %d bytes/sec",
            potentialLeakCount, potentialLeakBytes, memoryGrowthRateBytesPerSec);
    }
    
    /**
     * Get a detailed report string.
     */
    public String getDetailedReport() {
        StringBuilder sb = new StringBuilder();
        sb.append("=== Leak Detection Report ===\n");
        sb.append("Timestamp: ").append(timestampMs).append("\n");
        sb.append("Leaks detected: ").append(leaksDetected).append("\n");
        sb.append("Potential leak count: ").append(potentialLeakCount).append("\n");
        sb.append("Potential leak bytes: ").append(potentialLeakBytes).append("\n");
        sb.append("Memory growth rate: ").append(memoryGrowthRateBytesPerSec).append(" bytes/sec\n");
        sb.append("Analysis duration: ").append(analysisDurationMs).append(" ms\n");
        
        if (!potentialLeaks.isEmpty()) {
            sb.append("\nPotential leaks:\n");
            for (PotentialLeak leak : potentialLeaks) {
                sb.append("  - ").append(leak.toString()).append("\n");
            }
        }
        
        return sb.toString();
    }
}
