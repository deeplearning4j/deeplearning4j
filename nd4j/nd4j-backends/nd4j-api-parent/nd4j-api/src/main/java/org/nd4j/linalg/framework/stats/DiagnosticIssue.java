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

package org.nd4j.linalg.framework.stats;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Diagnostic issue detected by the framework.
 * 
 * @author ND4J Team
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class DiagnosticIssue {
    
    /**
     * Issue severity
     */
    public enum Severity {
        INFO,       // Informational, no action needed
        WARNING,    // Potential issue, monitor
        ERROR,      // Problem detected, action recommended
        CRITICAL    // Serious problem, immediate action required
    }
    
    /**
     * Issue category
     */
    public enum Category {
        MEMORY,         // Memory-related issue
        LEAK,           // Potential memory leak
        WORKSPACE,      // Workspace configuration issue
        PROFILING,      // Profiling-related issue
        PERFORMANCE,    // Performance degradation
        DEVICE,         // Device/GPU issue
        DEGRADATION     // General degradation over time
    }
    
    /**
     * Issue ID (unique identifier)
     */
    private String issueId;
    
    /**
     * Issue severity
     */
    private Severity severity;
    
    /**
     * Issue category
     */
    private Category category;
    
    /**
     * Short issue title
     */
    private String title;
    
    /**
     * Detailed description
     */
    private String description;
    
    /**
     * Recommended action to resolve
     */
    private String recommendedAction;
    
    /**
     * Timestamp when issue was first detected
     */
    private long firstDetectedMs;
    
    /**
     * Timestamp when issue was last updated
     */
    private long lastUpdatedMs;
    
    /**
     * Additional metadata
     */
    private java.util.Map<String, String> metadata;
    
    /**
     * Get issue summary
     */
    public String getSummary() {
        return String.format("[%s] %s: %s", severity, category, title);
    }
}
