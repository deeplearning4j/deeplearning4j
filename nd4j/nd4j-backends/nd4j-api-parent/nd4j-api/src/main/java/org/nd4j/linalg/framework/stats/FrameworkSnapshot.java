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
import org.nd4j.linalg.framework.history.LifecycleStats;

import java.util.List;

/**
 * Snapshot of entire framework state.
 * 
 * @author ND4J Team
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class FrameworkSnapshot {
    
    /**
     * Timestamp when snapshot was taken
     */
    private long timestampMs;
    
    /**
     * Memory statistics
     */
    private MemoryStats memoryStats;
    
    /**
     * Profiling statistics
     */
    private ProfilingStats profilingStats;
    
    /**
     * Lifecycle statistics
     */
    private LifecycleStats lifecycleStats;
    
    /**
     * Workspace statistics
     */
    private WorkspaceStats workspaceStats;
    
    /**
     * Active diagnostic issues
     */
    private List<DiagnosticIssue> diagnosticIssues;
    
    /**
     * Get summary string
     */
    public String getSummary() {
        StringBuilder sb = new StringBuilder();
        sb.append("Framework Snapshot:\n");
        
        if (memoryStats != null) {
            sb.append("  Memory: ").append(memoryStats.getSummary()).append("\n");
        }
        
        if (profilingStats != null) {
            sb.append("  Profiling: ").append(profilingStats.getTotalOps())
              .append(" ops, ").append(String.format("%.2f", profilingStats.getTotalTimeSeconds()))
              .append("s total\n");
        }
        
        if (lifecycleStats != null) {
            sb.append("  Lifecycle: ").append(lifecycleStats.getCurrentLive())
              .append(" live arrays\n");
        }
        
        if (workspaceStats != null) {
            sb.append("  Workspaces: ").append(workspaceStats.getActiveWorkspaceCount())
              .append(" active, ").append(String.format("%.1f", workspaceStats.getTotalWorkspaceBytes() / (1024.0 * 1024.0)))
              .append(" MB\n");
        }
        
        if (diagnosticIssues != null && !diagnosticIssues.isEmpty()) {
            sb.append("  Issues: ").append(diagnosticIssues.size()).append(" active\n");
        }
        
        return sb.toString();
    }
}
