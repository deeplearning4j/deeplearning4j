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
import org.nd4j.linalg.framework.device.DeviceState;
import org.nd4j.linalg.framework.exec.ExecutionState;
import org.nd4j.linalg.framework.lifecycle.LifecycleState;
import org.nd4j.linalg.framework.memory.MemoryState;
import org.nd4j.linalg.framework.profiling.ProfilingState;
import org.nd4j.linalg.framework.workspace.WorkspaceState;
import org.nd4j.linalg.framework.constant.ConstantCacheState;

/**
 * Complete framework state snapshot.
 * 
 * Aggregates state from all subsystems into a single comprehensive snapshot
 * with human-readable toString() output.
 * 
 * @author ND4J Team
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class FrameworkSummary {
    
    /**
     * Timestamp when snapshot was taken
     */
    private long timestampMs;
    
    /**
     * Memory subsystem state
     */
    private MemoryState memory;
    
    /**
     * Device subsystem state
     */
    private DeviceState device;
    
    /**
     * Execution subsystem state
     */
    private ExecutionState execution;
    
    /**
     * Workspace subsystem state
     */
    private WorkspaceState workspace;
    
    /**
     * Profiling subsystem state
     */
    private ProfilingState profiling;
    
    /**
     * Lifecycle subsystem state
     */
    private LifecycleState lifecycle;
    
    /**
     * Diagnostic subsystem state
     */
    private DiagnosticState diagnostics;
    
    /**
     * Constant cache subsystem state
     */
    private ConstantCacheState constants;
    
    /**
     * Get human-readable summary
     */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("╔══════════════════════════════════════════════════════════╗\n");
        sb.append("║           ND4J FRAMEWORK STATE SNAPSHOT                  ║\n");
        sb.append("╠══════════════════════════════════════════════════════════╣\n");
        sb.append(String.format("║ Timestamp: %-47s ║\n", new java.util.Date(timestampMs)));
        sb.append("╠══════════════════════════════════════════════════════════╣\n");
        
        if (memory != null) {
            sb.append("║ MEMORY:                                                    ║\n");
            String[] memLines = memory.toString().split("\n");
            for (String line : memLines) {
                sb.append(String.format("║   %-55s ║\n", line));
            }
        }
        
        if (device != null) {
            sb.append("╠══════════════════════════════════════════════════════════╣\n");
            sb.append("║ DEVICE:                                                    ║\n");
            String[] devLines = device.toString().split("\n");
            for (String line : devLines) {
                sb.append(String.format("║   %-55s ║\n", line));
            }
        }
        
        if (execution != null) {
            sb.append("╠══════════════════════════════════════════════════════════╣\n");
            sb.append("║ EXECUTION:                                                 ║\n");
            String[] execLines = execution.toString().split("\n");
            for (String line : execLines) {
                sb.append(String.format("║   %-55s ║\n", line));
            }
        }
        
        if (workspace != null) {
            sb.append("╠══════════════════════════════════════════════════════════╣\n");
            sb.append("║ WORKSPACE:                                                 ║\n");
            String[] wsLines = workspace.toString().split("\n");
            for (String line : wsLines) {
                sb.append(String.format("║   %-55s ║\n", line));
            }
        }
        
        if (profiling != null) {
            sb.append("╠══════════════════════════════════════════════════════════╣\n");
            sb.append("║ PROFILING:                                                 ║\n");
            String[] profLines = profiling.toString().split("\n");
            for (String line : profLines) {
                sb.append(String.format("║   %-55s ║\n", line));
            }
        }
        
        if (lifecycle != null) {
            sb.append("╠══════════════════════════════════════════════════════════╣\n");
            sb.append("║ LIFECYCLE:                                                 ║\n");
            String[] lifeLines = lifecycle.toString().split("\n");
            for (String line : lifeLines) {
                sb.append(String.format("║   %-55s ║\n", line));
            }
        }
        
        if (diagnostics != null) {
            sb.append("╠══════════════════════════════════════════════════════════╣\n");
            sb.append("║ DIAGNOSTICS:                                               ║\n");
            String[] diagLines = diagnostics.toString().split("\n");
            for (String line : diagLines) {
                sb.append(String.format("║   %-55s ║\n", line));
            }
        }
        
        if (constants != null) {
            sb.append("╠══════════════════════════════════════════════════════════╣\n");
            sb.append("║ CONSTANT CACHES:                                           ║\n");
            String[] constLines = constants.toString().split("\n");
            for (String line : constLines) {
                sb.append(String.format("║   %-55s ║\n", line));
            }
        }
        
        sb.append("╚══════════════════════════════════════════════════════════╝");
        return sb.toString();
    }
    
    /**
     * Get compact one-line summary
     */
    public String getCompactSummary() {
        String constStr;
        if (constants != null) {
            long peakBytes = constants.getTotalPeakCacheBytes();
            if (peakBytes > 0) {
                constStr = String.format("%.1fKB(peak=%.1fKB)", constants.getTotalCacheKb(), constants.getTotalPeakCacheKb());
            } else {
                constStr = String.format("%.1fKB", constants.getTotalCacheKb());
            }
        } else {
            constStr = "0.0KB";
        }
        return String.format(
            "Framework[Mem=%.1fMB, Dev=%d, Exec=%s, WS=%d, Life=%d, Diag=%s, Const=%s]",
            memory != null ? memory.getHeapUsedMb() : 0,
            device != null ? device.getNumDevices() : 0,
            execution != null ? execution.getDspStatus() : "?",
            workspace != null ? workspace.getNumWorkspaces() : 0,
            lifecycle != null ? lifecycle.getLiveArrays() : 0,
            diagnostics != null ? (diagnostics.isHealthy() ? "OK" : "ISSUES") : "?",
            constStr
        );
    }
}
