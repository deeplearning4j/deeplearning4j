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

package org.nd4j.linalg.framework.workspace;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Workspace subsystem state snapshot.
 * 
 * @author ND4J Team
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class WorkspaceState {
    
    /**
     * Number of active workspaces
     */
    private int numWorkspaces;
    
    /**
     * Total workspace memory (bytes)
     */
    private long totalWorkspaceBytes;
    
    /**
     * Total spilled bytes
     */
    private long totalSpilledBytes;
    
    /**
     * Workspace initial size (bytes)
     */
    private long initialSizeBytes;
    
    /**
     * Whether workspace learning is enabled
     */
    private boolean learningEnabled;
    
    /**
     * Workspace debug mode
     */
    private String debugMode;
    
    /**
     * Get total workspace memory in MB
     */
    public double getTotalWorkspaceMb() {
        return totalWorkspaceBytes / (1024.0 * 1024.0);
    }
    
    /**
     * Get initial size in MB
     */
    public double getInitialSizeMb() {
        return initialSizeBytes / (1024.0 * 1024.0);
    }
    
    /**
     * Get spill percentage
     */
    public double getSpillPercent() {
        if (totalWorkspaceBytes <= 0) return 0.0;
        return (totalSpilledBytes * 100.0) / totalWorkspaceBytes;
    }
    
    @Override
    public String toString() {
        return String.format(
            "Workspaces: %d, Memory: %.1f MB, Spilled: %.1f MB (%.1f%%), " +
            "Initial: %.1f MB, Learning: %s, Debug: %s",
            numWorkspaces,
            getTotalWorkspaceMb(),
            totalSpilledBytes / (1024.0 * 1024.0),
            getSpillPercent(),
            getInitialSizeMb(),
            learningEnabled ? "ON" : "OFF",
            debugMode
        );
    }
}
