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
 * Workspace statistics.
 * 
 * @author ND4J Team
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class WorkspaceStats {
    
    /**
     * Number of active workspaces
     */
    private int activeWorkspaceCount;
    
    /**
     * Total workspace memory (bytes)
     */
    private long totalWorkspaceBytes;
    
    /**
     * Total spilled bytes
     */
    private long totalSpilledBytes;
    
    /**
     * Total pinned bytes
     */
    private long totalPinnedBytes;
    
    /**
     * Total external allocations
     */
    private long totalExternalAllocations;
    
    /**
     * Get average workspace size
     */
    public double getAverageWorkspaceSize() {
        if (activeWorkspaceCount <= 0) return 0.0;
        return (double) totalWorkspaceBytes / activeWorkspaceCount;
    }
    
    /**
     * Get spill percentage
     */
    public double getSpillPercent() {
        if (totalWorkspaceBytes <= 0) return 0.0;
        return (totalSpilledBytes * 100.0) / totalWorkspaceBytes;
    }
}
