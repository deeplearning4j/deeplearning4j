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

package org.nd4j.linalg.framework.history;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Time-series sample for workspace tracking.
 * 
 * @author ND4J Team
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class WorkspaceSample {
    
    /**
     * Timestamp when sample was taken (milliseconds since epoch)
     */
    private long timestampMs;
    
    /**
     * Workspace ID
     */
    private String workspaceId;
    
    /**
     * Current workspace size (bytes)
     */
    private long currentSizeBytes;
    
    /**
     * Maximum workspace size (bytes)
     */
    private long maxSizeBytes;
    
    /**
     * Bytes spilled to external allocation
     */
    private long spilledBytes;
    
    /**
     * Bytes in pinned allocations
     */
    private long pinnedBytes;
    
    /**
     * Number of allocations in this workspace
     */
    private long allocationCount;
    
    /**
     * Get utilization percentage
     */
    public double getUtilizationPercent() {
        if (maxSizeBytes <= 0) return 0.0;
        return (currentSizeBytes * 100.0) / maxSizeBytes;
    }
    
    /**
     * Get spill percentage
     */
    public double getSpillPercent() {
        if (currentSizeBytes <= 0) return 0.0;
        return (spilledBytes * 100.0) / currentSizeBytes;
    }
}
