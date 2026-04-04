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
 * Statistics summary for array lifecycle tracking.
 * 
 * @author ND4J Team
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class LifecycleStats {
    
    /**
     * Total arrays created
     */
    private long totalCreated;
    
    /**
     * Total arrays destroyed
     */
    private long totalDestroyed;
    
    /**
     * Total workspace allocations
     */
    private long totalWorkspaceAllocated;
    
    /**
     * Current live arrays (created - destroyed)
     */
    private long currentLive;
    
    /**
     * Get net array creation rate (positive = growing, negative = shrinking)
     */
    public long getNetCreation() {
        return totalCreated - totalDestroyed;
    }
    
    /**
     * Get destruction efficiency (destroyed / created)
     */
    public double getDestructionEfficiency() {
        if (totalCreated <= 0) return 1.0;
        return (double) totalDestroyed / (double) totalCreated;
    }
}
