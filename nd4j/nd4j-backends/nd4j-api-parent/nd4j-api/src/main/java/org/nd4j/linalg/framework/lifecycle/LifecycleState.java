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

package org.nd4j.linalg.framework.lifecycle;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Lifecycle subsystem state snapshot.
 * 
 * @author ND4J Team
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class LifecycleState {
    
    /**
     * Total arrays created
     */
    private long totalCreated;
    
    /**
     * Total arrays destroyed
     */
    private long totalDestroyed;
    
    /**
     * Currently live arrays
     */
    private long liveArrays;
    
    /**
     * Whether lifecycle tracking is enabled
     */
    private boolean trackingEnabled;
    
    /**
     * Whether stack trace capture is enabled
     */
    private boolean stackTraceCapture;
    
    /**
     * Leak detection age threshold (ms)
     */
    private long leakDetectionAgeThresholdMs;
    
    /**
     * Get live array count
     */
    public long getLiveArrays() {
        return liveArrays;
    }
    
    /**
     * Get destruction efficiency (destroyed/created)
     */
    public double getDestructionEfficiency() {
        if (totalCreated <= 0) return 1.0;
        return (double) totalDestroyed / totalCreated;
    }
    
    /**
     * Get net creation (created - destroyed)
     */
    public long getNetCreation() {
        return totalCreated - totalDestroyed;
    }
    
    /**
     * Get leak detection age threshold in seconds
     */
    public double getLeakDetectionAgeThresholdSeconds() {
        return leakDetectionAgeThresholdMs / 1000.0;
    }
    
    @Override
    public String toString() {
        return String.format(
            "Created: %d, Destroyed: %d, Live: %d, Efficiency: %.1f%%, " +
            "Tracking: %s, StackTrace: %s, Leak Threshold: %.0f s",
            totalCreated,
            totalDestroyed,
            liveArrays,
            getDestructionEfficiency() * 100,
            trackingEnabled ? "ON" : "OFF",
            stackTraceCapture ? "ON" : "OFF",
            getLeakDetectionAgeThresholdSeconds()
        );
    }
}
