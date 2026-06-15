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

package org.nd4j.linalg.framework.memory;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Memory subsystem state snapshot.
 * 
 * @author ND4J Team
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class MemoryState {
    
    /**
     * Heap memory used (bytes)
     */
    private long heapUsedBytes;
    
    /**
     * Heap memory max (bytes)
     */
    private long heapMaxBytes;
    
    /**
     * Off-heap memory used (bytes)
     */
    private long offHeapUsedBytes;
    
    /**
     * Device memory used (bytes)
     */
    private long deviceUsedBytes;
    
    /**
     * Workspace memory used (bytes)
     */
    private long workspaceUsedBytes;
    
    /**
     * Live array count
     */
    private long liveArrayCount;
    
    /**
     * GC frequency
     */
    private int gcFrequency;
    
    /**
     * Whether periodic GC is enabled
     */
    private boolean periodicGcEnabled;
    
    /**
     * Whether array GC is disabled
     */
    private boolean noArrayGc;
    
    /**
     * Memory pressure threshold (0.0-1.0)
     */
    private double memoryPressureThreshold;
    
    /**
     * Get heap used in MB
     */
    public double getHeapUsedMb() {
        return heapUsedBytes / (1024.0 * 1024.0);
    }
    
    /**
     * Get heap max in MB
     */
    public double getHeapMaxMb() {
        return heapMaxBytes / (1024.0 * 1024.0);
    }
    
    /**
     * Get heap utilization percentage
     */
    public double getHeapUtilizationPercent() {
        if (heapMaxBytes <= 0) return 0.0;
        return (heapUsedBytes * 100.0) / heapMaxBytes;
    }
    
    /**
     * Get off-heap in MB
     */
    public double getOffHeapMb() {
        return offHeapUsedBytes / (1024.0 * 1024.0);
    }
    
    /**
     * Get device memory in MB
     */
    public double getDeviceMb() {
        return deviceUsedBytes / (1024.0 * 1024.0);
    }
    
    /**
     * Get workspace memory in MB
     */
    public double getWorkspaceMb() {
        return workspaceUsedBytes / (1024.0 * 1024.0);
    }
    
    /**
     * Get total memory used in MB
     */
    public double getTotalUsedMb() {
        return (heapUsedBytes + offHeapUsedBytes + deviceUsedBytes + workspaceUsedBytes) / (1024.0 * 1024.0);
    }
    
    @Override
    public String toString() {
        return String.format(
            "Heap: %.1f/%.1f MB (%.1f%%), Off-heap: %.1f MB, Device: %.1f MB, " +
            "Workspace: %.1f MB, Arrays: %d, GC: %d/%s, Pressure: %.0f%%",
            getHeapUsedMb(), getHeapMaxMb(), getHeapUtilizationPercent(),
            getOffHeapMb(),
            getDeviceMb(),
            getWorkspaceMb(),
            liveArrayCount,
            gcFrequency,
            periodicGcEnabled ? "ON" : "OFF",
            memoryPressureThreshold * 100
        );
    }
}
