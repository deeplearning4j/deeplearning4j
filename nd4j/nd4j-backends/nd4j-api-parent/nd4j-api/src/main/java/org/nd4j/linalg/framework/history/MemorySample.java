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
 * Time-series sample for memory tracking.
 * 
 * Captures a point-in-time snapshot of memory usage across
 * different memory types (heap, off-heap, device, workspace).
 * 
 * @author ND4J Team
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class MemorySample {
    
    /**
     * Timestamp when sample was taken (milliseconds since epoch)
     */
    private long timestampMs;
    
    /**
     * JVM heap memory used (bytes)
     */
    private long heapUsedBytes;
    
    /**
     * JVM heap memory maximum (bytes)
     */
    private long heapMaxBytes;
    
    /**
     * Off-heap/direct buffer memory used (bytes)
     */
    private long offHeapUsedBytes;
    
    /**
     * Device (CUDA/GPU) memory used (bytes)
     */
    private long deviceUsedBytes;
    
    /**
     * Workspace memory used (bytes)
     */
    private long workspaceUsedBytes;
    
    /**
     * Memory pool reserved memory (bytes)
     */
    private long poolReservedBytes;
    
    /**
     * Number of live INDArray objects
     */
    private long liveArrayCount;
    
    /**
     * Get total memory used across all types
     */
    public long getTotalUsedBytes() {
        return heapUsedBytes + offHeapUsedBytes + deviceUsedBytes + workspaceUsedBytes;
    }
    
    /**
     * Get heap utilization percentage
     */
    public double getHeapUtilizationPercent() {
        if (heapMaxBytes <= 0) return 0.0;
        return (heapUsedBytes * 100.0) / heapMaxBytes;
    }
}
