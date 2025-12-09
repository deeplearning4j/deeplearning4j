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

package org.nd4j.autodiff.samediff;

import lombok.Data;

import java.util.HashMap;
import java.util.Map;

/**
 * Memory usage information for operations
 */
@Data
public class OperationMemoryInfo {
    /**
     * Memory usage for each input variable (in bytes)
     */
    private Map<String, Long> inputMemoryUsage = new HashMap<>();
    
    /**
     * Memory usage for each output variable (in bytes)
     */
    private Map<String, Long> outputMemoryUsage = new HashMap<>();
    
    /**
     * Total memory usage for this operation (in bytes)
     */
    private long totalMemoryUsage = 0;
    
    /**
     * Peak memory usage observed (in bytes)
     */
    private long peakMemoryUsage = 0;
    
    /**
     * Memory allocated during operation execution (in bytes)
     */
    private long allocatedMemory = 0;
    
    /**
     * Memory deallocated after operation execution (in bytes)
     */
    private long deallocatedMemory = 0;
    
    /**
     * Timestamp when memory usage was last updated
     */
    private long lastUpdated = 0;
    
    /**
     * Add memory usage for an input variable
     * 
     * @param inputName name of the input variable
     * @param memoryUsage memory usage in bytes
     */
    public void addInputMemoryUsage(String inputName, long memoryUsage) {
        inputMemoryUsage.put(inputName, memoryUsage);
        updateTotalMemoryUsage();
    }
    
    /**
     * Add memory usage for an output variable
     * 
     * @param outputName name of the output variable
     * @param memoryUsage memory usage in bytes
     */
    public void addOutputMemoryUsage(String outputName, long memoryUsage) {
        outputMemoryUsage.put(outputName, memoryUsage);
        updateTotalMemoryUsage();
    }
    
    /**
     * Update total memory usage calculation
     */
    public void updateTotalMemoryUsage() {
        long inputTotal = inputMemoryUsage.values().stream().mapToLong(Long::longValue).sum();
        long outputTotal = outputMemoryUsage.values().stream().mapToLong(Long::longValue).sum();
        
        totalMemoryUsage = inputTotal + outputTotal + allocatedMemory - deallocatedMemory;
        peakMemoryUsage = Math.max(peakMemoryUsage, totalMemoryUsage);
        lastUpdated = System.currentTimeMillis();
    }
    
    /**
     * Record memory allocation during operation
     * 
     * @param allocated bytes allocated
     */
    public void recordAllocation(long allocated) {
        this.allocatedMemory += allocated;
        updateTotalMemoryUsage();
    }
    
    /**
     * Record memory deallocation during operation
     * 
     * @param deallocated bytes deallocated
     */
    public void recordDeallocation(long deallocated) {
        this.deallocatedMemory += deallocated;
        updateTotalMemoryUsage();
    }
    
    /**
     * Get total input memory usage
     * 
     * @return total memory usage of all inputs in bytes
     */
    public long getTotalInputMemoryUsage() {
        return inputMemoryUsage.values().stream().mapToLong(Long::longValue).sum();
    }
    
    /**
     * Get total output memory usage
     * 
     * @return total memory usage of all outputs in bytes
     */
    public long getTotalOutputMemoryUsage() {
        return outputMemoryUsage.values().stream().mapToLong(Long::longValue).sum();
    }
    
    /**
     * Get memory usage for a specific input
     * 
     * @param inputName name of the input variable
     * @return memory usage in bytes, or 0 if not found
     */
    public long getInputMemoryUsage(String inputName) {
        return inputMemoryUsage.getOrDefault(inputName, 0L);
    }
    
    /**
     * Get memory usage for a specific output
     * 
     * @param outputName name of the output variable
     * @return memory usage in bytes, or 0 if not found
     */
    public long getOutputMemoryUsage(String outputName) {
        return outputMemoryUsage.getOrDefault(outputName, 0L);
    }
    
    /**
     * Check if memory usage is considered high
     * 
     * @param thresholdBytes threshold in bytes
     * @return true if total memory usage exceeds threshold
     */
    public boolean isHighMemoryUsage(long thresholdBytes) {
        return totalMemoryUsage > thresholdBytes;
    }
    
    /**
     * Get memory usage in MB
     * 
     * @return total memory usage in megabytes
     */
    public double getTotalMemoryUsageMB() {
        return totalMemoryUsage / (1024.0 * 1024.0);
    }
    
    /**
     * Get peak memory usage in MB
     * 
     * @return peak memory usage in megabytes
     */
    public double getPeakMemoryUsageMB() {
        return peakMemoryUsage / (1024.0 * 1024.0);
    }
    
    /**
     * Get memory efficiency (ratio of current to peak usage)
     * 
     * @return memory efficiency ratio (0.0 to 1.0)
     */
    public double getMemoryEfficiency() {
        if (peakMemoryUsage == 0) return 1.0;
        return (double) totalMemoryUsage / peakMemoryUsage;
    }
    
    /**
     * Get memory usage summary as formatted string
     * 
     * @return formatted string with memory statistics
     */
    public String getMemoryUsageSummary() {
        return String.format("Memory: %.2f MB (peak: %.2f MB), Inputs: %.2f MB, Outputs: %.2f MB",
                getTotalMemoryUsageMB(),
                getPeakMemoryUsageMB(),
                getTotalInputMemoryUsage() / (1024.0 * 1024.0),
                getTotalOutputMemoryUsage() / (1024.0 * 1024.0));
    }
    
    /**
     * Reset all memory usage information
     */
    public void reset() {
        inputMemoryUsage.clear();
        outputMemoryUsage.clear();
        totalMemoryUsage = 0;
        peakMemoryUsage = 0;
        allocatedMemory = 0;
        deallocatedMemory = 0;
        lastUpdated = 0;
    }
}
