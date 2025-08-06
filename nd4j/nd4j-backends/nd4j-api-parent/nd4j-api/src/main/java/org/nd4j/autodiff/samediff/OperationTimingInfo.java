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

import java.util.ArrayList;
import java.util.List;

/**
 * Timing information for operation execution
 */
@Data
public class OperationTimingInfo {
    /**
     * List of execution times for this operation (in nanoseconds)
     */
    private List<Long> executionTimes = new ArrayList<>();
    
    /**
     * Total accumulated execution time (in nanoseconds)
     */
    private long totalExecutionTime = 0;
    
    /**
     * Minimum execution time observed (in nanoseconds)
     */
    private long minExecutionTime = Long.MAX_VALUE;
    
    /**
     * Maximum execution time observed (in nanoseconds)
     */
    private long maxExecutionTime = Long.MIN_VALUE;
    
    /**
     * Number of times this operation has been executed
     */
    private int executionCount = 0;
    
    /**
     * Timestamp of first execution
     */
    private long firstExecutionTime = 0;
    
    /**
     * Timestamp of last execution
     */
    private long lastExecutionTime = 0;
    
    /**
     * Add a new execution time measurement
     * 
     * @param executionTime execution time in nanoseconds
     */
    public void addExecutionTime(long executionTime) {
        executionTimes.add(executionTime);
        totalExecutionTime += executionTime;
        
        minExecutionTime = Math.min(minExecutionTime, executionTime);
        maxExecutionTime = Math.max(maxExecutionTime, executionTime);
        
        executionCount++;
        lastExecutionTime = System.currentTimeMillis();
        
        if (firstExecutionTime == 0) {
            firstExecutionTime = lastExecutionTime;
        }
        
        // Keep only recent execution times to avoid memory issues
        if (executionTimes.size() > 1000) {
            executionTimes.remove(0);
        }
    }
    
    /**
     * Get average execution time in nanoseconds
     * 
     * @return average execution time, or 0 if no executions recorded
     */
    public double getAverageExecutionTime() {
        if (executionCount == 0) return 0.0;
        return (double) totalExecutionTime / executionCount;
    }
    
    /**
     * Get average execution time in milliseconds
     * 
     * @return average execution time in milliseconds
     */
    public double getAverageExecutionTimeMs() {
        return getAverageExecutionTime() / 1_000_000.0;
    }
    
    /**
     * Get total execution time in milliseconds
     * 
     * @return total execution time in milliseconds
     */
    public double getTotalExecutionTimeMs() {
        return totalExecutionTime / 1_000_000.0;
    }
    
    /**
     * Get minimum execution time in milliseconds
     * 
     * @return minimum execution time in milliseconds
     */
    public double getMinExecutionTimeMs() {
        if (minExecutionTime == Long.MAX_VALUE) return 0.0;
        return minExecutionTime / 1_000_000.0;
    }
    
    /**
     * Get maximum execution time in milliseconds
     * 
     * @return maximum execution time in milliseconds
     */
    public double getMaxExecutionTimeMs() {
        if (maxExecutionTime == Long.MIN_VALUE) return 0.0;
        return maxExecutionTime / 1_000_000.0;
    }
    
    /**
     * Check if execution time is abnormally high compared to average
     * 
     * @param threshold multiplier for average (e.g., 3.0 for 3x average)
     * @return true if last execution was abnormally high
     */
    public boolean isAbnormalExecutionTime(double threshold) {
        if (executionTimes.isEmpty()) return false;
        
        double avgTime = getAverageExecutionTime();
        if (avgTime <= 0) return false;
        
        long lastExecution = executionTimes.get(executionTimes.size() - 1);
        return lastExecution > avgTime * threshold;
    }
    
    /**
     * Get execution time statistics as a summary string
     * 
     * @return formatted string with timing statistics
     */
    public String getTimingStatistics() {
        if (executionCount == 0) {
            return "No executions recorded";
        }
        
        return String.format("Executions: %d, Avg: %.2fms, Min: %.2fms, Max: %.2fms, Total: %.2fms",
                executionCount,
                getAverageExecutionTimeMs(),
                getMinExecutionTimeMs(),
                getMaxExecutionTimeMs(),
                getTotalExecutionTimeMs());
    }
    
    /**
     * Reset all timing information
     */
    public void reset() {
        executionTimes.clear();
        totalExecutionTime = 0;
        minExecutionTime = Long.MAX_VALUE;
        maxExecutionTime = Long.MIN_VALUE;
        executionCount = 0;
        firstExecutionTime = 0;
        lastExecutionTime = 0;
    }
}
