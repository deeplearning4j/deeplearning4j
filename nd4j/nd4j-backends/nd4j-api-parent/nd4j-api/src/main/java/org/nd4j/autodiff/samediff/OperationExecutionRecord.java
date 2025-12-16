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

/**
 * Record of a single operation execution
 */
@Data
public class OperationExecutionRecord {
    /**
     * Timestamp when execution occurred
     */
    private long timestamp;
    
    /**
     * Execution time in nanoseconds
     */
    private long executionTime;
    
    /**
     * Status of the execution
     */
    private OperationExecutionStatus status;
    
    /**
     * Number of input variables
     */
    private int inputCount;
    
    /**
     * Number of output variables
     */
    private int outputCount;
    
    /**
     * Loop iteration when this execution occurred (if applicable)
     */
    private int iteration = -1;
    
    /**
     * Frame name when this execution occurred (if applicable)
     */
    private String frame;
    
    /**
     * Error message if execution failed
     */
    private String errorMessage;
    
    /**
     * Memory usage during execution (in bytes)
     */
    private long memoryUsage = 0;
    
    /**
     * Additional execution context information
     */
    private java.util.Map<String, Object> executionContext = new java.util.HashMap<>();
    
    /**
     * Constructor for successful execution
     */
    public OperationExecutionRecord(long timestamp, long executionTime, OperationExecutionStatus status) {
        this.timestamp = timestamp;
        this.executionTime = executionTime;
        this.status = status;
    }
    
    /**
     * Constructor for failed execution
     */
    public OperationExecutionRecord(long timestamp, long executionTime, OperationExecutionStatus status, String errorMessage) {
        this(timestamp, executionTime, status);
        this.errorMessage = errorMessage;
    }
    
    /**
     * Default constructor
     */
    public OperationExecutionRecord() {
        this.timestamp = System.currentTimeMillis();
    }
    
    /**
     * Get execution time in milliseconds
     * 
     * @return execution time in milliseconds
     */
    public double getExecutionTimeMs() {
        return executionTime / 1_000_000.0;
    }
    
    /**
     * Check if this execution was successful
     * 
     * @return true if execution was successful
     */
    public boolean isSuccessful() {
        return status == OperationExecutionStatus.SUCCESS || status == OperationExecutionStatus.ANALYZED;
    }
    
    /**
     * Check if this execution had errors
     * 
     * @return true if execution had errors
     */
    public boolean hasError() {
        return status == OperationExecutionStatus.ERROR || errorMessage != null;
    }
    
    /**
     * Add context information
     * 
     * @param key context key
     * @param value context value
     */
    public void addContext(String key, Object value) {
        executionContext.put(key, value);
    }
    
    /**
     * Get context information
     * 
     * @param key context key
     * @return context value or null if not found
     */
    public Object getContext(String key) {
        return executionContext.get(key);
    }
    
    /**
     * Get formatted summary of this execution record
     * 
     * @return formatted summary string
     */
    public String getSummary() {
        StringBuilder summary = new StringBuilder();
        summary.append("Execution at ").append(new java.util.Date(timestamp));
        summary.append(": ").append(status);
        summary.append(" (").append(String.format("%.2f", getExecutionTimeMs())).append("ms)");
        
        if (frame != null && iteration >= 0) {
            summary.append(" [").append(frame).append(":").append(iteration).append("]");
        }
        
        if (hasError()) {
            summary.append(" ERROR: ").append(errorMessage);
        }
        
        return summary.toString();
    }
}
