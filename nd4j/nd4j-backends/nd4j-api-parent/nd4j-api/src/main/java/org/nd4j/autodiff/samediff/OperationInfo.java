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
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.config.SDValue;
import org.nd4j.autodiff.samediff.internal.FrameIter;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.internal.VarId;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;

/**
 * Enhanced OperationInfo class for comprehensive operation analysis and error reporting.
 * 
 * This class provides detailed information about operations including their execution context,
 * input/output analysis, error conditions, and integration with loop termination analysis.
 */
@Data
@Slf4j
public class OperationInfo {
    
    // === BASIC OPERATION INFORMATION ===
    
    /**
     * The unique name of the operation
     */
    private final String operationName;
    
    /**
     * The type of operation (e.g., "Add", "MatMul", "LoopCond")
     */
    private final String operationType;
    
    /**
     * The full class name of the operation
     */
    private final String className;
    
    /**
     * List of input variable names
     */
    private final List<String> inputs;
    
    /**
     * List of output variable names
     */
    private final List<String> outputs;
    
    /**
     * Frame information for this operation
     */
    private FrameInfo frameInfo;
    
    // === ENHANCED EXECUTION CONTEXT ===
    
    /**
     * Current input values at the time of analysis
     */
    private Map<String, Object> inputValues = new HashMap<>();
    
    /**
     * Current output values at the time of analysis
     */
    private Map<String, Object> outputValues = new HashMap<>();
    
    /**
     * Execution status of this operation
     */
    private OperationExecutionStatus executionStatus = OperationExecutionStatus.UNKNOWN;
    
    /**
     * Error information if operation failed
     */
    private OperationErrorInfo errorInfo;
    
    /**
     * Execution timing information
     */
    private OperationTimingInfo timingInfo = new OperationTimingInfo();
    
    /**
     * Memory usage information for this operation
     */
    private OperationMemoryInfo memoryInfo = new OperationMemoryInfo();
    
    /**
     * Operation-specific metadata
     */
    private Map<String, Object> metadata = new HashMap<>();
    
    /**
     * Execution history for this operation
     */
    private List<OperationExecutionRecord> executionHistory = new ArrayList<>();
    
    /**
     * Dependencies and relationships
     */
    private OperationDependencyInfo dependencyInfo = new OperationDependencyInfo();
    
    // === LOOP-SPECIFIC INFORMATION ===
    
    /**
     * Role of this operation in loop control flow
     */
    private LoopOperationRole loopRole = LoopOperationRole.REGULAR;
    
    /**
     * Loop iteration context when this operation was analyzed
     */
    private FrameIter loopContext;
    
    /**
     * Whether this operation is critical for loop termination
     */
    private boolean isTerminationCritical = false;
    
    /**
     * Loop-specific execution patterns
     */
    private Map<Integer, Object> iterationResults = new HashMap<>();
    
    // === CONSTRUCTORS ===
    
    /**
     * Basic constructor maintaining backward compatibility
     */
    public OperationInfo(String name, String opType, String className, List<String> inputs, List<String> outputs) {
        this.operationName = name;
        this.operationType = opType;
        this.className = className;
        this.inputs = inputs != null ? new ArrayList<>(inputs) : new ArrayList<>();
        this.outputs = outputs != null ? new ArrayList<>(outputs) : new ArrayList<>();
        
        // Initialize enhanced properties
        initializeDefaults();
    }
    
    /**
     * Enhanced constructor with frame context
     */
    public OperationInfo(String name, String opType, String className, List<String> inputs, List<String> outputs,
                        FrameInfo frameInfo) {
        this(name, opType, className, inputs, outputs);
        this.frameInfo = frameInfo;
        
        // Determine loop role based on operation type
        this.loopRole = OperationAnalysisUtils.determineLoopRole(opType);
        this.isTerminationCritical = OperationAnalysisUtils.isTerminationCriticalOperation(opType);
    }
    
    /**
     * Full constructor with execution context
     */
    public OperationInfo(String name, String opType, String className, List<String> inputs, List<String> outputs,
                        FrameInfo frameInfo, FrameIter loopContext) {
        this(name, opType, className, inputs, outputs, frameInfo);
        this.loopContext = loopContext;
    }
    
    // === INITIALIZATION ===
    
    /**
     * Initialize default values
     */
    private void initializeDefaults() {
        this.metadata.put("created_at", System.currentTimeMillis());
        this.metadata.put("analysis_version", "2.0");
        
        // Initialize loop role
        this.loopRole = OperationAnalysisUtils.determineLoopRole(operationType);
        this.isTerminationCritical = OperationAnalysisUtils.isTerminationCriticalOperation(operationType);
    }
    
    // === ANALYSIS METHODS ===
    
    /**
     * Analyze operation with current SameDiff state
     */
    public void analyzeWithCurrentState(SameDiff sameDiff, Map<VarId, SDValue> nodeValueOutputs) {
        analyzeWithCurrentState(sameDiff, nodeValueOutputs, null);
    }
    
    /**
     * Analyze operation with current SameDiff state and specific frame context
     */
    public void analyzeWithCurrentState(SameDiff sameDiff, Map<VarId, SDValue> nodeValueOutputs,
                                       FrameIter frameContext) {
        long startTime = System.nanoTime();
        
        try {
            // Update loop context if provided
            if (frameContext != null) {
                this.loopContext = frameContext;
            }
            
            // Analyze input values
            analyzeInputValues(nodeValueOutputs, frameContext);
            
            // Analyze output values
            analyzeOutputValues(nodeValueOutputs, frameContext);
            
            // Analyze operation dependencies
            analyzeDependencies(sameDiff);
            
            // Update memory usage
            memoryInfo.updateTotalMemoryUsage();
            
            // Record execution
            recordExecution(startTime, OperationExecutionStatus.ANALYZED);
            
        } catch (Exception e) {
            recordError(e, startTime);
            log.warn("Error analyzing operation '{}': {}", operationName, e.getMessage());
        }
    }
    
    /**
     * Analyze input values from node outputs
     */
    private void analyzeInputValues(Map<VarId, SDValue> nodeValueOutputs, FrameIter frameContext) {
        inputValues.clear();
        
        for (String inputName : inputs) {
            try {
                VarId varId = OperationAnalysisUtils.createVarId(inputName, frameContext, frameInfo);
                SDValue value = nodeValueOutputs.get(varId);
                
                if (value != null) {
                    Object extractedValue = OperationAnalysisUtils.extractValueForAnalysis(value);
                    inputValues.put(inputName, extractedValue);
                    
                    // Update memory usage
                    long memUsage = OperationAnalysisUtils.estimateValueMemoryUsage(value);
                    memoryInfo.addInputMemoryUsage(inputName, memUsage);
                } else {
                    inputValues.put(inputName, null);
                }
            } catch (Exception e) {
                log.debug("Could not analyze input '{}' for operation '{}': {}", inputName, operationName, e.getMessage());
                inputValues.put(inputName, "ERROR: " + e.getMessage());
            }
        }
    }
    
    /**
     * Analyze output values from node outputs
     */
    private void analyzeOutputValues(Map<VarId, SDValue> nodeValueOutputs, FrameIter frameContext) {
        outputValues.clear();
        
        for (String outputName : outputs) {
            try {
                VarId varId = OperationAnalysisUtils.createVarId(outputName, frameContext, frameInfo);
                SDValue value = nodeValueOutputs.get(varId);
                
                if (value != null) {
                    Object extractedValue = OperationAnalysisUtils.extractValueForAnalysis(value);
                    outputValues.put(outputName, extractedValue);
                    
                    // Update memory usage
                    long memUsage = OperationAnalysisUtils.estimateValueMemoryUsage(value);
                    memoryInfo.addOutputMemoryUsage(outputName, memUsage);
                    
                    // Store iteration result for loop analysis
                    if (loopContext != null && isTerminationCritical) {
                        iterationResults.put(loopContext.getIteration(), extractedValue);
                    }
                } else {
                    outputValues.put(outputName, null);
                }
            } catch (Exception e) {
                log.debug("Could not analyze output '{}' for operation '{}': {}", outputName, operationName, e.getMessage());
                outputValues.put(outputName, "ERROR: " + e.getMessage());
            }
        }
    }
    
    /**
     * Analyze operation dependencies
     */
    private void analyzeDependencies(SameDiff sameDiff) {
        dependencyInfo.clearDependencies();
        
        // Find operations that produce our inputs
        for (String inputName : inputs) {
            SameDiffOp producerOp = OperationAnalysisUtils.findProducerOperation(sameDiff, inputName);
            if (producerOp != null) {
                dependencyInfo.addInputDependency(inputName, producerOp.getName());
            }
        }
        
        // Find operations that consume our outputs
        for (String outputName : outputs) {
            List<SameDiffOp> consumerOps = OperationAnalysisUtils.findConsumerOperations(sameDiff, outputName);
            for (SameDiffOp consumerOp : consumerOps) {
                dependencyInfo.addOutputDependency(outputName, consumerOp.getName());
            }
        }
    }
    
    /**
     * Record successful execution
     */
    private void recordExecution(long startTime, OperationExecutionStatus status) {
        long endTime = System.nanoTime();
        long executionTime = endTime - startTime;
        
        this.executionStatus = status;
        this.timingInfo.addExecutionTime(executionTime);
        
        OperationExecutionRecord record = new OperationExecutionRecord();
        record.setTimestamp(System.currentTimeMillis());
        record.setExecutionTime(executionTime);
        record.setStatus(status);
        record.setInputCount(inputs.size());
        record.setOutputCount(outputs.size());
        
        if (loopContext != null) {
            record.setIteration(loopContext.getIteration());
            record.setFrame(loopContext.getFrame());
        }
        
        executionHistory.add(record);
        
        // Keep history limited
        if (executionHistory.size() > 100) {
            executionHistory.remove(0);
        }
    }
    
    /**
     * Record execution error
     */
    private void recordError(Exception error, long startTime) {
        long endTime = System.nanoTime();
        long executionTime = endTime - startTime;
        
        this.executionStatus = OperationExecutionStatus.ERROR;
        
        this.errorInfo = new OperationErrorInfo();
        this.errorInfo.setErrorMessage(error.getMessage());
        this.errorInfo.setErrorType(error.getClass().getSimpleName());
        this.errorInfo.setTimestamp(System.currentTimeMillis());
        this.errorInfo.setStackTrace(getStackTrace(error));
        
        OperationExecutionRecord record = new OperationExecutionRecord();
        record.setTimestamp(System.currentTimeMillis());
        record.setExecutionTime(executionTime);
        record.setStatus(OperationExecutionStatus.ERROR);
        record.setErrorMessage(error.getMessage());
        
        if (loopContext != null) {
            record.setIteration(loopContext.getIteration());
            record.setFrame(loopContext.getFrame());
        }
        
        executionHistory.add(record);
    }
    
    /**
     * Get stack trace as string
     */
    private String getStackTrace(Exception e) {
        java.io.StringWriter sw = new java.io.StringWriter();
        java.io.PrintWriter pw = new java.io.PrintWriter(sw);
        e.printStackTrace(pw);
        return sw.toString();
    }
    
    // === QUERY METHODS ===
    
    /**
     * Check if this operation is a loop control operation
     */
    public boolean isLoopControlOperation() {
        return loopRole != LoopOperationRole.REGULAR;
    }
    
    /**
     * Check if this operation has execution errors
     */
    public boolean hasExecutionErrors() {
        return executionStatus == OperationExecutionStatus.ERROR || errorInfo != null;
    }
    
    /**
     * Get the most recent execution record
     */
    public OperationExecutionRecord getLatestExecutionRecord() {
        if (executionHistory.isEmpty()) return null;
        return executionHistory.get(executionHistory.size() - 1);
    }
    
    /**
     * Get execution records for a specific iteration (loop context)
     */
    public List<OperationExecutionRecord> getExecutionRecordsForIteration(int iteration) {
        return executionHistory.stream()
                .filter(record -> record.getIteration() == iteration)
                .collect(java.util.stream.Collectors.toList());
    }
    
    /**
     * Get input value for a specific input name
     */
    public Object getInputValue(String inputName) {
        return inputValues.get(inputName);
    }
    
    /**
     * Get output value for a specific output name
     */
    public Object getOutputValue(String outputName) {
        return outputValues.get(outputName);
    }
    
    /**
     * Check if operation has problematic values (NaN, Inf, etc.)
     */
    public boolean hasProblematicValues() {
        // Check input values
        for (Object value : inputValues.values()) {
            if (OperationAnalysisUtils.isProblematicValue(value)) return true;
        }
        
        // Check output values
        for (Object value : outputValues.values()) {
            if (OperationAnalysisUtils.isProblematicValue(value)) return true;
        }
        
        return false;
    }
    
    /**
     * Get problematic value details
     */
    public List<String> getProblematicValueDetails() {
        List<String> details = new ArrayList<>();
        
        // Check inputs
        for (Map.Entry<String, Object> entry : inputValues.entrySet()) {
            if (OperationAnalysisUtils.isProblematicValue(entry.getValue())) {
                details.add("Input '" + entry.getKey() + "': " + 
                           OperationAnalysisUtils.describeProblem(entry.getValue()));
            }
        }
        
        // Check outputs
        for (Map.Entry<String, Object> entry : outputValues.entrySet()) {
            if (OperationAnalysisUtils.isProblematicValue(entry.getValue())) {
                details.add("Output '" + entry.getKey() + "': " + 
                           OperationAnalysisUtils.describeProblem(entry.getValue()));
            }
        }
        
        return details;
    }
    
    /**
     * Get average execution time
     */
    public double getAverageExecutionTime() {
        return timingInfo.getAverageExecutionTime();
    }
    
    /**
     * Get total memory usage
     */
    public long getTotalMemoryUsage() {
        return memoryInfo.getTotalMemoryUsage();
    }
    
    /**
     * Check if operation execution time is abnormally high
     */
    public boolean hasAbnormalExecutionTime() {
        return timingInfo.isAbnormalExecutionTime(3.0); // 3x average threshold
    }
    
    /**
     * Get execution summary for loop analysis
     */
    public String getExecutionSummary() {
        StringBuilder summary = new StringBuilder();
        summary.append("Operation: ").append(operationName);
        summary.append(" (").append(operationType).append(")");
        summary.append(" [").append(loopRole).append("]");
        
        if (loopContext != null) {
            summary.append(" Frame: ").append(loopContext.getFrame());
            summary.append(" Iter: ").append(loopContext.getIteration());
        }
        
        summary.append(" Status: ").append(executionStatus);
        
        if (hasExecutionErrors()) {
            summary.append(" ERROR: ").append(errorInfo.getErrorMessage());
        }
        
        if (hasProblematicValues()) {
            summary.append(" [PROBLEMATIC VALUES]");
        }
        
        return summary.toString();
    }
    
    /**
     * Generate detailed analysis report
     */
    public String generateDetailedReport() {
        StringBuilder report = new StringBuilder();
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
        
        report.append("=== OPERATION ANALYSIS REPORT ===\n");
        report.append("Name: ").append(operationName).append("\n");
        report.append("Type: ").append(operationType).append("\n");
        report.append("Class: ").append(className).append("\n");
        report.append("Loop Role: ").append(loopRole).append("\n");
        report.append("Termination Critical: ").append(isTerminationCritical).append("\n");
        report.append("Status: ").append(executionStatus).append("\n");
        
        if (loopContext != null) {
            report.append("Frame: ").append(loopContext.getFrame()).append("\n");
            report.append("Iteration: ").append(loopContext.getIteration()).append("\n");
        }
        
        // Input/Output summary
        report.append("\nInputs (").append(inputs.size()).append("): ").append(inputs).append("\n");
        report.append("Outputs (").append(outputs.size()).append("): ").append(outputs).append("\n");
        
        // Values (if available)
        if (!inputValues.isEmpty()) {
            report.append("\nInput Values:\n");
            for (Map.Entry<String, Object> entry : inputValues.entrySet()) {
                report.append("  ").append(entry.getKey()).append(" = ")
                      .append(OperationAnalysisUtils.formatValue(entry.getValue())).append("\n");
            }
        }
        
        if (!outputValues.isEmpty()) {
            report.append("\nOutput Values:\n");
            for (Map.Entry<String, Object> entry : outputValues.entrySet()) {
                report.append("  ").append(entry.getKey()).append(" = ")
                      .append(OperationAnalysisUtils.formatValue(entry.getValue())).append("\n");
            }
        }
        
        // Error information
        if (hasExecutionErrors()) {
            report.append("\nERROR INFORMATION:\n");
            report.append("Error Type: ").append(errorInfo.getErrorType()).append("\n");
            report.append("Error Message: ").append(errorInfo.getErrorMessage()).append("\n");
            report.append("Error Time: ").append(
                LocalDateTime.ofInstant(java.time.Instant.ofEpochMilli(errorInfo.getTimestamp()),
                                      java.time.ZoneId.systemDefault()).format(formatter)).append("\n");
        }
        
        // Performance metrics
        report.append("\nPERFORMANCE METRICS:\n");
        report.append("Executions: ").append(executionHistory.size()).append("\n");
        report.append(timingInfo.getTimingStatistics()).append("\n");
        report.append(memoryInfo.getMemoryUsageSummary()).append("\n");
        
        // Problematic values
        List<String> problems = getProblematicValueDetails();
        if (!problems.isEmpty()) {
            report.append("\nPROBLEMATIC VALUES:\n");
            for (String problem : problems) {
                report.append("  ⚠️  ").append(problem).append("\n");
            }
        }
        
        // Dependencies
        if (dependencyInfo.hasInputDependencies() || dependencyInfo.hasOutputDependencies()) {
            report.append("\nDEPENDENCIES:\n");
            report.append(dependencyInfo.getDependencySummary()).append("\n");
        }
        
        return report.toString();
    }
    
    // === DEPRECATED COMPATIBILITY ===
    
    /**
     * @deprecated Use operationName instead
     */
    @Deprecated
    public String getName() {
        return operationName;
    }
    
    /**
     * @deprecated Use operationType instead
     */
    @Deprecated
    public String getOpType() {
        return operationType;
    }
    
    /**
     * @deprecated Access frameInfo directly
     */
    @Deprecated
    public FrameInfo getFrameInfo() {
        return frameInfo;
    }
    
    /**
     * @deprecated Use inputs field directly
     */
    @Deprecated
    public List<String> getInputs() {
        return inputs;
    }
    
    /**
     * @deprecated Use outputs field directly
     */
    @Deprecated
    public List<String> getOutputs() {
        return outputs;
    }
}
