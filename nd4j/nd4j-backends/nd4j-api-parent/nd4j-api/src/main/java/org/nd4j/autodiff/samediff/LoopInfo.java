package org.nd4j.autodiff.samediff;/*
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


import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.linalg.api.ops.impl.controlflow.compat.*;

import java.util.*;
import java.util.stream.Collectors;

/**
 * LoopInfo holds comprehensive information about a loop during execution.
 * This class tracks all aspects of loop behavior including operations, variables,
 * execution state, termination predictions, and performance metrics.
 */
@Data
@NoArgsConstructor
public class LoopInfo {

    // === BASIC LOOP IDENTIFICATION ===

    /**
     * The name of the loop frame (e.g., "while_loop_1", "for_loop_2")
     */
    private String frameName;

    /**
     * Unique identifier for this loop instance
     */
    private String loopId;

    /**
     * The parent frame name, if this is a nested loop
     */
    private String parentFrameName;

    /**
     * The depth of loop nesting (0 for outermost loop)
     */
    private int nestingDepth = 0;

    // === LOOP OPERATIONS ===

    /**
     * The main loop condition operation (typically LoopCond)
     */
    private String loopCondOperation;

    /**
     * All exit operations that can terminate this loop
     */
    private List<String> exitOperations = new ArrayList<>();

    /**
     * All switch operations within this loop
     */
    private List<String> switchOperations = new ArrayList<>();

    /**
     * All NextIteration operations that advance the loop
     */
    private List<String> nextIterationOperations = new ArrayList<>();

    /**
     * All Enter operations that feed into this loop
     */
    private List<String> enterOperations = new ArrayList<>();

    /**
     * All Merge operations within this loop
     */
    private List<String> mergeOperations = new ArrayList<>();

    /**
     * All other operations contained within this loop frame
     */
    private Set<String> loopOperations = new HashSet<>();

    // === LOOP VARIABLES ===

    /**
     * Variables that are modified within the loop (loop variables)
     */
    private List<String> loopVariables = new ArrayList<>();

    /**
     * Variables that are constants within the loop
     */
    private List<String> loopConstants = new ArrayList<>();

    /**
     * Variables that are passed into the loop from outside
     */
    private List<String> inputVariables = new ArrayList<>();

    /**
     * Variables that are produced by the loop
     */
    private List<String> outputVariables = new ArrayList<>();

    /**
     * Variables that serve as loop invariants
     */
    private List<String> invariantVariables = new ArrayList<>();

    // === EXECUTION STATE ===

    /**
     * Current iteration number
     */
    private int currentIteration = 0;

    /**
     * Maximum number of iterations observed during execution
     */
    private int maxIterationsObserved = 0;

    /**
     * Minimum number of iterations observed (for loops that reset)
     */
    private int minIterationsObserved = 0;

    /**
     * Total number of times this loop has been executed
     */
    private int executionCount = 0;

    /**
     * Time when loop execution started
     */
    private long startTime;

    /**
     * Time when loop execution ended
     */
    private long endTime;

    /**
     * Current status of the loop
     */
    private LoopTerminationStatus status = LoopTerminationStatus.ACTIVE;

    /**
     * Reason for loop termination
     */
    private String terminationReason;

    // === TERMINATION PREDICTIONS ===

    /**
     * Predictions about when this loop will terminate
     */
    private List<TerminationPrediction> terminationPredictions = new ArrayList<>();

    /**
     * Whether early termination has been detected
     */
    private boolean earlyTerminationDetected = false;

    /**
     * Expected number of iterations (if known)
     */
    private int expectedIterations = -1; // -1 means unknown

    /**
     * Confidence in the expected iteration count
     */
    private double expectedIterationsConfidence = 0.0;

    // === PERFORMANCE METRICS ===

    /**
     * Total execution time in milliseconds
     */
    private long totalExecutionTime = 0;

    /**
     * Average time per iteration in milliseconds
     */
    private double averageIterationTime = 0.0;

    /**
     * Peak memory usage during loop execution
     */
    private long peakMemoryUsage = 0;

    /**
     * Average memory usage during loop execution
     */
    private long averageMemoryUsage = 0;

    /**
     * Number of operations executed per iteration
     */
    private Map<Integer, Integer> operationsPerIteration = new HashMap<>();

    // === ANALYSIS DATA ===

    /**
     * General metadata about the loop
     */
    private Map<String, Object> metadata = new HashMap<>();

    /**
     * Statistical information about loop behavior
     */
    private Map<String, Double> statistics = new HashMap<>();

    /**
     * Flags indicating various loop characteristics
     */
    private Map<String, Boolean> flags = new HashMap<>();

    /**
     * Custom properties that can be set during analysis
     */
    private Map<String, Object> customProperties = new HashMap<>();

    // === NESTED ENUMS ===

    // === CONSTRUCTORS ===

    /**
     * Create a new LoopInfo with frame name
     */
    public LoopInfo(String frameName) {
        this.frameName = frameName;
        this.loopId = generateLoopId(frameName);
        this.startTime = System.currentTimeMillis();
        initializeDefaults();
    }

    /**
     * Create a new LoopInfo with frame name and parent
     */
    public LoopInfo(String frameName, String parentFrameName) {
        this.frameName = frameName;
        this.parentFrameName = parentFrameName;
        this.loopId = generateLoopId(frameName);
        this.startTime = System.currentTimeMillis();
        initializeDefaults();
    }

    /**
     * Create a new LoopInfo with full details
     */
    public LoopInfo(String frameName, String parentFrameName, int nestingDepth) {
        this.frameName = frameName;
        this.parentFrameName = parentFrameName;
        this.nestingDepth = nestingDepth;
        this.loopId = generateLoopId(frameName);
        this.startTime = System.currentTimeMillis();
        initializeDefaults();
    }

    // === INITIALIZATION ===

    /**
     * Initialize default values and flags
     */
    private void initializeDefaults() {
        // Initialize flags
        flags.put("hasCondition", false);
        flags.put("hasExit", false);
        flags.put("hasSwitches", false);
        flags.put("hasNextIteration", false);
        flags.put("isNested", parentFrameName != null);
        flags.put("isInfinite", false);
        flags.put("isConverging", false);
        flags.put("isOscillating", false);
        flags.put("hasNumericalIssues", false);

        // Initialize statistics
        statistics.put("iterationsPerSecond", 0.0);
        statistics.put("convergenceRate", 0.0);
        statistics.put("memoryGrowthRate", 0.0);
        statistics.put("operationEfficiency", 0.0);

        // Initialize metadata
        metadata.put("createdAt", System.currentTimeMillis());
        metadata.put("version", "1.0");
    }

    /**
     * Generate a unique loop ID
     */
    private String generateLoopId(String frameName) {
        return frameName + "_" + System.nanoTime();
    }

    // === OPERATION DISCOVERY ===

    /**
     * Discover and categorize all operations related to this loop
     */
    public void discoverLoopOperations(SameDiff sameDiff) {
        for (Map.Entry<String, SameDiffOp> entry : sameDiff.getOps().entrySet()) {
            String opName = entry.getKey();
            SameDiffOp op = entry.getValue();
            DifferentialFunction func = op.getOp();

            // Check if this operation is associated with this loop frame
            if (isOperationInLoop(opName, func)) {
                loopOperations.add(opName);

                // Categorize the operation
                if (func instanceof LoopCond) {
                    loopCondOperation = opName;
                    flags.put("hasCondition", true);
                } else if (func instanceof Exit) {
                    exitOperations.add(opName);
                    flags.put("hasExit", true);
                } else if (func instanceof Switch) {
                    switchOperations.add(opName);
                    flags.put("hasSwitches", true);
                } else if (func instanceof NextIteration) {
                    nextIterationOperations.add(opName);
                    flags.put("hasNextIteration", true);
                } else if (func instanceof Enter) {
                    enterOperations.add(opName);
                } else if (func instanceof Merge) {
                    mergeOperations.add(opName);
                }
            }
        }

        // Update operation counts
        metadata.put("totalOperations", loopOperations.size());
        metadata.put("controlFlowOperations",
                exitOperations.size() + switchOperations.size() +
                        nextIterationOperations.size() + enterOperations.size() +
                        mergeOperations.size() + (loopCondOperation != null ? 1 : 0));
    }

    /**
     * Check if an operation belongs to this loop (simplified)
     */
    private boolean isOperationInLoop(String opName, DifferentialFunction func) {
        // This is a simplified check - in practice, you'd need to analyze
        // the graph structure to determine frame associations
        return true;
    }

    // === VARIABLE DISCOVERY ===

    /**
     * Discover and categorize variables related to this loop
     */
    public void discoverLoopVariables(SameDiff sameDiff) {
        // Discover loop variables from NextIteration operations
        for (String nextIterOp : nextIterationOperations) {
            SameDiffOp op = sameDiff.getOps().get(nextIterOp);
            if (op != null) {
                List<String> inputs = op.getInputsToOp();
                if (inputs != null) {
                    for (String input : inputs) {
                        if (!loopVariables.contains(input)) {
                            loopVariables.add(input);
                        }
                    }
                }
            }
        }

        // Discover input variables from Enter operations
        for (String enterOp : enterOperations) {
            SameDiffOp op = sameDiff.getOps().get(enterOp);
            if (op != null) {
                List<String> inputs = op.getInputsToOp();
                if (inputs != null) {
                    for (String input : inputs) {
                        if (!inputVariables.contains(input)) {
                            inputVariables.add(input);
                        }
                    }
                }
            }
        }

        // Discover output variables from Exit operations
        for (String exitOp : exitOperations) {
            SameDiffOp op = sameDiff.getOps().get(exitOp);
            if (op != null) {
                List<String> outputs = op.getOutputsOfOp();
                if (outputs != null) {
                    for (String output : outputs) {
                        if (!outputVariables.contains(output)) {
                            outputVariables.add(output);
                        }
                    }
                }
            }
        }

        // Remove duplicates and update metadata
        loopVariables = loopVariables.stream().distinct().collect(Collectors.toList());
        inputVariables = inputVariables.stream().distinct().collect(Collectors.toList());
        outputVariables = outputVariables.stream().distinct().collect(Collectors.toList());

        metadata.put("loopVariableCount", loopVariables.size());
        metadata.put("inputVariableCount", inputVariables.size());
        metadata.put("outputVariableCount", outputVariables.size());
    }

    // === ITERATION TRACKING ===

    /**
     * Update iteration count and related metrics
     */
    public void updateIteration(int iteration) {
        this.currentIteration = iteration;
        this.maxIterationsObserved = Math.max(this.maxIterationsObserved, iteration);

        // Update timing statistics
        long currentTime = System.currentTimeMillis();
        this.totalExecutionTime = currentTime - startTime;

        if (iteration > 0) {
            this.averageIterationTime = (double) totalExecutionTime / iteration;
            statistics.put("iterationsPerSecond", 1000.0 / averageIterationTime);
        }

        // Update flags
        if (iteration > 1000) {
            flags.put("isLongRunning", true);
        }

        if (iteration > expectedIterations && expectedIterations > 0) {
            flags.put("exceededExpected", true);
        }
    }

    /**
     * Record operation count for an iteration
     */
    public void recordOperationCount(int iteration, int operationCount) {
        operationsPerIteration.put(iteration, operationCount);

        // Update efficiency metric
        if (!operationsPerIteration.isEmpty()) {
            double avgOpsPerIter = operationsPerIteration.values().stream()
                    .mapToInt(Integer::intValue)
                    .average()
                    .orElse(0.0);
            statistics.put("averageOperationsPerIteration", avgOpsPerIter);
        }
    }

    // === MEMORY TRACKING ===

    /**
     * Update memory usage statistics
     */
    public void updateMemoryUsage(long memoryUsage) {
        this.peakMemoryUsage = Math.max(this.peakMemoryUsage, memoryUsage);

        // Update average (simplified moving average)
        if (this.averageMemoryUsage == 0) {
            this.averageMemoryUsage = memoryUsage;
        } else {
            this.averageMemoryUsage = (long) ((this.averageMemoryUsage * 0.9) + (memoryUsage * 0.1));
        }

        statistics.put("memoryEfficiency", (double) averageMemoryUsage / Math.max(peakMemoryUsage, 1));

        // Check for memory growth
        if (memoryUsage > averageMemoryUsage * 1.5) {
            flags.put("highMemoryUsage", true);
        }
    }

    // === TERMINATION PREDICTION ===

    /**
     * Add a termination prediction
     */
    public void addTerminationPrediction(TerminationPrediction prediction) {
        terminationPredictions.add(prediction);

        // Update expected iterations based on highest confidence prediction
        TerminationPrediction bestPrediction = terminationPredictions.stream()
                .max(Comparator.comparingDouble(TerminationPrediction::getConfidence))
                .orElse(null);

        if (bestPrediction != null && bestPrediction.getConfidence() > expectedIterationsConfidence) {
            expectedIterations = bestPrediction.getPredictedTerminationIteration();
            expectedIterationsConfidence = bestPrediction.getConfidence();
        }
    }

    /**
     * Get the most confident termination prediction
     */
    public TerminationPrediction getBestTerminationPrediction() {
        return terminationPredictions.stream()
                .max(Comparator.comparingDouble(TerminationPrediction::getConfidence))
                .orElse(null);
    }

    // === TERMINATION HANDLING ===

    /**
     * Mark the loop as terminated
     */
    public void markTerminated(LoopTerminationStatus status, String reason) {
        this.status = status;
        this.terminationReason = reason;
        this.endTime = System.currentTimeMillis();

        // Update final statistics
        updateFinalStatistics();

        // Set termination flags
        flags.put("isTerminated", true);
        flags.put("terminatedNormally", status == LoopTerminationStatus.TERMINATED_NORMAL);
        flags.put("terminatedEarly", status == LoopTerminationStatus.TERMINATED_EARLY);
        flags.put("terminatedWithError", status == LoopTerminationStatus.TERMINATED_ERROR);
    }

    /**
     * Update final statistics when loop terminates
     */
    private void updateFinalStatistics() {
        if (endTime > 0) {
            totalExecutionTime = endTime - startTime;
            if (maxIterationsObserved > 0) {
                averageIterationTime = (double) totalExecutionTime / maxIterationsObserved;
                statistics.put("finalIterationsPerSecond", 1000.0 / averageIterationTime);
            }
        }

        // Calculate prediction accuracy
        if (expectedIterations > 0) {
            double accuracy = 1.0 - (double) Math.abs(maxIterationsObserved - expectedIterations) / expectedIterations;
            statistics.put("predictionAccuracy", Math.max(0.0, accuracy));
        }
    }

    // === ANALYSIS METHODS ===

    /**
     * Check if the loop appears to be converging
     */
    public boolean isConverging() {
        return flags.getOrDefault("isConverging", false);
    }

    /**
     * Check if the loop appears to be oscillating
     */
    public boolean isOscillating() {
        return flags.getOrDefault("isOscillating", false);
    }

    /**
     * Check if the loop has numerical issues
     */
    public boolean hasNumericalIssues() {
        return flags.getOrDefault("hasNumericalIssues", false);
    }

    /**
     * Check if the loop is running longer than expected
     */
    public boolean isRunningLongerThanExpected() {
        return expectedIterations > 0 && currentIteration > expectedIterations * 1.2;
    }

    /**
     * Get the loop efficiency (operations per second)
     */
    public double getLoopEfficiency() {
        return statistics.getOrDefault("iterationsPerSecond", 0.0);
    }

    /**
     * Get the prediction accuracy
     */
    public double getPredictionAccuracy() {
        return statistics.getOrDefault("predictionAccuracy", 0.0);
    }

    // === UTILITY METHODS ===

    /**
     * Get a summary of the loop information
     */
    public String getSummary() {
        StringBuilder summary = new StringBuilder();
        summary.append("Loop '").append(frameName).append("'");
        summary.append(" (").append(status).append(")");
        summary.append(" - Iterations: ").append(maxIterationsObserved);

        if (totalExecutionTime > 0) {
            summary.append(", Time: ").append(totalExecutionTime).append("ms");
        }

        if (expectedIterations > 0) {
            summary.append(", Expected: ").append(expectedIterations);
        }

        if (terminationReason != null) {
            summary.append(", Reason: ").append(terminationReason);
        }

        return summary.toString();
    }

    /**
     * Get detailed information about the loop
     */
    public String getDetailedInfo() {
        StringBuilder info = new StringBuilder();
        info.append("=== Loop Information ===\n");
        info.append("Frame: ").append(frameName).append("\n");
        info.append("Status: ").append(status).append("\n");
        info.append("Iterations: ").append(maxIterationsObserved).append("\n");
        info.append("Execution Time: ").append(totalExecutionTime).append("ms\n");

        if (parentFrameName != null) {
            info.append("Parent Frame: ").append(parentFrameName).append("\n");
            info.append("Nesting Depth: ").append(nestingDepth).append("\n");
        }

        info.append("\nOperations:\n");
        info.append("  Condition: ").append(loopCondOperation).append("\n");
        info.append("  Exit: ").append(exitOperations).append("\n");
        info.append("  Switch: ").append(switchOperations).append("\n");
        info.append("  NextIteration: ").append(nextIterationOperations).append("\n");
        info.append("  Total: ").append(loopOperations.size()).append("\n");

        info.append("\nVariables:\n");
        info.append("  Loop Variables: ").append(loopVariables.size()).append("\n");
        info.append("  Input Variables: ").append(inputVariables.size()).append("\n");
        info.append("  Output Variables: ").append(outputVariables.size()).append("\n");

        if (!terminationPredictions.isEmpty()) {
            info.append("\nPredictions: ").append(terminationPredictions.size()).append("\n");
            TerminationPrediction best = getBestTerminationPrediction();
            if (best != null) {
                info.append("  Best: ").append(best.getPredictedTerminationIteration());
                info.append(" (").append(String.format("%.2f", best.getConfidence())).append(")\n");
            }
        }

        return info.toString();
    }

    /**
     * Check if the loop has specific characteristics
     */
    public boolean hasCharacteristic(String characteristic) {
        return flags.getOrDefault(characteristic, false);
    }

    /**
     * Set a characteristic flag
     */
    public void setCharacteristic(String characteristic, boolean value) {
        flags.put(characteristic, value);
    }

    /**
     * Get a statistic value
     */
    public double getStatistic(String name) {
        return statistics.getOrDefault(name, 0.0);
    }

    /**
     * Set a statistic value
     */
    public void setStatistic(String name, double value) {
        statistics.put(name, value);
    }

    /**
     * Get metadata value
     */
    public Object getMetadataValue(String key) {
        return metadata.get(key);
    }

    /**
     * Set metadata value
     */
    public void setMetadataValue(String key, Object value) {
        metadata.put(key, value);
    }

    @Override
    public String toString() {
        return getSummary();
    }
}