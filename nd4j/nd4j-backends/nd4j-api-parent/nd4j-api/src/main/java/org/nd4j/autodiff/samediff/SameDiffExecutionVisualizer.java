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

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.stream.Collectors;

import lombok.Builder;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.internal.ExecType;
import org.nd4j.autodiff.samediff.internal.FrameIter;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.internal.VarId;
import org.nd4j.linalg.api.ops.impl.controlflow.compat.*;

/**
 * Terminal-based visualizer for SameDiff execution flow.
 * Tracks execution steps and provides formatted output showing the order of operations.
 * Includes integrated loop termination analysis capabilities.
 */
public class SameDiffExecutionVisualizer {

    private final List<ExecutionStep> executionSteps = new ArrayList<>();
    private final Map<String, Set<String>> frameDependencies = new HashMap<>();
    private final AtomicInteger stepCounter = new AtomicInteger(0);
    private final DateTimeFormatter timeFormatter = DateTimeFormatter.ofPattern("HH:mm:ss.SSS");

    // Integrated loop termination analyzer
    final LoopTerminationAnalyzer loopAnalyzer;
    private LoopTerminationErrorReporter errorReporter;

    private boolean enableTimestamps = true;
    private boolean enableFrameTracking = true;
    private boolean enableDependencyTracking = true;
    private boolean enableLoopAnalysis = true;

    public SameDiffExecutionVisualizer() {
        this.loopAnalyzer = null; // Will be set when SameDiff instance is available
    }

    @Builder
    public SameDiffExecutionVisualizer(SameDiff sameDiff, Map<VarId, org.nd4j.autodiff.samediff.config.SDValue> nodeValueOutputs) {
        this.loopAnalyzer = new LoopTerminationAnalyzer(sameDiff, nodeValueOutputs);
        this.errorReporter = new LoopTerminationErrorReporter(sameDiff, nodeValueOutputs, loopAnalyzer);
        configureErrorReporter();
    }

    private void configureErrorReporter() {
        if (errorReporter != null) {
            errorReporter.setIncludeVariableValues(true);
            errorReporter.setIncludeVariableShapes(true);
            errorReporter.setIncludeOperationHistory(true);
            errorReporter.setIncludeFrameContext(true);
            errorReporter.setIncludeMemoryMetrics(true);
            errorReporter.setIncludeVariableEvolution(true);
            errorReporter.setGenerateVisualizations(true);
            errorReporter.setMaxVariableValueDisplay(20);
            errorReporter.setMaxHistoryDepth(50);
        }
    }

    /**
     * MAIN ENTRY POINT: Print comprehensive execution and loop analysis report
     * This is the primary method that should be called to get all analysis information.
     */
    public void printCompleteAnalysisReport() {
        printReportHeader();
        printExecutionOverview();
        printExecutionTrace();
        printLoopAnalysisReport();
        printErrorAnalysisReport();
        printStatisticsReport();
        printReportFooter();
    }

    /**
     * Record an execution step with ACTUAL VALUES extracted and ENHANCED LOOP DETECTION
     */
    public void recordStep(ExecType type, String name, FrameIter frameIter,
                           List<String> inputs, List<String> outputs, String status) {
        String timestamp = enableTimestamps ? LocalDateTime.now().format(timeFormatter) : "";
        String frame = frameIter != null ? frameIter.getFrame() : "OUTER_FRAME";
        int iteration = frameIter != null ? frameIter.getIteration() : 0;
        FrameIter parentFrame = frameIter != null ? frameIter.getParentFrame() : null;

        // Store original names for logic, enhanced names for display
        List<String> originalInputs = inputs != null ? new ArrayList<>(inputs) : new ArrayList<>();
        List<String> originalOutputs = outputs != null ? new ArrayList<>(outputs) : new ArrayList<>();

        // EXTRACT ACTUAL VALUES for inputs and outputs (with safety)
        List<String> inputsWithValues = enhanceVariableListWithValuesSafe(originalInputs, frameIter);
        List<String> outputsWithValues = enhanceVariableListWithValuesSafe(originalOutputs, frameIter);

        ExecutionStep step = new ExecutionStep(
                stepCounter.incrementAndGet(),
                timestamp,
                type,
                name,
                frame,
                iteration,
                parentFrame,
                inputsWithValues,  // Enhanced names for display
                outputsWithValues, // Enhanced names for display
                status
        );

        executionSteps.add(step);

        // Track frame dependencies if enabled (use original names)
        if (enableFrameTracking) {
            trackFrameDependencies(frame, originalInputs);
        }

        // ENHANCED LOOP ANALYSIS - Proactive Detection and Tracking
        if (enableLoopAnalysis && loopAnalyzer != null) {
            try {
                // 1. PROACTIVE LOOP DETECTION - Track recent operations per frame
                if (!frame.equals("OUTER_FRAME") && !frame.equals("main")) {
                    trackFrameOperationHistory(frame, name, type);

                    // Auto-detect loop patterns if not already tracking this frame
                    if (!loopAnalyzer.getActiveLoops().containsKey(frame)) {
                        attemptLoopDetection(frame, name, frameIter, iteration);
                    }
                }

                // 2. ENHANCED INTEGRATION - Better pattern recognition
                if (!frame.equals("OUTER_FRAME")) {
                    integrateLoopAnalysisEnhanced(type, name, frameIter, originalInputs, originalOutputs, status, iteration);
                }

            } catch (Exception loopAnalysisError) {
                // Don't let loop analysis errors crash main execution
                System.err.println("VISUALIZER_WARNING: Loop analysis error at step " + stepCounter.get() +
                        ": " + loopAnalysisError.getMessage());
            }
        }
    }

    // ADD: Frame operation history tracking
    private final Map<String, List<String>> frameOperationHistory = new HashMap<>();
    private final Map<String, List<ExecType>> frameTypeHistory = new HashMap<>();

    /**
     * Track recent operations per frame for pattern detection
     */
    private void trackFrameOperationHistory(String frame, String operationName, ExecType type) {
        // Keep last 15 operations per frame for pattern analysis
        List<String> recentOps = frameOperationHistory.computeIfAbsent(frame, k -> new ArrayList<>());
        List<ExecType> recentTypes = frameTypeHistory.computeIfAbsent(frame, k -> new ArrayList<>());

        recentOps.add(operationName);
        recentTypes.add(type);

        // Maintain sliding window
        if (recentOps.size() > 15) {
            recentOps.remove(0);
            recentTypes.remove(0);
        }
    }

    /**
     * Attempt to auto-detect loop patterns in execution
     */
    private void attemptLoopDetection(String frame, String currentOp, FrameIter frameIter, int iteration) {
        List<String> recentOps = frameOperationHistory.get(frame);
        if (recentOps == null || recentOps.size() < 3) return;

        // Look for loop control flow patterns
        boolean hasLoopPattern = detectLoopControlPattern(recentOps, currentOp);

        if (hasLoopPattern) {
            System.out.println("LOOP_AUTO_DETECTION: Detected loop pattern in frame '" + frame + "' at operation: " + currentOp);

            try {
                // Create synthetic FrameIter for the detected loop
                org.nd4j.autodiff.samediff.internal.FrameIter parentFrameInternal = null;
                if (frameIter.getParentFrame() != null) {
                    parentFrameInternal = new org.nd4j.autodiff.samediff.internal.FrameIter(
                            frameIter.getParentFrame().getFrame(),
                            frameIter.getParentFrame().getIteration(),
                            null);
                }

                org.nd4j.autodiff.samediff.internal.FrameIter loopFrameIter =
                        new org.nd4j.autodiff.samediff.internal.FrameIter(frame, iteration, parentFrameInternal);

                loopAnalyzer.onLoopFrameEnter(frame, "auto_detected_from_pattern", loopFrameIter);

            } catch (Exception e) {
                System.err.println("LOOP_DETECTION_ERROR: Failed to register auto-detected loop: " + e.getMessage());
            }
        }
    }

    /**
     * Detect if recent operations form a loop control pattern
     */
    private boolean detectLoopControlPattern(List<String> recentOps, String currentOp) {
        // Check for specific loop patterns
        boolean hasMergePattern = recentOps.stream().anyMatch(op -> op.contains("merge"));
        boolean hasConditionPattern = recentOps.stream().anyMatch(op ->
                op.contains("cond/") || op.contains("LoopCond") ||
                        op.contains("less") || op.contains("equal") || op.contains("greater"));
        boolean hasSwitchPattern = recentOps.stream().anyMatch(op -> op.contains("switch")) ||
                currentOp.contains("switch");

        // Pattern 1: merge -> condition -> switch sequence
        if (hasMergePattern && hasConditionPattern && hasSwitchPattern) {
            return true;
        }

        // Pattern 2: "outputs" frame with control flow operations
        if (recentOps.get(0).startsWith("outputs/") &&
                (hasMergePattern || hasConditionPattern || hasSwitchPattern)) {
            return true;
        }

        // Pattern 3: Multiple operations in same non-main frame with control flow indicators
        boolean allSameFrame = recentOps.stream().allMatch(op ->
                op.contains("/") && op.split("/")[0].equals(recentOps.get(0).split("/")[0]));

        if (allSameFrame && recentOps.size() >= 5 &&
                (hasMergePattern || hasConditionPattern || hasSwitchPattern)) {
            return true;
        }

        return false;
    }

    /**
     * Enhanced loop analysis integration with better pattern recognition
     */
    private void integrateLoopAnalysisEnhanced(ExecType type, String name, FrameIter frameIter,
                                               List<String> inputs, List<String> outputs, String status, int iteration) {
        String frameName = frameIter.getFrame();

        // Extract ACTUAL variable values from nodeValueOutputs (with safety)
        Map<String, Object> actualVariableValues = extractActualValuesSafe(outputs, frameIter);
        Map<String, Object> actualInputValues = extractActualValuesSafe(inputs, frameIter);

        switch (type) {
            case OP:
                // ENHANCED: Broader condition detection patterns
                if (isConditionOperation(name)) {
                    Object conditionValue = extractConditionValue(name, outputs, frameIter);
                    loopAnalyzer.onLoopConditionEvaluation(frameName, name, conditionValue, actualInputValues, iteration);

                    // Check if this condition indicates termination
                    boolean terminationTriggered = isTerminationCondition(conditionValue);
                    if (terminationTriggered) {
                        System.out.println("LOOP_TERMINATION_DETECTED: Condition '" + name + "' triggered termination in frame '" + frameName + "'");
                    }

                } else if (isSwitchOperation(name)) {
                    Object predicateValue = extractSwitchPredicateValue(inputs, frameIter);
                    String branchTaken = determineSwitchBranch(outputs, frameIter);
                    loopAnalyzer.onSwitchOperation(frameName, name, predicateValue, branchTaken, iteration);

                    System.out.println("SWITCH_OPERATION: '" + name + "' took " + branchTaken + " branch in frame '" + frameName + "'");

                } else if (isMergeOperation(name)) {
                    // Track merge operations as loop iteration indicators
                    loopAnalyzer.onLoopIteration(frameName, iteration, actualVariableValues);

                } else if (isExitOperation(name)) {
                    Object exitValue = extractExitValue(outputs, frameIter);
                    loopAnalyzer.onExitOperation(frameName, name, exitValue, iteration);

                } else if (isEnterOperation(name)) {
                    // Handle explicit Enter operations
                    try {
                        org.nd4j.autodiff.samediff.internal.FrameIter parentFrameInternal = null;
                        if (frameIter.getParentFrame() != null) {
                            parentFrameInternal = new org.nd4j.autodiff.samediff.internal.FrameIter(
                                    frameIter.getParentFrame().getFrame(),
                                    frameIter.getParentFrame().getIteration(),
                                    null);
                        }

                        org.nd4j.autodiff.samediff.internal.FrameIter loopFrameIter =
                                new org.nd4j.autodiff.samediff.internal.FrameIter(frameName, iteration, parentFrameInternal);

                        loopAnalyzer.onLoopFrameEnter(frameName, name, loopFrameIter);
                    } catch (Exception e) {
                        System.err.println("ENTER_OPERATION_ERROR: " + e.getMessage());
                    }

                } else if (isNextIterationOperation(name)) {
                    loopAnalyzer.onLoopIteration(frameName, iteration + 1, actualVariableValues);

                } else {
                    // Regular operation in potential loop - track iteration
                    loopAnalyzer.onLoopIteration(frameName, iteration, actualVariableValues);
                }

                // Check for error status
                if (status != null && status.toLowerCase().contains("error")) {
                    Exception error = new RuntimeException("Execution error: " + status);
                    Map<String, Object> errorContext = new HashMap<>();
                    errorContext.put("actualInputs", actualInputValues);
                    errorContext.put("actualOutputs", actualVariableValues);
                    errorContext.put("executionStep", stepCounter.get());
                    loopAnalyzer.onLoopError(frameName, iteration, name, error, errorContext);
                }
                break;

            default:
                // For other types, just track iteration if in a potential loop
                if (iteration >= 0) {
                    loopAnalyzer.onLoopIteration(frameName, iteration, actualVariableValues);
                }
                break;
        }
    }

    /**
     * Enhanced operation type detection methods
     */
    private boolean isConditionOperation(String name) {
        return name.contains("LoopCond") ||
                name.contains("cond/") ||
                (name.contains("less") && name.contains("cond")) ||
                (name.contains("equal") && name.contains("cond")) ||
                (name.contains("greater") && name.contains("cond")) ||
                (name.contains("bitwise_and") && name.contains("cond"));
    }

    private boolean isSwitchOperation(String name) {
        return name.contains("Switch") || name.contains("switch");
    }

    private boolean isMergeOperation(String name) {
        return name.contains("Merge") || name.contains("merge");
    }

    private boolean isExitOperation(String name) {
        return name.contains("Exit") || name.contains("exit");
    }

    private boolean isEnterOperation(String name) {
        return name.contains("Enter") || name.contains("enter");
    }

    private boolean isNextIterationOperation(String name) {
        return name.contains("NextIteration") || name.contains("next_iteration");
    }

    /**
     * Check if a condition value indicates loop termination
     */
    private boolean isTerminationCondition(Object conditionValue) {
        if (conditionValue instanceof Boolean) {
            return !(Boolean) conditionValue;
        } else if (conditionValue instanceof org.nd4j.linalg.api.ndarray.INDArray) {
            org.nd4j.linalg.api.ndarray.INDArray arr = (org.nd4j.linalg.api.ndarray.INDArray) conditionValue;
            if (arr.isScalar()) {
                return arr.getDouble(0) == 0.0;
            }
        } else if (conditionValue instanceof Number) {
            return ((Number) conditionValue).doubleValue() == 0.0;
        }
        return false;
    }

    /**
     * Safe version of value enhancement that doesn't crash execution
     */
    private List<String> enhanceVariableListWithValuesSafe(List<String> variableNames, FrameIter frameIter) {
        if (variableNames == null || variableNames.isEmpty()) {
            return new ArrayList<>();
        }

        try {
            Map<VarId, org.nd4j.autodiff.samediff.config.SDValue> nodeValueOutputs = getNodeValueOutputsFromAnalyzer();

            if (nodeValueOutputs == null) {
                // Fallback: return original names if we can't access values
                return new ArrayList<>(variableNames);
            }

            List<String> enhancedList = new ArrayList<>();
            for (String varName : variableNames) {
                try {
                    Object actualValue = findActualValue(varName, frameIter, nodeValueOutputs);
                    String valueDisplay = formatValueForDisplaySafe(actualValue, varName);
                    enhancedList.add(varName + "=" + valueDisplay);
                } catch (Exception e) {
                    // If individual value enhancement fails, just use the variable name
                    enhancedList.add(varName + "=<format_error>");
                }
            }
            return enhancedList;

        } catch (Exception e) {
            // If entire enhancement fails, return original names
            System.err.println("VISUALIZER_WARNING: Value enhancement failed: " + e.getMessage());
            return new ArrayList<>(variableNames);
        }
    }

    /**
     * Safe version of value extraction that handles errors gracefully
     */
    private Map<String, Object> extractActualValuesSafe(List<String> variableNames, FrameIter frameIter) {
        Map<String, Object> actualValues = new HashMap<>();

        if (variableNames == null || loopAnalyzer == null) {
            return actualValues;
        }

        try {
            Map<VarId, org.nd4j.autodiff.samediff.config.SDValue> nodeValueOutputs = getNodeValueOutputsFromAnalyzer();
            if (nodeValueOutputs == null) {
                return actualValues;
            }

            for (String varName : variableNames) {
                try {
                    Object actualValue = findActualValue(varName, frameIter, nodeValueOutputs);
                    if (actualValue != null) {
                        actualValues.put(varName, actualValue);
                    } else {
                        actualValues.put(varName, "<value_not_found_in_frame:" + frameIter.getFrame() + ">");
                    }
                } catch (Exception e) {
                    actualValues.put(varName, "<extraction_error:" + e.getMessage() + ">");
                }
            }
        } catch (Exception e) {
            System.err.println("VISUALIZER_WARNING: Value extraction failed: " + e.getMessage());
        }

        return actualValues;
    }

    /**
     * Safe value formatting that doesn't crash on invalid arrays
     */
    private String formatValueForDisplaySafe(Object value, String varName) {
        try {
            return formatValueForDisplay(value, varName);
        } catch (Exception e) {
            if (value == null) {
                return "<null>";
            } else {
                return "<format_error:" + value.getClass().getSimpleName() + ">";
            }
        }
    }


    private EntityType classifyEntity(String entityName, SameDiff sameDiff) {
        if (sameDiff.getOps().containsKey(entityName)) {
            return EntityType.OPERATION;
        } else if (sameDiff.getVariables().containsKey(entityName)) {
            return EntityType.VARIABLE;
        } else if (sameDiff.getConstantArrays().hasArray(entityName)) {
            return EntityType.CONSTANT;
        } else if (sameDiff.isPlaceHolder((entityName))) {
            return EntityType.PLACEHOLDER;
        } else {
            return EntityType.UNKNOWN;
        }
    }

    /**
     * Analyze a missing variable entity
     */
    private void analyzeVariableEntity(String variableName, SameDiff sameDiff,
                                       Map<VarId, org.nd4j.autodiff.samediff.config.SDValue> nodeValueOutputs,
                                       Set<String> allExecuted) {

        String producerOp = findProducerForVariable(variableName, sameDiff);

        if (producerOp != null) {
            boolean producerExecuted = allExecuted.contains(producerOp);
            SameDiffOp op = sameDiff.getOps().get(producerOp);

            System.out.printf("  📋 PRODUCER OPERATION:%n");
            System.out.printf("    Operation: %s%n", producerOp);
            System.out.printf("    Type: %s%n", op != null ? op.getOp().getClass().getSimpleName() : "Unknown");
            System.out.printf("    Executed: %s%n", producerExecuted ? "✅ YES" : "❌ NO");

            if (op != null) {
                List<String> outputs = op.getOutputsOfOp();
                System.out.printf("    All Outputs: %s%n", outputs);
            }

            if (producerExecuted) {
                System.out.printf("  💡 DIAGNOSIS: Producer executed but variable missing from nodeValueOutputs%n");
                analyzeVariableAvailabilityDetailed(variableName, nodeValueOutputs, "current context");
            } else {
                System.out.printf("  💡 DIAGNOSIS: Variable missing because producer operation not executed%n");
            }
        } else {
            System.out.printf("  ❌ ERROR: No producer operation found for variable%n");
            System.out.printf("  💡 DIAGNOSIS: Graph construction issue - variable has no producer%n");
        }

        analyzeVariableAvailabilityDetailed(variableName, nodeValueOutputs, "any frame");
    }

    /**
     * Analyze a missing operation entity
     */
    private void analyzeOperationEntity(String operationName, SameDiff sameDiff,
                                        Map<VarId, org.nd4j.autodiff.samediff.config.SDValue> nodeValueOutputs,
                                        Set<String> allExecuted) {

        SameDiffOp op = sameDiff.getOps().get(operationName);
        if (op == null) {
            System.out.printf("  ❌ ERROR: Operation not found in SameDiff graph%n");
            return;
        }

        System.out.printf("  Operation Type: %s%n", op.getOp().getClass().getSimpleName());

        List<String> inputs = op.getInputsToOp();
        if (inputs != null && !inputs.isEmpty()) {
            System.out.printf("  📥 INPUTS (%d):%n", inputs.size());
            for (String input : inputs) {
                analyzeInputAvailability(input, nodeValueOutputs, allExecuted);
            }
        } else {
            System.out.printf("  📥 INPUTS: None%n");
        }

        List<String> outputs = op.getOutputsOfOp();
        if (outputs != null && !outputs.isEmpty()) {
            System.out.printf("  📤 EXPECTED OUTPUTS (%d): %s%n", outputs.size(), outputs);
        }

        String inferredFrame = inferOperationFrame(operationName);
        System.out.printf("  🏗️  INFERRED FRAME: %s%n", inferredFrame);
    }

    /**
     * Analyze a missing constant entity
     */
    private void analyzeConstantEntity(String constantName, SameDiff sameDiff) {
        System.out.printf("  ⚠️  WARNING: Constant missing - this should not happen%n");
        System.out.printf("  💡 DIAGNOSIS: Constants should always be available%n");
    }

    /**
     * Analyze a missing placeholder entity
     */
    private void analyzePlaceholderEntity(String placeholderName, SameDiff sameDiff) {
        System.out.printf("  ℹ️  INFO: Placeholder may need to be fed at runtime%n");
        System.out.printf("  💡 DIAGNOSIS: Check if placeholder value was provided%n");
    }

    /**
     * Analyze an unknown entity
     */
    private void analyzeUnknownEntity(String entityName, SameDiff sameDiff) {
        System.out.printf("  ❌ ERROR: Entity not found in operations, variables, constants, or placeholders%n");
        System.out.printf("  💡 DIAGNOSIS: Possible graph construction or naming issue%n");
    }

    /**
     * Variable availability analysis with detailed context
     */
    private void analyzeVariableAvailabilityDetailed(String variableName,
                                                     Map<VarId, org.nd4j.autodiff.samediff.config.SDValue> nodeValueOutputs,
                                                     String context) {

        List<VarId> foundInFrames = new ArrayList<>();
        for (VarId varId : nodeValueOutputs.keySet()) {
            if (varId.getVariable().equals(variableName)) {
                foundInFrames.add(varId);
            }
        }

        if (!foundInFrames.isEmpty()) {
            System.out.printf("  📍 VARIABLE AVAILABILITY (%s):%n", context);
            System.out.printf("    Found in %d frame locations:%n", foundInFrames.size());
            for (VarId varId : foundInFrames) {
                System.out.printf("      Frame: %s, Iteration: %d%n", varId.getFrame(), varId.getIteration());
            }
        } else {
            System.out.printf("  📍 VARIABLE AVAILABILITY (%s): ❌ Not found%n", context);
        }
    }

    /**
     * Classify and summarize all missing entities
     */
    private void classifyAndSummarizeMissingEntities(Set<String> missing, SameDiff sameDiff) {
        System.out.println("\n📊 ENTITY CLASSIFICATION SUMMARY:");
        System.out.println("─".repeat(50));

        int operations = 0, variables = 0, constants = 0, placeholders = 0, unknown = 0;

        List<String> variablesList = new ArrayList<>();
        List<String> operationsList = new ArrayList<>();

        for (String entity : missing) {
            EntityType type = classifyEntity(entity, sameDiff);
            switch (type) {
                case OPERATION:
                    operations++;
                    operationsList.add(entity);
                    break;
                case VARIABLE:
                    variables++;
                    variablesList.add(entity);
                    break;
                case CONSTANT:
                    constants++;
                    break;
                case PLACEHOLDER:
                    placeholders++;
                    break;
                case UNKNOWN:
                    unknown++;
                    break;
            }
        }

        System.out.printf("📈 BREAKDOWN:%n");
        System.out.printf("  Operations: %d%n", operations);
        System.out.printf("  Variables: %d%n", variables);
        System.out.printf("  Constants: %d%n", constants);
        System.out.printf("  Placeholders: %d%n", placeholders);
        System.out.printf("  Unknown: %d%n", unknown);

        if (variables > 0) {
            System.out.printf("\n❌ CRITICAL FINDING:%n");
            System.out.printf("  %d variables are missing (should be accessed via VarId, not executed):%n", variables);
            for (String var : variablesList) {
                System.out.printf("    • %s%n", var);
            }
            System.out.printf("  This indicates a potential execution framework issue.%n");
        }

        if (operations > 0) {
            System.out.printf("\n⚙️ LEGITIMATE MISSING OPERATIONS:%n");
            for (String op : operationsList) {
                System.out.printf("    • %s%n", op);
            }
        }
    }

    /**
     * Get display icon for entity type
     */
    private String getEntityTypeIcon(EntityType type) {
        switch (type) {
            case OPERATION: return "⚙️";
            case VARIABLE: return "📊";
            case CONSTANT: return "📋";
            case PLACEHOLDER: return "🔲";
            case UNKNOWN: return "❓";
            default: return "❓";
        }
    }

    /**
     * Get display name for entity type
     */
    private String getEntityTypeName(EntityType type) {
        switch (type) {
            case OPERATION: return "OPERATION";
            case VARIABLE: return "VARIABLE";
            case CONSTANT: return "CONSTANT";
            case PLACEHOLDER: return "PLACEHOLDER";
            case UNKNOWN: return "UNKNOWN ENTITY";
            default: return "UNKNOWN";
        }
    }
    /**
     * Analyze execution failure with proper entity classification
     */
    public void analyzeExecutionFailure(Set<String> allRequired, Set<String> allExecuted,
                                        int step, String currentFrame, int currentIteration,
                                        Map<VarId, org.nd4j.autodiff.samediff.config.SDValue> nodeValueOutputs,
                                        SameDiff sameDiff) {

        System.out.println("\n" + "═".repeat(100));
        System.out.println("🔍 DETAILED EXECUTION FAILURE ANALYSIS");
        System.out.println("═".repeat(100));

        Set<String> missing = new HashSet<>(allRequired);
        missing.removeAll(allExecuted);

        System.out.printf("📊 FAILURE SUMMARY:%n");
        System.out.printf("  Failed at step: %d%n", step);
        System.out.printf("  Current frame: %s (iteration: %d)%n", currentFrame, currentIteration);
        System.out.printf("  Required entities: %d%n", allRequired.size());
        System.out.printf("  Completed entities: %d%n", allExecuted.size());
        System.out.printf("  Missing entities: %d%n", missing.size());

        System.out.println("\n🔍 MISSING ENTITIES DETAILED ANALYSIS:");
        for (String missingEntity : missing) {
            analyzeMissingOperation(missingEntity, sameDiff, nodeValueOutputs, allExecuted);
        }

        classifyAndSummarizeMissingEntities(missing, sameDiff);

        analyzeFrameContext(nodeValueOutputs, currentFrame);

        System.out.println("\n🔗 DEPENDENCY CHAIN ANALYSIS:");
        for (String missingEntity : missing) {
            EntityType type = classifyEntity(missingEntity, sameDiff);
            if (type == EntityType.OPERATION) {
                traceDependencyChain(missingEntity, sameDiff, allExecuted, 0);
            }
        }

        if (loopAnalyzer != null && !loopAnalyzer.getActiveLoops().isEmpty()) {
            analyzePostLoopContext(missing, allExecuted);
        }

        System.out.println("═".repeat(100));
    }

        /**
         * Analyze a specific missing entity in detail
         */
    private void analyzeMissingOperation(String entityName, SameDiff sameDiff,
                                         Map<VarId, org.nd4j.autodiff.samediff.config.SDValue> nodeValueOutputs,
                                         Set<String> allExecuted) {

        EntityType entityType = classifyEntity(entityName, sameDiff);
        String icon = getEntityTypeIcon(entityType);
        String typeName = getEntityTypeName(entityType);

        System.out.printf("\n%s %s: %s%n", icon, typeName, entityName);

        boolean isOperation = sameDiff.getOps().containsKey(entityName);
        boolean isVariable = sameDiff.getVariables().containsKey(entityName);
        boolean isConstant = sameDiff.getConstantArrays().hasArray(entityName);
        boolean isPlaceholder = sameDiff.isPlaceHolder(entityName);

        System.out.printf("  🔍 CLASSIFICATION:%n");
        System.out.printf("    Operation: %s%n", isOperation ? "✅ YES" : "❌ NO");
        System.out.printf("    Variable: %s%n", isVariable ? "✅ YES" : "❌ NO");
        System.out.printf("    Constant: %s%n", isConstant ? "✅ YES" : "❌ NO");
        System.out.printf("    Placeholder: %s%n", isPlaceholder ? "✅ YES" : "❌ NO");

        switch (entityType) {
            case VARIABLE:
                analyzeVariableEntity(entityName, sameDiff, nodeValueOutputs, allExecuted);
                break;
            case OPERATION:
                analyzeOperationEntity(entityName, sameDiff, nodeValueOutputs, allExecuted);
                break;
            case CONSTANT:
                analyzeConstantEntity(entityName, sameDiff);
                break;
            case PLACEHOLDER:
                analyzePlaceholderEntity(entityName, sameDiff);
                break;
            case UNKNOWN:
                analyzeUnknownEntity(entityName, sameDiff);
                break;
        }
    }
    /**
     * Analyze input availability across all frames
     */
    private void analyzeInputAvailability(String inputVar,
                                          Map<VarId, org.nd4j.autodiff.samediff.config.SDValue> nodeValueOutputs,
                                          Set<String> allExecuted) {

        System.out.printf("    • %s: ", inputVar);

        // Check if a producer operation was executed
        boolean producerExecuted = false;
        String producerOp = findProducerOperation(inputVar, allExecuted);
        if (producerOp != null) {
            producerExecuted = true;
            System.out.printf("✅ Producer '%s' executed", producerOp);
        } else {
            System.out.printf("❌ No producer executed");
        }

        // Check availability across frames
        List<VarId> foundInFrames = new ArrayList<>();
        for (VarId varId : nodeValueOutputs.keySet()) {
            if (varId.getVariable().equals(inputVar)) {
                foundInFrames.add(varId);
            }
        }

        if (!foundInFrames.isEmpty()) {
            System.out.printf(" | 📍 Available in frames: ");
            for (int i = 0; i < foundInFrames.size(); i++) {
                VarId varId = foundInFrames.get(i);
                System.out.printf("%s:%d", varId.getFrame(), varId.getIteration());
                if (i < foundInFrames.size() - 1) System.out.printf(", ");
            }
        } else {
            System.out.printf(" | ❌ Not available in any frame");
        }

        System.out.println();
    }

    /**
     * Find which operation should produce a given variable
     */
    private String findProducerOperation(String variable, Set<String> allExecuted) {
        for (ExecutionStep step : executionSteps) {
            if (step.getOutputs().stream().anyMatch(output -> output.contains(variable))) {
                return step.getName();
            }
        }
        return null;
    }

    /**
     * Analyze current frame context and variable availability
     */
    private void analyzeFrameContext(Map<VarId, org.nd4j.autodiff.samediff.config.SDValue> nodeValueOutputs,
                                     String currentFrame) {

        System.out.println("\n🏗️  FRAME CONTEXT ANALYSIS:");

        // Group variables by frame
        Map<String, List<VarId>> variablesByFrame = new HashMap<>();
        for (VarId varId : nodeValueOutputs.keySet()) {
            variablesByFrame.computeIfAbsent(varId.getFrame(), k -> new ArrayList<>()).add(varId);
        }

        System.out.printf("  Current Frame: %s%n", currentFrame);
        System.out.printf("  Total Frames with Variables: %d%n", variablesByFrame.size());

        for (Map.Entry<String, List<VarId>> entry : variablesByFrame.entrySet()) {
            String frame = entry.getKey();
            List<VarId> vars = entry.getValue();

            System.out.printf("    📁 Frame '%s': %d variables", frame, vars.size());
            if (frame.equals(currentFrame)) {
                System.out.printf(" ⭐ (CURRENT)");
            }
            System.out.println();

            // Show key variables in this frame
            List<VarId> relevantVars = vars.stream()
                    .filter(v -> v.getVariable().contains("pooler") || v.getVariable().contains("1492"))
                    .collect(Collectors.toList());

            if (!relevantVars.isEmpty()) {
                System.out.printf("      🔍 Relevant variables: ");
                for (int i = 0; i < Math.min(3, relevantVars.size()); i++) {
                    System.out.printf("%s", relevantVars.get(i).getVariable());
                    if (i < Math.min(3, relevantVars.size()) - 1) System.out.printf(", ");
                }
                if (relevantVars.size() > 3) {
                    System.out.printf(" (+%d more)", relevantVars.size() - 3);
                }
                System.out.println();
            }
        }
    }

    /**
     * Trace dependency chain for failed operations
     */
    private void traceDependencyChain(String opName, SameDiff sameDiff, Set<String> allExecuted, int depth) {
        String indent = "  ".repeat(depth);

        if (depth == 0) {
            System.out.printf("🔗 Dependency chain for: %s%n", opName);
        }

        if (depth > 5) {
            System.out.printf("%s  ⚠️  Max depth reached%n", indent);
            return;
        }

        SameDiffOp op = sameDiff.getOps().get(opName);
        if (op == null) {
            System.out.printf("%s  ❌ Operation not found%n", indent);
            return;
        }

        List<String> inputs = op.getInputsToOp();
        if (inputs == null || inputs.isEmpty()) {
            System.out.printf("%s  ✅ No dependencies (leaf operation)%n", indent);
            return;
        }

        for (String input : inputs) {
            String producerOp = findProducerForVariable(input, sameDiff);
            boolean producerExecuted = producerOp != null && allExecuted.contains(producerOp);

            System.out.printf("%s  📥 %s: ", indent, input);
            if (producerExecuted) {
                System.out.printf("✅ Producer '%s' executed%n", producerOp);
            } else if (producerOp != null) {
                System.out.printf("❌ Producer '%s' NOT executed%n", producerOp);
                // Recursively trace why producer didn't execute
                traceDependencyChain(producerOp, sameDiff, allExecuted, depth + 1);
            } else {
                System.out.printf("❓ No producer found (constant/placeholder?)%n");
            }
        }
    }

    /**
     * Find which operation produces a given variable
     */
    private String findProducerForVariable(String variable, SameDiff sameDiff) {
        for (Map.Entry<String, SameDiffOp> entry : sameDiff.getOps().entrySet()) {
            SameDiffOp op = entry.getValue();
            List<String> outputs = op.getOutputsOfOp();
            if (outputs != null && outputs.contains(variable)) {
                return entry.getKey();
            }
        }
        return null;
    }

    /**
     * Analyze post-loop specific context
     */
    private void analyzePostLoopContext(Set<String> missing, Set<String> allExecuted) {
        System.out.println("\n🔄 POST-LOOP EXECUTION ANALYSIS:");

        Map<String, LoopInfo> activeLoops = loopAnalyzer.getActiveLoops();
        Map<String, List<LoopTerminationEvent>> terminationHistory = loopAnalyzer.getTerminationHistory();

        for (Map.Entry<String, LoopInfo> entry : activeLoops.entrySet()) {
            String frameName = entry.getKey();
            LoopInfo loopInfo = entry.getValue();

            System.out.printf("  📍 Frame '%s': Status=%s, Iterations=%d%n",
                    frameName, loopInfo.getStatus(), loopInfo.getMaxIterationsObserved());

            // Check if this loop should have exported variables
            List<LoopTerminationEvent> events = terminationHistory.get(frameName);
            if (events != null) {
                System.out.printf("    Termination events: %d%n", events.size());
                for (LoopTerminationEvent event : events) {
                    if (event.getTerminationType().toString().contains("EXIT")) {
                        System.out.printf("      ✅ Exit event: %s%n", event.getTriggerOperation());
                    }
                }
            }

            // Check for missing Exit operations
            checkForMissingExitOperations(frameName, missing, allExecuted);
        }
    }

    /**
     * Check if there should be Exit operations that didn't execute
     */
    private void checkForMissingExitOperations(String frameName, Set<String> missing, Set<String> allExecuted) {
        List<String> expectedExits = missing.stream()
                .filter(op -> op.contains(frameName) && op.contains("exit"))
                .collect(Collectors.toList());

        if (!expectedExits.isEmpty()) {
            System.out.printf("    ❌ Missing Exit operations: %s%n", expectedExits);
        } else {
            System.out.printf("    ℹ️  All expected Exit operations executed%n");
        }
    }

    /**
     * Infer which frame an operation should execute in based on its name
     */
    private String inferOperationFrame(String opName) {
        if (opName.contains("/")) {
            String[] parts = opName.split("/");
            if (parts.length > 1) {
                return parts[0];
            }
        }
        return "main";
    }


    /**
     * Enhance variable list with actual values for display
     */
    private List<String> enhanceVariableListWithValues(List<String> variableNames, FrameIter frameIter) {
        if (variableNames == null || variableNames.isEmpty()) {
            return new ArrayList<>();
        }

        List<String> enhancedList = new ArrayList<>();
        Map<VarId, org.nd4j.autodiff.samediff.config.SDValue> nodeValueOutputs = getNodeValueOutputsFromAnalyzer();

        for (String varName : variableNames) {
            Object actualValue = findActualValue(varName, frameIter, nodeValueOutputs);
            String valueDisplay = formatValueForDisplay(actualValue, varName);
            enhancedList.add(varName + "=" + valueDisplay);
        }

        return enhancedList;
    }

    /**
     * Format a value for compact display in execution trace
     */
    private String formatValueForDisplay(Object value, String varName) {
        if (value == null) {
            return "<null>";
        }

        if (value instanceof org.nd4j.linalg.api.ndarray.INDArray) {
            org.nd4j.linalg.api.ndarray.INDArray arr = (org.nd4j.linalg.api.ndarray.INDArray) value;

            if (arr.isScalar()) {
                return String.format("%.4f", arr.getDouble(0));
            } else if (arr.length() <= 5) {
                // Show all values for small arrays
                StringBuilder sb = new StringBuilder("[");
                for (int i = 0; i < arr.length(); i++) {
                    if (i > 0) sb.append(",");
                    sb.append(String.format("%.3f", arr.getDouble(i)));
                }
                sb.append("]");
                return sb.toString();
            } else {
                // Show shape and first/last values for larger arrays
                return String.format("shape%s[%.3f...%.3f]",
                        java.util.Arrays.toString(arr.shape()),
                        arr.getDouble(0),
                        arr.getDouble(arr.length() - 1));
            }
        } else if (value instanceof List) {
            List<?> list = (List<?>) value;
            return String.format("List[%d]", list.size());
        } else if (value instanceof Map) {
            Map<?, ?> map = (Map<?, ?>) value;
            return String.format("Map[%d]", map.size());
        } else {
            String str = value.toString();
            return str.length() > 30 ? str.substring(0, 27) + "..." : str;
        }
    }




    /**
     * Integrate execution step with loop termination analysis - ENHANCED WITH ACTUAL VALUES
     */
    private void integrateLoopAnalysis(ExecType type, String name, FrameIter frameIter,
                                       List<String> inputs, List<String> outputs, String status, int iteration) {
        String frameName = frameIter.getFrame();

        // Extract ACTUAL variable values from nodeValueOutputs
        Map<String, Object> actualVariableValues = extractActualValues(outputs, frameIter);
        Map<String, Object> actualInputValues = extractActualValues(inputs, frameIter);

        switch (type) {
            case OP:
                if (name.contains("Enter")) {
                    loopAnalyzer.onLoopFrameEnter(frameName, name,
                            new org.nd4j.autodiff.samediff.internal.FrameIter(frameName, iteration,
                                    frameIter.getParentFrame() != null ?
                                            new org.nd4j.autodiff.samediff.internal.FrameIter(
                                                    frameIter.getParentFrame().getFrame(),
                                                    frameIter.getParentFrame().getIteration(), null) : null));
                } else if (name.contains("LoopCond")) {
                    // For loop conditions, try to extract the actual condition result
                    Object conditionValue = extractConditionValue(name, outputs, frameIter);
                    loopAnalyzer.onLoopConditionEvaluation(frameName, name, conditionValue, actualInputValues, iteration);
                } else if (name.contains("Switch")) {
                    Object predicateValue = extractSwitchPredicateValue(inputs, frameIter);
                    String branchTaken = determineSwitchBranch(outputs, frameIter);
                    loopAnalyzer.onSwitchOperation(frameName, name, predicateValue, branchTaken, iteration);
                } else if (name.contains("Exit")) {
                    Object exitValue = extractExitValue(outputs, frameIter);
                    loopAnalyzer.onExitOperation(frameName, name, exitValue, iteration);
                } else if (name.contains("NextIteration")) {
                    loopAnalyzer.onLoopIteration(frameName, iteration + 1, actualVariableValues);
                } else {
                    // Regular operation in loop - pass actual values
                    loopAnalyzer.onLoopIteration(frameName, iteration, actualVariableValues);
                }

                // Check for error status
                if (status != null && status.toLowerCase().contains("error")) {
                    Exception error = new RuntimeException("Execution error: " + status);
                    Map<String, Object> errorContext = new HashMap<>();
                    errorContext.put("actualInputs", actualInputValues);
                    errorContext.put("actualOutputs", actualVariableValues);
                    errorContext.put("executionStep", stepCounter.get());
                    loopAnalyzer.onLoopError(frameName, iteration, name, error, errorContext);
                }
                break;

            default:
                // For other types, just track iteration if in a loop with actual values
                if (iteration > 0) {
                    loopAnalyzer.onLoopIteration(frameName, iteration, actualVariableValues);
                }
                break;
        }
    }


    /**
     * Extract actual condition value for loop conditions
     */
    private Object extractConditionValue(String conditionOpName, List<String> outputs, FrameIter frameIter) {
        if (outputs == null || outputs.isEmpty()) {
            return "<no_condition_output>";
        }

        // Loop condition typically outputs a boolean scalar
        String conditionOutputVar = outputs.get(0);
        Object actualValue = findActualValue(conditionOutputVar, frameIter, getNodeValueOutputsFromAnalyzer());

        if (actualValue instanceof org.nd4j.linalg.api.ndarray.INDArray) {
            org.nd4j.linalg.api.ndarray.INDArray arr = (org.nd4j.linalg.api.ndarray.INDArray) actualValue;
            if (arr.isScalar()) {
                return arr.getDouble(0) != 0.0; // Convert to boolean
            }
            return actualValue;
        }

        return actualValue != null ? actualValue : "<condition_value_not_found>";
    }

    /**
     * Extract switch predicate value
     */
    private Object extractSwitchPredicateValue(List<String> inputs, FrameIter frameIter) {
        if (inputs == null || inputs.size() < 2) {
            return "<no_predicate_input>";
        }

        // Switch predicate is typically the second input
        String predicateVar = inputs.get(1);
        return findActualValue(predicateVar, frameIter, getNodeValueOutputsFromAnalyzer());
    }

    /**
     * Extract exit value
     */
    private Object extractExitValue(List<String> outputs, FrameIter frameIter) {
        if (outputs == null || outputs.isEmpty()) {
            return "<no_exit_output>";
        }

        String exitVar = outputs.get(0);
        Object actualValue = findActualValue(exitVar, frameIter, getNodeValueOutputsFromAnalyzer());
        return actualValue != null ? actualValue : "<exit_value_not_found>";
    }

    /**
     * Determine which branch was taken in a switch operation
     */
    private String determineSwitchBranch(List<String> outputs, FrameIter frameIter) {
        if (outputs == null || outputs.isEmpty()) {
            return "unknown_branch";
        }

        Map<VarId, org.nd4j.autodiff.samediff.config.SDValue> nodeValueOutputs = getNodeValueOutputsFromAnalyzer();
        if (nodeValueOutputs == null) {
            return "no_node_values";
        }

        // Check which output actually has a value
        for (int i = 0; i < outputs.size(); i++) {
            String outputVar = outputs.get(i);
            Object value = findActualValue(outputVar, frameIter, nodeValueOutputs);
            if (value != null) {
                return i == 0 ? "LEFT" : "RIGHT";
            }
        }

        return "no_active_branch";
    }

    /**
     * Extract actual values from nodeValueOutputs for given variable names
     */
    private Map<String, Object> extractActualValues(List<String> variableNames, FrameIter frameIter) {
        Map<String, Object> actualValues = new HashMap<>();

        if (variableNames == null || loopAnalyzer == null) {
            return actualValues;
        }

        // Access the nodeValueOutputs from the loop analyzer's SameDiff instance
        Map<VarId, org.nd4j.autodiff.samediff.config.SDValue> nodeValueOutputs =
                getNodeValueOutputsFromAnalyzer();

        if (nodeValueOutputs == null) {
            return actualValues;
        }

        for (String varName : variableNames) {
            // Try multiple VarId combinations to find the actual value
            Object actualValue = findActualValue(varName, frameIter, nodeValueOutputs);
            if (actualValue != null) {
                actualValues.put(varName, actualValue);
            } else {
                actualValues.put(varName, "<value_not_found_in_frame:" + frameIter.getFrame() + ">");
            }
        }

        return actualValues;
    }

    /**
     * Find actual value for a variable across different frame/iteration combinations
     */
    private Object findActualValue(String varName, FrameIter frameIter,
                                   Map<VarId, org.nd4j.autodiff.samediff.config.SDValue> nodeValueOutputs) {

        // Try current frame and iteration
        VarId currentVarId = new VarId(varName, frameIter.getFrame(), frameIter.getIteration(), frameIter.getParentFrame());
        org.nd4j.autodiff.samediff.config.SDValue value = nodeValueOutputs.get(currentVarId);
        if (value != null) {
            return extractValueFromSDValue(value, varName);
        }

        // Try current frame, iteration 0
        VarId frameVarId = new VarId(varName, frameIter.getFrame(), 0, frameIter.getParentFrame());
        value = nodeValueOutputs.get(frameVarId);
        if (value != null) {
            return extractValueFromSDValue(value, varName);
        }

        // Try outer frame
        VarId outerVarId = new VarId(varName, "main", 0, null);
        value = nodeValueOutputs.get(outerVarId);
        if (value != null) {
            return extractValueFromSDValue(value, varName);
        }

        // Search all VarIds for this variable name as fallback
        for (Map.Entry<VarId, org.nd4j.autodiff.samediff.config.SDValue> entry : nodeValueOutputs.entrySet()) {
            if (entry.getKey().getVariable().equals(varName)) {
                return extractValueFromSDValue(entry.getValue(), varName);
            }
        }

        return null;
    }

    /**
     * Extract the underlying value from an SDValue for display
     */
    private Object extractValueFromSDValue(org.nd4j.autodiff.samediff.config.SDValue sdValue, String varName) {
        if (sdValue == null) {
            return null;
        }

        switch (sdValue.getSdValueType()) {
            case TENSOR:
                return sdValue.getTensorValue(); // This is an INDArray
            case LIST:
                return sdValue.getListValue(); // This is a List<INDArray>
            case DICT:
                return sdValue.getDictValue(); // This is a Map<String, INDArray>
            default:
                return sdValue.toString();
        }
    }

    /**
     * Get nodeValueOutputs from the loop analyzer - requires access to the underlying SameDiff
     */
    private Map<VarId, org.nd4j.autodiff.samediff.config.SDValue> getNodeValueOutputsFromAnalyzer() {
        if (loopAnalyzer == null) {
            return null;
        }

        // The loop analyzer should expose the nodeValueOutputs it was constructed with
        return loopAnalyzer.getNodeValueOutputs(); // This method needs to be added to LoopTerminationAnalyzer
    }

    /**
     * Track dependencies between frames
     */
    private void trackFrameDependencies(String frame, List<String> inputs) {
        if (!enableDependencyTracking || inputs == null) return;
        frameDependencies.computeIfAbsent(frame, k -> new HashSet<>()).addAll(inputs);
    }

    // ===================================================================
    // REPORTING METHODS - Called by printCompleteAnalysisReport()
    // ===================================================================

    private void printReportHeader() {
        System.out.println("\n" + "═".repeat(100));
        System.out.println("🔄 SAMEDIFF EXECUTION & LOOP ANALYSIS COMPREHENSIVE REPORT");
        System.out.println("═".repeat(100));
        System.out.println("Generated: " + LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")));
        System.out.println("═".repeat(100));
    }

    private void printExecutionOverview() {
        System.out.println("\n📋 EXECUTION OVERVIEW");
        System.out.println("─".repeat(50));
        System.out.printf("Total Execution Steps: %d%n", executionSteps.size());

        // Count by status
        Map<String, Long> statusCounts = new HashMap<>();
        for (ExecutionStep step : executionSteps) {
            String status = step.getStatus() != null ? step.getStatus() : "SUCCESS";
            statusCounts.merge(status, 1L, Long::sum);
        }

        System.out.println("\nStatus Breakdown:");
        statusCounts.entrySet().stream()
                .sorted(Map.Entry.<String, Long>comparingByValue().reversed())
                .forEach(entry -> System.out.printf("  %s: %d%n", entry.getKey(), entry.getValue()));

        // Frame summary
        Set<String> frames = new HashSet<>();
        for (ExecutionStep step : executionSteps) {
            frames.add(step.getFrame());
        }
        System.out.printf("Frames Processed: %d%n", frames.size());

        if (enableDependencyTracking && !frameDependencies.isEmpty()) {
            System.out.println("\n🔗 Frame Dependencies:");
            for (Map.Entry<String, Set<String>> entry : frameDependencies.entrySet()) {
                System.out.printf("  %s → %s%n", entry.getKey(), String.join(", ", entry.getValue()));
            }
        }
    }

    private void printExecutionTrace() {
        System.out.println("\n🔍 EXECUTION TRACE");
        System.out.println("─".repeat(50));

        String currentFrame = "";
        int currentIteration = -1;

        for (ExecutionStep step : executionSteps) {
            // Print frame/iteration separator if changed
            if (!step.getFrame().equals(currentFrame) || step.getIteration() != currentIteration) {
                if (!currentFrame.isEmpty()) {
                    System.out.println(); // Add spacing between frame sections
                }
                System.out.printf("📍 FRAME: %s | ITERATION: %d%n", step.getFrame(), step.getIteration());
                System.out.println("  " + "─".repeat(40));
                currentFrame = step.getFrame();
                currentIteration = step.getIteration();
            }

            printExecutionStep(step);
        }
    }

    private void printLoopAnalysisReport() {
        if (!enableLoopAnalysis || loopAnalyzer == null) {
            System.out.println("\n🔄 LOOP ANALYSIS: Disabled");
            return;
        }

        System.out.println("\n🔄 LOOP ANALYSIS REPORT");
        System.out.println("─".repeat(50));

        Map<String, Object> loopStats = loopAnalyzer.getLoopStatistics();
        System.out.printf("Total Loops Tracked: %d%n", loopStats.get("totalLoopsTracked"));
        System.out.printf("Total Termination Events: %d%n", loopStats.get("totalTerminationEvents"));
        System.out.printf("Early Terminations: %d%n", loopStats.get("earlyTerminations"));

        // Print early terminated loops
        List<String> earlyTerminated = loopAnalyzer.getEarlyTerminatedLoops();
        if (!earlyTerminated.isEmpty()) {
            System.out.println("\n❌ EARLY TERMINATED LOOPS:");
            for (String frameName : earlyTerminated) {
                System.out.printf("  📍 Frame: %s%n", frameName);
                String terminationReport = loopAnalyzer.generateTerminationReport(frameName);
                // Print first few lines of the report
                String[] lines = terminationReport.split("\n");
                for (int i = 0; i < Math.min(5, lines.length); i++) {
                    System.out.println("    " + lines[i]);
                }
                if (lines.length > 5) {
                    System.out.println("    ... (see detailed analysis below)");
                }
            }
        }

        // Print active loops
        Map<String, LoopInfo> activeLoops = loopAnalyzer.getActiveLoops();
        if (!activeLoops.isEmpty()) {
            System.out.println("\n🔄 ACTIVE LOOPS:");
            for (Map.Entry<String, LoopInfo> entry : activeLoops.entrySet()) {
                LoopInfo loopInfo = entry.getValue();
                System.out.printf("  📍 Frame: %s%n", entry.getKey());
                System.out.printf("    Status: %s%n", loopInfo.getStatus());
                System.out.printf("    Current Iteration: %d%n", loopInfo.getCurrentIteration());
                System.out.printf("    Max Iterations: %d%n", loopInfo.getMaxIterationsObserved());

                if (loopInfo.isEarlyTerminationDetected()) {
                    System.out.println("    ⚠️  Early termination detected!");
                }
            }
        }
    }

    private void printErrorAnalysisReport() {
        System.out.println("\n❌ ERROR ANALYSIS REPORT");
        System.out.println("─".repeat(50));

        if (loopAnalyzer == null) {
            System.out.println("Loop analyzer not available - no error analysis possible");
            return;
        }

        Map<String, List<LoopTerminationEvent>> history = loopAnalyzer.getTerminationHistory();

        if (history.isEmpty()) {
            System.out.println("No loop termination events detected");
            return;
        }

        // Print summary statistics
        Map<String, Object> stats = LoopTerminationEventUtils.calculateTerminationStatistics(history);
        System.out.println("Error Summary:");
        System.out.println("  Total Events: " + stats.get("totalEvents"));
        System.out.println("  Unique Frames: " + stats.get("uniqueFrames"));
        System.out.println("  Early Termination Rate: " +
                String.format("%.1f%%", (Double) stats.get("earlyTerminationRate") * 100));

        @SuppressWarnings("unchecked")
        Map<TerminationType, Long> typeCounts = (Map<TerminationType, Long>) stats.get("eventsByType");
        System.out.println("\nTermination Types:");
        for (Map.Entry<TerminationType, Long> entry : typeCounts.entrySet()) {
            System.out.println("  " + entry.getKey() + ": " + entry.getValue());
        }

        // Print individual error reports for frames with errors
        boolean hasErrors = false;
        for (String frameName : history.keySet()) {
            List<LoopTerminationEvent> events = history.get(frameName);
            List<LoopTerminationEvent> errorEvents = events.stream()
                    .filter(event -> event.getTerminationType() == TerminationType.ERROR_TERMINATION ||
                            event.isWasEarlyTermination())
                    .collect(Collectors.toList());

            if (!errorEvents.isEmpty()) {
                if (!hasErrors) {
                    System.out.println("\nDetailed Error Reports:");
                    hasErrors = true;
                }
                System.out.printf("\n  Frame %s (%d error events):%n", frameName, errorEvents.size());
                for (LoopTerminationEvent event : errorEvents) {
                    System.out.printf("    • Iteration %d: %s - %s%n",
                            event.getIteration(), event.getTerminationType(), event.getTerminationReason());
                }
            }
        }

        if (!hasErrors) {
            System.out.println("No error events detected - all loops terminated normally");
        }
    }

    private void printStatisticsReport() {
        System.out.println("\n📊 DETAILED STATISTICS");
        System.out.println("─".repeat(50));

        Map<ExecType, Long> typeStats = new HashMap<>();
        Map<String, Long> frameStats = new HashMap<>();

        for (ExecutionStep step : executionSteps) {
            typeStats.merge(step.getType(), 1L, Long::sum);
            frameStats.merge(step.getFrame(), 1L, Long::sum);
        }

        System.out.println("📈 Operations by Type:");
        typeStats.entrySet().stream()
                .sorted(Map.Entry.<ExecType, Long>comparingByValue().reversed())
                .forEach(entry -> System.out.printf("  %s: %d%n", entry.getKey(), entry.getValue()));

        System.out.println("\n🎯 Operations by Frame:");
        frameStats.entrySet().stream()
                .sorted(Map.Entry.<String, Long>comparingByValue().reversed())
                .forEach(entry -> System.out.printf("  %s: %d%n", entry.getKey(), entry.getValue()));

        // Add loop statistics
        if (enableLoopAnalysis && loopAnalyzer != null) {
            Map<String, Object> loopStats = loopAnalyzer.getLoopStatistics();
            System.out.println("\n🔄 Loop Statistics:");
            for (Map.Entry<String, Object> entry : loopStats.entrySet()) {
                System.out.printf("  %s: %s%n", entry.getKey(), entry.getValue());
            }
        }
    }

    private void printReportFooter() {
        System.out.println("\n" + "═".repeat(100));
        System.out.println("END OF COMPREHENSIVE ANALYSIS REPORT");
        System.out.println("═".repeat(100));
    }

    // ===================================================================
    // UTILITY METHODS
    // ===================================================================

    /**
     * Print a single execution step
     */
    private void printExecutionStep(ExecutionStep step) {
        String typeIcon = getTypeIcon(step.getType());
        String statusIcon = getStatusIcon(step.getStatus());

        // Main step line
        System.out.printf("  %s [%03d] %s%s %s%n",
                typeIcon,
                step.getStepNumber(),
                enableTimestamps ? step.getTimestamp() + " " : "",
                statusIcon,
                step.getName()
        );

        // Parent frame info
        if (step.getParentFrame() != null) {
            System.out.printf("      ↳ Parent: %s (iter: %d)%n",
                    step.getParentFrame().getFrame(),
                    step.getParentFrame().getIteration());
        }

        // Inputs
        if (!step.getInputs().isEmpty()) {
            System.out.printf("      📥 Inputs: %s%n", String.join(", ", step.getInputs()));
        }

        // Outputs
        if (!step.getOutputs().isEmpty()) {
            System.out.printf("      📤 Outputs: %s%n", String.join(", ", step.getOutputs()));
        }

        // Status details
        if (step.getStatus() != null && !step.getStatus().equals("SUCCESS")) {
            System.out.printf("      ℹ️  Status: %s%n", step.getStatus());
        }
    }

    /**
     * Get icon for execution type
     */
    private String getTypeIcon(ExecType type) {
        switch (type) {
            case OP: return "⚙️";
            case VARIABLE: return "📊";
            case CONSTANT: return "📋";
            case PLACEHOLDER: return "🔲";
            case SWITCH_L: return "↙️";
            case SWITCH_R: return "↘️";
            case EXEC_START: return "🚀";
            case CONTROL_DEP: return "🔗";
            default: return "❓";
        }
    }

    /**
     * Get icon for status
     */
    private String getStatusIcon(String status) {
        if (status == null || status.equals("SUCCESS")) return "✅";
        if (status.toLowerCase().contains("error")) return "❌";
        if (status.toLowerCase().contains("warn")) return "⚠️";
        if (status.toLowerCase().contains("skip")) return "⏭️";
        return "ℹ️";
    }

    // ===================================================================
    // ADDITIONAL PUBLIC METHODS FOR SPECIFIC USE CASES
    // ===================================================================

    /**
     * Print execution trace for a specific frame
     */
    public void printFrameTrace(String frameName) {
        System.out.println("\n🔍 FRAME-SPECIFIC TRACE: " + frameName);
        System.out.println("─".repeat(50));

        List<ExecutionStep> frameSteps = executionSteps.stream()
                .filter(step -> step.getFrame().equals(frameName))
                .collect(Collectors.toList());

        if (frameSteps.isEmpty()) {
            System.out.println("No execution steps found for frame: " + frameName);
            return;
        }

        frameSteps.forEach(this::printExecutionStep);

        System.out.printf("\nFrame Summary - Total Steps: %d%n", frameSteps.size());

        // Print loop analysis for this frame
        if (enableLoopAnalysis && loopAnalyzer != null) {
            String terminationReport = loopAnalyzer.generateTerminationReport(frameName);
            if (!terminationReport.contains("No loop information found")) {
                System.out.println("\n🔄 Loop Analysis for Frame:");
                System.out.println(terminationReport);
            }
        }
    }

    /**
     * Print only error steps
     */
    public void printErrorTrace() {
        System.out.println("\n❌ ERROR-ONLY TRACE");
        System.out.println("─".repeat(50));

        List<ExecutionStep> errorSteps = executionSteps.stream()
                .filter(step -> step.getStatus() != null &&
                        (step.getStatus().toLowerCase().contains("error") ||
                                step.getStatus().toLowerCase().contains("failed")))
                .collect(Collectors.toList());

        if (errorSteps.isEmpty()) {
            System.out.println("No error steps detected");
            return;
        }

        errorSteps.forEach(this::printExecutionStep);
        System.out.printf("\nTotal Error Steps: %d%n", errorSteps.size());
    }

    // ===================================================================
    // CONFIGURATION AND UTILITY METHODS
    // ===================================================================

    /**
     * Clear all recorded steps and loop analysis data
     */
    public void clear() {
        executionSteps.clear();
        frameDependencies.clear();
        stepCounter.set(0);

        if (loopAnalyzer != null) {
            loopAnalyzer.clearAll();
        }
    }

    /**
     * Configure error reporting options
     */
    public void configureErrorReporting(boolean includeValues, boolean includeShapes,
                                        boolean includeHistory, boolean generateViz) {
        if (errorReporter != null) {
            errorReporter.setIncludeVariableValues(includeValues);
            errorReporter.setIncludeVariableShapes(includeShapes);
            errorReporter.setIncludeOperationHistory(includeHistory);
            errorReporter.setGenerateVisualizations(generateViz);
        }
    }

    // Configuration setters
    public void setEnableTimestamps(boolean enable) { this.enableTimestamps = enable; }
    public void setEnableFrameTracking(boolean enable) { this.enableFrameTracking = enable; }
    public void setEnableDependencyTracking(boolean enable) { this.enableDependencyTracking = enable; }
    public void setEnableLoopAnalysis(boolean enable) { this.enableLoopAnalysis = enable; }

    // Getters
    public int getStepCount() { return executionSteps.size(); }
    public List<ExecutionStep> getStepsForFrame(String frameName) {
        return executionSteps.stream()
                .filter(step -> step.getFrame().equals(frameName))
                .collect(Collectors.toList());
    }

    /**
     * Enhanced failure analysis specifically for control flow operations
     * This method provides detailed analysis when Switch, Merge, Enter, Exit, NextIteration, or LoopCond operations fail
     */
    public void analyzeControlFlowFailure(DifferentialFunction op, Set<VarId> opInputs, Set<VarId> allIterInputs,
                                          Set<String> constAndPhInputs, FrameIter outputFrameIter,
                                          Map<VarId, org.nd4j.autodiff.samediff.config.SDValue> nodeValueOutputs,
                                          Exception exception) {

        String opName = op.getOwnName();
        String opType = op.getClass().getSimpleName();
        String frame = outputFrameIter.getFrame();
        int iteration = outputFrameIter.getIteration();

        System.out.println("\n" + "🔄".repeat(20) + " CONTROL FLOW FAILURE ANALYSIS " + "🔄".repeat(20));
        System.out.printf("📍 FAILED OPERATION: %s (%s)%n", opName, opType);
        System.out.printf("📍 FRAME CONTEXT: %s (iteration: %d)%n", frame, iteration);
        System.out.printf("📍 EXCEPTION: %s%n", exception.getMessage());

        // Analyze specific control flow operation type
        if (op instanceof Switch) {
            analyzeSwitchFailure((Switch) op, opInputs, nodeValueOutputs, outputFrameIter);
        } else if (op instanceof Merge) {
            analyzeMergeFailure((Merge) op, opInputs, nodeValueOutputs, outputFrameIter);
        } else if (op instanceof Enter) {
            analyzeEnterFailure((Enter) op, opInputs, nodeValueOutputs, outputFrameIter);
        } else if (op instanceof Exit) {
            analyzeExitFailure((Exit) op, opInputs, nodeValueOutputs, outputFrameIter);
        } else if (op instanceof NextIteration) {
            analyzeNextIterationFailure((NextIteration) op, opInputs, nodeValueOutputs, outputFrameIter);
        } else if (op instanceof LoopCond) {
            analyzeLoopCondFailure((LoopCond) op, opInputs, nodeValueOutputs, outputFrameIter);
        } else {
            analyzeGenericControlFlowFailure(op, opInputs, nodeValueOutputs, outputFrameIter);
        }

        // Analyze frame state and loop context
        analyzeFrameStateAtFailure(frame, iteration, nodeValueOutputs);

        // Analyze input availability and values
        analyzeControlFlowInputs(opInputs, allIterInputs, constAndPhInputs, nodeValueOutputs, outputFrameIter);

        // Check for loop termination patterns
        if (loopAnalyzer != null && loopAnalyzer.getActiveLoops().containsKey(frame)) {
            analyzeLoopTerminationContext(frame, iteration, opName, exception);
        }

        // Provide recovery suggestions
        provideControlFlowRecoverySuggestions(op, frame, iteration, exception);

        System.out.println("🔄".repeat(80));
    }

    /**
     * Analyze Switch operation failure
     */
    private void analyzeSwitchFailure(Switch switchOp, Set<VarId> opInputs,
                                      Map<VarId, org.nd4j.autodiff.samediff.config.SDValue> nodeValueOutputs,
                                      FrameIter frameIter) {
        System.out.println("\n🔀 SWITCH OPERATION FAILURE ANALYSIS:");

        String[] argNames = switchOp.argNames();
        if (argNames.length >= 2) {
            String dataInput = argNames[0];
            String predicateInput = argNames[1];

            System.out.printf("  📥 Data Input: %s%n", dataInput);
            System.out.printf("  🎯 Predicate Input: %s%n", predicateInput);

            // Check data input availability
            VarId dataVarId = frameIter.toVarId(dataInput);
            org.nd4j.autodiff.samediff.config.SDValue dataValue = nodeValueOutputs.get(dataVarId);
            System.out.printf("  📊 Data Available: %s%n", dataValue != null ? "✅ YES" : "❌ NO");

            // Check predicate availability and value
            VarId predicateVarId = frameIter.toVarId(predicateInput);
            org.nd4j.autodiff.samediff.config.SDValue predicateValue = nodeValueOutputs.get(predicateVarId);
            System.out.printf("  🎯 Predicate Available: %s%n", predicateValue != null ? "✅ YES" : "❌ NO");

            if (predicateValue != null && predicateValue.getTensorValue() != null) {
                org.nd4j.linalg.api.ndarray.INDArray predArray = predicateValue.getTensorValue();
                System.out.printf("  🎯 Predicate Value: %s%n", formatPredicateValue(predArray));
                System.out.printf("  🎯 Predicate Shape: %s%n", java.util.Arrays.toString(predArray.shape()));
                System.out.printf("  🎯 Predicate Type: %s%n", predArray.dataType());

                if (!predArray.isScalar()) {
                    System.out.println("  ❌ ISSUE: Predicate should be scalar boolean");
                }
                if (predArray.dataType() != org.nd4j.linalg.api.buffer.DataType.BOOL) {
                    System.out.println("  ❌ ISSUE: Predicate should be boolean type");
                }
            } else {
                System.out.println("  ❌ CRITICAL: Predicate value is null or invalid");
            }
        }

        // Check for output variable conflicts
        String[] outputNames = switchOp.outputVariablesNames();
        if (outputNames != null) {
            System.out.printf("  📤 Expected Outputs: %s%n", java.util.Arrays.toString(outputNames));
        }
    }

    /**
     * Analyze Merge operation failure
     */
    private void analyzeMergeFailure(Merge mergeOp, Set<VarId> opInputs,
                                     Map<VarId, org.nd4j.autodiff.samediff.config.SDValue> nodeValueOutputs,
                                     FrameIter frameIter) {
        System.out.println("\n🔗 MERGE OPERATION FAILURE ANALYSIS:");

        String[] argNames = mergeOp.argNames();
        if (argNames.length >= 2) {
            System.out.printf("  📥 Input 1: %s%n", argNames[0]);
            System.out.printf("  📥 Input 2: %s%n", argNames[1]);

            // Check availability of both inputs
            VarId input1VarId = frameIter.toVarId(argNames[0]);
            VarId input2VarId = frameIter.toVarId(argNames[1]);

            org.nd4j.autodiff.samediff.config.SDValue value1 = nodeValueOutputs.get(input1VarId);
            org.nd4j.autodiff.samediff.config.SDValue value2 = nodeValueOutputs.get(input2VarId);

            System.out.printf("  📊 Input 1 Available: %s%n", value1 != null ? "✅ YES" : "❌ NO");
            System.out.printf("  📊 Input 2 Available: %s%n", value2 != null ? "✅ YES" : "❌ NO");

            if (value1 == null && value2 == null) {
                System.out.println("  ❌ CRITICAL: Both merge inputs are unavailable");

                // Check if inputs are from different iterations
                analyzeInputsAcrossIterations(argNames[0], argNames[1], frameIter, nodeValueOutputs);
            } else {
                System.out.printf("  ✅ MERGE SOURCE: %s%n", value1 != null ? argNames[0] : argNames[1]);
            }
        }
    }

    /**
     * Analyze Enter operation failure
     */
    private void analyzeEnterFailure(Enter enterOp, Set<VarId> opInputs,
                                     Map<VarId, org.nd4j.autodiff.samediff.config.SDValue> nodeValueOutputs,
                                     FrameIter frameIter) {
        System.out.println("\n📥 ENTER OPERATION FAILURE ANALYSIS:");

        String targetFrame = enterOp.getFrameName();
        boolean isConstant = enterOp.isConstant();

        System.out.printf("  🎯 Target Frame: %s%n", targetFrame);
        System.out.printf("  📋 Is Constant: %s%n", isConstant);

        String[] argNames = enterOp.argNames();
        if (argNames.length >= 1) {
            String inputVar = argNames[0];
            System.out.printf("  📥 Input Variable: %s%n", inputVar);

            // Check input availability in current frame
            VarId inputVarId = frameIter.toVarId(inputVar);
            org.nd4j.autodiff.samediff.config.SDValue inputValue = nodeValueOutputs.get(inputVarId);
            System.out.printf("  📊 Input Available in Current Frame: %s%n", inputValue != null ? "✅ YES" : "❌ NO");

            // Check in parent/outer frames
            if (inputValue == null) {
                System.out.println("  🔍 Searching in other frames:");
                VarId outerVarId = new VarId(inputVar, "main", 0, null);
                org.nd4j.autodiff.samediff.config.SDValue outerValue = nodeValueOutputs.get(outerVarId);
                System.out.printf("    📊 Available in 'main' frame: %s%n", outerValue != null ? "✅ YES" : "❌ NO");

                // Search across all frames
                searchVariableAcrossFrames(inputVar, nodeValueOutputs);
            }
        }
    }

    /**
     * Analyze Exit operation failure
     */
    private void analyzeExitFailure(Exit exitOp, Set<VarId> opInputs,
                                    Map<VarId, org.nd4j.autodiff.samediff.config.SDValue> nodeValueOutputs,
                                    FrameIter frameIter) {
        System.out.println("\n📤 EXIT OPERATION FAILURE ANALYSIS:");

        String currentFrame = frameIter.getFrame();
        FrameIter parentFrame = frameIter.getParentFrame();

        System.out.printf("  📍 Current Frame: %s%n", currentFrame);
        System.out.printf("  📍 Parent Frame: %s%n", parentFrame != null ? parentFrame.getFrame() : "None");

        if (parentFrame == null) {
            System.out.println("  ❌ ISSUE: No parent frame for Exit operation");
        }

        String[] argNames = exitOp.argNames();
        if (argNames.length >= 1) {
            String inputVar = argNames[0];
            System.out.printf("  📥 Input Variable: %s%n", inputVar);

            VarId inputVarId = frameIter.toVarId(inputVar);
            org.nd4j.autodiff.samediff.config.SDValue inputValue = nodeValueOutputs.get(inputVarId);
            System.out.printf("  📊 Input Available: %s%n", inputValue != null ? "✅ YES" : "❌ NO");

            if (inputValue != null && inputValue.getTensorValue() != null) {
                org.nd4j.linalg.api.ndarray.INDArray arr = inputValue.getTensorValue();
                System.out.printf("  📊 Input Shape: %s%n", java.util.Arrays.toString(arr.shape()));
                System.out.printf("  📊 Input Type: %s%n", arr.dataType());
            }
        }
    }

    /**
     * Analyze NextIteration operation failure
     */
    private void analyzeNextIterationFailure(NextIteration nextIterOp, Set<VarId> opInputs,
                                             Map<VarId, org.nd4j.autodiff.samediff.config.SDValue> nodeValueOutputs,
                                             FrameIter frameIter) {
        System.out.println("\n🔄 NEXT_ITERATION OPERATION FAILURE ANALYSIS:");

        String currentFrame = frameIter.getFrame();
        int currentIter = frameIter.getIteration();

        System.out.printf("  📍 Current Frame: %s%n", currentFrame);
        System.out.printf("  📍 Current Iteration: %d%n", currentIter);
        System.out.printf("  📍 Expected Output Iteration: %d%n", currentIter + 1);

        String[] argNames = nextIterOp.argNames();
        if (argNames.length >= 1) {
            String inputVar = argNames[0];
            System.out.printf("  📥 Input Variable: %s%n", inputVar);

            VarId inputVarId = frameIter.toVarId(inputVar);
            org.nd4j.autodiff.samediff.config.SDValue inputValue = nodeValueOutputs.get(inputVarId);
            System.out.printf("  📊 Input Available: %s%n", inputValue != null ? "✅ YES" : "❌ NO");

            // Check frame consistency
            if (opInputs != null) {
                for (VarId vid : opInputs) {
                    if (!vid.getFrame().equals(currentFrame)) {
                        System.out.printf("  ❌ FRAME MISMATCH: Input %s in frame %s, expected %s%n",
                                vid.getVariable(), vid.getFrame(), currentFrame);
                    }
                    if (vid.getIteration() != currentIter - 1) {
                        System.out.printf("  ❌ ITERATION MISMATCH: Input %s at iteration %d, expected %d%n",
                                vid.getVariable(), vid.getIteration(), currentIter - 1);
                    }
                }
            }
        }
    }

    /**
     * Analyze LoopCond operation failure
     */
    private void analyzeLoopCondFailure(LoopCond loopCondOp, Set<VarId> opInputs,
                                        Map<VarId, org.nd4j.autodiff.samediff.config.SDValue> nodeValueOutputs,
                                        FrameIter frameIter) {
        System.out.println("\n🔁 LOOP_CONDITION OPERATION FAILURE ANALYSIS:");

        String[] argNames = loopCondOp.argNames();
        if (argNames.length >= 1) {
            String inputVar = argNames[0];
            System.out.printf("  📥 Condition Input: %s%n", inputVar);

            VarId inputVarId = frameIter.toVarId(inputVar);
            org.nd4j.autodiff.samediff.config.SDValue inputValue = nodeValueOutputs.get(inputVarId);
            System.out.printf("  📊 Input Available: %s%n", inputValue != null ? "✅ YES" : "❌ NO");

            if (inputValue != null && inputValue.getTensorValue() != null) {
                org.nd4j.linalg.api.ndarray.INDArray condArray = inputValue.getTensorValue();
                System.out.printf("  📊 Condition Value: %s%n", formatPredicateValue(condArray));
                System.out.printf("  📊 Condition Shape: %s%n", java.util.Arrays.toString(condArray.shape()));
                System.out.printf("  📊 Condition Type: %s%n", condArray.dataType());

                if (!condArray.isScalar()) {
                    System.out.println("  ❌ ISSUE: Loop condition should be scalar boolean");
                }
                if (condArray.dataType() != org.nd4j.linalg.api.buffer.DataType.BOOL) {
                    System.out.println("  ❌ ISSUE: Loop condition should be boolean type");
                }

                if (condArray.isScalar() && condArray.dataType() == org.nd4j.linalg.api.buffer.DataType.BOOL) {
                    boolean condValue = condArray.getDouble(0) != 0.0;
                    System.out.printf("  🎯 Loop Should Continue: %s%n", condValue ? "✅ YES" : "❌ NO (TERMINATE)");

                    if (!condValue) {
                        System.out.println("  💡 DIAGNOSIS: Loop condition is FALSE - this should trigger termination");
                    }
                }
            } else {
                System.out.println("  ❌ CRITICAL: Loop condition value is null or invalid");
            }
        }
    }

    /**
     * Analyze generic control flow operation failure
     */
    private void analyzeGenericControlFlowFailure(DifferentialFunction op, Set<VarId> opInputs,
                                                  Map<VarId, org.nd4j.autodiff.samediff.config.SDValue> nodeValueOutputs,
                                                  FrameIter frameIter) {
        System.out.println("\n⚙️ GENERIC CONTROL FLOW FAILURE ANALYSIS:");

        String[] argNames = op.argNames();
        if (argNames != null) {
            System.out.printf("  📥 Expected Inputs (%d): %s%n", argNames.length, java.util.Arrays.toString(argNames));

            for (String argName : argNames) {
                VarId argVarId = frameIter.toVarId(argName);
                org.nd4j.autodiff.samediff.config.SDValue argValue = nodeValueOutputs.get(argVarId);
                System.out.printf("    • %s: %s%n", argName, argValue != null ? "✅ Available" : "❌ Missing");
            }
        }

        String[] outputNames = op.outputVariablesNames();
        if (outputNames != null) {
            System.out.printf("  📤 Expected Outputs (%d): %s%n", outputNames.length, java.util.Arrays.toString(outputNames));
        }
    }

    /**
     * Analyze frame state at time of failure
     */
    private void analyzeFrameStateAtFailure(String frame, int iteration,
                                            Map<VarId, org.nd4j.autodiff.samediff.config.SDValue> nodeValueOutputs) {
        System.out.println("\n🏗️ FRAME STATE AT FAILURE:");

        // Count variables in current frame
        int currentFrameVars = 0;
        int currentIterVars = 0;

        for (VarId varId : nodeValueOutputs.keySet()) {
            if (varId.getFrame().equals(frame)) {
                currentFrameVars++;
                if (varId.getIteration() == iteration) {
                    currentIterVars++;
                }
            }
        }

        System.out.printf("  📊 Variables in frame '%s': %d total%n", frame, currentFrameVars);
        System.out.printf("  📊 Variables at iteration %d: %d%n", iteration, currentIterVars);

        // Show a few key variables if they exist
        List<String> keyVarNames = java.util.Arrays.asList("1492", "pooler", "merge", "cond", "switch");
        for (String keyVar : keyVarNames) {
            VarId keyVarId = new VarId(keyVar, frame, iteration, null);
            if (nodeValueOutputs.containsKey(keyVarId)) {
                System.out.printf("  🔍 Key variable '%s' present in frame%n", keyVar);
            }
        }
    }

    /**
     * Analyze control flow operation inputs
     */
    private void analyzeControlFlowInputs(Set<VarId> opInputs, Set<VarId> allIterInputs, Set<String> constAndPhInputs,
                                          Map<VarId, org.nd4j.autodiff.samediff.config.SDValue> nodeValueOutputs,
                                          FrameIter frameIter) {
        System.out.println("\n📥 INPUT ANALYSIS:");

        if (opInputs != null && !opInputs.isEmpty()) {
            System.out.printf("  🔗 Operation Inputs (%d):%n", opInputs.size());
            for (VarId vid : opInputs) {
                org.nd4j.autodiff.samediff.config.SDValue value = nodeValueOutputs.get(vid);
                System.out.printf("    • %s [%s:%d]: %s%n",
                        vid.getVariable(), vid.getFrame(), vid.getIteration(),
                        value != null ? "✅ Available" : "❌ Missing");
            }
        }

        if (allIterInputs != null && !allIterInputs.isEmpty()) {
            System.out.printf("  🔄 Iteration Inputs (%d):%n", allIterInputs.size());
            for (VarId vid : allIterInputs) {
                org.nd4j.autodiff.samediff.config.SDValue value = nodeValueOutputs.get(vid);
                System.out.printf("    • %s [%s:%d]: %s%n",
                        vid.getVariable(), vid.getFrame(), vid.getIteration(),
                        value != null ? "✅ Available" : "❌ Missing");
            }
        }

        if (constAndPhInputs != null && !constAndPhInputs.isEmpty()) {
            System.out.printf("  📋 Constants/Placeholders (%d): %s%n",
                    constAndPhInputs.size(), String.join(", ", constAndPhInputs));
        }
    }

    /**
     * Analyze loop termination context if available
     */
    private void analyzeLoopTerminationContext(String frame, int iteration, String failedOp, Exception exception) {
        System.out.println("\n🔄 LOOP TERMINATION CONTEXT:");

        LoopInfo loopInfo = loopAnalyzer.getActiveLoops().get(frame);
        if (loopInfo != null) {
            System.out.printf("  📊 Loop Status: %s%n", loopInfo.getStatus());
            System.out.printf("  📊 Current Iteration: %d%n", loopInfo.getCurrentIteration());
            System.out.printf("  📊 Max Iterations Observed: %d%n", loopInfo.getMaxIterationsObserved());
            System.out.printf("  📊 Early Termination Detected: %s%n", loopInfo.isEarlyTerminationDetected());
        }

        List<LoopTerminationEvent> events = loopAnalyzer.getTerminationHistory().get(frame);
        if (events != null && !events.isEmpty()) {
            System.out.printf("  📊 Recent Termination Events (%d):%n", events.size());
            events.stream().limit(3).forEach(event -> {
                System.out.printf("    • Iter %d: %s - %s%n",
                        event.getIteration(),
                        event.getTerminationType(),
                        event.getTerminationReason());
            });
        }
    }

    /**
     * Provide recovery suggestions based on the failure
     */
    private void provideControlFlowRecoverySuggestions(DifferentialFunction op, String frame, int iteration, Exception exception) {
        System.out.println("\n💡 RECOVERY SUGGESTIONS:");

        String opType = op.getClass().getSimpleName();

        if (opType.contains("Switch")) {
            System.out.println("  🔀 Switch Operation Recovery:");
            System.out.println("    • Verify predicate input is scalar boolean");
            System.out.println("    • Check that data input is available");
            System.out.println("    • Ensure predicate comes from a valid condition operation");

        } else if (opType.contains("Merge")) {
            System.out.println("  🔗 Merge Operation Recovery:");
            System.out.println("    • Verify at least one input is available");
            System.out.println("    • Check input sources are from correct iterations");
            System.out.println("    • Validate that input types match");

        } else if (opType.contains("Enter")) {
            System.out.println("  📥 Enter Operation Recovery:");
            System.out.println("    • Check input variable is available in source frame");
            System.out.println("    • Verify target frame name is correct");
            System.out.println("    • Ensure frame hierarchy is properly established");

        } else if (opType.contains("Exit")) {
            System.out.println("  📤 Exit Operation Recovery:");
            System.out.println("    • Verify input is available in current frame");
            System.out.println("    • Check parent frame exists and is valid");
            System.out.println("    • Ensure proper loop termination conditions");

        } else if (opType.contains("NextIteration")) {
            System.out.println("  🔄 NextIteration Recovery:");
            System.out.println("    • Verify input is from previous iteration");
            System.out.println("    • Check frame consistency");
            System.out.println("    • Validate iteration sequencing");

        } else if (opType.contains("LoopCond")) {
            System.out.println("  🔁 LoopCondition Recovery:");
            System.out.println("    • Verify condition input is scalar boolean");
            System.out.println("    • Check condition logic is correct");
            System.out.println("    • Validate termination behavior");
        }

        System.out.println("\n  🔧 General Debugging Steps:");
        System.out.println("    • Enable detailed logging for control flow operations");
        System.out.println("    • Check execution order and dependencies");
        System.out.println("    • Validate input shapes and data types");
        System.out.println("    • Review loop structure and termination conditions");

        if (frame.equals("outputs") && iteration == 0) {
            System.out.println("\n  ⚠️  SPECIFIC TO 'outputs' FRAME:");
            System.out.println("    • This appears to be a dynamic sequence processing loop");
            System.out.println("    • Check sequence length initialization");
            System.out.println("    • Verify loop bounds are > 0");
            System.out.println("    • Consider using static execution if sequence length is known");
        }
    }

// Helper methods

    private String formatPredicateValue(org.nd4j.linalg.api.ndarray.INDArray predicate) {
        if (predicate.isScalar()) {
            if (predicate.dataType() == org.nd4j.linalg.api.buffer.DataType.BOOL) {
                return predicate.getDouble(0) != 0.0 ? "TRUE" : "FALSE";
            } else {
                return String.valueOf(predicate.getDouble(0));
            }
        } else {
            return "Non-scalar: " + java.util.Arrays.toString(predicate.shape());
        }
    }

    private void analyzeInputsAcrossIterations(String input1, String input2, FrameIter frameIter,
                                               Map<VarId, org.nd4j.autodiff.samediff.config.SDValue> nodeValueOutputs) {
        System.out.println("  🔍 CHECKING INPUTS ACROSS ITERATIONS:");

        for (int iter = 0; iter <= frameIter.getIteration() + 1; iter++) {
            VarId vid1 = new VarId(input1, frameIter.getFrame(), iter, frameIter.getParentFrame());
            VarId vid2 = new VarId(input2, frameIter.getFrame(), iter, frameIter.getParentFrame());

            boolean has1 = nodeValueOutputs.containsKey(vid1);
            boolean has2 = nodeValueOutputs.containsKey(vid2);

            if (has1 || has2) {
                System.out.printf("    Iteration %d: %s=%s, %s=%s%n",
                        iter, input1, has1 ? "✅" : "❌", input2, has2 ? "✅" : "❌");
            }
        }
    }

    private void searchVariableAcrossFrames(String varName, Map<VarId, org.nd4j.autodiff.samediff.config.SDValue> nodeValueOutputs) {
        System.out.println("  🔍 SEARCHING ACROSS ALL FRAMES:");

        Set<String> foundFrames = new HashSet<>();
        for (VarId varId : nodeValueOutputs.keySet()) {
            if (varId.getVariable().equals(varName)) {
                foundFrames.add(String.format("%s:%d", varId.getFrame(), varId.getIteration()));
            }
        }

        if (foundFrames.isEmpty()) {
            System.out.println("    ❌ Variable not found in any frame");
        } else {
            System.out.printf("    ✅ Found in: %s%n", String.join(", ", foundFrames));
        }
    }


}