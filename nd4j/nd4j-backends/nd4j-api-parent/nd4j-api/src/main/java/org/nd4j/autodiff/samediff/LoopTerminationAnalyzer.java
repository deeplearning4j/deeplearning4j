package org.nd4j.autodiff.samediff;

import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.config.SDValue;
import org.nd4j.autodiff.samediff.internal.FrameIter;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.internal.VarId;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.controlflow.compat.*;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * Loop Termination Analysis System
 *
 * Provides comprehensive tracking and analysis of loop termination behavior,
 * focusing on early termination detection and root cause analysis.
 */
@Slf4j
@Data
public class LoopTerminationAnalyzer {

    // Core tracking structures
    private final Map<String, LoopInfo> activeLoops = new ConcurrentHashMap<>();
    private final Map<String, List<LoopTerminationEvent>> terminationHistory = new ConcurrentHashMap<>();
    private final Map<String, LoopIterationTrace> iterationTraces = new ConcurrentHashMap<>();
    private final SameDiff sameDiff;
    private final Map<VarId, SDValue> nodeValueOutputs;

    // Configuration
    private boolean enableDetailedTracing = true;
    private boolean enableValueCapture = true;
    private int maxIterationsToTrace = 1000;
    private boolean enableEarlyTerminationDetection = true;


    // Configuration constants
    private int maxElementsToDisplay = 5;
    private int maxPrecisionDigits = 4;
    private boolean enableValueStatistics = true;

    public LoopTerminationAnalyzer(SameDiff sameDiff, Map<VarId, SDValue> nodeValueOutputs) {
        this.sameDiff = sameDiff;
        this.nodeValueOutputs = nodeValueOutputs;
    }




    /**
     * Format an SDValue for display
     */
    private String formatSDValue(SDValue value, String varName, int iteration) {
        if (value == null) {
            return "null";
        }

        StringBuilder formatted = new StringBuilder();
        formatted.append(String.format("SDValue(type=%s", value.getSdValueType()));

        switch (value.getSdValueType()) {
            case TENSOR:
                INDArray tensor = value.getTensorValue();
                if (tensor != null) {
                    formatted.append(", ").append(formatArrayValue(tensor, varName + "_iter" + iteration));
                } else {
                    formatted.append(", tensor=null");
                }
                break;
            case LIST:
                List<INDArray> list = value.getListValue();
                if (list != null) {
                    formatted.append(String.format(", list_size=%d", list.size()));
                    if (!list.isEmpty()) {
                        formatted.append(", first_element=");
                        if (list.get(0) != null) {
                            formatted.append(formatArrayValue(list.get(0), varName + "_list0"));
                        } else {
                            formatted.append("null");
                        }
                        if (list.size() > 1) {
                            formatted.append(", last_element=");
                            if (list.get(list.size() - 1) != null) {
                                formatted.append(formatArrayValue(list.get(list.size() - 1), varName + "_listN"));
                            } else {
                                formatted.append("null");
                            }
                        }
                    }
                } else {
                    formatted.append(", list=null");
                }
                break;
            case DICT:
                Map<String, INDArray> dict = value.getDictValue();
                if (dict != null) {
                    formatted.append(String.format(", dict_size=%d", dict.size()));
                    if (!dict.isEmpty()) {
                        formatted.append(", keys=[").append(String.join(", ", dict.keySet())).append("]");
                    }
                } else {
                    formatted.append(", dict=null");
                }
                break;
        }

        formatted.append(")");
        return formatted.toString();
    }

    /**
     * Format an INDArray value for display with configurable detail level
     */
    private String formatArrayValue(INDArray array, String context) {
        if (array == null) {
            return "null";
        }

        StringBuilder formatted = new StringBuilder();
        formatted.append(String.format("shape=%s, dtype=%s",
                Arrays.toString(array.shape()), array.dataType()));

        if (array.isScalar()) {
            formatted.append(String.format(", value=%s", formatNumber(array.getNumber())));
        } else if (array.length() <= maxElementsToDisplay) {
            // Show all elements for small arrays
            formatted.append(String.format(", values=%s", Arrays.toString(array.toDoubleVector())));
        } else {
            // Show preview for large arrays
            long[] shape = array.shape();
            if (shape.length == 1) {
                // 1D array preview
                formatted.append(String.format(", values=[%s, ..., %s] (length=%d)",
                        formatNumber(array.getDouble(0)),
                        formatNumber(array.getDouble(array.length() - 1)),
                        array.length()));
            } else if (shape.length == 2) {
                // 2D array preview
                formatted.append(String.format(", values=[[%s, ...], ...] (%dx%d)",
                        formatNumber(array.getDouble(0, 0)), shape[0], shape[1]));
            } else {
                // Multi-dimensional preview
                formatted.append(String.format(", values=[...] (total_elements=%d)", array.length()));
            }
        }

        // Add statistics for numeric arrays
        if (enableValueStatistics && array.dataType().isNumerical() && array.length() > 1) {
            try {
                double mean = array.meanNumber().doubleValue();
                double std = array.stdNumber().doubleValue();
                double min = array.minNumber().doubleValue();
                double max = array.maxNumber().doubleValue();
                formatted.append(String.format(", stats=[mean=%s, std=%s, min=%s, max=%s]",
                        formatNumber(mean), formatNumber(std), formatNumber(min), formatNumber(max)));
            } catch (Exception e) {
                // Skip statistics if calculation fails
            }
        }

        return formatted.toString();
    }

    /**
     * Format variable value for readable output
     */
    private String formatVariableValue(Object value, String varName) {
        if (value == null) {
            return "null";
        }

        if (value instanceof INDArray) {
            return formatArrayValue((INDArray) value, varName);
        } else if (value instanceof SDValue) {
            return formatSDValue((SDValue) value, varName, 0);
        } else if (value instanceof Number) {
            return formatNumber((Number) value);
        } else {
            return value.toString();
        }
    }

    /**
     * Format a number with appropriate precision
     */
    private String formatNumber(Number number) {
        if (number == null) return "null";

        double val = number.doubleValue();
        if (Math.abs(val) < 1e-6 && val != 0.0) {
            return String.format("%.2e", val);
        } else if (Math.abs(val) > 1e6) {
            return String.format("%.2e", val);
        } else {
            return String.format("%." + maxPrecisionDigits + "f", val);
        }
    }


    /**
     * Main entry points for loop tracking
     */

    /**
     * Called when a loop frame is entered
     */
    public void onLoopFrameEnter(String frameName, String enterOperation, FrameIter frameIter) {
        if (!activeLoops.containsKey(frameName)) {
            LoopInfo loopInfo = new LoopInfo();
            loopInfo.setFrameName(frameName);
            loopInfo.setStartTime(System.currentTimeMillis());
            loopInfo.setCurrentIteration(frameIter.getIteration());

            // Discover loop-related operations
            discoverLoopOperations(frameName, loopInfo);

            activeLoops.put(frameName, loopInfo);
            iterationTraces.put(frameName, new LoopIterationTrace());
            iterationTraces.get(frameName).setFrameName(frameName);

            if (enableDetailedTracing) {
                log.info("LOOP_ANALYSIS: Started tracking loop frame '{}' at iteration {}",
                        frameName, frameIter.getIteration());
                log.info("LOOP_ANALYSIS: Discovered operations - LoopCond: {}, Exit: {}, Switch: {}, NextIter: {}",
                        loopInfo.getLoopCondOperation(), loopInfo.getExitOperations().size(),
                        loopInfo.getSwitchOperations().size(), loopInfo.getNextIterationOperations().size());
            }
        }
    }

    /**
     * Called when a loop iteration begins
     */
    /**
     * Called when a loop iteration begins
     */
    public void onLoopIteration(String frameName, int iteration, Map<String, Object> variableValues) {
        LoopInfo loopInfo = activeLoops.get(frameName);
        if (loopInfo != null) {
            loopInfo.setCurrentIteration(iteration);
            loopInfo.setMaxIterationsObserved(Math.max(loopInfo.getMaxIterationsObserved(), iteration));

            // Create iteration snapshot
            IterationSnapshot snapshot = new IterationSnapshot();
            snapshot.setIteration(iteration);
            snapshot.setTimestamp(System.currentTimeMillis());

            if (enableValueCapture && variableValues != null) {
                snapshot.setVariableValues(new HashMap<>(variableValues));

                // Capture variable shapes and add detailed value information to debugInfo
                for (Map.Entry<String, Object> entry : variableValues.entrySet()) {
                    String varName = entry.getKey();
                    Object value = entry.getValue();
                    if (value instanceof INDArray) {
                        INDArray arr = (INDArray) value;
                        snapshot.getVariableShapes().put(varName, Arrays.toString(arr.shape()));

                        // Store detailed value formatting in debugInfo
                        String valueString = formatArrayValue(arr, "iteration_" + iteration);
                        snapshot.getDebugInfo().put(varName + "_detailed", valueString);
                    } else if (value instanceof SDValue) {
                        SDValue sdValue = (SDValue) value;
                        String valueString = formatSDValue(sdValue, varName, iteration);
                        snapshot.getDebugInfo().put(varName + "_detailed", valueString);
                    } else {
                        snapshot.getDebugInfo().put(varName + "_detailed", formatVariableValue(value, varName));
                    }
                }
            }

            LoopIterationTrace trace = iterationTraces.get(frameName);
            if (trace != null) {
                trace.getIterations().add(snapshot);

                // Track variable evolution
                if (enableValueCapture && variableValues != null) {
                    for (Map.Entry<String, Object> entry : variableValues.entrySet()) {
                        trace.getVariableEvolution().computeIfAbsent(entry.getKey(), k -> new ArrayList<>())
                                .add(entry.getValue());
                    }
                }
            }

            // Check for early termination prediction
            if (enableEarlyTerminationDetection && iteration > 0) {
                checkForEarlyTerminationPrediction(frameName, loopInfo, iteration);
            }

            if (enableDetailedTracing) {
                String valuesSummary = "";
                if (variableValues != null) {
                    valuesSummary = generateIterationValuesSummary(variableValues);
                }
                log.debug("LOOP_ANALYSIS: Iteration {} in frame '{}' - {} variables captured | Values: {}",
                        iteration, frameName, variableValues != null ? variableValues.size() : 0, valuesSummary);
            }
        }
    }

    /**
     * Called when a loop condition is evaluated
     */
    public void onLoopConditionEvaluation(String frameName, String conditionOp, Object conditionValue,
                                          Map<String, Object> inputValues, int iteration) {
        LoopInfo loopInfo = activeLoops.get(frameName);
        if (loopInfo != null) {
            ConditionEvaluation evaluation = new ConditionEvaluation();
            evaluation.setIteration(iteration);
            evaluation.setConditionOperation(conditionOp);
            evaluation.setConditionValue(conditionValue);
            evaluation.setInputValues(inputValues != null ? new HashMap<>(inputValues) : new HashMap<>());
            evaluation.setTimestamp(System.currentTimeMillis());
            evaluation.setEvaluationContext(String.format("Frame: %s, Iteration: %d", frameName, iteration));

            // Determine if this evaluation will trigger termination
            boolean terminationTriggered = false;
            if (conditionValue instanceof Boolean) {
                terminationTriggered = !(Boolean) conditionValue;
            } else if (conditionValue instanceof INDArray) {
                INDArray arr = (INDArray) conditionValue;
                if (arr.isScalar()) {
                    terminationTriggered = arr.getDouble(0) == 0.0;
                }
            }
            evaluation.setTerminationTriggered(terminationTriggered);

            LoopIterationTrace trace = iterationTraces.get(frameName);
            if (trace != null) {
                trace.getConditionEvaluations().add(evaluation);

                // Update current iteration snapshot with condition details
                List<IterationSnapshot> iterations = trace.getIterations();
                if (!iterations.isEmpty()) {
                    IterationSnapshot currentSnapshot = iterations.get(iterations.size() - 1);
                    currentSnapshot.setConditionEvaluated(true);
                    currentSnapshot.setConditionValue(conditionValue);
                    currentSnapshot.setConditionSource(conditionOp);

                    // Add formatted values to debugInfo
                    String formattedConditionValue = formatVariableValue(conditionValue, "condition");
                    currentSnapshot.getDebugInfo().put("condition_formatted", formattedConditionValue);

                    if (inputValues != null && !inputValues.isEmpty()) {
                        for (Map.Entry<String, Object> entry : inputValues.entrySet()) {
                            String formattedInput = formatVariableValue(entry.getValue(), entry.getKey());
                            currentSnapshot.getDebugInfo().put("input_" + entry.getKey() + "_formatted", formattedInput);
                        }
                    }
                }
            }

            if (enableDetailedTracing) {
                String inputsSummary = "";
                if (inputValues != null && !inputValues.isEmpty()) {
                    inputsSummary = generateValuesSummary(inputValues);
                }
                String formattedConditionValue = formatVariableValue(conditionValue, "condition");
                log.debug("LOOP_ANALYSIS: Condition evaluation in frame '{}' iteration {}: {} = {} (termination: {}) | Inputs: {}",
                        frameName, iteration, conditionOp, formattedConditionValue, terminationTriggered, inputsSummary);
            }

            // If termination is triggered, record it with enhanced details
            if (terminationTriggered) {
                String enhancedReason = String.format("Loop condition '%s' evaluated to %s with inputs: %s",
                        conditionOp, formatVariableValue(conditionValue, "condition"),
                        inputValues != null ? generateValuesSummary(inputValues) : "none");
                recordLoopTermination(frameName, iteration, TerminationType.CONDITION_FALSE,
                        conditionOp, conditionValue, enhancedReason);
            }
        }
    }

    /**
     * Called when a switch operation is executed in a loop
     */
    public void onSwitchOperation(String frameName, String switchOp, Object predicateValue,
                                  String branchTaken, int iteration) {
        LoopInfo loopInfo = activeLoops.get(frameName);
        if (loopInfo != null) {
            // Check if this switch operation affects loop termination
            boolean affectsTermination = checkSwitchAffectsTermination(frameName, switchOp, branchTaken);

            if (enableDetailedTracing) {
                log.debug("LOOP_ANALYSIS: Switch operation '{}' in frame '{}' iteration {}: predicate={}, branch={}, affects_termination={}",
                        switchOp, frameName, iteration, predicateValue, branchTaken, affectsTermination);
            }

            if (affectsTermination) {
                String reason = String.format("Switch operation '%s' took '%s' branch with predicate value: %s",
                        switchOp, branchTaken, predicateValue);
                recordLoopTermination(frameName, iteration, TerminationType.SWITCH_TERMINATION,
                        switchOp, predicateValue, reason);
            }
        }
    }

    /**
     * Called when an exit operation is executed
     */
    public void onExitOperation(String frameName, String exitOp, Object exitValue, int iteration) {
        LoopInfo loopInfo = activeLoops.get(frameName);
        if (loopInfo != null) {
            String reason = String.format("Exit operation '%s' executed with value: %s", exitOp, exitValue);
            recordLoopTermination(frameName, iteration, TerminationType.CONDITION_TRUE_EXIT,
                    exitOp, exitValue, reason);

            if (enableDetailedTracing) {
                log.info("LOOP_ANALYSIS: Exit operation '{}' in frame '{}' iteration {}: value={}",
                        exitOp, frameName, iteration, exitValue);
            }
        }
    }

    /**
     * Called when a loop encounters an error
     */
    public void onLoopError(String frameName, int iteration, String errorOperation,
                            Exception error, Map<String, Object> errorContext) {
        LoopInfo loopInfo = activeLoops.get(frameName);
        if (loopInfo != null) {
            String reason = String.format("Error in operation '%s': %s", errorOperation, error.getMessage());
            recordLoopTermination(frameName, iteration, TerminationType.ERROR_TERMINATION,
                    errorOperation, error, reason);

            // Add error context to the termination event
            LoopTerminationEvent event = getLatestTerminationEvent(frameName);
            if (event != null && errorContext != null) {
                event.getContextData().putAll(errorContext);
            }

            if (enableDetailedTracing) {
                log.error("LOOP_ANALYSIS: Error in frame '{}' iteration {} operation '{}': {}",
                        frameName, iteration, errorOperation, error.getMessage());
            }
        }
    }

    /**
     * Core analysis methods
     */

    /**
     * Discover loop-related operations in a frame
     */
    private void discoverLoopOperations(String frameName, LoopInfo loopInfo) {
        for (Map.Entry<String, SameDiffOp> entry : sameDiff.getOps().entrySet()) {
            String opName = entry.getKey();
            SameDiffOp op = entry.getValue();
            DifferentialFunction func = op.getOp();

            // Check if this operation is associated with the loop frame
            if (isOperationInFrame(opName, frameName)) {
                if (func instanceof LoopCond) {
                    loopInfo.setLoopCondOperation(opName);
                } else if (func instanceof Exit) {
                    loopInfo.getExitOperations().add(opName);
                } else if (func instanceof Switch) {
                    loopInfo.getSwitchOperations().add(opName);
                } else if (func instanceof NextIteration) {
                    loopInfo.getNextIterationOperations().add(opName);
                }
            }
        }

        // Discover loop variables
        loopInfo.setLoopVariables(discoverLoopVariables(frameName));
    }

    /**
     * Check if an operation is associated with a specific frame
     */
    private boolean isOperationInFrame(String opName, String frameName) {
        // FIXED: Proper frame detection logic

        // 1. Check if operation name contains frame reference
        if (opName.startsWith(frameName + "/")) {
            return true;
        }

        // 2. For "outputs" frame specifically, check for control flow patterns
        if (frameName.equals("outputs")) {
            if (opName.contains("outputs/merge") ||
                    opName.contains("outputs/cond/") ||
                    opName.contains("outputs/switch") ||
                    opName.contains("outputs/enter") ||
                    opName.contains("outputs/exit") ||
                    opName.contains("outputs/next_iteration")) {
                return true;
            }
        }

        // 3. Check operation's inputs/outputs for frame-specific variables
        SameDiffOp op = sameDiff.getOps().get(opName);
        if (op != null) {
            // Check inputs
            List<String> inputs = op.getInputsToOp();
            if (inputs != null) {
                for (String input : inputs) {
                    if (input.startsWith(frameName + "/")) {
                        return true;
                    }
                }
            }

            // Check outputs
            List<String> outputs = op.getOutputsOfOp();
            if (outputs != null) {
                for (String output : outputs) {
                    if (output.startsWith(frameName + "/")) {
                        return true;
                    }
                }
            }
        }

        return false;
    }

    // ADD TO LoopTerminationAnalyzer:
    public void detectLoopFromExecutionPattern(String frameName, List<String> recentOperations) {
        // Look for loop patterns in recent operations
        boolean hasLoopPattern = false;

        // Check for control flow pattern: merge -> condition -> switch sequence
        boolean hasMerge = recentOperations.stream().anyMatch(op -> op.contains("merge"));
        boolean hasCondition = recentOperations.stream().anyMatch(op -> op.contains("cond/"));
        boolean hasSwitch = recentOperations.stream().anyMatch(op -> op.contains("switch"));

        if (hasMerge && hasCondition && hasSwitch) {
            hasLoopPattern = true;
        }

        // Check for specific loop frame markers
        if (frameName.equals("outputs") && hasLoopPattern) {
            if (!activeLoops.containsKey(frameName)) {
                log.info("LOOP_DETECTION: Auto-detected loop in frame '{}' from execution pattern", frameName);

                // Create synthetic FrameIter for the detected loop
                FrameIter syntheticFrame = new FrameIter(frameName, 0, new FrameIter("main", 0, null));
                onLoopFrameEnter(frameName, "auto_detected_loop", syntheticFrame);
            }
        }
    }


    /**
     * Discover variables that are part of the loop
     */
    private List<String> discoverLoopVariables(String frameName) {
        List<String> loopVariables = new ArrayList<>();

        // Look for variables that are updated by NextIteration operations
        for (Map.Entry<String, SameDiffOp> entry : sameDiff.getOps().entrySet()) {
            SameDiffOp op = entry.getValue();
            if (op.getOp() instanceof NextIteration) {
                List<String> inputs = op.getInputsToOp();
                if (inputs != null) {
                    loopVariables.addAll(inputs);
                }
            }
        }

        return loopVariables.stream().distinct().collect(Collectors.toList());
    }

    /**
     * Check if a switch operation affects loop termination
     */
    private boolean checkSwitchAffectsTermination(String frameName, String switchOp, String branchTaken) {
        LoopInfo loopInfo = activeLoops.get(frameName);
        if (loopInfo == null) return false;

        // Check if the switch operation feeds into any exit operations
        SameDiffOp op = sameDiff.getOps().get(switchOp);
        if (op != null) {
            List<String> outputs = op.getOutputsOfOp();
            if (outputs != null) {
                for (String output : outputs) {
                    if (variableFeedsIntoExitOps(output, loopInfo.getExitOperations())) {
                        return true;
                    }
                }
            }
        }

        return false;
    }

    /**
     * Check if a variable feeds into any exit operations
     */
    private boolean variableFeedsIntoExitOps(String variable, List<String> exitOps) {
        for (String exitOp : exitOps) {
            SameDiffOp op = sameDiff.getOps().get(exitOp);
            if (op != null) {
                List<String> inputs = op.getInputsToOp();
                if (inputs != null && inputs.contains(variable)) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * Early termination prediction and detection
     */
    private void checkForEarlyTerminationPrediction(String frameName, LoopInfo loopInfo, int iteration) {
        LoopIterationTrace trace = iterationTraces.get(frameName);
        if (trace == null || trace.getIterations().size() < 3) return;

        // Analyze recent condition evaluations for patterns
        List<ConditionEvaluation> recentEvaluations = trace.getConditionEvaluations().stream()
                .filter(eval -> eval.getIteration() >= iteration - 5 && eval.getIteration() <= iteration)
                .collect(Collectors.toList());

        if (recentEvaluations.size() >= 2) {
            // Check for value convergence pattern
            TerminationPrediction prediction = analyzeConvergencePattern(recentEvaluations, iteration);
            if (prediction != null) {
                loopInfo.getTerminationPredictions().add(prediction);

                if (enableDetailedTracing) {
                    log.info("LOOP_ANALYSIS: Early termination prediction for frame '{}': {} (confidence: {:.2f})",
                            frameName, prediction.getReasoning(), prediction.getConfidence());
                }
            }
        }

        // Check for variable evolution patterns
        checkVariableEvolutionPatterns(frameName, loopInfo, iteration);
    }

    /**
     * Analyze convergence patterns in condition evaluations
     */
    private TerminationPrediction analyzeConvergencePattern(List<ConditionEvaluation> evaluations, int currentIteration) {
        if (evaluations.size() < 2) return null;

        // Check for numeric convergence
        List<Double> numericValues = new ArrayList<>();
        for (ConditionEvaluation eval : evaluations) {
            if (eval.getConditionValue() instanceof Number) {
                numericValues.add(((Number) eval.getConditionValue()).doubleValue());
            } else if (eval.getConditionValue() instanceof INDArray) {
                INDArray arr = (INDArray) eval.getConditionValue();
                if (arr.isScalar()) {
                    numericValues.add(arr.getDouble(0));
                }
            }
        }

        if (numericValues.size() >= 2) {
            double convergenceRate = calculateConvergenceRate(numericValues);
            if (convergenceRate > 0.1) { // Threshold for significant convergence
                TerminationPrediction prediction = new TerminationPrediction();
                prediction.setPredictedAtIteration(currentIteration);
                prediction.setPredictedTerminationIteration(currentIteration + (int)(1.0 / convergenceRate));
                prediction.setConfidence(Math.min(0.95, convergenceRate * 2));
                prediction.setPredictionMethod("convergence_analysis");
                prediction.setReasoning(String.format("Convergence rate: %.4f suggests termination in ~%d iterations",
                        convergenceRate, (int)(1.0 / convergenceRate)));

                // Store evidence
                prediction.getEvidenceData().put("convergenceRate", convergenceRate);
                prediction.getEvidenceData().put("recentValues", numericValues);

                return prediction;
            }
        }

        return null;
    }

    /**
     * Calculate convergence rate from a series of values
     */
    private double calculateConvergenceRate(List<Double> values) {
        if (values.size() < 2) return 0.0;

        double totalChange = 0.0;
        for (int i = 1; i < values.size(); i++) {
            totalChange += Math.abs(values.get(i) - values.get(i-1));
        }

        return totalChange / (values.size() - 1);
    }

    /**
     * Check for patterns in variable evolution
     */
    private void checkVariableEvolutionPatterns(String frameName, LoopInfo loopInfo, int iteration) {
        LoopIterationTrace trace = iterationTraces.get(frameName);
        if (trace == null) return;

        // Analyze each loop variable for patterns
        for (String varName : loopInfo.getLoopVariables()) {
            List<Object> evolution = trace.getVariableEvolution().get(varName);
            if (evolution != null && evolution.size() >= 3) {
                VariablePattern pattern = analyzeVariablePattern(evolution);
                if (pattern != null && pattern.isTerminationIndicator()) {
                    loopInfo.setEarlyTerminationDetected(true);

                    if (enableDetailedTracing) {
                        log.info("LOOP_ANALYSIS: Variable '{}' pattern indicates early termination: {}",
                                varName, pattern.getDescription());
                    }
                }
            }
        }
    }

    /**
     * Analyze patterns in variable evolution
     */
    private VariablePattern analyzeVariablePattern(List<Object> evolution) {
        // Check for monotonic decrease towards zero
        if (evolution.size() >= 3) {
            List<Double> numericValues = new ArrayList<>();
            for (Object value : evolution) {
                if (value instanceof Number) {
                    numericValues.add(((Number) value).doubleValue());
                } else if (value instanceof INDArray) {
                    INDArray arr = (INDArray) value;
                    if (arr.isScalar()) {
                        numericValues.add(arr.getDouble(0));
                    }
                }
            }

            if (numericValues.size() >= 3) {
                boolean monotonic = true;
                for (int i = 1; i < numericValues.size(); i++) {
                    if (numericValues.get(i) >= numericValues.get(i-1)) {
                        monotonic = false;
                        break;
                    }
                }

                if (monotonic && numericValues.get(numericValues.size() - 1) < 0.1) {
                    return new VariablePattern("monotonic_decrease_to_zero", true,
                            "Variable shows monotonic decrease approaching zero");
                }
            }
        }

        return null;
    }

    /**
     * Record a loop termination event
     */
    private void recordLoopTermination(String frameName, int iteration, TerminationType terminationType,
                                       String triggerOperation, Object terminationValue, String reason) {
        LoopInfo loopInfo = activeLoops.get(frameName);
        if (loopInfo == null) return;

        LoopTerminationEvent event = new LoopTerminationEvent();
        event.setFrameName(frameName);
        event.setIteration(iteration);
        event.setTimestamp(System.currentTimeMillis());
        event.setTerminationType(terminationType);
        event.setTriggerOperation(triggerOperation);
        event.setTerminationValue(terminationValue);
        event.setTerminationReason(reason);

        // Determine if this was an early termination
        boolean isEarlyTermination = determineIfEarlyTermination(loopInfo, iteration);
        event.setWasEarlyTermination(isEarlyTermination);

        if (isEarlyTermination) {
            event.setEarlyTerminationCause(analyzeEarlyTerminationCause(loopInfo, iteration, terminationType));
        }

        // Capture current loop state
        event.setLoopStateAtTermination(captureLoopState(frameName, iteration));

        // Store the event
        terminationHistory.computeIfAbsent(frameName, k -> new ArrayList<>()).add(event);

        // Update loop info
        loopInfo.setStatus(mapTerminationTypeToStatus(terminationType));
        loopInfo.setTerminationReason(reason);

        if (enableDetailedTracing) {
            log.info("LOOP_ANALYSIS: Recorded termination for frame '{}': {} at iteration {} (early: {})",
                    frameName, terminationType, iteration, isEarlyTermination);
        }
    }

    /**
     * Determine if a termination was early
     */
    private boolean determineIfEarlyTermination(LoopInfo loopInfo, int iteration) {
        // If we have predictions, check against them
        if (!loopInfo.getTerminationPredictions().isEmpty()) {
            TerminationPrediction bestPrediction = loopInfo.getTerminationPredictions()
                    .stream()
                    .max(Comparator.comparingDouble(TerminationPrediction::getConfidence))
                    .orElse(null);

            if (bestPrediction != null) {
                return iteration < bestPrediction.getPredictedTerminationIteration() - 2;
            }
        }

        // If we have an expected iteration count, check against it
        if (loopInfo.getExpectedIterations() > 0) {
            return iteration < loopInfo.getExpectedIterations() * 0.8;
        }

        // Heuristic: if we've seen significantly more iterations before, this might be early
        if (loopInfo.getMaxIterationsObserved() > iteration * 1.5) {
            return true;
        }

        return false;
    }

    /**
     * Analyze the cause of early termination
     */
    private String analyzeEarlyTerminationCause(LoopInfo loopInfo, int iteration, TerminationType terminationType) {
        return LoopAnalysisHelpers.analyzeEarlyTerminationCause(loopInfo, iteration, terminationType, iterationTraces);
    }

    /**
     * Map termination type to loop status
     */
    private LoopTerminationStatus mapTerminationTypeToStatus(TerminationType terminationType) {
        return LoopAnalysisHelpers.mapTerminationTypeToStatus(terminationType);
    }

    /**
     * Capture current loop state for debugging
     */
    private LoopState captureLoopState(String frameName, int iteration) {
        return LoopAnalysisHelpers.captureLoopState(frameName, iteration, nodeValueOutputs, sameDiff);
    }

    /**
     * Get the latest termination event for a frame
     */
    private LoopTerminationEvent getLatestTerminationEvent(String frameName) {
        return LoopTerminationEventUtils.getLatestTerminationEvent(frameName, terminationHistory,TerminationType.ERROR_TERMINATION);
    }



    /**
     * Generate a generic values summary
     */
    private String generateValuesSummary(Map<String, Object> values) {
        if (values == null || values.isEmpty()) {
            return "none";
        }

        StringBuilder summary = new StringBuilder();
        int count = 0;
        for (Map.Entry<String, Object> entry : values.entrySet()) {
            if (count > 0) summary.append(", ");
            if (count >= 2) { // Limit to first 2 for compactness
                summary.append("... (+").append(values.size() - 2).append(" more)");
                break;
            }

            summary.append(entry.getKey()).append("=").append(formatVariableValue(entry.getValue(), entry.getKey()));
            count++;
        }

        return summary.toString();
    }

    /**
     * Analyze trend in variable evolution
     */
    private String analyzeTrend(List<Object> values, String varName) {
        if (values == null || values.size() < 2) {
            return "";
        }

        // Try to extract numeric trends
        List<Double> numericValues = new ArrayList<>();
        for (Object value : values) {
            if (value instanceof Number) {
                numericValues.add(((Number) value).doubleValue());
            } else if (value instanceof INDArray) {
                INDArray arr = (INDArray) value;
                if (arr.isScalar()) {
                    numericValues.add(arr.getDouble(0));
                } else {
                    // Use mean for non-scalar arrays
                    numericValues.add(arr.meanNumber().doubleValue());
                }
            }
        }

        if (numericValues.size() >= 2) {
            double start = numericValues.get(0);
            double end = numericValues.get(numericValues.size() - 1);
            double change = end - start;
            double percentChange = start != 0 ? (change / start) * 100 : 0;

            StringBuilder trend = new StringBuilder();
            if (Math.abs(change) < 1e-9) {
                trend.append("stable");
            } else if (change > 0) {
                trend.append("increasing");
            } else {
                trend.append("decreasing");
            }

            trend.append(String.format(" (from %s to %s, change=%s",
                    formatNumber(start), formatNumber(end), formatNumber(change)));

            if (Math.abs(percentChange) > 0.01) {
                trend.append(String.format(", %+.2f%%", percentChange));
            }
            trend.append(")");

            return trend.toString();
        }

        return "non-numeric_trend";
    }

    /**
     * Generate evolution summary for a variable
     */
    private String generateEvolutionSummary(String varName, List<Object> evolution) {
        if (evolution == null || evolution.isEmpty()) {
            return "no_evolution";
        }

        StringBuilder summary = new StringBuilder();
        summary.append(String.format("Variable '%s': %d iterations", varName, evolution.size()));

        if (evolution.size() == 1) {
            summary.append(", final_value=").append(formatVariableValue(evolution.get(0), varName));
        } else {
            summary.append(", first_value=").append(formatVariableValue(evolution.get(0), varName));
            summary.append(", final_value=").append(formatVariableValue(evolution.get(evolution.size() - 1), varName));

            String trend = analyzeTrend(evolution, varName);
            if (!trend.isEmpty()) {
                summary.append(", trend=").append(trend);
            }
        }

        return summary.toString();
    }


    /**
     * Generate a comprehensive termination report for a loop with enhanced value details
     */
    public String generateTerminationReport(String frameName) {
        LoopInfo loopInfo = activeLoops.get(frameName);
        List<LoopTerminationEvent> events = terminationHistory.get(frameName);
        LoopIterationTrace trace = iterationTraces.get(frameName);

        if (loopInfo == null) {
            return "No loop information found for frame: " + frameName;
        }

        StringBuilder report = new StringBuilder();
        report.append("=== LOOP TERMINATION ANALYSIS REPORT ===\n");
        report.append("Frame: ").append(frameName).append("\n");
        report.append("Status: ").append(loopInfo.getStatus()).append("\n");
        report.append("Total Iterations: ").append(loopInfo.getMaxIterationsObserved()).append("\n");
        report.append("Duration: ").append(System.currentTimeMillis() - loopInfo.getStartTime()).append("ms\n");

        if (loopInfo.getTerminationReason() != null) {
            report.append("Termination Reason: ").append(loopInfo.getTerminationReason()).append("\n");
        }

        // Add detailed variable evolution analysis
        if (trace != null && !trace.getVariableEvolution().isEmpty()) {
            report.append("\n=== VARIABLE EVOLUTION ANALYSIS ===\n");
            for (Map.Entry<String, List<Object>> evolution : trace.getVariableEvolution().entrySet()) {
                String varName = evolution.getKey();
                List<Object> values = evolution.getValue();

                report.append(String.format("Variable '%s' (%d iterations):\n", varName, values.size()));

                // Show first, middle, and last values
                if (!values.isEmpty()) {
                    report.append("  First value: ").append(formatVariableValue(values.get(0), varName)).append("\n");

                    if (values.size() > 2) {
                        int midIndex = values.size() / 2;
                        report.append("  Middle value: ").append(formatVariableValue(values.get(midIndex), varName)).append("\n");
                    }

                    if (values.size() > 1) {
                        report.append("  Final value: ").append(formatVariableValue(values.get(values.size() - 1), varName)).append("\n");
                    }

                    // Add trend analysis
                    String trendAnalysis = analyzeTrend(values, varName);
                    if (!trendAnalysis.isEmpty()) {
                        report.append("  Trend: ").append(trendAnalysis).append("\n");
                    }
                }
                report.append("\n");
            }
        }

        // Add condition evaluation details
        if (trace != null && !trace.getConditionEvaluations().isEmpty()) {
            report.append("=== CONDITION EVALUATION HISTORY ===\n");
            for (ConditionEvaluation eval : trace.getConditionEvaluations()) {
                String formattedConditionValue = formatVariableValue(eval.getConditionValue(), "condition");
                report.append(String.format("Iteration %d: %s = %s",
                        eval.getIteration(), eval.getConditionOperation(), formattedConditionValue));
                if (eval.isTerminationTriggered()) {
                    report.append(" [TERMINATION TRIGGERED]");
                }
                report.append("\n");

                if (eval.getInputValues() != null && !eval.getInputValues().isEmpty()) {
                    report.append("  Inputs: ");
                    for (Map.Entry<String, Object> input : eval.getInputValues().entrySet()) {
                        String formattedInput = formatVariableValue(input.getValue(), input.getKey());
                        report.append(String.format("%s=%s ", input.getKey(), formattedInput));
                    }
                    report.append("\n");
                }
            }
            report.append("\n");
        }

        // Add termination event details
        if (events != null && !events.isEmpty()) {
            report.append("=== TERMINATION EVENTS ===\n");
            for (LoopTerminationEvent event : events) {
                report.append(String.format("Type: %s, Iteration: %d\n", event.getTerminationType(), event.getIteration()));
                report.append("Reason: ").append(event.getTerminationReason()).append("\n");
                if (event.getTerminationValue() != null) {
                    report.append("Termination Value: ").append(formatVariableValue(event.getTerminationValue(), "termination")).append("\n");
                }
                report.append("\n");
            }
        }

        return report.toString();
    }


    /**
     * Generate a summary of values for iteration display
     */
    private String generateIterationValuesSummary(Map<String, Object> variableValues) {
        if (variableValues == null || variableValues.isEmpty()) {
            return "no_values";
        }

        StringBuilder summary = new StringBuilder();
        int count = 0;
        for (Map.Entry<String, Object> entry : variableValues.entrySet()) {
            if (count > 0) summary.append(", ");
            if (count >= 3) { // Limit summary to first 3 variables
                summary.append("... (+").append(variableValues.size() - 3).append(" more)");
                break;
            }

            String varName = entry.getKey();
            Object value = entry.getValue();
            String valueStr = formatVariableValue(value, varName);

            // Truncate very long value strings
            if (valueStr.length() > 50) {
                valueStr = valueStr.substring(0, 47) + "...";
            }

            summary.append(varName).append("=").append(valueStr);
            count++;
        }

        return summary.toString();
    }


    /**
     * Get summary statistics for all tracked loops
     */
    public Map<String, Object> getLoopStatistics() {
        Map<String, Object> stats = new HashMap<>();

        stats.put("totalLoopsTracked", activeLoops.size());
        stats.put("totalTerminationEvents", terminationHistory.values().stream()
                .mapToInt(List::size).sum());

        // Count by termination type
        Map<TerminationType, Long> terminationCounts = terminationHistory.values().stream()
                .flatMap(List::stream)
                .collect(Collectors.groupingBy(LoopTerminationEvent::getTerminationType, Collectors.counting()));

        stats.put("terminationTypeBreakdown", terminationCounts);

        // Count early terminations
        long earlyTerminations = terminationHistory.values().stream()
                .flatMap(List::stream)
                .filter(LoopTerminationEvent::isWasEarlyTermination)
                .count();

        stats.put("earlyTerminations", earlyTerminations);

        // Average iterations per loop
        double avgIterations = activeLoops.values().stream()
                .mapToInt(LoopInfo::getMaxIterationsObserved)
                .average()
                .orElse(0.0);

        stats.put("averageIterationsPerLoop", avgIterations);

        return stats;
    }

    /**
     * Get all loops that terminated early
     */
    public List<String> getEarlyTerminatedLoops() {
        return terminationHistory.entrySet().stream()
                .filter(entry -> entry.getValue().stream().anyMatch(LoopTerminationEvent::isWasEarlyTermination))
                .map(Map.Entry::getKey)
                .collect(Collectors.toList());
    }

    /**
     * Get active loops
     */
    public Map<String, LoopInfo> getActiveLoops() {
        return activeLoops;
    }

    /**
     * Get termination history
     */
    public Map<String, List<LoopTerminationEvent>> getTerminationHistory() {
        return terminationHistory;
    }

    /**
     * Get iteration traces
     */
    public Map<String, LoopIterationTrace> getIterationTraces() {
        return iterationTraces;
    }

    /**
     * Clear all tracking data
     */
    public void clearAll() {
        activeLoops.clear();
        terminationHistory.clear();
        iterationTraces.clear();
    }

    /**
     * Configuration methods
     */
    public void setEnableDetailedTracing(boolean enable) {
        this.enableDetailedTracing = enable;
    }

    public void setEnableValueCapture(boolean enable) {
        this.enableValueCapture = enable;
    }

    public void setMaxIterationsToTrace(int maxIterations) {
        this.maxIterationsToTrace = maxIterations;
    }

    public void setEnableEarlyTerminationDetection(boolean enable) {
        this.enableEarlyTerminationDetection = enable;
    }
}