package org.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.config.SDValue;
import org.nd4j.autodiff.samediff.config.SDValueType;
import org.nd4j.autodiff.samediff.internal.FrameIter;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.internal.VarId;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Enhanced Loop Termination Error Reporter
 *
 * Provides comprehensive error reporting for loop terminations, including:
 * - Complete variable state capture at termination
 * - Input/output variable tracing
 * - Operation execution history
 * - Frame and iteration context
 * - Variable evolution patterns
 * - Memory and performance metrics
 * - Detailed root cause analysis
 */
@Slf4j
public class LoopTerminationErrorReporter {

    private final SameDiff sameDiff;
    private final Map<VarId, SDValue> nodeValueOutputs;
    private final LoopTerminationAnalyzer loopAnalyzer;
    private final DateTimeFormatter timeFormatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss.SSS");

    // Configuration options
    private boolean includeVariableValues = true;
    private boolean includeVariableShapes = true;
    private boolean includeOperationHistory = true;
    private boolean includeFrameContext = true;
    private boolean includeMemoryMetrics = true;
    private boolean includeVariableEvolution = true;
    private boolean generateVisualizations = true;
    private int maxVariableValueDisplay = 10; // Maximum elements to show for arrays
    private int maxHistoryDepth = 20; // Maximum iterations to include in history

    public LoopTerminationErrorReporter(SameDiff sameDiff,
                                        Map<VarId, SDValue> nodeValueOutputs,
                                        LoopTerminationAnalyzer loopAnalyzer) {
        this.sameDiff = sameDiff;
        this.nodeValueOutputs = nodeValueOutputs;
        this.loopAnalyzer = loopAnalyzer;
    }

    /**
     * Generate a comprehensive error report for a loop termination event
     */
    public LoopTerminationErrorReport generateErrorReport(LoopTerminationEvent event) {
        LoopTerminationErrorReport report = new LoopTerminationErrorReport();

        // Basic event information
        populateBasicInfo(report, event);

        // Variable state analysis
        populateVariableStateAnalysis(report, event);

        // Operation execution analysis
        populateOperationAnalysis(report, event);

        // Frame and iteration context
        populateFrameContext(report, event);

        // Variable evolution analysis
        populateVariableEvolution(report, event);

        // Memory and performance metrics
        populatePerformanceMetrics(report, event);

        // Root cause analysis
        populateRootCauseAnalysis(report, event);

        // Generate visualizations
        if (generateVisualizations) {
            populateVisualizations(report, event);
        }

        return report;
    }

    /**
     * Generate a human-readable error report string
     */
    public String generateErrorReportString(LoopTerminationEvent event) {
        LoopTerminationErrorReport report = generateErrorReport(event);
        return formatErrorReportAsString(report);
    }

    /**
     * Generate error report for multiple related termination events
     */
    public MultiLoopTerminationErrorReport generateMultiLoopErrorReport(List<LoopTerminationEvent> events) {
        MultiLoopTerminationErrorReport multiReport = new MultiLoopTerminationErrorReport();

        // Generate individual reports
        for (LoopTerminationEvent event : events) {
            LoopTerminationErrorReport report = generateErrorReport(event);
            multiReport.getIndividualReports().put(event.getFrameName(), report);
        }

        // Cross-loop analysis
        populateCrossLoopAnalysis(multiReport, events);

        return multiReport;
    }

    private void populateBasicInfo(LoopTerminationErrorReport report, LoopTerminationEvent event) {
        report.setFrameName(event.getFrameName());
        report.setIteration(event.getIteration());
        report.setTimestamp(event.getTimestamp());
        report.setTerminationType(event.getTerminationType());
        report.setTriggerOperation(event.getTriggerOperation());
        report.setTerminationReason(event.getTerminationReason());
        report.setWasEarlyTermination(event.isWasEarlyTermination());
        report.setEarlyTerminationCause(event.getEarlyTerminationCause());

        // Add execution context
        LoopInfo loopInfo = loopAnalyzer.getActiveLoops().get(event.getFrameName());
        if (loopInfo != null) {
            report.setLoopExecutionTime(System.currentTimeMillis() - loopInfo.getStartTime());
            report.setExpectedIterations(loopInfo.getExpectedIterations());
            report.setMaxIterationsObserved(loopInfo.getMaxIterationsObserved());
        }
    }

    private void populateVariableStateAnalysis(LoopTerminationErrorReport report, LoopTerminationEvent event) {
        if (!includeVariableValues) return;

        VariableStateAnalysis varAnalysis = new VariableStateAnalysis();
        String frameName = event.getFrameName();
        int iteration = event.getIteration();

        // Get all variables in the current frame and iteration
        Map<String, VariableStateInfo> currentVariables = new HashMap<>();
        Map<String, VariableStateInfo> inputVariables = new HashMap<>();
        Map<String, VariableStateInfo> outputVariables = new HashMap<>();

        // Analyze variables from nodeValueOutputs
        for (Map.Entry<VarId, SDValue> entry : nodeValueOutputs.entrySet()) {
            VarId varId = entry.getKey();
            SDValue value = entry.getValue();

            if (frameName.equals(varId.getFrame()) && varId.getIteration() == iteration) {
                VariableStateInfo varInfo = createVariableStateInfo(varId.getVariable(), value, varId);
                currentVariables.put(varId.getVariable(), varInfo);
            }
        }

        // Categorize variables based on loop structure
        LoopInfo loopInfo = loopAnalyzer.getActiveLoops().get(frameName);
        if (loopInfo != null) {
            // Separate input and output variables
            for (String inputVar : loopInfo.getInputVariables()) {
                if (currentVariables.containsKey(inputVar)) {
                    inputVariables.put(inputVar, currentVariables.get(inputVar));
                }
            }

            for (String outputVar : loopInfo.getOutputVariables()) {
                if (currentVariables.containsKey(outputVar)) {
                    outputVariables.put(outputVar, currentVariables.get(outputVar));
                }
            }

            // Add loop variables
            varAnalysis.setLoopVariables(loopInfo.getLoopVariables());
            varAnalysis.setLoopConstants(loopInfo.getLoopConstants());
            varAnalysis.setInvariantVariables(loopInfo.getInvariantVariables());
        }

        varAnalysis.setCurrentVariables(currentVariables);
        varAnalysis.setInputVariables(inputVariables);
        varAnalysis.setOutputVariables(outputVariables);

        // Analyze problematic variables
        varAnalysis.setProblematicVariables(identifyProblematicVariables(currentVariables));

        // Add variable dependency analysis
        varAnalysis.setVariableDependencies(analyzeVariableDependencies(frameName, currentVariables.keySet()));

        report.setVariableStateAnalysis(varAnalysis);
    }

    private VariableStateInfo createVariableStateInfo(String varName, SDValue value, VarId varId) {
        VariableStateInfo info = new VariableStateInfo();
        info.setVariableName(varName);
        info.setFrame(varId.getFrame());
        info.setIteration(varId.getIteration());
        info.setValueType(value.getSdValueType());

        if (includeVariableValues) {
            info.setValue(extractDisplayValue(value));
        }

        if (includeVariableShapes && value.getSdValueType() == SDValueType.TENSOR) {
            INDArray tensor = value.getTensorValue();
            if (tensor != null) {
                info.setShape(Arrays.toString(tensor.shape()));
                info.setDataType(tensor.dataType().toString());
                info.setLength(tensor.length());
                info.setMemoryUsage(estimateMemoryUsage(tensor));

                // Analyze tensor health
                info.setNumericalHealth(analyzeTensorHealth(tensor));
            }
        }

        // Add variable metadata
        info.getMetadata().put("created_at", System.currentTimeMillis());
        info.getMetadata().put("variable_id", varId.toString());

        return info;
    }

    private Object extractDisplayValue(SDValue value) {
        switch (value.getSdValueType()) {
            case TENSOR:
                INDArray tensor = value.getTensorValue();
                if (tensor == null) return "null";

                if (tensor.isScalar()) {
                    return tensor.getDouble(0);
                } else if (tensor.length() <= maxVariableValueDisplay) {
                    return tensor.toString();
                } else {
                    // Show first few elements
                    StringBuilder sb = new StringBuilder();
                    sb.append("Array[").append(Arrays.toString(tensor.shape())).append("] = [");
                    for (int i = 0; i < Math.min(maxVariableValueDisplay, tensor.length()); i++) {
                        if (i > 0) sb.append(", ");
                        sb.append(String.format("%.6f", tensor.getDouble(i)));
                    }
                    if (tensor.length() > maxVariableValueDisplay) {
                        sb.append(", ... (").append(tensor.length() - maxVariableValueDisplay).append(" more)");
                    }
                    sb.append("]");
                    return sb.toString();
                }

            case LIST:
                return value.getListValue();

            default:
                return value.toString();
        }
    }

    private String analyzeTensorHealth(INDArray tensor) {
        if (tensor == null) return "NULL";

        long nanCount = 0;
        long infCount = 0;
        double minVal = Double.MAX_VALUE;
        double maxVal = Double.MIN_VALUE;

        if (tensor.length() <= 10000) { // Only analyze small tensors directly
            for (int i = 0; i < tensor.length(); i++) {
                double val = tensor.getDouble(i);
                if (Double.isNaN(val)) nanCount++;
                else if (Double.isInfinite(val)) infCount++;
                else {
                    minVal = Math.min(minVal, val);
                    maxVal = Math.max(maxVal, val);
                }
            }
        }

        List<String> issues = new ArrayList<>();
        if (nanCount > 0) issues.add(nanCount + " NaN values");
        if (infCount > 0) issues.add(infCount + " Inf values");
        if (maxVal - minVal > 1e10) issues.add("Extreme value range");

        if (issues.isEmpty()) {
            return "HEALTHY";
        } else {
            return "ISSUES: " + String.join(", ", issues);
        }
    }

    private long estimateMemoryUsage(INDArray tensor) {
        if (tensor == null) return 0;
        return tensor.length() * tensor.dataType().width();
    }

    private Map<String, List<String>> identifyProblematicVariables(Map<String, VariableStateInfo> variables) {
        Map<String, List<String>> problematic = new HashMap<>();

        for (Map.Entry<String, VariableStateInfo> entry : variables.entrySet()) {
            String varName = entry.getKey();
            VariableStateInfo info = entry.getValue();
            List<String> issues = new ArrayList<>();

            // Check for numerical issues
            if (info.getNumericalHealth() != null && !info.getNumericalHealth().equals("HEALTHY")) {
                issues.add("Numerical instability: " + info.getNumericalHealth());
            }

            // Check for extreme memory usage
            if (info.getMemoryUsage() > 1024 * 1024 * 100) { // > 100MB
                issues.add("High memory usage: " + (info.getMemoryUsage() / 1024 / 1024) + "MB");
            }

            // Check for null values
            if (info.getValue() == null || "null".equals(info.getValue().toString())) {
                issues.add("Null value");
            }

            if (!issues.isEmpty()) {
                problematic.put(varName, issues);
            }
        }

        return problematic;
    }

    private Map<String, List<String>> analyzeVariableDependencies(String frameName, Set<String> variables) {
        Map<String, List<String>> dependencies = new HashMap<>();

        // Analyze operation dependencies
        for (Map.Entry<String, SameDiffOp> entry : sameDiff.getOps().entrySet()) {
            String opName = entry.getKey();
            SameDiffOp op = entry.getValue();

            List<String> inputs = op.getInputsToOp();
            List<String> outputs = op.getOutputsOfOp();

            if (inputs != null && outputs != null) {
                for (String output : outputs) {
                    if (variables.contains(output)) {
                        List<String> inputDeps = inputs.stream()
                                .filter(variables::contains)
                                .collect(Collectors.toList());
                        if (!inputDeps.isEmpty()) {
                            dependencies.put(output, inputDeps);
                        }
                    }
                }
            }
        }

        return dependencies;
    }

    private void populateOperationAnalysis(LoopTerminationErrorReport report, LoopTerminationEvent event) {
        if (!includeOperationHistory) return;

        OperationAnalysis opAnalysis = new OperationAnalysis();
        String frameName = event.getFrameName();

        // Get loop operations
        LoopInfo loopInfo = loopAnalyzer.getActiveLoops().get(frameName);
        if (loopInfo != null) {
            opAnalysis.setLoopConditionOp(loopInfo.getLoopCondOperation());
            opAnalysis.setExitOperations(loopInfo.getExitOperations());
            opAnalysis.setSwitchOperations(loopInfo.getSwitchOperations());
            opAnalysis.setNextIterationOperations(loopInfo.getNextIterationOperations());
            opAnalysis.setEnterOperations(loopInfo.getEnterOperations());
            opAnalysis.setMergeOperations(loopInfo.getMergeOperations());
        }

        // Analyze operation execution patterns
        LoopIterationTrace trace = loopAnalyzer.getIterationTraces().get(frameName);
        if (trace != null) {
            opAnalysis.setOperationExecutionCounts(trace.getOperationExecutionCounts());

            // Get recent operation history
            List<IterationSnapshot> recentSnapshots = trace.getIterations().stream()
                    .filter(snap -> snap.getIteration() >= event.getIteration() - maxHistoryDepth)
                    .collect(Collectors.toList());

            Map<Integer, List<String>> executionHistory = new HashMap<>();
            for (IterationSnapshot snapshot : recentSnapshots) {
                executionHistory.put(snapshot.getIteration(), snapshot.getExecutedOperations());
            }
            opAnalysis.setRecentExecutionHistory(executionHistory);
        }

        // Analyze trigger operation
        if (event.getTriggerOperation() != null) {
            OperationInfo triggerInfo = analyzeTriggerOperation(event.getTriggerOperation(), event);
            opAnalysis.setTriggerOperationInfo(triggerInfo);
        }

        report.setOperationAnalysis(opAnalysis);
    }

    private OperationInfo analyzeTriggerOperation(String triggerOp, LoopTerminationEvent event) {
        SameDiffOp op = sameDiff.getOps().get(triggerOp);
        if (op == null) {
            // Create minimal OperationInfo for unknown operation
            OperationInfo info = new OperationInfo(triggerOp, "UNKNOWN", "UNKNOWN",
                    new ArrayList<>(), new ArrayList<>());
            info.setExecutionStatus(OperationExecutionStatus.ERROR);
            info.setErrorInfo(new OperationErrorInfo("Operation not found in SameDiff graph", "OperationNotFound"));
            return info;
        }

        // Create OperationInfo with basic information
        String operationType = op.getOp().getClass().getSimpleName();
        String className = op.getOp().getClass().getName();
        List<String> inputs = op.getInputsToOp() != null ? op.getInputsToOp() : new ArrayList<>();
        List<String> outputs = op.getOutputsOfOp() != null ? op.getOutputsOfOp() : new ArrayList<>();

        // Create frame context for this termination event
        FrameIter frameContext = new FrameIter(event.getFrameName(), event.getIteration(), null);

        OperationInfo info = new OperationInfo(triggerOp, operationType, className, inputs, outputs,
                null, frameContext);

        // Perform comprehensive analysis with current state
        try {
            info.analyzeWithCurrentState(sameDiff, nodeValueOutputs, frameContext);

            // Mark as termination critical since this operation triggered termination
            info.setTerminationCritical(true);
            info.setExecutionStatus(OperationExecutionStatus.SUCCESS);

        } catch (Exception e) {
            // If analysis fails, record the error but continue
            OperationErrorInfo errorInfo = new OperationErrorInfo(
                    "Failed to analyze trigger operation: " + e.getMessage(),
                    e.getClass().getSimpleName()
            );
            errorInfo.setErrorContext("Loop termination analysis");
            errorInfo.setSeverity(OperationErrorInfo.ErrorSeverity.WARNING);
            info.setErrorInfo(errorInfo);
            info.setExecutionStatus(OperationExecutionStatus.ERROR);

            // Fallback: manually set input values if analysis failed
            if (!inputs.isEmpty()) {
                Map<String, Object> inputValues = new HashMap<>();
                for (String input : inputs) {
                    try {
                        VarId varId = new VarId(input, event.getFrameName(), event.getIteration(),null);
                        SDValue value = nodeValueOutputs.get(varId);
                        if (value != null) {
                            inputValues.put(input, OperationAnalysisUtils.extractValueForAnalysis(value));
                        } else {
                            inputValues.put(input, null);
                        }
                    } catch (Exception valueError) {
                        inputValues.put(input, "ERROR: " + valueError.getMessage());
                    }
                }
                info.getInputValues().putAll(inputValues);
            }
        }

        // Add termination-specific context to metadata
        info.getMetadata().put("terminationValue", event.getTerminationValue());
        info.getMetadata().put("terminationType", event.getTerminationType());
        info.getMetadata().put("terminationIteration", event.getIteration());
        info.getMetadata().put("terminationTimestamp", event.getTimestamp());
        info.getMetadata().put("wasEarlyTermination", event.isWasEarlyTermination());
        info.getMetadata().put("terminationReason", event.getTerminationReason());

        // Add loop state context if available
        if (event.getLoopStateAtTermination() != null) {
            LoopState loopState = event.getLoopStateAtTermination();
            info.getMetadata().put("loopStateVariableCount", loopState.getVariableStates().size());
            info.getMetadata().put("loopStateOperationCount", loopState.getOperationStates().size());
            info.getMetadata().put("loopStateActiveOps", loopState.getActiveOperations().size());
        }

        // Record this as a critical execution event
        OperationExecutionRecord terminationRecord = new OperationExecutionRecord();
        terminationRecord.setTimestamp(event.getTimestamp());
        terminationRecord.setStatus(OperationExecutionStatus.SUCCESS);
        terminationRecord.setIteration(event.getIteration());
        terminationRecord.setFrame(event.getFrameName());
        terminationRecord.addContext("terminationType", event.getTerminationType());
        terminationRecord.addContext("terminationValue", event.getTerminationValue());
        terminationRecord.addContext("triggerOperation", true);

        info.getExecutionHistory().add(terminationRecord);

        return info;
    }
    private void populateFrameContext(LoopTerminationErrorReport report, LoopTerminationEvent event) {
        if (!includeFrameContext) return;

        FrameContextInfo frameContext = new FrameContextInfo();
        frameContext.setFrameName(event.getFrameName());
        frameContext.setIteration(event.getIteration());

        // Get loop nesting information
        LoopInfo loopInfo = loopAnalyzer.getActiveLoops().get(event.getFrameName());
        if (loopInfo != null) {
            frameContext.setParentFrame(loopInfo.getParentFrameName());
            frameContext.setNestingDepth(loopInfo.getNestingDepth());

            // Get sibling and child frames
            List<String> relatedFrames = loopAnalyzer.getActiveLoops().keySet().stream()
                    .filter(frame -> !frame.equals(event.getFrameName()))
                    .collect(Collectors.toList());
            frameContext.setRelatedFrames(relatedFrames);
        }

        // Add cross-frame variable references
        frameContext.setCrossFrameReferences(analyzeCrossFrameReferences(event.getFrameName()));

        report.setFrameContext(frameContext);
    }

    private Map<String, List<String>> analyzeCrossFrameReferences(String frameName) {
        Map<String, List<String>> references = new HashMap<>();

        // This would require additional implementation to track cross-frame references
        // For now, return empty map

        return references;
    }

    private void populateVariableEvolution(LoopTerminationErrorReport report, LoopTerminationEvent event) {
        if (!includeVariableEvolution) return;

        VariableEvolutionAnalysis evolution = new VariableEvolutionAnalysis();

        LoopIterationTrace trace = loopAnalyzer.getIterationTraces().get(event.getFrameName());
        if (trace != null) {
            // Get recent evolution for key variables
            LoopInfo loopInfo = loopAnalyzer.getActiveLoops().get(event.getFrameName());
            if (loopInfo != null) {
                for (String varName : loopInfo.getLoopVariables()) {
                    List<Object> varEvolution = trace.getVariableEvolution().get(varName);
                    if (varEvolution != null) {
                        // Take only recent history
                        int startIndex = Math.max(0, varEvolution.size() - maxHistoryDepth);
                        List<Object> recentEvolution = varEvolution.subList(startIndex, varEvolution.size());
                        evolution.getVariableEvolution().put(varName, recentEvolution);

                        // Analyze patterns
                        VariablePattern pattern = analyzeEvolutionPattern(varName, recentEvolution);
                        if (pattern != null) {
                            evolution.getDetectedPatterns().put(varName, pattern);
                        }
                    }
                }
            }

            // Include condition evaluation history
            List<ConditionEvaluation> recentConditions = trace.getConditionEvaluations().stream()
                    .filter(cond -> cond.getIteration() >= event.getIteration() - maxHistoryDepth)
                    .collect(Collectors.toList());
            evolution.setConditionEvaluationHistory(recentConditions);
        }

        report.setVariableEvolution(evolution);
    }

    private VariablePattern analyzeEvolutionPattern(String varName, List<Object> evolution) {
        if (evolution.size() < 3) return null;

        // Check for monotonic decrease
        List<Double> numericValues = evolution.stream()
                .map(this::extractNumericValue)
                .filter(Objects::nonNull)
                .collect(Collectors.toList());

        if (numericValues.size() >= 3) {
            boolean monotonic = true;
            for (int i = 1; i < numericValues.size(); i++) {
                if (numericValues.get(i) >= numericValues.get(i-1)) {
                    monotonic = false;
                    break;
                }
            }

            if (monotonic) {
                return VariablePattern.createMonotonicDecrease(varName, evolution, 0, evolution.size() - 1);
            }

            // Check for convergence
            double convergenceRate = calculateConvergenceRate(numericValues);
            if (convergenceRate > 0.1) {
                return VariablePattern.createConvergence(varName, evolution, convergenceRate, 0, evolution.size() - 1);
            }
        }

        return null;
    }

    private Double extractNumericValue(Object value) {
        if (value instanceof Number) {
            return ((Number) value).doubleValue();
        } else if (value instanceof INDArray) {
            INDArray arr = (INDArray) value;
            if (arr.isScalar()) {
                return arr.getDouble(0);
            }
        }
        return null;
    }

    private double calculateConvergenceRate(List<Double> values) {
        if (values.size() < 2) return 0.0;

        double totalChange = 0.0;
        for (int i = 1; i < values.size(); i++) {
            totalChange += Math.abs(values.get(i) - values.get(i-1));
        }

        return totalChange / (values.size() - 1);
    }

    private void populatePerformanceMetrics(LoopTerminationErrorReport report, LoopTerminationEvent event) {
        if (!includeMemoryMetrics) return;

        PerformanceMetrics metrics = new PerformanceMetrics();

        // Memory metrics
        Runtime runtime = Runtime.getRuntime();
        metrics.setTotalMemory(runtime.totalMemory());
        metrics.setFreeMemory(runtime.freeMemory());
        metrics.setUsedMemory(runtime.totalMemory() - runtime.freeMemory());
        metrics.setMaxMemory(runtime.maxMemory());

        // Execution timing
        LoopInfo loopInfo = loopAnalyzer.getActiveLoops().get(event.getFrameName());
        if (loopInfo != null) {
            metrics.setLoopExecutionTime(System.currentTimeMillis() - loopInfo.getStartTime());
            metrics.setAverageIterationTime(loopInfo.getAverageIterationTime());
            metrics.setIterationsPerSecond(loopInfo.getLoopEfficiency());
        }

        // Variable memory usage
        long totalVariableMemory = 0;
        if (report.getVariableStateAnalysis() != null) {
            for (VariableStateInfo varInfo : report.getVariableStateAnalysis().getCurrentVariables().values()) {
                totalVariableMemory += varInfo.getMemoryUsage();
            }
        }
        metrics.setTotalVariableMemory(totalVariableMemory);

        report.setPerformanceMetrics(metrics);
    }

    private void populateRootCauseAnalysis(LoopTerminationErrorReport report, LoopTerminationEvent event) {
        RootCauseAnalysis rootCause = new RootCauseAnalysis();

        // Primary cause analysis
        String primaryCause = analyzePrimaryCause(event, report);
        rootCause.setPrimaryCause(primaryCause);

        // Contributing factors
        List<String> contributingFactors = identifyContributingFactors(event, report);
        rootCause.setContributingFactors(contributingFactors);

        // Recommended actions
        List<String> recommendations = generateRecommendations(event, report);
        rootCause.setRecommendedActions(recommendations);

        // Similar patterns in history
        List<String> similarPatterns = findSimilarTerminationPatterns(event);
        rootCause.setSimilarPatternsInHistory(similarPatterns);

        // Confidence assessment
        double confidence = assessCauseConfidence(event, report);
        rootCause.setConfidenceLevel(confidence);

        report.setRootCauseAnalysis(rootCause);
    }

    private String analyzePrimaryCause(LoopTerminationEvent event, LoopTerminationErrorReport report) {
        StringBuilder cause = new StringBuilder();

        switch (event.getTerminationType()) {
            case CONDITION_FALSE:
                cause.append("Loop condition became false");
                if (event.isWasEarlyTermination()) {
                    cause.append(" earlier than expected");

                    // Analyze condition evolution
                    if (report.getVariableEvolution() != null &&
                            !report.getVariableEvolution().getConditionEvaluationHistory().isEmpty()) {

                        List<ConditionEvaluation> conditions = report.getVariableEvolution().getConditionEvaluationHistory();
                        if (conditions.size() >= 2) {
                            ConditionEvaluation latest = conditions.get(conditions.size() - 1);
                            ConditionEvaluation previous = conditions.get(conditions.size() - 2);

                            cause.append(". Condition changed from ")
                                    .append(formatValue(previous.getConditionValue()))
                                    .append(" to ")
                                    .append(formatValue(latest.getConditionValue()));
                        }
                    }
                }
                break;

            case ERROR_TERMINATION:
                cause.append("Error occurred during loop execution: ").append(event.getTerminationReason());

                // Check for numerical issues in variables
                if (report.getVariableStateAnalysis() != null &&
                        !report.getVariableStateAnalysis().getProblematicVariables().isEmpty()) {
                    cause.append(". Problematic variables detected: ");
                    cause.append(String.join(", ", report.getVariableStateAnalysis().getProblematicVariables().keySet()));
                }
                break;

            case SWITCH_TERMINATION:
                cause.append("Switch operation took unexpected branch: ").append(event.getTriggerOperation());
                break;

            case TIMEOUT_TERMINATION:
                cause.append("Loop exceeded maximum allowed iterations (").append(event.getIteration()).append(")");
                break;

            default:
                cause.append("Loop terminated due to: ").append(event.getTerminationType());
        }

        return cause.toString();
    }

    private List<String> identifyContributingFactors(LoopTerminationEvent event, LoopTerminationErrorReport report) {
        List<String> factors = new ArrayList<>();

        // Check for numerical instability
        if (report.getVariableStateAnalysis() != null) {
            Map<String, List<String>> problematic = report.getVariableStateAnalysis().getProblematicVariables();
            if (!problematic.isEmpty()) {
                factors.add("Numerical instability in variables: " + String.join(", ", problematic.keySet()));
            }
        }

        // Check for memory pressure
        if (report.getPerformanceMetrics() != null) {
            PerformanceMetrics metrics = report.getPerformanceMetrics();
            double memoryUsagePercent = (double) metrics.getUsedMemory() / metrics.getTotalMemory() * 100;
            if (memoryUsagePercent > 90) {
                factors.add("High memory usage: " + String.format("%.1f%%", memoryUsagePercent));
            }
        }

        // Check for variable evolution patterns
        if (report.getVariableEvolution() != null) {
            Map<String, VariablePattern> patterns = report.getVariableEvolution().getDetectedPatterns();
            for (Map.Entry<String, VariablePattern> entry : patterns.entrySet()) {
                if (entry.getValue().isTerminationIndicator()) {
                    factors.add("Variable pattern in " + entry.getKey() + ": " + entry.getValue().getDescription());
                }
            }
        }

        // Check for execution time anomalies
        if (report.getPerformanceMetrics() != null && report.getPerformanceMetrics().getLoopExecutionTime() > 0) {
            LoopInfo loopInfo = loopAnalyzer.getActiveLoops().get(event.getFrameName());
            if (loopInfo != null && loopInfo.getExpectedIterations() > 0) {
                double expectedTime = loopInfo.getExpectedIterations() * loopInfo.getAverageIterationTime();
                if (report.getPerformanceMetrics().getLoopExecutionTime() > expectedTime * 2) {
                    factors.add("Execution took significantly longer than expected");
                }
            }
        }

        return factors;
    }

    private List<String> generateRecommendations(LoopTerminationEvent event, LoopTerminationErrorReport report) {
        List<String> recommendations = new ArrayList<>();

        switch (event.getTerminationType()) {
            case CONDITION_FALSE:
                if (event.isWasEarlyTermination()) {
                    recommendations.add("Review loop condition logic for unexpected early termination");
                    recommendations.add("Check input data for anomalies that might cause early convergence");
                }
                recommendations.add("Add condition value logging to track termination triggers");
                break;

            case ERROR_TERMINATION:
                recommendations.add("Add error handling and validation within the loop");
                recommendations.add("Check for numerical stability in loop variables");
                if (report.getVariableStateAnalysis() != null &&
                        !report.getVariableStateAnalysis().getProblematicVariables().isEmpty()) {
                    recommendations.add("Address numerical issues in problematic variables");
                }
                break;

            case TIMEOUT_TERMINATION:
                recommendations.add("Increase maximum iteration limit if convergence is expected");
                recommendations.add("Optimize loop operations for better performance");
                recommendations.add("Add convergence detection to terminate early when possible");
                break;

            case SWITCH_TERMINATION:
                recommendations.add("Review switch operation logic and branch conditions");
                recommendations.add("Add logging for switch decisions to understand branching patterns");
                break;
        }

        // General recommendations based on analysis
        if (report.getPerformanceMetrics() != null) {
            double memoryUsagePercent = (double) report.getPerformanceMetrics().getUsedMemory() /
                    report.getPerformanceMetrics().getTotalMemory() * 100;
            if (memoryUsagePercent > 80) {
                recommendations.add("Consider memory optimization - current usage is " +
                        String.format("%.1f%%", memoryUsagePercent));
            }
        }

        return recommendations;
    }

    private List<String> findSimilarTerminationPatterns(LoopTerminationEvent event) {
        List<String> patterns = new ArrayList<>();

        // Search through termination history for similar patterns
        Map<String, List<LoopTerminationEvent>> history = loopAnalyzer.getTerminationHistory();

        for (Map.Entry<String, List<LoopTerminationEvent>> entry : history.entrySet()) {
            String frameName = entry.getKey();
            if (frameName.equals(event.getFrameName())) continue; // Skip same frame

            for (LoopTerminationEvent pastEvent : entry.getValue()) {
                if (pastEvent.getTerminationType() == event.getTerminationType()) {
                    patterns.add("Similar " + event.getTerminationType() + " in frame " + frameName +
                            " at iteration " + pastEvent.getIteration());
                }
            }
        }

        return patterns;
    }

    private double assessCauseConfidence(LoopTerminationEvent event, LoopTerminationErrorReport report) {
        double confidence = 0.5; // Base confidence

        // Increase confidence based on available data
        if (report.getVariableStateAnalysis() != null) confidence += 0.2;
        if (report.getVariableEvolution() != null) confidence += 0.2;
        if (report.getOperationAnalysis() != null) confidence += 0.1;

        // Decrease confidence for complex termination types
        switch (event.getTerminationType()) {
            case CONDITION_FALSE:
            case ERROR_TERMINATION:
                confidence += 0.1; // These are usually clear
                break;
            case SWITCH_TERMINATION:
                confidence -= 0.1; // More complex to analyze
                break;
        }

        return Math.min(1.0, confidence);
    }

    private void populateVisualizations(LoopTerminationErrorReport report, LoopTerminationEvent event) {
        VisualizationData vizData = new VisualizationData();

        // Generate variable evolution plots
        if (report.getVariableEvolution() != null) {
            Map<String, String> variablePlots = new HashMap<>();

            for (Map.Entry<String, List<Object>> entry : report.getVariableEvolution().getVariableEvolution().entrySet()) {
                String varName = entry.getKey();
                List<Object> evolution = entry.getValue();

                String plotData = generateVariableEvolutionPlot(varName, evolution);
                variablePlots.put(varName, plotData);
            }

            vizData.setVariableEvolutionPlots(variablePlots);
        }

        // Generate condition evaluation timeline
        if (report.getVariableEvolution() != null &&
                !report.getVariableEvolution().getConditionEvaluationHistory().isEmpty()) {
            String conditionTimeline = generateConditionTimeline(
                    report.getVariableEvolution().getConditionEvaluationHistory());
            vizData.setConditionEvaluationTimeline(conditionTimeline);
        }

        // Generate memory usage visualization
        if (report.getPerformanceMetrics() != null) {
            String memoryViz = generateMemoryVisualization(report.getPerformanceMetrics());
            vizData.setMemoryUsageVisualization(memoryViz);
        }

        report.setVisualizationData(vizData);
    }

    private String generateVariableEvolutionPlot(String varName, List<Object> evolution) {
        StringBuilder plot = new StringBuilder();
        plot.append("Variable Evolution for ").append(varName).append(":\n");

        List<Double> numericValues = evolution.stream()
                .map(this::extractNumericValue)
                .filter(Objects::nonNull)
                .collect(Collectors.toList());

        if (!numericValues.isEmpty()) {
            double min = numericValues.stream().mapToDouble(Double::doubleValue).min().orElse(0);
            double max = numericValues.stream().mapToDouble(Double::doubleValue).max().orElse(1);
            double range = max - min;

            for (int i = 0; i < numericValues.size(); i++) {
                double value = numericValues.get(i);
                double normalized = range > 0 ? (value - min) / range : 0.5;
                int barLength = (int) (normalized * 40);

                plot.append(String.format("Iter %2d: ", i));
                plot.append("█".repeat(Math.max(0, barLength)));
                plot.append(String.format(" %.6f\n", value));
            }
        } else {
            plot.append("No numeric data available for plotting\n");
        }

        return plot.toString();
    }

    private String generateConditionTimeline(List<ConditionEvaluation> conditions) {
        StringBuilder timeline = new StringBuilder();
        timeline.append("Condition Evaluation Timeline:\n");

        for (ConditionEvaluation cond : conditions) {
            timeline.append(String.format("Iter %2d: %s = %s %s\n",
                    cond.getIteration(),
                    cond.getConditionOperation(),
                    formatValue(cond.getConditionValue()),
                    cond.isTerminationTriggered() ? "[TERMINATION]" : ""
            ));
        }

        return timeline.toString();
    }

    private String generateMemoryVisualization(PerformanceMetrics metrics) {
        StringBuilder viz = new StringBuilder();
        viz.append("Memory Usage Visualization:\n");

        double usedPercent = (double) metrics.getUsedMemory() / metrics.getTotalMemory() * 100;
        int usedBars = (int) (usedPercent / 2.5); // Scale to 40 chars
        int freeBars = 40 - usedBars;

        viz.append("Used:  [").append("█".repeat(usedBars)).append(" ".repeat(freeBars)).append("] ");
        viz.append(String.format("%.1f%% (%d MB)\n", usedPercent, metrics.getUsedMemory() / 1024 / 1024));

        viz.append("Total: ").append(metrics.getTotalMemory() / 1024 / 1024).append(" MB\n");
        viz.append("Max:   ").append(metrics.getMaxMemory() / 1024 / 1024).append(" MB\n");

        return viz.toString();
    }

    private void populateCrossLoopAnalysis(MultiLoopTerminationErrorReport multiReport, List<LoopTerminationEvent> events) {
        CrossLoopAnalysis crossAnalysis = new CrossLoopAnalysis();

        // Identify common patterns across loops
        Map<TerminationType, Long> terminationCounts = events.stream()
                .collect(Collectors.groupingBy(LoopTerminationEvent::getTerminationType, Collectors.counting()));
        crossAnalysis.setTerminationTypeDistribution(terminationCounts);

        // Find correlation between loop terminations
        List<String> correlations = findTerminationCorrelations(events);
        crossAnalysis.setTerminationCorrelations(correlations);

        // Identify system-wide issues
        List<String> systemIssues = identifySystemWideIssues(events, multiReport);
        crossAnalysis.setSystemWideIssues(systemIssues);

        multiReport.setCrossLoopAnalysis(crossAnalysis);
    }

    private List<String> findTerminationCorrelations(List<LoopTerminationEvent> events) {
        List<String> correlations = new ArrayList<>();

        // Sort events by timestamp
        List<LoopTerminationEvent> sortedEvents = events.stream()
                .sorted(Comparator.comparingLong(LoopTerminationEvent::getTimestamp))
                .collect(Collectors.toList());

        // Look for events that occurred close in time
        for (int i = 0; i < sortedEvents.size() - 1; i++) {
            LoopTerminationEvent event1 = sortedEvents.get(i);
            LoopTerminationEvent event2 = sortedEvents.get(i + 1);

            long timeDiff = event2.getTimestamp() - event1.getTimestamp();
            if (timeDiff < 1000) { // Within 1 second
                correlations.add(String.format("Frames %s and %s terminated within %dms of each other",
                        event1.getFrameName(), event2.getFrameName(), timeDiff));
            }
        }

        return correlations;
    }

    private List<String> identifySystemWideIssues(List<LoopTerminationEvent> events, MultiLoopTerminationErrorReport multiReport) {
        List<String> issues = new ArrayList<>();

        // Check for widespread error terminations
        long errorCount = events.stream()
                .filter(e -> e.getTerminationType() == TerminationType.ERROR_TERMINATION)
                .count();

        if (errorCount > events.size() / 2) {
            issues.add("More than half of loops terminated with errors - possible system-wide issue");
        }

        // Check for memory issues across loops
        boolean hasMemoryIssues = multiReport.getIndividualReports().values().stream()
                .filter(report -> report.getPerformanceMetrics() != null)
                .anyMatch(report -> {
                    double memUsage = (double) report.getPerformanceMetrics().getUsedMemory() /
                            report.getPerformanceMetrics().getTotalMemory();
                    return memUsage > 0.9;
                });

        if (hasMemoryIssues) {
            issues.add("High memory usage detected across multiple loops");
        }

        return issues;
    }

    private String formatErrorReportAsString(LoopTerminationErrorReport report) {
        StringBuilder sb = new StringBuilder();

        // Header
        sb.append("═".repeat(80)).append("\n");
        sb.append("LOOP TERMINATION ERROR REPORT\n");
        sb.append("═".repeat(80)).append("\n");
        sb.append("Frame: ").append(report.getFrameName()).append("\n");
        sb.append("Iteration: ").append(report.getIteration()).append("\n");
        sb.append("Timestamp: ").append(LocalDateTime.ofInstant(
                java.time.Instant.ofEpochMilli(report.getTimestamp()),
                java.time.ZoneId.systemDefault()).format(timeFormatter)).append("\n");
        sb.append("Termination Type: ").append(report.getTerminationType()).append("\n");
        sb.append("Early Termination: ").append(report.isWasEarlyTermination() ? "YES" : "NO").append("\n");
        sb.append("\n");

        // Root Cause Analysis
        if (report.getRootCauseAnalysis() != null) {
            sb.append("ROOT CAUSE ANALYSIS\n");
            sb.append("─".repeat(40)).append("\n");
            sb.append("Primary Cause: ").append(report.getRootCauseAnalysis().getPrimaryCause()).append("\n");
            sb.append("Confidence: ").append(String.format("%.1f%%",
                    report.getRootCauseAnalysis().getConfidenceLevel() * 100)).append("\n");

            if (!report.getRootCauseAnalysis().getContributingFactors().isEmpty()) {
                sb.append("\nContributing Factors:\n");
                for (String factor : report.getRootCauseAnalysis().getContributingFactors()) {
                    sb.append("  • ").append(factor).append("\n");
                }
            }

            if (!report.getRootCauseAnalysis().getRecommendedActions().isEmpty()) {
                sb.append("\nRecommended Actions:\n");
                for (String action : report.getRootCauseAnalysis().getRecommendedActions()) {
                    sb.append("  → ").append(action).append("\n");
                }
            }
            sb.append("\n");
        }

        // Variable State Analysis
        if (report.getVariableStateAnalysis() != null) {
            sb.append("VARIABLE STATE ANALYSIS\n");
            sb.append("─".repeat(40)).append("\n");

            VariableStateAnalysis varAnalysis = report.getVariableStateAnalysis();

            if (!varAnalysis.getProblematicVariables().isEmpty()) {
                sb.append("Problematic Variables:\n");
                for (Map.Entry<String, List<String>> entry : varAnalysis.getProblematicVariables().entrySet()) {
                    sb.append("  ⚠️  ").append(entry.getKey()).append(": ");
                    sb.append(String.join(", ", entry.getValue())).append("\n");
                }
                sb.append("\n");
            }

            sb.append("Variable Summary:\n");
            sb.append("  Total Variables: ").append(varAnalysis.getCurrentVariables().size()).append("\n");
            sb.append("  Input Variables: ").append(varAnalysis.getInputVariables().size()).append("\n");
            sb.append("  Output Variables: ").append(varAnalysis.getOutputVariables().size()).append("\n");
            sb.append("  Loop Variables: ").append(varAnalysis.getLoopVariables().size()).append("\n");
            sb.append("\n");

            // Show key variable values
            if (includeVariableValues && !varAnalysis.getCurrentVariables().isEmpty()) {
                sb.append("Key Variable Values:\n");
                varAnalysis.getCurrentVariables().entrySet().stream()
                        .limit(10) // Show only first 10
                        .forEach(entry -> {
                            VariableStateInfo info = entry.getValue();
                            sb.append("  ").append(entry.getKey()).append(" = ").append(info.getValue());
                            if (info.getShape() != null) {
                                sb.append(" ").append(info.getShape());
                            }
                            if (info.getNumericalHealth() != null && !info.getNumericalHealth().equals("HEALTHY")) {
                                sb.append(" [").append(info.getNumericalHealth()).append("]");
                            }
                            sb.append("\n");
                        });
                sb.append("\n");
            }
        }

        // Operation Analysis
        if (report.getOperationAnalysis() != null) {
            sb.append("OPERATION ANALYSIS\n");
            sb.append("─".repeat(40)).append("\n");

            OperationAnalysis opAnalysis = report.getOperationAnalysis();
            sb.append("Trigger Operation: ").append(report.getTriggerOperation()).append("\n");

            if (opAnalysis.getTriggerOperationInfo() != null) {
                OperationInfo triggerInfo = opAnalysis.getTriggerOperationInfo();
                sb.append("  Type: ").append(triggerInfo.getOperationType()).append("\n");
                if (triggerInfo.getInputs() != null) {
                    sb.append("  Inputs: ").append(String.join(", ", triggerInfo.getInputs())).append("\n");
                }
                if (triggerInfo.getInputValues() != null && !triggerInfo.getInputValues().isEmpty()) {
                    sb.append("  Input Values:\n");
                    for (Map.Entry<String, Object> entry : triggerInfo.getInputValues().entrySet()) {
                        sb.append("    ").append(entry.getKey()).append(" = ").append(entry.getValue()).append("\n");
                    }
                }
            }
            sb.append("\n");
        }

        // Performance Metrics
        if (report.getPerformanceMetrics() != null) {
            sb.append("PERFORMANCE METRICS\n");
            sb.append("─".repeat(40)).append("\n");

            PerformanceMetrics metrics = report.getPerformanceMetrics();
            sb.append("Execution Time: ").append(metrics.getLoopExecutionTime()).append("ms\n");

            if (metrics.getAverageIterationTime() > 0) {
                sb.append("Average Iteration Time: ").append(String.format("%.2f", metrics.getAverageIterationTime())).append("ms\n");
            }

            double memUsagePercent = (double) metrics.getUsedMemory() / metrics.getTotalMemory() * 100;
            sb.append("Memory Usage: ").append(String.format("%.1f%%", memUsagePercent));
            sb.append(" (").append(metrics.getUsedMemory() / 1024 / 1024).append(" MB used)\n");
            sb.append("\n");
        }

        // Variable Evolution Visualizations
        if (report.getVisualizationData() != null && generateVisualizations) {
            sb.append("VISUALIZATIONS\n");
            sb.append("─".repeat(40)).append("\n");

            VisualizationData vizData = report.getVisualizationData();

            if (vizData.getConditionEvaluationTimeline() != null) {
                sb.append(vizData.getConditionEvaluationTimeline()).append("\n");
            }

            if (vizData.getMemoryUsageVisualization() != null) {
                sb.append(vizData.getMemoryUsageVisualization()).append("\n");
            }

            if (vizData.getVariableEvolutionPlots() != null && !vizData.getVariableEvolutionPlots().isEmpty()) {
                // Show evolution for first few variables
                vizData.getVariableEvolutionPlots().entrySet().stream()
                        .limit(3)
                        .forEach(entry -> sb.append(entry.getValue()).append("\n"));
            }
        }

        sb.append("═".repeat(80)).append("\n");
        sb.append("END OF REPORT\n");
        sb.append("═".repeat(80)).append("\n");

        return sb.toString();
    }

    private String formatValue(Object value) {
        if (value == null) return "null";
        if (value instanceof Number) {
            return String.format("%.6f", ((Number) value).doubleValue());
        }
        if (value instanceof INDArray) {
            INDArray arr = (INDArray) value;
            if (arr.isScalar()) {
                return String.format("%.6f", arr.getDouble(0));
            }
        }
        return value.toString();
    }

    // Configuration methods
    public void setIncludeVariableValues(boolean include) { this.includeVariableValues = include; }
    public void setIncludeVariableShapes(boolean include) { this.includeVariableShapes = include; }
    public void setIncludeOperationHistory(boolean include) { this.includeOperationHistory = include; }
    public void setIncludeFrameContext(boolean include) { this.includeFrameContext = include; }
    public void setIncludeMemoryMetrics(boolean include) { this.includeMemoryMetrics = include; }
    public void setIncludeVariableEvolution(boolean include) { this.includeVariableEvolution = include; }
    public void setGenerateVisualizations(boolean generate) { this.generateVisualizations = generate; }
    public void setMaxVariableValueDisplay(int max) { this.maxVariableValueDisplay = max; }
    public void setMaxHistoryDepth(int depth) { this.maxHistoryDepth = depth; }

}


