package org.nd4j.autodiff.samediff;

import org.nd4j.autodiff.samediff.config.SDValue;
import org.nd4j.autodiff.samediff.config.SDValueType;
import org.nd4j.autodiff.samediff.internal.VarId;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Helper methods for loop analysis in the LoopTerminationAnalyzer
 */
public class LoopAnalysisHelpers {

    /**
     * Analyze the cause of early termination
     *
     * @param loopInfo The loop information
     * @param iteration The iteration at which termination occurred
     * @param terminationType The type of termination that occurred
     * @return Detailed analysis of the early termination cause
     */
    public static String analyzeEarlyTerminationCause(LoopInfo loopInfo, int iteration,
                                                      TerminationType terminationType,
                                                      Map<String, LoopIterationTrace> iterationTraces) {
        StringBuilder cause = new StringBuilder();

        switch (terminationType) {
            case CONDITION_FALSE:
                cause.append("Loop condition became false earlier than expected");

                // Analyze condition evolution
                LoopIterationTrace trace = iterationTraces.get(loopInfo.getFrameName());
                if (trace != null && !trace.getConditionEvaluations().isEmpty()) {
                    List<ConditionEvaluation> recent = trace.getConditionEvaluations().stream()
                            .filter(eval -> eval.getIteration() >= iteration - 3 && eval.getIteration() <= iteration)
                            .collect(Collectors.toList());

                    if (recent.size() >= 2) {
                        cause.append(" - Recent condition values: ");
                        for (ConditionEvaluation eval : recent) {
                            cause.append("[").append(eval.getIteration()).append(": ").append(formatValue(eval.getConditionValue())).append("] ");
                        }

                        // Analyze sudden changes
                        if (recent.size() >= 2) {
                            Object lastValue = recent.get(recent.size() - 1).getConditionValue();
                            Object secondLastValue = recent.get(recent.size() - 2).getConditionValue();

                            if (valueChangedSuddenly(secondLastValue, lastValue)) {
                                cause.append("\n  SUDDEN CHANGE DETECTED: ");
                                cause.append("Previous: ").append(formatValue(secondLastValue));
                                cause.append(" → Current: ").append(formatValue(lastValue));

                                // Calculate change magnitude
                                Double prevNum = extractNumericValue(secondLastValue);
                                Double currNum = extractNumericValue(lastValue);
                                if (prevNum != null && currNum != null && prevNum != 0) {
                                    double changePercent = Math.abs((currNum - prevNum) / prevNum) * 100;
                                    cause.append(" (").append(String.format("%.1f%%", changePercent)).append(" change)");
                                }
                            }
                        }

                        // Check for convergence patterns
                        if (recent.size() >= 3) {
                            List<Double> numericValues = recent.stream()
                                    .map(eval -> extractNumericValue(eval.getConditionValue()))
                                    .filter(Objects::nonNull)
                                    .collect(Collectors.toList());

                            if (numericValues.size() >= 3) {
                                double convergenceRate = calculateConvergenceRate(numericValues);
                                if (convergenceRate > 0.1) {
                                    cause.append("\n  RAPID CONVERGENCE: Rate = ").append(String.format("%.4f", convergenceRate));
                                }
                            }
                        }
                    }
                }

                // Check for variable-based causes
                analyzeVariableBasedCauses(cause, loopInfo, iteration, trace);
                break;

            case CONDITION_TRUE_EXIT:
                cause.append("Exit condition met sooner than expected");

                // Analyze what caused the exit condition to be true
                analyzeExitConditionCauses(cause, loopInfo, iteration, iterationTraces);
                break;

            case SWITCH_TERMINATION:
                cause.append("Switch operation took unexpected branch leading to early exit");

                // Analyze switch decision patterns
                analyzeSwitchTerminationCauses(cause, loopInfo, iteration, iterationTraces);
                break;

            case ERROR_TERMINATION:
                cause.append("Error occurred before normal termination condition");

                // Analyze error patterns
                analyzeErrorTerminationCauses(cause, loopInfo, iteration, iterationTraces);
                break;

            case TIMEOUT_TERMINATION:
                cause.append("Loop exceeded maximum allowed iterations");

                // Analyze timeout patterns
                analyzeTimeoutCauses(cause, loopInfo, iteration);
                break;

            case EARLY_BREAK:
                cause.append("Loop was terminated by an early break condition");

                // Analyze break patterns
                analyzeEarlyBreakCauses(cause, loopInfo, iteration, iterationTraces);
                break;

            case RESOURCE_EXHAUSTION:
                cause.append("Loop terminated due to resource exhaustion");

                // Analyze resource usage patterns
                analyzeResourceExhaustionCauses(cause, loopInfo, iteration);
                break;

            case MANUAL_TERMINATION:
                cause.append("Loop was manually terminated");
                break;

            default:
                cause.append("Unexpected termination type: ").append(terminationType);
                cause.append(" - This may indicate a new termination pattern");
        }

        // Add general early termination indicators
        addGeneralEarlyTerminationIndicators(cause, loopInfo, iteration, iterationTraces);

        return cause.toString();
    }

    /**
     * Map termination type to loop status
     *
     * @param terminationType The type of termination
     * @return The corresponding loop status
     */
    public static LoopTerminationStatus mapTerminationTypeToStatus(
            TerminationType terminationType) {
        switch (terminationType) {
            case CONDITION_FALSE:
            case CONDITION_TRUE_EXIT:
                return LoopTerminationStatus.TERMINATED_NORMAL;

            case SWITCH_TERMINATION:
            case EARLY_BREAK:
            case RESOURCE_EXHAUSTION:
                return LoopTerminationStatus.TERMINATED_EARLY;

            case ERROR_TERMINATION:
                return LoopTerminationStatus.TERMINATED_ERROR;

            case TIMEOUT_TERMINATION:
                return LoopTerminationStatus.TERMINATED_TIMEOUT;

            case MANUAL_TERMINATION:
                return LoopTerminationStatus.TERMINATED_EARLY;

            default:
                return LoopTerminationStatus.TERMINATED_EARLY;
        }
    }

    /**
     * Capture current loop state for debugging
     *
     * @param frameName The name of the loop frame
     * @param iteration The current iteration
     * @param nodeValueOutputs The current node value outputs
     * @param sameDiff The SameDiff instance for additional context
     * @return A comprehensive snapshot of the loop state
     */
    public static LoopState captureLoopState(String frameName, int iteration,
                                             Map<VarId, SDValue> nodeValueOutputs,
                                             org.nd4j.autodiff.samediff.SameDiff sameDiff) {
        LoopState state = new LoopState();
        state.setIteration(iteration);

        // Capture variable states in the current frame
        Map<String, Object> variableStates = new HashMap<>();
        Map<String, String> operationStates = new HashMap<>();
        List<String> activeOperations = new ArrayList<>();
        Map<String, Object> frameContext = new HashMap<>();

        // Collect variables from the current frame and iteration
        for (Map.Entry<VarId, SDValue> entry : nodeValueOutputs.entrySet()) {
            VarId varId = entry.getKey();

            // Check if this variable belongs to our frame
            if (frameName.equals(varId.getFrame()) && varId.getIteration() == iteration) {
                String varName = varId.getVariable();
                SDValue value = entry.getValue();

                if (value != null) {
                    Object extractedValue = extractValueFromSDValue(value);
                    variableStates.put(varName, extractedValue);

                    // Add variable metadata
                    frameContext.put(varName + "_type", value.getSdValueType().toString());
                    if (value.getSdValueType() == SDValueType.TENSOR && value.getTensorValue() != null) {
                        INDArray tensor = value.getTensorValue();
                        frameContext.put(varName + "_shape", Arrays.toString(tensor.shape()));
                        frameContext.put(varName + "_dataType", tensor.dataType().toString());
                        frameContext.put(varName + "_length", tensor.length());
                    }
                }
            }
        }

        // Capture operation states
        for (Map.Entry<String, org.nd4j.autodiff.samediff.internal.SameDiffOp> opEntry : sameDiff.getOps().entrySet()) {
            String opName = opEntry.getKey();
            org.nd4j.autodiff.samediff.internal.SameDiffOp op = opEntry.getValue();

            // Check if this operation is relevant to our frame
            if (isOperationRelevantToFrame(op, frameName)) {
                String opType = op.getOp().getClass().getSimpleName();
                operationStates.put(opName, opType);

                // Check if operation is currently active/executed
                if (isOperationActive(opName, frameName, iteration)) {
                    activeOperations.add(opName);
                }
            }
        }

        // Add frame-specific context
        frameContext.put("frameName", frameName);
        frameContext.put("iteration", iteration);
        frameContext.put("timestamp", System.currentTimeMillis());
        frameContext.put("variableCount", variableStates.size());
        frameContext.put("operationCount", operationStates.size());
        frameContext.put("activeOperationCount", activeOperations.size());

        // Add memory usage information if available
        try {
            Runtime runtime = Runtime.getRuntime();
            long totalMemory = runtime.totalMemory();
            long freeMemory = runtime.freeMemory();
            long usedMemory = totalMemory - freeMemory;

            frameContext.put("memoryUsed", usedMemory);
            frameContext.put("memoryTotal", totalMemory);
            frameContext.put("memoryFree", freeMemory);
        } catch (Exception e) {
            frameContext.put("memoryError", e.getMessage());
        }

        // Set all collected data
        state.setVariableStates(variableStates);
        state.setOperationStates(operationStates);
        state.setActiveOperations(activeOperations);
        state.setFrameContext(frameContext);

        return state;
    }

    // Helper methods for analyzing specific termination causes

    private static void analyzeVariableBasedCauses(StringBuilder cause, LoopInfo loopInfo, int iteration,
                                                   LoopIterationTrace trace) {
        if (trace == null || loopInfo.getLoopVariables().isEmpty()) return;

        cause.append("\n  VARIABLE ANALYSIS:");
        for (String varName : loopInfo.getLoopVariables()) {
            List<Object> evolution = trace.getVariableEvolution().get(varName);
            if (evolution != null && evolution.size() >= 2) {
                Object lastValue = evolution.get(evolution.size() - 1);
                Object secondLastValue = evolution.get(evolution.size() - 2);

                if (isNumericallyUnstable(lastValue)) {
                    cause.append("\n    Variable '").append(varName).append("' became unstable: ").append(formatValue(lastValue));
                } else if (valueChangedSuddenly(secondLastValue, lastValue)) {
                    cause.append("\n    Variable '").append(varName).append("' changed suddenly: ");
                    cause.append(formatValue(secondLastValue)).append(" → ").append(formatValue(lastValue));
                }
            }
        }
    }

    private static void analyzeExitConditionCauses(StringBuilder cause, LoopInfo loopInfo, int iteration,
                                                   Map<String, LoopIterationTrace> iterationTraces) {
        cause.append("\n  EXIT ANALYSIS:");

        // Check what led to the exit condition being true
        if (!loopInfo.getExitOperations().isEmpty()) {
            cause.append("\n    Exit operations: ").append(loopInfo.getExitOperations());
        }

        // Look for patterns in recent iterations
        LoopIterationTrace trace = iterationTraces.get(loopInfo.getFrameName());
        if (trace != null && !trace.getIterations().isEmpty()) {
            List<IterationSnapshot> recentSnapshots = trace.getIterations().stream()
                    .filter(snap -> snap.getIteration() >= iteration - 2 && snap.getIteration() <= iteration)
                    .collect(Collectors.toList());

            if (!recentSnapshots.isEmpty()) {
                cause.append("\n    Recent iterations showed: ");
                for (IterationSnapshot snap : recentSnapshots) {
                    cause.append("iter").append(snap.getIteration()).append("(").append(snap.getVariableValues().size()).append(" vars) ");
                }
            }
        }
    }

    private static void analyzeSwitchTerminationCauses(StringBuilder cause, LoopInfo loopInfo, int iteration,
                                                       Map<String, LoopIterationTrace> iterationTraces) {
        cause.append("\n  SWITCH ANALYSIS:");

        if (!loopInfo.getSwitchOperations().isEmpty()) {
            cause.append("\n    Switch operations in loop: ").append(loopInfo.getSwitchOperations());
        }

        // Analyze switch decision patterns
        LoopIterationTrace trace = iterationTraces.get(loopInfo.getFrameName());
        if (trace != null) {
            // Look for switch-related patterns in recent iterations
            cause.append("\n    Switch operations may have taken unexpected branches");
            cause.append("\n    This could indicate: predicate values changed unexpectedly,");
            cause.append("\n    or control flow logic differs from expected patterns");
        }
    }

    private static void analyzeErrorTerminationCauses(StringBuilder cause, LoopInfo loopInfo, int iteration,
                                                      Map<String, LoopIterationTrace> iterationTraces) {
        cause.append("\n  ERROR ANALYSIS:");
        cause.append("\n    Error occurred at iteration ").append(iteration);
        cause.append("\n    This suggests: numerical instability, invalid operations,");
        cause.append("\n    or resource constraints during loop execution");

        // Check for numerical instability patterns
        LoopIterationTrace trace = iterationTraces.get(loopInfo.getFrameName());
        if (trace != null) {
            for (String varName : loopInfo.getLoopVariables()) {
                List<Object> evolution = trace.getVariableEvolution().get(varName);
                if (evolution != null && !evolution.isEmpty()) {
                    Object lastValue = evolution.get(evolution.size() - 1);
                    if (isNumericallyUnstable(lastValue)) {
                        cause.append("\n    Variable '").append(varName).append("' shows instability: ").append(formatValue(lastValue));
                    }
                }
            }
        }
    }

    private static void analyzeTimeoutCauses(StringBuilder cause, LoopInfo loopInfo, int iteration) {
        cause.append("\n  TIMEOUT ANALYSIS:");
        cause.append("\n    Iteration ").append(iteration).append(" exceeded maximum allowed");
        cause.append("\n    This suggests: infinite loop, very slow convergence,");
        cause.append("\n    or incorrect termination conditions");

        if (loopInfo.getExpectedIterations() > 0) {
            cause.append("\n    Expected iterations: ").append(loopInfo.getExpectedIterations());
            cause.append(" | Actual iterations: ").append(iteration);
        }
    }

    private static void analyzeEarlyBreakCauses(StringBuilder cause, LoopInfo loopInfo, int iteration,
                                                Map<String, LoopIterationTrace> iterationTraces) {
        cause.append("\n  EARLY BREAK ANALYSIS:");
        cause.append("\n    Loop was terminated by an early break condition");
        cause.append("\n    This could indicate: optimization stopping criteria met,");
        cause.append("\n    convergence threshold reached, or manual intervention");
    }

    private static void analyzeResourceExhaustionCauses(StringBuilder cause, LoopInfo loopInfo, int iteration) {
        cause.append("\n  RESOURCE EXHAUSTION ANALYSIS:");
        cause.append("\n    Loop consumed too many resources");
        cause.append("\n    This suggests: memory leak, excessive computation,");
        cause.append("\n    or inefficient algorithms within the loop");

        // Add memory information if available
        try {
            Runtime runtime = Runtime.getRuntime();
            long totalMemory = runtime.totalMemory();
            long freeMemory = runtime.freeMemory();
            long usedMemory = totalMemory - freeMemory;

            cause.append("\n    Current memory usage: ").append(usedMemory / 1024 / 1024).append(" MB");
            cause.append(" of ").append(totalMemory / 1024 / 1024).append(" MB");
        } catch (Exception e) {
            cause.append("\n    Memory information unavailable");
        }
    }

    private static void addGeneralEarlyTerminationIndicators(StringBuilder cause, LoopInfo loopInfo, int iteration,
                                                             Map<String, LoopIterationTrace> iterationTraces) {
        cause.append("\n\n  GENERAL INDICATORS:");

        // Check if this was much earlier than expected
        if (loopInfo.getExpectedIterations() > 0 && iteration < loopInfo.getExpectedIterations() * 0.5) {
            cause.append("\n    SIGNIFICANTLY EARLY: Terminated at ").append(iteration);
            cause.append(" iterations (expected ~").append(loopInfo.getExpectedIterations()).append(")");
        }

        // Check for prediction accuracy
        if (!loopInfo.getTerminationPredictions().isEmpty()) {
            TerminationPrediction bestPrediction = loopInfo.getTerminationPredictions()
                    .stream()
                    .max(Comparator.comparingDouble(TerminationPrediction::getConfidence))
                    .orElse(null);

            if (bestPrediction != null) {
                int predictedIteration = bestPrediction.getPredictedTerminationIteration();
                int actualIteration = iteration;
                int difference = Math.abs(predictedIteration - actualIteration);

                cause.append("\n    PREDICTION ACCURACY: Predicted ").append(predictedIteration);
                cause.append(", Actual ").append(actualIteration);
                cause.append(" (difference: ").append(difference).append(" iterations)");

                if (difference > 5) {
                    cause.append("\n    PREDICTION MISS: Large difference suggests unexpected behavior");
                }
            }
        }

        // Check execution time
        long executionTime = System.currentTimeMillis() - loopInfo.getStartTime();
        if (executionTime < 100) { // Less than 100ms
            cause.append("\n    RAPID TERMINATION: Loop completed in ").append(executionTime).append("ms");
        }
    }

    // Utility methods

    private static Object extractValueFromSDValue(SDValue value) {
        if (value == null) return null;

        switch (value.getSdValueType()) {
            case TENSOR:
                return value.getTensorValue();
            case LIST:
                return value.getListValue();
            default:
                return value.toString();
        }
    }

    private static boolean isOperationRelevantToFrame(org.nd4j.autodiff.samediff.internal.SameDiffOp op, String frameName) {
        // This is a simplified check - in practice, you'd need to track frame associations
        // For now, we'll consider all operations potentially relevant
        return true;
    }

    private static boolean isOperationActive(String opName, String frameName, int iteration) {
        // This would need integration with the actual execution state
        // For now, we'll return false as a placeholder
        return false;
    }

    private static String formatValue(Object value) {
        if (value == null) return "null";

        if (value instanceof INDArray) {
            INDArray arr = (INDArray) value;
            if (arr.isScalar()) {
                return String.format("%.6f", arr.getDouble(0));
            } else if (arr.length() <= 5) {
                return arr.toString();
            } else {
                return String.format("Array[%s] (length: %d)", Arrays.toString(arr.shape()), arr.length());
            }
        } else if (value instanceof Number) {
            return String.format("%.6f", ((Number) value).doubleValue());
        } else if (value instanceof Boolean) {
            return value.toString();
        }

        return value.toString();
    }

    private static boolean valueChangedSuddenly(Object oldValue, Object newValue) {
        Double oldNum = extractNumericValue(oldValue);
        Double newNum = extractNumericValue(newValue);

        if (oldNum != null && newNum != null && Math.abs(oldNum) > 1e-10) {
            double changeRatio = Math.abs((newNum - oldNum) / oldNum);
            return changeRatio > 0.5; // Changed by more than 50%
        }

        return false;
    }

    private static Double extractNumericValue(Object value) {
        if (value == null) return null;

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

    private static boolean isNumericallyUnstable(Object value) {
        if (value instanceof Number) {
            double d = ((Number) value).doubleValue();
            return Double.isNaN(d) || Double.isInfinite(d) || Math.abs(d) > 1e10;
        } else if (value instanceof INDArray) {
            INDArray arr = (INDArray) value;
            if (arr.isScalar()) {
                double d = arr.getDouble(0);
                return Double.isNaN(d) || Double.isInfinite(d) || Math.abs(d) > 1e10;
            }
        }
        return false;
    }

    private static double calculateConvergenceRate(List<Double> values) {
        if (values.size() < 2) return 0.0;

        double totalChange = 0.0;
        for (int i = 1; i < values.size(); i++) {
            totalChange += Math.abs(values.get(i) - values.get(i-1));
        }

        return totalChange / (values.size() - 1);
    }
}