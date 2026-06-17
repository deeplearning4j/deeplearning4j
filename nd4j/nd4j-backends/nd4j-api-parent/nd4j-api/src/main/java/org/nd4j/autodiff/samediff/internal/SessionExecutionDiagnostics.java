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

package org.nd4j.autodiff.samediff.internal;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.SameDiffExecutionVisualizer;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.config.SDValue;
import org.nd4j.autodiff.samediff.config.ExecutionResult;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.controlflow.compat.*;

import java.util.*;
import java.util.stream.Collectors;

import static org.nd4j.imports.VariableUtils.stripVarSuffix;

/**
 * Helper class that contains all diagnostic and analysis methods extracted from
 * {@link AbstractSession}. These methods are only invoked on execution failure or
 * when visualization is enabled, and share no mutable state with the hot execution
 * path.
 *
 * <p>Construct once per session execution by passing the relevant state references.
 * All state is read-only from the helper's perspective — it never writes back to
 * {@code nodeValueOutputs} or the dependency tracker.
 */
@Slf4j
public class SessionExecutionDiagnostics {

    private final SameDiff sameDiff;
    private final Map<VarId, SDValue> nodeValueOutputs;
    private final ExecStepDependencyTracker dt;
    /** Supplier of the current frame name so diagnostics always reflect the live value. */
    private final java.util.function.Supplier<String> currentFrameSupplier;

    /**
     * @param sameDiff            The SameDiff graph being executed.
     * @param nodeValueOutputs    The live map of computed variable outputs.
     * @param dt                  The dependency tracker.
     * @param currentFrameSupplier A supplier returning the current execution frame name.
     */
    public SessionExecutionDiagnostics(SameDiff sameDiff,
                                       Map<VarId, SDValue> nodeValueOutputs,
                                       ExecStepDependencyTracker dt,
                                       java.util.function.Supplier<String> currentFrameSupplier) {
        this.sameDiff = sameDiff;
        this.nodeValueOutputs = nodeValueOutputs;
        this.dt = dt;
        this.currentFrameSupplier = currentFrameSupplier;
    }

    // -------------------------------------------------------------------------
    // Switch / Merge analysis
    // -------------------------------------------------------------------------

    /**
     * Analyze Switch operation execution and return detailed results.
     */
    public SwitchResult analyzeSwitchOperation(SameDiffOp op, ExecutionResult opOutputValues,
                                               List<String> inputNames) {
        SwitchResult result = new SwitchResult();
        result.operationName = op.getName();

        if (inputNames != null && inputNames.size() >= 2) {
            result.affectedVariables.addAll(inputNames);

            if (opOutputValues.hasValues()) {
                Map<String, SDValue> outputs = opOutputValues.getValueOutputs();
                for (Map.Entry<String, SDValue> entry : outputs.entrySet()) {
                    if (entry.getValue() != null) {
                        if (entry.getKey().endsWith(":0")) {
                            result.branchTaken = "LEFT";
                            result.outputIndex = 0;
                        } else if (entry.getKey().endsWith(":1")) {
                            result.branchTaken = "RIGHT";
                            result.outputIndex = 1;
                        }
                    }
                }
            } else if (opOutputValues.hasSingle()) {
                for (int i = 0; i < opOutputValues.numResults(); i++) {
                    if (opOutputValues.resultAt(i) != null) {
                        result.branchTaken = (i == 0) ? "LEFT" : "RIGHT";
                        result.outputIndex = i;
                        break;
                    }
                }
            }
        }

        return result;
    }

    /**
     * Analyze Merge operation execution and return detailed results.
     */
    public MergeResult analyzeMergeOperation(SameDiffOp op, ExecutionResult opOutputValues,
                                             Set<VarId> inputs) {
        MergeResult result = new MergeResult();
        result.operationName = op.getName();
        result.totalInputs = inputs != null ? inputs.size() : 0;

        if (opOutputValues.hasValues()) {
            Map<String, SDValue> outputs = opOutputValues.getValueOutputs();
            if (!outputs.isEmpty()) {
                SDValue mergedOutput = outputs.values().iterator().next();
                result.mergedValue = mergedOutput;
                result.selectedInputIndex = 0;
            }
        }

        return result;
    }

    // -------------------------------------------------------------------------
    // Failure context / summary generation
    // -------------------------------------------------------------------------

    /**
     * Generate a brief failure context string for the visualizer.
     */
    public String generateFailureContext(Set<String> allRequired, Set<String> allExecuted,
                                         Map<String, ControlFlowState> controlFlowStates) {
        StringBuilder context = new StringBuilder();
        context.append("EXECUTION_FAILED | ");

        Set<String> remaining = new HashSet<>(allRequired);
        remaining.removeAll(allExecuted);

        context.append("Remaining operations: ").append(remaining.size());
        if (remaining.size() <= 5) {
            context.append(" (").append(String.join(", ", remaining)).append(")");
        }

        long totalSwitches = controlFlowStates.values().stream()
                .mapToLong(state -> state.switchDecisions.size()).sum();
        long totalMerges = controlFlowStates.values().stream()
                .mapToLong(state -> state.mergeDecisions.size()).sum();

        if (totalSwitches > 0 || totalMerges > 0) {
            context.append(" | Control flow: ").append(totalSwitches).append(" switches, ")
                    .append(totalMerges).append(" merges");
        }

        return context.toString();
    }

    /**
     * Generate a comprehensive execution summary string.
     */
    public String generateExecutionSummary(Map<String, ControlFlowState> controlFlowStates,
                                            Set<String> allExecuted,
                                            Map<String, SDValue> outValues) {
        StringBuilder summary = new StringBuilder();
        summary.append("SUCCESS: ").append(outValues.size()).append(" outputs generated");
        summary.append(" | Total operations: ").append(allExecuted.size());

        long totalSwitches = controlFlowStates.values().stream()
                .mapToLong(state -> state.switchDecisions.size()).sum();
        long totalMerges = controlFlowStates.values().stream()
                .mapToLong(state -> state.mergeDecisions.size()).sum();

        if (totalSwitches > 0 || totalMerges > 0) {
            summary.append(" | Control flow: ").append(totalSwitches).append(" switches, ")
                    .append(totalMerges).append(" merges");
        }

        Set<String> uniqueFrames = controlFlowStates.values().stream()
                .map(state -> state.currentFrame)
                .filter(Objects::nonNull)
                .collect(Collectors.toSet());

        if (uniqueFrames.size() > 1) {
            summary.append(" | Frames used: ").append(uniqueFrames.size());
        }

        summary.append(" | Avg ops per control flow op: ");
        if (totalSwitches + totalMerges > 0) {
            summary.append(String.format("%.1f", (double) allExecuted.size() / (totalSwitches + totalMerges)));
        } else {
            summary.append("N/A");
        }

        return summary.toString();
    }

    /**
     * Print detailed control flow analysis at DEBUG level.
     */
    public void printControlFlowAnalysis(Map<String, ControlFlowState> controlFlowStates) {
        if (log.isDebugEnabled()) {
            log.debug("=== CONTROL FLOW ANALYSIS ===");

            for (Map.Entry<String, ControlFlowState> entry : controlFlowStates.entrySet()) {
                String opName = entry.getKey();
                ControlFlowState state = entry.getValue();

                if (!state.switchDecisions.isEmpty() || !state.mergeDecisions.isEmpty() ||
                        !state.frameTransitions.isEmpty()) {

                    log.debug("Operation: {}", opName);
                    log.debug("  Execution count: {}", state.executionCount);
                    log.debug("  Current frame: {}:{}", state.currentFrame, state.currentIteration);

                    if (!state.frameTransitions.isEmpty()) {
                        log.debug("  Frame transitions: {}", String.join(" → ", state.frameTransitions));
                    }

                    if (!state.switchDecisions.isEmpty()) {
                        log.debug("  Switch decisions:");
                        for (SwitchResult switch_result : state.switchDecisions) {
                            log.debug("    {}", switch_result);
                        }
                    }

                    if (!state.mergeDecisions.isEmpty()) {
                        log.debug("  Merge decisions:");
                        for (MergeResult merge : state.mergeDecisions) {
                            log.debug("    {}", merge);
                        }
                    }

                    log.debug("");
                }
            }

            long totalControlFlowOps = controlFlowStates.values().stream()
                    .mapToLong(state -> state.switchDecisions.size() + state.mergeDecisions.size())
                    .sum();

            if (totalControlFlowOps > 0) {
                log.debug("=== CONTROL FLOW SUMMARY ===");
                log.debug("Total control flow operations: {}", totalControlFlowOps);

                Map<String, Long> branchCounts = controlFlowStates.values().stream()
                        .flatMap(state -> state.switchDecisions.stream())
                        .collect(Collectors.groupingBy(
                                result -> result.branchTaken != null ? result.branchTaken : "UNKNOWN",
                                Collectors.counting()
                        ));

                if (!branchCounts.isEmpty()) {
                    log.debug("Branch distribution: {}", branchCounts);
                }

                Set<String> framesUsed = controlFlowStates.values().stream()
                        .map(state -> state.currentFrame)
                        .filter(Objects::nonNull)
                        .collect(Collectors.toSet());

                log.debug("Frames involved in control flow: {}", framesUsed);
                log.debug("===============================");
            }
        }
    }

    // -------------------------------------------------------------------------
    // ExecType / FrameIter conversion helpers (for visualizer)
    // -------------------------------------------------------------------------

    /**
     * Convert an ExecType to the same ExecType (identity mapping kept for API
     * symmetry with the original private helper).
     */
    public ExecType convertExecType(ExecType originalType) {
        switch (originalType) {
            case OP: return ExecType.OP;
            case VARIABLE: return ExecType.VARIABLE;
            case CONSTANT: return ExecType.CONSTANT;
            case PLACEHOLDER: return ExecType.PLACEHOLDER;
            case SWITCH_L: return ExecType.SWITCH_L;
            case SWITCH_R: return ExecType.SWITCH_R;
            case EXEC_START: return ExecType.EXEC_START;
            case CONTROL_DEP: return ExecType.CONTROL_DEP;
            default: return ExecType.OP;
        }
    }

    /**
     * Deep-copy a FrameIter (preserving parent chain).
     */
    public FrameIter convertFrameIter(FrameIter originalFrameIter) {
        if (originalFrameIter == null) {
            return null;
        }
        FrameIter parentFrame = null;
        if (originalFrameIter.getParentFrame() != null) {
            parentFrame = convertFrameIter(originalFrameIter.getParentFrame());
        }
        return new FrameIter(
                originalFrameIter.getFrame(),
                originalFrameIter.getIteration(),
                parentFrame
        );
    }

    // -------------------------------------------------------------------------
    // Step input / output name helpers
    // -------------------------------------------------------------------------

    /**
     * Get the list of input names for an execution step.
     */
    public List<String> getStepInputs(ExecStep es) {
        List<String> inputs = new ArrayList<>();
        if (es.getType() == ExecType.OP) {
            SameDiffOp op = sameDiff.getOps().get(es.getName());
            if (op != null) {
                List<String> inputNames = op.getInputsToOp();
                if (inputNames != null) {
                    inputs.addAll(inputNames);
                }
            }
        }
        return inputs;
    }

    /**
     * Get the list of output names for an execution step.
     */
    public List<String> getStepOutputs(ExecStep es) {
        List<String> outputs = new ArrayList<>();
        if (es.getType() == ExecType.OP) {
            SameDiffOp op = sameDiff.getOps().get(es.getName());
            if (op != null) {
                List<String> outputNames = op.getOutputsOfOp();
                if (outputNames != null) {
                    outputs.addAll(outputNames);
                }
            }
        } else if (es.getType() == ExecType.VARIABLE || es.getType() == ExecType.CONSTANT
                || es.getType() == ExecType.PLACEHOLDER) {
            outputs.add(es.getName());
        }
        return outputs;
    }

    /**
     * Collect input names for visualization from VarIds and constant/placeholder names.
     */
    public List<String> getInputNamesForVisualization(Set<VarId> inputs, Set<String> constAndPhInputs) {
        List<String> result = new ArrayList<>();
        if (inputs != null) {
            for (VarId vid : inputs) {
                result.add(vid.getVariable());
            }
        }
        if (constAndPhInputs != null) {
            result.addAll(constAndPhInputs);
        }
        return result;
    }

    // -------------------------------------------------------------------------
    // Comprehensive failure report
    // -------------------------------------------------------------------------

    /**
     * Generate a comprehensive failure report string.
     */
    public String generateFailureReport(Set<String> allRequired, Set<String> allExecuted,
                                         Map<String, ControlFlowState> controlFlowStates, int step) {
        StringBuilder report = new StringBuilder();

        report.append("=== EXECUTION FAILURE DETAILED REPORT ===\n");
        report.append("Timestamp: ").append(new java.util.Date()).append("\n");
        report.append("Failed at step: ").append(step).append("\n");
        report.append("Progress: ").append(allExecuted.size()).append("/").append(allRequired.size())
                .append(" (").append(String.format("%.1f%%", (double) allExecuted.size() / allRequired.size() * 100))
                .append(")\n\n");

        Set<String> remaining = new HashSet<>(allRequired);
        remaining.removeAll(allExecuted);

        report.append("=== REMAINING OPERATIONS (").append(remaining.size()).append(") ===\n");
        for (String remainingOp : remaining) {
            report.append(analyzeRemainingOperation(remainingOp)).append("\n\n");
        }

        if (!controlFlowStates.isEmpty()) {
            report.append("=== CONTROL FLOW ANALYSIS ===\n");
            for (Map.Entry<String, ControlFlowState> entry : controlFlowStates.entrySet()) {
                ControlFlowState state = entry.getValue();
                if (!state.switchDecisions.isEmpty() || !state.mergeDecisions.isEmpty()) {
                    report.append("Operation: ").append(entry.getKey()).append("\n");
                    report.append("  Executions: ").append(state.executionCount).append("\n");
                    report.append("  Frame: ").append(state.currentFrame).append(":").append(state.currentIteration).append("\n");

                    if (!state.switchDecisions.isEmpty()) {
                        report.append("  Switch decisions: ").append(state.switchDecisions.size()).append("\n");
                        for (SwitchResult result : state.switchDecisions) {
                            report.append("    ").append(result).append("\n");
                        }
                    }

                    if (!state.mergeDecisions.isEmpty()) {
                        report.append("  Merge decisions: ").append(state.mergeDecisions.size()).append("\n");
                        for (MergeResult result : state.mergeDecisions) {
                            report.append("    ").append(result).append("\n");
                        }
                    }
                    report.append("\n");
                }
            }
        }

        report.append("=== DEPENDENCY TRACKER STATE ===\n");
        report.append("Total satisfied dependencies: ").append(dt.getSatisfiedDependencies().size()).append("\n");
        report.append("Has new satisfied: ").append(dt.hasNewAllSatisfied()).append("\n");
        report.append("All satisfied count: ").append(dt.getAllSatisfied().size()).append("\n");
        report.append("Queue size: ").append(dt.getAllSatisfiedQueue().size()).append("\n");

        int opCount = 0, varCount = 0, constCount = 0, placeholderCount = 0, otherCount = 0;
        for (Object satisfied : dt.getAllSatisfied()) {
            if (satisfied instanceof ExecStep) {
                ExecStep execStep = (ExecStep) satisfied;
                switch (execStep.getType()) {
                    case OP:
                    case SWITCH_L:
                    case SWITCH_R:
                        opCount++;
                        break;
                    case VARIABLE:
                        varCount++;
                        break;
                    case CONSTANT:
                        constCount++;
                        break;
                    case PLACEHOLDER:
                        placeholderCount++;
                        break;
                    default:
                        otherCount++;
                        break;
                }
            }
        }
        report.append("Satisfied breakdown: ").append(opCount).append(" ops, ")
                .append(varCount).append(" vars, ").append(constCount).append(" constants, ")
                .append(placeholderCount).append(" placeholders, ").append(otherCount).append(" other\n");

        report.append("\n=== VARIABLE AVAILABILITY ===\n");
        Set<String> allVariables = sameDiff.variableMap().keySet();
        int availableCount = 0;
        for (String var : allVariables) {
            if (isVariableAvailable(var)) {
                availableCount++;
            }
        }
        report.append("Available variables: ").append(availableCount).append("/").append(allVariables.size()).append("\n");

        List<String> unavailableVars = new ArrayList<>();
        for (String var : allVariables) {
            if (!isVariableAvailable(var)) {
                unavailableVars.add(var);
                if (unavailableVars.size() >= 10) break;
            }
        }
        if (!unavailableVars.isEmpty()) {
            report.append("Some unavailable variables: ").append(unavailableVars).append("\n");
        }

        String frame = getCurrentFrame();
        if (frame != null) {
            report.append("\n=== FRAME INFORMATION ===\n");
            report.append("Current frame: ").append(frame).append("\n");

            Set<String> frameVars = getVariablesInCurrentFrame();
            if (frameVars != null) {
                report.append("Variables in current frame: ").append(frameVars.size()).append("\n");
            }
        }

        report.append("\n=== END FAILURE REPORT ===\n");

        return report.toString();
    }

    // -------------------------------------------------------------------------
    // Dependency info
    // -------------------------------------------------------------------------

    /**
     * Get comprehensive dependency information for debugging a given op.
     */
    public Map<String, Object> getDebugDependencyInfo(String opName) {
        Map<String, Object> info = new HashMap<>();

        SameDiffOp op = sameDiff.getOps().get(opName);
        if (op != null) {
            info.put("operationType", op.getOp().getClass().getSimpleName());
            info.put("inputs", op.getInputsToOp());
            info.put("outputs", op.getOutputsOfOp());
            info.put("controlDeps", op.getControlDeps());
        }

        List<Map<String, Object>> stepInfo = new ArrayList<>();

        for (Object satisfied : dt.getAllSatisfied()) {
            if (satisfied instanceof ExecStep) {
                ExecStep execStep = (ExecStep) satisfied;
                if (execStep.getName().equals(opName)) {
                    Map<String, Object> stepData = new HashMap<>();
                    stepData.put("type", execStep.getType());
                    stepData.put("satisfied", true);
                    stepData.put("location", "satisfied_set");
                    if (execStep.getFrameIter() != null) {
                        stepData.put("frame", execStep.getFrameIter().getFrame());
                        stepData.put("iteration", execStep.getFrameIter().getIteration());
                    }
                    stepInfo.add(stepData);
                }
            }
        }

        for (Object queued : dt.getAllSatisfiedQueue()) {
            if (queued instanceof ExecStep) {
                ExecStep execStep = (ExecStep) queued;
                if (execStep.getName().equals(opName)) {
                    Map<String, Object> stepData = new HashMap<>();
                    stepData.put("type", execStep.getType());
                    stepData.put("satisfied", false);
                    stepData.put("location", "queue");
                    if (execStep.getFrameIter() != null) {
                        stepData.put("frame", execStep.getFrameIter().getFrame());
                        stepData.put("iteration", execStep.getFrameIter().getIteration());
                    }
                    stepInfo.add(stepData);
                }
            }
        }

        info.put("executionSteps", stepInfo);

        if (op != null && op.getInputsToOp() != null) {
            Map<String, Object> inputInfo = new HashMap<>();
            for (String input : op.getInputsToOp()) {
                Map<String, Object> inputData = new HashMap<>();
                inputData.put("available", isVariableAvailable(input));
                String producer = findVariableProducer(input);
                inputData.put("producer", producer);
                if (producer != null) {
                    inputData.put("producerExecuted", isOperationExecuted(producer));
                }
                inputInfo.put(input, inputData);
            }
            info.put("inputDetails", inputInfo);
        }

        Map<String, Object> trackerInfo = new HashMap<>();
        trackerInfo.put("totalSatisfiedDeps", dt.getSatisfiedDependencies().size());
        trackerInfo.put("hasNewSatisfied", dt.hasNewAllSatisfied());
        trackerInfo.put("allSatisfiedCount", dt.getAllSatisfied().size());
        trackerInfo.put("queueSize", dt.getAllSatisfiedQueue().size());
        info.put("dependencyTracker", trackerInfo);

        return info;
    }

    // -------------------------------------------------------------------------
    // Control flow context analysis
    // -------------------------------------------------------------------------

    /**
     * Analyze if an item is part of a control flow structure.
     */
    public String analyzeControlFlowContext(String itemName) {
        StringBuilder context = new StringBuilder();

        List<String> controlFlowConsumers = findControlFlowConsumers(itemName);
        if (!controlFlowConsumers.isEmpty()) {
            context.append("CONTROL FLOW CONTEXT: Variable feeds into control flow operations: ");
            context.append(controlFlowConsumers);

            for (String consumer : controlFlowConsumers) {
                SameDiffOp consumerOp = sameDiff.getOps().get(consumer);
                if (consumerOp != null) {
                    DifferentialFunction func = consumerOp.getOp();
                    context.append("\n  Consumer: ").append(consumer)
                            .append(" (").append(func.getClass().getSimpleName()).append(")");

                    if (func instanceof Switch) {
                        context.append(" - SWITCH OPERATION");
                        Switch switchOp = (Switch) func;
                        if (switchOp.getPredicate() != null && switchOp.getPredicate().name().equals(itemName)) {
                            context.append(" - THIS IS THE PREDICATE!");
                        }
                    } else if (func instanceof Merge) {
                        context.append(" - MERGE OPERATION");
                    } else if (func instanceof NextIteration) {
                        context.append(" - NEXT ITERATION OPERATION");
                    } else if (func instanceof LoopCond) {
                        context.append(" - LOOP CONDITION OPERATION");
                    }

                    boolean consumerExecuted = isOperationExecuted(consumer);
                    context.append(" - Executed: ").append(consumerExecuted ? "✓" : "✗");
                }
            }
        }

        return context.length() > 0 ? context.toString() : null;
    }

    /**
     * Analyze potential loop condition issues for an item.
     */
    public String analyzeLoopConditionIssues(String itemName) {
        StringBuilder analysis = new StringBuilder();

        String producer = findVariableProducer(itemName);
        if (producer != null) {
            SameDiffOp producerOp = sameDiff.getOps().get(producer);
            if (producerOp != null) {
                String producerFrame = getOperationFrame(producer);
                if (!AbstractSession.OUTER_FRAME.equals(producerFrame)) {
                    analysis.append("LOOP CONDITION ANALYSIS:");
                    analysis.append("\n  Producer is in frame: ").append(producerFrame);

                    List<String> loopCondOps = findLoopConditionOperations(producerFrame);
                    if (!loopCondOps.isEmpty()) {
                        analysis.append("\n  Loop condition operations in frame: ").append(loopCondOps);

                        for (String loopCondOp : loopCondOps) {
                            boolean executed = isOperationExecuted(loopCondOp);
                            analysis.append("\n    ").append(loopCondOp)
                                    .append(" executed: ").append(executed ? "✓" : "✗");

                            if (!executed) {
                                analysis.append(" <- POTENTIAL ISSUE: Loop condition not evaluated");

                                SameDiffOp loopOp = sameDiff.getOps().get(loopCondOp);
                                if (loopOp != null) {
                                    List<String> loopInputs = loopOp.getInputsToOp();
                                    if (loopInputs != null) {
                                        analysis.append("\n      Loop condition inputs: ");
                                        for (String input : loopInputs) {
                                            boolean inputAvailable = isVariableAvailable(input);
                                            analysis.append(input).append(inputAvailable ? "✓" : "✗").append(" ");
                                        }
                                    }
                                }
                            }
                        }
                    }

                    String infiniteLoopCheck = checkForInfiniteLoop(producerFrame);
                    if (infiniteLoopCheck != null) {
                        analysis.append("\n  ").append(infiniteLoopCheck);
                    }
                }
            }
        }

        return analysis.length() > 0 ? analysis.toString() : null;
    }

    // -------------------------------------------------------------------------
    // Remaining operation analysis
    // -------------------------------------------------------------------------

    /**
     * Analyze a single remaining item (variable or operation) for the failure report.
     */
    public String analyzeRemainingOperation(String itemName) {
        StringBuilder analysis = new StringBuilder();

        if (sameDiff.variableMap().containsKey(itemName)) {
            analysis.append("Variable: ").append(itemName);

            Variable var = sameDiff.getVariables().get(itemName);
            if (var != null) {
                SDVariable sdVar = var.getVariable();
                analysis.append(" (").append(sdVar.getVariableType()).append(")");

                boolean available = isVariableAvailable(itemName);
                analysis.append("\n  Available: ").append(available ? "✓" : "✗");

                if (!available) {
                    String producer = findVariableProducer(itemName);
                    if (producer != null) {
                        analysis.append("\n  Producer operation: '").append(producer).append("'");

                        boolean producerExecuted = isOperationExecuted(producer);
                        analysis.append("\n  Producer executed: ").append(producerExecuted ? "✓" : "✗");

                        if (!producerExecuted) {
                            SameDiffOp producerOp = sameDiff.getOps().get(producer);
                            if (producerOp != null) {
                                analysis.append("\n  Producer analysis:");
                                analysis.append("\n    Type: ").append(producerOp.getOp().getClass().getSimpleName());

                                List<String> producerInputs = producerOp.getInputsToOp();
                                if (producerInputs != null && !producerInputs.isEmpty()) {
                                    analysis.append("\n    Producer inputs: ");
                                    for (String input : producerInputs) {
                                        boolean inputAvailable = isVariableAvailable(input);
                                        analysis.append(input).append(inputAvailable ? "✓" : "✗").append(" ");
                                    }
                                }

                                boolean inQueue = false;
                                for (Object queued : dt.getAllSatisfiedQueue()) {
                                    if (queued instanceof ExecStep) {
                                        ExecStep execStep = (ExecStep) queued;
                                        if (execStep.getName().equals(producer)) {
                                            inQueue = true;
                                            break;
                                        }
                                    }
                                }
                                analysis.append("\n    Producer in execution queue: ").append(inQueue ? "✓" : "✗");

                            } else {
                                analysis.append("\n    ERROR: Producer operation not found in graph");
                            }
                        }
                    } else {
                        analysis.append("\n  ERROR: No producer operation found for this variable");
                    }
                } else {
                    analysis.append("\n  WARNING: Variable is available but still in remaining set");
                }
            }

            analysis.append("\n\n=== CONTROL FLOW ANALYSIS ===");

            String controlFlowContext = analyzeControlFlowContext(itemName);
            if (controlFlowContext != null) {
                analysis.append("\n").append(controlFlowContext);
            }

            String loopAnalysis = analyzeLoopConditionIssues(itemName);
            if (loopAnalysis != null) {
                analysis.append("\n").append(loopAnalysis);
            }

            return analysis.toString();
        }

        SameDiffOp op = sameDiff.getOps().get(itemName);
        if (op != null) {
            analysis.append("Operation: ").append(itemName);

            DifferentialFunction opFunc = op.getOp();
            analysis.append(" (").append(opFunc.getClass().getSimpleName()).append(")");

            List<String> inputs = op.getInputsToOp();
            if (inputs != null && !inputs.isEmpty()) {
                analysis.append("\n  Inputs: ");
                for (int i = 0; i < inputs.size(); i++) {
                    String input = inputs.get(i);
                    boolean available = isVariableAvailable(input);
                    analysis.append(input).append(available ? "✓" : "✗");
                    if (i < inputs.size() - 1) analysis.append(", ");
                }

                List<String> missingInputs = new ArrayList<>();
                for (String input : inputs) {
                    if (!isVariableAvailable(input)) {
                        missingInputs.add(input);
                    }
                }

                if (!missingInputs.isEmpty()) {
                    analysis.append("\n  Missing inputs: ").append(missingInputs);
                }
            } else {
                analysis.append("\n  No inputs required");
            }

            boolean executed = isOperationExecuted(itemName);
            analysis.append("\n  Executed: ").append(executed ? "✓" : "✗");

            if (!executed) {
                boolean foundInQueue = false;
                for (Object queued : dt.getAllSatisfiedQueue()) {
                    if (queued instanceof ExecStep) {
                        ExecStep execStep = (ExecStep) queued;
                        if (execStep.getName().equals(itemName)) {
                            foundInQueue = true;
                            break;
                        }
                    }
                }
                analysis.append("\n  In execution queue: ").append(foundInQueue ? "✓" : "✗");
            }

            return analysis.toString();
        }

        analysis.append("Unknown item: ").append(itemName);
        analysis.append("\n  ERROR: Item not found in variables or operations");

        return analysis.toString();
    }

    // -------------------------------------------------------------------------
    // Variable / operation state queries
    // -------------------------------------------------------------------------

    /**
     * Find the operation that produces a given variable.
     */
    public String findVariableProducer(String varName) {
        if (sameDiff.getVariables().containsKey(varName)) {
            Variable var = sameDiff.getVariables().get(varName);
            if (var.getVariable().isConstant() || var.getVariable().getVariableType() == VariableType.VARIABLE) {
                return varName;
            }
        }

        for (Map.Entry<String, SameDiffOp> entry : sameDiff.getOps().entrySet()) {
            SameDiffOp op = entry.getValue();
            List<String> outputs = op.getOutputsOfOp();
            if (outputs != null && outputs.contains(varName)) {
                return entry.getKey();
            }
        }

        return null;
    }

    /**
     * Check if a variable is available (non-null) in the current execution context.
     */
    public boolean isVariableAvailable(String varName) {
        for (VarId vid : nodeValueOutputs.keySet()) {
            if (vid.getVariable().equals(varName)) {
                SDValue value = nodeValueOutputs.get(vid);
                return value != null;
            }
        }
        return false;
    }

    /**
     * Check if an operation has been executed (appears in the satisfied set).
     */
    public boolean isOperationExecuted(String opName) {
        for (Object satisfied : dt.getAllSatisfied()) {
            if (satisfied instanceof ExecStep) {
                ExecStep execStep = (ExecStep) satisfied;
                if (execStep.getName().equals(opName) &&
                        (execStep.getType() == ExecType.OP
                                || execStep.getType() == ExecType.SWITCH_L
                                || execStep.getType() == ExecType.SWITCH_R)) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * Check if there is a direct dependency path from {@code from} to {@code to}.
     */
    public boolean hasDependencyPath(String from, String to) {
        if (from.equals(to)) {
            return true;
        }

        SameDiffOp fromOp = sameDiff.getOps().get(from);
        if (fromOp != null) {
            List<String> inputs = fromOp.getInputsToOp();
            if (inputs != null) {
                for (String input : inputs) {
                    String producer = findVariableProducer(input);
                    if (to.equals(producer)) {
                        return true;
                    }
                }
            }
        }

        return false;
    }

    /**
     * Log (at ERROR) detailed execution step status for an op.
     */
    public void analyzeExecutionStepStatus(String opName) {
        log.error("    Execution step analysis for '{}':", opName);

        boolean foundStep = false;
        for (Object satisfied : dt.getAllSatisfied()) {
            if (satisfied instanceof ExecStep) {
                ExecStep execStep = (ExecStep) satisfied;
                if (execStep.getName().equals(opName)) {
                    foundStep = true;
                    boolean isSatisfied = dt.isSatisfied(execStep);

                    log.error("      Step {}: satisfied={}, frame={}:{}",
                            execStep.getType(), isSatisfied,
                            execStep.getFrameIter() != null ? execStep.getFrameIter().getFrame() : "null",
                            execStep.getFrameIter() != null ? execStep.getFrameIter().getIteration() : "null");

                    DependencyList<ExecStep, ExecStep> deps = dt.getDependencies(execStep);
                    if (deps != null && deps.getDependencies() != null) {
                        for (ExecStep dep : deps.getDependencies()) {
                            boolean depSatisfied = dt.isSatisfied(dep);
                            log.error("        Depends on {}: satisfied={}", dep.getName(), depSatisfied);
                        }
                    }
                }
            }
        }

        if (!foundStep) {
            log.error("      No execution steps found in satisfied set");
        }
    }

    /**
     * Get the current value of a variable (across any frame/iteration).
     */
    public Object getVariableValue(String varName) {
        for (Map.Entry<VarId, SDValue> entry : nodeValueOutputs.entrySet()) {
            if (entry.getKey().getVariable().equals(varName)) {
                SDValue value = entry.getValue();
                if (value != null) {
                    switch (value.getSdValueType()) {
                        case TENSOR:
                            return value.getTensorValue();
                        case LIST:
                            return value.getListValue();
                        default:
                            return value.toString();
                    }
                }
            }
        }

        if (sameDiff.getConstantArrays().hasArray(varName)) {
            return sameDiff.getConstantArrays().getArray(varName);
        }

        return null;
    }

    /**
     * Check if a control dependency (by op name) is satisfied.
     */
    public boolean isControlDependencySatisfied(String controlDep) {
        return isOperationExecuted(controlDep);
    }

    /**
     * Get the current execution frame name.
     */
    public String getCurrentFrame() {
        String frame = currentFrameSupplier.get();
        return frame != null ? frame : AbstractSession.OUTER_FRAME;
    }

    /**
     * Determine the frame for a given operation from the dependency tracker state.
     */
    public String getOperationFrame(String opName) {
        SameDiffOp op = sameDiff.getOps().get(opName);
        if (op != null && op.getOp() instanceof Enter) {
            return ((Enter) op.getOp()).getFrameName();
        }

        for (Object satisfied : dt.getAllSatisfied()) {
            if (satisfied instanceof ExecStep) {
                ExecStep execStep = (ExecStep) satisfied;
                if (execStep.getName().equals(opName) && execStep.getFrameIter() != null) {
                    return execStep.getFrameIter().getFrame();
                }
            }
        }

        for (Object queued : dt.getAllSatisfiedQueue()) {
            if (queued instanceof ExecStep) {
                ExecStep execStep = (ExecStep) queued;
                if (execStep.getName().equals(opName) && execStep.getFrameIter() != null) {
                    return execStep.getFrameIter().getFrame();
                }
            }
        }

        return AbstractSession.OUTER_FRAME;
    }

    /**
     * Check if a named dependency (variable or op) is satisfied.
     */
    public boolean isDependencySatisfied(String dep) {
        if (sameDiff.getVariables().containsKey(dep)) {
            return isVariableAvailable(dep);
        }
        if (sameDiff.getOps().containsKey(dep)) {
            return isOperationExecuted(dep);
        }
        return false;
    }

    /**
     * Get the set of direct dependencies (inputs + control deps) for an op.
     */
    public Set<String> getDependenciesFor(String opName) {
        Set<String> dependencies = new HashSet<>();

        SameDiffOp op = sameDiff.getOps().get(opName);
        if (op != null) {
            List<String> inputs = op.getInputsToOp();
            if (inputs != null) {
                dependencies.addAll(inputs);
            }

            List<String> controlDeps = op.getControlDeps();
            if (controlDeps != null) {
                dependencies.addAll(controlDeps);
            }
        }

        return dependencies;
    }

    /**
     * Get all variable names that are available in the current frame.
     */
    public Set<String> getVariablesInCurrentFrame() {
        Set<String> frameVariables = new HashSet<>();
        String frame = getCurrentFrame();

        for (VarId vid : nodeValueOutputs.keySet()) {
            if (frame.equals(vid.getFrame())) {
                frameVariables.add(vid.getVariable());
            }
        }

        return frameVariables;
    }

    /**
     * Find all execution steps related to a given op name in the tracker.
     */
    public List<ExecStep> findExecutionSteps(String opName) {
        List<ExecStep> steps = new ArrayList<>();

        for (Object satisfied : dt.getAllSatisfied()) {
            if (satisfied instanceof ExecStep) {
                ExecStep execStep = (ExecStep) satisfied;
                if (execStep.getName().equals(opName)) {
                    steps.add(execStep);
                }
            }
        }

        for (Object queued : dt.getAllSatisfiedQueue()) {
            if (queued instanceof ExecStep) {
                ExecStep execStep = (ExecStep) queued;
                if (execStep.getName().equals(opName)) {
                    steps.add(execStep);
                }
            }
        }

        return steps;
    }

    /**
     * Log (at ERROR) the overall dependency graph state.
     */
    public void analyzeDependencyGraphState(Set<String> remainingOps) {
        log.error("Dependency tracker state:");
        log.error("  Total satisfied dependencies: {}", dt.getSatisfiedDependencies().size());
        log.error("  Steps with new satisfied dependencies: {}", dt.hasNewAllSatisfied());
        log.error("  All satisfied count: {}", dt.getAllSatisfied().size());
        log.error("  Queue size: {}", dt.getAllSatisfiedQueue().size());

        int opCount = 0, varCount = 0, constCount = 0, placeholderCount = 0;
        for (Object satisfied : dt.getAllSatisfied()) {
            if (satisfied instanceof ExecStep) {
                ExecStep execStep = (ExecStep) satisfied;
                switch (execStep.getType()) {
                    case OP:
                    case SWITCH_L:
                    case SWITCH_R:
                        opCount++;
                        break;
                    case VARIABLE:
                        varCount++;
                        break;
                    case CONSTANT:
                        constCount++;
                        break;
                    case PLACEHOLDER:
                        placeholderCount++;
                        break;
                }
            }
        }

        log.error("  Satisfied breakdown: {} ops, {} vars, {} constants, {} placeholders",
                opCount, varCount, constCount, placeholderCount);

        if (remainingOps.size() > 1) {
            log.error("Multiple operations remaining - potential circular dependency or missing prerequisites");
        }
    }

    // -------------------------------------------------------------------------
    // execFailed
    // -------------------------------------------------------------------------

    /**
     * Build the failure message and throw {@link IllegalStateException}.
     *
     * @param visualizer   The visualizer (may be null when visualization is disabled).
     */
    public void execFailed(Set<String> userRequestedUnique, Map<String, SDValue> outValues,
                           Set<String> allRequired, Set<String> allExecuted, int step,
                           SameDiffExecutionVisualizer visualizer) {

        Set<String> remaining = new HashSet<>(allRequired);
        remaining.removeAll(allExecuted);

        StringBuilder sb = new StringBuilder();
        sb.append("Execution failed at step ").append(step).append("\n");
        sb.append("Total operations required: ").append(allRequired.size()).append("\n");
        sb.append("Operations completed: ").append(allExecuted.size()).append("\n");
        sb.append("Operations remaining: ").append(remaining.size()).append("\n");

        if (remaining.size() <= 10) {
            sb.append("Remaining operations: ").append(remaining).append("\n");
        }

        sb.append("Dependency tracker state:\n");
        sb.append("  Total satisfied dependencies: ").append(dt.getSatisfiedDependencies().size()).append("\n");
        sb.append("  Has new satisfied: ").append(dt.hasNewAllSatisfied()).append("\n");
        sb.append("  All satisfied count: ").append(dt.getAllSatisfied().size()).append("\n");

        sb.append("\n=== FAILURE PATTERN DETECTION ===\n");
        if (remaining.size() == 1) {
            sb.append("PATTERN: Single stuck item - likely control flow issue\n");
        } else if (remaining.size() > 1) {
            sb.append("PATTERN: Multiple stuck items - possible dependency cycle\n");
        }

        List<String> unexecutedControlFlow = findUnexecutedControlFlowOperations();
        if (!unexecutedControlFlow.isEmpty()) {
            sb.append("CRITICAL: Unexecuted control flow operations: ").append(unexecutedControlFlow).append("\n");
        }

        sb.append("\n=== DETAILED ANALYSIS ===\n");
        for (String remainingItem : remaining) {
            sb.append(analyzeRemainingOperation(remainingItem)).append("\n\n");
        }

        if (visualizer != null) {
            visualizer.printCompleteAnalysisReport();
        }

        throw new IllegalStateException(sb.toString());
    }

    // -------------------------------------------------------------------------
    // Control flow finders
    // -------------------------------------------------------------------------

    /**
     * Find control flow operations (Switch/Merge/Enter/Exit/NextIteration/LoopCond)
     * that have not yet been executed.
     */
    public List<String> findUnexecutedControlFlowOperations() {
        List<String> unexecuted = new ArrayList<>();

        for (Map.Entry<String, SameDiffOp> entry : sameDiff.getOps().entrySet()) {
            String opName = entry.getKey();
            SameDiffOp op = entry.getValue();
            DifferentialFunction func = op.getOp();

            if (func instanceof Switch || func instanceof Merge ||
                    func instanceof Enter || func instanceof Exit ||
                    func instanceof NextIteration || func instanceof LoopCond) {

                if (!isOperationExecuted(opName)) {
                    unexecuted.add(opName + "(" + func.getClass().getSimpleName() + ")");
                }
            }
        }

        return unexecuted;
    }

    /**
     * Find operations that consume a variable and are control flow operations.
     */
    public List<String> findControlFlowConsumers(String variableName) {
        List<String> consumers = new ArrayList<>();

        for (Map.Entry<String, SameDiffOp> entry : sameDiff.getOps().entrySet()) {
            SameDiffOp op = entry.getValue();
            List<String> inputs = op.getInputsToOp();

            if (inputs != null && inputs.contains(variableName)) {
                DifferentialFunction func = op.getOp();
                if (func instanceof Switch || func instanceof Merge ||
                        func instanceof NextIteration || func instanceof LoopCond) {
                    consumers.add(entry.getKey());
                }
            }
        }

        return consumers;
    }

    /**
     * Find loop condition operations in a specific frame.
     */
    public List<String> findLoopConditionOperations(String frameName) {
        List<String> loopCondOps = new ArrayList<>();

        for (Map.Entry<String, SameDiffOp> entry : sameDiff.getOps().entrySet()) {
            SameDiffOp op = entry.getValue();
            if (op.getOp() instanceof LoopCond) {
                String opFrame = getOperationFrame(entry.getKey());
                if (frameName.equals(opFrame)) {
                    loopCondOps.add(entry.getKey());
                }
            }
        }

        return loopCondOps;
    }

    /**
     * Check for infinite loop indicators in a frame.
     */
    public String checkForInfiniteLoop(String frameName) {
        int nextIterCount = 0;
        for (Map.Entry<String, SameDiffOp> entry : sameDiff.getOps().entrySet()) {
            SameDiffOp op = entry.getValue();
            if (op.getOp() instanceof NextIteration) {
                String opFrame = getOperationFrame(entry.getKey());
                if (frameName.equals(opFrame)) {
                    nextIterCount++;
                }
            }
        }

        if (nextIterCount > 0) {
            return "POTENTIAL INFINITE LOOP: Frame has " + nextIterCount + " NextIteration operations";
        }

        return null;
    }

    // -------------------------------------------------------------------------
    // Detailed context analysis (verbose path)
    // -------------------------------------------------------------------------

    /**
     * Detailed control flow context analysis including per-frame values.
     */
    public String analyzeControlFlowContextDetailed(String itemName) {
        StringBuilder context = new StringBuilder();

        List<String> controlFlowConsumers = findControlFlowConsumers(itemName);
        if (!controlFlowConsumers.isEmpty()) {
            context.append("CONTROL FLOW CONTEXT: Variable feeds into control flow operations");

            List<VarId> instances = getVariableInstances(itemName);
            if (!instances.isEmpty()) {
                context.append("\n  Variable values across frames/iterations:");
                for (VarId vid : instances) {
                    SDValue value = nodeValueOutputs.get(vid);
                    context.append("\n    Frame: ").append(vid.getFrame())
                            .append(", Iter: ").append(vid.getIteration())
                            .append(", Value: ").append(formatValue(value));
                }
            }

            for (String consumer : controlFlowConsumers) {
                SameDiffOp consumerOp = sameDiff.getOps().get(consumer);
                if (consumerOp != null) {
                    DifferentialFunction func = consumerOp.getOp();
                    context.append("\n  Consumer: ").append(consumer)
                            .append(" (").append(func.getClass().getSimpleName()).append(")");

                    boolean consumerExecuted = isOperationExecuted(consumer);
                    context.append("\n    Executed: ").append(consumerExecuted ? "✓" : "✗");

                    context.append("\n    Execution steps:");
                    boolean foundConsumerSteps = false;

                    for (Object satisfied : dt.getAllSatisfied()) {
                        if (satisfied instanceof ExecStep) {
                            ExecStep execStep = (ExecStep) satisfied;
                            if (execStep.getName().equals(consumer)) {
                                foundConsumerSteps = true;
                                FrameIter frameIter = execStep.getFrameIter();
                                context.append("\n      SATISFIED: ").append(execStep.getType());
                                if (frameIter != null) {
                                    context.append(" Frame: ").append(frameIter.getFrame())
                                            .append(", Iter: ").append(frameIter.getIteration());
                                }
                            }
                        }
                    }

                    for (Object queued : dt.getAllSatisfiedQueue()) {
                        if (queued instanceof ExecStep) {
                            ExecStep execStep = (ExecStep) queued;
                            if (execStep.getName().equals(consumer)) {
                                foundConsumerSteps = true;
                                FrameIter frameIter = execStep.getFrameIter();
                                context.append("\n      QUEUED: ").append(execStep.getType());
                                if (frameIter != null) {
                                    context.append(" Frame: ").append(frameIter.getFrame())
                                            .append(", Iter: ").append(frameIter.getIteration());
                                }
                            }
                        }
                    }

                    if (!foundConsumerSteps) {
                        context.append("\n      No execution steps found");
                    }

                    if (func instanceof Switch) {
                        Switch switchOp = (Switch) func;
                        context.append("\n    SWITCH DETAILS:");

                        if (switchOp.getPredicate() != null && switchOp.getPredicate().name().equals(itemName)) {
                            context.append("\n      THIS VARIABLE IS THE SWITCH PREDICATE!");

                            for (VarId vid : instances) {
                                SDValue value = nodeValueOutputs.get(vid);
                                context.append("\n        Frame: ").append(vid.getFrame())
                                        .append(", Iter: ").append(vid.getIteration())
                                        .append(", Predicate value: ").append(formatValue(value));
                            }
                        }

                        List<String> switchOutputs = consumerOp.getOutputsOfOp();
                        if (switchOutputs != null) {
                            context.append("\n      Switch outputs:");
                            for (String output : switchOutputs) {
                                boolean outputAvailable = isVariableAvailable(output);
                                context.append("\n        ").append(output).append(": ")
                                        .append(outputAvailable ? "✓" : "✗");

                                if (outputAvailable) {
                                    List<VarId> outputInstances = getVariableInstances(output);
                                    for (VarId vid : outputInstances) {
                                        SDValue value = nodeValueOutputs.get(vid);
                                        context.append("\n          Frame: ").append(vid.getFrame())
                                                .append(", Iter: ").append(vid.getIteration())
                                                .append(", Value: ").append(formatValue(value));
                                    }
                                }
                            }
                        }

                    } else if (func instanceof Merge) {
                        context.append("\n    MERGE DETAILS:");

                        List<String> mergeInputs = consumerOp.getInputsToOp();
                        if (mergeInputs != null) {
                            context.append("\n      Merge inputs:");
                            for (String input : mergeInputs) {
                                boolean inputAvailable = isVariableAvailable(input);
                                context.append("\n        ").append(input).append(": ")
                                        .append(inputAvailable ? "✓" : "✗");

                                if (inputAvailable) {
                                    List<VarId> inputInstances = getVariableInstances(input);
                                    for (VarId vid : inputInstances) {
                                        SDValue value = nodeValueOutputs.get(vid);
                                        context.append("\n          Frame: ").append(vid.getFrame())
                                                .append(", Iter: ").append(vid.getIteration())
                                                .append(", Value: ").append(formatValue(value));
                                    }
                                }
                            }
                        }

                    } else if (func instanceof LoopCond) {
                        context.append("\n    LOOP CONDITION DETAILS:");
                        context.append("\n      This operation determines if the loop should continue");

                        List<String> loopInputs = consumerOp.getInputsToOp();
                        if (loopInputs != null) {
                            context.append("\n      Loop condition inputs:");
                            for (String input : loopInputs) {
                                boolean inputAvailable = isVariableAvailable(input);
                                context.append("\n        ").append(input).append(": ")
                                        .append(inputAvailable ? "✓" : "✗");

                                if (inputAvailable) {
                                    Object value = getVariableValue(input);
                                    context.append(" Value: ").append(formatValue(value));
                                }
                            }
                        }

                        List<String> loopOutputs = consumerOp.getOutputsOfOp();
                        if (loopOutputs != null) {
                            context.append("\n      Loop condition outputs:");
                            for (String output : loopOutputs) {
                                boolean outputAvailable = isVariableAvailable(output);
                                context.append("\n        ").append(output).append(": ")
                                        .append(outputAvailable ? "✓" : "✗");

                                if (outputAvailable) {
                                    Object value = getVariableValue(output);
                                    context.append(" Value: ").append(formatValue(value));
                                }
                            }
                        }
                    }
                }
            }
        }

        return context.length() > 0 ? context.toString() : null;
    }

    /**
     * Detailed loop condition issue analysis with per-frame values.
     */
    public String analyzeLoopConditionIssuesDetailed(String itemName) {
        StringBuilder analysis = new StringBuilder();

        String producer = findVariableProducer(itemName);
        if (producer != null) {
            SameDiffOp producerOp = sameDiff.getOps().get(producer);
            if (producerOp != null) {
                String producerFrame = getOperationFrame(producer);
                if (!AbstractSession.OUTER_FRAME.equals(producerFrame)) {
                    analysis.append("LOOP CONDITION ANALYSIS:");
                    analysis.append("\n  Producer '").append(producer)
                            .append("' is in frame: ").append(producerFrame);

                    boolean producerExecuted = isOperationExecuted(producer);
                    analysis.append("\n  Producer executed: ").append(producerExecuted ? "✓" : "✗");

                    analysis.append("\n  Producer execution steps:");
                    boolean foundProducerSteps = false;

                    for (Object satisfied : dt.getAllSatisfied()) {
                        if (satisfied instanceof ExecStep) {
                            ExecStep execStep = (ExecStep) satisfied;
                            if (execStep.getName().equals(producer)) {
                                foundProducerSteps = true;
                                FrameIter frameIter = execStep.getFrameIter();
                                analysis.append("\n    SATISFIED: ").append(execStep.getType());
                                if (frameIter != null) {
                                    analysis.append(" Frame: ").append(frameIter.getFrame())
                                            .append(", Iter: ").append(frameIter.getIteration());
                                }
                            }
                        }
                    }

                    for (Object queued : dt.getAllSatisfiedQueue()) {
                        if (queued instanceof ExecStep) {
                            ExecStep execStep = (ExecStep) queued;
                            if (execStep.getName().equals(producer)) {
                                foundProducerSteps = true;
                                FrameIter frameIter = execStep.getFrameIter();
                                analysis.append("\n    QUEUED: ").append(execStep.getType());
                                if (frameIter != null) {
                                    analysis.append(" Frame: ").append(frameIter.getFrame())
                                            .append(", Iter: ").append(frameIter.getIteration());
                                }
                            }
                        }
                    }

                    if (!foundProducerSteps) {
                        analysis.append("\n    No execution steps found for producer");
                    }

                    List<String> loopCondOps = findLoopConditionOperations(producerFrame);
                    if (!loopCondOps.isEmpty()) {
                        analysis.append("\n  Loop condition operations in frame: ").append(loopCondOps);

                        for (String loopCondOp : loopCondOps) {
                            boolean executed = isOperationExecuted(loopCondOp);
                            analysis.append("\n    ").append(loopCondOp)
                                    .append(" executed: ").append(executed ? "✓" : "✗");

                            if (!executed) {
                                analysis.append(" <- CRITICAL: Loop condition not evaluated");
                            }

                            analysis.append("\n      Execution steps:");
                            boolean foundLoopSteps = false;

                            for (Object satisfied : dt.getAllSatisfied()) {
                                if (satisfied instanceof ExecStep) {
                                    ExecStep execStep = (ExecStep) satisfied;
                                    if (execStep.getName().equals(loopCondOp)) {
                                        foundLoopSteps = true;
                                        FrameIter frameIter = execStep.getFrameIter();
                                        analysis.append("\n        SATISFIED: ").append(execStep.getType());
                                        if (frameIter != null) {
                                            analysis.append(" Frame: ").append(frameIter.getFrame())
                                                    .append(", Iter: ").append(frameIter.getIteration());
                                        }
                                    }
                                }
                            }

                            for (Object queued : dt.getAllSatisfiedQueue()) {
                                if (queued instanceof ExecStep) {
                                    ExecStep execStep = (ExecStep) queued;
                                    if (execStep.getName().equals(loopCondOp)) {
                                        foundLoopSteps = true;
                                        FrameIter frameIter = execStep.getFrameIter();
                                        analysis.append("\n        QUEUED: ").append(execStep.getType());
                                        if (frameIter != null) {
                                            analysis.append(" Frame: ").append(frameIter.getFrame())
                                                    .append(", Iter: ").append(frameIter.getIteration());
                                        }
                                    }
                                }
                            }

                            if (!foundLoopSteps) {
                                analysis.append("\n        No execution steps found");
                            }

                            SameDiffOp loopOp = sameDiff.getOps().get(loopCondOp);
                            if (loopOp != null) {
                                List<String> loopInputs = loopOp.getInputsToOp();
                                if (loopInputs != null) {
                                    analysis.append("\n      Loop condition inputs:");
                                    for (String input : loopInputs) {
                                        boolean inputAvailable = isVariableAvailable(input);
                                        analysis.append("\n        ").append(input).append(": ")
                                                .append(inputAvailable ? "✓" : "✗");

                                        if (inputAvailable) {
                                            Object value = getVariableValue(input);
                                            analysis.append(" Value: ").append(formatValue(value));

                                            List<VarId> inputInstances = getVariableInstances(input);
                                            for (VarId vid : inputInstances) {
                                                SDValue sdValue = nodeValueOutputs.get(vid);
                                                analysis.append("\n          Frame: ").append(vid.getFrame())
                                                        .append(", Iter: ").append(vid.getIteration())
                                                        .append(", Value: ").append(formatValue(sdValue));
                                            }
                                        }
                                    }
                                }

                                List<String> loopOutputs = loopOp.getOutputsOfOp();
                                if (loopOutputs != null) {
                                    analysis.append("\n      Loop condition outputs:");
                                    for (String output : loopOutputs) {
                                        boolean outputAvailable = isVariableAvailable(output);
                                        analysis.append("\n        ").append(output).append(": ")
                                                .append(outputAvailable ? "✓" : "✗");

                                        if (outputAvailable) {
                                            Object value = getVariableValue(output);
                                            analysis.append(" Value: ").append(formatValue(value));
                                        }
                                    }
                                }
                            }
                        }
                    }

                    List<String> frameTransOps = findFrameTransitionOperationsInFrame(producerFrame);
                    if (!frameTransOps.isEmpty()) {
                        analysis.append("\n  Frame transition operations in frame:");
                        for (String transOp : frameTransOps) {
                            boolean executed = isOperationExecuted(transOp);
                            analysis.append("\n    ").append(transOp)
                                    .append(" executed: ").append(executed ? "✓" : "✗");

                            SameDiffOp transOpObj = sameDiff.getOps().get(transOp);
                            if (transOpObj != null) {
                                DifferentialFunction transFunc = transOpObj.getOp();
                                analysis.append(" (").append(transFunc.getClass().getSimpleName()).append(")");

                                if (transFunc instanceof NextIteration) {
                                    analysis.append(" - Advances loop iteration");
                                } else if (transFunc instanceof Enter) {
                                    Enter enter = (Enter) transFunc;
                                    analysis.append(" - Enters frame: ").append(enter.getFrameName());
                                } else if (transFunc instanceof Exit) {
                                    analysis.append(" - Exits current frame");
                                }
                            }
                        }
                    }

                    String infiniteLoopCheck = checkForInfiniteLoop(producerFrame);
                    if (infiniteLoopCheck != null) {
                        analysis.append("\n  ").append(infiniteLoopCheck);
                    }
                }
            }
        }

        return analysis.length() > 0 ? analysis.toString() : null;
    }

    // -------------------------------------------------------------------------
    // Variable instance helpers
    // -------------------------------------------------------------------------

    /**
     * Get all VarId instances for a variable across different frames/iterations.
     */
    public List<VarId> getVariableInstances(String variableName) {
        List<VarId> instances = new ArrayList<>();

        for (VarId vid : nodeValueOutputs.keySet()) {
            if (vid.getVariable().equals(variableName)) {
                instances.add(vid);
            }
        }

        instances.sort((a, b) -> {
            int frameComp = a.getFrame().compareTo(b.getFrame());
            if (frameComp != 0) return frameComp;
            return Integer.compare(a.getIteration(), b.getIteration());
        });

        return instances;
    }

    // -------------------------------------------------------------------------
    // Value / array formatting
    // -------------------------------------------------------------------------

    /**
     * Format a value (SDValue, INDArray, or arbitrary Object) for display.
     */
    public String formatValue(Object value) {
        if (value == null) {
            return "null";
        }

        if (value instanceof SDValue) {
            SDValue sdValue = (SDValue) value;
            switch (sdValue.getSdValueType()) {
                case TENSOR:
                    INDArray tensor = sdValue.getTensorValue();
                    if (tensor == null) {
                        return "null tensor";
                    }
                    return formatINDArray(tensor);
                case LIST:
                    List<INDArray> list = sdValue.getListValue();
                    if (list.isEmpty()) {
                        return "List[empty]";
                    }
                    StringBuilder sb = new StringBuilder("List[").append(list.size()).append("]: [");
                    for (int i = 0; i < Math.min(3, list.size()); i++) {
                        if (i > 0) sb.append(", ");
                        sb.append(formatINDArray(list.get(i)));
                    }
                    if (list.size() > 3) sb.append("...");
                    sb.append("]");
                    return sb.toString();
                default:
                    return sdValue.toString();
            }
        }

        if (value instanceof INDArray) {
            return formatINDArray((INDArray) value);
        }

        return value.toString();
    }

    /**
     * Format an INDArray for display with actual values.
     */
    public String formatINDArray(INDArray arr) {
        if (arr == null) {
            return "null";
        }

        StringBuilder sb = new StringBuilder();
        sb.append("Array").append(Arrays.toString(arr.shape())).append(" ").append(arr.dataType());

        if (arr.isScalar()) {
            sb.append(" = ").append(arr.getDouble(0));
        } else if (arr.length() <= 10) {
            sb.append(" = [");
            for (int i = 0; i < arr.length(); i++) {
                if (i > 0) sb.append(", ");
                sb.append(arr.getDouble(i));
            }
            sb.append("]");
        } else {
            sb.append(" = [");
            for (int i = 0; i < Math.min(3, arr.length()); i++) {
                if (i > 0) sb.append(", ");
                sb.append(arr.getDouble(i));
            }
            if (arr.length() > 6) {
                sb.append("...");
                for (int i = (int) Math.max(3, arr.length() - 3); i < arr.length(); i++) {
                    sb.append(", ").append(arr.getDouble(i));
                }
            } else {
                for (int i = 3; i < arr.length(); i++) {
                    sb.append(", ").append(arr.getDouble(i));
                }
            }
            sb.append("]");
        }

        return sb.toString();
    }

    // -------------------------------------------------------------------------
    // Frame transition finders
    // -------------------------------------------------------------------------

    /**
     * Find frame transition operations (Enter/Exit/NextIteration) in a specific frame.
     */
    public List<String> findFrameTransitionOperationsInFrame(String frameName) {
        List<String> transOps = new ArrayList<>();

        for (Map.Entry<String, SameDiffOp> entry : sameDiff.getOps().entrySet()) {
            SameDiffOp op = entry.getValue();
            DifferentialFunction func = op.getOp();

            if (func instanceof Enter || func instanceof Exit || func instanceof NextIteration) {
                String opFrame = getOperationFrame(entry.getKey());
                if (frameName.equals(opFrame)) {
                    transOps.add(entry.getKey());
                }
            }
        }

        return transOps;
    }

    /**
     * Analyze a remaining item (simplified form, without enhanced control flow section).
     */
    public String analyzeRemainingItem(String itemName) {
        StringBuilder analysis = new StringBuilder();

        if (sameDiff.variableMap().containsKey(itemName)) {
            analysis.append("Variable: ").append(itemName);

            Variable var = sameDiff.getVariables().get(itemName);
            if (var != null) {
                SDVariable sdVar = var.getVariable();
                analysis.append(" (").append(sdVar.getVariableType()).append(")");

                boolean available = isVariableAvailable(itemName);
                analysis.append("\n  Available: ").append(available ? "✓" : "✗");

                if (!available) {
                    String producer = findVariableProducer(itemName);
                    if (producer != null) {
                        analysis.append("\n  Producer operation: '").append(producer).append("'");

                        boolean producerExecuted = isOperationExecuted(producer);
                        analysis.append("\n  Producer executed: ").append(producerExecuted ? "✓" : "✗");

                        if (!producerExecuted) {
                            SameDiffOp producerOp = sameDiff.getOps().get(producer);
                            if (producerOp != null) {
                                analysis.append("\n  Producer analysis:");
                                analysis.append("\n    Type: ").append(producerOp.getOp().getClass().getSimpleName());

                                List<String> producerInputs = producerOp.getInputsToOp();
                                if (producerInputs != null && !producerInputs.isEmpty()) {
                                    analysis.append("\n    Producer inputs: ");
                                    for (String input : producerInputs) {
                                        boolean inputAvailable = isVariableAvailable(input);
                                        analysis.append(input).append(inputAvailable ? "✓" : "✗").append(" ");
                                    }
                                }

                                boolean inQueue = false;
                                for (Object queued : dt.getAllSatisfiedQueue()) {
                                    if (queued instanceof ExecStep) {
                                        ExecStep execStep = (ExecStep) queued;
                                        if (execStep.getName().equals(producer)) {
                                            inQueue = true;
                                            break;
                                        }
                                    }
                                }
                                analysis.append("\n    Producer in execution queue: ").append(inQueue ? "✓" : "✗");

                            } else {
                                analysis.append("\n    ERROR: Producer operation not found in graph");
                            }
                        }
                    } else {
                        analysis.append("\n  ERROR: No producer operation found for this variable");
                    }
                } else {
                    analysis.append("\n  WARNING: Variable is available but still in remaining set");
                }
            }

            return analysis.toString();
        }

        SameDiffOp op = sameDiff.getOps().get(itemName);
        if (op != null) {
            analysis.append("Operation: ").append(itemName);

            DifferentialFunction opFunc = op.getOp();
            analysis.append(" (").append(opFunc.getClass().getSimpleName()).append(")");

            List<String> inputs = op.getInputsToOp();
            if (inputs != null && !inputs.isEmpty()) {
                analysis.append("\n  Inputs: ");
                for (int i = 0; i < inputs.size(); i++) {
                    String input = inputs.get(i);
                    boolean available = isVariableAvailable(input);
                    analysis.append(input).append(available ? "✓" : "✗");
                    if (i < inputs.size() - 1) analysis.append(", ");
                }

                List<String> missingInputs = new ArrayList<>();
                for (String input : inputs) {
                    if (!isVariableAvailable(input)) {
                        missingInputs.add(input);
                    }
                }

                if (!missingInputs.isEmpty()) {
                    analysis.append("\n  Missing inputs: ").append(missingInputs);
                }
            } else {
                analysis.append("\n  No inputs required");
            }

            boolean executed = isOperationExecuted(itemName);
            analysis.append("\n  Executed: ").append(executed ? "✓" : "✗");

            if (!executed) {
                boolean foundInQueue = false;
                for (Object queued : dt.getAllSatisfiedQueue()) {
                    if (queued instanceof ExecStep) {
                        ExecStep execStep = (ExecStep) queued;
                        if (execStep.getName().equals(itemName)) {
                            foundInQueue = true;
                            break;
                        }
                    }
                }
                analysis.append("\n  In execution queue: ").append(foundInQueue ? "✓" : "✗");
            }

            return analysis.toString();
        }

        analysis.append("Unknown item: ").append(itemName);
        analysis.append("\n  ERROR: Item not found in variables or operations");

        return analysis.toString();
    }

    // -------------------------------------------------------------------------
    // Variable dependency helpers
    // -------------------------------------------------------------------------

    /**
     * Get the input dependencies for a variable by looking at its producer op.
     */
    public String[] getVariableDependencies(String varName) {
        List<String> dependencies = new ArrayList<>();
        for (Map.Entry<String, SameDiffOp> entry : sameDiff.getOps().entrySet()) {
            SameDiffOp op = entry.getValue();
            if (op.getOutputsOfOp() != null && op.getOutputsOfOp().contains(varName)) {
                List<String> inputs = op.getInputsToOp();
                if (inputs != null) {
                    dependencies.addAll(inputs);
                }
            }
        }
        return dependencies.isEmpty() ? null : dependencies.toArray(new String[0]);
    }

    /**
     * Check whether {@code variable} transitively depends on {@code target}.
     */
    public boolean isVariableDependentOn(String variable, String target) {
        Set<String> visited = new HashSet<>();
        return checkDependencyPath(variable, target, visited);
    }

    /**
     * Recursive helper for {@link #isVariableDependentOn}.
     */
    public boolean checkDependencyPath(String current, String target, Set<String> visited) {
        if (visited.contains(current) || current.equals(target)) {
            return current.equals(target);
        }
        visited.add(current);

        Variable var = sameDiff.getVariables().get(current);
        if (var != null && var.getOutputOfOp() != null) {
            SameDiffOp op = sameDiff.getOps().get(var.getOutputOfOp());
            if (op != null && op.getInputsToOp() != null) {
                for (String input : op.getInputsToOp()) {
                    if (checkDependencyPath(input, target, visited)) {
                        return true;
                    }
                }
            }
        }

        return false;
    }

    // -------------------------------------------------------------------------
    // DAG visualization helpers
    // -------------------------------------------------------------------------

    /**
     * Log the accumulated DAG data at INFO level.
     */
    public void visualizeDAG(Set<String> requestedOutputs,
                              Map<String, Set<String>> dagFlow,
                              Map<String, String> variableTypes,
                              Map<String, String> producerOps) {

        log.info("=== EXECUTION ORDER ===");

        List<String> executionOrder = topologicalSort(dagFlow);

        for (int i = 0; i < executionOrder.size(); i++) {
            String var = executionOrder.get(i);
            String type = variableTypes.get(var);
            String producer = producerOps.get(var);

            if (producer != null) {
                log.info("{}: [{}] {} <- {}", i, type, var, producer);
            } else {
                log.info("{}: [{}] {}", i, type, var);
            }
        }
    }

    /**
     * Topological sort of a DAG represented as an adjacency map.
     */
    public List<String> topologicalSort(Map<String, Set<String>> dagFlow) {
        List<String> result = new ArrayList<>();
        Set<String> visited = new HashSet<>();
        Set<String> visiting = new HashSet<>();

        for (String node : dagFlow.keySet()) {
            if (!visited.contains(node)) {
                topologicalSortHelper(node, dagFlow, visited, visiting, result);
            }
        }

        return result;
    }

    /**
     * Recursive helper for topological sort.
     */
    public void topologicalSortHelper(String node, Map<String, Set<String>> dagFlow,
                                       Set<String> visited, Set<String> visiting,
                                       List<String> result) {
        if (visiting.contains(node)) {
            return; // Cycle detected, skip
        }
        if (visited.contains(node)) {
            return;
        }

        visiting.add(node);

        Set<String> dependencies = dagFlow.get(node);
        if (dependencies != null) {
            for (String dep : dependencies) {
                topologicalSortHelper(dep, dagFlow, visited, visiting, result);
            }
        }

        visiting.remove(node);
        visited.add(node);
        result.add(node);
    }

    /**
     * Find all nodes reachable from {@code start} in the DAG.
     */
    public Set<String> findReachableNodes(String start, Map<String, Set<String>> dagFlow,
                                           Set<String> visited) {
        if (visited.contains(start)) {
            return new HashSet<>();
        }

        visited.add(start);
        Set<String> reachable = new HashSet<>();
        reachable.add(start);

        Set<String> dependencies = dagFlow.get(start);
        if (dependencies != null) {
            for (String dep : dependencies) {
                reachable.addAll(findReachableNodes(dep, dagFlow, visited));
            }
        }

        return reachable;
    }

    // -------------------------------------------------------------------------
    // Variable lookup helpers
    // -------------------------------------------------------------------------

    /**
     * Enhanced variable lookup with suffix-stripping and type-conversion fallbacks.
     */
    public Variable findVariable(String varName) {
        Variable var = sameDiff.getVariables().get(varName);
        if (var != null) {
            return var;
        }

        String baseVarName = stripVarSuffix(varName);
        if (!baseVarName.equals(varName)) {
            var = sameDiff.getVariables().get(baseVarName);
            if (var != null) {
                log.debug("Found variable using base name: {} -> {}", varName, baseVarName);
                return var;
            }
        }

        if (!varName.contains(":")) {
            for (String suffix : Arrays.asList(":0", ":1", ":2")) {
                String candidateName = varName + suffix;
                var = sameDiff.getVariables().get(candidateName);
                if (var != null) {
                    log.debug("Found variable using suffix: {} -> {}", varName, candidateName);
                    return var;
                }
            }
        }

        var = findThroughTypeConversions(varName);
        if (var != null) {
            return var;
        }

        log.debug("Variable not found: {}", varName);
        return null;
    }

    /**
     * Try to find a variable through common type-conversion naming patterns.
     */
    public Variable findThroughTypeConversions(String varName) {
        String[] patterns = {
                varName + "_int32",
                varName + "_float",
                varName + "_double",
                varName + "_long",
                varName.replace("_int32", "").replace("_float", "").replace("_double", "").replace("_long", "")
        };

        for (String pattern : patterns) {
            if (!pattern.equals(varName)) {
                Variable var = sameDiff.getVariables().get(pattern);
                if (var != null) {
                    log.debug("Found variable through type conversion: {} -> {}", varName, pattern);
                    return var;
                }
            }
        }

        return null;
    }

    /**
     * Find the operation that produces a given variable with enhanced fallback strategies.
     */
    public SameDiffOp findProducerOperation(String varName) {
        Variable var = findVariable(varName);
        if (var != null && var.getOutputOfOp() != null) {
            return sameDiff.getOps().get(var.getOutputOfOp());
        }

        for (SameDiffOp op : sameDiff.getOps().values()) {
            List<String> outputs = op.getOutputsOfOp();
            if (outputs != null && outputs.contains(varName)) {
                log.debug("Found producer operation for {}: {}", varName, op.getName());
                return op;
            }

            String baseVarName = stripVarSuffix(varName);
            if (!baseVarName.equals(varName) && outputs != null && outputs.contains(baseVarName)) {
                log.debug("Found producer operation for {} using base name {}: {}",
                        varName, baseVarName, op.getName());
                return op;
            }
        }

        log.debug("No producer operation found for: {}", varName);
        return null;
    }
}
