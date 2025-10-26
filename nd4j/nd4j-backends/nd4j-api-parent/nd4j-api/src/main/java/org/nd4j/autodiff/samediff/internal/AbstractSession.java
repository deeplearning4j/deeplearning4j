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

import lombok.*;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.listeners.At;
import org.nd4j.autodiff.listeners.Listener;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.SameDiffExecutionVisualizer;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.config.ExecutionResult;
import org.nd4j.autodiff.samediff.config.SDValue;
import org.nd4j.autodiff.samediff.config.SDValueType;
import org.nd4j.autodiff.samediff.execution.ExecutionNode;
import org.nd4j.autodiff.samediff.execution.ForwardExecutionDAG;
import org.nd4j.autodiff.samediff.execution.ForwardExecutionDAGBuilder;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.controlflow.compat.*;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.common.function.Predicate;

import java.util.*;
import java.util.stream.Collectors;

import static org.nd4j.imports.VariableUtils.stripVarSuffix;

@Slf4j
public abstract class AbstractSession<T, O> {

    /**
     * All execution in Samediff happens in a frame... this is the name of the
     * main/outer frame - i.e., the "default" frame
     * Other frames (such as for loops) may be nested within this frame
     */
    public static final String OUTER_FRAME = "main";

    protected final SameDiff sameDiff;
    @Getter
    protected final Map<VarId, SDValue> nodeValueOutputs = new LinkedHashMap<>(); // Key: variable (at a given frame +
    protected   SameDiffExecutionVisualizer visualizer;
    protected boolean visualizationEnabled = true;                                              // iteration). Value: the calculated

    /*
     * The dependency tracker is responsible for determining what ops (at what
     * frame/iteration) can be executed next, given
     * what has been executed so far.
     * For static graphs, such as abstraction would not be necessary; for dynamic
     * graphs (i.e., nested loops, of arbitrary
     * number of iterations and depth - and also switch ops which can cause whole
     * subgraphs to not be executed) this is necessary
     * Note: the ExecStep represents one step for execution - some steps are as
     * simple as "execute an op (at the given frame/iter)"
     * It works by adding dependencies (X -> Y - such as
     * "op Y depends on the output of op X") and then marking them as
     * satisfied ("op X has been calculated"). Once all dependencies for an
     * execution step have been satisfied, the execution step
     * is added to a queue - outputs of which can be accessed with
     * dt.getNewAllSatisfied() and dt.getNewAllSatisfiedList(),
     * at which point it is removed from the dependency tracker
     */
    protected final ExecStepDependencyTracker dt = new ExecStepDependencyTracker();

    /**
     * Contains variables we *might* need to execute in process of getting outputs
     * we want.
     * Variables not in this set are definitely not needed to get the requested
     * output variables, but variables that are
     * in this set may not be executed depending on the graph structure - i.e.,
     * switch ops, etc
     */
    protected final Set<String> subgraph = new LinkedHashSet<>();
    /**
     * As per subgraph set, but for ops instead
     */
    protected final Set<String> subgraphOps = new LinkedHashSet<>();

    protected String currentFrame = OUTER_FRAME;
    protected int currentFrameIter = 0;
    protected FrameIter currParentFrame = null;

    /**
     * Contains the names of ops that don't have any inputs. Kept because normally
     * ops are triggered for execution when
     * their all their inputs have been calculated; we'll trigger that step manually
     * during execution initialization
     */
    protected final Set<String> zeroInputOpsInSubgraph = new HashSet<>();

    public AbstractSession(@NonNull SameDiff sameDiff) {
        this.sameDiff = sameDiff;
    }

    public boolean contains(String variable, String frame, int iteration, FrameIter parentFrameIter) {
        VarId varId = new VarId(variable, frame, iteration, parentFrameIter);
        return nodeValueOutputs.containsKey(varId);
    }

    /**
     * Get a previously calculated output; throws an exception if the output does
     * not exist
     */
    public SDValue get(String variable, String frame, int iteration, FrameIter parentFrameIter) {
        return get(variable, frame, iteration, parentFrameIter, true);
    }

    /**
     * Get a previously calculated output
     *
     * @param enforceExistence If true: throw an exception if the array does not
     *                         exist
     */
    public SDValue get(String variable, String frame, int iteration, FrameIter parentFrameIter,
                       boolean enforceExistence) {
        // TODO eventually we'll cache and reuse VarId objects here to avoid garbage
        // generation on lookup etc
        VarId varId = new VarId(variable, frame, iteration, parentFrameIter);
        SDValue out = nodeValueOutputs.get(varId);
        if (enforceExistence) {
            Preconditions.checkNotNull(out, "No output found for variable %s (frame %s, iteration %s)", variable, frame,
                    iteration);
        }
        return out;
    }

    /**
     * Get the output of the session - i.e., perform inference/forward pass and
     * return the outputs for the specified variables
     *
     * @param variables           Name of the variables we want the
     *                            arrays/activations for
     * @param placeholderValues   The placeholder values (if any). May be null.
     * @param batch               The batch data, used to call Listener.opExecution
     * @param requiredActivations Additional activations that are required. Won't be
     *                            output, but opExecution will be called. May be
     *                            null.
     * @return The specified variable values, optionally in the specified workspace
     */
    public Map<String, T> output(@NonNull List<String> variables, Map<String, T> placeholderValues,
                                 MultiDataSet batch, Collection<String> requiredActivations, List<Listener> listeners, At at) {
        ExecutionResult output = output(variables, placeholderValues, Collections.emptyMap(), batch,
                requiredActivations, listeners, at);
        if (output.hasSingle())
            return (Map<String, T>) output.getOutputs();
        else if (output.hasValues()) {
            Map<String, SDValue> outputs = output.getValueOutputs();
            Map<String, INDArray> ret = new LinkedHashMap<>();
            for (Map.Entry<String, SDValue> value : outputs.entrySet()) {
                ret.put(value.getKey(), value.getValue().getTensorValue());
            }

            return (Map<String, T>) ret;
        }

        throw new IllegalStateException("No result output! Expected values or tensors.");
    }




    /**
     * Get the output of the session - i.e., perform inference/forward pass and
     * return the outputs for the specified variables
     *
     * @param variables              Name of the variables we want the
     *                               arrays/activations for
     * @param placeholderValues      The placeholder values (if any). May be null.
     * @param otherPlaceHolderValues other placeholder values that may not be
     *                               ndarrays.
     * @param batch                  The batch data, used to call
     *                               Listener.opExecution
     * @param requiredActivations    Additional activations that are required. Won't
     *                               be output, but opExecution will be called. May
     *                               be null.
     * @return The specified variable values, optionally in the specified workspace
     */
    public ExecutionResult output(@NonNull List<String> variables,
                                  Map<String, T> placeholderValues,
                                  Map<String, SDValue> otherPlaceHolderValues,
                                  MultiDataSet batch,
                                  Collection<String> requiredActivations,
                                  List<Listener> listeners, At at) {

        // Initialize visualization if enabled
        if (visualizationEnabled) {
            this.visualizer = SameDiffExecutionVisualizer.builder()
                    .nodeValueOutputs(nodeValueOutputs)
                    .sameDiff(sameDiff)
                    .build();

            visualizer.clear();
            visualizer.recordStep(
                    ExecType.EXEC_START,
                    "EXECUTION_START",
                    new FrameIter(OUTER_FRAME, 0, null),
                    new ArrayList<>(variables),
                    Collections.emptyList(),
                    "INITIALIZING"
            );
        }

        Preconditions.checkState(!variables.isEmpty() || !requiredActivations.isEmpty(),
                "Variables to perform forward pass for must not be empty");

        // ensure all placeholders are in a mutable map
        otherPlaceHolderValues = new LinkedHashMap<>(otherPlaceHolderValues);

        // ensure all placeholders passed in are placed with the other placeholder
        // values for consistency
        // later in execution we only use other place holder values
        if (placeholderValues != null && !placeholderValues.isEmpty()) {
            for (Map.Entry<String, T> placeHolderValue : placeholderValues.entrySet()) {
                if (otherPlaceHolderValues.containsKey(placeHolderValue.getKey())) {
                    throw new IllegalArgumentException(
                            "Unable to determine which placeholder to use. Please ensure all names across both placeholders are unique");
                }

                otherPlaceHolderValues.put(placeHolderValue.getKey(),
                        SDValue.create((INDArray) placeHolderValue.getValue()));
            }
        }

        if (requiredActivations == null)
            requiredActivations = Collections.emptySet();

        if (at == null)
            at = At.defaultAt();

        // Step 0: validation - that variables exist, placeholders have arrays, etc
        for (String s : variables) {
            Preconditions.checkState(sameDiff.variableMap().containsKey(s),
                    "Requested output variable %s does not exist in SameDiff instance", s);
        }

        Set<String> reqOutputVariablesSet = new LinkedHashSet<>(variables);

        placeholderValues = preprocessPlaceholders(placeholderValues, at);
        otherPlaceHolderValues = preprocessValuePlaceholders(otherPlaceHolderValues, at);

        // Clear state from past iterations, if any
        dt.clear();
        subgraph.clear();
        subgraphOps.clear();

        // Step 1: determine subgraph structure we actually need to execute
        Set<String> userRequestedUnique = new LinkedHashSet<>(variables);
        Set<String> allRequired = new LinkedHashSet<>(requiredActivations);
        allRequired.addAll(variables);
        initSubgraph(allRequired);

        // Visualize subgraph initialization
        if (visualizationEnabled) {
            visualizer.recordStep(
                    ExecType.OP,
                    "SUBGRAPH_INIT",
                    new FrameIter(OUTER_FRAME, 0, null),
                    new ArrayList<>(allRequired),
                    new ArrayList<>(subgraph),
                    "SUBGRAPH_DETERMINED"
            );
        }

        // Step 2: Check that we have required placeholders
        List<String> phNames = sameDiff.inputs();
        Set<String> presentPlaceholders = new HashSet<>();
        // add all placeholder values together
        if (placeholderValues != null && !placeholderValues.isEmpty())
            presentPlaceholders.addAll(placeholderValues.keySet());
        if (otherPlaceHolderValues != null && !otherPlaceHolderValues.isEmpty())
            presentPlaceholders.addAll(otherPlaceHolderValues.keySet());

        if (presentPlaceholders.isEmpty() || !presentPlaceholders.containsAll(phNames)) {
            /*
             * We only have a subset of all placeholders
             * Validate that we have all *required* placeholder values. Some might not be
             * needed to calculate the requested outputs
             * A placeholder is required if:
             * (a) It's one of the requested outputs
             * (b) It's required to calculate any of the ops in the subgraph
             * For example, we might have a label placeholder, and we're doing inference not
             * training
             */
            for (String s : phNames) {
                boolean required = false;
                if (variables.contains(s)) {
                    required = true;
                }
                if (!required) {
                    Variable v = sameDiff.getVariables().get(s);
                    if (v.getInputsForOp() != null) {
                        for (String s2 : v.getInputsForOp()) {
                            if (subgraph.contains(s2)) {
                                // Placeholder is required
                                required = true;
                                break;
                            }
                        }
                    }
                }

                if (required && (presentPlaceholders.isEmpty() || !presentPlaceholders.contains(s))) {
                    // Visualize placeholder validation error
                    if (visualizationEnabled) {
                        visualizer.recordStep(
                                ExecType.PLACEHOLDER,
                                s,
                                new FrameIter(OUTER_FRAME, 0, null),
                                Collections.emptyList(),
                                Collections.emptyList(),
                                "ERROR: Missing required placeholder"
                        );
                    }
                    throw new IllegalStateException(
                            "An input placeholder \"" + s + "\" is required to calculate the requested outputs," +
                                    " but a placeholder value was not provided");
                }
            }
        }

        // Step 3: Mark the (required) variables, constants and placeholders as
        // available via dependency tracker
        // And also any "zero dependency" ops - i.e., those without any inputs
        ExecStep start = new ExecStep(ExecType.EXEC_START, "", null); // Dummy dependency to trigger the variables and
        // constants
        for (SDVariable v : sameDiff.variables()) {
            VariableType vt = v.getVariableType();
            if (vt == VariableType.VARIABLE || vt == VariableType.CONSTANT) {
                ExecType et = vt == VariableType.VARIABLE ? ExecType.VARIABLE : ExecType.CONSTANT;
                ExecStep es = new ExecStep(et, v.name(), new FrameIter(OUTER_FRAME, 0, null));
                dt.addDependency(es, start);

                Variable var = sameDiff.getVariables().get(v.name());
                if (var.getControlDeps() != null) {
                    addVarControlDeps(es, var); // Before this variable can be considered available for use, we need
                    // specified op to be executed
                }
            }
        }

        for (String s : phNames) {
            ExecStep es = new ExecStep(ExecType.PLACEHOLDER, s, new FrameIter(OUTER_FRAME, 0, null));
            dt.addDependency(es, start);

            Variable var = sameDiff.getVariables().get(s);
            if (var.getControlDeps() != null) {
                addVarControlDeps(es, var); // Before this variable can be considered available for use, we need
                // specified op to be executed
            }
        }

        for (String s : zeroInputOpsInSubgraph) {
            ExecStep es = new ExecStep(ExecType.OP, s, new FrameIter(OUTER_FRAME, 0, null));
            dt.addDependency(es, start);
        }
        dt.markSatisfied(start, true);

        // Step 4: execute in any order, but not switching to new frame/iteration until
        // all from current frame/iter ops are done - until we have all required nodeOutputs
        Map<String, SDValue> outValues = new LinkedHashMap<>();
        Set<String> allExecuted = new LinkedHashSet<>();
        int step = 0; // Number of execution steps
        // Next 3: current execution frame
        int currentFrameIter = 0;
        ExecStepPredicate predicate = new ExecStepPredicate();

        // Enhanced tracking for control flow visualization
        Map<String, ControlFlowState> controlFlowStates = new HashMap<>();

        while (allExecuted.size() < allRequired.size()) {
            if (!dt.hasNewAllSatisfied()) {
                // Visualize execution failure with detailed context
                if (visualizationEnabled) {
                    visualizer.recordStep(
                            ExecType.OP,
                            "EXECUTION_FAILED",
                            new FrameIter(currentFrame, currentFrameIter, currParentFrame),
                            new ArrayList<>(allRequired),
                            new ArrayList<>(allExecuted),
                            generateFailureContext(allRequired, allExecuted, controlFlowStates)
                    );

                    visualizer.analyzeExecutionFailure(allRequired, allExecuted, step,
                            currentFrame, currentFrameIter,
                            nodeValueOutputs, sameDiff);
                }

                execFailed(userRequestedUnique, outValues, allRequired, allExecuted, step);
                break;
            }

            // Get variable in the current frame/iteration and execute it's corresponding op
             predicate.setCurrentFrame(currentFrame);
            predicate.setCurrentFrameIter(currentFrameIter);
            predicate.setCurrParentFrame(currParentFrame);

            ExecStep es = dt.getFirstNewAllSatisfiedMatching(predicate);
            if (es == null) {
                // We must have finished the current frame/iter, and are switching to the next one
                es = dt.getNewAllSatisfied();
            }

            currentFrame = es.getFrameIter().getFrame();
            currentFrameIter = es.getFrameIter().getIteration();
            currParentFrame = es.getFrameIter().getParentFrame();

            log.trace("Beginning execution step {}: {}", step, es);

            // Prepare visualization data
            List<String> stepInputs = getStepInputs(es);
            List<String> stepOutputs = getStepOutputs(es);
            String executionStatus = "SUCCESS";
            String detailedStatus = "";

            FrameIter outFrameIter;
            boolean skipDepUpdate = false; // Only used for Switch ops, which have slightly different handling...
            boolean skipMarkSatisfied = false; // Only for enter ops, because of different frame/iter

            try {
                if (es.getType() == ExecType.CONSTANT || es.getType() == ExecType.VARIABLE) {
                    VarId vid = new VarId(es.getName(), currentFrame, currentFrameIter, currParentFrame);
                    T arr = getConstantOrVariable(es.getName());
                    Preconditions.checkNotNull(arr, "Encountered null placeholder array for constant: %s", vid);
                    putNodeValue(SDValue.create((INDArray) arr), vid);
                    outFrameIter = new FrameIter(OUTER_FRAME, 0, null);
                    if (userRequestedUnique.contains(es.getName())) {
                        // User requested const/variable as one of the outputs
                        outValues.put(es.getName(), SDValue.create((INDArray) arr));
                    }

                    if (allRequired.contains(es.getName())) {
                        allExecuted.add(es.getName());
                    }

                    stepOutputs = Arrays.asList(es.getName());
                    detailedStatus = "Variable/Constant loaded: " + es.getName();

                } else if (es.getType() == ExecType.PLACEHOLDER) {
                    VarId vid = new VarId(es.getName(), currentFrame, currentFrameIter, currParentFrame);
                    if (placeholderValues != null && placeholderValues.containsKey(es.getName())) {
                        T phVal = placeholderValues == null ? null : placeholderValues.get(es.getName());
                        SDValue valueCreate = SDValue.create((INDArray) phVal);
                        putNodeValue(valueCreate, vid);
                        detailedStatus = "Placeholder value assigned: " + es.getName();
                    } else if (otherPlaceHolderValues != null && otherPlaceHolderValues.containsKey(es.getName())) {
                        SDValue value = otherPlaceHolderValues.get(es.getName());
                        switch (value.getSdValueType()) {
                            default:
                                putNodeValue(value, vid);
                                detailedStatus = "Placeholder value assigned: " + es.getName() + " (type: " + value.getSdValueType() + ")";
                                break;
                            case DICT:
                                throw new UnsupportedOperationException("Unable to process dictionary types.");
                        }
                    } else {
                        putNodeValue(null, vid);
                        detailedStatus = "Placeholder set to null: " + es.getName();
                    }

                    outFrameIter = new FrameIter(OUTER_FRAME, 0, null);
                    if (allRequired.contains(es.getName())) {
                        Preconditions.checkState(placeholderValues != null
                                        && !placeholderValues.containsKey(es.getName())
                                        || otherPlaceHolderValues != null &&
                                        otherPlaceHolderValues.containsKey(es.getName()),
                                "No array was provided for the placeholder variable \"%s\" that is required for execution",
                                es.getName());
                        // User requested placeholder value as one of the outputs
                        if (placeholderValues.containsKey(es.getName()))
                            outValues.put(es.getName(), SDValue.create((INDArray) placeholderValues.get(es.getName())));
                        else if (otherPlaceHolderValues.containsKey(es.getName())) {
                            outValues.put(es.getName(), otherPlaceHolderValues.get(es.getName()));
                        }
                    }

                    if (allRequired.contains(es.getName())) {
                        allExecuted.add(es.getName());
                    }

                    stepOutputs = Arrays.asList(es.getName());

                } else if (es.getType() == ExecType.OP) {
                    String opName = es.getName();
                    SameDiffOp op = sameDiff.getOps().get(opName);
                    DifferentialFunction o = op.getOp();

                    // Enhanced control flow tracking for visualization
                    ControlFlowState controlState = controlFlowStates.computeIfAbsent(opName, k -> new ControlFlowState());
                    controlState.executionCount++;
                    controlState.currentFrame = currentFrame;
                    controlState.currentIteration = currentFrameIter;

                    if (o instanceof Enter) {
                        // Enter op: output is variable in a new (specified) frame, iteration 0.
                        // Parent is current (input) frame
                        String outFrame = ((Enter) o).getFrameName();
                        outFrameIter = new FrameIter(outFrame, 0, es.getFrameIter());
                        detailedStatus = String.format("ENTER frame '%s' (is_constant: %s)",
                                outFrame, ((Enter) o).isConstant());
                        controlState.frameTransitions.add("ENTER -> " + outFrame);

                    } else if (o instanceof Exit) {
                        outFrameIter = getExitIter(es);
                        detailedStatus = String.format("EXIT from frame '%s' (iter: %d)",
                                currentFrame, currentFrameIter);
                        controlState.frameTransitions.add("EXIT <- " + currentFrame);

                    } else if (o instanceof NextIteration) {
                        // NextIteration op: forwards its single input to its output variable in the
                        // current frame, but increments the iteration number
                        outFrameIter = es.getFrameIter().clone();
                        outFrameIter.setIteration(outFrameIter.getIteration());
                        detailedStatus = String.format("NEXT_ITERATION in frame '%s' (iter: %d -> %d)",
                                currentFrame, currentFrameIter, outFrameIter.getIteration());
                        controlState.frameTransitions.add("NEXT_ITER: " + currentFrameIter + " -> " + outFrameIter.getIteration());

                    } else {
                        // Standard ops - output variable has same frame and iteration number as the
                        // input(s)
                        // Also loopCond, merge, while, etc
                        outFrameIter = es.getFrameIter();
                        detailedStatus = String.format("Standard operation in frame '%s' (iter: %d)", currentFrame, currentFrameIter);
                    }

                    // Resolve the inputs to this execution step (op) to actual arrays
                    Set<VarId> inputs = null;
                    Set<VarId> allIterInputs = null;
                    Set<String> constAndPhInputs = null;
                    DependencyList<ExecStep, ExecStep> dl = dt.getDependencies(es);

                    List<String> inputNames = op.getInputsToOp();
                    if (inputNames != null && !inputNames.isEmpty()) {
                        inputs = new LinkedHashSet<>();
                        allIterInputs = new LinkedHashSet<>();
                        constAndPhInputs = new LinkedHashSet<>();
                        Iterable<ExecStep> deps = dl.getDependencies();
                        if (deps != null) {
                            for (ExecStep dep : deps) {
                                switch (dep.getType()) {
                                    case OP:
                                    case SWITCH_L:
                                    case SWITCH_R:
                                        // The current execution step depends on one output of the op "dep"
                                        SameDiffOp toExecOp = sameDiff.getOps().get(es.getName());
                                        List<String> inputsToExecOp = toExecOp.getInputsToOp();
                                        SameDiffOp inputOp = sameDiff.getOps().get(dep.getName());
                                        List<String> inputOpOutNames = inputOp.getOutputsOfOp();
                                        for (String s : inputsToExecOp) {
                                            if (inputOpOutNames.contains(s)) {
                                                VarId vid = new VarId(s, dep.getFrameIter().getFrame(),
                                                        dep.getFrameIter().getIteration(),
                                                        dep.getFrameIter().getParentFrame());
                                                inputs.add(vid);
                                            }
                                        }
                                        break;
                                    case VARIABLE:
                                        inputs.add(new VarId(dep.getName(), dep.getFrameIter().getFrame(),
                                                dep.getFrameIter().getIteration(), dep.getFrameIter().getParentFrame()));
                                        break;
                                    case CONSTANT:
                                    case PLACEHOLDER:
                                        constAndPhInputs.add(dep.getName());
                                        break;
                                    default:
                                        throw new UnsupportedOperationException("Not yet implemented: " + dep.getType());
                                }
                            }
                        }
                    }

                    // Get actual input names for visualization
                    stepInputs = getInputNamesForVisualization(inputs, constAndPhInputs);

                    // Do execution of the op, in 2 steps
                    // (a) "Parameterize" the op - i.e., find and set the arrays on the op, allocate
                    // outputs, etc ready for execution
                    // (b) actually execute the operation
                    O parameterizedOp = getAndParameterizeOp(opName, outFrameIter, inputs, allIterInputs, constAndPhInputs,
                            placeholderValues, reqOutputVariablesSet, otherPlaceHolderValues);
                    ExecutionResult opOutputValues = getOutputs(parameterizedOp, outFrameIter, inputs, allIterInputs,
                            constAndPhInputs, listeners, at, batch, reqOutputVariablesSet, otherPlaceHolderValues);
                    List<String> opOutVarNames = op.getOutputsOfOp();
                    stepOutputs = new ArrayList<>(opOutVarNames);

                    int lengthToCheck = opOutputValues.numResults();
                    if (!opOutVarNames.isEmpty() && opOutputValues.hasSingle()) {
                        Preconditions.checkState(lengthToCheck == opOutVarNames.size(),
                                "Unexpected number of outputs from executed op %s:" +
                                        " got %s outputs when %s outputs were expected (%s)",
                                parameterizedOp.getClass().getSimpleName(), opOutputValues.numResults(),
                                opOutVarNames.size(), opOutVarNames);
                    }
                    // Store the op outputs
                    for (int i = 0; i < lengthToCheck; i++) {
                        if (opOutputValues.hasSingle() && opOutputValues.resultAt(i) == null
                                || opOutputValues.hasValues() && !opOutputValues.valueExistsAtIndex(i)
                                && op.getOp() instanceof Switch) {
                            // Switch op only forwards the input to one of the outputs
                            continue;
                        }

                        // control flow ops are actually variables from the input forwarding to the next
                        // frame
                        String n = opOutVarNames.get(i);

                        VarId vid = new VarId(n, outFrameIter.getFrame(), outFrameIter.getIteration(),
                                outFrameIter.getParentFrame());
                        if (opOutputValues.hasValues()) {
                            SDValue sdValue = opOutputValues.valueWithKeyAtIndex(i, false);
                            // values can be null
                            if (sdValue != null)
                                switch (sdValue.getSdValueType()) {
                                    case LIST:
                                        // tensor array op
                                        // note: we leave this out since we already update node value outputs earlier
                                        putNodeValue(sdValue, vid);
                                        break;

                                    case TENSOR:
                                        putNodeValue(sdValue, vid);
                                        // tensorflow import case where 2 input names are the same and 1 output will be
                                        // null
                                        if (op.getOp() instanceof Switch && inputNames.size() > 1
                                                && inputNames.get(0).equals(inputNames.get(1))) {
                                            putNodeValue(sdValue, vid);
                                            putNodeValue(sdValue, outFrameIter.toVarId(vid.getVariable() + ":1"));
                                        } else {
                                            putNodeValue(sdValue, vid);
                                        }
                                        break;
                                }

                            if (userRequestedUnique.contains(n)) {
                                outValues.put(n, sdValue);
                            }

                        } else {
                            SDValue currValueOutput = SDValue.create(opOutputValues.resultAt(i));
                            putNodeValue(currValueOutput, vid);
                            // ensure a singular value is populated in case the user uses the node value
                            // outputs
                            if (userRequestedUnique.contains(n)) {
                                outValues.put(n, currValueOutput);
                            }

                        }

                        if (allRequired.contains(n)) {
                            allExecuted.add(n);
                        }


                    }

                    // Post execution: update dependency tracker so we know what is available to
                    // execute next, given we now have these new values
                    if (o instanceof Switch) {
                        /*
                         * Switch is a special case: only one output/branch is considered to exist post
                         * execution.
                         * Unlike every other type of op, only 1 of 2 output arrays is actually
                         * executed.
                         * For dependency tracking purposes, this is why we have SWITCH_L and _R
                         * execution types.
                         * If we just depended on the op, the dependency tracker would incorrectly
                         * conclude that ops relying on
                         * both branches (i.e., including the unavailable one) can now be executed
                         */
                        skipDepUpdate = true;
                        skipMarkSatisfied = true;

                        // Enhanced Switch visualization tracking
                        SwitchResult switchResult = analyzeSwitchOperation(op, opOutputValues, inputNames);
                        controlState.switchDecisions.add(switchResult);

                        String[] argNames = o.argNames();
                        // tensorflow import case: this means we output a list with a single name and
                        // need to extract the null value from that singular list
                        if (argNames[0].equals(argNames[1])) {
                            SDValue sdValue = opOutputValues.getValueOutputs().get(argNames[0]);
                            List<INDArray> inputList = sdValue.getListValue();
                            int nullCount = (inputList.get(0) != null ? 1 : 0) + (inputList.get(1) != null ? 1 : 0);
                            Preconditions.checkState(nullCount == 1,
                                    "Expected exactly one output to be present for switch ops, got %s", nullCount);
                            boolean left = inputList.get(0) != null;

                            ExecStep branch;
                            if (left) {
                                branch = new ExecStep(ExecType.SWITCH_L, es.getName(), es.getFrameIter());
                            } else {
                                branch = new ExecStep(ExecType.SWITCH_R, es.getName(), es.getFrameIter());
                            }
                            updateDescendantDeps(branch, outFrameIter);
                            dt.markSatisfied(branch, true);

                            executionStatus = "SWITCH_" + (left ? "LEFT" : "RIGHT");
                            detailedStatus = String.format("SWITCH decision: %s branch taken (frame: %s)",
                                    left ? "LEFT" : "RIGHT", currentFrame);
                            switchResult.branchTaken = left ? "LEFT" : "RIGHT";
                            switchResult.predicateValue = inputList.get(left ? 0 : 1);

                        } else {
                            int nullCount = (opOutputValues.valueExistsAtIndex(0) ? 1 : 0)
                                    + (opOutputValues.valueExistsAtIndex(1) ? 1 : 0);
                            Preconditions.checkState(nullCount == 1,
                                    "Expected exactly one output to be present for switch ops, got %s", nullCount);
                            boolean left = opOutputValues.valueExistsAtIndex(0);
                            ExecStep branch;
                            if (left) {
                                branch = new ExecStep(ExecType.SWITCH_L, es.getName(), es.getFrameIter());
                            } else {
                                branch = new ExecStep(ExecType.SWITCH_R, es.getName(), es.getFrameIter());
                            }
                            updateDescendantDeps(branch, outFrameIter);
                            dt.markSatisfied(branch, true);

                            executionStatus = "SWITCH_" + (left ? "LEFT" : "RIGHT");
                            detailedStatus = String.format("SWITCH decision: %s branch taken (frame: %s, iter: %d)",
                                    left ? "LEFT" : "RIGHT", currentFrame, currentFrameIter);
                            switchResult.branchTaken = left ? "LEFT" : "RIGHT";
                            switchResult.outputIndex = left ? 0 : 1;
                        }

                    } else if (o instanceof Enter) {
                        // Enter op: we want to say that the inner frame is executed...
                        skipDepUpdate = true;
                        skipMarkSatisfied = true;
                        Enter e = (Enter) o;
                        FrameIter fi = new FrameIter(e.getFrameName(), 0, es.getFrameIter());
                        ExecStep exec = new ExecStep(ExecType.OP, es.getName(), fi);
                        updateDescendantDeps(exec, fi);
                        dt.markSatisfied(exec, true);

                        executionStatus = "ENTER_FRAME";
                        detailedStatus += String.format(" | Variables entering: %d",
                                stepInputs != null ? stepInputs.size() : 0);

                    } else if (o instanceof Exit) {
                        // Exit op: we want to say that the parent frame is executed...
                        skipDepUpdate = true;
                        skipMarkSatisfied = true;
                        FrameIter fi = es.getFrameIter().getParentFrame();
                        ExecStep exec = new ExecStep(ExecType.OP, es.getName(), fi);
                        updateDescendantDeps(exec, fi);
                        dt.markSatisfied(exec, true);

                        executionStatus = "EXIT_FRAME";
                        detailedStatus += String.format(" | Variables exiting: %d",
                                stepOutputs != null ? stepOutputs.size() : 0);

                    } else if (o instanceof Merge) {
                        // Enhanced Merge operation tracking
                        MergeResult mergeResult = analyzeMergeOperation(op, opOutputValues, inputs);
                        controlState.mergeDecisions.add(mergeResult);

                        detailedStatus += String.format(" | MERGE: %d inputs, selected input: %d",
                                inputs != null ? inputs.size() : 0, mergeResult.selectedInputIndex);
                    }

                    /*
                     * Edge case for TensorFlow import control dependencies: for some reason, TF
                     * allows op control dependencies
                     * like /while/x -> SomeConstant - i.e., a constant depending on something
                     * inside a scope.
                     * This should be handled with an enter op, but TF doesn't always use this :/
                     * Note that this is equivalent to marking the control dependency as satisfied
                     * on the first iteration
                     * TODO double check that this is exactly the same behaviour as TF - otherwise
                     * this approach might fail in
                     * some rare cases that rely on the constant/variable not being available
                     */
                    List<String> cdFor = op.getControlDepFor();
                    if (cdFor != null) {
                        ExecStep cdEs = new ExecStep(ExecType.CONTROL_DEP, opName, null);
                        if (!dt.isSatisfied(cdEs)) {
                            dt.markSatisfied(cdEs, true);
                        }
                        detailedStatus += String.format(" | Control deps: %d", cdFor.size());
                    }

                } else {
                    // Should never happen
                    throw new RuntimeException("Unknown ExecStep: " + es);
                }

            } catch (Exception e) {
                executionStatus = "ERROR: " + e.getMessage();
                detailedStatus = "Exception during execution: " + e.getClass().getSimpleName();
                if (visualizationEnabled) {
                    visualizer.recordStep(
                            convertExecType(es.getType()),
                            es.getName(),
                            convertFrameIter(es.getFrameIter()),
                            stepInputs,
                            stepOutputs,
                            executionStatus + " | " + detailedStatus
                    );
                }
                throw e; // Re-throw the exception
            }

            // Record the successful execution step with enhanced context
            if (visualizationEnabled) {
                String combinedStatus = executionStatus;
                if (!detailedStatus.isEmpty()) {
                    combinedStatus += " | " + detailedStatus;
                }

                // Add control flow statistics
                ControlFlowState state = controlFlowStates.get(es.getName());
                if (state != null && (state.executionCount > 1 || !state.switchDecisions.isEmpty() || !state.mergeDecisions.isEmpty())) {
                    combinedStatus += String.format(" | Exec count: %d", state.executionCount);
                    if (!state.switchDecisions.isEmpty()) {
                        combinedStatus += String.format(", Switches: %d", state.switchDecisions.size());
                    }
                    if (!state.mergeDecisions.isEmpty()) {
                        combinedStatus += String.format(", Merges: %d", state.mergeDecisions.size());
                    }
                }

                visualizer.recordStep(
                        convertExecType(es.getType()),
                        es.getName(),
                        convertFrameIter(es.getFrameIter()),
                        stepInputs,
                        stepOutputs,
                        combinedStatus
                );
            }

            // Standard ops
            if (!skipDepUpdate) {
                updateDescendantDeps(es, outFrameIter);
            }
            if (!skipMarkSatisfied) {
                dt.markSatisfied(es, true);
            }

            step++;
        }

        // Final visualization step with comprehensive summary
        if (visualizationEnabled) {
            String executionSummary = generateExecutionSummary(controlFlowStates, allExecuted,outValues);

            visualizer.recordStep(
                    ExecType.OP,
                    "EXECUTION_COMPLETE",
                    new FrameIter(OUTER_FRAME, 0, null),
                    new ArrayList<>(allExecuted),
                    new ArrayList<>(outValues.keySet()),
                    executionSummary
            );

            // Print the execution trace with enhanced control flow information
            visualizer.printCompleteAnalysisReport();

            // Generate and print control flow analysis
            if (!controlFlowStates.isEmpty()) {
                printControlFlowAnalysis(controlFlowStates);
            }
        }

        // TODO we should clear the node outputs map to get rid of the invalid (closed,
        // out of workspace, etc) arrays

        outValues = postProcessOutputValues(outValues);
        return ExecutionResult.builder()
                .valueOutputs(outValues).build();
    }


    /**
     * Resolve VarId aliases for cross-frame variable access
     */
    protected VarId resolveVarIdAlias(VarId varId) {
        if (varId == null) return null;

        // Create corresponding ExecStep and resolve using dependency tracker
        ExecStep step = new ExecStep(ExecType.OP, varId.getVariable(),
                new FrameIter(varId.getFrame(), varId.getIteration(), varId.getParentFrame()));

        // Use the dependency tracker's alias resolution
        ExecStep resolved = null;
        try {
            // The dependency tracker's resolveDependeeAlias method returns the resolved dependee
            Object resolvedObj = dt.getClass().getDeclaredMethod("resolveDependeeAlias", Object.class).invoke(dt, step);
            if (resolvedObj instanceof ExecStep) {
                resolved = (ExecStep) resolvedObj;
            }
        } catch (Exception e) {
            log.debug("Could not resolve alias for VarId {}: {}", varId, e.getMessage());
            return null;
        }

        if (resolved != null && !resolved.equals(step)) {
            return new VarId(resolved.getName(),
                    resolved.getFrameIter().getFrame(),
                    resolved.getFrameIter().getIteration(),
                    resolved.getFrameIter().getParentFrame());
        }

        return null;
    }

    /**
     * Create VarId aliases for frame transitions
     */
    protected void createVarIdAliases(Map<VarId, VarId> aliasMapping) {
        Map<ExecStep, ExecStep> stepMappings = new HashMap<>();

        for (Map.Entry<VarId, VarId> entry : aliasMapping.entrySet()) {
            VarId from = entry.getKey();
            VarId to = entry.getValue();

            ExecStep fromStep = new ExecStep(ExecType.OP, from.getVariable(),
                    new FrameIter(from.getFrame(), from.getIteration(), from.getParentFrame()));
            ExecStep toStep = new ExecStep(ExecType.OP, to.getVariable(),
                    new FrameIter(to.getFrame(), to.getIteration(), to.getParentFrame()));

            stepMappings.put(fromStep, toStep);
        }

        dt.batchCreateDependeeAliases(stepMappings);
    }

    /**
     * Handle frame transitions for Enter operations and update cross-frame dependencies
     */
    protected void handleFrameTransition(String operationName, FrameIter fromFrame, FrameIter toFrame, List<String> transferredVariables) {
        if (transferredVariables == null || transferredVariables.isEmpty()) {
            return;
        }

        Map<String, FrameIter> frameTransitions = new HashMap<>();
        for (String varName : transferredVariables) {
            frameTransitions.put(varName, toFrame);
        }

        // Update dependency tracker with frame transitions
        dt.resolveCrossFrameDependencies(frameTransitions);

        // Create VarId aliases for the transferred variables
        Map<VarId, VarId> varIdAliases = new HashMap<>();
        for (String varName : transferredVariables) {
            VarId fromVid = new VarId(varName, fromFrame.getFrame(), fromFrame.getIteration(), fromFrame.getParentFrame());
            VarId toVid = new VarId(varName, toFrame.getFrame(), toFrame.getIteration(), toFrame.getParentFrame());
            varIdAliases.put(fromVid, toVid);
        }

        createVarIdAliases(varIdAliases);

        log.debug("Handled frame transition for operation {}: {} variables from {} to {}",
                operationName, transferredVariables.size(), fromFrame.getFrame(), toFrame.getFrame());
    }

    /**
     * Validate cross-frame dependencies for Merge operations
     */
    protected void validateMergeDependencies(FrameIter frameIter, List<String> outputs) {
        if (outputs == null) return;

        for (String output : outputs) {
            for (SameDiffOp candidate : sameDiff.getOps().values()) {
                if (candidate.getOp() instanceof Merge) {
                    List<String> mergeInputs = candidate.getInputsToOp();
                    if (mergeInputs != null && mergeInputs.contains(output)) {
                        log.debug("Validating Merge {} can access Enter output {}",
                                candidate.getName(), output);

                        VarId expectedVid = new VarId(output, frameIter.getFrame(),
                                frameIter.getIteration(), frameIter.getParentFrame());

                        if (!nodeValueOutputs.containsKey(expectedVid)) {
                            log.warn("Merge operation {} may not find input {} in frame {}",
                                    candidate.getName(), output, frameIter.getFrame());
                        }
                    }
                }
            }
        }
    }

    /**
     * Propagate variables from child frame to parent frame when loops terminate early
     * This ensures post-loop operations can access variables even when loops exit via switch operations
     */
    protected void propagateFrameVariables(String fromFrame, FrameIter toFrame) {
        log.debug("FRAME_PROPAGATION: From '{}' to '{}'", fromFrame, toFrame);

        int propagatedCount = 0;
        for (Map.Entry<VarId, SDValue> entry : nodeValueOutputs.entrySet()) {
            VarId sourceVid = entry.getKey();

            // Find variables in the source frame
            if (sourceVid.getFrame().equals(fromFrame) && entry.getValue() != null) {
                // Create target VarId in parent frame
                VarId targetVid = new VarId(sourceVid.getVariable(), toFrame.getFrame(),
                        toFrame.getIteration(), toFrame.getParentFrame());

                // Only propagate if target doesn't exist
                if (!nodeValueOutputs.containsKey(targetVid)) {
                    putNodeValue(entry.getValue(), targetVid);
                    propagatedCount++;
                    log.debug("  Propagated: '{}' -> {}", sourceVid.getVariable(), targetVid);
                }
            }
        }

        log.debug("FRAME_PROPAGATION: Propagated {} variables", propagatedCount);
    }


    /**
     * Analyze Switch operation execution and return detailed results
     */
    private SwitchResult analyzeSwitchOperation(SameDiffOp op, ExecutionResult opOutputValues, List<String> inputNames) {
        SwitchResult result = new SwitchResult();
        result.operationName = op.getName();

        if (inputNames != null && inputNames.size() >= 2) {
            // Get predicate input (usually the second input)
            String predicateVar = inputNames.get(1);
            result.affectedVariables.addAll(inputNames);

            // Analyze which branch was taken based on outputs
            if (opOutputValues.hasValues()) {
                Map<String, SDValue> outputs = opOutputValues.getValueOutputs();
                int nonNullOutputs = 0;
                for (Map.Entry<String, SDValue> entry : outputs.entrySet()) {
                    if (entry.getValue() != null) {
                        nonNullOutputs++;
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
                // For single output mode, determine branch from which output is non-null
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
     * Analyze Merge operation execution and return detailed results
     */
    private MergeResult analyzeMergeOperation(SameDiffOp op, ExecutionResult opOutputValues, Set<VarId> inputs) {
        MergeResult result = new MergeResult();
        result.operationName = op.getName();
        result.totalInputs = inputs != null ? inputs.size() : 0;

        // Determine which input was selected by the merge operation
        if (opOutputValues.hasValues()) {
            Map<String, SDValue> outputs = opOutputValues.getValueOutputs();
            if (!outputs.isEmpty()) {
                SDValue mergedOutput = outputs.values().iterator().next();
                result.mergedValue = mergedOutput;

                // Try to determine which input was selected (this is implementation-specific)
                if (inputs != null) {
                    int inputIndex = 0;
                    for (VarId inputVar : inputs) {
                        // Compare with actual input values to determine selection
                        // This would require access to the actual input values
                        inputIndex++;
                    }
                    result.selectedInputIndex = 0; // Default to first input if undetermined
                }
            }
        }

        return result;
    }

    /**
     * Generate failure context information for debugging
     */
    private String generateFailureContext(Set<String> allRequired, Set<String> allExecuted,
                                          Map<String, ControlFlowState> controlFlowStates) {
        StringBuilder context = new StringBuilder();
        context.append("EXECUTION_FAILED | ");

        Set<String> remaining = new HashSet<>(allRequired);
        remaining.removeAll(allExecuted);

        context.append("Remaining operations: ").append(remaining.size());
        if (remaining.size() <= 5) {
            context.append(" (").append(String.join(", ", remaining)).append(")");
        }

        // Add control flow state summary
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
     * Generate comprehensive execution summary
     */
    private String generateExecutionSummary(Map<String, ControlFlowState> controlFlowStates,
                                            Set<String> allExecuted,
                                            Map<String, SDValue> outValues) {
        StringBuilder summary = new StringBuilder();
        summary.append("SUCCESS: ").append(outValues.size()).append(" outputs generated");

        // Add execution statistics
        summary.append(" | Total operations: ").append(allExecuted.size());

        // Add control flow statistics
        long totalExecutions = controlFlowStates.values().stream()
                .mapToLong(state -> state.executionCount).sum();
        long totalSwitches = controlFlowStates.values().stream()
                .mapToLong(state -> state.switchDecisions.size()).sum();
        long totalMerges = controlFlowStates.values().stream()
                .mapToLong(state -> state.mergeDecisions.size()).sum();

        if (totalSwitches > 0 || totalMerges > 0) {
            summary.append(" | Control flow: ").append(totalSwitches).append(" switches, ")
                    .append(totalMerges).append(" merges");
        }

        // Add frame information
        Set<String> uniqueFrames = controlFlowStates.values().stream()
                .map(state -> state.currentFrame)
                .filter(Objects::nonNull)
                .collect(Collectors.toSet());

        if (uniqueFrames.size() > 1) {
            summary.append(" | Frames used: ").append(uniqueFrames.size());
        }

        // Add performance metrics
        summary.append(" | Avg ops per control flow op: ");
        if (totalSwitches + totalMerges > 0) {
            summary.append(String.format("%.1f", (double) allExecuted.size() / (totalSwitches + totalMerges)));
        } else {
            summary.append("N/A");
        }

        return summary.toString();
    }

    /**
     * Print detailed control flow analysis
     */
    private void printControlFlowAnalysis(Map<String, ControlFlowState> controlFlowStates) {
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

            // Print summary statistics
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
    /**
     * Helper method to convert original ExecType to visualizer ExecType
     */
    private ExecType convertExecType(ExecType originalType) {
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
     * Helper method to convert original FrameIter to visualizer FrameIter
     */
    private FrameIter convertFrameIter(FrameIter originalFrameIter) {
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

    /**
     * Helper method to get input names for an execution step
     */
    private List<String> getStepInputs(ExecStep es) {
        List<String> inputs = new ArrayList<>();

        if (es.getType() == ExecType.OP) {
            String opName = es.getName();
            SameDiffOp op = sameDiff.getOps().get(opName);
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
     * Helper method to get output names for an execution step
     */
    private List<String> getStepOutputs(ExecStep es) {
        List<String> outputs = new ArrayList<>();

        if (es.getType() == ExecType.OP) {
            String opName = es.getName();
            SameDiffOp op = sameDiff.getOps().get(opName);
            if (op != null) {
                List<String> outputNames = op.getOutputsOfOp();
                if (outputNames != null) {
                    outputs.addAll(outputNames);
                }
            }
        } else if (es.getType() == ExecType.VARIABLE || es.getType() == ExecType.CONSTANT || es.getType() == ExecType.PLACEHOLDER) {
            outputs.add(es.getName());
        }

        return outputs;
    }

    /**
     * Helper method to get input names for visualization from VarIds and constants
     */
    private List<String> getInputNamesForVisualization(Set<VarId> inputs, Set<String> constAndPhInputs) {
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

    private FrameIter getExitIter(ExecStep es) {
        FrameIter outFrameIter;
        // Exit node forwards input to parent frame
        String outFrame = es.getFrameIter().getParentFrame().getFrame();
        int outIter = es.getFrameIter().getParentFrame().getIteration();
        FrameIter outParentFrame = es.getFrameIter().getParentFrame().getParentFrame();
        outFrameIter = new FrameIter(outFrame, outIter, outParentFrame);
        return outFrameIter;
    }

    /**
     * Add the control dependency from Op -> variable
     *
     * @param es Execution step for the variable
     * @param v  Variable
     */
    protected void addVarControlDeps(ExecStep es, Variable v) {
        List<String> cds = v.getControlDeps();
        if (cds != null) {
            for (String s : cds) {
                ExecStep controlES = new ExecStep(ExecType.CONTROL_DEP, s, null);
                dt.addDependency(es, controlES); // Before this variable can be considered available for use, we need
                // specified op to be executed
            }
        }
    }

    protected SDValue getSdValue(VarId tArr) {
        return nodeValueOutputs.get(tArr);
    }

    protected void setArrayAtIndex(List<INDArray> l, int i, INDArray sub) {
        l.set(i, sub);
    }

    protected void putNodeValue(SDValue sdValue, VarId varId) {
        nodeValueOutputs.put(varId, sdValue);
    }

    protected INDArray getTensorFromOutputs(VarId varId) {
        if (nodeValueOutputs.containsKey(varId) && getSdValue(varId).getTensorValue() != null)
            return getSdValue(varId).getTensorValue();
        return null;
    }

    /**
     * Generate a comprehensive failure report
     */
    /**
     * Generate a comprehensive failure report
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

        // Remaining operations summary
        Set<String> remaining = new HashSet<>(allRequired);
        remaining.removeAll(allExecuted);

        report.append("=== REMAINING OPERATIONS (").append(remaining.size()).append(") ===\n");
        for (String remainingOp : remaining) {
            report.append(analyzeRemainingOperation(remainingOp)).append("\n\n");
        }

        // Control flow analysis
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

        // Dependency tracker state
        report.append("=== DEPENDENCY TRACKER STATE ===\n");
        report.append("Total satisfied dependencies: ").append(dt.getSatisfiedDependencies().size()).append("\n");
        report.append("Has new satisfied: ").append(dt.hasNewAllSatisfied()).append("\n");
        report.append("All satisfied count: ").append(dt.getAllSatisfied().size()).append("\n");
        report.append("Queue size: ").append(dt.getAllSatisfiedQueue().size()).append("\n");

        // Break down satisfied items by type
        int opCount = 0, varCount = 0, constCount = 0, placeholderCount = 0, otherCount = 0;
        for (Object satisfied : dt.getAllSatisfied()) {
            if (satisfied instanceof ExecStep) {
                ExecStep step2 = (ExecStep) satisfied;
                switch (step2.getType()) {
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

        // Variable availability summary
        report.append("\n=== VARIABLE AVAILABILITY ===\n");
        Set<String> allVariables = sameDiff.variableMap().keySet();
        int availableCount = 0;
        for (String var : allVariables) {
            if (isVariableAvailable(var)) {
                availableCount++;
            }
        }
        report.append("Available variables: ").append(availableCount).append("/").append(allVariables.size()).append("\n");

        // Show some unavailable variables if there are any
        List<String> unavailableVars = new ArrayList<>();
        for (String var : allVariables) {
            if (!isVariableAvailable(var)) {
                unavailableVars.add(var);
                if (unavailableVars.size() >= 10) break; // Limit to first 10
            }
        }
        if (!unavailableVars.isEmpty()) {
            report.append("Some unavailable variables: ").append(unavailableVars).append("\n");
        }

        // Frame information if available
        if (getCurrentFrame() != null) {
            report.append("\n=== FRAME INFORMATION ===\n");
            report.append("Current frame: ").append(getCurrentFrame()).append("\n");

            Set<String> frameVars = getVariablesInCurrentFrame();
            if (frameVars != null) {
                report.append("Variables in current frame: ").append(frameVars.size()).append("\n");
            }
        }

        report.append("\n=== END FAILURE REPORT ===\n");

        return report.toString();
    }


    /**
     * Get comprehensive dependency information for debugging
     */
    /**
     * Get comprehensive dependency information for debugging
     */
    private Map<String, Object> getDebugDependencyInfo(String opName) {
        Map<String, Object> info = new HashMap<>();

        // Basic operation info
        SameDiffOp op = sameDiff.getOps().get(opName);
        if (op != null) {
            info.put("operationType", op.getOp().getClass().getSimpleName());
            info.put("inputs", op.getInputsToOp());
            info.put("outputs", op.getOutputsOfOp());
            info.put("controlDeps", op.getControlDeps());
        }

        // Execution step info - find steps related to this operation
        List<Map<String, Object>> stepInfo = new ArrayList<>();

        // Check in satisfied set
        for (Object satisfied : dt.getAllSatisfied()) {
            if (satisfied instanceof ExecStep) {
                ExecStep step = (ExecStep) satisfied;
                if (step.getName().equals(opName)) {
                    Map<String, Object> stepData = new HashMap<>();
                    stepData.put("type", step.getType());
                    stepData.put("satisfied", true); // It's in satisfied set
                    stepData.put("location", "satisfied_set");
                    if (step.getFrameIter() != null) {
                        stepData.put("frame", step.getFrameIter().getFrame());
                        stepData.put("iteration", step.getFrameIter().getIteration());
                    }
                    stepInfo.add(stepData);
                }
            }
        }

        // Check in queue
        for (Object queued : dt.getAllSatisfiedQueue()) {
            if (queued instanceof ExecStep) {
                ExecStep step = (ExecStep) queued;
                if (step.getName().equals(opName)) {
                    Map<String, Object> stepData = new HashMap<>();
                    stepData.put("type", step.getType());
                    stepData.put("satisfied", false);
                    stepData.put("location", "queue");
                    if (step.getFrameIter() != null) {
                        stepData.put("frame", step.getFrameIter().getFrame());
                        stepData.put("iteration", step.getFrameIter().getIteration());
                    }
                    stepInfo.add(stepData);
                }
            }
        }

        info.put("executionSteps", stepInfo);

        // Input availability
        if (op != null && op.getInputsToOp() != null) {
            Map<String, Object> inputInfo = new HashMap<>();
            for (String input : op.getInputsToOp()) {
                Map<String, Object> inputData = new HashMap<>();
                inputData.put("available", isVariableAvailable(input));
                inputData.put("producer", findVariableProducer(input));
                if (findVariableProducer(input) != null) {
                    inputData.put("producerExecuted", isOperationExecuted(findVariableProducer(input)));
                }
                inputInfo.put(input, inputData);
            }
            info.put("inputDetails", inputInfo);
        }

        // Dependency tracker specific info
        Map<String, Object> trackerInfo = new HashMap<>();
        trackerInfo.put("totalSatisfiedDeps", dt.getSatisfiedDependencies().size());
        trackerInfo.put("hasNewSatisfied", dt.hasNewAllSatisfied());
        trackerInfo.put("allSatisfiedCount", dt.getAllSatisfied().size());
        trackerInfo.put("queueSize", dt.getAllSatisfiedQueue().size());
        info.put("dependencyTracker", trackerInfo);

        return info;
    }

    /**
     * Analyze if this item is part of a control flow structure
     */
    private String analyzeControlFlowContext(String itemName) {
        StringBuilder context = new StringBuilder();

        // Check if this is a variable that feeds into control flow operations
        List<String> controlFlowConsumers = findControlFlowConsumers(itemName);
        if (!controlFlowConsumers.isEmpty()) {
            context.append("CONTROL FLOW CONTEXT: Variable feeds into control flow operations: ");
            context.append(controlFlowConsumers);

            // Analyze each control flow consumer
            for (String consumer : controlFlowConsumers) {
                SameDiffOp consumerOp = sameDiff.getOps().get(consumer);
                if (consumerOp != null) {
                    DifferentialFunction func = consumerOp.getOp();
                    context.append("\n  Consumer: ").append(consumer).append(" (").append(func.getClass().getSimpleName()).append(")");

                    if (func instanceof Switch) {
                        context.append(" - SWITCH OPERATION");
                        // Check if this is the predicate for the switch
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

                    // Check if consumer has executed
                    boolean consumerExecuted = isOperationExecuted(consumer);
                    context.append(" - Executed: ").append(consumerExecuted ? "✓" : "✗");
                }
            }
        }

        return context.length() > 0 ? context.toString() : null;
    }

    /**
     * Analyze potential loop condition issues
     */
    private String analyzeLoopConditionIssues(String itemName) {
        StringBuilder analysis = new StringBuilder();

        // Look for loop condition patterns
        String producer = findVariableProducer(itemName);
        if (producer != null) {
            SameDiffOp producerOp = sameDiff.getOps().get(producer);
            if (producerOp != null) {
                DifferentialFunction func = producerOp.getOp();

                // Check if producer is in a loop structure
                String producerFrame = getOperationFrame(producer);
                if (!OUTER_FRAME.equals(producerFrame)) {
                    analysis.append("LOOP CONDITION ANALYSIS:");
                    analysis.append("\n  Producer is in frame: ").append(producerFrame);

                    // Check if there are any loop condition operations in this frame
                    List<String> loopCondOps = findLoopConditionOperations(producerFrame);
                    if (!loopCondOps.isEmpty()) {
                        analysis.append("\n  Loop condition operations in frame: ").append(loopCondOps);

                        // Check if loop conditions have been satisfied
                        for (String loopCondOp : loopCondOps) {
                            boolean executed = isOperationExecuted(loopCondOp);
                            analysis.append("\n    ").append(loopCondOp).append(" executed: ").append(executed ? "✓" : "✗");

                            if (!executed) {
                                // This might be the problem!
                                analysis.append(" <- POTENTIAL ISSUE: Loop condition not evaluated");

                                // Check what the loop condition depends on
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

                    // Check for infinite loop indicators
                    String infiniteLoopCheck = checkForInfiniteLoop(producerFrame);
                    if (infiniteLoopCheck != null) {
                        analysis.append("\n  ").append(infiniteLoopCheck);
                    }
                }
            }
        }

        return analysis.length() > 0 ? analysis.toString() : null;
    }



    /**
     * Analyze a single remaining item (could be operation or variable) for the failure report
     */
    private String analyzeRemainingOperation(String itemName) {
        StringBuilder analysis = new StringBuilder();

        // First check if it's a variable
        if (sameDiff.variableMap().containsKey(itemName)) {
            analysis.append("Variable: ").append(itemName);

            Variable var = sameDiff.getVariables().get(itemName);
            if (var != null) {
                SDVariable sdVar = var.getVariable();
                analysis.append(" (").append(sdVar.getVariableType()).append(")");

                // Check if variable is available
                boolean available = isVariableAvailable(itemName);
                analysis.append("\n  Available: ").append(available ? "✓" : "✗");

                if (!available) {
                    // Find what operation should produce this variable
                    String producer = findVariableProducer(itemName);
                    if (producer != null) {
                        analysis.append("\n  Producer operation: '").append(producer).append("'");

                        // Check if producer has been executed
                        boolean producerExecuted = isOperationExecuted(producer);
                        analysis.append("\n  Producer executed: ").append(producerExecuted ? "✓" : "✗");

                        if (!producerExecuted) {
                            // Analyze why the producer hasn't executed
                            SameDiffOp producerOp = sameDiff.getOps().get(producer);
                            if (producerOp != null) {
                                analysis.append("\n  Producer analysis:");
                                analysis.append("\n    Type: ").append(producerOp.getOp().getClass().getSimpleName());

                                // Check producer's inputs
                                List<String> producerInputs = producerOp.getInputsToOp();
                                if (producerInputs != null && !producerInputs.isEmpty()) {
                                    analysis.append("\n    Producer inputs: ");
                                    for (String input : producerInputs) {
                                        boolean inputAvailable = isVariableAvailable(input);
                                        analysis.append(input).append(inputAvailable ? "✓" : "✗").append(" ");
                                    }
                                }

                                // Check if producer is in execution queue
                                boolean inQueue = false;
                                for (Object queued : dt.getAllSatisfiedQueue()) {
                                    if (queued instanceof ExecStep) {
                                        ExecStep step = (ExecStep) queued;
                                        if (step.getName().equals(producer)) {
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
                    // Variable is available, so why is it in remaining?
                    analysis.append("\n  WARNING: Variable is available but still in remaining set");
                }
            }

            // Add enhanced control flow analysis
            analysis.append("\n\n=== CONTROL FLOW ANALYSIS ===");

            // Check if this item is part of a control flow structure
            String controlFlowContext = analyzeControlFlowContext(itemName);
            if (controlFlowContext != null) {
                analysis.append("\n").append(controlFlowContext);
            }

            // Check for loop condition issues
            String loopAnalysis = analyzeLoopConditionIssues(itemName);
            if (loopAnalysis != null) {
                analysis.append("\n").append(loopAnalysis);
            }

            return analysis.toString();
        }

        // Check if it's an operation
        SameDiffOp op = sameDiff.getOps().get(itemName);
        if (op != null) {
            analysis.append("Operation: ").append(itemName);

            DifferentialFunction opFunc = op.getOp();
            analysis.append(" (").append(opFunc.getClass().getSimpleName()).append(")");

            // Check inputs
            List<String> inputs = op.getInputsToOp();
            if (inputs != null && !inputs.isEmpty()) {
                analysis.append("\n  Inputs: ");
                for (int i = 0; i < inputs.size(); i++) {
                    String input = inputs.get(i);
                    boolean available = isVariableAvailable(input);
                    analysis.append(input).append(available ? "✓" : "✗");
                    if (i < inputs.size() - 1) analysis.append(", ");
                }

                // Check for missing inputs
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

            // Check if operation has been executed
            boolean executed = isOperationExecuted(itemName);
            analysis.append("\n  Executed: ").append(executed ? "✓" : "✗");

            if (!executed) {
                // Check execution step status
                boolean foundInQueue = false;
                for (Object queued : dt.getAllSatisfiedQueue()) {
                    if (queued instanceof ExecStep) {
                        ExecStep step = (ExecStep) queued;
                        if (step.getName().equals(itemName)) {
                            foundInQueue = true;
                            break;
                        }
                    }
                }
                analysis.append("\n  In execution queue: ").append(foundInQueue ? "✓" : "✗");
            }

            return analysis.toString();
        }

        // Neither variable nor operation found
        analysis.append("Unknown item: ").append(itemName);
        analysis.append("\n  ERROR: Item not found in variables or operations");

        return analysis.toString();
    }

    /**
     * Find the operation that produces a given variable
     */
    public String findVariableProducer(String varName) {
        // Check constants and variables first
        if (sameDiff.getVariables().containsKey(varName)) {
            Variable var = sameDiff.getVariables().get(varName);
            if (var.getVariable().isConstant() || var.getVariable().getVariableType() == VariableType.VARIABLE) {
                return varName; // Self-produced
            }
        }

        // Check operations that produce this variable
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
     * Check if a variable is available in the current execution context
     */
    private boolean isVariableAvailable(String varName) {
        // Check if variable exists in nodeVarOutputs (your variable storage)
        for (VarId vid : nodeValueOutputs.keySet()) {
            if (vid.getVariable().equals(varName)) {
                SDValue value = nodeValueOutputs.get(vid);
                return value != null;
            }
        }
        return false;
    }


    /**
     * Check if an operation has been executed
     */
    private boolean isOperationExecuted(String opName) {
        // Check if the operation exists in allSatisfied set of dependency tracker
        // We need to check for ExecStep objects with this operation name
        for (Object satisfied : dt.getAllSatisfied()) {
            if (satisfied instanceof ExecStep) {
                ExecStep step = (ExecStep) satisfied;
                if (step.getName().equals(opName) &&
                        (step.getType() == ExecType.OP || step.getType() == ExecType.SWITCH_L || step.getType() == ExecType.SWITCH_R)) {
                    return true;
                }
            }
        }
        return false;
    }



    /**
     * Check if there's a dependency path between two operations
     * Simplified version that checks direct dependencies only
     */
    private boolean hasDependencyPath(String from, String to) {
        if (from.equals(to)) {
            return true;
        }

        // Get direct dependencies for the 'from' operation
        SameDiffOp fromOp = sameDiff.getOps().get(from);
        if (fromOp != null) {
            List<String> inputs = fromOp.getInputsToOp();
            if (inputs != null) {
                // Check if 'to' operation produces any of the inputs for 'from'
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
     * Analyze execution step status in dependency tracker
     */
    private void analyzeExecutionStepStatus(String opName) {
        log.error("    Execution step analysis for '{}':", opName);

        // Find all execution steps for this operation by checking satisfied dependencies
        boolean foundStep = false;
        for (Object satisfied : dt.getAllSatisfied()) {
            if (satisfied instanceof ExecStep) {
                ExecStep step = (ExecStep) satisfied;
                if (step.getName().equals(opName)) {
                    foundStep = true;
                    boolean isSatisfied = dt.isSatisfied(step);

                    log.error("      Step {}: satisfied={}, frame={}:{}",
                            step.getType(), isSatisfied,
                            step.getFrameIter() != null ? step.getFrameIter().getFrame() : "null",
                            step.getFrameIter() != null ? step.getFrameIter().getIteration() : "null");

                    // Check dependencies of this step
                    DependencyList<ExecStep, ExecStep> deps = dt.getDependencies(step);
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

    private Object getVariableValue(String varName) {
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
        return null;
    }

    /**
     * Check if a control dependency is satisfied
     */
    private boolean isControlDependencySatisfied(String controlDep) {
        // Control dependencies are usually operations that need to complete first
        return isOperationExecuted(controlDep);
    }


    /**
     * Get current execution frame
     */
    private String getCurrentFrame() {
        // Return the current frame being executed
        // This should be tracked in your execution loop
        return currentFrame != null ? currentFrame : OUTER_FRAME;
    }

    /**
     * Get the frame for an operation
     */
    /**
     * Get the frame for an operation
     */
    private String getOperationFrame(String opName) {
        // First check if this is a special frame operation
        SameDiffOp op = sameDiff.getOps().get(opName);
        if (op != null && op.getOp() instanceof Enter) {
            return ((Enter) op.getOp()).getFrameName();
        }

        // Check in the satisfied execution steps for frame information
        for (Object satisfied : dt.getAllSatisfied()) {
            if (satisfied instanceof ExecStep) {
                ExecStep step = (ExecStep) satisfied;
                if (step.getName().equals(opName) && step.getFrameIter() != null) {
                    return step.getFrameIter().getFrame();
                }
            }
        }

        // Check in the queue for frame information
        for (Object queued : dt.getAllSatisfiedQueue()) {
            if (queued instanceof ExecStep) {
                ExecStep step = (ExecStep) queued;
                if (step.getName().equals(opName) && step.getFrameIter() != null) {
                    return step.getFrameIter().getFrame();
                }
            }
        }

        // Default to outer frame if no specific frame found
        return OUTER_FRAME;
    }

    /**
     * Check if a dependency is satisfied
     */
    private boolean isDependencySatisfied(String dep) {
        // For variable dependencies, check if the variable is available
        if (sameDiff.getVariables().containsKey(dep)) {
            return isVariableAvailable(dep);
        }

        // For operation dependencies, check if the operation has been executed
        if (sameDiff.getOps().containsKey(dep)) {
            return isOperationExecuted(dep);
        }

        return false;
    }


    private Set<String> getDependenciesFor(String opName) {
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
     * Get all variables in the current frame
     */
    private Set<String> getVariablesInCurrentFrame() {
        Set<String> frameVariables = new HashSet<>();
        String currentFrame = getCurrentFrame();

        for (VarId vid : nodeValueOutputs.keySet()) {
            if (currentFrame.equals(vid.getFrame())) {
                frameVariables.add(vid.getVariable());
            }
        }

        return frameVariables;
    }

    /**
     * Find all execution steps related to an operation
     */
    private List<ExecStep> findExecutionSteps(String opName) {
        List<ExecStep> steps = new ArrayList<>();

        // Check in satisfied dependencies
        for (Object satisfied : dt.getAllSatisfied()) {
            if (satisfied instanceof ExecStep) {
                ExecStep step = (ExecStep) satisfied;
                if (step.getName().equals(opName)) {
                    steps.add(step);
                }
            }
        }

        // Check in the allSatisfiedQueue (new satisfied items)
        for (Object queued : dt.getAllSatisfiedQueue()) {
            if (queued instanceof ExecStep) {
                ExecStep step = (ExecStep) queued;
                if (step.getName().equals(opName)) {
                    steps.add(step);
                }
            }
        }

        return steps;
    }

    /**
     * Analyze the overall dependency graph state
     */
    private void analyzeDependencyGraphState(Set<String> remainingOps) {
        log.error("Dependency tracker state:");
        log.error("  Total satisfied dependencies: {}", dt.getSatisfiedDependencies().size());
        log.error("  Steps with new satisfied dependencies: {}", dt.hasNewAllSatisfied());
        log.error("  All satisfied count: {}", dt.getAllSatisfied().size());
        log.error("  Queue size: {}", dt.getAllSatisfiedQueue().size());

        // Show some details about what's in the satisfied set
        int opCount = 0, varCount = 0, constCount = 0, placeholderCount = 0;
        for (Object satisfied : dt.getAllSatisfied()) {
            if (satisfied instanceof ExecStep) {
                ExecStep step = (ExecStep) satisfied;
                switch (step.getType()) {
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

        // Note: Circular dependency detection would require traversing the dependency graph
        // which is complex without access to the internal structure
        if (remainingOps.size() > 1) {
            log.error("Multiple operations remaining - potential circular dependency or missing prerequisites");
        }
    }

    /**
     * Enhanced execution failed method with detailed logging
     */
    /**
     * Enhanced execution failed method with comprehensive control flow analysis
     */
    private void execFailed(Set<String> userRequestedUnique, Map<String, SDValue> outValues,
                            Set<String> allRequired, Set<String> allExecuted, int step) {

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

        // Add dependency tracker state
        sb.append("Dependency tracker state:\n");
        sb.append("  Total satisfied dependencies: ").append(dt.getSatisfiedDependencies().size()).append("\n");
        sb.append("  Has new satisfied: ").append(dt.hasNewAllSatisfied()).append("\n");
        sb.append("  All satisfied count: ").append(dt.getAllSatisfied().size()).append("\n");

        // Quick failure pattern detection
        sb.append("\n=== FAILURE PATTERN DETECTION ===\n");
        if (remaining.size() == 1) {
            sb.append("PATTERN: Single stuck item - likely control flow issue\n");
        } else if (remaining.size() > 1) {
            sb.append("PATTERN: Multiple stuck items - possible dependency cycle\n");
        }

        // Check for control flow operations that haven't executed
        List<String> unexecutedControlFlow = findUnexecutedControlFlowOperations();
        if (!unexecutedControlFlow.isEmpty()) {
            sb.append("CRITICAL: Unexecuted control flow operations: ").append(unexecutedControlFlow).append("\n");
        }

        // Add detailed analysis for each remaining item
        sb.append("\n=== DETAILED ANALYSIS ===\n");
        for (String remainingItem : remaining) {
            sb.append(analyzeRemainingOperation(remainingItem)).append("\n\n");
        }


        visualizer.printCompleteAnalysisReport();


        throw new IllegalStateException(sb.toString());
    }

    /**
     * Find control flow operations that haven't been executed
     */
    private List<String> findUnexecutedControlFlowOperations() {
        List<String> unexecuted = new ArrayList<>();

        for (Map.Entry<String, SameDiffOp> entry : sameDiff.getOps().entrySet()) {
            String opName = entry.getKey();
            SameDiffOp op = entry.getValue();
            DifferentialFunction func = op.getOp();

            // Check if this is a control flow operation
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
     * Find operations that consume a variable and are control flow operations
     */
    private List<String> findControlFlowConsumers(String variableName) {
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
     * Find loop condition operations in a specific frame
     */
    private List<String> findLoopConditionOperations(String frameName) {
        List<String> loopCondOps = new ArrayList<>();

        for (Map.Entry<String, SameDiffOp> entry : sameDiff.getOps().entrySet()) {
            SameDiffOp op = entry.getValue();
            if (op.getOp() instanceof LoopCond) {
                // Check if this loop condition is in the specified frame
                String opFrame = getOperationFrame(entry.getKey());
                if (frameName.equals(opFrame)) {
                    loopCondOps.add(entry.getKey());
                }
            }
        }

        return loopCondOps;
    }

    /**
     * Check for infinite loop indicators in a frame
     */
    private String checkForInfiniteLoop(String frameName) {
        // Check if there are NextIteration operations that keep executing
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


    private String analyzeControlFlowContextDetailed(String itemName) {
        StringBuilder context = new StringBuilder();

        // Check if this is a variable that feeds into control flow operations
        List<String> controlFlowConsumers = findControlFlowConsumers(itemName);
        if (!controlFlowConsumers.isEmpty()) {
            context.append("CONTROL FLOW CONTEXT: Variable feeds into control flow operations");

            // Show the current value of this variable across all frames/iterations
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

            // Analyze each control flow consumer in detail
            for (String consumer : controlFlowConsumers) {
                SameDiffOp consumerOp = sameDiff.getOps().get(consumer);
                if (consumerOp != null) {
                    DifferentialFunction func = consumerOp.getOp();
                    context.append("\n  Consumer: ").append(consumer).append(" (").append(func.getClass().getSimpleName()).append(")");

                    // Check execution status with frame information
                    boolean consumerExecuted = isOperationExecuted(consumer);
                    context.append("\n    Executed: ").append(consumerExecuted ? "✓" : "✗");

                    // Show execution steps for this consumer
                    context.append("\n    Execution steps:");
                    boolean foundConsumerSteps = false;

                    for (Object satisfied : dt.getAllSatisfied()) {
                        if (satisfied instanceof ExecStep) {
                            ExecStep step = (ExecStep) satisfied;
                            if (step.getName().equals(consumer)) {
                                foundConsumerSteps = true;
                                FrameIter frameIter = step.getFrameIter();
                                context.append("\n      SATISFIED: ").append(step.getType());
                                if (frameIter != null) {
                                    context.append(" Frame: ").append(frameIter.getFrame())
                                            .append(", Iter: ").append(frameIter.getIteration());
                                }
                            }
                        }
                    }

                    for (Object queued : dt.getAllSatisfiedQueue()) {
                        if (queued instanceof ExecStep) {
                            ExecStep step = (ExecStep) queued;
                            if (step.getName().equals(consumer)) {
                                foundConsumerSteps = true;
                                FrameIter frameIter = step.getFrameIter();
                                context.append("\n      QUEUED: ").append(step.getType());
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

                    // Detailed analysis based on operation type
                    if (func instanceof Switch) {
                        Switch switchOp = (Switch) func;
                        context.append("\n    SWITCH DETAILS:");

                        if (switchOp.getPredicate() != null && switchOp.getPredicate().name().equals(itemName)) {
                            context.append("\n      THIS VARIABLE IS THE SWITCH PREDICATE!");

                            // Show all predicate values across frames
                            for (VarId vid : instances) {
                                SDValue value = nodeValueOutputs.get(vid);
                                context.append("\n        Frame: ").append(vid.getFrame())
                                        .append(", Iter: ").append(vid.getIteration())
                                        .append(", Predicate value: ").append(formatValue(value));
                            }
                        }

                        // Show switch outputs if they exist
                        List<String> switchOutputs = consumerOp.getOutputsOfOp();
                        if (switchOutputs != null) {
                            context.append("\n      Switch outputs:");
                            for (String output : switchOutputs) {
                                boolean outputAvailable = isVariableAvailable(output);
                                context.append("\n        ").append(output).append(": ").append(outputAvailable ? "✓" : "✗");

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

                        // Show all inputs to the merge
                        List<String> mergeInputs = consumerOp.getInputsToOp();
                        if (mergeInputs != null) {
                            context.append("\n      Merge inputs:");
                            for (String input : mergeInputs) {
                                boolean inputAvailable = isVariableAvailable(input);
                                context.append("\n        ").append(input).append(": ").append(inputAvailable ? "✓" : "✗");

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

                        // Show loop condition inputs and outputs
                        List<String> loopInputs = consumerOp.getInputsToOp();
                        if (loopInputs != null) {
                            context.append("\n      Loop condition inputs:");
                            for (String input : loopInputs) {
                                boolean inputAvailable = isVariableAvailable(input);
                                context.append("\n        ").append(input).append(": ").append(inputAvailable ? "✓" : "✗");

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
                                context.append("\n        ").append(output).append(": ").append(outputAvailable ? "✓" : "✗");

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
     * Analyze loop condition issues with detailed values and frame information
     */
    private String analyzeLoopConditionIssuesDetailed(String itemName) {
        StringBuilder analysis = new StringBuilder();

        // Look for loop condition patterns
        String producer = findVariableProducer(itemName);
        if (producer != null) {
            SameDiffOp producerOp = sameDiff.getOps().get(producer);
            if (producerOp != null) {
                DifferentialFunction func = producerOp.getOp();

                // Check if producer is in a loop structure
                String producerFrame = getOperationFrame(producer);
                if (!OUTER_FRAME.equals(producerFrame)) {
                    analysis.append("LOOP CONDITION ANALYSIS:");
                    analysis.append("\n  Producer '").append(producer).append("' is in frame: ").append(producerFrame);

                    // Show producer execution details
                    boolean producerExecuted = isOperationExecuted(producer);
                    analysis.append("\n  Producer executed: ").append(producerExecuted ? "✓" : "✗");

                    // Show all execution steps for the producer
                    analysis.append("\n  Producer execution steps:");
                    boolean foundProducerSteps = false;

                    for (Object satisfied : dt.getAllSatisfied()) {
                        if (satisfied instanceof ExecStep) {
                            ExecStep step = (ExecStep) satisfied;
                            if (step.getName().equals(producer)) {
                                foundProducerSteps = true;
                                FrameIter frameIter = step.getFrameIter();
                                analysis.append("\n    SATISFIED: ").append(step.getType());
                                if (frameIter != null) {
                                    analysis.append(" Frame: ").append(frameIter.getFrame())
                                            .append(", Iter: ").append(frameIter.getIteration());
                                }
                            }
                        }
                    }

                    for (Object queued : dt.getAllSatisfiedQueue()) {
                        if (queued instanceof ExecStep) {
                            ExecStep step = (ExecStep) queued;
                            if (step.getName().equals(producer)) {
                                foundProducerSteps = true;
                                FrameIter frameIter = step.getFrameIter();
                                analysis.append("\n    QUEUED: ").append(step.getType());
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

                    // Check if there are any loop condition operations in this frame
                    List<String> loopCondOps = findLoopConditionOperations(producerFrame);
                    if (!loopCondOps.isEmpty()) {
                        analysis.append("\n  Loop condition operations in frame: ").append(loopCondOps);

                        // Analyze each loop condition operation in detail
                        for (String loopCondOp : loopCondOps) {
                            boolean executed = isOperationExecuted(loopCondOp);
                            analysis.append("\n    ").append(loopCondOp).append(" executed: ").append(executed ? "✓" : "✗");

                            if (!executed) {
                                analysis.append(" <- CRITICAL: Loop condition not evaluated");
                            }

                            // Show execution steps for loop condition
                            analysis.append("\n      Execution steps:");
                            boolean foundLoopSteps = false;

                            for (Object satisfied : dt.getAllSatisfied()) {
                                if (satisfied instanceof ExecStep) {
                                    ExecStep step = (ExecStep) satisfied;
                                    if (step.getName().equals(loopCondOp)) {
                                        foundLoopSteps = true;
                                        FrameIter frameIter = step.getFrameIter();
                                        analysis.append("\n        SATISFIED: ").append(step.getType());
                                        if (frameIter != null) {
                                            analysis.append(" Frame: ").append(frameIter.getFrame())
                                                    .append(", Iter: ").append(frameIter.getIteration());
                                        }
                                    }
                                }
                            }

                            for (Object queued : dt.getAllSatisfiedQueue()) {
                                if (queued instanceof ExecStep) {
                                    ExecStep step = (ExecStep) queued;
                                    if (step.getName().equals(loopCondOp)) {
                                        foundLoopSteps = true;
                                        FrameIter frameIter = step.getFrameIter();
                                        analysis.append("\n        QUEUED: ").append(step.getType());
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

                            // Check what the loop condition depends on
                            SameDiffOp loopOp = sameDiff.getOps().get(loopCondOp);
                            if (loopOp != null) {
                                List<String> loopInputs = loopOp.getInputsToOp();
                                if (loopInputs != null) {
                                    analysis.append("\n      Loop condition inputs:");
                                    for (String input : loopInputs) {
                                        boolean inputAvailable = isVariableAvailable(input);
                                        analysis.append("\n        ").append(input).append(": ").append(inputAvailable ? "✓" : "✗");

                                        if (inputAvailable) {
                                            Object value = getVariableValue(input);
                                            analysis.append(" Value: ").append(formatValue(value));

                                            // Show all instances of this input
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
                                        analysis.append("\n        ").append(output).append(": ").append(outputAvailable ? "✓" : "✗");

                                        if (outputAvailable) {
                                            Object value = getVariableValue(output);
                                            analysis.append(" Value: ").append(formatValue(value));
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Show frame transition operations
                    List<String> frameTransOps = findFrameTransitionOperationsInFrame(producerFrame);
                    if (!frameTransOps.isEmpty()) {
                        analysis.append("\n  Frame transition operations in frame:");
                        for (String transOp : frameTransOps) {
                            boolean executed = isOperationExecuted(transOp);
                            analysis.append("\n    ").append(transOp).append(" executed: ").append(executed ? "✓" : "✗");

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

                    // Check for infinite loop indicators
                    String infiniteLoopCheck = checkForInfiniteLoop(producerFrame);
                    if (infiniteLoopCheck != null) {
                        analysis.append("\n  ").append(infiniteLoopCheck);
                    }
                }
            }
        }

        return analysis.length() > 0 ? analysis.toString() : null;
    }

    /**
     * Get all instances of a variable across different frames/iterations
     */
    private List<VarId> getVariableInstances(String variableName) {
        List<VarId> instances = new ArrayList<>();

        for (VarId vid : nodeValueOutputs.keySet()) {
            if (vid.getVariable().equals(variableName)) {
                instances.add(vid);
            }
        }

        // Sort by frame and iteration for consistent output
        instances.sort((a, b) -> {
            int frameComp = a.getFrame().compareTo(b.getFrame());
            if (frameComp != 0) return frameComp;
            return Integer.compare(a.getIteration(), b.getIteration());
        });

        return instances;
    }

    /**
     * Format a value for display, handling different types appropriately
     */
    /**
     * Format a value for display, handling different types appropriately
     */
    private String formatValue(Object value) {
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
     * Format an INDArray for display with actual values
     */
    private String formatINDArray(INDArray arr) {
        if (arr == null) {
            return "null";
        }

        StringBuilder sb = new StringBuilder();
        sb.append("Array").append(Arrays.toString(arr.shape())).append(" ").append(arr.dataType());

        if (arr.isScalar()) {
            sb.append(" = ").append(arr.getDouble(0));
        } else if (arr.length() <= 10) {
            // Show all values for small arrays
            sb.append(" = [");
            for (int i = 0; i < arr.length(); i++) {
                if (i > 0) sb.append(", ");
                sb.append(arr.getDouble(i));
            }
            sb.append("]");
        } else {
            // Show first few and last few values for larger arrays
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



    /**
     * Find frame transition operations in a specific frame
     */
    private List<String> findFrameTransitionOperationsInFrame(String frameName) {
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
     * Analyze a single remaining item (could be operation or variable) for the failure report
     */
    private String analyzeRemainingItem(String itemName) {
        StringBuilder analysis = new StringBuilder();

        // First check if it's a variable
        if (sameDiff.variableMap().containsKey(itemName)) {
            analysis.append("Variable: ").append(itemName);

            Variable var = sameDiff.getVariables().get(itemName);
            if (var != null) {
                SDVariable sdVar = var.getVariable();
                analysis.append(" (").append(sdVar.getVariableType()).append(")");

                // Check if variable is available
                boolean available = isVariableAvailable(itemName);
                analysis.append("\n  Available: ").append(available ? "✓" : "✗");

                if (!available) {
                    // Find what operation should produce this variable
                    String producer = findVariableProducer(itemName);
                    if (producer != null) {
                        analysis.append("\n  Producer operation: '").append(producer).append("'");

                        // Check if producer has been executed
                        boolean producerExecuted = isOperationExecuted(producer);
                        analysis.append("\n  Producer executed: ").append(producerExecuted ? "✓" : "✗");

                        if (!producerExecuted) {
                            // Analyze why the producer hasn't executed
                            SameDiffOp producerOp = sameDiff.getOps().get(producer);
                            if (producerOp != null) {
                                analysis.append("\n  Producer analysis:");
                                analysis.append("\n    Type: ").append(producerOp.getOp().getClass().getSimpleName());

                                // Check producer's inputs
                                List<String> producerInputs = producerOp.getInputsToOp();
                                if (producerInputs != null && !producerInputs.isEmpty()) {
                                    analysis.append("\n    Producer inputs: ");
                                    for (String input : producerInputs) {
                                        boolean inputAvailable = isVariableAvailable(input);
                                        analysis.append(input).append(inputAvailable ? "✓" : "✗").append(" ");
                                    }
                                }

                                // Check if producer is in execution queue
                                boolean inQueue = false;
                                for (Object queued : dt.getAllSatisfiedQueue()) {
                                    if (queued instanceof ExecStep) {
                                        ExecStep step = (ExecStep) queued;
                                        if (step.getName().equals(producer)) {
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
                    // Variable is available, so why is it in remaining?
                    analysis.append("\n  WARNING: Variable is available but still in remaining set");
                }
            }

            return analysis.toString();
        }

        // Check if it's an operation
        SameDiffOp op = sameDiff.getOps().get(itemName);
        if (op != null) {
            analysis.append("Operation: ").append(itemName);

            DifferentialFunction opFunc = op.getOp();
            analysis.append(" (").append(opFunc.getClass().getSimpleName()).append(")");

            // Check inputs
            List<String> inputs = op.getInputsToOp();
            if (inputs != null && !inputs.isEmpty()) {
                analysis.append("\n  Inputs: ");
                for (int i = 0; i < inputs.size(); i++) {
                    String input = inputs.get(i);
                    boolean available = isVariableAvailable(input);
                    analysis.append(input).append(available ? "✓" : "✗");
                    if (i < inputs.size() - 1) analysis.append(", ");
                }

                // Check for missing inputs
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

            // Check if operation has been executed
            boolean executed = isOperationExecuted(itemName);
            analysis.append("\n  Executed: ").append(executed ? "✓" : "✗");

            if (!executed) {
                // Check execution step status
                boolean foundInQueue = false;
                for (Object queued : dt.getAllSatisfiedQueue()) {
                    if (queued instanceof ExecStep) {
                        ExecStep step = (ExecStep) queued;
                        if (step.getName().equals(itemName)) {
                            foundInQueue = true;
                            break;
                        }
                    }
                }
                analysis.append("\n  In execution queue: ").append(foundInQueue ? "✓" : "✗");
            }

            return analysis.toString();
        }

        // Neither variable nor operation found
        analysis.append("Unknown item: ").append(itemName);
        analysis.append("\n  ERROR: Item not found in variables or operations");

        return analysis.toString();
    }

    /**
     * Update the descendant dependencies
     * So if the graph structure is X -> A, then add all (X,Y,Z,...) -> A to the
     * dependency tracker
     * This is for a specific frame and iteration, for both sides of the dependency
     * (in and out)
     */
    protected void updateDescendantDeps(ExecStep justExecuted, FrameIter outFrameIter) {
        ExecType t = justExecuted.getType();
        String n = justExecuted.getName();
        if (justExecuted.getType() == ExecType.OP) {
            SameDiffOp op = sameDiff.getOps().get(n);
            List<String> outNames = op.getOutputsOfOp();
            for (String s : outNames) {
                Variable v = sameDiff.getVariables().get(s);
                if (v != null) {
                    List<String> inputsToOps = v.getInputsForOp();
                    if (inputsToOps != null) {
                        for (String opName : inputsToOps) {
                            if (subgraphOps.contains(opName)) {
                                // We've just executed X, and there's dependency X -> Y
                                // But, there also might be a Z -> Y that we should mark as needed for Y
                                addDependenciesForOp(opName, outFrameIter);
                            }
                        }
                    }

                    // Also add control dependencies (variable)
                    List<String> cdForOps = v.getControlDepsForOp();
                    if (cdForOps != null) {
                        for (String opName : cdForOps) {
                            if (subgraphOps.contains(opName)) {
                                // We've just executed X, and there's dependency X -> Y
                                // But, there also might be a Z -> Y that we should mark as needed for Y
                                addDependenciesForOp(opName, outFrameIter);
                            }
                        }
                    }
                }

            }
        } else if (t == ExecType.VARIABLE || t == ExecType.CONSTANT || t == ExecType.PLACEHOLDER) {
            Variable v = sameDiff.getVariables().get(n);
            if (v != null) {
                List<String> inputsToOps = v.getInputsForOp();
                if (inputsToOps != null) {
                    for (String opName : inputsToOps) {
                        if (subgraphOps.contains(opName)) {
                            addDependenciesForOp(opName, outFrameIter);
                        }
                    }
                }
            }

        } else if (justExecuted.getType() == ExecType.SWITCH_L || justExecuted.getType() == ExecType.SWITCH_R) {
            SameDiffOp op = sameDiff.getOps().get(n);
            List<String> outNames = op.getOutputsOfOp();
            String branchVarName = (justExecuted.getType() == ExecType.SWITCH_L ? outNames.get(0) : outNames.get(1));
            Variable v = sameDiff.getVariables().get(branchVarName);
            if (v != null) {
                List<String> inputsToOps = v.getInputsForOp();
                if (inputsToOps != null) {
                    for (String opName : inputsToOps) {
                        if (subgraphOps.contains(opName)) {
                            // We've just executed X, and there's dependency X -> Y
                            // But, there also might be a Z -> Y that we should mark as needed for Y
                            addDependenciesForOp(opName, outFrameIter);
                        }
                    }
                }
            }

        } else {
            throw new UnsupportedOperationException("Unknown or not yet implemented exec type: " + justExecuted);
        }
    }

    /**
     * Suppose operation X has just been executed.
     * For X -> someOp, add all dependencies for someOp, i.e., all Z -> someOp
     * (which includes X, but may not only be X)
     *
     * @param opName       Name of the op
     * @param depFrameIter Frame/iteration of the op instance to be executed
     */
    protected void addDependenciesForOp(String opName, FrameIter depFrameIter) {
        SameDiffOp op = sameDiff.getOps().get(opName);
        List<String> inputs = op.getInputsToOp();
        List<String> cdOps = op.getControlDeps();
        List<String> cdVars = op.getVarControlDeps();

        ExecStep es = new ExecStep(ExecType.OP, opName, depFrameIter);
        if (!(op.getOp() instanceof NextIteration) && dt.hasDependency(es)) {
            // Already processed this once. We only add dependencies once per op (for a
            // given frame/iteration)
            return;
        }

        if (op.getOp() instanceof Merge) {
            // Merge ops are a special case: they can be executed with EITHER ONE of the
            // inputs available - unlike every
            // other op, we don't need all inputs, just one, before it can be executed
            Variable v0 = sameDiff.getVariables().get(inputs.get(0));
            Variable v1 = sameDiff.getVariables().get(inputs.get(1));

            ExecStep or0 = getExecStepForVar(v0.getName(), depFrameIter);
            ExecStep or1 = getExecStepForVar(v1.getName(), depFrameIter);
            dt.addOrDependency(es, or0, or1);
        } else if (op.getOp() instanceof NextIteration) {
            // For NextIteration, dependencies should be of the form X(iter) ->
            // NextIter(iter+1)
            FrameIter fi = depFrameIter.clone();
            fi.setIteration(fi.getIteration() + 1);
            es = new ExecStep(ExecType.OP, opName, fi);
            for (String s : inputs) {
                ExecStep req = getExecStepForVar(s, depFrameIter);
                dt.addDependency(es, req);
            }
        } else {
            for (String s : inputs) {
                ExecStep req = getExecStepForVar(s, depFrameIter);
                dt.addDependency(es, req);
            }
        }

        if (cdOps != null) {
            for (String s : cdOps) {
                ExecStep req = getExecStepForVar(s, depFrameIter);
                dt.addDependency(es, req);
            }
        }

    }

    /**
     * Get the ExecStep for the given variable, given execution is happening at the
     * specified frame/iteration
     */
    protected ExecStep getExecStepForVar(String varName, FrameIter frameIter) {
        Variable v = sameDiff.getVariables().get(varName);
        if (v == null) {
            SameDiffOp op = sameDiff.getOps().get(varName);
            if (op != null) {
                // redirect because of rename
                v = sameDiff.getVariables().get(op.getOutputsOfOp().get(0));
            } else {
                throw new IllegalArgumentException("Variable name " + varName + " not found! Renamed?");
            }
        }
        VariableType vt = v.getVariable().getVariableType();
        if (vt == VariableType.VARIABLE) {
            return new ExecStep(ExecType.VARIABLE, v.getVariable().name(), new FrameIter(OUTER_FRAME, 0, null));
        } else if (vt == VariableType.PLACEHOLDER) {
            return new ExecStep(ExecType.PLACEHOLDER, v.getVariable().name(), new FrameIter(OUTER_FRAME, 0, null));
        } else if (vt == VariableType.CONSTANT) {
            return new ExecStep(ExecType.CONSTANT, v.getVariable().name(), new FrameIter(OUTER_FRAME, 0, null));
        } else {
            // Array type. Must be output of an op
            if (v.getOutputOfOp() == null) {
                v = sameDiff.getVariables().get(stripVarSuffix(v.getName()));
            }

            String outOfOp = v.getOutputOfOp();
            SameDiffOp sdo = sameDiff.getOps().get(outOfOp);

            if (sdo == null) {
                throw new IllegalStateException(
                        "Samediff output op named " + v.getName() + " did not have any ops associated with it.");
            }

            if (sdo.getOp() instanceof Switch) {
                // For dependency tracking purposes, we track left and right output branches of
                // switch op separately
                // Otherwise, ops depending both branches will be marked as available if we just
                // rely on "op has been executed"
                List<String> opOutputs = sdo.getOutputsOfOp();
                int idx = opOutputs.indexOf(v.getName());
                if (idx == 0) {
                    // Left branch
                    return new ExecStep(ExecType.SWITCH_L, outOfOp, frameIter);
                } else if (idx == 1) {
                    // Right branch
                    return new ExecStep(ExecType.SWITCH_R, outOfOp, frameIter);
                } else {
                    // Should never happen
                    throw new IllegalStateException(
                            "Expected variable \"" + v.getName() + "\" to be an output of operation \"" +
                                    outOfOp + "\", but op output variables are: " + opOutputs);
                }
            } else if (sdo.getOp() instanceof Enter) {
                Enter e = (Enter) sdo.getOp();

                // For enter ops, "constant=true" enter ops are available for ALL iterations,
                // hence use iter=0
                // For constant=false, these are only available at iteration 0 - so use
                // *current* iteration, same as all other ops
                // (which is this case, won't be triggered on iter > 0 - as desired/expected)
                if (e.isConstant()) {
                    FrameIter fi = frameIter.clone();
                    fi.setIteration(0);

                    // Nested constant enter case: Iteration 0 all the way down...
                    String inVarName = sdo.getInputsToOp().get(0);
                    FrameIter parentFrame = fi.getParentFrame();
                    while (parentFrame != null) {
                        Variable var = sameDiff.getVariables().get(inVarName);
                        if (var.getOutputOfOp() != null) {
                            String opName = var.getOutputOfOp();
                            SameDiffOp sdo2 = sameDiff.getOps().get(opName);
                            if (sdo2.getOp() instanceof Enter) {
                                Enter e2 = (Enter) sdo.getOp();
                                if (e2.isConstant()) {
                                    parentFrame.setIteration(0);
                                    parentFrame = parentFrame.getParentFrame();
                                    inVarName = sdo2.getInputsToOp().get(0);
                                } else {
                                    break;
                                }
                            } else {
                                break;
                            }
                        } else {
                            break;
                        }
                    }

                    return new ExecStep(ExecType.OP, outOfOp, fi);
                }

                // Intentional fall-through to default case
            }
            return new ExecStep(ExecType.OP, outOfOp, frameIter);
        }
    }


    /**
     * Initialize the subgraph - the subgraph and subgraphOps sets
     * This works our what ops and variables we might need to execute to get the
     * requested outputs.
     * In general, this is a subset of the graph.
     *
     * @param variables Set of output variables we need
     */
    protected void initSubgraph(Set<String> variables) {
        log.info("Initializing corrected subgraph for {} variables", variables.size());

        // Build corrected DAG instead of broken mixed dependencies
        ForwardExecutionDAGBuilder builder = new ForwardExecutionDAGBuilder(sameDiff);
        ForwardExecutionDAG dag = builder.buildForwardDAG(variables);

        // Clear the broken data structures
        subgraph.clear();
        subgraphOps.clear();
        zeroInputOpsInSubgraph.clear();

        // Populate with corrected data
        subgraph.addAll(dag.getVariableProducers().keySet());
        subgraphOps.addAll(dag.getOperationNodes().keySet());

        // Find operations with no dependencies (can execute immediately)
        for (ExecutionNode node : dag.getExecutionOrder()) {
            if (node.hasNoDependencies() &&
                    (node.getNodeType() == ExecutionNode.ExecutionNodeType.STANDARD_OP ||
                            node.getNodeType() == ExecutionNode.ExecutionNodeType.CONTROL_FLOW_OP)) {
                zeroInputOpsInSubgraph.add(node.getOperationName());
            }
        }

        log.info("Corrected subgraph: {} variables, {} operations, {} zero-input ops",
                subgraph.size(), subgraphOps.size(), zeroInputOpsInSubgraph.size());
    }


    /**
     *  Get variable dependencies including transformations
     */
    private String[] getVariableDependencies(String varName) {
        // Check if any operations consume this variable and trace their inputs
        List<String> dependencies = new ArrayList<>();
        for (Map.Entry<String, SameDiffOp> entry : sameDiff.getOps().entrySet()) {
            SameDiffOp op = entry.getValue();
            if (op.getOutputsOfOp() != null && op.getOutputsOfOp().contains(varName)) {
                // This operation produces varName, check its inputs
                List<String> inputs = op.getInputsToOp();
                if (inputs != null) {
                    dependencies.addAll(inputs);
                }
            }
        }

        return dependencies.isEmpty() ? null : dependencies.toArray(new String[0]);
    }

    /**
     * FIX 4: Helper method to check variable dependencies
     */
    private boolean isVariableDependentOn(String variable, String target) {
        Set<String> visited = new HashSet<>();
        return checkDependencyPath(variable, target, visited);
    }

    private boolean checkDependencyPath(String current, String target, Set<String> visited) {
        if (visited.contains(current) || current.equals(target)) {
            return current.equals(target);
        }
        visited.add(current);

        // Check through operations
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

    /**
     * Visualize the accumulated DAG data
     */
    private void visualizeDAG(Set<String> requestedOutputs,
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
     * Topological sort to get execution order
     */
    private List<String> topologicalSort(Map<String, Set<String>> dagFlow) {
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
     * Helper for topological sort
     */
    private void topologicalSortHelper(String node, Map<String, Set<String>> dagFlow,
                                       Set<String> visited, Set<String> visiting, List<String> result) {
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
     * Find all nodes reachable from a given starting node
     */
    private Set<String> findReachableNodes(String start, Map<String, Set<String>> dagFlow, Set<String> visited) {
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


    /**
     * Enhanced variable lookup with multiple fallback strategies
     */
    private Variable findVariable(String varName) {
        // Try exact match first
        Variable var = sameDiff.getVariables().get(varName);
        if (var != null) {
            return var;
        }

        // Try without suffix (handles :0, :1, :2 cases)
        String baseVarName = stripVarSuffix(varName);
        if (!baseVarName.equals(varName)) {
            var = sameDiff.getVariables().get(baseVarName);
            if (var != null) {
                log.debug("Found variable using base name: {} -> {}", varName, baseVarName);
                return var;
            }
        }

        // Try with common suffixes if the base name was provided
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

        // Try to find through type conversion chains (e.g., input_ids -> input_ids_int32)
        var = findThroughTypeConversions(varName);
        if (var != null) {
            return var;
        }

        log.debug("Variable not found: {}", varName);
        return null;
    }


    /**
     * Find variables through common type conversion patterns
     */
    private Variable findThroughTypeConversions(String varName) {
        // Common conversion patterns
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
     * Find the operation that produces a given variable
     */
    private SameDiffOp findProducerOperation(String varName) {
        // First, try direct lookup through variable's outputOfOp
        Variable var = findVariable(varName);
        if (var != null && var.getOutputOfOp() != null) {
            return sameDiff.getOps().get(var.getOutputOfOp());
        }

        // Search through all operations to find one that produces this variable
        for (SameDiffOp op : sameDiff.getOps().values()) {
            List<String> outputs = op.getOutputsOfOp();
            if (outputs != null && outputs.contains(varName)) {
                log.debug("Found producer operation for {}: {}", varName, op.getName());
                return op;
            }

            // Also check with base name (without suffix)
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

    /**
     * Preprocess the placeholder values, if required.
     * Mainly reserved for casting in the case of InferenceSession
     *
     * @param placeholders Placeholders to preprocess.
     * @return Preprocessed placeholders
     */
    protected Map<String, SDValue> preprocessValuePlaceholders(Map<String, SDValue> placeholders, At at) {
        return placeholders;
    }

    /**
     * Preprocess the placeholder values, if required.
     * Mainly reserved for casting in the case of InferenceSession
     *
     * @param placeholders Placeholders to preprocess.
     * @return Preprocessed placeholders
     */
    protected Map<String, T> preprocessPlaceholders(Map<String, T> placeholders, At at) {
        return placeholders;
    }

    /**
     * Post process the session output values, if required.
     * Override if required in session subclasses
     *
     * @param output Output to be returned to the user
     * @return Post processed output
     */
    protected Map<String, SDValue> postProcessOutputValues(Map<String, SDValue> output) {
        for (Map.Entry<String, SDValue> entry : output.entrySet()) {
            switch (entry.getValue().getSdValueType()) {
                case DICT:
                    for (Map.Entry<String, INDArray> arr : entry.getValue().getDictValue().entrySet()) {
                        arr.getValue().setCloseable(false);
                    }
                    break;
                case LIST:
                    for (INDArray arr : entry.getValue().getListValue()) {
                        arr.setCloseable(false);
                    }
                    break;
                case TENSOR:
                    entry.getValue().getTensorValue().setCloseable(false);
                    break;
            }

        }

        return output;
    }


    /**
     * Get the constant or variable output - for example, constant array or constant
     * shape.
     * Note that both constants and variables (i.e., VariableType.CONSTANT and
     * VariableType.VARIABLE) are the same
     * for all frames and iterations.
     *
     * @param variableName The name of the variable to get the constant for
     * @return The constant
     */
    public abstract T getConstantOrVariable(String variableName);

    /**
     * Get the parameterized op to execute - for example, the
     * op/DifferentialFunction with all inputs set
     *
     * @param opName            Name of the op
     * @param frameIter         The frame and iteration of the op outputs
     * @param inputs            The inputs to the op (excluding
     *                          constants/placeholders) - for the specific frame +
     *                          iteration
     * @param allIterInputs     The inputs - those that are not iteration-specific
     *                          (mainly Enter op vars, which might be used in all
     *                          iterations but are only executed once on iter 0)
     * @param constAndPhInputs  The constant and placeholder inputs - used for all
     *                          frames/iterations
     * @param allReqVariables   All required variables requested for the current
     *                          session execution (not just the current op outputs)
     * @param otherPlaceholders
     * @return The parameterized op
     */
    public abstract O getAndParameterizeOp(String opName, FrameIter frameIter, Set<VarId> inputs,
                                           Set<VarId> allIterInputs, Set<String> constAndPhInputs,
                                           Map<String, T> placeholderValues, Set<String> allReqVariables, Map<String, SDValue> otherPlaceholders);

    /**
     * Execute the op - calculate INDArrays, or shape info, etc
     *
     * @param op                Operation to exit. This should be parameterized
     *                          (i.e., all inputs set)
     * @param outputFrameIter   The frame and iteration of the outputs
     * @param inputs            The specific input arrays for the op
     * @param allReqVariables   All required variables requested for the current
     *                          session execution (not just the current op outputs)
     * @param otherPlaceHolders
     * @return The outputs of the op
     */
    public abstract ExecutionResult getOutputs(O op, FrameIter outputFrameIter, Set<VarId> inputs,
                                               Set<VarId> allIterInputs, Set<String> constAndPhInputs,
                                               List<Listener> listeners, At at, MultiDataSet batch, Set<String> allReqVariables,
                                               Map<String, SDValue> otherPlaceHolders);

    /**
     * Get the VarId from the specified name. The VarId should be in one or the
     * other of the collections,
     * and only one VarId with that name should exist
     */
    protected static VarId lookup(String name, Collection<VarId> varIds, Collection<VarId> varIds2,
                                  boolean exceptionOnNotFound) {
        VarId vid = varIds == null ? null : lookup(name, varIds, false);
        if (vid == null && varIds2 != null)
            vid = lookup(name, varIds2, false);

        if (vid == null && exceptionOnNotFound) {
            throw new RuntimeException("Could not find VarId for input \"" + name + "\"");
        }
        return vid;
    }

    /**
     * Get the {@link INDArray}
     * associated with the given variable name
     *
     * @param name the variable name
     * @return the list of {@link INDArray}
     */
    public List<INDArray> getTensorArraysInSession(String name, String frame, int iteration, FrameIter parentFrame) {
        DifferentialFunction op = sameDiff.getVariableOutputOp(name);
        if (op == null)
            return null;
        String[] inputs = sameDiff.getInputsForOp(op);
        String[] outputs = sameDiff.getOutputsForOp(op);
        Set<VarId> varIds = new LinkedHashSet<>();
        for (String input : inputs) {
            VarId varId = new VarId(input, frame, iteration, parentFrame);
            varIds.add(varId);
        }

        varIds.addAll(nodeValueOutputs.entrySet().stream().filter(input -> input.getValue() != null &&
                        input.getValue().getSdValueType() == SDValueType.LIST).map(input -> input.getKey())
                .collect(Collectors.toList()));

        VarId lookup = lookup(op.getOwnName(), varIds, false);
        if (lookup == null && op.args().length > 0) {
            SDVariable inTensorArray = op.arg(0); // Dummy variable representing the tensor array
            lookup = lookup(inTensorArray.name(), varIds, false);
            if (lookup != null) {
                List<INDArray> ret = nodeValueOutputs.containsKey(lookup) ? nodeValueOutputs.get(lookup).getListValue()
                        : null;
                if (ret == null && parentFrame != null)
                    return getTensorArraysInSession(name);
            }
            return null;
        }
        List<INDArray> ret = nodeValueOutputs.get(lookup).getListValue();
        if (ret == null && parentFrame != null)
            return getTensorArraysInSession(name);
        return null;
    }

    /**
     * Get the {@link INDArray}
     * associated with the given variable name
     *
     * @param name the variable name
     * @return the list of {@link INDArray}
     */
    public List<INDArray> getTensorArraysInSession(String name) {
        return getTensorArraysInSession(name, OUTER_FRAME, 0, null);
    }

    /**
     * Get the VarId from the specified name. The VarId should be in the collection,
     * and only one VarId with that name should exist
     */
    protected static VarId lookup(String name, Collection<VarId> varIds, boolean exceptionOnNotFound) {
        for (VarId vid : varIds) {
            if (vid.getVariable().equals(name)) {
                return vid;
            }
        }
        if (exceptionOnNotFound) {
            throw new RuntimeException("Could not find VarId to input " + name);
        }
        return null;
    }

    ;

    /**
     * Used in getting the next ExecStep that matches the specified (current)
     * frame/iteration
     */
    @Data
    @AllArgsConstructor
    @NoArgsConstructor
    protected class ExecStepPredicate implements Predicate<ExecStep> {

        protected String currentFrame;
        protected int currentFrameIter;
        protected FrameIter currParentFrame;

        @Override
        public boolean test(ExecStep execStep) {
            // Handle null execStep or frameIter
            if (execStep == null || execStep.getFrameIter() == null) {
                return false;
            }

            // Handle null currentFrame
            String stepFrame = execStep.getFrameIter().getFrame();
            boolean frameMatches = (currentFrame == null && stepFrame == null) ||
                    (currentFrame != null && currentFrame.equals(stepFrame));

            // Check iteration match
            boolean iterationMatches = currentFrameIter == execStep.getFrameIter().getIteration();

            // Handle null parent frames
            FrameIter stepParentFrame = execStep.getFrameIter().getParentFrame();
            boolean parentFrameMatches = (currParentFrame == null && stepParentFrame == null) ||
                    (currParentFrame != null && currParentFrame.equals(stepParentFrame));

            return frameMatches && iterationMatches && parentFrameMatches;
        }
    }

    ;
}
