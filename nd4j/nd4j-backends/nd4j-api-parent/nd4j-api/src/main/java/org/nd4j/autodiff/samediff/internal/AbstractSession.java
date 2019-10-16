/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.autodiff.samediff.internal;

import lombok.*;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.listeners.At;
import org.nd4j.autodiff.listeners.Listener;
import org.nd4j.autodiff.listeners.Operation;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ops.impl.controlflow.compat.*;

import java.util.*;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.function.Predicate;

/**
 * Additional functionality to add:
 * - Workspaces support
 * - Proper cache support
 *
 * @param <T> Node output type - for example, INDArray, shape, etc depending on what we're calculating
 * @param <O> Op type
 */
@Slf4j
public abstract class AbstractSession<T, O> {

    //All execution happens in a frame... this is the name of the main/outer frame
    public static final String OUTER_FRAME = "main";

    protected final SameDiff sameDiff;
    @Getter
    protected final Map<VarId, T> nodeOutputs = new HashMap<>();
    @Getter
    protected final Map<VarId, List<T>> tensorArrays = new HashMap<>(); //Stores the outputs for a TensorArray ops
//    protected final Queue<VarId> availableForExec = new LinkedList<>();
//    protected final Set<VarId> availableForExecSet = new HashSet<>();       //Same content as the queue, but used for O(1) contains instead of ordered removal
    protected final DependencyTracker<ExecStep, ExecStep> dt = new DependencyTracker<>();

    /**
     * Contains variables we *might* need to execute in process of getting outputs we want.
     * Variables not in this set are definitely not needed to get the requested output variables, but variables that are
     * in this set may not be executed depending on the graph structure - i.e., switch ops, etc
     */
    protected final Set<String> subgraph = new HashSet<>();

    protected final Set<String> subgraphOps = new HashSet<>();

    /**
     * Stores what variables are required to calculate the specific variable. These inputs could be inputs to an op that
     * calculates the variable's value, or it could be a control dependenci
     * Keys: variable (in specific frame/iteration) to be executed
     * Values: inputs to that node (inc. frame and iteration), unordered - needed for execution of op giving variable
     */
    protected final Map<VarId, Set<VarId>> execInputs = new HashMap<>();

    /**
     * As per execInputs map - with the different that the iteration number should be ignored (i.e., always 0)
     * Reason: Enter nodes - these are executed once
     * Example: EnterOp(x) -> LoopCondition(less(x,y)): less op requires "X" on all iterations which is the output of the
     * enter op, which is only executed for iteration 0 in a frame.
     */
    protected final Map<VarId, Set<VarId>> execInputsAllIter = new HashMap<>();

    /**
     * Contains the set set of constant and placeholders inputs
     * Essentially the same as the execInputs map, but the constants and placeholders are used for calculating all instances
     * of a variable - i.e., the input (constant/placeholder) applies to all frames and iterations.
     * Keys: variable (any/all frame/iteration) to be executed
     * Values: constant or placeholder needed for execution of op giving variable
     */
    protected final Map<String, Set<String>> execConstInputs = new HashMap<>();
    /**
     * Map for exit ops. This is used to determine where an exit op should exit to.
     * Values added on enter ops. Note that it's not sufficient to
     * Key: frame name (for enter/exit nodes).
     * Value: parent frame name + iteration
     */
    @Getter
    protected final Map<String, FrameIter> frameParents = new HashMap<>();


    public AbstractSession(@NonNull SameDiff sameDiff) {
        this.sameDiff = sameDiff;
    }

    public boolean contains(String variable, String frame, int iteration, FrameIter parentFrameIter){
        VarId varId = newVarId(variable, frame, iteration, parentFrameIter);
        return nodeOutputs.containsKey(varId);
    }

    /**
     * Get a previously calculated output; throws an exception if the output does not exist
     */
    public T get(String variable, String frame, int iteration, FrameIter parentFrameIter) {
        return get(variable, frame, iteration, parentFrameIter, true);
    }

    /**
     * Get a previously calculated output
     * @param enforceExistence If true: throw an exception if the array does not exist
     */
    public T get(String variable, String frame, int iteration, FrameIter parentFrameIter, boolean enforceExistence) {
        //TODO eventually we'll cache and reuse VarId objects here to avoid garbage generation on lookup etc
        VarId varId = newVarId(variable, frame, iteration, parentFrameIter);
        T out = nodeOutputs.get(varId);
        if(enforceExistence) {
            Preconditions.checkNotNull(out, "No output found for variable %s (frame %s, iteration %s)", variable, frame, iteration);
        }
        return out;
    }

    public VarId newVarId(String variable, String frame, int iteration, FrameIter parentFrameIter) {
        //TODO eventually we'll cache and reuse VarId objects here to avoid garbage generation on lookup
        return new VarId(variable, frame, iteration, parentFrameIter);
    }

    public VarId newVarId(String variable, FrameIter frameIter) {
        return newVarId(variable, frameIter.getFrame(), frameIter.getIteration(), frameIter.getParentFrame());
    }

    /**
     * @deprecated Use {@link #output(List, Map, MultiDataSet, Collection, List, At)}.
     *
     * @param training Uses Operation.TRAINING if true, otherwise Operation.INFERENCE
     */
    @Deprecated
    public Map<String, T> output(@NonNull List<String> variables, Map<String, T> placeholderValues,
            MultiDataSet batch, Collection<String> requiredActivations, boolean training, At at){
        if(at == null){
            if(training)
                at = At.defaultAt(Operation.TRAINING);
            else
                at = At.defaultAt(Operation.INFERENCE);
        }
        return output(variables, placeholderValues, batch, requiredActivations, Collections.<Listener>emptyList(), at);
    }

    /**
     * Get the output of the session - i.e., perform inference/forward pass
     *
     * @param variables         Name of the variables we want the arrays/activations for
     * @param placeholderValues The placeholder values (if any).
     * @param batch             The batch data, used to call Listener.opExecution
     * @param requiredActivations  Additional activations that are required.  Won't be outputed, but opExecution will be called.  May be null.
     * @return The specified variable values, optionally in the specified workspace
     */
    public Map<String, T> output(@NonNull List<String> variables, Map<String, T> placeholderValues,
            MultiDataSet batch, Collection<String> requiredActivations, List<Listener> listeners, At at) {

        Preconditions.checkState(!variables.isEmpty() || !requiredActivations.isEmpty(), "Variables to perform forward pass for must not be empty");

        if(requiredActivations == null)
            requiredActivations = Collections.emptyList();

        if(at == null)
            at = At.defaultAt();

        //Step 0: validation - that variables exist, placeholders have arrays, etc
        for (String s : variables) {
            Preconditions.checkState(sameDiff.variableMap().containsKey(s), "Requested output variable %s does not exist in SameDiff instance", s);
        }

        Set<String> reqOutputVariablesSet = new HashSet<>(variables);

        placeholderValues = preprocessPlaceholders(placeholderValues);

        //Clear state from past
        dt.clear();
        subgraph.clear();
        subgraphOps.clear();
        execInputs.clear();
        execInputsAllIter.clear();
        execConstInputs.clear();
        nodeOutputs.clear();            //TODO eventually we'll have (optional) cache here for later execs... main challenge is detecting in-place array modifications and invalidating old results. And overall memory use...
        tensorArrays.clear();

        //Step 1: determine subgraph structure we actually need to execute
        //Basic plan: work backwards from the variables we want, based on the graph structure, to work out what
        // we actually need to execute
        Set<String> userRequestedUnique = new HashSet<>(variables);
        Set<String> allRequired = new HashSet<>(requiredActivations);
        allRequired.addAll(variables);
        initSubgraph(allRequired);

        //Step 1a: Check that we have required placeholders
        List<String> phNames = sameDiff.inputs();
        if(placeholderValues == null || !placeholderValues.keySet().containsAll(phNames)){
            /* We only have a subset of all placeholders
            Validate that we have all *required* placeholder values. Some might not be needed to calculate the requested outputs
            A placeholder is required if:
            (a) It's one of the requested outputs
            (b) It's required to calculate any of the ops in the subgraph
             */
            for(String s : phNames){
                boolean required = false;
                if(variables.contains(s)){      //TODO List.contains - O(N)
                    required = true;
                }
                if(!required){
                    Variable v = sameDiff.getVariables().get(s);
                    if(v.getInputsForOp() != null){
                        for(String s2 : v.getInputsForOp()){
                            if(subgraph.contains(s2)){
                                //Placeholder is required
                                required = true;
                                break;
                            }
                        }
                    }
                }

                if(required && (placeholderValues == null || !placeholderValues.containsKey(s))){

                    // Some Keras layers (like GRU) do different things depending on whether the model is training.
                    // We provide this value directly.
                    if(s.endsWith("keras_learning_phase")){
                        placeholderValues.put(s, (T) Nd4j.scalar(at.operation().isTrainingPhase()));
                    } else {
                        throw new IllegalStateException(
                                "An input placeholder \"" + s + "\" is required to calculate the requested outputs," +
                                        " but a placeholder value was not provided");
                    }
                }
            }
        }

        //Mark the (required) variables, constants and placeholders as available via dependency tracker
        ExecStep start = new ExecStep(ExecType.EXEC_START, "", null);   //Dummy dependency to trigger the variables and constants
        for(SDVariable v : sameDiff.variables()){
            VariableType vt = v.getVariableType();
            if(vt == VariableType.VARIABLE || vt == VariableType.CONSTANT){
                ExecType et = vt == VariableType.VARIABLE ? ExecType.VARIABLE : ExecType.CONSTANT;
                ExecStep es = new ExecStep(et, v.getVarName(), new FrameIter(OUTER_FRAME, 0, null));
                dt.addDependency(es, start);
            }
        }
        dt.markSatisfied(start, true);



        //Step 2: execute (in any order, but not switching to new frame/iteration until all from current frame/iter are done),
        // until we have all required nodeOutputs
        /*
        The idea is simple: we start off with a set of "available to execute" variables - just the placeholders and
        constants at this point.

        Then, we remove an "available to execute" node and execute it. Execution may be:
        (a) For constants and placeholders: just looking up the value
        (b) For variables as outputs of ops: actually executing the op

        After execution, we look at the graph structure and determine what that now executed/calculated variable is
        an input to. If all inputs are available for the op, we mark all output variables of that op as available for execution.

        We stop computation once all the required outputs are available. At this point, subgraph may NOT be empty - for example,
        switch ops may cause entire branches of the graph to be skipped.
         */

        Map<String, T> out = new HashMap<>();
        int step = 0;
        String currentFrame = OUTER_FRAME;
        final int currentFrameIter = 0;
        FrameIter currParentFrame = null;
        ExecStepPredicate predicate = new ExecStepPredicate();
        while (out.size() < userRequestedUnique.size()) {
            if(!dt.hasNewAllSatisfied()){
                int missingCount = userRequestedUnique.size() - out.size();
                StringBuilder sb = new StringBuilder();
                sb.append("No variable are available for execution at step ")
                        .append(step).append(": ").append(missingCount).append(" values remaining");
                Set<String> missing = new HashSet<>();
                for(String s : userRequestedUnique){
                    if(!out.containsKey(s)){
                        missing.add(s);
                    }
                }
                if(missingCount <= 10){
                    sb.append(". Missing variables: ");
                    sb.append(missing);
                } else {
                    sb.append(". First 10 missing variables: ");
                    Iterator<String> iter = missing.iterator();
                    for( int i=0; i<10 && iter.hasNext(); i++ ){
                        if(i > 0)
                            sb.append(",");
                        sb.append(iter.next());
                    }
                }
                String s = sb.toString();
                throw new IllegalStateException(s);
            }

            //Get variable in the current frame/iteration and execute it's corresponding op
            //If no more ops exist for the current frame/iter, we'll switch to the next frame/iter
            //The idea is to not mix the order of execution of ops in different frames/iters
            predicate.setCurrentFrame(currentFrame);
            predicate.setCurrentFrameIter(currentFrameIter);
            predicate.setCurrParentFrame(currParentFrame);

            ExecStep es = dt.getFirstNewAllSatisfiedMatching(predicate);
            if(es == null){
                es = dt.getNewAllSatisfied();

                if(es.getType() == ExecType.OP) {
                    //Trigger frame/iter transition
                    FrameIter fi = es.getFrameIter();
                    onFrameIterTransition(currentFrame, currentFrameIter, currParentFrame,
                            fi.getFrame(), fi.getIteration(), fi.getParentFrame());
                }
            }

            log.trace("Beginning execution step {}: {}", step, es);

            FrameIter outFrameIter;
            if(es.getType() == ExecType.CONSTANT || es.getType() == ExecType.VARIABLE ) {
                VarId vid = newVarId(es.getName(), OUTER_FRAME, 0, null );
                T arr = getConstantOrVariable(es.getName());
                Preconditions.checkNotNull(arr, "Encountered null placeholder array for constant: %s", vid);
                nodeOutputs.put(vid, arr);
                dt.markSatisfied(es, true);
                outFrameIter = new FrameIter(OUTER_FRAME, 0, null);
            } else if(es.getType() == ExecType.PLACEHOLDER) {
                VarId vid = newVarId(es.getName(), OUTER_FRAME, 0, null);
                nodeOutputs.put(vid, placeholderValues.get(es.getName()));
                dt.markSatisfied(es, true);
                outFrameIter = new FrameIter(OUTER_FRAME, 0, null);
            } else if(es.getType() == ExecType.OP){
                String opName = es.getName();
                SameDiffOp op = sameDiff.getOps().get(opName);
                DifferentialFunction o = op.getOp();



                if (o instanceof Enter) {
                    //Enter op: output is variable in a new (specified) frame, iteration 0.
                    //Parent is current (input) frame
                    String outFrame = ((Enter) o).getFrameName();
                    outFrameIter = new FrameIter(outFrame, 0, es.getFrameIter());
                } else if (o instanceof Exit) {
                    //Exit node forwards input to parent frame
                    String outFrame = es.getFrameIter().getParentFrame().getFrame();
                    int outIter = es.getFrameIter().getParentFrame().getIteration();
                    FrameIter outParentFrame = es.getFrameIter().getParentFrame().getParentFrame();
                    outFrameIter = new FrameIter(outFrame, outIter, outParentFrame);
                } else if (o instanceof NextIteration) {
                    //NextIteration op: forwards its single input to its output varible in the current frame, but increments the iteration number
                    outFrameIter = es.getFrameIter().clone();
                    outFrameIter.setIteration(outFrameIter.getIteration() + 1);
                } else {
                    //Standard ops - output variable has same frame and iteration number as the input(s)
                    //Also loopCond, merge, while, etc
                    outFrameIter = es.getFrameIter();
                }


                //Resolve the inputs to this execution step (op) to actual arrays
                Set<VarId> inputs = null;
                Set<VarId> allIterInputs = null;
                Set<String> constAndPhInputs = null;
                DependencyList<ExecStep, ExecStep> dl = dt.getDependencies(es);

                List<String> inputNames = op.getInputsToOp();
                if(inputs != null && !inputs.isEmpty()) {
                    inputs = new HashSet<>();
                    allIterInputs = new HashSet<>();
                    constAndPhInputs = new HashSet<>();
                    List<ExecStep> deps = dl.getDependencies();
                    if(deps != null && !deps.isEmpty()) {
                        for(ExecStep dep : deps) {
                            switch (dep.getType()) {
                                case OP:
                                    inputs.add(es.toVarId());
                                    break;
                                case VARIABLE:
                                    inputs.add(new VarId(dep.getName(), OUTER_FRAME, 0, null));
                                    break;
                                case CONSTANT:
                                case PLACEHOLDER:
                                    constAndPhInputs.add(dep.getName());
                                    break;
                                case SWITCH_L:
                                case SWITCH_R:
                                default:
                                    throw new UnsupportedOperationException("Not yet implemented: " + dep.getType());
                            }
                        }
                    }
                }



                O parameterizedOp = getAndParameterizeOp(opName, outFrameIter, inputs, allIterInputs, constAndPhInputs, placeholderValues, reqOutputVariablesSet);
                T[] opOutputValues = getOutputs(parameterizedOp, outFrameIter, inputs, allIterInputs, constAndPhInputs, listeners, at, batch, reqOutputVariablesSet);
                List<String> opOutVarNames = op.getOutputsOfOp();

                Preconditions.checkState(opOutputValues.length == opOutVarNames.size(), "Unexpected number of outputs from executed op %s:" +
                                " got %s outputs when %s outputs were expected (%s)", parameterizedOp.getClass().getSimpleName(), opOutputValues.length,
                        opOutVarNames.size(), opOutVarNames);

                for( int i=0; i<opOutputValues.length; i++ ){
                    String n = opOutVarNames.get(i);
                    VarId vid = new VarId(n, outFrameIter.getFrame(), outFrameIter.getIteration(), outFrameIter.getParentFrame());
                    nodeOutputs.put(vid, opOutputValues[i]);

                    if(allRequired.contains(n)){
                        out.put(n, opOutputValues[i]);
                    }
                }

                //Post execution: update dependency tracker so we know what is available to execute next
                if(o instanceof Switch){
                    throw new IllegalStateException("Not yet implemented: SWITCH op");
                } else {
                    dt.markSatisfied(es, true);
                }

            } else {
                throw new RuntimeException("Unknown ExecStep: " + es);
            }

            updateDescendantDeps(es, outFrameIter);

            step++;
        }


        //TODO under what circumstances should we clear the nodeOutputs map?

        out = postProcessOutput(out);
        return out;
    }

    protected void updateDescendantDeps(ExecStep justExecuted, FrameIter outFrameIter){

        ExecType t = justExecuted.getType();
        String n = justExecuted.getName();
        if(justExecuted.getType() == ExecType.OP){
            SameDiffOp op = sameDiff.getOps().get(n);
            List<String> outNames = op.getOutputsOfOp();
            for(String s : outNames){
                Variable v = sameDiff.getVariables().get(s);
                List<String> inputsToOps = v.getInputsForOp();
                if(inputsToOps != null ){
                    for(String opName : inputsToOps){
                        if(subgraph.contains(opName)){
                            //We've just executed X, and there's dependency X -> Y
                            //But, there also might be a Z -> Y that we should mark as needed for Y



                            ExecStep es = new ExecStep(ExecType.OP, opName, outFrameIter);

                        }
                    }
                }
            }
        } else if( t == ExecType.VARIABLE || t == ExecType.CONSTANT || t == ExecType.PLACEHOLDER ){
            Variable v = sameDiff.getVariables().get(n);
            List<String> inputsToOps = v.getInputsForOp();
            if(inputsToOps != null ){
                for(String opName : inputsToOps){
                    if(subgraph.contains(opName)){
                        ExecStep es = new ExecStep(ExecType.OP, opName, outFrameIter);
                    }
                }
            }
        } else if(justExecuted.getType() == ExecType.SWITCH_L || justExecuted.getType() == ExecType.SWITCH_R){
            throw new UnsupportedOperationException("Not yet implemented");
        } else {
            throw new UnsupportedOperationException("Not yet implemented");
        }


    }

    protected void initSubgraph(Set<String> variables) {
        //Step 1: determine subgraph structure we actually need to execute
        Queue<String> processingQueue = new LinkedList<>(variables);

        //Note subgraph initially should include placeholders and constants
        while (!processingQueue.isEmpty()) {
            String varName = processingQueue.remove();
            String opName = (sameDiff.getVariableOutputOp(varName) == null ? null : sameDiff.getVariableOutputOp(varName).getOwnName());

            if (!subgraph.contains(varName)) {
                String[] opInputs = opName == null ? null : sameDiff.getInputsForOp(sameDiff.getOpById(opName));
                List<String> controlDeps = sameDiff.getVariables().get(varName).getControlDeps();
                int numInputs = (opInputs == null ? 0 : opInputs.length);
                if (controlDeps != null) {
                    //Also count variable control dependencies as inputs - even a constant may not be available for use
                    // until after execution of some other ops (for example, in conditional operations)
                    numInputs += controlDeps.size();
                }
//                if (numInputs == 0) {
//                    VarId vid = newVarId(varName, OUTER_FRAME, 0, null);
//                    if(!availableForExecSet.contains(vid)) {
//                        availableForExec.add(vid);
//                        availableForExecSet.add(vid);
//                    }
//                    execInputs.put(vid, new HashSet<VarId>());
//                }
                subgraph.add(varName);

                if(opName != null){
                    subgraphOps.add(opName);
                }

                if(controlDeps != null){
                    //If variable has control dependencies, it's not available right away... to make it available,
                    // we need the "inputs" to be available first. This is mainly used for TF import.
                    for(String s : controlDeps){
                        if(!subgraph.contains(s)){
                            processingQueue.add(s);
                        }
                    }
                }
            }

            if (opName != null) {
                //To execute op - and hence get this variable: need inputs to that op
                String[] inputs = sameDiff.getInputsForOp(sameDiff.getOpById(opName));
                for (String s2 : inputs) {
                    if (!subgraph.contains(s2)) {
                        processingQueue.add(s2);
                    }
                }

                //To execute op - and hence get this variable - we also need control deps
                List<String> opControlDeps = sameDiff.getOps().get(opName).getControlDeps();
                if (opControlDeps != null) {
                    for (String s2 : opControlDeps) {
                        if (!subgraph.contains(s2)) {
                            processingQueue.add(s2);
                        }
                    }
                }
            }
        }
    }

    /**
     * Preprocess the placeholder values, if required.
     * Mainly reserved for casting in the case of InferenceSession
     * @param placeholders Placeholders to preprocess.
     * @return Preprocessed placeholders
     */
    protected Map<String,T> preprocessPlaceholders(Map<String,T> placeholders){
        return placeholders;
    }

    protected Map<String,T> postProcessOutput(Map<String,T> output){
        return output;
    }

    /**
     * Get the constant or variable output - for example, constant array or constant shape.
     * Note that both constants and variables (i.e., VariableType.CONSTANT and VariableType.VARIABLE) are the same
     * for all frames and iterations.
     *
     * @param variableName The name of the variable to get the constant for
     * @return The constant
     */
    public abstract T getConstantOrVariable(String variableName);

    /**
     * Get the parameterized op to execute - for example, the op/DifferentialFunction with all inputs set
     *
     * @param opName           Name of the op
     * @param frameIter        The frame and iteration of the op outputs
     * @param inputs           The inputs to the op (excluding constants/placeholders) - for the specific frame + iteration
     * @param allIterInputs    The inputs - those that are not iteration-specific (mainly Enter op vars, which might be used in all iterations but are only executed once on iter 0)
     * @param constAndPhInputs The constant and placeholder inputs - used for all frames/iterations
     * @param allReqVariables  All required variables requested for the current session execution (not just the current op outputs)
     * @return The parameterized op
     */
    public abstract O getAndParameterizeOp(String opName, FrameIter frameIter, Set<VarId> inputs, Set<VarId> allIterInputs, Set<String> constAndPhInputs,
                                           Map<String,T> placeholderValues, Set<String> allReqVariables);

    /**
     * Execute the op - calculate INDArrays, or shape info, etc
     *
     * @param op              Operation to exit. This should be parameterized (i.e., all inputs set)
     * @param outputFrameIter The frame and iteration of the outputs
     * @param inputs          The specific input arrays for the op
     * @param allReqVariables All required variables requested for the current session execution (not just the current op outputs)
     * @return The outputs of the op
     */
    public abstract T[] getOutputs(O op, FrameIter outputFrameIter, Set<VarId> inputs, Set<VarId> allIterInputs, Set<String> constAndPhInputs,
                                   List<Listener> listeners, At at, MultiDataSet batch, Set<String> allReqVariables);

    /**
     * This method is used to record that the specified input is required for calculating the specified output.
     * While the graph structure itself provides us with the (input vars) -> op -> (output vars) type structure, it
     * doesn't tell us exactly which array copy (i.e., variable + frame + iteration) to use as which copy of the output
     * variable (variable + frame + iteration).
     * <p>
     * This method is basically used to store information we need to parameterize ops for execution later
     *
     * @param isConstOrPh If true: inputVar is either a constant or a placeholder
     * @param inputVar    Input variable (i.e., the X in (X, ...) -> op -> (forVariable,...))
     * @param forVariable Output variable (i.e., the Y in (inputVar, ...) -> op -> (Y,...))
     */
    protected void addToExecInputs(boolean isConstOrPh, VarId inputVar, VarId forVariable) {
        if (!subgraph.contains(forVariable.getVariable()))
            return;     //Not needed to calculate requested outputs, so no need to record it's inputs

        if (isConstOrPh) {
            //Mark that outVar needs to use placeholder/constant (same regardless of frame/iter)
            if (!execConstInputs.containsKey(forVariable.getVariable()))
                execConstInputs.put(forVariable.getVariable(), new HashSet<String>());
            execConstInputs.get(forVariable.getVariable()).add(inputVar.getVariable());
        } else {
            //Mark that outVar needs this specific executedVar (i.e., specific frame/iteration)
            //However, in the case of enter nodes, they are available for ALL iterations (used in loop conditions, for example)
            Variable v = sameDiff.getVariables().get(inputVar.getVariable());
            boolean isEnter = sameDiff.getVariableOutputOp(v.getVariable().getVarName()) instanceof Enter;

            if(isEnter){
                VarId iter0 = forVariable;
                if(iter0.getIteration() != 0){
                    iter0 = newVarId(iter0.getVariable(), iter0.getFrame(), 0, forVariable.getParentFrame());
                }

                Variable var = sameDiff.getVariables().get(inputVar.getVariable());
                Enter e = (Enter) sameDiff.getOps().get(var.getOutputOfOp()).getOp();
                if(e.isConstant()){
                    //For enter nodes that are constants, we want iteration 0 in all frames in the heirarchy
                    //For example, const -> Enter(a) -> Enter(b) -> op; in this case, the input to Op (at any frame/iteration) should should
                    // be the constant value - which is recorded as (frame="a",iter=0,parent=(frame="b",iter=0))
                    iter0.setParentFrame(iter0.getParentFrame().clone());
                    FrameIter toZero = iter0.getParentFrame();
                    while(toZero != null){
                        toZero.setIteration(0);
                        toZero = toZero.getParentFrame();
                    }
                }

                if(!execInputsAllIter.containsKey(iter0))
                    execInputsAllIter.put(iter0, new HashSet<VarId>());
                execInputsAllIter.get(iter0).add(inputVar);
            } else {
                //Most variables
                if (!execInputs.containsKey(forVariable))
                    execInputs.put(forVariable, new HashSet<VarId>());
                execInputs.get(forVariable).add(inputVar);
            }
        }
    }


    protected void onFrameIterTransition(String fromFrame, int fromIter, FrameIter parentFrom, String toFrame, int toIter, FrameIter parentTo){
        //No-op by default
    }

    protected static VarId lookup(String name, Collection<VarId> varIds, Collection<VarId> varIds2, boolean exceptionOnNotFound){
        VarId vid = varIds == null ? null : lookup(name, varIds, false);
        if(vid == null && varIds2 != null)
            vid = lookup(name, varIds2, false);

        if(vid == null && exceptionOnNotFound) {
            throw new RuntimeException("Could not find VarId for input \"" + name + "\"");
        }
        return vid;
    }


    protected static VarId lookup(String name, Collection<VarId> varIds, boolean exceptionOnNotFound){
        for(VarId vid : varIds){
            if(vid.getVariable().equals(name)){
                return vid;
            }
        }
        if(exceptionOnNotFound) {
            throw new RuntimeException("Could not find VarId to input " + name);
        }
        return null;
    }

    /*
    VarId: identifies a variable in a specific frame and frame iteration
    Used for 2 places:
    (a) to identify variables that are available for execution
    (b) to store results
     */
    @Data
    @AllArgsConstructor
    public static class VarId {
        private String variable;
        private String frame;
        private int iteration;
        private FrameIter parentFrame;

        @Override
        public String toString() {
            return "VarId(\"" + variable + "\",\"" + frame + "\"," + iteration + ",parent=" + parentFrame + ")";
        }

        public FrameIter toFrameIter() {
            return new FrameIter(frame, iteration, parentFrame);
        }
    }

    /*
    FrameIter: Identifies frame + iteration. Used mainly for for exit nodes
     */
    @Data
    @AllArgsConstructor
    public static class FrameIter {
        private String frame;
        private int iteration;
        private FrameIter parentFrame;

        @Override
        public String toString(){
            return "(\"" + frame + "\"," + iteration + (parentFrame == null ? "" : ",parent=" + parentFrame.toString()) + ")";
        }

        @Override
        public FrameIter clone(){
            return new FrameIter(frame, iteration, (parentFrame == null ? null : parentFrame.clone()));
        }
    }

    /**
     * Note on SWITCH_L and SWITCH_R: This is a bit of a hack to account for the fact that only one of
     * the branches (left or right) will ever be available; without this, once the switch op is executed, we'll
     * (incorrectly) conclude that *both* branches can be executed
     */
    protected enum ExecType {OP, VARIABLE, CONSTANT, PLACEHOLDER, SWITCH_L, SWITCH_R, EXEC_START};

    @Getter
    @EqualsAndHashCode
    protected static class ExecStep {
        protected final ExecType type;
        protected final String name;
        protected final FrameIter frameIter;

        protected ExecStep(@NonNull ExecType execType, @NonNull String name, FrameIter frameIter){
            this.type = execType;
            this.name = name;
            this.frameIter = frameIter;
        }

        protected VarId toVarId(){
            return new VarId(name, frameIter.getFrame(), frameIter.getIteration(), frameIter.getParentFrame());
        }

        @Override
        public String toString(){
            return "ExecStep(" + type + ",name=\"" + name + "\"," + frameIter + ")";
        }
    }

    @Data
    @AllArgsConstructor
    @NoArgsConstructor
    protected class ExecStepPredicate implements Predicate<ExecStep> {

        protected String currentFrame;
        protected int currentFrameIter;
        protected FrameIter currParentFrame;

        @Override
        public boolean test(ExecStep execStep) {
            return currentFrame.equals(execStep.getFrameIter().getFrame()) &&
                    currentFrameIter == execStep.getFrameIter().getIteration() &&
                    (currParentFrame == null && execStep.getFrameIter().getParentFrame() == null ||
                            currParentFrame.equals(execStep.getFrameIter().getParentFrame()));
        }
    };

}
