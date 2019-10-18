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
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.function.Predicate;

import java.util.*;

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
    protected final DependencyTracker<ExecStep, ExecStep> dt = new DependencyTracker<>();

    /**
     * Contains variables we *might* need to execute in process of getting outputs we want.
     * Variables not in this set are definitely not needed to get the requested output variables, but variables that are
     * in this set may not be executed depending on the graph structure - i.e., switch ops, etc
     */
    protected final Set<String> subgraph = new HashSet<>();

    protected final Set<String> subgraphOps = new HashSet<>();
    protected final Set<String> zeroInputOpsInSubgraph = new HashSet<>();

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
        //And also any "zero dependency" ops - i.e., those without any inputs
        ExecStep start = new ExecStep(ExecType.EXEC_START, "", null);   //Dummy dependency to trigger the variables and constants
        for(SDVariable v : sameDiff.variables()){
            VariableType vt = v.getVariableType();
            if(vt == VariableType.VARIABLE || vt == VariableType.CONSTANT){
                ExecType et = vt == VariableType.VARIABLE ? ExecType.VARIABLE : ExecType.CONSTANT;
                ExecStep es = new ExecStep(et, v.getVarName(), new FrameIter(OUTER_FRAME, 0, null));
                dt.addDependency(es, start);

                Variable var = sameDiff.getVariables().get(v.getVarName());
                if(var.getControlDeps() != null){
                    addVarControlDeps(es, var);     //Before this variable can be considered available for use, we need specified op to be executed
                }
            }
        }
        for(String s : phNames){
            ExecStep es = new ExecStep(ExecType.PLACEHOLDER, s, new FrameIter(OUTER_FRAME, 0, null));
            dt.addDependency(es, start);

            Variable var = sameDiff.getVariables().get(s);
            if(var.getControlDeps() != null){
                addVarControlDeps(es, var);     //Before this variable can be considered available for use, we need specified op to be executed
            }
        }
        for(String s : zeroInputOpsInSubgraph){
            ExecStep es = new ExecStep(ExecType.OP, s, new FrameIter(OUTER_FRAME, 0, null));
            dt.addDependency(es, start);
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
        int currentFrameIter = 0;
        FrameIter currParentFrame = null;
        ExecStepPredicate predicate = new ExecStepPredicate();
        while (out.size() < userRequestedUnique.size()) {
            if(!dt.hasNewAllSatisfied()){
                //Haven't got all of the outputs the user requested, but there's nothing left that we can execute. Should not happen.
                execFailed(userRequestedUnique, out, step);
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

            currentFrame = es.getFrameIter().getFrame();
            currentFrameIter = es.getFrameIter().getIteration();
            currParentFrame = es.getFrameIter().getParentFrame();

            log.trace("Beginning execution step {}: {}", step, es);

            FrameIter outFrameIter;
            boolean skipDepUpdate = false;      //Only used for Switch ops, which have slighly different handling...
            boolean skipMarkSatisfied = false;  //Only for enter ops, because of different frame/iter
            if(es.getType() == ExecType.CONSTANT || es.getType() == ExecType.VARIABLE ) {
                VarId vid = newVarId(es.getName(), OUTER_FRAME, 0, null );
                T arr = getConstantOrVariable(es.getName());
                Preconditions.checkNotNull(arr, "Encountered null placeholder array for constant: %s", vid);
                nodeOutputs.put(vid, arr);
                outFrameIter = new FrameIter(OUTER_FRAME, 0, null);
                if(allRequired.contains(es.getName())){
                    //User requested const/variable as one of the outputs
                    out.put(es.getName(), arr);
                }
            } else if(es.getType() == ExecType.PLACEHOLDER) {
                VarId vid = newVarId(es.getName(), OUTER_FRAME, 0, null);
                nodeOutputs.put(vid, placeholderValues.get(es.getName()));
                outFrameIter = new FrameIter(OUTER_FRAME, 0, null);
                if(allRequired.contains(es.getName())){
                    //User requested const/variable as one of the outputs
                    out.put(es.getName(), placeholderValues.get(es.getName()));
                }
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
                    outFrameIter.setIteration(outFrameIter.getIteration());
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
                if(inputNames != null && !inputNames.isEmpty()) {
                    inputs = new HashSet<>();
                    allIterInputs = new HashSet<>();
                    constAndPhInputs = new HashSet<>();
                    List<ExecStep> deps = dl.getDependencies();
                    if(deps != null && !deps.isEmpty()) {
                        for(ExecStep dep : deps) {
                            switch (dep.getType()) {
                                case OP:
                                case SWITCH_L:
                                case SWITCH_R:
                                    //The current execution step depends on one output of the op "dep"
                                    SameDiffOp toExecOp = sameDiff.getOps().get(es.getName());
                                    List<String> inputsToExecOp = toExecOp.getInputsToOp();
                                    SameDiffOp inputOp = sameDiff.getOps().get(dep.getName());
                                    List<String> inputOpOutNames = inputOp.getOutputsOfOp();
                                    for(String s : inputsToExecOp){
                                        if(inputOpOutNames.contains(s)){
                                            VarId vid = new VarId(s, dep.getFrameIter().getFrame(), dep.getFrameIter().getIteration(), dep.getFrameIter().getParentFrame());
                                            inputs.add(vid);
                                        }
                                    }
                                    break;
                                case VARIABLE:
                                    inputs.add(new VarId(dep.getName(), OUTER_FRAME, 0, null));
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



                O parameterizedOp = getAndParameterizeOp(opName, outFrameIter, inputs, allIterInputs, constAndPhInputs, placeholderValues, reqOutputVariablesSet);
                T[] opOutputValues = getOutputs(parameterizedOp, outFrameIter, inputs, allIterInputs, constAndPhInputs, listeners, at, batch, reqOutputVariablesSet);
                List<String> opOutVarNames = op.getOutputsOfOp();

                Preconditions.checkState(opOutputValues.length == opOutVarNames.size(), "Unexpected number of outputs from executed op %s:" +
                                " got %s outputs when %s outputs were expected (%s)", parameterizedOp.getClass().getSimpleName(), opOutputValues.length,
                        opOutVarNames.size(), opOutVarNames);

                for( int i=0; i<opOutputValues.length; i++ ){
                    if(opOutputValues[i] == null && op.getOp() instanceof Switch){
                        //Switch op only forwards the input to one of the outputs
                        continue;
                    }

                    String n = opOutVarNames.get(i);
                    VarId vid = new VarId(n, outFrameIter.getFrame(), outFrameIter.getIteration(), outFrameIter.getParentFrame());
                    nodeOutputs.put(vid, opOutputValues[i]);

                    if(allRequired.contains(n)){
                        out.put(n, opOutputValues[i]);
                    }
                }

                //Post execution: update dependency tracker so we know what is available to execute next
                if(o instanceof Switch){
                    /*
                    Switch is a special case: only one output/branch is considered to exist post execution.
                    Unlike every other type of op, only 1 of 2 output arrays is actually executed
                     */
                    skipDepUpdate = true;
                    skipMarkSatisfied = true;
                    int nullCount = (opOutputValues[0] == null ? 1 : 0) + (opOutputValues[1] == null ? 1 : 0);
                    Preconditions.checkState(nullCount == 1,"Expected exactly one output to be present for switch ops, got %s", nullCount);
                    boolean left = opOutputValues[0] != null;
                    ExecStep branch;
                    if(left){
                        branch = new ExecStep(ExecType.SWITCH_L, es.getName(), es.getFrameIter());
                    } else {
                        branch = new ExecStep(ExecType.SWITCH_R, es.getName(), es.getFrameIter());
                    }
                    updateDescendantDeps(branch, outFrameIter);
                    dt.markSatisfied(branch, true);
                } else if(o instanceof Enter) {
                    //Enter op: we want to say that the inner frame is executed...
                    skipDepUpdate = true;
                    skipMarkSatisfied = true;
                    Enter e = (Enter) o;
                    FrameIter fi = new FrameIter(e.getFrameName(), 0, es.getFrameIter());
                    ExecStep exec = new ExecStep(ExecType.OP, es.getName(), fi);
                    updateDescendantDeps(exec, fi);
                    dt.markSatisfied(exec, true);
                } else if(o instanceof Exit){
                    //Exit op: we want to say that the parent frame is executed...
                    skipDepUpdate = true;
                    skipMarkSatisfied = true;
                    FrameIter fi = es.getFrameIter().getParentFrame();
                    ExecStep exec = new ExecStep(ExecType.OP, es.getName(), fi);
                    updateDescendantDeps(exec, fi);
                    dt.markSatisfied(exec, true);
                }

                /*
                Edge case for TensorFlow import control dependencies: for some reason, TF allows op control dependencies
                like /while/x -> SomeConstant - i.e., a constant depending on something inside a scope.
                This should be handled with an enter op, but TF doesn't always use this :/
                Note that this is equivalent to marking the control dependency as satisfied on the first iteration
                TODO double check that this is exactly the same behaviour as TF - otherwise this approach might fail in
                     some rare cases that rely on the constant/variable not being available
                 */
                List<String> cdFor = op.getControlDepFor();
                if(cdFor != null){
                    ExecStep cdEs = new ExecStep(ExecType.CONTROL_DEP, opName, null);
                    if(!dt.isSatisfied(cdEs)){
                        dt.markSatisfied(cdEs, true);
                    }
                }

            } else {
                throw new RuntimeException("Unknown ExecStep: " + es);
            }

            if(!skipDepUpdate) {
                updateDescendantDeps(es, outFrameIter);
            }
            if(!skipMarkSatisfied) {
                dt.markSatisfied(es, true);
            }




            step++;
        }


        //TODO under what circumstances should we clear the nodeOutputs map? Doesn't matter for memory, because we already close/deallocate though...

        out = postProcessOutput(out);
        return out;
    }

    protected void addVarControlDeps(ExecStep es, Variable v){
        List<String> cds = v.getControlDeps();
        for(String s : cds){
            ExecStep controlES = new ExecStep(ExecType.CONTROL_DEP, s, null);
            dt.addDependency(es, controlES);    //Before this variable can be considered available for use, we need specified op to be executed
        }
    }

    protected void execFailed(Set<String> userRequestedUnique, Map<String,T> out, int step){
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
//        System.out.println(sameDiff.summary());
        throw new IllegalStateException(s);
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
                        if(subgraphOps.contains(opName)){
                            //We've just executed X, and there's dependency X -> Y
                            //But, there also might be a Z -> Y that we should mark as needed for Y
                            addDependenciesForOp(opName, outFrameIter);
                        }
                    }
                }

                //Also add control dependencies
                List<String> cdForOps = v.getControlDepsForOp();
                if(cdForOps != null){
                    for(String opName : cdForOps){
                        if(subgraphOps.contains(opName)){
                            //We've just executed X, and there's dependency X -> Y
                            //But, there also might be a Z -> Y that we should mark as needed for Y
                            addDependenciesForOp(opName, outFrameIter);
                        }
                    }
                }
            }
        } else if( t == ExecType.VARIABLE || t == ExecType.CONSTANT || t == ExecType.PLACEHOLDER ){
            Variable v = sameDiff.getVariables().get(n);
            List<String> inputsToOps = v.getInputsForOp();
            if(inputsToOps != null ){
                for(String opName : inputsToOps){
                    if(subgraphOps.contains(opName)){
                        addDependenciesForOp(opName, outFrameIter);
                    }
                }
            }
        } else if(justExecuted.getType() == ExecType.SWITCH_L || justExecuted.getType() == ExecType.SWITCH_R){
            SameDiffOp op = sameDiff.getOps().get(n);
            List<String> outNames = op.getOutputsOfOp();
            String branchVarName = (justExecuted.getType() == ExecType.SWITCH_L ? outNames.get(0) : outNames.get(1));
            Variable v = sameDiff.getVariables().get(branchVarName);
            List<String> inputsToOps = v.getInputsForOp();
            if(inputsToOps != null ){
                for(String opName : inputsToOps){
                    if(subgraphOps.contains(opName)){
                        //We've just executed X, and there's dependency X -> Y
                        //But, there also might be a Z -> Y that we should mark as needed for Y
                        addDependenciesForOp(opName, outFrameIter);
                    }
                }
            }
        } else {
            throw new UnsupportedOperationException("Not yet implemented exec type - execution: " + justExecuted);
        }
    }

    /**
     * Suppose operation X has just been executed.
     * For X -> someOp, add all dependencies for someOp, i.e., all Z -> someOp
     * (which includes X, but may not only be X)
     *
     * @param opName
     * @param depFrameIter
     */
    protected void addDependenciesForOp(String opName, FrameIter depFrameIter){
        SameDiffOp op = sameDiff.getOps().get(opName);
        List<String> inputs = op.getInputsToOp();

        ExecStep es = new ExecStep(ExecType.OP, opName, depFrameIter);
        if(!(op.getOp() instanceof NextIteration) && dt.hasDependency(es)){
            //Already processed this once. We only add dependencies once per op (for a given frame/iteration)
            return;
        }

        if(op.getOp() instanceof Merge) {
            //Merge ops are a special case: they can be executed with EITHER ONE of the inputs available - unlike every
            // other op, we don't need all inputs, just one, before it can be executed
            Variable v0 = sameDiff.getVariables().get(inputs.get(0));
            Variable v1 = sameDiff.getVariables().get(inputs.get(1));

            ExecStep or0 = getExecStepForVar(v0.getName(), depFrameIter);
            ExecStep or1 = getExecStepForVar(v1.getName(), depFrameIter);
            dt.addOrDependency(es, or0, or1);
        } else if(op.getOp() instanceof NextIteration ){
            //For NextIteration, dependencies should be of the form X(iter) -> NextIter(iter+1)
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
    }

    protected ExecStep getExecStepForVar(String varName, FrameIter frameIter){
        Variable v = sameDiff.getVariables().get(varName);
        VariableType vt = v.getVariable().getVariableType();
        if (vt == VariableType.VARIABLE) {
            return new ExecStep(ExecType.VARIABLE, v.getVariable().getVarName(), new FrameIter(OUTER_FRAME, 0, null));
        } else if (vt == VariableType.PLACEHOLDER) {
            return new ExecStep(ExecType.PLACEHOLDER, v.getVariable().getVarName(), new FrameIter(OUTER_FRAME, 0, null));
        } else if (vt == VariableType.CONSTANT) {
            return new ExecStep(ExecType.CONSTANT, v.getVariable().getVarName(), new FrameIter(OUTER_FRAME, 0, null));
        } else {
            //Array type. Must be output of an op
            String outOfOp = v.getOutputOfOp();
            SameDiffOp sdo = sameDiff.getOps().get(outOfOp);
            if(sdo.getOp() instanceof Switch){
                //For dependency tracking purposes, we track left and right output branches of switch op separately
                //Otherwise, ops depending both branches will be marked as available if we just rely on "op has been executed"
                List<String> opOutputs = sdo.getOutputsOfOp();
                int idx = opOutputs.indexOf(v.getName());
                if(idx == 0){
                    //Left branch
                    return new ExecStep(ExecType.SWITCH_L, outOfOp, frameIter);
                } else if(idx == 1){
                    //Right branch
                    return new ExecStep(ExecType.SWITCH_R, outOfOp, frameIter);
                } else {
                    //Should never happen
                    throw new IllegalStateException("Expected variable \"" + v.getName() + "\" to be an output of operation \"" +
                            outOfOp + "\", but op output variables are: " + opOutputs);
                }
            } else if(sdo.getOp() instanceof Enter){
                Enter e = (Enter) sdo.getOp();

                //For enter ops, "constant=true" enter ops are available for ALL iterations, hence use iter=0
                //For constant=false, these are only available at iteration 0 - so use *current* iteration, same as all other ops
                // (which is this case, won't be triggered on iter > 0 - as desired/expected)
                if(e.isConstant()){
                    FrameIter fi = frameIter;
                    if(fi.getIteration() != 0){
                        fi = fi.clone();
                        fi.setIteration(0);
                    }
                    return new ExecStep(ExecType.OP, outOfOp, fi);
                }
            }
            return new ExecStep(ExecType.OP, outOfOp, frameIter);
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
                if (numInputs == 0 && opName != null) {
                    zeroInputOpsInSubgraph.add(opName);
                }
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
     * ExecType: Execution type, as used in ExecStep<br>
     * OP: Operation execution<br>
     * VARIABLE: Variable "execution", mainly used to trigger ops that depend on the variable<br>
     * CONSTANT: As per variable<br>
     * PLACEHOLDER: As per variable<br>
     * SWITCH_L and SWITCH_R: This is a bit of a hack to account for the fact that only one of
     * the switch branches (left or right) will ever be available; without this, once the switch op is executed, we'll
     * (incorrectly) conclude that *both* branches can be executed<br>
     * EXEC_START: Start of execution<br>
     * CONTROL_DEP: Control dependency for op. Used for TF import, due to its odd "constant depends on op in a frame" behaviour
     */
    protected enum ExecType {OP, VARIABLE, CONSTANT, PLACEHOLDER, SWITCH_L, SWITCH_R, EXEC_START, CONTROL_DEP};

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
