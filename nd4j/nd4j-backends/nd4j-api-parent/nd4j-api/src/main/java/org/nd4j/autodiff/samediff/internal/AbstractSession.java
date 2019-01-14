package org.nd4j.autodiff.samediff.internal;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.Getter;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ops.impl.controlflow.compat.*;

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
    protected final Queue<VarId> availableForExec = new LinkedList<>();
    /**
     * Contains variables we *might* need to execute in process of getting outputs we want.
     * Variables not in this set are definitely not needed to get the requested output variables, but variables that are
     * in this set may not be executed depending on the graph structure - i.e., switch ops, etc
     */
    protected final Set<String> subgraph = new HashSet<>();
    /**
     * Stores what variables are required to calculate the specific variable. These inputs could be inputs to an op that
     * calculates the variable's value, or it could be a control dependenci
     * Keys: variable (in specific frame/iteration) to be executed
     * Values: inputs to that node (inc. frame and iteration), unordered - needed for execution of op giving variable
     */
    protected final Map<VarId, Set<VarId>> execInputs = new HashMap<>();
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

    public boolean contains(String variable, String frame, int iteration){
        VarId varId = newVarId(variable, frame, iteration);
        return nodeOutputs.containsKey(varId);
    }

    /**
     * Get a previously calculated output; throws an exception if the output does not exist
     */
    public T get(String variable, String frame, int iteration) {
        return get(variable, frame, iteration, true);
    }

    /**
     * Get a previously calculated output
     * @param enforceExistence If true: throw an exception if the array does not exist
     */
    public T get(String variable, String frame, int iteration, boolean enforceExistence) {
        //TODO eventually we'll cache and reuse VarId objects here to avoid garbage generation on lookup etc
        VarId varId = newVarId(variable, frame, iteration);
        T out = nodeOutputs.get(varId);
        if(enforceExistence) {
            Preconditions.checkNotNull(out, "No output found for variable %s (frame %s, iteration %s)", variable, frame, iteration);
        }
        return out;
    }

    public VarId newVarId(String variable, String frame, int iteration) {
        //TODO eventually we'll cache and reuse VarId objects here to avoid garbage generation on lookup
        return new VarId(variable, frame, iteration);
    }

    public VarId newVarId(String variable, FrameIter frameIter) {
        return newVarId(variable, frameIter.getFrame(), frameIter.getIteration());
    }

    /**
     * Get the output of the session - i.e., perform inference/forward pass
     *
     * @param variables         Name of the variables we want the arrays/activations for
     * @param placeholderValues The placeholder values (if any).
     * @return The specified variable values, optionally in the specified workspace
     */
    public Map<String, T> output(@NonNull List<String> variables, Map<String, T> placeholderValues) {
        Preconditions.checkState(!variables.isEmpty(), "Variables to perform forward pass for must not be empty");
        List<String> sdPlaceholders = sameDiff.inputs();
        Preconditions.checkState(sdPlaceholders == null || sdPlaceholders.isEmpty()
                        || (placeholderValues != null && placeholderValues.size() == sdPlaceholders.size() &&
                        placeholderValues.keySet().containsAll(sdPlaceholders)),
                "Invalid placeholders: SameDiff instance has placeholders %s, got placeholders %s", sdPlaceholders,
                (placeholderValues == null ? null : placeholderValues.keySet()));


        //Step 0: validation - that variables exist, placeholders have arrays, etc
        for (String s : variables) {
            Preconditions.checkState(sameDiff.variableMap().containsKey(s), "Requested output variable %s does not exist in SameDiff instance", s);
        }

        placeholderValues = preprocessPlaceholders(placeholderValues);

        //Clear state from past
        availableForExec.clear();
        subgraph.clear();
        execInputs.clear();
        execConstInputs.clear();
        nodeOutputs.clear();            //TODO eventually we'll have cache here for later execs... main challenge is detecting in-place array modifications and invalidating old results
        tensorArrays.clear();

        //Step 1: determine subgraph structure we actually need to execute
        //Basic plan: work backwards from the variables we want, based on the graph structure, to work out what
        // we actually need to execute
        initSubgraph(variables);

        //Step 2: execute in any order, until we have all required nodeOutputs
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
        while (out.size() < variables.size()) {
            Preconditions.checkState(availableForExec.size() > 0, "No variables are available for execution at execution step %s", step);

            //Get any variable and execute it's corresponding op
            VarId varToExec = availableForExec.remove();
            if (nodeOutputs.containsKey(varToExec)) {
                //Already processed this one. May occur if execution was triggered by a different output of a multi-output op
                //But we'll still update its descendants to ensure they are marked as available
                if (variables.contains(varToExec.getVariable())) {  //Check if required output
                    out.put(varToExec.getVariable(), nodeOutputs.get(varToExec));
                }
                updateDescendentsForExec(varToExec);
                continue;
            }

            //Get inputs to this variable. May be actual op inputs, or just control dependencies
            Set<VarId> inputsToVar = execInputs.get(varToExec);
            Set<String> constPhForVar = execConstInputs.get(varToExec.getVariable());

            log.trace("Beginning execution step {}: variable {}", (step++), varToExec);

            if (sameDiff.getVariable(varToExec.getVariable()).isPlaceHolder()) {
                //Variable is placeholder: do lookup
                nodeOutputs.put(varToExec, placeholderValues.get(varToExec.getVariable()));
                updateDescendentsForExec(varToExec); //Check + mark descendants as available for exec
                if (variables.contains(varToExec.getVariable())) {  //Check if required output
                    out.put(varToExec.getVariable(), placeholderValues.get(varToExec.getVariable()));
                }
            } else if (sameDiff.getVariable(varToExec.getVariable()).isConstant() ||
                    sameDiff.getVariable(varToExec.getVariable()).getVariableType() == VariableType.VARIABLE) {
                //Variable is constant: do lookup
                //OR variable is VARIABLE type - i.e., a trainable parameter...
                T phArr = getConstantOrVariable(varToExec.getVariable());
                Preconditions.checkNotNull(phArr, "Encountered null placeholder array for constant: %s", varToExec);
                nodeOutputs.put(varToExec, phArr);
                updateDescendentsForExec(varToExec); //Check + mark descendants as available for exec
                if (variables.contains(varToExec.getVariable())) {  //Check if required output
                    out.put(varToExec.getVariable(), phArr);
                }


            } else if (sameDiff.getVariableOutputFunction(varToExec.getVariable()) != null) {
                //Variable is the output of an op -> execute op
                String opName = sameDiff.getVariables().get(varToExec.getVariable()).getOutputOfOp();

                //Execute op
                FrameIter frameIter = varToExec.toFrameIter();
                O parameterizedOp = getAndParameterizeOp(opName, frameIter, inputsToVar, constPhForVar, placeholderValues);
                T[] opOutputValues = getOutputs(parameterizedOp, frameIter, inputsToVar, constPhForVar);


                //Post execution: work out what is now available for exec
                String[] opOutputVarNames = sameDiff.getFunctionById(opName).outputVariablesNames();

                Preconditions.checkState(opOutputValues.length == opOutputVarNames.length, "Unexpected number of outputs from executed op %s:" +
                                " got %s outputs when %s outputs were expected (%s)", parameterizedOp.getClass().getSimpleName(), opOutputValues.length,
                        opOutputVarNames.length, opOutputVarNames);

                for (int i = 0; i < opOutputVarNames.length; i++) {
                    if (opOutputValues[i] == null && parameterizedOp instanceof Switch) {
                        //Skip null - for switch op only. Switch op forwards input to only one of its outputs
                        //All other ops should not
                        continue;
                    }

                    Preconditions.checkNotNull(opOutputValues[i], "Encountered null output (output %s) for op %s at execution step %s", i, parameterizedOp.getClass().getSimpleName(), step);

                    VarId outputVarId;
                    if (parameterizedOp instanceof Enter) {
                        //Enter op: output is variable in a new (specified) frame, iteration 0
                        String frame = ((Enter) parameterizedOp).getFrameName();
                        outputVarId = newVarId(opOutputVarNames[i], frame, varToExec.getIteration());
                    } else if (parameterizedOp instanceof Exit) {
                        //Exit node forwards input to parent frame (which is already reflected in varToExec)
                        outputVarId = newVarId(opOutputVarNames[i], varToExec.getFrame(), varToExec.getIteration());
                    } else if (parameterizedOp instanceof NextIteration) {
                        //NextIteration op: forwards its single input to its output varible in the current frame, but increments the iteration number
                        //Note that varToExec has already had its iteration number incremented by 1 (relative to its input) in updateDescendentsForExec... so don't increment here
                        outputVarId = newVarId(opOutputVarNames[i], varToExec.getFrame(), varToExec.getIteration());
                    } else if (parameterizedOp instanceof LoopCond) {
                        //LoopCond just forwards input to output
                        outputVarId = newVarId(opOutputVarNames[i], varToExec.getFrame(), varToExec.getIteration());
                    } else {
                        //Standard ops - output variable has same frame and iteration number as the input(s)
                        outputVarId = newVarId(opOutputVarNames[i], varToExec.getFrame(), varToExec.getIteration());
                    }

                    nodeOutputs.put(outputVarId, opOutputValues[i]);
                    updateDescendentsForExec(outputVarId); //Check + mark descendants as available for exec

                    if (variables.contains(opOutputVarNames[i])) {  //Check if required output
                        out.put(opOutputVarNames[i], opOutputValues[i]);
                    }
                }
            } else {
                Variable v = sameDiff.getVariables().get(varToExec.getVariable());
                throw new IllegalStateException("Unable to execute variable " + varToExec + " of type " + v.getVariable().getVariableType());
            }
        }


        //TODO under what circumstances should we clear the nodeOutputs map?
        //TODO when should we close the workspace? (Might want to leave it open if we expect to re-use)

        return out;
    }

    protected void initSubgraph(List<String> variables) {
        //Step 1: determine subgraph structure we actually need to execute
        Queue<String> processingQueue = new LinkedList<>(variables);

        //Note subgraph initially should include placeholders and constants
        while (!processingQueue.isEmpty()) {
            String varName = processingQueue.remove();
            String opName = (sameDiff.getVariableOutputFunction(varName) == null ? null : sameDiff.getVariableOutputFunction(varName).getOwnName());

            if (!subgraph.contains(varName)) {
                String[] opInputs = opName == null ? null : sameDiff.getInputsForFunction(sameDiff.getFunctionById(opName));
                List<String> controlDeps = sameDiff.getVariables().get(varName).getControlDeps();
                int numInputs = (opInputs == null ? 0 : opInputs.length);
                if (controlDeps != null) {
                    //Also count variable control dependencies as inputs - even a constant may not be available for use
                    // until after execution of some other ops (for example, in conditional operations)
                    numInputs += controlDeps.size();
                }
                if (numInputs == 0) {
                    VarId vid = newVarId(varName, OUTER_FRAME, 0);
                    availableForExec.add(vid);
                    execInputs.put(vid, new HashSet<VarId>());
                }
                subgraph.add(varName);

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
                String[] inputs = sameDiff.getInputsForFunction(sameDiff.getFunctionById(opName));
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
     * This method should be called for a variable once it's array is ready for use.
     * For example, post op execution, etc
     *
     * @param executedVar Variable that was just executed
     */
    protected void updateDescendentsForExec(VarId executedVar) {
        String varName = executedVar.getVariable();
        Variable var = sameDiff.getVariables().get(executedVar.getVariable());
        //Find any ops (or variables with control dependencies) that this is required for execution of and check if now available for exec
        List<String> l = sameDiff.getVariables().get(executedVar.getVariable()).getInputsForOp();
        String[] inputForOps = l == null ? null : l.toArray(new String[l.size()]);  //Just executed variable is input to these ops
        List<String> controlDepForVars = var.getControlDepsForVar();                //Just executed variable is a control dependency for these variables
        List<String> controlDepForOps = var.getControlDepsForOp();                  //Just executed variable is a control dependency for these ops


        SDVariable v = var.getVariable();
        boolean isConstOrPhInput = v.isPlaceHolder() || v.isConstant();

        //After a variable becomes available, we should look at the ops this is an input to, and check if we can execute this op now...
        if (inputForOps != null) {
            for (String opName : inputForOps) {


                if (sameDiff.getFunctionById(opName) instanceof Merge) {
                    //Merge op: available for execution when *any* of its inputs are available. But only mark it for exec once...
                    List<String> opOutputs = sameDiff.getOps().get(opName).getOutputsOfOp();
                    Preconditions.checkState(opOutputs.size() == 1, "Expected only 1 output variable for merge op, got %s", opOutputs);
                    VarId outVarId = newVarId(opOutputs.get(0), executedVar.getFrame(), executedVar.getIteration());
                    if (!nodeOutputs.containsKey(outVarId) && subgraph.contains(outVarId.getVariable())) {
                        availableForExec.add(outVarId);
                        log.trace("Marked merge op ({}) variable {} as available for execution: input {} is now available", opName, outVarId, executedVar);
                    }

                    //Mark that we need the specified input to calculate this output
                    addToExecInputs(isConstOrPhInput, executedVar, outVarId);
                    continue;
                } else if (sameDiff.getFunctionById(opName) instanceof Enter) {
                    //Enter node: available for exec when any of its inputs are available for exec
                    // Note input feeds from one frame to another
                    List<String> opOutputs = sameDiff.getOps().get(opName).getOutputsOfOp();
                    Preconditions.checkState(opOutputs.size() == 1, "Expected only 1 output variable for enter op, got %s", opOutputs);
                    Enter e = (Enter) sameDiff.getFunctionById(opName);
                    VarId outVarId = newVarId(opOutputs.get(0), e.getFrameName(), 0);
                    if (!nodeOutputs.containsKey(outVarId) && subgraph.contains(outVarId.getVariable())) {
                        availableForExec.add(outVarId);
                        log.trace("Marked enter op ({}) variable {} as available for execution: input {} is now available", opName, outVarId, executedVar);
                    }

                    //Also record the parent frame: we'll need this when we get to the corresponding exit ops
                    frameParents.put(e.getFrameName(), executedVar.toFrameIter());

                    //Mark that we need the specified input to calculate this output
                    addToExecInputs(isConstOrPhInput, executedVar, outVarId);
                    continue;
                } else if (sameDiff.getFunctionById(opName) instanceof Exit) {
                    //Exit node forwards input to parent frame
                    List<String> opOutputs = sameDiff.getOps().get(opName).getOutputsOfOp();
                    FrameIter parentFrame = frameParents.get(executedVar.getFrame());
                    Preconditions.checkNotNull(parentFrame, "Parent frame must not be null for exit op: variable to exec is %s", executedVar);

                    VarId outVarId = new VarId(opOutputs.get(0), parentFrame.getFrame(), parentFrame.getIteration());
                    if (!nodeOutputs.containsKey(outVarId) && subgraph.contains(outVarId.getVariable())) {
                        availableForExec.add(outVarId);
                        log.trace("Marked Exit op ({}) variable {} as available for execution: input {} is now available", opName, outVarId, executedVar);
                    }

                    addToExecInputs(isConstOrPhInput, executedVar, outVarId);
                } else if (sameDiff.getFunctionById(opName) instanceof NextIteration) {
                    //NextIteration is available for execution when its single input is available
                    //NextIteration op: forwards its single input to the output of the current frame, but increments the iteration number
                    List<String> opOutputs = sameDiff.getOps().get(opName).getOutputsOfOp();
                    Preconditions.checkState(opOutputs.size() == 1, "Expected exactly 1 output for NextIteration op: got %s", opOutputs);
                    VarId outVarId = newVarId(opOutputs.get(0), executedVar.getFrame(), executedVar.getIteration() + 1);

                    if (!nodeOutputs.containsKey(outVarId) && subgraph.contains(outVarId.getVariable())) {
                        availableForExec.add(outVarId);
                        log.trace("Marked NextIteration op ({}) variable {} as available for execution: input {} is now available", opName, outVarId, executedVar);
                    }

                    //Mark that we need the specified input to calculate this output
                    addToExecInputs(isConstOrPhInput, executedVar, outVarId);
                    continue;
                }
                //Note for LoopCond: just forwards input to output - so basically handle it the same as other ops here


                //Can execute this op - and hence get it's output variables - if all inputs (and control deps) are available
                String[] inputsThisOp = sameDiff.getFunctionById(opName).argNames();
                boolean allInputsAvailable = true;
                if (inputsThisOp != null) {
                    allInputsAvailable = allInputsAvailable(inputsThisOp, executedVar);
                }

                //Check Op control dependencies
                List<String> opControlDeps = sameDiff.getOps().get(opName).getControlDeps();
                if (opControlDeps != null && allInputsAvailable) {
                    for (String cd : opControlDeps) {
                        VarId vcd = newVarId(cd, executedVar.getFrame(), executedVar.getIteration());
                        if (!nodeOutputs.containsKey(vcd)) {
                            allInputsAvailable = false;
                            break;
                        }
                    }
                }

                List<String> opOutputs = sameDiff.getOps().get(opName).getOutputsOfOp();
                if (opOutputs != null) {

                    for (String s : opOutputs) {
                        //The input (for normal ops - not Enter/Exit/NextIteration) have the same frame and iteration number as the just executed var
                        //Exception 1 to this: constants. If variable is a constant, then it's always iteration 0 of the main frame  (unless variable control dep exists)
                        //Exception 2 to this: placeholders. As above
                        SDVariable sdv = sameDiff.getVariable(s);
                        Variable variable = sameDiff.getVariables().get(s);
                        VarId outVarId;
                        if (sdv.isConstant() || sdv.isPlaceHolder()) {
                            //Constant
                            if(variable.getControlDeps() == null || var.getControlDeps().isEmpty()){
                                //Standard case - do a lookup of placeholder/constant
                                outVarId = newVarId(s, OUTER_FRAME, 0);
                            } else {
                                //Edge case: control dependency x -> constant exists
                                //We should look up based on x's frame/iteration
                                outVarId = newVarId(s, executedVar.getFrame(), executedVar.getIteration());
                            }
                        } else {
                            //Normal (non-constant)
                            outVarId = newVarId(s, executedVar.getFrame(), executedVar.getIteration());
                        }

                        //Mark that we need the specified input to calculate this output
                        addToExecInputs(isConstOrPhInput, executedVar, outVarId);

                        //Check variable control dependencies, for each of the op outputs
                        if(allInputsAvailable && variable.getControlDeps() != null && !variable.getControlDeps().isEmpty()){
                            //If one of the op outputs has a control dependency input, make sure this is available
                            // before executing the op
                            //For example, if z=add(x,y) and control dependency A->z exists, then don't execute op until A is available
                            for(String cd : variable.getControlDeps()){
                                Variable cdVar = sameDiff.getVariables().get(cd);
                                VarId cdVarId = null;
                                if (cdVar.getVariable().isConstant() || cdVar.getVariable().isPlaceHolder()) {
                                    //Constant
                                    if(variable.getControlDeps() == null || var.getControlDeps().isEmpty()){
                                        //Standard case - do a lookup of placeholder/constant
                                        cdVarId = newVarId(cd, OUTER_FRAME, 0);
                                    } else {
                                        //Edge case: control dependency x -> constant -> thisOutput exists
                                        //We should look up based on x's frame/iteration
                                        cdVarId = newVarId(cd, executedVar.getFrame(), executedVar.getIteration());
                                    }
                                } else {
                                    //Normal (non-constant)
                                    cdVarId = newVarId(cd, executedVar.getFrame(), executedVar.getIteration());
                                }
                                allInputsAvailable &= nodeOutputs.containsKey(cdVarId);
                                if(!allInputsAvailable)
                                    break;
                            }
                        }
                    }

                    if (allInputsAvailable) {
                        //Op can be executed -> variables as output are available for exec

                        for (String s : opOutputs) {
                            if (!subgraph.contains(s))
                                continue;       //Don't need this variable to calculate requested outputs - so don't mark as available for execution
                            VarId vid = newVarId(s, executedVar.getFrame(), executedVar.getIteration());
                            availableForExec.add(vid);      //TODO let's avoid adding and logging duplicates... result is same (checks if already executed) but logged multiple times here...
                            log.trace("Marked variable as available for execution: {} - output of op {} ({}) with op inputs {}", vid, opName,
                                    sameDiff.getFunctionById(opName).getClass().getSimpleName(), (inputsThisOp == null ? "<none>" : Arrays.toString(inputsThisOp)));
                        }
                    }
                }

            }
        }

        //Also check variable control dependencies... if control dependency varX->varY exists and varY is a constant/placeholder/variable,
        // then it's not going to be triggered by the op-based check above
        if(controlDepForVars != null){
            for(String s : controlDepForVars){
                if (!subgraph.contains(s))
                    continue;       //Don't need this variable to calculate requested outputs - so don't mark as available for execution

                SDVariable depFor = sameDiff.getVariable(s);
                if(depFor.getVariableType() != VariableType.ARRAY){
                    //Control dependency executedVar -> s exists, where "s" is not the output of an op
                    //Even thought this is a constant, we'll inherit the frame and iteration from the control dependency
                    // otherwise, we lose this frame/iteration information for any downstream variables using the constant within a frame
                    VarId outVarId = newVarId(s, executedVar.getFrame(), executedVar.getIteration());
                    availableForExec.add(outVarId);
                    log.trace("Marked variable as available for execution: {} - control dependency {} -> {} exists", outVarId, executedVar.getVariable(), s);
                } else {
                    //Another edge case: OpX has output varY (with no inputs), and control dependency executedVar -> varY exists
                    //We should check if OpX is now available for execution...
                    //Similarly, if we have OpX with inputs, but we're only waiting on a varible control dependency Z -> X
                    // then we might not get triggered as available for exec above either
                    String opName = sameDiff.getVariables().get(s).getOutputOfOp();
                    if(opName != null){
                        SameDiffOp op = sameDiff.getOps().get(opName);
                        boolean allInputsAvailable = true;
                        if(op.getInputsToOp() != null && !op.getInputsToOp().isEmpty()){
                            List<String> inputList = op.getInputsToOp();
                            allInputsAvailable = allInputsAvailable(inputList.toArray(new String[inputList.size()]), executedVar);
                        }

                        if(allInputsAvailable && op.getControlDeps() != null){
                            for(String cd : op.getControlDeps()){
                                VarId vid = newVarId(cd, executedVar.getFrame(), executedVar.getIteration());     //Note: is array type, therefore has same frame/iter as parent
                                allInputsAvailable &= nodeOutputs.containsKey(vid);
                                if(!allInputsAvailable)
                                    break;
                            }
                        }
                        if(allInputsAvailable){
                            for(String opOutput : op.getOutputsOfOp()){
                                Variable v2 = sameDiff.getVariables().get(opOutput);
                                if(v2.getControlDeps() != null){
                                    for(String s2 : v2.getControlDeps()){
                                        VarId vid = newVarId(s2, executedVar.getFrame(), executedVar.getIteration());     //Note: is array type, therefore has same frame/iter as parent
                                        allInputsAvailable &= nodeOutputs.containsKey(vid);
                                        if(!allInputsAvailable)
                                            break;
                                    }
                                }
                            }
                        }

                        if(allInputsAvailable){
                            VarId outVarId = newVarId(s, executedVar.getFrame(), executedVar.getIteration());
                            availableForExec.add(outVarId);
                            log.trace("Marked variable as available for execution: {} - is output of op {} with no inputs (but has control dependencies)", outVarId, op.getName());
                        }
                    }
                }
            }
        }

        //Edge case: if control dependency varX->opY exists, and opY doesn't have any inputs, it also can't be triggeered
        // (made available for execution) by any of the previous checks. For any ops that DO have inputs, they will
        // be triggered already
        if(controlDepForOps != null){
            for(String opName : controlDepForOps){
                SameDiffOp op = sameDiff.getOps().get(opName);
                if(op.getInputsToOp() == null || op.getInputsToOp().isEmpty()){
                    for(String out : op.getOutputsOfOp()){
                        if (!subgraph.contains(out))
                            continue;       //Don't need this variable to calculate requested outputs - so don't mark as available for execution

                        //TODO is it possible to have both variable and op control dependencies??
                        VarId outVarId = newVarId(out, OUTER_FRAME, 0);
                        availableForExec.add(outVarId);
                        log.trace("Marked variable as available for execution: {} - op control dependency variable {} -> op {} exists", outVarId, executedVar.getVariable(), opName);
                    }
                }
            }
        }
    }

    protected boolean allInputsAvailable(String[] inputsThisOp, VarId executedVar){
        for (String in : inputsThisOp) {
            //The input (for normal ops - not Enter/Exit/NextIteration) have the same frame and iteration number as the just executed var
            //Exception 1 to this: constants. If variable is a constant, then it's always iteration 0 of the main frame (unless variable control dep exists)
            //Exception 2 to this: placeholders. As above
            //TODO Add SameDiff.isConstant(String) method... or SDVariable.isConstant() (or both)
            SDVariable sdv = sameDiff.getVariable(in);
            Variable variable = sameDiff.getVariables().get(in);
            VarId vid;
            if (sdv.isConstant() || sdv.isPlaceHolder()) {
                //Constant
                if(variable.getControlDeps() == null || variable.getControlDeps().isEmpty()){
                    //Standard case - do a lookup of placeholder/constant
                    vid = newVarId(in, OUTER_FRAME, 0);
                } else {
                    //Edge case: control dependency x -> constant exists
                    //We should look up based on x's frame/iteration
                    vid = newVarId(in, executedVar.getFrame(), executedVar.getIteration());
                }
            } else {
                //Normal (non-constant)
                vid = newVarId(in, executedVar.getFrame(), executedVar.getIteration());
            }

            if (!nodeOutputs.containsKey(vid)) {
                return false;
            }
        }
        return true;
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
     * @param constAndPhInputs The constant and placeholder inputs - used for all frames/iterations
     * @return The parameterized op
     */
    public abstract O getAndParameterizeOp(String opName, FrameIter frameIter, Set<VarId> inputs, Set<String> constAndPhInputs, Map<String,T> placeholderValues);

    /**
     * Execute the op - calculate INDArrays, or shape info, etc
     *
     * @param op              Operation to exit. This should be parameterized (i.e., all inputs set)
     * @param outputFrameIter The frame and iteration of the outputs
     * @param inputs          The specific input arrays for the op
     * @return The outputs of the op
     */
    public abstract T[] getOutputs(O op, FrameIter outputFrameIter, Set<VarId> inputs, Set<String> constAndPhInputs);

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
            if (!execInputs.containsKey(forVariable))
                execInputs.put(forVariable, new HashSet<VarId>());
            execInputs.get(forVariable).add(inputVar);
        }
    }


    protected static VarId lookup(String name, Collection<VarId> varIds){
        for(VarId vid : varIds){
            if(vid.getVariable().equals(name)){
                return vid;
            }
        }
        throw new RuntimeException("Could not find VarId to input " + name);
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

        @Override
        public String toString() {
            return "VarId(\"" + variable + "\",\"" + frame + "\"," + iteration + ")";
        }

        public FrameIter toFrameIter() {
            return new FrameIter(frame, iteration);
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
    }

}
