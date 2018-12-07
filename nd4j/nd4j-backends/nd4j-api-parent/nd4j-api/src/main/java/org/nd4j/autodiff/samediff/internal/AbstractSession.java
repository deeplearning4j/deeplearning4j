package org.nd4j.autodiff.samediff.internal;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.controlflow.compat.Merge;

import java.util.*;

/**
 *
 * @param <T> Node output type - for example, INDArray, shape, etc depending on what we're calculating
 * @param <O> Op type
 */
@Slf4j
public abstract class AbstractSession<T,O> {

    //All execution happens in a frame... this is the name of the main/outer frame
    protected static final String OUTER_FRAME = "main";

    /*
    VarId: identifies a variable in a specific frame and frame iteration
    Used for 2 places: (a) to identify variables that are available for execution
    (b) to store results
     */
    @Data
    @AllArgsConstructor
    protected static class VarId {
        private String variable;
        private String frame;
        private int iteration;
    }

    protected final SameDiff sameDiff;
    protected final Map<VarId, T> nodeOutputs = new HashMap<>();

    public AbstractSession(@NonNull SameDiff sameDiff) {
        this.sameDiff = sameDiff;
    }

    /**
     * Execute the op - calculate INDArrays, or shape info, etc
     * @param op
     * @return
     */
    public abstract T[] getOutputs(O op);

    /**
     * Get the parameterized op to execute - for example, the op/DifferentialFunction with all inputs set
     * @return
     */
    public abstract O getAndParameterizeOp(String opName);

    //TODO we might not need this method eventually...
    public abstract void preprocessPlaceholderValues(Map<String,T> placeholderValues);

    public T get(String variable, String frame, int iteration){
        //TODO eventually we'll cache and reuse VarId objects here to avoid garbage generation on lookup etc
        VarId varId = newVarId(variable, frame, iteration);
        return nodeOutputs.get(vid);
    }

    public VarId newVarId(String variable, String frame, int iteration){
        //TODO eventually we'll cache and reuse VarId objects here to avoid garbage generation on lookup
        return new VarId(variable, frame, iteration);
    }


    /**
     * @param variables       Name of the variables we want the arrays/activations for
     * @param outputWorkspace May be null. If null: returned arrays will be detached. If non-null: arrays will be in the specified workspace
     * @return The specified variable values, optionally in the specified workspace
     */
    //TODO CHANGE SIGNATURE TO USE OPERANDS CLASS OR SIMILAR
    public Map<String, T> output(@NonNull List<String> variables, Map<String, T> placeholderValues, MemoryWorkspace outputWorkspace) {
        Preconditions.checkState(!variables.isEmpty(), "Variables to perform forward pass for must not be empty");
        Preconditions.checkState(sameDiff.getPlaceHolderVarNames() == null || sameDiff.getPlaceHolderVarNames().isEmpty()
                        || (placeholderValues != null && placeholderValues.size() == sameDiff.getPlaceHolderVarNames().size() &&
                        placeholderValues.keySet().containsAll(sameDiff.getPlaceHolderVarNames())),
                "Invalid placeholders: SameDiff instance has placeholders %s, got placeholders %s", sameDiff.getPlaceHolderVarNames(),
                (placeholderValues == null ? null : placeholderValues.keySet()));

        //Basic plan: work backwards from the variables we want, based on the graph structure
        //Eventually, we'll cache this 'execution plan' information to avoid recalculating it - LRU cache etc

        /*
        Algorithm for determining ops to execute:
        First, determine list of variables we *might* need to get the requested output (also accounting for control dependencies).
         This gives us a sub-graph - specifically, a subgraph where nodes are variables and edges are ops
        - Basically, input to requested variable, or input of input, etc.
        - Not all inputs are required - consider switch op, only need one branch

        Second: have a mutable representation of the subgraph - i.e., "what remains to be executed"

        To determine what op to execute next, we simply look for leaves in the subgraph.
        After executing an op, we update the subgraph to remove the newly calculated variables.
         */

        //Step 0: validation - that variables exist, placeholders have arrays, etc
        for(String s : variables){
            Preconditions.checkState(sameDiff.variableMap().containsKey(s), "Requested output variable %s does not exist in SameDiff instance", s);
        }


        //Step 1: determine subgraph structure we actually need to execute
        Queue<String> processingQueue = new LinkedList<>(variables);
        Set<String> subgraph = new HashSet<>();     //Contains variables we *might* need to execute in process of getting outputs we want
        Queue<VarId> availableForExec = new LinkedList<>();

        //Note subgraph initially should include placeholders and constants
        while(!processingQueue.isEmpty()){
            String varName = processingQueue.remove();
            String opName = (sameDiff.getVariableOutputFunction(varName) == null ? null : sameDiff.getVariableOutputFunction(varName).getOwnName());

            if(!subgraph.contains(varName)){
                String[] opInputs = opName == null ? null : sameDiff.getInputsForFunction(sameDiff.getFunctionById(opName));
                String[] controlDeps = null;
                int numInputs = opInputs == null ? 0 : opInputs.length;
                if( controlDeps != null){
                    numInputs += controlDeps.length;
                }
                if(numInputs == 0){
                    availableForExec.add(newVarId(varName, OUTER_FRAME, 0));
                }
                subgraph.add(varName);
            }

            if(opName != null) {
                //To execute op - and hence get this variable: need inputs to that op
                String[] inputs = sameDiff.getInputsForFunction(sameDiff.getFunctionById(opName));
                for (String s2 : inputs) {
                    if (!subgraph.contains(s2)) {
                        processingQueue.add(s2);
                    }
                }

                //TODO To execute op - and hence get this variable - we also need control deps
                String[] opControlDeps = null;
                if(opControlDeps != null){
                    for(String s2 : opControlDeps){
                        if(!subgraph.contains(s2)){
                            processingQueue.add(s2);
                        }
                    }
                }
            }
        }


        //Step 3: execute in any order, until we have all required nodeOutputs
        /*
        Idea for execution is simple: we have subgraph of variables, which is whatever is left to execute.
        We look for leaf elements in subgraph - these are variables with either no inputs (placeholders/constants), or
        variables where all inputs required to calculate are available.

        After finding leaf, we execute associated op (for non placeholders/constants). Then, we remove the newly computed
        variable from the subgraph, exposing one or more new leaves to be executed in the next steps.

        We stop computation once all the required outputs are available. At this point, subgraph may NOT be empty - for example,
        switch ops may cause entire branches of the graph to be skipped.
         */

        //TODO we'll not do this in the future, but it's necessary for execution for now...
        preprocessPlaceholderValues(placeholderValues);

        Map<String,T> out = new HashMap<>();
        int step = 0;
        while(out.size() < variables.size()){

            Preconditions.checkState(availableForExec.size() > 0, "No variables are available for execution");

            //Get any variable and execute it's corresponding op
            VarId varToExec = availableForExec.remove();
            if(nodeOutputs.containsKey(varToExec))
                continue;   //Already processed this one?

            log.debug("Beginning execution step {}: variable {}", (step++), varToExec);

            if(sameDiff.isPlaceHolder(varToExec.getVariable()) ){
                nodeOutputs.put(varToExec, placeholderValues.get(varToExec.getVariable()));
                updateDescendentsForExec(varToExec, availableForExec);
                if(variables.contains(varToExec.getVariable())){  //Check if required output
                    out.put(varToExec.getVariable(), placeholderValues.get(varToExec.getVariable()));
                }
            } else if( sameDiff.getImportedConstants().contains(varToExec.getVariable()) ){
                //TODO let's remove the "importad constants" field, just have constants
                //TODO let's add an 'isConstant(String)'?

                nodeOutputs.put(varToExec, placeholderValues.get(varToExec.getVariable()));
                updateDescendentsForExec(varToExec, availableForExec);
                if(variables.contains(varToExec.getVariable())){  //Check if required output
                    out.put(varToExec.getVariable(), placeholderValues.get(varToExec.getVariable()));
                }
            } else if(sameDiff.getVariableOutputFunction(varToExec.getVariable()) != null){
                //Need to execute op to get this variable... which might have already happened in a previous step for multi-op variables
                if(!nodeOutputs.containsKey(varToExec)){
                    String opName = sameDiff.getFunctionOutputFor().get(varToExec.getVariable()).get(0).getOwnName();

                    //Execute op
                    //TODO

                    O parameterizedOp = getAndParameterizeOp(opName);
                    T[] opOutputValues = getOutputs(parameterizedOp);


                    //Post execution: work out what is now available for exec
                    String[] opOutputVarNames = sameDiff.getFunctionById(opName).outputVariablesNames();

                    Preconditions.checkState(opOutputValues.length == opOutputVarNames.length);

                    for( int i=0; i<opOutputVarNames.length; i++ ){
                        if(opOutputValues[i] == null){
                            //Skip - for switch op. Maybe better way to implement this?
                            continue;
                        }

                        VarId vid = newVarId(opOutputVarNames[i], varToExec.getFrame(), varToExec.getIteration());      //In same frame as input
                        nodeOutputs.put(vid, opOutputValues[i]);
                        updateDescendentsForExec(vid, availableForExec);

                        if(variables.contains(opOutputVarNames[i])){  //Check if required output
                            out.put(opOutputVarNames[i], opOutputValues[i]);
                        }
                    }
                }
            }

            //TODO check for invalid graph structure
        }


        //TODO under what circumstances should we clear the nodeOutputs map?
        //TODO when should we close the workspace? (Might want to leave it open if we expect to re-use)

        return out;
    }


    /**
     * This method should be called for a variable once it's array is ready for use.
     * For example, post op execution, etc
     *
     * @param varName          Name of the variable
     * @param availableForExec Any other variables that are now available for execution
     */
    protected void updateDescendentsForExec(String varName, Queue<String> availableForExec){
        //Find any ops (or variables with control dependencies) that this is required for execution of and check if now available for exec
        List<DifferentialFunction> l = sameDiff.getFunctionsArgsFor().get(varName);
        String[] inputForOps = l == null ? null : new String[l.size()];
        if(l != null){
            for(int i=0; i<inputForOps.length; i++ ){
                inputForOps[i] = l.get(i).getOwnName();
            }
        }

        //Check if we can execute this op now...
        if(inputForOps != null){
            for(String opName : inputForOps) {
                if(sameDiff.getFunctionById(opName) instanceof Merge){
                    //Merge op: available for execution when *any* of its inputs are available. But only mark it for exec once...
                    String[] opOutputs = sameDiff.getOutgoingArgsReverse().get(opName);
                    Preconditions.checkState(opOutputs.length == 1, "Expected only 1 output variable for merge op, got %s", opOutputs);
                    if(!nodeOutputs.containsKey(opName)){
                        Collections.addAll(availableForExec, opOutputs);
                        log.info("Marked merge op ({}) as available for execution: input {} is now available", opName, varName);
                    }
                    continue;
                }

                //Can execute this op - and hence get it's output variables - if all inputs (and control deps) are available
                String[] inputsThisOp = sameDiff.getFunctionById(opName).argNames();
                boolean allInputsAvailable = true;
                if(inputsThisOp != null) {
                    for (String in : inputsThisOp) {
                        if (!nodeOutputs.containsKey(in)) {
                            allInputsAvailable = false;
                            break;
                        }
                    }
                }

                String[] opControlDeps = null;
                if(opControlDeps != null && allInputsAvailable){
                    for(String cd : opControlDeps){
                        if(!nodeOutputs.containsKey(cd)) {
                            allInputsAvailable = false;
                            break;
                        }
                    }
                }

                String[] opOutputs = sameDiff.getOutgoingArgsReverse().get(opName);
                if(allInputsAvailable && opOutputs != null){
                    //Op can be executed -> variables as output are available for exec
                    //TODO what about variable control depes?

                    Collections.addAll(availableForExec, opOutputs);
                    log.info("Marked variables as available for execution: {} - output of op {} ({}) with op inputs {}", Arrays.toString(opOutputs), opName,
                            sameDiff.getFunctionById(opName).getClass().getSimpleName(), (inputsThisOp == null ? "<none>" : Arrays.toString(inputsThisOp)));
                }
            }
        }
    }


}
