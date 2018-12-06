package org.nd4j.autodiff.samediff.internal;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.memory.MemoryWorkspace;

import java.util.*;

/**
 *
 * @param <T> Node output type - for example, INDArray, shape, etc depending on what we're calculating
 * @param <O> Op type
 */
@Slf4j
public abstract class AbstractSession<T,O> {

    protected final SameDiff sameDiff;
    protected final Map<String, T> nodeOutputs = new HashMap<>();      //INDArrays for ARRAY type SDVariables only

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
     * @param op
     * @return
     */
    public abstract O getAndParameterizeOp(String opName);

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


        //Step 1: determine subgraph structure we actually need to execute
        Queue<String> processingQueue = new LinkedList<>(variables);
        Set<String> subgraph = new HashSet<>();     //Contains variables we *might* need to execute in process of getting outputs we want
        Queue<String> availableForExec = new LinkedList<>();

        //Note subgraph initially should include placeholders and constants
        while(!processingQueue.isEmpty()){
            String s = processingQueue.remove();
            Variable v = sameDiff.getVariables().get(s);

            if(!subgraph.contains(v.getName())){
                String opName = v.getOutputOfOp();
                SameDiffOp op = sameDiff.getOps().get(opName);
                int numInputs = op.getInputsToOp() == null ? 0 : op.getInputsToOp().length;
                if(op.getControlDeps() != null){
                    numInputs += op.getControlDeps().length;
                }
                if(numInputs == 0){
                    availableForExec.add(v.getName());
                }
            }

            if(v.getOutputOfOp() != null) {
                String opName = v.getOutputOfOp();
                SameDiffOp op = sameDiff.getOps().get(opName);

                //To execute op - and hence get this variable: need inputs to that op
                String[] inputs = op.getInputsToOp();
                for (String s2 : inputs) {
                    if (!subgraph.contains(s2)) {
                        processingQueue.add(s2);
                    }
                }

                //To execute op - and hence get this variable - we also need control deps
                String[] opControlDeps = op.getControlDeps();
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

        Map<String,T> out = new HashMap<>();
        int step = 0;
        while(out.size() < variables.size()){
            //Get any variable and execute it's corresponding op
            String varToExec = availableForExec.remove();
            Variable v = sameDiff.getVariables().get(varToExec);

            log.debug("Beginning execution step {}: variable {}", (step++), varToExec);

            if(sameDiff.isPlaceHolder(varToExec) ){
                nodeOutputs.put(varToExec, placeholderValues.get(varToExec));
                updateDescendentsForExec(varToExec, availableForExec);
                if(variables.contains(varToExec)){  //Check if required output
                    out.put(varToExec, placeholderValues.get(varToExec));
                }
            }
            /*
            else if( isConstant ){

                continue;
            }
             */
            else if(v.getOutputOfOp() != null){
                //Need to execute op to get this variable... which might have already happened in a previous step for multi-op variables

                if(!nodeOutputs.containsKey(varToExec)){
                    SameDiffOp op = sameDiff.getOps().get(v.getOutputOfOp());

                    //Execute op
                    //TODO

                    O parameterizedOp = getAndParameterizeOp(op.getName());
                    T[] opOutputValues = getOutputs(parameterizedOp);


                    //Post execution: work out what is now available for exec
                    String[] opOutputNames = op.getOutputsOfOp();

                    Preconditions.checkState(opOutputValues.length == opOutputNames.length);

                    for( int i=0; i<opOutputNames.length; i++ ){
                        nodeOutputs.put(opOutputNames[i], opOutputValues[i]);
                        updateDescendentsForExec(opOutputNames[i], availableForExec);

                        if(variables.contains(opOutputNames[i])){  //Check if required output
                            out.put(opOutputNames[i], opOutputValues[i]);
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
        //Find any ops (or variables with control dependencies) that this is required for execution of and check if now available
        Variable v = sameDiff.getVariables().get(varName);

        //Check if we can execute this op now...
        if(v.getInputsForOp() != null){
            String[] ops = v.getInputsForOp();

            for(String s : ops) {
                SameDiffOp o = sameDiff.getOps().get(s);
                //Can execute this op - and hence get it's output variables - if all inputs (and control deps) are available
                String[] inputs = o.getInputsToOp();
                boolean allInputsAvailable = true;
                for(String in : inputs){
                    if(!nodeOutputs.containsKey(in)) {
                        allInputsAvailable = false;
                        break;
                    }
                }

                String[] opControlDeps = o.getControlDeps();
                if(opControlDeps != null && allInputsAvailable){
                    for(String cd : opControlDeps){
                        if(!nodeOutputs.containsKey(cd)) {
                            allInputsAvailable = false;
                            break;
                        }
                    }
                }

                if(allInputsAvailable && o.getOutputsOfOp() != null){
                    //Op can be executed -> variables as output are available for exec
                    //TODO what about variable control depes?

                    Collections.addAll(availableForExec, o.getOutputsOfOp());
                    if(log.isTraceEnabled()){
                        log.trace("Marked variables as available for execution: {}", Arrays.toString(o.getOutputsOfOp()));
                    }
                }
            }
        }
    }


}
