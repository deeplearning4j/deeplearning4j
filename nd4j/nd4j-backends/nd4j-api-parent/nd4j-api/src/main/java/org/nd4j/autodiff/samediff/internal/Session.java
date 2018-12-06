package org.nd4j.autodiff.samediff.internal;

import com.google.common.collect.Iterables;
import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

public class Session {

    protected final SameDiff sameDiff;
    protected final Map<String, INDArray> arrays = new HashMap<>();      //INDArrays for ARRAY type SDVariables only

    public Session(@NonNull SameDiff sameDiff) {
        this.sameDiff = sameDiff;
    }

    /**
     * @param variables       Name of the variables we want the arrays/activations for
     * @param outputWorkspace May be null. If null: returned arrays will be detached. If non-null: arrays will be in the specified workspace
     * @return The specified variable values, optionally in the specified workspace
     */
    public Map<String, INDArray> output(@NonNull List<String> variables, Map<String, INDArray> placeholderValues, MemoryWorkspace outputWorkspace) {
        Preconditions.checkState(!variables.isEmpty(), "Variables to perform forward pass for must not be empty");
        Preconditions.checkState(sameDiff.getPlaceHolderVarNames() == null || sameDiff.getPlaceHolderVarNames().isEmpty()
                        || (placeholderValues != null && placeholderValues.size() == sameDiff.getPlaceHolderVarNames().size() &&
                        placeholderValues.keySet().containsAll(sameDiff.getPlaceHolderVarNames()),
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


        //Step 1: determine subgraph structure
        Queue<String> processingQueue = new LinkedList<>(variables);
        Set<String> subgraph = new HashSet<>();     //Contains variables we *might* need to execute in process of getting outputs we want
        Queue<String> availableForExec = new LinkedList<>();

        //Note subgraph initially should include placeholders and constants?
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


        //Step 3: execute in any order, until we have all required arrays
        /*
        Idea for execution is simple: we have subgraph of variables, which is whatever is left to execute.
        We look for leaf elements in subgraph - these are variables with either no inputs (placeholders/constants), or
        variables where all inputs required to calculate are available.

        After finding leaf, we execute associated op (for non placeholders/constants). Then, we remove the newly computed
        variable from the subgraph, exposing one or more new leaves to be executed in the next steps.

        We stop computation once all the required outputs are available. At this point, subgraph may NOT be empty - for example,
        switch ops may cause entire branches of the graph to be skipped.
         */

        Map<String,INDArray> out = new HashMap<>();
        while(out.size() < variables.size()){
            //Get any variable and execute it's corresponding op
            String varToExec = availableForExec.remove();
            Variable v = sameDiff.getVariables().get(varToExec);

            if(sameDiff.isPlaceHolder(varToExec) ){
                arrays.put(varToExec, placeholderValues.get(varToExec));
                updateDescendentsForExec(varToExec);
            }
            /*
            else if( isConstant ){

                continue;
            }
             */
            else if(v.getOutputOfOp() != null){
                //Need to execute op to get this variable... which might have already happened in a previous step for multi-op variables

                if(!arrays.containsKey(varToExec)){
                    SameDiffOp op = sameDiff.getOps().get(v.getOutputOfOp());

                    //Execute op
                    //TODO

                    //Post execution: work out what is now available for exec
                    String[] opOutputs = op.getOutputsOfOp();
                    for(String s : opOutputs){
                        updateDescendentsForExec(s);
                    }
                }


            }



            String[] opOutputVars = op.getOutputsOfOp();            //All of these variables are now available
            for(String var : opOutputVars){
                Variable v2 = sameDiff.getVariables().get(var);
                String[] inputsFor = v2.getInputsForOp();           //This variable is input to other ops

            }


        }


        //TODO under what circumstances should we clear the arrays map?
        //TODO when should we close the workspace? (Might want to leave it open if we expect to re-use)

        return out;
    }


    protected void updateDescendentsForExec(String varName){

        //Find any ops (or variables with control dependencies) that this is required for execution of and check if now available
        Variable v = sameDiff.getVariables().get(varName);
        if(v.getInputsForOp() != null){
            String[] ops = v.getInputsForOp();

            for(String s : ops) {
                SameDiffOp o = sameDiff.getOps().get(s);
                //Can execute this op - and hence get it's output variables - if all inputs are available

            }
        }
    }


}
