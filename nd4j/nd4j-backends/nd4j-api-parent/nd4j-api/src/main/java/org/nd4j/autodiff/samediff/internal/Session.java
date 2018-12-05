package org.nd4j.autodiff.samediff.internal;

import com.google.common.collect.Iterables;
import lombok.NonNull;
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
        Map<String, AtomicInteger> subgraph = new HashMap<>();       //Key: variable name. Value: number of missing inputs - TODO more useful representation?
        //Note subgraph initially should include placeholders and constants?
        while(!processingQueue.isEmpty()){
            String s = processingQueue.remove();
            Variable v = sameDiff.getVariables().get(s);

            if(!subgraph.containsKey(v.getName())){
                String opName = v.getOutputOfOp();
                SameDiffOp op = sameDiff.getOps().get(opName);
                int numInputs = op.getInputsToOp() == null ? 0 : op.getInputsToOp().length;
                subgraph.put(v.getName(), new AtomicInteger(numInputs));
            }

            //To execute op, need all that inputs -> v depends on op inputs
            SameDiffOp op = sameDiff.getOps().get(v.getOutputOfOp());
            String[] inputs = op.getInputsToOp();
            for(String s2 : inputs){
                if(!subgraph.containsKey(s2) ){     // && !placeholder || placeholder with control deps
                    processingQueue.add(s2);
                }
            }

            //Variables can have control dependencies (TODO TF uses Constant nodes for this; we might consider doing the same instead of allowing variable control inputs?)
            String[] controlDeps = v.getControlDeps();
            if(controlDeps != null ){
                for(String cd : controlDeps) {
                    if (!subgraph.containsKey(cd)) {    // && !placeholder || placeholder with control deps
                        processingQueue.add(cd);
                    }
                }
            }
        }


        //Step 3: execute in any order, until we have all required arrays
        while(subgraph.size() > 0){
            //Get any variable and execute it's corresponding op
            String toExecute = Iterables.getFirst(leaves, null);       //TODO anything more efficient without object creation?


            //Post execution: update the graph structure
            Variable v = sameDiff.getVariables().get(toExecute);
            String[] inputsTo = v.getInputsForOp();
        }


        //Return
        Map<String,INDArray> out = new HashMap<>();
        for(String s : variables){
            if(sameDiff.isPlaceHolder(s)){

            } else if( false /*sameDiff.isConstant(s)*/){

            } else {
                out.put(s, arrays.get(s));
            }
        }

        //TODO under what circumstances should we clear the arrays map?
        //TODO when should we close the workspace? (Might want to leave it open

        throw new UnsupportedOperationException("Not yet implemented");
    }


}
