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

package org.nd4j.autodiff.samediff.transform;

import lombok.NonNull;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.internal.Variable;
import org.nd4j.autodiff.samediff.ops.SDOps;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.function.Consumer;

import java.util.*;

public class GraphTransformUtil {

    private GraphTransformUtil(){ }

    public static List<SubGraph> getSubgraphsMatching(SameDiff sd, SubGraphPredicate p){

        List<SubGraph> out = new ArrayList<>();
        for(DifferentialFunction df : sd.functions()){
            if(p.matches(sd, df)){
                SubGraph sg = p.getSubGraph(sd, df);
                out.add(sg);
            }
        }

        return out;
    }

    public static SameDiff replaceSubgraphsMatching(@NonNull SameDiff sd, @NonNull SubGraphPredicate p, @NonNull SubGraphProcessor processor){
        //Make a copy so that if the transform fails part way through, we don't leave user with broken graph
        sd = sd.dup();

        List<SubGraph> subgraphs = getSubgraphsMatching(sd, p);

        for(SubGraph sg : subgraphs){
            List<SDVariable> newOutputs = processor.processSubgraph(sd, sg);
            List<SDVariable> oldOutputs = sg.outputs();
            Preconditions.checkState(oldOutputs.size() == newOutputs.size(), "Error applying subgraph processor: " +
                    "different number of outputs for subgraph (%s) vs. returned by preprocessor (%s)", oldOutputs.size(), newOutputs.size());

            /*
            Now that we've processed subgraph to add new components, let's remove the old DifferentialFunction instances that
            comprise the subgraph. We've got a few things to fix up:
            1. Variable objects - any references to the old DifferentialFunction
            2. SameDiffOp objects - any references
            4. Clear out sessions etc, if required
            5. Validate graph structure
            */


            //Step 1: replace the old outputs with new outputs
            //So for initial graph (x -> y -> z) and post application of processor we now have (x -> (y, A); y->z),
            // we want to end up with (x -> A -> z)
            List<DifferentialFunction> allSubGraphFns = sg.allFunctionsInSubgraph();
            for(int i=0; i<oldOutputs.size(); i++ ){
                String oldOutVarName = oldOutputs.get(i).getVarName();
                String newOutVarName = newOutputs.get(i).getVarName();
                Preconditions.checkState(!oldOutVarName.equals(newOutVarName), "Reusing old variables not yet implemented");

                //Update inputs for ops: if X->opA, and now Y->opA, then X.inputsForOps contains "opA"; Y.inputsForOps should be updated
                List<String> oldInputsForOps = sd.getVariables().get(oldOutVarName).getInputsForOp();
                if(oldInputsForOps != null){
                    List<String> newInputsForOps = new ArrayList<>();
                    for(String s : oldInputsForOps){
                        DifferentialFunction df = sd.getFunctionById(s);
                        if(!allSubGraphFns.contains(df)){
                            newInputsForOps.add(s);
                        }
                    }
                    sd.getVariables().get(newOutVarName).setInputsForOp(newInputsForOps);
                }


                //Basically: anywhere that oldName exists, newName should be substituted
                for(Variable v : sd.getVariables().values()){
                    // if control dep v -> oldOutput exists, replace it
                    if(v.getControlDepsForVar() != null){
                        List<String> cds = v.getControlDepsForVar();
                        int idx;
                        while( (idx = cds.indexOf(oldOutVarName)) > 0){
                            cds.set(idx, newOutVarName);
                        }
                    }

                    if(v.getControlDeps() != null){
                        List<String> cds = v.getControlDeps();
                        //Control dependency oldOutput -> v exists, replace it
                        int idx;
                        while( (idx= cds.indexOf(oldOutVarName)) > 0){
                            cds.set(idx, newOutVarName);
                        }
                    }
                }

                for(SameDiffOp op : sd.getOps().values()){
                    List<String> inputsToOp = op.getInputsToOp();
                    if(inputsToOp != null) {
                        int idx;
                        while ((idx = inputsToOp.indexOf(oldOutVarName)) >= 0) {
                            //Previous Op.inputs = {oldVarName, ...} - now {newVarName, ...}
                            inputsToOp.set(idx, newOutVarName);
                        }
                    }

                    //Don't need to modify outputsOfOp - old outputs are only on functions to be removed anyway
                    List<String> controlDeps = op.getControlDeps();
                    if(controlDeps != null){
                        int idx;
                        while ((idx = controlDeps.indexOf(oldOutVarName)) >= 0) {
                            //Previous Op.inputs = {oldVarName, ...} - now {newVarName, ...}
                            controlDeps.set(idx, newOutVarName);
                        }
                    }
                }
            }

            //Step 2: Update input variables: if X -> (subgraph) exists, then X.inputsForOp needs to be updated
            List<SDVariable> inputs = sg.inputs();
            for(SDVariable v : inputs){
                Variable var = sd.getVariables().get(v.getVarName());
                if(var.getInputsForOp() != null){
                    List<String> newInputsForOp = new ArrayList<>(var.getInputsForOp());
                    for(String opName : var.getInputsForOp()){
                        //Two possibilities here:
                        // (1) variable is (was) input to op that has been removed - just remove from list
                        // (2) variable is now connected directly as an output: (A->B->C) becomes (A->C)
                        // For the latter case, this
                        DifferentialFunction df = sd.getFunctionById(opName);
                        if(allSubGraphFns.contains(df)){
                            newInputsForOp.remove(opName);
                        }
                    }
                    var.setInputsForOp(newInputsForOp);
                }
            }


            //Step 3: Remove the old variables and old functions
            Map<String,SameDiffOp> ops = sd.getOps();
            Map<String,Variable> vars = sd.getVariables();

            for(DifferentialFunction df : sg.allFunctionsInSubgraph()){
                ops.remove(df.getOwnName());
                SDVariable[] outputs = df.outputVariables();
                if(outputs != null){
                    for(SDVariable v : outputs){
                        vars.remove(v.getVarName());
                    }
                }
            }
        }

        return sd;
    }

}
