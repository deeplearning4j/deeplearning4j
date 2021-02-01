/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.autodiff.samediff.transform;

import lombok.NonNull;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.internal.Variable;
import org.nd4j.common.base.Preconditions;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * GraphTransformUtil provides a number of utility methods to modify graphs - replacing nodes and subgraphs, etc.<br>
 * See the individual methods for futher details
 *
 * @author Alex Black
 */
public class GraphTransformUtil {

    private GraphTransformUtil() {
    }

    /**
     * Find all of the subgraphs that match the specified SubGraphPredicate and then replace them with a different subgraph.<br>
     * Note that the original SameDiff instance is not modified; a copy is made, which is then modified and returned.
     * <br>
     * Note: For each subgraph to be replaced by SubGraphProcessor, its replacement should have the same number of output
     * SDVariables.
     *
     * @param sd        SameDiff instance to copy and modify
     * @param p         SubGraphPredicate to define and select the subgraphs that should be modified or replaced
     * @param processor SubGraphProcessor is used to define how the subgraphs (selected by the SubGraphPredicate) should
     *                  be modified/replaced
     * @return A SameDiff instance that has been modified
     */
    public static SameDiff replaceSubgraphsMatching(@NonNull SameDiff sd, @NonNull SubGraphPredicate p, @NonNull SubGraphProcessor processor) {
        //Make a copy so that if the transform fails part way through, we don't leave user with broken graph
        sd = sd.dup();

        List<SubGraph> subgraphs = getSubgraphsMatching(sd, p);

        for (SubGraph sg : subgraphs) {
            List<SDVariable> newOutputs = processor.processSubgraph(sd, sg);
            List<SDVariable> oldOutputs = sg.outputs();
            Preconditions.checkState(oldOutputs.size() == newOutputs.size(), "Error applying subgraph processor: " +
                    "different number of outputs for subgraph (%s) vs. returned by preprocessor (%s)", oldOutputs.size(), newOutputs.size());

            //Step 1: replace the old outputs with new outputs
            //So for initial graph (x -> y -> z) and post application of processor we now have (x -> (y, A); y->z),
            // we want to end up with (x -> A -> z)
            List<DifferentialFunction> allSubGraphFns = sg.allFunctionsInSubgraph();
            for (int i = 0; i < oldOutputs.size(); i++) {
                String oldOutVarName = oldOutputs.get(i).name();
                String newOutVarName = newOutputs.get(i).name();
                Preconditions.checkState(!oldOutVarName.equals(newOutVarName), "Reusing old variables not yet implemented");

                //Update inputs for ops: if X->opA, and now Y->opA, then X.inputsForOps contains "opA"; Y.inputsForOps should be updated
                List<String> oldInputsForOps = sd.getVariables().get(oldOutVarName).getInputsForOp();
                if (oldInputsForOps != null) {
                    List<String> newInputsForOps = new ArrayList<>();
                    for (String s : oldInputsForOps) {
                        DifferentialFunction df = sd.getOpById(s);
                        if (!allSubGraphFns.contains(df)) {
                            newInputsForOps.add(s);
                        }
                    }
                    sd.getVariables().get(newOutVarName).setInputsForOp(newInputsForOps);
                }


                //Basically: anywhere that oldName exists, newName should be substituted
                for (Variable v : sd.getVariables().values()) {
                    // if control dep v -> oldOutput exists, replace it
                    if (v.getControlDepsForVar() != null) {
                        List<String> cds = v.getControlDepsForVar();
                        int idx;
                        while ((idx = cds.indexOf(oldOutVarName)) > 0) {
                            cds.set(idx, newOutVarName);
                        }
                    }

                    if (v.getControlDeps() != null) {
                        List<String> cds = v.getControlDeps();
                        //Control dependency oldOutput -> v exists, replace it
                        int idx;
                        while ((idx = cds.indexOf(oldOutVarName)) > 0) {
                            cds.set(idx, newOutVarName);
                        }
                    }
                }

                for (SameDiffOp op : sd.getOps().values()) {
                    List<String> inputsToOp = op.getInputsToOp();
                    if (inputsToOp != null) {
                        int idx;
                        while ((idx = inputsToOp.indexOf(oldOutVarName)) >= 0) {
                            //Previous Op.inputs = {oldVarName, ...} - now {newVarName, ...}
                            inputsToOp.set(idx, newOutVarName);
                        }
                    }

                    //Don't need to modify outputsOfOp - old outputs are only on functions to be removed anyway
                    List<String> controlDeps = op.getControlDeps();
                    if (controlDeps != null) {
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
            for (SDVariable v : inputs) {
                Variable var = sd.getVariables().get(v.name());
                if (var.getInputsForOp() != null) {
                    List<String> newInputsForOp = new ArrayList<>(var.getInputsForOp());
                    for (String opName : var.getInputsForOp()) {
                        //Two possibilities here:
                        // (1) variable is (was) input to op that has been removed - just remove from list
                        // (2) variable is now connected directly as an output: (A->B->C) becomes (A->C)
                        // For the latter case, this
                        DifferentialFunction df = sd.getOpById(opName);
                        if (allSubGraphFns.contains(df)) {
                            newInputsForOp.remove(opName);
                        }
                    }
                    var.setInputsForOp(newInputsForOp);
                }
            }


            //Step 3: Remove the old variables and old functions
            Map<String, SameDiffOp> ops = sd.getOps();
            Map<String, Variable> vars = sd.getVariables();

            for (DifferentialFunction df : sg.allFunctionsInSubgraph()) {
                ops.remove(df.getOwnName());
                SDVariable[] outputs = df.outputVariables();
                if (outputs != null) {
                    for (SDVariable v : outputs) {
                        vars.remove(v.name());
                    }
                }
            }
        }

        return sd;
    }

    /**
     * Get a list of all the subgraphs that match the specified predicate
     *
     * @param sd SameDiff instance to get the subgraphs for
     * @param p  Subgraph predicate. This defines the subgraphs that should be selected in the SameDiff instance
     * @return Subgraphs
     */
    public static List<SubGraph> getSubgraphsMatching(SameDiff sd, SubGraphPredicate p) {
        List<SubGraph> out = new ArrayList<>();
        for (DifferentialFunction df : sd.ops()) {
            if (p.matches(sd, df)) {
                SubGraph sg = p.getSubGraph(sd, df);
                out.add(sg);
            }
        }

        return out;
    }
}
