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
import org.nd4j.base.Preconditions;

import java.util.*;

/**
 * SubGraphPredicate defines a subgraph - a set of connected functions that are part of a SameDiff instance.
 *
 *
 * @author Alex Black
 */
public class SubGraphPredicate extends OpPredicate {

    protected final OpPredicate root;
    protected Integer inputCount = null;
    protected Integer outputCount = null;
    protected Map<Integer,OpPredicate> opInputMatchPredicates = new HashMap<>();     //Must match - but these are NOT included in the resultant subgraph
    protected Map<Integer,OpPredicate> opInputSubgraphPredicates = new HashMap<>();  //Must match - and thare ARE incrluded in the resultant subgraph

    protected SubGraphPredicate(OpPredicate root){
        this.root = root;
    }

    /**
     * Determine if the subgraph, starting with the root function, matches the predicate
     *
     * @param sameDiff SameDiff instance the function belongs to
     * @param rootFn   Function that defines the root of the subgraph
     * @return True if the subgraph mathes the predicate
     */
    public boolean matches(SameDiff sameDiff, DifferentialFunction rootFn){

        if(!root.matches(sameDiff, rootFn)){
            return false;
        }

        SDVariable[] inputs = rootFn.args();
        int inCount = inputs == null ? 0 : inputs.length;
        if(inputCount != null){
            if(inCount != this.inputCount)
                return false;
        }

        SDVariable[] outputs = rootFn.outputVariables();
        int outCount = outputs == null ? 0 : outputs.length;
        if(outputCount != null){
            if(outCount != outputCount)
                return false;
        }

        for(Map<Integer,OpPredicate> m : Arrays.asList(opInputMatchPredicates, opInputSubgraphPredicates)) {
            for (Map.Entry<Integer, OpPredicate> e : m.entrySet()) {
                int inNum = e.getKey();
                if (inNum >= inCount) {
                    return false;
                }

                SDVariable in = inputs[inNum];
                DifferentialFunction df = sameDiff.getVariableOutputOp(in.getVarName());
                if (df == null || !e.getValue().matches(sameDiff, df)) {
                    return false;
                }
            }
        }

        return true;
    }

    /**
     * Get the SubGraph that matches the predicate
     *
     * @param sd SameDiff instance the function belongs to
     * @param rootFn   Function that defines the root of the subgraph
     * @return The subgraph that matches the predicate
     */
    public SubGraph getSubGraph(SameDiff sd, DifferentialFunction rootFn){
        Preconditions.checkState(matches(sd, rootFn), "Root function does not match predicate");

        List<DifferentialFunction> childNodes = new ArrayList<>();
        //Need to work out child nodes
        if(!opInputSubgraphPredicates.isEmpty()){
            for(Map.Entry<Integer,OpPredicate> entry : opInputSubgraphPredicates.entrySet()){
                OpPredicate p2 = entry.getValue();
                SDVariable arg = rootFn.arg(entry.getKey());
                DifferentialFunction df = sd.getVariableOutputOp(arg.getVarName());
                if(df != null){
                    childNodes.add(df);

                    if(p2 instanceof SubGraphPredicate){
                        SubGraph sg = ((SubGraphPredicate) p2).getSubGraph(sd, df);
                        childNodes.addAll(sg.childNodes);
                    }
                }
            }
        }

        SubGraph sg = SubGraph.builder()
                .sameDiff(sd)
                .rootNode(rootFn)
                .childNodes(childNodes)
                .build();

        return sg;
    }


    /**
     * Create a SubGraphPredicate with the specified root predicate
     * @param root Predicate for matching the root
     */
    public static SubGraphPredicate withRoot(@NonNull OpPredicate root){
        return new SubGraphPredicate(root);
    }

    /**
     * Modify the current subgraph to match only if the function has the specified number of inputs
     * @param inputCount Match only if the function has the specified number of inputs
     */
    public SubGraphPredicate withInputCount(int inputCount){
        this.inputCount = inputCount;
        return this;
    }

    /**
     * Modify the current subgraph to match only if the function has the specified number of outputs
     * @param outputCount Match only if the function has the specified number of outputs
     */
    public SubGraphPredicate withOutputCount(int outputCount){
        this.outputCount = outputCount;
        return this;
    }

    /**
     * Require the subgraph to match the specified predicate for the specified input.<br>
     * Note that this does NOT add the specified input to part of the subgraph<br>
     * i.e., the subgraph matches if the input matches the predicate, but when returning the SubGraph itself, the
     * function for this input is not added to the SubGraph
     * @param inputNum    Input number
     * @param opPredicate Predicate that the input must match
     * @return This predicate with the additional requirement added
     */
    public SubGraphPredicate withInputMatching(int inputNum, @NonNull OpPredicate opPredicate){
        opInputMatchPredicates.put(inputNum, opPredicate);
        return this;
    }

    /**
     * Require the subgraph to match the specified predicate for the specified input.<br>
     * Note that this DOES add the specified input to part of the subgraph<br>
     * i.e., the subgraph matches if the input matches the predicate, and when returning the SubGraph itself, the
     * function for this input IS added to the SubGraph
     * @param inputNum    Input number
     * @param opPredicate Predicate that the input must match
     * @return This predicate with the additional requirement added
     */
    public SubGraphPredicate withInputSubgraph(int inputNum, @NonNull OpPredicate opPredicate){
        opInputSubgraphPredicates.put(inputNum, opPredicate);
        return this;
    }
}
