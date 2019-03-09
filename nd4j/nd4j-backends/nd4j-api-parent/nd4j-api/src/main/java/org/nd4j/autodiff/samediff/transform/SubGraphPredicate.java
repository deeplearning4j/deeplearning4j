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

public class SubGraphPredicate extends OpPredicate {

    protected final OpPredicate root;
    protected Integer inputCount = null;
    protected Integer outputCount = null;
    protected Map<Integer,OpPredicate> opInputMatchPredicates = new HashMap<>();     //Must match - but these are NOT included in the resultant subgraph
    protected Map<Integer,OpPredicate> opInputSubgraphPredicates = new HashMap<>();  //Must match - and thare ARE incrluded in the resultant subgraph

    protected SubGraphPredicate(OpPredicate root){
        this.root = root;
    }

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
                DifferentialFunction df = sameDiff.getVariableOutputFunction(in.getVarName());
                if (df == null || !e.getValue().matches(sameDiff, df)) {
                    return false;
                }
            }
        }

        return true;
    }

    public SubGraph getSubGraph(SameDiff sd, DifferentialFunction rootFn){
        Preconditions.checkState(matches(sd, rootFn), "Root function does not match predicate");

        List<DifferentialFunction> childNodes = new ArrayList<>();
        //Need to work out child nodes
        if(!opInputSubgraphPredicates.isEmpty()){
            for(Map.Entry<Integer,OpPredicate> entry : opInputSubgraphPredicates.entrySet()){
                OpPredicate p2 = entry.getValue();
                SDVariable arg = rootFn.arg(entry.getKey());
                DifferentialFunction df = sd.getVariableOutputFunction(arg.getVarName());
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



    public static SubGraphPredicate withRoot(@NonNull OpPredicate root){
        return new SubGraphPredicate(root);
    }

    public SubGraphPredicate withInputCount(int inputCount){
        this.inputCount = inputCount;
        return this;
    }

    public SubGraphPredicate withOutputCount(int outputCount){
        this.outputCount = outputCount;
        return this;
    }

    public SubGraphPredicate withInputMatching(int inputNum, @NonNull OpPredicate opPredicate){
        opInputMatchPredicates.put(inputNum, opPredicate);
        return this;
    }

    public SubGraphPredicate withInputSubgraph(int inputNum, @NonNull OpPredicate opPredicate){
        opInputSubgraphPredicates.put(inputNum, opPredicate);
        return this;
    }
}
