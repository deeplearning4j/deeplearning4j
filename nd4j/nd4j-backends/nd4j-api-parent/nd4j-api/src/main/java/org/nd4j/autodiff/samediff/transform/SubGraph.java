/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
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

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.Variable;

import java.util.*;

@AllArgsConstructor
@NoArgsConstructor
@Builder
@Data
public class SubGraph {

    protected SameDiff sameDiff;
    protected DifferentialFunction rootNode;
    protected List<DifferentialFunction> childNodes;


    public List<SDVariable> outputs(){
        //Outputs: the SDVariables of the root OR child nodes that are not consumed *ONLY* by another op within the subgraph
        List<SDVariable> allOutputs = new ArrayList<>();
        if(rootNode.outputVariables() != null)
            Collections.addAll(allOutputs, rootNode.outputVariables());
        if(childNodes != null && !childNodes.isEmpty()){

            Set<SDVariable> seenAsInput = new HashSet<>();
            if(rootNode.args() != null)
                Collections.addAll(seenAsInput, rootNode.args());

            for(DifferentialFunction df : childNodes){
                if(df.args() != null)
                    Collections.addAll(seenAsInput, df.args());
                if(df.outputVariables() != null)
                    Collections.addAll(allOutputs, df.outputVariables());
            }
        }

        //Now: filter all output variables that are consumed *only* by
        //Example subgraph: x -> y -> z... then Y is not an output
        //But suppose same subgraph, but connection y -> a exists; then Y must be an output, because it's used somewhere else
        List<SDVariable> filteredOutputs = new ArrayList<>(allOutputs.size());
        for(SDVariable v : allOutputs){
            Variable var = sameDiff.getVariables().get(v.name());
            List<String> inputsFor = var.getInputsForOp();
            boolean allInSubgraph = true;
            if(inputsFor != null){
                for(String opOwnName : inputsFor) {
                    if (!inSubgraph(sameDiff.getOpById(opOwnName))){
                        allInSubgraph = false;
                        break;
                    }
                }
            }
            if(!allInSubgraph){
                filteredOutputs.add(v);
            }
        }

        return filteredOutputs;
    }

    public List<SDVariable> inputs(){
        //Inputs: the SDVariables that are inputs to this subgraph are those used by any of the differential functions
        // (root or child nodes) that are NOT outputs of any of the child nodes

        Set<SDVariable> outputsOfSubgraphNodes = new HashSet<>();
        for(DifferentialFunction df : allFunctionsInSubgraph()){
            SDVariable[] outputVars = df.outputVariables();
            if(outputVars != null){
                Collections.addAll(outputsOfSubgraphNodes, outputVars);
            }
        }

        List<SDVariable> inputs = new ArrayList<>();
        for(DifferentialFunction df : allFunctionsInSubgraph()){
            SDVariable[] args = df.args();
            if(args != null){
                for(SDVariable arg : args){
                    if(!outputsOfSubgraphNodes.contains(arg)){
                        inputs.add(arg);
                    }
                }
            }
        }


        return inputs;
    }

    public boolean inSubgraph(DifferentialFunction df){
        if(rootNode == df)
            return true;
        if(childNodes != null){
            for(DifferentialFunction d : childNodes){
                if(d == df){
                    return true;
                }
            }
        }
        return false;
    }

    public List<DifferentialFunction> allFunctionsInSubgraph(){
        List<DifferentialFunction> out = new ArrayList<>();
        out.add(rootNode);
        if(childNodes != null){
            out.addAll(childNodes);
        }
        return out;
    }
}
