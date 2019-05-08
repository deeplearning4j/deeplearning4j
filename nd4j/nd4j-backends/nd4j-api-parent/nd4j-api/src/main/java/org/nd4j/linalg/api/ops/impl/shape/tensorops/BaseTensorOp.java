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

package org.nd4j.linalg.api.ops.impl.shape.tensorops;

import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Collections;
import java.util.List;
import java.util.Map;

public abstract  class BaseTensorOp extends DynamicCustomOp {

    public BaseTensorOp(String name, SameDiff sameDiff, SDVariable[] args){
        super(name, sameDiff, args);
    }
    public BaseTensorOp(SameDiff sameDiff, SDVariable[] args){
        super(null, sameDiff, args);
    }

    public BaseTensorOp(){}

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        val inputOne = nodeDef.getInput(1);
        val varFor = initWith.getVariable(inputOne);
        val nodeWithIndex = TFGraphMapper.getInstance().getNodeWithNameFromGraph(graph,inputOne);
        val var = TFGraphMapper.getInstance().getArrayFrom(nodeWithIndex,graph);
        if(var != null) {
            val idx = var.getInt(0);
            addIArgument(idx);
        }
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        throw new UnsupportedOperationException("Differentiation not yet implemented for " + getClass().getName());
    }

    @Override
    public Op.Type opType() {
        return Op.Type.CUSTOM;
    }


    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op name found for " + opName());
    }

    @Override
    public String toString() {
        return opName();
    }

    @Override
    public List<LongShapeDescriptor> calculateOutputShape() {
        //Not used/not required
        return Collections.emptyList();
    }

    @Override
    public int getNumOutputs(){
        //1 output in allay cases - sometimes just a dummy output, however
        return 1;
    }

}
