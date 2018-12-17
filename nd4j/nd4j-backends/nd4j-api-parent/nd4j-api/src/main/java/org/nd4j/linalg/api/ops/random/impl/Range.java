/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
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

package org.nd4j.linalg.api.ops.random.impl;

import lombok.val;
import onnx.OnnxProto3;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.factory.Nd4j;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Range Op implementation, generates from..to distribution within Z
 *
 * @author raver119@gmail.com
 */
public class Range extends DynamicCustomOp {
    private Double from;
    private Double to;
    private Double delta;
    //used for initWithArrays when there are place holder
    //values that need to be resolved
//    private String fromVertexId,toVertexId,deltaVertexId;
    public Range() {
        // no-op
    }

    public Range(SameDiff sd, double from, double to, double step){
        super(null, sd, new SDVariable[0]);
        addTArgument(from, to, step);
        this.from = from;
        this.to = to;
        this.delta = step;

    }


    @Override
    public int opNum() {
        return 4;
    }

    @Override
    public String opName() {
        return "range";
    }

    @Override
    public String onnxName() {
        return "Range";
    }

    @Override
    public String tensorflowName() {
        return "Range";
    }



    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        super.initFromTensorFlow(nodeDef, initWith, attributesForNode, graph);
/*
        NodeDef startNode = null,endNode = null,deltaNode = null;
        for(val  node : graph.getNodeList()) {
            if(node.getName().equals(nodeDef.getInput(0))) {
                startNode = node;
            }
            if(node.getName().equals(nodeDef.getInput(1))) {
                endNode = node;
            }
            if(node.getName().equals(nodeDef.getInput(2))) {
                deltaNode = node;
            }

            if(startNode != null && endNode != null && deltaNode != null)
                break;
        }

        val start = TFGraphMapper.getInstance().getNDArrayFromTensor("value",startNode,graph);
        val end = TFGraphMapper.getInstance().getNDArrayFromTensor("value",endNode,graph);
        val delta = TFGraphMapper.getInstance().getNDArrayFromTensor("value",deltaNode,graph);
        if(start != null && end != null && delta != null) {

            if(endNode.getName() != null && endNode.getName().equalsIgnoreCase("Rank")){
                //Edge case: TF seems to produce rank 0 arrays for range op sometimes, in spite of docs - properties are [range/start, Rank, range/delta]
                this.from = start.getDouble(0);
                this.to = from+1;
                this.delta = 1.0;
            } else {
                this.from = start.getDouble(0);
                this.to = end.getDouble(0);
                this.delta = delta.getDouble(0);
            }

            val outputVars = outputVariables();
            addTArgument(this.from,this.to,this.delta);
            val outputVertexId = outputVars[0].getVarName();
            if(sameDiff.getArrForVarName(outputVertexId) == null) {
                if(outputVars[0].getShape() == null) {
                    val calcShape = calculateOutputShape();
                    sameDiff.putShapeForVarName(outputVars[0].getVarName(),calcShape.get(0));
                }

                long[] shape = outputVars[0].getShape();
                val arr = Nd4j.create(shape);
                initWith.putArrayForVarName(outputVertexId, arr);
                addOutputArgument(arr);

            }
        }

        val fromVar = initWith.getVariable(TFGraphMapper.getInstance().getNodeName(startNode.getName()));
        val toVar = initWith.getVariable(TFGraphMapper.getInstance().getNodeName(endNode.getName()));
        val deltaVar =  initWith.getVariable(TFGraphMapper.getInstance().getNodeName(deltaNode.getName()));
*/
//        this.fromVertexId = fromVar.getVarName();
//        this.toVertexId = toVar.getVarName();
//        this.deltaVertexId = deltaVar.getVarName();

    }



    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {
        super.initFromOnnx(node, initWith, attributesForNode, graph);
    }




    @Override
    public List<LongShapeDescriptor> calculateOutputShape() {
        val iArgs = iArgs();
        val tArgs = tArgs();
        val inputArgs = inputArguments();
        int cnt = 0;

        if(args().length > 1) {
            if (inputArgs.length > 0)
                return Nd4j.getExecutioner().calculateOutputShape(this);
        } else if (iArgs.length > 0) {
            return Nd4j.getExecutioner().calculateOutputShape(this);
        } else if (tArgs.length > 0) {
            return Nd4j.getExecutioner().calculateOutputShape(this);
        }

       return Collections.emptyList();
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return Collections.emptyList();
    }

    @Override
    public Op.Type opType() {
        return Op.Type.CUSTOM;
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        Preconditions.checkState(inputDataTypes == null || inputDataTypes.isEmpty(), "Expected no input datatypes (no args), got %s", inputDataTypes);
        //Input data type specifies the shape; output data type should be any float
        //TODO MAKE CONFIGUREABLE - https://github.com/deeplearning4j/deeplearning4j/issues/6854
        return Collections.singletonList(DataType.FLOAT);
    }
}
