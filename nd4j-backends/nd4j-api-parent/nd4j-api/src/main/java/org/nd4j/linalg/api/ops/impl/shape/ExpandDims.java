/*-
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *
 */

package org.nd4j.linalg.api.ops.impl.shape;

import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * ExpandDims function
 *
 * @author Adam Gibson
 */
public class ExpandDims extends DynamicCustomOp {
    private int axis;


    public ExpandDims() {
    }

    public ExpandDims(SameDiff sameDiff, SDVariable[] args, int axis) {
        super(null, sameDiff, args);
        this.axis = axis;
    }

    public ExpandDims(SameDiff sameDiff, SDVariable[] args) {
        super(null, sameDiff, args);
    }

    public ExpandDims(INDArray[] inputs, INDArray[] outputs) {
        super(null, inputs, outputs);
    }

    public ExpandDims(SameDiff sameDiff, SDVariable[] args, boolean inPlace) {
        super(null, sameDiff, args, inPlace);
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
         val targetNode = TFGraphMapper.getInstance().getNodeWithNameFromGraph(graph,nodeDef.getInput(1));
        val dimArr = TFGraphMapper.getInstance().getNDArrayFromTensor("value",targetNode,graph);

        if(dimArr != null) {
             int axis = dimArr.data().asInt()[0];
             this.axis = axis;
             addIArgument(this.axis);
         }
         else {
            this.axis = Integer.MAX_VALUE;
            addIArgument(this.axis);
        }
    }

    @Override
    public String opName() {
        return "expand_dims";
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " +  opName());

    }

    @Override
    public String tensorflowName() {
        return "ExpandDims";
    }




    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        SDVariable ret = f().div(arg(),f().abs(arg()));
        return Collections.singletonList(ret);
    }

}
