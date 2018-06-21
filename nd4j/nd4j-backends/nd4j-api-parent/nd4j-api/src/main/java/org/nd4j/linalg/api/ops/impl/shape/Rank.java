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

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import onnx.OnnxProto3;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.*;

/**
 * Rank function
 *
 * @author Adam Gibson
 */
@Slf4j
public class Rank extends DynamicCustomOp {


    public Rank() {
    }

    public Rank(SameDiff sameDiff, SDVariable input, boolean inPlace) {
        super(null, sameDiff, new SDVariable[] {input}, inPlace);
    }


    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        val name = TFGraphMapper.getInstance().getNodeName(nodeDef.getName());
        val input = initWith.getVariable(name);
        val outputVertex = input.getVarName();
        if (!initWith.isPlaceHolder(input.getVarName()) && initWith.shapeAlreadyExistsForVarName(outputVertex)) {
            val inputShape = initWith.getShapeForVarName(input.getVarName());
            val resultLength = Nd4j.scalar(inputShape.length);
            val thisResultId = outputVertex;
            initWith.putArrayForVarName(thisResultId, resultLength);
            initWith.putShapeForVarName(thisResultId, new long[]{1, 1});
        }
    }

    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {

    }

    @Override
    public String opName() {
        return "rank";
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx found for op " + opName());
    }

    @Override
    public String tensorflowName() {
        return "Rank";
    }

    @Override
    public List<long[]> calculateOutputShape() {
        List<long[]> ret = new ArrayList<>();
        ret.add(new long[]{});
        return ret;
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        return Collections.singletonList(sameDiff.zerosLike(arg()));
    }

}
