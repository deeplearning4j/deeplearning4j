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

import com.google.common.primitives.Ints;
import lombok.val;
import onnx.OnnxProto3;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.util.ArrayUtil;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Transpose function
 *
 * @author Adam Gibson
 */
public class Transpose extends DynamicCustomOp {
    protected int[] permuteDims;

    public Transpose(SameDiff sameDiff, DifferentialFunction i_v) {
        super(null,sameDiff,new DifferentialFunction[]{i_v});

    }

    public Transpose() {}




   @Override
    public String opName() {
        return "transpose";
    }

    @Override
    public String onnxName() {
        return "Transpose";
    }

    @Override
    public String tensorflowName() {
        return "Transpose";
    }

    @Override
    public void initWithArrays(Map<String, INDArray> arrayMap) {
        if(permuteDims == null) {
            val permuteArrayOp = sameDiff.getArrForVertexId(args()[1].resultVertexId());
            if(permuteArrayOp != null) {
                this.permuteDims = permuteArrayOp.data().asInt();
            }
        }
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        super.initFromTensorFlow(nodeDef, initWith, attributesForNode, graph);
        NodeDef permuteDimsNode = null;
        for(int i = 0; i < graph.getNodeCount(); i++) {
            if(graph.getNode(i).getName().equals(nodeDef.getInput(1))) {
                permuteDimsNode = graph.getNode(i);
            }
        }

        val permuteArrayOp = TFGraphMapper.getInstance().getNDArrayFromTensor("value",permuteDimsNode,graph);
        if(permuteArrayOp != null) {
            this.permuteDims = permuteArrayOp.data().asInt();
        }
    }

    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {
        if(!attributesForNode.containsKey("perm")) {

        }
        else
            this.permuteDims = Ints.toArray(attributesForNode.get("perm").getIntsList());
    }

    @Override
    public List<int[]> calculateOutputShape() {
        if(permuteDims == null && arg() != null && arg().getResultShape() != null) {
            this.permuteDims = ArrayUtil.reverseCopy(ArrayUtil.range(0,arg().getResultShape().length));
            val permutedShape = ArrayUtil.permute(arg().getResultShape(),permuteDims);
            return Arrays.asList(permutedShape);
        }
        else if(permuteDims != null) {
            val permutedShape = ArrayUtil.permute(arg().getResultShape(),permuteDims);
            return Arrays.asList(permutedShape);
        }

        return Collections.emptyList();
    }

    @Override
    public int[] getResultShape() {
        val shapeList = calculateOutputShape();
        if(!shapeList.isEmpty())
            return shapeList.get(0);
        return null;
    }



    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> i_v) {
        return Collections.<DifferentialFunction>singletonList(this);
    }

}
