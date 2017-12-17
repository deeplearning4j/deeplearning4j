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
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
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

    public Transpose(SameDiff sameDiff, SDVariable i_v) {
        super(null,sameDiff,new SDVariable[]{i_v});

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
    public void initWithArrays(Map<String, INDArray> arrayMap, Object... extraArgs) {
        if(permuteDims == null) {
            val args = args();
            if(args().length > 1) {
                val permuteArrayOp = sameDiff.getArrForVarName(args[1].getVarName());
                if(permuteArrayOp != null) {
                    this.permuteDims = permuteArrayOp.data().asInt();
                    if(ArrayUtil.prod(permuteDims) == 0 || permuteDims.length < args[0].getShape().length) {
                        this.permuteDims = ArrayUtil.reverseCopy(ArrayUtil.range(0,args[0].getShape().length));
                    }
                    else {
                        for(int i = 0; i < permuteDims.length; i++) {
                            addIArgument(permuteDims[i]);
                        }
                    }

                }
                else
                    this.permuteDims = ArrayUtil.reverseCopy(ArrayUtil.range(0,args[0].getShape().length));
            }
            else {
                this.permuteDims = ArrayUtil.reverseCopy(ArrayUtil.range(0,args[0].getShape().length));
            }

        }

        //only add if empty
        if(numOutputArguments() == 0)
            for(int i = 0; i < permuteDims.length; i++) {
                addIArgument(permuteDims[i]);
            }


        if(permuteDims != null && permuteDims.length < arg().getShape().length)
            throw new ND4JIllegalStateException("Illegal permute found. Not all dimensions specified");
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        super.initFromTensorFlow(nodeDef, initWith, attributesForNode, graph);
        //permute dimensions re not specified as second input
        if(nodeDef.getInputCount() < 2)
            return;
        NodeDef permuteDimsNode = null;
        for(int i = 0; i < graph.getNodeCount(); i++) {
            if(graph.getNode(i).getName().equals(nodeDef.getInput(1))) {
                permuteDimsNode = graph.getNode(i);
            }

        }

        val permuteArrayOp = TFGraphMapper.getInstance().getNDArrayFromTensor("value",permuteDimsNode,graph);
        if(permuteArrayOp != null) {
            this.permuteDims = permuteArrayOp.data().asInt();
            for(int i = 0; i < permuteDims.length; i++) {
                addIArgument(permuteDims[i]);
            }
        }

        //handle once properly mapped
        if(arg().getShape() == null) {
           return;
        }


        INDArray arr = sameDiff.getArrForVarName(arg().getVarName());
        if(arr == null) {
            val  arrVar = sameDiff.getVariable(arg().getVarName());
            arr = arrVar.getWeightInitScheme().create(arrVar.getShape());
            sameDiff.putArrayForVarName(arg().getVarName(),arr);
        }

        addInputArgument(arr);


        if(permuteDims != null && permuteDims.length < arg().getShape().length)
            throw new ND4JIllegalStateException("Illegal permute found. Not all dimensions specified");


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
        if(permuteDims == null && arg() != null && arg().getShape() != null) {
            this.permuteDims = ArrayUtil.reverseCopy(ArrayUtil.range(0,arg().getShape().length));
            val permutedShape = ArrayUtil.permute(arg().getShape(),permuteDims);
            return Arrays.asList(permutedShape);
        }
        else if(permuteDims != null) {
            val permutedShape = ArrayUtil.permute(arg().getShape(),permuteDims);
            return Arrays.asList(permutedShape);
        }

        throw new ND4JIllegalStateException("Unable to compute shape!");
    }




    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        return Collections.singletonList(outputVariables()[0]);
    }

}
