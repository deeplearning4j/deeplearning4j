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
import org.nd4j.imports.graphmapper.onnx.OnnxGraphMapper;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Reshape function
 *
 * @author Adam Gibson
 */
@Slf4j
public class Reshape extends DynamicCustomOp {

    private int[] shape;

    public Reshape(SameDiff sameDiff, SDVariable i_v,int[] shape) {
        super(null,sameDiff, new SDVariable[]{i_v});
        this.shape = shape;
        addIArgument(shape);
    }


    public Reshape() {}





    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        if(!nodeDef.containsAttr("TShape") && nodeDef.getInputCount() == 1) {
            this.shape = new int[] {1,1};
            return;
        }
        else if(nodeDef.getInputCount() > 1) {
            val shapeNode = nodeDef.getInput(1);
            NodeDef shapeNodeInGraph = null;
            for(int i = 0; i < graph.getNodeCount(); i++) {
                if (graph.getNode(i).getName().equals(shapeNode)) {
                    shapeNodeInGraph = graph.getNode(i);

                }
            }

            val arr = TFGraphMapper.getInstance().getNDArrayFromTensor("value",shapeNodeInGraph,graph);
            if(arr != null) {
                this.shape = arr.data().asInt();
                addIArgument(this.shape);
            }
        }
        else {
            val shape = nodeDef.getAttrOrThrow("Tshape");
            if(!shape.hasShape()) {
                val shapeRet = new int[2];
                shapeRet[0] = 1;
                shapeRet[1] = shape.getValueCase().getNumber();
                this.shape = shapeRet;
            }
            else {
                val shapeVals = shape.getShape().getDimList();
                if(shapeVals.size() > 1) {
                    this.shape = new int[shapeVals.size()];
                    for(int i = 0; i < shapeVals.size(); i++) {
                        this.shape[i] = (int) shapeVals.get(i).getSize();
                    }
                }
                else {
                    this.shape = new int[2];
                    this.shape[0] = 1;
                    this.shape[1] = (int) shapeVals.get(0).getSize();
                }

            }

            if(this.shape != null)
                addIArgument(this.shape);


        }



    }

    @Override
    public void initWithArrays(Map<String, INDArray> arrayMap, Object... extraArgs) {
        super.initWithArrays(arrayMap);
        if(numIArguments() == 0) {
            if(args().length > 1) {
                val arr = sameDiff.getArrForVertexId(args()[1].getVertexId());
                if(arr == null) {
                    throw new ND4JIllegalStateException("Unable to infer shape for reshape. No array found for getting shape data from!");
                }

                this.shape = arr.data().asInt();
                addIArgument(this.shape);

            }
            else if(this.shape != null)
                addIArgument(this.shape);
            else
                throw new ND4JIllegalStateException("Unable to map shape for reshape. No shape found!");
        }
    }

    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {
        val shape = new OnnxGraphMapper().getShape(node);
        this.shape = shape;

    }

    @Override
    public List<int[]> calculateOutputShape() {
        return Arrays.asList(shape);
    }

    @Override
    public void addInputArgument(INDArray... arg) {
        if(numInputArguments() > 1) {
            throw new ND4JIllegalStateException("Unable to add more input. Reshape should only have 1.");
        }

        super.addInputArgument(arg);
    }

    @Override
    public String opName() {
        return "reshape";
    }

    @Override
    public String onnxName() {
        return "Reshape";
    }

    @Override
    public String tensorflowName() {
        return "Reshape";
    }





    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        SDVariable ret = outputVariables()[0];
        return Collections.singletonList(ret);
    }

}
