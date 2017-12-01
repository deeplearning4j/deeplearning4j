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
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.graphmapper.onnx.OnnxGraphMapper;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

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

    public Reshape(SameDiff sameDiff, DifferentialFunction i_v,int[] shape) {
        super(null,sameDiff, new DifferentialFunction[]{i_v});
        this.shape = shape;
    }


    public Reshape() {}





    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        if(!nodeDef.containsAttr("TShape")) {
            this.shape = new int[] {1,1};
            return;
        }

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

    }

    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {
        val shape = new OnnxGraphMapper().getShape(node);
        this.shape = shape;

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
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> i_v) {
        DifferentialFunction ret = this;

        return Collections.singletonList(ret);
    }

}
