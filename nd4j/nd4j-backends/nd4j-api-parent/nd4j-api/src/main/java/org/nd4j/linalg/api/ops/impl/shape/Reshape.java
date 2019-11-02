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

package org.nd4j.linalg.api.ops.impl.shape;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import onnx.Onnx;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Reshape function
 *
 * @author Adam Gibson
 */
@Slf4j
public class Reshape extends DynamicCustomOp {

    private long[] shape;
    private String arrName;

    public Reshape(SameDiff sameDiff, SDVariable i_v, long[] shape) {
        super(null, sameDiff, new SDVariable[]{i_v});
        this.shape = shape;
        addIArgument(shape);
    }

    public Reshape(SameDiff sameDiff, SDVariable i_v, SDVariable shape) {
        super(null, sameDiff, new SDVariable[]{i_v, shape});
    }

    public Reshape(INDArray in, INDArray shape, INDArray out){
        super(null, new INDArray[]{in, shape}, new INDArray[]{out}, null, (List<Integer>)null);
    }

    public Reshape() {
    }


    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        if (!nodeDef.containsAttr("TShape") && nodeDef.getInputCount() == 1) {
            this.shape = new long[]{};
            return;
        } else if(nodeDef.getInputCount() == 1){
            val shape = nodeDef.getAttrOrThrow("Tshape");
            if (!shape.hasShape()) {
                val shapeRet = new long[2];
                shapeRet[0] = 1;
                shapeRet[1] = shape.getValueCase().getNumber();
                this.shape = shapeRet;
            } else {
                val shapeVals = shape.getShape().getDimList();
                if (shapeVals.size() > 1) {
                    this.shape = new long[shapeVals.size()];
                    for (int i = 0; i < shapeVals.size(); i++) {
                        this.shape[i] = (int) shapeVals.get(i).getSize();
                    }
                } else {
                    this.shape = new long[2];
                    this.shape[0] = 1;
                    this.shape[1] = (int) shapeVals.get(0).getSize();
                }

            }

            //all TF is c

            if (this.shape != null) {
                addIArgument(this.shape);
            }
        }
    }

    @Override
    public void initFromOnnx(Onnx.NodeProto node, SameDiff initWith, Map<String, Onnx.AttributeProto> attributesForNode, Onnx.GraphProto graph) {

    }


    @Override
    public Map<String, Map<String, PropertyMapping>> mappingsForFunction() {
        Map<String, Map<String, PropertyMapping>> ret = new HashMap<>();
        Map<String, PropertyMapping> map = new HashMap<>();

        val shapeMapping = PropertyMapping.builder()
                .onnxAttrName("shape")
                .tfInputPosition(-1)
                .propertyNames(new String[]{"shape"})
                .build();

        map.put("shape", shapeMapping);

        ret.put(tensorflowName(), map);
        ret.put(onnxName(), map);

        return ret;
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
        SDVariable origShape = f().shape(arg());
        SDVariable ret = f().reshape(i_v.get(0), origShape);
        return Collections.singletonList(ret);
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        //Output type is always same as input type
        return Collections.singletonList(dataTypes.get(0));
    }

}
