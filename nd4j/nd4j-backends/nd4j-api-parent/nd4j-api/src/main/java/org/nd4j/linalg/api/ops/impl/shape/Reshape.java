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
import onnx.OnnxProto3;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.imports.graphmapper.onnx.OnnxGraphMapper;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.util.ArrayUtil;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.*;

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
        } else if (nodeDef.getInputCount() > 1) {
            val shapeNode = nodeDef.getInput(1);
            NodeDef shapeNodeInGraph = null;
            for (int i = 0; i < graph.getNodeCount(); i++) {
                if (graph.getNode(i).getName().equals(shapeNode)) {
                    shapeNodeInGraph = graph.getNode(i);

                }
            }

            val arr = TFGraphMapper.getInstance().getNDArrayFromTensor("value", shapeNodeInGraph, graph);
            if (arr != null && arr.isEmpty()) {
                // special case: empty array
                this.shape = new long[0];

            } else if (arr != null) {
                this.shape = arr.data().asLong();
                //all TF is c
                if (!ArrayUtil.containsAnyNegative(this.shape))
                    addIArgument(this.shape);
                else {
                    arrName = nodeDef.getName();
                }

            }
        } else {
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
    public void resolvePropertiesFromSameDiffBeforeExecution() {
        super.resolvePropertiesFromSameDiffBeforeExecution();
        if (arrName != null) {
            val args = args();
            val firstInputShape = args[0].getShape();
            val shapeInput = args[1].getArr().data().asLong();
            for (int i = 0; i < shapeInput.length; i++) {
                if (shapeInput[i] < 0) {
                    shapeInput[i] = firstInputShape[i];
                }
            }

            this.shape = shapeInput;
            addIArgument(shapeInput);
        }


    }

    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {
        val shape = new OnnxGraphMapper().getShape(node);
        this.shape = shape;
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
        val origShape = arg().getShape();
        if (origShape == null) {
            //TODO need a more robust way to do this
            throw new ND4JIllegalStateException("Cannot reshape: original array input shape is null");
        }
        SDVariable ret = f().reshape(i_v.get(0), origShape);
        return Collections.singletonList(ret);
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        //Output type is always same as input type
        return Collections.singletonList(dataTypes.get(0));
    }

}
