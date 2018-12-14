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

import lombok.NoArgsConstructor;
import lombok.val;
import onnx.OnnxProto3;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.imports.graphmapper.onnx.OnnxGraphMapper;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.*;

/**
 * Gather op
 */
@NoArgsConstructor
public class Gather extends DynamicCustomOp {

    protected int[] indices;
    protected int jaxis = 0;


    public Gather(SameDiff sameDiff, SDVariable input, int[] indices, int axis, boolean inPlace) {
        super(null, sameDiff, new SDVariable[] {input}, inPlace);

        addIArgument(axis);
        addIArgument(indices);
        this.jaxis = axis;
        this.indices = indices;
    }

    public Gather(SameDiff sameDiff, SDVariable input, SDVariable indices, int axis, boolean inPlace) {
        super(null, sameDiff, new SDVariable[] {input, indices}, inPlace);
        addIArgument(axis);
        this.jaxis = axis;
    }

    @Override
    public String onnxName() {
        return "Gather";
    }


    @Override
    public String[] tensorflowNames() {
        return new String[]{"Gather", "GatherV2"};
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        TFGraphMapper.getInstance().initFunctionFromProperties(nodeDef.getOp(), this, attributesForNode, nodeDef, graph);

    }

    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {
        OnnxGraphMapper.getInstance().initFunctionFromProperties(node.getOpType(), this, attributesForNode, node, graph);
    }


    @Override
    public void resolvePropertiesFromSameDiffBeforeExecution() {
        super.resolvePropertiesFromSameDiffBeforeExecution();
        if (indices != null && numInputArguments() < 2) {
            if (numInputArguments() == 0) {
                INDArray a = Nd4j.create(indices, new long[]{indices.length}, new long[]{1}, 'c', DataType.INT);
                if (indices.length > 1)
                    a = a.reshape(indices.length);
                else
                    a = a.reshape(new int[]{});

                addInputArgument(args()[0].getArr(), a);
            } else if (numInputArguments() == 1) {
                addInputArgument(Nd4j.create(indices, new long[]{indices.length}, new long[]{1}, 'c', DataType.INT));
            }

        }

        if (numIArguments() < 1) {
            addIArgument(jaxis);
        }

        if (numOutputArguments() < getDescriptor().getNumOutputs()) {
            val outputs = outputVariables();
            //Check that ALL variables have an array before setting
            for(SDVariable v : outputs){
                if(v.getArr() == null){
                    return;
                }
            }

            for (int i = 0; i < outputs.length; i++) {
                val output = outputs[i].getArr();
                addOutputArgument(output);
            }
        }
    }

    @Override
    public Map<String, Map<String, PropertyMapping>> mappingsForFunction() {
        Map<String, Map<String, PropertyMapping>> ret = new HashMap<>();
        Map<String, PropertyMapping> map = new HashMap<>();
        val broadcast = PropertyMapping.builder()
                .onnxAttrName("indices")
                .tfInputPosition(1)
                .propertyNames(new String[]{"indices"}).build();

        map.put("indices", broadcast);

        ret.put(tensorflowNames()[0], map);
        ret.put(onnxName(), map);

        Map<String, PropertyMapping> map2 = new HashMap<>();
        val broadcast2 = PropertyMapping.builder()
                .tfInputPosition(1)
                .propertyNames(new String[]{"indices"}).build();
        map2.put("indices", broadcast2);

        val axis2 = PropertyMapping.builder()
                .tfInputPosition(2)
                .propertyNames(new String[]{"axis"}).build();
        map2.put("axis", axis2);

        ret.put("GatherV2", map2);


        return ret;
    }

    @Override
    public String opName() {
        return "gather";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v){
        //2 args: input and indices. Plus integer dimension arg
        //Gather backprop is just scatter add

        SDVariable indicesGrad = sameDiff.zerosLike(arg(1));
        SDVariable inputGrad = sameDiff.zerosLike(arg(0));

        int ndim = arg(0).getShape().length;
        int a = jaxis;
        if(a < 0){
            a += ndim;
        }

        if(a == 0){
            inputGrad = sameDiff.scatterAdd(inputGrad, arg(1), i_v.get(0));
        } else {
            int[] permDims = new int[ndim];
            permDims[0] = a;
            for(int i=0; i<a; i++){
                permDims[i+1] = i;
            }
            for(int i=a+1; i<ndim; i++){
                permDims[i] = i;
            }
            inputGrad = sameDiff.permute(inputGrad, permDims);
            SDVariable i_v_transposed = sameDiff.permute(i_v.get(0), permDims);
            inputGrad = sameDiff.scatterAdd(inputGrad, arg(1), i_v_transposed);
            int[] reverseDims = new int[ndim];
            for(int i=0; i<ndim; i++){
                reverseDims[permDims[i]] = i;
            }
            inputGrad = sameDiff.permute(inputGrad, reverseDims);
        }

        return Arrays.asList(inputGrad, indicesGrad);
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        //Output type is same as (first) input type
        return Collections.singletonList(dataTypes.get(0));
    }
}
