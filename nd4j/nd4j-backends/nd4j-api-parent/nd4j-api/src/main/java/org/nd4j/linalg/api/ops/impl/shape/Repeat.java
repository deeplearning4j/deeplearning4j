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
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.*;

/**
 * Repeat function
 *
 * @author Adam Gibson
 */
@NoArgsConstructor
public class Repeat extends DynamicCustomOp {
    private int axis;

    public Repeat(int axis) {
        this.axis = axis;
    }

    public Repeat(SameDiff sameDiff, SDVariable[] args, int axis) {
        super(null, sameDiff, args);
        this.axis = axis;
    }

    public Repeat(INDArray[] inputs, INDArray[] outputs, List<Double> tArguments, List<Integer> iArguments, int axis) {
        super(null, inputs, outputs, tArguments, iArguments);
        this.axis = axis;
    }

    public Repeat(INDArray[] inputs, INDArray[] outputs, int axis) {
        super(null, inputs, outputs);
        this.axis = axis;
    }

    public Repeat(SameDiff sameDiff, SDVariable[] args, boolean inPlace, int axis) {
        super(null, sameDiff, args, inPlace);
        this.axis = axis;
    }


    @Override
    public Map<String, Object> propertiesForFunction() {
        Map<String, Object> ret = new LinkedHashMap<>();
        ret.put("axis", axis);
        return ret;
    }


    @Override
    public String opName() {
        return "repeat";
    }


    @Override
    public Map<String, Map<String, PropertyMapping>> mappingsForFunction() {
        Map<String, Map<String, PropertyMapping>> ret = new HashMap<>();
        Map<String, PropertyMapping> map = new HashMap<>();

        val axisMapping = PropertyMapping.builder()
                .onnxAttrName("axis")
                .tfInputPosition(-1)
                .propertyNames(new String[]{"axis"})
                .build();

        map.put("axis", axisMapping);

        ret.put(tensorflowName(), map);
        ret.put(onnxName(), map);

        return ret;
    }


    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        TFGraphMapper.getInstance().initFunctionFromProperties(nodeDef.getOp(), this, attributesForNode, nodeDef, graph);
        addIArgument(axis);
    }

    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {
        super.initFromOnnx(node, initWith, attributesForNode, graph);
    }

    @Override
    public void resolvePropertiesFromSameDiffBeforeExecution() {
        if (numOutputArguments() < getDescriptor().getNumOutputs()) {
            for (val output : outputVariables()) {
                addOutputArgument(output.getArr());
            }
        }
    }

    @Override
    public String onnxName() {
        return "Repeat";
    }

    @Override
    public String tensorflowName() {
        return "Repeat";
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        SDVariable ret = outputVariables()[0];
        return Arrays.asList(ret);
    }

}
