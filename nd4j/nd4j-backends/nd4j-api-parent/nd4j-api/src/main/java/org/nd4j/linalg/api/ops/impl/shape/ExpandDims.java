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

import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.exception.ND4JIllegalArgumentException;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.*;

/**
 * ExpandDims function
 *
 * @author Adam Gibson
 */
public class ExpandDims extends DynamicCustomOp {
    private int jaxis;


    public ExpandDims() {
    }

    public ExpandDims(SameDiff sameDiff, SDVariable[] args, int axis) {
        super(null, sameDiff, args);
        if (axis == Integer.MAX_VALUE) {
            throw new ND4JIllegalArgumentException("Cannot perform ExpandDims with axis == Integer.MAX_VALUE");
        }
        this.jaxis = axis;
        addIArgument(this.jaxis);
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
        val targetNode = TFGraphMapper.getInstance().getNodeWithNameFromGraph(graph, nodeDef.getInput(1));
        val dimArr = TFGraphMapper.getInstance().getNDArrayFromTensor("value", targetNode, graph);

        if (dimArr != null) {
            int axis = dimArr.data().asInt()[0];
            this.jaxis = axis;
            addIArgument(this.jaxis);
        } else {
            this.jaxis = Integer.MAX_VALUE;
            addIArgument(this.jaxis);
        }
    }

    @Override
    public Map<String, Object> propertiesForFunction() {
        Map<String, Object> ret = new LinkedHashMap<>();
        ret.put("axis", axis);
        return ret;
    }

    @Override
    public Map<String, Map<String, PropertyMapping>> mappingsForFunction() {
        Map<String, Map<String, PropertyMapping>> ret = new HashMap<>();
        val axisMapping = PropertyMapping.builder()
                .tfInputPosition(1)
                .propertyNames(new String[]{"axis"})
                .build();
        Map<String, PropertyMapping> map = new HashMap<>();
        map.put("axis", axisMapping);

        ret.put(tensorflowName(), map);
        return ret;
    }

    @Override
    public void assertValidForExecution() {
        val descriptor = getDescriptor();
        if (descriptor.getNumInputs() > 0 && numInputArguments() > 2 || numInputArguments() < 1)
            throw new ND4JIllegalStateException("Op failure for " + opName() + " Number of inputs is invalid for execution. Specified " + numInputArguments() + " but should be " + descriptor.getNumInputs());

        if (descriptor.getNumOutputs() > 0 && numOutputArguments() != descriptor.getNumOutputs())
            throw new ND4JIllegalStateException("Op failure for " + opName() + " Number of outputs is invalid for execution. Specified " + numOutputArguments() + " but should be " + descriptor.getNumInputs());

        //< 0 means dynamic size
        if (descriptor.getNumIArgs() >= 0 && numIArguments() != descriptor.getNumIArgs())
            throw new ND4JIllegalStateException("Op failure for " + opName() + " Number of integer arguments is invalid for execution. Specified " + numIArguments() + " but should be " + descriptor.getNumIArgs());

        if (descriptor.getNumTArgs() >= 0 && numTArguments() != descriptor.getNumTArgs())
            throw new ND4JIllegalStateException("Op failure for " + opName() + " Number of inputs is invalid for execution. Specified " + numTArguments() + " but should be " + descriptor.getNumTArgs());

    }

    @Override
    public String opName() {
        return "expand_dims";
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " + opName());

    }

    @Override
    public String tensorflowName() {
        return "ExpandDims";
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        //Simply need a reshape to remove the dimension...
        SDVariable ret = sameDiff.squeeze(i_v.get(0), jaxis);
        return Arrays.asList(ret);
    }

    @Override
    public List<org.nd4j.linalg.api.buffer.DataType> calculateOutputDataTypes(List<org.nd4j.linalg.api.buffer.DataType> dataTypes){
        Preconditions.checkState(dataTypes.size() == 1, "Expected list with exactly 1 datatype, got %s", dataTypes);
        //Output type is same as input type
        return dataTypes;
    }

}
