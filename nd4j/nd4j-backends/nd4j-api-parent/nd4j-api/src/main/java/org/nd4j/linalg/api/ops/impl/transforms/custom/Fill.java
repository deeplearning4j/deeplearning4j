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

package org.nd4j.linalg.api.ops.impl.transforms.custom;

import lombok.val;
import onnx.Onnx;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.imports.descriptors.properties.adapters.DataTypeAdapter;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Collections;
import java.util.List;
import java.util.Map;


/**
 * Fill an array of given "shape" with the provided "value", e.g.
 * shape [2, 2] and value 42 returns [[42, 42], [42, 42]].
 *
 * @author Max Pumperla
 */
public class Fill extends DynamicCustomOp {

    private double value;
    private DataType outputDataType;

    public Fill() {
    }


    public Fill(SameDiff sameDiff, SDVariable shape, DataType outputDataType, double value) {
        super(null,sameDiff, new SDVariable[] {shape}, false);
        this.value = value;
        this.outputDataType = outputDataType;
        this.outputDataType = outputDataType;
        addArgs();
    }

    public Fill(INDArray shape, DataType outputDataType, double value) {
        super(new INDArray[]{shape, Nd4j.scalar(outputDataType, value)}, null);
        this.value = value;
        this.outputDataType = outputDataType;
    }

    public Fill(INDArray shape, INDArray result, double value) {
        super(null, shape, result, Collections.singletonList(value), null);
        this.value = value;
    }

    public Fill(INDArray shape, INDArray value, INDArray result) {
        super(null, new INDArray[]{shape, value}, new INDArray[]{result});
    }


    protected void addArgs() {
        addTArgument(value);
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        org.tensorflow.framework.DataType dt = attributesForNode.get("T").getType();
        this.outputDataType = DataTypeAdapter.dtypeConv(dt);
    }

    @Override
    public void initFromOnnx(Onnx.NodeProto node, SameDiff initWith, Map<String, Onnx.AttributeProto> attributesForNode, Onnx.GraphProto graph) {
        super.initFromOnnx(node, initWith, attributesForNode, graph);
    }

    @Override
    public void assertValidForExecution() {
        val descriptor = getDescriptor();
        if(descriptor.getNumInputs() > 0 && numInputArguments() >  2 || numInputArguments() < 1)
            throw new ND4JIllegalStateException("Op failure for " + opName() + " Number of inputs is invalid for execution. Specified " + numInputArguments() + " but should be " + descriptor.getNumInputs());

        if(descriptor.getNumOutputs() > 0 && numOutputArguments() != descriptor.getNumOutputs())
            throw new ND4JIllegalStateException("Op failure for " + opName() + " Number of outputs is invalid for execution. Specified " + numOutputArguments() + " but should be " + descriptor.getNumInputs());

        //< 0 means dynamic size
        if(descriptor.getNumIArgs() >= 0 && numIArguments() != descriptor.getNumIArgs())
            throw new ND4JIllegalStateException("Op failure for " + opName() + " Number of integer arguments is invalid for execution. Specified " + numIArguments() + " but should be " + descriptor.getNumIArgs());

        if(descriptor.getNumTArgs() >= 0 && numTArguments() < 1)
            throw new ND4JIllegalStateException("Op failure for " + opName() + " Number of inputs is invalid for execution. Specified " + numTArguments() + " but should be " + descriptor.getNumTArgs());

    }

    @Override
    public String opName() {
        return "fill";
    }

    @Override
    public String onnxName() {
        return "ConstantFill";
    }

    @Override
    public String tensorflowName() {
        return "Fill";
    }

    @Override
    public Op.Type opType() {
        return Op.Type.CUSTOM;
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradients){
        return Collections.singletonList(sameDiff.zerosLike(arg()));
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        //1 or 2 possible: 2 for TF import (fill with specified value
        Preconditions.checkState(dataTypes != null && (dataTypes.size() == 1 || dataTypes.size() == 2),
                "Expected 1 or 2 input datatypes for %s, got %s", getClass(), dataTypes);
        Preconditions.checkNotNull(outputDataType, "Output datatype was null (not set)");
        return Collections.singletonList(outputDataType);
    }
}
