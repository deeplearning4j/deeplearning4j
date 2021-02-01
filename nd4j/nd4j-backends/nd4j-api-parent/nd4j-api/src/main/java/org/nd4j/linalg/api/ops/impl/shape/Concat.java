/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.api.ops.impl.shape;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.impl.shape.bp.ConcatBp;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.*;

@Slf4j
public class Concat extends DynamicCustomOp {
    private int concatDimension = -1;
    private boolean isDynamicAxis = false;

    public Concat(){

    }

    public Concat(int concatDimension, INDArray... arrays) {
        super(null, arrays, null);
        this.concatDimension = concatDimension;
        addIArgument(concatDimension);
    }

    public Concat(INDArray[] arrays, int concatDimension) {
        this(concatDimension, arrays);
    }

    public Concat(SameDiff sameDiff, SDVariable[] inputs, int concatDimension){
        this(sameDiff, concatDimension, inputs);
    }

    public Concat(SameDiff sameDiff, int concatDimension, SDVariable... inputs){
        super(null, sameDiff, inputs);
        addIArgument(concatDimension);
        this.concatDimension = concatDimension;
    }

    @Override
    public String opName() {
        return "concat";
    }

    @Override
    public void assertValidForExecution() {
        val descriptor = getDescriptor();
        if(descriptor == null)
            throw new NoOpNameFoundException("No descriptor found for op name " + opName());


        if(descriptor.getNumInputs() > 0 && numInputArguments() < 2)
            throw new ND4JIllegalStateException("Op failure for " + opName() + " Number of inputs is invalid for execution. Specified " + numInputArguments() + " but should be " + descriptor.getNumInputs());

        if(descriptor.getNumOutputs() > 0 && numOutputArguments() != descriptor.getNumOutputs())
            throw new ND4JIllegalStateException("Op failure for " + opName() + " Number of outputs is invalid for execution. Specified " + numOutputArguments() + " but should be " + descriptor.getNumOutputs());

        //< 0 means dynamic size
        if(descriptor.getNumIArgs() >= 0 && numIArguments() != descriptor.getNumIArgs())
            throw new ND4JIllegalStateException("Op failure for " + opName() + " Number of integer arguments is invalid for execution. Specified " + numIArguments() + " but should be " + descriptor.getNumIArgs());

        if(descriptor.getNumTArgs() >= 0 && numTArguments() != descriptor.getNumTArgs())
            throw new ND4JIllegalStateException("Op failure for " + opName() + " Number of inputs is invalid for execution. Specified " + numTArguments() + " but should be " + descriptor.getNumTArgs());

    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        //TF uses dynamic axis - last argument is a scalar integer array for axis
        addBArgument(true);
        isDynamicAxis = true;
    }

    @Override
    public Map<String, Object> propertiesForFunction() {
        Map<String,Object> ret = new LinkedHashMap<>();
        ret.put("concatDimension",concatDimension);
        return ret;
    }

    @Override
    public String onnxName() {
        return "Concat";
    }

    @Override
    public String tensorflowName() {
        return "Concat";
    }

    @Override
    public String[] tensorflowNames() {
        return new String[]  {"Concat","ConcatV2"};
    }

    @Override
    public Op.Type opType() {
        return Op.Type.CUSTOM;
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        SDVariable[] args = args();
        SDVariable[] bpArgs;
        if(isDynamicAxis){
            bpArgs = Arrays.copyOf(args, args.length + 2);
            bpArgs[bpArgs.length - 1] = bpArgs[bpArgs.length - 3];      //Last input is axis -> move to end of bp args too
            bpArgs[bpArgs.length - 2] = i_v.get(0);
            return Arrays.asList(new ConcatBp(sameDiff, concatDimension, bpArgs).outputVariables());
        } else {
            bpArgs = Arrays.copyOf(args, args.length + 1);
            bpArgs[bpArgs.length - 1] = i_v.get(0);
            return Arrays.asList(new ConcatBp(sameDiff, concatDimension, bpArgs).outputVariables());
        }
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        DataType first = dataTypes.get(0);

        for( int i = 1; i < dataTypes.size() - (isDynamicAxis ? 1 : 0); i++) {
            DataType dt = dataTypes.get(i);
            Preconditions.checkState(first == dt, "All inputs must have same datatype - got %s and %s for inputs 0 and %s respectively", first, dt, i);
        }

        if(isDynamicAxis) {
            Preconditions.checkState(dataTypes.get(dataTypes.size() - 1).isIntType(),
                    "For dynamic axis case, last datatype must be an integer type, got input types %s");
        }

        //Output type is same as input types
        return Collections.singletonList(first);
    }
}
