/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
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

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;

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

    public Concat(SameDiff sameDiff, SDVariable[] inputs, int concatDimension) {
        this(sameDiff, concatDimension, inputs);
    }

    public Concat(SameDiff sameDiff, int concatDimension, SDVariable... inputs) {
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
        ret.put("isDynamicAxis",isDynamicAxis);
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
        if(!dArguments.isEmpty()) {
            return Collections.singletonList(dArguments.get(0));
        }

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

    /**
     * Concat output shape: same as inputs except concat dimension is sum of all inputs.
     * For inputs [D0, D1, ..., Dk, ...] concatenated along axis k,
     * output is [D0, D1, ..., sum(Dk), ...]
     */
    @Override
    public List<DataBuffer> calculateOutputShapeFromInputs(OpContext oc) {
        if (oc == null || oc.numInputArguments() < 1) {
            return null;
        }

        // Get concat dimension from iArgs or field
        List<Long> iArgs = oc.getIArguments();
        int axis;

        // For dynamic axis (TF import), last input is the axis - fall back to C++
        if (isDynamicAxis) {
            return null;
        }

        if (iArgs != null && !iArgs.isEmpty()) {
            axis = iArgs.get(0).intValue();
        } else {
            axis = this.concatDimension;
        }

        // Get first input to determine rank and base shape
        INDArray first = oc.getInputArray(0);
        if (first == null) {
            return null;
        }

        long[] firstShape = first.shape();
        int rank = firstShape.length;

        // Normalize negative axis
        if (axis < 0) {
            axis += rank;
        }

        if (axis < 0 || axis >= rank) {
            return null; // Invalid axis - fall back to C++
        }

        // Sum up the concat dimension across all inputs
        long concatDimSize = firstShape[axis];
        int numInputs = oc.numInputArguments();

        for (int i = 1; i < numInputs; i++) {
            INDArray input = oc.getInputArray(i);
            if (input == null) {
                return null;
            }
            long[] inputShape = input.shape();
            if (inputShape.length != rank) {
                return null; // Rank mismatch - fall back to C++
            }
            concatDimSize += inputShape[axis];
        }

        // Build output shape
        long[] outputShape = firstShape.clone();
        outputShape[axis] = concatDimSize;

        DataType dtype = first.dataType();
        long[] strides = Nd4j.getStrides(outputShape, 'c');
        boolean isEmpty = false;
        for (long dim : outputShape) {
            if (dim == 0) { isEmpty = true; break; }
        }
        LongShapeDescriptor descriptor = LongShapeDescriptor.fromShape(outputShape, strides, 1, 'c', dtype, isEmpty);
        DataBuffer shapeInfo = Shape.createShapeInformation(descriptor);
        return Collections.singletonList(shapeInfo);
    }
}
