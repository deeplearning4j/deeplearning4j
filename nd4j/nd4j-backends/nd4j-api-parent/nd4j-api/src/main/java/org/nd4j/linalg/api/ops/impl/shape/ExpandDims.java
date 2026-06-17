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

import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.exception.ND4JIllegalArgumentException;
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

public class ExpandDims extends DynamicCustomOp {
    private int jaxis;


    public ExpandDims() {
    }

    public ExpandDims(SameDiff sameDiff, SDVariable args, int axis) {
        this(sameDiff, new SDVariable[]{args}, axis);
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

    public ExpandDims(INDArray x, int axis){
        super(new INDArray[]{x}, null);
        this.jaxis = axis;
        addIArgument(axis);
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        throw new UnsupportedOperationException("Use the new Tensorflow Importer instead. This method is now removed.");

    }

    @Override
    public void setPropertiesForFunction(Map<String, Object> properties) {
        if(properties.containsKey("axis")) {
            Long value = (Long) properties.get("axis");
            if(value != null) {
                this.jaxis = value.intValue();
            }
        }
    }

    @Override
    public void configureFromArguments() {
        if(!iArguments.isEmpty()) {
            this.jaxis = iArguments.get(0).intValue();
        }
    }

    @Override
    public Map<String, Object> propertiesForFunction() {
        Map<String, Object> ret = new LinkedHashMap<>();
        ret.put("axis", (long) jaxis);
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
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        //Axis may be defined either as integer or as an array
        Preconditions.checkState(dataTypes != null && (dataTypes.size() == 1 || dataTypes.size() == 2), "Expected list with 1 or 2 datatype for %s, got %s", getClass(), dataTypes);
        //Output type is same as input type
        return Collections.singletonList(dataTypes.get(0));
    }

    @Override
    public boolean outputShapeDependsOnInputData() {
        return true;
    }

    /**
     * ExpandDims output shape: inserts a new dimension of size 1 at the specified axis.
     * For input [D0, D1, ..., Dn] with axis k, output is [D0, ..., D(k-1), 1, Dk, ..., Dn]
     */
    @Override
    public List<DataBuffer> calculateOutputShapeFromInputs(OpContext oc) {
        if (oc == null || oc.numInputArguments() < 1) {
            return null;
        }

        // If axis comes from second input (dynamic), fall back to C++
        if (oc.numInputArguments() > 1) {
            return null;
        }

        // Get axis from iArgs or field
        List<Long> iArgs = oc.getIArguments();
        int axis;
        if (iArgs != null && !iArgs.isEmpty()) {
            axis = iArgs.get(0).intValue();
        } else {
            axis = this.jaxis;
        }

        INDArray input = oc.getInputArray(0);
        if (input == null) {
            return null;
        }

        long[] inputShape = input.shape();
        int inputRank = inputShape.length;
        int outputRank = inputRank + 1;

        // Normalize negative axis
        if (axis < 0) {
            axis += outputRank;
        }

        if (axis < 0 || axis > inputRank) {
            return null; // Invalid axis - fall back to C++
        }

        // Build output shape: insert 1 at axis position
        long[] outputShape = new long[outputRank];
        int inIdx = 0;
        for (int i = 0; i < outputRank; i++) {
            if (i == axis) {
                outputShape[i] = 1;
            } else {
                outputShape[i] = inputShape[inIdx++];
            }
        }

        DataType dtype = input.dataType();
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
