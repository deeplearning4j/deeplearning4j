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

import onnx.Onnx;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.Op;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Collections;
import java.util.List;
import java.util.Map;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Returns the shape of the input array.
 *
 * @author Adam Gibson
 */
public class Shape extends DynamicCustomOp {

    protected DataType dataType;

    public Shape() {}

    public Shape(SameDiff sameDiff, SDVariable input) {
        this(sameDiff, input, false);
    }

    public Shape(SameDiff sameDiff, SDVariable input, boolean inPlace) {
        super(null, sameDiff, new SDVariable[] {input}, inPlace);
    }

    public Shape(INDArray in, INDArray out){
        super(null, in, out, null, null);
    }

    public Shape(INDArray in){
        this(in, null);
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx name found for shape " + opName());
    }


    @Override
    public String opName() {
        return "shape_of";
    }

    @Override
    public String tensorflowName() {
        return "Shape";
    }

    @Override
    public Op.Type opType() {
        return Op.Type.CUSTOM;
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        throw new UnsupportedOperationException("Use the new Tensorflow Importer instead. This method is now removed.");

    }

    @Override
    public void initFromOnnx(Onnx.NodeProto node, SameDiff initWith, Map<String, Onnx.AttributeProto> attributesForNode, Onnx.GraphProto graph) {
        throw new NoOpNameFoundException("No onnx name found for shape " + opName());
    }

    @Override
    public void configureFromArguments() {
        if(!dArguments.isEmpty()) {
            this.dataType = dArguments.get(0);
        }
    }

    @Override
    public void setPropertiesForFunction(Map<String, Object> properties) {
        super.setPropertiesForFunction(properties);
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        return Collections.singletonList(sameDiff.zerosLike(arg()));
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        Preconditions.checkState(dataTypes.size() == 1, "Expected list with exactly 1 datatype for %s, got %s", getClass(), dataTypes);
        if(!dArguments.isEmpty())
            return Collections.singletonList(dArguments.get(0));
        else if(dataType != null && dArguments.isEmpty()) {
            dArguments.add(dataType);
        }
        return Collections.singletonList(dataType == null ? DataType.LONG : dataType);
    }

    /**
     * Shape op output is always a 1D tensor containing the input shape.
     * Output shape = [input.rank()], dtype = LONG (or configured dataType).
     */
    @Override
    public List<DataBuffer> calculateOutputShapeFromInputs(OpContext oc) {
        if (oc == null || oc.numInputArguments() < 1) {
            return null;
        }

        INDArray input = oc.getInputArray(0);
        if (input == null) {
            return null;
        }

        // Output is 1D tensor with length = input rank
        long[] outputShape = new long[]{input.rank()};

        // Use configured dataType or default to LONG
        List<DataType> dArgs = oc.getDArguments();
        DataType dtype = (dArgs != null && !dArgs.isEmpty()) ? dArgs.get(0) :
                (this.dataType != null ? this.dataType : DataType.LONG);

        long[] strides = Nd4j.getStrides(outputShape, 'c');
        LongShapeDescriptor descriptor = LongShapeDescriptor.fromShape(outputShape, strides, 1, 'c', dtype, false);
        DataBuffer shapeInfo = org.nd4j.linalg.api.shape.Shape.createShapeInformation(descriptor);
        return Collections.singletonList(shapeInfo);
    }
}
