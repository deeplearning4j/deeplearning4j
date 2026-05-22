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

package org.nd4j.linalg.api.ops.impl.transforms.custom;

import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collections;
import java.util.List;

public class Trace extends DynamicCustomOp {

    public Trace(SameDiff sd, SDVariable in){
        super(null, sd, new SDVariable[]{in});
    }

    public Trace(@NonNull INDArray in){
        super(wrapOrNull(in), null);
    }

    public Trace(){ }

    @Override
    public String opName(){
        return "trace";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradAtOutput){
        SDVariable rows = sameDiff.reshape(sameDiff.sizeAt(arg(), -2), 1);
        SDVariable cols = sameDiff.reshape(sameDiff.sizeAt(arg(), -1), 1);
        SDVariable eye = sameDiff.math().eye(/*sameDiff.shape(gradAtOutput.get(0)),*/ rows, cols);
        //Reshape gradient from [x,y,z] to [x,y,z,1,1]
        SDVariable reshapedGrad = sameDiff.expandDims(gradAtOutput.get(0), -1);
        reshapedGrad = sameDiff.expandDims(reshapedGrad, -1);
        return Collections.singletonList(reshapedGrad.mul(eye));
    }

    @Override
    public List<DataBuffer> calculateOutputShapeFromInputs(OpContext oc) {
        INDArray input = null;
        if (oc != null && oc.numInputArguments() >= 1) {
            input = oc.getInputArray(0);
        } else if (numInputArguments() >= 1) {
            input = getInputArgument(0);
        }
        if (input == null) {
            return null;
        }

        int inRank = input.rank();
        if (inRank < 2) {
            return null;
        }

        // Output rank = input rank - 2 (remove the last two matrix dimensions)
        int outRank = inRank - 2;
        DataType dtype = input.dataType();

        if (outRank == 0) {
            // Scalar output — use Nd4j scalar shape info
            LongShapeDescriptor descriptor = LongShapeDescriptor.fromShape(new long[0], dtype);
            DataBuffer shapeInfo = Shape.createShapeInformation(descriptor);
            return Collections.singletonList(shapeInfo);
        }

        long[] outputShape = new long[outRank];
        long[] inputShape = input.shape();
        for (int i = 0; i < outRank; i++) {
            outputShape[i] = inputShape[i];
        }

        long[] strides = Nd4j.getStrides(outputShape, 'c');
        boolean isEmpty = false;
        for (long dim : outputShape) {
            if (dim == 0) { isEmpty = true; break; }
        }
        LongShapeDescriptor descriptor = LongShapeDescriptor.fromShape(outputShape, strides, 1, 'c', dtype, isEmpty);
        DataBuffer shapeInfo = Shape.createShapeInformation(descriptor);
        return Collections.singletonList(shapeInfo);
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        Preconditions.checkState(dataTypes != null && dataTypes.size() == 1, "Expected exactly 1 input datatype for %s, got %s", getClass(), dataTypes);
        return Collections.singletonList(dataTypes.get(0));
    }

}
