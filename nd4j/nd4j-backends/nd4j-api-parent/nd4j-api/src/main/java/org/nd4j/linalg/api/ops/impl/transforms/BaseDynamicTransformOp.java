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

package org.nd4j.linalg.api.ops.impl.transforms;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collections;
import java.util.List;

@Slf4j
public abstract class BaseDynamicTransformOp extends DynamicCustomOp {

    public BaseDynamicTransformOp() {}

    public BaseDynamicTransformOp(SameDiff sameDiff, SDVariable[] args, boolean inPlace) {
        super(null, sameDiff, args, inPlace);
    }

    public BaseDynamicTransformOp(INDArray[] inputs, INDArray[] outputs) {
        super(null, inputs, outputs);
    }



    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        Preconditions.checkState(dataTypes != null && dataTypes.size() == 2, "Expected exactly 2 input datatypes for %s, got input %s", getClass(), dataTypes);

        DataType z = Shape.pickPairwiseDataType(dataTypes.get(0), dataTypes.get(1));
        return Collections.singletonList(z);
    }

    /**
     * Calculate output shape directly from input arrays in Java, avoiding JNI overhead.
     * Binary elementwise ops compute broadcast shape from the two inputs.
     */
    @Override
    public List<DataBuffer> calculateOutputShapeFromInputs(OpContext oc) {
        if (oc == null || oc.numInputArguments() < 2) {
            return null;
        }

        INDArray x = oc.getInputArray(0);
        INDArray y = oc.getInputArray(1);
        if (x == null || y == null) {
            return null;
        }

        long[] xShape = x.shape();
        long[] yShape = y.shape();

        // If either input is empty, the output is empty (TF broadcast semantics)
        if (x.isEmpty() || y.isEmpty()) {
            return null; // Fall back to C++ which handles empty array broadcasting correctly
        }

        // Shape.broadcastOutputShape has a bug with rank-1 arrays, so only use Java-side
        // inference when ranks are equal (most common case during inference)
        if (xShape.length != yShape.length) {
            return null; // Different ranks - fall back to C++ for correct broadcasting
        }

        // Use calculateOutputDataTypes to get the correct output dtype
        // (e.g., comparison ops like equals return BOOL, not the input dtype)
        List<DataType> inputDtypes = java.util.Arrays.asList(x.dataType(), y.dataType());
        List<DataType> outputDtypes;
        try {
            outputDtypes = calculateOutputDataTypes(inputDtypes);
        } catch (Exception e) {
            // Fall back to default pairwise dtype if calculation fails
            outputDtypes = Collections.singletonList(Shape.pickPairwiseDataType(x.dataType(), y.dataType()));
        }
        DataType outputDtype = outputDtypes != null && !outputDtypes.isEmpty()
                ? outputDtypes.get(0)
                : Shape.pickPairwiseDataType(x.dataType(), y.dataType());

        // Compute broadcast shape (safe now since ranks are equal).
        // broadcastOutputShape throws on non-broadcastable shapes — catch and
        // fall back to C++ shape function which may handle the case differently.
        long[] outputShape;
        try {
            outputShape = Shape.broadcastOutputShape(xShape, yShape);
        } catch (Exception e) {
            return null; // Shapes not broadcastable - fall back to C++
        }
        if (outputShape == null) {
            return null; // Shapes not broadcastable - fall back to C++
        }

        // Create shape info buffer using LongShapeDescriptor (routes through native cache)
        long[] strides = Nd4j.getStrides(outputShape, 'c');
        boolean isEmpty = false;
        for (long dim : outputShape) {
            if (dim == 0) { isEmpty = true; break; }
        }
        LongShapeDescriptor descriptor = LongShapeDescriptor.fromShape(outputShape, strides, 1, 'c', outputDtype, isEmpty);
        DataBuffer shapeInfo = Shape.createShapeInformation(descriptor);
        return Collections.singletonList(shapeInfo);
    }
}
