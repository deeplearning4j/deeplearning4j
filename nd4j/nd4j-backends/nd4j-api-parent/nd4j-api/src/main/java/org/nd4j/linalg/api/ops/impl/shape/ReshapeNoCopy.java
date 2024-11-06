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
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.api.shape.options.ArrayOptionsHelper;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collections;
import java.util.List;

public class ReshapeNoCopy extends DynamicCustomOp {

    public static final int RESHAPE_NO_COPY_C_ORDER_MARKER = -99;
    public static final int RESHAPE_NO_COPY_F_ORDER_MARKER = -102;

    public ReshapeNoCopy() {
        // Default constructor
    }

    public ReshapeNoCopy(INDArray input, INDArray output) {
        addInputArgument(input);
        addOutputArgument(output);
    }

    public ReshapeNoCopy(INDArray input, INDArray shape, INDArray output) {
        addInputArgument(input, shape);
        addOutputArgument(output);
    }

    public ReshapeNoCopy(INDArray input, long[] shape, INDArray output) {
        this(input, shape, output, 'c');
    }

    public ReshapeNoCopy(INDArray input, long[] shape, INDArray output, char order) {
        addInputArgument(input);
        if(output != null) {
            addOutputArgument(output);
        }

        addShapeOrder(shape, order);
    }

    private void addShapeOrder(long[] shape, char order) {
        if (order == 'c') {
            addIArgument(shape);
            addIArgument(RESHAPE_NO_COPY_C_ORDER_MARKER);
        } else if (order == 'f') {
            addIArgument(shape);
            addIArgument(RESHAPE_NO_COPY_F_ORDER_MARKER);
        } else {
            throw new IllegalArgumentException("No order defined for reshape!");
        }
    }

    public ReshapeNoCopy(SameDiff sameDiff, SDVariable input, SDVariable shape) {
        super(null, sameDiff, new SDVariable[]{input, shape});
    }

    public ReshapeNoCopy(SameDiff sameDiff, SDVariable input, long[] shape) {
        this(sameDiff, input, shape, 'c');
    }

    public ReshapeNoCopy(SameDiff sameDiff, SDVariable input, long[] shape, char order) {
        super(null, sameDiff, new SDVariable[]{input});
        addShapeOrder(shape, order);
    }

    @Override
    public boolean initializeOutputs(OpContext ctx) {
        boolean shapeOverride = false;
        if (numOutputArguments() == 0 && !isInplaceCall()) {
            try {
                val list = Nd4j.getExecutioner().calculateOutputShape(this,ctx);
                if (list.isEmpty())
                    throw new ND4JIllegalStateException("Op name " + opName() + " failed to calculate output shape and data types.");

                LongShapeDescriptor needsCopyShape = list.get(0);
                if(ArrayOptionsHelper.arrayNeedsCopy(needsCopyShape.getExtras())) {
                    INDArray newOut = Nd4j.create(needsCopyShape, false);
                    addOutputArgument(newOut);
                } else {
                    INDArray newOut = Nd4j.create(inputArguments.get(0).data(),needsCopyShape);
                    addOutputArgument(newOut);
                }

                shapeOverride = true;
            } catch (ND4JIllegalStateException e) {
                throw e;
            } catch (Exception e) {
                String lastErrorMessage = Nd4j.getNativeOps().lastErrorMessage();
                throw new ND4JIllegalStateException("Op name " + opName() + " - no output arrays were provided and calculateOutputShape failed to execute error message: " + lastErrorMessage, e);
            }
        }

        return shapeOverride;
    }

    @Override
    public String opName() {
        return "reshape_no_copy";
    }

    @Override
    public String tensorflowName() {
        return "ReshapeNoCopy";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradients) {
        // The gradient of a reshape operation is just a reshape of the gradient back to the original shape
        SDVariable grad = gradients.get(0);
        SDVariable originalShape = arg(0).shape();
        return Collections.singletonList(grad.reshape(originalShape));
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        // The output type is the same as the input type
        return Collections.singletonList(inputDataTypes.get(0));
    }
}