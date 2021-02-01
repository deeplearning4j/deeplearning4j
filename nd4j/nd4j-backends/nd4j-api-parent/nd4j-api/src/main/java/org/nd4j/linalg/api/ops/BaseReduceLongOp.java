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

package org.nd4j.linalg.api.ops;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.api.shape.Shape;

import java.util.Collections;
import java.util.List;

public abstract class BaseReduceLongOp extends BaseReduceOp implements ReduceLongOp {

    public BaseReduceLongOp(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, int[] dimensions) {
        super(sameDiff, i_v, i_v2, dimensions);
    }

    protected BaseReduceLongOp(SameDiff sameDiff, SDVariable input, int[] dimensions, boolean keepDims) {
        super(sameDiff, input, dimensions, keepDims);
    }

    protected BaseReduceLongOp(SameDiff sameDiff, SDVariable input, int... dimensions) {
        super(sameDiff, input, dimensions);
    }

    public BaseReduceLongOp(INDArray x, int... dimensions) {
        super(x, dimensions);
    }

    public BaseReduceLongOp(INDArray x, INDArray z, int... dimensions) {
        super(x, z, dimensions);
    }

    protected BaseReduceLongOp() {
        super();
    }

    @Override
    public Type opType() {
        return Type.REDUCE_LONG;
    }

    @Override
    public Type getOpType() {
        return opType();
    }

    @Override
    public DataType resultType() {
        return DataType.LONG;
    }

    @Override
    public DataType resultType(OpContext oc) {
        return DataType.LONG;
    }

    @Override
    public boolean validateDataTypes(OpContext oc) {
        INDArray x = oc != null ? oc.getInputArray(0) : x();
        INDArray y = oc != null ? oc.getInputArray(1) : y();
        if (y != null)
            Preconditions.checkArgument(x.dataType() == y.dataType(), "Op.X type must be the same as Op.Y:" +
                    " x.dataType=%s, y.dataType=%s, op=%s", x.dataType(), y.dataType(), getClass().getName());

        INDArray z = oc != null ? oc.getOutputArray(0) : z();
        if (z != null)
            Preconditions.checkArgument( z.dataType() == DataType.LONG,"Op.Z must be long: has type %s for op %s", z.dataType(), getClass());

        return true;
    }

    @Override
    public List<LongShapeDescriptor> calculateOutputShape() {
        return calculateOutputShape(null);
    }

    @Override
    public List<LongShapeDescriptor> calculateOutputShape(OpContext oc) {
        INDArray x = oc != null ? oc.getInputArray(0) : x();
        if(x == null)
            return Collections.emptyList();

        //Calculate reduction shape. Note that reduction on scalar - returns a scalar
        long[] reducedShape = x.rank() == 0 ? x.shape() : Shape.getReducedShape(x.shape(),dimensions, isKeepDims());
        return Collections.singletonList(LongShapeDescriptor.fromShape(reducedShape, DataType.LONG));
    }

    @Override
    public List<org.nd4j.linalg.api.buffer.DataType> calculateOutputDataTypes(List<org.nd4j.linalg.api.buffer.DataType> dataTypes){
        //All reduce long ops: always long output type
        //Second input is dynamic axis arg
        Preconditions.checkState(dataTypes != null && (dataTypes.size() == 1 || dataTypes.size() == 2),
                "Expected 1 or input datatype for %s, got input %s", getClass(), dataTypes);
        Preconditions.checkState(dataTypes.size() == 1 || dataTypes.get(1).isIntType(), "When executing reductions" +
                "with 2 inputs, second input (axis) must be an integer datatype for %s, got %s", getClass(), dataTypes);
        return Collections.singletonList(DataType.LONG);
    }
}
