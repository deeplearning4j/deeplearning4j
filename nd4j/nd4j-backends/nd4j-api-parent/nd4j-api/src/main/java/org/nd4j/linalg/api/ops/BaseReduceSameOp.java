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

package org.nd4j.linalg.api.ops;

import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.exception.ND4JIllegalArgumentException;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public abstract class BaseReduceSameOp extends BaseReduceOp implements ReduceSameOp {

    public BaseReduceSameOp(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, int[] dimensions) {
        super(sameDiff, i_v, i_v2, dimensions);
    }

    protected BaseReduceSameOp(SameDiff sameDiff, SDVariable input, int[] dimensions, boolean keepDims) {
        super(sameDiff, input, dimensions, keepDims);
    }

    protected BaseReduceSameOp(SameDiff sameDiff, SDVariable input, int... dimensions) {
        super(sameDiff, input, dimensions);
    }

    public BaseReduceSameOp(INDArray x, INDArray z, boolean keepDims, int[] dimensions) {
        super(x, null, z, keepDims, dimensions);
    }

    public BaseReduceSameOp(INDArray x, INDArray y, INDArray z, int... dimensions) {
        super(x, y, z, dimensions);
    }

    public BaseReduceSameOp(INDArray x, int... dimensions) {
        super(x, dimensions);
    }

    public BaseReduceSameOp(INDArray x, boolean keepDims, int... dimensions) {
        super(x, keepDims, dimensions);
    }

    protected BaseReduceSameOp() {
        super();
    }

    @Override
    public Type opType() {
        return Type.REDUCE_SAME;
    }

    @Override
    public Type getOpType() {
        return opType();
    }

    @Override
    public DataType resultType() {
        return this.x().dataType();
    }

    @Override
    public boolean validateDataTypes() {
        if (y() != null)
            Preconditions.checkArgument(x().dataType() == y().dataType(),"Op.X type must be the same as Op.Y type:" +
                    " x.dataType=%s, y.dataType=%s, op=%s", x.dataType(), y.dataType(), getClass().getName());

        if (z() != null)
            Preconditions.checkArgument(z().dataType() == x().dataType(), "Op.Z must be the same as Op.X type. Op.X.datatype=%s, " +
                    "Op.Z.datatype=%s", x().dataType(), z.dataType());

        return true;
    }

    @Override
    public List<LongShapeDescriptor> calculateOutputShape() {
        if(x == null)
            return Collections.emptyList();

        //Calculate reduction shape. Note that reduction on scalar - returns a scalar
        long[] reducedShape = x.rank() == 0 ? x.shape() : Shape.getReducedShape(x.shape(),dimensions, isKeepDims());
        return Collections.singletonList(LongShapeDescriptor.fromShape(reducedShape, this.resultType()));
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        //Output type: same as input type for BaseReduceSameOp
        //Note TF uses 2 inputs - i.e., axis arg is a variable or constant
        Preconditions.checkState(dataTypes != null && (dataTypes.size() == 1 || dataTypes.size() == 2),
                "Expected 1 or 2 input datatypes for %s, got %s", getClass(), dataTypes);
        Preconditions.checkState(dataTypes.size() == 1 || dataTypes.get(1).isIntType(), "When executing reductions" +
                "with 2 inputs, second input (axis) must be an integer datatype for %s, got %s", getClass(), dataTypes);
        return Collections.singletonList(dataTypes.get(0));
    }
}
