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

package org.nd4j.linalg.api.ops;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collections;
import java.util.List;

public abstract class BaseReduceBoolOp extends BaseReduceOp implements ReduceBoolOp {
    public BaseReduceBoolOp(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, long[] dimensions) {
        super(sameDiff, i_v, i_v2, dimensions);
    }

    public BaseReduceBoolOp(SameDiff sameDiff, SDVariable i_v, boolean keepDims) {
        super(sameDiff, i_v, keepDims);
    }

    public BaseReduceBoolOp(SameDiff sameDiff, SDVariable i_v, SDVariable dimensions, boolean keepDims) {
        super(sameDiff, i_v, dimensions, keepDims);
    }

    public BaseReduceBoolOp(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2) {
        super(sameDiff, i_v, i_v2);
    }

    protected BaseReduceBoolOp(SameDiff sameDiff, SDVariable input, long[] dimensions, boolean keepDims) {
        super(sameDiff, input, dimensions, keepDims);
    }

    public BaseReduceBoolOp(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, long[] dimensions, boolean keepDims) {
        super(sameDiff, i_v, i_v2, dimensions, keepDims);
    }

    public BaseReduceBoolOp(SameDiff sameDiff, SDVariable i_v) {
        super(sameDiff, i_v);
    }

    protected BaseReduceBoolOp(SameDiff sameDiff, SDVariable input, long... dimensions) {
        super(sameDiff, input, dimensions);
    }


    public BaseReduceBoolOp(INDArray x, INDArray z, boolean keepDims, long[] dimensions) {
        super(x, null, z, keepDims, dimensions);
    }

    public BaseReduceBoolOp(INDArray x, long... dimensions) {
        this(x, null, false, dimensions);
    }

    public BaseReduceBoolOp(INDArray x, boolean keepDims, long... dimensions) {
        super(x, keepDims, dimensions);
    }

    public BaseReduceBoolOp(INDArray x, INDArray z, long... dimensions) {
        this(x, z, false, dimensions);
    }

    public BaseReduceBoolOp(INDArray x, INDArray y, INDArray z, long... dimensions) {
        super(x, y, z, dimensions);
    }

    public BaseReduceBoolOp(SameDiff sameDiff) {
        super(sameDiff);
    }

    public BaseReduceBoolOp(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, SDVariable dimensions) {
        super(sameDiff, i_v, i_v2, dimensions);
    }

    protected BaseReduceBoolOp() {
        super();
    }

    public BaseReduceBoolOp(INDArray x, INDArray y, INDArray z, boolean keepDims, long[] dimensions) {
        super(x, y, z, keepDims, dimensions);
    }

    @Override
    public Type opType() {
        return Type.REDUCE_BOOL;
    }

    @Override
    public Type getOpType() {
        return opType();
    }

    @Override
    public DataType resultType() {
        return DataType.BOOL;
    }

    @Override
    public DataType resultType(OpContext oc) {
        return DataType.BOOL;
    }

    @Override
    public boolean validateDataTypes(OpContext oc) {
        INDArray x = oc != null ? oc.getInputArray(0) : x();
        INDArray y = oc != null ? oc.getInputArray(1) : y();
        if (y != null)
            Preconditions.checkArgument(x.dataType()  == y.dataType(),"Op.X type must be the same as Op.Y:" +
                            " x.dataType=%s, y.dataType=%s, op=%s", x.dataType(), y.dataType(), getClass().getName());

        INDArray z = oc != null ? oc.getOutputArray(0) : z();
        if (z != null)
            Preconditions.checkArgument(z.isB(), "Op.Z type must be bool: got type %s for op %s", z.dataType(), getClass());

        return true;
    }

    @Override
    public List<DataBuffer> calculateOutputShape() {
        return calculateOutputShape(null);
    }

    @Override
    public List<DataBuffer> calculateOutputShape(OpContext oc) {
        INDArray x = oc != null ? oc.getInputArray(0) : x();
        if(x == null)
            return Collections.emptyList();

        //Calculate reduction shape. Note that reduction on scalar - returns a scalar
        long[] reducedShape = x.rank() == 0 ? x.shape() : Shape.getReducedShape(x.shape(),dimensions, isKeepDims());
        return Collections.singletonList(Nd4j.createBuffer(LongShapeDescriptor.fromShape(reducedShape, DataType.BOOL).toShapeInfo()));
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        //All reduce bool: always bool output type. 2nd input is axis arg
        Preconditions.checkState(dataTypes != null && (dataTypes.size() == 1 || dataTypes.size() == 2),
                "Expected 1 or input datatype for %s, got input %s", getClass(), dataTypes);
        Preconditions.checkState(dataTypes.size() == 1 || dataTypes.get(1).isIntType(), "When executing reductions" +
                "with 2 inputs, second input (axis) must be an integer datatype for %s, got %s", getClass(), dataTypes);
        return Collections.singletonList(DataType.BOOL);
    }


    public abstract boolean emptyValue();
}
