/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public abstract class BaseTransformFloatOp extends BaseTransformOp implements TransformFloatOp {

    public BaseTransformFloatOp(SameDiff sameDiff, SDVariable i_v, int[] shape, boolean inPlace, Object[] extraArgs) {
        super(sameDiff, i_v, shape, inPlace, extraArgs);
    }

    public BaseTransformFloatOp(SameDiff sameDiff, SDVariable i_v, boolean inPlace) {
        super(sameDiff, i_v, inPlace);
    }

    public BaseTransformFloatOp(SameDiff sameDiff, SDVariable i_v, long[] shape, boolean inPlace, Object[] extraArgs) {
        super(sameDiff, i_v, shape, inPlace, extraArgs);
    }

    public BaseTransformFloatOp(SameDiff sameDiff, SDVariable i_v, Object[] extraArgs) {
        super(sameDiff, i_v, extraArgs);
    }

    public BaseTransformFloatOp(INDArray x, INDArray z) {
        super(x, z);
    }

    public BaseTransformFloatOp() {
        super();
    }

    public BaseTransformFloatOp(INDArray x) {
        super(x);
    }

    @Override
    public Type getOpType() {
        return Type.TRANSFORM_FLOAT;
    }

    @Override
    public Type opType() {
        return Type.TRANSFORM_FLOAT;
    }

    @Override
    public DataType resultType() {
        if (this.x() != null && this.x().isR())
            return this.x().dataType();

        return Nd4j.defaultFloatingPointType();
    }

    @Override
    public boolean validateDataTypes(boolean experimentalMode) {
        if (y() != null && !experimentalMode) {
            Preconditions.checkArgument(x.dataType() == y.dataType(), "Op.X must have same data type as Op.Y");
        }

        if (z() != null)
            Preconditions.checkArgument(z().isR(),"Op.Z must be one of floating types: z.datatype=%s for op %s", z().dataType(), getClass());

        return true;
    }

    @Override
    public List<LongShapeDescriptor> calculateOutputShape() {
        if(x == null)
            return Collections.emptyList();
        return Collections.singletonList(LongShapeDescriptor.fromShape(x.shape(), x.isR() ? x.dataType() : Nd4j.defaultFloatingPointType()));
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        Preconditions.checkState(dataTypes != null && dataTypes.size() == 1, "Expected exactly 1 input datatype for %s, got input %s", getClass(), dataTypes);
        if(dataTypes.get(0).isFPType())
            return Collections.singletonList(dataTypes.get(0));
        //TODO is this what we want for all cases?
        return Collections.singletonList(DataType.FLOAT);
    }
}
