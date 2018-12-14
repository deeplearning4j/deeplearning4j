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
import org.nd4j.linalg.exception.ND4JIllegalArgumentException;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public abstract class BaseTransformStrictOp extends BaseTransformOp implements TransformStrictOp {

    public BaseTransformStrictOp(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2) {
        super(sameDiff, i_v1, i_v2);
    }

    public BaseTransformStrictOp(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2, boolean inPlace) {
        super(sameDiff, i_v1, i_v2, inPlace);
    }

    public BaseTransformStrictOp(SameDiff sameDiff, SDVariable i_v, int[] shape, boolean inPlace, Object[] extraArgs) {
        super(sameDiff, i_v, shape, inPlace, extraArgs);
    }

    public BaseTransformStrictOp(SameDiff sameDiff, SDVariable i_v, boolean inPlace) {
        super(sameDiff, i_v, inPlace);
    }

    public BaseTransformStrictOp(SameDiff sameDiff, SDVariable i_v, long[] shape, boolean inPlace, Object[] extraArgs) {
        super(sameDiff, i_v, shape, inPlace, extraArgs);
    }

    public BaseTransformStrictOp(SameDiff sameDiff, SDVariable i_v, Object[] extraArgs) {
        super(sameDiff, i_v, extraArgs);
    }

    public BaseTransformStrictOp(INDArray x, INDArray z) {
        super(x, z);
    }

    public BaseTransformStrictOp() {
        super();
    }

    public BaseTransformStrictOp(INDArray x, INDArray z, long n) {
        super(x, z, n);
    }


    public BaseTransformStrictOp(INDArray x) {
        super(x);
    }

    @Override
    public Type getOpType() {
        return Type.TRANSFORM_STRICT;
    }

    @Override
    public Type opType() {
        return Type.TRANSFORM_STRICT;
    }

    @Override
    public DataType resultType() {
        return this.x().dataType();
    }


    @Override
    public boolean validateDataTypes(boolean experimentalMode) {
        Preconditions.checkArgument(x().isR(), "Op.X must be one of floating types: x.datatype=%s for op %s", x().dataType(), getClass());

        if (y() != null) {
            Preconditions.checkArgument(y().isR(), "Op.Y must be one of floating types: y.datatype=%s for op %s", y().dataType(), getClass());

            if (!experimentalMode)
                Preconditions.checkArgument(x.dataType() == y.dataType(), "Op.X must have same data type as Op.Y");
        }

        if (z() != null)
            Preconditions.checkArgument(z().dataType() == x().dataType(), "Op.Z must have the same type as Op.X: x.datatype=%s, z.datatype=%s for op %s",
                    x.dataType(), z.dataType(), getClass());

        return true;
    }

    @Override
    public List<LongShapeDescriptor> calculateOutputShape() {
        val ret = new ArrayList<LongShapeDescriptor>(1);
        if(arg() == null)
            throw new ND4JIllegalStateException("No arg found for op!");

        val arr = sameDiff.getArrForVarName(arg().getVarName());
        if(arr == null)
            return Collections.emptyList();

        ret.add(LongShapeDescriptor.fromShape(arr.shape(), arr.dataType()));
        this.n = arr.length();
        return ret;
    }

    @Override
    public List<org.nd4j.linalg.api.buffer.DataType> calculateOutputDataTypes(List<org.nd4j.linalg.api.buffer.DataType> dataTypes){
        //All strict tranform ops: FP in, FP out
        Preconditions.checkState(dataTypes != null && dataTypes.size() == 1, "Expected exactly 1 input datatype, got input %s", dataTypes);
        Preconditions.checkState(dataTypes.get(0).isFPType(), "Only floating point types are supported for strict tranform ops - got %s", dataTypes.get(0));
        return dataTypes;
    }
}
