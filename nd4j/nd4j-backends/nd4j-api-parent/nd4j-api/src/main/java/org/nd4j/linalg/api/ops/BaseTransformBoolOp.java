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

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public abstract class BaseTransformBoolOp extends BaseTransformOp implements TransformSameOp {

    public BaseTransformBoolOp(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2) {
        super(sameDiff, i_v1, i_v2);
    }

    public BaseTransformBoolOp(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2, boolean inPlace) {
        super(sameDiff, i_v1, i_v2, inPlace);
    }

    public BaseTransformBoolOp(SameDiff sameDiff, SDVariable i_v, int[] shape, boolean inPlace, Object[] extraArgs) {
        super(sameDiff, i_v, shape, inPlace, extraArgs);
    }

    public BaseTransformBoolOp(SameDiff sameDiff, SDVariable i_v, boolean inPlace) {
        super(sameDiff, i_v, inPlace);
    }

    public BaseTransformBoolOp(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }


    public BaseTransformBoolOp(SameDiff sameDiff) {
        super(sameDiff);
    }

    public BaseTransformBoolOp(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2, Object[] extraArgs) {
        super(sameDiff, i_v1, i_v2, extraArgs);
    }

    public BaseTransformBoolOp(SameDiff sameDiff, SDVariable i_v, long[] shape, boolean inPlace, Object[] extraArgs) {
        super(sameDiff, i_v, shape, inPlace, extraArgs);
    }

    public BaseTransformBoolOp(SameDiff sameDiff, SDVariable i_v, Object[] extraArgs) {
        super(sameDiff, i_v, extraArgs);
    }

    public BaseTransformBoolOp(INDArray x, INDArray z) {
        super(x, z);
    }

    public BaseTransformBoolOp() {
        super();
    }

    public BaseTransformBoolOp(INDArray x, INDArray z, long n) {
        super(x, z, n);
    }


    public BaseTransformBoolOp(INDArray x) {
        super(x);
    }

    @Override
    public Type getOpType() {
        return Type.TRANSFORM_BOOL;
    }

    @Override
    public Type opType() {
        return Type.TRANSFORM_BOOL;
    }

    @Override
    public DataType resultType() {
        return DataType.BOOL;
    }

    @Override
    public boolean validateDataTypes(boolean experimentalMode) {
        if (y() != null)
            Preconditions.checkArgument(x().dataType() == y().dataType(), "Op.X must be the same type as Op.Y: " +
                    "x.datatype=%s, y.datatype=%s", x().dataType(), y.dataType());


        if (z() != null)
            Preconditions.checkArgument(z().isB(),"Op.Z type must be bool: z.datatype=%s for op %s", z().dataType(), getClass());

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

        ret.add(LongShapeDescriptor.fromShape(arr.shape(), DataType.BOOL));
        this.n = arr.length();
        return ret;
    }

    @Override
    public List<org.nd4j.linalg.api.buffer.DataType> calculateOutputDataTypes(List<org.nd4j.linalg.api.buffer.DataType> dataTypes){
        //All bool tranform ops: always bool output type
        SDVariable[] args = args();
        Preconditions.checkState(dataTypes != null && dataTypes.size() == args.length, "Expected exactly %s input datatype(s), got input %s", args.length, dataTypes);
        return Collections.singletonList(DataType.BOOL);
    }
}
