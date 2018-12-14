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

public abstract class BaseTransformAnyOp extends BaseTransformOp implements TransformSameOp {

    public BaseTransformAnyOp(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2) {
        super(sameDiff, i_v1, i_v2);
    }

    public BaseTransformAnyOp(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2, boolean inPlace) {
        super(sameDiff, i_v1, i_v2, inPlace);
    }

    public BaseTransformAnyOp(SameDiff sameDiff) {
        super(sameDiff);
    }

    public BaseTransformAnyOp(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }

    public BaseTransformAnyOp(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2, Object[] extraArgs) {
        super(sameDiff, i_v1, i_v2, extraArgs);
    }

    public BaseTransformAnyOp(SameDiff sameDiff, SDVariable i_v, int[] shape, boolean inPlace, Object[] extraArgs) {
        super(sameDiff, i_v, shape, inPlace, extraArgs);
    }

    public BaseTransformAnyOp(SameDiff sameDiff, SDVariable i_v, boolean inPlace) {
        super(sameDiff, i_v, inPlace);
    }

    public BaseTransformAnyOp(SameDiff sameDiff, SDVariable i_v, long[] shape, boolean inPlace, Object[] extraArgs) {
        super(sameDiff, i_v, shape, inPlace, extraArgs);
    }

    public BaseTransformAnyOp(SameDiff sameDiff, SDVariable i_v, Object[] extraArgs) {
        super(sameDiff, i_v, extraArgs);
    }

    public BaseTransformAnyOp(INDArray x, INDArray z) {
        super(x, z);
    }

    public BaseTransformAnyOp() {
        super();
    }

    public BaseTransformAnyOp(INDArray x, INDArray z, long n) {
        super(x, z, n);
    }


    public BaseTransformAnyOp(INDArray x) {
        super(x);
    }

    @Override
    public Type getOpType() {
        return Type.TRANSFORM_ANY;
    }

    @Override
    public Type opType() {
        return Type.TRANSFORM_ANY;
    }

    @Override
    public DataType resultType() {
        return this.x().dataType();
    }

    @Override
    public boolean validateDataTypes(boolean experimentalMode) {
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
        //Transform any: for the purposes of samediff datatype calculation, treat as same in/out
        Preconditions.checkState(dataTypes != null && dataTypes.size() == 1, "Expected exactly 1 input datatype, got input %s", dataTypes);
        return dataTypes;
    }
}
