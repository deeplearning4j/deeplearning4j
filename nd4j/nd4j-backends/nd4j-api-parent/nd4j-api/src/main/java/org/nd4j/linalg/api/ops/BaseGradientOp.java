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

/**
 * A gradient op always makes the following assumptions:
 * there is always a y (beacuse of backpropagating
 * or using the chain rule)
 *
 * and that it is special exec (for now)
 *
 * This op opType sis meant to be used
 * to build derivative operations.
 *
 *
 * @author Adam Gibson
 */
public abstract class BaseGradientOp extends BaseTransformOp implements GradientOp {
    public BaseGradientOp(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2) {
        super(sameDiff, i_v1, i_v2);
    }

    public BaseGradientOp(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2, boolean inPlace) {
        super(sameDiff, i_v1, i_v2, inPlace);
    }

    public BaseGradientOp(INDArray x, INDArray z) {
        super(x, z);
        assertWrt();
    }

    public BaseGradientOp() {
    }

    public BaseGradientOp(INDArray x, INDArray z, long n) {
        super(x, z, n);
        assertWrt();
    }

    public BaseGradientOp(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
        assertWrt();
    }

    public BaseGradientOp(INDArray x) {
        super(x);
        assertWrt();
    }

    /**
     * The array
     * to the gradient with respect to
     *
     * @return
     */
    @Override
    public INDArray wrt() {
        return y();
    }

    @Override
    public boolean isExecSpecial() {
        return true;
    }

    @Override
    public boolean isPassThrough() {
        return true;
    }

    private void assertWrt() {
        Preconditions.checkState(y != null,"A gradient op must define a wrt variable as a Y. ");
    }

    @Override
    public Type getOpType() {
        return Type.TRANSFORM_STRICT;
    }

    @Override
    public boolean validateDataTypes(boolean experimentalMode) {
        if (!x().isR())
            throw new ND4JIllegalArgumentException("Op.X must be one of floating types");

        if (y() != null && !y().isR())
            throw new ND4JIllegalArgumentException("Op.Y must be one of floating types");

        if (z() != null && z().dataType() != x().dataType())
            throw new ND4JIllegalArgumentException("Op.Z type must be the same as Op.X type");

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
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == 1, "Expected exactly 1 input datatype, got %s");
        return inputDataTypes;
    }
}
