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

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.util.SameDiffUtils;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Base scalar operation
 *
 * @author Adam Gibson
 */
@Slf4j
public abstract class BaseScalarOp extends BaseOp implements ScalarOp {

    public BaseScalarOp() {
        this.scalarValue = Nd4j.scalar(0.f);
    }

    public BaseScalarOp(INDArray x, INDArray y, INDArray z, Number num) {
        super(x, y, z);
        if (x.isCompressed())
            Nd4j.getCompressor().decompressi(x);

        try(MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
            this.scalarValue = Nd4j.scalar(x.dataType(), num);
        }
    }

    public BaseScalarOp(INDArray x, Number num) {
        super(x);
        if (x.isCompressed())
            Nd4j.getCompressor().decompressi(x);

        try(MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
            this.scalarValue = Nd4j.scalar(x.dataType(), num);
        }

    }
    public BaseScalarOp(INDArray x, INDArray z, Number set) {
        super(x, null, z);
        if (x.isCompressed())
            Nd4j.getCompressor().decompressi(x);

        try(MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
            this.scalarValue = Nd4j.scalar(x.dataType(), set);
        }
    }




    public BaseScalarOp(SameDiff sameDiff,SDVariable i_v,Number scalar) {
        this(sameDiff,i_v,scalar,false,null);
    }

    public BaseScalarOp(SameDiff sameDiff,SDVariable i_v,Number scalar,boolean inPlace) {
        this(sameDiff,i_v,scalar,inPlace,null);
    }

    public BaseScalarOp(SameDiff sameDiff,
                        @NonNull SDVariable i_v,
                        Number scalar,
                        boolean inPlace,
                        Object[] extraArgs) {
        super(sameDiff,inPlace,extraArgs);
        this.scalarValue = Nd4j.scalar(i_v.dataType(), scalar);
        this.xVertexId = i_v.name();
        sameDiff.addArgsFor(new String[]{xVertexId},this);
        SameDiffUtils.validateDifferentialFunctionSameDiff(sameDiff, i_v, this);
    }


    public BaseScalarOp(SameDiff sameDiff,
                        SDVariable i_v,
                        Number scalar,
                        Object[] extraArgs) {
        this(sameDiff,i_v,scalar,false,extraArgs);
    }



    @Override
    public INDArray z() {
        return z;
    }


    @Override
    public List<LongShapeDescriptor> calculateOutputShape() {
        return calculateOutputShape(null);
    }

    @Override
    public List<LongShapeDescriptor> calculateOutputShape(OpContext oc) {
        INDArray x = oc != null ? oc.getInputArray(0) : x();

        val ret = new ArrayList<LongShapeDescriptor>(1);

        long[] s;
        if(x != null){
            s = x.shape();
        } else {
            s = arg().getShape();
        }

        val aT = arg().dataType();
        val sT = scalarValue.dataType();

        ret.add(LongShapeDescriptor.fromShape(s, Shape.pickPairwiseDataType(aT, sT)));
        return ret;
    }

    @Override
    public Type opType() {
        return Type.SCALAR;
    }

    @Override
    public void setScalar(Number scalar) {
        this.scalarValue = Nd4j.scalar(x.dataType(), scalar);
    }

    @Override
    public void setScalar(INDArray scalar){
        this.scalarValue = scalar;
    }

    @Override
    public INDArray scalar() {
        if(y() != null && y().isScalar())
            return y();
        return scalarValue;
    }

    @Override
    public int[] getDimension() {
        return dimensions;
    }

    @Override
    public void setDimension(int... dimension) {
        defineDimensions(dimension);
    }

    @Override
    public boolean validateDataTypes(boolean experimentalMode) {
        if (y() != null) {
            if (y().isR() || x().isR())
                Preconditions.checkArgument(z().isR(), "Op.Z must have floating point type, since one of operands is floating point:" +
                        " x.dataType=%s, y.dataType=%s, z.dataType=%s, op=%s", x.dataType(), y.dataType(), z.dataType(), getClass().getName());

            if (!experimentalMode)
                Preconditions.checkArgument(x.dataType() == y.dataType()  || y.dataType() == DataType.BOOL, "Op.X must have same data type as Op.Y");
        } else if (x().isR())
            Preconditions.checkArgument(z().isR(), "Op.Z must have floating point type, since one of operands is floating point:" +
                    " x.dataType=%s, z.dataType=%s, op=%s", x.dataType(), z.dataType(), getClass().getName());


        return true;
    }

    @Override
    public Type getOpType() {
        return Type.SCALAR;
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        //All scalar ops: output type is same as input type
        Preconditions.checkState(dataTypes != null && dataTypes.size() >= 1, "Expected 1 or more input datatype %s, got input %s", getClass(), dataTypes);
        return Collections.singletonList(dataTypes.get(0));
    }

}
