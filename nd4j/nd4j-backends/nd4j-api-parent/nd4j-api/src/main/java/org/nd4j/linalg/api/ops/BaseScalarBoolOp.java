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

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.util.SameDiffUtils;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collections;
import java.util.List;

/**
 * Base scalar boolean operation
 *
 * @author Adam Gibson
 */
@Slf4j
public abstract class BaseScalarBoolOp extends BaseOp implements ScalarOp {
    public BaseScalarBoolOp() {}

    public BaseScalarBoolOp(INDArray x, INDArray y, INDArray z, Number num) {
        super(x, y, z);
        this.scalarValue = Nd4j.scalar(x.dataType(), num);
    }

    public BaseScalarBoolOp(INDArray x, Number num) {
        super(x);
        this.scalarValue = Nd4j.scalar(x.dataType(), num);
    }

    public BaseScalarBoolOp(INDArray x, INDArray z, Number set) {
        super(x, null, z);
        this.scalarValue= Nd4j.scalar(x.dataType(), set);
    }




    public BaseScalarBoolOp(SameDiff sameDiff, SDVariable i_v, Number scalar) {
        this(sameDiff,i_v,scalar,false,null);
    }

    public BaseScalarBoolOp(SameDiff sameDiff, SDVariable i_v, Number scalar, boolean inPlace) {
        this(sameDiff,i_v,scalar,inPlace,null);
    }

    public BaseScalarBoolOp(SameDiff sameDiff,
                            SDVariable i_v,
                            Number scalar,
                            boolean inPlace,
                            Object[] extraArgs) {
        super(sameDiff,inPlace,extraArgs);
        this.scalarValue = Nd4j.scalar(i_v.dataType(), scalar);
        if (i_v != null) {
            this.xVertexId = i_v.name();
            sameDiff.addArgsFor(new String[]{xVertexId},this);
            SameDiffUtils.validateDifferentialFunctionSameDiff(sameDiff, i_v, this);
        } else {
            throw new IllegalArgumentException("Input not null variable.");
        }

    }


    public BaseScalarBoolOp(SameDiff sameDiff,
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
        if(x == null)
            return Collections.emptyList();

        //Calculate reduction shape. Note that reduction on scalar - returns a scalar
        return Collections.singletonList(LongShapeDescriptor.fromShape(x.shape(), DataType.BOOL));
    }

    @Override
    public Type opType() {
        return Type.SCALAR_BOOL;
    }

    @Override
    public void setScalar(Number scalar) {
        this.scalarValue = Nd4j.scalar(scalar);
    }

    @Override
    public void setScalar(INDArray scalar){
        this.scalarValue = scalar;
    }

    @Override
    public INDArray scalar() {
        if(scalarValue == null && y() != null && y().isScalar())
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
        Preconditions.checkArgument(z().isB(), "Op.Z must have floating point type, since one of operands is floating point." +
                " op.z.datatype=" + z().dataType());

        return true;
    }

    @Override
    public Type getOpType() {
        return Type.SCALAR_BOOL;
    }

    @Override
    public List<org.nd4j.linalg.api.buffer.DataType> calculateOutputDataTypes(List<org.nd4j.linalg.api.buffer.DataType> dataTypes){
        //All scalar bool ops: output type is always bool
        Preconditions.checkState(dataTypes != null && dataTypes.size() == 1, "Expected exactly 1 input datatype for %s, got input %s", getClass(), dataTypes);
        return Collections.singletonList(DataType.BOOL);
    }
}
