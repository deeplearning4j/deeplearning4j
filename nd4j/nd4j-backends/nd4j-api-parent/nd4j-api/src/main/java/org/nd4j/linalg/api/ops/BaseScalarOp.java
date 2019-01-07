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

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
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

    public BaseScalarOp(INDArray x, INDArray y, INDArray z, long n, Number num) {
        super(x, y, z, n);
        if (x.isCompressed())
            Nd4j.getCompressor().decompressi(x);

        this.scalarValue = Nd4j.scalar(x.dataType(), num);

        init(x, y, z, n);
    }

    public BaseScalarOp(INDArray x, Number num) {
        super(x);
        if (x.isCompressed())
            Nd4j.getCompressor().decompressi(x);

        this.scalarValue = Nd4j.scalar(x.dataType(), num);
        init(x, y, z, n);

    }
    public BaseScalarOp(INDArray x, INDArray z, Number set) {
        super(x, null, z, x.length());
        if (x.isCompressed())
            Nd4j.getCompressor().decompressi(x);

        this.scalarValue= Nd4j.scalar(x.dataType(), set);
    }




    public BaseScalarOp(SameDiff sameDiff,SDVariable i_v,Number scalar) {
        this(sameDiff,i_v,scalar,false,null);
    }

    public BaseScalarOp(SameDiff sameDiff,SDVariable i_v,Number scalar,boolean inPlace) {
        this(sameDiff,i_v,scalar,inPlace,null);
    }

    public BaseScalarOp(SameDiff sameDiff,
                        SDVariable i_v,
                        Number scalar,
                        boolean inPlace,
                        Object[] extraArgs) {
        super(sameDiff,inPlace,extraArgs);
        this.scalarValue = Nd4j.scalar(scalar);
        if (i_v != null) {
            this.xVertexId = i_v.getVarName();
            sameDiff.addArgsFor(new String[]{xVertexId},this);
            if(Shape.isPlaceholderShape(i_v.getShape())) {
                sameDiff.addPropertyToResolve(this,i_v.getVarName());
            }
            f().validateDifferentialFunctionsameDiff(i_v);
        } else {
            throw new IllegalArgumentException("Input not null variable.");
        }

    }


    public BaseScalarOp(SameDiff sameDiff,
                        SDVariable i_v,
                        Number scalar,
                        Object[] extraArgs) {
        this(sameDiff,i_v,scalar,false,extraArgs);
    }



    @Override
    public INDArray z() {
        if(z == null) {
            if(sameDiff != null) {
                this.z = outputVariables()[0].getArr();
                if(this.z == null) {
                    val var = outputVariables()[0];
                    if(var.getShape() != null)
                        this. z = var.storeAndAllocateNewArray();
                    else {
                        val argsShape = args()[0].getShape();
                        if(argsShape != null) {
                            sameDiff.putShapeForVarName(var.getVarName(),argsShape);
                            this. z = var.storeAndAllocateNewArray();
                        }
                    }
                }
            }
        }

        return z;
    }


    @Override
    public List<LongShapeDescriptor> calculateOutputShape() {
        val ret = new ArrayList<LongShapeDescriptor>(1);

        val s = arg().getShape();

        if (s == null)
            return Collections.emptyList();

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

}
