/*-
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *
 */

package org.nd4j.linalg.api.ops;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.linalg.util.LinAlgExceptions;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * A base op for basic getters and setters
 *
 * @author Adam Gibson
 */
@Slf4j
public abstract class BaseTransformOp extends BaseOp implements TransformOp {


    public BaseTransformOp(SameDiff sameDiff,
                           SDVariable i_v1,
                           SDVariable i_v2) {
        this(sameDiff,i_v1,i_v2,false);
    }

    public BaseTransformOp(SameDiff sameDiff,
                           SDVariable i_v1,
                           SDVariable i_v2,
                           boolean inPlace) {
        super(sameDiff,inPlace,new Object[] {i_v2});
        if (i_v1 != null && i_v2 != null) {
            f().validateDifferentialFunctionsameDiff(i_v1);
            f().validateDifferentialFunctionsameDiff(i_v2);
            this.sameDiff = sameDiff;
            this.inPlace = inPlace;
            this.xVertexId = i_v1.getVarName();
            this.yVertexId = i_v2.getVarName();
            sameDiff.addArgsFor(new SDVariable[]{i_v1,i_v2},this);
            if(Shape.isPlaceholderShape(i_v1.getShape())) {
                sameDiff.addPropertyToResolve(this,i_v1.getVarName());
            }

            if(Shape.isPlaceholderShape(i_v2.getShape())) {
                sameDiff.addPropertyToResolve(this,i_v2.getVarName());
            }
            if(i_v1.getShape() != null)
                this.n = ArrayUtil.prod(i_v1.getShape());


        } else {
            throw new IllegalArgumentException("Input not null variables.");
        }


    }

    public BaseTransformOp(SameDiff sameDiff) {
        this.sameDiff = sameDiff;
    }

    public BaseTransformOp(SameDiff sameDiff,
                           SDVariable i_v1,
                           SDVariable i_v2,
                           Object[] extraArgs) {
        super(sameDiff,extraArgs);
        if (i_v1 != null && i_v2 != null) {

            f().validateDifferentialFunctionsameDiff(i_v1);
            f().validateDifferentialFunctionsameDiff(i_v2);
            this.sameDiff = sameDiff;
            this.xVertexId = i_v1.getVarName();
            this.yVertexId = i_v2.getVarName();
            sameDiff.addArgsFor(new SDVariable[]{i_v1,i_v2},this);
            if(i_v1.getShape() != null)
                this.n = ArrayUtil.prod(i_v1.getShape());

            if(Shape.isPlaceholderShape(i_v1.getShape())) {
                sameDiff.addPropertyToResolve(this,i_v1.getVarName());
            }

            if(Shape.isPlaceholderShape(i_v2.getShape())) {
                sameDiff.addPropertyToResolve(this,i_v2.getVarName());
            }

        } else {
            throw new IllegalArgumentException("Input not null variables.");
        }

    }




    public BaseTransformOp(SameDiff sameDiff,SDVariable i_v,boolean inPlace) {
        this(sameDiff,i_v,i_v.getShape(),inPlace,null);
    }

    public BaseTransformOp(SameDiff sameDiff,
                           SDVariable i_v,
                           int[] shape,
                           boolean inPlace,
                           Object[] extraArgs) {
        // FIXME: int cast !
        this(sameDiff, i_v, ArrayUtil.toLongArray(shape), inPlace, extraArgs);
    }

    public BaseTransformOp(SameDiff sameDiff,
                           SDVariable i_v,
                           long[] shape,
                           boolean inPlace,
                           Object[] extraArgs) {
        super(sameDiff,inPlace,extraArgs);

        if (i_v != null) {
            f().validateDifferentialFunctionsameDiff(i_v);
            this.xVertexId = i_v.getVarName();
            sameDiff.addArgsFor(new SDVariable[]{i_v},this);
            if(i_v.getShape() != null) {
                this.n = ArrayUtil.prod(i_v.getShape());
            }

            if(Shape.isPlaceholderShape(i_v.getShape())) {
                sameDiff.addPropertyToResolve(this,i_v.getVarName());
            }


        } else {
            throw new IllegalArgumentException("Input must not null variable.");
        }

    }


    public BaseTransformOp(SameDiff sameDiff,
                           SDVariable i_v,
                           Object[] extraArgs) {
        this(sameDiff,i_v,i_v.getShape(),false,extraArgs);
    }



    public BaseTransformOp(INDArray x, INDArray z) {
        super(x, z);
        LinAlgExceptions.assertSameShape(x, z);
    }

    public BaseTransformOp() {}

    public BaseTransformOp(INDArray x, INDArray z, long n) {
        super(x, z, n);
    }

    public BaseTransformOp(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
        if (y != null)
            LinAlgExceptions.assertSameLength(x, y, z);
        else
            LinAlgExceptions.assertSameLength(x, z);

    }

    public BaseTransformOp(INDArray x) {
        super(x);
    }

    @Override
    public Type opType() {
        if(args() == null || args().length == 1)
            return Type.TRANSFORM;
        else if(args().length == 2)
            return Type.PAIRWISE;

        else throw new ND4JIllegalStateException("Illegal number of args (can only be 1 or 2)");
    }



    @Override
    public List<long[]> calculateOutputShape() {
        List<long[]> ret = new ArrayList<>(1);
        if(arg() == null)
            throw new ND4JIllegalStateException("No arg found for op!");

        val arr = sameDiff.getArrForVarName(arg().getVarName());
        if(arr == null)
            return Collections.emptyList();
        ret.add(arr.shape());
        this.n = arr.length();
        return ret;
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


}
