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

package org.nd4j.linalg.api.ops.impl.shape;

import org.apache.commons.math3.util.FastMath;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.ShapeOp;
import org.nd4j.linalg.util.ComplexUtil;

import java.util.List;

/**
 * Broadcast function
 *
 * @author Adam Gibson
 */
public class Broadcast extends ShapeOp {

    public Broadcast(SameDiff sameDiff, int[] shape) {
        super(sameDiff);
    }

    public Broadcast(SameDiff sameDiff, DifferentialFunction i_v, int[] shape) {
        super(sameDiff, i_v, shape,false,null);
    }

    public Broadcast() {}

    public Broadcast(INDArray x, INDArray z) {
        super(x, z);
    }

    public Broadcast(INDArray x, INDArray z, long n) {
        super(x, z, n);
    }

    public Broadcast(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }

    public Broadcast(INDArray x) {
        super(x);
    }

    @Override
    public void exec(int... dimensions) {
        exec();
    }

    @Override
    public boolean isExecSpecial() {
        return true;
    }

    @Override
    public void exec() {
        int[] permuteDims = extraArgs == null ? z().shape() : (int[]) extraArgs[0];
        if(x != z) {
            if(x.isScalar() && !z.isScalar()) {
                z.assign(x.getDouble(0));
            }
            else
                z.assign(x.broadcast(permuteDims));
        }
        else {
            if(x.isScalar() && !z.isScalar()) {
                z.assign(x.getDouble(0));
            }
            else
                this.z = x.broadcast(permuteDims);
        }

    }


    @Override
    public int opNum() {
        return 0;
    }

    @Override
    public String name() {
        return "broadcast";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        return ComplexUtil.abs(origin);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        return ComplexUtil.abs(origin);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        return ComplexUtil.abs(origin);
    }

    @Override
    public float op(float origin, float other) {
        return FastMath.abs(origin);
    }

    @Override
    public double op(double origin, double other) {
        return FastMath.abs(origin);
    }

    @Override
    public double op(double origin) {
        return FastMath.abs(origin);
    }

    @Override
    public float op(float origin) {
        return FastMath.abs(origin);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
        return ComplexUtil.abs(origin);
    }

    @Override
    public Op opForDimension(int index, int dimension) {
        INDArray xAlongDimension = x.vectorAlongDimension(index, dimension);

        if (y() != null)
            return new Broadcast(xAlongDimension, y.vectorAlongDimension(index, dimension),
                    z.vectorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new Broadcast(xAlongDimension, z.vectorAlongDimension(index, dimension), xAlongDimension.length());

    }



    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);

        if (y() != null)
            return new Broadcast(xAlongDimension, y.tensorAlongDimension(index, dimension),
                    z.tensorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new Broadcast(xAlongDimension, z.tensorAlongDimension(index, dimension), xAlongDimension.length());

    }

    @Override
    public int[] getResultShape() {
        return shape;
    }


    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> i_v) {
        throw new UnsupportedOperationException();
    }
}
