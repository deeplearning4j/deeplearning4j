/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.nd4j.linalg.api.ops.impl.transforms;

import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Stabilization function, forces values to be within a range
 * @author Adam Gibson
 */
public class Stabilize extends BaseTransformOp {
    double realMin = 1.1755e-38f;
    double cutOff = FastMath.log(realMin);

    public Stabilize(INDArray x, INDArray z) {
        super(x, z);
    }

    public Stabilize(INDArray x, INDArray z, int n) {
        super(x, z, n);
    }

    public Stabilize(INDArray x, INDArray y, INDArray z, int n) {
        super(x, y, z, n);
    }

    public Stabilize(INDArray x) {
        super(x);
    }


    @Override
    public String name() {
        return "stabilize";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other, Object[] extraArgs) {
        return stabilize(origin,k(extraArgs));
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other, Object[] extraArgs) {
        return stabilize(origin,k(extraArgs));
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other, Object[] extraArgs) {
        return stabilize(origin,k(extraArgs));
    }

    @Override
    public float op(float origin, float other, Object[] extraArgs) {
        return (float) stabilize(origin,k(extraArgs));
    }

    @Override
    public double op(double origin, double other, Object[] extraArgs) {
        return stabilize(origin,k(extraArgs));
    }

    @Override
    public double op(double origin, Object[] extraArgs) {
        return stabilize(origin,k(extraArgs));
    }

    @Override
    public float op(float origin, Object[] extraArgs) {
        return (float) stabilize(origin,k(extraArgs));
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, Object[] extraArgs) {
        return stabilize(origin,k(extraArgs));
    }

    private double stabilize(double curr,double k) {

        if (curr * k > -cutOff)
            return -cutOff / k;
        else if (curr * k < cutOff)
            return cutOff / k;
        return curr;
    }

    private IComplexNumber stabilize(IComplexNumber c,double k) {
        double realMin = 1.1755e-38f;
        double cutOff = FastMath.log(realMin);
        double curr = c.realComponent().doubleValue();
        if (curr * k > -cutOff)
            return Nd4j.createDouble(-cutOff / k, c.imaginaryComponent().doubleValue());
        else if (curr * k < cutOff)
            return Nd4j.createDouble(cutOff / k, c.imaginaryComponent().doubleValue());
        return c;

    }

    private double k(Object[] extraArgs) {
        if(extraArgs == null || extraArgs.length < 1)
            throw new IllegalArgumentException("Please specify a k");
        return Double.valueOf(extraArgs[0].toString());
    }

    @Override
    public Op opForDimension(int index,int dimension) {
        INDArray xAlongDimension = x.vectorAlongDimension(index,dimension);
        if(y() != null)
            return new Stabilize(xAlongDimension,y.vectorAlongDimension(index,dimension),z.vectorAlongDimension(index,dimension),xAlongDimension.length());
        else
            return new Stabilize(xAlongDimension,z.vectorAlongDimension(index,dimension),xAlongDimension.length());

    }

}
