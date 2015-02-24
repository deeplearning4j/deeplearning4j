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
import org.nd4j.linalg.api.ops.TransformOp;
import org.nd4j.linalg.util.ComplexUtil;

/**
 * Exp elementwise function
 * @author Adam Gibson
 */
public class Exp extends BaseTransformOp {
    public Exp(INDArray x, INDArray z) {
        super(x, z);
    }

    public Exp(INDArray x, INDArray z, int n) {
        super(x, z, n);
    }

    public Exp(INDArray x, INDArray y, INDArray z, int n) {
        super(x, y, z, n);
    }

    public Exp(INDArray x) {
        super(x);
    }

    @Override
    public String name() {
        return "exp";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other, Object[] extraArgs) {
        return ComplexUtil.exp(origin);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other, Object[] extraArgs) {
        return ComplexUtil.exp(origin);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other, Object[] extraArgs) {
        return ComplexUtil.exp(origin);
    }

    @Override
    public float op(float origin, float other, Object[] extraArgs) {
        return (float) FastMath.exp(origin);
    }

    @Override
    public double op(double origin, double other, Object[] extraArgs) {
        return FastMath.exp(origin);
    }

    @Override
    public double op(double origin, Object[] extraArgs) {
        return FastMath.exp(origin);
    }

    @Override
    public float op(float origin, Object[] extraArgs) {
        return (float) FastMath.exp(origin);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, Object[] extraArgs) {
        return ComplexUtil.exp(origin);
    }

    @Override
    public TransformOp derivative() {
        return this;
    }

    @Override
    public Op opForDimension(int index,int dimension) {
        if(y() != null)
            return new Exp(x.vectorAlongDimension(index,dimension),y.vectorAlongDimension(index,dimension),z.vectorAlongDimension(index,dimension),x.length());
        else
            return new Exp(x.vectorAlongDimension(index,dimension),z.vectorAlongDimension(index,dimension),x.length());

    }
}
