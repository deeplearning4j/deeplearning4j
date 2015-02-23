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

import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Signum function
 *
 * @author Adam Gibson
 */
public class Sign extends BaseTransformOp {

    public Sign(INDArray x, INDArray z) {
        super(x, z);
    }

    public Sign(INDArray x, INDArray z, int n) {
        super(x, z, n);
    }

    public Sign(INDArray x, INDArray y, INDArray z, int n) {
        super(x, y, z, n);
    }

    public Sign(INDArray x) {
        super(x);
    }

    @Override
    public String name() {
        return "sign";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other, Object[] extraArgs) {
        return sign(origin);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other, Object[] extraArgs) {
        return sign(origin);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other, Object[] extraArgs) {
        return sign(origin);
    }

    @Override
    public float op(float origin, float other, Object[] extraArgs) {
        return (float) sign(origin);
    }

    @Override
    public double op(double origin, double other, Object[] extraArgs) {
        return sign(origin);
    }

    @Override
    public double op(double origin, Object[] extraArgs) {
        return sign(origin);
    }

    @Override
    public float op(float origin, Object[] extraArgs) {
        return (float) sign(origin);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, Object[] extraArgs) {
        return sign(origin);
    }

    private double sign(double n) {
        if (n < 0)
            return (double) -1;
        else if (n > 0)
            return (double) 1;
        return (double) 0;
    }

    private IComplexNumber sign(IComplexNumber n) {
        if (n.realComponent().doubleValue() > 0)
            return Nd4j.createDouble(1, 0);
        else if (n.realComponent().doubleValue() < 0)
            return Nd4j.createDouble(-1, 0);

        return Nd4j.createDouble(0, 0);
    }

}
