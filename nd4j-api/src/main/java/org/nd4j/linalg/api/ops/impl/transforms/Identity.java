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

/**
 * Identity function
 * @author Adam Gibson
 */
public class Identity extends BaseTransformOp {

    public Identity(INDArray x, INDArray z) {
        super(x, z);
    }

    public Identity(INDArray x, INDArray z, int n) {
        super(x, z, n);
    }

    public Identity(INDArray x, INDArray y, INDArray z, int n) {
        super(x, y, z, n);
    }

    public Identity(INDArray x) {
        super(x);
    }

    @Override
    public String name() {
        return "identity";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other, Object[] extraArgs) {
        return origin;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other, Object[] extraArgs) {
        return origin;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other, Object[] extraArgs) {
        return origin;
    }

    @Override
    public float op(float origin, float other, Object[] extraArgs) {
        return origin;
    }

    @Override
    public double op(double origin, double other, Object[] extraArgs) {
        return origin;
    }

    @Override
    public double op(double origin, Object[] extraArgs) {
        return origin;
    }

    @Override
    public float op(float origin, Object[] extraArgs) {
        return origin;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, Object[] extraArgs) {
        return origin;
    }
}
