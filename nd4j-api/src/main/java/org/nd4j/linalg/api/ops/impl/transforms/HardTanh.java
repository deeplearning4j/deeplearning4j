/*
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

package org.nd4j.linalg.api.ops.impl.transforms;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.TransformOp;
import org.nd4j.linalg.util.ComplexUtil;

/**
 * Hard tanh elementwise function
 *
 * @author Adam Gibson
 */
public class HardTanh extends BaseTransformOp {
    public HardTanh() {
    }

    public HardTanh(INDArray x, INDArray z) {
        super(x, z);
    }

    public HardTanh(INDArray x, INDArray z, int n) {
        super(x, z, n);
    }

    public HardTanh(INDArray x, INDArray y, INDArray z, int n) {
        super(x, y, z, n);
    }

    public HardTanh(INDArray x) {
        super(x);
    }

    @Override
    public int opNum() {
        return 19;
    }

    @Override
    public String name() {
        return "hardtanh";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        return ComplexUtil.hardTanh(origin);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        return ComplexUtil.hardTanh(origin);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        return ComplexUtil.hardTanh(origin);
    }

    @Override
    public float op(float origin, float other) {
        return hardTanh(origin);
    }

    @Override
    public double op(double origin, double other) {
        return hardTanh(origin);
    }

    @Override
    public double op(double origin) {
        return hardTanh(origin);
    }

    @Override
    public float op(float origin) {
        return hardTanh(origin);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
        return ComplexUtil.hardTanh(origin);
    }


    @Override
    public TransformOp derivative() {
        return new HardTanhDerivative(x, y, z, n);
    }

    private static float hardTanh(float num) {
    	return num < -1.0f ? -1.0f : num > 1.0f ? 1.0f : num;
    }

    private static double hardTanh(double num) {
        return num < -1.0 ? -1.0 : num > 1.0 ? 1.0 : num;

    }

    @Override
    public Op opForDimension(int index, int dimension) {
        INDArray xAlongDimension = x.vectorAlongDimension(index, dimension);

        if (y() != null)
            return new HardTanh(xAlongDimension, y.vectorAlongDimension(index, dimension), z.vectorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new HardTanh(xAlongDimension, z.vectorAlongDimension(index, dimension), xAlongDimension.length());

    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);

        if (y() != null)
            return new HardTanh(xAlongDimension, y.tensorAlongDimension(index, dimension), z.tensorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new HardTanh(xAlongDimension, z.tensorAlongDimension(index, dimension), xAlongDimension.length());

    }

}
