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

package org.nd4j.linalg.api.ops.impl.transforms;

import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.TransformOp;

/**
 * HardSigmoid function
 *
 * @author raver119@gmail.com
 */
public class HardSigmoid extends BaseTransformOp {
    public HardSigmoid() {}

    public HardSigmoid(INDArray x, INDArray z) {
        super(x, z);
    }

    public HardSigmoid(INDArray x, INDArray z, long n) {
        super(x, z, n);
    }

    public HardSigmoid(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }

    public HardSigmoid(INDArray x, INDArray y, INDArray z) {
        super(x, y, z, x.lengthLong());
    }

    public HardSigmoid(INDArray ndArray) {
        super(ndArray);
    }

    @Override
    public int opNum() {
        return 51;
    }

    @Override
    public String name() {
        return "hard_sigmoid";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        return sigmoid(origin);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        return sigmoid(origin);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        return sigmoid(origin);
    }

    @Override
    public float op(float origin, float other) {
        return (float) hardSigmoid(origin);
    }

    @Override
    public double op(double origin, double other) {
        return hardSigmoid(origin);
    }

    @Override
    public double op(double origin) {
        return hardSigmoid(origin);
    }

    @Override
    public float op(float origin) {
        return (float) hardSigmoid(origin);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
        return sigmoid(origin);
    }


    private double hardSigmoid(double input) {
        return Math.min(1.0, Math.max(0, 0.2 * input + 0.5));
    }

    @Override
    public TransformOp derivative() {
        return new HardSigmoidDerivative(x, y, z, n);
    }

    private IComplexNumber sigmoid(IComplexNumber number) {
        return null;
    }

    @Override
    public Op opForDimension(int index, int dimension) {
        INDArray xAlongDimension = x.vectorAlongDimension(index, dimension);

        if (y() != null)
            return new HardSigmoid(x.vectorAlongDimension(index, dimension), y.vectorAlongDimension(index, dimension),
                            z.vectorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new HardSigmoid(x.vectorAlongDimension(index, dimension), z.vectorAlongDimension(index, dimension),
                            xAlongDimension.length());

    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);

        if (y() != null)
            return new HardSigmoid(x.tensorAlongDimension(index, dimension), y.tensorAlongDimension(index, dimension),
                            z.tensorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new HardSigmoid(x.tensorAlongDimension(index, dimension), z.tensorAlongDimension(index, dimension),
                            xAlongDimension.length());

    }

}
